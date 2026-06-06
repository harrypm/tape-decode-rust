use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::time::Instant;

use anyhow::{Context as _, Result};
use tape_decode::{DecoderMetadata, LumaOutput, WriteableField};

use crate::metadata::metadata_to_tbc;

pub struct DecodeWriter {
    outfile_video: File,
    outfile_chroma: Option<File>,
    /// JSON sidecar, kept open and rewritten incrementally: after every field the
    /// file on disk is a complete, valid document (see [`Self::write_writeable`]).
    json_file: Option<File>,
    /// Byte offset of the array-closing `]` in the JSON file — i.e. just past the
    /// last field object written. The next field, or the final metadata, is
    /// written starting here, overwriting the previous `]` + trailing metadata.
    json_field_end: u64,
    field_count: usize,
    first_field_write: Option<Instant>,
    last_field_write: Option<Instant>,
}

fn fps(field_count: usize, elapsed: f64) -> f64 {
    if elapsed > 0.0 {
        field_count as f64 / (elapsed * 2.0)
    } else {
        0.0
    }
}

impl DecodeWriter {
    pub fn new(luma: File, chroma: Option<File>, mut metadata: Option<File>) -> Result<Self> {
        const FIELDS_OPEN: &[u8] = b"{\"fields\":[";
        if let Some(metadata) = metadata.as_mut() {
            let mut chunk = FIELDS_OPEN.to_vec();
            append_tail(&mut chunk, None, 0)?;
            metadata.write_all(&chunk)?;
        }
        Ok(Self {
            outfile_video: luma,
            outfile_chroma: chroma,
            json_file: metadata,
            json_field_end: FIELDS_OPEN.len() as u64,
            field_count: 0,
            first_field_write: None,
            last_field_write: None,
        })
    }

    pub(crate) fn write_writeable(
        &mut self,
        field: &WriteableField,
        metadata: Option<&DecoderMetadata>,
    ) -> Result<()> {
        match field.luma() {
            LumaOutput::Encoded(values) => write_native_slice(&mut self.outfile_video, values)?,
            LumaOutput::Raw(values) => write_native_slice(&mut self.outfile_video, values)?,
        }
        if let Some(outfile) = &mut self.outfile_chroma {
            let chroma = field
                .chroma()
                .context("missing chroma output for chroma file")?;
            write_native_slice(outfile, chroma)?;
        }
        let now = Instant::now();
        let start = self.first_field_write.get_or_insert(now);
        self.last_field_write = Some(now);

        self.field_count += 1;

        if let Some(json_file) = self.json_file.as_mut() {
            // Starting at the array-closing `]`, append this field's object followed
            // by a fresh `]` and the metadata "as if decoding had finished here", so
            // the file is a complete, valid document the instant it hits disk. Track
            // where this field's object ends and seek back there: the next field
            // overwrites the tail with `,<field>]...`, and the final pass overwrites
            // it with the authoritative metadata. Field + tail go out as one write.

            let mut chunk = Vec::new();
            if self.field_count > 1 {
                chunk.push(b',');
            }
            serde_json::to_writer(&mut chunk, &field.info)?;
            let field_end = self.json_field_end + chunk.len() as u64;

            json_file.seek(SeekFrom::Start(self.json_field_end))?;
            append_tail(&mut chunk, metadata, self.field_count)?;
            json_file.write_all(&chunk)?;

            self.json_field_end = field_end;
        }
        
        const LOG_INTERVAL: usize = 500;

        let field_num = self.field_count;
        tracing::debug!("Written field {}", self.field_count);
        if field_num.is_multiple_of(LOG_INTERVAL) {
            let elapsed = now.duration_since(*start).as_secs_f64();
            let fps = fps(field_num, elapsed);
            tracing::info!("Decoded {} fields so far in {:.3}s ({:.2} FPS)", field_num, elapsed, fps);
        }

        Ok(())
    }

    pub(crate) fn close(&mut self, metadata: Option<DecoderMetadata>) -> Result<()> {
        let field_count = self.field_count;
        let elapsed = match (self.first_field_write, self.last_field_write) {
            (Some(first), Some(last)) => last.duration_since(first).as_secs_f64(),
            _ => 0.0,
        };
        let fps = fps(field_count, elapsed);
        tracing::info!(
            "Decode finished: {} fields in {:.3}s ({:.2} FPS)",
            field_count,
            elapsed,
            fps,
        );

        if let Some(json_file) = self.json_file.as_mut() {
            // Overwrite the trailing `]` + metadata with the final, authoritative
            // metadata and truncate, so the file matches a one-shot serialization
            // exactly (the array was opened up front by `open`).
            let mut chunk = Vec::new();
            append_tail(&mut chunk, metadata.as_ref(), field_count)?;
            json_file.seek(SeekFrom::Start(self.json_field_end))?;
            json_file.write_all(&chunk)?;
        }

        Ok(())
    }
}

/// Append the array-closing `]`, the flattened metadata (when present), and the
/// document-closing `}` plus newline to `chunk`. With `metadata` set this yields
/// `],"pcmAudioParameters":{…},"videoParameters":{…}}\n`; without it, `]}\n`.
fn append_tail(
    chunk: &mut Vec<u8>,
    metadata: Option<&DecoderMetadata>,
    field_count: usize,
) -> Result<()> {
    chunk.push(b']');
    if let Some(metadata) = metadata {
        // Serialize the metadata object on its own, then splice its body into the
        // root by replacing its leading `{` with `,` and dropping its trailing `}`.
        let bytes = serde_json::to_vec(&metadata_to_tbc(metadata, field_count))?;
        chunk.push(b',');
        chunk.extend_from_slice(&bytes[1..bytes.len() - 1]);
    }
    chunk.push(b'}');
    chunk.push(b'\n');
    Ok(())
}

fn write_native_slice<T>(file: &mut dyn Write, values: &[T]) -> Result<()> {
    // Write the values' native-endian bytes directly, without an intermediate buffer.
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    };
    file.write_all(bytes)?;
    Ok(())
}
