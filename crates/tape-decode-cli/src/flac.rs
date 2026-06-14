//! FLAC input via symphonia. Decodes a mono FLAC stream to `f32` samples behind
//! the same [`SampleSource`] interface as the raw formats. Signed integer FLAC
//! samples are normalized to ~[-1, 1], so a 16-bit FLAC reads identically to the
//! equivalent raw `s16` capture. File-backed input seeks via
//! symphonia's binary search; stdin can only skip forward by decoding.

use anyhow::{bail, Context as _, Result};
use symphonia_bundle_flac::{FlacDecoder, FlacReader};
use symphonia_core::audio::{Audio, GenericAudioBufferRef};
use symphonia_core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
use symphonia_core::codecs::CodecParameters;
use symphonia_core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia_core::io::{MediaSource, MediaSourceStream};
use symphonia_core::units::Timestamp;

use crate::reader::SampleSource;

/// Decode `source` (a seekable file or a forward-only pipe) as a FLAC sample
/// source.
pub fn open(source: Box<dyn MediaSource>) -> Result<Box<dyn SampleSource>> {
    Ok(Box::new(FlacSource::new(source)?))
}

struct FlacSource {
    reader: FlacReader<'static>,
    decoder: FlacDecoder,
    track_id: u32,
    /// Right shift undoing symphonia's normalization of samples to the full i32
    /// range (`32 - bits_per_sample`), then `scale` normalizes the native signed
    /// integer amplitude to match raw signed PCM input.
    shift: u32,
    scale: f32,
    /// Whether the source supports real seeking (a regular file); pipes can only
    /// skip forward by decoding.
    seekable: bool,
    /// Decoded samples of the current packet not yet returned.
    pending: Vec<f32>,
    pending_pos: usize,
    /// Absolute index of the next sample `read` will return.
    position: u64,
    eof: bool,
}

impl FlacSource {
    fn new(source: Box<dyn MediaSource>) -> Result<Self> {
        let seekable = source.is_seekable();
        let mss = MediaSourceStream::new(source, Default::default());
        let reader =
            FlacReader::try_new(mss, FormatOptions::default()).context("failed to read FLAC")?;

        let (track_id, params) = {
            let track = reader.tracks().first().context("FLAC has no tracks")?;
            let params = track
                .codec_params
                .as_ref()
                .and_then(CodecParameters::audio)
                .context("FLAC track is not audio")?
                .clone();
            (track.id, params)
        };

        let bits = params
            .bits_per_sample
            .context("FLAC stream info missing bits per sample")?;
        if !(1..=32).contains(&bits) {
            bail!("unsupported FLAC bit depth: {bits}");
        }
        let channels = params.channels.as_ref().map_or(1, |c| c.count());
        if channels != 1 {
            bail!("only mono FLAC input is supported, found {channels} channels");
        }

        let decoder = FlacDecoder::try_new(&params, &AudioDecoderOptions::default())
            .context("failed to initialize FLAC decoder")?;

        let scale = 1.0 / ((1u64 << (bits - 1)) as f32);

        Ok(Self {
            reader,
            decoder,
            track_id,
            shift: 32 - bits,
            scale,
            seekable,
            pending: Vec::new(),
            pending_pos: 0,
            position: 0,
            eof: false,
        })
    }

    /// Decode the next packet into `pending`. Returns false at end of input.
    fn decode_next(&mut self) -> Result<bool> {
        let shift = self.shift;
        let scale = self.scale;
        loop {
            let Some(packet) = self.reader.next_packet().context("FLAC read error")? else {
                self.eof = true;
                return Ok(false);
            };
            if packet.track_id != self.track_id {
                continue;
            }
            let decoded = self.decoder.decode(&packet).context("FLAC decode error")?;
            if decoded.frames() == 0 {
                continue;
            }
            let GenericAudioBufferRef::S32(buf) = decoded else {
                bail!("unexpected FLAC sample format (expected signed 32-bit)");
            };
            let plane = buf.plane(0).expect("mono FLAC has one plane");
            self.pending.clear();
            self.pending
                .extend(plane.iter().map(|&s| ((s >> shift) as f32) * scale));
            self.pending_pos = 0;
            return Ok(true);
        }
    }

    /// Ensure `pending` has unconsumed samples, decoding if needed. False at EOF.
    fn fill(&mut self) -> Result<bool> {
        if self.pending_pos < self.pending.len() {
            return Ok(true);
        }
        if self.eof {
            return Ok(false);
        }
        self.decode_next()
    }

    /// Drop samples until `position` reaches `target` (forward only).
    fn skip_to(&mut self, target: u64) -> Result<()> {
        while self.position < target {
            if !self.fill()? {
                bail!("FLAC input ended before reaching sample offset {target}");
            }
            let avail = self.pending.len() - self.pending_pos;
            let n = avail.min((target - self.position) as usize);
            self.pending_pos += n;
            self.position += n as u64;
        }
        Ok(())
    }
}

impl SampleSource for FlacSource {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let mut written = 0;
        while written < out.len() {
            if !self.fill()? {
                break;
            }
            let avail = &self.pending[self.pending_pos..];
            let n = avail.len().min(out.len() - written);
            out[written..written + n].copy_from_slice(&avail[..n]);
            self.pending_pos += n;
            written += n;
        }
        self.position += written as u64;
        Ok(written)
    }

    fn seek_samples(&mut self, sample: u64) -> Result<()> {
        if self.seekable {
            // Jump to the packet containing `sample` via symphonia's binary
            // search, then drop the handful of samples up to it.
            let ts = Timestamp::try_from(sample).context("seek offset too large")?;
            let seeked = self
                .reader
                .seek(
                    SeekMode::Coarse,
                    SeekTo::Timestamp {
                        ts,
                        track_id: self.track_id,
                    },
                )
                .context("FLAC seek failed")?;
            self.decoder.reset();
            self.pending.clear();
            self.pending_pos = 0;
            self.eof = false;
            self.position = seeked.actual_ts.get() as u64;
        } else if sample < self.position {
            bail!("cannot seek backward on a non-seekable FLAC input (stdin)");
        }
        self.skip_to(sample)
    }
}
