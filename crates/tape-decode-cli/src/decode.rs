use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};

use anyhow::Result;
use tape_decode::{Decoder, DecoderMetadata, DecoderSpec, LumaOutput, WriteableField, BLOCKSIZE};

use crate::reader::DecodeReader;
use crate::writer::DecodeWriter;

/// Decode the whole input serially. Like the multithreaded path, the input is
/// streamed once from the stream start and the decoder skips past everything
/// before `start_offset` itself (so this works on non-seekable inputs such as
/// pipes); `start_offset` is taken directly rather than inferred from the
/// reader's position.
pub fn decode_all(
    reader: &mut DecodeReader,
    writer: &mut DecodeWriter,
    spec: Arc<DecoderSpec>,
    start_offset: u64,
) -> Result<()> {
    let mut decoder = Decoder::new(Arc::clone(&spec), start_offset);

    // Feed the decoder one chunk at a time over a sliding window starting at
    // absolute sample `base` (0, since reading begins at the stream start). A
    // chunk exceeds one field's span, so each refill makes progress. `decode`
    // reports the offset before which input is no longer needed; we drop that
    // prefix, or seek forward past it when `start_offset` lands beyond the window.
    let chunk = spec.readlen() + 4 * BLOCKSIZE;
    let mut window: Vec<f32> = Vec::new();
    let mut read_buffer = vec![0.0f32; chunk];
    let mut base: u64 = 0;
    let mut read_pos: u64 = 0;

    let mut fields_written = 0usize;
    loop {
        let read = reader.read(&mut read_buffer)?;
        window.extend_from_slice(&read_buffer[..read]);
        let final_chunk = read < chunk;
        read_pos += read as u64;

        let (consumed, fields) = decoder.decode(&window, base, final_chunk)?;
        let metadata = decoder.metadata();
        for field in &fields {
            writer.write_writeable(field, metadata.as_ref())?;
        }
        fields_written += fields.len();

        if final_chunk {
            break;
        }
        if consumed > read_pos {
            // The decoder skipped past everything buffered (e.g. an initial --start
            // offset); seek the input forward and start a fresh window there.
            reader.seek_samples(consumed)?;
            read_pos = consumed;
            window.clear();
        } else {
            window.drain(..(consumed - base) as usize);
        }
        base = consumed;
    }

    if fields_written != 0 {
        tracing::info!("Completed: saving JSON and exiting.");
    } else {
        tracing::info!("Completed without handling any frames.");
    }

    writer.close(decoder.metadata())?;
    Ok(())
}

// ============================================================================
// Multithreaded decoding
// ============================================================================
//
// Decoding is inherently serial: each field's levels and phase are built up from
// the fields before it. To parallelize we run several decoders concurrently,
// each seeked `distance_size` fields apart, and stitch their outputs back into a
// single stream. A decoder that starts mid-file produces a garbage first field
// while it locks on, then converges onto exactly the output a single serial
// decode would have produced for that part of the file. We exploit that by
// having the earlier ("authoritative") decoder overlap into the next decoder's
// region and, once `overlap_count` consecutive fields agree, hand off to the
// later decoder — which has already decoded ahead in parallel.
//
// Determinism: segment boundaries are fixed (`seg * distance_size` fields), each
// decoder is deterministic, and every stitch decision (earlier decoder wins,
// first `overlap_count`-in-a-row match stitches) is independent of thread timing.
// So the output depends only on the parameters and input, never on scheduling.

/// Multithreading parameters. See the `--mt-*` CLI flags.
pub struct MtParams {
    /// Number of concurrent decoder threads. Must be >= 1 here (the `== 0` case
    /// is handled by the serial [`decode_all`]).
    pub threads: usize,
    /// Fields of distance between each thread's start offset, and the width of
    /// the overlap window searched for a stitch.
    pub distance_size: u64,
    /// Number of consecutive matching fields required to stitch.
    pub overlap_count: usize,
    /// Per-field MSRE threshold below which two fields are considered equal.
    pub threshold: f64,
    /// Fraction of the largest per-sample squared deviations discarded before.
    pub trim_fraction: f64,
}

/// 4-field phase ID from field order and color-frame phase, matching the
/// decoder's own derivation so a renumbered global sequence reproduces the value
/// a serial decode would have written.
fn field_phase_id(first_field: bool, second_phase: bool) -> i64 {
    match (first_field, second_phase) {
        (true, true) => 1,
        (false, false) => 2,
        (true, false) => 3,
        (false, true) => 4,
    }
}

// --- Field comparison --------------------------------------------------------

use crate::fields_match::{f32_msre, wrapped_u16_msre};

fn luma_matches(a: &LumaOutput, b: &LumaOutput, mt: &MtParams) -> bool {
    match (a, b) {
        (LumaOutput::Encoded(x), LumaOutput::Encoded(y)) => {
            x.len() == y.len() && wrapped_u16_msre(x, y, mt.trim_fraction) < mt.threshold
        }
        (LumaOutput::Raw(x), LumaOutput::Raw(y)) => {
            x.len() == y.len() && f32_msre(x, y, mt.trim_fraction) < mt.threshold
        }
        // Different output kinds can never be stitched together.
        _ => false,
    }
}

/// Whether two fields are equal enough to stitch.
fn fields_match(a: &WriteableField, b: &WriteableField, mt: &MtParams) -> bool {
    if !luma_matches(a.luma(), b.luma(), mt) {
        return false;
    }
    match (a.chroma(), b.chroma()) {
        (Some(x), Some(y)) => {
            x.len() == y.len() && wrapped_u16_msre(x, y, mt.trim_fraction) < mt.threshold
        }
        (None, None) => true,
        _ => false,
    }
}

// --- Shared sequential input ------------------------------------------------
//
// Every worker shares one view of the input through `Tape`, which streams the
// source exactly once, in order, behind a mutex. A worker asks for the samples
// at an absolute offset; if they are already buffered it copies them out,
// otherwise it reads the next sequential block from the source itself. Reading
// is therefore cooperative — no dedicated reader thread — and any byte is read
// at most once. The orchestrator marks how far the lowest-numbered live decoder
// has advanced via `set_drop_threshold`; everything before that is dropped, so
// memory stays bounded to the span the live decoders actually cover. Because
// the source is only ever read forward, this works on non-seekable inputs such
// as pipes.

/// Largest block read from the source while holding the lock, so other workers
/// can interleave cached reads between blocks during the initial fan-out.
const TAPE_READ_BLOCK: usize = 1 << 20;

struct Tape {
    // The buffered samples sit behind an `RwLock` so the lagging decoders, which
    // only copy out already-buffered data, can do so concurrently; extending the
    // frontier or dropping the prefix takes the exclusive lock. The source lives
    // behind its own mutex (it is `Send` but not `Sync`) and is only touched
    // while extending, always after taking the write lock, so reads stay strictly
    // ordered and there is a single lock order (buffer then source).
    buf: RwLock<BufState>,
    source: Mutex<DecodeReader>,
}

struct BufState {
    /// Buffered input samples (widened to `f32`) covering `[start, start + buf.len())`.
    buf: Vec<f32>,
    /// Absolute sample offset of `buf[0]`.
    start: u64,
    /// Set once the source has reported end of input.
    eof: bool,
    /// Total input length in samples, known once `eof` is reached.
    len: Option<u64>,
    /// Samples before this offset are no longer needed by any decoder.
    drop_threshold: u64,
}

impl Tape {
    fn new(source: DecodeReader) -> Self {
        Self {
            buf: RwLock::new(BufState {
                buf: Vec::new(),
                start: 0,
                eof: false,
                len: None,
                drop_threshold: 0,
            }),
            source: Mutex::new(source),
        }
    }

    /// Total input length in samples, or `None` until the source hits EOF.
    fn known_len(&self) -> Option<u64> {
        self.buf.read().unwrap().len
    }

    /// Mark that no decoder needs samples before `offset`, dropping them.
    fn set_drop_threshold(&self, offset: u64) {
        let mut state = self.buf.write().unwrap();
        if offset > state.drop_threshold {
            state.drop_threshold = offset;
        }
        Self::drop_prefix(&mut state);
    }

    fn drop_prefix(state: &mut BufState) {
        let drop_to = state
            .drop_threshold
            .min(state.start + state.buf.len() as u64);
        if drop_to > state.start {
            let n = (drop_to - state.start) as usize;
            state.buf.drain(..n);
            state.start = drop_to;
        }
    }

    /// Copy the samples `[from, from + dst.len())` out of `state` into `dst`,
    /// truncated at the buffered frontier. Returns the count written.
    fn copy_out(state: &BufState, from: u64, dst: &mut [f32]) -> usize {
        let frontier = state.start + state.buf.len() as u64;
        let lo = from.saturating_sub(state.start).min(state.buf.len() as u64) as usize;
        let hi = (from + dst.len() as u64)
            .min(frontier)
            .saturating_sub(state.start)
            .min(state.buf.len() as u64) as usize;
        let got = hi - lo;
        dst[..got].copy_from_slice(&state.buf[lo..hi]);
        got
    }

    /// Fill `dst` with samples starting at absolute offset `from`, reading the
    /// source forward as needed. Returns the number of samples written and
    /// whether end of input was reached (fewer than `dst.len()` were available),
    /// or `None` if `stop` was raised. `from` must be at or after the current
    /// drop threshold, which every active worker satisfies.
    fn read_into(
        &self,
        from: u64,
        dst: &mut [f32],
        stop: &AtomicBool,
    ) -> Result<Option<(usize, bool)>> {
        let want = dst.len() as u64;
        let target = from + want;
        loop {
            if stop.load(Ordering::Relaxed) {
                return Ok(None);
            }
            // Fast path: the samples are already buffered (or input has ended), so
            // copy them out under a shared read lock, concurrently with the other
            // lagging decoders.
            {
                let state = self.buf.read().unwrap();
                if state.eof || state.start + state.buf.len() as u64 >= target {
                    let got = Self::copy_out(&state, from, dst);
                    return Ok(Some((got, (got as u64) < want)));
                }
            }
            // Slow path: extend the frontier by reading the next sequential block
            // straight into the buffer's tail. Releasing the lock after each block
            // lets readers copy cached data between blocks instead of waiting for
            // one big read.
            let mut state = self.buf.write().unwrap();
            Self::drop_prefix(&mut state);
            let frontier = state.start + state.buf.len() as u64;
            if !state.eof && frontier < target {
                let read_len = (target - frontier).min(TAPE_READ_BLOCK as u64) as usize;
                let old = state.buf.len();
                state.buf.resize(old + read_len, 0.0);
                let mut source = self.source.lock().unwrap();
                let n = source.read(&mut state.buf[old..])?;
                drop(source);
                state.buf.truncate(old + n);
                if n == 0 {
                    state.eof = true;
                    state.len = Some(state.start + state.buf.len() as u64);
                }
            }
        }
    }
}

/// The first absolute sample offset a decoder started at `start_offset` will
/// read. A decode call with an empty window does no work and leaves the decoder
/// untouched, but reports exactly this offset — the point a serial decode would
/// skip its input to. Used to start each worker's window without reading or
/// seeking earlier input, and to tell the tape how much it may drop.
fn first_needed_offset(spec: &Arc<DecoderSpec>, start_offset: u64) -> Result<u64> {
    let mut probe = Decoder::new(Arc::clone(spec), start_offset);
    Ok(probe.decode(&[], 0, false)?.0)
}

// --- Worker thread ----------------------------------------------------------

enum WorkerMsg {
    Field(WriteableField),
    /// Sent once, as soon as the decoder has locked on far enough to report it, so
    /// the orchestrator can write metadata into the JSON while decoding continues.
    Meta(DecoderMetadata),
    /// Sent once on natural end-of-input: the decoder's final metadata, or an
    /// error if the decode failed. Stopped workers send nothing.
    Done(Result<Option<DecoderMetadata>>),
}

/// Decode forward from `start_offset` to end-of-input, sending each field over
/// `tx`. Input comes from the shared `tape`, read once and only forward, so this
/// works on non-seekable inputs. `start_needed` is the first sample the decoder
/// reads (see [`first_needed_offset`]); feeding the window from there reproduces
/// the single-threaded byte stream without touching earlier input. Returns the
/// final metadata; `Ok(None)` if asked to stop early.
fn decode_segment(
    spec: &Arc<DecoderSpec>,
    tape: &Tape,
    start_offset: u64,
    start_needed: u64,
    tx: &SyncSender<WorkerMsg>,
    stop: &AtomicBool,
) -> Result<Option<DecoderMetadata>> {
    let mut decoder = Decoder::new(Arc::clone(spec), start_offset);
    let chunk = spec.readlen() + 4 * BLOCKSIZE;
    let mut window: Vec<f32> = Vec::new();
    let mut read_buffer = vec![0.0f32; chunk];
    let mut base: u64 = start_needed;
    let mut meta_sent = false;

    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let frontier = base + window.len() as u64;
        let Some((read, final_chunk)) = tape.read_into(frontier, &mut read_buffer, stop)? else {
            return Ok(None);
        };
        window.extend_from_slice(&read_buffer[..read]);

        let (consumed, fields) = decoder.decode(&window, base, final_chunk)?;
        if !meta_sent {
            if let Some(metadata) = decoder.metadata() {
                if tx.send(WorkerMsg::Meta(metadata)).is_err() {
                    return Ok(None);
                }
                meta_sent = true;
            }
        }
        for field in fields {
            if stop.load(Ordering::Relaxed) {
                return Ok(None);
            }
            if tx.send(WorkerMsg::Field(field)).is_err() {
                // The orchestrator dropped our receiver: it no longer wants us.
                return Ok(None);
            }
        }

        if final_chunk {
            break;
        }
        if consumed > base + window.len() as u64 {
            // The decoder skipped past everything buffered; resume at `consumed`.
            window.clear();
        } else {
            window.drain(..(consumed - base) as usize);
        }
        base = consumed;
    }

    Ok(decoder.metadata())
}

/// Orchestrator-side handle for one worker thread, with a small look-ahead
/// buffer so fields can be peeked (for `file_loc` alignment) before being taken.
struct Worker {
    rx: Receiver<WorkerMsg>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    buf: VecDeque<WriteableField>,
    finished: bool,
    produced_any: bool,
    outcome: Option<Result<Option<DecoderMetadata>>>,
    /// This worker's decoder metadata, once it has reported it (see
    /// [`WorkerMsg::Meta`]); used to keep the JSON sidecar's metadata populated.
    metadata: Option<DecoderMetadata>,
    /// First input sample this worker reads (see [`first_needed_offset`]); the
    /// orchestrator uses the lowest live worker's value as the tape drop point.
    start_needed: u64,
}

impl Worker {
    fn spawn(
        spec: &Arc<DecoderSpec>,
        tape: &Arc<Tape>,
        start_offset: u64,
        start_needed: u64,
        capacity: usize,
    ) -> Self {
        let (tx, rx) = sync_channel(capacity);
        let stop = Arc::new(AtomicBool::new(false));
        let spec = Arc::clone(spec);
        let tape = Arc::clone(tape);
        let worker_stop = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            let outcome =
                decode_segment(&spec, &tape, start_offset, start_needed, &tx, &worker_stop);
            // If we were stopped, the receiver is gone and the result is moot.
            if !worker_stop.load(Ordering::Relaxed) {
                let _ = tx.send(WorkerMsg::Done(outcome));
            }
        });
        Self {
            rx,
            stop,
            handle: Some(handle),
            buf: VecDeque::new(),
            finished: false,
            produced_any: false,
            outcome: None,
            metadata: None,
            start_needed,
        }
    }

    /// Ensure at least one field is buffered, blocking on the worker if needed.
    /// Returns false once the worker has finished with nothing more to give.
    fn fill(&mut self) -> bool {
        if !self.buf.is_empty() {
            return true;
        }
        if self.finished {
            return false;
        }
        loop {
            match self.rx.recv() {
                Ok(WorkerMsg::Field(field)) => {
                    self.produced_any = true;
                    self.buf.push_back(field);
                    return true;
                }
                // Metadata arrives out of band; record it and keep waiting for the
                // field this call promised to buffer.
                Ok(WorkerMsg::Meta(metadata)) => {
                    self.metadata = Some(metadata);
                }
                Ok(WorkerMsg::Done(result)) => {
                    self.outcome = Some(result);
                    self.finished = true;
                    return false;
                }
                // Channel closed without a Done (e.g. the thread panicked).
                Err(_) => {
                    self.finished = true;
                    return false;
                }
            }
        }
    }

    fn peek(&mut self) -> Option<&WriteableField> {
        if self.fill() {
            self.buf.front()
        } else {
            None
        }
    }

    fn pop(&mut self) -> Option<WriteableField> {
        if self.fill() {
            self.buf.pop_front()
        } else {
            None
        }
    }

    /// Signal the worker to stop and join it. Dropping the receiver also unblocks
    /// a worker parked on a full channel.
    fn shutdown(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let handle = self.handle.take();
        // Dropping the receiver closes the channel, unblocking a worker parked on
        // a full send so the join below cannot hang.
        drop(self.rx);
        if let Some(handle) = handle {
            let _ = handle.join();
        }
    }
}

// --- Orchestrator -----------------------------------------------------------

/// Stitches the output of several [`Worker`]s into a single in-order field
/// stream written through `writer`.
struct MtOrchestrator<'a> {
    writer: &'a mut DecodeWriter,
    spec: Arc<DecoderSpec>,
    /// Shared single-pass view of the input, streamed once to all workers.
    tape: Arc<Tape>,
    mt: MtParams,
    /// Absolute sample offset where decoding begins (honors `--start-fileloc`).
    global_base: u64,
    /// Samples per field, for translating field distances into seek offsets.
    spf: u64,
    capacity: usize,
    /// Active workers keyed by segment index.
    pool: HashMap<u64, Worker>,
    /// Lowest segment index not yet assigned to a worker.
    next_unassigned: u64,
    /// Fields committed to the output so far (the running global sequence count).
    committed: usize,
    /// Decoder metadata to embed in the JSON sidecar while decoding runs, taken
    /// from the first worker to report it (it is the same for every worker). The
    /// final, authoritative metadata is written from `final_metadata` at close.
    metadata: Option<DecoderMetadata>,
    final_metadata: Option<DecoderMetadata>,
}

impl<'a> MtOrchestrator<'a> {
    /// Absolute start sample offset of a segment.
    fn seg_start(&self, seg: u64) -> u64 {
        self.global_base
            + seg
                .saturating_mul(self.mt.distance_size)
                .saturating_mul(self.spf)
    }

    /// Keep up to `threads` workers decoding ahead, spawning the lowest
    /// unassigned segments. The input length is unknown until the source hits
    /// EOF; once known, segments past it are skipped (a worker started past EOF
    /// would simply produce nothing, so this is only an optimization).
    fn refill(&mut self) -> Result<()> {
        while self.pool.len() < self.mt.threads {
            let seg = self.next_unassigned;
            if let Some(len) = self.tape.known_len() {
                if self.seg_start(seg) >= len {
                    break;
                }
            }
            let start_offset = self.seg_start(seg);
            let start_needed = first_needed_offset(&self.spec, start_offset)?;
            let worker = Worker::spawn(
                &self.spec,
                &self.tape,
                start_offset,
                start_needed,
                self.capacity,
            );
            self.pool.insert(seg, worker);
            self.next_unassigned += 1;
        }
        Ok(())
    }

    /// Tell the tape that no decoder needs input before segment `seg`'s start,
    /// so the buffered prefix below it can be dropped. Call only after any
    /// lower-numbered worker has been shut down (joined), since the threshold
    /// must never pass a worker that is still reading.
    fn mark_drop_to(&self, seg: u64) {
        if let Some(worker) = self.pool.get(&seg) {
            self.tape.set_drop_threshold(worker.start_needed);
        }
    }

    /// Take a finished worker's outcome: surface a decode error, or capture the
    /// metadata of the decoder that reached this part of the file.
    fn absorb_outcome(&mut self, seg: u64) -> Result<()> {
        let outcome = self.pool.get_mut(&seg).and_then(|w| w.outcome.take());
        match outcome {
            Some(Ok(Some(metadata))) => self.final_metadata = Some(metadata),
            Some(Ok(None)) | None => {}
            Some(Err(error)) => return Err(error),
        }
        Ok(())
    }

    /// Renumber a field's sequence-derived fields against the global position and
    /// write it out.
    fn commit(&mut self, mut field: WriteableField) -> Result<()> {
        let global_seq = self.committed + 1;
        field.info.seq_no = global_seq;
        field.info.field_phase_id = field.field_phase_id_raw.unwrap_or_else(|| {
            field_phase_id(
                field.info.is_first_field,
                (global_seq / 2).is_multiple_of(2),
            )
        });
        if self.metadata.is_none() {
            self.metadata = self
                .pool
                .values()
                .find_map(|worker| worker.metadata.clone());
        }
        self.writer
            .write_writeable(&field, self.metadata.as_ref())?;
        self.committed += 1;
        Ok(())
    }

    fn shutdown_worker(&mut self, seg: u64) {
        if let Some(worker) = self.pool.remove(&seg) {
            worker.shutdown();
        }
    }

    /// Advance `next`'s buffer to the field aligned with `file_loc` and report
    /// whether it matches `current_field`. The aligned field is consumed so the
    /// later decoder, if it takes over, resumes immediately after it.
    fn compare_with_next(&mut self, next_seg: u64, current_field: &WriteableField) -> bool {
        let tol = self.spf / 2;
        let file_loc = current_field.info.file_loc;
        let aligned = {
            let Some(worker) = self.pool.get_mut(&next_seg) else {
                return false;
            };
            // Drop any of `next`'s fields that fall before this one (extra fields
            // it produced while still locking on).
            while worker
                .peek()
                .is_some_and(|g| g.info.file_loc + tol < file_loc)
            {
                worker.pop();
            }
            match worker.peek() {
                Some(g) if g.info.file_loc.abs_diff(file_loc) <= tol => worker.pop().unwrap(),
                // `next` has no field at this location yet (or is finished).
                _ => return false,
            }
        };
        fields_match(current_field, &aligned, &self.mt)
    }

    /// Run the decode, always tearing down workers before returning (even on
    /// error), and writing the output only on success.
    fn run(&mut self) -> Result<()> {
        let result = self.decode_loop();
        // Tear down every still-running worker regardless of outcome.
        for (_, worker) in std::mem::take(&mut self.pool) {
            worker.shutdown();
        }
        result?;
        self.write_output()
    }

    /// The stitching loop. Commits fields in order, leaving `final_metadata` set
    /// to the metadata of whichever decoder reached end of input.
    fn decode_loop(&mut self) -> Result<()> {
        self.refill()?;
        let mut current_seg = 0u64;
        if !self.pool.contains_key(&current_seg) {
            // Nothing decodable (input shorter than the start offset).
            return Ok(());
        }
        let mut next_seg = 1u64;
        // The lowest live segment fixes how far the tape may drop its prefix.
        self.mark_drop_to(current_seg);

        loop {
            let region_start = self.seg_start(next_seg);
            let region_end = self.seg_start(next_seg + 1);

            // No further segment exists: the current decoder is the final
            // authority; drain it to end of input.
            if !self.pool.contains_key(&next_seg) {
                while let Some(field) = self.pool.get_mut(&current_seg).unwrap().pop() {
                    self.commit(field)?;
                }
                self.absorb_outcome(current_seg)?;
                return Ok(());
            }

            // Phase 1: commit the current decoder's fields up to the next
            // decoder's region.
            loop {
                let reached = match self.pool.get_mut(&current_seg).unwrap().peek() {
                    Some(field) => field.info.file_loc >= region_start,
                    // Current reached end of input before the next region; it is
                    // the final authority.
                    None => {
                        self.absorb_outcome(current_seg)?;
                        return Ok(());
                    }
                };
                if reached {
                    break;
                }
                let field = self.pool.get_mut(&current_seg).unwrap().pop().unwrap();
                self.commit(field)?;
            }

            // Phase 2: in the overlap window, commit the current decoder's fields
            // while comparing against the next decoder, until `overlap_count`
            // consecutive fields match (stitch) or the window is exhausted.
            let mut run = 0usize;
            let mut stitched = false;
            loop {
                let in_window = match self.pool.get_mut(&current_seg).unwrap().peek() {
                    Some(field) => field.info.file_loc < region_end,
                    None => {
                        self.absorb_outcome(current_seg)?;
                        return Ok(());
                    }
                };
                if !in_window {
                    break;
                }
                let field = self.pool.get_mut(&current_seg).unwrap().pop().unwrap();
                let matched = self.compare_with_next(next_seg, &field);
                self.commit(field)?;
                if matched {
                    run += 1;
                    if run >= self.mt.overlap_count {
                        stitched = true;
                        break;
                    }
                } else {
                    run = 0;
                }
            }

            if stitched {
                // The next decoder, already past the matched fields, takes over.
                // Shut down (join) the old decoder before advancing the drop
                // point past it, so the tape never drops input it is still reading.
                self.shutdown_worker(current_seg);
                current_seg = next_seg;
                next_seg += 1;
                self.mark_drop_to(current_seg);
            } else {
                // The whole overlap disagreed: the earlier decoder stays
                // authoritative and the later decoder's work is thrown away.
                let produced = self.pool.get(&next_seg).is_some_and(|w| w.produced_any);
                let start = self.seg_start(next_seg);
                self.shutdown_worker(next_seg);
                if produced {
                    tracing::warn!(
                        segment_start_sample = start,
                        "Discarded a thread's work: no {}-field match found within {} overlapping fields; the earlier thread remains authoritative",
                        self.mt.overlap_count,
                        self.mt.distance_size,
                    );
                }
                next_seg += 1;
            }
            self.refill()?;
        }
    }

    fn write_output(&mut self) -> Result<()> {
        if self.committed != 0 {
            tracing::info!("Completed: saving JSON and exiting.");
        } else {
            tracing::info!("Completed without handling any frames.");
        }
        self.writer.close(self.final_metadata.take())?;
        Ok(())
    }
}

/// Multithreaded counterpart to [`decode_all`]. Requires `mt.threads >= 1`;
/// `mt.threads == 0` is handled by the serial path. `start_offset` is the
/// absolute sample where decoding begins (`--start-fileloc`, 0 by default).
///
/// The input is streamed once through a shared [`Tape`]; the workers never
/// reopen, seek, or stat it, so this runs on non-seekable inputs such as pipes.
/// The decoders themselves skip past the input before `start_offset`, so
/// reading still begins at the stream's start.
pub fn decode_all_mt(
    reader: DecodeReader,
    writer: &mut DecodeWriter,
    spec: Arc<DecoderSpec>,
    mt: MtParams,
    start_offset: u64,
) -> Result<()> {
    let tape = Arc::new(Tape::new(reader));
    let spf = spec.samples_per_field();
    // A worker only needs to bank its own segment plus the handful of fields it
    // overlaps into the next segment to hand off (a one-field warm-up plus
    // `overlap_count` matches). Anything decoded beyond that is thrown away when
    // the worker is stitched out, so the buffer is kept tight to avoid wasted
    // decode work; a small margin keeps the hand-off from stalling on a live
    // decode. The cap bounds memory when --mt-distance-size is large; correctness
    // is unaffected, as workers stream further fields on demand. It also caps how
    // far ahead a worker reads the shared tape, bounding the tape's buffered span.
    const HANDOFF_MARGIN: usize = 3;
    const MAX_BUFFERED_FIELDS: usize = 256;
    let capacity = (mt.distance_size as usize)
        .saturating_add(mt.overlap_count)
        .saturating_add(HANDOFF_MARGIN)
        .clamp(4, MAX_BUFFERED_FIELDS);

    let mut orchestrator = MtOrchestrator {
        writer,
        spec,
        tape,
        mt,
        global_base: start_offset,
        spf,
        capacity,
        pool: HashMap::new(),
        next_unassigned: 0,
        committed: 0,
        metadata: None,
        final_metadata: None,
    };
    orchestrator.run()
}
