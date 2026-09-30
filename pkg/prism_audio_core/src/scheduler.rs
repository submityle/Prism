//! Sample-accurate event scheduling and named musical clocks.
//!
//! The [`Transport`](crate::time::Transport) answers "where are we?"; this
//! module answers "when should this happen, exactly?". It provides two
//! primitives that together give the engine UE-Quartz-class timing:
//!
//! - [`NamedClock`] — an independent tempo grid (its own BPM, time signature,
//!   and sample origin) that can [`quantize`](NamedClock::quantize) an
//!   arbitrary sample position up to the next beat/bar/subdivision boundary.
//!   Several clocks can run concurrently (a 128 BPM music clock next to an
//!   ambient pulse clock), each with its own grid.
//! - [`EventScheduler`] — a bounded, allocation-free min-heap of timestamped
//!   payloads. Each block, [`drain_due`](EventScheduler::drain_due) yields the
//!   events that fall inside the block together with their exact frame offset,
//!   so voices and parameter changes start on the right sample rather than on
//!   the frame-rate-jittered game clock.
//!
//! # Real-time contract
//!
//! [`EventScheduler::schedule`], [`EventScheduler::drain_due`], and every
//! [`NamedClock`] query are allocation-free, lock-free, and panic-free: the
//! heap storage is reserved once at construction and `schedule` refuses (rather
//! than reallocates) once full. Both are therefore safe to call from the audio
//! callback thread.

use alloc::vec::Vec;

use crate::time::TimeSignature;

/// A musical quantization grid used by [`NamedClock::quantize`].
///
/// Boundaries are always measured relative to the clock's
/// [`sample_origin`](NamedClock::sample_origin), so restarting a clock realigns
/// every future boundary to the new origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Grid {
    /// No quantization: the target sample is returned unchanged.
    Immediate,
    /// Snap to the next beat boundary.
    Beat,
    /// Snap to the next bar boundary (beat count taken from the signature).
    Bar,
    /// Snap to the next `1/n`-of-a-beat subdivision (e.g. `Nth(4)` = 16th notes
    /// in 4/4). The subdivision count is clamped to at least one.
    Nth(u32),
}

/// An independent tempo grid with its own origin, tempo, and signature.
///
/// Unlike the global [`Transport`](crate::time::Transport), several clocks can
/// coexist, each quantizing events to a different musical grid. All arithmetic
/// is done in `f64` and rounded deterministically so boundaries are stable
/// across platforms.
#[derive(Debug, Clone, Copy)]
pub struct NamedClock {
    /// Sample rate in Hz (shared with the render context).
    sample_rate: u32,
    /// Absolute sample position this clock's grid is anchored to.
    sample_origin: u64,
    /// Tempo in beats per minute (clamped to a musical range).
    tempo_bpm: f32,
    /// Meter used to compute bar length from beat length.
    signature: TimeSignature,
}

impl NamedClock {
    /// Creates a clock anchored at `sample_origin`, running at `tempo_bpm` in
    /// the given `signature`.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is zero.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        sample_origin: u64,
        tempo_bpm: f32,
        signature: TimeSignature,
    ) -> Self {
        assert!(sample_rate > 0, "sample_rate must be non-zero");
        Self {
            sample_rate,
            sample_origin,
            tempo_bpm: tempo_bpm.clamp(1.0, 1000.0),
            signature,
        }
    }

    /// Returns the sample rate in Hz.
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the absolute sample position the grid is anchored to.
    #[inline]
    #[must_use]
    pub fn sample_origin(&self) -> u64 {
        self.sample_origin
    }

    /// Re-anchors the grid so future boundaries are measured from `origin`.
    #[inline]
    pub fn set_origin(&mut self, origin: u64) {
        self.sample_origin = origin;
    }

    /// Returns the tempo in beats per minute.
    #[inline]
    #[must_use]
    pub fn tempo_bpm(&self) -> f32 {
        self.tempo_bpm
    }

    /// Sets the tempo in beats per minute (clamped to a musical range).
    #[inline]
    pub fn set_tempo_bpm(&mut self, bpm: f32) {
        self.tempo_bpm = bpm.clamp(1.0, 1000.0);
    }

    /// Returns the time signature.
    #[inline]
    #[must_use]
    pub fn signature(&self) -> TimeSignature {
        self.signature
    }

    /// Sets the time signature.
    #[inline]
    pub fn set_signature(&mut self, signature: TimeSignature) {
        self.signature = signature;
    }

    /// Samples in one beat at the current tempo.
    #[inline]
    #[must_use]
    pub fn samples_per_beat(&self) -> f64 {
        (f64::from(self.sample_rate) * 60.0) / f64::from(self.tempo_bpm)
    }

    /// Samples in one bar at the current tempo and signature.
    #[inline]
    #[must_use]
    pub fn samples_per_bar(&self) -> f64 {
        self.samples_per_beat() * f64::from(self.signature.beats_per_bar)
    }

    /// Length in samples of one `grid` step, or `None` for
    /// [`Grid::Immediate`].
    #[must_use]
    fn grid_samples(&self, grid: Grid) -> Option<f64> {
        match grid {
            Grid::Immediate => None,
            Grid::Beat => Some(self.samples_per_beat()),
            Grid::Bar => Some(self.samples_per_bar()),
            Grid::Nth(n) => Some(self.samples_per_beat() / f64::from(n.max(1))),
        }
    }

    /// Returns the absolute sample position of the next `grid` boundary at or
    /// after `target_sample`.
    ///
    /// Targets before the clock origin snap to the origin. [`Grid::Immediate`]
    /// returns `target_sample` unchanged.
    #[must_use]
    pub fn quantize(&self, target_sample: u64, grid: Grid) -> u64 {
        let Some(step) = self.grid_samples(grid) else {
            return target_sample;
        };
        if step <= 0.0 || target_sample <= self.sample_origin {
            return self.sample_origin.max(target_sample.min(self.sample_origin));
        }
        // Relative position from the grid origin, rounded up to the next step.
        #[expect(
            clippy::cast_precision_loss,
            reason = "sample counts stay well within f64's 53-bit exact integer range for any realistic session length"
        )]
        let rel = (target_sample - self.sample_origin) as f64;
        let steps = libm::ceil(rel / step);
        let boundary = libm::round(steps * step);
        #[expect(
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            reason = "boundary is non-negative and bounded by realistic session length"
        )]
        let offset = boundary as u64;
        self.sample_origin + offset
    }
}

/// A single timestamped payload held by an [`EventScheduler`].
#[derive(Debug, Clone, Copy)]
struct Slot<E> {
    /// Absolute sample position at which the event fires.
    at_sample: u64,
    /// Monotonic sequence number: FIFO tie-breaker for equal timestamps so
    /// dispatch order is deterministic.
    seq: u64,
    /// The user payload delivered when the event fires.
    payload: E,
}

/// A bounded, allocation-free min-heap of sample-timestamped events.
///
/// The heap is ordered by `(at_sample, seq)` so events fire in timestamp order
/// and, for identical timestamps, in the order they were scheduled. Storage is
/// reserved once at construction; [`schedule`](Self::schedule) returns the
/// payload back to the caller instead of growing when the heap is full, keeping
/// the audio thread allocation-free.
///
/// `E` is the caller's event payload (a note trigger, a parameter change, a
/// voice-start command, ...).
pub struct EventScheduler<E> {
    /// Backing storage, kept at `len <= capacity` so pushes never reallocate.
    heap: Vec<Slot<E>>,
    /// Reserved capacity; `schedule` refuses beyond this.
    capacity: usize,
    /// Next sequence number handed out to break timestamp ties.
    next_seq: u64,
}

impl<E> EventScheduler<E> {
    /// Creates a scheduler that can hold up to `capacity` pending events.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: Vec::with_capacity(capacity),
            capacity,
            next_seq: 0,
        }
    }

    /// Returns the number of pending events.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` if no events are pending.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Returns the maximum number of pending events the heap can hold.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Schedules `payload` to fire at absolute sample `at_sample`.
    ///
    /// # Errors
    ///
    /// Returns `Err(payload)` (handing the payload back) if the heap is already
    /// at capacity, so the real-time thread never allocates.
    pub fn schedule(&mut self, at_sample: u64, payload: E) -> Result<(), E> {
        if self.heap.len() >= self.capacity {
            return Err(payload);
        }
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        self.heap.push(Slot {
            at_sample,
            seq,
            payload,
        });
        self.sift_up(self.heap.len() - 1);
        Ok(())
    }

    /// Absolute sample of the earliest pending event, if any.
    #[inline]
    #[must_use]
    pub fn peek_at(&self) -> Option<u64> {
        self.heap.first().map(|s| s.at_sample)
    }

    /// Removes every pending event whose timestamp falls in
    /// `[block_start, block_start + block_len)` and invokes `f(frame_offset,
    /// payload)` in ascending `(at_sample, seq)` order.
    ///
    /// `frame_offset` is the event's position within the block in `[0,
    /// block_len)`; events that were due before `block_start` (late) are
    /// clamped to offset `0` so they still fire this block rather than being
    /// dropped. This method performs no allocation.
    pub fn drain_due<F>(&mut self, block_start: u64, block_len: usize, mut f: F)
    where
        F: FnMut(usize, E),
    {
        let block_end = block_start.saturating_add(block_len as u64);
        while let Some(slot) = self.heap.first() {
            if slot.at_sample >= block_end {
                break;
            }
            let Some(slot) = self.pop_min() else { break };
            let offset = slot
                .at_sample
                .saturating_sub(block_start)
                .min(block_len.saturating_sub(1) as u64);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "offset is clamped below block_len (a usize) so the cast is exact"
            )]
            f(offset as usize, slot.payload);
        }
    }

    /// Drops all pending events without firing them.
    #[inline]
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Removes and returns the minimum element, or `None` when empty.
    ///
    /// Panic-free (RT-safe): an empty heap short-circuits via `checked_sub`
    /// instead of underflowing, so no path can panic on the audio thread.
    fn pop_min(&mut self) -> Option<Slot<E>> {
        let last = self.heap.len().checked_sub(1)?;
        self.heap.swap(0, last);
        let min = self.heap.pop();
        if !self.heap.is_empty() {
            self.sift_down(0);
        }
        min
    }

    /// Restores the heap invariant upward from `index` after a push.
    fn sift_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if Self::less(&self.heap[index], &self.heap[parent]) {
                self.heap.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    /// Restores the heap invariant downward from `index` after a pop.
    fn sift_down(&mut self, mut index: usize) {
        let len = self.heap.len();
        loop {
            let left = index * 2 + 1;
            let right = left + 1;
            let mut smallest = index;
            if left < len && Self::less(&self.heap[left], &self.heap[smallest]) {
                smallest = left;
            }
            if right < len && Self::less(&self.heap[right], &self.heap[smallest]) {
                smallest = right;
            }
            if smallest == index {
                break;
            }
            self.heap.swap(index, smallest);
            index = smallest;
        }
    }

    /// Strict `(at_sample, seq)` ordering used to keep the min-heap stable.
    #[inline]
    fn less(a: &Slot<E>, b: &Slot<E>) -> bool {
        (a.at_sample, a.seq) < (b.at_sample, b.seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig() -> TimeSignature {
        TimeSignature::default()
    }

    #[test]
    fn quantize_beat_and_bar() {
        // 120 BPM @ 48k => 24000 samples/beat, 96000 samples/bar.
        let clock = NamedClock::new(48_000, 0, 120.0, sig());
        assert!((clock.samples_per_beat() - 24_000.0).abs() < 1e-6);
        assert_eq!(clock.quantize(0, Grid::Beat), 0);
        assert_eq!(clock.quantize(1, Grid::Beat), 24_000);
        assert_eq!(clock.quantize(24_000, Grid::Beat), 24_000);
        assert_eq!(clock.quantize(24_001, Grid::Bar), 96_000);
        assert_eq!(clock.quantize(1, Grid::Nth(4)), 6_000); // 16th note grid
    }

    #[test]
    fn quantize_respects_origin() {
        let clock = NamedClock::new(48_000, 10_000, 120.0, sig());
        // Targets before the origin snap to the origin.
        assert_eq!(clock.quantize(0, Grid::Beat), 10_000);
        assert_eq!(clock.quantize(10_000, Grid::Beat), 10_000);
        assert_eq!(clock.quantize(10_001, Grid::Beat), 34_000);
    }

    #[test]
    fn immediate_grid_is_identity() {
        let clock = NamedClock::new(44_100, 5, 90.0, sig());
        assert_eq!(clock.quantize(12_345, Grid::Immediate), 12_345);
    }

    #[test]
    fn scheduler_fires_in_timestamp_order() {
        let mut sched = EventScheduler::<u32>::with_capacity(8);
        sched.schedule(30, 3).unwrap();
        sched.schedule(10, 1).unwrap();
        sched.schedule(20, 2).unwrap();
        let mut fired = Vec::new();
        sched.drain_due(0, 64, |off, payload| fired.push((off, payload)));
        assert_eq!(fired, [(10, 1), (20, 2), (30, 3)]);
        assert!(sched.is_empty());
    }

    #[test]
    fn scheduler_ties_are_fifo() {
        let mut sched = EventScheduler::<char>::with_capacity(8);
        sched.schedule(5, 'a').unwrap();
        sched.schedule(5, 'b').unwrap();
        sched.schedule(5, 'c').unwrap();
        let mut order = Vec::new();
        sched.drain_due(0, 16, |_, c| order.push(c));
        assert_eq!(order, ['a', 'b', 'c']);
    }

    #[test]
    fn scheduler_only_drains_current_block() {
        let mut sched = EventScheduler::<u32>::with_capacity(8);
        sched.schedule(5, 1).unwrap();
        sched.schedule(100, 2).unwrap();
        let mut fired = Vec::new();
        sched.drain_due(0, 64, |off, p| fired.push((off, p)));
        assert_eq!(fired, [(5, 1)]);
        assert_eq!(sched.len(), 1);
        assert_eq!(sched.peek_at(), Some(100));
    }

    #[test]
    fn late_events_clamp_to_block_start() {
        let mut sched = EventScheduler::<u32>::with_capacity(4);
        // Scheduled before the block started: should still fire at offset 0.
        sched.schedule(50, 7).unwrap();
        let mut fired = Vec::new();
        sched.drain_due(100, 64, |off, p| fired.push((off, p)));
        assert_eq!(fired, [(0, 7)]);
    }

    #[test]
    fn scheduler_refuses_when_full() {
        let mut sched = EventScheduler::<u32>::with_capacity(2);
        assert!(sched.schedule(1, 1).is_ok());
        assert!(sched.schedule(2, 2).is_ok());
        assert_eq!(sched.schedule(3, 3), Err(3));
        assert_eq!(sched.len(), 2);
    }
}
