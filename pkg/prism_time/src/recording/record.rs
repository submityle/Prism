//! [`Recorder`]: the **record** state of the record/replay system.
//!
//! Call [`record_frame`](Recorder::record_frame) once per frame with the step,
//! the sampled input, and the frame's seed; the recorder appends them in order.
//! When the capture is done, [`finish`](Recorder::finish) yields the immutable
//! [`Recording`] a [`Player`](crate::recording::Player) replays.

use crate::recording::{RecordedFrame, Recording};
use crate::Duration;

/// Accumulates a [`Recording`] one frame at a time (the record state).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recorder<I> {
    recording: Recording<I>,
}

impl<I> Recorder<I> {
    /// A fresh recorder with an empty timeline.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            recording: Recording::new(),
        }
    }

    /// Record one frame from its parts (step, input snapshot, seed).
    #[inline]
    pub fn record_frame(&mut self, dt: Duration, input: I, seed: u64) {
        self.recording.push(RecordedFrame::new(dt, input, seed));
    }

    /// Record one already-built [`RecordedFrame`].
    #[inline]
    pub fn record(&mut self, frame: RecordedFrame<I>) {
        self.recording.push(frame);
    }

    /// Number of frames recorded so far.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.recording.len()
    }

    /// Whether nothing has been recorded yet.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.recording.is_empty()
    }

    /// Borrow the recording accumulated so far (without consuming the recorder).
    #[inline]
    #[must_use]
    pub fn recording(&self) -> &Recording<I> {
        &self.recording
    }

    /// Discard all recorded frames and start over.
    #[inline]
    pub fn clear(&mut self) {
        self.recording.clear();
    }

    /// Finish recording and take ownership of the immutable [`Recording`].
    #[inline]
    #[must_use]
    pub fn finish(self) -> Recording<I> {
        self.recording
    }
}

impl<I> Default for Recorder<I> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
