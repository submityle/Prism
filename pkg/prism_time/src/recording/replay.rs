//! [`Player`]: the **playback** state of the record/replay system.
//!
//! A player owns a [`Recording`] and a cursor. [`next_frame`](Player::next_frame)
//! yields the frames in order — the "frame feeder": the caller takes each
//! frame's `dt` and feeds it to the clock instead of reading the wall clock, so
//! the replay reproduces the original run exactly. [`seek`](Player::seek) and
//! [`reset`](Player::reset) allow scrubbing for frame-by-frame debugging.

use crate::recording::{RecordedFrame, Recording};

/// Walks a [`Recording`] one frame at a time (the playback state).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Player<I> {
    recording: Recording<I>,
    cursor: usize,
}

impl<I> Player<I> {
    /// Start a player at the beginning of `recording`.
    #[inline]
    #[must_use]
    pub fn new(recording: Recording<I>) -> Self {
        Self {
            recording,
            cursor: 0,
        }
    }

    /// Yield the next frame and advance the cursor, or `None` at the end.
    ///
    /// This is the frame feeder: drive the clock with the returned frame's
    /// `dt`, apply its `input`, and seed the frame RNG with its `seed`.
    #[inline]
    pub fn next_frame(&mut self) -> Option<&RecordedFrame<I>> {
        let frame = self.recording.frames().get(self.cursor)?;
        self.cursor += 1;
        Some(frame)
    }

    /// The frame the cursor points at without advancing, or `None` at the end.
    #[inline]
    #[must_use]
    pub fn peek(&self) -> Option<&RecordedFrame<I>> {
        self.recording.frame(self.cursor)
    }

    /// The next frame index to be returned.
    #[inline]
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Frames not yet returned.
    #[inline]
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.recording.len().saturating_sub(self.cursor)
    }

    /// Whether every frame has been returned.
    #[inline]
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.cursor >= self.recording.len()
    }

    /// Total number of frames in the recording.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.recording.len()
    }

    /// Whether the backing recording has no frames.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.recording.is_empty()
    }

    /// Rewind to the first frame.
    #[inline]
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Move the cursor to `index`, clamped to one past the last frame so the
    /// player cannot overrun.
    #[inline]
    pub fn seek(&mut self, index: usize) {
        self.cursor = index.min(self.recording.len());
    }

    /// Borrow the backing recording.
    #[inline]
    #[must_use]
    pub fn recording(&self) -> &Recording<I> {
        &self.recording
    }

    /// Consume the player and return its recording (e.g. to re-wrap in a fresh
    /// player for a second independent replay).
    #[inline]
    #[must_use]
    pub fn into_recording(self) -> Recording<I> {
        self.recording
    }
}
