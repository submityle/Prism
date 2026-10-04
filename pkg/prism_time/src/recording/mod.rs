//! **§24.1 — input + time record / replay.** A deterministic frame timeline.
//!
//! This module records, for every frame, the exact triple the design doc calls
//! out — the time step (`dt`), the sampled input snapshot, and the random seed
//! used that frame — into an ordered [`Recording`]. Replaying feeds those
//! recorded deltas back into the clock instead of reading the wall clock, so a
//! run reproduces bit-for-bit: same input + same step + same seed => same
//! result (position-equal), closing the deterministic-debugging loop with the
//! [`determinism`](crate::determinism) tick clock and the
//! [`multiworld`](crate::multiworld) audit trail.
//!
//! The system has two states, each its own type:
//! - **record** — [`Recorder`] appends one [`RecordedFrame`] per frame and
//!   hands back a [`Recording`] with [`Recorder::finish`].
//! - **playback** — [`Player`] walks a [`Recording`] one frame at a time via
//!   [`Player::next_frame`], the "frame feeder" the clock is driven from.
//!
//! Everything here is pure data and pure integer/`Duration` arithmetic: no wall
//! clock, no allocation on the per-frame hot path beyond the recording's own
//! growth, and `no_std + alloc` friendly.

mod record;
mod replay;

pub use record::Recorder;
pub use replay::Player;

use crate::Duration;
use alloc::vec::Vec;

/// One recorded frame: the exact time step applied, the input snapshot sampled
/// that frame, and the random seed used to drive the frame's stochastic logic.
///
/// `I` is the caller's input type (a button mask, an analog-stick struct, a
/// serialized command list, ...). It only needs to be [`Clone`] to record and
/// replay; deriving richer bounds (`Copy`/`Eq`/`Hash`) follows `I`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecordedFrame<I> {
    /// Exact time step for this frame. On replay this is fed to the clock in
    /// place of a wall-clock delta, guaranteeing identical stepping.
    pub dt: Duration,
    /// Input snapshot sampled at the start of this frame.
    pub input: I,
    /// Random seed used to drive this frame's deterministic RNG.
    pub seed: u64,
}

impl<I> RecordedFrame<I> {
    /// Build a frame from its parts.
    #[inline]
    pub const fn new(dt: Duration, input: I, seed: u64) -> Self {
        Self { dt, input, seed }
    }
}

/// An ordered, immutable-by-convention timeline of [`RecordedFrame`]s.
///
/// A [`Recorder`] produces one; a [`Player`] consumes one. It is plain data and
/// [`Clone`], so the same recording can seed two independent replays (run A /
/// run B) for double-run determinism checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recording<I> {
    frames: Vec<RecordedFrame<I>>,
}

impl<I> Recording<I> {
    /// An empty recording.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Build a recording from pre-collected frames.
    #[inline]
    #[must_use]
    pub fn from_frames(frames: Vec<RecordedFrame<I>>) -> Self {
        Self { frames }
    }

    /// Append one frame to the end of the timeline.
    #[inline]
    pub fn push(&mut self, frame: RecordedFrame<I>) {
        self.frames.push(frame);
    }

    /// Number of recorded frames.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the recording has no frames.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The frame at `index`, if any.
    #[inline]
    #[must_use]
    pub fn frame(&self, index: usize) -> Option<&RecordedFrame<I>> {
        self.frames.get(index)
    }

    /// All recorded frames in order.
    #[inline]
    #[must_use]
    pub fn frames(&self) -> &[RecordedFrame<I>] {
        &self.frames
    }

    /// Sum of every frame's `dt`, saturating. The exact wall-clock length a
    /// replay reproduces.
    #[inline]
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.frames
            .iter()
            .fold(Duration::ZERO, |acc, f| acc.saturating_add(f.dt))
    }

    /// Drop all frames, keeping allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

impl<I> Default for Recording<I> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
