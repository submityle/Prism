//! Input + random-seed recording stream for precise replay reproduction
//! (§24.4).
//!
//! Rollback networking, recordings, and competitive-game replays all share one
//! requirement: capture exactly what fed the simulation each frame so a run can
//! be reproduced bit-for-bit later. For a deterministic simulation that is the
//! per-frame **input** plus the **random seed** consumed that frame — nothing
//! else needs to be stored, because the simulation re-derives all other state
//! from them.
//!
//! [`InputRecorder`] records one [`FrameInput`] per frame (frame number, a
//! deterministic hash of that frame's input, and the seed). To reproduce a bug,
//! replay the recorded stream back into the simulation in order via
//! [`InputReplay`]; feeding the identical input + seed sequence drives the
//! identical state, so the resulting [`DeterminismTrace`] matches the original.
//! [`InputRecorder::to_trace`] folds the recorded stream into such a trace so a
//! recording and its replay can be compared directly with
//! [`compare`](crate::determinism::compare).
//!
//! Everything here is pure `core`/`alloc` integer arithmetic — deterministic,
//! `no_std` + `alloc`, no `unsafe`.

extern crate alloc;

use alloc::vec::Vec;

use super::hash::StateHasher;
use super::trace::DeterminismTrace;

/// One recorded frame of simulation input: the frame number, a deterministic
/// digest of that frame's input, and the random seed consumed that frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameInput {
    /// Frame index this input was recorded on.
    pub frame: u64,
    /// Deterministic digest of the frame's raw input (buttons, axes, commands).
    pub input_hash: u64,
    /// Random seed consumed by the simulation on this frame.
    pub seed: u64,
}

impl FrameInput {
    /// Construct a recorded input frame.
    #[inline]
    #[must_use]
    pub const fn new(frame: u64, input_hash: u64, seed: u64) -> Self {
        Self {
            frame,
            input_hash,
            seed,
        }
    }

    /// Fold this frame into a single deterministic per-frame digest (frame
    /// number, input hash, and seed, in that fixed order).
    ///
    /// This is the digest [`InputRecorder::to_trace`] records per frame, so a
    /// replay that re-derives the same inputs produces an identical trace.
    #[inline]
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut hasher = StateHasher::new();
        hasher.write_u64(self.frame);
        hasher.write_u64(self.input_hash);
        hasher.write_u64(self.seed);
        hasher.finish()
    }
}

/// An ordered recording of per-frame input + seed for one run.
///
/// Frame numbers auto-increment from `0` as frames are recorded with
/// [`record`](Self::record); use [`push`](Self::push) to supply explicit frame
/// numbers (e.g. when recording a sub-range).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InputRecorder {
    /// Recorded frames, in recording order.
    entries: Vec<FrameInput>,
    /// Next auto-assigned frame number for [`record`](Self::record).
    next_frame: u64,
}

impl InputRecorder {
    /// An empty recorder starting at frame `0`.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_frame: 0,
        }
    }

    /// Deterministically hash a frame's raw input bytes.
    ///
    /// A thin wrapper over [`StateHasher`] so callers that hold the raw input
    /// as bytes get a stable digest without re-implementing the fold.
    #[inline]
    #[must_use]
    pub fn hash_input(bytes: &[u8]) -> u64 {
        let mut hasher = StateHasher::new();
        hasher.write_bytes(bytes);
        hasher.finish()
    }

    /// Record raw input bytes + `seed` for the next auto-numbered frame,
    /// returning the recorded [`FrameInput`].
    #[inline]
    pub fn record(&mut self, input: &[u8], seed: u64) -> FrameInput {
        self.record_hashed(Self::hash_input(input), seed)
    }

    /// Record a pre-hashed input digest + `seed` for the next auto-numbered
    /// frame, returning the recorded [`FrameInput`].
    #[inline]
    pub fn record_hashed(&mut self, input_hash: u64, seed: u64) -> FrameInput {
        let entry = FrameInput::new(self.next_frame, input_hash, seed);
        self.next_frame += 1;
        self.entries.push(entry);
        entry
    }

    /// Append a precomposed frame with an explicit frame number.
    ///
    /// The auto-numbering cursor advances to `frame + 1` so a following
    /// [`record`](Self::record) continues after this frame.
    #[inline]
    pub fn push(&mut self, entry: FrameInput) {
        self.next_frame = entry.frame + 1;
        self.entries.push(entry);
    }

    /// Borrow the recorded frames in recording order.
    #[inline]
    #[must_use]
    pub fn entries(&self) -> &[FrameInput] {
        &self.entries
    }

    /// Number of recorded frames.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no frames have been recorded.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The next frame number [`record`](Self::record) will assign.
    #[inline]
    #[must_use]
    pub fn next_frame(&self) -> u64 {
        self.next_frame
    }

    /// Drop all recorded frames and reset the auto-numbering cursor to `0`.
    #[inline]
    pub fn clear(&mut self) {
        self.entries.clear();
        self.next_frame = 0;
    }

    /// A single deterministic digest over the whole recorded stream.
    ///
    /// Two recordings with identical frame/input/seed sequences produce the
    /// same digest; any difference changes it. Useful as a cheap equality check
    /// before a full [`compare`](crate::determinism::compare).
    #[inline]
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut hasher = StateHasher::new();
        hasher.write_u64(self.entries.len() as u64);
        for entry in &self.entries {
            hasher.combine(entry.digest());
        }
        hasher.finish()
    }

    /// Fold the recorded stream into a per-frame [`DeterminismTrace`].
    ///
    /// Each recorded frame contributes one [`FrameHash`](super::trace::FrameHash)
    /// whose hash is [`FrameInput::digest`]. Replaying the same input + seed
    /// sequence reproduces this trace, so recording vs. replay can be compared
    /// with [`compare`](crate::determinism::compare).
    #[inline]
    #[must_use]
    pub fn to_trace(&self) -> DeterminismTrace {
        let mut trace = DeterminismTrace::new();
        for entry in &self.entries {
            trace.record(entry.frame, entry.digest());
        }
        trace
    }

    /// A replay cursor over the recorded stream for precise reproduction.
    #[inline]
    #[must_use]
    pub fn replay(&self) -> InputReplay<'_> {
        InputReplay {
            entries: &self.entries,
            cursor: 0,
        }
    }
}

/// A forward cursor over an [`InputRecorder`]'s recorded frames, feeding the
/// identical input + seed sequence back into a simulation for precise replay.
#[derive(Clone, Copy, Debug)]
pub struct InputReplay<'a> {
    /// Recorded frames being replayed.
    entries: &'a [FrameInput],
    /// Index of the next frame to yield.
    cursor: usize,
}

impl<'a> InputReplay<'a> {
    /// Yield the next recorded frame and advance, or `None` at the end.
    #[inline]
    pub fn next_frame(&mut self) -> Option<FrameInput> {
        let entry = self.entries.get(self.cursor).copied()?;
        self.cursor += 1;
        Some(entry)
    }

    /// Peek the next recorded frame without advancing.
    #[inline]
    #[must_use]
    pub fn peek(&self) -> Option<FrameInput> {
        self.entries.get(self.cursor).copied()
    }

    /// Number of frames not yet yielded.
    #[inline]
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.entries.len() - self.cursor
    }

    /// Whether every recorded frame has been yielded.
    #[inline]
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.cursor >= self.entries.len()
    }

    /// The index of the next frame to yield.
    #[inline]
    #[must_use]
    pub fn position(&self) -> usize {
        self.cursor
    }

    /// Rewind to the start so the stream can be replayed again.
    #[inline]
    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}

impl Iterator for InputReplay<'_> {
    type Item = FrameInput;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_frame()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for InputReplay<'_> {}
