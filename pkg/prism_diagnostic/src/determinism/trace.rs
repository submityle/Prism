//! Per-frame deterministic trace recording and double-run comparison (§24.4).
//!
//! The core anti-desync workflow: under a deterministic build, record one key
//! state digest per frame into a [`DeterminismTrace`], then compare two runs'
//! traces frame-by-frame. The earliest frame whose digest differs is the desync
//! root cause ([`TraceDiff::Diverged`]). A trace that stops early surfaces as
//! [`TraceDiff::LengthMismatch`]; two identical runs report
//! [`TraceDiff::Identical`].
//!
//! Each recorded [`FrameHash`] carries both the caller-assigned `frame` number
//! and the state `hash`. Comparison matches by position *and* value: two frames
//! match only when their `frame` and `hash` are both equal, so a run that drops
//! or renumbers a frame surfaces as a divergence at that position rather than
//! silently realigning.
//!
//! Everything here is pure `core`/`alloc` integer arithmetic — deterministic,
//! `no_std` + `alloc`, no `unsafe`.

extern crate alloc;

use alloc::vec::Vec;

/// One frame's recorded state digest: the caller-assigned frame number plus the
/// deterministic hash of that frame's key state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameHash {
    /// Frame index this digest was recorded on (caller-assigned, informational
    /// but also part of the match test).
    pub frame: u64,
    /// Deterministic key-state digest for this frame.
    pub hash: u64,
}

impl FrameHash {
    /// Construct a frame digest.
    #[inline]
    #[must_use]
    pub const fn new(frame: u64, hash: u64) -> Self {
        Self { frame, hash }
    }
}

/// An ordered log of per-frame state digests for one run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeterminismTrace {
    /// One digest per recorded frame, in recording order.
    frames: Vec<FrameHash>,
}

impl DeterminismTrace {
    /// An empty trace.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Record a digest for `frame`.
    #[inline]
    pub fn record(&mut self, frame: u64, hash: u64) {
        self.frames.push(FrameHash::new(frame, hash));
    }

    /// Append a precomposed frame digest.
    #[inline]
    pub fn push(&mut self, frame: FrameHash) {
        self.frames.push(frame);
    }

    /// Borrow the recorded frame digests in recording order.
    #[inline]
    #[must_use]
    pub fn frames(&self) -> &[FrameHash] {
        &self.frames
    }

    /// Number of recorded frames.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether no frames have been recorded.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The most recently recorded frame digest, if any.
    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<FrameHash> {
        self.frames.last().copied()
    }

    /// Drop all recorded frames, keeping capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Compare this trace (the reference run) against `other` (the replay run)
    /// frame-by-frame. See [`compare`].
    #[inline]
    #[must_use]
    pub fn compare(&self, other: &DeterminismTrace) -> TraceDiff {
        compare(self, other)
    }
}

/// The outcome of comparing two [`DeterminismTrace`]s with [`compare`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceDiff {
    /// Both traces have equal length and every frame digest matches.
    Identical,
    /// A common frame diverged first — the desync root cause.
    Diverged {
        /// Zero-based position of the first diverging frame.
        frame: usize,
        /// The reference run's digest at that position.
        left: FrameHash,
        /// The replay run's digest at that position.
        right: FrameHash,
    },
    /// Every common frame matched, but the traces have different lengths (one
    /// run stopped early or ran longer).
    LengthMismatch {
        /// Count of leading frames that matched (the common prefix length).
        matched: usize,
        /// Length of the left (reference) trace.
        left_len: usize,
        /// Length of the right (replay) trace.
        right_len: usize,
    },
}

impl TraceDiff {
    /// Whether the two runs are identical.
    #[inline]
    #[must_use]
    pub const fn is_identical(&self) -> bool {
        matches!(self, Self::Identical)
    }

    /// The zero-based position of the first diverging frame, if the runs
    /// diverged on a common frame.
    #[inline]
    #[must_use]
    pub const fn diverged_frame(&self) -> Option<usize> {
        match self {
            Self::Diverged { frame, .. } => Some(*frame),
            _ => None,
        }
    }

    /// The position where comparison stopped making progress, if the runs are
    /// not identical: the diverging position for [`TraceDiff::Diverged`] and
    /// the common-prefix length for [`TraceDiff::LengthMismatch`].
    #[inline]
    #[must_use]
    pub const fn divergence_position(&self) -> Option<usize> {
        match self {
            Self::Identical => None,
            Self::Diverged { frame, .. } => Some(*frame),
            Self::LengthMismatch { matched, .. } => Some(*matched),
        }
    }
}

/// Compare two traces frame-by-frame and report the first divergence.
///
/// Scans the common prefix first: the earliest position whose [`FrameHash`]
/// differs (by frame number or hash) is reported as [`TraceDiff::Diverged`]. If
/// every common frame matched but the traces differ in length, reports
/// [`TraceDiff::LengthMismatch`]. Otherwise [`TraceDiff::Identical`].
#[must_use]
pub fn compare(left: &DeterminismTrace, right: &DeterminismTrace) -> TraceDiff {
    let left_len = left.frames.len();
    let right_len = right.frames.len();
    let common = left_len.min(right_len);
    for index in 0..common {
        let l = left.frames[index];
        let r = right.frames[index];
        if l != r {
            return TraceDiff::Diverged {
                frame: index,
                left: l,
                right: r,
            };
        }
    }
    if left_len != right_len {
        return TraceDiff::LengthMismatch {
            matched: common,
            left_len,
            right_len,
        };
    }
    TraceDiff::Identical
}
