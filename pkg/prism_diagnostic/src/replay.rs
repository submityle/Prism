//! Deterministic replay markers (design §17 / §24.4, M6).
//!
//! Rollback-networking, recordings, and deterministic simulation share one
//! brutal failure mode: two runs that should be identical diverge. The fix is
//! to record a tagged marker stream per run — each marker pairs a frame index,
//! a stable label, and a state hash — then compare two runs marker-by-marker to
//! localize the *first* point of divergence (the desync root cause).
//!
//! This module is the recording + comparison core. Hashing of actual engine
//! state (ECS world, physics) lives in those crates; [`fnv1a_64`] is provided as
//! a small, fully specified, deterministic hash so callers that just need a
//! byte-stream digest have one without pulling a dependency.

extern crate alloc;

use alloc::vec::Vec;

/// One tagged point on a run's deterministic timeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayMarker {
    /// Frame index this marker was emitted on.
    pub frame: u64,
    /// Stable label identifying what was hashed (e.g. `"world"`, `"physics"`).
    pub label: &'static str,
    /// Deterministic state hash captured at this marker.
    pub hash: u64,
}

/// An ordered sequence of [`ReplayMarker`]s recorded over one run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplayTimeline {
    markers: Vec<ReplayMarker>,
}

impl ReplayTimeline {
    /// Create an empty timeline.
    pub fn new() -> Self {
        Self {
            markers: Vec::new(),
        }
    }

    /// Append a marker for `frame`/`label` carrying `hash`.
    pub fn mark(&mut self, frame: u64, label: &'static str, hash: u64) {
        self.markers.push(ReplayMarker { frame, label, hash });
    }

    /// Append a precomposed marker.
    pub fn push(&mut self, marker: ReplayMarker) {
        self.markers.push(marker);
    }

    /// Borrow the recorded markers in emission order.
    pub fn markers(&self) -> &[ReplayMarker] {
        &self.markers
    }

    /// Number of recorded markers.
    pub fn len(&self) -> usize {
        self.markers.len()
    }

    /// Whether no markers have been recorded.
    pub fn is_empty(&self) -> bool {
        self.markers.is_empty()
    }

    /// Compare this timeline (the reference run) against `other` (the replay
    /// run). See [`compare_timelines`].
    pub fn diff(&self, other: &ReplayTimeline) -> ReplayDivergence {
        compare_timelines(self, other)
    }
}

/// The result of comparing two [`ReplayTimeline`]s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayDivergence {
    /// The timelines are identical (same length, same markers).
    Match,
    /// The markers differ at `index`; `left`/`right` are the mismatched pair.
    /// This is the first divergent frame — the desync root cause.
    Diverged {
        /// Marker index of the first mismatch.
        index: usize,
        /// Marker from the reference run.
        left: ReplayMarker,
        /// Marker from the replay run.
        right: ReplayMarker,
    },
    /// Every marker over the common prefix matched, but the runs recorded
    /// different numbers of markers (one ran longer / stopped early).
    LengthMismatch {
        /// Number of markers that matched before the shorter run ended.
        common: usize,
        /// Reference-run marker count.
        left_len: usize,
        /// Replay-run marker count.
        right_len: usize,
    },
}

impl ReplayDivergence {
    /// Whether the two runs matched exactly.
    pub fn is_match(&self) -> bool {
        matches!(self, ReplayDivergence::Match)
    }

    /// The marker index of the first divergence, if any (both a content
    /// mismatch and a length mismatch report where comparison stopped).
    pub fn divergence_index(&self) -> Option<usize> {
        match self {
            ReplayDivergence::Match => None,
            ReplayDivergence::Diverged { index, .. } => Some(*index),
            ReplayDivergence::LengthMismatch { common, .. } => Some(*common),
        }
    }
}

/// Compare two timelines marker-by-marker and report the first divergence.
///
/// Two markers match when their `label` and `hash` are both equal. The `frame`
/// field is informational and not part of the match test, so a run that drops a
/// frame still surfaces as a label/hash mismatch at that index rather than
/// silently realigning.
pub fn compare_timelines(left: &ReplayTimeline, right: &ReplayTimeline) -> ReplayDivergence {
    let common = left.markers.len().min(right.markers.len());
    for index in 0..common {
        let l = left.markers[index];
        let r = right.markers[index];
        if l.label != r.label || l.hash != r.hash {
            return ReplayDivergence::Diverged {
                index,
                left: l,
                right: r,
            };
        }
    }
    if left.markers.len() != right.markers.len() {
        return ReplayDivergence::LengthMismatch {
            common,
            left_len: left.markers.len(),
            right_len: right.markers.len(),
        };
    }
    ReplayDivergence::Match
}

/// A fully specified `FNV`-1a 64-bit hash over `bytes`.
///
/// `FNV`-1a is deterministic and endianness-independent (it consumes a byte
/// stream), making it a sound building block for cross-run / cross-platform
/// state hashing. It is not cryptographic.
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}
