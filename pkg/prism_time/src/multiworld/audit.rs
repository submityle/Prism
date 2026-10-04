//! Deterministic audit: per-frame state hashing and double-run divergence
//! location.
//!
//! Hashing uses 64-bit FNV-1a — a small, allocation-free, fully deterministic
//! hash (no `RandomState`, no per-process seed) so two runs that compute the
//! same bytes produce the same digest on every platform. Record one digest per
//! frame into an [`AuditTrail`], then [`compare_trails`] two runs to find the
//! first frame whose state diverged.

use alloc::vec::Vec;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Hash a byte slice with 64-bit FNV-1a. Deterministic and `const`.
#[inline]
#[must_use]
pub const fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash
}

/// An incremental 64-bit FNV-1a hasher for composing a frame's key state.
///
/// Feed the deterministic state fields (tick, accumulator, step, scale bits,
/// pause flag, ...) in a fixed order, then read [`finish`](Self::finish). Field
/// order is part of the contract: both runs must write identically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateHasher {
    /// Running FNV-1a state.
    state: u64,
}

impl StateHasher {
    /// A hasher primed with the FNV-1a offset basis.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { state: FNV_OFFSET }
    }

    /// Fold one byte into the hash.
    #[inline]
    pub fn write_u8(&mut self, byte: u8) {
        self.state ^= byte as u64;
        self.state = self.state.wrapping_mul(FNV_PRIME);
    }

    /// Fold a byte slice into the hash.
    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u8(b);
        }
    }

    /// Fold a `u32` (little-endian) into the hash.
    #[inline]
    pub fn write_u32(&mut self, value: u32) {
        self.write_bytes(&value.to_le_bytes());
    }

    /// Fold a `u64` (little-endian) into the hash.
    #[inline]
    pub fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }

    /// Fold a `u128` (little-endian) into the hash.
    #[inline]
    pub fn write_u128(&mut self, value: u128) {
        self.write_bytes(&value.to_le_bytes());
    }

    /// Fold an `f64` by its raw bits (so `NaN`/`-0.0` hash deterministically).
    #[inline]
    pub fn write_f64_bits(&mut self, value: f64) {
        self.write_u64(value.to_bits());
    }

    /// The current digest.
    #[inline]
    #[must_use]
    pub const fn finish(&self) -> u64 {
        self.state
    }
}

impl Default for StateHasher {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// An ordered log of per-frame state digests for one run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditTrail {
    /// One digest per recorded frame, in frame order.
    digests: Vec<u64>,
}

impl AuditTrail {
    /// An empty trail.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { digests: Vec::new() }
    }

    /// Append one frame's digest.
    #[inline]
    pub fn record(&mut self, digest: u64) {
        self.digests.push(digest);
    }

    /// Number of recorded frames.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.digests.len()
    }

    /// Whether no frames have been recorded.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.digests.is_empty()
    }

    /// The recorded digests in frame order.
    #[inline]
    #[must_use]
    pub fn digests(&self) -> &[u64] {
        &self.digests
    }

    /// The most recently recorded digest, if any.
    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<u64> {
        self.digests.last().copied()
    }

    /// Drop all digests, keeping capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.digests.clear();
    }
}

impl Default for AuditTrail {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// The outcome of diffing two [`AuditTrail`]s with [`compare_trails`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditDiff {
    /// Both trails have equal length and every frame digest matches.
    Identical,
    /// A common frame diverged first.
    Diverged {
        /// Zero-based index of the first diverging frame.
        frame: usize,
        /// Digest from the left (first) trail at that frame.
        left: u64,
        /// Digest from the right (second) trail at that frame.
        right: u64,
    },
    /// Every common frame matched, but the trails have different lengths.
    LengthMismatch {
        /// Count of leading frames that matched (the common prefix length).
        matched: usize,
        /// Length of the left (first) trail.
        left_len: usize,
        /// Length of the right (second) trail.
        right_len: usize,
    },
}

impl AuditDiff {
    /// Whether the two runs are identical.
    #[inline]
    #[must_use]
    pub const fn is_identical(&self) -> bool {
        matches!(self, Self::Identical)
    }

    /// The first diverging frame index, if the runs diverged on a common frame.
    #[inline]
    #[must_use]
    pub const fn diverged_frame(&self) -> Option<usize> {
        match self {
            Self::Diverged { frame, .. } => Some(*frame),
            _ => None,
        }
    }
}

/// Diff two audit trails and locate the first point of divergence.
///
/// Scans the common prefix first: the earliest frame whose digests differ is
/// reported as [`AuditDiff::Diverged`]. If every common frame matched but the
/// trails differ in length, reports [`AuditDiff::LengthMismatch`]. Otherwise
/// [`AuditDiff::Identical`].
#[inline]
#[must_use]
pub fn compare_trails(left: &AuditTrail, right: &AuditTrail) -> AuditDiff {
    let la = left.digests.len();
    let lb = right.digests.len();
    let common = if la < lb { la } else { lb };
    let mut i = 0;
    while i < common {
        let l = left.digests[i];
        let r = right.digests[i];
        if l != r {
            return AuditDiff::Diverged {
                frame: i,
                left: l,
                right: r,
            };
        }
        i += 1;
    }
    if la != lb {
        return AuditDiff::LengthMismatch {
            matched: common,
            left_len: la,
            right_len: lb,
        };
    }
    AuditDiff::Identical
}
