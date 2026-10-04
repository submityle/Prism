//! A stable, fully specified 64-bit hash for deterministic state digests
//! (§24.4).
//!
//! Reproducing a desync requires that two runs computing the *same* bytes
//! produce the *same* digest on every platform and every process launch. The
//! standard-library hashers are unsuitable: `RandomState` seeds per process and
//! `Hash` layouts are unspecified. [`fnv1a_64`] and [`StateHasher`] are a small,
//! allocation-free, endianness-explicit `FNV`-1a implementation with no hidden
//! seed, so they are a sound building block for cross-run / cross-platform state
//! hashing. They are not cryptographic.
//!
//! [`StateHasher`] folds fields in the exact order they are written: field
//! order is part of the contract, and both runs must write identically for
//! their digests to compare equal. All multi-byte integers are folded
//! little-endian and all floats by their raw bits, so `NaN`/`-0.0` hash
//! deterministically.
//!
//! This mirrors the auditing concept in `prism_time`'s `multiworld` layer
//! (same `FNV`-1a constants, same fixed-order fold) without taking a dependency
//! edge: this is the diagnostic layer's general-purpose trace primitive.

/// `FNV`-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// `FNV`-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Hash a byte slice with 64-bit `FNV`-1a.
///
/// Deterministic, `const`, and endianness-independent (it consumes a raw byte
/// stream). Equivalent to feeding `bytes` to a [`StateHasher`] via
/// [`StateHasher::write_bytes`] and reading [`StateHasher::finish`].
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

/// An incremental 64-bit `FNV`-1a hasher for composing a frame's key state.
///
/// Feed the deterministic state fields (tick, entity transforms, physics
/// bodies, input, seed, ...) in a fixed order, then read
/// [`finish`](Self::finish). Field order is part of the contract: both runs
/// must write identically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateHasher {
    /// Running `FNV`-1a state.
    state: u64,
}

impl StateHasher {
    /// A hasher primed with the `FNV`-1a offset basis.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { state: FNV_OFFSET }
    }

    /// A hasher continuing from a previously produced digest.
    ///
    /// Useful for chaining a running digest across frames (feed the previous
    /// frame's [`finish`](Self::finish) in as the starting state).
    #[inline]
    #[must_use]
    pub const fn from_state(state: u64) -> Self {
        Self { state }
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

    /// Fold a `bool` as a single `0`/`1` byte.
    #[inline]
    pub fn write_bool(&mut self, value: bool) {
        self.write_u8(u8::from(value));
    }

    /// Fold a `u16` (little-endian) into the hash.
    #[inline]
    pub fn write_u16(&mut self, value: u16) {
        self.write_bytes(&value.to_le_bytes());
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

    /// Fold an `i32` (two's-complement little-endian) into the hash.
    #[inline]
    pub fn write_i32(&mut self, value: i32) {
        self.write_bytes(&value.to_le_bytes());
    }

    /// Fold an `i64` (two's-complement little-endian) into the hash.
    #[inline]
    pub fn write_i64(&mut self, value: i64) {
        self.write_bytes(&value.to_le_bytes());
    }

    /// Fold an `f32` by its raw bits (so `NaN`/`-0.0` hash deterministically).
    #[inline]
    pub fn write_f32_bits(&mut self, value: f32) {
        self.write_u32(value.to_bits());
    }

    /// Fold an `f64` by its raw bits (so `NaN`/`-0.0` hash deterministically).
    #[inline]
    pub fn write_f64_bits(&mut self, value: f64) {
        self.write_u64(value.to_bits());
    }

    /// Fold a length-prefixed `&str` (`u64` byte length, then the `UTF`-8
    /// bytes).
    ///
    /// The length prefix makes the fold unambiguous: `"ab" + "c"` and
    /// `"a" + "bc"` produce different digests, which matters when hashing a
    /// sequence of labels or names.
    #[inline]
    pub fn write_str(&mut self, value: &str) {
        self.write_u64(value.len() as u64);
        self.write_bytes(value.as_bytes());
    }

    /// Fold another digest into this one (combine two sub-hashes in order).
    #[inline]
    pub fn combine(&mut self, digest: u64) {
        self.write_u64(digest);
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
