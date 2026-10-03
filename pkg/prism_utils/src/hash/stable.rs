//! Stable (seed-free) hashing and content-addressed hashing.
//!
//! The maps in [`map`](super::map) use the fast [`FxHasher`](super::FxHasher)
//! for cache-friendly lookups, but Fx is a *moving target*: its constants may
//! change and it is tuned for speed, not cross-run reproducibility. Asset IDs,
//! type IDs (consumed by `prism_reflect`), and network protocol versions need a
//! hash that is **identical across runs** of the same build, so this module
//! provides a fixed-seed [`StableHasher`] (FNV-1a 64-bit) and a 128-bit
//! [`ContentHash`] for content-addressed deduplication.
//!
//! ## Stability contract
//! - [`stable_hash_bytes`] / [`stable_hash_str`] / [`ContentHash::of`] operate
//!   directly on bytes and are stable across runs **and** platforms (they never
//!   depend on pointer width or endianness of the input type).
//! - [`stable_hash`] hashes any [`Hash`] value through [`StableHasher`]; it is
//!   stable across runs of the same build, but width-dependent types such as
//!   `usize` can differ between 32- and 64-bit targets. Prefer the byte/str
//!   entry points for cross-platform identity.
//!
//! The algorithm and seeds are a **versioned contract**: changing them breaks
//! previously stored IDs, so they must only change behind an explicit format
//! version bump.

use core::hash::{Hash, Hasher};

/// FNV-1a 64-bit offset basis (the canonical seed).
const FNV64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 128-bit offset basis.
const FNV128_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
/// FNV-1a 128-bit prime.
const FNV128_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

/// A deterministic, seed-free [`Hasher`] (FNV-1a, 64-bit).
///
/// Unlike the default `RandomState`/`FxHasher`, this hasher yields the same
/// value for the same input on every run, which is what stable IDs require. It
/// is **not** DoS-resistant; use it only for trusted, internal identity.
#[derive(Clone, Copy, Debug)]
pub struct StableHasher {
    state: u64,
}

impl StableHasher {
    /// Create a hasher seeded with the canonical FNV-1a offset basis.
    #[inline]
    pub const fn new() -> Self {
        Self {
            state: FNV64_OFFSET,
        }
    }
}

impl Default for StableHasher {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for StableHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut hash = self.state;
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV64_PRIME);
        }
        self.state = hash;
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.state
    }
}

/// A [`BuildHasher`](core::hash::BuildHasher) that always produces a fresh,
/// identically-seeded [`StableHasher`].
#[derive(Clone, Copy, Debug, Default)]
pub struct StableBuildHasher;

impl core::hash::BuildHasher for StableBuildHasher {
    type Hasher = StableHasher;
    #[inline]
    fn build_hasher(&self) -> StableHasher {
        StableHasher::new()
    }
}

/// Stable FNV-1a 64-bit hash of a byte slice. Cross-run and cross-platform
/// stable.
#[inline]
pub fn stable_hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = StableHasher::new();
    h.write(bytes);
    h.finish()
}

/// Stable FNV-1a 64-bit hash of a string's UTF-8 bytes.
#[inline]
pub fn stable_hash_str(s: &str) -> u64 {
    stable_hash_bytes(s.as_bytes())
}

/// Stable hash of any [`Hash`] value via [`StableHasher`]. Stable across runs
/// of the same build (see the module stability contract for the cross-platform
/// caveat on width-dependent types).
#[inline]
pub fn stable_hash<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut h = StableHasher::new();
    value.hash(&mut h);
    h.finish()
}

/// A 128-bit content hash for content-addressed deduplication (asset blocks,
/// meshes, textures). Computed with FNV-1a 128-bit directly over bytes, so it
/// is stable across runs and platforms. Not cryptographic.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentHash(u128);

impl ContentHash {
    /// Hash a byte slice into a content hash.
    #[inline]
    pub fn of(bytes: &[u8]) -> Self {
        let mut hash = FNV128_OFFSET;
        for &b in bytes {
            hash ^= b as u128;
            hash = hash.wrapping_mul(FNV128_PRIME);
        }
        Self(hash)
    }

    /// Hash a string's UTF-8 bytes into a content hash.
    #[inline]
    pub fn of_str(s: &str) -> Self {
        Self::of(s.as_bytes())
    }

    /// The raw 128-bit value.
    #[inline]
    pub const fn to_u128(self) -> u128 {
        self.0
    }

    /// Reconstruct a content hash from a raw 128-bit value (e.g. read back from
    /// storage).
    #[inline]
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }
}

impl core::fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ContentHash({:032x})", self.0)
    }
}

impl core::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_matches_known_vectors() {
        // Canonical FNV-1a 64 test vectors.
        assert_eq!(stable_hash_bytes(b""), FNV64_OFFSET);
        assert_eq!(stable_hash_str("a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(stable_hash_str("foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn stable_across_calls() {
        let a = stable_hash_str("prism::Transform");
        let b = stable_hash_str("prism::Transform");
        assert_eq!(a, b);
        assert_ne!(a, stable_hash_str("prism::GlobalTransform"));
    }

    #[test]
    fn content_hash_dedup_and_roundtrip() {
        let a = ContentHash::of(b"the quick brown fox");
        let b = ContentHash::of(b"the quick brown fox");
        let c = ContentHash::of(b"the quick brown fo");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(ContentHash::from_u128(a.to_u128()), a);
        assert_eq!(format!("{a}").len(), 32);
    }

    #[test]
    fn empty_content_hash_is_offset() {
        assert_eq!(ContentHash::of(b"").to_u128(), FNV128_OFFSET);
    }
}
