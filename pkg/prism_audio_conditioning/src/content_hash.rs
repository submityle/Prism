//! Deterministic content-addressed hashing and the conditioning cache.
//!
//! Conditioning is expensive, so results are memoized by a content hash of the
//! source bytes folded together with the stage parameters: identical inputs
//! must always map to the identical artifact. The hash is a 64-bit `FNV-1a`
//! accumulator computed with wrapping integer arithmetic only (no floating
//! point), which makes it bit-for-bit reproducible across platforms.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. `FNV-1a`
//! is a public-domain non-cryptographic hash; only the published constants and
//! update rule are used.
//!
//! # Relationship
//! Keys the conditioning cache of design section 51 so repeated authoring runs
//! are golden-reproducible and skip redundant work.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// `FNV-1a` 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A 64-bit content hash produced by [`Hasher`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ContentHash(pub u64);

/// Hashes a byte slice in one shot with [`Hasher`].
#[must_use]
pub fn hash_bytes(bytes: &[u8]) -> ContentHash {
    let mut hasher = Hasher::new();
    hasher.write(bytes);
    hasher.finish()
}

/// An `FNV-1a` accumulator that folds in bytes and primitive values.
///
/// Mix order is significant: callers must write the same fields in the same
/// order to obtain the same hash. Every update uses wrapping arithmetic so the
/// result is identical on every target.
#[derive(Debug, Clone, Copy)]
pub struct Hasher {
    /// Running hash state.
    state: u64,
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher {
    /// Creates a hasher seeded with the `FNV-1a` offset basis.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: FNV_OFFSET_BASIS,
        }
    }

    /// Folds a byte slice into the running hash.
    pub fn write(&mut self, bytes: &[u8]) {
        let mut state = self.state;
        for &byte in bytes {
            state ^= u64::from(byte);
            state = state.wrapping_mul(FNV_PRIME);
        }
        self.state = state;
    }

    /// Folds a single byte into the running hash.
    pub fn write_u8(&mut self, value: u8) {
        self.write(&[value]);
    }

    /// Folds a `u32` (little-endian) into the running hash.
    pub fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    /// Folds a `u64` (little-endian) into the running hash.
    pub fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    /// Folds an `f32` into the running hash by its raw bit pattern.
    ///
    /// Bit-pattern hashing keeps the hash deterministic without any floating
    /// point arithmetic; `NaN` payloads are preserved verbatim.
    pub fn write_f32(&mut self, value: f32) {
        self.write(&value.to_bits().to_le_bytes());
    }

    /// Returns the finished hash. The accumulator may continue to be written.
    #[must_use]
    pub const fn finish(&self) -> ContentHash {
        ContentHash(self.state)
    }
}

/// A memoized conditioning result stored in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CacheEntry {
    /// The content hash this entry is keyed by (stored for round-tripping).
    pub hash: ContentHash,
    /// Opaque payload (for example a serialized artifact or a blob offset).
    pub payload: Vec<u8>,
}

impl CacheEntry {
    /// Creates a cache entry.
    #[must_use]
    pub fn new(hash: ContentHash, payload: Vec<u8>) -> Self {
        Self { hash, payload }
    }
}

/// A content-addressed cache keyed by [`ContentHash`].
///
/// A [`BTreeMap`] is used instead of a hash map so iteration order (and any
/// serialized form) is deterministic across runs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ConditioningCache {
    /// Hash-keyed entries in sorted order.
    map: BTreeMap<ContentHash, CacheEntry>,
}

impl ConditioningCache {
    /// Creates an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    /// Inserts or replaces the entry for `entry.hash`, returning the previous
    /// entry if one was present.
    pub fn put(&mut self, entry: CacheEntry) -> Option<CacheEntry> {
        self.map.insert(entry.hash, entry)
    }

    /// Returns the entry for `hash`, if present.
    #[must_use]
    pub fn get(&self, hash: ContentHash) -> Option<&CacheEntry> {
        self.map.get(&hash)
    }

    /// Returns `true` when an entry exists for `hash`.
    #[must_use]
    pub fn contains(&self, hash: ContentHash) -> bool {
        self.map.contains_key(&hash)
    }

    /// The number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` when the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn same_bytes_same_hash() {
        assert_eq!(hash_bytes(b"prism audio"), hash_bytes(b"prism audio"));
    }

    #[test]
    fn one_byte_change_changes_hash() {
        let a = hash_bytes(b"prism audio");
        let b = hash_bytes(b"prism Audio");
        assert_ne!(a, b);
    }

    #[test]
    fn empty_hash_is_offset_basis() {
        assert_eq!(hash_bytes(&[]), ContentHash(FNV_OFFSET_BASIS));
    }

    #[test]
    fn folding_order_matters() {
        let mut a = Hasher::new();
        a.write_u32(1);
        a.write_u32(2);
        let mut b = Hasher::new();
        b.write_u32(2);
        b.write_u32(1);
        assert_ne!(a.finish(), b.finish());
    }

    #[test]
    fn float_bits_are_hashed() {
        let mut a = Hasher::new();
        a.write_f32(0.5);
        let mut b = Hasher::new();
        b.write_f32(0.5);
        assert_eq!(a.finish(), b.finish());
        let mut c = Hasher::new();
        c.write_f32(0.500_001);
        assert_ne!(a.finish(), c.finish());
    }

    #[test]
    fn cache_put_and_get() {
        let mut cache = ConditioningCache::new();
        let hash = hash_bytes(b"asset");
        assert!(cache.is_empty());
        assert!(cache.put(CacheEntry::new(hash, vec![1, 2, 3])).is_none());
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(hash));
        assert_eq!(cache.get(hash).unwrap().payload, vec![1, 2, 3]);
        let prev = cache.put(CacheEntry::new(hash, vec![4])).unwrap();
        assert_eq!(prev.payload, vec![1, 2, 3]);
    }
}
