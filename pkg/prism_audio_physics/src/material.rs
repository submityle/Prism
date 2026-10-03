//! Acoustic-material identity and ordered-pair resolution for the bridge.
//!
//! The physics world labels every body with an opaque acoustic material id
//! ([`AudioMaterialId`]). A sounding contact needs the symmetric
//! [`prism_audio_procedural::material::MaterialPairId`] that keys the modal
//! table and friction template. [`MaterialResolver`] performs that mapping: it
//! holds an explicit registry of authored pairs and, for every unregistered
//! pair, derives a stable pair id from the normalised material ids so the
//! material space never has a silent hole.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the acoustic-material coupling edge of design section 47.5 on the
//! physics-bridge side: it feeds a resolved
//! [`prism_audio_procedural::material::MaterialPairId`] into every
//! [`crate::translator::ContactAudioTranslator`] event so the procedural stages
//! can look up timbre with no gaps.

use alloc::collections::BTreeMap;

use prism_audio_procedural::material::MaterialPairId;

/// Opaque identifier of a body's acoustic material.
///
/// The bridge never interprets the number beyond equality, ordering, and
/// pairing; its acoustic meaning is resolved into a
/// [`MaterialPairId`] by [`MaterialResolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioMaterialId(pub u32);

/// Normalised ordered key for a material pair, used as the registry key.
///
/// Construction sorts the two ids so `(a, b)` and `(b, a)` map to the same key,
/// matching the symmetry of [`MaterialPairId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
struct PairKey {
    lo: u32,
    hi: u32,
}

impl PairKey {
    #[inline]
    fn new(a: AudioMaterialId, b: AudioMaterialId) -> Self {
        if a.0 <= b.0 {
            Self { lo: a.0, hi: b.0 }
        } else {
            Self { lo: b.0, hi: a.0 }
        }
    }
}

/// Folds a 32-bit material id into the 16-bit space of [`MaterialPairId`].
///
/// Uses a single FNV-1a style round over the four bytes so distinct ids spread
/// across the 16-bit range deterministically; the mapping is pure integer math
/// with no floating point.
#[inline]
#[must_use]
fn fold_u32_to_u16(value: u32) -> u16 {
    let mut hash: u32 = 0x811c_9dc5;
    let mut v = value;
    let mut i = 0;
    while i < 4 {
        let byte = v & 0xff;
        hash ^= byte;
        hash = hash.wrapping_mul(0x0100_0193);
        v >>= 8;
        i += 1;
    }
    // XOR-fold the 32-bit hash into 16 bits to keep both halves significant.
    ((hash >> 16) ^ (hash & 0xffff)) as u16
}

/// Ordered-pair resolver from acoustic material ids to procedural pair ids.
///
/// Explicitly registered pairs win; everything else falls back to a stable
/// derived id so [`MaterialResolver::resolve`] is total (never returns `None`).
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MaterialResolver {
    registry: BTreeMap<PairKey, MaterialPairId>,
}

impl MaterialResolver {
    /// Builds an empty resolver that resolves every pair through the fallback.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: BTreeMap::new(),
        }
    }

    /// Registers an explicit pair id for an unordered material pair.
    ///
    /// The pair is normalised, so registering `(a, b)` also serves `(b, a)`.
    /// A later registration of the same pair overrides the earlier one.
    #[inline]
    pub fn register(&mut self, a: AudioMaterialId, b: AudioMaterialId, pair: MaterialPairId) {
        self.registry.insert(PairKey::new(a, b), pair);
    }

    /// Returns the number of explicitly registered pairs.
    #[inline]
    #[must_use]
    pub fn registered_len(&self) -> usize {
        self.registry.len()
    }

    /// Resolves an unordered material pair into a procedural pair id.
    ///
    /// Returns the registered pair when present, otherwise a deterministic
    /// derived pair id built from the folded, normalised material ids. The
    /// derived id is symmetric in `a` and `b`.
    #[inline]
    #[must_use]
    pub fn resolve(&self, a: AudioMaterialId, b: AudioMaterialId) -> MaterialPairId {
        let key = PairKey::new(a, b);
        if let Some(found) = self.registry.get(&key) {
            *found
        } else {
            MaterialPairId::new(fold_u32_to_u16(key.lo), fold_u32_to_u16(key.hi))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_is_symmetric() {
        let r = MaterialResolver::new();
        let a = AudioMaterialId(11);
        let b = AudioMaterialId(97);
        assert_eq!(r.resolve(a, b), r.resolve(b, a));
    }

    #[test]
    fn registered_pair_wins() {
        let mut r = MaterialResolver::new();
        let a = AudioMaterialId(3);
        let b = AudioMaterialId(8);
        let explicit = MaterialPairId::new(1, 2);
        r.register(a, b, explicit);
        assert_eq!(r.resolve(a, b), explicit);
        assert_eq!(r.resolve(b, a), explicit);
        assert_eq!(r.registered_len(), 1);
    }

    #[test]
    fn fallback_is_deterministic_and_total() {
        let r = MaterialResolver::new();
        let a = AudioMaterialId(123_456);
        let b = AudioMaterialId(789_012);
        let first = r.resolve(a, b);
        let second = r.resolve(a, b);
        assert_eq!(first, second);
    }

    #[test]
    fn distinct_pairs_tend_to_differ() {
        let r = MaterialResolver::new();
        let p0 = r.resolve(AudioMaterialId(1), AudioMaterialId(2));
        let p1 = r.resolve(AudioMaterialId(1), AudioMaterialId(3));
        assert_ne!(p0, p1);
    }
}
