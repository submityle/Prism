//! Deterministic mapping from a body pair to a procedural contact id.
//!
//! The procedural synthesis stages track a contact's impact -> roll -> slide ->
//! separate life cycle by a stable [`prism_audio_procedural::contact::ContactId`].
//! A persistent manifold between the same two bodies must therefore map to the
//! same id across blocks, regardless of which body the physics engine listed
//! first. [`ContactKey`] normalises the body pair, and [`procedural_id`] mixes
//! the two 64-bit body ids into a single 64-bit id with a SplitMix64-style
//! avalanche built entirely from wrapping integer operations, so the mapping is
//! pure, symmetric, and replayable with no floating point.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the stable-identity requirement of design section 47.1: the id
//! returned here is carried by every impact, sustain, and separation event so
//! the continuous state machine in `prism_audio_procedural` can follow a
//! contact over time.

use prism_audio_procedural::contact::ContactId;

use crate::body::BodyAudioId;

/// Normalised, order-independent key for a contact between two bodies.
///
/// `ContactKey::new(a, b)` equals `ContactKey::new(b, a)`, so a manifold keeps
/// one identity no matter which body the narrow phase reports first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ContactKey {
    lo: u64,
    hi: u64,
}

impl ContactKey {
    /// Builds a normalised key from two body ids.
    #[inline]
    #[must_use]
    pub fn new(a: BodyAudioId, b: BodyAudioId) -> Self {
        if a.0 <= b.0 {
            Self { lo: a.0, hi: b.0 }
        } else {
            Self { lo: b.0, hi: a.0 }
        }
    }

    /// Returns the lower body id of the pair.
    #[inline]
    #[must_use]
    pub fn lo(self) -> BodyAudioId {
        BodyAudioId(self.lo)
    }

    /// Returns the higher body id of the pair.
    #[inline]
    #[must_use]
    pub fn hi(self) -> BodyAudioId {
        BodyAudioId(self.hi)
    }
}

/// `SplitMix64` finaliser: a well-known integer avalanche with good diffusion.
#[inline]
#[must_use]
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Maps a normalised contact key to a stable procedural contact id.
///
/// Mixes the low id, then folds the high id in and mixes again, so the result
/// depends on both bodies and is symmetric because the key is normalised.
#[inline]
#[must_use]
pub fn procedural_id(key: ContactKey) -> ContactId {
    let first = mix64(key.lo);
    let combined = mix64(first ^ key.hi.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    ContactId(combined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_symmetric() {
        let a = BodyAudioId(5);
        let b = BodyAudioId(42);
        assert_eq!(ContactKey::new(a, b), ContactKey::new(b, a));
    }

    #[test]
    fn id_is_symmetric_and_stable() {
        let a = BodyAudioId(5);
        let b = BodyAudioId(42);
        let id_ab = procedural_id(ContactKey::new(a, b));
        let id_ba = procedural_id(ContactKey::new(b, a));
        assert_eq!(id_ab, id_ba);
        assert_eq!(id_ab, procedural_id(ContactKey::new(a, b)));
    }

    #[test]
    fn distinct_pairs_differ() {
        let p0 = procedural_id(ContactKey::new(BodyAudioId(1), BodyAudioId(2)));
        let p1 = procedural_id(ContactKey::new(BodyAudioId(1), BodyAudioId(3)));
        let p2 = procedural_id(ContactKey::new(BodyAudioId(2), BodyAudioId(3)));
        assert_ne!(p0, p1);
        assert_ne!(p1, p2);
        assert_ne!(p0, p2);
    }

    #[test]
    fn key_exposes_sorted_ids() {
        let key = ContactKey::new(BodyAudioId(9), BodyAudioId(4));
        assert_eq!(key.lo(), BodyAudioId(4));
        assert_eq!(key.hi(), BodyAudioId(9));
    }
}
