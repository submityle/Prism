//! Deterministic golden digest of a baked [`PhysicsCache`].
//!
//! A golden-replay test guards against silent changes to the soft-body solver:
//! bake a known scene, hash the trajectory into a [`GoldenDigest`], and compare
//! it against a previously recorded value. Because the cache stores quantized
//! integers, the hash is exact and platform-independent, so a mismatch means
//! the simulation genuinely changed rather than merely drifting in the last
//! floating-point bit.
//!
//! The digest is a plain FNV-1a over the cache metadata and every track's
//! reconstructed quantized components in frame order. Reconstructing (rather
//! than hashing the compressed records directly) makes the digest independent
//! of the keyframe interval: two caches that describe the same motion hash
//! equal even if one used denser keyframes.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. FNV-1a is
//! a public-domain, non-cryptographic hash implemented from its published
//! specification.

use crate::cache::asset::PhysicsCache;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A 64-bit deterministic digest of a [`PhysicsCache`] trajectory.
///
/// Two caches produce the same [`GoldenDigest`] iff their frame step, track
/// shapes, and every reconstructed quantized position match exactly.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GoldenDigest(pub u64);

/// Incremental FNV-1a accumulator over raw bytes.
#[derive(Clone, Copy, Debug)]
struct Fnv1a(u64);

impl Fnv1a {
    /// Starts a new accumulator at the FNV offset basis.
    const fn new() -> Fnv1a {
        Fnv1a(FNV_OFFSET)
    }

    /// Folds a single byte into the digest.
    fn write_u8(&mut self, byte: u8) {
        self.0 ^= u64::from(byte);
        self.0 = self.0.wrapping_mul(FNV_PRIME);
    }

    /// Folds all four bytes of a `u32` (little-endian) into the digest.
    fn write_u32(&mut self, value: u32) {
        for byte in value.to_le_bytes() {
            self.write_u8(byte);
        }
    }

    /// Folds all four bytes of an `i32` (little-endian) into the digest.
    fn write_i32(&mut self, value: i32) {
        self.write_u32(value as u32);
    }

    /// Folds an `f32` by hashing its IEEE-754 bit pattern.
    fn write_f32(&mut self, value: f32) {
        self.write_u32(value.to_bits());
    }

    /// Returns the accumulated digest.
    const fn finish(self) -> u64 {
        self.0
    }
}

/// Computes the deterministic [`GoldenDigest`] of `cache`.
///
/// The frame step, frame count, and track count are mixed in first, then each
/// track's particle count and every frame's reconstructed quantized components
/// are folded in order. A track that fails to reconstruct contributes a
/// sentinel marker so a corrupt cache cannot silently collide with a valid one.
#[must_use]
pub fn trajectory_hash(cache: &PhysicsCache) -> GoldenDigest {
    let mut h = Fnv1a::new();
    h.write_f32(cache.frame_dt());
    h.write_u32(cache.frame_count());
    h.write_u32(cache.track_count() as u32);
    let frames = cache.frame_count();
    for (track_index, track) in cache.tracks().enumerate() {
        h.write_u32(track_index as u32);
        h.write_u32(track.particle_count());
        for frame in 0..frames {
            match track.reconstruct(frame) {
                Some(positions) => {
                    for q in positions {
                        h.write_i32(q[0]);
                        h.write_i32(q[1]);
                        h.write_i32(q[2]);
                    }
                }
                // Distinct marker for an unreconstructable frame so a corrupt
                // track never hashes equal to a well-formed one.
                None => h.write_u32(u32::MAX),
            }
        }
    }
    GoldenDigest(h.finish())
}

/// Returns `true` when `cache`'s trajectory hashes to `expected`.
#[must_use]
pub fn verify(cache: &PhysicsCache, expected: GoldenDigest) -> bool {
    trajectory_hash(cache) == expected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::bake::{BakeConfig, Baker};
    use crate::cache::quantize::PositionQuantizer;
    use crate::cache::track::CacheTrack;
    use crate::soft::build::ClothGrid;

    fn bake(frames: u32, keyframe_interval: u32) -> PhysicsCache {
        let mut cloth = ClothGrid {
            columns: 4,
            rows: 4,
            ..ClothGrid::default()
        }
        .build_default();
        cloth.pin(0, 0);
        let baker = Baker::new(BakeConfig {
            frames,
            keyframe_interval,
            ..BakeConfig::default()
        });
        baker.bake_body(&mut cloth.body)
    }

    #[test]
    fn identical_bakes_hash_equal() {
        let a = bake(16, 8);
        let b = bake(16, 8);
        assert_eq!(trajectory_hash(&a), trajectory_hash(&b));
        assert!(verify(&b, trajectory_hash(&a)));
    }

    #[test]
    fn digest_is_independent_of_keyframe_interval() {
        let dense = bake(16, 1);
        let sparse = bake(16, 8);
        assert_eq!(trajectory_hash(&dense), trajectory_hash(&sparse));
    }

    #[test]
    fn different_motion_hashes_differ() {
        let still = bake(16, 8);
        // A shifted quantizer changes every stored index, so the digest moves.
        let mut cloth = ClothGrid {
            columns: 4,
            rows: 4,
            ..ClothGrid::default()
        }
        .build_default();
        cloth.pin(0, 0);
        let baker = Baker::new(BakeConfig {
            frames: 16,
            keyframe_interval: 8,
            quantizer: PositionQuantizer::new(2.0e-4),
            ..BakeConfig::default()
        });
        let shifted = baker.bake_body(&mut cloth.body);
        assert_ne!(trajectory_hash(&still), trajectory_hash(&shifted));
    }

    #[test]
    fn extra_track_changes_digest() {
        let one = bake(8, 4);
        let mut two = one.clone();
        // Appending an independent track must change the whole-cache digest.
        let mut track = CacheTrack::new(1, 4);
        for i in 0..one.frame_count() {
            track.push_frame(&[[i32::try_from(i).unwrap(), 0, 0]]);
        }
        two.push_track(track);
        assert_ne!(trajectory_hash(&one), trajectory_hash(&two));
    }
}
