//! Deterministic seeded pseudo-random generator shared across the crate.
//!
//! Every stochastic choice in this crate (granular grain scattering, rolling
//! pulse jitter, soundscape one-shot placement) is driven through this
//! generator so that an identical seed plus an identical sequence of inputs
//! reproduces bit-identical output. This is the property that makes the engine
//! golden-testable and network-consistent.
//!
//! The algorithm is a `xorshift64*` core (Marsaglia 2003 / Vigna 2016) seeded
//! through a `SplitMix64` mixing step (Steele et al. 2014). Both are standard,
//! publicly documented integer generators; the whole module is pure integer
//! arithmetic with no floating-point state, so stream order is identical on
//! every platform.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Backs section 47 (contact/granular stochastic shaping) and section 37
//! (procedural soundscape scattering); mirrors the integer-only RNG style used
//! by `prism_physics_core::fracture::rng` without sharing code.

use bevy_math::ops;

use crate::dsp::TWO_PI;
use prism_audio_core::math::Sample;

/// A reproducible `xorshift64*` generator seeded via `SplitMix64`.
///
/// The generator is `Clone` and holds a single 64-bit word, so a snapshot of
/// its state can be captured and restored to replay a stream (used by the
/// block renderers to rewind on a dry/wet reset).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ProceduralRng {
    state: u64,
}

impl ProceduralRng {
    /// Creates a generator from `seed`.
    ///
    /// The seed is passed through a `SplitMix64` step so that even nearby seeds
    /// (`0`, `1`, `2`, ...) produce well-separated streams. A resulting zero
    /// state is remapped to a fixed non-zero constant because `xorshift` cannot
    /// escape the all-zero state.
    #[inline]
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Self {
            state: if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z },
        }
    }

    /// Returns the next 64-bit pseudo-random word and advances the state.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Returns the next 32-bit pseudo-random word (the high bits of the 64-bit
    /// word, which have the best statistical quality for `xorshift64*`).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Returns a uniform [`Sample`] in the half-open range `[0, 1)`.
    #[inline]
    pub fn next_unit(&mut self) -> Sample {
        // Use the top 24 bits so every representable f32 mantissa step is hit.
        let bits = self.next_u64() >> 40;
        (bits as Sample) / ((1u32 << 24) as Sample)
    }

    /// Returns a uniform [`Sample`] in the bipolar range `[-1, 1)`.
    #[inline]
    pub fn next_bipolar(&mut self) -> Sample {
        self.next_unit() * 2.0 - 1.0
    }

    /// Returns a uniform [`Sample`] in the half-open range `[lo, hi)`.
    ///
    /// If `hi <= lo` the function returns `lo`, so a degenerate range never
    /// produces a value outside its endpoints.
    #[inline]
    pub fn next_range(&mut self, lo: Sample, hi: Sample) -> Sample {
        if hi <= lo {
            lo
        } else {
            lo + (hi - lo) * self.next_unit()
        }
    }

    /// Returns a point drawn uniformly from the disc of `radius` centred on the
    /// origin, as an `(x, z)` pair on the ground plane.
    ///
    /// Rejection-free: the radius is drawn as `sqrt(u)` so the areal density is
    /// uniform (naive `radius * u` clusters samples at the centre).
    #[inline]
    pub fn next_in_disc(&mut self, radius: Sample) -> (Sample, Sample) {
        let r = radius * ops::sqrt(self.next_unit());
        let theta = TWO_PI * self.next_unit();
        let (s, c) = ops::sin_cos(theta);
        (r * c, r * s)
    }

    /// Returns `true` with probability `p` (clamped to `[0, 1]`).
    #[inline]
    pub fn chance(&mut self, p: Sample) -> bool {
        self.next_unit() < p.clamp(0.0, 1.0)
    }

    /// Captures the current state so it can later be restored with
    /// [`ProceduralRng::restore`], replaying the identical stream.
    #[inline]
    #[must_use]
    pub fn snapshot(&self) -> u64 {
        self.state
    }

    /// Restores a state previously captured with [`ProceduralRng::snapshot`].
    #[inline]
    pub fn restore(&mut self, state: u64) {
        self.state = if state == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            state
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = ProceduralRng::new(0xDEAD_BEEF);
        let mut b = ProceduralRng::new(0xDEAD_BEEF);
        for _ in 0..256 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seed_differs() {
        let mut a = ProceduralRng::new(1);
        let mut b = ProceduralRng::new(2);
        let mut diff = false;
        for _ in 0..32 {
            if a.next_u64() != b.next_u64() {
                diff = true;
                break;
            }
        }
        assert!(diff);
    }

    #[test]
    fn unit_is_in_range() {
        let mut rng = ProceduralRng::new(42);
        for _ in 0..10_000 {
            let u = rng.next_unit();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn disc_stays_inside_radius() {
        let mut rng = ProceduralRng::new(7);
        for _ in 0..10_000 {
            let (x, z) = rng.next_in_disc(3.0);
            assert!(ops::sqrt(x * x + z * z) <= 3.0 + 1e-4);
        }
    }

    #[test]
    fn snapshot_replays_stream() {
        let mut rng = ProceduralRng::new(99);
        let snap = rng.snapshot();
        let first: [u64; 8] = core::array::from_fn(|_| rng.next_u64());
        rng.restore(snap);
        let second: [u64; 8] = core::array::from_fn(|_| rng.next_u64());
        assert_eq!(first, second);
    }

    #[test]
    fn zero_seed_is_non_degenerate() {
        let mut rng = ProceduralRng::new(0);
        assert_ne!(rng.next_u64(), 0);
    }
}
