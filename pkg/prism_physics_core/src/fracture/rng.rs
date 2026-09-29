//! A tiny deterministic pseudo-random generator for reproducible fracture
//! seed scattering.
//!
//! Fracture patterns must be reproducible across runs and machines so that
//! baked caches and networked simulations stay in sync, so the crate ships its
//! own fixed-algorithm generator rather than depending on the platform RNG.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! `xorshift64*` generator is a standard, publicly documented pseudo-random
//! algorithm (Marsaglia 2003; Vigna 2016).

use glam::Vec3;

use crate::math::scalar::Real;

/// A reproducible `xorshift64*` pseudo-random generator.
#[derive(Clone, Debug)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Creates a generator from `seed`. A zero seed is remapped to a fixed
    /// non-zero constant because `xorshift` cannot escape the all-zero state.
    #[must_use]
    pub fn new(seed: u64) -> DeterministicRng {
        DeterministicRng {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// Returns the next 64-bit pseudo-random word.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Returns a uniform [`Real`] in the half-open range `[0, 1)`.
    pub fn next_unit(&mut self) -> Real {
        // Use the high 24 bits so every representable f32 mantissa step is hit.
        let bits = self.next_u64() >> 40;
        (bits as Real) / ((1u32 << 24) as Real)
    }

    /// Returns a uniform [`Real`] in the half-open range `[lo, hi)`.
    pub fn next_range(&mut self, lo: Real, hi: Real) -> Real {
        lo + (hi - lo) * self.next_unit()
    }

    /// Returns a point drawn uniformly from the axis-aligned box `[min, max]`.
    pub fn next_in_box(&mut self, min: Vec3, max: Vec3) -> Vec3 {
        Vec3::new(
            self.next_range(min.x, max.x),
            self.next_range(min.y, max.y),
            self.next_range(min.z, max.z),
        )
    }
}
