//! Deterministic pseudo-random generators and sampling helpers (M5).
//!
//! Everything here is **explicitly seeded, reproducible, and free of any ML**.
//! The [`Rng`] trait defines the raw bit source (`next_u32`/`next_u64`) and
//! provides portable, allocation-free helpers on top: uniform floats, bounded
//! integers, booleans, and geometric/Gaussian distributions.
//!
//! Three classic generators are provided:
//! - [`SplitMix64`] — tiny seed expander and lightweight PRNG.
//! - [`Pcg32`] — small-state, stream-selectable, 32-bit output.
//! - [`Xoshiro256StarStar`] — the general-purpose default (large period,
//!   `jump`-able into independent sub-streams).
//!
//! Float generation uses the standard "upper-bits × 2⁻ᵐ" construction and is
//! bit-identical across platforms given the same `u64`/`u32` stream, satisfying
//! the determinism contract from the design doc.

mod pcg;
mod splitmix;
mod xoshiro;

pub use pcg::Pcg32;
pub use splitmix::SplitMix64;
pub use xoshiro::Xoshiro256StarStar;

use crate::float::f32 as mf;
use crate::{Vec2, Vec3};

/// A deterministic source of uniformly-distributed random bits, plus derived
/// sampling helpers.
///
/// Implementors only need to provide [`Rng::next_u32`] and [`Rng::next_u64`];
/// every other method has a portable default built on those two.
pub trait Rng {
    /// Next uniformly-distributed 32-bit value.
    fn next_u32(&mut self) -> u32;

    /// Next uniformly-distributed 64-bit value.
    fn next_u64(&mut self) -> u64;

    /// Uniform `f32` in `[0, 1)` (24 bits of mantissa randomness).
    #[inline]
    fn next_f32(&mut self) -> f32 {
        // Top 24 bits -> [0, 1); exact and never reaches 1.0.
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Uniform `f64` in `[0, 1)` (53 bits of mantissa randomness).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform `u64` in `[0, bound)`. Returns `0` if `bound == 0`.
    ///
    /// Uses Lemire's multiply-high mapping, which is fast and effectively
    /// unbiased for game-scale bounds while remaining fully deterministic.
    #[inline]
    fn range_u64(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        ((u128::from(self.next_u64()) * u128::from(bound)) >> 64) as u64
    }

    /// Uniform `i64` in `[min, max)`. Panics (debug) if `min >= max`.
    #[inline]
    fn range_i64(&mut self, min: i64, max: i64) -> i64 {
        debug_assert!(min < max, "range_i64 requires min < max");
        let span = (max as i128 - min as i128) as u64;
        min.wrapping_add(self.range_u64(span) as i64)
    }

    /// Uniform `u32` in `[0, bound)`. Returns `0` if `bound == 0`.
    #[inline]
    fn range_u32(&mut self, bound: u32) -> u32 {
        if bound == 0 {
            return 0;
        }
        (((u64::from(self.next_u32())) * u64::from(bound)) >> 32) as u32
    }

    /// Uniform `f32` in `[min, max)`.
    #[inline]
    fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    /// Uniform `f64` in `[min, max)`.
    #[inline]
    fn range_f64(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next_f64()
    }

    /// Random boolean that is `true` with probability `p` (clamped to `[0,1]`).
    #[inline]
    fn gen_bool(&mut self, p: f32) -> bool {
        self.next_f32() < p.clamp(0.0, 1.0)
    }

    /// Fair coin flip.
    #[inline]
    fn flip(&mut self) -> bool {
        (self.next_u32() >> 31) == 1
    }

    /// A point uniformly distributed on the unit circle (length `1`).
    #[inline]
    fn unit_circle(&mut self) -> Vec2 {
        let theta = self.range_f32(0.0, core::f32::consts::TAU);
        let (s, c) = mf::sin_cos(theta);
        Vec2::new(c, s)
    }

    /// A point uniformly distributed inside the unit disk (length `<= 1`).
    #[inline]
    fn in_unit_disk(&mut self) -> Vec2 {
        // Radius `sqrt(u)` makes the area distribution uniform.
        let r = mf::sqrt(self.next_f32());
        let theta = self.range_f32(0.0, core::f32::consts::TAU);
        let (s, c) = mf::sin_cos(theta);
        Vec2::new(r * c, r * s)
    }

    /// A direction uniformly distributed on the unit sphere (length `1`).
    #[inline]
    fn unit_sphere(&mut self) -> Vec3 {
        // Marsaglia / inverse-CDF method: uniform `z`, uniform azimuth.
        let z = self.range_f32(-1.0, 1.0);
        let theta = self.range_f32(0.0, core::f32::consts::TAU);
        let r = mf::sqrt((1.0 - z * z).max(0.0));
        let (s, c) = mf::sin_cos(theta);
        Vec3::new(r * c, r * s, z)
    }

    /// A point uniformly distributed inside the unit ball (length `<= 1`).
    #[inline]
    fn in_unit_sphere(&mut self) -> Vec3 {
        // Uniform direction scaled by `u^(1/3)` for uniform volume density.
        let dir = self.unit_sphere();
        let r = mf::cbrt(self.next_f32());
        dir * r
    }

    /// A standard-normal `f32` (mean `0`, variance `1`) via Box–Muller.
    #[inline]
    fn gaussian(&mut self) -> f32 {
        self.gaussian_pair().0
    }

    /// Two independent standard-normal `f32`s via one Box–Muller transform.
    #[inline]
    fn gaussian_pair(&mut self) -> (f32, f32) {
        // Avoid `ln(0)` by excluding the zero endpoint.
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        let r = mf::sqrt(-2.0 * mf::ln(u1));
        let (s, c) = mf::sin_cos(core::f32::consts::TAU * u2);
        (r * c, r * s)
    }

    /// A normal `f32` with the given `mean` and standard deviation `std_dev`.
    #[inline]
    fn normal(&mut self, mean: f32, std_dev: f32) -> f32 {
        mean + std_dev * self.gaussian()
    }
}
