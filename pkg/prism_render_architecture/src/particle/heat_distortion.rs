//! Screen-space heat-haze / refraction `UV` offset model (design §16-§21).
//!
//! This module is the deterministic `CPU` reference for the post-process heat
//! distortion that production `VFX` stacks layer over hot emitters (engine
//! exhaust, fire, explosions): a particle contributes a small screen-space `UV`
//! perturbation that bends whatever `HDR` scene color is sampled behind it,
//! producing the shimmering refraction look. It owns four independent pieces
//! the compositor combines:
//!
//! 1. a normal-driven base offset, `normal_xy * strength * scale`;
//! 2. a rational-polynomial distance falloff that fades the effect with camera
//!    or emitter distance;
//! 3. a rolling phase perturbation that animates the shimmer, built from an
//!    integer-hash value-noise field rather than a trigonometric wave; and
//! 4. clamps that bound the offset magnitude and keep the distorted `UV` inside
//!    the sampled `[0, 1]` texture domain.
//!
//! Determinism rules (design §29) are inherited: the only floating-point
//! primitives beyond ordinary arithmetic are `f32::floor` (lattice location and
//! fractional phase) and `f32::sqrt` (offset magnitude). There are no
//! transcendental calls (`sin` / `cos` / `exp` / `ln` / `pow`); the phase roll
//! is hash-seeded value noise faded with the multiply-only smoothstep
//! `t * t * (3 - 2 t)`, the falloff is the rational polynomial
//! `1 / (1 + k1 d + k2 d^2)`, and interpolation is linear. The `GPU` shimmer
//! kernel can reproduce these results bit for bit by hashing the same lattice
//! cells. The `std430` packing anticipates that kernel's uniform block.

use crate::particle::gpu_layout::VEC4_STRIDE;

/// Scale that turns a 24-bit hash mantissa into the half-open range `[0, 1)`.
const INV_2POW24: f32 = 1.0 / 16_777_216.0;

/// Odd-integer salt seeding the horizontal rolling-noise channel.
const ROLL_SEED_X: u32 = 0x68E3_1DA4;

/// Odd-integer salt seeding the vertical rolling-noise channel, so the two
/// channels are statistically independent.
const ROLL_SEED_Y: u32 = 0xB543_9C13;

/// Smallest denominator allowed in the distance falloff, so the rational
/// falloff never divides by (near) zero for adversarial coefficients.
const MIN_FALLOFF_DENOM: f32 = 1.0e-3;

/// Byte size of the `std430` packing of [`HeatParams`]: the five scalars occupy
/// the first `vec4` plus one scalar of the second, padded up to two `vec4`
/// slots so the block honors the 16-byte `std430` base alignment.
pub const HEAT_STD430_SIZE: usize = 2 * VEC4_STRIDE;

/// Absolute tolerance for the `f32` equality comparisons used by the tests;
/// direct `==` on floating point is intentionally avoided.
#[cfg(test)]
const CMP_EPS: f32 = 1.0e-6;

/// One folding step of the hash: xor-in a multiplied input word, then rotate
/// and multiply to spread the bits before the next word is folded.
#[must_use]
fn mix(mut h: u32, v: u32) -> u32 {
    h ^= v.wrapping_mul(0x9E37_79B1);
    h = h.rotate_left(15).wrapping_mul(0x85EB_CA6B);
    h
}

/// Final avalanche (an integer bit-mixer) applied once after all inputs are
/// folded, giving a well-distributed 32-bit result.
#[must_use]
fn finalize(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h
}

/// Stateless integer hash of a 1D lattice cell and seed (the pseudo-noise
/// generator). No state, no transcendental call, and bit-identical between the
/// `CPU` reference and a future `GPU` kernel; negative cells address the whole
/// signed lattice through a two's-complement cast.
#[must_use]
fn hash_cell(i: i32, seed: u32) -> u32 {
    let mut h = seed ^ 0x811C_9DC5;
    h = mix(h, i as u32);
    finalize(h)
}

/// The reproducible scalar value assigned to a lattice cell, in `[-1, 1)`.
#[must_use]
fn cell_value(i: i32, seed: u32) -> f32 {
    let h = hash_cell(i, seed);
    let unit = ((h >> 8) as f32) * INV_2POW24;
    unit * 2.0 - 1.0
}

/// The smoothstep fade `t * t * (3 - 2 t)`: a multiply-only Hermite polynomial
/// with zero first derivative at `0` and `1`, so interpolation across cells is
/// `C1`-continuous without any transcendental call.
#[must_use]
fn fade(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation `a + (b - a) * t`.
#[must_use]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Splits a coordinate into its floor cell index and the fractional offset in
/// `[0, 1)`.
#[must_use]
fn floor_split(x: f32) -> (i32, f32) {
    let f = x.floor();
    (f as i32, x - f)
}

/// Smoothstep-faded 1D value noise in `[-1, 1]`: a pure function of the
/// coordinate and seed used to roll the shimmer phase without a sine wave.
#[must_use]
fn value_noise_1d(t: f32, seed: u32) -> f32 {
    let (i, f) = floor_split(t);
    let c0 = cell_value(i, seed);
    let c1 = cell_value(i + 1, seed);
    lerp(c0, c1, fade(f))
}

/// The base screen-space `UV` offset from a surface normal.
///
/// The `xy` components of the (view-space) normal are scaled by the product of
/// the emitter `strength` and the global `distortion_scale`. A zero normal
/// yields a zero offset, so a flat-facing particle refracts nothing.
#[must_use]
pub fn uv_offset(normal_xy: [f32; 2], strength: f32, scale: f32) -> [f32; 2] {
    let k = strength * scale;
    [normal_xy[0] * k, normal_xy[1] * k]
}

/// Animated rolling perturbation sampled from the hash value-noise field.
///
/// The `base_uv` is scaled by `freq` and advanced by `phase`, then each channel
/// reads an independent value-noise realization, giving a smooth, periodic-free
/// shimmer that is fully deterministic (equal inputs return equal output) and
/// contains no trigonometric call. The result lies in `[-1, 1]` per component.
#[must_use]
pub fn rolling_offset(base_uv: [f32; 2], phase: f32, freq: f32) -> [f32; 2] {
    let tx = base_uv[0] * freq + phase;
    let ty = base_uv[1] * freq + phase;
    [
        value_noise_1d(tx, ROLL_SEED_X),
        value_noise_1d(ty, ROLL_SEED_Y),
    ]
}

/// Clamps each component of `offset` to `[-max_offset, max_offset]`.
///
/// `max_offset` is treated as a non-negative magnitude bound; a negative bound
/// collapses the offset to zero, which is the safe (no-distortion) fallback.
#[must_use]
pub fn clamp_offset(offset: [f32; 2], max_offset: f32) -> [f32; 2] {
    let m = max_offset.max(0.0);
    [offset[0].clamp(-m, m), offset[1].clamp(-m, m)]
}

/// Applies an `offset` to a `UV` and clamps the result into the `[0, 1]`
/// sampled texture domain, so the distortion never reads outside the source.
#[must_use]
pub fn apply_to_uv(uv: [f32; 2], offset: [f32; 2]) -> [f32; 2] {
    [
        (uv[0] + offset[0]).clamp(0.0, 1.0),
        (uv[1] + offset[1]).clamp(0.0, 1.0),
    ]
}

/// One evaluation of the heat distortion: the bounded `UV` `offset` and the
/// `distorted_uv` it produces when applied to the sampled coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeatSample {
    /// The clamped screen-space `UV` offset the compositor adds.
    pub offset: [f32; 2],
    /// The distorted `UV`, clamped into the `[0, 1]` sampled domain.
    pub distorted_uv: [f32; 2],
}

/// Artist-facing parameters of a heat-distortion contribution.
///
/// The refraction strength and global `distortion_scale` set the base offset,
/// the two falloff coefficients shape the rational distance fade, and
/// `max_offset` bounds the final screen-space displacement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeatParams {
    /// Per-emitter refraction strength multiplying the normal offset.
    pub strength: f32,
    /// Global screen-space displacement scale (in `UV` units).
    pub distortion_scale: f32,
    /// Linear distance-falloff coefficient `k1` (per unit distance).
    pub falloff_k1: f32,
    /// Quadratic distance-falloff coefficient `k2` (per unit distance squared).
    pub falloff_k2: f32,
    /// Maximum absolute `UV` offset per component after falloff.
    pub max_offset: f32,
}

impl HeatParams {
    /// Builds a parameter set from its fields.
    #[must_use]
    pub const fn new(
        strength: f32,
        distortion_scale: f32,
        falloff_k1: f32,
        falloff_k2: f32,
        max_offset: f32,
    ) -> Self {
        Self {
            strength,
            distortion_scale,
            falloff_k1,
            falloff_k2,
            max_offset,
        }
    }

    /// The rational-polynomial distance falloff at distance `d`, in `(0, 1]`.
    ///
    /// Uses `1 / (1 + k1 d + k2 d^2)` on the clamped distance `max(d, 0)` so the
    /// factor is exactly `1` at `d = 0` and decreases monotonically as `d`
    /// grows (for non-negative coefficients). The denominator is floored at
    /// [`MIN_FALLOFF_DENOM`] and the result clamped to `[0, 1]`, so the division
    /// is always well defined.
    #[must_use]
    pub fn distance_falloff(&self, d: f32) -> f32 {
        let dd = d.max(0.0);
        let denom = (1.0 + self.falloff_k1 * dd + self.falloff_k2 * dd * dd).max(MIN_FALLOFF_DENOM);
        (1.0 / denom).clamp(0.0, 1.0)
    }

    /// Evaluates the distortion for a particle with view-space normal
    /// `normal_xy` at `distance`, sampling the scene at `uv`.
    ///
    /// Chains the base normal offset, the distance falloff, the magnitude
    /// clamp, and the `UV`-domain clamp into a single [`HeatSample`].
    #[must_use]
    pub fn evaluate(&self, normal_xy: [f32; 2], distance: f32, uv: [f32; 2]) -> HeatSample {
        let base = uv_offset(normal_xy, self.strength, self.distortion_scale);
        let fall = self.distance_falloff(distance);
        let scaled = [base[0] * fall, base[1] * fall];
        let offset = clamp_offset(scaled, self.max_offset);
        let distorted_uv = apply_to_uv(uv, offset);
        HeatSample {
            offset,
            distorted_uv,
        }
    }

    /// Packs the parameters into their `std430` uniform-block bytes.
    ///
    /// The five scalars are laid out little-endian as `f32`s across two `vec4`
    /// slots ([`HEAT_STD430_SIZE`] bytes); the trailing three scalars are
    /// zero padding so the block honors the 16-byte `std430` base alignment
    /// expected by the `GPU` kernel.
    #[must_use]
    pub fn to_std430(&self) -> [u8; HEAT_STD430_SIZE] {
        let fields = [
            self.strength,
            self.distortion_scale,
            self.falloff_k1,
            self.falloff_k2,
            self.max_offset,
        ];
        let mut bytes = [0u8; HEAT_STD430_SIZE];
        for (slot, value) in bytes.chunks_exact_mut(4).zip(fields.iter()) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::gpu_layout::storage_bytes;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn approx2(a: [f32; 2], b: [f32; 2]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1])
    }

    #[test]
    fn zero_normal_gives_zero_offset() {
        let off = uv_offset([0.0, 0.0], 3.0, 2.0);
        assert!(approx2(off, [0.0, 0.0]));
        let p = HeatParams::new(5.0, 4.0, 0.1, 0.01, 0.2);
        let s = p.evaluate([0.0, 0.0], 1.0, [0.5, 0.5]);
        assert!(approx2(s.offset, [0.0, 0.0]));
        assert!(approx2(s.distorted_uv, [0.5, 0.5]));
    }

    #[test]
    fn uv_offset_scales_with_strength_and_scale() {
        let off = uv_offset([1.0, -2.0], 3.0, 2.0);
        assert!(approx2(off, [6.0, -12.0]));
    }

    #[test]
    fn falloff_is_one_at_zero_distance() {
        let p = HeatParams::new(1.0, 1.0, 0.5, 0.25, 1.0);
        assert!(approx(p.distance_falloff(0.0), 1.0));
    }

    #[test]
    fn falloff_is_monotonically_decreasing_in_distance() {
        let p = HeatParams::new(1.0, 1.0, 0.5, 0.25, 1.0);
        let d0 = p.distance_falloff(0.0);
        let d1 = p.distance_falloff(1.0);
        let d2 = p.distance_falloff(5.0);
        let d3 = p.distance_falloff(50.0);
        assert!(d0 > d1);
        assert!(d1 > d2);
        assert!(d2 > d3);
        assert!(d3 > 0.0);
    }

    #[test]
    fn falloff_negative_distance_clamps_to_zero_distance() {
        let p = HeatParams::new(1.0, 1.0, 0.5, 0.25, 1.0);
        assert!(approx(p.distance_falloff(-10.0), 1.0));
    }

    #[test]
    fn clamp_offset_bounds_each_component() {
        let clamped = clamp_offset([0.9, -0.7], 0.2);
        assert!(approx2(clamped, [0.2, -0.2]));
        let within = clamp_offset([0.05, -0.1], 0.2);
        assert!(approx2(within, [0.05, -0.1]));
    }

    #[test]
    fn clamp_offset_negative_bound_is_zero() {
        let clamped = clamp_offset([0.9, -0.7], -1.0);
        assert!(approx2(clamped, [0.0, 0.0]));
    }

    #[test]
    fn apply_to_uv_stays_in_unit_square() {
        let hi = apply_to_uv([0.95, 0.5], [0.5, -0.9]);
        assert!(hi[0] >= 0.0 && hi[0] <= 1.0);
        assert!(hi[1] >= 0.0 && hi[1] <= 1.0);
        assert!(approx2(hi, [1.0, 0.0]));
    }

    #[test]
    fn evaluate_offset_respects_max_offset() {
        let p = HeatParams::new(100.0, 100.0, 0.0, 0.0, 0.03);
        let s = p.evaluate([1.0, -1.0], 0.0, [0.5, 0.5]);
        assert!(s.offset[0].abs() <= 0.03 + CMP_EPS);
        assert!(s.offset[1].abs() <= 0.03 + CMP_EPS);
        assert!(s.distorted_uv[0] >= 0.0 && s.distorted_uv[0] <= 1.0);
        assert!(s.distorted_uv[1] >= 0.0 && s.distorted_uv[1] <= 1.0);
    }

    #[test]
    fn rolling_offset_is_deterministic_and_bounded() {
        let a = rolling_offset([0.3, 0.7], 1.25, 4.0);
        let b = rolling_offset([0.3, 0.7], 1.25, 4.0);
        assert!(approx2(a, b));
        assert!(a[0] >= -1.0 && a[0] <= 1.0);
        assert!(a[1] >= -1.0 && a[1] <= 1.0);
    }

    #[test]
    fn rolling_offset_changes_with_phase() {
        let a = rolling_offset([0.3, 0.7], 0.0, 4.0);
        let b = rolling_offset([0.3, 0.7], 3.5, 4.0);
        assert!(!approx2(a, b));
    }

    #[test]
    fn rolling_channels_are_decorrelated() {
        // A symmetric input would collapse to equal channels only if the two
        // seeds were shared; independent seeds keep them distinct.
        let o = rolling_offset([2.0, 2.0], 0.0, 1.0);
        assert!(!approx(o[0], o[1]));
    }

    #[test]
    fn std430_packing_is_two_vec4_slots() {
        let p = HeatParams::new(1.0, 2.0, 3.0, 4.0, 5.0);
        let bytes = p.to_std430();
        assert_eq!(bytes.len(), HEAT_STD430_SIZE);
        assert_eq!(HEAT_STD430_SIZE % VEC4_STRIDE, 0);
        assert_eq!(storage_bytes(HEAT_STD430_SIZE, 1), HEAT_STD430_SIZE);
    }

    #[test]
    fn std430_round_trips_the_five_scalars() {
        let p = HeatParams::new(1.5, -2.25, 3.0, 0.5, 0.125);
        let bytes = p.to_std430();
        let fields = [
            p.strength,
            p.distortion_scale,
            p.falloff_k1,
            p.falloff_k2,
            p.max_offset,
        ];
        for (slot, value) in bytes.chunks_exact(4).zip(fields.iter()) {
            let mut word = [0u8; 4];
            word.copy_from_slice(slot);
            assert!(approx(f32::from_le_bytes(word), *value));
        }
        // The padding tail (three scalars) must be zero.
        for byte in &bytes[20..] {
            assert_eq!(*byte, 0);
        }
    }
}
