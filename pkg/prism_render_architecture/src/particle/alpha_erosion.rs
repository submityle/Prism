//! Life-driven *alpha erosion* (dissolve) for particle shading (design §17).
//!
//! Many production `VFX` looks fade a particle out not by a flat alpha ramp but
//! by *eroding* it against a per-particle noise field: a rising threshold sweeps
//! across the noise so the sprite dissolves in irregular holes, with an emissive
//! rim glowing along the moving dissolve edge. This module owns the `CPU`
//! reference for that model so a future `GPU` kernel can match it bit for bit.
//!
//! The pipeline is: (1) a normalized age `t` in `0..=1` drives an erosion
//! `threshold` via [`threshold_over_age`]; (2) a per-particle noise value `n` in
//! `0..=1` comes from a self-contained integer-hash pseudo-noise
//! ([`hash_noise01`]); (3) [`erosion_alpha`] compares `n` against the threshold
//! through a `smoothstep` soft edge; and (4) [`edge_glow_factor`] lights a
//! band-shaped rim right at the dissolve boundary. [`ErosionParams::evaluate`]
//! wires these together into an [`ErosionSample`].
//!
//! Only `sqrt`-free rational/`smoothstep` arithmetic and integer hashing are
//! used — no transcendental functions — so the result is deterministic and
//! reproducible. `GPU` packing follows the shared `std430` `vec4` alignment from
//! [`super::gpu_layout`].

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Minimum edge width (and generic denominator guard) below which a soft edge
/// collapses to a hard step, so no division by zero can produce a `NaN`.
const MIN_EDGE: f32 = 1e-6;

/// Byte stride of one [`ErosionParams`] record in a `std430` storage buffer.
///
/// The seven scalars pack into two `vec4` slots: `vec4(glow_rgb, glow_intensity)`
/// followed by `vec4(edge_width, threshold_start, threshold_end, pad)`.
pub const EROSION_PARAMS_STRIDE: usize = 2 * VEC4_STRIDE;

/// Clamps a scalar into the `0..=1` range without branching on equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Hermite `smoothstep` from `edge0` to `edge1` evaluated at `x`.
///
/// Returns `0.0` at or below `edge0`, `1.0` at or above `edge1`, and the
/// `t * t * (3 - 2 * t)` interpolation in between. `edge0` is expected to be
/// less than `edge1`; a degenerate (near-equal) interval collapses to a hard
/// step at `edge1` rather than dividing by zero.
#[must_use]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span < MIN_EDGE {
        return if x < edge1 { 0.0 } else { 1.0 };
    }
    let t = clamp01((x - edge0) / span);
    t * t * (3.0 - 2.0 * t)
}

/// Integer avalanche hash mixing a `u32` seed into a well-distributed `u32`.
///
/// This is a self-contained mixer (no dependency on any noise module): a
/// sequence of xor-shifts and odd-constant multiplies, giving a deterministic,
/// platform-independent pseudo-random word for the erosion `RNG`.
#[must_use]
fn hash_u32(seed: u32) -> u32 {
    let mut x = seed;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Deterministic per-particle pseudo-noise in `0..=1` from an integer `seed`.
///
/// The `seed` is mixed by [`hash_u32`] and normalized by [`u32::MAX`], so the
/// same seed always yields the same value and distinct seeds decorrelate.
#[must_use]
pub fn hash_noise01(seed: u32) -> f32 {
    let hashed = hash_u32(seed);
    // u32 -> f32 for a `0..=1` normalization; the modest precision loss above
    // 2^24 is irrelevant to a dissolve mask and matches the GPU normalize.
    let numerator = hashed as f32;
    let denominator = u32::MAX as f32;
    numerator / denominator
}

/// Erosion threshold at normalized age `t`, biased from `start` to `end`.
///
/// Linearly advances the dissolve threshold across the particle's life:
/// `start` at `t = 0` and `end` at `t = 1`. Both `t` and the result are clamped
/// to `0..=1`, so an out-of-range age or an inverted `start`/`end` pair stays
/// well defined.
#[must_use]
pub fn threshold_over_age(t: f32, start: f32, end: f32) -> f32 {
    let tt = clamp01(t);
    clamp01(start + (end - start) * tt)
}

/// Dissolve alpha for noise `n` against `threshold` with a `smoothstep` edge.
///
/// `n <= threshold` is fully eroded (`0.0`); `n >= threshold + edge_width` is
/// fully opaque (`1.0`); the transition is a `smoothstep`. A non-positive
/// `edge_width` degenerates to a hard cutoff at `threshold`.
#[must_use]
pub fn erosion_alpha(n: f32, threshold: f32, edge_width: f32) -> f32 {
    if edge_width < MIN_EDGE {
        return if n <= threshold { 0.0 } else { 1.0 };
    }
    smoothstep(threshold, threshold + edge_width, n)
}

/// Rim-glow weight in `0..=1` for the dissolve edge window of `n`.
///
/// The weight is a band that is zero at both ends of the erosion window
/// (`threshold` and `threshold + edge_width`) and peaks at the window centre,
/// formed by subtracting a rising `smoothstep` from an earlier rising
/// `smoothstep`. Outside the window it is zero. A non-positive `edge_width`
/// yields no glow.
#[must_use]
pub fn edge_glow_factor(n: f32, threshold: f32, edge_width: f32) -> f32 {
    if edge_width < MIN_EDGE {
        return 0.0;
    }
    let u = clamp01((n - threshold) / edge_width);
    let rising = smoothstep(0.0, 0.5, u);
    let falling = smoothstep(0.5, 1.0, u);
    clamp01(rising - falling)
}

/// A fully evaluated erosion result for one particle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErosionSample {
    /// Dissolve alpha in `0..=1` (`0` fully eroded, `1` fully opaque).
    pub alpha: f32,
    /// Emissive rim colour contribution in linear `RGB`.
    pub glow_rgb: [f32; 3],
}

/// Parameters controlling life-driven alpha erosion for a renderer (design §17).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErosionParams {
    /// Width of the `smoothstep` dissolve edge in noise units.
    pub edge_width: f32,
    /// Linear `RGB` colour of the dissolve rim glow.
    pub glow_color: [f32; 3],
    /// Scalar multiplier applied to the rim-glow band weight.
    pub glow_intensity: f32,
    /// Erosion threshold at birth (normalized age `0`).
    pub threshold_start: f32,
    /// Erosion threshold at death (normalized age `1`).
    pub threshold_end: f32,
}

impl ErosionParams {
    /// Creates erosion parameters from all fields.
    #[must_use]
    pub const fn new(
        edge_width: f32,
        glow_color: [f32; 3],
        glow_intensity: f32,
        threshold_start: f32,
        threshold_end: f32,
    ) -> Self {
        Self {
            edge_width,
            glow_color,
            glow_intensity,
            threshold_start,
            threshold_end,
        }
    }

    /// Evaluates the dissolve alpha and rim glow for normalized age `age` and
    /// per-particle noise `n`.
    ///
    /// The threshold is advanced by [`threshold_over_age`], the alpha by
    /// [`erosion_alpha`], and the glow by [`edge_glow_factor`] scaled by
    /// `glow_intensity` and multiplied into `glow_color`.
    #[must_use]
    pub fn evaluate(&self, age: f32, n: f32) -> ErosionSample {
        let threshold = threshold_over_age(age, self.threshold_start, self.threshold_end);
        let alpha = erosion_alpha(n, threshold, self.edge_width);
        let glow = edge_glow_factor(n, threshold, self.edge_width) * self.glow_intensity;
        ErosionSample {
            alpha,
            glow_rgb: [
                self.glow_color[0] * glow,
                self.glow_color[1] * glow,
                self.glow_color[2] * glow,
            ],
        }
    }

    /// Packs the parameters into their `std430` `vec4`-aligned scalar layout.
    ///
    /// Layout: `[glow_r, glow_g, glow_b, glow_intensity, edge_width,
    /// threshold_start, threshold_end, pad]` — two `vec4` slots, matching
    /// [`EROSION_PARAMS_STRIDE`]. The trailing scalar is padding.
    #[must_use]
    pub fn to_std430(&self) -> [f32; 8] {
        [
            self.glow_color[0],
            self.glow_color[1],
            self.glow_color[2],
            self.glow_intensity,
            self.edge_width,
            self.threshold_start,
            self.threshold_end,
            0.0,
        ]
    }
}

/// Total byte size of a `std430` storage buffer holding `count` packed
/// [`ErosionParams`] records.
///
/// Uses [`EROSION_PARAMS_STRIDE`] and the shared clamp-to-one-element rule from
/// [`storage_bytes`], so an empty set still yields a valid `GPU` binding.
#[must_use]
pub fn erosion_params_buffer_bytes(count: usize) -> usize {
    storage_bytes(EROSION_PARAMS_STRIDE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the float assertions in this module's tests.
    const CMP_EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    #[test]
    fn threshold_over_age_is_clamped_and_linear() {
        assert!(approx_eq(threshold_over_age(0.0, 0.2, 0.8), 0.2));
        assert!(approx_eq(threshold_over_age(1.0, 0.2, 0.8), 0.8));
        assert!(approx_eq(threshold_over_age(0.5, 0.2, 0.8), 0.5));
        // Out-of-range age clamps.
        assert!(approx_eq(threshold_over_age(-1.0, 0.2, 0.8), 0.2));
        assert!(approx_eq(threshold_over_age(2.0, 0.2, 0.8), 0.8));
        // Result clamps into 0..=1 even for an aggressive bias.
        assert!(approx_eq(threshold_over_age(1.0, 0.0, 2.0), 1.0));
    }

    #[test]
    fn smoothstep_endpoints_are_zero_and_one() {
        assert!(approx_eq(smoothstep(0.0, 1.0, -0.5), 0.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 0.0), 0.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 1.0), 1.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 1.5), 1.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 0.5), 0.5));
        // Degenerate interval is a hard step.
        assert!(approx_eq(smoothstep(0.5, 0.5, 0.4), 0.0));
        assert!(approx_eq(smoothstep(0.5, 0.5, 0.5), 1.0));
    }

    #[test]
    fn erosion_alpha_endpoints_and_hard_cutoff() {
        assert!(approx_eq(erosion_alpha(0.3, 0.4, 0.2), 0.0));
        assert!(approx_eq(erosion_alpha(0.4, 0.4, 0.2), 0.0));
        assert!(approx_eq(erosion_alpha(0.6, 0.4, 0.2), 1.0));
        assert!(approx_eq(erosion_alpha(0.5, 0.4, 0.2), 0.5));
        // Zero edge width collapses to a hard cutoff.
        assert!(approx_eq(erosion_alpha(0.4, 0.4, 0.0), 0.0));
        assert!(approx_eq(erosion_alpha(0.5, 0.4, 0.0), 1.0));
    }

    #[test]
    fn alpha_is_monotone_non_increasing_over_age() {
        let params = ErosionParams::new(0.2, [1.0, 0.5, 0.25], 2.0, 0.0, 1.0);
        let n = 0.5;
        let mut prev = 2.0;
        for step in 0..=20u32 {
            let age = f32::from(u16::try_from(step).unwrap_or(0)) / 20.0;
            let sample = params.evaluate(age, n);
            assert!(sample.alpha <= prev + CMP_EPS);
            prev = sample.alpha;
        }
        // Sanity: fully opaque at birth, fully eroded at death for this noise.
        assert!(approx_eq(params.evaluate(0.0, n).alpha, 1.0));
        assert!(approx_eq(params.evaluate(1.0, n).alpha, 0.0));
    }

    #[test]
    fn edge_glow_is_a_band_peaking_at_the_edge() {
        let threshold = 0.3;
        let edge = 0.4;
        let at_start = edge_glow_factor(threshold, threshold, edge);
        let at_end = edge_glow_factor(threshold + edge, threshold, edge);
        let at_center = edge_glow_factor(threshold + 0.5 * edge, threshold, edge);
        assert!(approx_eq(at_start, 0.0));
        assert!(approx_eq(at_end, 0.0));
        assert!(approx_eq(at_center, 1.0));
        // Strictly banded: interior samples exceed the edges.
        assert!(at_center > at_start + CMP_EPS);
        assert!(at_center > at_end + CMP_EPS);
        // Outside the window there is no glow.
        assert!(approx_eq(edge_glow_factor(0.0, threshold, edge), 0.0));
        assert!(approx_eq(edge_glow_factor(1.0, threshold, edge), 0.0));
        // Non-positive edge width yields no glow.
        assert!(approx_eq(edge_glow_factor(0.5, 0.3, 0.0), 0.0));
    }

    #[test]
    fn hash_noise_is_deterministic_and_normalized() {
        for seed in 0..64u32 {
            let a = hash_noise01(seed);
            let b = hash_noise01(seed);
            assert!(approx_eq(a, b));
            assert!(a >= 0.0);
            assert!(a <= 1.0);
        }
        // Distinct seeds decorrelate.
        assert!(!approx_eq(hash_noise01(1), hash_noise01(2)));
        assert!(!approx_eq(hash_noise01(100), hash_noise01(101)));
    }

    #[test]
    fn evaluate_scales_glow_color_by_intensity() {
        let params = ErosionParams::new(0.4, [0.4, 0.6, 0.8], 3.0, 0.0, 1.0);
        let threshold = 0.3;
        let n = threshold + 0.2; // window centre => band weight 1.0.
        let sample = params.evaluate(threshold, n);
        // glow = 1.0 * intensity = 3.0.
        assert!(approx_eq(sample.glow_rgb[0], 0.4 * 3.0));
        assert!(approx_eq(sample.glow_rgb[1], 0.6 * 3.0));
        assert!(approx_eq(sample.glow_rgb[2], 0.8 * 3.0));
    }

    #[test]
    fn std430_packing_layout_and_bytes() {
        assert_eq!(EROSION_PARAMS_STRIDE, 32);
        let params = ErosionParams::new(0.2, [0.1, 0.2, 0.3], 1.5, 0.25, 0.75);
        let packed = params.to_std430();
        assert!(approx_eq(packed[0], 0.1));
        assert!(approx_eq(packed[3], 1.5));
        assert!(approx_eq(packed[4], 0.2));
        assert!(approx_eq(packed[6], 0.75));
        assert!(approx_eq(packed[7], 0.0));

        assert_eq!(erosion_params_buffer_bytes(3), 96);
        // Empty set still reserves one element.
        assert_eq!(erosion_params_buffer_bytes(0), EROSION_PARAMS_STRIDE);
    }
}
