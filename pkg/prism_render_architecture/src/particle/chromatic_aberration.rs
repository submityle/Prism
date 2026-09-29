//! Screen-space `chromatic` aberration model: per-channel radial `RGB` split
//! (design §16-§21).
//!
//! This module is the deterministic `CPU` reference for the post-process
//! `chromatic` aberration that production `VFX` stacks layer over lens-like
//! renderers (bloom halos, energy shields, glitch transitions). Real lenses
//! refract short and long wavelengths by slightly different amounts, so the
//! red, green, and blue records of a point land at slightly different radii
//! from the optical center. The screen-space approximation reproduces that
//! look by sampling the scene color three times per pixel, offsetting each
//! `RGB` channel along the radial direction away from an optical center, with
//! the offset growing toward the frame edges where lens dispersion is worst.
//!
//! It owns four independent pieces the compositor combines:
//!
//! 1. a center-relative *radial* vector, `uv - center`, whose direction points
//!    outward from the optical center and whose length is the sampled radius;
//! 2. a rational-polynomial *radial intensity* that is weakest at the center
//!    and strengthens toward the edges, soft-clamped into `[0, 1]`;
//! 3. three per-channel *offsets* where red pushes outward, blue pulls inward
//!    (or the reverse via the sign of the scales), and green stays on the
//!    baseline; and
//! 4. per-channel sampled `UV`s clamped into the `[0, 1]` texture domain so a
//!    split sample never reads outside the source target.
//!
//! Determinism rules (design §29) are inherited: the only floating-point
//! primitive beyond ordinary arithmetic is `f32::sqrt` (the radius). There are
//! no transcendental calls (`sin` / `cos` / `tan` / `atan` / `exp` / `ln` /
//! `pow`); powers are unrolled integer multiplies, the intensity is the
//! rational polynomial `r^2 (1 + k r^2) / (1 + r^2)` shaped by the
//! multiply-only smoothstep `t^2 (3 - 2 t)`, and every sample clamps into the
//! unit square. A future `GPU` kernel reproduces these results bit for bit, and
//! the `std430` packing anticipates that kernel's uniform block.

use crate::particle::gpu_layout::VEC4_STRIDE;
use alloc::vec::Vec;

/// Number of scalar fields packed into the [`ChromaParams`] `std430` block:
/// the five scalars plus the two `center` components.
const CHROMA_FIELD_COUNT: usize = 7;

/// Byte size of the `std430` packing of [`ChromaParams`]: [`CHROMA_FIELD_COUNT`]
/// scalars rounded up to whole `vec4` slots so the block honors the 16-byte
/// `std430` base alignment. Seven scalars occupy two `vec4` slots (32 bytes),
/// leaving a one-scalar padding tail.
pub const CHROMA_STD430_SIZE: usize = CHROMA_FIELD_COUNT.div_ceil(4) * VEC4_STRIDE;

/// Absolute tolerance for the `f32` equality comparisons used by the tests;
/// direct `==` on floating point is intentionally avoided.
#[cfg(test)]
const CMP_EPS: f32 = 1.0e-6;

/// Clamps a scalar into the closed unit interval `[0, 1]`.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Clamps a `UV` coordinate component-wise into the `[0, 1]` texture domain so
/// a split sample never reads outside the source target.
#[must_use]
fn clamp_uv(uv: [f32; 2]) -> [f32; 2] {
    [clamp01(uv[0]), clamp01(uv[1])]
}

/// Integer power `base^exp` built from an unrolled multiply loop, avoiding the
/// forbidden `powf`/`powi` transcendental-style path while staying bit-exact.
#[must_use]
fn pow_u32(base: f32, exp: u32) -> f32 {
    let mut acc = 1.0;
    for _ in 0..exp {
        acc *= base;
    }
    acc
}

/// The smoothstep shaper `t^2 (3 - 2 t)` after clamping `t` into `[0, 1]`: a
/// multiply-only Hermite polynomial with zero first derivative at `0` and `1`,
/// so the intensity ramps softly toward the saturated edge without any
/// transcendental call. It is monotonically non-decreasing on `[0, 1]`.
#[must_use]
fn smoothstep01(t: f32) -> f32 {
    let c = clamp01(t);
    c * c * (3.0 - 2.0 * c)
}

/// Optical-dispersion parameters for the screen-space `chromatic` aberration
/// split (design §16-§21).
///
/// The `center` is the optical center in `UV` space (typically `[0.5, 0.5]`).
/// `strength` is the global offset gain; `radial_falloff_k` shapes how fast the
/// intensity climbs toward the edges; and `r_scale` / `g_scale` / `b_scale` are
/// the per-channel signed radial gains (red outward, blue inward when the
/// scales share a sign, green on the baseline).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChromaParams {
    /// Global gain applied to every channel's radial offset.
    pub strength: f32,
    /// Edge-emphasis coefficient of the rational-polynomial radial intensity.
    pub radial_falloff_k: f32,
    /// Signed radial gain of the red channel (positive pushes outward).
    pub r_scale: f32,
    /// Signed radial gain of the green channel (baseline, typically `0`).
    pub g_scale: f32,
    /// Signed radial gain of the blue channel (positive pulls inward).
    pub b_scale: f32,
    /// Optical center in `UV` space that the split radiates from.
    pub center: [f32; 2],
}

impl ChromaParams {
    /// Builds a parameter set from its raw fields.
    #[must_use]
    pub fn new(
        strength: f32,
        radial_falloff_k: f32,
        r_scale: f32,
        g_scale: f32,
        b_scale: f32,
        center: [f32; 2],
    ) -> Self {
        Self {
            strength,
            radial_falloff_k,
            r_scale,
            g_scale,
            b_scale,
            center,
        }
    }

    /// The rational-polynomial radial intensity at radius `r`, soft-clamped into
    /// `[0, 1]`.
    ///
    /// The raw curve `r^2 (1 + k r^2) / (1 + r^2)` is `0` at the optical center
    /// (its minimum) and increases monotonically with `r`, so dispersion is
    /// weakest at the center and strongest toward the edges. A non-negative
    /// `k` (negatives are floored to `0`) sharpens the edge emphasis, and
    /// [`smoothstep01`] both bounds the result to `[0, 1]` and gives it a soft
    /// saturated edge.
    #[must_use]
    pub fn radial_intensity(&self, r: f32) -> f32 {
        let k = self.radial_falloff_k.max(0.0);
        let r2 = pow_u32(r, 2);
        let num = r2 * (1.0 + k * r2);
        let den = 1.0 + r2;
        smoothstep01(num / den)
    }

    /// The three per-channel radial `UV` offsets `[red, green, blue]`.
    ///
    /// Each offset is the outward radial vector `uv - center` scaled by the
    /// global `strength`, the shared [`radial_intensity`](Self::radial_intensity),
    /// and the channel's signed gain. Red uses `+r_scale` (outward), blue uses
    /// `-b_scale` (inward, so red and blue separate in opposite directions when
    /// the scales share a sign), and green uses `+g_scale` (baseline). At the
    /// optical center the radial vector is zero, so all three offsets vanish.
    #[must_use]
    pub fn channel_offsets(&self, uv: [f32; 2]) -> [[f32; 2]; 3] {
        let dx = uv[0] - self.center[0];
        let dy = uv[1] - self.center[1];
        let r = (dx * dx + dy * dy).sqrt();
        let base = self.strength * self.radial_intensity(r);
        let r_gain = base * self.r_scale;
        let g_gain = base * self.g_scale;
        let b_gain = -base * self.b_scale;
        [
            [dx * r_gain, dy * r_gain],
            [dx * g_gain, dy * g_gain],
            [dx * b_gain, dy * b_gain],
        ]
    }

    /// The three per-channel sampled `UV`s `[red, green, blue]`, each the source
    /// `uv` shifted by its channel offset and clamped into the `[0, 1]` texture
    /// domain.
    #[must_use]
    pub fn sample_uvs(&self, uv: [f32; 2]) -> [[f32; 2]; 3] {
        let offsets = self.channel_offsets(uv);
        [
            clamp_uv([uv[0] + offsets[0][0], uv[1] + offsets[0][1]]),
            clamp_uv([uv[0] + offsets[1][0], uv[1] + offsets[1][1]]),
            clamp_uv([uv[0] + offsets[2][0], uv[1] + offsets[2][1]]),
        ]
    }

    /// Batches [`sample_uvs`](Self::sample_uvs) over many source `UV`s, one
    /// `[red, green, blue]` triple per input, preserving order.
    #[must_use]
    pub fn sample_uv_batch(&self, uvs: &[[f32; 2]]) -> Vec<[[f32; 2]; 3]> {
        uvs.iter().map(|&uv| self.sample_uvs(uv)).collect()
    }

    /// Packs the parameters into their `std430` uniform-block byte layout.
    ///
    /// The five scalars followed by the two `center` components fill the first
    /// [`CHROMA_FIELD_COUNT`] scalar slots; the remaining padding tail (one
    /// scalar) stays zero so the block is a whole number of `vec4` slots.
    #[must_use]
    pub fn to_std430(&self) -> [u8; CHROMA_STD430_SIZE] {
        let fields = [
            self.strength,
            self.radial_falloff_k,
            self.r_scale,
            self.g_scale,
            self.b_scale,
            self.center[0],
            self.center[1],
        ];
        let mut bytes = [0u8; CHROMA_STD430_SIZE];
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

    fn sample_params() -> ChromaParams {
        ChromaParams::new(1.0, 2.0, 0.6, 0.0, 0.4, [0.5, 0.5])
    }

    fn radial_len(offset: [f32; 2]) -> f32 {
        (offset[0] * offset[0] + offset[1] * offset[1]).sqrt()
    }

    #[test]
    fn optical_center_has_zero_offsets() {
        let p = sample_params();
        let offsets = p.channel_offsets(p.center);
        assert!(approx2(offsets[0], [0.0, 0.0]));
        assert!(approx2(offsets[1], [0.0, 0.0]));
        assert!(approx2(offsets[2], [0.0, 0.0]));
        let uvs = p.sample_uvs(p.center);
        assert!(approx2(uvs[0], p.center));
        assert!(approx2(uvs[1], p.center));
        assert!(approx2(uvs[2], p.center));
    }

    #[test]
    fn edge_offsets_grow_beyond_center() {
        let p = sample_params();
        let near = radial_len(p.channel_offsets([0.55, 0.5])[0]);
        let far = radial_len(p.channel_offsets([0.95, 0.5])[0]);
        assert!(far > near);
        assert!(near > 0.0);
    }

    #[test]
    fn red_and_blue_separate_in_opposite_directions() {
        let p = sample_params();
        let offsets = p.channel_offsets([0.9, 0.5]);
        // Radial points in +x here, so red offsets outward (+x) and blue
        // inward (-x): their x components must carry opposite signs.
        assert!(offsets[0][0] > 0.0);
        assert!(offsets[2][0] < 0.0);
        assert!(offsets[0][0] * offsets[2][0] < 0.0);
    }

    #[test]
    fn green_stays_on_the_baseline_uv() {
        let p = sample_params();
        let uv = [0.85, 0.35];
        let offsets = p.channel_offsets(uv);
        assert!(approx2(offsets[1], [0.0, 0.0]));
        let uvs = p.sample_uvs(uv);
        assert!(approx2(uvs[1], uv));
    }

    #[test]
    fn sampled_uvs_stay_in_unit_square() {
        // A large strength drives the raw offsets well past the frame; every
        // sampled UV must still clamp into [0, 1].
        let p = ChromaParams::new(50.0, 4.0, 3.0, 1.0, 2.5, [0.5, 0.5]);
        for uv in [[0.0, 0.0], [1.0, 1.0], [0.02, 0.98], [0.97, 0.03]] {
            let uvs = p.sample_uvs(uv);
            for channel in uvs {
                assert!((0.0..=1.0).contains(&channel[0]));
                assert!((0.0..=1.0).contains(&channel[1]));
            }
        }
    }

    #[test]
    fn radial_intensity_is_minimal_at_center() {
        let p = sample_params();
        assert!(approx(p.radial_intensity(0.0), 0.0));
    }

    #[test]
    fn radial_intensity_is_monotonic_in_radius() {
        let p = sample_params();
        let samples = [0.0, 0.1, 0.2, 0.35, 0.5, 0.65];
        let mut prev = p.radial_intensity(samples[0]);
        for &r in &samples[1..] {
            let cur = p.radial_intensity(r);
            assert!(cur > prev);
            prev = cur;
        }
    }

    #[test]
    fn radial_intensity_stays_within_unit_range() {
        let p = ChromaParams::new(1.0, 100.0, 1.0, 0.0, 1.0, [0.5, 0.5]);
        for &r in &[0.0, 0.25, 0.5, 0.707, 5.0] {
            let v = p.radial_intensity(r);
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn negative_falloff_k_is_floored_to_zero() {
        let neg = ChromaParams::new(1.0, -8.0, 0.6, 0.0, 0.4, [0.5, 0.5]);
        let zero = ChromaParams::new(1.0, 0.0, 0.6, 0.0, 0.4, [0.5, 0.5]);
        assert!(approx(
            neg.radial_intensity(0.4),
            zero.radial_intensity(0.4)
        ));
    }

    #[test]
    fn sampling_is_deterministic() {
        let p = sample_params();
        let uv = [0.73, 0.41];
        assert_eq!(p.sample_uvs(uv), p.sample_uvs(uv));
        assert_eq!(p.channel_offsets(uv), p.channel_offsets(uv));
    }

    #[test]
    fn batch_matches_scalar_sampling() {
        let p = sample_params();
        let uvs = [[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]];
        let batch = p.sample_uv_batch(&uvs);
        assert_eq!(batch.len(), uvs.len());
        for (triple, uv) in batch.iter().zip(uvs.iter()) {
            assert_eq!(*triple, p.sample_uvs(*uv));
        }
    }

    #[test]
    fn std430_packing_is_two_vec4_slots() {
        let p = sample_params();
        let bytes = p.to_std430();
        assert_eq!(bytes.len(), CHROMA_STD430_SIZE);
        assert_eq!(CHROMA_STD430_SIZE, 2 * VEC4_STRIDE);
        assert_eq!(CHROMA_STD430_SIZE % VEC4_STRIDE, 0);
        assert_eq!(storage_bytes(CHROMA_STD430_SIZE, 1), CHROMA_STD430_SIZE);
    }

    #[test]
    fn std430_round_trips_the_seven_scalars() {
        let p = ChromaParams::new(1.5, 2.25, 0.75, -0.5, 0.125, [0.25, 0.6]);
        let bytes = p.to_std430();
        let fields = [
            p.strength,
            p.radial_falloff_k,
            p.r_scale,
            p.g_scale,
            p.b_scale,
            p.center[0],
            p.center[1],
        ];
        for (slot, value) in bytes.chunks_exact(4).zip(fields.iter()) {
            let mut word = [0u8; 4];
            word.copy_from_slice(slot);
            assert!(approx(f32::from_le_bytes(word), *value));
        }
        // The padding tail (one scalar) must be zero.
        for byte in &bytes[CHROMA_FIELD_COUNT * 4..] {
            assert_eq!(*byte, 0);
        }
    }
}
