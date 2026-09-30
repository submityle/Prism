//! Texture-space anisotropic sampling-footprint geometry (design §16-§17).
//!
//! When a billboarded or ribbon particle is shaded, the texture coordinate
//! (`UV`) it reads is a function of the screen pixel, so a single pixel covers
//! an *ellipse* in texture space rather than a square texel. The screen-space
//! partial derivatives `ddx`/`ddy` (how the `UV` moves for a one-pixel step in
//! screen x / y, measured in texels) describe that ellipse: the classic
//! anisotropic-filtering setup a `GPU` sampler performs implicitly, reproduced
//! here as a device-free contract so the `CPU` reference and a `WGSL` kernel
//! agree bit for bit.
//!
//! This module is deliberately **orthogonal to [`super::lod`]**: that sibling
//! owns the *particle-simulation* quality ladder (how much work a system spends
//! per frame), whereas this file only computes *texture-sampling* geometry —
//! the ellipse axes, the trilinear / anisotropic `LOD` (`mip` level) the axes
//! imply, and the along-axis sample offsets. It shares no types with `lod` and
//! never touches the deformation budget.
//!
//! The ellipse is recovered from the `2x2` metric `M = Jᵀ·J` where the Jacobian
//! `J`'s columns are `ddx` and `ddy`; its eigenvalues are the squared ellipse
//! half-axes, obtained in closed form using only a single `f32::sqrt`. The
//! `LOD` uses a `log2` built purely from the `IEEE754` `f32` bit pattern — no
//! transcendental (`log2`/`ln`) call is made anywhere in this module.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE, VEC4_STRIDE};

/// Epsilon for degeneracy guards; `f32` magnitudes below this are treated as
/// zero so a vanishing gradient or minor axis never divides by zero.
const CMP_EPS: f32 = 1e-6;

/// Quadratic correction coefficient for the mantissa term of [`log2_via_bits`]:
/// `log2(1 + f) ≈ f + LOG2_MANTISSA_K * f * (1 - f)` for `f` in `[0, 1)`. The
/// value reproduces exact powers of two (`f == 0`) and keeps the residual well
/// under `0.01` across an octave.
const LOG2_MANTISSA_K: f32 = 0.346_573_6;

/// `2^23`, the width of the `f32` mantissa field, as a float divisor.
const MANTISSA_SCALE: f32 = 8_388_608.0;

/// `log2` result for non-positive / subnormal inputs: far below any physical
/// `mip` level, so a degenerate footprint pins to the coarsest sampling after
/// clamping instead of yielding `-inf`/`NaN`.
const LOG2_ZERO_FLOOR: f32 = -1000.0;

/// Fixed-point scale used to turn the fractional `anisotropy` ratio into an
/// integer numerator so the sample count is a `div_ceil` (see
/// [`Footprint::sample_count`]).
const SAMPLE_FIXED_SCALE: f32 = 256.0;

/// Integer form of [`SAMPLE_FIXED_SCALE`], the `div_ceil` denominator.
const SAMPLE_FIXED_SCALE_U32: u32 = 256;

/// `std430` byte size of a serialized [`Footprint`]: two `vec4` slots (`32`
/// bytes). The first slot holds `major_dir` (a `vec2`) then `major_len` and
/// `minor_len`; the second slot holds `anisotropy` and a zeroed padding tail.
pub const ANISO_FOOTPRINT_STD430_SIZE: usize = 2 * VEC4_STRIDE;

/// Screen-space partial derivatives of a texture coordinate, in texels.
///
/// `ddx` is the `UV` change for a one-pixel step along screen x, `ddy` the
/// change for a one-pixel step along screen y. Together they are the columns of
/// the Jacobian whose metric this module diagonalizes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gradients {
    /// `UV` derivative with respect to screen x, in texels.
    pub ddx: [f32; 2],
    /// `UV` derivative with respect to screen y, in texels.
    pub ddy: [f32; 2],
}

/// The anisotropic sampling ellipse recovered from a [`Gradients`] pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Footprint {
    /// Half-length of the ellipse's long axis, in texels (`= sqrt(λ_major)`).
    pub major_len: f32,
    /// Half-length of the ellipse's short axis, in texels (`= sqrt(λ_minor)`).
    pub minor_len: f32,
    /// Unit direction of the long axis: the metric eigenvector for `λ_major`.
    pub major_dir: [f32; 2],
    /// `major_len / minor_len`, clamped to `[1, max_anisotropy]`.
    pub anisotropy: f32,
}

impl Footprint {
    /// Builds the footprint ellipse from a [`Gradients`] pair.
    ///
    /// Forms the symmetric metric `M = [[A, B], [B, C]]` with `A = ddx·ddx`,
    /// `B = ddx·ddy`, `C = ddy·ddy`, then solves its eigenvalues in closed form:
    /// `λ = (A + C)/2 ± sqrt(((A - C)/2)² + B²)`. The half-axes are the square
    /// roots of the eigenvalues, the long-axis direction is the `λ_major`
    /// eigenvector, and `anisotropy` is the axis ratio clamped into
    /// `[1, max_anisotropy]`. A vanishing gradient collapses to a centered,
    /// isotropic footprint with unit direction `[1, 0]`.
    #[must_use]
    pub fn from_gradients(g: &Gradients, max_anisotropy: f32) -> Self {
        let a = g.ddx[0] * g.ddx[0] + g.ddx[1] * g.ddx[1];
        let c = g.ddy[0] * g.ddy[0] + g.ddy[1] * g.ddy[1];
        let b = g.ddx[0] * g.ddy[0] + g.ddx[1] * g.ddy[1];

        let mean = (a + c) * 0.5;
        let diff = (a - c) * 0.5;
        let disc = (diff * diff + b * b).sqrt();
        let lam_major = (mean + disc).max(0.0);
        let lam_minor = (mean - disc).max(0.0);

        let major_len = lam_major.sqrt();
        let minor_len = lam_minor.sqrt();

        if major_len < CMP_EPS {
            return Self {
                major_len: 0.0,
                minor_len: 0.0,
                major_dir: [1.0, 0.0],
                anisotropy: 1.0,
            };
        }

        // Two algebraically equivalent eigenvectors for `λ_major`; the one with
        // the larger squared norm is the numerically better conditioned pick.
        let cand_a = [b, lam_major - a];
        let cand_c = [lam_major - c, b];
        let norm_a = cand_a[0] * cand_a[0] + cand_a[1] * cand_a[1];
        let norm_c = cand_c[0] * cand_c[0] + cand_c[1] * cand_c[1];
        let raw = if norm_a >= norm_c { cand_a } else { cand_c };
        let major_dir = normalize2(raw);

        let max_a = max_anisotropy.max(1.0);
        let anisotropy = if minor_len < CMP_EPS {
            max_a
        } else {
            (major_len / minor_len).clamp(1.0, max_a)
        };

        Self {
            major_len,
            minor_len,
            major_dir,
            anisotropy,
        }
    }

    /// Anisotropic `LOD`: the `log2` of the short-axis half-length.
    ///
    /// Anisotropic filtering selects the `mip` level from the *minor* axis and
    /// then walks samples along the major axis, so the returned level is finer
    /// (smaller) than the isotropic [`lod_trilinear`] whenever the footprint is
    /// elongated.
    #[must_use]
    pub fn lod_anisotropic(&self) -> f32 {
        log2_via_bits(self.minor_len)
    }

    /// Number of samples to take along the major axis: `ceil(anisotropy)`,
    /// always at least one.
    ///
    /// `anisotropy` is already clamped to `max_anisotropy` in
    /// [`Footprint::from_gradients`], so the count never exceeds
    /// `ceil(max_anisotropy)`. The ceiling is expressed as a `div_ceil` over a
    /// fixed-point numerator so no transcendental rounding is used.
    #[must_use]
    pub fn sample_count(&self) -> u32 {
        let scaled = (self.anisotropy * SAMPLE_FIXED_SCALE).floor();
        let fixed = f32_to_u32(scaled);
        fixed.div_ceil(SAMPLE_FIXED_SCALE_U32).max(1)
    }

    /// Sample positions spread symmetrically along the major axis about
    /// `center_uv`.
    ///
    /// Returns exactly [`Footprint::sample_count`] points. A single sample sits
    /// at the center; multiple samples fan out to `± (major_len - minor_len)`
    /// along `major_dir`, so `offsets[i]` and `offsets[n - 1 - i]` are mirror
    /// images about the center and their mean is the center. An isotropic
    /// footprint (`major_len == minor_len`) collapses every offset onto the
    /// center.
    #[must_use]
    pub fn sample_offsets(&self, center_uv: [f32; 2]) -> Vec<[f32; 2]> {
        let n = self.sample_count();
        if n == 1 {
            return alloc::vec![center_uv];
        }
        let reach = (self.major_len - self.minor_len).max(0.0);
        let denom = u32_to_f32(n - 1);
        (0..n)
            .map(|i| {
                let t = u32_to_f32(i) / denom * 2.0 - 1.0;
                let d = t * reach;
                [
                    center_uv[0] + self.major_dir[0] * d,
                    center_uv[1] + self.major_dir[1] * d,
                ]
            })
            .collect()
    }
}

/// Isotropic (trilinear) `LOD`: the `log2` of the longest gradient length.
///
/// This is the fallback a plain trilinear sampler uses — it picks the `mip`
/// level from whichever screen axis stretches the `UV` most, ignoring
/// direction. Compare with [`Footprint::lod_anisotropic`], which sharpens the
/// level using the minor axis.
#[must_use]
pub fn lod_trilinear(g: &Gradients) -> f32 {
    let dx = (g.ddx[0] * g.ddx[0] + g.ddx[1] * g.ddx[1]).sqrt();
    let dy = (g.ddy[0] * g.ddy[0] + g.ddy[1] * g.ddy[1]).sqrt();
    log2_via_bits(dx.max(dy))
}

/// Serializes a [`Footprint`] into its `std430` byte layout
/// ([`ANISO_FOOTPRINT_STD430_SIZE`] bytes).
///
/// Slot layout: `major_dir.x`, `major_dir.y`, `major_len`, `minor_len` fill the
/// first `vec4`; `anisotropy` occupies the first lane of the second `vec4` and
/// the remaining twelve bytes are zero padding.
#[must_use]
pub fn to_std430(footprint: &Footprint) -> [u8; ANISO_FOOTPRINT_STD430_SIZE] {
    let mut bytes = [0u8; ANISO_FOOTPRINT_STD430_SIZE];
    bytes[0..U32_STRIDE].copy_from_slice(&footprint.major_dir[0].to_le_bytes());
    bytes[U32_STRIDE..2 * U32_STRIDE].copy_from_slice(&footprint.major_dir[1].to_le_bytes());
    bytes[2 * U32_STRIDE..3 * U32_STRIDE].copy_from_slice(&footprint.major_len.to_le_bytes());
    bytes[3 * U32_STRIDE..4 * U32_STRIDE].copy_from_slice(&footprint.minor_len.to_le_bytes());
    bytes[4 * U32_STRIDE..5 * U32_STRIDE].copy_from_slice(&footprint.anisotropy.to_le_bytes());
    bytes
}

/// Total `GPU` storage size for `count` serialized footprints, clamped up to a
/// single element so an empty batch still yields a valid storage binding.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(ANISO_FOOTPRINT_STD430_SIZE, count)
}

/// Normalizes a `2`-vector, returning `[1, 0]` for a vanishing input so a
/// degenerate direction is still a valid unit vector.
#[must_use]
fn normalize2(v: [f32; 2]) -> [f32; 2] {
    let len_sq = v[0] * v[0] + v[1] * v[1];
    if len_sq < CMP_EPS {
        return [1.0, 0.0];
    }
    let inv = 1.0 / len_sq.sqrt();
    [v[0] * inv, v[1] * inv]
}

/// `log2(x)` built purely from the `IEEE754` `f32` bit pattern.
///
/// For a positive normal `x = (1 + f) * 2^e` the biased exponent field yields
/// `e` exactly and the mantissa fraction `f` in `[0, 1)` is closed with the
/// quadratic `log2(1 + f) ≈ f + LOG2_MANTISSA_K * f * (1 - f)`, so powers of two
/// are reproduced exactly. Non-positive or subnormal inputs return
/// [`LOG2_ZERO_FLOOR`]. No `log2`/`ln` floating-point function is called.
#[must_use]
fn log2_via_bits(x: f32) -> f32 {
    if x <= 0.0 {
        return LOG2_ZERO_FLOOR;
    }
    let bits = x.to_bits();
    let exp_field = i32::try_from((bits >> 23) & 0xff).unwrap_or(0);
    if exp_field == 0 {
        return LOG2_ZERO_FLOOR;
    }
    let mantissa = bits & 0x007f_ffff;
    #[expect(
        clippy::cast_precision_loss,
        reason = "mantissa < 2^23 is represented exactly in f32"
    )]
    let frac = mantissa as f32 / MANTISSA_SCALE;
    let log_mant = frac + LOG2_MANTISSA_K * frac * (1.0 - frac);
    #[expect(
        clippy::cast_precision_loss,
        reason = "an unbiased f32 exponent lies in [-126, 127], exact in f32"
    )]
    let exponent = (exp_field - 127) as f32;
    exponent + log_mant
}

/// Truncates a known-non-negative, finite `f32` to `u32`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "callers pass a floored, non-negative, small magnitude"
)]
#[must_use]
fn f32_to_u32(x: f32) -> u32 {
    x as u32
}

/// Widens a small `u32` (a sample index or count) to `f32`.
#[expect(
    clippy::cast_precision_loss,
    reason = "sample counts are small, well below the f32 exact-integer limit"
)]
#[must_use]
fn u32_to_f32(x: u32) -> f32 {
    x as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only float tolerance (production code compares with [`CMP_EPS`]).
    const TEST_EPS: f32 = 1e-4;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn decode_f32(bytes: &[u8], offset: usize) -> f32 {
        let mut word = [0u8; 4];
        word.copy_from_slice(&bytes[offset..offset + 4]);
        f32::from_le_bytes(word)
    }

    #[test]
    fn isotropic_gradient_has_unit_anisotropy() {
        let g = Gradients {
            ddx: [1.0, 0.0],
            ddy: [0.0, 1.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        assert!(close(f.anisotropy, 1.0, TEST_EPS));
        assert!(close(f.major_len, 1.0, TEST_EPS));
        assert!(close(f.minor_len, 1.0, TEST_EPS));
    }

    #[test]
    fn stretched_gradient_has_anisotropy_above_one() {
        let g = Gradients {
            ddx: [4.0, 0.0],
            ddy: [0.0, 1.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        assert!(f.anisotropy > 1.0);
        assert!(close(f.anisotropy, 4.0, TEST_EPS));
    }

    #[test]
    fn anisotropy_is_clamped_to_max() {
        let g = Gradients {
            ddx: [100.0, 0.0],
            ddy: [0.0, 1.0],
        };
        let f = Footprint::from_gradients(&g, 8.0);
        assert!(close(f.anisotropy, 8.0, TEST_EPS));
    }

    #[test]
    fn max_anisotropy_below_one_is_lifted_to_one() {
        let g = Gradients {
            ddx: [4.0, 0.0],
            ddy: [0.0, 1.0],
        };
        let f = Footprint::from_gradients(&g, 0.25);
        assert!(close(f.anisotropy, 1.0, TEST_EPS));
    }

    #[test]
    fn sample_count_is_ceil_of_anisotropy() {
        let g = Gradients {
            ddx: [3.5, 0.0],
            ddy: [0.0, 1.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        assert_eq!(f.sample_count(), 4);
    }

    #[test]
    fn sample_count_is_clamped_by_max_anisotropy() {
        let g = Gradients {
            ddx: [100.0, 0.0],
            ddy: [0.0, 1.0],
        };
        let f = Footprint::from_gradients(&g, 6.0);
        assert_eq!(f.sample_count(), 6);
    }

    #[test]
    fn isotropic_footprint_takes_one_sample() {
        let g = Gradients {
            ddx: [1.0, 0.0],
            ddy: [0.0, 1.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        assert_eq!(f.sample_count(), 1);
    }

    #[test]
    fn sample_offsets_length_matches_count() {
        let g = Gradients {
            ddx: [5.0, 0.0],
            ddy: [0.0, 1.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        let offsets = f.sample_offsets([0.5, 0.5]);
        assert_eq!(u32::try_from(offsets.len()).unwrap(), f.sample_count());
    }

    #[test]
    fn sample_offsets_are_symmetric_about_center() {
        let g = Gradients {
            ddx: [7.0, 0.0],
            ddy: [0.0, 1.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        let center = [0.25, 0.75];
        let offsets = f.sample_offsets(center);
        let n = offsets.len();
        for i in 0..n {
            let lo = offsets[i];
            let hi = offsets[n - 1 - i];
            assert!(close(lo[0] + hi[0], 2.0 * center[0], TEST_EPS));
            assert!(close(lo[1] + hi[1], 2.0 * center[1], TEST_EPS));
        }
    }

    #[test]
    fn sample_offsets_lie_along_major_axis() {
        let g = Gradients {
            ddx: [6.0, 2.0],
            ddy: [1.0, 1.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        let center = [0.1, 0.2];
        let offsets = f.sample_offsets(center);
        for o in &offsets {
            let rel = [o[0] - center[0], o[1] - center[1]];
            // Cross product with the unit major direction must vanish.
            let cross = rel[0] * f.major_dir[1] - rel[1] * f.major_dir[0];
            assert!(close(cross, 0.0, TEST_EPS));
        }
    }

    #[test]
    fn isotropic_offsets_collapse_to_center() {
        let g = Gradients {
            ddx: [1.0, 0.0],
            ddy: [0.0, 1.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        let center = [0.3, 0.6];
        let offsets = f.sample_offsets(center);
        assert_eq!(offsets.len(), 1);
        assert!(close(offsets[0][0], center[0], TEST_EPS));
        assert!(close(offsets[0][1], center[1], TEST_EPS));
    }

    #[test]
    fn log2_is_exact_on_powers_of_two() {
        assert!(close(log2_via_bits(1.0), 0.0, 1e-6));
        assert!(close(log2_via_bits(2.0), 1.0, 1e-6));
        assert!(close(log2_via_bits(4.0), 2.0, 1e-6));
        assert!(close(log2_via_bits(0.5), -1.0, 1e-6));
        assert!(close(log2_via_bits(1024.0), 10.0, 1e-6));
    }

    #[test]
    fn log2_approximates_non_powers() {
        // Reference values from log2(3) and log2(6).
        assert!(close(log2_via_bits(3.0), 1.584_962_5, 1e-2));
        assert!(close(log2_via_bits(6.0), 2.584_962_5, 1e-2));
    }

    #[test]
    fn log2_of_non_positive_is_floored() {
        assert!(close(log2_via_bits(0.0), LOG2_ZERO_FLOOR, TEST_EPS));
        assert!(close(log2_via_bits(-4.0), LOG2_ZERO_FLOOR, TEST_EPS));
    }

    #[test]
    fn zero_gradient_is_protected() {
        let g = Gradients {
            ddx: [0.0, 0.0],
            ddy: [0.0, 0.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        assert!(close(f.major_len, 0.0, TEST_EPS));
        assert!(close(f.minor_len, 0.0, TEST_EPS));
        assert!(close(f.anisotropy, 1.0, TEST_EPS));
        assert!(!f.major_len.is_nan());
        assert!(!f.major_dir[0].is_nan());
        assert_eq!(f.sample_count(), 1);
    }

    #[test]
    fn eigenvalues_are_non_negative() {
        let g = Gradients {
            ddx: [2.0, -3.0],
            ddy: [1.0, 4.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        assert!(f.major_len >= 0.0);
        assert!(f.minor_len >= 0.0);
    }

    #[test]
    fn major_axis_is_at_least_minor_axis() {
        let cases = [
            ([3.0_f32, 1.0], [0.5_f32, 2.0]),
            ([1.0, 0.0], [0.0, 1.0]),
            ([5.0, 5.0], [1.0, -1.0]),
        ];
        for (ddx, ddy) in cases {
            let f = Footprint::from_gradients(&Gradients { ddx, ddy }, 16.0);
            assert!(f.major_len >= f.minor_len - TEST_EPS);
        }
    }

    #[test]
    fn eigenvalue_sum_equals_metric_trace() {
        let g = Gradients {
            ddx: [2.0, 1.0],
            ddy: [-1.0, 3.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        let trace = (g.ddx[0] * g.ddx[0] + g.ddx[1] * g.ddx[1])
            + (g.ddy[0] * g.ddy[0] + g.ddy[1] * g.ddy[1]);
        let sum = f.major_len * f.major_len + f.minor_len * f.minor_len;
        assert!(close(sum, trace, 1e-3));
    }

    #[test]
    fn major_direction_is_unit_length() {
        let g = Gradients {
            ddx: [3.0, 4.0],
            ddy: [-2.0, 1.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        let len = (f.major_dir[0] * f.major_dir[0] + f.major_dir[1] * f.major_dir[1]).sqrt();
        assert!(close(len, 1.0, TEST_EPS));
    }

    #[test]
    fn anisotropic_lod_is_finer_than_trilinear() {
        let g = Gradients {
            ddx: [8.0, 0.0],
            ddy: [0.0, 1.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        assert!(f.lod_anisotropic() < lod_trilinear(&g) + TEST_EPS);
        // Minor axis is length 1 -> log2(1) == 0.
        assert!(close(f.lod_anisotropic(), 0.0, TEST_EPS));
    }

    #[test]
    fn trilinear_lod_matches_log2_of_longest_gradient() {
        let g = Gradients {
            ddx: [4.0, 0.0],
            ddy: [0.0, 2.0],
        };
        // Longest gradient length is 4 -> log2(4) == 2.
        assert!(close(lod_trilinear(&g), 2.0, TEST_EPS));
    }

    #[test]
    fn std430_size_is_thirty_two() {
        assert_eq!(ANISO_FOOTPRINT_STD430_SIZE, 32);
    }

    #[test]
    fn std430_round_trip_preserves_fields() {
        let g = Gradients {
            ddx: [6.0, 1.0],
            ddy: [1.0, 2.0],
        };
        let f = Footprint::from_gradients(&g, 16.0);
        let bytes = to_std430(&f);
        assert!(close(decode_f32(&bytes, 0), f.major_dir[0], TEST_EPS));
        assert!(close(decode_f32(&bytes, 4), f.major_dir[1], TEST_EPS));
        assert!(close(decode_f32(&bytes, 8), f.major_len, TEST_EPS));
        assert!(close(decode_f32(&bytes, 12), f.minor_len, TEST_EPS));
        assert!(close(decode_f32(&bytes, 16), f.anisotropy, TEST_EPS));
        // Padding tail is zeroed.
        assert_eq!(&bytes[20..32], &[0u8; 12]);
    }

    #[test]
    fn gpu_storage_bytes_reserves_one_element_when_empty() {
        assert_eq!(gpu_storage_bytes(0), ANISO_FOOTPRINT_STD430_SIZE);
    }

    #[test]
    fn gpu_storage_bytes_scales_with_count() {
        assert_eq!(gpu_storage_bytes(4), 4 * ANISO_FOOTPRINT_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(1), ANISO_FOOTPRINT_STD430_SIZE);
    }

    #[test]
    fn degenerate_minor_axis_uses_max_anisotropy() {
        // A rank-one gradient (both derivatives colinear) has a zero minor axis.
        let g = Gradients {
            ddx: [2.0, 0.0],
            ddy: [4.0, 0.0],
        };
        let f = Footprint::from_gradients(&g, 12.0);
        assert!(close(f.minor_len, 0.0, TEST_EPS));
        assert!(close(f.anisotropy, 12.0, TEST_EPS));
    }
}
