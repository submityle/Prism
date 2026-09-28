//! Backend-neutral CPU golden for AAA display tone mapping.
//!
//! The visibility -> classify -> shade -> resolve chain writes *pre-exposed*
//! linear HDR radiance (see [`crate::exposure`] for the multiplier that lands
//! it there). This module is the next step: it turns that open-domain HDR
//! radiance into a display-referred `[0, 1]` colour by applying a tone-map
//! curve. Tone mapping is distinct from exposure — exposure sets the working
//! range, tone mapping compresses it for a screen — and both run before any
//! sRGB / PQ transfer encode.
//!
//! Five operators are provided, spanning the range an AAA title ships with:
//!
//! * **Reinhard** (`x / (1 + x)`), the classic global operator, and its
//!   **extended** white-point variant that lets highlights above `white` clip
//!   to `1.0` instead of asymptotically approaching it.
//! * **ACES Narkowicz**, the cheap Krzysztof Narkowicz RRT+ODT fit
//!   (`(x(2.51x + 0.03)) / (x(2.43x + 0.59) + 0.14)`) applied per channel.
//! * **ACES fitted**, Stephen Hill's higher-fidelity fit: an sRGB->`ACEScg`
//!   input matrix, the `RRTAndODTFit` rational curve, then an `ACEScg`->sRGB
//!   output matrix (the real matrices, not a stand-in).
//! * **`AgX`**, Troy Sobotka's minimal formulation: an input matrix, a `log2`
//!   encode normalised into a fixed EV window, a 6th-order polynomial sigmoid,
//!   an optional look/saturation grade (identity by default) and an output
//!   matrix.
//!
//! Everything is pure `f32` maths mirrored arm-for-arm by
//! `shaders/tonemap.wesl`, so the CPU golden and the GPU twin agree.

use bevy_math::ops;

/// Rec. 709 luminance weights (linear sRGB primaries), reused for the `AgX` look
/// saturation grade.
const LUMINANCE_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Rec. 709 relative luminance of a linear RGB radiance sample.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * LUMINANCE_WEIGHTS[0] + rgb[1] * LUMINANCE_WEIGHTS[1] + rgb[2] * LUMINANCE_WEIGHTS[2]
}

/// Row-major 3x3 times a column vector: `result[i] = dot(matrix[i], v)`.
fn mat3_mul_vec3(matrix: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        matrix[0][0] * v[0] + matrix[0][1] * v[1] + matrix[0][2] * v[2],
        matrix[1][0] * v[0] + matrix[1][1] * v[1] + matrix[1][2] * v[2],
        matrix[2][0] * v[0] + matrix[2][1] * v[1] + matrix[2][2] * v[2],
    ]
}

// --- Reinhard -------------------------------------------------------------

/// Reinhard global tone map `x / (1 + x)` applied per channel.
///
/// Negative inputs are clamped to zero first so the curve stays monotonic and
/// never produces a negative or out-of-range result.
#[must_use]
pub fn tonemap_reinhard(rgb: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for (o, &c) in out.iter_mut().zip(rgb.iter()) {
        let x = c.max(0.0);
        *o = x / (1.0 + x);
    }
    out
}

/// Extended Reinhard with a white point:
/// `x * (1 + x / white^2) / (1 + x)`, applied per channel.
///
/// A radiance equal to `white` maps to exactly `1.0`; brighter values clip.
/// A non-positive `white` degrades gracefully to plain [`tonemap_reinhard`].
#[must_use]
pub fn tonemap_reinhard_extended(rgb: [f32; 3], white: f32) -> [f32; 3] {
    if white <= 0.0 {
        return tonemap_reinhard(rgb);
    }
    let white_sq = white * white;
    let mut out = [0.0_f32; 3];
    for (o, &c) in out.iter_mut().zip(rgb.iter()) {
        let x = c.max(0.0);
        *o = x * (1.0 + x / white_sq) / (1.0 + x);
    }
    out
}

// --- ACES (Narkowicz) -----------------------------------------------------

/// Krzysztof Narkowicz's cheap ACES RRT+ODT fit applied per channel and
/// clamped to `[0, 1]`: `(x(2.51x + 0.03)) / (x(2.43x + 0.59) + 0.14)`.
#[must_use]
pub fn tonemap_aces_narkowicz(rgb: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for (o, &c) in out.iter_mut().zip(rgb.iter()) {
        let x = c.max(0.0);
        let numerator = x * (2.51 * x + 0.03);
        let denominator = x * (2.43 * x + 0.59) + 0.14;
        *o = (numerator / denominator).clamp(0.0, 1.0);
    }
    out
}

// --- ACES (Hill fitted) ---------------------------------------------------

/// sRGB (linear) -> `ACEScg` input matrix for Stephen Hill's ACES fit. Each row
/// sums to `1.0`, so a neutral grey survives the transform.
pub const ACES_INPUT_MATRIX: [[f32; 3]; 3] = [
    [0.59719, 0.35458, 0.04823],
    [0.07600, 0.90834, 0.01566],
    [0.02840, 0.13383, 0.83777],
];

/// `ACEScg` -> sRGB (linear) output matrix for Stephen Hill's ACES fit. Each row
/// sums to `1.0`.
pub const ACES_OUTPUT_MATRIX: [[f32; 3]; 3] = [
    [1.60475, -0.53108, -0.07367],
    [-0.10208, 1.10813, -0.00605],
    [-0.00327, -0.07276, 1.07602],
];

/// Hill's `RRTAndODTFit` rational curve, applied per channel in `ACEScg` space.
fn rrt_and_odt_fit(v: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for (o, &x) in out.iter_mut().zip(v.iter()) {
        let a = x * (x + 0.0245786) - 0.000090537;
        let b = x * (0.983729 * x + 0.432951) + 0.238081;
        *o = a / b;
    }
    out
}

/// Stephen Hill's higher-fidelity ACES fit: input matrix, `RRTAndODTFit`,
/// output matrix, then a `[0, 1]` clamp for the display.
#[must_use]
pub fn tonemap_aces_fitted(rgb: [f32; 3]) -> [f32; 3] {
    let clamped = [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)];
    let acescg = mat3_mul_vec3(ACES_INPUT_MATRIX, clamped);
    let fitted = rrt_and_odt_fit(acescg);
    let srgb = mat3_mul_vec3(ACES_OUTPUT_MATRIX, fitted);
    [
        srgb[0].clamp(0.0, 1.0),
        srgb[1].clamp(0.0, 1.0),
        srgb[2].clamp(0.0, 1.0),
    ]
}

// --- AgX (Troy Sobotka minimal) -------------------------------------------

/// `AgX` input matrix (Troy Sobotka's minimal fit). Rows sum to approximately
/// `1.0`.
pub const AGX_INPUT_MATRIX: [[f32; 3]; 3] = [
    [0.84247905, 0.0784336, 0.079223745],
    [0.042328242, 0.87846863, 0.07916613],
    [0.042375654, 0.0784336, 0.879143],
];

/// `AgX` output (inverse) matrix.
pub const AGX_OUTPUT_MATRIX: [[f32; 3]; 3] = [
    [1.196879, -0.09802088, -0.09902974],
    [-0.052896854, 1.1519032, -0.098961174],
    [-0.052971635, -0.09804345, 1.1510737],
];

/// Lowest EV of the `AgX` log encode window.
pub const AGX_MIN_EV: f32 = -12.47393;
/// Highest EV of the `AgX` log encode window.
pub const AGX_MAX_EV: f32 = 4.026069;

/// Optional `AgX` look grade. The default is the neutral (identity) look, so
/// `tonemap_agx` matches the reference minimal `AgX` out of the box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AgxLook {
    /// Per-channel slope (gain). `[1, 1, 1]` is neutral.
    pub slope: [f32; 3],
    /// Per-channel offset (lift). `[0, 0, 0]` is neutral.
    pub offset: [f32; 3],
    /// Per-channel power (gamma). `[1, 1, 1]` is neutral.
    pub power: [f32; 3],
    /// Saturation about luminance. `1.0` is neutral.
    pub saturation: f32,
}

impl Default for AgxLook {
    fn default() -> Self {
        Self {
            slope: [1.0, 1.0, 1.0],
            offset: [0.0, 0.0, 0.0],
            power: [1.0, 1.0, 1.0],
            saturation: 1.0,
        }
    }
}

/// 6th-order polynomial sigmoid (`agxDefaultContrastApprox`) on a `[0, 1]`
/// log-encoded value, applied per channel.
fn agx_contrast_sigmoid(v: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for (o, &x) in out.iter_mut().zip(v.iter()) {
        let x2 = x * x;
        let x4 = x2 * x2;
        *o = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x
            + 0.4298 * x2
            + 0.1191 * x
            - 0.00232;
    }
    out
}

/// `AgX` input transform: matrix into `AgX` working space, then a `log2` encode
/// normalised into the `[AGX_MIN_EV, AGX_MAX_EV]` window mapped to `[0, 1]`.
fn agx_encode(rgb: [f32; 3]) -> [f32; 3] {
    let clamped = [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)];
    let working = mat3_mul_vec3(AGX_INPUT_MATRIX, clamped);
    let span = AGX_MAX_EV - AGX_MIN_EV;
    let mut out = [0.0_f32; 3];
    for (o, &c) in out.iter_mut().zip(working.iter()) {
        let safe = c.max(1.0e-10);
        let log_ev = ops::log2(safe).clamp(AGX_MIN_EV, AGX_MAX_EV);
        *o = (log_ev - AGX_MIN_EV) / span;
    }
    out
}

/// `AgX` look grade `pow(v * slope + offset, power)` then a saturation blend
/// about luminance. Neutral parameters return `v` unchanged.
fn agx_apply_look(v: [f32; 3], look: AgxLook) -> [f32; 3] {
    let mut graded = [0.0_f32; 3];
    for i in 0..3 {
        let base = (v[i] * look.slope[i] + look.offset[i]).max(0.0);
        graded[i] = ops::powf(base, look.power[i]);
    }
    let luma = luminance(graded);
    [
        luma + look.saturation * (graded[0] - luma),
        luma + look.saturation * (graded[1] - luma),
        luma + look.saturation * (graded[2] - luma),
    ]
}

/// `AgX` with the neutral (default) look. See [`tonemap_agx_with_look`].
#[must_use]
pub fn tonemap_agx(rgb: [f32; 3]) -> [f32; 3] {
    tonemap_agx_with_look(rgb, AgxLook::default())
}

/// Full `AgX` pipeline: input encode, contrast sigmoid, look grade, output
/// matrix, then a `[0, 1]` clamp for the display.
#[must_use]
pub fn tonemap_agx_with_look(rgb: [f32; 3], look: AgxLook) -> [f32; 3] {
    let encoded = agx_encode(rgb);
    let contrast = agx_contrast_sigmoid(encoded);
    let graded = agx_apply_look(contrast, look);
    let out = mat3_mul_vec3(AGX_OUTPUT_MATRIX, graded);
    [
        out[0].clamp(0.0, 1.0),
        out[1].clamp(0.0, 1.0),
        out[2].clamp(0.0, 1.0),
    ]
}

// --- Unified dispatch -----------------------------------------------------

/// Tunable inputs shared by the tone-map operators. Only the extended-Reinhard
/// white point and the `AgX` look are configurable today; the rest of the
/// operators are parameter-free curves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TonemapParams {
    /// White point for [`TonemapOperator::ReinhardExtended`] (radiance mapping
    /// to `1.0`).
    pub reinhard_white_point: f32,
    /// Look grade for [`TonemapOperator::AgX`].
    pub agx_look: AgxLook,
}

impl Default for TonemapParams {
    fn default() -> Self {
        Self {
            reinhard_white_point: 4.0,
            agx_look: AgxLook::default(),
        }
    }
}

/// The tone-map operator to apply. [`TonemapOperator::Reinhard`] is the
/// default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TonemapOperator {
    /// Plain Reinhard `x / (1 + x)`.
    #[default]
    Reinhard,
    /// Extended Reinhard with a white point.
    ReinhardExtended,
    /// Narkowicz ACES RRT+ODT fit.
    AcesNarkowicz,
    /// Hill ACES fit (matrices + `RRTAndODTFit`).
    AcesFitted,
    /// Troy Sobotka minimal `AgX`.
    AgX,
}

/// Applies `operator` to `rgb` using `params`, returning display-referred
/// `[0, 1]` colour.
#[must_use]
pub fn apply_tonemap(operator: TonemapOperator, rgb: [f32; 3], params: TonemapParams) -> [f32; 3] {
    match operator {
        TonemapOperator::Reinhard => tonemap_reinhard(rgb),
        TonemapOperator::ReinhardExtended => {
            tonemap_reinhard_extended(rgb, params.reinhard_white_point)
        }
        TonemapOperator::AcesNarkowicz => tonemap_aces_narkowicz(rgb),
        TonemapOperator::AcesFitted => tonemap_aces_fitted(rgb),
        TonemapOperator::AgX => tonemap_agx_with_look(rgb, params.agx_look),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
    }

    fn row_sum(matrix: [[f32; 3]; 3], row: usize) -> f32 {
        matrix[row][0] + matrix[row][1] + matrix[row][2]
    }

    #[test]
    fn luminance_matches_rec709() {
        approx(luminance([1.0, 1.0, 1.0]), 1.0);
        approx(luminance([1.0, 0.0, 0.0]), 0.2126);
        approx(luminance([0.0, 1.0, 0.0]), 0.7152);
        approx(luminance([0.0, 0.0, 1.0]), 0.0722);
    }

    #[test]
    fn reinhard_maps_black_to_black() {
        approx(tonemap_reinhard([0.0, 0.0, 0.0])[0], 0.0);
        approx(tonemap_aces_narkowicz([0.0, 0.0, 0.0])[0], 0.0);
        approx(tonemap_aces_fitted([0.0, 0.0, 0.0])[0], 0.0);
    }

    #[test]
    fn reinhard_is_monotonic_and_bounded() {
        let low = tonemap_reinhard([0.5, 0.5, 0.5])[0];
        let mid = tonemap_reinhard([2.0, 2.0, 2.0])[0];
        let high = tonemap_reinhard([50.0, 50.0, 50.0])[0];
        assert!(low < mid && mid < high);
        // x / (1 + x) is strictly below 1 for all finite inputs.
        assert!(high < 1.0);
    }

    #[test]
    fn reinhard_half_input_matches_formula() {
        // 1 / (1 + 1) = 0.5.
        approx(tonemap_reinhard([1.0, 1.0, 1.0])[0], 0.5);
    }

    #[test]
    fn reinhard_extended_hits_one_at_white() {
        let white = 4.0;
        let out = tonemap_reinhard_extended([white, white, white], white);
        approx(out[0], 1.0);
        approx(out[1], 1.0);
        approx(out[2], 1.0);
    }

    #[test]
    fn reinhard_extended_below_white_is_under_one() {
        let out = tonemap_reinhard_extended([1.0, 1.0, 1.0], 4.0);
        assert!(out[0] < 1.0);
        // And it lifts highlights above plain Reinhard for the same input.
        assert!(out[0] > tonemap_reinhard([1.0, 1.0, 1.0])[0]);
    }

    #[test]
    fn reinhard_extended_degrades_to_plain_for_bad_white() {
        let bad = tonemap_reinhard_extended([2.0, 3.0, 4.0], 0.0);
        let plain = tonemap_reinhard([2.0, 3.0, 4.0]);
        approx(bad[0], plain[0]);
        approx(bad[1], plain[1]);
        approx(bad[2], plain[2]);
    }

    #[test]
    fn aces_narkowicz_saturates_and_is_monotonic() {
        let low = tonemap_aces_narkowicz([0.2, 0.2, 0.2])[0];
        let mid = tonemap_aces_narkowicz([1.0, 1.0, 1.0])[0];
        let big = tonemap_aces_narkowicz([1000.0, 1000.0, 1000.0])[0];
        assert!(low < mid);
        assert!(mid < big);
        // Large inputs clamp to at most 1.0.
        assert!(big <= 1.0);
        approx(big, 1.0);
    }

    #[test]
    fn aces_narkowicz_middle_grey_is_reasonable() {
        // 0.18 linear should land in a plausible mid-tone band.
        let grey = tonemap_aces_narkowicz([0.18, 0.18, 0.18])[0];
        assert!(grey > 0.1 && grey < 0.45, "{grey}");
    }

    #[test]
    fn aces_fitted_is_monotonic_and_bounded() {
        let low = tonemap_aces_fitted([0.1, 0.1, 0.1])[0];
        let mid = tonemap_aces_fitted([0.5, 0.5, 0.5])[0];
        let high = tonemap_aces_fitted([8.0, 8.0, 8.0])[0];
        assert!(low < mid && mid < high);
        assert!((0.0..=1.0).contains(&high));
    }

    #[test]
    fn aces_matrix_rows_sum_to_one() {
        for row in 0..3 {
            approx(row_sum(ACES_INPUT_MATRIX, row), 1.0);
            approx(row_sum(ACES_OUTPUT_MATRIX, row), 1.0);
        }
    }

    #[test]
    fn agx_input_matrix_rows_sum_to_about_one() {
        for row in 0..3 {
            let sum = row_sum(AGX_INPUT_MATRIX, row);
            assert!((sum - 1.0).abs() < 1.0e-3, "row {row} sum {sum}");
        }
    }

    #[test]
    fn agx_maps_black_near_zero() {
        let out = tonemap_agx([0.0, 0.0, 0.0]);
        assert!(out[0] < 1.0e-3, "{}", out[0]);
    }

    #[test]
    fn agx_is_monotonic_on_grey_ramp() {
        let a = luminance(tonemap_agx([0.05, 0.05, 0.05]));
        let b = luminance(tonemap_agx([0.3, 0.3, 0.3]));
        let c = luminance(tonemap_agx([1.0, 1.0, 1.0]));
        let d = luminance(tonemap_agx([8.0, 8.0, 8.0]));
        assert!(a < b && b < c && c < d, "{a} {b} {c} {d}");
    }

    #[test]
    fn agx_output_stays_in_unit_range() {
        for &v in &[0.0_f32, 0.01, 0.18, 1.0, 100.0, 10000.0] {
            let out = tonemap_agx([v, v, v]);
            for ch in out {
                assert!((0.0..=1.0).contains(&ch), "v={v} ch={ch}");
            }
        }
    }

    #[test]
    fn agx_neutral_look_matches_default_pipeline() {
        let rgb = [0.4, 0.9, 1.7];
        let default = tonemap_agx(rgb);
        let neutral = tonemap_agx_with_look(rgb, AgxLook::default());
        for i in 0..3 {
            approx(default[i], neutral[i]);
        }
    }

    #[test]
    fn agx_zero_saturation_is_grey() {
        let look = AgxLook {
            saturation: 0.0,
            ..AgxLook::default()
        };
        let out = tonemap_agx_with_look([0.2, 0.8, 1.5], look);
        // Zero saturation collapses the graded colour to a single luminance;
        // the only residual channel spread is the AgX output matrix's tiny
        // row-sum difference, so the result is grey to within ~2e-3.
        assert!((out[0] - out[1]).abs() < 2.0e-3, "{out:?}");
        assert!((out[1] - out[2]).abs() < 2.0e-3, "{out:?}");
    }

    #[test]
    fn apply_tonemap_dispatches_each_operator() {
        let rgb = [0.5, 0.5, 0.5];
        let params = TonemapParams::default();
        approx(
            apply_tonemap(TonemapOperator::Reinhard, rgb, params)[0],
            tonemap_reinhard(rgb)[0],
        );
        approx(
            apply_tonemap(TonemapOperator::ReinhardExtended, rgb, params)[0],
            tonemap_reinhard_extended(rgb, params.reinhard_white_point)[0],
        );
        approx(
            apply_tonemap(TonemapOperator::AcesNarkowicz, rgb, params)[0],
            tonemap_aces_narkowicz(rgb)[0],
        );
        approx(
            apply_tonemap(TonemapOperator::AcesFitted, rgb, params)[0],
            tonemap_aces_fitted(rgb)[0],
        );
        approx(
            apply_tonemap(TonemapOperator::AgX, rgb, params)[0],
            tonemap_agx(rgb)[0],
        );
    }

    #[test]
    fn operator_default_is_reinhard() {
        assert_eq!(TonemapOperator::default(), TonemapOperator::Reinhard);
    }
}
