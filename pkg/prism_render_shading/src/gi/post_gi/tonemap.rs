//! Filmic display tone mapping — backend-neutral CPU golden.
//!
//! Tone mapping turns the open-domain, pre-exposed linear HDR radiance that the
//! GI pipeline produces into a display-referred `[0, 1]` colour. It is distinct
//! from exposure (which sets the working range, see
//! [`super::auto_exposure`]) and runs before the sRGB transfer encode. This
//! module collects the operators an AAA title ships with so the WESL/GPU twin
//! has a bit-exact reference to match:
//!
//! * **Reinhard** (`x / (1 + x)`) and **extended Reinhard** with a white point
//!   that lets highlights above `white` clip to `1.0`. Cheap, provided mainly
//!   as a monotone reference to cross-check the filmic curves against.
//! * **ACES Narkowicz**, Krzysztof Narkowicz's cheap RRT+ODT rational fit
//!   `(x(2.51x + 0.03)) / (x(2.43x + 0.59) + 0.14)` applied per channel.
//! * **ACES fitted**, Stephen Hill's higher-fidelity fit: an sRGB->`ACEScg`
//!   input matrix, the `RRTAndODTFit` rational curve, then an `ACEScg`->sRGB
//!   output matrix (the real matrices, not a stand-in).
//! * **`AgX`**, Troy Sobotka's minimal formulation: an input matrix, a `log2`
//!   encode normalised into a fixed EV window, a 6th-order polynomial sigmoid,
//!   an optional look/saturation grade, and an output matrix.
//! * **sRGB transfer functions**: the IEC 61966-2-1 OETF (linear -> encoded)
//!   and its inverse EOTF (encoded -> linear) with the real piecewise curve.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe / global state.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Inputs are floored to zero where a tone curve expects non-negative light;
//!   outputs are clamped to `[0, 1]`. No path can emit `NaN` or `inf`.
//! * `f32` storage mirrors the layout the WESL/GPU twin consumes.

use bevy_math::ops;

/// Rec. 709 luminance weights (linear sRGB primaries), reused for the `AgX`
/// look saturation grade.
pub const LUMINANCE_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

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

/// Reinhard global tone map `x / (1 + x)` applied per channel. Negative inputs
/// are clamped to zero first so the curve stays monotonic and non-negative.
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
/// A radiance equal to `white` maps to exactly `1.0`; brighter values clip. A
/// non-positive `white` degrades gracefully to plain [`tonemap_reinhard`].
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

/// Krzysztof Narkowicz's cheap ACES RRT+ODT fit, per channel, clamped to
/// `[0, 1]`: `(x(2.51x + 0.03)) / (x(2.43x + 0.59) + 0.14)`.
#[must_use]
pub fn tonemap_aces_narkowicz(rgb: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for (o, &c) in out.iter_mut().zip(rgb.iter()) {
        let x = c.max(0.0);
        let numerator = x * (2.51 * x + 0.03);
        let denominator = x * (2.43 * x + 0.59) + 0.14;
        *o = (numerator / denominator.max(1.0e-6)).clamp(0.0, 1.0);
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
        let b = x * (0.983729 * x + 0.4329510) + 0.238081;
        *o = a / b.max(1.0e-6);
    }
    out
}

/// Stephen Hill's higher-fidelity ACES fit: input matrix, `RRTAndODTFit`,
/// output matrix, final clamp to `[0, 1]`.
#[must_use]
pub fn tonemap_aces_fitted(rgb: [f32; 3]) -> [f32; 3] {
    let clamped = [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)];
    let acescg = mat3_mul_vec3(ACES_INPUT_MATRIX, clamped);
    let fit = rrt_and_odt_fit(acescg);
    let srgb = mat3_mul_vec3(ACES_OUTPUT_MATRIX, fit);
    [
        srgb[0].clamp(0.0, 1.0),
        srgb[1].clamp(0.0, 1.0),
        srgb[2].clamp(0.0, 1.0),
    ]
}

// --- AgX ------------------------------------------------------------------

/// `AgX` input matrix (sRGB linear -> `AgX` working space), Troy Sobotka's
/// minimal fit. Rows sum to about `1.0`.
pub const AGX_INPUT_MATRIX: [[f32; 3]; 3] = [
    [0.842479062253094, 0.0784335999999992, 0.0792237451477643],
    [0.0423282422610123, 0.878468636469772, 0.0791661274605434],
    [0.0423756549057051, 0.0784336, 0.879142973793104],
];

/// `AgX` output matrix (`AgX` working space -> sRGB linear). Rows sum to about
/// `1.0`.
pub const AGX_OUTPUT_MATRIX: [[f32; 3]; 3] = [
    [1.19687900512017, -0.0980208811401368, -0.0990297440797205],
    [-0.0528968517574562, 1.15190312990417, -0.0989611768448433],
    [-0.0529716355144438, -0.0980434501171241, 1.15107367264116],
];

/// `AgX` log2 encode window minimum in EV (`-12.47393` stops).
pub const AGX_MIN_EV: f32 = -12.47393;
/// `AgX` log2 encode window maximum in EV (`+4.026069` stops).
pub const AGX_MAX_EV: f32 = 4.026069;

/// Sobotka's 6th-order polynomial sigmoid approximating the `AgX` contrast
/// curve, evaluated per channel on the normalised `[0, 1]` log value.
fn agx_sigmoid(v: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for (o, &x) in out.iter_mut().zip(v.iter()) {
        let x = x.clamp(0.0, 1.0);
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x2 * x2;
        let x5 = x4 * x;
        let x6 = x3 * x3;
        // Minimal 6th-order fit of the AgX display sigmoid (Sobotka / Hill).
        *o = (15.5 * x6 - 40.14 * x5 + 31.96 * x4 - 6.868 * x3 + 0.4298 * x2
            + 0.1191 * x
            - 0.00232)
            .clamp(0.0, 1.0);
    }
    out
}

/// `AgX` log2 encode: map linear light into the `[AGX_MIN_EV, AGX_MAX_EV]`
/// window normalised to `[0, 1]`. Non-positive channels floor to a tiny value
/// so the log stays finite.
fn agx_log_encode(v: [f32; 3]) -> [f32; 3] {
    let span = (AGX_MAX_EV - AGX_MIN_EV).max(1.0e-6);
    let mut out = [0.0_f32; 3];
    for (o, &x) in out.iter_mut().zip(v.iter()) {
        let l = ops::log2(x.max(1.0e-10));
        *o = ((l - AGX_MIN_EV) / span).clamp(0.0, 1.0);
    }
    out
}

/// Optional `AgX` "look" grade: offset, slope (per-channel power), and a
/// luminance-preserving saturation. `Default` is the neutral identity look.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AgxLook {
    /// Additive offset applied to the sigmoid output per channel.
    pub offset: f32,
    /// Per-channel power (`pow(x, 1/slope)`-style contrast); `1.0` is neutral.
    pub slope: f32,
    /// Saturation about the luminance; `1.0` is neutral, `0.0` fully grey.
    pub saturation: f32,
}

impl Default for AgxLook {
    fn default() -> Self {
        Self {
            offset: 0.0,
            slope: 1.0,
            saturation: 1.0,
        }
    }
}

impl AgxLook {
    /// Returns `true` when the look is the neutral identity (so callers can skip
    /// the grade entirely and keep the fast path bit-exact).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.offset == 0.0 && self.slope == 1.0 && self.saturation == 1.0
    }
}

fn apply_agx_look(v: [f32; 3], look: AgxLook) -> [f32; 3] {
    if look.is_identity() {
        return v;
    }
    let slope = look.slope.max(1.0e-4);
    let mut graded = [0.0_f32; 3];
    for (g, &x) in graded.iter_mut().zip(v.iter()) {
        let x = (x + look.offset).max(0.0);
        *g = ops::powf(x, slope);
    }
    let luma = luminance(graded);
    let sat = look.saturation.max(0.0);
    [
        (luma + (graded[0] - luma) * sat).clamp(0.0, 1.0),
        (luma + (graded[1] - luma) * sat).clamp(0.0, 1.0),
        (luma + (graded[2] - luma) * sat).clamp(0.0, 1.0),
    ]
}

/// `AgX` tone map with a neutral look. Input matrix, log2 encode, sigmoid,
/// output matrix, final `[0, 1]` clamp.
#[must_use]
pub fn tonemap_agx(rgb: [f32; 3]) -> [f32; 3] {
    tonemap_agx_with_look(rgb, AgxLook::default())
}

/// `AgX` tone map with an explicit [`AgxLook`] grade applied after the sigmoid.
#[must_use]
pub fn tonemap_agx_with_look(rgb: [f32; 3], look: AgxLook) -> [f32; 3] {
    let clamped = [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)];
    let working = mat3_mul_vec3(AGX_INPUT_MATRIX, clamped);
    let encoded = agx_log_encode(working);
    let sigmoid = agx_sigmoid(encoded);
    let looked = apply_agx_look(sigmoid, look);
    let srgb = mat3_mul_vec3(AGX_OUTPUT_MATRIX, looked);
    [
        srgb[0].clamp(0.0, 1.0),
        srgb[1].clamp(0.0, 1.0),
        srgb[2].clamp(0.0, 1.0),
    ]
}

// --- sRGB transfer functions ----------------------------------------------

/// sRGB OETF (IEC 61966-2-1): linear light -> gamma-encoded value, per channel.
///
/// `v <= 0.0031308 -> 12.92 * v`, else `1.055 * v^(1/2.4) - 0.055`. Inputs are
/// clamped to `[0, 1]`.
#[must_use]
pub fn linear_to_srgb(rgb: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for (o, &c) in out.iter_mut().zip(rgb.iter()) {
        let x = c.clamp(0.0, 1.0);
        *o = if x <= 0.0031308 {
            12.92 * x
        } else {
            1.055 * ops::powf(x, 1.0 / 2.4) - 0.055
        }
        .clamp(0.0, 1.0);
    }
    out
}

/// sRGB EOTF (inverse of [`linear_to_srgb`]): gamma-encoded value -> linear
/// light, per channel.
///
/// `v <= 0.04045 -> v / 12.92`, else `((v + 0.055) / 1.055)^2.4`. Inputs are
/// clamped to `[0, 1]`.
#[must_use]
pub fn srgb_to_linear(rgb: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for (o, &c) in out.iter_mut().zip(rgb.iter()) {
        let x = c.clamp(0.0, 1.0);
        *o = if x <= 0.04045 {
            x / 12.92
        } else {
            ops::powf((x + 0.055) / 1.055, 2.4)
        }
        .clamp(0.0, 1.0);
    }
    out
}

// --- Operator dispatch ----------------------------------------------------

/// Selects which tone-map operator [`apply_tonemap`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TonemapOperator {
    /// Plain Reinhard `x / (1 + x)`.
    #[default]
    Reinhard,
    /// Extended Reinhard with the configured white point.
    ReinhardExtended,
    /// Cheap Narkowicz ACES fit.
    AcesNarkowicz,
    /// Stephen Hill's fitted ACES.
    AcesFitted,
    /// Troy Sobotka's `AgX`.
    AgX,
}

/// Parameters shared by the dispatch helper. `Default` matches each operator's
/// own defaults (white point `4.0`, neutral `AgX` look).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TonemapParams {
    /// White point for [`TonemapOperator::ReinhardExtended`].
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

/// Dispatches to the selected [`TonemapOperator`] with `params`.
#[must_use]
pub fn apply_tonemap(
    operator: TonemapOperator,
    rgb: [f32; 3],
    params: TonemapParams,
) -> [f32; 3] {
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

    fn row_sum(m: [[f32; 3]; 3], r: usize) -> f32 {
        m[r][0] + m[r][1] + m[r][2]
    }

    #[test]
    fn reinhard_zero_is_zero_and_monotonic() {
        approx(tonemap_reinhard([0.0, 0.0, 0.0])[0], 0.0);
        let a = tonemap_reinhard([0.5, 0.5, 0.5])[0];
        let b = tonemap_reinhard([2.0, 2.0, 2.0])[0];
        assert!(a < b && b < 1.0);
    }

    #[test]
    fn reinhard_extended_hits_one_at_white() {
        let out = tonemap_reinhard_extended([4.0, 4.0, 4.0], 4.0);
        approx(out[0], 1.0);
    }

    #[test]
    fn reinhard_extended_degrades_for_bad_white() {
        let bad = tonemap_reinhard_extended([2.0, 3.0, 4.0], 0.0);
        let plain = tonemap_reinhard([2.0, 3.0, 4.0]);
        approx(bad[0], plain[0]);
        approx(bad[2], plain[2]);
    }

    #[test]
    fn aces_narkowicz_zero_is_zero() {
        approx(tonemap_aces_narkowicz([0.0, 0.0, 0.0])[0], 0.0);
    }

    #[test]
    fn aces_narkowicz_saturates_and_is_monotonic() {
        let low = tonemap_aces_narkowicz([0.2, 0.2, 0.2])[0];
        let mid = tonemap_aces_narkowicz([1.0, 1.0, 1.0])[0];
        let big = tonemap_aces_narkowicz([1000.0, 1000.0, 1000.0])[0];
        assert!(low < mid && mid < big);
        approx(big, 1.0);
    }

    #[test]
    fn aces_fitted_zero_is_zero_and_bounded() {
        let z = tonemap_aces_fitted([0.0, 0.0, 0.0]);
        assert!(z[0] < 1.0e-3);
        let low = tonemap_aces_fitted([0.1, 0.1, 0.1])[0];
        let mid = tonemap_aces_fitted([0.5, 0.5, 0.5])[0];
        let high = tonemap_aces_fitted([8.0, 8.0, 8.0])[0];
        assert!(low < mid && mid < high);
        assert!((0.0..=1.0).contains(&high));
    }

    #[test]
    fn aces_matrix_rows_sum_to_one() {
        for r in 0..3 {
            approx(row_sum(ACES_INPUT_MATRIX, r), 1.0);
            approx(row_sum(ACES_OUTPUT_MATRIX, r), 1.0);
        }
    }

    #[test]
    fn agx_input_matrix_rows_sum_to_about_one() {
        for r in 0..3 {
            let s = row_sum(AGX_INPUT_MATRIX, r);
            assert!((s - 1.0).abs() < 1.0e-3, "row {r} sum {s}");
        }
    }

    #[test]
    fn agx_maps_black_near_zero() {
        let out = tonemap_agx([0.0, 0.0, 0.0]);
        assert!(out[0] < 1.0e-2, "{}", out[0]);
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
    fn agx_neutral_look_matches_default() {
        let rgb = [0.4, 0.9, 1.7];
        let a = tonemap_agx(rgb);
        let b = tonemap_agx_with_look(rgb, AgxLook::default());
        for i in 0..3 {
            approx(a[i], b[i]);
        }
    }

    #[test]
    fn agx_zero_saturation_is_grey() {
        let look = AgxLook {
            saturation: 0.0,
            ..AgxLook::default()
        };
        let out = tonemap_agx_with_look([0.2, 0.8, 1.5], look);
        assert!((out[0] - out[1]).abs() < 5.0e-3, "{out:?}");
        assert!((out[1] - out[2]).abs() < 5.0e-3, "{out:?}");
    }

    #[test]
    fn srgb_round_trips() {
        for &v in &[0.0_f32, 0.001, 0.0031, 0.05, 0.5, 1.0] {
            let enc = linear_to_srgb([v, v, v]);
            let dec = srgb_to_linear(enc);
            approx(dec[0], v);
        }
    }

    #[test]
    fn srgb_endpoints_exact() {
        approx(linear_to_srgb([0.0, 0.0, 0.0])[0], 0.0);
        approx(linear_to_srgb([1.0, 1.0, 1.0])[0], 1.0);
        approx(srgb_to_linear([0.0, 0.0, 0.0])[0], 0.0);
        approx(srgb_to_linear([1.0, 1.0, 1.0])[0], 1.0);
    }

    #[test]
    fn dispatch_matches_direct_calls() {
        let rgb = [0.5, 0.5, 0.5];
        let p = TonemapParams::default();
        approx(
            apply_tonemap(TonemapOperator::Reinhard, rgb, p)[0],
            tonemap_reinhard(rgb)[0],
        );
        approx(
            apply_tonemap(TonemapOperator::ReinhardExtended, rgb, p)[0],
            tonemap_reinhard_extended(rgb, p.reinhard_white_point)[0],
        );
        approx(
            apply_tonemap(TonemapOperator::AcesNarkowicz, rgb, p)[0],
            tonemap_aces_narkowicz(rgb)[0],
        );
        approx(
            apply_tonemap(TonemapOperator::AcesFitted, rgb, p)[0],
            tonemap_aces_fitted(rgb)[0],
        );
        approx(
            apply_tonemap(TonemapOperator::AgX, rgb, p)[0],
            tonemap_agx(rgb)[0],
        );
    }

    #[test]
    fn operator_default_is_reinhard() {
        assert_eq!(TonemapOperator::default(), TonemapOperator::Reinhard);
    }
}
