//! Backend-neutral CPU golden for industry-standard HDR colour grading.
//!
//! Colour grading is a shared post-processing base that runs on the resolved,
//! pre-exposed *linear* HDR radiance (see [`crate::exposure`]) **before** the
//! display tone-map curve. Grading in the scene-referred linear domain — rather
//! than after tone-mapping — is what keeps hue shifts stable across the dynamic
//! range and matches the `DaVinci` / ACES workflow film pipelines rely on. Every
//! illumination model, physical or stylized, writes into the same HDR buffer,
//! so one grade serves them all.
//!
//! The operators, applied in a fixed order by [`apply_color_grade`], are the
//! ones every AAA grading stack exposes:
//!
//! * **White balance.** A von Kries chromatic adaptation in LMS space driven by
//!   a normalised temperature / tint pair. The reference-white maths is the
//!   transcendental-free formulation popularised by Unity's Post Processing
//!   Stack v2 (`StandardIlluminantY` on the Planckian-locus approximation plus
//!   the `CIExy`->LMS conversion), so the neutral setting is the identity to
//!   floating-point tolerance.
//! * **Lift / gamma / gain.** The ASC CDL transfer `out = (in * gain + lift)^gamma`
//!   applied per channel (slope = gain, offset = lift, power = gamma), the
//!   colourist's primary wheels. The neutral `(0, 1, 1)` returns the input.
//! * **Contrast.** A linear expansion about a `pivot` (default 0.18 middle
//!   grey): `pivot + (in - pivot) * contrast`. Neutral contrast `1` is the
//!   identity and the pivot is a fixed point for every contrast.
//! * **Saturation.** A luma-preserving mix about the Rec. 709 luma:
//!   `luma + (in - luma) * sat`. Neutral `sat = 1` returns the input and the
//!   Rec. 709 luminance is preserved for *every* saturation.
//! * **Offset** (`in + offset`) is provided for completeness but sits outside
//!   the fixed pipeline order so the four canonical stages stay unambiguous.
//!
//! Only lift/gamma/gain touches a transcendental (`ops::powf` for the CDL
//! power); everything else is polynomial. The whole module is mirrored
//! arm-for-arm by `shaders/color_grade.wesl` (same operation order, same
//! constants) so the CPU golden and the GPU twin agree.

use bevy_math::ops;

/// Rec. 709 luma weights (linear `sRGB` primaries), used by the saturation mix.
pub const COLOR_GRADE_LUMA_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// D65 reference white expressed in the same LMS basis as [`lin_to_lms`], the
/// target of the von Kries adaptation (Unity `PPv2` constant).
pub const D65_LMS: [f32; 3] = [0.949237, 1.03542, 1.08728];

/// Rec. 709 relative luma of a linear RGB sample.
#[must_use]
pub fn luma(rgb: [f32; 3]) -> f32 {
    rgb[0] * COLOR_GRADE_LUMA_WEIGHTS[0]
        + rgb[1] * COLOR_GRADE_LUMA_WEIGHTS[1]
        + rgb[2] * COLOR_GRADE_LUMA_WEIGHTS[2]
}

/// Planckian-locus `y` chromaticity approximation `2.87x - 3x^2 - 0.27509507`
/// (Unity `PPv2` `StandardIlluminantY`). At the D65 `x = 0.31271` it returns
/// `~0.32902`, the D65 `y`.
#[must_use]
pub fn standard_illuminant_y(x: f32) -> f32 {
    2.87 * x - 3.0 * x * x - 0.275_095_07
}

/// Linear `sRGB` -> LMS (Unity `PPv2` `LIN_2_LMS_MAT`, row-major).
#[must_use]
pub fn lin_to_lms(c: [f32; 3]) -> [f32; 3] {
    [
        3.904_05e-1 * c[0] + 5.499_41e-1 * c[1] + 8.926_32e-3 * c[2],
        7.084_16e-2 * c[0] + 9.631_72e-1 * c[1] + 1.357_75e-3 * c[2],
        2.310_82e-2 * c[0] + 1.280_21e-1 * c[1] + 9.362_45e-1 * c[2],
    ]
}

/// LMS -> linear `sRGB` (Unity `PPv2` `LMS_2_LIN_MAT`, row-major).
#[must_use]
pub fn lms_to_lin(c: [f32; 3]) -> [f32; 3] {
    [
        2.858_47e0 * c[0] + -1.628_79e0 * c[1] + -2.489_10e-2 * c[2],
        -2.101_82e-1 * c[0] + 1.158_20e0 * c[1] + 3.242_81e-4 * c[2],
        -4.181_20e-2 * c[0] + -1.181_69e-1 * c[1] + 1.068_67e0 * c[2],
    ]
}

/// White balance by a von Kries adaptation in LMS space.
///
/// `temperature` and `tint` are normalised knobs (roughly `[-1, 1]`, usable to
/// `~±1.67`): positive `temperature` warms (boosts red, cuts blue). Tint follows
/// the Unity `PPv2` convention — positive `tint` pushes toward magenta (green
/// falls), negative `tint` pushes toward green. The neutral `(0, 0)` is the identity to
/// floating-point tolerance because the derived white point then coincides with
/// [`D65_LMS`]. Mirrors Unity `PPv2` arm-for-arm.
#[must_use]
pub fn white_balance(rgb: [f32; 3], temperature: f32, tint: f32) -> [f32; 3] {
    let t1 = temperature * 10.0 / 6.0;
    let t2 = tint * 10.0 / 6.0;

    // Reference-white chromaticity offset from D65 along the Planckian locus.
    let scale = if t1 < 0.0 { 0.1 } else { 0.05 };
    let x = 0.31271 - t1 * scale;
    let y = standard_illuminant_y(x) + t2 * 0.05;

    // CIExy -> LMS at unit luminance.
    let cie_y = 1.0;
    let cie_x = cie_y * x / y;
    let cie_z = cie_y * (1.0 - x - y) / y;
    let lms_l = 0.7328 * cie_x + 0.4296 * cie_y - 0.1624 * cie_z;
    let lms_m = -0.7036 * cie_x + 1.6975 * cie_y + 0.0061 * cie_z;
    let lms_s = 0.0030 * cie_x + 0.0136 * cie_y + 0.9834 * cie_z;

    // Diagonal von Kries transform toward the D65 reference white.
    let balance = [D65_LMS[0] / lms_l, D65_LMS[1] / lms_m, D65_LMS[2] / lms_s];

    let lms = lin_to_lms(rgb);
    let adapted = [
        lms[0] * balance[0],
        lms[1] * balance[1],
        lms[2] * balance[2],
    ];
    lms_to_lin(adapted)
}

/// ASC CDL lift / gamma / gain per channel: `out = (in * gain + lift)^gamma`.
///
/// `gain` is the CDL slope, `lift` the offset, `gamma` the power. Negative
/// bases are clamped to `0` before the power (ASC CDL convention) so the result
/// is always finite. The neutral `lift = 0, gamma = 1, gain = 1` returns the
/// input. `ops::powf` keeps the transcendental off the disallowed `f32` path.
#[must_use]
pub fn lift_gamma_gain(rgb: [f32; 3], lift: [f32; 3], gamma: [f32; 3], gain: [f32; 3]) -> [f32; 3] {
    [
        ops::powf((rgb[0] * gain[0] + lift[0]).max(0.0), gamma[0]),
        ops::powf((rgb[1] * gain[1] + lift[1]).max(0.0), gamma[1]),
        ops::powf((rgb[2] * gain[2] + lift[2]).max(0.0), gamma[2]),
    ]
}

/// Saturation as a luma-preserving mix about the Rec. 709 luma:
/// `luma + (in - luma) * sat`.
///
/// `sat = 1` is the identity, `sat = 0` collapses to neutral grey, `sat > 1`
/// over-saturates and `sat < 0` inverts chroma. Because the mix pivots on the
/// luma, the Rec. 709 luminance of the result equals that of the input for
/// *every* `sat`.
#[must_use]
pub fn color_saturation(rgb: [f32; 3], sat: f32) -> [f32; 3] {
    let l = luma(rgb);
    [
        l + (rgb[0] - l) * sat,
        l + (rgb[1] - l) * sat,
        l + (rgb[2] - l) * sat,
    ]
}

/// Linear contrast expansion about `pivot`: `pivot + (in - pivot) * contrast`.
///
/// `contrast = 1` is the identity; `> 1` pushes values away from the pivot,
/// `< 1` pulls them toward it. `pivot` (typically 0.18 middle grey) is a fixed
/// point for every contrast.
#[must_use]
pub fn contrast(rgb: [f32; 3], contrast: f32, pivot: f32) -> [f32; 3] {
    [
        pivot + (rgb[0] - pivot) * contrast,
        pivot + (rgb[1] - pivot) * contrast,
        pivot + (rgb[2] - pivot) * contrast,
    ]
}

/// Per-channel additive offset `in + offset`. Provided for completeness; not
/// part of the fixed [`apply_color_grade`] pipeline. Neutral offset `0` is the
/// identity.
#[must_use]
pub fn offset(rgb: [f32; 3], offset: [f32; 3]) -> [f32; 3] {
    [rgb[0] + offset[0], rgb[1] + offset[1], rgb[2] + offset[2]]
}

/// Artist controls for the full grade. `Default` is a neutral, identity grade
/// (see the per-field defaults), so grading a pixel with the default params
/// returns it unchanged to floating-point tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorGradeParams {
    /// Normalised white-balance temperature (warm > 0, cool < 0). Neutral `0`.
    pub temperature: f32,
    /// Normalised white-balance tint (green > 0, magenta < 0). Neutral `0`.
    pub tint: f32,
    /// ASC CDL offset per channel. Neutral `[0, 0, 0]`.
    pub lift: [f32; 3],
    /// ASC CDL power per channel. Neutral `[1, 1, 1]`.
    pub gamma: [f32; 3],
    /// ASC CDL slope per channel. Neutral `[1, 1, 1]`.
    pub gain: [f32; 3],
    /// Linear contrast about `pivot`. Neutral `1`.
    pub contrast: f32,
    /// Contrast pivot (middle grey). Default `0.18`.
    pub pivot: f32,
    /// Luma-preserving saturation. Neutral `1`.
    pub saturation: f32,
}

impl Default for ColorGradeParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            tint: 0.0,
            lift: [0.0, 0.0, 0.0],
            gamma: [1.0, 1.0, 1.0],
            gain: [1.0, 1.0, 1.0],
            contrast: 1.0,
            pivot: 0.18,
            saturation: 1.0,
        }
    }
}

/// Apply the full grade in the fixed pipeline order
/// white balance -> lift/gamma/gain -> contrast -> saturation.
#[must_use]
pub fn apply_color_grade(rgb: [f32; 3], params: &ColorGradeParams) -> [f32; 3] {
    let c = white_balance(rgb, params.temperature, params.tint);
    let c = lift_gamma_gain(c, params.lift, params.gamma, params.gain);
    let c = contrast(c, params.contrast, params.pivot);
    color_saturation(c, params.saturation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) {
        approx(a[0], b[0]);
        approx(a[1], b[1]);
        approx(a[2], b[2]);
    }

    // --- white balance ---

    #[test]
    fn white_balance_neutral_is_identity() {
        // Neutral temperature/tint derive the D65 white point, so the von Kries
        // transform is the identity to fp tolerance.
        let rgb = [0.3, 0.5, 0.8];
        approx3(white_balance(rgb, 0.0, 0.0), rgb);
    }

    #[test]
    fn white_balance_neutral_preserves_grey() {
        let grey = [0.5, 0.5, 0.5];
        approx3(white_balance(grey, 0.0, 0.0), grey);
    }

    #[test]
    fn white_balance_warm_boosts_red_over_blue() {
        // Positive temperature warms: red rises relative to blue.
        let grey = [0.5, 0.5, 0.5];
        let warm = white_balance(grey, 0.4, 0.0);
        assert!(warm[0] > grey[0], "red should rise when warming");
        assert!(warm[2] < grey[2], "blue should fall when warming");
    }

    #[test]
    fn white_balance_cool_boosts_blue_over_red() {
        let grey = [0.5, 0.5, 0.5];
        let cool = white_balance(grey, -0.4, 0.0);
        assert!(cool[2] > grey[2], "blue should rise when cooling");
        assert!(cool[0] < grey[0], "red should fall when cooling");
    }

    #[test]
    fn white_balance_tint_shifts_green() {
        // Unity PPv2 tint convention: positive tint pushes toward magenta
        // (green falls) and negative tint pushes toward green (green rises).
        let grey = [0.5, 0.5, 0.5];
        let base = white_balance(grey, 0.0, 0.0);
        let magenta = white_balance(grey, 0.0, 0.4);
        let green = white_balance(grey, 0.0, -0.4);
        assert!(
            magenta[1] < base[1],
            "green should fall with positive (magenta) tint"
        );
        assert!(
            green[1] > base[1],
            "green should rise with negative (green) tint"
        );
    }

    // --- lift / gamma / gain ---

    #[test]
    fn lift_gamma_gain_neutral_is_identity() {
        let rgb = [0.2, 0.5, 0.9];
        approx3(lift_gamma_gain(rgb, [0.0; 3], [1.0; 3], [1.0; 3]), rgb);
    }

    #[test]
    fn lift_raises_shadows() {
        let rgb = [0.2, 0.2, 0.2];
        let lifted = lift_gamma_gain(rgb, [0.1; 3], [1.0; 3], [1.0; 3]);
        approx3(lifted, [0.3, 0.3, 0.3]);
    }

    #[test]
    fn gain_scales_linearly() {
        let rgb = [0.2, 0.4, 0.6];
        let g = lift_gamma_gain(rgb, [0.0; 3], [1.0; 3], [2.0; 3]);
        approx3(g, [0.4, 0.8, 1.2]);
    }

    #[test]
    fn gamma_power_darkens_midtones() {
        // Base in (0, 1): a power > 1 darkens it.
        let rgb = [0.5, 0.5, 0.5];
        let g = lift_gamma_gain(rgb, [0.0; 3], [2.0; 3], [1.0; 3]);
        assert!(g[0] < rgb[0], "power > 1 should darken a sub-unit value");
        approx(g[0], 0.25); // 0.5^2
    }

    #[test]
    fn lift_gamma_gain_is_per_channel() {
        let rgb = [0.5, 0.5, 0.5];
        let g = lift_gamma_gain(rgb, [0.1, 0.0, -0.1], [1.0; 3], [1.0; 3]);
        approx3(g, [0.6, 0.5, 0.4]);
    }

    #[test]
    fn lift_gamma_gain_clamps_negative_base() {
        // gain*in + lift < 0 clamps to 0 before the power (finite, non-NaN).
        let rgb = [0.1, 0.1, 0.1];
        let g = lift_gamma_gain(rgb, [-0.5; 3], [1.0; 3], [1.0; 3]);
        approx3(g, [0.0, 0.0, 0.0]);
    }

    // --- saturation ---

    #[test]
    fn saturation_one_is_identity() {
        let rgb = [0.2, 0.6, 0.9];
        approx3(color_saturation(rgb, 1.0), rgb);
    }

    #[test]
    fn saturation_zero_is_neutral_grey() {
        let rgb = [0.2, 0.6, 0.9];
        let l = luma(rgb);
        approx3(color_saturation(rgb, 0.0), [l, l, l]);
    }

    #[test]
    fn saturation_preserves_luma_for_any_sat() {
        let rgb = [0.1, 0.7, 0.4];
        let l = luma(rgb);
        for &sat in &[0.0, 0.5, 1.0, 1.5, 2.0] {
            approx(luma(color_saturation(rgb, sat)), l);
        }
    }

    // --- contrast ---

    #[test]
    fn contrast_one_is_identity() {
        let rgb = [0.05, 0.18, 0.9];
        approx3(contrast(rgb, 1.0, 0.18), rgb);
    }

    #[test]
    fn contrast_pivot_is_fixed_point() {
        let pivot = 0.18;
        let rgb = [pivot, pivot, pivot];
        for &amt in &[0.0, 0.5, 1.0, 2.0] {
            approx3(contrast(rgb, amt, pivot), rgb);
        }
    }

    #[test]
    fn contrast_increases_spread() {
        // > 1 pushes below-pivot down and above-pivot up.
        let pivot = 0.18;
        let dark = contrast([0.08, 0.08, 0.08], 2.0, pivot);
        let bright = contrast([0.5, 0.5, 0.5], 2.0, pivot);
        assert!(dark[0] < 0.08, "below-pivot should darken");
        assert!(bright[0] > 0.5, "above-pivot should brighten");
    }

    // --- offset ---

    #[test]
    fn offset_zero_is_identity() {
        let rgb = [0.2, 0.4, 0.6];
        approx3(offset(rgb, [0.0; 3]), rgb);
    }

    #[test]
    fn offset_shifts_per_channel() {
        let rgb = [0.2, 0.4, 0.6];
        approx3(offset(rgb, [0.1, -0.1, 0.05]), [0.3, 0.3, 0.65]);
    }

    // --- full pipeline ---

    #[test]
    fn apply_color_grade_default_is_identity() {
        let params = ColorGradeParams::default();
        let rgb = [0.15, 0.4, 0.85];
        approx3(apply_color_grade(rgb, &params), rgb);
    }

    #[test]
    fn color_grade_params_default_is_neutral() {
        let p = ColorGradeParams::default();
        approx(p.temperature, 0.0);
        approx(p.tint, 0.0);
        approx3(p.lift, [0.0; 3]);
        approx3(p.gamma, [1.0; 3]);
        approx3(p.gain, [1.0; 3]);
        approx(p.contrast, 1.0);
        approx(p.pivot, 0.18);
        approx(p.saturation, 1.0);
    }

    #[test]
    fn apply_color_grade_composes_stages() {
        // A non-neutral grade should equal the hand-composed stage order.
        let params = ColorGradeParams {
            temperature: 0.2,
            tint: -0.1,
            lift: [0.02, 0.0, 0.0],
            gamma: [1.1, 1.0, 0.9],
            gain: [1.05, 1.0, 0.95],
            contrast: 1.2,
            pivot: 0.18,
            saturation: 0.8,
        };
        let rgb = [0.3, 0.5, 0.7];
        let expected = {
            let c = white_balance(rgb, params.temperature, params.tint);
            let c = lift_gamma_gain(c, params.lift, params.gamma, params.gain);
            let c = contrast(c, params.contrast, params.pivot);
            color_saturation(c, params.saturation)
        };
        approx3(apply_color_grade(rgb, &params), expected);
    }

    #[test]
    fn luma_matches_rec709() {
        approx(luma([1.0, 1.0, 1.0]), 1.0);
        approx(luma([1.0, 0.0, 0.0]), 0.2126);
        approx(luma([0.0, 1.0, 0.0]), 0.7152);
        approx(luma([0.0, 0.0, 1.0]), 0.0722);
    }
}
