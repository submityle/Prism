//! Backend-neutral CPU golden for NPR posterization (tone separation / colour
//! quantization).
//!
//! Posterize is a stylized post-processing base that runs on the resolved,
//! pre-exposed *linear* HDR radiance (see [`crate::exposure`]) before the
//! display tone-map curve. It collapses the continuous tonal range into a small
//! set of flat bands, the poster-print / cel-shaded look, by snapping values to
//! a quantization grid. Every illumination model, physical or stylized, writes
//! into the same HDR buffer, so one posterize serves them all.
//!
//! Two quantizers are offered, selected by [`PosterizeParams::use_luma`] and
//! composed by [`apply_posterize`]:
//!
//! * **Per-channel RGB.** [`quantize_rgb`] snaps each channel independently to
//!   `levels` bands via [`quantize`]. Cheap and punchy, but it can shift hue
//!   because the channels band at different values.
//! * **Luma-preserving.** [`quantize_luma_preserving`] quantizes only the Rec.
//!   709 luminance to `luma_levels` bands and rescales the original colour by
//!   `quant_luma / luma`, so the chroma ratios (and therefore the hue) survive
//!   the banding — the result reads as posterized brightness at the original
//!   colour.
//!
//! [`quantize`] uses `floor(x * (levels - 1) + 0.5) / (levels - 1)` — a
//! round-to-nearest grid snap built from `floor` alone, so nothing touches the
//! disallowed `f32` transcendental path (the module is pure arithmetic, no
//! `ops`). The quantized result is blended back over the input by `strength`
//! (`a + (b - a) * t`) so the effect can be dialled in. The whole module is
//! mirrored arm-for-arm by `shaders/posterize.wesl` (same operation order, same
//! constants, same guards) so the CPU golden and the GPU twin agree.

/// Rec. 709 luma weights (linear `sRGB` primaries), used by [`luminance`].
pub const POSTERIZE_LUMA_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Luminance floor below which the luma-preserving rescale is skipped (a black
/// or near-black sample has no stable chroma ratio to preserve).
pub const POSTERIZE_LUMA_EPS: f32 = 1.0e-5;

/// Rec. 709 relative luminance of a linear RGB sample.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * POSTERIZE_LUMA_WEIGHTS[0]
        + rgb[1] * POSTERIZE_LUMA_WEIGHTS[1]
        + rgb[2] * POSTERIZE_LUMA_WEIGHTS[2]
}

/// Snap a scalar to `levels` evenly spaced bands on `[0, 1]`.
///
/// Uses the round-to-nearest grid `floor(x * (levels - 1) + 0.5) / (levels - 1)`,
/// so `0` and `1` are always exact band endpoints. `levels <= 1` has no grid and
/// returns the input unchanged. Values outside `[0, 1]` snap to the extended
/// grid (they are not clamped here).
#[must_use]
pub fn quantize(x: f32, levels: f32) -> f32 {
    if levels <= 1.0 {
        return x;
    }
    let n = levels - 1.0;
    (x * n + 0.5).floor() / n
}

/// Per-channel RGB posterize: [`quantize`] applied independently per channel.
#[must_use]
pub fn quantize_rgb(rgb: [f32; 3], levels: f32) -> [f32; 3] {
    [
        quantize(rgb[0], levels),
        quantize(rgb[1], levels),
        quantize(rgb[2], levels),
    ]
}

/// Luma-preserving posterize: quantize the Rec. 709 luminance to `luma_levels`
/// bands and rescale the original colour by `quant_luma / luma`.
///
/// The chroma ratios (hence the hue) are preserved, and the result's luminance
/// equals the quantized luma. A luminance at or below [`POSTERIZE_LUMA_EPS`]
/// has no stable ratio, so the sample is returned unchanged to stay finite.
#[must_use]
pub fn quantize_luma_preserving(rgb: [f32; 3], luma_levels: f32) -> [f32; 3] {
    let l = luminance(rgb);
    if l <= POSTERIZE_LUMA_EPS {
        return rgb;
    }
    let ql = quantize(l, luma_levels);
    let scale = ql / l;
    [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
}

/// Artist controls for the posterize. `Default` is disabled, i.e. a bit-exact
/// identity for every input (see the per-field defaults).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosterizeParams {
    /// Band count for the per-channel RGB quantizer. Default `4`.
    pub levels: f32,
    /// Band count for the luma-preserving quantizer. Default `4`.
    pub luma_levels: f32,
    /// Select the luma-preserving quantizer (`true`) over per-channel RGB
    /// (`false`). Default `false`.
    pub use_luma: bool,
    /// Blend of the quantized result over the input, `a + (b - a) * strength`.
    /// Neutral `1` is full posterize. Default `1`.
    pub strength: f32,
    /// Master enable. When `false` the effect is a hard identity regardless of
    /// the other fields. Default `false`.
    pub enabled: bool,
}

impl Default for PosterizeParams {
    fn default() -> Self {
        Self {
            levels: 4.0,
            luma_levels: 4.0,
            use_luma: false,
            strength: 1.0,
            enabled: false,
        }
    }
}

/// Apply the posterize in the fixed order
/// quantize (per-channel RGB or luma-preserving) -> blend by `strength`.
///
/// When [`PosterizeParams::enabled`] is `false` the blend weight is `0`, so the
/// result is the untouched input.
#[must_use]
pub fn apply_posterize(rgb: [f32; 3], params: &PosterizeParams) -> [f32; 3] {
    let base = if params.use_luma {
        quantize_luma_preserving(rgb, params.luma_levels)
    } else {
        quantize_rgb(rgb, params.levels)
    };
    let t = if params.enabled { params.strength } else { 0.0 };
    [
        rgb[0] + (base[0] - rgb[0]) * t,
        rgb[1] + (base[1] - rgb[1]) * t,
        rgb[2] + (base[2] - rgb[2]) * t,
    ]
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

    // --- luminance ---

    #[test]
    fn luminance_matches_rec709() {
        approx(luminance([1.0, 1.0, 1.0]), 1.0);
        approx(luminance([1.0, 0.0, 0.0]), 0.2126);
        approx(luminance([0.0, 1.0, 0.0]), 0.7152);
        approx(luminance([0.0, 0.0, 1.0]), 0.0722);
    }

    #[test]
    fn luminance_is_weighted_sum() {
        let rgb = [0.2, 0.5, 0.9];
        let expected = 0.2 * 0.2126 + 0.5 * 0.7152 + 0.9 * 0.0722;
        approx(luminance(rgb), expected);
    }

    // --- scalar quantize ---

    #[test]
    fn quantize_levels_le_one_is_identity() {
        approx(quantize(0.37, 1.0), 0.37);
        approx(quantize(0.37, 0.5), 0.37);
        approx(quantize(0.37, 0.0), 0.37);
    }

    #[test]
    fn quantize_two_levels_is_round() {
        // levels = 2 -> n = 1 -> round to {0, 1}.
        approx(quantize(0.3, 2.0), 0.0);
        approx(quantize(0.7, 2.0), 1.0);
        approx(quantize(0.5, 2.0), 1.0); // floor(1.0) = 1
    }

    #[test]
    fn quantize_three_levels_grid() {
        // levels = 3 -> bands {0, 0.5, 1}.
        approx(quantize(0.1, 3.0), 0.0);
        approx(quantize(0.3, 3.0), 0.5);
        approx(quantize(0.6, 3.0), 0.5);
        approx(quantize(0.9, 3.0), 1.0);
    }

    #[test]
    fn quantize_preserves_endpoints() {
        for &levels in &[2.0, 3.0, 4.0, 8.0] {
            approx(quantize(0.0, levels), 0.0);
            approx(quantize(1.0, levels), 1.0);
        }
    }

    #[test]
    fn quantize_is_monotonic_nondecreasing() {
        let mut prev = quantize(0.0, 5.0);
        let mut x = 0.0;
        while x <= 1.0 {
            let q = quantize(x, 5.0);
            assert!(q >= prev, "quantize must be non-decreasing: {q} < {prev}");
            prev = q;
            x += 0.02;
        }
    }

    // --- quantize_rgb ---

    #[test]
    fn quantize_rgb_is_per_channel() {
        approx3(quantize_rgb([0.1, 0.3, 0.9], 3.0), [0.0, 0.5, 1.0]);
    }

    #[test]
    fn quantize_rgb_levels_le_one_is_identity() {
        let rgb = [0.2, 0.55, 0.83];
        approx3(quantize_rgb(rgb, 1.0), rgb);
    }

    // --- luma-preserving quantize ---

    #[test]
    fn quantize_luma_preserving_keeps_chroma_ratio() {
        let rgb = [0.2, 0.4, 0.8];
        let out = quantize_luma_preserving(rgb, 4.0);
        // Output is a uniform scale of the input, so channel ratios survive.
        let scale = out[0] / rgb[0];
        approx(out[1] / rgb[1], scale);
        approx(out[2] / rgb[2], scale);
    }

    #[test]
    fn quantize_luma_preserving_sets_quantized_luma() {
        let rgb = [0.2, 0.4, 0.8];
        let out = quantize_luma_preserving(rgb, 4.0);
        approx(luminance(out), quantize(luminance(rgb), 4.0));
    }

    #[test]
    fn quantize_luma_preserving_black_is_unchanged() {
        let rgb = [0.0, 0.0, 0.0];
        approx3(quantize_luma_preserving(rgb, 4.0), rgb);
        let tiny = [1.0e-7, 0.0, 1.0e-7];
        approx3(quantize_luma_preserving(tiny, 4.0), tiny);
    }

    // --- full pipeline ---

    #[test]
    fn params_default_is_disabled_identity() {
        let p = PosterizeParams::default();
        approx(p.levels, 4.0);
        approx(p.luma_levels, 4.0);
        assert!(!p.use_luma);
        approx(p.strength, 1.0);
        assert!(!p.enabled);
    }

    #[test]
    fn apply_default_is_identity() {
        let params = PosterizeParams::default();
        let rgb = [0.15, 0.4, 0.85];
        approx3(apply_posterize(rgb, &params), rgb);
    }

    #[test]
    fn apply_disabled_ignores_quantizer() {
        let params = PosterizeParams {
            levels: 3.0,
            luma_levels: 3.0,
            use_luma: false,
            strength: 1.0,
            enabled: false,
        };
        let rgb = [0.31, 0.62, 0.9];
        approx3(apply_posterize(rgb, &params), rgb);
    }

    #[test]
    fn apply_rgb_mode_matches_quantize_rgb() {
        let params = PosterizeParams {
            levels: 3.0,
            luma_levels: 3.0,
            use_luma: false,
            strength: 1.0,
            enabled: true,
        };
        let rgb = [0.1, 0.3, 0.9];
        approx3(apply_posterize(rgb, &params), quantize_rgb(rgb, 3.0));
    }

    #[test]
    fn apply_luma_mode_matches_quantize_luma() {
        let params = PosterizeParams {
            levels: 3.0,
            luma_levels: 4.0,
            use_luma: true,
            strength: 1.0,
            enabled: true,
        };
        let rgb = [0.2, 0.4, 0.8];
        approx3(
            apply_posterize(rgb, &params),
            quantize_luma_preserving(rgb, 4.0),
        );
    }

    #[test]
    fn apply_strength_zero_is_identity() {
        let params = PosterizeParams {
            levels: 2.0,
            luma_levels: 2.0,
            use_luma: false,
            strength: 0.0,
            enabled: true,
        };
        let rgb = [0.31, 0.62, 0.9];
        approx3(apply_posterize(rgb, &params), rgb);
    }

    #[test]
    fn apply_strength_blends_halfway() {
        let params = PosterizeParams {
            levels: 2.0,
            luma_levels: 2.0,
            use_luma: false,
            strength: 0.5,
            enabled: true,
        };
        let rgb = [0.3, 0.7, 0.5];
        let full = quantize_rgb(rgb, 2.0); // [0, 1, 1]
        let expected = [
            rgb[0] + (full[0] - rgb[0]) * 0.5,
            rgb[1] + (full[1] - rgb[1]) * 0.5,
            rgb[2] + (full[2] - rgb[2]) * 0.5,
        ];
        approx3(apply_posterize(rgb, &params), expected);
    }

    #[test]
    fn apply_matches_hand_composed_stages() {
        let params = PosterizeParams {
            levels: 5.0,
            luma_levels: 3.0,
            use_luma: true,
            strength: 0.75,
            enabled: true,
        };
        let rgb = [0.25, 0.5, 0.7];
        let base = quantize_luma_preserving(rgb, params.luma_levels);
        let t = params.strength;
        let expected = [
            rgb[0] + (base[0] - rgb[0]) * t,
            rgb[1] + (base[1] - rgb[1]) * t,
            rgb[2] + (base[2] - rgb[2]) * t,
        ];
        approx3(apply_posterize(rgb, &params), expected);
    }
}
