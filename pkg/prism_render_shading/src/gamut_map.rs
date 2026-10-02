//! Backend-neutral CPU golden for perceptual gamut mapping (gamut compression).
//!
//! Gamut mapping is a shared post-processing base that runs on the resolved,
//! pre-exposed *linear* HDR radiance (see [`crate::exposure`]) after grading and
//! before the display tone-map curve. Wide-gamut or over-saturated illumination
//! routinely produces colours that fall outside the target display gamut
//! (linear `sRGB` / Rec. 709): a channel goes negative, or a ratio pushes past a
//! primary. Naively clamping such colours to `[0, 1]` shifts their hue, so this
//! module instead *compresses* the out-of-gamut region smoothly back toward the
//! achromatic axis, the approach `OpenColorIO` and the `ACES` gamut-compress
//! transform take (as exposed in tools like Nuke). Every illumination model,
//! physical or stylized, writes into the same HDR buffer, so one gamut map
//! serves them all.
//!
//! The operators, applied in a fixed order by [`apply_gamut_compress`], are:
//!
//! * **Achromatic anchor.** [`achromatic`] returns `max(RGB)`, the neutral axis
//!   the compression pivots around. The max channel is a fixed point of the map.
//! * **Relative distance.** [`distance_from_achromatic`] measures each channel's
//!   `(ac - c) / ac` offset from the anchor. In-gamut positive colours sit in
//!   `[0, 1]`; a negative (out-of-gamut) channel exceeds `1`.
//! * **Distance compression.** [`compress_distance`] leaves distances below a
//!   per-channel `threshold` untouched (so the working gamut is bit-exact) and
//!   pulls larger distances along a rational knee that asymptotes to `limit`, so
//!   the far out-of-gamut region is bounded without a hard clip.
//! * **Reconstruction.** The compressed distance rebuilds the channel as
//!   `ac - dist_compressed * ac`, preserving the anchor and the hue.
//!
//! The knee is the transcendental-free rational curve
//! `threshold + x / (1 + x / (limit - threshold))` (with `x = dist - threshold`)
//! rather than the `ACES` `pow`/`tanh` form, so nothing touches the disallowed
//! `f32` transcendental path and the whole module stays polynomial/rational. It
//! is mirrored arm-for-arm by `shaders/gamut_map.wesl` (same operation order,
//! same constants, same guards) so the CPU golden and the GPU twin agree.

/// Rec. 709 luma weights (linear `sRGB` primaries), used by [`luminance`].
pub const GAMUT_MAP_LUMA_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Rec. 709 relative luminance of a linear RGB sample.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * GAMUT_MAP_LUMA_WEIGHTS[0]
        + rgb[1] * GAMUT_MAP_LUMA_WEIGHTS[1]
        + rgb[2] * GAMUT_MAP_LUMA_WEIGHTS[2]
}

/// Largest of the three linear channels.
#[must_use]
pub fn max_channel(rgb: [f32; 3]) -> f32 {
    rgb[0].max(rgb[1]).max(rgb[2])
}

/// Smallest of the three linear channels.
#[must_use]
pub fn min_channel(rgb: [f32; 3]) -> f32 {
    rgb[0].min(rgb[1]).min(rgb[2])
}

/// Achromatic (neutral-axis) anchor the compression pivots around.
///
/// Uses `max(RGB)` (the `ACES` gamut-compress convention): the max channel then
/// has zero distance from the anchor and is a fixed point of the whole map, so
/// the brightest primary is preserved exactly.
#[must_use]
pub fn achromatic(rgb: [f32; 3]) -> f32 {
    max_channel(rgb)
}

/// Per-channel relative distance from the achromatic anchor `ac`.
///
/// Each channel maps to `(ac - c) / ac`: `0` on the anchor, `1` at a primary
/// (channel `0`) and `> 1` for an out-of-gamut (negative) channel. A
/// non-positive anchor is degenerate (a black or negative sample), so the
/// distance collapses to `0` there to stay finite.
#[must_use]
pub fn distance_from_achromatic(rgb: [f32; 3], ac: f32) -> [f32; 3] {
    if ac <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [(ac - rgb[0]) / ac, (ac - rgb[1]) / ac, (ac - rgb[2]) / ac]
}

/// Compress a single relative distance along a rational knee.
///
/// Distances below `threshold` (and the degenerate `limit <= threshold`) are
/// returned unchanged, keeping the working gamut bit-exact. Above `threshold`
/// the excess `x = dist - threshold` follows
/// `threshold + x / (1 + x / (limit - threshold))`, a smooth monotone curve that
/// asymptotes to `limit` as `dist` grows — the transcendental-free stand-in for
/// the `ACES` / Nuke gamut-compress knee. Pick `limit <= 1` to pull the far
/// out-of-gamut region back onto or inside the gamut boundary.
#[must_use]
pub fn compress_distance(dist: f32, threshold: f32, limit: f32) -> f32 {
    let span = limit - threshold;
    if span <= 0.0 || dist < threshold {
        return dist;
    }
    let x = dist - threshold;
    threshold + x / (1.0 + x / span)
}

/// Clamp a scalar to the `[0, 1]` display range.
#[must_use]
pub fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Artist controls for the gamut compression. `Default` is disabled, i.e. a
/// bit-exact identity for every input (see the per-field defaults).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GamutMapParams {
    /// Per-channel distance below which colours are left untouched (the working
    /// gamut). Neutral `[1, 1, 1]`.
    pub threshold: [f32; 3],
    /// Per-channel asymptotic compressed distance (the gamut boundary target).
    /// Default `[1.1, 1.1, 1.1]`; use `<= 1` to fully contain the gamut.
    pub limit: [f32; 3],
    /// Reserved perceptual shaping exponent for future knee tuning. The current
    /// rational curve does not consume it; carried through the ABI so the GPU
    /// twin's uniform layout matches. Neutral `1`.
    pub power: f32,
    /// Master enable. When `false` the map is a hard identity regardless of the
    /// other fields. Default `false`.
    pub enabled: bool,
}

impl Default for GamutMapParams {
    fn default() -> Self {
        Self {
            threshold: [1.0, 1.0, 1.0],
            limit: [1.1, 1.1, 1.1],
            power: 1.0,
            enabled: false,
        }
    }
}

/// Apply the full gamut compression in the fixed order
/// anchor -> distance -> per-channel compress -> reconstruct, then blend by the
/// master enable.
///
/// In-gamut colours (every channel distance below `threshold`) reconstruct to
/// themselves, so the working gamut is preserved bit-exactly; only the
/// out-of-gamut excess is pulled in. When [`GamutMapParams::enabled`] is
/// `false` the result is the untouched input.
#[must_use]
pub fn apply_gamut_compress(rgb: [f32; 3], params: &GamutMapParams) -> [f32; 3] {
    let ac = achromatic(rgb);
    let dist = distance_from_achromatic(rgb, ac);
    let cd = [
        compress_distance(dist[0], params.threshold[0], params.limit[0]),
        compress_distance(dist[1], params.threshold[1], params.limit[1]),
        compress_distance(dist[2], params.threshold[2], params.limit[2]),
    ];
    let compressed = [ac - cd[0] * ac, ac - cd[1] * ac, ac - cd[2] * ac];
    let t = if params.enabled { 1.0 } else { 0.0 };
    [
        rgb[0] + (compressed[0] - rgb[0]) * t,
        rgb[1] + (compressed[1] - rgb[1]) * t,
        rgb[2] + (compressed[2] - rgb[2]) * t,
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

    // --- max / min channel ---

    #[test]
    fn max_channel_picks_largest() {
        approx(max_channel([0.2, 0.9, 0.4]), 0.9);
        approx(max_channel([-0.3, -0.1, -0.5]), -0.1);
    }

    #[test]
    fn min_channel_picks_smallest() {
        approx(min_channel([0.2, 0.9, 0.4]), 0.2);
        approx(min_channel([-0.3, -0.1, -0.5]), -0.5);
    }

    // --- achromatic anchor ---

    #[test]
    fn achromatic_is_max_channel() {
        let rgb = [1.2, 0.5, -0.2];
        approx(achromatic(rgb), max_channel(rgb));
        approx(achromatic(rgb), 1.2);
    }

    // --- distance from achromatic ---

    #[test]
    fn distance_is_zero_for_grey() {
        // A neutral colour sits on the achromatic axis: every distance is 0.
        approx3(distance_from_achromatic([0.5, 0.5, 0.5], 0.5), [0.0; 3]);
    }

    #[test]
    fn distance_is_zero_at_max_channel() {
        let rgb = [1.2, 0.5, -0.2];
        let d = distance_from_achromatic(rgb, achromatic(rgb));
        approx(d[0], 0.0); // the max channel is always on the anchor
    }

    #[test]
    fn distance_positive_below_anchor() {
        let ac = 1.0;
        let d = distance_from_achromatic([1.0, 0.5, 0.0], ac);
        approx(d[0], 0.0);
        approx(d[1], 0.5);
        approx(d[2], 1.0); // a channel at 0 sits exactly on the primary
    }

    #[test]
    fn distance_exceeds_one_for_negative_channel() {
        let ac = 1.0;
        let d = distance_from_achromatic([1.0, 0.5, -0.2], ac);
        assert!(d[2] > 1.0, "a negative channel is out of gamut (dist > 1)");
        approx(d[2], 1.2);
    }

    #[test]
    fn distance_degenerate_when_anchor_nonpositive() {
        approx3(distance_from_achromatic([-0.1, -0.2, 0.0], 0.0), [0.0; 3]);
        approx3(distance_from_achromatic([-0.1, -0.2, -0.3], -0.1), [0.0; 3]);
    }

    // --- distance compression ---

    #[test]
    fn compress_below_threshold_is_identity() {
        approx(compress_distance(0.3, 0.8, 1.0), 0.3);
        approx(compress_distance(0.0, 0.8, 1.0), 0.0);
    }

    #[test]
    fn compress_at_threshold_is_identity() {
        approx(compress_distance(0.8, 0.8, 1.0), 0.8);
    }

    #[test]
    fn compress_above_threshold_is_reduced() {
        // dist 1.0, threshold 0.8, limit 1.0 -> 0.9 (pulled below the input).
        let c = compress_distance(1.0, 0.8, 1.0);
        approx(c, 0.9);
        assert!(c < 1.0, "compressed distance is pulled below the input");
        assert!(c > 0.8, "and stays above the threshold");
    }

    #[test]
    fn compress_asymptotes_to_limit() {
        let threshold = 0.8;
        let limit = 1.0;
        let c = compress_distance(1.0e4, threshold, limit);
        assert!(c < limit, "the knee never reaches the limit");
        assert!(limit - c < 1.0e-3, "but approaches it for large distances");
    }

    #[test]
    fn compress_is_monotonic_increasing() {
        let mut prev = compress_distance(0.0, 0.8, 1.0);
        let mut d = 0.8;
        while d <= 5.0 {
            let c = compress_distance(d, 0.8, 1.0);
            assert!(c >= prev, "compression must be monotone: {c} < {prev}");
            prev = c;
            d += 0.1;
        }
    }

    #[test]
    fn compress_degenerate_limit_is_identity() {
        // limit <= threshold has no valid span, so the curve is the identity.
        approx(compress_distance(2.0, 1.0, 1.0), 2.0);
        approx(compress_distance(2.0, 1.0, 0.5), 2.0);
    }

    // --- clamp01 ---

    #[test]
    fn clamp01_clamps_to_unit_range() {
        approx(clamp01(-0.5), 0.0);
        approx(clamp01(0.3), 0.3);
        approx(clamp01(1.7), 1.0);
    }

    // --- full pipeline ---

    #[test]
    fn params_default_is_disabled_identity() {
        let p = GamutMapParams::default();
        approx3(p.threshold, [1.0; 3]);
        approx3(p.limit, [1.1; 3]);
        approx(p.power, 1.0);
        assert!(!p.enabled);
    }

    #[test]
    fn apply_default_is_identity() {
        let params = GamutMapParams::default();
        let rgb = [1.4, 0.5, -0.2];
        approx3(apply_gamut_compress(rgb, &params), rgb);
    }

    #[test]
    fn apply_disabled_ignores_out_of_gamut() {
        // enabled = false is a hard identity even for out-of-gamut input.
        let params = GamutMapParams {
            threshold: [0.6, 0.6, 0.6],
            limit: [1.0, 1.0, 1.0],
            power: 1.0,
            enabled: false,
        };
        let rgb = [1.2, 0.5, -0.2];
        approx3(apply_gamut_compress(rgb, &params), rgb);
    }

    #[test]
    fn apply_in_gamut_is_identity_when_enabled() {
        // Every channel distance is below threshold, so nothing moves.
        let params = GamutMapParams {
            threshold: [0.6, 0.6, 0.6],
            limit: [1.0, 1.0, 1.0],
            power: 1.0,
            enabled: true,
        };
        let rgb = [1.0, 0.8, 0.7];
        approx3(apply_gamut_compress(rgb, &params), rgb);
    }

    #[test]
    fn apply_preserves_grey() {
        let params = GamutMapParams {
            threshold: [0.6, 0.6, 0.6],
            limit: [1.0, 1.0, 1.0],
            power: 1.0,
            enabled: true,
        };
        let grey = [0.5, 0.5, 0.5];
        approx3(apply_gamut_compress(grey, &params), grey);
    }

    #[test]
    fn apply_preserves_max_channel() {
        // The achromatic anchor (max channel) is a fixed point of the map.
        let params = GamutMapParams {
            threshold: [0.6, 0.6, 0.6],
            limit: [1.0, 1.0, 1.0],
            power: 1.0,
            enabled: true,
        };
        let rgb = [1.2, 0.5, -0.2];
        let out = apply_gamut_compress(rgb, &params);
        approx(out[0], 1.2);
    }

    #[test]
    fn apply_pulls_out_of_gamut_channel_inward() {
        let params = GamutMapParams {
            threshold: [0.6, 0.6, 0.6],
            limit: [1.0, 1.0, 1.0],
            power: 1.0,
            enabled: true,
        };
        let rgb = [1.2, 0.5, -0.2];
        let out = apply_gamut_compress(rgb, &params);
        assert!(
            out[2] > rgb[2],
            "the negative channel is raised toward gamut"
        );
        assert!(out[2] >= 0.0, "and brought back into the displayable range");
        // The in-gamut middle channel (dist 0.5833 < threshold) is untouched.
        approx(out[1], 0.5);
    }

    #[test]
    fn apply_matches_hand_composed_stages() {
        let params = GamutMapParams {
            threshold: [0.7, 0.7, 0.7],
            limit: [1.0, 1.0, 1.0],
            power: 1.0,
            enabled: true,
        };
        let rgb = [1.5, 0.4, -0.3];
        let expected = {
            let ac = achromatic(rgb);
            let dist = distance_from_achromatic(rgb, ac);
            let cd = [
                compress_distance(dist[0], params.threshold[0], params.limit[0]),
                compress_distance(dist[1], params.threshold[1], params.limit[1]),
                compress_distance(dist[2], params.threshold[2], params.limit[2]),
            ];
            [ac - cd[0] * ac, ac - cd[1] * ac, ac - cd[2] * ac]
        };
        approx3(apply_gamut_compress(rgb, &params), expected);
    }
}
