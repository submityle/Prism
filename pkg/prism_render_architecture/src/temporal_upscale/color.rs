//! Color-space primitives for temporal reconstruction.
//!
//! A temporal upsampler (`DLSS` / `FSR` / `XeSS`-class) decides how much of the
//! accumulated history to keep by comparing it against the current frame's
//! local neighborhood. Those comparisons are far more robust in a
//! luma/chroma space than in raw linear `RGB`, and `HDR` averaging needs a
//! firefly-suppressing weight so a single bright sample cannot dominate the
//! accumulation.
//!
//! This module owns the deterministic, transcendental-free building blocks for
//! that: the Rec. 709 luma, the reversible `YCoCg` transform used for
//! neighborhood clipping, and the exactly-invertible Karis tone-map pair used
//! to blend in a bounded domain. Every operation is `+`, `-`, `*`, `/`, and
//! `min`/`max` only, so a future `GPU` kernel can reproduce the results
//! bit-for-bit.

/// Rec. 709 relative luminance of a linear `RGB` color.
///
/// These are the standard luma coefficients for the Rec. 709 / sRGB primaries;
/// the result is the perceived brightness used for anti-flicker weighting and
/// history-rejection heuristics.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Converts a linear `RGB` color to the `YCoCg` luma/chroma space.
///
/// `YCoCg` is a cheap, reversible rotation of `RGB` whose axes (luma `Y`,
/// orange-chroma `Co`, green-chroma `Cg`) decorrelate brightness from color.
/// Temporal reconstructors clip history in this space because an axis-aligned
/// box there is a far tighter bound on plausible colors than one in `RGB`.
///
/// The transform uses the lifting form `Y = R/4 + G/2 + B/4`,
/// `Co = R/2 - B/2`, `Cg = -R/4 + G/2 - B/4`, which [`ycocg_to_rgb`] inverts
/// exactly.
#[must_use]
pub fn rgb_to_ycocg(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb;
    let y = 0.25 * r + 0.5 * g + 0.25 * b;
    let co = 0.5 * r - 0.5 * b;
    let cg = -0.25 * r + 0.5 * g - 0.25 * b;
    [y, co, cg]
}

/// Inverse of [`rgb_to_ycocg`]: maps `YCoCg` back to linear `RGB`.
///
/// The inverse is `R = Y + Co - Cg`, `G = Y + Cg`, `B = Y - Co - Cg`, which
/// recovers the original `RGB` triple exactly (up to floating-point rounding)
/// for any input produced by [`rgb_to_ycocg`].
#[must_use]
pub fn ycocg_to_rgb(ycocg: [f32; 3]) -> [f32; 3] {
    let [y, co, cg] = ycocg;
    let r = y + co - cg;
    let g = y + cg;
    let b = y - co - cg;
    [r, g, b]
}

/// Firefly-suppressing weight for averaging an `HDR` sample, in `(0, 1]`.
///
/// Weighting each sample by `1 / (1 + luma)` before averaging (Karis) pulls the
/// mean away from rare, very bright pixels so a single firefly cannot dominate
/// the accumulated color. The weight is `1` for black and tends to `0` as the
/// sample brightens; a non-finite or negative luma is treated as `0` brightness
/// (full weight) so the fallback stays deterministic.
#[must_use]
pub fn tonemap_weight(luma: f32) -> f32 {
    if luma.is_nan() || luma <= 0.0 {
        // Covers NaN and non-positive luma: a dark/degenerate sample gets the
        // maximum weight of 1 and never divides by a value below 1.
        1.0
    } else {
        1.0 / (1.0 + luma)
    }
}

/// Maps a linear `HDR` color into the bounded Karis tone-map domain.
///
/// The operator `c / (1 + m)`, where `m` is the largest color channel,
/// compresses `[0, inf)` into `[0, 1)` per channel while remaining exactly
/// invertible by [`untonemap`]. Blending history and current color in this
/// bounded domain keeps a bright background from smearing across a moving edge
/// (the classic `TAA` ghost). Negative channels are clamped to `0` first so the
/// denominator stays `>= 1` and the mapping is monotonic.
#[must_use]
pub fn tonemap(rgb: [f32; 3]) -> [f32; 3] {
    let m = max_channel(rgb).max(0.0);
    let inv = 1.0 / (1.0 + m);
    [rgb[0] * inv, rgb[1] * inv, rgb[2] * inv]
}

/// Inverse of [`tonemap`]: maps a tone-mapped color back to linear `HDR`.
///
/// Given a tone-mapped color whose largest channel is `m' = m / (1 + m)`, the
/// denominator `1 - m'` equals `1 / (1 + m)`, so dividing by it recovers the
/// original linear color exactly for any value produced by [`tonemap`]. The
/// largest channel of a tone-mapped color is strictly below `1`, so `1 - m'`
/// is always positive and the division is well-defined.
#[must_use]
pub fn untonemap(rgb: [f32; 3]) -> [f32; 3] {
    let m = max_channel(rgb).max(0.0);
    // `m` is in [0, 1) for any tone-mapped input, so `1 - m` is positive; the
    // `min` guards against a caller passing an out-of-domain value.
    let denom = (1.0 - m).max(f32::MIN_POSITIVE);
    let inv = 1.0 / denom;
    [rgb[0] * inv, rgb[1] * inv, rgb[2] * inv]
}

/// The largest of the three color channels.
///
/// Used by the tone-map pair to pick the compression factor; written as nested
/// `max` calls so `NaN` handling matches the standard library's `f32::max`.
#[must_use]
fn max_channel(rgb: [f32; 3]) -> f32 {
    rgb[0].max(rgb[1]).max(rgb[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-error tolerance for the round-trip and value checks.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    /// Asserts two colors agree channel-wise within [`approx`].
    fn approx_rgb(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    #[test]
    fn luminance_matches_rec709_weights() {
        assert!(approx(luminance([1.0, 0.0, 0.0]), 0.2126));
        assert!(approx(luminance([0.0, 1.0, 0.0]), 0.7152));
        assert!(approx(luminance([0.0, 0.0, 1.0]), 0.0722));
        // White sums to unity.
        assert!(approx(luminance([1.0, 1.0, 1.0]), 1.0));
    }

    #[test]
    fn ycocg_round_trips() {
        let colors = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.8, 0.1, 0.3],
            [0.2, 0.9, 0.4],
            [3.5, 0.25, 7.0],
        ];
        for c in colors {
            let back = ycocg_to_rgb(rgb_to_ycocg(c));
            assert!(
                approx_rgb(back, c),
                "round trip failed for {c:?} -> {back:?}"
            );
        }
    }

    #[test]
    fn ycocg_luma_channel_is_box_filtered_brightness() {
        // The Y channel of YCoCg is the (1/4, 1/2, 1/4) box luma, distinct from
        // the Rec. 709 luma; check the exact definition.
        let y = rgb_to_ycocg([0.4, 0.6, 0.8])[0];
        assert!(approx(y, 0.25 * 0.4 + 0.5 * 0.6 + 0.25 * 0.8));
    }

    /// Relative round-trip check: floating-point error in the tone-map pair
    /// scales with the color magnitude, so an `HDR` value in the hundreds needs
    /// a magnitude-relative tolerance rather than a fixed absolute one.
    fn approx_rel(back: [f32; 3], c: [f32; 3]) -> bool {
        (0..3).all(|i| (back[i] - c[i]).abs() <= 1e-4 * c[i].abs().max(1.0))
    }

    #[test]
    fn tonemap_round_trips_for_hdr() {
        let colors = [
            [0.0, 0.0, 0.0],
            [0.5, 0.25, 0.75],
            [1.0, 1.0, 1.0],
            [10.0, 2.0, 0.5],
            [100.0, 50.0, 25.0],
        ];
        for c in colors {
            let back = untonemap(tonemap(c));
            assert!(
                approx_rel(back, c),
                "tonemap round trip failed for {c:?} -> {back:?}"
            );
        }
    }

    #[test]
    fn tonemap_is_bounded_below_one() {
        // Even an extreme sample maps strictly inside the unit cube.
        let t = tonemap([1000.0, 1.0, 0.0]);
        assert!(t[0] < 1.0 && t[0] > 0.0, "max channel {t:?} not in (0, 1)");
    }

    #[test]
    fn tonemap_clamps_negative_input() {
        // A negative channel must not drive the denominator below 1.
        let t = tonemap([-5.0, 2.0, 0.0]);
        assert!(t.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn tonemap_weight_falls_with_brightness() {
        assert!(approx(tonemap_weight(0.0), 1.0));
        assert!(approx(tonemap_weight(1.0), 0.5));
        assert!(approx(tonemap_weight(9.0), 0.1));
        // Degenerate luma resolves to full weight.
        assert!(approx(tonemap_weight(f32::NAN), 1.0));
        assert!(approx(tonemap_weight(-2.0), 1.0));
    }
}
