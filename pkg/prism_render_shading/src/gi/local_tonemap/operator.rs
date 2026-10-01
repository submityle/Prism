//! Local tone-mapping operators: Reinhard-local base compression and
//! Mertens-style exposure-fusion weights — backend-neutral CPU golden.
//!
//! Once [`super::bilateral`] has split log-luminance into a coarse base and a
//! fine detail layer, the operator stage decides *how much* to squeeze the base
//! and *how* to recombine it. Two classic ingredients live here:
//!
//! * **Reinhard-local** — the photographic operator `L / (1 + L)` evaluated
//!   against a *local* adaptation luminance instead of a single global key, with
//!   an optional white point so a chosen highlight maps to display white. Local
//!   adaptation is what gives the operator its dodge-and-burn behaviour: bright
//!   regions are compressed harder than their dark neighbours.
//! * **Exposure-fusion weights** — Mertens, Kautz & Van Reeth's three quality
//!   measures (contrast, saturation, well-exposedness) that score how "good" a
//!   pixel looks at a given exposure. We expose each measure and their product
//!   so a caller can blend differently exposed renders of a scene without ever
//!   building an explicit HDR radiance map.
//!
//! The headline entry point, [`compress_base_detail`], ties the two together:
//! it compresses the base layer in the linear domain, re-adds the gained detail
//! in the log domain, and returns a display-referred luminance in `[0, 1]`.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe / global state.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Inputs are floored where a curve expects non-negative light; luminance
//!   outputs are clamped to `[0, 1]`. No path can emit `NaN` or `inf`.
//!
//! # References
//! * Reinhard et al., "Photographic Tone Reproduction for Digital Images",
//!   SIGGRAPH 2002 — local operator with per-pixel adaptation and white point.
//! * Mertens, Kautz & Van Reeth, "Exposure Fusion", Pacific Graphics 2007 —
//!   contrast / saturation / well-exposedness quality weights.

use bevy_math::ops;

use super::bilateral::BaseDetail;
use super::luminance::{exp_luminance, log_luminance};

/// Numerical floor shared by the operator stage.
pub const OPERATOR_EPSILON: f32 = 1.0e-6;

/// Reinhard operator driven by a *local* adaptation luminance.
///
/// `l` is the pixel luminance, `local_adaptation` is the surrounding (base-layer)
/// luminance that sets the local key. With `key` controlling overall brightness
/// the operator is `s / (1 + s)` where `s = key * l / local_adaptation`. A
/// non-positive adaptation degrades gracefully to a global Reinhard on `l`.
#[must_use]
pub fn reinhard_local(l: f32, local_adaptation: f32, key: f32) -> f32 {
    let l = l.max(0.0);
    let key = if key.is_finite() { key.max(OPERATOR_EPSILON) } else { 1.0 };
    let adapt = if local_adaptation.is_finite() {
        local_adaptation.max(OPERATOR_EPSILON)
    } else {
        OPERATOR_EPSILON
    };
    let scaled = key * l / adapt;
    let out = scaled / (1.0 + scaled);
    if out.is_finite() { out.clamp(0.0, 1.0) } else { 0.0 }
}

/// Reinhard-local with a white point, so a luminance equal to `white` (after the
/// local scaling) maps to exactly `1.0` and brighter values clip.
///
/// `s = key * l / local_adaptation`; the curve is
/// `s * (1 + s / white^2) / (1 + s)`. A non-positive `white` falls back to the
/// plain [`reinhard_local`].
#[must_use]
pub fn reinhard_local_white(l: f32, local_adaptation: f32, key: f32, white: f32) -> f32 {
    if !(white.is_finite() && white > 0.0) {
        return reinhard_local(l, local_adaptation, key);
    }
    let l = l.max(0.0);
    let key = if key.is_finite() { key.max(OPERATOR_EPSILON) } else { 1.0 };
    let adapt = if local_adaptation.is_finite() {
        local_adaptation.max(OPERATOR_EPSILON)
    } else {
        OPERATOR_EPSILON
    };
    let s = key * l / adapt;
    let white_sq = white * white;
    let out = s * (1.0 + s / white_sq) / (1.0 + s);
    if out.is_finite() { out.clamp(0.0, 1.0) } else { 0.0 }
}

/// Mertens *well-exposedness* weight for a single channel value in `[0, 1]`.
///
/// A Gaussian centred on mid-grey (`0.5`): pixels near `0.5` score highest,
/// crushed shadows and blown highlights score near zero. `sigma` is floored to a
/// small positive value.
#[must_use]
pub fn well_exposedness(value: f32, sigma: f32) -> f32 {
    let s = if sigma.is_finite() { sigma.max(OPERATOR_EPSILON) } else { 0.2 };
    let d = value.clamp(0.0, 1.0) - 0.5;
    let e = -(d * d) / (2.0 * s * s);
    if e.is_finite() { ops::exp(e) } else { 0.0 }
}

/// Mertens well-exposedness over an RGB triple: the product of the per-channel
/// weights, which favours pixels that are well exposed in *every* channel.
#[must_use]
pub fn well_exposedness_rgb(rgb: [f32; 3], sigma: f32) -> f32 {
    well_exposedness(rgb[0], sigma)
        * well_exposedness(rgb[1], sigma)
        * well_exposedness(rgb[2], sigma)
}

/// Mertens *saturation* weight: the standard deviation of the three channels.
///
/// Vivid pixels (channels far apart) score high; greys score near zero. The
/// result is non-negative and finite.
#[must_use]
pub fn saturation_weight(rgb: [f32; 3]) -> f32 {
    let r = rgb[0].max(0.0);
    let g = rgb[1].max(0.0);
    let b = rgb[2].max(0.0);
    let mean = (r + g + b) / 3.0;
    let var = ((r - mean) * (r - mean) + (g - mean) * (g - mean) + (b - mean) * (b - mean)) / 3.0;
    let s = var.max(0.0).sqrt();
    if s.is_finite() { s } else { 0.0 }
}

/// Mertens *contrast* weight from a 3-tap Laplacian response.
///
/// The caller supplies the centre luminance and its two axis neighbours'
/// luminances; the weight is `|4*c - sum(neighbours) * 2|`-style second
/// derivative magnitude. Here we take the absolute discrete Laplacian over the
/// provided neighbourhood (centre plus up to four neighbours), normalised by the
/// neighbour count so windows of different sizes stay comparable.
#[must_use]
pub fn contrast_weight(center: f32, neighbors: &[f32]) -> f32 {
    if neighbors.is_empty() {
        return 0.0;
    }
    let c = sanitize(center);
    let mut acc = 0.0_f32;
    let mut n = 0.0_f32;
    for &raw in neighbors {
        acc += c - sanitize(raw);
        n += 1.0;
    }
    // Discrete Laplacian magnitude: |n*c - sum(neighbors)| == |sum(c - ni)|.
    let lap = (acc / n.max(1.0)).abs();
    if lap.is_finite() { lap } else { 0.0 }
}

/// Combined Mertens fusion weight: `contrast^wc * saturation^ws * exposure^we`.
///
/// Exponents let a caller weight the three measures (Mertens' default is `1.0`
/// each). All exponents are clamped to `[0, 8]`; the result is non-negative and
/// finite.
#[must_use]
pub fn fusion_weight(
    contrast: f32,
    saturation: f32,
    exposure: f32,
    exponents: [f32; 3],
) -> f32 {
    let term = |base: f32, exp: f32| -> f32 {
        let b = base.max(0.0);
        let e = if exp.is_finite() { exp.clamp(0.0, 8.0) } else { 1.0 };
        // Add a tiny epsilon so a zero measure does not annihilate the product
        // unless its exponent genuinely demands it.
        let v = ops::powf(b + OPERATOR_EPSILON, e);
        if v.is_finite() { v } else { 0.0 }
    };
    let w = term(contrast, exponents[0]) * term(saturation, exponents[1]) * term(exposure, exponents[2]);
    if w.is_finite() { w.max(0.0) } else { 0.0 }
}

/// Compress a base/detail pair into a display-referred luminance in `[0, 1]`.
///
/// Steps:
/// 1. Exponentiate the base back to linear luminance.
/// 2. Apply [`reinhard_local_white`] using `local_adaptation` as the key source,
///    yielding a compressed display luminance in `[0, 1]`.
/// 3. Re-add the (gained) detail in the log domain so fine contrast survives the
///    base squeeze, then clamp back to `[0, 1]`.
///
/// This keeps large-scale lighting under control while preserving the local
/// texture that global operators tend to wash out.
#[must_use]
pub fn compress_base_detail(
    bd: BaseDetail,
    local_adaptation: f32,
    key: f32,
    white: f32,
    detail_gain: f32,
) -> f32 {
    // Linear base luminance from the log-domain base layer.
    let base_linear = exp_luminance(bd.base);
    // Compressed display luminance for the base.
    let compressed = reinhard_local_white(base_linear, local_adaptation, key, white);
    // Re-add detail multiplicatively (additive in log) so texture survives.
    let gain = if detail_gain.is_finite() {
        detail_gain.clamp(0.0, 16.0)
    } else {
        1.0
    };
    let log_compressed = log_luminance(compressed);
    let recombined = exp_luminance(log_compressed + bd.detail * gain);
    if recombined.is_finite() {
        recombined.clamp(0.0, 1.0)
    } else {
        compressed.clamp(0.0, 1.0)
    }
}

/// Replace non-finite values with zero.
#[inline]
fn sanitize(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
    }

    #[test]
    fn reinhard_local_zero_is_zero() {
        approx(reinhard_local(0.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn reinhard_local_saturates() {
        let big = reinhard_local(1.0e6, 1.0, 1.0);
        assert!(big < 1.0 && big > 0.99);
    }

    #[test]
    fn reinhard_local_is_monotonic_in_luminance() {
        let a = reinhard_local(0.1, 1.0, 1.0);
        let b = reinhard_local(1.0, 1.0, 1.0);
        let c = reinhard_local(10.0, 1.0, 1.0);
        assert!(a < b && b < c);
    }

    #[test]
    fn local_adaptation_dodges_and_burns() {
        // Same pixel luminance, brighter surround -> stronger compression
        // (lower output). This is the dodge/burn behaviour.
        let dark_surround = reinhard_local(1.0, 0.2, 1.0);
        let bright_surround = reinhard_local(1.0, 5.0, 1.0);
        assert!(bright_surround < dark_surround);
    }

    #[test]
    fn reinhard_local_handles_bad_adaptation() {
        assert!(reinhard_local(1.0, 0.0, 1.0).is_finite());
        assert!(reinhard_local(1.0, -3.0, 1.0).is_finite());
        assert!(reinhard_local(1.0, f32::NAN, 1.0).is_finite());
    }

    #[test]
    fn reinhard_white_hits_one_at_white() {
        // With adaptation = key = 1, s == l; choosing white == l maps to 1.0.
        let out = reinhard_local_white(4.0, 1.0, 1.0, 4.0);
        approx(out, 1.0);
    }

    #[test]
    fn reinhard_white_falls_back_for_bad_white() {
        let a = reinhard_local_white(2.0, 1.0, 1.0, 0.0);
        let b = reinhard_local(2.0, 1.0, 1.0);
        approx(a, b);
    }

    #[test]
    fn well_exposedness_peaks_at_mid_grey() {
        let mid = well_exposedness(0.5, 0.2);
        let dark = well_exposedness(0.0, 0.2);
        let bright = well_exposedness(1.0, 0.2);
        assert!(mid > dark && mid > bright);
        approx(mid, 1.0);
    }

    #[test]
    fn well_exposedness_bad_sigma_is_finite() {
        assert!(well_exposedness(0.5, 0.0).is_finite());
        assert!(well_exposedness(0.5, f32::NAN).is_finite());
    }

    #[test]
    fn well_exposedness_rgb_is_product() {
        let rgb = [0.5, 0.5, 0.5];
        approx(well_exposedness_rgb(rgb, 0.2), 1.0);
        let off = [0.0, 0.5, 1.0];
        let expected = well_exposedness(0.0, 0.2)
            * well_exposedness(0.5, 0.2)
            * well_exposedness(1.0, 0.2);
        approx(well_exposedness_rgb(off, 0.2), expected);
    }

    #[test]
    fn saturation_zero_for_grey() {
        approx(saturation_weight([0.4, 0.4, 0.4]), 0.0);
    }

    #[test]
    fn saturation_positive_for_colorful() {
        assert!(saturation_weight([1.0, 0.0, 0.0]) > 0.0);
    }

    #[test]
    fn contrast_zero_on_flat_region() {
        approx(contrast_weight(1.0, &[1.0, 1.0, 1.0, 1.0]), 0.0);
    }

    #[test]
    fn contrast_positive_on_edge() {
        assert!(contrast_weight(1.0, &[0.0, 0.0]) > 0.0);
    }

    #[test]
    fn contrast_empty_neighbors_is_zero() {
        approx(contrast_weight(1.0, &[]), 0.0);
    }

    #[test]
    fn fusion_weight_product_of_measures() {
        let w = fusion_weight(0.5, 0.5, 0.5, [1.0, 1.0, 1.0]);
        assert!(w > 0.0 && w.is_finite());
    }

    #[test]
    fn fusion_weight_zero_exponent_ignores_measure() {
        // Exponent 0 -> that measure contributes a factor of 1.
        let with = fusion_weight(0.0, 0.5, 0.5, [0.0, 1.0, 1.0]);
        let expected = fusion_weight(1.0, 0.5, 0.5, [0.0, 1.0, 1.0]);
        approx(with, expected);
    }

    #[test]
    fn fusion_weight_clamps_bad_exponent() {
        assert!(fusion_weight(0.5, 0.5, 0.5, [f32::NAN, 1.0e9, -2.0]).is_finite());
    }

    #[test]
    fn compress_outputs_unit_range() {
        for &b in &[-5.0_f32, 0.0, 2.0, 10.0] {
            let bd = BaseDetail {
                base: b,
                detail: 0.3,
            };
            let out = compress_base_detail(bd, 1.0, 1.0, 6.0, 1.0);
            assert!((0.0..=1.0).contains(&out), "b={b} out={out}");
        }
    }

    #[test]
    fn compress_detail_gain_increases_output_on_positive_detail() {
        let bd = BaseDetail {
            base: 0.0,
            detail: 0.5,
        };
        let low = compress_base_detail(bd, 1.0, 1.0, 6.0, 0.5);
        let high = compress_base_detail(bd, 1.0, 1.0, 6.0, 2.0);
        assert!(high >= low);
    }

    #[test]
    fn compress_never_emits_non_finite() {
        let bd = BaseDetail {
            base: f32::INFINITY,
            detail: f32::NAN,
        };
        let out = compress_base_detail(bd, f32::NAN, f32::NAN, f32::NAN, f32::NAN);
        assert!(out.is_finite());
        assert!((0.0..=1.0).contains(&out));
    }
}
