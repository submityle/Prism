//! Physically-aware bloom — backend-neutral CPU golden.
//!
//! Bloom scatters energy from the brightest pixels of the resolved, pre-exposed
//! HDR scene into their neighbourhood, approximating the light bleeding of a
//! real lens / sensor. This module is the numerical reference for the modern
//! AAA "dual filter" chain (Jimenez, *Next Generation Post Processing in Call
//! of Duty: Advanced Warfare*, SIGGRAPH 2014), the same structure UE's bloom
//! pass uses:
//!
//! * **Soft-knee threshold.** A Karis prefilter isolates the bright tail while
//!   keeping the transition smooth: below the threshold nothing bleeds, above
//!   it the surplus passes through, and a quadratic knee eases the two together
//!   so the contribution is continuous (a hard knee crushes the histogram and
//!   flickers under motion).
//! * **Karis average.** The first downsample weights each tap by
//!   `1 / (1 + luma)` so a single fireflies-bright sub-pixel cannot dominate a
//!   mip and pump energy every frame — the standard anti-flicker guard.
//! * **13-tap downsample.** The COD partial-Karis kernel: a centre 2x2 box at
//!   weight `0.5` plus four overlapping corner 2x2 boxes at weight `0.125`
//!   each. The weights sum to `1`, so the pyramid conserves energy.
//! * **9-tap tent upsample.** A 3x3 tent (`1/16` weights) spreads each coarse
//!   mip back up while adding it to the finer one; a `radius` widens or narrows
//!   that spread without changing the total weight.
//! * **Energy-conserving blend.** Per-mip weights sum to `1` so recombining the
//!   pyramid neither gains nor loses total energy, and the final composite is a
//!   `lerp` by an artist `intensity`.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe / global state.
//! * All kernel maths is plain `f32` arithmetic; no transcendental calls are
//!   needed, so the curve is exact.
//! * Brightness uses the max component (the COD/Karis choice); luminance uses
//!   Rec. 709 weights. Negative light is floored to zero; no path emits `NaN`.
//! * `f32` storage mirrors the layout the WESL/GPU twin consumes.

use alloc::vec::Vec;

/// Rec. 709 luminance weights (linear sRGB primaries).
pub const BLOOM_LUMINANCE_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Small denominator guard shared by the soft-knee prefilter and Karis weight.
pub const BLOOM_EPSILON: f32 = 1.0e-6;

/// Rec. 709 relative luminance of a linear RGB sample, floored to zero.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    (rgb[0] * BLOOM_LUMINANCE_WEIGHTS[0]
        + rgb[1] * BLOOM_LUMINANCE_WEIGHTS[1]
        + rgb[2] * BLOOM_LUMINANCE_WEIGHTS[2])
        .max(0.0)
}

/// Max-component "brightness" of a sample (`max(r, g, b)`), floored to zero.
#[must_use]
pub fn brightness(rgb: [f32; 3]) -> f32 {
    rgb[0].max(rgb[1]).max(rgb[2]).max(0.0)
}

/// Karis fireflies-suppression weight `1 / (1 + luma)`, strictly decreasing in
/// `luma`, so a lone very bright tap is pulled down before it can bloom the
/// whole mip. Negative `luma` is floored to zero (weight `1`).
#[must_use]
pub fn karis_average_weight(luma: f32) -> f32 {
    1.0 / (1.0 + luma.max(0.0))
}

/// Soft-knee threshold prefilter: the fraction of `color` that bleeds into the
/// bloom pyramid.
///
/// With max-component brightness `br`, threshold `t` and knee `k >= 0`:
///
/// ```text
/// soft         = clamp(br - (t - k), 0, 2k)
/// soft         = soft^2 / (4k + eps)
/// contribution = max(soft, br - t) / max(br, eps)
/// out          = color * contribution
/// ```
///
/// `contribution` is continuous in `br`, so there is no seam at the threshold.
/// A knee of `0` degrades to a hard threshold. Negative knees clamp to `0` so
/// the inner `clamp` bounds never invert, and sub-threshold pixels return pure
/// black (zero bleed).
#[must_use]
pub fn prefilter(color: [f32; 3], threshold: f32, knee: f32) -> [f32; 3] {
    let knee = knee.max(0.0);
    let br = brightness(color);
    let soft = (br - (threshold - knee)).clamp(0.0, 2.0 * knee);
    let soft = soft * soft / (4.0 * knee + BLOOM_EPSILON);
    let contribution = soft.max(br - threshold).max(0.0) / br.max(BLOOM_EPSILON);
    [
        color[0] * contribution,
        color[1] * contribution,
        color[2] * contribution,
    ]
}

/// Average of a 2x2 box of taps.
#[inline]
fn box_avg(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> [f32; 3] {
    [
        (a[0] + b[0] + c[0] + d[0]) * 0.25,
        (a[1] + b[1] + c[1] + d[1]) * 0.25,
        (a[2] + b[2] + c[2] + d[2]) * 0.25,
    ]
}

/// COD 13-tap downsample. The taps are laid out on a 5x5 grid (`.` = unused):
///
/// ```text
///   0 . 1 . 2
///   . 3 . 4 .
///   5 . 6 . 7     (6 = centre)
///   . 8 . 9 .
///  10 .11 .12
/// ```
///
/// The result is the centre 2x2 box `{3,4,8,9}` at weight `0.5` plus the four
/// overlapping corner boxes at weight `0.125` each. Weights sum to exactly `1`,
/// so a flat input passes through unchanged.
#[must_use]
pub fn downsample_13tap(samples: &[[f32; 3]; 13]) -> [f32; 3] {
    let center = box_avg(samples[3], samples[4], samples[8], samples[9]);
    let tl = box_avg(samples[0], samples[1], samples[5], samples[6]);
    let tr = box_avg(samples[1], samples[2], samples[6], samples[7]);
    let bl = box_avg(samples[5], samples[6], samples[10], samples[11]);
    let br = box_avg(samples[6], samples[7], samples[11], samples[12]);
    let mut out = [0.0_f32; 3];
    for c in 0..3 {
        out[c] = 0.5 * center[c] + 0.125 * (tl[c] + tr[c] + bl[c] + br[c]);
    }
    out
}

/// COD 13-tap downsample with per-box Karis weighting for the brightest mip.
///
/// Each 2x2 box average is weighted by `1 / (1 + luma(box))` (see
/// [`karis_average_weight`]) and the weighted boxes are renormalised, so a
/// fireflies-bright tap is attenuated before it can dominate. On a flat input
/// every box has equal weight and the result matches [`downsample_13tap`].
#[must_use]
pub fn downsample_13tap_karis(samples: &[[f32; 3]; 13]) -> [f32; 3] {
    let center = box_avg(samples[3], samples[4], samples[8], samples[9]);
    let tl = box_avg(samples[0], samples[1], samples[5], samples[6]);
    let tr = box_avg(samples[1], samples[2], samples[6], samples[7]);
    let bl = box_avg(samples[5], samples[6], samples[10], samples[11]);
    let br = box_avg(samples[6], samples[7], samples[11], samples[12]);

    // Base kernel weights (sum to 1) folded with per-box Karis attenuation.
    let boxes = [center, tl, tr, bl, br];
    let base = [0.5_f32, 0.125, 0.125, 0.125, 0.125];
    let mut out = [0.0_f32; 3];
    let mut total = 0.0_f32;
    for (b, &w) in boxes.iter().zip(base.iter()) {
        let kw = w * karis_average_weight(luminance(*b));
        for c in 0..3 {
            out[c] += b[c] * kw;
        }
        total += kw;
    }
    let inv = 1.0 / total.max(BLOOM_EPSILON);
    [out[0] * inv, out[1] * inv, out[2] * inv]
}

/// 9-tap 3x3 tent upsample with a `1/16` kernel (corners `1`, edges `2`, centre
/// `4`). Taps are row-major:
///
/// ```text
///   0 1 2
///   3 4 5     (4 = centre)
///   6 7 8
/// ```
///
/// `radius` (clamped to `[0, 1]`) blends the centre tap toward the full tent:
/// `0` returns just the centre, `1` is the canonical tent. Both endpoints are
/// weight-`1` filters, so the blend preserves total energy at every radius.
#[must_use]
pub fn upsample_tent(samples: &[[f32; 3]; 9], radius: f32) -> [f32; 3] {
    let r = radius.clamp(0.0, 1.0);
    let center = samples[4];
    let mut out = [0.0_f32; 3];
    for c in 0..3 {
        let tent = (samples[0][c] + samples[2][c] + samples[6][c] + samples[8][c])
            + 2.0 * (samples[1][c] + samples[3][c] + samples[5][c] + samples[7][c])
            + 4.0 * samples[4][c];
        let tent = tent / 16.0;
        out[c] = center[c] + (tent - center[c]) * r;
    }
    out
}

/// Per-mip contribution weights for an `levels`-deep pyramid, normalised to
/// sum to `1`. A geometric `falloff` (`> 0`) controls how fast coarse mips lose
/// weight: `weight[i] ∝ falloff^i`. `falloff == 1` is a flat (equal) blend,
/// `< 1` favours the finer mips (tight glow). Returns an empty `Vec` for `0`
/// levels; a non-positive `falloff` falls back to `0.5`.
#[must_use]
pub fn mip_blend_weights(levels: u32, falloff: f32) -> Vec<f32> {
    if levels == 0 {
        return Vec::new();
    }
    let f = if falloff > 0.0 { falloff } else { 0.5 };
    let mut raw = Vec::with_capacity(levels as usize);
    let mut w = 1.0_f32;
    let mut total = 0.0_f32;
    for _ in 0..levels {
        raw.push(w);
        total += w;
        w *= f;
    }
    let inv = 1.0 / total.max(BLOOM_EPSILON);
    for v in &mut raw {
        *v *= inv;
    }
    raw
}

/// Blends a stack of per-mip bloom colours into one, using energy-conserving
/// weights from [`mip_blend_weights`] with the given `falloff`. An empty stack
/// returns black.
#[must_use]
pub fn blend_mips(mips: &[[f32; 3]], falloff: f32) -> [f32; 3] {
    if mips.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let weights = mip_blend_weights(mips.len() as u32, falloff);
    let mut out = [0.0_f32; 3];
    for (m, &w) in mips.iter().zip(weights.iter()) {
        for c in 0..3 {
            out[c] += m[c] * w;
        }
    }
    out
}

/// Blends the accumulated `bloom` over the `scene` by an artist `intensity`:
/// `scene + (bloom - scene) * intensity`.
///
/// `intensity == 0` returns the untouched scene, `1` returns pure bloom. The
/// `intensity` is not clamped so callers may push past `1` for an over-bright
/// look; the output is floored to zero to stay non-negative.
#[must_use]
pub fn combine(scene: [f32; 3], bloom: [f32; 3], intensity: f32) -> [f32; 3] {
    [
        (scene[0] + (bloom[0] - scene[0]) * intensity).max(0.0),
        (scene[1] + (bloom[1] - scene[1]) * intensity).max(0.0),
        (scene[2] + (bloom[2] - scene[2]) * intensity).max(0.0),
    ]
}

/// Additive bloom composite often used when energy should only ever be added:
/// `scene + bloom * intensity`, floored to zero. Unlike [`combine`] this never
/// darkens the scene.
#[must_use]
pub fn combine_additive(scene: [f32; 3], bloom: [f32; 3], intensity: f32) -> [f32; 3] {
    let i = intensity.max(0.0);
    [
        (scene[0] + bloom[0] * i).max(0.0),
        (scene[1] + bloom[1] * i).max(0.0),
        (scene[2] + bloom[2] * i).max(0.0),
    ]
}

/// Artist controls for the bloom pass. `Default` is a neutral disabled bloom
/// (`intensity == 0` leaves the scene untouched) with a mild falloff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomParams {
    /// Brightness above which pixels start to bloom.
    pub threshold: f32,
    /// Width of the soft-threshold knee (`>= 0`; `0` is a hard threshold).
    pub knee: f32,
    /// Blend weight of the accumulated bloom over the scene in [`combine`].
    pub intensity: f32,
    /// Tent-upsample spread radius passed to [`upsample_tent`].
    pub radius: f32,
    /// Geometric per-mip falloff passed to [`mip_blend_weights`].
    pub mip_falloff: f32,
}

impl Default for BloomParams {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            knee: 0.0,
            intensity: 0.0,
            radius: 1.0,
            mip_falloff: 0.65,
        }
    }
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

    #[test]
    fn luminance_matches_rec709() {
        approx(luminance([1.0, 1.0, 1.0]), 1.0);
        approx(luminance([1.0, 0.0, 0.0]), 0.2126);
        approx(luminance([-1.0, -1.0, -1.0]), 0.0);
    }

    #[test]
    fn karis_weight_is_decreasing() {
        assert!(karis_average_weight(0.0) > karis_average_weight(1.0));
        assert!(karis_average_weight(1.0) > karis_average_weight(100.0));
        approx(karis_average_weight(0.0), 1.0);
        approx(karis_average_weight(-5.0), 1.0);
    }

    #[test]
    fn prefilter_below_threshold_is_black() {
        let out = prefilter([0.5, 0.5, 0.5], 1.0, 0.0);
        approx3(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn prefilter_hard_threshold_passes_surplus() {
        // br = 2, t = 1, knee 0 -> contribution = (2-1)/2 = 0.5.
        let out = prefilter([2.0, 2.0, 2.0], 1.0, 0.0);
        approx3(out, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn prefilter_is_continuous_at_threshold() {
        let t = 1.0;
        let k = 0.5;
        let just_below = prefilter([t - 1.0e-4, t - 1.0e-4, t - 1.0e-4], t, k)[0];
        let just_above = prefilter([t + 1.0e-4, t + 1.0e-4, t + 1.0e-4], t, k)[0];
        assert!((just_below - just_above).abs() < 1.0e-3);
    }

    #[test]
    fn prefilter_negative_knee_is_clamped() {
        let a = prefilter([2.0, 2.0, 2.0], 1.0, -3.0);
        let b = prefilter([2.0, 2.0, 2.0], 1.0, 0.0);
        approx3(a, b);
    }

    #[test]
    fn downsample_flat_input_passes_through() {
        let samples = [[0.7, 0.3, 0.1]; 13];
        approx3(downsample_13tap(&samples), [0.7, 0.3, 0.1]);
    }

    #[test]
    fn downsample_weights_sum_to_one() {
        // A delta at the centre tap sees total weight 0.5*0.25 + 4*(0.125*0.25)
        // across the boxes it belongs to; summing a unit flat field is the
        // cleaner invariant and is covered above. Here check energy of a mixed
        // field equals the weighted mean, i.e. no gain for a constant.
        let samples = [[2.0, 2.0, 2.0]; 13];
        approx3(downsample_13tap(&samples), [2.0, 2.0, 2.0]);
    }

    #[test]
    fn karis_downsample_matches_plain_on_flat() {
        let samples = [[1.5, 0.5, 0.25]; 13];
        approx3(downsample_13tap_karis(&samples), downsample_13tap(&samples));
    }

    #[test]
    fn karis_downsample_attenuates_firefly() {
        // One tap is extremely bright; Karis weighting must keep the result
        // well below the plain average, which the firefly would dominate.
        let mut samples = [[0.1, 0.1, 0.1]; 13];
        samples[6] = [1000.0, 1000.0, 1000.0];
        let plain = downsample_13tap(&samples)[0];
        let karis = downsample_13tap_karis(&samples)[0];
        assert!(karis < plain, "karis {karis} plain {plain}");
    }

    #[test]
    fn tent_flat_input_passes_through() {
        let samples = [[0.4, 0.5, 0.6]; 9];
        approx3(upsample_tent(&samples, 1.0), [0.4, 0.5, 0.6]);
    }

    #[test]
    fn tent_zero_radius_returns_center() {
        let mut samples = [[0.0, 0.0, 0.0]; 9];
        samples[4] = [0.9, 0.8, 0.7];
        approx3(upsample_tent(&samples, 0.0), [0.9, 0.8, 0.7]);
    }

    #[test]
    fn mip_weights_sum_to_one() {
        for levels in [1_u32, 2, 5, 8] {
            let w = mip_blend_weights(levels, 0.65);
            let sum: f32 = w.iter().sum();
            approx(sum, 1.0);
            assert_eq!(w.len(), levels as usize);
        }
        assert!(mip_blend_weights(0, 0.5).is_empty());
    }

    #[test]
    fn mip_weights_falloff_favours_fine() {
        let w = mip_blend_weights(4, 0.5);
        assert!(w[0] > w[1] && w[1] > w[2] && w[2] > w[3]);
    }

    #[test]
    fn mip_weights_flat_falloff_is_uniform() {
        let w = mip_blend_weights(4, 1.0);
        for v in &w {
            approx(*v, 0.25);
        }
    }

    #[test]
    fn blend_mips_conserves_constant() {
        let mips = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        approx3(blend_mips(&mips, 0.65), [1.0, 1.0, 1.0]);
        approx3(blend_mips(&[], 0.65), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn combine_endpoints() {
        let scene = [0.2, 0.4, 0.6];
        let bloom = [1.0, 1.0, 1.0];
        approx3(combine(scene, bloom, 0.0), scene);
        approx3(combine(scene, bloom, 1.0), bloom);
    }

    #[test]
    fn combine_additive_never_darkens() {
        let scene = [0.2, 0.4, 0.6];
        let bloom = [0.5, 0.5, 0.5];
        let out = combine_additive(scene, bloom, 1.0);
        assert!(out[0] >= scene[0] && out[1] >= scene[1] && out[2] >= scene[2]);
        approx3(combine_additive(scene, bloom, 0.0), scene);
    }
}
