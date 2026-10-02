//! Backend-neutral CPU golden for a physically-motivated, energy-preserving
//! bloom (light "glow" around bright features).
//!
//! Bloom is a shared post-processing base, not a peer of the PBR/NPR shading
//! fronts: it runs on the resolved, pre-exposed HDR radiance (see
//! [`crate::exposure`]) and scatters energy from the brightest pixels into
//! their neighbourhood, approximating light bleeding in a real lens/sensor.
//! Every illumination model — physically based or stylized — writes into the
//! same HDR buffer this pass reads, so one implementation serves them all.
//!
//! The pipeline is the modern AAA "dual filter" chain popularised by Jimenez's
//! *Next Generation Post Processing in Call of Duty: Advanced Warfare*
//! (SIGGRAPH 2014) and used by UE's `FBloomPass`:
//!
//! * **Prefilter.** A Karis soft-threshold curve isolates the bright tail while
//!   keeping the transition smooth (a hard knee crushes the histogram and
//!   flickers under motion). Below the threshold nothing bleeds; above it the
//!   surplus over the threshold passes through, with a quadratic "knee" easing
//!   the two together so the contribution is continuous at the threshold.
//! * **Karis average.** The first downsample weights each tap by
//!   `1 / (1 + luma)` so a single fireflies-bright sub-pixel cannot dominate a
//!   whole mip and pump energy every frame; this is the standard
//!   fireflies/anti-flicker guard.
//! * **13-tap downsample.** The COD partial-Karis kernel: a centre 2x2 box
//!   (weight `0.5`) plus four overlapping corner 2x2 boxes (weight `0.125`
//!   each). The weights sum to `1`, so the pyramid neither gains nor loses
//!   energy as it is built.
//! * **9-tap tent upsample.** A 3x3 tent (`1/16` weights) spreads each coarse
//!   mip back up while adding it to the finer one; a `radius` widens or narrows
//!   that spread without changing the total weight.
//! * **Combine.** The accumulated bloom is blended over the scene by an
//!   artist `intensity`.
//!
//! Everything is pure `f32` maths — no transcendental calls — mirrored
//! arm-for-arm by `shaders/bloom.wesl`, so the CPU golden and the GPU twin
//! agree bit-for-bit in intent.

use alloc::vec::Vec;

/// Rec. 709 luminance weights (linear sRGB primaries), shared with
/// [`crate::exposure`]. Kept local so the bloom golden is self-contained.
pub const BLOOM_LUMINANCE_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Rec. 709 relative luminance of a linear RGB radiance sample.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * BLOOM_LUMINANCE_WEIGHTS[0]
        + rgb[1] * BLOOM_LUMINANCE_WEIGHTS[1]
        + rgb[2] * BLOOM_LUMINANCE_WEIGHTS[2]
}

/// Karis soft-threshold prefilter. Returns the fraction of `color` that bleeds
/// into the bloom pyramid.
///
/// The channel "brightness" is the max component (`max(r, g, b)`, the classic
/// COD/Karis choice). With `threshold` `t` and `knee` `k >= 0`:
///
/// ```text
/// soft         = clamp(br - (t - k), 0, 2k)
/// soft         = soft^2 / (4k + 1e-6)
/// contribution = max(soft, br - t) / max(br, 1e-6)
/// out          = color * contribution
/// ```
///
/// `contribution` is a continuous function of `br`, so the output has no seam
/// at the threshold. A `knee` of `0` degrades gracefully to a hard threshold
/// (`soft == 0`, so `contribution = max(0, br - t) / br`). Negative knees are
/// clamped to `0` so the `clamp` bounds never invert.
#[must_use]
pub fn prefilter(color: [f32; 3], threshold: f32, knee: f32) -> [f32; 3] {
    let knee = knee.max(0.0);
    let br = color[0].max(color[1]).max(color[2]);
    let soft = (br - (threshold - knee)).clamp(0.0, 2.0 * knee);
    let soft = soft * soft / (4.0 * knee + 1.0e-6);
    let contribution = soft.max(br - threshold) / br.max(1.0e-6);
    [
        color[0] * contribution,
        color[1] * contribution,
        color[2] * contribution,
    ]
}

/// Karis fireflies-suppression weight `1 / (1 + luminance)` applied per tap on
/// the first (brightest) downsample. Strictly decreasing in `luminance`, so a
/// lone very bright sample is pulled down before it can bloom the whole mip.
#[must_use]
pub fn karis_average_weight(luminance: f32) -> f32 {
    1.0 / (1.0 + luminance.max(0.0))
}

/// COD 13-tap downsample. The taps are laid out on a 5x5 grid (`.` = unused):
///
/// ```text
///   0 . 1 . 2      a . b . c
///   . 3 . 4 .      . j . k .
///   5 . 6 . 7      d . e . f     (6 = centre)
///   . 8 . 9 .      . l . m .
///  10 .11 .12      g . h . i
/// ```
///
/// The result is the centre 2x2 box `{j,k,l,m}` at weight `0.5` plus the four
/// overlapping corner boxes `{a,b,d,e}`, `{b,c,e,f}`, `{d,e,g,h}`, `{e,f,h,i}`
/// at weight `0.125` each (every box averages its four taps). The weights sum
/// to exactly `1`, so a flat input passes through unchanged.
#[must_use]
pub fn downsample_13tap(samples: &[[f32; 3]; 13]) -> [f32; 3] {
    // Box averages, each 0.25 * (four taps).
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

/// 9-tap 3x3 tent upsample with a `1/16` kernel (corners `1`, edges `2`, centre
/// `4`). Taps are row-major:
///
/// ```text
///   0 1 2
///   3 4 5     (4 = centre)
///   6 7 8
/// ```
///
/// `radius` (clamped to `[0, 1]`) controls the spread by blending the centre
/// tap toward the full tent: `0` returns just the centre, `1` is the canonical
/// `1/16` tent. Because both endpoints are weight-`1` filters, the blend
/// preserves total energy for every radius.
#[must_use]
pub fn upsample_tent(samples: &[[f32; 3]; 9], radius: f32) -> [f32; 3] {
    let r = radius.clamp(0.0, 1.0);
    let center = samples[4];
    let mut out = [0.0_f32; 3];
    for c in 0..3 {
        // 3x3 tent, weights sum to 16.
        let tent = (samples[0][c] + samples[2][c] + samples[6][c] + samples[8][c])
            + 2.0 * (samples[1][c] + samples[3][c] + samples[5][c] + samples[7][c])
            + 4.0 * samples[4][c];
        let tent = tent / 16.0;
        // Blend centre -> full tent by radius (energy preserved: both sum to 1).
        out[c] = center[c] + (tent - center[c]) * r;
    }
    out
}

/// Blend the accumulated `bloom` over the `scene` by an artist `intensity`,
/// linearly interpolating per channel: `scene + (bloom - scene) * intensity`.
///
/// `intensity == 0` returns the untouched scene, `intensity == 1` returns pure
/// bloom; values in between cross-fade. `intensity` is not clamped so callers
/// may push past `1` for an over-bright look.
#[must_use]
pub fn combine(scene: [f32; 3], bloom: [f32; 3], intensity: f32) -> [f32; 3] {
    [
        scene[0] + (bloom[0] - scene[0]) * intensity,
        scene[1] + (bloom[1] - scene[1]) * intensity,
        scene[2] + (bloom[2] - scene[2]) * intensity,
    ]
}

/// Per-mip contribution weights for an `levels`-deep bloom pyramid, normalised
/// to sum to `1`. Uses a linear ramp so finer mips (lower index, tighter glow)
/// contribute more than coarse ones (wide halo): `weight[i] = (levels - i) / S`
/// with `S = levels * (levels + 1) / 2`. Returns an empty `Vec` for `0` levels.
#[must_use]
pub fn mip_blend_weights(levels: u32) -> Vec<f32> {
    if levels == 0 {
        return Vec::new();
    }
    let total = (levels * (levels + 1) / 2) as f32;
    (0..levels).map(|i| (levels - i) as f32 / total).collect()
}

/// Average of a 2x2 box of taps (helper for [`downsample_13tap`]).
#[inline]
fn box_avg(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> [f32; 3] {
    [
        (a[0] + b[0] + c[0] + d[0]) * 0.25,
        (a[1] + b[1] + c[1] + d[1]) * 0.25,
        (a[2] + b[2] + c[2] + d[2]) * 0.25,
    ]
}

/// Artist controls for the bloom pass. `Default` is a neutral, disabled bloom
/// (all zero): `intensity == 0` leaves the scene untouched.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct BloomParams {
    /// Luminance/brightness above which pixels start to bloom.
    pub threshold: f32,
    /// Width of the soft-threshold knee (`>= 0`; `0` is a hard threshold).
    pub knee: f32,
    /// Blend weight of the accumulated bloom over the scene in [`combine`].
    pub intensity: f32,
    /// Tent-upsample spread radius passed to [`upsample_tent`].
    pub radius: f32,
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
        approx(luminance([0.0, 1.0, 0.0]), 0.7152);
        approx(luminance([0.0, 0.0, 1.0]), 0.0722);
    }

    #[test]
    fn prefilter_below_threshold_is_zero() {
        // Hard knee: anything dimmer than the threshold contributes nothing.
        let out = prefilter([0.4, 0.3, 0.2], 1.0, 0.0);
        approx3(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn prefilter_zero_knee_is_hard_threshold() {
        // br = 2.0, threshold 1.0 -> contribution = (2 - 1) / 2 = 0.5.
        let out = prefilter([2.0, 1.0, 0.0], 1.0, 0.0);
        approx3(out, [1.0, 0.5, 0.0]);
    }

    #[test]
    fn prefilter_soft_knee_is_continuous_at_threshold() {
        // The mapping is continuous in brightness, so left/right limits agree.
        let t = 1.0;
        let k = 0.5;
        let eps = 1.0e-6;
        let lo = prefilter([t - eps, t - eps, t - eps], t, k);
        let hi = prefilter([t + eps, t + eps, t + eps], t, k);
        approx3(lo, hi);
    }

    #[test]
    fn prefilter_pure_white_passes_above_threshold() {
        // Well above the threshold the contribution approaches (br - t) / br.
        let out = prefilter([4.0, 4.0, 4.0], 1.0, 0.5);
        let expected = (4.0 - 1.0) / 4.0;
        approx3(out, [4.0 * expected, 4.0 * expected, 4.0 * expected]);
    }

    #[test]
    fn prefilter_preserves_hue() {
        // Contribution is a scalar, so the output keeps the input's ratios.
        let color = [3.0, 1.5, 0.75];
        let out = prefilter(color, 1.0, 0.25);
        approx(out[0] / out[1], color[0] / color[1]);
        approx(out[1] / out[2], color[1] / color[2]);
    }

    #[test]
    fn prefilter_negative_knee_is_safe() {
        // A negative knee must not invert the clamp bounds / panic.
        let out = prefilter([2.0, 2.0, 2.0], 1.0, -3.0);
        approx3(out, prefilter([2.0, 2.0, 2.0], 1.0, 0.0));
    }

    #[test]
    fn karis_weight_endpoints() {
        approx(karis_average_weight(0.0), 1.0);
        approx(karis_average_weight(1.0), 0.5);
        approx(karis_average_weight(3.0), 0.25);
    }

    #[test]
    fn karis_weight_is_monotonic_decreasing() {
        let mut prev = f32::INFINITY;
        for i in 0..64 {
            let w = karis_average_weight(i as f32 * 0.5);
            assert!(w < prev, "weight not decreasing at {i}");
            prev = w;
        }
    }

    #[test]
    fn downsample_13tap_conserves_energy() {
        // Flat white input must pass through unchanged (weights sum to 1).
        let samples = [[1.0, 1.0, 1.0]; 13];
        approx3(downsample_13tap(&samples), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn downsample_13tap_weights_sum_to_one() {
        // A flat coloured input is likewise preserved for every channel.
        let samples = [[0.3, 0.6, 0.9]; 13];
        approx3(downsample_13tap(&samples), [0.3, 0.6, 0.9]);
    }

    #[test]
    fn downsample_13tap_center_box_dominates() {
        // Only the inner box {3,4,8,9} lit -> 0.5 weight * average 1.0 = 0.5.
        let mut samples = [[0.0, 0.0, 0.0]; 13];
        for &i in &[3, 4, 8, 9] {
            samples[i] = [1.0, 1.0, 1.0];
        }
        approx3(downsample_13tap(&samples), [0.5, 0.5, 0.5]);
    }

    #[test]
    fn upsample_tent_conserves_energy_any_radius() {
        let samples = [[1.0, 1.0, 1.0]; 9];
        for &r in &[0.0, 0.25, 0.5, 1.0, 2.0] {
            approx3(upsample_tent(&samples, r), [1.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn upsample_tent_full_radius_is_1_16_kernel() {
        // Only the centre lit at radius 1 -> centre weight 4/16 = 0.25.
        let mut samples = [[0.0, 0.0, 0.0]; 9];
        samples[4] = [1.0, 1.0, 1.0];
        approx3(upsample_tent(&samples, 1.0), [0.25, 0.25, 0.25]);
        // Only a corner lit -> weight 1/16.
        let mut corner = [[0.0, 0.0, 0.0]; 9];
        corner[0] = [1.0, 1.0, 1.0];
        approx3(
            upsample_tent(&corner, 1.0),
            [1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0],
        );
    }

    #[test]
    fn upsample_tent_zero_radius_is_center() {
        let mut samples = [[0.0, 0.0, 0.0]; 9];
        samples[4] = [0.7, 0.2, 0.9];
        approx3(upsample_tent(&samples, 0.0), [0.7, 0.2, 0.9]);
    }

    #[test]
    fn combine_intensity_zero_returns_scene() {
        let scene = [0.2, 0.4, 0.6];
        let bloom = [1.0, 1.0, 1.0];
        approx3(combine(scene, bloom, 0.0), scene);
    }

    #[test]
    fn combine_intensity_one_returns_bloom() {
        let scene = [0.2, 0.4, 0.6];
        let bloom = [1.0, 0.5, 0.25];
        approx3(combine(scene, bloom, 1.0), bloom);
    }

    #[test]
    fn combine_midpoint_is_average() {
        let scene = [0.0, 0.0, 0.0];
        let bloom = [1.0, 0.5, 0.25];
        approx3(combine(scene, bloom, 0.5), [0.5, 0.25, 0.125]);
    }

    #[test]
    fn mip_blend_weights_sum_to_one() {
        for levels in 1..=8 {
            let w = mip_blend_weights(levels);
            assert_eq!(w.len(), levels as usize);
            approx(w.iter().sum::<f32>(), 1.0);
        }
    }

    #[test]
    fn mip_blend_weights_monotonic_decreasing() {
        let w = mip_blend_weights(5);
        for pair in w.windows(2) {
            assert!(pair[0] > pair[1], "weights not decreasing: {w:?}");
        }
    }

    #[test]
    fn mip_blend_weights_empty_for_zero_levels() {
        assert!(mip_blend_weights(0).is_empty());
    }

    #[test]
    fn bloom_params_default_is_disabled() {
        let p = BloomParams::default();
        approx(p.threshold, 0.0);
        approx(p.knee, 0.0);
        approx(p.intensity, 0.0);
        approx(p.radius, 0.0);
        // Disabled intensity leaves the scene untouched regardless of bloom.
        approx3(
            combine([0.3, 0.3, 0.3], [9.0, 9.0, 9.0], p.intensity),
            [0.3, 0.3, 0.3],
        );
    }
}
