//! Backend-neutral CPU golden for Contrast-Adaptive Sharpening (`CAS`).
//!
//! `CAS` is AMD `FidelityFX` Contrast-Adaptive Sharpening, the sharpening pass
//! every modern AAA pipeline reaches for because it is *adaptive*: it sharpens
//! low-contrast regions strongly and high-contrast edges weakly, so it lifts
//! perceived detail without the ringing haloes a fixed unsharp-mask leaves. It
//! is a shared post-processing base, not a peer of the PBR/NPR shading fronts:
//! it runs on the display-referred (tone-mapped, roughly `[0, 1]`) buffer that
//! every illumination model writes into, so one sharpen pass serves them all. A
//! stylized front can crank it for a crisp illustrative read while a
//! photographic front keeps it gentle.
//!
//! The kernel is the `FidelityFX` "no-scaling" path: a 3x3 neighbourhood
//!
//! ```text
//! a b c
//! d e f
//! g h i
//! ```
//!
//! with centre `e`. Per channel it forms a *soft* min and max (the min/max of
//! the five-tap cross plus the min/max of the four corners, i.e. the
//! `FidelityFX` doubled soft-limit), derives an adaptive amplitude
//! `sqrt(saturate(min(soft_min, 2 - soft_max) / soft_max))`, shapes it by a
//! sharpness knob into a negative cross weight `w`, and blends the cross taps
//! against the centre as `saturate(((b + d + f + h) * w + e) / (1 + 4 * w))`.
//! A flat neighbourhood is a fixed point for every sharpness, so the pass is a
//! true identity where there is no detail to lift.
//!
//! Everything is plain `f32` (only `sqrt`, which the determinism lint allows),
//! so the module needs no `bevy_math::ops` import and stays bit-reproducible
//! against `shaders/cas.wesl`, which mirrors it arm-for-arm.

/// Minimum of three values.
#[must_use]
pub fn min3(a: f32, b: f32, c: f32) -> f32 {
    a.min(b).min(c)
}

/// Maximum of three values.
#[must_use]
pub fn max3(a: f32, b: f32, c: f32) -> f32 {
    a.max(b).max(c)
}

/// A 3x3 neighbourhood in row-major order (`a b c / d e f / g h i`), each tap
/// an RGB triple. `neigh[4]` is the centre `e`.
pub type Neighborhood = [[f32; 3]; 9];

/// `FidelityFX` doubled soft minimum of one channel over the neighbourhood:
/// the min of the five-tap cross (`b, d, e, f, h`) plus the min of the four
/// corners (`a, c, g, i`). The doubling keeps the `2 - soft_max` limit maths
/// consistent for a display-referred `[0, 1]` signal.
#[must_use]
pub fn soft_min_channel(n: &Neighborhood, ch: usize) -> f32 {
    let (a, b, c) = (n[0][ch], n[1][ch], n[2][ch]);
    let (d, e, f) = (n[3][ch], n[4][ch], n[5][ch]);
    let (g, h, i) = (n[6][ch], n[7][ch], n[8][ch]);
    let cross = min3(min3(d, e, f), b, h);
    let corners = min3(min3(a, c, g), i, i);
    cross + corners
}

/// `FidelityFX` doubled soft maximum of one channel (cross max plus corner max).
#[must_use]
pub fn soft_max_channel(n: &Neighborhood, ch: usize) -> f32 {
    let (a, b, c) = (n[0][ch], n[1][ch], n[2][ch]);
    let (d, e, f) = (n[3][ch], n[4][ch], n[5][ch]);
    let (g, h, i) = (n[6][ch], n[7][ch], n[8][ch]);
    let cross = max3(max3(d, e, f), b, h);
    let corners = max3(max3(a, c, g), i, i);
    cross + corners
}

/// Adaptive sharpening amplitude for one channel from its doubled soft limits:
/// `sqrt(saturate(min(soft_min, 2 - soft_max) / soft_max))`.
///
/// Near a hard edge `soft_max` is large so the amplitude (and thus the applied
/// sharpening) shrinks; in a smooth low-contrast region it approaches `1`. The
/// division guards a zero `soft_max` (pure black) by returning `0`.
#[must_use]
pub fn cas_amplitude(soft_min: f32, soft_max: f32) -> f32 {
    if soft_max <= 0.0 {
        return 0.0;
    }
    let limited = soft_min.min(2.0 - soft_max);
    let amp = (limited / soft_max).clamp(0.0, 1.0);
    amp.sqrt()
}

/// Shapes an amplitude into the negative cross weight `w` used by the blend.
///
/// `sharpness` in `[0, 1]` maps the peak from `-1/8` (gentle) to `-1/5`
/// (strong) via `peak = -1 / (8 + (5 - 8) * sharpness)`; `w = amplitude * peak`.
#[must_use]
pub fn cas_weight(amplitude: f32, sharpness: f32) -> f32 {
    let s = sharpness.clamp(0.0, 1.0);
    let denom = 8.0 + (5.0 - 8.0) * s;
    let peak = -1.0 / denom;
    amplitude * peak
}

/// Blends one channel's cross taps against the centre with weight `w`:
/// `saturate((b + d + f + h) * w + e) / (1 + 4 * w)`.
#[must_use]
pub fn cas_blend_channel(b: f32, d: f32, e: f32, f: f32, h: f32, w: f32) -> f32 {
    let rcp = 1.0 / (1.0 + 4.0 * w);
    (((b + d + f + h) * w + e) * rcp).clamp(0.0, 1.0)
}

/// Full `CAS` kernel over a 3x3 neighbourhood at the given `sharpness`.
///
/// Runs [`soft_min_channel`]/[`soft_max_channel`] -> [`cas_amplitude`] ->
/// [`cas_weight`] -> [`cas_blend_channel`] per channel. A flat neighbourhood
/// returns the centre unchanged for every sharpness.
#[must_use]
pub fn cas_sharpen(n: &Neighborhood, sharpness: f32) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for (ch, slot) in out.iter_mut().enumerate() {
        let smn = soft_min_channel(n, ch);
        let smx = soft_max_channel(n, ch);
        let amp = cas_amplitude(smn, smx);
        let w = cas_weight(amp, sharpness);
        *slot = cas_blend_channel(n[1][ch], n[3][ch], n[4][ch], n[5][ch], n[7][ch], w);
    }
    out
}

/// `CAS` parameters. `strength` in `[0, 1]` blends the sharpened result against
/// the untouched centre (`0` is a bit-exact identity, the disabled default);
/// `sharpness` in `[0, 1]` selects the adaptive peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CasParams {
    /// Blend of the sharpened result over the original centre, `[0, 1]`.
    pub strength: f32,
    /// Adaptive sharpening peak selector, `[0, 1]`.
    pub sharpness: f32,
}

impl Default for CasParams {
    fn default() -> Self {
        // Disabled: strength 0 returns the centre untouched.
        Self {
            strength: 0.0,
            sharpness: 0.0,
        }
    }
}

/// Applies [`cas_sharpen`] and blends it over the centre by `params.strength`.
#[must_use]
pub fn apply_cas(n: &Neighborhood, params: CasParams) -> [f32; 3] {
    let e = n[4];
    let s = params.strength.clamp(0.0, 1.0);
    if s <= 0.0 {
        return e;
    }
    let sharp = cas_sharpen(n, params.sharpness);
    [
        e[0] + (sharp[0] - e[0]) * s,
        e[1] + (sharp[1] - e[1]) * s,
        e[2] + (sharp[2] - e[2]) * s,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(v: f32) -> Neighborhood {
        [[v, v, v]; 9]
    }

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) {
        for k in 0..3 {
            approx(a[k], b[k]);
        }
    }

    #[test]
    fn min3_max3_are_correct() {
        approx(min3(0.3, 0.1, 0.2), 0.1);
        approx(max3(0.3, 0.1, 0.2), 0.3);
    }

    #[test]
    fn soft_limits_double_on_flat_region() {
        let n = flat(0.5);
        approx(soft_min_channel(&n, 0), 1.0);
        approx(soft_max_channel(&n, 0), 1.0);
    }

    #[test]
    fn soft_min_picks_the_darkest_tap() {
        let mut n = flat(0.6);
        n[0][0] = 0.1; // corner a
        // cross min stays 0.6, corner min is 0.1 -> 0.7
        approx(soft_min_channel(&n, 0), 0.7);
    }

    #[test]
    fn soft_max_picks_the_brightest_tap() {
        let mut n = flat(0.4);
        n[5][0] = 0.9; // cross tap f
        // cross max 0.9, corner max 0.4 -> 1.3
        approx(soft_max_channel(&n, 0), 1.3);
    }

    #[test]
    fn amplitude_is_one_at_mid_grey_flat() {
        // soft_min = soft_max = 1 -> min(1, 2-1)/1 = 1 -> sqrt = 1
        approx(cas_amplitude(1.0, 1.0), 1.0);
    }

    #[test]
    fn amplitude_shrinks_near_a_hard_edge() {
        // High soft_max (bright edge) reduces the limit and thus the amplitude.
        let edge = cas_amplitude(0.2, 1.9);
        let smooth = cas_amplitude(0.9, 1.1);
        assert!(edge < smooth, "edge {edge} should sharpen less than smooth {smooth}");
    }

    #[test]
    fn amplitude_is_zero_for_black() {
        approx(cas_amplitude(0.0, 0.0), 0.0);
    }

    #[test]
    fn amplitude_is_clamped_to_unit_range() {
        let a = cas_amplitude(5.0, 1.0);
        assert!((0.0..=1.0).contains(&a), "amplitude {a} out of range");
    }

    #[test]
    fn weight_is_negative_and_stronger_with_sharpness() {
        let gentle = cas_weight(1.0, 0.0);
        let strong = cas_weight(1.0, 1.0);
        assert!(gentle < 0.0 && strong < 0.0, "weights must be negative");
        assert!(strong < gentle, "sharper knob should give a more negative weight");
        approx(gentle, -1.0 / 8.0);
        approx(strong, -1.0 / 5.0);
    }

    #[test]
    fn weight_scales_with_amplitude() {
        let full = cas_weight(1.0, 0.5);
        let half = cas_weight(0.5, 0.5);
        approx(half, full * 0.5);
    }

    #[test]
    fn blend_is_identity_on_flat_channel() {
        // b=d=e=f=h=0.5, w=-1/8 -> (4*0.5*w+0.5)/(1+4w) = 0.5
        approx(cas_blend_channel(0.5, 0.5, 0.5, 0.5, 0.5, -1.0 / 8.0), 0.5);
    }

    #[test]
    fn blend_lifts_a_bright_centre() {
        // Centre brighter than neighbours: sharpening should raise it further.
        let out = cas_blend_channel(0.4, 0.4, 0.6, 0.4, 0.4, -1.0 / 8.0);
        assert!(out > 0.6, "bright centre should be lifted, got {out}");
    }

    #[test]
    fn blend_darkens_a_dim_centre() {
        let out = cas_blend_channel(0.6, 0.6, 0.4, 0.6, 0.6, -1.0 / 8.0);
        assert!(out < 0.4, "dim centre should be pushed darker, got {out}");
    }

    #[test]
    fn blend_stays_in_unit_range() {
        let out = cas_blend_channel(0.0, 0.0, 1.0, 0.0, 0.0, -1.0 / 5.0);
        assert!((0.0..=1.0).contains(&out), "blend {out} out of range");
    }

    #[test]
    fn sharpen_is_identity_on_flat_neighbourhood() {
        let n = flat(0.5);
        approx3(cas_sharpen(&n, 0.0), [0.5, 0.5, 0.5]);
        approx3(cas_sharpen(&n, 1.0), [0.5, 0.5, 0.5]);
    }

    #[test]
    fn sharpen_enhances_local_contrast() {
        // Bright centre on a darker cross -> brighter after sharpening.
        let mut n = flat(0.4);
        n[4] = [0.6, 0.6, 0.6];
        let out = cas_sharpen(&n, 0.5);
        assert!(out[0] > 0.6, "centre should be lifted, got {:?}", out);
    }

    #[test]
    fn sharpen_is_per_channel() {
        let mut n = flat(0.5);
        n[4] = [0.7, 0.5, 0.3];
        let out = cas_sharpen(&n, 0.5);
        assert!(out[0] > 0.7, "red centre lifted");
        approx(out[1], 0.5); // green flat -> identity
        assert!(out[2] < 0.3, "blue centre pushed down");
    }

    #[test]
    fn stronger_sharpness_enhances_more() {
        let mut n = flat(0.4);
        n[4] = [0.6, 0.6, 0.6];
        let gentle = cas_sharpen(&n, 0.0)[0];
        let strong = cas_sharpen(&n, 1.0)[0];
        assert!(strong > gentle, "strong {strong} should exceed gentle {gentle}");
    }

    #[test]
    fn params_default_is_disabled_identity() {
        let mut n = flat(0.3);
        n[4] = [0.8, 0.2, 0.5];
        approx3(apply_cas(&n, CasParams::default()), [0.8, 0.2, 0.5]);
    }

    #[test]
    fn apply_blends_by_strength() {
        let mut n = flat(0.4);
        n[4] = [0.6, 0.6, 0.6];
        let full = cas_sharpen(&n, 0.5);
        let half = apply_cas(
            &n,
            CasParams {
                strength: 0.5,
                sharpness: 0.5,
            },
        );
        // Half strength sits between the centre and the full sharpen.
        for k in 0..3 {
            approx(half[k], 0.6 + (full[k] - 0.6) * 0.5);
        }
    }

    #[test]
    fn apply_full_strength_equals_sharpen() {
        let mut n = flat(0.4);
        n[4] = [0.65, 0.55, 0.45];
        let full = apply_cas(
            &n,
            CasParams {
                strength: 1.0,
                sharpness: 0.7,
            },
        );
        approx3(full, cas_sharpen(&n, 0.7));
    }
}
