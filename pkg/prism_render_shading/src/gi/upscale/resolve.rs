//! Reconstruction and sharpening for temporal super-resolution.
//!
//! The final resolve stage turns the splatted/accumulated high-resolution grid
//! (see [`super::accumulate`]) into a crisp output image.  It performs three
//! classical steps:
//!
//! 1. **Checkerboard reconstruction** — in checkerboard rendering only half the
//!    pixels are shaded each frame (a quincunx/checker lattice).  Missing pixels
//!    are filled by an *edge-aware* weighted interpolation of their four axial
//!    neighbours, so the fill follows edges instead of blurring across them.
//! 2. **Temporal / spatial blend** — where temporal history is confident the
//!    resolve trusts the accumulated value; where it is thin (recent
//!    disocclusion) it falls back to the spatially reconstructed estimate.
//! 3. **Sharpening** — a Catmull-Rom-style negative-lobe / unsharp-mask filter
//!    restores the high-frequency detail that bilinear splatting softens.  The
//!    sharpened value is clamped to the local neighbourhood colour box so the
//!    filter can never *overshoot* into ringing, and the operator is energy
//!    aware (it adds and subtracts equal weight around the centre).
//!
//! All of it is deterministic signal processing — Catmull-Rom cubics, weighted
//! means and clamps — with no learned components whatsoever.
//!
//! # Conventions
//! * Colours are linear-RGB [`Vec3`], kept finite and non-negative.
//! * Checkerboard parity: a pixel is *rendered* this frame when
//!   `(x + y + frame_parity)` is even; the opposite phase is reconstructed.
//! * Sharpening clamps to the neighbourhood `[min, max]` box (no overshoot) and
//!   uses a signed amount in `[0, ~1]`.
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the value
//!   method.  Every function is a deterministic pure function — no RNG, I/O,
//!   GPU, or `unsafe` — and never produces `NaN`.

use bevy_math::Vec3;

/// Small epsilon guarding weight normalisation against divide-by-zero.
const EPSILON: f32 = 1.0e-6;

/// Rec.709 luminance weights, used for edge-aware neighbour weighting.
const LUMA_709: Vec3 = Vec3::new(0.212_639, 0.715_169, 0.072_192);

/// Returns `true` when pixel `(x, y)` is natively *rendered* this frame under
/// checkerboard rendering with the given `frame_parity` (`0` or `1`).
///
/// Rendered pixels have `(x + y + frame_parity)` even; the complementary set is
/// reconstructed by [`reconstruct_checkerboard`].
#[inline]
pub fn is_rendered(x: u32, y: u32, frame_parity: u32) -> bool {
    (x.wrapping_add(y).wrapping_add(frame_parity) & 1) == 0
}

/// Clamps a colour to be component-wise finite and non-negative.
#[inline]
pub fn sanitize(color: Vec3) -> Vec3 {
    Vec3::new(
        if color.x.is_finite() { color.x.max(0.0) } else { 0.0 },
        if color.y.is_finite() { color.y.max(0.0) } else { 0.0 },
        if color.z.is_finite() { color.z.max(0.0) } else { 0.0 },
    )
}

/// Rec.709 luminance of a linear-RGB colour, clamped non-negative.
#[inline]
pub fn luma(color: Vec3) -> f32 {
    sanitize(color).dot(LUMA_709).max(0.0)
}

/// An axial neighbour sample for checkerboard reconstruction: its `color` and a
/// `valid` flag (false for off-screen or still-missing neighbours).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    /// Linear-RGB colour of the neighbour.
    pub color: Vec3,
    /// Whether this neighbour carries a usable sample.
    pub valid: bool,
}

impl Neighbor {
    /// A usable neighbour with the given colour.
    #[inline]
    pub fn present(color: Vec3) -> Self {
        Self { color: sanitize(color), valid: true }
    }

    /// A missing / off-screen neighbour (ignored by the reconstruction).
    #[inline]
    pub fn missing() -> Self {
        Self { color: Vec3::ZERO, valid: false }
    }
}

/// Reconstructs a missing checkerboard pixel from its four axial neighbours
/// (`left`, `right`, `up`, `down`) using edge-aware weighted interpolation.
///
/// Opposite pairs that agree in luminance (a smooth region) contribute evenly;
/// a pair that straddles an edge (large luma gap) is down-weighted so the fill
/// follows the edge rather than averaging across it.  Invalid neighbours are
/// skipped.  If no neighbour is valid the result is `Vec3::ZERO`.
pub fn reconstruct_checkerboard(
    left: Neighbor,
    right: Neighbor,
    up: Neighbor,
    down: Neighbor,
) -> Vec3 {
    // Horizontal and vertical pair confidences: a smaller intra-pair luma gap
    // means a smoother direction, which earns a higher interpolation weight.
    let h_weight = pair_confidence(left, right);
    let v_weight = pair_confidence(up, down);

    let mut acc = Vec3::ZERO;
    let mut wsum = 0.0f32;
    accumulate_neighbor(&mut acc, &mut wsum, left, h_weight);
    accumulate_neighbor(&mut acc, &mut wsum, right, h_weight);
    accumulate_neighbor(&mut acc, &mut wsum, up, v_weight);
    accumulate_neighbor(&mut acc, &mut wsum, down, v_weight);

    if wsum > EPSILON {
        sanitize(acc / wsum)
    } else {
        // Fall back to a plain average of whatever is valid.
        unweighted_mean(&[left, right, up, down])
    }
}

/// Blends the temporally accumulated colour with the spatially reconstructed one
/// according to `temporal_confidence ∈ [0, 1]`.
///
/// Confidence `1` returns the temporal value (mature, stable history); `0`
/// returns the spatial estimate (fresh disocclusion).  Inputs are sanitized and
/// confidence is clamped.
#[inline]
pub fn blend_temporal_spatial(temporal: Vec3, spatial: Vec3, temporal_confidence: f32) -> Vec3 {
    let c = finite_or(temporal_confidence, 1.0).clamp(0.0, 1.0);
    let t = sanitize(temporal);
    let s = sanitize(spatial);
    sanitize(s + (t - s) * c)
}

/// Catmull-Rom basis weights for the four samples straddling fractional
/// position `t ∈ [0, 1]` (samples at relative offsets `-1, 0, 1, 2`).
///
/// Returns `[w_{-1}, w_0, w_1, w_2]`; the weights sum to `1`.  The spline is the
/// standard tension-`0.5` Catmull-Rom, which interpolates the control points and
/// introduces the mild negative lobe responsible for its edge-preserving
/// sharpness.  `t` is clamped to `[0, 1]`.
#[inline]
pub fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t = finite_or(t, 0.0).clamp(0.0, 1.0);
    let t2 = t * t;
    let t3 = t2 * t;
    // Catmull-Rom (a = -0.5) basis.
    let w0 = -0.5 * t3 + t2 - 0.5 * t;
    let w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
    let w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
    let w3 = 0.5 * t3 - 0.5 * t2;
    [w0, w1, w2, w3]
}

/// Evaluates the Catmull-Rom cubic through four scalar `samples` at fractional
/// position `t ∈ [0, 1]`.
///
/// `samples` are at relative positions `-1, 0, 1, 2`; the spline interpolates
/// `samples[1]` at `t = 0` and `samples[2]` at `t = 1`.
#[inline]
pub fn catmull_rom_1d(samples: [f32; 4], t: f32) -> f32 {
    let w = catmull_rom_weights(t);
    let v = w[0] * samples[0] + w[1] * samples[1] + w[2] * samples[2] + w[3] * samples[3];
    finite_or(v, samples[1])
}

/// Per-channel Catmull-Rom interpolation of four colour `samples` at `t`.
#[inline]
pub fn catmull_rom_rgb(samples: [Vec3; 4], t: f32) -> Vec3 {
    let w = catmull_rom_weights(t);
    let v = samples[0] * w[0] + samples[1] * w[1] + samples[2] * w[2] + samples[3] * w[3];
    sanitize(v)
}

/// Sharpens `center` against a low-pass `blur` estimate with an unsharp mask,
/// then clamps the result into the neighbourhood box `[nbr_min, nbr_max]` to
/// forbid overshoot (ringing).
///
/// `amount >= 0` is the sharpening strength: `out = center + amount·(center -
/// blur)`.  The unsharp operator is energy aware — it adds and subtracts the
/// same high-frequency residual around the centre — and the final clamp keeps
/// the output within the colours already present locally, so energy cannot run
/// away.  `amount` of `0` returns `center` unchanged.
#[inline]
pub fn unsharp_mask(center: Vec3, blur: Vec3, amount: f32, nbr_min: Vec3, nbr_max: Vec3) -> Vec3 {
    let a = finite_or(amount, 0.0).max(0.0);
    let c = sanitize(center);
    let b = sanitize(blur);
    let sharp = c + (c - b) * a;
    let lo = sanitize(nbr_min);
    let hi = sanitize(nbr_max);
    Vec3::new(
        clamp_ordered(sharp.x, lo.x, hi.x),
        clamp_ordered(sharp.y, lo.y, hi.y),
        clamp_ordered(sharp.z, lo.z, hi.z),
    )
}

/// Computes the component-wise neighbourhood colour box `[min, max]` over a slice
/// of colours (typically the 3x3 window around a pixel).
///
/// Empty input yields a zero box.  Non-finite components are sanitized before
/// the min/max so the box is always valid.
pub fn neighborhood_box(colors: &[Vec3]) -> (Vec3, Vec3) {
    if colors.is_empty() {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &c in colors {
        let s = sanitize(c);
        lo = lo.min(s);
        hi = hi.max(s);
    }
    (sanitize(lo), sanitize(hi))
}

/// Convenience sharpen: builds the neighbourhood box from `window`, uses its
/// mean as the low-pass `blur`, and applies [`unsharp_mask`] to `center`.
///
/// This is the one-call resolve-sharpen used by the pass: it both derives the
/// anti-overshoot clamp box and the blur reference from the same local window,
/// guaranteeing the sharpened pixel stays within locally observed colours.
pub fn sharpen_in_neighborhood(center: Vec3, window: &[Vec3], amount: f32) -> Vec3 {
    if window.is_empty() {
        return sanitize(center);
    }
    let (lo, hi) = neighborhood_box(window);
    let mut mean = Vec3::ZERO;
    for &c in window {
        mean += sanitize(c);
    }
    mean /= window.len() as f32;
    unsharp_mask(center, mean, amount, lo, hi)
}

/// Edge-aware confidence for an opposite neighbour pair: higher when the pair
/// agrees in luminance (smooth), lower across an edge.
#[inline]
fn pair_confidence(a: Neighbor, b: Neighbor) -> f32 {
    if !a.valid || !b.valid {
        // A half-present pair carries a small baseline weight so its single
        // valid member can still contribute if nothing better exists.
        return if a.valid || b.valid { 0.25 } else { 0.0 };
    }
    let la = luma(a.color);
    let lb = luma(b.color);
    let denom = la.max(lb) + EPSILON;
    let gap = (la - lb).abs() / denom;
    // Smooth falloff: identical luma -> 1, large gap -> toward 0.
    (1.0 / (1.0 + 4.0 * gap)).clamp(0.0, 1.0)
}

/// Adds a valid neighbour's colour to the weighted accumulator.
#[inline]
fn accumulate_neighbor(acc: &mut Vec3, wsum: &mut f32, n: Neighbor, weight: f32) {
    if n.valid && weight > 0.0 {
        *acc += sanitize(n.color) * weight;
        *wsum += weight;
    }
}

/// Plain average of the valid neighbours (fallback when all pair weights are 0).
#[inline]
fn unweighted_mean(neighbors: &[Neighbor]) -> Vec3 {
    let mut acc = Vec3::ZERO;
    let mut count = 0.0f32;
    for n in neighbors {
        if n.valid {
            acc += sanitize(n.color);
            count += 1.0;
        }
    }
    if count > 0.0 {
        sanitize(acc / count)
    } else {
        Vec3::ZERO
    }
}

/// Returns `v` when finite, else the `fallback`.
#[inline]
fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() { v } else { fallback }
}

/// Clamps `v` to `[a, b]`, swapping inverted bounds.
#[inline]
fn clamp_ordered(v: f32, a: f32, b: f32) -> f32 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    v.clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkerboard_parity_alternates() {
        assert!(is_rendered(0, 0, 0));
        assert!(!is_rendered(1, 0, 0));
        assert!(!is_rendered(0, 1, 0));
        assert!(is_rendered(1, 1, 0));
        // Flipping the frame parity flips the whole lattice.
        assert!(!is_rendered(0, 0, 1));
        assert!(is_rendered(1, 0, 1));
    }

    #[test]
    fn reconstruct_constant_signal_is_exact() {
        // A constant field must reconstruct to the exact constant everywhere.
        let c = Vec3::new(0.3, 0.5, 0.7);
        let out = reconstruct_checkerboard(
            Neighbor::present(c),
            Neighbor::present(c),
            Neighbor::present(c),
            Neighbor::present(c),
        );
        assert!((out - c).length() < 1e-6, "got {out:?}");
    }

    #[test]
    fn reconstruct_follows_smooth_direction() {
        // Vertical pair agrees (smooth), horizontal straddles a bright edge.
        // The fill should lean toward the smooth vertical pair's value.
        let dark = Vec3::splat(0.1);
        let bright = Vec3::splat(0.9);
        let smooth = Vec3::splat(0.5);
        let out = reconstruct_checkerboard(
            Neighbor::present(dark),
            Neighbor::present(bright),
            Neighbor::present(smooth),
            Neighbor::present(smooth),
        );
        assert!((out - smooth).length() < 0.2, "edge bled through: {out:?}");
    }

    #[test]
    fn reconstruct_ignores_missing_neighbors() {
        let c = Vec3::new(0.2, 0.4, 0.6);
        let out = reconstruct_checkerboard(
            Neighbor::present(c),
            Neighbor::missing(),
            Neighbor::missing(),
            Neighbor::missing(),
        );
        assert!((out - c).length() < 1e-6, "got {out:?}");
    }

    #[test]
    fn reconstruct_all_missing_is_zero() {
        let out = reconstruct_checkerboard(
            Neighbor::missing(),
            Neighbor::missing(),
            Neighbor::missing(),
            Neighbor::missing(),
        );
        assert_eq!(out, Vec3::ZERO);
    }

    #[test]
    fn temporal_spatial_blend_endpoints() {
        let t = Vec3::splat(0.8);
        let s = Vec3::splat(0.2);
        assert!((blend_temporal_spatial(t, s, 1.0) - t).length() < 1e-6);
        assert!((blend_temporal_spatial(t, s, 0.0) - s).length() < 1e-6);
        let mid = blend_temporal_spatial(t, s, 0.5);
        assert!((mid - Vec3::splat(0.5)).length() < 1e-6, "{mid:?}");
    }

    #[test]
    fn catmull_rom_weights_sum_to_one() {
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let w = catmull_rom_weights(t);
            let s: f32 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "t {t} sum {s}");
        }
    }

    #[test]
    fn catmull_rom_interpolates_control_points() {
        let s = [1.0, 2.0, 5.0, 9.0];
        assert!((catmull_rom_1d(s, 0.0) - 2.0).abs() < 1e-6);
        assert!((catmull_rom_1d(s, 1.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn catmull_rom_reproduces_linear_ramp() {
        // On a linear ramp the cubic must stay linear (no overshoot).
        let s = [0.0, 1.0, 2.0, 3.0];
        for i in 0..=8 {
            let t = i as f32 / 8.0;
            let want = 1.0 + t;
            assert!((catmull_rom_1d(s, t) - want).abs() < 1e-5, "t {t}");
        }
    }

    #[test]
    fn catmull_rom_rgb_matches_scalar() {
        let s = [Vec3::splat(1.0), Vec3::splat(2.0), Vec3::splat(5.0), Vec3::splat(9.0)];
        let v = catmull_rom_rgb(s, 0.5);
        let scalar = catmull_rom_1d([1.0, 2.0, 5.0, 9.0], 0.5);
        assert!((v.x - scalar).abs() < 1e-6, "{v:?} vs {scalar}");
    }

    #[test]
    fn unsharp_amount_zero_is_identity() {
        let c = Vec3::new(0.4, 0.5, 0.6);
        let out = unsharp_mask(c, Vec3::splat(0.2), 0.0, Vec3::ZERO, Vec3::ONE);
        assert!((out - c).length() < 1e-6, "{out:?}");
    }

    #[test]
    fn unsharp_increases_contrast_against_blur() {
        // Center brighter than blur -> sharpened center should brighten further
        // (up to the clamp box).
        let c = Vec3::splat(0.6);
        let blur = Vec3::splat(0.5);
        let out = unsharp_mask(c, blur, 0.5, Vec3::ZERO, Vec3::ONE);
        assert!(out.x > c.x, "did not sharpen: {out:?}");
    }

    #[test]
    fn unsharp_clamps_overshoot() {
        // A huge amount must not push past the neighbourhood max (no ringing).
        let c = Vec3::splat(0.9);
        let blur = Vec3::splat(0.1);
        let out = unsharp_mask(c, blur, 10.0, Vec3::ZERO, Vec3::splat(0.95));
        assert!(out.x <= 0.95 + 1e-6, "overshoot: {out:?}");
    }

    #[test]
    fn unsharp_never_nan_or_negative() {
        let out = unsharp_mask(
            Vec3::new(f32::NAN, 0.5, 0.5),
            Vec3::new(0.2, f32::INFINITY, 0.1),
            2.0,
            Vec3::ZERO,
            Vec3::ONE,
        );
        assert!(out.x.is_finite() && out.y.is_finite() && out.z.is_finite());
        assert!(out.x >= 0.0 && out.y >= 0.0 && out.z >= 0.0, "{out:?}");
    }

    #[test]
    fn neighborhood_box_brackets_members() {
        let w = [Vec3::splat(0.2), Vec3::new(0.9, 0.1, 0.5), Vec3::splat(0.4)];
        let (lo, hi) = neighborhood_box(&w);
        assert!(lo.x <= 0.2 && hi.x >= 0.9, "lo {lo:?} hi {hi:?}");
        assert!((lo.y - 0.1).abs() < 1e-6);
    }

    #[test]
    fn sharpen_in_neighborhood_stays_in_box() {
        let window = [
            Vec3::splat(0.2),
            Vec3::splat(0.4),
            Vec3::splat(0.6),
            Vec3::splat(0.5),
        ];
        let out = sharpen_in_neighborhood(Vec3::splat(0.6), &window, 5.0);
        let (lo, hi) = neighborhood_box(&window);
        assert!(out.x >= lo.x - 1e-6 && out.x <= hi.x + 1e-6, "{out:?}");
    }

    #[test]
    fn sharpen_empty_window_is_identity() {
        let c = Vec3::new(0.3, 0.3, 0.3);
        assert_eq!(sharpen_in_neighborhood(c, &[], 2.0), c);
    }

    #[test]
    fn is_deterministic() {
        let a = Neighbor::present(Vec3::splat(0.3));
        let b = Neighbor::present(Vec3::splat(0.7));
        assert_eq!(
            reconstruct_checkerboard(a, b, a, b),
            reconstruct_checkerboard(a, b, a, b),
        );
    }
}
