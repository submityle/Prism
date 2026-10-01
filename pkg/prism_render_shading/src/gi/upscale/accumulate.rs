//! Motion-compensated history accumulation for temporal super-resolution.
//!
//! Once per frame the renderer rasterizes a *jittered* low-resolution image (see
//! [`super::jitter`]).  To recover a stable high-resolution result we re-project
//! the previous frame's accumulated history along per-pixel motion vectors and
//! blend the freshly rasterized low-res samples into it.  Because each low-res
//! sample lands at a sub-pixel position inside the high-res grid, it is
//! *splatted* with bilinear weights across the surrounding high-res texels — a
//! scatter-style resolve that fills the grid as the jitter sweeps the pixel.
//!
//! The accumulation is driven entirely by classical confidence heuristics:
//!
//! * a running **sample count** that lowers the new-sample blend weight as
//!   history matures (`α ≈ 1 / n`), bounded by a floor so moving content keeps
//!   adapting;
//! * a per-pixel **confidence** in `[0, 1]` from motion-vector agreement and
//!   depth/normal consistency that attenuates history on uncertain reprojection;
//! * a hard **disocclusion reset** that throws history away where reprojection
//!   is invalid (newly revealed geometry);
//! * a **luma-stability** neighbourhood clamp (variance clipping) that reins in
//!   history toward the current frame's local colour box, killing ghosting.
//!
//! Nothing here is learned: it is bilinear splatting plus exponential moving
//! averages plus an AABB colour clamp.
//!
//! # Conventions
//! * Colours are linear-RGB [`Vec3`]; they are kept non-negative and finite.
//! * `sample_count` is a float so the `1/n` blend decays smoothly; it is clamped
//!   to `[1, max_samples]`.
//! * Blend weights (`α`) are the weight of the *incoming* sample, in `[0, 1]`.
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the value
//!   method.  Every function is a deterministic pure function — no RNG, I/O,
//!   GPU, or `unsafe` — and never produces `NaN`.

use alloc::vec::Vec;
use bevy_math::{Vec2, Vec3};

/// Smallest accumulated weight treated as a filled high-res texel.
const WEIGHT_EPSILON: f32 = 1.0e-6;

/// Rec.709 luminance weights for linear-RGB luma stability.
const LUMA_709: Vec3 = Vec3::new(0.212_639, 0.715_169, 0.072_192);

/// Luminance (Rec.709) of a linear-RGB colour, clamped non-negative.
#[inline]
pub fn luma(color: Vec3) -> f32 {
    sanitize(color).dot(LUMA_709).max(0.0)
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

/// Bilinear splat weights for a sample whose sub-texel position within a cell is
/// `frac ∈ [0, 1)^2`, ordered `[w00, w10, w01, w11]`.
///
/// `w00` is the lower-left texel `(0, 0)`, `w10` the right neighbour, `w01` the
/// upper neighbour and `w11` the diagonal; the four weights sum to `1`.  `frac`
/// is clamped to `[0, 1]` so out-of-range inputs degrade gracefully.
#[inline]
pub fn bilinear_splat_weights(frac: Vec2) -> [f32; 4] {
    let fx = if frac.x.is_finite() { frac.x.clamp(0.0, 1.0) } else { 0.0 };
    let fy = if frac.y.is_finite() { frac.y.clamp(0.0, 1.0) } else { 0.0 };
    let (ix, iy) = (1.0 - fx, 1.0 - fy);
    [ix * iy, fx * iy, ix * fy, fx * fy]
}

/// Incoming-sample blend weight `α` for an exponential moving average.
///
/// With a running `sample_count` of `n`, the unbiased average weight is `1 / n`.
/// `confidence ∈ [0, 1]` scales how much the history is trusted: low confidence
/// raises `α` toward `1` (favour the fresh sample).  A `min_alpha` floor keeps
/// the filter responsive to moving content so it never fully freezes.
///
/// Inputs are clamped defensively; the result lies in `[min_alpha, 1]`.
#[inline]
pub fn accumulation_alpha(sample_count: f32, confidence: f32, min_alpha: f32) -> f32 {
    let n = clamp_count(sample_count, f32::INFINITY);
    let c = finite_or(confidence, 1.0).clamp(0.0, 1.0);
    let floor = finite_or(min_alpha, 0.0).clamp(0.0, 1.0);
    // Base unbiased weight, raised as confidence drops toward zero.
    let base = 1.0 / n;
    let alpha = base + (1.0 - base) * (1.0 - c);
    alpha.clamp(floor, 1.0)
}

/// Blends a `history` colour with an incoming `current` sample using blend
/// weight `alpha` (the weight of `current`).
///
/// Result = `lerp(history, current, alpha)`; both inputs are sanitized and the
/// output is non-negative and finite.
#[inline]
pub fn blend_history(history: Vec3, current: Vec3, alpha: f32) -> Vec3 {
    let a = finite_or(alpha, 1.0).clamp(0.0, 1.0);
    let h = sanitize(history);
    let c = sanitize(current);
    sanitize(h + (c - h) * a)
}

/// Advances the running sample count by one frame, saturating at `max_samples`.
///
/// `max_samples` bounds the effective history length (and therefore the minimum
/// blend weight `1 / max_samples`).  The count is clamped to `[1, max_samples]`.
#[inline]
pub fn advance_sample_count(sample_count: f32, max_samples: f32) -> f32 {
    let max = finite_or(max_samples, 1.0).max(1.0);
    clamp_count(sample_count + 1.0, max)
}

/// Resets accumulation at a disoccluded pixel.
///
/// When reprojection is invalid (`disoccluded == true`) there is no usable
/// history, so the current sample becomes the new history and the sample count
/// restarts at `1`.  Otherwise the inputs pass through, with the count advanced
/// by one frame bounded by `max_samples`.
///
/// Returns `(new_history, new_sample_count)`.
#[inline]
pub fn resolve_disocclusion(
    disoccluded: bool,
    history: Vec3,
    current: Vec3,
    sample_count: f32,
    max_samples: f32,
) -> (Vec3, f32) {
    if disoccluded {
        (sanitize(current), 1.0)
    } else {
        (sanitize(history), advance_sample_count(sample_count, max_samples))
    }
}

/// Clamps `history` into the local colour neighbourhood box `[min, max]`
/// (variance-clipping style anti-ghosting).
///
/// The box is normally derived from the current frame's local mean ± `k·σ`; any
/// history colour outside it is a stale/ghosted value and is clamped back onto
/// the box.  Degenerate boxes (min > max per component) are repaired by
/// swapping the bounds.
#[inline]
pub fn clamp_history_to_box(history: Vec3, box_min: Vec3, box_max: Vec3) -> Vec3 {
    let h = sanitize(history);
    let lo = sanitize(box_min);
    let hi = sanitize(box_max);
    Vec3::new(
        clamp_ordered(h.x, lo.x, hi.x),
        clamp_ordered(h.y, lo.y, hi.y),
        clamp_ordered(h.z, lo.z, hi.z),
    )
}

/// Builds a luma-aware colour clamp box from a local `mean` and standard
/// deviation `sigma`, widened by `gamma` (the variance-clipping aggressiveness).
///
/// Returns `(box_min, box_max)` where each component is `mean ± gamma·sigma`,
/// clamped non-negative.  Larger `gamma` keeps more history (softer); smaller
/// `gamma` clamps harder (less ghosting, more flicker).
#[inline]
pub fn luma_clip_box(mean: Vec3, sigma: Vec3, gamma: f32) -> (Vec3, Vec3) {
    let m = sanitize(mean);
    let s = sanitize(sigma);
    let g = finite_or(gamma, 1.0).max(0.0);
    let half = s * g;
    (sanitize(m - half), sanitize(m + half))
}

/// A luma-stability factor in `[0, 1]`: how much to trust history given the
/// luminance disagreement between `history` and `current`.
///
/// Large relative luma differences (fast changing content) drive the factor
/// toward `0` so the resolve leans on the current frame; agreement keeps it near
/// `1`.  `sensitivity` scales how quickly trust falls off.
#[inline]
pub fn luma_stability(history: Vec3, current: Vec3, sensitivity: f32) -> f32 {
    let lh = luma(history);
    let lc = luma(current);
    let s = finite_or(sensitivity, 1.0).max(0.0);
    let denom = lh.max(lc) + WEIGHT_EPSILON;
    let rel = ((lh - lc).abs() / denom) * s;
    (1.0 - rel).clamp(0.0, 1.0)
}

/// A sparse high-resolution accumulation grid for scatter-style ("splat")
/// resolve.
///
/// Low-resolution jittered samples are splatted into `color` with bilinear
/// weights, their coverage tracked in `weight`.  [`HighResAccumulator::resolve`]
/// normalizes by coverage; texels that never received a sample report as holes
/// for the spatial reconstruction pass (see [`super::resolve`]).
///
/// Storage is row-major, `width * height` texels.
#[derive(Clone, Debug, PartialEq)]
pub struct HighResAccumulator {
    width: usize,
    height: usize,
    color: Vec<Vec3>,
    weight: Vec<f32>,
}

impl HighResAccumulator {
    /// Creates an empty accumulator of the given grid dimensions.
    ///
    /// Zero in either dimension yields an empty (all-hole) grid.
    #[inline]
    pub fn new(width: usize, height: usize) -> Self {
        let len = width.saturating_mul(height);
        Self {
            width,
            height,
            color: alloc::vec![Vec3::ZERO; len],
            weight: alloc::vec![0.0; len],
        }
    }

    /// Grid width in high-res texels.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Grid height in high-res texels.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Row-major index for `(x, y)`, or `None` when out of bounds.
    #[inline]
    fn index(&self, x: usize, y: usize) -> Option<usize> {
        if x < self.width && y < self.height {
            Some(y * self.width + x)
        } else {
            None
        }
    }

    /// Accumulated coverage weight at `(x, y)` (`0` for an untouched hole).
    #[inline]
    pub fn weight_at(&self, x: usize, y: usize) -> f32 {
        self.index(x, y).map(|i| self.weight[i]).unwrap_or(0.0)
    }

    /// Splats a low-res sample of `color` to the continuous high-res position
    /// `pos` (in high-res texel space) using bilinear weights, scaled by an
    /// optional `confidence` in `[0, 1]`.
    ///
    /// The lower-left integer texel is `floor(pos)`; the four surrounding texels
    /// receive the bilinear share of `confidence`.  Out-of-grid texels are
    /// skipped.  Non-finite inputs are ignored (no-op) to protect the grid.
    pub fn splat(&mut self, pos: Vec2, color: Vec3, confidence: f32) {
        if !pos.x.is_finite() || !pos.y.is_finite() {
            return;
        }
        let c = finite_or(confidence, 1.0).clamp(0.0, 1.0);
        if c <= 0.0 {
            return;
        }
        let col = sanitize(color);
        let base_x = pos.x.floor();
        let base_y = pos.y.floor();
        let frac = Vec2::new(pos.x - base_x, pos.y - base_y);
        let w = bilinear_splat_weights(frac);
        let bx = base_x as i64;
        let by = base_y as i64;
        // Order mirrors `bilinear_splat_weights`: (0,0),(1,0),(0,1),(1,1).
        let offsets = [(0i64, 0i64), (1, 0), (0, 1), (1, 1)];
        for (k, &(ox, oy)) in offsets.iter().enumerate() {
            let tx = bx + ox;
            let ty = by + oy;
            if tx < 0 || ty < 0 {
                continue;
            }
            if let Some(i) = self.index(tx as usize, ty as usize) {
                let wk = w[k] * c;
                if wk > 0.0 {
                    self.color[i] += col * wk;
                    self.weight[i] += wk;
                }
            }
        }
    }

    /// Resolves the splatted grid into normalized colours.
    ///
    /// Returns `(colors, filled)`: each covered texel's accumulated colour is
    /// divided by its weight; texels below [`WEIGHT_EPSILON`] coverage are
    /// `Vec3::ZERO` with `filled == false`, marking holes for the spatial
    /// reconstruction pass.
    pub fn resolve(&self) -> (Vec<Vec3>, Vec<bool>) {
        let mut colors = alloc::vec![Vec3::ZERO; self.color.len()];
        let mut filled = alloc::vec![false; self.color.len()];
        for i in 0..self.color.len() {
            if self.weight[i] > WEIGHT_EPSILON {
                colors[i] = sanitize(self.color[i] / self.weight[i]);
                filled[i] = true;
            }
        }
        (colors, filled)
    }
}

/// Clamps a sample count to `[1, max]`, repairing non-finite inputs to `1`.
#[inline]
fn clamp_count(n: f32, max: f32) -> f32 {
    let v = finite_or(n, 1.0);
    let hi = if max.is_finite() { max.max(1.0) } else { f32::INFINITY };
    v.clamp(1.0, hi)
}

/// Returns `v` when finite, else the `fallback`.
#[inline]
fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() { v } else { fallback }
}

/// Clamps `v` to `[a, b]`, swapping the bounds if they are inverted.
#[inline]
fn clamp_ordered(v: f32, a: f32, b: f32) -> f32 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    v.clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_weights_sum_to_one() {
        for &(fx, fy) in &[(0.0, 0.0), (0.5, 0.5), (0.25, 0.75), (0.9, 0.1)] {
            let w = bilinear_splat_weights(Vec2::new(fx, fy));
            let s: f32 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "sum {s} for ({fx},{fy})");
            assert!(w.iter().all(|&x| x >= 0.0));
        }
    }

    #[test]
    fn bilinear_corner_weights() {
        // frac (0,0) puts all weight on the lower-left texel.
        let w = bilinear_splat_weights(Vec2::ZERO);
        assert!((w[0] - 1.0).abs() < 1e-6);
        // frac (1,1) puts all weight on the diagonal texel.
        let w = bilinear_splat_weights(Vec2::ONE);
        assert!((w[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn alpha_decays_as_one_over_n() {
        // With full confidence and no floor, α == 1/n.
        for n in 1..=16u32 {
            let a = accumulation_alpha(n as f32, 1.0, 0.0);
            assert!((a - 1.0 / n as f32).abs() < 1e-6, "n {n} => {a}");
        }
    }

    #[test]
    fn alpha_respects_floor_and_confidence() {
        // Low confidence pushes α toward 1.
        let a = accumulation_alpha(100.0, 0.0, 0.0);
        assert!((a - 1.0).abs() < 1e-6, "no-confidence α {a}");
        // Floor keeps the filter responsive.
        let a = accumulation_alpha(1000.0, 1.0, 0.1);
        assert!(a >= 0.1 - 1e-6, "floor not applied: {a}");
    }

    #[test]
    fn ema_converges_to_true_mean() {
        // Feeding a constant signal must converge the history to that signal.
        let truth = Vec3::new(0.4, 0.7, 0.2);
        let mut hist = Vec3::ZERO;
        let mut count = 1.0f32;
        for _ in 0..256 {
            let a = accumulation_alpha(count, 1.0, 0.0);
            hist = blend_history(hist, truth, a);
            count = advance_sample_count(count, 1024.0);
        }
        assert!((hist - truth).length() < 1e-3, "did not converge: {hist:?}");
    }

    #[test]
    fn ema_averages_two_level_signal() {
        // Alternating two colours should converge near their mean under 1/n.
        let a_col = Vec3::new(0.2, 0.2, 0.2);
        let b_col = Vec3::new(0.8, 0.8, 0.8);
        let mut hist = a_col;
        let mut count = 1.0f32;
        for i in 0..4096 {
            let s = if i % 2 == 0 { a_col } else { b_col };
            let alpha = accumulation_alpha(count, 1.0, 0.0);
            hist = blend_history(hist, s, alpha);
            count = advance_sample_count(count, 1.0e9);
        }
        assert!((hist.x - 0.5).abs() < 0.02, "mean {hist:?}");
    }

    #[test]
    fn disocclusion_resets_history() {
        let (h, n) = resolve_disocclusion(
            true,
            Vec3::new(9.0, 9.0, 9.0),
            Vec3::new(0.3, 0.3, 0.3),
            500.0,
            512.0,
        );
        assert_eq!(h, Vec3::new(0.3, 0.3, 0.3));
        assert_eq!(n, 1.0);
    }

    #[test]
    fn no_disocclusion_advances_count() {
        let (_h, n) = resolve_disocclusion(false, Vec3::ZERO, Vec3::ONE, 7.0, 64.0);
        assert_eq!(n, 8.0);
    }

    #[test]
    fn count_saturates_at_max() {
        let n = advance_sample_count(64.0, 64.0);
        assert_eq!(n, 64.0);
    }

    #[test]
    fn history_clamp_box_pulls_outliers_in() {
        let clamped = clamp_history_to_box(
            Vec3::new(2.0, -1.0, 0.5),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        assert_eq!(clamped, Vec3::new(1.0, 0.0, 0.5));
    }

    #[test]
    fn luma_clip_box_brackets_mean() {
        let (lo, hi) = luma_clip_box(Vec3::splat(0.5), Vec3::splat(0.1), 2.0);
        assert!(lo.x <= 0.3 + 1e-6 && hi.x >= 0.7 - 1e-6, "lo {lo:?} hi {hi:?}");
    }

    #[test]
    fn luma_stability_high_for_agreement() {
        let f = luma_stability(Vec3::splat(0.5), Vec3::splat(0.5), 1.0);
        assert!((f - 1.0).abs() < 1e-6, "stable f {f}");
    }

    #[test]
    fn luma_stability_low_for_disagreement() {
        let f = luma_stability(Vec3::splat(0.1), Vec3::splat(0.9), 1.0);
        assert!(f < 0.3, "unstable f {f}");
    }

    #[test]
    fn splat_constant_resolves_to_constant() {
        // Splatting the same colour everywhere must resolve to that colour on
        // covered texels (energy / value preservation).
        let mut acc = HighResAccumulator::new(4, 4);
        let col = Vec3::new(0.3, 0.6, 0.9);
        for gy in 0..4 {
            for gx in 0..4 {
                acc.splat(Vec2::new(gx as f32 + 0.3, gy as f32 + 0.7), col, 1.0);
            }
        }
        let (colors, filled) = acc.resolve();
        for (i, &f) in filled.iter().enumerate() {
            if f {
                assert!((colors[i] - col).length() < 1e-5, "texel {i} {:?}", colors[i]);
            }
        }
    }

    #[test]
    fn splat_marks_holes() {
        let acc = HighResAccumulator::new(2, 2);
        let (_c, filled) = acc.resolve();
        assert!(filled.iter().all(|&f| !f), "empty grid should be all holes");
    }

    #[test]
    fn splat_ignores_non_finite_position() {
        let mut acc = HighResAccumulator::new(2, 2);
        acc.splat(Vec2::new(f32::NAN, 0.0), Vec3::ONE, 1.0);
        assert_eq!(acc.weight_at(0, 0), 0.0);
    }

    #[test]
    fn splat_weight_proportional_to_bilinear_share() {
        // A sample at the exact centre of a texel deposits its full weight there.
        let mut acc = HighResAccumulator::new(3, 3);
        acc.splat(Vec2::new(1.0, 1.0), Vec3::ONE, 1.0);
        assert!((acc.weight_at(1, 1) - 1.0).abs() < 1e-6, "{}", acc.weight_at(1, 1));
    }

    #[test]
    fn sanitize_repairs_nan_and_negatives() {
        let s = sanitize(Vec3::new(f32::NAN, -2.0, 0.5));
        assert_eq!(s, Vec3::new(0.0, 0.0, 0.5));
    }

    #[test]
    fn is_deterministic() {
        let mut a = HighResAccumulator::new(4, 4);
        let mut b = HighResAccumulator::new(4, 4);
        for i in 0..8 {
            let p = Vec2::new(0.5 + i as f32 * 0.11, 1.5);
            a.splat(p, Vec3::splat(0.4), 0.8);
            b.splat(p, Vec3::splat(0.4), 0.8);
        }
        assert_eq!(a.resolve().0, b.resolve().0);
    }
}
