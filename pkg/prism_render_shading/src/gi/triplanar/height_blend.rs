//! Height-map-aware material blending — CPU golden reference.
//!
//! Where two tiling materials meet (rock over sand, moss over stone), a plain
//! linear cross-fade looks like a translucent decal: the boundary is a soft,
//! obviously artificial gradient.  Artists instead want the *higher* material
//! to poke through the lower one along their natural relief — pebbles standing
//! above the sand, grout sitting below the tiles.  Height blending achieves
//! that by biasing the blend toward whichever layer has the greater local
//! height, producing a crisp, interlocking seam that follows the texture.
//!
//! For a set of layers each carrying a sampled relief height `h_i` and a
//! control weight `w_i` (splat-map opacity, mask, vertex weight), the blend
//! weight is
//!
//! ```text
//! peak_i   = h_i + w_i
//! max_peak = max_i(peak_i)
//! b_i      = max(0, peak_i - max_peak + depth)
//! weight_i = b_i / sum_j(b_j)
//! ```
//!
//! `depth` is the transition width.  Only layers whose peak lies within `depth`
//! of the tallest survive the `max(0, …)` clamp, so a small `depth` yields a
//! razor-sharp seam (effectively a height-ordered `max`) and a large `depth`
//! relaxes toward the ordinary linear blend.  The subtraction of `max_peak`
//! keeps the operator numerically stable regardless of the absolute heights.
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; none are needed here, so
//!   the module stays in plain arithmetic.
//! * Weights are non-negative and sum to one (within rounding) whenever at
//!   least one layer has positive control weight.
//! * Defensive clamping everywhere: non-finite heights/weights are treated as
//!   absent, `depth` is clamped to a small positive floor, an empty or
//!   all-zero input returns an empty / uniform result, and no `NaN`/`inf`
//!   ever escapes.

use alloc::vec::Vec;

/// Smallest transition width.  A zero width would make the operator a hard
/// `argmax`, which is discontinuous and loses the single-winner normalisation;
/// a tiny positive floor keeps exactly the tallest layer(s) selected.
const MIN_DEPTH: f32 = 1.0e-4;

/// Largest transition width honoured.  Beyond this the blend has already
/// relaxed into an essentially linear cross-fade, so clamping bounds the math.
const MAX_DEPTH: f32 = 16.0;

/// One material layer entering the blend.
///
/// `height` is the sampled relief value (any finite scale; larger stands
/// higher) and `control` is the layer's external opacity / mask weight in
/// `[0, 1]`.  A `control` of zero removes the layer from the blend entirely.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightSample {
    /// Local relief height of the layer at this texel.
    pub height: f32,
    /// External control weight (splat opacity / mask), clamped to `[0, 1]`.
    pub control: f32,
}

impl HeightSample {
    /// Builds a sample, sanitising both fields.
    pub fn new(height: f32, control: f32) -> Self {
        Self {
            height: finite_or(height, 0.0),
            control: clamp01(finite_or(control, 0.0)),
        }
    }

    /// The layer "peak" = relief height plus control weight.
    fn peak(self) -> f32 {
        self.height + self.control
    }
}

/// Computes normalised height-blend weights for an arbitrary layer stack.
///
/// Returns one weight per input layer, in input order, all non-negative and
/// summing to one when any layer has positive control.  If the stack is empty,
/// returns an empty vector.  If every layer has zero control (nothing to show),
/// returns a uniform distribution so downstream code still gets a valid blend.
pub fn height_blend_weights(layers: &[HeightSample], depth: f32) -> Vec<f32> {
    let count = layers.len();
    let mut out = Vec::with_capacity(count);
    if count == 0 {
        return out;
    }

    let depth = sanitize_depth(depth);

    // Peaks, with fully masked-out layers forced below any real contender.
    let mut max_peak = f32::NEG_INFINITY;
    let mut any_active = false;
    for layer in layers {
        let s = HeightSample::new(layer.height, layer.control);
        if s.control > 0.0 {
            any_active = true;
            let p = s.peak();
            if p > max_peak {
                max_peak = p;
            }
        }
    }

    if !any_active {
        // Nothing to display — hand back a uniform, valid partition of unity.
        let uniform = 1.0 / count as f32;
        out.resize(count, uniform);
        return out;
    }

    // Raised blend factors b_i, then normalise.
    let mut sum = 0.0f32;
    for layer in layers {
        let s = HeightSample::new(layer.height, layer.control);
        let b = if s.control > 0.0 {
            (s.peak() - max_peak + depth).max(0.0)
        } else {
            0.0
        };
        out.push(b);
        sum += b;
    }

    if sum > f32::MIN_POSITIVE {
        let inv = 1.0 / sum;
        for w in out.iter_mut() {
            *w *= inv;
        }
    } else {
        // Degenerate (all b_i collapsed to zero): fall back to the tallest.
        let mut best = 0usize;
        let mut best_peak = f32::NEG_INFINITY;
        for (i, layer) in layers.iter().enumerate() {
            let s = HeightSample::new(layer.height, layer.control);
            if s.control > 0.0 && s.peak() > best_peak {
                best_peak = s.peak();
                best = i;
            }
        }
        for (i, w) in out.iter_mut().enumerate() {
            *w = if i == best { 1.0 } else { 0.0 };
        }
    }

    out
}

/// Two-layer height blend returning the weight of the *second* layer.
///
/// A convenience wrapper over [`height_blend_weights`] for the common
/// over/under case.  The returned factor `t` is in `[0, 1]`; the shader uses
/// `mix(layer_a, layer_b, t)`.  Both layers are given unit control weight, so
/// the blend is driven purely by the relief heights and `depth`.
pub fn height_blend_factor(height_a: f32, height_b: f32, depth: f32) -> f32 {
    let layers = [
        HeightSample::new(height_a, 1.0),
        HeightSample::new(height_b, 1.0),
    ];
    let w = height_blend_weights(&layers, depth);
    // `w` always has two finite entries summing to one here.
    clamp01(w[1])
}

/// Clamps the transition width to a finite, positive, bounded range.
fn sanitize_depth(d: f32) -> f32 {
    if d.is_finite() {
        d.clamp(MIN_DEPTH, MAX_DEPTH)
    } else {
        MIN_DEPTH
    }
}

/// Returns `x` when finite, otherwise `fallback`.
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        fallback
    }
}

/// Clamps to the unit interval.
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(v: &[f32]) -> f32 {
        v.iter().copied().sum()
    }

    /// Every weight set is a partition of unity, over varied stacks.
    #[test]
    fn weights_are_energy_normalized() {
        let stacks: [&[HeightSample]; 3] = [
            &[
                HeightSample::new(0.2, 1.0),
                HeightSample::new(0.8, 1.0),
            ],
            &[
                HeightSample::new(0.5, 0.5),
                HeightSample::new(0.4, 0.7),
                HeightSample::new(0.9, 0.2),
            ],
            &[
                HeightSample::new(0.3, 1.0),
                HeightSample::new(0.3, 1.0),
                HeightSample::new(0.3, 1.0),
                HeightSample::new(0.3, 1.0),
            ],
        ];
        for stack in stacks {
            for &depth in &[0.05f32, 0.2, 1.0] {
                let w = height_blend_weights(stack, depth);
                assert_eq!(w.len(), stack.len());
                assert!((sum(&w) - 1.0).abs() < 1e-5, "sum {} depth {}", sum(&w), depth);
                assert!(w.iter().all(|&x| x >= 0.0), "negative weight {:?}", w);
            }
        }
    }

    /// The taller layer takes the larger share, and a sharp seam nearly wins
    /// outright.
    #[test]
    fn higher_layer_dominates() {
        let layers = [
            HeightSample::new(0.2, 1.0),
            HeightSample::new(0.9, 1.0),
        ];
        // Narrow transition: the taller (second) layer should dominate.
        let sharp = height_blend_weights(&layers, 0.05);
        assert!(sharp[1] > sharp[0], "taller must win: {:?}", sharp);
        assert!(sharp[1] > 0.99, "sharp seam nearly all second layer: {:?}", sharp);

        // The convenience factor agrees.
        assert!(height_blend_factor(0.2, 0.9, 0.05) > 0.99);
        assert!(height_blend_factor(0.9, 0.2, 0.05) < 0.01);
    }

    /// Widening `depth` broadens the transition: equal-ish layers approach 0.5.
    #[test]
    fn depth_controls_transition_width() {
        // Heights 0.4 vs 0.6: the gap is 0.2.
        let a = 0.4f32;
        let b = 0.6f32;

        // Very narrow: the taller layer (b) is almost exclusively chosen.
        let narrow = height_blend_factor(a, b, 0.02);
        // Wide: the transition softens toward an even split.
        let wide = height_blend_factor(a, b, 4.0);

        assert!(narrow > wide, "narrow {} should favor taller more than wide {}", narrow, wide);
        assert!(wide > 0.49 && wide < 0.6, "wide blend near even: {}", wide);
        assert!(narrow > 0.95, "narrow blend near winner-takes-all: {}", narrow);
    }

    /// Height blending is sharper than a plain linear height-ratio blend near a
    /// seam: for close heights the dominant layer gets more than its linear
    /// share.
    #[test]
    fn sharper_than_linear() {
        let a = 0.45f32;
        let b = 0.55f32;
        // A linear height-weighted blend would give b a share of
        // b / (a + b) ~= 0.55; height blending with a narrow depth exceeds it.
        let linear_share = b / (a + b);
        let hb = height_blend_factor(a, b, 0.05);
        assert!(hb > linear_share, "height blend {} should beat linear {}", hb, linear_share);
    }

    /// Equal heights and controls give an exactly even split.
    #[test]
    fn ties_split_evenly() {
        let layers = [
            HeightSample::new(0.5, 1.0),
            HeightSample::new(0.5, 1.0),
            HeightSample::new(0.5, 1.0),
        ];
        let w = height_blend_weights(&layers, 0.3);
        for &x in &w {
            assert!((x - 1.0 / 3.0).abs() < 1e-5, "even split expected: {:?}", w);
        }
    }

    /// Control weight gates a layer: zero control removes it from the blend.
    #[test]
    fn zero_control_excludes_layer() {
        let layers = [
            HeightSample::new(0.9, 0.0), // tall but fully masked out
            HeightSample::new(0.2, 1.0),
        ];
        let w = height_blend_weights(&layers, 0.1);
        assert!(w[0].abs() < 1e-6, "masked layer must not contribute: {:?}", w);
        assert!((w[1] - 1.0).abs() < 1e-5, "visible layer takes everything: {:?}", w);
    }

    /// Empty input yields an empty result; all-masked yields a uniform one.
    #[test]
    fn degenerate_inputs_are_safe() {
        let empty = height_blend_weights(&[], 0.2);
        assert!(empty.is_empty());

        let masked = [
            HeightSample::new(0.5, 0.0),
            HeightSample::new(0.5, 0.0),
        ];
        let w = height_blend_weights(&masked, 0.2);
        assert!((sum(&w) - 1.0).abs() < 1e-5, "uniform fallback sums to one: {:?}", w);
        assert!(w.iter().all(|&x| (x - 0.5).abs() < 1e-5), "uniform fallback: {:?}", w);
    }

    /// Non-finite inputs are sanitised, never producing NaN weights.
    #[test]
    fn non_finite_inputs_are_sanitized() {
        let layers = [
            HeightSample::new(f32::NAN, 1.0),
            HeightSample::new(0.6, f32::INFINITY),
        ];
        let w = height_blend_weights(&layers, f32::NAN);
        assert!(w.iter().all(|x| x.is_finite()), "weights finite: {:?}", w);
        assert!((sum(&w) - 1.0).abs() < 1e-5, "still normalized: {:?}", w);
    }
}
