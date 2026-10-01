//! Compositing helpers for order-independent transparency — CPU golden reference.
//!
//! This module collects the small, well-defined blending operations that the
//! OIT resolve passes share: premultiplied-alpha "over" compositing, folding a
//! transparent result onto an opaque background, converting the various
//! coverage encodings (revealage, transmittance) into blend weights, and a
//! ground-truth sorted compositor used to *verify* that an OIT estimate is
//! genuinely order independent.
//!
//! The reference "over" operator (Porter & Duff 1984) is the correct — but
//! order *dependent* — compositing both [`crate::gi::oit::weighted_blended`] and
//! [`crate::gi::oit::moment_based`] approximate.  [`composite_sorted`] evaluates
//! it exactly by sorting fragments back-to-front, giving the baseline the
//! sort-free estimators are compared against in tests and tooling.
//!
//! # Conventions
//! * **Premultiplied** colours store `rgb` already multiplied by `a`; the `over`
//!   operators here operate on that representation, which composes associatively.
//! * **Straight** colours store un-multiplied `rgb` with a separate `a`;
//!   [`premultiply`] / [`unpremultiply`] convert between the two.
//! * `revealage ∈ [0, 1]` is the fraction of background still visible through a
//!   transparent stack; `coverage = 1 - revealage`.  `transmittance ∈ [0, 1]` is
//!   the equivalent light-throughput encoding used by moment-based OIT.
//! * Depth increases away from the eye; [`composite_sorted`] blends from the
//!   largest depth (farthest) to the smallest (nearest).
//! * Every function clamps alpha/coverage to `[0, 1]` and guards against
//!   `NaN`/`inf`, so outputs are always finite.
//!
//! # References
//! * Porter & Duff 1984, *Compositing Digital Images*, SIGGRAPH.
//! * McGuire & Bavoil 2013, *Weighted Blended Order-Independent Transparency*.

use bevy_math::{Vec3, Vec4};

/// Sanitises a scalar to a finite value, substituting `fallback` otherwise.
#[inline]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Sanitises a colour component-wise, replacing non-finite channels with `0`.
#[inline]
fn finite_vec3(v: Vec3) -> Vec3 {
    Vec3::new(finite_or(v.x, 0.0), finite_or(v.y, 0.0), finite_or(v.z, 0.0))
}

/// Premultiplies a straight `color`/`alpha` pair into a premultiplied `Vec4`.
///
/// `alpha` is clamped to `[0, 1]`; the returned `xyz` is `color·alpha` and `w`
/// is the alpha.
#[inline]
pub fn premultiply(color: Vec3, alpha: f32) -> Vec4 {
    let a = finite_or(alpha, 0.0).clamp(0.0, 1.0);
    let c = finite_vec3(color) * a;
    Vec4::new(c.x, c.y, c.z, a)
}

/// Recovers a straight `(color, alpha)` pair from a premultiplied `Vec4`.
///
/// Divides `rgb` by the alpha when it is positive; a zero-alpha input yields a
/// black colour.  The returned alpha is clamped to `[0, 1]`.
#[inline]
pub fn unpremultiply(premult: Vec4) -> (Vec3, f32) {
    let a = finite_or(premult.w, 0.0).clamp(0.0, 1.0);
    let rgb = Vec3::new(premult.x, premult.y, premult.z);
    if a > 0.0 {
        (finite_vec3(rgb) / a, a)
    } else {
        (Vec3::ZERO, 0.0)
    }
}

/// Premultiplied-alpha "over": `src over dst = src + (1 - src.a)·dst`.
///
/// Both operands are premultiplied `(rgb·a, a)` vectors; the result is likewise
/// premultiplied.  This operator is associative, which is what lets layered
/// transparency be composited in groups.
#[inline]
pub fn over_premultiplied(src: Vec4, dst: Vec4) -> Vec4 {
    let s = Vec4::new(
        finite_or(src.x, 0.0),
        finite_or(src.y, 0.0),
        finite_or(src.z, 0.0),
        finite_or(src.w, 0.0).clamp(0.0, 1.0),
    );
    let d = Vec4::new(
        finite_or(dst.x, 0.0),
        finite_or(dst.y, 0.0),
        finite_or(dst.z, 0.0),
        finite_or(dst.w, 0.0).clamp(0.0, 1.0),
    );
    s + (1.0 - s.w) * d
}

/// Straight-alpha "over" of a `src` layer onto a `dst` layer.
///
/// Internally premultiplies, applies [`over_premultiplied`], and converts back
/// to straight `(color, alpha)`.
#[inline]
pub fn over_straight(src_color: Vec3, src_alpha: f32, dst_color: Vec3, dst_alpha: f32) -> (Vec3, f32) {
    let out = over_premultiplied(
        premultiply(src_color, src_alpha),
        premultiply(dst_color, dst_alpha),
    );
    unpremultiply(out)
}

/// Composites a premultiplied transparent result over an opaque background.
///
/// `src_premult` is the premultiplied transparent colour and `coverage` is the
/// transparent coverage `1 - revealage`.  The background is opaque, so
/// `out = src_premult.rgb + (1 - coverage)·background`.
#[inline]
pub fn composite_over_opaque(src_premult: Vec4, coverage: f32, background: Vec3) -> Vec3 {
    let cov = finite_or(coverage, 0.0).clamp(0.0, 1.0);
    let src = Vec3::new(
        finite_or(src_premult.x, 0.0),
        finite_or(src_premult.y, 0.0),
        finite_or(src_premult.z, 0.0),
    );
    src + (1.0 - cov) * finite_vec3(background)
}

/// Composites an average transparent `color` with `coverage` over `background`.
///
/// Straight-colour counterpart of [`composite_over_opaque`]:
/// `out = color·coverage + background·(1 - coverage)`.
#[inline]
pub fn composite_average(color: Vec3, coverage: f32, background: Vec3) -> Vec3 {
    let cov = finite_or(coverage, 0.0).clamp(0.0, 1.0);
    finite_vec3(color) * cov + finite_vec3(background) * (1.0 - cov)
}

/// Converts a revealage value to transparent coverage `1 - revealage`.
#[inline]
pub fn revealage_to_coverage(revealage: f32) -> f32 {
    1.0 - finite_or(revealage, 1.0).clamp(0.0, 1.0)
}

/// Converts a transmittance value to opacity/coverage `1 - transmittance`.
#[inline]
pub fn transmittance_to_coverage(transmittance: f32) -> f32 {
    1.0 - finite_or(transmittance, 1.0).clamp(0.0, 1.0)
}

/// A single transparent fragment for the ground-truth sorted compositor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SortableFragment {
    /// Straight linear RGB.
    pub color: Vec3,
    /// Coverage `∈ [0, 1]`.
    pub alpha: f32,
    /// View-space depth (larger = farther from the eye).
    pub depth: f32,
}

impl SortableFragment {
    /// Builds a fragment, clamping `alpha` to `[0, 1]`.
    #[inline]
    pub fn new(color: Vec3, alpha: f32, depth: f32) -> Self {
        Self {
            color: finite_vec3(color),
            alpha: finite_or(alpha, 0.0).clamp(0.0, 1.0),
            depth: finite_or(depth, 0.0),
        }
    }
}

/// Exact back-to-front "over" compositing — the order-*dependent* ground truth.
///
/// Sorts `fragments` in place by descending depth (farthest first) and folds
/// them over `background` with the straight-alpha operator.  This is the
/// reference result the sort-free OIT estimators approximate; it is used by
/// [`max_channel_error`] and the verification tests to quantify their error.
pub fn composite_sorted(fragments: &mut [SortableFragment], background: Vec3) -> Vec3 {
    fragments.sort_by(|a, b| {
        b.depth
            .partial_cmp(&a.depth)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut out = finite_vec3(background);
    for frag in fragments.iter() {
        out = frag.color * frag.alpha + out * (1.0 - frag.alpha);
    }
    out
}

/// Largest absolute per-channel difference between two colours.
///
/// Handy for asserting that an OIT estimate stays within tolerance of the
/// sorted ground truth, or that two orderings agree.
#[inline]
pub fn max_channel_error(a: Vec3, b: Vec3) -> f32 {
    let d = (finite_vec3(a) - finite_vec3(b)).abs();
    d.x.max(d.y).max(d.z)
}

/// Checks that every colour in `results` agrees within `eps` per channel.
///
/// Order-independent resolves must produce the same colour for any input
/// permutation; feeding the resolves of several permutations here verifies that
/// property.  An empty or single-element slice is trivially consistent.
pub fn is_order_independent(results: &[Vec3], eps: f32) -> bool {
    let tol = finite_or(eps, 0.0).max(0.0);
    match results.split_first() {
        None => true,
        Some((first, rest)) => rest.iter().all(|r| max_channel_error(*first, *r) <= tol),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn premultiply_round_trips() {
        let (c, a) = unpremultiply(premultiply(Vec3::new(0.2, 0.5, 0.8), 0.4));
        assert!((c - Vec3::new(0.2, 0.5, 0.8)).length() < 1e-6);
        assert!((a - 0.4).abs() < 1e-6);
    }

    #[test]
    fn over_opaque_source_hides_background() {
        let src = premultiply(Vec3::new(1.0, 0.0, 0.0), 1.0);
        let out = over_premultiplied(src, premultiply(Vec3::ONE, 1.0));
        let (c, a) = unpremultiply(out);
        assert!((c - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
        assert!((a - 1.0).abs() < 1e-6);
    }

    #[test]
    fn over_transparent_source_shows_background() {
        let out = over_premultiplied(premultiply(Vec3::X, 0.0), premultiply(Vec3::Y, 1.0));
        let (c, _) = unpremultiply(out);
        assert!((c - Vec3::Y).length() < 1e-6);
    }

    #[test]
    fn over_is_associative() {
        let a = premultiply(Vec3::new(0.8, 0.1, 0.1), 0.5);
        let b = premultiply(Vec3::new(0.1, 0.8, 0.1), 0.4);
        let c = premultiply(Vec3::new(0.1, 0.1, 0.8), 0.3);
        let left = over_premultiplied(over_premultiplied(a, b), c);
        let right = over_premultiplied(a, over_premultiplied(b, c));
        let (lc, _) = unpremultiply(left);
        let (rc, _) = unpremultiply(right);
        assert!(max_channel_error(lc, rc) < 1e-6);
    }

    #[test]
    fn coverage_conversions_round_trip() {
        assert!((revealage_to_coverage(0.25) - 0.75).abs() < 1e-6);
        assert!((transmittance_to_coverage(0.3) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn composite_over_opaque_matches_average_form() {
        let color = Vec3::new(0.6, 0.3, 0.1);
        let coverage = 0.4;
        let bg = Vec3::new(0.1, 0.1, 0.1);
        let premult = premultiply(color, coverage);
        let a = composite_over_opaque(premult, coverage, bg);
        let b = composite_average(color, coverage, bg);
        assert!(max_channel_error(a, b) < 1e-6);
    }

    #[test]
    fn sorted_compositing_is_order_independent_as_ground_truth() {
        let bg = Vec3::new(0.05, 0.05, 0.05);
        let mut forward = [
            SortableFragment::new(Vec3::X, 0.5, 1.0),
            SortableFragment::new(Vec3::Y, 0.4, 5.0),
            SortableFragment::new(Vec3::Z, 0.6, 3.0),
        ];
        let mut reverse = forward;
        reverse.reverse();
        let a = composite_sorted(&mut forward, bg);
        let b = composite_sorted(&mut reverse, bg);
        assert!(max_channel_error(a, b) < 1e-6, "a={a:?} b={b:?}");
    }

    #[test]
    fn sorted_matches_hand_computed_two_layers() {
        // Near red (a=0.5) over far green (a=1.0) over black.
        let bg = Vec3::ZERO;
        let mut frags = [
            SortableFragment::new(Vec3::X, 0.5, 1.0),
            SortableFragment::new(Vec3::Y, 1.0, 5.0),
        ];
        let out = composite_sorted(&mut frags, bg);
        // far: green. near over: red*0.5 + green*0.5.
        let expected = Vec3::new(0.5, 0.5, 0.0);
        assert!(max_channel_error(out, expected) < 1e-6, "out={out:?}");
    }

    #[test]
    fn order_independence_checker() {
        assert!(is_order_independent(&[], 1e-6));
        assert!(is_order_independent(
            &[Vec3::splat(0.5), Vec3::splat(0.5 + 1e-7)],
            1e-6
        ));
        assert!(!is_order_independent(
            &[Vec3::ZERO, Vec3::splat(0.5)],
            1e-6
        ));
    }

    #[test]
    fn degenerate_inputs_never_nan() {
        let out = composite_over_opaque(
            Vec4::splat(f32::NAN),
            f32::NAN,
            Vec3::splat(f32::INFINITY),
        );
        assert!(out.is_finite());
        let (c, a) = unpremultiply(Vec4::new(f32::NAN, 0.0, 0.0, 0.0));
        assert!(c.is_finite() && a.is_finite());
    }
}
