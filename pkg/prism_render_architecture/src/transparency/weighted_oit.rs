//! Weighted-blended order-independent transparency (`WBOIT`) resolve.
//!
//! `WBOIT` (`McGuire` & Bavoil, *Weighted Blended Order-Independent
//! Transparency*, JCGT 2013) approximates the back-to-front "over" operator
//! without sorting. Each transparent fragment contributes to two targets:
//!
//! - an **accumulation** target `accum = (sum(C_i * a_i * w_i), sum(a_i * w_i))`
//!   written with additive blending, where `C_i` is the straight (un-premultiplied)
//!   linear `RGB` color, `a_i` the coverage, and `w_i` a depth/alpha weight; and
//! - a **revealage** target `R = prod(1 - a_i)` written with multiplicative
//!   blending, i.e. the fraction of the background that survives.
//!
//! The resolve reconstructs the frame color from those two moments:
//!
//! ```text
//! average = accum.rgb / max(accum.a, epsilon)
//! out     = average * (1 - R) + background * R
//! ```
//!
//! Because both targets are built only from a commutative sum and a commutative
//! product, the result is independent of the order fragments arrive in — the
//! defining property an `OIT` resolve must provide and the invariant the `GPU`
//! twin is validated against. This module is the pure-`CPU` golden reference for
//! that resolve: it owns the weighting functions and the moment accumulation so
//! the shipping compute path can be checked against bit-for-bit identical math.
//!
//! The weighting functions trade near/far separation against the risk of the
//! accumulation target overflowing a 16-bit float. All of them are evaluated
//! with integer-power multiplication so the resolve stays free of transcendental
//! intrinsics and is reproducible across backends.

/// Raises `base` to a small non-negative integer `exp` by repeated
/// multiplication.
///
/// Used instead of a transcendental power so the weighting functions remain
/// exactly reproducible on every backend. `exp` is a compile-time-sized small
/// integer (2..=6 for the shipped weights), so the loop is trivially bounded.
#[must_use]
fn ipow(base: f32, exp: u32) -> f32 {
    let mut acc = 1.0_f32;
    let mut i = 0;
    while i < exp {
        acc *= base;
        i += 1;
    }
    acc
}

/// Depth/alpha weighting function `w(z, a)` applied to each fragment.
///
/// Equations 7–10 are the view-space-depth weights proposed by `McGuire` &
/// Bavoil; they clamp the raw weight to `[1e-2, 3e3]` before scaling by alpha so
/// a 16-bit float accumulation target cannot overflow. [`WeightFunction::Uniform`]
/// drops the depth term entirely (`w = a`) and is appropriate for co-planar 2D
/// or screen-space layers where every fragment shares a depth.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightFunction {
    /// `w = a * clamp(10 / (1e-5 + (|z|/5)^2 + (|z|/200)^6), 1e-2, 3e3)`.
    ///
    /// Strong near-field emphasis; good for scenes with lots of close,
    /// high-coverage transparency.
    Equation7,
    /// `w = a * clamp(10 / (1e-5 + (|z|/10)^3 + (|z|/200)^6), 1e-2, 3e3)`.
    ///
    /// Slightly gentler near-field falloff than [`WeightFunction::Equation7`].
    Equation8,
    /// `w = a * clamp(10 / (1e-5 + (|z|/200)^4), 1e-2, 3e3)`.
    ///
    /// A single mid-range term; a reasonable general-purpose default.
    Equation9,
    /// `w = a * clamp(0.03 / (1e-5 + (|z|/200)^4), 1e-2, 3e3)`.
    ///
    /// The most conservative of the depth weights; least prone to overflow at
    /// the cost of weaker near/far separation.
    Equation10,
    /// `w = a`. Depth-independent weighting for co-planar layers.
    Uniform,
}

/// Lower clamp applied to the raw (pre-alpha) weight.
const WEIGHT_MIN: f32 = 1.0e-2;
/// Upper clamp applied to the raw (pre-alpha) weight.
const WEIGHT_MAX: f32 = 3.0e3;
/// Guards the `accum.a` divisor in the resolve against division by zero.
const RESOLVE_EPSILON: f32 = 1.0e-5;

impl WeightFunction {
    /// Evaluates the weight for a fragment at the given positive view-space
    /// depth and coverage.
    ///
    /// `view_depth` is treated as a distance, so its sign is ignored. `alpha` is
    /// used as supplied; callers that cannot guarantee `[0, 1]` should clamp
    /// first (the accumulator does).
    #[must_use]
    pub fn weight(self, view_depth: f32, alpha: f32) -> f32 {
        let z = view_depth.abs();
        let raw = match self {
            Self::Equation7 => 10.0 / (RESOLVE_EPSILON + ipow(z / 5.0, 2) + ipow(z / 200.0, 6)),
            Self::Equation8 => 10.0 / (RESOLVE_EPSILON + ipow(z / 10.0, 3) + ipow(z / 200.0, 6)),
            Self::Equation9 => 10.0 / (RESOLVE_EPSILON + ipow(z / 200.0, 4)),
            Self::Equation10 => 0.03 / (RESOLVE_EPSILON + ipow(z / 200.0, 4)),
            Self::Uniform => return alpha,
        };
        alpha * raw.clamp(WEIGHT_MIN, WEIGHT_MAX)
    }
}

/// A single transparent fragment to composite.
#[derive(Clone, Copy, Debug)]
pub struct OitFragment {
    /// Straight (un-premultiplied) linear `RGB` color.
    pub color: [f32; 3],
    /// Coverage / opacity. Clamped to `[0, 1]` on accumulation.
    pub alpha: f32,
    /// Positive view-space depth (distance from the camera).
    pub view_depth: f32,
}

impl OitFragment {
    /// Convenience constructor.
    #[must_use]
    pub const fn new(color: [f32; 3], alpha: f32, view_depth: f32) -> Self {
        Self {
            color,
            alpha,
            view_depth,
        }
    }
}

/// Accumulates the two `WBOIT` moments and resolves them to a frame color.
///
/// The accumulator starts at the cleared state (`accum = 0`, `revealage = 1`),
/// matching a target cleared to transparent-black with the revealage target
/// cleared to one. Fragments may be [`accumulate`](Self::accumulate)d in any
/// order; the resolved color is order-independent up to floating-point rounding.
#[derive(Clone, Copy, Debug)]
pub struct OitAccumulator {
    /// `rgb = sum(C_i * a_i * w_i)`, `a = sum(a_i * w_i)`.
    accum: [f32; 4],
    /// `prod(1 - a_i)`; the surviving background fraction.
    revealage: f32,
}

impl Default for OitAccumulator {
    fn default() -> Self {
        Self {
            accum: [0.0; 4],
            revealage: 1.0,
        }
    }
}

impl OitAccumulator {
    /// Creates an empty accumulator in the cleared state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one fragment's contribution under the given weighting function.
    pub fn accumulate(&mut self, fragment: OitFragment, weight_fn: WeightFunction) {
        let alpha = fragment.alpha.clamp(0.0, 1.0);
        let w = weight_fn.weight(fragment.view_depth, alpha);
        // Premultiplied color times weight into the additive accumulation.
        self.accum[0] += fragment.color[0] * alpha * w;
        self.accum[1] += fragment.color[1] * alpha * w;
        self.accum[2] += fragment.color[2] * alpha * w;
        self.accum[3] += alpha * w;
        // Revealage multiplies by the fragment's transparency.
        self.revealage *= 1.0 - alpha;
    }

    /// The fraction of this pixel covered by transparency: `1 - revealage`.
    #[must_use]
    pub fn coverage(&self) -> f32 {
        1.0 - self.revealage
    }

    /// The surviving background fraction: `prod(1 - a_i)`.
    #[must_use]
    pub fn revealage(&self) -> f32 {
        self.revealage
    }

    /// Resolves the accumulated moments over an opaque background color.
    ///
    /// With no fragments accumulated this returns `background` unchanged.
    #[must_use]
    pub fn resolve(&self, background: [f32; 3]) -> [f32; 3] {
        let denom = self.accum[3].max(RESOLVE_EPSILON);
        let reveal = self.revealage;
        let cover = 1.0 - reveal;
        let mut out = [0.0_f32; 3];
        for c in 0..3 {
            let average = self.accum[c] / denom;
            out[c] = average * cover + background[c] * reveal;
        }
        out
    }
}

/// Composites a slice of fragments in a single call.
///
/// Equivalent to feeding every fragment to an [`OitAccumulator`] and resolving;
/// provided for the common one-shot case. The result does not depend on the
/// order of `fragments`.
#[must_use]
pub fn composite(
    fragments: &[OitFragment],
    weight_fn: WeightFunction,
    background: [f32; 3],
) -> [f32; 3] {
    let mut acc = OitAccumulator::new();
    for &fragment in fragments {
        acc.accumulate(fragment, weight_fn);
    }
    acc.resolve(background)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const BG: [f32; 3] = [0.1, 0.2, 0.3];

    fn approx(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
    }

    #[test]
    fn empty_returns_background() {
        let acc = OitAccumulator::new();
        assert_eq!(acc.resolve(BG), BG);
        assert_eq!(acc.revealage(), 1.0);
        assert_eq!(acc.coverage(), 0.0);
    }

    #[test]
    fn single_opaque_fragment_returns_its_color() {
        // alpha = 1 -> revealage 0 -> background fully occluded, average = color.
        let frag = OitFragment::new([0.8, 0.4, 0.2], 1.0, 12.0);
        for wf in [
            WeightFunction::Equation7,
            WeightFunction::Equation8,
            WeightFunction::Equation9,
            WeightFunction::Equation10,
            WeightFunction::Uniform,
        ] {
            let out = composite(&[frag], wf, BG);
            assert!(approx(out, [0.8, 0.4, 0.2], 1e-6), "{wf:?} -> {out:?}");
        }
    }

    #[test]
    fn fully_transparent_fragment_leaves_background() {
        let frag = OitFragment::new([1.0, 1.0, 1.0], 0.0, 5.0);
        let out = composite(&[frag], WeightFunction::Equation9, BG);
        assert!(approx(out, BG, 1e-6), "{out:?}");
    }

    #[test]
    fn resolve_is_order_independent() {
        let frags = vec![
            OitFragment::new([0.9, 0.1, 0.1], 0.6, 3.0),
            OitFragment::new([0.1, 0.9, 0.1], 0.4, 8.0),
            OitFragment::new([0.1, 0.1, 0.9], 0.7, 15.0),
            OitFragment::new([0.5, 0.5, 0.0], 0.3, 40.0),
        ];
        for wf in [
            WeightFunction::Equation7,
            WeightFunction::Equation8,
            WeightFunction::Equation9,
            WeightFunction::Equation10,
            WeightFunction::Uniform,
        ] {
            let forward = composite(&frags, wf, BG);
            let mut reversed: Vec<OitFragment> = frags.clone();
            reversed.reverse();
            let backward = composite(&reversed, wf, BG);
            // A rotated permutation too.
            let mut rotated: Vec<OitFragment> = frags.clone();
            rotated.rotate_left(2);
            let rot = composite(&rotated, wf, BG);
            assert!(approx(forward, backward, 1e-5), "{wf:?} fwd/bwd differ");
            assert!(approx(forward, rot, 1e-5), "{wf:?} fwd/rot differ");
        }
    }

    #[test]
    fn revealage_is_product_of_transparencies() {
        let mut acc = OitAccumulator::new();
        acc.accumulate(OitFragment::new([1.0, 0.0, 0.0], 0.5, 2.0), WeightFunction::Uniform);
        acc.accumulate(OitFragment::new([0.0, 1.0, 0.0], 0.25, 2.0), WeightFunction::Uniform);
        // (1 - 0.5) * (1 - 0.25) = 0.375
        assert!((acc.revealage() - 0.375).abs() <= 1e-6);
        assert!((acc.coverage() - 0.625).abs() <= 1e-6);
    }

    #[test]
    fn weight_clamps_and_scales_by_alpha() {
        // Near depth saturates the raw weight to WEIGHT_MAX, then * alpha.
        let near = WeightFunction::Equation9.weight(0.0, 0.5);
        assert!((near - WEIGHT_MAX * 0.5).abs() <= 1e-3, "near = {near}");
        // Zero alpha yields zero weight regardless of depth.
        assert_eq!(WeightFunction::Equation9.weight(50.0, 0.0), 0.0);
        // Uniform ignores depth entirely.
        assert_eq!(WeightFunction::Uniform.weight(999.0, 0.42), 0.42);
    }

    #[test]
    fn nearer_fragment_dominates_the_average() {
        // Two equal-coverage fragments at different depths: the nearer one
        // should pull the resolved color toward its color under a depth weight.
        let near = OitFragment::new([1.0, 0.0, 0.0], 0.5, 2.0);
        let far = OitFragment::new([0.0, 0.0, 1.0], 0.5, 120.0);
        let out = composite(&[near, far], WeightFunction::Equation9, [0.0, 0.0, 0.0]);
        assert!(out[0] > out[2], "near red should outweigh far blue: {out:?}");
    }

    #[test]
    fn alpha_is_clamped_on_accumulation() {
        // Out-of-range alpha must not corrupt the revealage product.
        let mut acc = OitAccumulator::new();
        acc.accumulate(OitFragment::new([1.0, 1.0, 1.0], 1.5, 2.0), WeightFunction::Uniform);
        assert_eq!(acc.revealage(), 0.0);
        let out = acc.resolve(BG);
        assert!(approx(out, [1.0, 1.0, 1.0], 1e-6), "{out:?}");
    }

    #[test]
    fn ipow_matches_repeated_multiplication() {
        assert_eq!(ipow(2.0, 0), 1.0);
        assert_eq!(ipow(2.0, 1), 2.0);
        assert_eq!(ipow(3.0, 2), 9.0);
        assert_eq!(ipow(2.0, 6), 64.0);
    }
}
