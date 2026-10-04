//! Exact depth-sorted transparency resolve (the `A-buffer` ground truth).
//!
//! The sorted path composites transparent fragments with the Porter–Duff
//! *over* operator in depth order, which is the physically correct result for
//! non-refractive blended surfaces and the reference both approximate
//! order-independent resolves are measured against:
//!
//! - [`weighted_oit`](super::weighted_oit) reconstructs an order-independent
//!   *approximation* of this result from two moments; and
//! - a moment-based resolve reconstructs a closer approximation from several.
//!
//! This module owns the exact answer. It collects every fragment for a pixel,
//! sorts front-to-back, and folds them with a running transmittance:
//!
//! ```text
//! out = 0 ; T = 1
//! for fragment in ascending view-depth:
//!     out += T * a * color      // premultiplied "over", front-to-back
//!     T   *= 1 - a
//! out += T * background
//! ```
//!
//! Front-to-back folding with premultiplied color is algebraically identical to
//! the familiar back-to-front *over*, but it keeps a single running
//! transmittance `T = prod(1 - a_i)` that doubles as the surviving background
//! fraction (the same quantity `weighted_oit` calls *revealage*), so the two
//! resolvers can be cross-checked exactly on coverage.
//!
//! The resolve sorts its inputs, so the result does not depend on the order
//! fragments were submitted in; it depends only on their depths. Ties keep
//! their submitted order (a stable comparison on equal depths), matching how a
//! real `A-buffer` would resolve coplanar fragments. All arithmetic is add/
//! multiply/compare, so the reference is bit-reproducible across backends.

use alloc::vec::Vec;

use super::weighted_oit::OitFragment;

/// Collects transparent fragments for a single pixel and resolves them exactly.
///
/// Fragments may be pushed in any order; [`resolve`](Self::resolve) sorts them
/// front-to-back before compositing.
#[derive(Clone, Debug, Default)]
pub struct SortedResolve {
    fragments: Vec<OitFragment>,
}

impl SortedResolve {
    /// Creates an empty resolver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fragments: Vec::new(),
        }
    }

    /// Adds a fragment to the pixel's list.
    pub fn push(&mut self, fragment: OitFragment) {
        self.fragments.push(fragment);
    }

    /// Number of fragments collected so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fragments.len()
    }

    /// `true` when no fragments have been collected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }

    /// Composites the collected fragments over `background` with the exact
    /// depth-sorted *over* operator.
    ///
    /// With no fragments this returns `background` unchanged.
    #[must_use]
    pub fn resolve(&self, background: [f32; 3]) -> [f32; 3] {
        let mut order: Vec<usize> = (0..self.fragments.len()).collect();
        // Stable sort on view depth: nearer fragments composite first. Equal
        // depths keep submitted order so coplanar fragments resolve as an
        // `A-buffer` would. `total_cmp` gives a deterministic total order even
        // for NaN/-0.0 depths.
        order.sort_by(|&i, &j| {
            self.fragments[i]
                .view_depth
                .total_cmp(&self.fragments[j].view_depth)
        });

        let mut out = [0.0_f32; 3];
        let mut transmittance = 1.0_f32;
        for &idx in &order {
            let frag = self.fragments[idx];
            let alpha = frag.alpha.clamp(0.0, 1.0);
            let contribution = transmittance * alpha;
            out[0] += contribution * frag.color[0];
            out[1] += contribution * frag.color[1];
            out[2] += contribution * frag.color[2];
            transmittance *= 1.0 - alpha;
        }
        out[0] += transmittance * background[0];
        out[1] += transmittance * background[1];
        out[2] += transmittance * background[2];
        out
    }

    /// The surviving background fraction `prod(1 - a_i)` after resolve.
    ///
    /// This equals the *revealage* an order-independent resolve accumulates, so
    /// it is the exact value those approximations are checked against. It does
    /// not depend on fragment order.
    #[must_use]
    pub fn revealage(&self) -> f32 {
        let mut transmittance = 1.0_f32;
        for frag in &self.fragments {
            transmittance *= 1.0 - frag.alpha.clamp(0.0, 1.0);
        }
        transmittance
    }
}

/// Composites a slice of fragments exactly in a single call.
///
/// Equivalent to pushing every fragment into a [`SortedResolve`] and resolving.
/// The result depends only on fragment depths, not submission order.
#[must_use]
pub fn composite_sorted(fragments: &[OitFragment], background: [f32; 3]) -> [f32; 3] {
    let mut resolve = SortedResolve::new();
    for &fragment in fragments {
        resolve.push(fragment);
    }
    resolve.resolve(background)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use crate::transparency::weighted_oit::{self, WeightFunction};

    const BG: [f32; 3] = [0.1, 0.2, 0.3];

    fn approx(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
    }

    #[test]
    fn empty_returns_background() {
        let r = SortedResolve::new();
        assert!(r.is_empty());
        assert_eq!(r.resolve(BG), BG);
        assert_eq!(r.revealage(), 1.0);
    }

    #[test]
    fn single_opaque_returns_its_color() {
        let out = composite_sorted(&[OitFragment::new([0.8, 0.4, 0.2], 1.0, 7.0)], BG);
        assert!(approx(out, [0.8, 0.4, 0.2], 1e-6), "{out:?}");
    }

    #[test]
    fn near_opaque_occludes_far() {
        // A near opaque red fully hides a far blue and the background.
        let near = OitFragment::new([1.0, 0.0, 0.0], 1.0, 2.0);
        let far = OitFragment::new([0.0, 0.0, 1.0], 1.0, 50.0);
        let out = composite_sorted(&[far, near], BG);
        assert!(approx(out, [1.0, 0.0, 0.0], 1e-6), "{out:?}");
    }

    #[test]
    fn matches_hand_computed_over() {
        // Near 50% white over far 50% black over grey background.
        let near = OitFragment::new([1.0, 1.0, 1.0], 0.5, 1.0);
        let far = OitFragment::new([0.0, 0.0, 0.0], 0.5, 2.0);
        let bg = [0.4, 0.4, 0.4];
        // out = 0.5*1 + 0.5*(0.5*0 + 0.5*0.4) = 0.5 + 0.5*0.2 = 0.6
        let out = composite_sorted(&[near, far], bg);
        assert!(approx(out, [0.6, 0.6, 0.6], 1e-6), "{out:?}");
    }

    #[test]
    fn resolve_sorts_so_input_order_does_not_matter() {
        let frags = vec![
            OitFragment::new([0.9, 0.1, 0.1], 0.6, 3.0),
            OitFragment::new([0.1, 0.9, 0.1], 0.4, 8.0),
            OitFragment::new([0.1, 0.1, 0.9], 0.7, 1.5),
        ];
        let forward = composite_sorted(&frags, BG);
        let mut shuffled: Vec<OitFragment> = frags.clone();
        shuffled.rotate_left(1);
        let a = composite_sorted(&shuffled, BG);
        shuffled.reverse();
        let b = composite_sorted(&shuffled, BG);
        assert!(approx(forward, a, 1e-6), "rotate changed result");
        assert!(approx(forward, b, 1e-6), "reverse changed result");
    }

    #[test]
    fn depth_order_changes_the_exact_result() {
        // Unlike the OIT approximations, swapping depths of differently colored
        // semi-transparent fragments changes the exact composite.
        let a = OitFragment::new([1.0, 0.0, 0.0], 0.5, 1.0);
        let b = OitFragment::new([0.0, 0.0, 1.0], 0.5, 2.0);
        let red_front = composite_sorted(&[a, b], [0.0; 3]);
        let blue_front = composite_sorted(
            &[
                OitFragment::new([1.0, 0.0, 0.0], 0.5, 2.0),
                OitFragment::new([0.0, 0.0, 1.0], 0.5, 1.0),
            ],
            [0.0; 3],
        );
        assert!(red_front[0] > blue_front[0], "red should lead when in front");
        assert!(blue_front[2] > red_front[2], "blue should lead when in front");
    }

    #[test]
    fn revealage_matches_weighted_oit_exactly() {
        // The exact revealage is the quantity the weighted-blended resolve
        // accumulates; they must agree bit-for-bit on coverage.
        let frags = [
            OitFragment::new([0.9, 0.1, 0.1], 0.6, 3.0),
            OitFragment::new([0.1, 0.9, 0.1], 0.4, 8.0),
            OitFragment::new([0.1, 0.1, 0.9], 0.7, 15.0),
        ];
        let exact = {
            let mut r = SortedResolve::new();
            for &f in &frags {
                r.push(f);
            }
            r.revealage()
        };
        let mut acc = weighted_oit::OitAccumulator::new();
        for &f in &frags {
            acc.accumulate(f, WeightFunction::Equation9);
        }
        assert_eq!(exact, acc.revealage());
    }

    #[test]
    fn weighted_oit_approximates_exact_within_tolerance() {
        // For low-overlap fragments the cheap approximation should land close to
        // the exact composite. This is a sanity bound, not an equality.
        let frags = [
            OitFragment::new([0.8, 0.2, 0.1], 0.3, 4.0),
            OitFragment::new([0.1, 0.7, 0.2], 0.25, 9.0),
        ];
        let exact = composite_sorted(&frags, BG);
        let approxd = weighted_oit::composite(&frags, WeightFunction::Equation9, BG);
        assert!(
            approx(exact, approxd, 0.2),
            "exact {exact:?} vs approx {approxd:?}"
        );
    }

    #[test]
    fn alpha_is_clamped() {
        let out = composite_sorted(&[OitFragment::new([0.2, 0.4, 0.6], 2.0, 1.0)], BG);
        assert!(approx(out, [0.2, 0.4, 0.6], 1e-6), "{out:?}");
    }
}
