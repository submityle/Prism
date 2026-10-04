//! Adaptive transparency (Salvi & Vaidyanathan 2011).
//!
//! Adaptive transparency keeps a *bounded* approximation of a pixel's exact
//! visibility function `V(z) = prod_{d_i <= z} (1 - a_i)`, the running
//! fraction of background light that survives to depth `z`. The exact
//! depth-sorted `A-buffer` in [`sorted_oit`](super::sorted_oit) stores one node
//! per fragment and is the ground truth; weighted (`WBOIT`) and moment resolves
//! compress that function into a handful of fixed statistics. Adaptive
//! transparency sits between the two: it stores an explicit, piecewise-constant
//! visibility curve but caps it at `max_nodes` control points, discarding the
//! control point whose removal adds the least area error.
//!
//! The curve is order-independent: inserting the same fragments in any order
//! yields the same uncompressed curve, and with `max_nodes >= fragment_count`
//! no compression happens, so the resolve reproduces the exact `A-buffer`
//! answer bit-for-bit. All arithmetic is multiply / compare, so the result is
//! reproducible across backends with no transcendental calls.
//!
//! ```text
//! insert(d, a):  f = 1 - a
//!   V(z) *= f  for every control point with z >= d   (this fragment occludes them)
//!   new point at d has value V(d-) * f                (V(d-) = survivor just in front)
//! compress (when over budget): drop interior point i minimising
//!   cost(i) = (z[i+1] - z[i]) * (V[i-1] - V[i])       // the rectangle it bounds
//! ```

use alloc::vec::Vec;

use super::weighted_oit::OitFragment;

/// One control point of the piecewise-constant visibility curve.
///
/// `vis` is the surviving transmittance `V(depth)` *just after* the fragment at
/// `depth`, i.e. the product of `1 - a_i` over every fragment at or in front of
/// `depth`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisibilityNode {
    /// View-space depth of the control point.
    pub depth: f32,
    /// Surviving transmittance just behind this depth.
    pub vis: f32,
}

/// A bounded-memory approximation of a pixel's visibility function.
///
/// Fragments may be inserted in any order. The curve never exceeds `max_nodes`
/// control points; inserting past the budget drops the least-significant
/// interior point.
#[derive(Clone, Debug)]
pub struct AdaptiveVisibility {
    nodes: Vec<VisibilityNode>,
    max_nodes: usize,
}

impl AdaptiveVisibility {
    /// Creates an empty curve capped at `max_nodes` control points.
    ///
    /// `max_nodes` is clamped up to 2 so the first and last points (which carry
    /// the near-plane and total transmittance) are always retained.
    #[must_use]
    pub fn new(max_nodes: usize) -> Self {
        Self {
            nodes: Vec::new(),
            max_nodes: if max_nodes < 2 { 2 } else { max_nodes },
        }
    }

    /// Control points, sorted by ascending depth.
    #[must_use]
    pub fn nodes(&self) -> &[VisibilityNode] {
        &self.nodes
    }

    /// Number of control points currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// `true` when no fragments have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Inserts a fragment of opacity `alpha` at `depth`, updating the curve.
    ///
    /// Every control point at or behind `depth` is attenuated by `1 - alpha`,
    /// and a new point is added carrying the transmittance just in front of
    /// `depth` times `1 - alpha`. If the budget is exceeded the curve is
    /// compressed back down by one point.
    pub fn insert(&mut self, depth: f32, alpha: f32) {
        let f = 1.0 - alpha.clamp(0.0, 1.0);

        // Walk to the first point whose depth is >= the new fragment, tracking
        // the surviving transmittance just in front of it.
        let mut vis_before = 1.0_f32;
        let mut idx = self.nodes.len();
        for (i, node) in self.nodes.iter().enumerate() {
            if node.depth >= depth {
                idx = i;
                break;
            }
            vis_before = node.vis;
        }

        // Everything at or behind the new fragment is further occluded by it.
        for node in &mut self.nodes[idx..] {
            node.vis *= f;
        }

        self.nodes.insert(
            idx,
            VisibilityNode {
                depth,
                vis: vis_before * f,
            },
        );

        if self.nodes.len() > self.max_nodes {
            self.compress();
        }
    }

    /// Drops the interior control point whose removal adds the least area error.
    ///
    /// The error of removing point `i` is the rectangle `(z[i+1] - z[i]) *
    /// (V[i-1] - V[i])`: once `i` is gone, the segment it covered inherits the
    /// higher transmittance `V[i-1]` of its predecessor. The first and last
    /// points are never removed.
    fn compress(&mut self) {
        let n = self.nodes.len();
        if n <= 2 {
            return;
        }

        let mut best_idx = 1;
        let mut best_cost = f32::INFINITY;
        for i in 1..n - 1 {
            let width = self.nodes[i + 1].depth - self.nodes[i].depth;
            let drop = self.nodes[i - 1].vis - self.nodes[i].vis;
            let cost = width * drop;
            if cost < best_cost {
                best_cost = cost;
                best_idx = i;
            }
        }
        self.nodes.remove(best_idx);
    }

    /// Surviving transmittance at `query`: the value of the last control point
    /// at or in front of `query`, or `1.0` before the first point.
    #[must_use]
    pub fn transmittance_at(&self, query: f32) -> f32 {
        let mut vis = 1.0_f32;
        for node in &self.nodes {
            if node.depth <= query {
                vis = node.vis;
            } else {
                break;
            }
        }
        vis
    }

    /// Surviving transmittance *strictly in front of* `query`: the value of the
    /// last control point with depth `< query`, or `1.0` if none precede it.
    ///
    /// This is the fraction of light reaching a fragment at `query` from the
    /// camera and is what a resolve multiplies the fragment's own colour by.
    #[must_use]
    pub fn transmittance_before(&self, query: f32) -> f32 {
        let mut vis = 1.0_f32;
        for node in &self.nodes {
            if node.depth < query {
                vis = node.vis;
            } else {
                break;
            }
        }
        vis
    }

    /// Total surviving background transmittance after every fragment.
    #[must_use]
    pub fn total_transmittance(&self) -> f32 {
        self.nodes.last().map_or(1.0, |node| node.vis)
    }
}

/// Resolves transparent `fragments` over `background` with adaptive
/// transparency, capping the visibility curve at `max_nodes` control points.
///
/// Each fragment contributes `T_i * a_i * color_i`, where `T_i` is the
/// surviving transmittance just in front of it, and the background contributes
/// the total surviving transmittance. The result is order-independent, and with
/// `max_nodes >= fragments.len()` it equals the exact depth-sorted resolve.
#[must_use]
pub fn resolve(fragments: &[OitFragment], background: [f32; 3], max_nodes: usize) -> [f32; 3] {
    let mut curve = AdaptiveVisibility::new(max_nodes);
    for fragment in fragments {
        curve.insert(fragment.view_depth, fragment.alpha);
    }

    let mut out = [0.0_f32; 3];
    for fragment in fragments {
        let transmittance = curve.transmittance_before(fragment.view_depth);
        let alpha = fragment.alpha.clamp(0.0, 1.0);
        let contribution = transmittance * alpha;
        out[0] += contribution * fragment.color[0];
        out[1] += contribution * fragment.color[1];
        out[2] += contribution * fragment.color[2];
    }

    let survivor = curve.total_transmittance();
    out[0] += survivor * background[0];
    out[1] += survivor * background[1];
    out[2] += survivor * background[2];
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transparency::sorted_oit::composite_sorted;
    use alloc::vec;
    use alloc::vec::Vec;

    fn frag(color: [f32; 3], alpha: f32, depth: f32) -> OitFragment {
        OitFragment {
            color,
            alpha,
            view_depth: depth,
        }
    }

    fn approx(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (a[0] - b[0]).abs() <= eps && (a[1] - b[1]).abs() <= eps && (a[2] - b[2]).abs() <= eps
    }

    #[test]
    fn empty_curve_passes_background_through() {
        let bg = [0.2, 0.4, 0.6];
        assert_eq!(resolve(&[], bg, 8), bg);
        let curve = AdaptiveVisibility::new(8);
        assert!(curve.is_empty());
        assert_eq!(curve.total_transmittance(), 1.0);
        assert_eq!(curve.transmittance_at(0.5), 1.0);
    }

    #[test]
    fn generous_budget_matches_exact_abuffer() {
        let frags = vec![
            frag([1.0, 0.0, 0.0], 0.5, 3.0),
            frag([0.0, 1.0, 0.0], 0.25, 1.0),
            frag([0.0, 0.0, 1.0], 0.75, 2.0),
            frag([1.0, 1.0, 0.0], 0.1, 4.0),
        ];
        let bg = [0.1, 0.1, 0.1];
        let exact = composite_sorted(&frags, bg);
        let got = resolve(&frags, bg, frags.len() + 1);
        assert!(approx(got, exact, 1e-6), "{got:?} vs {exact:?}");
    }

    #[test]
    fn resolve_is_order_independent_without_compression() {
        let a = vec![
            frag([0.9, 0.1, 0.2], 0.3, 5.0),
            frag([0.1, 0.8, 0.3], 0.6, 2.0),
            frag([0.2, 0.2, 0.9], 0.4, 8.0),
        ];
        let b = vec![a[2], a[0], a[1]];
        let bg = [0.3, 0.3, 0.3];
        let ra = resolve(&a, bg, 16);
        let rb = resolve(&b, bg, 16);
        assert!(approx(ra, rb, 1e-6));
    }

    #[test]
    fn curve_is_monotone_decreasing() {
        let mut curve = AdaptiveVisibility::new(16);
        for i in 0..10 {
            curve.insert(i as f32, 0.2);
        }
        let mut prev = f32::INFINITY;
        for node in curve.nodes() {
            assert!(node.vis <= prev + 1e-7, "vis must not increase with depth");
            prev = node.vis;
        }
        assert!(curve.total_transmittance() > 0.0);
    }

    #[test]
    fn node_count_stays_within_budget() {
        let mut curve = AdaptiveVisibility::new(4);
        for i in 0..32 {
            curve.insert(i as f32 * 0.5, 0.15);
        }
        assert!(curve.len() <= 4, "len {} exceeds budget", curve.len());
    }

    #[test]
    fn more_nodes_never_increase_error() {
        // Error against the exact resolve should be non-increasing as the budget
        // grows, and reach ~0 once the budget covers every fragment.
        let mut frags = Vec::new();
        for i in 0..12 {
            let t = i as f32;
            frags.push(frag([0.6, 0.3, 0.1 + t * 0.02], 0.2, t));
        }
        let bg = [0.05, 0.05, 0.05];
        let exact = composite_sorted(&frags, bg);

        let err = |budget: usize| -> f32 {
            let got = resolve(&frags, bg, budget);
            (got[0] - exact[0]).abs() + (got[1] - exact[1]).abs() + (got[2] - exact[2]).abs()
        };

        let coarse = err(3);
        let mid = err(6);
        let fine = err(frags.len() + 1);
        assert!(mid <= coarse + 1e-6, "mid {mid} > coarse {coarse}");
        assert!(fine <= mid + 1e-6, "fine {fine} > mid {mid}");
        assert!(fine <= 1e-6, "full budget should be exact, got {fine}");
    }

    #[test]
    fn transmittance_queries_track_fragments() {
        let mut curve = AdaptiveVisibility::new(16);
        curve.insert(2.0, 0.5); // vis 0.5 at depth 2
        curve.insert(4.0, 0.5); // vis 0.25 at depth 4
        assert_eq!(curve.transmittance_at(1.0), 1.0);
        assert_eq!(curve.transmittance_at(2.0), 0.5);
        assert_eq!(curve.transmittance_at(3.0), 0.5);
        assert_eq!(curve.transmittance_at(5.0), 0.25);
        // Strictly-in-front query excludes the fragment exactly at the depth.
        assert_eq!(curve.transmittance_before(2.0), 1.0);
        assert_eq!(curve.transmittance_before(4.0), 0.5);
    }

    #[test]
    fn insertion_order_does_not_change_uncompressed_curve() {
        let mut forward = AdaptiveVisibility::new(16);
        for d in [1.0, 2.0, 3.0, 4.0] {
            forward.insert(d, 0.3);
        }
        let mut shuffled = AdaptiveVisibility::new(16);
        for d in [3.0, 1.0, 4.0, 2.0] {
            shuffled.insert(d, 0.3);
        }
        assert_eq!(forward.nodes(), shuffled.nodes());
    }
}
