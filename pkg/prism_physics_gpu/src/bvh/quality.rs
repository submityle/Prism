//! Refit quality tracking and the refit-versus-rebuild policy for a linear
//! `BVH`.
//!
//! An incremental [`cpu_refit_lbvh`](crate::bvh::cpu_refit_lbvh) keeps the tree
//! topology frozen and only re-derives node bounds, which is cheap but slowly
//! loses quality as primitives drift: leaves that the last full build grouped
//! into one subtree can wander apart, inflating the internal-node boxes that
//! share them while the overall scene extent — and therefore the root box —
//! barely moves. A broad-phase or ray query then visits ever-larger internal
//! boxes and tests more primitives than a freshly built tree would. The
//! standard real-time remedy is to refit most frames and periodically rebuild
//! from scratch; this module supplies the metric and the policy that decide
//! *when* that rebuild is worth paying for.
//!
//! # Surface Area Heuristic cost
//!
//! [`lbvh_sah_cost`] evaluates the Surface Area Heuristic (`SAH`): the expected
//! cost of a query is proportional to the probability a random ray or box hits
//! each node, which for a convex child inside its parent is the ratio of their
//! surface areas. Summing surface area over every internal node (weighted by a
//! traversal cost) and every leaf (weighted by an intersection cost), then
//! normalising by the root surface area, yields a scale- and
//! translation-invariant number that rises precisely when the tree degrades.
//! A fresh build minimises it; a stale refit inflates it.
//!
//! # Rebuild policy
//!
//! [`RefitQualityTracker`] remembers the `SAH` cost captured at the last full
//! rebuild and fires a [`RebuildDecision::Rebuild`] on either of two independent
//! triggers, matching production acceleration-structure managers:
//!
//! - **cost growth** — the current `SAH` cost exceeds the rebuild-time baseline
//!   times a factor (default [`DEFAULT_REBUILD_COST_FACTOR`]), catching sudden
//!   quality collapse from fast-moving primitives.
//! - **staleness** — a fixed number of refits have elapsed since the last
//!   rebuild (default [`DEFAULT_MAX_REFITS_BETWEEN_REBUILDS`]), bounding the slow
//!   drift that never trips the cost trigger on its own.
//!
//! The decision is deliberately host-side control flow: it chooses, between
//! frames, whether the next update runs the cheap refit or the full
//! [`cpu_build_lbvh`](crate::bvh::cpu_build_lbvh). The `CPU` cost here is the
//! golden evaluator; a resident `GPU` pipeline would compute the same sum with a
//! surface-area reduction over the node buffers and apply the identical policy.
//!
//! # Provenance
//!
//! The Surface Area Heuristic is Goldsmith and Salmon, "Automatic Creation of
//! Object Hierarchies for Ray Tracing" (IEEE CG&A 1987), as refined by
//! `MacDonald` and Booth (1990). The refit-then-periodically-rebuild strategy is
//! standard real-time practice. No Unreal Engine source or derived code.

use crate::bvh::config::Aabb;
use crate::bvh::cpu::Lbvh;
use glam::Vec3;

/// Default internal-node traversal weight in the `SAH` sum.
pub const DEFAULT_TRAVERSAL_COST: f32 = 1.0;

/// Default leaf intersection weight in the `SAH` sum.
pub const DEFAULT_INTERSECTION_COST: f32 = 1.0;

/// Default cost-growth factor past which a refit is abandoned for a rebuild.
///
/// A value of `2.0` means "rebuild once a stale tree is twice as expensive to
/// query as it was at the last rebuild".
pub const DEFAULT_REBUILD_COST_FACTOR: f32 = 2.0;

/// Default number of refits tolerated between full rebuilds.
pub const DEFAULT_MAX_REFITS_BETWEEN_REBUILDS: u32 = 60;

/// The surface area of a box: `2·(dx·dy + dy·dz + dz·dx)`.
///
/// A degenerate box whose `max` falls below its `min` on any axis clamps that
/// axis to zero extent, so the result is always non-negative; a point box has
/// zero area and a planar box keeps the area of its single non-degenerate face.
#[must_use]
pub fn surface_area(aabb: &Aabb) -> f32 {
    let d = (aabb.max - aabb.min).max(Vec3::ZERO);
    2.0 * d.x.mul_add(d.y, d.y.mul_add(d.z, d.z * d.x))
}

/// The box of an encoded node id, leaf or internal.
///
/// The caller must guarantee the tree is non-empty, so `encoded` decodes to a
/// real node.
fn node_box(tree: &Lbvh, encoded: u32) -> Aabb {
    if tree.is_leaf(encoded) {
        tree.leaf_aabb[encoded as usize - tree.num_internal]
    } else {
        tree.internal_aabb[encoded as usize]
    }
}

/// The Surface Area Heuristic cost of `tree` with the default weights.
///
/// See [`lbvh_sah_cost_weighted`] for the definition and the weight meaning.
#[must_use]
pub fn lbvh_sah_cost(tree: &Lbvh) -> f32 {
    lbvh_sah_cost_weighted(tree, DEFAULT_TRAVERSAL_COST, DEFAULT_INTERSECTION_COST)
}

/// The Surface Area Heuristic cost of `tree` with explicit weights.
///
/// The cost is
///
/// ``text
/// (c_traversal · Σ_internal SA(node) + c_intersection · Σ_leaf SA(leaf)) / SA(root)
/// ``
///
/// Normalising by the root surface area makes the value invariant to uniform
/// scaling and to translation of the whole scene, so it reflects tree
/// *shape* rather than world units: a freshly built tree minimises it and a
/// stale refit inflates it. An empty tree has no query cost and returns `0`; a
/// degenerate point root (zero surface area) cannot be normalised, so the cost
/// falls back to the intersection-weighted leaf count.
#[must_use]
pub fn lbvh_sah_cost_weighted(tree: &Lbvh, c_traversal: f32, c_intersection: f32) -> f32 {
    if tree.num_leaves == 0 {
        return 0.0;
    }
    let root_sa = surface_area(&node_box(tree, tree.root));
    if root_sa <= 0.0 {
        // A point-sized root defeats surface-area normalisation; fall back to
        // the primitive count so the cost is still a finite, monotone proxy.
        return c_intersection * tree.num_leaves as f32;
    }
    let internal_sa: f32 = tree.internal_aabb.iter().map(surface_area).sum();
    let leaf_sa: f32 = tree.leaf_aabb.iter().map(surface_area).sum();
    c_traversal.mul_add(internal_sa, c_intersection * leaf_sa) / root_sa
}

/// Whether the next update should refit the existing tree or rebuild it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebuildDecision {
    /// Quality is still acceptable; keep the topology and refit bounds.
    Refit,
    /// Quality has degraded past the policy; rebuild the tree from scratch.
    Rebuild,
}

/// Tracks refit quality against the cost captured at the last full rebuild and
/// decides when to rebuild instead of refitting again.
///
/// Drive it once per update: refit the tree, measure its [`lbvh_sah_cost`], and
/// call [`RefitQualityTracker::observe_refit`]. When it returns
/// [`RebuildDecision::Rebuild`], run a full build and report the rebuilt cost
/// through [`RefitQualityTracker::record_rebuild`] to reset the baseline and the
/// staleness counter.
#[derive(Clone, Debug)]
pub struct RefitQualityTracker {
    baseline_cost: f32,
    rebuild_cost_factor: f32,
    max_refits_between_rebuilds: u32,
    refits_since_rebuild: u32,
}

impl RefitQualityTracker {
    /// Creates a tracker seeded with the `SAH` cost of a freshly built tree.
    ///
    /// # Panics
    ///
    /// Panics if `rebuild_cost_factor` is below `1.0`, since a factor under one
    /// would demand a rebuild even when quality has not degraded, or if it is
    /// not finite.
    #[must_use]
    pub fn new(
        baseline_cost: f32,
        rebuild_cost_factor: f32,
        max_refits_between_rebuilds: u32,
    ) -> RefitQualityTracker {
        assert!(
            rebuild_cost_factor.is_finite() && rebuild_cost_factor >= 1.0,
            "rebuild_cost_factor must be finite and at least 1.0"
        );
        RefitQualityTracker {
            baseline_cost,
            rebuild_cost_factor,
            max_refits_between_rebuilds,
            refits_since_rebuild: 0,
        }
    }

    /// Creates a tracker from a freshly built tree with the default policy
    /// constants, measuring its baseline cost directly.
    #[must_use]
    pub fn from_tree(tree: &Lbvh) -> RefitQualityTracker {
        RefitQualityTracker::new(
            lbvh_sah_cost(tree),
            DEFAULT_REBUILD_COST_FACTOR,
            DEFAULT_MAX_REFITS_BETWEEN_REBUILDS,
        )
    }

    /// The `SAH` cost recorded at the last rebuild.
    #[must_use]
    pub fn baseline_cost(&self) -> f32 {
        self.baseline_cost
    }

    /// The cost-growth factor that triggers a rebuild.
    #[must_use]
    pub fn rebuild_cost_factor(&self) -> f32 {
        self.rebuild_cost_factor
    }

    /// The staleness bound: refits tolerated between rebuilds.
    #[must_use]
    pub fn max_refits_between_rebuilds(&self) -> u32 {
        self.max_refits_between_rebuilds
    }

    /// The number of refits observed since the last rebuild.
    #[must_use]
    pub fn refits_since_rebuild(&self) -> u32 {
        self.refits_since_rebuild
    }

    /// Decides, without mutating the tracker, whether `current_cost` warrants a
    /// rebuild under the cost-growth trigger alone.
    ///
    /// The staleness trigger is intentionally excluded here because it depends
    /// on the refit counter that only [`RefitQualityTracker::observe_refit`]
    /// advances; use this for a read-only "would a rebuild help?" probe.
    #[must_use]
    pub fn evaluate(&self, current_cost: f32) -> RebuildDecision {
        if self.cost_grown(current_cost) {
            RebuildDecision::Rebuild
        } else {
            RebuildDecision::Refit
        }
    }

    /// Records that one refit happened at `current_cost` and returns the policy
    /// decision for the next update, combining the cost-growth and staleness
    /// triggers.
    pub fn observe_refit(&mut self, current_cost: f32) -> RebuildDecision {
        self.refits_since_rebuild = self.refits_since_rebuild.saturating_add(1);
        if self.cost_grown(current_cost) || self.is_stale() {
            RebuildDecision::Rebuild
        } else {
            RebuildDecision::Refit
        }
    }

    /// Resets the baseline to a freshly rebuilt tree's cost and clears the
    /// staleness counter.
    pub fn record_rebuild(&mut self, new_baseline_cost: f32) {
        self.baseline_cost = new_baseline_cost;
        self.refits_since_rebuild = 0;
    }

    /// Whether the cost-growth trigger fires for `current_cost`.
    ///
    /// A non-positive baseline (an empty or point-root tree) disables this
    /// trigger, leaving only staleness, since there is no meaningful ratio to
    /// grow against.
    fn cost_grown(&self, current_cost: f32) -> bool {
        self.baseline_cost > 0.0 && current_cost > self.baseline_cost * self.rebuild_cost_factor
    }

    /// Whether the staleness trigger fires.
    fn is_stale(&self) -> bool {
        self.refits_since_rebuild >= self.max_refits_between_rebuilds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::cpu::cpu_build_lbvh;

    fn unit_cube() -> Aabb {
        Aabb::new(Vec3::ZERO, Vec3::ONE)
    }

    /// A line of unit boxes spaced two units apart along x, with small y/z
    /// jitter so Morton codes are well separated.
    fn line_boxes(n: usize) -> Vec<Aabb> {
        (0..n)
            .map(|i| {
                let c = Vec3::new(i as f32 * 2.0, (i % 3) as f32 * 0.25, (i % 2) as f32 * 0.25);
                Aabb::new(c, c + Vec3::ONE)
            })
            .collect()
    }

    #[test]
    fn surface_area_of_unit_cube_is_six() {
        assert_eq!(surface_area(&unit_cube()), 6.0);
    }

    #[test]
    fn surface_area_clamps_degenerate_box_to_zero() {
        let inverted = Aabb::new(Vec3::ONE, Vec3::ZERO);
        assert_eq!(surface_area(&inverted), 0.0);
    }

    #[test]
    fn surface_area_of_planar_box_keeps_one_face() {
        // A 2x3 box flattened on z: only the x-y face pair survives, area 2*2*3.
        let planar = Aabb::new(Vec3::ZERO, Vec3::new(2.0, 3.0, 0.0));
        assert_eq!(surface_area(&planar), 12.0);
    }

    #[test]
    fn single_leaf_cost_is_the_intersection_weight() {
        let tree = cpu_build_lbvh(&[unit_cube()]);
        // Root is the lone leaf: internal sum is zero, leaf sum equals the root,
        // so the normalised cost is exactly the intersection weight.
        assert_eq!(lbvh_sah_cost(&tree), DEFAULT_INTERSECTION_COST);
    }

    #[test]
    fn empty_tree_has_zero_cost() {
        assert_eq!(lbvh_sah_cost(&cpu_build_lbvh(&[])), 0.0);
    }

    #[test]
    fn cost_is_invariant_to_uniform_scale() {
        let base = line_boxes(16);
        let scaled: Vec<Aabb> = base
            .iter()
            .map(|b| Aabb::new(b.min * 7.0, b.max * 7.0))
            .collect();
        let c0 = lbvh_sah_cost(&cpu_build_lbvh(&base));
        let c1 = lbvh_sah_cost(&cpu_build_lbvh(&scaled));
        assert!((c0 - c1).abs() <= 1e-3 * c0, "scale changed cost: {c0} vs {c1}");
    }

    #[test]
    fn cost_is_invariant_to_translation() {
        let base = line_boxes(16);
        let shift = Vec3::new(123.0, -45.0, 67.0);
        let moved: Vec<Aabb> = base
            .iter()
            .map(|b| Aabb::new(b.min + shift, b.max + shift))
            .collect();
        let c0 = lbvh_sah_cost(&cpu_build_lbvh(&base));
        let c1 = lbvh_sah_cost(&cpu_build_lbvh(&moved));
        assert!((c0 - c1).abs() <= 1e-3 * c0, "translation changed cost: {c0} vs {c1}");
    }

    #[test]
    fn stale_refit_costs_more_than_a_rebuild() {
        use crate::bvh::refit::cpu_refit_lbvh;
        // Build over a locality-ordered line, then swap the two end boxes so the
        // occupied positions (and hence the scene extent and root box) are
        // unchanged, but the frozen topology now straddles the whole line.
        let base = line_boxes(16);
        let built = cpu_build_lbvh(&base);
        let fresh_cost = lbvh_sah_cost(&built);

        let mut swapped = base.clone();
        swapped.swap(0, 15);
        let refit = cpu_refit_lbvh(&built, &swapped);
        let refit_cost = lbvh_sah_cost(&refit);

        // Rebuilding from the swapped set regroups neighbours and recovers
        // quality, so the stale refit must cost strictly more.
        let rebuilt = cpu_build_lbvh(&swapped);
        let rebuilt_cost = lbvh_sah_cost(&rebuilt);

        assert!(
            refit_cost > fresh_cost,
            "stale refit should exceed the fresh cost: refit={refit_cost} fresh={fresh_cost}"
        );
        assert!(
            refit_cost > rebuilt_cost,
            "rebuild should beat the stale refit: refit={refit_cost} rebuilt={rebuilt_cost}"
        );
    }

    #[test]
    fn tracker_refits_while_cost_is_low() {
        let mut t = RefitQualityTracker::new(10.0, 2.0, 60);
        assert_eq!(t.observe_refit(12.0), RebuildDecision::Refit);
        assert_eq!(t.observe_refit(19.9), RebuildDecision::Refit);
        assert_eq!(t.refits_since_rebuild(), 2);
    }

    #[test]
    fn tracker_rebuilds_when_cost_doubles() {
        let mut t = RefitQualityTracker::new(10.0, 2.0, 60);
        assert_eq!(t.observe_refit(20.01), RebuildDecision::Rebuild);
    }

    #[test]
    fn tracker_rebuilds_when_stale() {
        let mut t = RefitQualityTracker::new(10.0, 2.0, 3);
        assert_eq!(t.observe_refit(10.0), RebuildDecision::Refit);
        assert_eq!(t.observe_refit(10.0), RebuildDecision::Refit);
        // Third refit hits the staleness bound even though cost never grew.
        assert_eq!(t.observe_refit(10.0), RebuildDecision::Rebuild);
    }

    #[test]
    fn record_rebuild_resets_baseline_and_counter() {
        let mut t = RefitQualityTracker::new(10.0, 2.0, 3);
        let _ = t.observe_refit(10.0);
        let _ = t.observe_refit(10.0);
        t.record_rebuild(25.0);
        assert_eq!(t.refits_since_rebuild(), 0);
        assert_eq!(t.baseline_cost(), 25.0);
        // The new baseline shifts the growth threshold up accordingly.
        assert_eq!(t.observe_refit(40.0), RebuildDecision::Refit);
        assert_eq!(t.observe_refit(50.01), RebuildDecision::Rebuild);
    }

    #[test]
    fn evaluate_ignores_staleness() {
        let mut t = RefitQualityTracker::new(10.0, 2.0, 1);
        let _ = t.observe_refit(10.0); // now stale by the observe path
        // The read-only probe only looks at cost growth, not the counter.
        assert_eq!(t.evaluate(10.0), RebuildDecision::Refit);
        assert_eq!(t.evaluate(20.01), RebuildDecision::Rebuild);
    }

    #[test]
    fn zero_baseline_disables_cost_trigger() {
        let mut t = RefitQualityTracker::new(0.0, 2.0, 5);
        // No ratio to grow against, so only staleness can fire.
        assert_eq!(t.observe_refit(1_000.0), RebuildDecision::Refit);
    }

    #[test]
    #[should_panic(expected = "rebuild_cost_factor")]
    fn new_rejects_factor_below_one() {
        let _ = RefitQualityTracker::new(10.0, 0.5, 60);
    }
}
