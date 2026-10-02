//! Broad-phase-accelerated collide-shape query: find every static target a
//! query convex overlaps and the contact manifold against each, the overlap
//! analogue of [`cast_shape_bvh`](super::shape_cast_bvh::cast_shape_bvh). This
//! is the one-against-many scene query `AAA` engines expose as Jolt
//! `NarrowPhaseQuery::CollideShape`, `PhysX` `PxScene::overlap` with contact
//! generation, and Unreal `Chaos` shape overlaps: descend an acceleration
//! structure over the scene with the query's bounds to collect candidates, then
//! build the exact convex-versus-convex manifold only on those candidates.
//!
//! # Body layout
//!
//! The input tables are body-indexed exactly like the manifold narrow phase:
//! body `0` is the query shape and bodies `1..=n` are the targets, so target
//! slot `i` lives at body index `i + 1`. The returned
//! [`CollideShapeHit::target`] is the zero-based target index.
//!
//! # Why the result matches the brute force
//!
//! The convex-versus-convex manifold is penetration-only: a separated couple
//! yields [`None`] and is dropped. The gather descends the identical
//! [`gather_boxes`] geometry a brute-force walk would, so its candidate set is a
//! conservative superset of the truly-overlapping targets; running the exact
//! manifold on that superset and dropping the [`None`] slots therefore recovers
//! exactly the brute-force hit set, re-sorted by ascending target index. The
//! `margin` only widens the conservative gather bounds and never changes which
//! manifolds survive, since a non-penetrating candidate still returns [`None`].
//!
//! # Provenance
//!
//! Broad-phase overlap gather (Jolt / `PhysX` / `Chaos` pattern), `LBVH` per
//! Karras 2012, stackless traversal per Hapala 2011, convex-versus-convex
//! manifold via Gilbert-Johnson-Keerthi 1988 distance / expanding-polytope
//! (van den Bergen 2001) with Sutherland-Hodgman face clipping. No Unreal
//! Engine source or derived code.

use glam::Vec3;

use crate::bvh::{cpu_build_lbvh, cpu_bvh_aabb_overlap, Aabb};

use super::convex_convex_manifold::{cpu_convex_convex_manifold, ConvexConvexPair};
use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::manifold::ContactManifold;

/// A target overlapped by a collide-shape query, paired with the contact
/// manifold between the query shape (body `0`) and that target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CollideShapeHit {
    /// Zero-based index of the overlapped target in the `1..` body range.
    pub target: u32,
    /// Contact manifold for the couple: shared normal (query toward target),
    /// one to four coplanar points, and their penetration depths.
    pub manifold: ContactManifold,
}

/// World axis-aligned bounds of `hull` placed at `pose`, grown by `margin` on
/// every axis. A degenerate hull with no vertices collapses to the `pose`
/// translation (then grown), so the box is always valid.
#[must_use]
pub(super) fn world_aabb(hull: &ConvexHull, pose: &ConvexPose, margin: f32) -> Aabb {
    let verts = hull.vertices();
    let (mut min, mut max) = match verts.first() {
        Some(&v) => {
            let p = pose.transform_point(v);
            (p, p)
        }
        None => (pose.translation, pose.translation),
    };
    for &v in verts.iter().skip(1) {
        let p = pose.transform_point(v);
        min = min.min(p);
        max = max.max(p);
    }
    let grow = Vec3::splat(margin.max(0.0));
    Aabb::new(min - grow, max + grow)
}

/// Builds the conservative overlap bounds the gather descends: one box per
/// target (body `1..`, no extra margin) and the query shape's box (body `0`,
/// grown by `margin`). Shared by the `CPU` [`collide_shape_bvh`] walk and the
/// `GPU`-driven query so both descend identical geometry, keeping the `GPU`
/// candidate set a provable superset of the `CPU` one.
#[must_use]
pub(super) fn gather_boxes(
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    margin: f32,
) -> (Vec<Aabb>, Aabb) {
    let target_boxes = (1..hulls.len())
        .map(|i| world_aabb(&hulls[i], &poses[i], 0.0))
        .collect();
    let query = world_aabb(&hulls[0], &poses[0], margin.max(0.0));
    (target_boxes, query)
}

/// Orders a hit list by ascending target index, the deterministic order both
/// the `CPU` and `GPU` queries report so their outputs compare slot for slot.
pub(super) fn sort_hits(hits: &mut [CollideShapeHit]) {
    hits.sort_by_key(|hit| hit.target);
}

/// Gathers the candidate target indices (into the `0`-based target space) by
/// descending a `BVH` over the targets' bounds with the query shape's grown
/// box. The result is a conservative superset of every overlapping target; the
/// `Err` arm falls back to the whole scene so correctness never depends on the
/// gather succeeding.
fn candidates(hulls: &[ConvexHull], poses: &[ConvexPose], margin: f32) -> Vec<u32> {
    let num_targets = hulls.len() - 1;
    if num_targets == 0 {
        return Vec::new();
    }
    let (target_boxes, query) = gather_boxes(hulls, poses, margin);
    let lbvh = cpu_build_lbvh(&target_boxes);
    let capacity = u32::try_from(num_targets).unwrap_or(u32::MAX);
    match cpu_bvh_aabb_overlap(&lbvh, &[query], capacity) {
        Ok(mut per_query) => per_query.pop().unwrap_or_default(),
        Err(_) => (0..capacity).collect(),
    }
}

/// Collapses the manifolds of a candidate set into the overlapping hits, in
/// ascending target order. Shared by the brute-force and `BVH` walks so both
/// drop the [`None`] slots and sort identically.
fn hits_from(candidates: &[u32], manifolds: Vec<Option<ContactManifold>>) -> Vec<CollideShapeHit> {
    let mut hits: Vec<CollideShapeHit> = candidates
        .iter()
        .zip(manifolds)
        .filter_map(|(&target, slot)| slot.map(|manifold| CollideShapeHit { target, manifold }))
        .collect();
    sort_hits(&mut hits);
    hits
}

/// Brute-force collide-shape golden: the contact manifold of the query shape
/// (body `0`) against every target (bodies `1..`) that it overlaps, in
/// ascending target order. Builds a manifold for every couple with no broad
/// phase, so it is the reference the accelerated [`collide_shape_bvh`] must
/// reproduce.
#[must_use]
pub fn collide_shape(hulls: &[ConvexHull], poses: &[ConvexPose]) -> Vec<CollideShapeHit> {
    if hulls.len() < 2 {
        return Vec::new();
    }
    let all: Vec<u32> = (0..u32::try_from(hulls.len() - 1).unwrap_or(u32::MAX)).collect();
    let pairs: Vec<ConvexConvexPair> = all
        .iter()
        .map(|&target| ConvexConvexPair::new(0, target + 1))
        .collect();
    let manifolds = cpu_convex_convex_manifold(hulls, poses, &pairs);
    hits_from(&all, manifolds)
}

/// Broad-phase-accelerated collide-shape query: the same overlapping-target
/// manifolds as [`collide_shape`], but gathered through a `BVH` over the
/// targets' bounds so only candidates near the query shape have a manifold
/// built. `margin` grows the query's gather box to catch targets within a
/// speculative distance; it never changes which manifolds survive, since the
/// penetration-only narrow phase drops any non-overlapping candidate. The
/// result equals [`collide_shape`] exactly, hit for hit.
#[must_use]
pub fn collide_shape_bvh(
    hulls: &[ConvexHull],
    poses: &[ConvexPose],
    margin: f32,
) -> Vec<CollideShapeHit> {
    if hulls.len() < 2 {
        return Vec::new();
    }
    let candidates = candidates(hulls, poses, margin);
    let pairs: Vec<ConvexConvexPair> = candidates
        .iter()
        .map(|&target| ConvexConvexPair::new(0, target + 1))
        .collect();
    let manifolds = cpu_convex_convex_manifold(hulls, poses, &pairs);
    hits_from(&candidates, manifolds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    /// A unit box core, reused across bodies.
    fn unit_box() -> ConvexHull {
        ConvexHull::from_box(Vec3::splat(0.5))
    }

    /// A pose at `center` with no rotation.
    fn at(center: Vec3) -> ConvexPose {
        ConvexPose::new(center, Quat::IDENTITY)
    }

    #[test]
    fn bvh_collide_matches_brute_force_across_a_mixed_scene() {
        // A query box overlapping two of four targets; the other two are far.
        let hulls = [unit_box(), unit_box(), unit_box(), unit_box(), unit_box()];
        let poses = [
            at(Vec3::new(0.0, 0.0, 0.0)),
            at(Vec3::new(0.6, 0.03, 0.0)),   // overlaps
            at(Vec3::new(-0.6, 0.0, 0.05)),  // overlaps
            at(Vec3::new(40.0, 0.0, 0.0)),   // far
            at(Vec3::new(0.0, 40.0, 0.0)),   // far
        ];
        let brute = collide_shape(&hulls, &poses);
        let bvh = collide_shape_bvh(&hulls, &poses, 0.0);
        assert_eq!(brute, bvh, "BVH collide must equal brute force exactly");
        assert_eq!(brute.len(), 2, "exactly two targets overlap the query");
        assert_eq!(brute[0].target, 0);
        assert_eq!(brute[1].target, 1);
    }

    #[test]
    fn bvh_collide_reports_no_hits_when_nothing_overlaps() {
        let hulls = [unit_box(), unit_box(), unit_box()];
        let poses = [
            at(Vec3::ZERO),
            at(Vec3::new(10.0, 0.0, 0.0)),
            at(Vec3::new(0.0, -10.0, 0.0)),
        ];
        assert!(collide_shape(&hulls, &poses).is_empty());
        assert_eq!(
            collide_shape_bvh(&hulls, &poses, 0.0),
            collide_shape(&hulls, &poses)
        );
    }

    #[test]
    fn bvh_collide_handles_an_empty_scene() {
        let hulls = [unit_box()];
        let poses = [at(Vec3::ZERO)];
        assert!(collide_shape(&hulls, &poses).is_empty());
        assert!(collide_shape_bvh(&hulls, &poses, 0.0).is_empty());
    }

    #[test]
    fn bvh_collide_prunes_a_large_scene_and_matches_brute_force() {
        // One query box at the origin; a dense cluster of overlapping boxes plus
        // a far wall the gather must prune. Distinct tiny offsets avoid ties.
        let mut hulls = vec![unit_box()];
        let mut poses = vec![at(Vec3::ZERO)];
        for k in 0..5 {
            let d = 0.3 + 0.02 * (k as f32);
            hulls.push(unit_box());
            poses.push(at(Vec3::new(d, 0.01 * (k as f32), 0.0)));
        }
        for k in 0..40 {
            hulls.push(unit_box());
            poses.push(at(Vec3::new(5.0 + 1.5 * (k as f32), 80.0, 0.0)));
        }
        let brute = collide_shape(&hulls, &poses);
        let bvh = collide_shape_bvh(&hulls, &poses, 0.0);
        assert_eq!(brute, bvh, "BVH collide must equal brute force on a large scene");
        assert_eq!(brute.len(), 5, "only the five clustered targets overlap");
    }
}
