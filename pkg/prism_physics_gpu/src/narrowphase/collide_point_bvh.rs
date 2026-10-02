//! Broad-phase-accelerated collide-point scene query: find every static target
//! whose convex volume contains a world-space query point. This is the
//! one-against-many containment query `AAA` engines expose as Jolt
//! `NarrowPhaseQuery::CollidePoint`, `PhysX` `PxScene::overlap` with a
//! zero-extent probe, and Unreal `Chaos` point overlaps.
//!
//! # A point test is a still, zero-radius sweep
//!
//! A point-containment test is exactly a shape cast of a zero-radius point core
//! that never moves: the core carries no convex radius, sits at the query
//! point, and has zero displacement over the unit substep. Conservative
//! advancement reports an initially-overlapping couple at substep time `0`, so a
//! target contains the point precisely when the still point core "hits" it at
//! time zero and is otherwise missed. This module is therefore a thin,
//! semantically-clearer façade over the proven
//! [`cast_shape_bvh`](super::shape_cast_bvh::cast_shape_bvh) path — it reuses
//! that path's `BVH` gather and conservative-advancement overlap test verbatim
//! and only discards the (degenerate, always-zero) travel distance, keeping just
//! the containing target index. The `GPU` twin
//! [`GpuSceneCollidePoint`](super::collide_point_bvh_gpu::GpuSceneCollidePoint)
//! reuses the device shape-cast path the same way.
//!
//! # Containment convention
//!
//! A target counts as containing the point when the point lies in the target's
//! closed convex volume (a point strictly outside is never a hit; a point on the
//! exact surface is boundary-degenerate and left to the underlying `GJK` overlap
//! test). The containing targets are reported in ascending target index, the
//! deterministic order the still sweep's zero-distance ties resolve to.
//!
//! # Provenance
//!
//! Broad-phase overlap gather (Jolt / `PhysX` / `Chaos` pattern), `LBVH` per
//! Karras 2012, stackless traversal per Hapala 2011, conservative advancement
//! per Mirtich 2000 over a Gilbert-Johnson-Keerthi distance walk (van den
//! Bergen, 2004). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};

use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::shape_cast::{cast_shape_all, RoundedConvex, ShapeCastHit};
use super::shape_cast_bvh::cast_shape_all_bvh;

/// A world-space query point for a collide-point containment query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScenePoint {
    /// World position tested for containment against every target.
    pub position: Vec3,
}

impl ScenePoint {
    /// Builds a query point at `position`.
    #[must_use]
    pub fn new(position: Vec3) -> ScenePoint {
        ScenePoint { position }
    }

    /// The point core's pose: an identity-oriented point at the query position.
    #[must_use]
    pub(super) fn pose(&self) -> ConvexPose {
        ConvexPose::new(self.position, Quat::IDENTITY)
    }
}

/// A target whose convex volume contains a collide-point query point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CollidePointHit {
    /// Index of the containing target in the `targets` slice handed to the query.
    pub target: u32,
}

/// Rewrites a shape-cast hit (a still, zero-distance overlap) into a
/// collide-point hit by keeping only the containing target index, shared by the
/// `CPU` queries here and the `GPU` twin so both map the result identically.
#[must_use]
pub(super) fn collide_point_hit(hit: ShapeCastHit) -> CollidePointHit {
    CollidePointHit { target: hit.target }
}

/// Builds the still, zero-radius point core that represents the query point,
/// borrowing `point_hull` (a single-vertex [`ConvexHull::from_point`] hull the
/// caller owns).
#[must_use]
fn point_shape<'a>(point_hull: &'a ConvexHull, point: &ScenePoint) -> RoundedConvex<'a> {
    RoundedConvex::still(point_hull, point.pose(), 0.0)
}

/// Brute-force collide-point golden: every target whose convex volume contains
/// `point`, in ascending target index. Tests the still point core against every
/// target with no broad phase, so it is the reference the accelerated
/// [`collide_point_bvh`] must reproduce.
#[must_use]
pub fn collide_point(targets: &[RoundedConvex], point: &ScenePoint) -> Vec<CollidePointHit> {
    let hull = ConvexHull::from_point();
    let shape = point_shape(&hull, point);
    cast_shape_all(&shape, targets, 1.0, 0.0)
        .into_iter()
        .map(collide_point_hit)
        .collect()
}

/// Broad-phase-accelerated form of [`collide_point`]: the same containing
/// targets, but gathered through a `BVH` over the targets' bounds so only
/// candidates whose bounds enclose the point run the exact overlap test. The
/// result equals [`collide_point`] exactly, target for target.
#[must_use]
pub fn collide_point_bvh(targets: &[RoundedConvex], point: &ScenePoint) -> Vec<CollidePointHit> {
    let hull = ConvexHull::from_point();
    let shape = point_shape(&hull, point);
    cast_shape_all_bvh(&shape, targets, 1.0, 0.0)
        .into_iter()
        .map(collide_point_hit)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit box core, reused across targets.
    fn unit_box() -> ConvexHull {
        ConvexHull::from_box(Vec3::splat(0.5))
    }

    /// A stationary target: a zero-radius box centred at `center`.
    fn target_at(hull: &ConvexHull, center: Vec3) -> RoundedConvex<'_> {
        RoundedConvex::still(hull, ConvexPose::new(center, Quat::IDENTITY), 0.0)
    }

    #[test]
    fn collide_point_bvh_matches_brute_force_single_containment() {
        // Three disjoint boxes; the point sits inside the middle one only.
        let h = [unit_box(), unit_box(), unit_box()];
        let targets = [
            target_at(&h[0], Vec3::new(0.0, 0.0, 0.0)),
            target_at(&h[1], Vec3::new(3.0, 0.0, 0.0)),
            target_at(&h[2], Vec3::new(6.0, 0.0, 0.0)),
        ];
        let point = ScenePoint::new(Vec3::new(3.1, 0.05, -0.1));
        let brute = collide_point(&targets, &point);
        let bvh = collide_point_bvh(&targets, &point);
        assert_eq!(brute, bvh, "BVH collide-point must equal brute force exactly");
        assert_eq!(brute.len(), 1, "the point lies inside exactly one box");
        assert_eq!(brute[0].target, 1, "the middle box contains the point");
    }

    #[test]
    fn collide_point_reports_no_hit_when_the_point_is_outside_every_target() {
        let h = [unit_box(), unit_box()];
        let targets = [
            target_at(&h[0], Vec3::new(0.0, 0.0, 0.0)),
            target_at(&h[1], Vec3::new(10.0, 0.0, 0.0)),
        ];
        let point = ScenePoint::new(Vec3::new(5.0, 5.0, 5.0));
        assert!(collide_point(&targets, &point).is_empty());
        assert_eq!(
            collide_point_bvh(&targets, &point),
            collide_point(&targets, &point)
        );
    }

    #[test]
    fn collide_point_reports_every_overlapping_target_containing_the_point() {
        // Two boxes overlapping near the origin plus a far one; the point sits in
        // the shared overlap region of the first two.
        let h = [unit_box(), unit_box(), unit_box()];
        let targets = [
            target_at(&h[0], Vec3::new(0.0, 0.0, 0.0)),
            target_at(&h[1], Vec3::new(0.4, 0.0, 0.0)),
            target_at(&h[2], Vec3::new(20.0, 0.0, 0.0)),
        ];
        let point = ScenePoint::new(Vec3::new(0.2, 0.0, 0.0));
        let brute = collide_point(&targets, &point);
        let bvh = collide_point_bvh(&targets, &point);
        assert_eq!(brute, bvh, "BVH collide-point must equal brute force exactly");
        assert_eq!(brute.len(), 2, "both overlapping boxes contain the point");
        assert_eq!(brute[0].target, 0, "containing targets are ascending");
        assert_eq!(brute[1].target, 1);
    }

    #[test]
    fn collide_point_bvh_prunes_a_large_scene_and_matches_brute_force() {
        // One box containing the point amid a far wall the gather must prune.
        let mut hulls = vec![unit_box()];
        for _ in 0..40 {
            hulls.push(unit_box());
        }
        let mut targets = vec![target_at(&hulls[0], Vec3::new(0.0, 0.0, 0.0))];
        for k in 0..40 {
            targets.push(target_at(&hulls[k + 1], Vec3::new(1.0 + 1.5 * (k as f32), 60.0, 0.0)));
        }
        let point = ScenePoint::new(Vec3::new(0.1, -0.1, 0.2));
        let brute = collide_point(&targets, &point);
        let bvh = collide_point_bvh(&targets, &point);
        assert_eq!(brute, bvh, "BVH collide-point must equal brute force on a large scene");
        assert_eq!(brute.len(), 1, "only the origin box contains the point");
        assert_eq!(brute[0].target, 0);
    }
}
