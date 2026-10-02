//! Broad-phase-accelerated ray-cast scene query: find the nearest static
//! target a ray pierces (or every target along it) and the exact entry point
//! and surface normal. This is the one-against-many ray query `AAA` engines
//! expose as Jolt `NarrowPhaseQuery::CastRay`, `PhysX` `PxScene::raycast`, and
//! Unreal `Chaos` ray casts.
//!
//! # A ray is a swept point
//!
//! A ray cast is exactly a shape cast of a zero-radius point core swept across
//! the ray's displacement: the point core carries no convex radius, starts at
//! the ray origin, and moves `direction * max_distance` over the unit substep.
//! This module is therefore a thin, semantically-clearer façade over the proven
//! [`cast_shape_bvh`](super::shape_cast_bvh::cast_shape_bvh) path — it reuses
//! that path's `BVH` gather and conservative-advancement sweep verbatim and only
//! rewrites the result from a substep fraction into a travelled distance. The
//! `GPU` twin [`GpuSceneRayCast`](super::ray_cast_bvh_gpu::GpuSceneRayCast)
//! reuses the device shape-cast path the same way.
//!
//! # Distance convention
//!
//! [`SceneRay::direction`] must be a unit vector. The impact is reported as a
//! travelled [`RayCastHit::distance`] in `[0, max_distance]` (the substep
//! fraction scaled by `max_distance`), with the world entry
//! [`RayCastHit::point`] and the surface [`RayCastHit::normal`] pointing back
//! toward the ray origin, matching the shape-cast normal convention.
//!
//! # Provenance
//!
//! Broad-phase ray/sweep gather (Jolt / `PhysX` / `Chaos` pattern), `LBVH` per
//! Karras 2012, stackless traversal per Hapala 2011, conservative advancement
//! per Mirtich 2000 over a Gilbert-Johnson-Keerthi distance walk (van den
//! Bergen, 2004). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};

use super::body_motion::BodyMotion;
use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::shape_cast::{cast_shape, cast_shape_all, RoundedConvex, ShapeCastHit};
use super::shape_cast_bvh::{cast_shape_all_bvh, cast_shape_bvh};

/// A ray in world space: an origin, a unit direction, and a maximum travel
/// distance. The cast reaches targets within `max_distance` of the origin along
/// `direction` and no farther.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneRay {
    /// World origin the ray is cast from.
    pub origin: Vec3,
    /// Unit direction the ray travels. A non-unit direction skews the reported
    /// distance, so normalise before constructing.
    pub direction: Vec3,
    /// Farthest distance along `direction` the cast reaches.
    pub max_distance: f32,
}

impl SceneRay {
    /// Builds a ray from its origin, unit direction, and maximum travel
    /// distance.
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3, max_distance: f32) -> SceneRay {
        SceneRay {
            origin,
            direction,
            max_distance,
        }
    }

    /// The point core's displacement over the unit substep: the full ray span.
    #[must_use]
    pub(super) fn displacement(&self) -> Vec3 {
        self.direction * self.max_distance
    }

    /// The point core's start pose: an identity-oriented point at the origin.
    #[must_use]
    pub(super) fn pose(&self) -> ConvexPose {
        ConvexPose::new(self.origin, Quat::IDENTITY)
    }

    /// The point core's motion over the unit substep: linear travel along the
    /// full ray span, no rotation.
    #[must_use]
    pub(super) fn motion(&self) -> BodyMotion {
        BodyMotion::new(self.displacement(), Vec3::ZERO)
    }
}

/// A target pierced by a ray cast, paired with the entry distance, world point,
/// and surface normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayCastHit {
    /// Index of the struck target in the `targets` slice handed to the cast.
    pub target: u32,
    /// Travelled distance from the ray origin to the entry point, in
    /// `[0, max_distance]`.
    pub distance: f32,
    /// World-space entry point on the struck target.
    pub point: Vec3,
    /// Surface normal at the entry point, pointing back toward the ray origin.
    pub normal: Vec3,
}

/// Rewrites a shape-cast hit (substep fraction) into a ray-cast hit (travelled
/// distance), shared by the `CPU` queries here and the `GPU` twin so both map
/// the result identically.
#[must_use]
pub(super) fn ray_cast_hit(ray: &SceneRay, hit: ShapeCastHit) -> RayCastHit {
    RayCastHit {
        target: hit.target,
        distance: hit.toi.time * ray.max_distance,
        point: hit.toi.point,
        normal: hit.toi.normal,
    }
}

/// Builds the moving point core that represents the ray, borrowing `point`
/// (a single-vertex [`ConvexHull::from_point`] hull the caller owns).
#[must_use]
fn ray_shape<'a>(point: &'a ConvexHull, ray: &SceneRay) -> RoundedConvex<'a> {
    RoundedConvex::new(point, ray.pose(), ray.motion(), 0.0)
}

/// Brute-force nearest ray hit: the `golden` the `BVH` query is checked against.
/// Sweeps the ray's point core past every target and keeps the earliest entry.
#[must_use]
pub fn ray_cast(targets: &[RoundedConvex], ray: &SceneRay) -> Option<RayCastHit> {
    let point = ConvexHull::from_point();
    let shape = ray_shape(&point, ray);
    cast_shape(&shape, targets, 1.0, 0.0).map(|hit| ray_cast_hit(ray, hit))
}

/// Brute-force all-hits ray cast: every target the ray pierces, ordered by
/// increasing distance with ties broken by ascending target index.
#[must_use]
pub fn ray_cast_all(targets: &[RoundedConvex], ray: &SceneRay) -> Vec<RayCastHit> {
    let point = ConvexHull::from_point();
    let shape = ray_shape(&point, ray);
    cast_shape_all(&shape, targets, 1.0, 0.0)
        .into_iter()
        .map(|hit| ray_cast_hit(ray, hit))
        .collect()
}

/// Broad-phase-accelerated form of [`ray_cast`]: the nearest target the ray
/// pierces, found by gathering candidates from a `BVH` over the targets' swept
/// bounds and sweeping only those. The result matches [`ray_cast`] exactly,
/// including the lower-index rule on an exact distance tie.
#[must_use]
pub fn ray_cast_bvh(targets: &[RoundedConvex], ray: &SceneRay) -> Option<RayCastHit> {
    let point = ConvexHull::from_point();
    let shape = ray_shape(&point, ray);
    cast_shape_bvh(&shape, targets, 1.0, 0.0).map(|hit| ray_cast_hit(ray, hit))
}

/// Broad-phase-accelerated form of [`ray_cast_all`]: every target the ray
/// pierces, ordered by increasing distance with ties broken by ascending target
/// index, matching [`ray_cast_all`] exactly.
#[must_use]
pub fn ray_cast_all_bvh(targets: &[RoundedConvex], ray: &SceneRay) -> Vec<RayCastHit> {
    let point = ConvexHull::from_point();
    let shape = ray_shape(&point, ray);
    cast_shape_all_bvh(&shape, targets, 1.0, 0.0)
        .into_iter()
        .map(|hit| ray_cast_hit(ray, hit))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit box core, reused across targets.
    fn unit_box() -> ConvexHull {
        ConvexHull::from_box(Vec3::splat(0.5))
    }

    /// A stationary target: a unit box centred at `center`.
    fn target_at(hull: &ConvexHull, center: Vec3) -> RoundedConvex<'_> {
        RoundedConvex::still(hull, ConvexPose::new(center, Quat::IDENTITY), 0.0)
    }

    #[test]
    fn ray_cast_bvh_matches_brute_force_nearest_hit() {
        // Three boxes strung along +x; the ray from the origin must pick the
        // first one and report the distance to its near face at x = 2.5.
        let h = [unit_box(), unit_box(), unit_box()];
        let targets = [
            target_at(&h[0], Vec3::new(3.0, 0.0, 0.0)),
            target_at(&h[1], Vec3::new(6.0, 0.0, 0.0)),
            target_at(&h[2], Vec3::new(9.0, 0.0, 0.0)),
        ];
        let ray = SceneRay::new(Vec3::ZERO, Vec3::X, 20.0);
        let brute = ray_cast(&targets, &ray);
        let bvh = ray_cast_bvh(&targets, &ray);
        assert_eq!(brute, bvh, "BVH ray cast must equal brute force exactly");
        let hit = bvh.expect("the ray pierces the first box");
        assert_eq!(hit.target, 0, "the nearest box is struck first");
        assert!(
            (hit.distance - 2.5).abs() <= 1e-4,
            "entry distance to the near face is 2.5, got {}",
            hit.distance
        );
    }

    #[test]
    fn ray_cast_reports_no_hit_when_the_ray_misses() {
        let h = [unit_box()];
        let targets = [target_at(&h[0], Vec3::new(0.0, 20.0, 0.0))];
        let ray = SceneRay::new(Vec3::ZERO, Vec3::X, 20.0);
        assert!(ray_cast(&targets, &ray).is_none());
        assert_eq!(ray_cast_bvh(&targets, &ray), ray_cast(&targets, &ray));
    }

    #[test]
    fn ray_cast_stops_at_max_distance() {
        let h = [unit_box()];
        // Near face at x = 9.5 is beyond a max distance of 5.
        let targets = [target_at(&h[0], Vec3::new(10.0, 0.0, 0.0))];
        let ray = SceneRay::new(Vec3::ZERO, Vec3::X, 5.0);
        assert!(ray_cast(&targets, &ray).is_none());
        assert_eq!(ray_cast_bvh(&targets, &ray), ray_cast(&targets, &ray));
    }

    #[test]
    fn ray_cast_all_bvh_matches_brute_force_ordered_list() {
        // A corridor of boxes the ray passes through, plus an off-path box the
        // gather must prune. Distinct spacings keep the distances apart.
        let hulls: Vec<ConvexHull> = (0..6).map(|_| unit_box()).collect();
        let mut targets: Vec<RoundedConvex> = (0..5)
            .map(|k| target_at(&hulls[k], Vec3::new(3.0 + 2.0 * (k as f32), 0.0, 0.0)))
            .collect();
        // Off-path box far in +y the swept point never reaches.
        targets.push(target_at(&hulls[5], Vec3::new(4.0, 40.0, 0.0)));
        let ray = SceneRay::new(Vec3::ZERO, Vec3::X, 50.0);
        let brute = ray_cast_all(&targets, &ray);
        let bvh = ray_cast_all_bvh(&targets, &ray);
        assert_eq!(brute, bvh, "BVH all-hits must equal brute force exactly");
        assert_eq!(brute.len(), 5, "the five on-path boxes are pierced");
        for pair in brute.windows(2) {
            assert!(
                pair[0].distance <= pair[1].distance,
                "hits are ordered by increasing distance"
            );
        }
    }

    #[test]
    fn ray_cast_bvh_prunes_a_large_scene() {
        // One on-path box amid a wall of off-path boxes the gather must prune.
        let mut hulls = vec![unit_box()];
        for _ in 0..40 {
            hulls.push(unit_box());
        }
        let mut targets = vec![target_at(&hulls[0], Vec3::new(5.0, 0.0, 0.0))];
        for k in 0..40 {
            targets.push(target_at(&hulls[k + 1], Vec3::new(1.0 + 1.5 * (k as f32), 60.0, 0.0)));
        }
        let ray = SceneRay::new(Vec3::ZERO, Vec3::X, 50.0);
        let brute = ray_cast(&targets, &ray);
        let bvh = ray_cast_bvh(&targets, &ray);
        assert_eq!(brute, bvh, "BVH ray cast must equal brute force on a large scene");
        assert_eq!(bvh.expect("the on-path box is struck").target, 0);
    }
}
