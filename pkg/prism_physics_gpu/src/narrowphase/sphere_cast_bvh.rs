//! Broad-phase-accelerated sphere cast scene query: sweep a sphere of a given
//! radius along a ray and find the nearest static target it first touches (or
//! every target it touches), with the exact contact point and surface normal.
//! This is the one-against-many swept-sphere query `AAA` engines expose as Jolt
//! `NarrowPhaseQuery::CastShape` with a sphere, `PhysX` `PxScene::sweep` with a
//! `PxSphereGeometry`, and Unreal `Chaos` sphere sweeps.
//!
//! # A sphere cast is a swept rounded point
//!
//! A sphere cast is exactly a shape cast of a point core inflated by the sphere
//! radius, swept across the ray's displacement: the point core carries the
//! sphere's convex radius, its centre starts at the cast origin, and it moves
//! `direction * max_distance` over the unit substep. This module is therefore a
//! thin, semantically-clearer façade over the proven
//! [`cast_shape_bvh`](super::shape_cast_bvh::cast_shape_bvh) path — it reuses
//! that path's `BVH` gather and conservative-advancement sweep verbatim and only
//! rewrites the result from a substep fraction into a travelled distance. The
//! `GPU` twin
//! [`GpuSceneSphereCast`](super::sphere_cast_bvh_gpu::GpuSceneSphereCast) reuses
//! the device shape-cast path the same way. A radius of `0` degenerates exactly
//! to the [`ray_cast`](super::ray_cast_bvh) point query.
//!
//! # Distance convention
//!
//! [`SceneSphereCast::direction`] must be a unit vector. The impact is reported
//! as the travelled [`SphereCastHit::distance`] of the sphere centre in
//! `[0, max_distance]` (the substep fraction scaled by `max_distance`), with the
//! world contact [`SphereCastHit::point`] on the struck target and the surface
//! [`SphereCastHit::normal`] pointing back toward the cast origin, matching the
//! shape-cast normal convention.
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

/// A swept sphere in world space: a start centre, a unit direction, a sphere
/// radius, and a maximum travel distance. The cast reaches targets within
/// `max_distance` of the origin along `direction` and no farther.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneSphereCast {
    /// World origin the sphere centre is cast from.
    pub origin: Vec3,
    /// Unit direction the sphere travels. A non-unit direction skews the
    /// reported distance, so normalise before constructing.
    pub direction: Vec3,
    /// Radius of the swept sphere. A radius of `0` degenerates to a ray cast.
    pub radius: f32,
    /// Farthest distance along `direction` the sphere centre travels.
    pub max_distance: f32,
}

impl SceneSphereCast {
    /// Builds a sphere cast from its origin, unit direction, sphere radius, and
    /// maximum travel distance.
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3, radius: f32, max_distance: f32) -> SceneSphereCast {
        SceneSphereCast {
            origin,
            direction,
            radius,
            max_distance,
        }
    }

    /// The sphere centre's displacement over the unit substep: the full span.
    #[must_use]
    pub(super) fn displacement(&self) -> Vec3 {
        self.direction * self.max_distance
    }

    /// The core's start pose: an identity-oriented point at the origin, inflated
    /// by [`radius`](Self::radius) into a sphere by the rounded-convex core.
    #[must_use]
    pub(super) fn pose(&self) -> ConvexPose {
        ConvexPose::new(self.origin, Quat::IDENTITY)
    }

    /// The core's motion over the unit substep: linear travel along the full
    /// span, no rotation.
    #[must_use]
    pub(super) fn motion(&self) -> BodyMotion {
        BodyMotion::new(self.displacement(), Vec3::ZERO)
    }
}

/// A target first touched by a sphere cast, paired with the entry distance,
/// world contact point, and surface normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereCastHit {
    /// Index of the struck target in the `targets` slice handed to the cast.
    pub target: u32,
    /// Travelled distance of the sphere centre from the origin to first contact,
    /// in `[0, max_distance]`.
    pub distance: f32,
    /// World-space contact point on the struck target.
    pub point: Vec3,
    /// Surface normal at the contact point, pointing back toward the origin.
    pub normal: Vec3,
}

/// Rewrites a shape-cast hit (substep fraction) into a sphere-cast hit
/// (travelled distance), shared by the `CPU` queries here and the `GPU` twin so
/// both map the result identically.
#[must_use]
pub(super) fn sphere_cast_hit(cast: &SceneSphereCast, hit: ShapeCastHit) -> SphereCastHit {
    SphereCastHit {
        target: hit.target,
        distance: hit.toi.time * cast.max_distance,
        point: hit.toi.point,
        normal: hit.toi.normal,
    }
}

/// Builds the moving sphere core that represents the cast, borrowing `point`
/// (a single-vertex [`ConvexHull::from_point`] hull the caller owns) and
/// inflating it by the cast radius.
#[must_use]
fn sphere_shape<'a>(point: &'a ConvexHull, cast: &SceneSphereCast) -> RoundedConvex<'a> {
    RoundedConvex::new(point, cast.pose(), cast.motion(), cast.radius)
}

/// Brute-force nearest sphere hit: the `golden` the `BVH` query is checked
/// against. Sweeps the sphere against every target and keeps the earliest
/// contact, ties broken by ascending target index.
#[must_use]
pub fn sphere_cast(targets: &[RoundedConvex], cast: &SceneSphereCast) -> Option<SphereCastHit> {
    let point = ConvexHull::from_point();
    let shape = sphere_shape(&point, cast);
    cast_shape(&shape, targets, 1.0, 0.0).map(|hit| sphere_cast_hit(cast, hit))
}

/// Brute-force all-hits sphere cast: every target the sphere touches, ordered by
/// increasing distance with ties broken by ascending target index.
#[must_use]
pub fn sphere_cast_all(targets: &[RoundedConvex], cast: &SceneSphereCast) -> Vec<SphereCastHit> {
    let point = ConvexHull::from_point();
    let shape = sphere_shape(&point, cast);
    cast_shape_all(&shape, targets, 1.0, 0.0)
        .into_iter()
        .map(|hit| sphere_cast_hit(cast, hit))
        .collect()
}

/// Broad-phase-accelerated form of [`sphere_cast`]: the nearest target the
/// sphere touches, found by gathering candidates from a `BVH` over the targets'
/// swept bounds and sweeping only those. The result matches [`sphere_cast`]
/// exactly, including the lower-index rule on an exact distance tie.
#[must_use]
pub fn sphere_cast_bvh(targets: &[RoundedConvex], cast: &SceneSphereCast) -> Option<SphereCastHit> {
    let point = ConvexHull::from_point();
    let shape = sphere_shape(&point, cast);
    cast_shape_bvh(&shape, targets, 1.0, 0.0).map(|hit| sphere_cast_hit(cast, hit))
}

/// Broad-phase-accelerated form of [`sphere_cast_all`]: every target the sphere
/// touches, ordered by increasing distance with ties broken by ascending target
/// index, matching [`sphere_cast_all`] exactly.
#[must_use]
pub fn sphere_cast_all_bvh(
    targets: &[RoundedConvex],
    cast: &SceneSphereCast,
) -> Vec<SphereCastHit> {
    let point = ConvexHull::from_point();
    let shape = sphere_shape(&point, cast);
    cast_shape_all_bvh(&shape, targets, 1.0, 0.0)
        .into_iter()
        .map(|hit| sphere_cast_hit(cast, hit))
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
    fn sphere_cast_bvh_matches_brute_force_nearest_hit() {
        // Three boxes strung along +x; the swept sphere (radius 0.5) from the
        // origin must pick the first one. Its near face sits at x = 2.5, so the
        // sphere surface meets it when the centre has travelled 2.0.
        let h = [unit_box(), unit_box(), unit_box()];
        let targets = [
            target_at(&h[0], Vec3::new(3.0, 0.0, 0.0)),
            target_at(&h[1], Vec3::new(6.0, 0.0, 0.0)),
            target_at(&h[2], Vec3::new(9.0, 0.0, 0.0)),
        ];
        let cast = SceneSphereCast::new(Vec3::ZERO, Vec3::X, 0.5, 20.0);
        let brute = sphere_cast(&targets, &cast);
        let bvh = sphere_cast_bvh(&targets, &cast);
        assert_eq!(brute, bvh, "BVH sphere cast must equal brute force exactly");
        let hit = bvh.expect("the sphere touches the first box");
        assert_eq!(hit.target, 0, "the nearest box is struck first");
        assert!(
            (hit.distance - 2.0).abs() <= 1e-4,
            "centre travels 2.0 before the radius-0.5 sphere meets the x=2.5 face, got {}",
            hit.distance
        );
    }

    #[test]
    fn sphere_radius_brings_the_impact_earlier_than_a_ray() {
        // The same box, cast with a growing radius: a fatter sphere touches the
        // near face sooner, so the travelled distance strictly decreases.
        let h = [unit_box()];
        let targets = [target_at(&h[0], Vec3::new(5.0, 0.0, 0.0))];
        let thin = SceneSphereCast::new(Vec3::ZERO, Vec3::X, 0.1, 20.0);
        let fat = SceneSphereCast::new(Vec3::ZERO, Vec3::X, 1.0, 20.0);
        let d_thin = sphere_cast_bvh(&targets, &thin)
            .expect("thin sphere hits")
            .distance;
        let d_fat = sphere_cast_bvh(&targets, &fat)
            .expect("fat sphere hits")
            .distance;
        assert!(
            d_fat + 1e-4 < d_thin,
            "a fatter sphere must contact earlier: fat {d_fat} vs thin {d_thin}"
        );
    }

    #[test]
    fn sphere_cast_reports_no_hit_when_the_sweep_misses() {
        let h = [unit_box()];
        let targets = [target_at(&h[0], Vec3::new(0.0, 20.0, 0.0))];
        let cast = SceneSphereCast::new(Vec3::ZERO, Vec3::X, 0.5, 20.0);
        assert!(sphere_cast(&targets, &cast).is_none());
        assert_eq!(sphere_cast_bvh(&targets, &cast), sphere_cast(&targets, &cast));
    }

    #[test]
    fn sphere_cast_stops_at_max_distance() {
        let h = [unit_box()];
        // Near face at x = 9.5; a radius-0.5 sphere would meet it at centre
        // travel 9.0, beyond a max distance of 5: both miss.
        let targets = [target_at(&h[0], Vec3::new(10.0, 0.0, 0.0))];
        let cast = SceneSphereCast::new(Vec3::ZERO, Vec3::X, 0.5, 5.0);
        assert!(sphere_cast(&targets, &cast).is_none());
        assert_eq!(sphere_cast_bvh(&targets, &cast), sphere_cast(&targets, &cast));
    }

    #[test]
    fn sphere_cast_all_bvh_matches_brute_force_ordered_list() {
        // A corridor of boxes the sphere passes through, plus an off-path box the
        // gather must prune. Distinct spacings keep the distances apart.
        let hulls: Vec<ConvexHull> = (0..6).map(|_| unit_box()).collect();
        let mut targets: Vec<RoundedConvex> = (0..5)
            .map(|k| target_at(&hulls[k], Vec3::new(3.0 + 2.0 * (k as f32), 0.0, 0.0)))
            .collect();
        // Off-path box far in +y the swept sphere never reaches.
        targets.push(target_at(&hulls[5], Vec3::new(4.0, 40.0, 0.0)));
        let cast = SceneSphereCast::new(Vec3::ZERO, Vec3::X, 0.5, 50.0);
        let brute = sphere_cast_all(&targets, &cast);
        let bvh = sphere_cast_all_bvh(&targets, &cast);
        assert_eq!(brute, bvh, "BVH all-hits must equal brute force exactly");
        assert_eq!(brute.len(), 5, "the five on-path boxes are touched");
        for pair in brute.windows(2) {
            assert!(
                pair[0].distance <= pair[1].distance,
                "hits are ordered by increasing distance"
            );
        }
    }

    #[test]
    fn sphere_cast_bvh_prunes_a_large_scene() {
        // One on-path box amid a wall of off-path boxes the gather must prune.
        let mut hulls = vec![unit_box()];
        for _ in 0..40 {
            hulls.push(unit_box());
        }
        let mut targets = vec![target_at(&hulls[0], Vec3::new(5.0, 0.0, 0.0))];
        for k in 0..40 {
            targets.push(target_at(&hulls[k + 1], Vec3::new(1.0 + 1.5 * (k as f32), 60.0, 0.0)));
        }
        let cast = SceneSphereCast::new(Vec3::ZERO, Vec3::X, 0.5, 50.0);
        let brute = sphere_cast(&targets, &cast);
        let bvh = sphere_cast_bvh(&targets, &cast);
        assert_eq!(brute, bvh, "BVH sphere cast must equal brute force on a large scene");
        assert_eq!(bvh.expect("the on-path box is struck").target, 0);
    }

    #[test]
    fn zero_radius_sphere_cast_degenerates_to_a_ray() {
        // With radius 0 the sphere cast must land on the near face exactly where
        // a ray would: centre travel 2.5 to the x=2.5 face.
        let h = [unit_box()];
        let targets = [target_at(&h[0], Vec3::new(3.0, 0.0, 0.0))];
        let cast = SceneSphereCast::new(Vec3::ZERO, Vec3::X, 0.0, 20.0);
        let hit = sphere_cast_bvh(&targets, &cast).expect("the zero-radius sphere hits");
        assert!(
            (hit.distance - 2.5).abs() <= 1e-4,
            "a zero-radius sphere reaches the x=2.5 face at travel 2.5, got {}",
            hit.distance
        );
    }
}
