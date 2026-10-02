//! Broad-phase-accelerated capsule cast scene query: sweep a capsule (a line
//! segment inflated by a radius) along a ray and find the nearest static target
//! it first touches (or every target it touches), with the exact contact point
//! and surface normal. This is the one-against-many swept-capsule query `AAA`
//! engines expose as Jolt `NarrowPhaseQuery::CastShape` with a capsule, `PhysX`
//! `PxScene::sweep` with a `PxCapsuleGeometry`, and Unreal `Chaos` capsule
//! sweeps. The swept capsule is the character-controller sweep every `AAA`
//! engine leans on for movement, step-up, and ground probing.
//!
//! # A capsule cast is a swept rounded segment
//!
//! A capsule cast is exactly a shape cast of a two-vertex segment core inflated
//! by the capsule radius, swept across the ray's displacement: the segment core
//! carries the capsule's convex radius, its centre starts at the cast origin,
//! and it moves `direction * max_distance` over the unit substep. This module is
//! therefore a thin, semantically-clearer façade over the proven
//! [`cast_shape_bvh`](super::shape_cast_bvh::cast_shape_bvh) path — it reuses
//! that path's `BVH` gather and conservative-advancement sweep verbatim and only
//! rewrites the result from a substep fraction into a travelled distance. The
//! `GPU` twin
//! [`GpuSceneCapsuleCast`](super::capsule_cast_bvh_gpu::GpuSceneCapsuleCast)
//! reuses the device shape-cast path the same way. A segment of zero half-height
//! degenerates to the [`sphere_cast`](super::sphere_cast_bvh) point core, and a
//! segment with a zero radius sweeps the bare segment.
//!
//! # Distance convention
//!
//! [`SceneCapsuleCast::direction`] must be a unit vector. The impact is reported
//! as the travelled [`CapsuleCastHit::distance`] of the capsule centre in
//! `[0, max_distance]` (the substep fraction scaled by `max_distance`), with the
//! world contact [`CapsuleCastHit::point`] on the struck target and the surface
//! [`CapsuleCastHit::normal`] pointing back toward the cast origin, matching the
//! shape-cast normal convention. The capsule's own [`SceneCapsuleCast::axis`]
//! and [`SceneCapsuleCast::half_height`] fix its orientation and length, so a
//! segment aligned across the sweep contacts like a sphere while a segment
//! aligned along the sweep contacts a half-height sooner.
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

/// A swept capsule in world space: a start centre, a unit travel direction, a
/// segment axis and half-height that fix the capsule's orientation and length, a
/// cap radius, and a maximum travel distance. The cast reaches targets within
/// `max_distance` of the origin along `direction` and no farther.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneCapsuleCast {
    /// World origin the capsule centre is cast from.
    pub origin: Vec3,
    /// Unit direction the capsule travels. A non-unit direction skews the
    /// reported distance, so normalise before constructing.
    pub direction: Vec3,
    /// Direction of the capsule's segment axis. Need not be normalised; only its
    /// direction matters. Together with [`half_height`](Self::half_height) it
    /// fixes the segment core endpoints.
    pub axis: Vec3,
    /// Half the length of the capsule's internal segment, measured along
    /// [`axis`](Self::axis). A half-height of `0` degenerates to a sphere cast.
    pub half_height: f32,
    /// Cap radius of the swept capsule. A radius of `0` sweeps the bare segment.
    pub radius: f32,
    /// Farthest distance along `direction` the capsule centre travels.
    pub max_distance: f32,
}

impl SceneCapsuleCast {
    /// Builds a capsule cast from its origin, unit travel direction, segment
    /// axis and half-height, cap radius, and maximum travel distance.
    #[must_use]
    pub fn new(
        origin: Vec3,
        direction: Vec3,
        axis: Vec3,
        half_height: f32,
        radius: f32,
        max_distance: f32,
    ) -> SceneCapsuleCast {
        SceneCapsuleCast {
            origin,
            direction,
            axis,
            half_height,
            radius,
            max_distance,
        }
    }

    /// The capsule centre's displacement over the unit substep: the full span.
    #[must_use]
    pub(super) fn displacement(&self) -> Vec3 {
        self.direction * self.max_distance
    }

    /// The core's start pose: an identity-oriented segment centred at the origin.
    /// The segment's orientation is baked into its vertices by
    /// [`core`](Self::core), so the pose carries no rotation.
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

    /// The capsule's segment core: a two-vertex hull running `+-half_height`
    /// along [`axis`](Self::axis), inflated by [`radius`](Self::radius) into a
    /// capsule by the rounded-convex core.
    #[must_use]
    pub(super) fn core(&self) -> ConvexHull {
        ConvexHull::from_segment(self.axis, self.half_height)
    }
}

/// A target first touched by a capsule cast, paired with the entry distance,
/// world contact point, and surface normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapsuleCastHit {
    /// Index of the struck target in the `targets` slice handed to the cast.
    pub target: u32,
    /// Travelled distance of the capsule centre from the origin to first
    /// contact, in `[0, max_distance]`.
    pub distance: f32,
    /// World-space contact point on the struck target.
    pub point: Vec3,
    /// Surface normal at the contact point, pointing back toward the origin.
    pub normal: Vec3,
}

/// Rewrites a shape-cast hit (substep fraction) into a capsule-cast hit
/// (travelled distance), shared by the `CPU` queries here and the `GPU` twin so
/// both report the identical distance, point, and normal.
#[must_use]
pub(super) fn capsule_cast_hit(cast: &SceneCapsuleCast, hit: ShapeCastHit) -> CapsuleCastHit {
    CapsuleCastHit {
        target: hit.target,
        distance: hit.toi.time * cast.max_distance,
        point: hit.toi.point,
        normal: hit.toi.normal,
    }
}

/// Builds the moving capsule core that represents the cast, borrowing `segment`
/// (a two-vertex [`ConvexHull::from_segment`] hull the caller owns) and
/// inflating it by the cast radius.
#[must_use]
fn capsule_shape<'a>(segment: &'a ConvexHull, cast: &SceneCapsuleCast) -> RoundedConvex<'a> {
    RoundedConvex::new(segment, cast.pose(), cast.motion(), cast.radius)
}

/// Brute-force nearest capsule hit: the `golden` the `BVH` query is checked
/// against. Sweeps the capsule against every target and keeps the earliest
/// contact, ties broken by ascending target index.
#[must_use]
pub fn capsule_cast(targets: &[RoundedConvex], cast: &SceneCapsuleCast) -> Option<CapsuleCastHit> {
    let segment = cast.core();
    let shape = capsule_shape(&segment, cast);
    cast_shape(&shape, targets, 1.0, 0.0).map(|hit| capsule_cast_hit(cast, hit))
}

/// Brute-force all-hits capsule cast: every target the capsule touches, ordered
/// by increasing distance with ties broken by ascending target index.
#[must_use]
pub fn capsule_cast_all(targets: &[RoundedConvex], cast: &SceneCapsuleCast) -> Vec<CapsuleCastHit> {
    let segment = cast.core();
    let shape = capsule_shape(&segment, cast);
    cast_shape_all(&shape, targets, 1.0, 0.0)
        .into_iter()
        .map(|hit| capsule_cast_hit(cast, hit))
        .collect()
}

/// Broad-phase-accelerated form of [`capsule_cast`]: the nearest target the
/// capsule touches, found by gathering candidates from a `BVH` over the targets'
/// swept bounds and sweeping only those. The result matches [`capsule_cast`]
/// exactly, including the lower-index rule on an exact distance tie.
#[must_use]
pub fn capsule_cast_bvh(
    targets: &[RoundedConvex],
    cast: &SceneCapsuleCast,
) -> Option<CapsuleCastHit> {
    let segment = cast.core();
    let shape = capsule_shape(&segment, cast);
    cast_shape_bvh(&shape, targets, 1.0, 0.0).map(|hit| capsule_cast_hit(cast, hit))
}

/// Broad-phase-accelerated form of [`capsule_cast_all`]: every target the
/// capsule touches, ordered by increasing distance with ties broken by ascending
/// target index, matching [`capsule_cast_all`] exactly.
#[must_use]
pub fn capsule_cast_all_bvh(
    targets: &[RoundedConvex],
    cast: &SceneCapsuleCast,
) -> Vec<CapsuleCastHit> {
    let segment = cast.core();
    let shape = capsule_shape(&segment, cast);
    cast_shape_all_bvh(&shape, targets, 1.0, 0.0)
        .into_iter()
        .map(|hit| capsule_cast_hit(cast, hit))
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
    fn capsule_cast_bvh_matches_brute_force_nearest_hit() {
        // Three boxes strung along +x; the swept capsule (axis +y, half-height
        // 1.0, radius 0.5) from the origin must pick the first one. The segment
        // is perpendicular to the +x sweep, so it adds no forward reach: the
        // capsule surface meets the near face at x = 2.5 when the centre has
        // travelled 2.0, exactly like a radius-0.5 sphere.
        let h = [unit_box(), unit_box(), unit_box()];
        let targets = [
            target_at(&h[0], Vec3::new(3.0, 0.0, 0.0)),
            target_at(&h[1], Vec3::new(6.0, 0.0, 0.0)),
            target_at(&h[2], Vec3::new(9.0, 0.0, 0.0)),
        ];
        let cast = SceneCapsuleCast::new(Vec3::ZERO, Vec3::X, Vec3::Y, 1.0, 0.5, 20.0);
        let brute = capsule_cast(&targets, &cast);
        let bvh = capsule_cast_bvh(&targets, &cast);
        assert_eq!(brute, bvh, "BVH capsule cast must equal brute force exactly");
        let hit = bvh.expect("the capsule touches the first box");
        assert_eq!(hit.target, 0, "the nearest box is struck first");
        assert!(
            (hit.distance - 2.0).abs() <= 1e-4,
            "a +y segment adds no forward reach, so the centre travels 2.0, got {}",
            hit.distance
        );
    }

    #[test]
    fn capsule_axis_along_the_sweep_contacts_a_half_height_sooner() {
        // The same box with the segment turned to lie along the +x sweep. The
        // forward tip now reaches half_height + radius = 1.5 ahead of the centre,
        // so the near face at x = 4.5 is met when the centre has travelled 3.0,
        // a full half-height sooner than the perpendicular segment's 4.0.
        let h = [unit_box()];
        let targets = [target_at(&h[0], Vec3::new(5.0, 0.0, 0.0))];
        let along = SceneCapsuleCast::new(Vec3::ZERO, Vec3::X, Vec3::X, 1.0, 0.5, 20.0);
        let across = SceneCapsuleCast::new(Vec3::ZERO, Vec3::X, Vec3::Y, 1.0, 0.5, 20.0);
        let d_along = capsule_cast_bvh(&targets, &along)
            .expect("the along-sweep capsule hits")
            .distance;
        let d_across = capsule_cast_bvh(&targets, &across)
            .expect("the across-sweep capsule hits")
            .distance;
        assert!(
            (d_along - 3.0).abs() <= 1e-4,
            "an along-sweep segment reaches a half-height sooner: expected 3.0, got {d_along}"
        );
        assert!(
            (d_across - 4.0).abs() <= 1e-4,
            "an across-sweep segment adds no forward reach: expected 4.0, got {d_across}"
        );
    }

    #[test]
    fn tall_capsule_reaches_an_offset_target_a_sphere_would_miss() {
        // A box lifted to y in [1.3, 2.3]. A radius-0.5 sphere centred on the
        // sweep line never climbs past y = 0.5 and misses it, but a tall +y
        // capsule (half-height 2.0) spans up to y = 2.0 and contacts the near
        // face at x = 4.5 with its radius clearance, travelling 4.0.
        let h = [unit_box()];
        let targets = [target_at(&h[0], Vec3::new(5.0, 1.8, 0.0))];
        let sphere = SceneCapsuleCast::new(Vec3::ZERO, Vec3::X, Vec3::Y, 0.0, 0.5, 20.0);
        let tall = SceneCapsuleCast::new(Vec3::ZERO, Vec3::X, Vec3::Y, 2.0, 0.5, 20.0);
        assert!(
            capsule_cast_bvh(&targets, &sphere).is_none(),
            "a short capsule degenerating to a sphere never reaches the lifted box"
        );
        let hit = capsule_cast_bvh(&targets, &tall).expect("the tall capsule reaches the box");
        assert!(
            (hit.distance - 4.0).abs() <= 1e-4,
            "the tall capsule meets the x = 4.5 face at radius clearance, travel 4.0, got {}",
            hit.distance
        );
    }

    #[test]
    fn capsule_cast_reports_no_hit_when_the_sweep_misses() {
        let h = [unit_box()];
        let targets = [target_at(&h[0], Vec3::new(0.0, 20.0, 0.0))];
        let cast = SceneCapsuleCast::new(Vec3::ZERO, Vec3::X, Vec3::Y, 1.0, 0.5, 20.0);
        assert!(capsule_cast(&targets, &cast).is_none());
        assert_eq!(
            capsule_cast_bvh(&targets, &cast),
            capsule_cast(&targets, &cast)
        );
    }

    #[test]
    fn capsule_cast_stops_at_max_distance() {
        let h = [unit_box()];
        // Near face at x = 9.5; an across-sweep radius-0.5 capsule would meet it
        // at centre travel 9.0, beyond a max distance of 5: both miss.
        let targets = [target_at(&h[0], Vec3::new(10.0, 0.0, 0.0))];
        let cast = SceneCapsuleCast::new(Vec3::ZERO, Vec3::X, Vec3::Y, 1.0, 0.5, 5.0);
        assert!(capsule_cast(&targets, &cast).is_none());
        assert_eq!(
            capsule_cast_bvh(&targets, &cast),
            capsule_cast(&targets, &cast)
        );
    }

    #[test]
    fn capsule_cast_all_bvh_matches_brute_force_ordered_list() {
        // A corridor of boxes the capsule passes through, plus an off-path box
        // the gather must prune. Distinct spacings keep the distances apart.
        let hulls: Vec<ConvexHull> = (0..6).map(|_| unit_box()).collect();
        let mut targets: Vec<RoundedConvex> = (0..5)
            .map(|k| target_at(&hulls[k], Vec3::new(3.0 + 2.0 * (k as f32), 0.0, 0.0)))
            .collect();
        // Off-path box far in +y the swept capsule never reaches.
        targets.push(target_at(&hulls[5], Vec3::new(4.0, 40.0, 0.0)));
        let cast = SceneCapsuleCast::new(Vec3::ZERO, Vec3::X, Vec3::Y, 1.0, 0.5, 50.0);
        let brute = capsule_cast_all(&targets, &cast);
        let bvh = capsule_cast_all_bvh(&targets, &cast);
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
    fn capsule_cast_bvh_prunes_a_large_scene() {
        // One on-path box amid a wall of off-path boxes the gather must prune.
        let mut hulls = vec![unit_box()];
        for _ in 0..40 {
            hulls.push(unit_box());
        }
        let mut targets = vec![target_at(&hulls[0], Vec3::new(5.0, 0.0, 0.0))];
        for k in 0..40 {
            targets.push(target_at(&hulls[k + 1], Vec3::new(1.0 + 1.5 * (k as f32), 60.0, 0.0)));
        }
        let cast = SceneCapsuleCast::new(Vec3::ZERO, Vec3::X, Vec3::Y, 1.0, 0.5, 50.0);
        let brute = capsule_cast(&targets, &cast);
        let bvh = capsule_cast_bvh(&targets, &cast);
        assert_eq!(
            brute, bvh,
            "BVH capsule cast must equal brute force on a large scene"
        );
        assert_eq!(bvh.expect("the on-path box is struck").target, 0);
    }

    #[test]
    fn zero_half_height_capsule_cast_degenerates_to_a_sphere() {
        // With half-height 0 the segment collapses to a point, so the capsule
        // cast lands exactly where a radius-0.5 sphere cast would: the near face
        // at x = 2.5 met at centre travel 2.0.
        let h = [unit_box()];
        let targets = [target_at(&h[0], Vec3::new(3.0, 0.0, 0.0))];
        let cast = SceneCapsuleCast::new(Vec3::ZERO, Vec3::X, Vec3::Y, 0.0, 0.5, 20.0);
        let hit = capsule_cast_bvh(&targets, &cast).expect("the degenerate capsule hits");
        assert!(
            (hit.distance - 2.0).abs() <= 1e-4,
            "a zero-half-height capsule reaches the x = 2.5 face at travel 2.0, got {}",
            hit.distance
        );
    }
}
