//! Broad-phase-accelerated box cast scene query: sweep an oriented box along a
//! ray and find the nearest static target it first touches (or every target it
//! touches), with the exact contact point and surface normal. This is the
//! one-against-many swept-box query `AAA` engines expose as Jolt
//! `NarrowPhaseQuery::CastShape` with a box, `PhysX` `PxScene::sweep` with a
//! `PxBoxGeometry`, and Unreal `Chaos` box sweeps. The swept box is the standard
//! fourth sweep primitive alongside the ray, sphere, and capsule, used for
//! chunky volume probes, trigger overlaps in flight, and blocky
//! character/vehicle hulls.
//!
//! # A box cast is a swept oriented hull
//!
//! A box cast is exactly a shape cast of a six-face box hull swept across the
//! ray's displacement: the hull's half-extents and orientation fix the box, its
//! centre starts at the cast origin, and it moves `direction * max_distance`
//! over the unit substep. This module is therefore a thin,
//! semantically-clearer façade over the proven
//! [`cast_shape_bvh`](super::shape_cast_bvh::cast_shape_bvh) path — it reuses
//! that path's `BVH` gather and conservative-advancement sweep verbatim and only
//! rewrites the result from a substep fraction into a travelled distance. The
//! `GPU` twin [`GpuSceneBoxCast`](super::box_cast_bvh_gpu::GpuSceneBoxCast)
//! reuses the device shape-cast path the same way.
//!
//! # Distance convention
//!
//! [`SceneBoxCast::direction`] must be a unit vector. The impact is reported as
//! the travelled [`BoxCastHit::distance`] of the box centre in
//! `[0, max_distance]` (the substep fraction scaled by `max_distance`), with the
//! world contact [`BoxCastHit::point`] on the struck target and the surface
//! [`BoxCastHit::normal`] pointing back toward the cast origin, matching the
//! shape-cast normal convention. The box's own
//! [`SceneBoxCast::orientation`] turns the hull, so a box rotated off-axis
//! presents a corner into the sweep and contacts a target farther away than its
//! face-on half-extent would.
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

/// A swept oriented box in world space: a start centre, a unit travel direction,
/// the box half-extents and orientation, and a maximum travel distance. The cast
/// reaches targets within `max_distance` of the origin along `direction` and no
/// farther.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneBoxCast {
    /// World origin the box centre is cast from.
    pub origin: Vec3,
    /// Unit direction the box travels. A non-unit direction skews the reported
    /// distance, so normalise before constructing.
    pub direction: Vec3,
    /// Half-extents of the box along its own local axes before orientation.
    pub half_extents: Vec3,
    /// Orientation of the box's local axes in world space. Identity leaves the
    /// box axis-aligned.
    pub orientation: Quat,
    /// Farthest distance along `direction` the box centre travels.
    pub max_distance: f32,
}

impl SceneBoxCast {
    /// Builds a box cast from its origin, unit travel direction, half-extents,
    /// orientation, and maximum travel distance.
    #[must_use]
    pub fn new(
        origin: Vec3,
        direction: Vec3,
        half_extents: Vec3,
        orientation: Quat,
        max_distance: f32,
    ) -> SceneBoxCast {
        SceneBoxCast {
            origin,
            direction,
            half_extents,
            orientation,
            max_distance,
        }
    }

    /// The box centre's displacement over the unit substep: the full span.
    #[must_use]
    pub(super) fn displacement(&self) -> Vec3 {
        self.direction * self.max_distance
    }

    /// The core's start pose: the box's orientation at the cast origin.
    #[must_use]
    pub(super) fn pose(&self) -> ConvexPose {
        ConvexPose::new(self.origin, self.orientation)
    }

    /// The core's motion over the unit substep: linear travel along the full
    /// span, no rotation.
    #[must_use]
    pub(super) fn motion(&self) -> BodyMotion {
        BodyMotion::new(self.displacement(), Vec3::ZERO)
    }

    /// The box's six-face hull core, axis-aligned in local space; the pose's
    /// orientation turns it into world space.
    #[must_use]
    pub(super) fn core(&self) -> ConvexHull {
        ConvexHull::from_box(self.half_extents)
    }
}

/// A target first touched by a box cast, paired with the entry distance, world
/// contact point, and surface normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxCastHit {
    /// Index of the struck target in the `targets` slice handed to the cast.
    pub target: u32,
    /// Travelled distance of the box centre from the origin to first contact, in
    /// `[0, max_distance]`.
    pub distance: f32,
    /// World-space contact point on the struck target.
    pub point: Vec3,
    /// Surface normal at the contact point, pointing back toward the origin.
    pub normal: Vec3,
}

/// Rewrites a shape-cast hit (substep fraction) into a box-cast hit (travelled
/// distance), shared by the `CPU` queries here and the `GPU` twin so both report
/// the identical distance, point, and normal.
#[must_use]
pub(super) fn box_cast_hit(cast: &SceneBoxCast, hit: ShapeCastHit) -> BoxCastHit {
    BoxCastHit {
        target: hit.target,
        distance: hit.toi.time * cast.max_distance,
        point: hit.toi.point,
        normal: hit.toi.normal,
    }
}

/// Builds the moving box core that represents the cast, borrowing `hull` (a
/// six-face [`ConvexHull::from_box`] hull the caller owns). A box carries no
/// convex rounding, so the core radius is zero.
#[must_use]
fn box_shape<'a>(hull: &'a ConvexHull, cast: &SceneBoxCast) -> RoundedConvex<'a> {
    RoundedConvex::new(hull, cast.pose(), cast.motion(), 0.0)
}

/// Brute-force nearest box hit: the `golden` the `BVH` query is checked against.
/// Sweeps the box against every target and keeps the earliest contact, ties
/// broken by ascending target index.
#[must_use]
pub fn box_cast(targets: &[RoundedConvex], cast: &SceneBoxCast) -> Option<BoxCastHit> {
    let hull = cast.core();
    let shape = box_shape(&hull, cast);
    cast_shape(&shape, targets, 1.0, 0.0).map(|hit| box_cast_hit(cast, hit))
}

/// Brute-force all-hits box cast: every target the box touches, ordered by
/// increasing distance with ties broken by ascending target index.
#[must_use]
pub fn box_cast_all(targets: &[RoundedConvex], cast: &SceneBoxCast) -> Vec<BoxCastHit> {
    let hull = cast.core();
    let shape = box_shape(&hull, cast);
    cast_shape_all(&shape, targets, 1.0, 0.0)
        .into_iter()
        .map(|hit| box_cast_hit(cast, hit))
        .collect()
}

/// Broad-phase-accelerated form of [`box_cast`]: the nearest target the box
/// touches, found by gathering candidates from a `BVH` over the targets' swept
/// bounds and sweeping only those. The result matches [`box_cast`] exactly,
/// including the lower-index rule on an exact distance tie.
#[must_use]
pub fn box_cast_bvh(targets: &[RoundedConvex], cast: &SceneBoxCast) -> Option<BoxCastHit> {
    let hull = cast.core();
    let shape = box_shape(&hull, cast);
    cast_shape_bvh(&shape, targets, 1.0, 0.0).map(|hit| box_cast_hit(cast, hit))
}

/// Broad-phase-accelerated form of [`box_cast_all`]: every target the box
/// touches, ordered by increasing distance with ties broken by ascending target
/// index, matching [`box_cast_all`] exactly.
#[must_use]
pub fn box_cast_all_bvh(targets: &[RoundedConvex], cast: &SceneBoxCast) -> Vec<BoxCastHit> {
    let hull = cast.core();
    let shape = box_shape(&hull, cast);
    cast_shape_all_bvh(&shape, targets, 1.0, 0.0)
        .into_iter()
        .map(|hit| box_cast_hit(cast, hit))
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
    fn box_cast_bvh_matches_brute_force_nearest_hit() {
        // Three boxes strung along +x; the swept unit box from the origin must
        // pick the first one. Its forward face sits at centre + 0.5, so it meets
        // the near face at x = 2.5 when the centre has travelled 2.0.
        let h = [unit_box(), unit_box(), unit_box()];
        let targets = [
            target_at(&h[0], Vec3::new(3.0, 0.0, 0.0)),
            target_at(&h[1], Vec3::new(6.0, 0.0, 0.0)),
            target_at(&h[2], Vec3::new(9.0, 0.0, 0.0)),
        ];
        let cast = SceneBoxCast::new(Vec3::ZERO, Vec3::X, Vec3::splat(0.5), Quat::IDENTITY, 20.0);
        let brute = box_cast(&targets, &cast);
        let bvh = box_cast_bvh(&targets, &cast);
        assert_eq!(brute, bvh, "BVH box cast must equal brute force exactly");
        let hit = bvh.expect("the box touches the first box");
        assert_eq!(hit.target, 0, "the nearest box is struck first");
        assert!(
            (hit.distance - 2.0).abs() <= 1e-4,
            "an axis-aligned unit box meets the x = 2.5 face at travel 2.0, got {}",
            hit.distance
        );
    }

    #[test]
    fn box_rotated_to_present_a_corner_contacts_later() {
        // The same box turned 45 degrees about +z presents a corner into the +x
        // sweep: its forward reach grows from 0.5 to 0.5 * sqrt(2), so the near
        // face at x = 2.5 is met a touch sooner in centre travel, 2.5 - 0.707.
        let h = [unit_box()];
        let targets = [target_at(&h[0], Vec3::new(3.0, 0.0, 0.0))];
        let aligned =
            SceneBoxCast::new(Vec3::ZERO, Vec3::X, Vec3::splat(0.5), Quat::IDENTITY, 20.0);
        let turned = SceneBoxCast::new(
            Vec3::ZERO,
            Vec3::X,
            Vec3::splat(0.5),
            Quat::from_rotation_z(core::f32::consts::FRAC_PI_4),
            20.0,
        );
        let d_aligned = box_cast_bvh(&targets, &aligned)
            .expect("the aligned box hits")
            .distance;
        let d_turned = box_cast_bvh(&targets, &turned)
            .expect("the turned box hits")
            .distance;
        let reach = 0.5 * core::f32::consts::SQRT_2;
        assert!(
            (d_aligned - 2.0).abs() <= 1e-4,
            "the aligned box reaches 0.5 ahead, travel 2.0, got {d_aligned}"
        );
        assert!(
            (d_turned - (2.5 - reach)).abs() <= 1e-4,
            "the 45-degree box reaches a corner {reach} ahead, got {d_turned}"
        );
        assert!(
            d_turned + 1e-4 < d_aligned,
            "a box presenting a corner contacts sooner: {d_turned} vs {d_aligned}"
        );
    }

    #[test]
    fn box_cast_reports_no_hit_when_the_sweep_misses() {
        let h = [unit_box()];
        let targets = [target_at(&h[0], Vec3::new(0.0, 20.0, 0.0))];
        let cast = SceneBoxCast::new(Vec3::ZERO, Vec3::X, Vec3::splat(0.5), Quat::IDENTITY, 20.0);
        assert!(box_cast(&targets, &cast).is_none());
        assert_eq!(box_cast_bvh(&targets, &cast), box_cast(&targets, &cast));
    }

    #[test]
    fn box_cast_stops_at_max_distance() {
        let h = [unit_box()];
        // Near face at x = 9.5; a unit box meets it at centre travel 9.0, beyond
        // a max distance of 5: both miss.
        let targets = [target_at(&h[0], Vec3::new(10.0, 0.0, 0.0))];
        let cast = SceneBoxCast::new(Vec3::ZERO, Vec3::X, Vec3::splat(0.5), Quat::IDENTITY, 5.0);
        assert!(box_cast(&targets, &cast).is_none());
        assert_eq!(box_cast_bvh(&targets, &cast), box_cast(&targets, &cast));
    }

    #[test]
    fn box_cast_all_bvh_matches_brute_force_ordered_list() {
        // A corridor of boxes the box passes through, plus an off-path box the
        // gather must prune. Distinct spacings keep the distances apart.
        let hulls: Vec<ConvexHull> = (0..6).map(|_| unit_box()).collect();
        let mut targets: Vec<RoundedConvex> = (0..5)
            .map(|k| target_at(&hulls[k], Vec3::new(3.0 + 2.0 * (k as f32), 0.0, 0.0)))
            .collect();
        targets.push(target_at(&hulls[5], Vec3::new(4.0, 40.0, 0.0)));
        let cast = SceneBoxCast::new(Vec3::ZERO, Vec3::X, Vec3::splat(0.5), Quat::IDENTITY, 50.0);
        let brute = box_cast_all(&targets, &cast);
        let bvh = box_cast_all_bvh(&targets, &cast);
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
    fn box_cast_bvh_prunes_a_large_scene() {
        let mut hulls = vec![unit_box()];
        for _ in 0..40 {
            hulls.push(unit_box());
        }
        let mut targets = vec![target_at(&hulls[0], Vec3::new(5.0, 0.0, 0.0))];
        for k in 0..40 {
            targets.push(target_at(&hulls[k + 1], Vec3::new(1.0 + 1.5 * (k as f32), 60.0, 0.0)));
        }
        let cast = SceneBoxCast::new(Vec3::ZERO, Vec3::X, Vec3::splat(0.5), Quat::IDENTITY, 50.0);
        let brute = box_cast(&targets, &cast);
        let bvh = box_cast_bvh(&targets, &cast);
        assert_eq!(
            brute, bvh,
            "BVH box cast must equal brute force on a large scene"
        );
        assert_eq!(bvh.expect("the on-path box is struck").target, 0);
    }
}
