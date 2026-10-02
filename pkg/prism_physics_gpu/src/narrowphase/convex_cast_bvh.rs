//! Broad-phase-accelerated convex cast scene query: sweep an arbitrary posed
//! convex hull along a ray and find the nearest static target it first touches
//! (or every target it touches), with the exact contact point and surface
//! normal. This is the general one-against-many swept-shape query `AAA` engines
//! expose as Jolt `NarrowPhaseQuery::CastShape` with a convex, `PhysX`
//! `PxScene::sweep` with a `PxConvexMeshGeometry`, and Unreal `Chaos` convex
//! sweeps. It is the arbitrary-hull member of the sweep family whose fixed-shape
//! members are the ray, sphere, capsule, and box: where those synthesise a
//! specific core, this one accepts any `ConvexHull` the caller already owns and,
//! unlike them, exposes a convex rounding `SceneConvexCast::radius` so a swept
//! shape can carry the inflated skin `PhysX` and `Chaos` keep on their convex
//! cores.
//!
//! # A convex cast is a swept rounded hull
//!
//! A convex cast is exactly a shape cast of the caller's hull swept across the
//! ray's displacement: the hull and its rounding radius fix the shape, its
//! origin starts at the cast origin under the given orientation, and it moves
//! `direction * max_distance` over the unit substep. This module is therefore a
//! thin, semantically-clearer façade over the proven
//! [`cast_shape_bvh`](super::shape_cast_bvh::cast_shape_bvh) path — it reuses
//! that path's `BVH` gather and conservative-advancement sweep verbatim and only
//! rewrites the result from a substep fraction into a travelled distance. The
//! `GPU` twin [`GpuSceneConvexCast`](super::convex_cast_bvh_gpu::GpuSceneConvexCast)
//! reuses the device shape-cast path the same way.
//!
//! # Distance convention
//!
//! [`SceneConvexCast::direction`] must be a unit vector. The impact is reported
//! as the travelled [`ConvexCastHit::distance`] of the hull origin in
//! `[0, max_distance]` (the substep fraction scaled by `max_distance`), with the
//! world contact [`ConvexCastHit::point`] on the struck target and the surface
//! [`ConvexCastHit::normal`] pointing back toward the cast origin, matching the
//! shape-cast normal convention. The hull's own
//! [`SceneConvexCast::orientation`] turns the shape and its
//! [`SceneConvexCast::radius`] inflates it, so an off-axis or rounded hull
//! contacts a target farther away than its unrotated, unrounded extent would.
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

/// A swept posed convex hull in world space: a borrowed core hull, a start
/// origin, a unit travel direction, an orientation, a convex rounding radius,
/// and a maximum travel distance. The cast reaches targets within
/// `max_distance` of the origin along `direction` and no farther.
#[derive(Clone, Copy, Debug)]
pub struct SceneConvexCast<'a> {
    /// Core convex hull swept by the cast, owned by the caller. Its vertices are
    /// in the hull's local frame before [`SceneConvexCast::orientation`].
    pub hull: &'a ConvexHull,
    /// World origin the hull's local frame is placed at.
    pub origin: Vec3,
    /// Unit direction the hull travels. A non-unit direction skews the reported
    /// distance, so normalise before constructing.
    pub direction: Vec3,
    /// Orientation of the hull's local axes in world space. Identity leaves the
    /// hull in its local frame.
    pub orientation: Quat,
    /// Convex rounding radius inflating the hull's surface. Zero sweeps the raw
    /// hull; a positive radius sweeps the rounded (spherically-swept) hull.
    pub radius: f32,
    /// Farthest distance along `direction` the hull origin travels.
    pub max_distance: f32,
}

impl<'a> SceneConvexCast<'a> {
    /// Builds a convex cast from its borrowed core hull, origin, unit travel
    /// direction, orientation, convex rounding radius, and maximum travel
    /// distance.
    #[must_use]
    pub fn new(
        hull: &'a ConvexHull,
        origin: Vec3,
        direction: Vec3,
        orientation: Quat,
        radius: f32,
        max_distance: f32,
    ) -> SceneConvexCast<'a> {
        SceneConvexCast {
            hull,
            origin,
            direction,
            orientation,
            radius,
            max_distance,
        }
    }

    /// The hull origin's displacement over the unit substep: the full span.
    #[must_use]
    pub(super) fn displacement(&self) -> Vec3 {
        self.direction * self.max_distance
    }

    /// The core's start pose: the hull's orientation at the cast origin.
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
}

/// A target first touched by a convex cast, paired with the entry distance,
/// world contact point, and surface normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvexCastHit {
    /// Index of the struck target in the `targets` slice handed to the cast.
    pub target: u32,
    /// Travelled distance of the hull origin from the origin to first contact,
    /// in `[0, max_distance]`.
    pub distance: f32,
    /// World-space contact point on the struck target.
    pub point: Vec3,
    /// Surface normal at the contact point, pointing back toward the origin.
    pub normal: Vec3,
}

/// Rewrites a shape-cast hit (substep fraction) into a convex-cast hit
/// (travelled distance), shared by the `CPU` queries here and the `GPU` twin so
/// both report the identical distance, point, and normal.
#[must_use]
pub(super) fn convex_cast_hit(cast: &SceneConvexCast, hit: ShapeCastHit) -> ConvexCastHit {
    ConvexCastHit {
        target: hit.target,
        distance: hit.toi.time * cast.max_distance,
        point: hit.toi.point,
        normal: hit.toi.normal,
    }
}

/// Builds the moving rounded-hull core that represents the cast, borrowing the
/// caller's [`SceneConvexCast::hull`] and carrying the cast's convex rounding
/// radius.
#[must_use]
fn convex_shape<'a>(cast: &SceneConvexCast<'a>) -> RoundedConvex<'a> {
    RoundedConvex::new(cast.hull, cast.pose(), cast.motion(), cast.radius)
}

/// Brute-force nearest convex hit: the `golden` the `BVH` query is checked
/// against. Sweeps the hull against every target and keeps the earliest
/// contact, ties broken by ascending target index.
#[must_use]
pub fn convex_cast(targets: &[RoundedConvex], cast: &SceneConvexCast) -> Option<ConvexCastHit> {
    let shape = convex_shape(cast);
    cast_shape(&shape, targets, 1.0, 0.0).map(|hit| convex_cast_hit(cast, hit))
}

/// Brute-force all-hits convex cast: every target the hull touches, ordered by
/// increasing distance with ties broken by ascending target index.
#[must_use]
pub fn convex_cast_all(targets: &[RoundedConvex], cast: &SceneConvexCast) -> Vec<ConvexCastHit> {
    let shape = convex_shape(cast);
    cast_shape_all(&shape, targets, 1.0, 0.0)
        .into_iter()
        .map(|hit| convex_cast_hit(cast, hit))
        .collect()
}

/// Broad-phase-accelerated form of [`convex_cast`]: the nearest target the hull
/// touches, found by gathering candidates from a `BVH` over the targets' swept
/// bounds and sweeping only those. The result matches [`convex_cast`] exactly,
/// including the lower-index rule on an exact distance tie.
#[must_use]
pub fn convex_cast_bvh(targets: &[RoundedConvex], cast: &SceneConvexCast) -> Option<ConvexCastHit> {
    let shape = convex_shape(cast);
    cast_shape_bvh(&shape, targets, 1.0, 0.0).map(|hit| convex_cast_hit(cast, hit))
}

/// Broad-phase-accelerated form of [`convex_cast_all`]: every target the hull
/// touches, ordered by increasing distance with ties broken by ascending target
/// index, matching [`convex_cast_all`] exactly.
#[must_use]
pub fn convex_cast_all_bvh(
    targets: &[RoundedConvex],
    cast: &SceneConvexCast,
) -> Vec<ConvexCastHit> {
    let shape = convex_shape(cast);
    cast_shape_all_bvh(&shape, targets, 1.0, 0.0)
        .into_iter()
        .map(|hit| convex_cast_hit(cast, hit))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An axis-aligned box hull with the given half-extents.
    fn box_hull(he: Vec3) -> ConvexHull {
        ConvexHull::from_box(he)
    }

    /// A stationary target hull centred at `center` with the given rounding.
    fn target_at(hull: &ConvexHull, center: Vec3, radius: f32) -> RoundedConvex<'_> {
        RoundedConvex::still(hull, ConvexPose::new(center, Quat::IDENTITY), radius)
    }

    #[test]
    fn convex_cast_bvh_matches_brute_force_nearest_hit() {
        // Three unit boxes strung along +x; the swept box from the origin must
        // pick the first. Its forward face sits at centre + 0.5 and the target's
        // near face at centre - 0.5, so contact is when the origin has travelled
        // 3.0 - 0.5 - 0.5 = 2.0.
        let swept = box_hull(Vec3::splat(0.5));
        let th = [
            box_hull(Vec3::splat(0.5)),
            box_hull(Vec3::splat(0.5)),
            box_hull(Vec3::splat(0.5)),
        ];
        let targets = [
            target_at(&th[0], Vec3::new(3.0, 0.0, 0.0), 0.0),
            target_at(&th[1], Vec3::new(6.0, 0.0, 0.0), 0.0),
            target_at(&th[2], Vec3::new(9.0, 0.0, 0.0), 0.0),
        ];
        let cast = SceneConvexCast::new(&swept, Vec3::ZERO, Vec3::X, Quat::IDENTITY, 0.0, 20.0);
        let brute = convex_cast(&targets, &cast);
        let bvh = convex_cast_bvh(&targets, &cast);
        assert_eq!(brute, bvh, "BVH convex cast must equal brute force");
        let hit = brute.expect("the swept box reaches the first target");
        assert_eq!(hit.target, 0, "the nearest target is struck first");
        assert!((hit.distance - 2.0).abs() < 1e-4, "distance was {}", hit.distance);
    }

    #[test]
    fn convex_cast_rounding_radius_contacts_earlier() {
        // A rounded swept box reaches the target a radius sooner than the raw
        // box: the inflated skin closes the gap by exactly the radius.
        let swept = box_hull(Vec3::splat(0.5));
        let th = [box_hull(Vec3::splat(0.5))];
        let targets = [target_at(&th[0], Vec3::new(5.0, 0.0, 0.0), 0.0)];

        let raw = SceneConvexCast::new(&swept, Vec3::ZERO, Vec3::X, Quat::IDENTITY, 0.0, 20.0);
        let rounded = SceneConvexCast::new(&swept, Vec3::ZERO, Vec3::X, Quat::IDENTITY, 0.25, 20.0);
        let raw_hit = convex_cast_bvh(&targets, &raw).expect("raw box reaches target");
        let rounded_hit = convex_cast_bvh(&targets, &rounded).expect("rounded box reaches target");
        assert!(
            (raw_hit.distance - rounded_hit.distance - 0.25).abs() < 1e-4,
            "rounding should save exactly the radius: raw {} rounded {}",
            raw_hit.distance,
            rounded_hit.distance
        );
    }

    #[test]
    fn convex_cast_oriented_hull_matches_brute_force() {
        // An asymmetric box rotated about z presents a different silhouette into
        // the sweep; the façade and brute force must still agree exactly.
        let swept = box_hull(Vec3::new(0.8, 0.3, 0.5));
        let th = [box_hull(Vec3::splat(0.5)), box_hull(Vec3::splat(0.5))];
        let targets = [
            target_at(&th[0], Vec3::new(4.0, 0.2, 0.0), 0.1),
            target_at(&th[1], Vec3::new(7.0, -0.1, 0.0), 0.0),
        ];
        let cast = SceneConvexCast::new(
            &swept,
            Vec3::ZERO,
            Vec3::X,
            Quat::from_rotation_z(core::f32::consts::FRAC_PI_6),
            0.05,
            20.0,
        );
        assert_eq!(
            convex_cast(&targets, &cast),
            convex_cast_bvh(&targets, &cast),
            "BVH convex cast must equal brute force for an oriented rounded hull"
        );
    }

    #[test]
    fn convex_cast_all_orders_every_hit_like_brute_force() {
        let swept = box_hull(Vec3::splat(0.4));
        let th: Vec<ConvexHull> = (0..5).map(|_| box_hull(Vec3::splat(0.5))).collect();
        let targets: Vec<RoundedConvex> = (0..5)
            .map(|i| target_at(&th[i], Vec3::new(3.0 + 2.0 * i as f32, 0.0, 0.0), 0.0))
            .collect();
        let cast = SceneConvexCast::new(&swept, Vec3::ZERO, Vec3::X, Quat::IDENTITY, 0.0, 40.0);
        assert_eq!(
            convex_cast_all(&targets, &cast),
            convex_cast_all_bvh(&targets, &cast),
            "ordered BVH list must equal brute force"
        );
    }

    #[test]
    fn convex_cast_misses_past_max_distance() {
        let swept = box_hull(Vec3::splat(0.5));
        let th = [box_hull(Vec3::splat(0.5))];
        let targets = [target_at(&th[0], Vec3::new(10.0, 0.0, 0.0), 0.0)];
        // Near face at x = 9.5, reached at origin travel 9.0, beyond max 5.0.
        let cast = SceneConvexCast::new(&swept, Vec3::ZERO, Vec3::X, Quat::IDENTITY, 0.0, 5.0);
        assert_eq!(convex_cast_bvh(&targets, &cast), None, "contact is past max distance");
        assert_eq!(convex_cast(&targets, &cast), convex_cast_bvh(&targets, &cast));
    }
}
