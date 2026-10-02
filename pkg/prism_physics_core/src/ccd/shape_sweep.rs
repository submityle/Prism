//! Shape-aware continuous-collision sweep of a single fast mover against the
//! rest of the scene.
//!
//! The baseline [`resolve_ccd`](super::resolve_ccd) sweep approximated every
//! mover by its inscribed core sphere and used the world's analytic
//! [`spherecast`](crate::world::PhysicsWorld::spherecast). That is cheap but
//! clamps boxes and capsules earlier than their true geometry would and ignores
//! orientation entirely, so a thin spinning plank reports contact against a
//! wall long before an edge actually reaches it -- and, worse, never catches a
//! body that *rotates* a corner through a wall without translating at all.
//!
//! This module replaces the sphere approximation with a true convex-vs-convex
//! time-of-impact query that accounts for the mover's full rigid sub-step
//! motion: both its linear displacement and its spin. The mover's real
//! [`ColliderShape`] is exposed as a [`CcdSupport`] support map in its
//! start-of-sub-step pose and advanced against each nearby target with
//! [`rotational_conservative_advancement`], which bounds the worst-case
//! closing speed by the linear term plus the angular point speed
//! `angle.abs() * radius`. This matches the fidelity of shape-cast CCD in
//! UE/Chaos, `PhysX`, and Jolt: an oriented box clamps exactly where its own
//! surface would touch, and a fast-spinning plank is caught by the arc its
//! corner sweeps even when its centre barely moves. Infinite
//! [`ColliderShape::Plane`] half-spaces cannot be support-mapped, so they are
//! handled by an exact analytic half-space sweep that folds the same angular
//! bound into its closing rate.
//!
//! Rotational advancement is a strict generalisation of pure-translation
//! advancement: when the sub-step rotation is the identity the angular speed is
//! zero and the swept pose reduces to a straight line, so purely translating
//! movers produce exactly the same time of impact as before.
//!
//! The query is strictly read-only: it gathers candidate targets through the
//! world's broad-phase [`overlap_aabb`](crate::world::PhysicsWorld::overlap_aabb)
//! and never mutates body state, so [`resolve_ccd`](super::resolve_ccd) can run
//! it inside its gather pass before applying any clamp.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is a
//! clean-room composition of this workspace's own support maps, rotational
//! conservative advancement, and broad-phase overlap query.

use crate::ccd::support::CcdSupport;
use crate::collider::ColliderShape;
use crate::query::QueryFilter;
use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;
use glam::{Quat, Vec3};
use prism_physics_geometry::{rotational_conservative_advancement, Aabb, SupportMap};

/// Separation (metres) at which the conservative-advancement query declares two
/// convex shapes to be in contact. A small positive band keeps the GJK witness
/// alive at the moment of impact instead of degenerating exactly on the
/// surface, and doubles as the stand-off for the analytic plane sweep.
const CONTACT_TOLERANCE: f32 = 1.0e-3;

/// The mover's rigid sub-step motion, resolved once and shared across every
/// target query so each target sees a consistent linear + angular bound.
struct MoverMotion {
    /// Centre of rotation: the mover's origin at the start of the sub-step.
    pivot: Vec3,
    /// Net linear translation of the mover's origin over the sub-step.
    displacement: Vec3,
    /// Rotation applied over the sub-step (curr * prev inverse), about the pivot.
    rotation: Quat,
    /// Bounding-sphere radius of the mover about the pivot.
    radius: f32,
    /// Worst-case angular point speed over the sub-step (`|angle| * radius`),
    /// used by the analytic plane sweep as the angular share of the closing
    /// rate.
    angular_speed: f32,
}

/// Sweeps `mover`'s actual convex shape through its full rigid sub-step motion
/// -- translating by `displacement` and rotating from `prev_rot` to `curr_rot`
/// about its start origin `prev_pos` -- against every other body and returns the
/// sub-step fraction `[0, 1]` at which it first makes contact, or [`None`] when
/// nothing is hit within the sub-step.
///
/// The caller rewinds the body to the interpolated pose at the returned
/// fraction `t`: position `prev_pos + displacement * t` and orientation
/// `prev_rot.slerp(curr_rot, t)`. A returned `0.0` means the mover already
/// overlaps a target at the start of the sub-step and must not advance at all.
///
/// Targets are gathered through the world broad-phase over the swept bounding
/// box of the mover, so the per-mover cost scales with local scene density
/// rather than the whole world.
#[must_use]
pub(super) fn sweep_toi(
    world: &PhysicsWorld,
    mover: BodyHandle,
    mover_shape: &ColliderShape,
    prev_pos: Vec3,
    prev_rot: Quat,
    curr_rot: Quat,
    displacement: Vec3,
) -> Option<f32> {
    let distance = displacement.length();
    let rotation = (curr_rot * prev_rot.inverse()).normalize();
    let (_, angle) = rotation.to_axis_angle();
    let radius = bounding_radius(mover_shape);
    let angular_speed = angle.abs() * radius;

    // Neither translating nor rotating: there is nothing to sweep.
    if distance <= 0.0 && angular_speed <= 0.0 {
        return None;
    }

    // The mover must be a bounded convex volume to be support-mapped. Planes are
    // immovable and are never movers, so this only rejects nonsensical input.
    // The support map is built in the *start* pose; rotational advancement
    // applies the sub-step spin on top of it.
    let mover_support = CcdSupport::from_shape(mover_shape, prev_pos, prev_rot)?;

    // Conservative box enclosing every intermediate pose. The mover's origin
    // travels the segment prev_pos -> curr_pos, and every point of the body
    // stays within `radius` of that origin at all times (rotation is about the
    // origin), so padding the segment's AABB by the bounding radius plus the
    // contact band covers the full swept -- and spun -- volume.
    let curr_pos = prev_pos + displacement;
    let pad = radius + CONTACT_TOLERANCE;
    let swept_aabb = Aabb::from_points(&[prev_pos, curr_pos])
        .expect("two points always form a valid AABB")
        .expanded_by(pad);
    let filter = QueryFilter::excluding(mover);

    let motion = MoverMotion {
        pivot: prev_pos,
        displacement,
        rotation,
        radius,
        angular_speed,
    };

    let mut best_toi: Option<f32> = None;
    for target in world.overlap_aabb(&swept_aabb, &filter) {
        let Some(toi) = target_toi(world, target, &mover_support, &motion) else {
            continue;
        };
        // Fraction of the sub-step; the queries only ever return a value in
        // `[0, 1]`.
        if best_toi.is_none_or(|b| toi < b) {
            best_toi = Some(toi);
        }
    }

    best_toi
}

/// Returns the sub-step fraction `[0, 1]` at which the mover, undergoing
/// [`motion`](MoverMotion), first contacts `target`, or [`None`] when they never
/// touch during the sub-step.
fn target_toi(
    world: &PhysicsWorld,
    target: BodyHandle,
    mover_support: &CcdSupport,
    motion: &MoverMotion,
) -> Option<f32> {
    let collider = world.bodies.collider(target)?;
    let shape = world.shapes.get(collider)?;

    // Start-of-sub-step pose of the target. Static and kinematic bodies do not
    // move within the sub-step, so these equal their current pose; for a moving
    // target the start pose keeps the relative-motion framing consistent.
    let target_pos = world
        .bodies
        .prev_position(target)
        .or_else(|| world.bodies.position(target))?;
    let target_rot = world
        .bodies
        .prev_orientation(target)
        .or_else(|| world.bodies.orientation(target))
        .unwrap_or(Quat::IDENTITY);

    match *shape {
        ColliderShape::Plane { normal, offset } => plane_toi(
            mover_support,
            motion.displacement,
            motion.angular_speed,
            target_pos,
            target_rot,
            normal,
            offset,
        ),
        _ => {
            let target_support = CcdSupport::from_shape(shape, target_pos, target_rot)?;
            // Fold the target's own linear sub-step motion into the relative
            // motion so the query can treat the target as stationary. (A moving
            // target's own spin is not folded in; the bodies being tunnelled
            // *into* are overwhelmingly static or kinematic-translating.)
            let target_disp = world
                .bodies
                .position(target)
                .zip(world.bodies.prev_position(target))
                .map_or(Vec3::ZERO, |(curr, prev)| curr - prev);
            let relative_motion = motion.displacement - target_disp;
            rotational_conservative_advancement(
                mover_support,
                &target_support,
                motion.pivot,
                relative_motion,
                motion.rotation,
                motion.radius,
                CONTACT_TOLERANCE,
            )
            .map(|hit| hit.toi)
        }
    }
}

/// Exact time of impact of a convex mover against a static infinite half-space,
/// expressed as a sub-step fraction `[0, 1]`.
///
/// The mover translates by `motion` and may also spin; `angular_speed` is the
/// worst-case speed at which any point of the mover can approach the plane
/// through that spin. The plane is `dot(normal, x) = offset` in its own frame;
/// `plane_pos`/`plane_rot` place that frame in the world. The solid side is
/// `dot(n, x) <= plane_d`, so the mover's support point toward the plane
/// (`support_point(-n)`) is the deepest point at the start of the sub-step, and
/// contact occurs when the deepest point's signed height above the plane drops
/// to the contact band.
///
/// Bounding the closing rate by `-n.dot(motion) + angular_speed` is
/// conservative: the deepest-point height is `angular_speed`-Lipschitz in time
/// plus the linear descent, so `(height - band) / closing` never steps past a
/// real contact even as the spin changes which corner is deepest.
fn plane_toi(
    mover_support: &CcdSupport,
    motion: Vec3,
    angular_speed: f32,
    plane_pos: Vec3,
    plane_rot: Quat,
    normal: Vec3,
    offset: f32,
) -> Option<f32> {
    let n = (plane_rot * normal).normalize_or_zero();
    if n == Vec3::ZERO {
        return None;
    }
    let point_on_plane = plane_pos + plane_rot * (normal * offset);
    let plane_d = n.dot(point_on_plane);

    // Height of the mover's deepest point above the plane at the start of the
    // sub-step. support_point(-n) is the farthest point in the direction into
    // the solid half-space, i.e. the one with the smallest n . x.
    let deepest = mover_support.support_point(-n);
    let height = n.dot(deepest) - plane_d;

    // Already within the contact band: contact is immediate.
    if height <= CONTACT_TOLERANCE {
        return Some(0.0);
    }

    // Worst-case rate at which the deepest point descends toward the plane: the
    // linear descent plus the maximum angular point speed. A non-positive value
    // means the mover is parallel to or receding from the plane and -- with no
    // spin to carry a corner in -- never reaches it.
    let closing = -n.dot(motion) + angular_speed;
    if closing <= 0.0 {
        return None;
    }

    let toi = (height - CONTACT_TOLERANCE) / closing;
    (toi <= 1.0).then_some(toi.max(0.0))
}

/// Radius of the smallest sphere centred on the body origin that encloses
/// `shape`, used both to pad the broad-phase swept box and as the lever arm
/// that converts the sub-step spin into a worst-case point speed.
#[must_use]
pub(super) fn bounding_radius(shape: &ColliderShape) -> f32 {
    match *shape {
        ColliderShape::Sphere { radius } => radius,
        ColliderShape::Capsule {
            half_height,
            radius,
        } => half_height + radius,
        ColliderShape::Cuboid { half_extents } => half_extents.length(),
        // Planes are never movers; a zero pad is harmless if one is ever passed.
        ColliderShape::Plane { .. } => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounding_radius_covers_each_primitive() {
        assert_eq!(bounding_radius(&ColliderShape::Sphere { radius: 2.0 }), 2.0);
        assert_eq!(
            bounding_radius(&ColliderShape::Capsule {
                half_height: 1.5,
                radius: 0.5,
            }),
            2.0
        );
        let r = bounding_radius(&ColliderShape::Cuboid {
            half_extents: Vec3::new(3.0, 4.0, 0.0),
        });
        assert!((r - 5.0).abs() < 1e-6, "cuboid radius was {r}");
    }

    #[test]
    fn plane_toi_detects_descending_sphere() {
        // Unit sphere one metre above the ground plane y = 0, moving down 2 m,
        // no spin.
        let support = CcdSupport::from_shape(
            &ColliderShape::Sphere { radius: 1.0 },
            Vec3::new(0.0, 2.0, 0.0),
            Quat::IDENTITY,
        )
        .unwrap();
        let toi = plane_toi(
            &support,
            Vec3::new(0.0, -2.0, 0.0),
            0.0,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::Y,
            0.0,
        )
        .expect("sphere descends into the plane");
        // Deepest point starts at y = 1; it reaches the band after ~0.5 of a 2 m
        // descent.
        assert!((toi - 0.5).abs() < 1e-2, "toi was {toi}");
    }

    #[test]
    fn plane_toi_ignores_receding_motion() {
        let support = CcdSupport::from_shape(
            &ColliderShape::Sphere { radius: 1.0 },
            Vec3::new(0.0, 2.0, 0.0),
            Quat::IDENTITY,
        )
        .unwrap();
        // Moving up, away from the ground, with no spin: never impacts.
        let toi = plane_toi(
            &support,
            Vec3::new(0.0, 2.0, 0.0),
            0.0,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::Y,
            0.0,
        );
        assert_eq!(toi, None);
    }

    #[test]
    fn plane_toi_angular_term_closes_a_non_translating_mover() {
        // A long thin plank whose deepest point sits 0.5 m above the ground and
        // which is not translating at all. With a pure linear bound it would
        // never touch (closing = 0), but a spin fast enough to carry a corner
        // down must produce a finite time of impact.
        let support = CcdSupport::from_shape(
            &ColliderShape::Cuboid {
                half_extents: Vec3::new(0.05, 0.5, 0.05),
            },
            Vec3::new(0.0, 1.0, 0.0),
            Quat::IDENTITY,
        )
        .unwrap();
        // Deepest point is at y = 0.5 (height 0.5 above the plane). An angular
        // point speed of 2 per sub-step closes the 0.5 m gap at ~0.25 of the
        // step.
        let toi = plane_toi(
            &support,
            Vec3::ZERO,
            2.0,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::Y,
            0.0,
        )
        .expect("spin must drive a time of impact even with zero translation");
        assert!((toi - 0.25).abs() < 1e-2, "angular toi was {toi}");
    }

    #[test]
    fn sweep_toi_is_none_for_a_body_at_rest() {
        // A body that neither translates nor rotates must not be swept, even if
        // a wall sits right next to it.
        use crate::collider::PhysicsMaterial;
        use crate::state::body::BodyDesc;

        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let wall = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::new(0.05, 1.0, 1.0),
        });
        world.spawn(
            BodyDesc::static_at(Vec3::ZERO)
                .with_collider(wall)
                .with_material(PhysicsMaterial::DEFAULT),
        );

        let shape = ColliderShape::Cuboid {
            half_extents: Vec3::splat(0.1),
        };
        let collider = world.shapes.insert(shape);
        let mover = world.spawn(
            BodyDesc::dynamic_at(Vec3::new(-0.5, 0.0, 0.0))
                .with_collider(collider)
                .with_material(PhysicsMaterial::DEFAULT),
        );

        assert_eq!(
            sweep_toi(
                &world,
                mover,
                &shape,
                Vec3::new(-0.5, 0.0, 0.0),
                Quat::IDENTITY,
                Quat::IDENTITY,
                Vec3::ZERO,
            ),
            None
        );
    }

    #[test]
    fn sweep_toi_catches_a_spin_a_linear_sweep_would_miss() {
        // A long thin plank centred just left of a thin wall, with its long axis
        // pointing *up* so no corner reaches the wall in the start pose and the
        // centre never translates. A quarter turn about Z swings the top corner
        // horizontally into the wall. A pure-translation sweep (zero
        // displacement) returns None; the rotational sweep must return a finite
        // sub-step fraction well before the end of the step.
        use crate::collider::PhysicsMaterial;
        use crate::state::body::BodyDesc;

        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let wall = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::new(0.05, 2.0, 2.0),
        });
        world.spawn(
            BodyDesc::static_at(Vec3::ZERO)
                .with_collider(wall)
                .with_material(PhysicsMaterial::DEFAULT),
        );

        let shape = ColliderShape::Cuboid {
            half_extents: Vec3::new(0.05, 0.6, 0.05),
        };
        let collider = world.shapes.insert(shape);
        // Centre at x = -0.5: the plank's upright half-extent (0.6) does not
        // reach the wall's near face (x = -0.05), but a 90 degree turn about Z
        // lays it flat so the former top corner swings to about x = -0.5 + 0.6 =
        // +0.1, through the wall.
        let center = Vec3::new(-0.5, 0.0, 0.0);
        let mover = world.spawn(
            BodyDesc::dynamic_at(center)
                .with_collider(collider)
                .with_material(PhysicsMaterial::DEFAULT),
        );

        let prev_rot = Quat::IDENTITY;
        let curr_rot = Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2);

        // Pure-translation sweep sees no motion and misses the tunnelling spin.
        assert_eq!(
            sweep_toi(
                &world,
                mover,
                &shape,
                center,
                prev_rot,
                prev_rot,
                Vec3::ZERO
            ),
            None,
            "a non-spinning, non-translating plank must not be clamped"
        );

        // The rotational sweep catches it before the step completes.
        let toi = sweep_toi(
            &world,
            mover,
            &shape,
            center,
            prev_rot,
            curr_rot,
            Vec3::ZERO,
        )
        .expect("the spinning corner must register a time of impact");
        assert!(
            (0.0..1.0).contains(&toi),
            "rotational toi should clamp before the end of the sub-step, got {toi}"
        );
    }
}
