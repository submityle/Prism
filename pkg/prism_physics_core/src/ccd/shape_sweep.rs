//! Shape-aware continuous-collision sweep of a single fast mover against the
//! rest of the scene.
//!
//! The baseline [`resolve_ccd`](super::resolve_ccd) sweep approximated every
//! mover by its inscribed core sphere and used the world's analytic
//! [`spherecast`](crate::world::PhysicsWorld::spherecast). That is cheap but
//! clamps boxes and capsules earlier than their true geometry would and ignores
//! orientation entirely, so a thin spinning plank reports contact against a
//! wall long before an edge actually reaches it.
//!
//! This module replaces the sphere approximation with a true convex-vs-convex
//! time-of-impact query. The mover's real [`ColliderShape`] is exposed as a
//! [`CcdSupport`] support map and advanced against each nearby target with
//! [`conservative_advancement`], matching the fidelity of shape-cast CCD in
//! UE/Chaos, `PhysX`, and Jolt. Infinite [`ColliderShape::Plane`] half-spaces
//! cannot be support-mapped, so they are handled by an exact analytic
//! half-space sweep instead.
//!
//! The query is strictly read-only: it gathers candidate targets through the
//! world's broad-phase [`overlap_aabb`](crate::world::PhysicsWorld::overlap_aabb)
//! and never mutates body state, so [`resolve_ccd`](super::resolve_ccd) can run
//! it inside its gather pass before applying any clamp.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is a
//! clean-room composition of this workspace's own support maps, conservative
//! advancement, and broad-phase overlap query.

use crate::ccd::support::CcdSupport;
use crate::collider::ColliderShape;
use crate::query::QueryFilter;
use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;
use glam::{Quat, Vec3};
use prism_physics_geometry::{conservative_advancement, Aabb, SupportMap};

/// Separation (metres) at which the conservative-advancement query declares two
/// convex shapes to be in contact. A small positive band keeps the GJK witness
/// alive at the moment of impact instead of degenerating exactly on the
/// surface, and doubles as the stand-off for the analytic plane sweep.
const CONTACT_TOLERANCE: f32 = 1.0e-3;

/// Sweeps `mover`'s actual convex shape along `displacement` (its sub-step
/// motion, starting from `prev_pos` with orientation `orientation`) against
/// every other body and returns the distance along `displacement` at which it
/// first makes contact, or `None` when nothing is hit within the sub-step.
///
/// The returned value is measured in the same units as `displacement.length()`,
/// so the caller clamps the mover to `prev_pos + displacement.normalize() * d`.
/// A returned `0.0` means the mover already overlaps a target at the start of
/// the sub-step and must not advance at all.
///
/// Targets are gathered through the world broad-phase over the swept bounding
/// box of the mover, so the per-mover cost scales with local scene density
/// rather than the whole world.
#[must_use]
pub(super) fn sweep_distance(
    world: &PhysicsWorld,
    mover: BodyHandle,
    mover_shape: &ColliderShape,
    prev_pos: Vec3,
    orientation: Quat,
    displacement: Vec3,
) -> Option<f32> {
    let distance = displacement.length();
    if distance <= 0.0 {
        return None;
    }

    // The mover must be a bounded convex volume to be support-mapped. Planes are
    // immovable and are never movers, so this only rejects nonsensical input.
    let mover_support = CcdSupport::from_shape(mover_shape, prev_pos, orientation)?;

    // Conservative box that encloses the mover at both ends of the sub-step,
    // padded by the mover's bounding radius and the contact band so no target
    // the swept shape could touch is missed by the broad-phase gather.
    let curr_pos = prev_pos + displacement;
    let pad = bounding_radius(mover_shape) + CONTACT_TOLERANCE;
    let swept_aabb = Aabb::from_points(&[prev_pos, curr_pos])
        .expect("two points always form a valid AABB")
        .expanded_by(pad);
    let filter = QueryFilter::excluding(mover);

    let mut best_toi: Option<f32> = None;
    for target in world.overlap_aabb(&swept_aabb, &filter) {
        let Some(toi) = target_toi(world, target, &mover_support, displacement) else {
            continue;
        };
        // Fraction of the sub-step; `conservative_advancement`/the plane sweep
        // only ever return a value in `[0, 1]`.
        if best_toi.is_none_or(|b| toi < b) {
            best_toi = Some(toi);
        }
    }

    best_toi.map(|toi| toi * distance)
}

/// Returns the sub-step fraction `[0, 1]` at which `mover_support`, translating
/// by `displacement`, first contacts `target`, or `None` when they never touch.
fn target_toi(
    world: &PhysicsWorld,
    target: BodyHandle,
    mover_support: &CcdSupport,
    displacement: Vec3,
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
            displacement,
            target_pos,
            target_rot,
            normal,
            offset,
        ),
        _ => {
            let target_support = CcdSupport::from_shape(shape, target_pos, target_rot)?;
            // Fold the target's own sub-step motion into the relative motion so
            // the query can treat the target as stationary.
            let target_disp = world
                .bodies
                .position(target)
                .zip(world.bodies.prev_position(target))
                .map_or(Vec3::ZERO, |(curr, prev)| curr - prev);
            let relative_motion = displacement - target_disp;
            conservative_advancement(
                mover_support,
                &target_support,
                relative_motion,
                CONTACT_TOLERANCE,
            )
            .map(|hit| hit.toi)
        }
    }
}

/// Exact time of impact of a convex mover translating by `motion` against a
/// static infinite half-space, expressed as a sub-step fraction `[0, 1]`.
///
/// The plane is `dot(normal, x) = offset` in its own frame; `pose` places that
/// frame in the world. The solid side is `dot(n, x) <= plane_d`, so the mover's
/// support point toward the plane (`support_point(-n)`) is the first point to
/// enter the solid region, and contact occurs when its signed height above the
/// plane drops to the contact band.
fn plane_toi(
    mover_support: &CcdSupport,
    motion: Vec3,
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
    // sub-step. `support_point(-n)` is the farthest point in the direction into
    // the solid half-space, i.e. the one with the smallest `n . x`.
    let deepest = mover_support.support_point(-n);
    let height = n.dot(deepest) - plane_d;

    // Already within the contact band: contact is immediate.
    if height <= CONTACT_TOLERANCE {
        return Some(0.0);
    }

    // Rate of descent toward the plane. A non-negative value means the mover is
    // parallel to or receding from the plane and never reaches it.
    let closing = -n.dot(motion);
    if closing <= 0.0 {
        return None;
    }

    let toi = (height - CONTACT_TOLERANCE) / closing;
    (toi <= 1.0).then_some(toi.max(0.0))
}

/// Radius of the smallest sphere centred on the body origin that encloses
/// `shape`, used to pad the broad-phase swept box.
fn bounding_radius(shape: &ColliderShape) -> f32 {
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
        // Unit sphere one metre above the ground plane y = 0, moving down 2 m.
        let support = CcdSupport::from_shape(
            &ColliderShape::Sphere { radius: 1.0 },
            Vec3::new(0.0, 2.0, 0.0),
            Quat::IDENTITY,
        )
        .unwrap();
        let toi = plane_toi(
            &support,
            Vec3::new(0.0, -2.0, 0.0),
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
        // Moving up, away from the ground: never impacts.
        let toi = plane_toi(
            &support,
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::Y,
            0.0,
        );
        assert_eq!(toi, None);
    }
}
