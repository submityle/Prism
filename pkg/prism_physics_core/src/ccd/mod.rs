//! Continuous collision detection (CCD) for fast-moving dynamic bodies.
//!
//! # Why CCD is needed
//!
//! The [`XpbdSolver`](crate::solver::XpbdSolver) detects contacts once per
//! sub-step, at each body's *predicted* end-of-sub-step pose. That is correct
//! for slow bodies, whose start and end poses overlap the same geometry, but it
//! misses collisions for a body that travels further than its own size in a
//! single sub-step: the predicted pose can sit entirely on the far side of a
//! thin wall, so no contact is ever generated and the body tunnels through.
//!
//! # Approach
//!
//! Between prediction and contact detection, [`resolve_ccd`] sweeps each opt-in
//! body's *true convex shape* along its sub-step displacement (see
//! [`shape_sweep`]). A fast mover's real
//! [`ColliderShape`](crate::collider::ColliderShape) is advanced against nearby
//! geometry with a *rotational* conservative-advancement time-of-impact query
//! that accounts for both its linear displacement and its sub-step spin, so an
//! oriented box or capsule clamps exactly where its own surface would touch
//! rather than where a bounding sphere would, and a fast-spinning body is
//! caught by the arc its corner sweeps even when its centre barely moves. If
//! the sweep hits something before the body reaches its predicted pose, the
//! body is rewound to the interpolated pose at the time of first contact --
//! both its position and its orientation. The very same sub-step's discrete detection then
//! resolves the touch normally and velocity recovery bleeds off the excess
//! speed, so the body comes to rest against the surface instead of passing
//! through it.
//!
//! Only CCD-flagged, awake, dynamic bodies whose linear displacement *or*
//! angular arc exceeds [`CcdConfig::min_motion_ratio`] times their core radius
//! are swept, so the slow-moving majority pay no cost.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! shape-aware conservative-advancement sweep and time-of-impact clamp is a
//! standard, publicly documented continuous-collision technique, implemented
//! here on top of this workspace's own support maps and geometry queries.

pub mod config;
mod shape_sweep;
pub mod support;
pub mod sweep;

pub use config::CcdConfig;
pub use support::CcdSupport;

use crate::state::body::BodyKind;
use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;
use glam::{Quat, Vec3};

/// Sweeps every opt-in fast-moving body and clamps it to the first surface it
/// would cross during this sub-step of length `h`.
///
/// This is meant to run once per sub-step, *after* pose prediction and *before*
/// discrete contact detection. It is a no-op when CCD is disabled globally
/// ([`CcdConfig::enabled`]) or when `h` is not positive.
///
/// The work is split into two passes so the read-only sweep query and the
/// pose write never borrow the world at the same time: the first pass gathers
/// the clamped pose (position and orientation) for each affected body, and the
/// second pass applies them.
pub fn resolve_ccd(world: &mut PhysicsWorld, h: f32) {
    let config = world.config.ccd;
    if !config.enabled || h <= 0.0 {
        return;
    }

    // Pass 1: gather clamps. Every access here is read-only, so the immutable
    // borrow taken by the shape sweep is safe.
    let mut clamps: Vec<(BodyHandle, Vec3, Quat)> = Vec::new();
    for slot in 0..world.bodies.slot_count() {
        let Some(handle) = world.bodies.handle_at_slot(slot) else {
            continue;
        };
        // CCD only guards dynamic bodies; static and kinematic bodies are the
        // things being tunnelled *into*, not through.
        if world.bodies.kind(handle) != Some(BodyKind::Dynamic) {
            continue;
        }
        if world.bodies.ccd(handle) != Some(true) {
            continue;
        }
        // Sleeping bodies are not predicted, so their displacement is zero and
        // the motion gate below would reject them anyway; skip them explicitly.
        if world.bodies.is_sleeping(handle) == Some(true) {
            continue;
        }
        let Some(collider) = world.bodies.collider(handle) else {
            continue;
        };
        let Some(shape) = world.shapes.get(collider) else {
            continue;
        };
        let Some(radius) = sweep::sweep_radius(shape) else {
            continue;
        };
        let Some(prev) = world.bodies.prev_position(handle) else {
            continue;
        };
        let Some(curr) = world.bodies.position(handle) else {
            continue;
        };
        let prev_rot = world
            .bodies
            .prev_orientation(handle)
            .or_else(|| world.bodies.orientation(handle))
            .unwrap_or(Quat::IDENTITY);
        let curr_rot = world.bodies.orientation(handle).unwrap_or(prev_rot);

        let displacement = curr - prev;
        let distance = displacement.length();

        // Motion gate: only sweep genuine fast movers. A body qualifies when
        // either its linear travel or the arc swept by its farthest point
        // exceeds a fraction of its core radius, so a body that tunnels purely
        // by spinning (zero translation) is still caught.
        let gate = config.min_motion_ratio * radius;
        let (_, delta_angle) = (curr_rot * prev_rot.inverse()).normalize().to_axis_angle();
        let angular_arc = delta_angle.abs() * shape_sweep::bounding_radius(shape);
        if distance <= gate && angular_arc <= gate {
            continue;
        }

        // Shape-aware rotational sweep of the body's true convex geometry
        // against the scene; falls back to no clamp when nothing is hit this
        // sub-step. The returned value is the sub-step fraction of first
        // contact, so a value < 1 means the predicted pose overshoots a surface.
        if let Some(toi) =
            shape_sweep::sweep_toi(world, handle, shape, prev, prev_rot, curr_rot, displacement)
            && toi < 1.0
        {
            // Convert the skin back-off into a sub-step fraction over the total
            // point travel (linear + angular) so the body stops just short of
            // the surface, then rewind both pose components to that fraction.
            let travel = distance + angular_arc;
            let skin_frac = if travel > 0.0 {
                (config.skin / travel).min(toi)
            } else {
                0.0
            };
            let clamped_toi = (toi - skin_frac).max(0.0);
            let clamped_pos = prev + displacement * clamped_toi;
            let clamped_rot = prev_rot.slerp(curr_rot, clamped_toi);
            clamps.push((handle, clamped_pos, clamped_rot));
        }
    }

    // Pass 2: apply the clamps. This is the only mutable access.
    for (handle, position, orientation) in clamps {
        world.bodies.set_position(handle, position);
        world.bodies.set_orientation(handle, orientation);
    }
}

#[cfg(test)]
mod tests {
    use crate::collider::{ColliderShape, PhysicsMaterial};
    use crate::solver::{Solver, XpbdSolver};
    use crate::state::body::BodyDesc;
    use crate::world::PhysicsWorld;
    use glam::{Quat, Vec3};

    /// Builds a world with zero gravity and a thin static wall in the `x = 0`
    /// plane, then fires a small fast sphere at it from `x = -2`. The sphere is
    /// created with CCD set to `ccd`. Returns the sphere's final x position
    /// after a single 1/60 s step.
    fn fire_bullet_at_wall(ccd: bool) -> f32 {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);

        // Thin static wall spanning x in [-0.05, 0.05].
        let wall = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::new(0.05, 5.0, 5.0),
        });
        world.spawn(
            BodyDesc::static_at(Vec3::ZERO)
                .with_collider(wall)
                .with_material(PhysicsMaterial::DEFAULT),
        );

        // Fast bullet sphere approaching from the left at 300 m/s.
        let ball_shape = ColliderShape::Sphere { radius: 0.1 };
        let ball = world.shapes.insert(ball_shape);
        let mp = ball_shape.mass_properties(1.0);
        let bullet = world.spawn(
            BodyDesc::dynamic_at(Vec3::new(-2.0, 0.0, 0.0))
                .with_collider(ball)
                .with_mass_properties(mp)
                .with_material(PhysicsMaterial::DEFAULT)
                .with_linear_velocity(Vec3::new(300.0, 0.0, 0.0))
                .with_ccd(ccd),
        );

        // A single sub-step displacement is 300 / 60 = 5 m, far past the wall.
        let mut solver = XpbdSolver::new();
        solver.step(&mut world, 1.0 / 60.0, 1);
        world.bodies.position(bullet).unwrap().x
    }

    #[test]
    fn ccd_bullet_does_not_tunnel_through_thin_wall() {
        let x = fire_bullet_at_wall(true);
        assert!(
            x < 0.0,
            "CCD bullet should stop on the near side of the wall, got x = {x}"
        );
    }

    #[test]
    fn without_ccd_bullet_tunnels_through_thin_wall() {
        // Documents the discrete-detection failure mode that CCD fixes: with no
        // sweep the predicted pose lands past the wall and no contact forms.
        let x = fire_bullet_at_wall(false);
        assert!(
            x > 0.0,
            "without CCD the bullet is expected to tunnel through, got x = {x}"
        );
    }

    #[test]
    fn resolve_ccd_is_noop_when_disabled() {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        world.config.ccd.enabled = false;
        // Should not panic and should leave an empty world untouched.
        super::resolve_ccd(&mut world, 1.0 / 60.0);
    }

    /// Fires a CCD-enabled dynamic body carrying `shape` (rotated by
    /// `orientation`) from `x = -2` at 300 m/s toward the thin static wall in
    /// the `x = 0` plane, steps once, and returns the body's final x position.
    ///
    /// Shares the wall setup with [`fire_bullet_at_wall`] but exercises the
    /// shape-aware sweep for non-spherical movers.
    fn fire_shape_at_wall(shape: ColliderShape, orientation: Quat) -> f32 {
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);

        let wall = world.shapes.insert(ColliderShape::Cuboid {
            half_extents: Vec3::new(0.05, 5.0, 5.0),
        });
        world.spawn(
            BodyDesc::static_at(Vec3::ZERO)
                .with_collider(wall)
                .with_material(PhysicsMaterial::DEFAULT),
        );

        let collider = world.shapes.insert(shape);
        let mp = shape.mass_properties(1.0);
        let mut desc = BodyDesc::dynamic_at(Vec3::new(-2.0, 0.0, 0.0))
            .with_collider(collider)
            .with_mass_properties(mp)
            .with_material(PhysicsMaterial::DEFAULT)
            .with_linear_velocity(Vec3::new(300.0, 0.0, 0.0))
            .with_ccd(true);
        desc.orientation = orientation;
        let bullet = world.spawn(desc);

        let mut solver = XpbdSolver::new();
        solver.step(&mut world, 1.0 / 60.0, 1);
        world.bodies.position(bullet).unwrap().x
    }

    #[test]
    fn ccd_box_bullet_does_not_tunnel_through_thin_wall() {
        let x = fire_shape_at_wall(
            ColliderShape::Cuboid {
                half_extents: Vec3::splat(0.1),
            },
            Quat::IDENTITY,
        );
        assert!(
            x < 0.0,
            "CCD box bullet should stop on the near side of the wall, got x = {x}"
        );
    }

    #[test]
    fn ccd_capsule_bullet_does_not_tunnel_through_thin_wall() {
        let x = fire_shape_at_wall(
            ColliderShape::Capsule {
                half_height: 0.3,
                radius: 0.1,
            },
            Quat::IDENTITY,
        );
        assert!(
            x < 0.0,
            "CCD capsule bullet should stop on the near side of the wall, got x = {x}"
        );
    }

    #[test]
    fn ccd_respects_true_box_face_not_inscribed_sphere() {
        // A long thin box travelling along its own long axis. Its front face
        // sits 0.4 m ahead of the centre, so a shape-aware sweep must stop the
        // centre near x = -0.45 (front face against the wall's -0.05 face).
        // The old inscribed-sphere sweep used only the 0.05 m core radius and
        // would have let the centre advance to about x = -0.10, driving the
        // front face 0.35 m into the wall. Checking x < -0.3 proves the true
        // geometry — not a bounding sphere — governs the clamp.
        let x = fire_shape_at_wall(
            ColliderShape::Cuboid {
                half_extents: Vec3::new(0.4, 0.05, 0.05),
            },
            Quat::IDENTITY,
        );
        assert!(
            (-0.6..-0.3).contains(&x),
            "shape-aware clamp should respect the box's front face, got x = {x}"
        );
    }
}
