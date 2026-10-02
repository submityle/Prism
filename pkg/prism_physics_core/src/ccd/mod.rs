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
//! Between prediction and contact detection, [`resolve_ccd`] sweeps the core
//! sphere (see [`sweep::sweep_radius`]) of each opt-in body along its sub-step
//! displacement using the world's existing [`spherecast`](crate::world::PhysicsWorld::spherecast).
//! If the sweep hits something before the body reaches its predicted pose, the
//! body is clamped back to the point of first contact. The very same sub-step's
//! discrete detection then resolves the touch normally and velocity recovery
//! bleeds off the excess speed, so the body comes to rest against the surface
//! instead of passing through it.
//!
//! Only CCD-flagged, awake, dynamic bodies whose displacement exceeds
//! [`CcdConfig::min_motion_ratio`] times their core radius are swept, so the
//! slow-moving majority pay no cost.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! speculative core-sphere sweep and time-of-impact clamp is a standard,
//! publicly documented continuous-collision technique, implemented here on top
//! of this crate's own spherecast query.

pub mod config;
pub mod support;
pub mod sweep;

pub use config::CcdConfig;
pub use support::CcdSupport;

use crate::query::QueryFilter;
use crate::state::body::BodyKind;
use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;
use glam::Vec3;
use prism_physics_geometry::Ray;

/// Sweeps every opt-in fast-moving body and clamps it to the first surface it
/// would cross during this sub-step of length `h`.
///
/// This is meant to run once per sub-step, *after* pose prediction and *before*
/// discrete contact detection. It is a no-op when CCD is disabled globally
/// ([`CcdConfig::enabled`]) or when `h` is not positive.
///
/// The work is split into two passes so the read-only sweep query and the
/// position write never borrow the world at the same time: the first pass
/// gathers the clamped position for each affected body, and the second pass
/// applies them.
pub fn resolve_ccd(world: &mut PhysicsWorld, h: f32) {
    let config = world.config.ccd;
    if !config.enabled || h <= 0.0 {
        return;
    }

    // Pass 1: gather clamps. Every access here is read-only, so the immutable
    // borrow taken by `spherecast` is safe.
    let mut clamps: Vec<(BodyHandle, Vec3)> = Vec::new();
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

        let displacement = curr - prev;
        let distance = displacement.length();
        // Motion gate: only sweep genuine fast movers. This also guarantees
        // `distance > 0.0` so the direction below is well defined.
        if distance <= config.min_motion_ratio * radius {
            continue;
        }

        let direction = displacement / distance;
        let ray = Ray::with_tmax(prev, direction, distance);
        let filter = QueryFilter::excluding(handle);
        if let Some(hit) = world.spherecast(&ray, radius, &filter)
            && hit.time_of_impact < distance
        {
            let toi = (hit.time_of_impact - config.skin).max(0.0);
            clamps.push((handle, prev + direction * toi));
        }
    }

    // Pass 2: apply the clamps. This is the only mutable access.
    for (handle, position) in clamps {
        world.bodies.set_position(handle, position);
    }
}

#[cfg(test)]
mod tests {
    use crate::collider::{ColliderShape, PhysicsMaterial};
    use crate::solver::{Solver, XpbdSolver};
    use crate::state::body::BodyDesc;
    use crate::world::PhysicsWorld;
    use glam::Vec3;

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
}
