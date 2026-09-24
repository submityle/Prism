//! The closed set of deferred physics commands.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Impulse
//! application (`Δv = m⁻¹ J`) and direct state assignment are textbook rigid
//! body operations implemented from scratch.

use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;
use glam::{Quat, Vec3};

/// A single deferred mutation of a body, applied at a step boundary.
///
/// Every variant names the [`BodyHandle`] it targets. Applying a command to a
/// stale or non-dynamic-eligible handle is a no-op that reports failure rather
/// than panicking, so a command referencing a despawned body is simply dropped.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum PhysicsCommand {
    /// Add a linear impulse `J` (kg·m/s), changing linear velocity by
    /// `m⁻¹ J`. A zero inverse mass (static/kinematic body) yields no change.
    ApplyLinearImpulse {
        /// Body the impulse is applied to.
        body: BodyHandle,
        /// Impulse vector in world space.
        impulse: Vec3,
    },
    /// Add an angular impulse `L`, changing angular velocity by `I⁻¹ L` using
    /// the body's diagonal inverse inertia in world axes (matching the engine's
    /// simplified inertia handling).
    ApplyAngularImpulse {
        /// Body the angular impulse is applied to.
        body: BodyHandle,
        /// Angular impulse vector in world space.
        impulse: Vec3,
    },
    /// Overwrite the body's linear velocity.
    SetLinearVelocity {
        /// Body whose linear velocity is set.
        body: BodyHandle,
        /// New linear velocity in world space.
        velocity: Vec3,
    },
    /// Overwrite the body's angular velocity.
    SetAngularVelocity {
        /// Body whose angular velocity is set.
        body: BodyHandle,
        /// New angular velocity in world space.
        velocity: Vec3,
    },
    /// Teleport the body to a new position without changing its velocity.
    SetPosition {
        /// Body to reposition.
        body: BodyHandle,
        /// New world-space position.
        position: Vec3,
    },
    /// Reorient the body; the quaternion is normalized before it is stored.
    SetOrientation {
        /// Body to reorient.
        body: BodyHandle,
        /// New orientation; normalized on apply.
        orientation: Quat,
    },
}

impl PhysicsCommand {
    /// Returns the body this command targets.
    #[must_use]
    pub fn body(&self) -> BodyHandle {
        match *self {
            PhysicsCommand::ApplyLinearImpulse { body, .. }
            | PhysicsCommand::ApplyAngularImpulse { body, .. }
            | PhysicsCommand::SetLinearVelocity { body, .. }
            | PhysicsCommand::SetAngularVelocity { body, .. }
            | PhysicsCommand::SetPosition { body, .. }
            | PhysicsCommand::SetOrientation { body, .. } => body,
        }
    }

    /// Applies the command to `world`, returning `true` if it took effect.
    ///
    /// Returns `false` when the target handle is stale/invalid so the caller can
    /// account for dropped commands; this never panics.
    pub fn apply(&self, world: &mut PhysicsWorld) -> bool {
        let applied = match *self {
            PhysicsCommand::ApplyLinearImpulse { body, impulse } => {
                let Some(mass) = world.bodies.mass_properties(body) else {
                    return false;
                };
                let Some(v) = world.bodies.linear_velocity(body) else {
                    return false;
                };
                world
                    .bodies
                    .set_linear_velocity(body, v + impulse * mass.inv_mass)
            }
            PhysicsCommand::ApplyAngularImpulse { body, impulse } => {
                let Some(mass) = world.bodies.mass_properties(body) else {
                    return false;
                };
                let Some(w) = world.bodies.angular_velocity(body) else {
                    return false;
                };
                world
                    .bodies
                    .set_angular_velocity(body, w + impulse * mass.inv_inertia)
            }
            PhysicsCommand::SetLinearVelocity { body, velocity } => {
                world.bodies.set_linear_velocity(body, velocity)
            }
            PhysicsCommand::SetAngularVelocity { body, velocity } => {
                world.bodies.set_angular_velocity(body, velocity)
            }
            PhysicsCommand::SetPosition { body, position } => {
                world.bodies.set_position(body, position)
            }
            PhysicsCommand::SetOrientation { body, orientation } => world
                .bodies
                .set_orientation(body, normalize_or_identity(orientation)),
        };
        // Any command that took effect disturbs its target, so wake it (and its
        // island next sub-step). Waking a static/kinematic or stale handle is a
        // harmless no-op.
        if applied {
            world.bodies.wake(self.body());
        }
        applied
    }
}

/// Normalizes `q`, falling back to the identity when it is degenerate.
fn normalize_or_identity(q: Quat) -> Quat {
    if q.length_squared() > 0.0 {
        q.normalize()
    } else {
        Quat::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::body::BodyDesc;

    #[test]
    fn linear_impulse_changes_velocity_by_inv_mass() {
        let mut world = PhysicsWorld::default();
        // Default dynamic body has inv_mass = 1.0.
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let ok = PhysicsCommand::ApplyLinearImpulse {
            body: h,
            impulse: Vec3::new(3.0, 0.0, 0.0),
        }
        .apply(&mut world);
        assert!(ok);
        assert_eq!(
            world.bodies.linear_velocity(h),
            Some(Vec3::new(3.0, 0.0, 0.0))
        );
    }

    #[test]
    fn impulse_on_static_body_has_no_effect() {
        let mut world = PhysicsWorld::default();
        let h = world.spawn(BodyDesc::static_at(Vec3::ZERO));
        // Static body has zero inverse mass, so the velocity stays zero.
        PhysicsCommand::ApplyLinearImpulse {
            body: h,
            impulse: Vec3::new(9.0, 9.0, 9.0),
        }
        .apply(&mut world);
        assert_eq!(world.bodies.linear_velocity(h), Some(Vec3::ZERO));
    }

    #[test]
    fn set_velocity_and_position_overwrite() {
        let mut world = PhysicsWorld::default();
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        PhysicsCommand::SetLinearVelocity {
            body: h,
            velocity: Vec3::new(1.0, 2.0, 3.0),
        }
        .apply(&mut world);
        PhysicsCommand::SetPosition {
            body: h,
            position: Vec3::new(-1.0, -2.0, -3.0),
        }
        .apply(&mut world);
        assert_eq!(
            world.bodies.linear_velocity(h),
            Some(Vec3::new(1.0, 2.0, 3.0))
        );
        assert_eq!(world.bodies.position(h), Some(Vec3::new(-1.0, -2.0, -3.0)));
    }

    #[test]
    fn command_on_stale_handle_reports_failure() {
        let mut world = PhysicsWorld::default();
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        world.bodies.remove(h);
        let ok = PhysicsCommand::SetPosition {
            body: h,
            position: Vec3::ONE,
        }
        .apply(&mut world);
        assert!(!ok);
    }
}
