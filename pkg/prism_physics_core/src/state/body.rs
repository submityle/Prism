//! Body descriptions and mass properties.
//!
//! These are the value types used to spawn and describe rigid bodies. Mass
//! properties are stored in inverse form (`inv_mass`, `inv_inertia`) because
//! that is the form the solver and integrator consume directly, and because it
//! represents an immovable body cleanly as a zero (infinite mass).

use crate::collider::{ColliderHandle, ColliderShape, PhysicsMaterial};
use glam::{Quat, Vec3};

/// The simulation category of a body, which controls how it responds to
/// forces and integration.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BodyKind {
    /// Never moves and has infinite mass. Not integrated.
    #[default]
    Static,
    /// Moves only via externally set velocity; has infinite mass and ignores
    /// forces. Not integrated by the free-body integrator in M0.
    Kinematic,
    /// Fully simulated: responds to gravity, damping, and (in later
    /// milestones) constraints and contacts.
    Dynamic,
}

/// Inverse mass and inverse principal inertia of a body.
///
/// Values are stored inverted so that an immovable body is represented by
/// zeros (infinite mass / inertia). [`inv_inertia`](MassProperties::inv_inertia)
/// holds the inverse of the three principal moments of inertia expressed in
/// the body's local frame.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MassProperties {
    /// Inverse mass. Zero means infinite mass (immovable under linear force).
    pub inv_mass: f32,
    /// Inverse of the principal moments of inertia in the local frame. Zero
    /// components mean infinite inertia about that axis.
    pub inv_inertia: Vec3,
}

impl MassProperties {
    /// Returns all-zero mass properties, representing an immovable body with
    /// infinite mass and inertia (suitable for static and kinematic bodies).
    #[must_use]
    pub const fn zero() -> Self {
        MassProperties {
            inv_mass: 0.0,
            inv_inertia: Vec3::ZERO,
        }
    }

    /// Returns the finite mass of the body, or `0.0` if the body has infinite
    /// mass (`inv_mass == 0`).
    #[must_use]
    pub fn mass(&self) -> f32 {
        if self.inv_mass > 0.0 {
            1.0 / self.inv_mass
        } else {
            0.0
        }
    }

    /// Computes mass properties for a shape of uniform `density`.
    ///
    /// This delegates to [`ColliderShape::mass_properties`], which uses
    /// closed-form inertia for spheres and cuboids, a documented approximation
    /// for capsules, and zero (immovable) for planes.
    #[must_use]
    pub fn from_shape(shape: &ColliderShape, density: f32) -> MassProperties {
        shape.mass_properties(density)
    }
}

impl Default for MassProperties {
    fn default() -> Self {
        MassProperties::zero()
    }
}

/// A full description of a body used when spawning it into storage.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BodyDesc {
    /// The simulation category of the body.
    pub kind: BodyKind,
    /// Initial world-space position.
    pub position: Vec3,
    /// Initial world-space orientation.
    pub orientation: Quat,
    /// Initial linear velocity.
    pub linear_velocity: Vec3,
    /// Initial angular velocity (axis-angle rate, radians per second).
    pub angular_velocity: Vec3,
    /// Mass and inertia of the body.
    pub mass_properties: MassProperties,
    /// Linear velocity damping coefficient (per second).
    pub linear_damping: f32,
    /// Angular velocity damping coefficient (per second).
    pub angular_damping: f32,
    /// Optional shared collision shape handle. `None` means the body has no
    /// collider and is skipped by collision detection.
    pub collider: Option<ColliderHandle>,
    /// Surface contact material used by the contact solver.
    pub material: PhysicsMaterial,
}

impl BodyDesc {
    /// Creates a dynamic body at `position` with unit mass and no inertia
    /// resistance, ready to be customised further.
    #[must_use]
    pub fn dynamic_at(position: Vec3) -> BodyDesc {
        BodyDesc {
            kind: BodyKind::Dynamic,
            position,
            mass_properties: MassProperties {
                inv_mass: 1.0,
                inv_inertia: Vec3::ONE,
            },
            ..BodyDesc::default()
        }
    }

    /// Creates a static (immovable) body at `position`.
    #[must_use]
    pub fn static_at(position: Vec3) -> BodyDesc {
        BodyDesc {
            kind: BodyKind::Static,
            position,
            mass_properties: MassProperties::zero(),
            ..BodyDesc::default()
        }
    }

    /// Sets the mass properties and returns the modified description.
    #[must_use]
    pub fn with_mass_properties(mut self, mass_properties: MassProperties) -> BodyDesc {
        self.mass_properties = mass_properties;
        self
    }

    /// Sets the linear velocity and returns the modified description.
    #[must_use]
    pub fn with_linear_velocity(mut self, linear_velocity: Vec3) -> BodyDesc {
        self.linear_velocity = linear_velocity;
        self
    }

    /// Sets the angular velocity and returns the modified description.
    #[must_use]
    pub fn with_angular_velocity(mut self, angular_velocity: Vec3) -> BodyDesc {
        self.angular_velocity = angular_velocity;
        self
    }

    /// Sets both damping coefficients and returns the modified description.
    #[must_use]
    pub fn with_damping(mut self, linear: f32, angular: f32) -> BodyDesc {
        self.linear_damping = linear;
        self.angular_damping = angular;
        self
    }

    /// Attaches a shared collider shape handle and returns the modified
    /// description.
    #[must_use]
    pub fn with_collider(mut self, collider: ColliderHandle) -> BodyDesc {
        self.collider = Some(collider);
        self
    }

    /// Sets the surface contact material and returns the modified description.
    #[must_use]
    pub fn with_material(mut self, material: PhysicsMaterial) -> BodyDesc {
        self.material = material;
        self
    }
}

impl Default for BodyDesc {
    fn default() -> Self {
        BodyDesc {
            kind: BodyKind::Static,
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass_properties: MassProperties::zero(),
            linear_damping: 0.0,
            angular_damping: 0.0,
            collider: None,
            material: PhysicsMaterial::DEFAULT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_mass_is_infinite() {
        let mp = MassProperties::zero();
        assert_eq!(mp.mass(), 0.0);
        assert_eq!(mp.inv_mass, 0.0);
    }

    #[test]
    fn finite_mass_round_trips() {
        let mp = MassProperties {
            inv_mass: 0.25,
            inv_inertia: Vec3::ONE,
        };
        assert!((mp.mass() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn builders_set_expected_fields() {
        let d = BodyDesc::dynamic_at(Vec3::new(1.0, 2.0, 3.0))
            .with_linear_velocity(Vec3::X)
            .with_damping(0.1, 0.2);
        assert_eq!(d.kind, BodyKind::Dynamic);
        assert_eq!(d.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(d.linear_velocity, Vec3::X);
        assert_eq!(d.linear_damping, 0.1);
        assert_eq!(d.angular_damping, 0.2);

        let s = BodyDesc::static_at(Vec3::Y);
        assert_eq!(s.kind, BodyKind::Static);
        assert_eq!(s.mass_properties.inv_mass, 0.0);
    }
}
