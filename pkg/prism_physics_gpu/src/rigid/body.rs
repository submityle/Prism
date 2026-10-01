//! The dense rigid-body state advanced by the 6-DOF integrator.
//!
//! A rigid body carries six degrees of freedom: a world-space centre-of-mass
//! `position` (3) and an `orientation` quaternion (3 independent). Its motion is
//! described by a `linear_velocity` and an `angular_velocity`, both in world
//! space, and its response to force and torque by an `inverse_mass` and a
//! body-frame `inverse_inertia` diagonal.
//!
//! # Why body-frame diagonal inertia
//!
//! Every rigid body has a symmetric positive-definite inertia tensor; by the
//! spectral theorem it is diagonal in its principal-axis frame. Production
//! engines (`PhysX`, Chaos) therefore store the inertia as that principal-axis
//! diagonal and bake the principal-axis rotation into the body frame, so a
//! diagonal `inverse_inertia` is fully general for a rigid body — not a
//! simplification. The world-space tensor is recovered on the fly as
//! `R · diag(I) · Rᵀ`, which the integrator does implicitly by rotating the
//! angular velocity and torque into the body frame, applying the diagonal, and
//! rotating back.
//!
//! # Static and locked bodies
//!
//! An `inverse_mass` of `0` marks a body whose translation is frozen (infinite
//! mass): the integrator skips its linear update. An all-zero `inverse_inertia`
//! marks a body whose rotation is frozen (infinite inertia): the integrator
//! skips its angular and orientation update. The two are independent, so a body
//! may translate without rotating or rotate in place without translating.
//!
//! Provenance: textbook rigid-body state (Baraff & Witkin; principal-axis
//! inertia). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};

/// A structure-of-arrays store of every rigid body's state, advanced in place
/// by the integrator.
///
/// All six arrays are kept the same length (one entry per body); the velocities
/// and the orientation are the mutable dynamic state, while `inverse_masses` and
/// `inverse_inertias` are the fixed per-body material properties.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RigidBodyState {
    /// World-space centre-of-mass position of each body.
    pub positions: Vec<Vec3>,
    /// World-space orientation of each body, as a unit quaternion.
    pub orientations: Vec<Quat>,
    /// World-space linear velocity of each body's centre of mass.
    pub linear_velocities: Vec<Vec3>,
    /// World-space angular velocity of each body.
    pub angular_velocities: Vec<Vec3>,
    /// Reciprocal mass of each body; `0` marks an immovable (static) body.
    pub inverse_masses: Vec<f32>,
    /// Body-frame principal-axis diagonal of each body's inverse inertia
    /// tensor; an all-zero entry marks a rotation-locked body.
    pub inverse_inertias: Vec<Vec3>,
}

impl RigidBodyState {
    /// Creates an empty rigid-body set.
    #[must_use]
    pub fn new() -> RigidBodyState {
        RigidBodyState {
            positions: Vec::new(),
            orientations: Vec::new(),
            linear_velocities: Vec::new(),
            angular_velocities: Vec::new(),
            inverse_masses: Vec::new(),
            inverse_inertias: Vec::new(),
        }
    }

    /// Appends one body at rest (zero velocity) with the given pose and material
    /// properties.
    ///
    /// `inverse_mass` is the reciprocal mass (`0` for a static body) and
    /// `inverse_inertia` is the body-frame principal-axis diagonal of the
    /// inverse inertia tensor (all-zero for a rotation-locked body).
    pub fn push(
        &mut self,
        position: Vec3,
        orientation: Quat,
        inverse_mass: f32,
        inverse_inertia: Vec3,
    ) {
        self.positions.push(position);
        self.orientations.push(orientation);
        self.linear_velocities.push(Vec3::ZERO);
        self.angular_velocities.push(Vec3::ZERO);
        self.inverse_masses.push(inverse_mass);
        self.inverse_inertias.push(inverse_inertia);
    }

    /// The number of bodies in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the set holds no bodies.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Whether every per-body array has the same length, the invariant the
    /// integrator relies on to index bodies by a single offset.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        let n = self.positions.len();
        self.orientations.len() == n
            && self.linear_velocities.len() == n
            && self.angular_velocities.len() == n
            && self.inverse_masses.len() == n
            && self.inverse_inertias.len() == n
    }
}
