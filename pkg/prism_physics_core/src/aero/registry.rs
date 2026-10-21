//! Per-body rigid-aerodynamics authoring data.
//!
//! [`AeroBodyConfig`] is deliberately just data. It can be authored by an
//! ECS/editor layer, copied into a [`PhysicsWorld`](crate::PhysicsWorld), and
//! evaluated by the pure functions in this module without coupling those
//! functions to a particular storage or solver.

use glam::Vec3;

use super::apply::{density, fluid_fraction, sanitized_state, surface_force};
use super::sanitize::{add, finite, non_negative, vec3};
use super::{
    compute_aero_wrench, AeroBodyState, AeroModel, AeroSurface, AeroWrench, BuoyancyVolume,
    FlowField, Real, Thruster,
};

/// All M12 aerodynamic elements attached to one rigid body.
#[derive(Clone, Debug, PartialEq)]
pub struct AeroBodyConfig {
    pub surfaces: Vec<AeroSurface>,
    pub thrusters: Vec<Thruster>,
    pub buoyancy: Vec<BuoyancyVolume>,
    pub model: AeroModel,
}

impl Default for AeroBodyConfig {
    fn default() -> Self {
        Self {
            surfaces: Vec::new(),
            thrusters: Vec::new(),
            buoyancy: Vec::new(),
            model: AeroModel::default(),
        }
    }
}

impl AeroBodyConfig {
    /// Creates an empty configuration using `model` for fluid constants and
    /// optional added-mass/Magnus effects.
    #[must_use]
    pub fn new(model: AeroModel) -> Self {
        Self {
            model,
            ..Self::default()
        }
    }

    /// Appends one surface element while retaining builder-style authoring.
    #[must_use]
    pub fn with_surface(mut self, surface: AeroSurface) -> Self {
        self.surfaces.push(surface);
        self
    }

    /// Appends one thruster while retaining builder-style authoring.
    #[must_use]
    pub fn with_thruster(mut self, thruster: Thruster) -> Self {
        self.thrusters.push(thruster);
        self
    }

    /// Appends one buoyancy volume while retaining builder-style authoring.
    #[must_use]
    pub fn with_buoyancy(mut self, volume: BuoyancyVolume) -> Self {
        self.buoyancy.push(volume);
        self
    }

    /// Evaluates this body's complete rigid-aero contribution at one field
    /// time, keeping world integration outside the authoring configuration.
    #[must_use]
    pub fn compute_wrench<F: FlowField + ?Sized>(
        &self,
        state: &AeroBodyState,
        field: &F,
        time: Real,
    ) -> AeroWrench {
        compute_aero_wrench(
            state,
            &self.surfaces,
            &self.thrusters,
            &self.buoyancy,
            field,
            &self.model,
            time,
        )
    }

    /// Evaluates and bounds one body's wrench for a fixed-step adapter.
    #[must_use]
    pub fn compute_bounded_wrench<F: FlowField + ?Sized>(
        &self,
        state: &AeroBodyState,
        field: &F,
        time: Real,
        maximum_force: Real,
        maximum_torque: Real,
    ) -> AeroWrench {
        self.compute_wrench(state, field, time)
            .bounded(maximum_force, maximum_torque)
    }

    /// Returns the local inverse mass and inverse inertia after applying this
    /// configuration's optional implicit added-mass correction.
    #[must_use]
    pub fn implicit_inverse_properties(
        &self,
        inverse_mass: Vec3,
        inverse_inertia: Vec3,
        immersion: Real,
    ) -> (Vec3, Vec3) {
        let Some(added) = self.model.added_mass else {
            return (
                inverse_mass.max(Vec3::ZERO),
                inverse_inertia.max(Vec3::ZERO),
            );
        };
        let density = density(&self.model, added.fluid);
        (
            added.implicit_inverse_mass(inverse_mass, density, immersion),
            added.implicit_inverse_inertia(inverse_inertia, density, immersion),
        )
    }

    /// Returns the fluid fraction used by this configuration's added-mass
    /// model at `position`. World and solver adapters should pass this exact
    /// value to [`Self::implicit_inverse_properties`] so force and impulse
    /// paths use the same water-plane and reference-radius rule.
    #[must_use]
    pub fn added_mass_immersion(&self, position: Vec3) -> Real {
        let Some(added) = self.model.added_mass else {
            return 0.0;
        };
        fluid_fraction(position, added.fluid, added.reference_radius, &self.model)
    }

    /// Fixed-step force adapter with bounded panel impulses and an implicit
    /// added-mass solve. `external` is the already queued non-aero wrench;
    /// `gravity` is the acceleration the predictor will apply separately.
    ///
    /// The returned wrench is additive: do not requeue `external`. Velocity
    /// history and the acceleration fields in `state` are deliberately ignored.
    /// Base inverse principal inertia is in body coordinates. Contacts and
    /// impulses still require the effective inverse properties at their solver
    /// boundary; this adapter corrects only force-driven acceleration.
    #[must_use]
    pub fn compute_substep_wrench<F: FlowField + ?Sized>(
        &self,
        state: &AeroBodyState,
        field: &F,
        time: Real,
        dt: Real,
        inverse_mass: Real,
        inverse_inertia: Vec3,
        external: AeroWrench,
        gravity: Vec3,
    ) -> AeroWrench {
        if !dt.is_finite() || dt <= 0.0 {
            return AeroWrench::ZERO;
        }
        let mut state = sanitized_state(*state);
        state.linear_acceleration = Vec3::ZERO;
        state.angular_acceleration = Vec3::ZERO;
        let inverse_mass = non_negative(inverse_mass);
        let inverse_inertia = vec3(inverse_inertia).max(Vec3::ZERO);
        let mut model = self.model.clone();
        model.added_mass = None;
        let mut wrench = compute_aero_wrench(
            &state,
            &[],
            &self.thrusters,
            &self.buoyancy,
            field,
            &model,
            time,
        );
        let orientation = state.orientation;
        let inverse = orientation.conjugate();
        let count = self.surfaces.len().max(1) as f64;
        for surface in &self.surfaces {
            let (drag, lift, point) = surface_force(&state, surface, field, &model, time);
            let offset = point - state.position;
            let relative = vec3(field.sample(point, time).velocity)
                - (state.linear_velocity + state.angular_velocity.cross(offset));
            let speed = length64(relative);
            let force = add(drag, lift);
            let magnitude = length64(force);
            if magnitude == 0.0 || !speed.is_finite() || speed == 0.0 {
                continue;
            }
            let direction = force / magnitude as Real;
            let moment = inverse * offset.cross(direction);
            let effective_inverse =
                f64::from(inverse_mass) + weighted_square_sum(moment, inverse_inertia);
            // A shared per-panel budget prevents coincident panels from each
            // consuming an entire relative-velocity change. The rational form
            // approaches but cannot exceed that budget even for huge forces.
            let ratio = f64::from(dt) * effective_inverse * magnitude * count / speed;
            let scale = (1.0 / (1.0 + ratio)) as Real;
            wrench.add_force_at_point(force * scale, point, state.position);
        }

        let Some(added) = self.model.added_mass else {
            return wrench;
        };
        let fraction = self.added_mass_immersion(state.position);
        let rho = density(&self.model, added.fluid);
        let added_mass = added.mass_diagonal(rho, fraction);
        let added_inertia = added.inertia_diagonal(rho, fraction);
        let force = inverse * add(wrench.force, external.force);
        let torque = inverse * add(wrench.torque, external.torque);
        let gravity = inverse * vec3(gravity);
        let fluid_acceleration = inverse * vec3(field.sample(state.position, time).acceleration);
        let correction = |load: Real, acceleration: Real, fluid: Real, inv: Real, added: Real| {
            // a = (inv*F + g + inv*M_added*a_fluid) / (1 + inv*M_added)
            // Return only the force correction, leaving queued loads unchanged.
            let added = f64::from(added);
            let inv = f64::from(inv);
            let value = added
                * (f64::from(fluid) - f64::from(acceleration) - inv * f64::from(finite(load)))
                / (1.0 + inv * added);
            value.clamp(-f64::from(Real::MAX), f64::from(Real::MAX)) as Real
        };
        let mut added_force = Vec3::ZERO;
        let mut added_torque = Vec3::ZERO;
        for axis in 0..3 {
            if inverse_mass > 0.0 {
                added_force[axis] = correction(
                    force[axis],
                    gravity[axis],
                    fluid_acceleration[axis],
                    inverse_mass,
                    added_mass[axis],
                );
            }
            if inverse_inertia[axis] > 0.0 {
                added_torque[axis] = correction(
                    torque[axis],
                    0.0,
                    0.0,
                    inverse_inertia[axis],
                    added_inertia[axis],
                );
            }
        }
        wrench.add_force_at_point(orientation * added_force, state.position, state.position);
        wrench.add_torque(orientation * added_torque);
        wrench
    }
}

fn length64(value: Vec3) -> f64 {
    let value = vec3(value);
    (f64::from(value.x) * f64::from(value.x)
        + f64::from(value.y) * f64::from(value.y)
        + f64::from(value.z) * f64::from(value.z))
    .sqrt()
}

fn weighted_square_sum(value: Vec3, weights: Vec3) -> f64 {
    f64::from(value.x) * f64::from(value.x) * f64::from(weights.x)
        + f64::from(value.y) * f64::from(value.y) * f64::from(weights.y)
        + f64::from(value.z) * f64::from(value.z) * f64::from(weights.z)
}
