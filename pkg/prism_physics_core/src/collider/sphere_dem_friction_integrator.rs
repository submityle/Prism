//! A frictional discrete-element (DEM) time integrator for a sphere cloud.
//!
//! [`SphereDemIntegrator`](crate::collider::sphere_dem_integrator::SphereDemIntegrator)
//! advances a cloud under the memoryless Hertz force. Real granular media also
//! carry *tangential history*: a grain resting on a pile develops a growing
//! static-friction force that keeps it in place until it reaches the Coulomb
//! cone and slips. Capturing that needs persistent per-contact springs, which
//! this integrator holds through a
//! [`SphereCundallStrackDriver`](crate::collider::sphere_cundall_strack_driver::SphereCundallStrackDriver).
//!
//! Each step rebuilds the contact set with
//! [`SphereNarrowPhase`](crate::collider::sphere_narrow_phase::SphereNarrowPhase),
//! evaluates the Cundall–Strack normal **and** tangential-history force on every
//! contact (advancing the stored springs), adds gravity, and advances the cloud
//! with a semi-implicit (symplectic) Euler update. The cloud representation is
//! shared with the frictionless integrator through
//! [`SphereDemState`], so a caller can swap force models without reshaping its
//! data.

use glam::Vec3;

use crate::collider::sphere_cundall_strack_driver::SphereCundallStrackDriver;
use crate::collider::sphere_dem_integrator::SphereDemState;
use crate::collider::sphere_narrow_phase::SphereNarrowPhase;
use crate::collider::tangential_history_contact::CundallStrackModel;

/// Diagnostics produced by a single [`SphereDemFrictionIntegrator::step`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereDemFrictionStepReport {
    contact_count: u32,
    max_normal_force: f32,
    sliding_count: u32,
    active_springs: usize,
    max_speed: f32,
    kinetic_energy: f32,
}

impl SphereDemFrictionStepReport {
    /// Number of load-bearing grain–grain contacts resolved this step.
    #[must_use]
    pub fn contact_count(&self) -> u32 {
        self.contact_count
    }

    /// Largest single-contact normal force magnitude this step.
    #[must_use]
    pub fn max_normal_force(&self) -> f32 {
        self.max_normal_force
    }

    /// Number of contacts whose friction reached the Coulomb limit (sliding).
    #[must_use]
    pub fn sliding_count(&self) -> u32 {
        self.sliding_count
    }

    /// Number of live tangential springs stored after the step.
    #[must_use]
    pub fn active_springs(&self) -> usize {
        self.active_springs
    }

    /// Largest grain speed after the update.
    #[must_use]
    pub fn max_speed(&self) -> f32 {
        self.max_speed
    }

    /// Total translational kinetic energy after the update.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        self.kinetic_energy
    }
}

/// A semi-implicit DEM time integrator for a sphere cloud under gravity and
/// Cundall–Strack frictional contact with persistent tangential history.
#[derive(Clone, Debug)]
pub struct SphereDemFrictionIntegrator {
    driver: SphereCundallStrackDriver,
    narrow_phase: SphereNarrowPhase,
    gravity: Vec3,
}

impl SphereDemFrictionIntegrator {
    /// Builds an integrator with the given Cundall–Strack `model` and uniform
    /// body acceleration `gravity` (applied as a force `mᵢ · gravity` per
    /// grain). The friction history starts empty.
    #[must_use]
    pub fn new(model: CundallStrackModel, gravity: Vec3) -> Self {
        Self {
            driver: SphereCundallStrackDriver::new(model),
            narrow_phase: SphereNarrowPhase::new(),
            gravity,
        }
    }

    /// The uniform body acceleration applied each step.
    #[must_use]
    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    /// Number of live tangential springs currently stored.
    #[must_use]
    pub fn active_springs(&self) -> usize {
        self.driver.active_springs()
    }

    /// Forgets all stored friction history.
    pub fn clear_history(&mut self) {
        self.driver.clear();
    }

    /// Advances `state` by `dt` with a semi-implicit (symplectic) Euler step.
    ///
    /// The grain–grain contact set is rebuilt from the current positions, the
    /// Cundall–Strack normal and tangential-history force on each contact is
    /// accumulated per grain (advancing the stored springs), gravity is added,
    /// velocities are updated from the resulting acceleration, and positions
    /// follow the *new* velocities. `dt` must be finite and strictly positive;
    /// returns `None` on invalid `dt` or if the contact pipeline rejects the
    /// cloud.
    #[must_use]
    pub fn step(
        &mut self,
        state: &mut SphereDemState,
        dt: f32,
    ) -> Option<SphereDemFrictionStepReport> {
        if !(dt.is_finite() && dt > 0.0) {
            return None;
        }

        let resolution = {
            let contacts = self
                .narrow_phase
                .detect(state.positions(), state.radii(), 0.0)?;
            self.driver
                .resolve(contacts, state.radii(), state.velocities(), dt)?
        };
        let forces = resolution.forces();

        let masses = state.masses().to_vec();
        let mut velocities = state.velocities().to_vec();
        let mut positions = state.positions().to_vec();
        let mut max_speed = 0.0_f32;
        for (i, force) in forces.iter().enumerate() {
            let total = *force + masses[i] * self.gravity;
            let acceleration = total / masses[i];
            velocities[i] += acceleration * dt;
            positions[i] += velocities[i] * dt;
            max_speed = max_speed.max(velocities[i].length());
        }

        // Rebuild the validated state from the advanced buffers; the radii and
        // masses are unchanged so this cannot fail for a state that was already
        // valid.
        let radii = state.radii().to_vec();
        let advanced = SphereDemState::new(positions, velocities, radii, masses)?;
        let kinetic_energy = advanced.kinetic_energy();
        *state = advanced;

        Some(SphereDemFrictionStepReport {
            contact_count: resolution.contact_count(),
            max_normal_force: resolution.max_normal_force(),
            sliding_count: resolution.sliding_count(),
            active_springs: self.driver.active_springs(),
            max_speed,
            kinetic_energy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CundallStrackModel {
        // kₙ, γₙ, k_t, γ_t, μ.
        CundallStrackModel::new(1.0e5, 5.0, 1.0e5, 5.0, 0.5).unwrap()
    }

    #[test]
    fn rejects_invalid_dt() {
        let mut integrator = SphereDemFrictionIntegrator::new(model(), Vec3::new(0.0, 0.0, -9.81));
        let mut state =
            SphereDemState::new(vec![Vec3::ZERO], vec![Vec3::ZERO], vec![1.0], vec![1.0]).unwrap();
        assert!(integrator.step(&mut state, 0.0).is_none());
        assert!(integrator.step(&mut state, -0.01).is_none());
        assert!(integrator.step(&mut state, f32::NAN).is_none());
    }

    #[test]
    fn free_grain_follows_gravity() {
        let gravity = Vec3::new(0.0, 0.0, -9.81);
        let mut integrator = SphereDemFrictionIntegrator::new(model(), gravity);
        let mut state =
            SphereDemState::new(vec![Vec3::ZERO], vec![Vec3::ZERO], vec![1.0], vec![2.0]).unwrap();
        let dt = 0.01_f32;
        let report = integrator.step(&mut state, dt).unwrap();
        assert_eq!(report.contact_count(), 0);
        assert_eq!(report.active_springs(), 0);
        let v = state.velocities()[0];
        assert!((v - gravity * dt).length() < 1.0e-6);
        assert!((state.positions()[0] - v * dt).length() < 1.0e-6);
    }

    #[test]
    fn overlapping_pair_repels_and_conserves_momentum() {
        let mut integrator = SphereDemFrictionIntegrator::new(model(), Vec3::ZERO);
        let positions = vec![Vec3::ZERO, Vec3::new(1.9, 0.0, 0.0)]; // overlap 0.1
        let velocities = vec![Vec3::ZERO, Vec3::ZERO];
        let radii = vec![1.0_f32, 1.0];
        let masses = vec![1.0_f32, 1.0];
        let mut state = SphereDemState::new(positions, velocities, radii, masses).unwrap();
        let momentum_before = state.linear_momentum();
        let report = integrator.step(&mut state, 0.001).unwrap();
        assert_eq!(report.contact_count(), 1);
        assert_eq!(report.active_springs(), 1);
        assert!(state.velocities()[0].x < 0.0);
        assert!(state.velocities()[1].x > 0.0);
        let momentum_after = state.linear_momentum();
        assert!((momentum_after - momentum_before).length() < 1.0e-4);
    }

    #[test]
    fn asymmetric_masses_conserve_momentum() {
        let mut integrator = SphereDemFrictionIntegrator::new(model(), Vec3::ZERO);
        let positions = vec![Vec3::ZERO, Vec3::new(1.9, 0.0, 0.0)];
        let velocities = vec![Vec3::new(0.2, 0.1, 0.0), Vec3::new(-0.1, -0.05, 0.0)];
        let radii = vec![1.0_f32, 1.0];
        let masses = vec![1.0_f32, 4.0];
        let mut state = SphereDemState::new(positions, velocities, radii, masses).unwrap();
        let momentum_before = state.linear_momentum();
        integrator.step(&mut state, 0.0005).unwrap();
        let momentum_after = state.linear_momentum();
        assert!((momentum_after - momentum_before).length() < 1.0e-4);
    }

    #[test]
    fn tangential_slip_builds_persistent_friction() {
        // Two grains overlapping along x, sheared along y. Friction opposes the
        // shear, so the relative tangential speed must shrink over the step and
        // the spring must persist across steps.
        let mut integrator = SphereDemFrictionIntegrator::new(model(), Vec3::ZERO);
        let positions = vec![Vec3::ZERO, Vec3::new(1.9, 0.0, 0.0)];
        let velocities = vec![Vec3::new(0.0, 0.1, 0.0), Vec3::new(0.0, -0.1, 0.0)];
        let radii = vec![1.0_f32, 1.0];
        let masses = vec![1.0_f32, 1.0];
        let mut state = SphereDemState::new(positions, velocities, radii, masses).unwrap();
        let rel_before = (state.velocities()[1] - state.velocities()[0]).y.abs();
        let dt = 1.0e-4_f32;
        for _ in 0..3 {
            let report = integrator.step(&mut state, dt).unwrap();
            assert_eq!(report.active_springs(), 1);
        }
        let rel_after = (state.velocities()[1] - state.velocities()[0]).y.abs();
        assert!(
            rel_after < rel_before,
            "friction must reduce tangential relative speed: {rel_after} !< {rel_before}"
        );
    }

    #[test]
    fn report_tracks_speed_and_energy() {
        let gravity = Vec3::new(0.0, 0.0, -9.81);
        let mut integrator = SphereDemFrictionIntegrator::new(model(), gravity);
        let mut state =
            SphereDemState::new(vec![Vec3::ZERO], vec![Vec3::ZERO], vec![1.0], vec![2.0]).unwrap();
        let dt = 0.02_f32;
        let report = integrator.step(&mut state, dt).unwrap();
        let v = state.velocities()[0].length();
        assert!((report.max_speed() - v).abs() < 1.0e-6);
        let expected_energy = 0.5 * 2.0 * v * v;
        assert!((report.kinetic_energy() - expected_energy).abs() < 1.0e-4);
    }

    #[test]
    fn clear_history_drops_springs() {
        let mut integrator = SphereDemFrictionIntegrator::new(model(), Vec3::ZERO);
        let positions = vec![Vec3::ZERO, Vec3::new(1.9, 0.0, 0.0)];
        let velocities = vec![Vec3::ZERO, Vec3::ZERO];
        let radii = vec![1.0_f32, 1.0];
        let masses = vec![1.0_f32, 1.0];
        let mut state = SphereDemState::new(positions, velocities, radii, masses).unwrap();
        integrator.step(&mut state, 0.001).unwrap();
        assert_eq!(integrator.active_springs(), 1);
        integrator.clear_history();
        assert_eq!(integrator.active_springs(), 0);
    }
}
