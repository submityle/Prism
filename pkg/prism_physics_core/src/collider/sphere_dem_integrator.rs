//! A translational discrete-element (DEM) time integrator for a sphere cloud.
//!
//! This composes the decoupled contact stack into a usable simulation step:
//! each call runs the broad-phase-accelerated
//! [`SphereNarrowPhase`](crate::collider::sphere_narrow_phase::SphereNarrowPhase)
//! to find the touching pairs, evaluates the Hertz force on each pair with
//! [`resolve_sphere_contact_forces`](crate::collider::sphere_contact_forces::resolve_sphere_contact_forces),
//! adds a uniform body force (gravity), and advances the cloud with a
//! semi-implicit (symplectic) Euler update.
//!
//! The integrator owns no force law of its own — it reuses the single Hertz
//! source of truth through the force resolver — and it tracks only translation
//! (no spin), so grain–grain friction torque and boundary contacts are handled
//! by separate drivers. Keeping the step this thin makes it a clean building
//! block: a caller can drop in different narrow phases, body forces, or
//! integration orders without the force law changing underneath them.

use glam::Vec3;

use crate::collider::hertz_contact::HertzModel;
use crate::collider::sphere_contact_forces::resolve_sphere_contact_forces;
use crate::collider::sphere_narrow_phase::SphereNarrowPhase;

/// The mutable state of a sphere cloud: index-aligned positions, velocities,
/// radii, and masses.
#[derive(Clone, Debug, PartialEq)]
pub struct SphereDemState {
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    radii: Vec<f32>,
    masses: Vec<f32>,
}

impl SphereDemState {
    /// Builds a cloud from explicit per-grain masses.
    ///
    /// All four slices must share one length. Every position and velocity must
    /// be finite, every radius finite and strictly positive, and every mass
    /// finite and strictly positive. Returns `None` otherwise.
    #[must_use]
    pub fn new(
        positions: Vec<Vec3>,
        velocities: Vec<Vec3>,
        radii: Vec<f32>,
        masses: Vec<f32>,
    ) -> Option<Self> {
        let n = positions.len();
        if velocities.len() != n || radii.len() != n || masses.len() != n {
            return None;
        }
        for i in 0..n {
            if !positions[i].is_finite() || !velocities[i].is_finite() {
                return None;
            }
            if !radii[i].is_finite() || radii[i] <= 0.0 {
                return None;
            }
            if !masses[i].is_finite() || masses[i] <= 0.0 {
                return None;
            }
        }
        Some(Self {
            positions,
            velocities,
            radii,
            masses,
        })
    }

    /// Builds a cloud whose masses follow from a uniform `density` and each
    /// grain's volume `4/3 · π · r³`.
    ///
    /// `density` must be finite and strictly positive; the position, velocity,
    /// and radius validation matches [`SphereDemState::new`]. Returns `None`
    /// otherwise.
    #[must_use]
    pub fn from_density(
        positions: Vec<Vec3>,
        velocities: Vec<Vec3>,
        radii: Vec<f32>,
        density: f32,
    ) -> Option<Self> {
        if !density.is_finite() || density <= 0.0 {
            return None;
        }
        let mut masses = Vec::with_capacity(radii.len());
        for &r in radii.iter() {
            if !r.is_finite() || r <= 0.0 {
                return None;
            }
            let volume = (4.0 / 3.0) * core::f32::consts::PI * r * r * r;
            masses.push(density * volume);
        }
        Self::new(positions, velocities, radii, masses)
    }

    /// Number of grains in the cloud.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the cloud is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Grain centres.
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Grain velocities.
    #[must_use]
    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }

    /// Grain radii.
    #[must_use]
    pub fn radii(&self) -> &[f32] {
        &self.radii
    }

    /// Grain masses.
    #[must_use]
    pub fn masses(&self) -> &[f32] {
        &self.masses
    }

    /// Total linear momentum `Σ mᵢ vᵢ`.
    #[must_use]
    pub fn linear_momentum(&self) -> Vec3 {
        self.masses
            .iter()
            .zip(self.velocities.iter())
            .fold(Vec3::ZERO, |acc, (&m, v)| acc + m * *v)
    }

    /// Total translational kinetic energy `Σ ½ mᵢ |vᵢ|²`.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        self.masses
            .iter()
            .zip(self.velocities.iter())
            .map(|(&m, v)| 0.5 * m * v.length_squared())
            .sum()
    }
}

/// Diagnostics produced by a single [`SphereDemIntegrator::step`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereDemStepReport {
    contact_count: u32,
    max_normal_force: f32,
    sliding_count: u32,
    max_speed: f32,
    kinetic_energy: f32,
}

impl SphereDemStepReport {
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

    /// Number of contacts whose friction reached the Coulomb limit.
    #[must_use]
    pub fn sliding_count(&self) -> u32 {
        self.sliding_count
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
/// Hertz grain–grain contact.
#[derive(Clone, Debug)]
pub struct SphereDemIntegrator {
    model: HertzModel,
    gravity: Vec3,
    narrow_phase: SphereNarrowPhase,
}

impl SphereDemIntegrator {
    /// Builds an integrator with the given contact `model` and uniform body
    /// acceleration `gravity` (applied as a force `mᵢ · gravity` per grain).
    #[must_use]
    pub fn new(model: HertzModel, gravity: Vec3) -> Self {
        Self {
            model,
            gravity,
            narrow_phase: SphereNarrowPhase::new(),
        }
    }

    /// The uniform body acceleration applied each step.
    #[must_use]
    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    /// Advances `state` by `dt` with a semi-implicit (symplectic) Euler step.
    ///
    /// The grain–grain contact set is rebuilt from the current positions, the
    /// Hertz force on each contact is accumulated per grain, gravity is added,
    /// velocities are updated from the resulting acceleration, and positions
    /// follow the *new* velocities. `dt` must be finite and strictly positive;
    /// returns `None` on invalid `dt` or if the contact pipeline rejects the
    /// cloud (which cannot happen for a state built through this module's
    /// validated constructors, but is surfaced defensively).
    #[must_use]
    pub fn step(&mut self, state: &mut SphereDemState, dt: f32) -> Option<SphereDemStepReport> {
        if !dt.is_finite() || dt <= 0.0 {
            return None;
        }

        let resolution = {
            let contacts = self
                .narrow_phase
                .detect(&state.positions, &state.radii, 0.0)?;
            resolve_sphere_contact_forces(contacts, &state.radii, &state.velocities, &self.model)?
        };
        let forces = resolution.forces();

        let mut max_speed = 0.0_f32;
        for (i, force) in forces.iter().enumerate() {
            let total = *force + state.masses[i] * self.gravity;
            let acceleration = total / state.masses[i];
            state.velocities[i] += acceleration * dt;
            state.positions[i] += state.velocities[i] * dt;
            max_speed = max_speed.max(state.velocities[i].length());
        }

        Some(SphereDemStepReport {
            contact_count: resolution.contact_count(),
            max_normal_force: resolution.max_normal_force(),
            sliding_count: resolution.sliding_count(),
            max_speed,
            kinetic_energy: state.kinetic_energy(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> HertzModel {
        HertzModel::new(1.0e7, 0.3, 10.0, 10.0, 0.5).unwrap()
    }

    #[test]
    fn new_validates_cloud() {
        // Mismatched lengths.
        assert!(SphereDemState::new(
            vec![Vec3::ZERO],
            vec![Vec3::ZERO, Vec3::ZERO],
            vec![1.0],
            vec![1.0],
        )
        .is_none());
        // Non-positive radius.
        assert!(
            SphereDemState::new(vec![Vec3::ZERO], vec![Vec3::ZERO], vec![0.0], vec![1.0],)
                .is_none()
        );
        // Non-positive mass.
        assert!(
            SphereDemState::new(vec![Vec3::ZERO], vec![Vec3::ZERO], vec![1.0], vec![0.0],)
                .is_none()
        );
        // Non-finite position.
        assert!(SphereDemState::new(
            vec![Vec3::new(f32::NAN, 0.0, 0.0)],
            vec![Vec3::ZERO],
            vec![1.0],
            vec![1.0],
        )
        .is_none());
    }

    #[test]
    fn from_density_matches_sphere_volume() {
        let radii = vec![2.0_f32];
        let density = 3.0_f32;
        let state =
            SphereDemState::from_density(vec![Vec3::ZERO], vec![Vec3::ZERO], radii, density)
                .unwrap();
        let expected = density * (4.0 / 3.0) * core::f32::consts::PI * 2.0 * 2.0 * 2.0;
        assert!((state.masses()[0] - expected).abs() < 1.0e-3);
    }

    #[test]
    fn from_density_rejects_bad_density() {
        assert!(
            SphereDemState::from_density(vec![Vec3::ZERO], vec![Vec3::ZERO], vec![1.0], 0.0,)
                .is_none()
        );
    }

    #[test]
    fn rejects_invalid_dt() {
        let mut integrator = SphereDemIntegrator::new(model(), Vec3::new(0.0, 0.0, -9.81));
        let mut state =
            SphereDemState::new(vec![Vec3::ZERO], vec![Vec3::ZERO], vec![1.0], vec![1.0]).unwrap();
        assert!(integrator.step(&mut state, 0.0).is_none());
        assert!(integrator.step(&mut state, -0.01).is_none());
        assert!(integrator.step(&mut state, f32::NAN).is_none());
    }

    #[test]
    fn free_grain_follows_gravity() {
        let gravity = Vec3::new(0.0, 0.0, -9.81);
        let mut integrator = SphereDemIntegrator::new(model(), gravity);
        let mut state =
            SphereDemState::new(vec![Vec3::ZERO], vec![Vec3::ZERO], vec![1.0], vec![2.0]).unwrap();
        let dt = 0.01_f32;
        let report = integrator.step(&mut state, dt).unwrap();
        // No contacts; velocity gains exactly gravity·dt regardless of mass.
        assert_eq!(report.contact_count(), 0);
        let v = state.velocities()[0];
        assert!((v - gravity * dt).length() < 1.0e-6);
        // Semi-implicit: position follows the updated velocity.
        assert!((state.positions()[0] - v * dt).length() < 1.0e-6);
    }

    #[test]
    fn overlapping_pair_repels_and_conserves_momentum() {
        // Zero gravity isolates the internal contact force.
        let mut integrator = SphereDemIntegrator::new(model(), Vec3::ZERO);
        let positions = vec![Vec3::ZERO, Vec3::new(1.9, 0.0, 0.0)]; // overlap 0.1
        let velocities = vec![Vec3::ZERO, Vec3::ZERO];
        let radii = vec![1.0_f32, 1.0];
        let masses = vec![1.0_f32, 1.0];
        let mut state = SphereDemState::new(positions, velocities, radii, masses).unwrap();
        let momentum_before = state.linear_momentum();
        let report = integrator.step(&mut state, 0.001).unwrap();
        assert_eq!(report.contact_count(), 1);
        // Grain 0 pushed toward -x, grain 1 toward +x.
        assert!(state.velocities()[0].x < 0.0);
        assert!(state.velocities()[1].x > 0.0);
        // Internal forces are equal and opposite: momentum is preserved.
        let momentum_after = state.linear_momentum();
        assert!((momentum_after - momentum_before).length() < 1.0e-5);
    }

    #[test]
    fn asymmetric_masses_conserve_momentum() {
        let mut integrator = SphereDemIntegrator::new(model(), Vec3::ZERO);
        let positions = vec![Vec3::ZERO, Vec3::new(1.9, 0.0, 0.0)];
        let velocities = vec![Vec3::new(0.2, 0.0, 0.0), Vec3::new(-0.1, 0.0, 0.0)];
        let radii = vec![1.0_f32, 1.0];
        let masses = vec![1.0_f32, 4.0];
        let mut state = SphereDemState::new(positions, velocities, radii, masses).unwrap();
        let momentum_before = state.linear_momentum();
        integrator.step(&mut state, 0.001).unwrap();
        let momentum_after = state.linear_momentum();
        assert!((momentum_after - momentum_before).length() < 1.0e-5);
    }

    #[test]
    fn report_tracks_speed_and_energy() {
        let gravity = Vec3::new(0.0, 0.0, -9.81);
        let mut integrator = SphereDemIntegrator::new(model(), gravity);
        let mut state =
            SphereDemState::new(vec![Vec3::ZERO], vec![Vec3::ZERO], vec![1.0], vec![2.0]).unwrap();
        let dt = 0.02_f32;
        let report = integrator.step(&mut state, dt).unwrap();
        let v = state.velocities()[0].length();
        assert!((report.max_speed() - v).abs() < 1.0e-6);
        let expected_energy = 0.5 * 2.0 * v * v;
        assert!((report.kinetic_energy() - expected_energy).abs() < 1.0e-4);
        assert!(report.kinetic_energy() > 0.0);
    }
}
