//! Explicit discrete-element driver for a packing of spheres colliding under
//! the rotational contact law (friction torque plus rolling resistance).
//!
//! This is the rotational counterpart of
//! [`tangential_history_integrator`](super::tangential_history_integrator). That
//! driver advances only each grain's linear velocity, so idealised spheres roll
//! almost freely and a pile slumps too flat. This driver additionally carries
//! each grain's *angular* velocity and integrates the spin torques produced by
//! the [`rotational_contact`](super::rotational_contact) law, so a packing can
//! resist both sliding and rolling and reach a realistic angle of repose.
//!
//! # State
//!
//! A [`RotationalContactBody`] owns the kinematic state of every particle —
//! position, linear velocity and angular velocity — together with its radius,
//! mass, and the isotropic solid-sphere moment of inertia `I = ⅖·m·r²`, plus
//! the [`RollingContactResolver`](super::rotational_contact_resolver::RollingContactResolver)
//! carrying the persistent friction and rolling springs. A particle with a
//! non-finite mass is treated as *pinned*: it feels neither linear nor angular
//! acceleration and keeps its prescribed velocities, the usual way to model
//! immovable walls or scripted grains.
//!
//! # Time step
//!
//! Each [`RotationalContactBody::step`] advances the packing by `dt`:
//!
//! ```text
//!   (Fᵢ, τᵢ) = resolver.resolve(x, r, v, ω, model, dt) + external
//!   vᵢ += dt·Fᵢ/mᵢ        (pinned particles unchanged)
//!   ωᵢ += dt·τᵢ/Iᵢ        (pinned particles unchanged)
//!   xᵢ += dt·vᵢ           (symplectic position update)
//! ```
//!
//! The grains' orientations are not tracked: the rolling-resistance spring
//! accumulates the *relative* rotation internally from `ω·dt`, so no absolute
//! orientation state is required. Because the contact forces act at the shared
//! contact point and the rolling couples are equal and opposite, a step with no
//! external loads conserves both linear and angular momentum (up to the usual
//! explicit-integration error).

use crate::collider::rotational_contact::RollingContactModel;
use crate::collider::rotational_contact_resolver::{
    RollingContactResolution, RollingContactResolver,
};
use glam::Vec3;

/// Diagnostic summary of one rotational contact-integration step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotationalContactStepReport {
    /// Number of contacting pairs resolved this step.
    pub contact_count: usize,
    /// Largest overlap across all contacts this step.
    pub max_overlap: f32,
    /// Largest normal force magnitude across all contacts this step.
    pub max_normal_force: f32,
    /// Largest rolling resistance torque magnitude across all contacts this
    /// step.
    pub max_rolling_torque: f32,
    /// Number of contacts at the Coulomb (sliding) limit this step.
    pub sliding_count: usize,
    /// Total kinetic energy (translational plus rotational) of the finite-mass
    /// particles after the step.
    pub kinetic_energy: f32,
}

/// A time-steppable packing of colliding spheres under the rotational contact
/// law, carrying linear and angular velocities and persistent contact springs.
#[derive(Clone, Debug)]
pub struct RotationalContactBody {
    model: RollingContactModel,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    angular: Vec<Vec3>,
    radii: Vec<f32>,
    masses: Vec<f32>,
    inertias: Vec<f32>,
    resolver: RollingContactResolver,
}

impl RotationalContactBody {
    /// Builds a packing from a `model`, initial `positions`, per-particle
    /// `radii`, and per-particle `masses`. Linear and angular velocities start
    /// at rest and the contact history is empty. Each finite-mass particle is
    /// assigned the isotropic solid-sphere inertia `I = ⅖·m·r²`.
    ///
    /// A particle whose mass is non-finite (infinite) is *pinned*. Returns
    /// `None` when the arrays disagree in length, when any position is
    /// non-finite, when any radius is not finite and strictly positive, or when
    /// any mass is NaN or finite but non-positive.
    #[must_use]
    pub fn new(
        model: RollingContactModel,
        positions: Vec<Vec3>,
        radii: Vec<f32>,
        masses: Vec<f32>,
    ) -> Option<Self> {
        let n = positions.len();
        if radii.len() != n || masses.len() != n {
            return None;
        }
        if positions
            .iter()
            .any(|p| !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()))
        {
            return None;
        }
        if !radii.iter().all(|&r| r.is_finite() && r > 0.0) {
            return None;
        }
        // A mass is valid if it is NaN-free and either infinite (pinned) or
        // strictly positive.
        let valid_mass = |m: f32| !m.is_nan() && (m.is_infinite() || m > 0.0);
        if !masses.iter().copied().all(valid_mass) {
            return None;
        }
        // Isotropic solid-sphere inertia. A pinned (infinite-mass) particle
        // gets an infinite inertia, so it is skipped in the angular update just
        // as it is in the linear update.
        let inertias: Vec<f32> = masses
            .iter()
            .zip(radii.iter())
            .map(|(&m, &r)| 0.4 * m * r * r)
            .collect();
        Some(Self {
            model,
            velocities: vec![Vec3::ZERO; n],
            angular: vec![Vec3::ZERO; n],
            positions,
            radii,
            masses,
            inertias,
            resolver: RollingContactResolver::new(),
        })
    }

    /// Number of particles in the packing.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Current particle positions.
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Current particle linear velocities.
    #[must_use]
    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }

    /// Current particle angular velocities.
    #[must_use]
    pub fn angular_velocities(&self) -> &[Vec3] {
        &self.angular
    }

    /// Particle radii.
    #[must_use]
    pub fn radii(&self) -> &[f32] {
        &self.radii
    }

    /// Isotropic solid-sphere moment of inertia `I = ⅖·m·r²` of each particle
    /// (infinite for pinned particles).
    #[must_use]
    pub fn inertias(&self) -> &[f32] {
        &self.inertias
    }

    /// Number of live contact-spring pairs currently tracked.
    #[must_use]
    pub fn active_contacts(&self) -> usize {
        self.resolver.active_contacts()
    }

    /// Whether particle `i` is pinned (non-finite mass).
    #[must_use]
    pub fn is_pinned(&self, i: usize) -> bool {
        self.masses.get(i).is_some_and(|m| !m.is_finite())
    }

    /// Sets the linear velocity of particle `i`, if it exists.
    pub fn set_velocity(&mut self, i: usize, velocity: Vec3) {
        if let Some(v) = self.velocities.get_mut(i) {
            *v = velocity;
        }
    }

    /// Sets the angular velocity of particle `i`, if it exists.
    pub fn set_angular_velocity(&mut self, i: usize, angular: Vec3) {
        if let Some(w) = self.angular.get_mut(i) {
            *w = angular;
        }
    }

    /// Total translational kinetic energy of the finite-mass particles.
    #[must_use]
    pub fn translational_kinetic_energy(&self) -> f32 {
        let mut energy = 0.0;
        for (m, v) in self.masses.iter().zip(self.velocities.iter()) {
            if m.is_finite() {
                energy += 0.5 * m * v.length_squared();
            }
        }
        energy
    }

    /// Total rotational kinetic energy of the finite-mass particles.
    #[must_use]
    pub fn rotational_kinetic_energy(&self) -> f32 {
        let mut energy = 0.0;
        for (inertia, w) in self.inertias.iter().zip(self.angular.iter()) {
            if inertia.is_finite() {
                energy += 0.5 * inertia * w.length_squared();
            }
        }
        energy
    }

    /// Total kinetic energy (translational plus rotational) of the finite-mass
    /// particles.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        self.translational_kinetic_energy() + self.rotational_kinetic_energy()
    }

    /// Total linear momentum of the finite-mass particles.
    #[must_use]
    pub fn linear_momentum(&self) -> Vec3 {
        let mut p = Vec3::ZERO;
        for (m, v) in self.masses.iter().zip(self.velocities.iter()) {
            if m.is_finite() {
                p += *m * *v;
            }
        }
        p
    }

    /// Total angular momentum of the finite-mass particles about the origin,
    /// `Σ (mᵢ·xᵢ×vᵢ + Iᵢ·ωᵢ)`.
    #[must_use]
    pub fn angular_momentum(&self) -> Vec3 {
        let mut l = Vec3::ZERO;
        for i in 0..self.positions.len() {
            if self.masses[i].is_finite() {
                l += self.masses[i] * self.positions[i].cross(self.velocities[i]);
                l += self.inertias[i] * self.angular[i];
            }
        }
        l
    }

    /// Advances the packing by `dt` under optional per-particle external forces
    /// and torques, re-resolving the contact set and advancing the springs
    /// once.
    ///
    /// `external_forces` and `external_torques` must each have one entry per
    /// particle. Returns `None` when `dt` is not strictly positive and finite,
    /// when either external slice has the wrong length, or when the internal
    /// resolver rejects the state (which cannot happen for a well-formed body).
    /// On `None` the body is left unchanged.
    #[must_use]
    pub fn step(
        &mut self,
        dt: f32,
        external_forces: &[Vec3],
        external_torques: &[Vec3],
    ) -> Option<RotationalContactStepReport> {
        let n = self.positions.len();
        if !(dt.is_finite() && dt > 0.0) {
            return None;
        }
        if external_forces.len() != n || external_torques.len() != n {
            return None;
        }

        // Disjoint field borrows: `resolver` mutably, kinematic state immutably.
        let resolution: RollingContactResolution = self.resolver.resolve(
            &self.positions,
            &self.radii,
            &self.velocities,
            &self.angular,
            &self.model,
            dt,
        )?;

        // Integrate linear and angular velocities (pinned particles untouched),
        // then positions, giving the step its symplectic character.
        for i in 0..n {
            if self.masses[i].is_finite() {
                let force = resolution.forces[i] + external_forces[i];
                self.velocities[i] += (dt / self.masses[i]) * force;
                let torque = resolution.torques[i] + external_torques[i];
                self.angular[i] += (dt / self.inertias[i]) * torque;
            }
        }
        for (pos, &vel) in self.positions.iter_mut().zip(self.velocities.iter()) {
            *pos += vel * dt;
        }

        Some(RotationalContactStepReport {
            contact_count: resolution.contact_count(),
            max_overlap: resolution.max_overlap(),
            max_normal_force: resolution.max_normal_force(),
            max_rolling_torque: resolution.max_rolling_torque(),
            sliding_count: resolution.sliding_count(),
            kinetic_energy: self.kinetic_energy(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> RollingContactModel {
        // (kₙ,γₙ), (k_t,γ_t,μ), (k_r,γ_r,μ_r) — damped so packings settle.
        RollingContactModel::new((1.0e4, 50.0), (1.0e4, 50.0, 0.6), (1.0e3, 20.0, 0.3)).unwrap()
    }

    fn zeros(n: usize) -> Vec<Vec3> {
        vec![Vec3::ZERO; n]
    }

    #[test]
    fn new_rejects_mismatched_arrays() {
        assert!(RotationalContactBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::X],
            vec![1.0],
            vec![1.0, 1.0],
        )
        .is_none());
    }

    #[test]
    fn new_rejects_bad_radius_mass_and_position() {
        assert!(
            RotationalContactBody::new(model(), vec![Vec3::ZERO], vec![0.0], vec![1.0]).is_none()
        );
        assert!(
            RotationalContactBody::new(model(), vec![Vec3::ZERO], vec![1.0], vec![0.0]).is_none()
        );
        assert!(RotationalContactBody::new(
            model(),
            vec![Vec3::new(f32::NAN, 0.0, 0.0)],
            vec![1.0],
            vec![1.0],
        )
        .is_none());
        // Infinite mass (pinned) is accepted.
        assert!(RotationalContactBody::new(
            model(),
            vec![Vec3::ZERO],
            vec![1.0],
            vec![f32::INFINITY]
        )
        .is_some());
    }

    #[test]
    fn inertia_is_two_fifths_m_r_squared() {
        let body = RotationalContactBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)],
            vec![2.0, 1.0],
            vec![3.0, f32::INFINITY],
        )
        .expect("valid");
        assert!((body.inertias()[0] - 0.4 * 3.0 * 2.0 * 2.0).abs() < 1e-5);
        assert!(
            !body.inertias()[1].is_finite(),
            "pinned inertia is infinite"
        );
    }

    #[test]
    fn step_rejects_bad_dt_and_external_length() {
        let mut body =
            RotationalContactBody::new(model(), vec![Vec3::ZERO], vec![1.0], vec![1.0]).unwrap();
        assert!(body.step(0.0, &zeros(1), &zeros(1)).is_none());
        assert!(body.step(f32::NAN, &zeros(1), &zeros(1)).is_none());
        assert!(body.step(1.0e-3, &zeros(2), &zeros(1)).is_none());
        assert!(body.step(1.0e-3, &zeros(1), &zeros(2)).is_none());
    }

    #[test]
    fn overlapping_pair_repels_and_conserves_linear_momentum() {
        let mut body = RotationalContactBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::new(1.9, 0.0, 0.0)],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .unwrap();
        let report = body.step(1.0e-4, &zeros(2), &zeros(2)).expect("valid");
        assert_eq!(report.contact_count, 1);
        assert!(report.max_normal_force > 0.0);
        // The normal penalty pushes the grains apart along x.
        assert!(body.velocities()[0].x < 0.0);
        assert!(body.velocities()[1].x > 0.0);
        // No external load, equal masses: total linear momentum stays ≈ 0.
        assert!(body.linear_momentum().length() < 1e-3);
    }

    #[test]
    fn external_torque_spins_up_a_free_grain() {
        let mut body =
            RotationalContactBody::new(model(), vec![Vec3::ZERO], vec![1.0], vec![2.0]).unwrap();
        let inertia = body.inertias()[0];
        let torque = Vec3::new(0.0, 0.0, 5.0);
        let dt = 1.0e-3;
        let _ = body.step(dt, &zeros(1), &[torque]).expect("valid");
        // ω_z = dt·τ_z/I for a free grain with no contacts.
        let expected = dt * torque.z / inertia;
        assert!((body.angular_velocities()[0].z - expected).abs() < 1e-6);
    }

    #[test]
    fn isolated_spinning_grain_conserves_angular_velocity() {
        let mut body =
            RotationalContactBody::new(model(), vec![Vec3::ZERO], vec![1.0], vec![1.0]).unwrap();
        body.set_angular_velocity(0, Vec3::new(0.0, 0.0, 2.0));
        for _ in 0..100 {
            let _ = body.step(1.0e-3, &zeros(1), &zeros(1)).expect("valid");
        }
        // No contacts, no external torque: spin is unchanged.
        assert!((body.angular_velocities()[0].z - 2.0).abs() < 1e-5);
    }

    #[test]
    fn pinned_grain_keeps_its_velocities() {
        let mut body = RotationalContactBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::new(1.9, 0.0, 0.0)],
            vec![1.0, 1.0],
            vec![f32::INFINITY, 1.0],
        )
        .unwrap();
        body.set_velocity(0, Vec3::new(0.3, 0.0, 0.0));
        body.set_angular_velocity(0, Vec3::new(0.0, 0.0, 1.0));
        let _ = body.step(1.0e-4, &zeros(2), &zeros(2)).expect("valid");
        // The pinned grain ignores contact forces and torques entirely.
        assert_eq!(body.velocities()[0], Vec3::new(0.3, 0.0, 0.0));
        assert_eq!(body.angular_velocities()[0], Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn rolling_resistance_damps_relative_spin() {
        let mut body = RotationalContactBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::new(1.9, 0.0, 0.0)],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .unwrap();
        // Counter-rotating grains in contact: both surface friction and rolling
        // resistance oppose the relative spin.
        body.set_angular_velocity(0, Vec3::new(0.0, 0.0, 2.0));
        body.set_angular_velocity(1, Vec3::new(0.0, 0.0, -2.0));
        let initial = (body.angular_velocities()[0] - body.angular_velocities()[1]).length();
        let mut saw_rolling = false;
        for _ in 0..2000 {
            let report = body.step(1.0e-4, &zeros(2), &zeros(2)).expect("valid");
            if report.max_rolling_torque > 0.0 {
                saw_rolling = true;
            }
        }
        let final_rel = (body.angular_velocities()[0] - body.angular_velocities()[1]).length();
        assert!(saw_rolling, "rolling resistance engaged at some step");
        assert!(
            final_rel < initial,
            "relative spin decayed: {initial} -> {final_rel}"
        );
    }

    #[test]
    fn kinetic_energy_counts_translation_and_rotation() {
        let mut body =
            RotationalContactBody::new(model(), vec![Vec3::ZERO], vec![1.0], vec![2.0]).unwrap();
        body.set_velocity(0, Vec3::new(3.0, 0.0, 0.0));
        body.set_angular_velocity(0, Vec3::new(0.0, 0.0, 4.0));
        let inertia = body.inertias()[0];
        let expected_trans = 0.5 * 2.0 * 9.0;
        let expected_rot = 0.5 * inertia * 16.0;
        assert!((body.translational_kinetic_energy() - expected_trans).abs() < 1e-4);
        assert!((body.rotational_kinetic_energy() - expected_rot).abs() < 1e-4);
        assert!((body.kinetic_energy() - (expected_trans + expected_rot)).abs() < 1e-4);
    }

    #[test]
    fn free_contact_step_dissipates_energy() {
        // Two overlapping grains with closing velocity: the damped contact must
        // remove kinetic energy over the step rather than inject it.
        let mut body = RotationalContactBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .unwrap();
        body.set_velocity(0, Vec3::new(0.5, 0.0, 0.0));
        body.set_velocity(1, Vec3::new(-0.5, 0.0, 0.0));
        let before = body.kinetic_energy();
        // Grains start exactly touching (no stored penalty energy), so the only
        // energy change comes from the damped collision.
        let mut energy = before;
        for _ in 0..500 {
            let report = body.step(1.0e-4, &zeros(2), &zeros(2)).expect("valid");
            energy = report.kinetic_energy;
        }
        assert!(energy < before, "damped collision dissipated energy");
    }
}
