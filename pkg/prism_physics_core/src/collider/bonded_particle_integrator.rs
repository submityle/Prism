//! Explicit discrete-element driver for a bonded-particle body.
//!
//! This closes the bonded-particle pipeline into something that advances in
//! time: the network builder
//! ([`build_bond_network`](super::bonded_particle_network_builder::build_bond_network))
//! installs bonds, the assembly layer
//! ([`assemble_bond_forces`](super::bonded_particle_assembly::assemble_bond_forces))
//! scatters per-bond loads onto particles, and this module integrates the
//! resulting particle motion with a symplectic (semi-implicit Euler) step while
//! carrying each bond's irreversible state across steps.
//!
//! # State
//!
//! A [`BondedParticleBody`] owns the kinematic state of every particle —
//! position, linear velocity, and angular velocity — together with its mass and
//! isotropic moment of inertia, the installed bonds, and one
//! [`BondState`](super::bonded_particle::BondState) per bond. A particle with a
//! non-finite mass is treated as *pinned*: it feels no acceleration and keeps
//! its prescribed velocity, which is the usual way to impose kinematic boundary
//! conditions on a bonded solid.
//!
//! # Time step
//!
//! Each [`BondedParticleBody::step`] advances the body by `dt` using the
//! standard explicit discrete-element scheme:
//!
//! ```text
//!   Δuᵢ = vᵢ·dt,  Δθᵢ = ωᵢ·dt                 (per-particle increments)
//!   (F, τ)       = assemble_bond_forces(Δu, Δθ) + external
//!   vᵢ += dt·Fᵢ/mᵢ,  ωᵢ += dt·τᵢ/Iᵢ            (pinned particles unchanged)
//!   xᵢ += dt·vᵢ                                 (symplectic position update)
//! ```
//!
//! The relative increments fed to the parallel-bond kernel come from the
//! *current* velocities, so the bond state is advanced explicitly; the position
//! update then uses the freshly updated velocity, giving the step its symplectic
//! character. Because the bond force/torque pairs are internal and
//! equal-and-opposite, a step with no external loads conserves linear momentum
//! exactly (`Σ mᵢ Δvᵢ = dt·Σ Fᵢ = 0`).

use crate::collider::bonded_particle::{BondModel, BondState};
use crate::collider::bonded_particle_assembly::{
    assemble_bond_forces, broken_bond_count, ParticleBond,
};
use glam::Vec3;

/// Diagnostic summary of one integration step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BondStepReport {
    /// Number of bonds that broke on this step.
    pub broke_this_step: usize,
    /// Total number of bonds broken so far (cumulative).
    pub broken_total: usize,
    /// Highest extreme-fibre tensile stress across all bonds this step.
    pub max_normal_stress: f32,
    /// Highest extreme-fibre shear stress across all bonds this step.
    pub max_shear_stress: f32,
    /// Total translational + rotational kinetic energy after the step.
    pub kinetic_energy: f32,
}

/// A time-steppable bonded-particle body.
#[derive(Clone, Debug, PartialEq)]
pub struct BondedParticleBody {
    model: BondModel,
    bonds: Vec<ParticleBond>,
    states: Vec<BondState>,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    angular_velocities: Vec<Vec3>,
    masses: Vec<f32>,
    moments: Vec<f32>,
}

impl BondedParticleBody {
    /// Builds a body from a bond `model`, the installed `bonds`, initial
    /// `positions`, per-particle `masses`, and per-particle isotropic moments of
    /// inertia `moments`. Velocities start at rest; bonds start intact.
    ///
    /// A particle whose mass (or moment) is non-finite is *pinned* in that
    /// degree of freedom. Returns `None` when the arrays disagree in length,
    /// when a bond is a self-bond or references an out-of-range particle, or
    /// when any position, mass, or moment is otherwise invalid (NaN, or a finite
    /// but non-positive mass/moment).
    #[must_use]
    pub fn new(
        model: BondModel,
        bonds: Vec<ParticleBond>,
        positions: Vec<Vec3>,
        masses: Vec<f32>,
        moments: Vec<f32>,
    ) -> Option<Self> {
        let n = positions.len();
        if masses.len() != n || moments.len() != n {
            return None;
        }
        if positions
            .iter()
            .any(|p| !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()))
        {
            return None;
        }
        // A mass is valid if it is NaN-free and either infinite (pinned) or
        // strictly positive.
        let valid_scalar = |v: f32| !v.is_nan() && (v.is_infinite() || v > 0.0);
        if !masses.iter().copied().all(valid_scalar) || !moments.iter().copied().all(valid_scalar) {
            return None;
        }
        let bound = n as u32;
        if bonds
            .iter()
            .any(|b| b.a == b.b || b.a >= bound || b.b >= bound)
        {
            return None;
        }
        let states = vec![BondState::intact(); bonds.len()];
        Some(Self {
            model,
            bonds,
            states,
            velocities: vec![Vec3::ZERO; n],
            angular_velocities: vec![Vec3::ZERO; n],
            positions,
            masses,
            moments,
        })
    }

    /// Number of particles in the body.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of bonds in the body.
    #[must_use]
    pub fn bond_count(&self) -> usize {
        self.bonds.len()
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
        &self.angular_velocities
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
    pub fn set_angular_velocity(&mut self, i: usize, omega: Vec3) {
        if let Some(w) = self.angular_velocities.get_mut(i) {
            *w = omega;
        }
    }

    /// Number of bonds that have broken so far.
    #[must_use]
    pub fn broken_bonds(&self) -> usize {
        broken_bond_count(&self.states)
    }

    /// Total translational + rotational kinetic energy of the finite-mass
    /// particles.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        let mut energy = 0.0;
        for i in 0..self.positions.len() {
            if self.masses[i].is_finite() {
                energy += 0.5 * self.masses[i] * self.velocities[i].length_squared();
            }
            if self.moments[i].is_finite() {
                energy += 0.5 * self.moments[i] * self.angular_velocities[i].length_squared();
            }
        }
        energy
    }

    /// Total linear momentum of the finite-mass particles.
    #[must_use]
    pub fn linear_momentum(&self) -> Vec3 {
        let mut p = Vec3::ZERO;
        for i in 0..self.positions.len() {
            if self.masses[i].is_finite() {
                p += self.masses[i] * self.velocities[i];
            }
        }
        p
    }

    /// Advances the body by `dt` under optional per-particle external forces and
    /// torques, re-evaluating and integrating the bond network once.
    ///
    /// `external_forces` and `external_torques` must each have one entry per
    /// particle. Returns `None` when `dt` is not strictly positive and finite,
    /// when the external arrays have the wrong length, or when the internal bond
    /// assembly rejects the state (which cannot happen for a well-formed body).
    /// On `None` the body is left unchanged.
    #[must_use]
    pub fn step(
        &mut self,
        dt: f32,
        external_forces: &[Vec3],
        external_torques: &[Vec3],
    ) -> Option<BondStepReport> {
        let n = self.positions.len();
        if !(dt.is_finite() && dt > 0.0) {
            return None;
        }
        if external_forces.len() != n || external_torques.len() != n {
            return None;
        }

        // Per-particle increments from the current velocities.
        let mut delta_disp = Vec::with_capacity(n);
        let mut delta_rot = Vec::with_capacity(n);
        for i in 0..n {
            delta_disp.push(self.velocities[i] * dt);
            delta_rot.push(self.angular_velocities[i] * dt);
        }

        let assembly = assemble_bond_forces(
            &self.bonds,
            &self.positions,
            &delta_disp,
            &delta_rot,
            &self.model,
            &mut self.states,
        )?;

        // Integrate velocities (pinned particles are left untouched), then
        // positions, giving the step its symplectic character.
        for i in 0..n {
            if self.masses[i].is_finite() {
                let force = assembly.forces[i] + external_forces[i];
                self.velocities[i] += (dt / self.masses[i]) * force;
            }
            if self.moments[i].is_finite() {
                let torque = assembly.torques[i] + external_torques[i];
                self.angular_velocities[i] += (dt / self.moments[i]) * torque;
            }
        }
        for i in 0..n {
            self.positions[i] += self.velocities[i] * dt;
        }

        Some(BondStepReport {
            broke_this_step: assembly.broke_this_step_count(),
            broken_total: broken_bond_count(&self.states),
            max_normal_stress: assembly.max_normal_stress(),
            max_shear_stress: assembly.max_shear_stress(),
            kinetic_energy: self.kinetic_energy(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);

    fn model() -> BondModel {
        BondModel::new(1.0e9, 1.0e9, 0.01, 1.0e6, 1.0e6, 0.5).unwrap()
    }

    fn two_particle_body() -> BondedParticleBody {
        BondedParticleBody::new(
            model(),
            vec![ParticleBond::new(0, 1)],
            vec![Vec3::ZERO, X],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .expect("valid body")
    }

    fn zeros(n: usize) -> Vec<Vec3> {
        vec![Vec3::ZERO; n]
    }

    #[test]
    fn new_rejects_mismatched_arrays() {
        assert!(BondedParticleBody::new(
            model(),
            vec![ParticleBond::new(0, 1)],
            vec![Vec3::ZERO, X],
            vec![1.0],
            vec![1.0, 1.0],
        )
        .is_none());
    }

    #[test]
    fn new_rejects_bad_mass_and_bonds() {
        // Finite non-positive mass is invalid.
        assert!(BondedParticleBody::new(
            model(),
            vec![],
            vec![Vec3::ZERO, X],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        )
        .is_none());
        // Out-of-range bond.
        assert!(BondedParticleBody::new(
            model(),
            vec![ParticleBond::new(0, 5)],
            vec![Vec3::ZERO, X],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .is_none());
        // Self-bond.
        assert!(BondedParticleBody::new(
            model(),
            vec![ParticleBond::new(0, 0)],
            vec![Vec3::ZERO, X],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .is_none());
    }

    #[test]
    fn pinned_particle_has_infinite_mass() {
        let body = BondedParticleBody::new(
            model(),
            vec![ParticleBond::new(0, 1)],
            vec![Vec3::ZERO, X],
            vec![f32::INFINITY, 1.0],
            vec![f32::INFINITY, 1.0],
        )
        .expect("valid");
        assert!(body.is_pinned(0));
        assert!(!body.is_pinned(1));
    }

    #[test]
    fn step_rejects_bad_dt_and_lengths() {
        let mut body = two_particle_body();
        assert!(body.step(0.0, &zeros(2), &zeros(2)).is_none());
        assert!(body.step(-1.0, &zeros(2), &zeros(2)).is_none());
        assert!(body.step(f32::NAN, &zeros(2), &zeros(2)).is_none());
        assert!(body.step(1.0e-6, &zeros(1), &zeros(2)).is_none());
    }

    #[test]
    fn rigid_translation_develops_no_bond_force() {
        let mut body = two_particle_body();
        let v = Vec3::new(0.3, -0.2, 0.1);
        body.set_velocity(0, v);
        body.set_velocity(1, v);
        let report = body.step(1.0e-4, &zeros(2), &zeros(2)).expect("ok");
        // No relative motion → no bond stress, both particles keep their speed.
        assert!(report.max_normal_stress < 1e-3);
        assert!(report.max_shear_stress < 1e-3);
        assert!((body.velocities()[0] - v).length() < 1e-6);
        assert!((body.velocities()[1] - v).length() < 1e-6);
    }

    #[test]
    fn internal_bond_forces_conserve_linear_momentum() {
        let mut body = two_particle_body();
        // Opposing velocities stretch the bond; no external loads.
        body.set_velocity(0, X * -0.5);
        body.set_velocity(1, X * 0.5);
        let p0 = body.linear_momentum();
        for _ in 0..50 {
            let r = body.step(1.0e-6, &zeros(2), &zeros(2)).expect("ok");
            assert_eq!(r.broke_this_step, 0, "small stretch stays intact");
        }
        let p1 = body.linear_momentum();
        assert!((p1 - p0).length() < 1e-3, "momentum drift {p0:?} -> {p1:?}");
    }

    #[test]
    fn stretched_bond_pulls_particles_back_together() {
        let mut body = two_particle_body();
        // Pull the two apart; the bond tension should decelerate the recession.
        body.set_velocity(0, X * -0.2);
        body.set_velocity(1, X * 0.2);
        let dt = 1.0e-6;
        // Relative recession speed starts at 0.4 m/s.
        let rel_start = (body.velocities()[1] - body.velocities()[0]).x;
        for _ in 0..30 {
            let _ = body.step(dt, &zeros(2), &zeros(2)).expect("ok");
        }
        let rel_end = (body.velocities()[1] - body.velocities()[0]).x;
        assert!(rel_end < rel_start, "tension slows separation");
    }

    #[test]
    fn sustained_overload_breaks_the_bond_then_particles_fly_free() {
        let mut body = two_particle_body();
        // Large opposing external pull along the axis on each particle.
        let mut fext = zeros(2);
        fext[0] = X * -5.0e4;
        fext[1] = X * 5.0e4;
        let dt = 1.0e-5;
        let mut broke_at = None;
        for s in 0..200 {
            let r = body.step(dt, &fext, &zeros(2)).expect("ok");
            if r.broken_total == 1 {
                broke_at = Some(s);
                break;
            }
        }
        assert!(broke_at.is_some(), "overload must snap the bond");
        assert_eq!(body.broken_bonds(), 1);
        // After the snap the bond carries nothing, so a further step produces no
        // bond stress at all.
        let after = body.step(dt, &zeros(2), &zeros(2)).expect("ok");
        assert!(after.max_normal_stress < 1e-3);
        assert!(after.max_shear_stress < 1e-3);
    }

    #[test]
    fn pinned_anchor_holds_while_free_particle_loads_the_bond() {
        let mut body = BondedParticleBody::new(
            model(),
            vec![ParticleBond::new(0, 1)],
            vec![Vec3::ZERO, X],
            vec![f32::INFINITY, 1.0],
            vec![f32::INFINITY, 1.0],
        )
        .expect("valid");
        // Moderate pull on the free particle; anchor must not move. The load
        // is sized to produce a drift that is comfortably resolvable in f32
        // (peak extreme-fibre stress stays well below the bond's tensile limit,
        // so the bond never breaks).
        let mut fext = zeros(2);
        fext[1] = X * 100.0;
        for _ in 0..200 {
            let r = body.step(1.0e-5, &fext, &zeros(2)).expect("ok");
            assert_eq!(r.broke_this_step, 0, "gentle load keeps the bond intact");
        }
        assert_eq!(body.positions()[0], Vec3::ZERO, "anchor is immovable");
        assert_eq!(body.velocities()[0], Vec3::ZERO);
        assert!(body.positions()[1].x > 1.0, "free particle drifted outward");
    }

    #[test]
    fn integrator_matches_builder_output() {
        use crate::collider::bonded_particle_network_builder::build_bond_network;
        // Install bonds on a short chain, then drive it: the two layers compose.
        let pos = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];
        let net = build_bond_network(&pos, &[1.0; 3], 0.0).expect("valid");
        assert_eq!(net.bond_count(), 2);
        let mut body = BondedParticleBody::new(model(), net.bonds, pos, vec![1.0; 3], vec![1.0; 3])
            .expect("valid");
        let r = body.step(1.0e-6, &zeros(3), &zeros(3)).expect("ok");
        assert_eq!(r.broke_this_step, 0);
        assert_eq!(body.bond_count(), 2);
    }
}
