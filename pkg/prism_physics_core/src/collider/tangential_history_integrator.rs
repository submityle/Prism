//! Explicit discrete-element driver for a packing of spheres colliding under
//! the Cundall–Strack tangential-history friction law.
//!
//! This is the stateful-friction counterpart of
//! [`hertz_contact_integrator`](super::hertz_contact_integrator). The Hertz and
//! soft-sphere drivers resolve friction from the instantaneous sliding velocity
//! alone, so a grain can never truly rest on a slope. This driver instead holds
//! a [`TangentialHistoryResolver`](super::tangential_history_resolver::TangentialHistoryResolver)
//! whose per-contact springs persist across steps, so a grain subjected to a
//! tangential load below the Coulomb limit `μ·F_n` settles to rest rather than
//! creeping — the behaviour that lets a granular pile stand at an angle of
//! repose.
//!
//! # State
//!
//! A [`TangentialHistoryBody`] owns the kinematic state of every particle —
//! position and linear velocity — together with its radius and mass, plus the
//! resolver carrying the friction springs. A particle with a non-finite mass is
//! treated as *pinned*: it feels no acceleration and keeps its prescribed
//! velocity, the usual way to model immovable walls or scripted grains.
//! Contacts carry no rotational response here; the model is a translational
//! discrete-element packing.
//!
//! # Time step
//!
//! Each [`TangentialHistoryBody::step`] advances the packing by `dt`:
//!
//! ```text
//!   (Fᵢ)   = resolver.resolve(x, r, v, model, dt) + external
//!   vᵢ += dt·Fᵢ/mᵢ        (pinned particles unchanged)
//!   xᵢ += dt·vᵢ           (symplectic position update)
//! ```
//!
//! The resolver advances every live contact's friction spring and prunes the
//! springs of pairs that have separated, so the friction force carries memory
//! from step to step. Because the contact forces are internal and
//! equal-and-opposite, a step with no external loads conserves linear momentum
//! exactly (`Σ mᵢ Δvᵢ = dt·Σ Fᵢ = 0`).

use crate::collider::tangential_history_contact::CundallStrackModel;
use crate::collider::tangential_history_resolver::{
    TangentialHistoryResolution, TangentialHistoryResolver,
};
use glam::Vec3;

/// Diagnostic summary of one Cundall–Strack contact-integration step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TangentialHistoryStepReport {
    /// Number of contacting pairs resolved this step.
    pub contact_count: usize,
    /// Largest overlap across all contacts this step.
    pub max_overlap: f32,
    /// Largest normal force magnitude across all contacts this step.
    pub max_normal_force: f32,
    /// Number of contacts at the Coulomb (sliding) limit this step.
    pub sliding_count: usize,
    /// Total translational kinetic energy of the finite-mass particles after
    /// the step.
    pub kinetic_energy: f32,
}

/// A time-steppable packing of colliding spheres under a Cundall–Strack
/// tangential-history contact law with persistent friction springs.
#[derive(Clone, Debug)]
pub struct TangentialHistoryBody {
    model: CundallStrackModel,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    radii: Vec<f32>,
    masses: Vec<f32>,
    resolver: TangentialHistoryResolver,
}

impl TangentialHistoryBody {
    /// Builds a packing from a `model`, initial `positions`, per-particle
    /// `radii`, and per-particle `masses`. Velocities start at rest and the
    /// friction history is empty.
    ///
    /// A particle whose mass is non-finite (infinite) is *pinned*. Returns
    /// `None` when the arrays disagree in length, when any position is
    /// non-finite, when any radius is not finite and strictly positive, or when
    /// any mass is NaN or finite but non-positive.
    #[must_use]
    pub fn new(
        model: CundallStrackModel,
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
        Some(Self {
            model,
            velocities: vec![Vec3::ZERO; n],
            positions,
            radii,
            masses,
            resolver: TangentialHistoryResolver::new(),
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

    /// Particle radii.
    #[must_use]
    pub fn radii(&self) -> &[f32] {
        &self.radii
    }

    /// Number of live friction-spring contacts currently tracked.
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

    /// Total translational kinetic energy of the finite-mass particles.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        let mut energy = 0.0;
        for (m, v) in self.masses.iter().zip(self.velocities.iter()) {
            if m.is_finite() {
                energy += 0.5 * m * v.length_squared();
            }
        }
        energy
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

    /// Advances the packing by `dt` under optional per-particle external forces,
    /// re-resolving the contact set and advancing the friction springs once.
    ///
    /// `external_forces` must have one entry per particle. Returns `None` when
    /// `dt` is not strictly positive and finite, when `external_forces` has the
    /// wrong length, or when the internal resolver rejects the state (which
    /// cannot happen for a well-formed body). On `None` the body is left
    /// unchanged.
    #[must_use]
    pub fn step(
        &mut self,
        dt: f32,
        external_forces: &[Vec3],
    ) -> Option<TangentialHistoryStepReport> {
        let n = self.positions.len();
        if !(dt.is_finite() && dt > 0.0) {
            return None;
        }
        if external_forces.len() != n {
            return None;
        }

        // Disjoint field borrows: `resolver` mutably, kinematic state immutably.
        let resolution: TangentialHistoryResolution = self.resolver.resolve(
            &self.positions,
            &self.radii,
            &self.velocities,
            &self.model,
            dt,
        )?;

        // Integrate velocities (pinned particles untouched), then positions,
        // giving the step its symplectic character.
        for (i, &ext) in external_forces.iter().enumerate() {
            if self.masses[i].is_finite() {
                let force = resolution.forces[i] + ext;
                self.velocities[i] += (dt / self.masses[i]) * force;
            }
        }
        for (pos, &vel) in self.positions.iter_mut().zip(self.velocities.iter()) {
            *pos += vel * dt;
        }

        Some(TangentialHistoryStepReport {
            contact_count: resolution.contact_count(),
            max_overlap: resolution.max_overlap(),
            max_normal_force: resolution.max_normal_force(),
            sliding_count: resolution.sliding_count(),
            kinetic_energy: self.kinetic_energy(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CundallStrackModel {
        // kₙ, γₙ, k_t, γ_t, μ — damped so packings settle.
        CundallStrackModel::new(1.0e4, 50.0, 1.0e4, 50.0, 0.6).unwrap()
    }

    fn zeros(n: usize) -> Vec<Vec3> {
        vec![Vec3::ZERO; n]
    }

    #[test]
    fn new_rejects_mismatched_arrays() {
        assert!(TangentialHistoryBody::new(
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
            TangentialHistoryBody::new(model(), vec![Vec3::ZERO], vec![0.0], vec![1.0]).is_none()
        );
        assert!(
            TangentialHistoryBody::new(model(), vec![Vec3::ZERO], vec![1.0], vec![0.0]).is_none()
        );
        assert!(TangentialHistoryBody::new(
            model(),
            vec![Vec3::new(f32::NAN, 0.0, 0.0)],
            vec![1.0],
            vec![1.0],
        )
        .is_none());
        // Infinite mass (pinned) is accepted.
        assert!(TangentialHistoryBody::new(
            model(),
            vec![Vec3::ZERO],
            vec![1.0],
            vec![f32::INFINITY]
        )
        .is_some());
    }

    #[test]
    fn step_rejects_bad_dt_and_external_length() {
        let mut body =
            TangentialHistoryBody::new(model(), vec![Vec3::ZERO], vec![1.0], vec![1.0]).unwrap();
        assert!(body.step(0.0, &zeros(1)).is_none());
        assert!(body.step(f32::NAN, &zeros(1)).is_none());
        assert!(body.step(1.0e-3, &zeros(2)).is_none());
    }

    #[test]
    fn free_particle_follows_external_force() {
        let mut body =
            TangentialHistoryBody::new(model(), vec![Vec3::ZERO], vec![1.0], vec![2.0]).unwrap();
        // Constant force of (4, 0, 0) on mass 2 → acceleration 2.
        let report = body.step(1.0e-2, &[Vec3::new(4.0, 0.0, 0.0)]).unwrap();
        assert_eq!(report.contact_count, 0);
        // v = a·dt = 2·1e-2 = 0.02; x = v·dt = 2e-4.
        assert!((body.velocities()[0].x - 0.02).abs() < 1e-6);
        assert!((body.positions()[0].x - 2.0e-4).abs() < 1e-7);
    }

    #[test]
    fn two_overlapping_grains_repel_and_conserve_momentum() {
        // Pure normal overlap, no external loads → they push apart but the
        // total momentum stays zero.
        let mut body = TangentialHistoryBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::new(1.9, 0.0, 0.0)],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .unwrap();
        let before = body.positions()[1].x - body.positions()[0].x;
        // First step: the pair is overlapping, so exactly one contact resolves.
        let first = body.step(1.0e-3, &zeros(2)).unwrap();
        assert_eq!(first.contact_count, 1);
        assert!(body.linear_momentum().length() < 1e-3, "momentum conserved");
        for _ in 0..60 {
            body.step(1.0e-3, &zeros(2)).unwrap();
            // Symmetric masses with equal-and-opposite contact forces: the
            // total momentum stays zero whether in contact or flying free.
            assert!(body.linear_momentum().length() < 1e-3, "momentum conserved");
        }
        let after = body.positions()[1].x - body.positions()[0].x;
        assert!(after > before, "repulsion increases separation");
    }

    #[test]
    fn pinned_particle_acts_as_immovable_wall() {
        // Grain 0 pinned, grain 1 overlapping it; grain 0 must not move despite
        // the contact force.
        let mut body = TangentialHistoryBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::new(1.9, 0.0, 0.0)],
            vec![1.0, 1.0],
            vec![f32::INFINITY, 1.0],
        )
        .unwrap();
        assert!(body.is_pinned(0));
        assert!(!body.is_pinned(1));
        for _ in 0..100 {
            body.step(1.0e-3, &zeros(2)).unwrap();
        }
        assert_eq!(body.positions()[0], Vec3::ZERO, "pinned grain never moves");
        assert_eq!(body.velocities()[0], Vec3::ZERO);
        assert!(body.positions()[1].x > 1.9, "free grain pushed away");
    }

    #[test]
    fn tangential_load_below_cap_settles_to_rest() {
        // A grain resting on a pinned floor grain, held down by a vertical load
        // and pushed horizontally below the Coulomb limit. Static friction must
        // bring it to rest at a bounded displacement rather than letting it
        // creep away indefinitely.
        let mut body = TangentialHistoryBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::new(0.0, 1.95, 0.0)],
            vec![1.0, 1.0],
            vec![f32::INFINITY, 1.0],
        )
        .unwrap();
        // Hold-down 100 → steady normal force ≈ 100, cap = μ·F_n = 60.
        // Horizontal push 20 < 60, so the contact must stick.
        let ext = vec![Vec3::ZERO, Vec3::new(20.0, -100.0, 0.0)];
        for _ in 0..40_000 {
            body.step(1.0e-3, &ext).unwrap();
        }
        let v = body.velocities()[1];
        assert!(v.length() < 1e-2, "grain settles to rest, got v={v:?}");
        // It stuck near the origin in x — a bounded elastic offset, not a slide.
        assert!(
            body.positions()[1].x.abs() < 0.2,
            "static friction holds the grain, x={}",
            body.positions()[1].x
        );
        let last = body.step(1.0e-3, &ext).unwrap();
        assert_eq!(last.sliding_count, 0, "resting contact is not sliding");
        assert_eq!(body.active_contacts(), 1);
    }

    #[test]
    fn tangential_load_above_cap_slides() {
        // Same geometry, but a horizontal push above the Coulomb limit makes the
        // grain slide: it keeps gaining horizontal speed and the contact reports
        // sliding.
        let mut body = TangentialHistoryBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::new(0.0, 1.95, 0.0)],
            vec![1.0, 1.0],
            vec![f32::INFINITY, 1.0],
        )
        .unwrap();
        // Push 200 ≫ cap 60.
        let ext = vec![Vec3::ZERO, Vec3::new(200.0, -100.0, 0.0)];
        let mut saw_sliding = false;
        for _ in 0..2_000 {
            let r = body.step(1.0e-3, &ext).unwrap();
            if r.sliding_count > 0 {
                saw_sliding = true;
            }
        }
        assert!(saw_sliding, "over-cap push must slide");
        assert!(
            body.positions()[1].x > 0.2,
            "sliding grain travels, x={}",
            body.positions()[1].x
        );
        assert!(body.velocities()[1].x > 0.1, "sliding grain keeps speed");
    }

    #[test]
    fn kinetic_energy_decays_under_damping() {
        // Give two overlapping grains opposing velocities; the damped contact
        // must bleed kinetic energy out of the pair.
        // Start exactly touching (overlap 0 → no stored penalty energy), moving
        // toward each other so the collision engages the damper.
        let mut body = TangentialHistoryBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .unwrap();
        body.set_velocity(0, Vec3::new(1.0, 0.0, 0.0));
        body.set_velocity(1, Vec3::new(-1.0, 0.0, 0.0));
        let start = body.kinetic_energy();
        let mut last = start;
        for _ in 0..200 {
            let r = body.step(1.0e-3, &zeros(2)).unwrap();
            last = r.kinetic_energy;
        }
        assert!(last < start, "damping dissipates kinetic energy");
    }
}
