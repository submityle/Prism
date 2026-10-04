//! Explicit discrete-element driver for a packing of colliding spheres.
//!
//! Where the [`bonded_particle_integrator`](super::bonded_particle_integrator)
//! advances the *cohesive* (parallel-bond) response of a bonded solid, this
//! module advances the *repulsive* response of free, colliding grains: the
//! broad-phase resolver
//! ([`resolve_contacts`](super::bonded_particle_contact_resolver::resolve_contacts))
//! accumulates the soft-sphere contact force on every particle, and this driver
//! integrates the resulting motion with a symplectic (semi-implicit Euler)
//! step. Together with the bonded driver it closes the discrete-element
//! pipeline on both sides: grains hold together while bonded, and once broken
//! the fragments collide and settle instead of passing through one another.
//!
//! # State
//!
//! A [`ContactBody`] owns the kinematic state of every particle — position and
//! linear velocity — together with its radius and mass. A particle with a
//! non-finite mass is treated as *pinned*: it feels no acceleration and keeps
//! its prescribed velocity, the usual way to model immovable walls or
//! kinematically scripted grains. Contacts carry no rotational response here;
//! the model is a translational soft-sphere DEM.
//!
//! # Time step
//!
//! Each [`ContactBody::step`] advances the packing by `dt`:
//!
//! ```text
//!   (Fᵢ)   = resolve_contacts(x, r, v, model) + external
//!   vᵢ += dt·Fᵢ/mᵢ        (pinned particles unchanged)
//!   xᵢ += dt·vᵢ           (symplectic position update)
//! ```
//!
//! Contacts are re-resolved from the *current* positions and velocities each
//! step, so the response is fully explicit; the position update then uses the
//! freshly updated velocity, giving the step its symplectic character. Because
//! the contact forces are internal and equal-and-opposite, a step with no
//! external loads conserves linear momentum exactly (`Σ mᵢ Δvᵢ = dt·Σ Fᵢ = 0`).

use crate::collider::bonded_particle_contact::ContactModel;
use crate::collider::bonded_particle_contact_resolver::{resolve_contacts, ContactResolution};
use glam::Vec3;

/// Diagnostic summary of one contact-integration step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactStepReport {
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

/// A time-steppable packing of colliding spheres under a soft-sphere contact
/// law.
#[derive(Clone, Debug, PartialEq)]
pub struct ContactBody {
    model: ContactModel,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    radii: Vec<f32>,
    masses: Vec<f32>,
}

impl ContactBody {
    /// Builds a packing from a contact `model`, initial `positions`,
    /// per-particle `radii`, and per-particle `masses`. Velocities start at
    /// rest.
    ///
    /// A particle whose mass is non-finite (infinite) is *pinned*. Returns
    /// `None` when the arrays disagree in length, when any position is
    /// non-finite, when any radius is not finite and strictly positive, or when
    /// any mass is NaN or finite but non-positive.
    #[must_use]
    pub fn new(
        model: ContactModel,
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
    /// re-resolving the contact set once.
    ///
    /// `external_forces` must have one entry per particle. Returns `None` when
    /// `dt` is not strictly positive and finite, when `external_forces` has the
    /// wrong length, or when the internal contact resolver rejects the state
    /// (which cannot happen for a well-formed body). On `None` the body is left
    /// unchanged.
    #[must_use]
    pub fn step(&mut self, dt: f32, external_forces: &[Vec3]) -> Option<ContactStepReport> {
        let n = self.positions.len();
        if !(dt.is_finite() && dt > 0.0) {
            return None;
        }
        if external_forces.len() != n {
            return None;
        }

        let resolution: ContactResolution =
            resolve_contacts(&self.positions, &self.radii, &self.velocities, &self.model)?;

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

        Some(ContactStepReport {
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

    fn model() -> ContactModel {
        ContactModel::new(1.0e5, 5.0, 5.0, 0.5).unwrap()
    }

    fn two_overlapping() -> ContactBody {
        // Centres 1.5 apart, radii 1 + 1 = 2 → overlap 0.5.
        ContactBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)],
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
        assert!(ContactBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::X],
            vec![1.0],
            vec![1.0, 1.0],
        )
        .is_none());
    }

    #[test]
    fn new_rejects_bad_radius_and_mass() {
        // Non-positive radius.
        assert!(ContactBody::new(model(), vec![Vec3::ZERO], vec![0.0], vec![1.0]).is_none());
        // Finite non-positive mass.
        assert!(ContactBody::new(model(), vec![Vec3::ZERO], vec![1.0], vec![0.0]).is_none());
        // NaN position.
        assert!(ContactBody::new(
            model(),
            vec![Vec3::new(f32::NAN, 0.0, 0.0)],
            vec![1.0],
            vec![1.0],
        )
        .is_none());
    }

    #[test]
    fn rejects_bad_dt_and_external_length() {
        let mut body = two_overlapping();
        assert!(body.step(0.0, &zeros(2)).is_none());
        assert!(body.step(-1.0, &zeros(2)).is_none());
        assert!(body.step(f32::NAN, &zeros(2)).is_none());
        assert!(body.step(1.0e-3, &zeros(1)).is_none());
    }

    #[test]
    fn overlapping_pair_separates_over_time() {
        let mut body = two_overlapping();
        let dt = 1.0e-4;
        let gap0 = (body.positions()[1] - body.positions()[0]).length();
        for _ in 0..200 {
            body.step(dt, &zeros(2)).expect("step");
        }
        let gap1 = (body.positions()[1] - body.positions()[0]).length();
        assert!(gap1 > gap0, "repulsion must push the pair apart");
    }

    #[test]
    fn free_pair_conserves_momentum() {
        let mut body = two_overlapping();
        // Give the pair a net drift so momentum is non-trivial.
        body.set_velocity(0, Vec3::new(0.0, 1.0, 0.0));
        body.set_velocity(1, Vec3::new(0.0, 1.0, 0.0));
        let p0 = body.linear_momentum();
        for _ in 0..50 {
            body.step(1.0e-4, &zeros(2)).expect("step");
        }
        let p1 = body.linear_momentum();
        assert!(
            (p1 - p0).length() < 1e-2,
            "internal contacts conserve momentum"
        );
    }

    #[test]
    fn pinned_particle_does_not_move() {
        // Particle 0 is pinned (infinite mass); the overlapping neighbour must
        // be pushed away while the wall stays put.
        let mut body = ContactBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)],
            vec![1.0, 1.0],
            vec![f32::INFINITY, 1.0],
        )
        .expect("valid body");
        assert!(body.is_pinned(0));
        for _ in 0..200 {
            body.step(1.0e-4, &zeros(2)).expect("step");
        }
        assert_eq!(body.positions()[0], Vec3::ZERO, "pinned wall is immovable");
        assert!(
            body.positions()[1].x > 1.5,
            "free grain is pushed off the wall"
        );
    }

    #[test]
    fn separated_pair_is_inert_without_external_force() {
        // Centres 3 apart, radii 1 + 1 = 2 < 3 → no contact, no motion.
        let mut body = ContactBody::new(
            model(),
            vec![Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0)],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .expect("valid body");
        let report = body.step(1.0e-3, &zeros(2)).expect("step");
        assert_eq!(report.contact_count, 0);
        assert_eq!(body.positions()[0], Vec3::ZERO);
        assert_eq!(body.positions()[1], Vec3::new(3.0, 0.0, 0.0));
        assert_eq!(body.kinetic_energy(), 0.0);
    }

    #[test]
    fn gravity_drives_a_grain_onto_a_pinned_floor() {
        // A single free grain above a pinned floor grain, pulled down by a
        // constant external force, must come to rest in contact rather than
        // sinking through. A well-damped contact (ζ = γₙ/(2·√(kₙ·m)) = 0.75)
        // is used so the grain actually settles instead of bouncing forever.
        let damped = ContactModel::new(1.0e4, 150.0, 50.0, 0.5).unwrap();
        let mut body = ContactBody::new(
            damped,
            vec![Vec3::ZERO, Vec3::new(0.0, 2.5, 0.0)],
            vec![1.0, 1.0],
            vec![f32::INFINITY, 1.0],
        )
        .expect("valid body");
        let gravity = vec![Vec3::ZERO, Vec3::new(0.0, -9.81, 0.0)];
        for _ in 0..4000 {
            body.step(1.0e-3, &gravity).expect("step");
        }
        // Equilibrium sits at a tiny penalty overlap δ = mg/kₙ ≈ 9.8e-4 below
        // one-radius contact, so the settled centre gap is just under 2·R = 2.
        let gap = body.positions()[1].y - body.positions()[0].y;
        assert!(gap > 1.9, "grain must rest on the floor, not fall through");
        assert!(gap < 2.01, "grain must actually settle onto the floor");
        // And it must be at rest, not mid-bounce.
        assert!(
            body.velocities()[1].length() < 1e-2,
            "grain must come to rest"
        );
    }
}
