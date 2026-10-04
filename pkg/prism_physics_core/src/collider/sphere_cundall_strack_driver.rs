//! Stateful Cundall–Strack friction over a precomputed sphere-contact set.
//!
//! [`sphere_contact_forces`](crate::collider::sphere_contact_forces) evaluates
//! the memoryless Hertz force on a shared contact set. The Cundall–Strack
//! tangential law is different: it carries a persistent elastic spring per
//! contact so that a stuck grain develops a growing static-friction force until
//! it reaches the Coulomb cone and slips. That history has to survive between
//! steps, so it cannot live in a stateless resolver call.
//!
//! This driver owns that history. It consumes an already-built contact set — for
//! example from
//! [`SphereNarrowPhase`](crate::collider::sphere_narrow_phase::SphereNarrowPhase)
//! — advances one spring per contacting pair, and prunes the springs of pairs
//! that have separated. The force law itself is reused verbatim from
//! [`evaluate_tangential_history`](crate::collider::tangential_history_contact::evaluate_tangential_history),
//! so there is a single source of truth for the contact physics, exactly as the
//! monolithic
//! [`TangentialHistoryResolver`](crate::collider::tangential_history_resolver::TangentialHistoryResolver)
//! does — the only difference is that this driver takes the geometry from a
//! shared narrow phase instead of rediscovering it with a private grid.

use std::collections::HashMap;

use glam::Vec3;

use crate::collider::sphere_narrow_phase::SphereContact;
use crate::collider::tangential_history_contact::{
    evaluate_tangential_history, CundallStrackModel,
};

/// Per-grain forces and summary diagnostics from one
/// [`SphereCundallStrackDriver::resolve`].
#[derive(Clone, Debug, PartialEq)]
pub struct SphereCundallStrackResolution {
    forces: Vec<Vec3>,
    contact_count: u32,
    max_normal_force: f32,
    sliding_count: u32,
}

impl SphereCundallStrackResolution {
    /// Per-grain accumulated contact force (index-aligned with the cloud).
    #[must_use]
    pub fn forces(&self) -> &[Vec3] {
        &self.forces
    }

    /// Number of contacts that carried a load this step.
    #[must_use]
    pub fn contact_count(&self) -> u32 {
        self.contact_count
    }

    /// Largest single-contact normal force magnitude.
    #[must_use]
    pub fn max_normal_force(&self) -> f32 {
        self.max_normal_force
    }

    /// Number of contacts whose friction reached the Coulomb (sliding) limit.
    #[must_use]
    pub fn sliding_count(&self) -> u32 {
        self.sliding_count
    }

    /// Vector sum of all grain forces; near zero because contacts are
    /// equal-and-opposite.
    #[must_use]
    pub fn total_force(&self) -> Vec3 {
        self.forces.iter().copied().fold(Vec3::ZERO, |a, f| a + f)
    }
}

/// A stateful Cundall–Strack friction driver that carries persistent tangential
/// springs across time steps and advances them over a shared contact set.
///
/// Reuse one instance across steps for a given cloud: it stores one spring per
/// contacting pair, advances it each [`resolve`](Self::resolve), and drops the
/// spring once the pair separates. A fresh driver has no history and behaves
/// like the stateless law on its first step.
#[derive(Clone, Debug)]
pub struct SphereCundallStrackDriver {
    model: CundallStrackModel,
    springs: HashMap<(u32, u32), Vec3>,
}

impl SphereCundallStrackDriver {
    /// Builds a driver with the given contact `model` and no stored history.
    #[must_use]
    pub fn new(model: CundallStrackModel) -> Self {
        Self {
            model,
            springs: HashMap::new(),
        }
    }

    /// Number of live contact springs currently stored.
    #[must_use]
    pub fn active_springs(&self) -> usize {
        self.springs.len()
    }

    /// The stored tangential spring for the unordered pair, or `None` when no
    /// live contact exists between those grains. Indices may be given in either
    /// order.
    #[must_use]
    pub fn spring(&self, i: u32, j: u32) -> Option<Vec3> {
        let key = if i <= j { (i, j) } else { (j, i) };
        self.springs.get(&key).copied()
    }

    /// Forgets all stored friction history.
    pub fn clear(&mut self) {
        self.springs.clear();
    }

    /// Advances the Cundall–Strack friction over `contacts`, returning the net
    /// per-grain force and summary diagnostics.
    ///
    /// `radii` and `velocities` describe the grain cloud and must have equal
    /// length; every radius must be finite and strictly positive and every
    /// velocity finite. `dt` must be finite and strictly positive. Each
    /// [`SphereContact`] references grain indices into that cloud; any
    /// out-of-range or self index makes the call fail. Contacts with
    /// non-positive penetration (near pairs reported under a detection margin)
    /// carry no load and have their spring dropped. The springs of pairs absent
    /// from `contacts` are also dropped, so the stored history always reflects
    /// the live contact set. Returns `None` on invalid input, leaving the
    /// stored history untouched.
    ///
    /// For a contact between `a` and `b` with unit normal pointing `a -> b`, the
    /// force on `b` is the Cundall–Strack response and `a` receives its
    /// negation, so linear momentum is conserved.
    #[must_use]
    pub fn resolve(
        &mut self,
        contacts: &[SphereContact],
        radii: &[f32],
        velocities: &[Vec3],
        dt: f32,
    ) -> Option<SphereCundallStrackResolution> {
        let n = radii.len();
        if velocities.len() != n {
            return None;
        }
        if !(dt.is_finite() && dt > 0.0) {
            return None;
        }
        for (&r, vel) in radii.iter().zip(velocities.iter()) {
            if !r.is_finite() || r <= 0.0 || !vel.is_finite() {
                return None;
            }
        }
        for contact in contacts.iter() {
            let a = contact.a as usize;
            let b = contact.b as usize;
            if a >= n || b >= n || a == b {
                return None;
            }
        }

        let mut forces = vec![Vec3::ZERO; n];
        let mut next_springs: HashMap<(u32, u32), Vec3> = HashMap::new();
        let mut contact_count: u32 = 0;
        let mut max_normal_force = 0.0_f32;
        let mut sliding_count: u32 = 0;

        for contact in contacts.iter() {
            if !(contact.penetration.is_finite() && contact.penetration > 0.0) {
                continue;
            }
            let a = contact.a as usize;
            let b = contact.b as usize;
            let key = (contact.a, contact.b);

            // Carry the stored spring forward; a fresh pair starts at rest.
            let mut spring = self.springs.get(&key).copied().unwrap_or(Vec3::ZERO);
            let rel_vel = velocities[b] - velocities[a];
            let force = evaluate_tangential_history(
                &self.model,
                contact.normal,
                contact.penetration,
                rel_vel,
                &mut spring,
                dt,
            );
            next_springs.insert(key, spring);

            forces[b] += force.force_on_b;
            forces[a] -= force.force_on_b;
            contact_count += 1;
            max_normal_force = max_normal_force.max(force.normal_magnitude);
            if force.sliding {
                sliding_count += 1;
            }
        }

        self.springs = next_springs;

        Some(SphereCundallStrackResolution {
            forces,
            contact_count,
            max_normal_force,
            sliding_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::sphere_narrow_phase::SphereNarrowPhase;
    use crate::collider::tangential_history_resolver::TangentialHistoryResolver;

    fn model() -> CundallStrackModel {
        // kₙ, γₙ, k_t, γ_t, μ.
        CundallStrackModel::new(1.0e5, 0.0, 1.0e5, 0.0, 0.5).unwrap()
    }

    fn contact(a: u32, b: u32, normal: Vec3, penetration: f32) -> SphereContact {
        SphereContact {
            a,
            b,
            normal,
            penetration,
            contact_point: Vec3::ZERO,
        }
    }

    #[test]
    fn empty_contacts_give_zero_forces() {
        let mut driver = SphereCundallStrackDriver::new(model());
        let res = driver
            .resolve(&[], &[1.0, 1.0], &[Vec3::ZERO, Vec3::ZERO], 1.0e-3)
            .unwrap();
        assert_eq!(res.forces(), &[Vec3::ZERO, Vec3::ZERO]);
        assert_eq!(res.contact_count(), 0);
        assert_eq!(driver.active_springs(), 0);
    }

    #[test]
    fn single_overlap_is_repulsive_and_newtonian() {
        let mut driver = SphereCundallStrackDriver::new(model());
        let radii = vec![1.0_f32, 1.0];
        let velocities = vec![Vec3::ZERO, Vec3::ZERO];
        let contacts = vec![contact(0, 1, Vec3::X, 0.1)];
        let res = driver
            .resolve(&contacts, &radii, &velocities, 1.0e-3)
            .unwrap();
        assert!(res.forces()[0].x < 0.0);
        assert!(res.forces()[1].x > 0.0);
        assert!((res.forces()[0] + res.forces()[1]).length() < 1.0e-5);
        assert!(res.max_normal_force() > 0.0);
        assert_eq!(res.contact_count(), 1);
        assert_eq!(driver.active_springs(), 1);
    }

    #[test]
    fn matches_direct_force_law_on_first_step() {
        let mut driver = SphereCundallStrackDriver::new(model());
        let radii = vec![1.0_f32, 1.0];
        // b slips tangentially so a non-zero friction force is exercised.
        let velocities = vec![Vec3::ZERO, Vec3::new(0.0, 0.2, 0.0)];
        let penetration = 0.1_f32;
        let dt = 1.0e-3_f32;
        let contacts = vec![contact(0, 1, Vec3::X, penetration)];
        let res = driver.resolve(&contacts, &radii, &velocities, dt).unwrap();

        let mut spring = Vec3::ZERO;
        let rel_vel = velocities[1] - velocities[0];
        let direct =
            evaluate_tangential_history(&model(), Vec3::X, penetration, rel_vel, &mut spring, dt);
        assert!((res.forces()[1] - direct.force_on_b).length() < 1.0e-5);
        assert!((res.forces()[0] + direct.force_on_b).length() < 1.0e-5);
        assert!(direct.tangential_magnitude > 0.0, "friction exercised");
    }

    #[test]
    fn negative_penetration_contributes_no_force() {
        let mut driver = SphereCundallStrackDriver::new(model());
        let radii = vec![1.0_f32, 1.0];
        let velocities = vec![Vec3::ZERO, Vec3::ZERO];
        let contacts = vec![contact(0, 1, Vec3::X, -0.05)];
        let res = driver
            .resolve(&contacts, &radii, &velocities, 1.0e-3)
            .unwrap();
        assert_eq!(res.forces(), &[Vec3::ZERO, Vec3::ZERO]);
        assert_eq!(res.contact_count(), 0);
        assert_eq!(driver.active_springs(), 0);
    }

    #[test]
    fn rejects_out_of_range_index_and_bad_input() {
        let mut driver = SphereCundallStrackDriver::new(model());
        let radii = vec![1.0_f32, 1.0];
        let velocities = vec![Vec3::ZERO, Vec3::ZERO];
        // Out-of-range index.
        assert!(driver
            .resolve(&[contact(0, 5, Vec3::X, 0.1)], &radii, &velocities, 1.0e-3)
            .is_none());
        // Mismatched lengths.
        assert!(driver
            .resolve(
                &[contact(0, 1, Vec3::X, 0.1)],
                &[1.0, 1.0],
                &[Vec3::ZERO],
                1.0e-3
            )
            .is_none());
        // Non-positive dt.
        assert!(driver
            .resolve(&[contact(0, 1, Vec3::X, 0.1)], &radii, &velocities, 0.0)
            .is_none());
        // Non-finite velocity.
        assert!(driver
            .resolve(
                &[contact(0, 1, Vec3::X, 0.1)],
                &radii,
                &[Vec3::ZERO, Vec3::new(f32::NAN, 0.0, 0.0)],
                1.0e-3
            )
            .is_none());
    }

    #[test]
    fn static_friction_grows_across_steps() {
        let mut driver = SphereCundallStrackDriver::new(model());
        let radii = vec![1.0_f32, 1.0];
        // Slow tangential slip stays under the Coulomb cap, so the pair sticks
        // and the spring force grows each step.
        let velocities = vec![Vec3::ZERO, Vec3::new(0.0, 0.01, 0.0)];
        let contacts = vec![contact(0, 1, Vec3::X, 0.5)];
        let dt = 1.0e-4_f32;

        let mut last_tangential = 0.0_f32;
        for step in 1..=5 {
            let res = driver.resolve(&contacts, &radii, &velocities, dt).unwrap();
            assert_eq!(driver.active_springs(), 1);
            assert_eq!(res.sliding_count(), 0, "slow slip must stick, step {step}");
            // Tangential force magnitude on grain 1 (minus its normal part).
            let f = res.forces()[1];
            let tangential = (f - Vec3::new(f.x, 0.0, 0.0)).length();
            assert!(tangential > last_tangential, "friction grows, step {step}");
            last_tangential = tangential;
        }
    }

    #[test]
    fn separation_prunes_the_spring() {
        let mut driver = SphereCundallStrackDriver::new(model());
        let radii = vec![1.0_f32, 1.0];
        let velocities = vec![Vec3::ZERO, Vec3::ZERO];
        // Step 1: overlapping, spring forms.
        driver
            .resolve(&[contact(0, 1, Vec3::X, 0.1)], &radii, &velocities, 1.0e-3)
            .unwrap();
        assert_eq!(driver.active_springs(), 1);
        assert!(driver.spring(1, 0).is_some(), "unordered lookup");
        // Step 2: no contacts reported (separated) → spring dropped.
        driver.resolve(&[], &radii, &velocities, 1.0e-3).unwrap();
        assert_eq!(driver.active_springs(), 0);
        assert!(driver.spring(0, 1).is_none());
    }

    #[test]
    fn pipeline_matches_monolithic_resolver_first_step() {
        // A small overlapping cluster with distinct velocities so friction is
        // exercised. On the first step both the shared-narrow-phase driver and
        // the monolithic resolver start from zero springs and must agree.
        let mut positions = Vec::new();
        let spacing = 1.8_f32;
        for ix in 0..3 {
            for iy in 0..3 {
                positions.push(Vec3::new(ix as f32 * spacing, iy as f32 * spacing, 0.0));
            }
        }
        let n = positions.len();
        let radii = vec![1.0_f32; n];
        let mut velocities = Vec::with_capacity(n);
        for i in 0..n {
            let s = i as f32;
            velocities.push(Vec3::new(0.05 * s, -0.03 * s, 0.02 * s));
        }
        let model = model();
        let dt = 1.0e-4_f32;

        let mut np = SphereNarrowPhase::new();
        let contacts = np.detect(&positions, &radii, 0.0).unwrap().to_vec();
        let mut driver = SphereCundallStrackDriver::new(model);
        let decoupled = driver.resolve(&contacts, &radii, &velocities, dt).unwrap();

        let mut monolithic_resolver = TangentialHistoryResolver::new();
        let monolithic = monolithic_resolver
            .resolve(&positions, &radii, &velocities, &model, dt)
            .unwrap();

        assert_eq!(decoupled.forces().len(), monolithic.forces.len());
        for (d, m) in decoupled.forces().iter().zip(monolithic.forces.iter()) {
            assert!(
                (*d - *m).length() < 1.0e-3,
                "per-grain force mismatch: {d:?} vs {m:?}"
            );
        }
        assert_eq!(
            decoupled.contact_count() as usize,
            monolithic.contact_count()
        );
    }
}
