//! Broad-phase accumulation of rotational discrete-element contacts with
//! persistent per-contact friction and rolling springs.
//!
//! The translational Cundall–Strack resolver
//! ([`tangential_history_resolver`](super::tangential_history_resolver))
//! advances only each grain's *linear* velocity: it carries a tangential
//! friction spring per pair, but the grains cannot spin. Real packings reach a
//! realistic angle of repose only when the contacts also resist the grains
//! *rolling* over one another, which requires the angular degrees of freedom
//! introduced by the
//! [`rotational_contact`](super::rotational_contact) law.
//!
//! This resolver is the stateful counterpart of that law. It owns a map of
//! [`ContactSprings`] (the tangential-history sliding displacement *and* the
//! accumulated rolling angle) keyed by the unordered particle pair `(a, b)`.
//! Each [`resolve`](RollingContactResolver::resolve) call advances every live
//! contact's springs and **drops the springs of any pair that is no longer in
//! contact**, so a reopened contact starts fresh.
//!
//! # Acceleration
//!
//! The broad phase is identical to the other resolvers: particle centres are
//! binned into a uniform spatial-hash grid whose cell size is the largest
//! possible contact distance `2·R_max`, and each particle tests only the 27
//! cells of its `3×3×3` neighbourhood. Each unordered pair is evaluated once
//! with the lower index first, and contacts are reported in a deterministic
//! `(a, b)` order independent of grid-bucket iteration order.
//!
//! # Forces, torques and momentum
//!
//! For each contacting pair the resolver calls
//! [`rotational_contact_between`](super::rotational_contact::rotational_contact_between),
//! which returns the force on `b` (the equal and opposite force acts on `a`)
//! plus the spin torque on each grain. The per-particle `forces` therefore sum
//! to zero up to floating-point error, so a resolved packing conserves linear
//! momentum exactly; because every contact force acts at the *shared* contact
//! point, the packing conserves angular momentum as well.

use crate::collider::rotational_contact::{
    rotational_contact_between, ContactSprings, RollingContactModel,
};
use glam::Vec3;
use std::collections::{HashMap, HashSet};

/// A single resolved rotational contact between two particles of a packing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RollingPairContact {
    /// Lower particle index of the contacting pair.
    pub a: u32,
    /// Higher particle index of the contacting pair.
    pub b: u32,
    /// Overlap `δ > 0` resolved by this contact.
    pub overlap: f32,
    /// Normal force magnitude `F_n ≥ 0`.
    pub normal_magnitude: f32,
    /// Tangential (sliding friction) force magnitude actually applied.
    pub tangential_magnitude: f32,
    /// Rolling resistance torque magnitude actually applied.
    pub rolling_magnitude: f32,
    /// Whether the tangential force reached the Coulomb limit (sliding).
    pub sliding: bool,
}

/// Result of resolving every rotational contact in a particle packing.
#[derive(Clone, Debug, PartialEq)]
pub struct RollingContactResolution {
    /// Net contact force on each particle, indexed in parallel with the input
    /// `positions`; always the same length as the packing.
    pub forces: Vec<Vec3>,
    /// Net contact torque on each particle, indexed in parallel with the input
    /// `positions`; always the same length as the packing.
    pub torques: Vec<Vec3>,
    /// The resolved contacts, in deterministic `(a, b)` order.
    pub contacts: Vec<RollingPairContact>,
}

impl RollingContactResolution {
    /// Number of resolved contacts.
    #[must_use]
    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }

    /// Largest overlap across all contacts, or `0.0` when there are none.
    #[must_use]
    pub fn max_overlap(&self) -> f32 {
        self.contacts
            .iter()
            .map(|c| c.overlap)
            .fold(0.0_f32, f32::max)
    }

    /// Largest normal force magnitude across all contacts, or `0.0` when there
    /// are none.
    #[must_use]
    pub fn max_normal_force(&self) -> f32 {
        self.contacts
            .iter()
            .map(|c| c.normal_magnitude)
            .fold(0.0_f32, f32::max)
    }

    /// Largest rolling resistance torque magnitude across all contacts, or
    /// `0.0` when there are none.
    #[must_use]
    pub fn max_rolling_torque(&self) -> f32 {
        self.contacts
            .iter()
            .map(|c| c.rolling_magnitude)
            .fold(0.0_f32, f32::max)
    }

    /// Sum of all per-particle forces. The contact forces are internal and
    /// equal-and-opposite, so this is zero up to floating-point error; it is
    /// exposed as a linear-momentum conservation diagnostic.
    #[must_use]
    pub fn total_force(&self) -> Vec3 {
        self.forces.iter().copied().sum()
    }

    /// Total rate of change of angular momentum about the origin,
    /// `Σ (xᵢ × Fᵢ) + Σ τᵢ`, given the packing `positions`. Because every
    /// contact force acts at the shared contact point and the rolling couples
    /// are equal and opposite, this is zero up to floating-point error; it is
    /// exposed as an angular-momentum conservation diagnostic. Returns
    /// [`Vec3::ZERO`] when `positions` does not match the packing length.
    #[must_use]
    pub fn total_angular_momentum_rate(&self, positions: &[Vec3]) -> Vec3 {
        if positions.len() != self.forces.len() {
            return Vec3::ZERO;
        }
        let moment: Vec3 = positions
            .iter()
            .zip(self.forces.iter())
            .map(|(x, f)| x.cross(*f))
            .sum();
        let spin: Vec3 = self.torques.iter().copied().sum();
        moment + spin
    }

    /// Number of contacts whose tangential force reached the Coulomb limit.
    #[must_use]
    pub fn sliding_count(&self) -> usize {
        self.contacts.iter().filter(|c| c.sliding).count()
    }
}

/// Validates that the packing is well formed: equal-length `positions`,
/// `radii`, `velocities`, and `angular`; every coordinate finite; every radius
/// finite and strictly positive; every velocity and angular-velocity component
/// finite.
fn is_valid(positions: &[Vec3], radii: &[f32], velocities: &[Vec3], angular: &[Vec3]) -> bool {
    if positions.len() != radii.len()
        || positions.len() != velocities.len()
        || positions.len() != angular.len()
    {
        return false;
    }
    let finite_vec = |v: &Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    positions.iter().all(finite_vec)
        && velocities.iter().all(finite_vec)
        && angular.iter().all(finite_vec)
        && radii.iter().all(|&r| r.is_finite() && r > 0.0)
}

/// Integer grid cell coordinate of a point relative to the packing's minimum
/// corner, at the given cell size.
fn cell_of(point: Vec3, origin: Vec3, cell_size: f32) -> [i32; 3] {
    let rel = (point - origin) / cell_size;
    [
        rel.x.floor() as i32,
        rel.y.floor() as i32,
        rel.z.floor() as i32,
    ]
}

/// A stateful resolver that carries persistent rotational contact springs
/// across time steps.
///
/// Create one with [`RollingContactResolver::new`] and call
/// [`resolve`](RollingContactResolver::resolve) once per step with the current
/// packing state; the resolver advances the stored springs and prunes contacts
/// that have separated. The same resolver instance (and therefore the same
/// spring map) must be reused across steps for a given packing — a fresh
/// resolver has no friction or rolling history and behaves like the stateless
/// law on its first step.
#[derive(Clone, Debug, Default)]
pub struct RollingContactResolver {
    /// Persistent springs per unordered pair `(a, b)` with `a < b`.
    springs: HashMap<(u32, u32), ContactSprings>,
}

impl RollingContactResolver {
    /// Builds an empty resolver with no stored contact history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of live contact springs currently stored.
    #[must_use]
    pub fn active_contacts(&self) -> usize {
        self.springs.len()
    }

    /// The stored springs for the unordered pair, or `None` when no live
    /// contact exists between those particles. The indices may be supplied in
    /// either order.
    #[must_use]
    pub fn springs(&self, i: u32, j: u32) -> Option<ContactSprings> {
        let key = if i <= j { (i, j) } else { (j, i) };
        self.springs.get(&key).copied()
    }

    /// Forgets all stored contact history, resetting the resolver to its
    /// freshly constructed state.
    pub fn clear(&mut self) {
        self.springs.clear();
    }

    /// Resolves every rotational contact in the packing described by
    /// `positions`, `radii`, `velocities`, and `angular` under the given
    /// `model`, advancing the stored springs over the time step `dt`.
    ///
    /// Returns `None` when the four slices disagree in length, when any
    /// coordinate, velocity, or angular-velocity component is non-finite, when
    /// any radius is not finite and strictly positive, or when `dt` is not
    /// finite and strictly positive. On success the returned
    /// [`RollingContactResolution`] carries the net force and torque on each
    /// particle (parallel to `positions`) and the list of resolved contacts in
    /// deterministic order; the springs of pairs that are no longer in contact
    /// are dropped.
    #[must_use]
    pub fn resolve(
        &mut self,
        positions: &[Vec3],
        radii: &[f32],
        velocities: &[Vec3],
        angular: &[Vec3],
        model: &RollingContactModel,
        dt: f32,
    ) -> Option<RollingContactResolution> {
        if !is_valid(positions, radii, velocities, angular) {
            return None;
        }
        if !(dt.is_finite() && dt > 0.0) {
            return None;
        }

        let n = positions.len();
        let mut forces = vec![Vec3::ZERO; n];
        let mut torques = vec![Vec3::ZERO; n];
        let mut contacts = Vec::new();
        if n == 0 {
            self.springs.clear();
            return Some(RollingContactResolution {
                forces,
                torques,
                contacts,
            });
        }

        // Cell size = largest possible contact distance.
        let max_radius = radii.iter().copied().fold(0.0_f32, f32::max);
        let cell_size = 2.0 * max_radius;
        if !(cell_size.is_finite() && cell_size > 0.0) {
            return None;
        }

        let origin = positions
            .iter()
            .copied()
            .reduce(Vec3::min)
            .unwrap_or(Vec3::ZERO);

        // Bin particle indices into their grid cells.
        let mut grid: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        for (i, &p) in positions.iter().enumerate() {
            grid.entry(cell_of(p, origin, cell_size))
                .or_default()
                .push(i as u32);
        }

        // Pairs that are still in contact this step; everything else is pruned.
        let mut touched: HashSet<(u32, u32)> = HashSet::new();

        for (i, &pi) in positions.iter().enumerate() {
            let base = cell_of(pi, origin, cell_size);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let cell = [base[0] + dx, base[1] + dy, base[2] + dz];
                        let Some(bucket) = grid.get(&cell) else {
                            continue;
                        };
                        for &jraw in bucket {
                            let j = jraw as usize;
                            // Evaluate each unordered pair once, lower index first.
                            if j <= i {
                                continue;
                            }
                            let key = (i as u32, j as u32);
                            let mut springs = self.springs.get(&key).copied().unwrap_or_default();
                            let Some(contact) = rotational_contact_between(
                                model,
                                (pi, positions[j]),
                                (radii[i], radii[j]),
                                (velocities[i], velocities[j]),
                                (angular[i], angular[j]),
                                &mut springs,
                                dt,
                            ) else {
                                continue;
                            };
                            self.springs.insert(key, springs);
                            touched.insert(key);
                            forces[j] += contact.force_on_b;
                            forces[i] -= contact.force_on_b;
                            torques[i] += contact.torque_on_a;
                            torques[j] += contact.torque_on_b;
                            contacts.push(RollingPairContact {
                                a: i as u32,
                                b: j as u32,
                                overlap: contact.overlap,
                                normal_magnitude: contact.normal_magnitude,
                                tangential_magnitude: contact.tangential_magnitude,
                                rolling_magnitude: contact.rolling_magnitude,
                                sliding: contact.sliding,
                            });
                        }
                    }
                }
            }
        }

        // Drop springs for pairs that have separated since the last step.
        self.springs.retain(|k, _| touched.contains(k));

        // Deterministic order independent of grid-bucket iteration order.
        contacts.sort_unstable_by_key(|c| (c.a, c.b));
        Some(RollingContactResolution {
            forces,
            torques,
            contacts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> RollingContactModel {
        // (kₙ,γₙ), (k_t,γ_t,μ), (k_r,γ_r,μ_r).
        RollingContactModel::new((1.0e5, 0.0), (1.0e5, 0.0, 0.5), (1.0e4, 0.0, 0.3)).unwrap()
    }

    #[test]
    fn empty_packing_resolves_to_nothing() {
        let mut r = RollingContactResolver::new();
        let out = r
            .resolve(&[], &[], &[], &[], &model(), 1.0e-3)
            .expect("empty is valid");
        assert_eq!(out.contact_count(), 0);
        assert!(out.forces.is_empty());
        assert!(out.torques.is_empty());
        assert_eq!(r.active_contacts(), 0);
    }

    #[test]
    fn mismatched_lengths_and_bad_dt_return_none() {
        let mut r = RollingContactResolver::new();
        let p = [Vec3::ZERO, Vec3::X];
        let rad = [1.0_f32];
        let v = [Vec3::ZERO, Vec3::ZERO];
        let w = [Vec3::ZERO, Vec3::ZERO];
        assert!(r.resolve(&p, &rad, &v, &w, &model(), 1.0e-3).is_none());
        let rad2 = [1.0_f32, 1.0];
        assert!(r.resolve(&p, &rad2, &v, &w, &model(), 0.0).is_none());
        assert!(r.resolve(&p, &rad2, &v, &w, &model(), f32::NAN).is_none());
        let bad = [Vec3::new(f32::NAN, 0.0, 0.0), Vec3::X];
        assert!(r.resolve(&bad, &rad2, &v, &w, &model(), 1.0e-3).is_none());
        let badw = [Vec3::new(f32::INFINITY, 0.0, 0.0), Vec3::ZERO];
        assert!(r.resolve(&p, &rad2, &v, &badw, &model(), 1.0e-3).is_none());
    }

    #[test]
    fn overlapping_pair_conserves_linear_momentum() {
        let mut r = RollingContactResolver::new();
        let p = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let rad = [1.0_f32, 1.0];
        // b slides tangentially so friction is non-zero.
        let v = [Vec3::ZERO, Vec3::new(0.0, 0.01, 0.0)];
        let w = [Vec3::ZERO, Vec3::ZERO];
        let out = r
            .resolve(&p, &rad, &v, &w, &model(), 1.0e-4)
            .expect("valid");
        assert_eq!(out.contact_count(), 1);
        assert_eq!(out.contacts[0].a, 0);
        assert_eq!(out.contacts[0].b, 1);
        assert!(out.contacts[0].normal_magnitude > 0.0);
        assert!(out.contacts[0].tangential_magnitude > 0.0);
        assert!(out.total_force().length() < 1e-2, "internal forces cancel");
        // Equal and opposite force on the two grains.
        assert!((out.forces[0] + out.forces[1]).length() < 1e-2);
    }

    #[test]
    fn spinning_pair_conserves_angular_momentum() {
        let mut r = RollingContactResolver::new();
        let p = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let rad = [1.0_f32, 1.0];
        // Both grains spin and slide so friction torques and rolling couples
        // are both excited.
        let v = [Vec3::ZERO, Vec3::new(0.0, 0.02, 0.0)];
        let w = [Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -0.5)];
        let out = r
            .resolve(&p, &rad, &v, &w, &model(), 1.0e-4)
            .expect("valid");
        assert_eq!(out.contact_count(), 1);
        assert!(out.max_rolling_torque() > 0.0, "rolling couple is active");
        let rate = out.total_angular_momentum_rate(&p);
        assert!(
            rate.length() < 1e-2,
            "angular momentum conserved, got {rate:?}"
        );
    }

    #[test]
    fn springs_persist_across_steps() {
        let mut r = RollingContactResolver::new();
        let p = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let rad = [1.0_f32, 1.0];
        let v = [Vec3::ZERO, Vec3::new(0.0, 0.01, 0.0)];
        let w = [Vec3::ZERO, Vec3::ZERO];
        let _ = r
            .resolve(&p, &rad, &v, &w, &model(), 1.0e-4)
            .expect("valid");
        assert_eq!(r.active_contacts(), 1);
        let s0 = r.springs(0, 1).expect("live contact");
        // The tangential spring accumulated some sliding displacement.
        assert!(s0.tangential.length() > 0.0);
        // Index order does not matter.
        assert_eq!(r.springs(1, 0), r.springs(0, 1));
        // A second step grows the spring further.
        let _ = r
            .resolve(&p, &rad, &v, &w, &model(), 1.0e-4)
            .expect("valid");
        let s1 = r.springs(0, 1).expect("still live");
        assert!(s1.tangential.length() > s0.tangential.length());
    }

    #[test]
    fn separated_pair_prunes_its_springs() {
        let mut r = RollingContactResolver::new();
        let rad = [1.0_f32, 1.0];
        let v = [Vec3::ZERO, Vec3::new(0.0, 0.01, 0.0)];
        let w = [Vec3::ZERO, Vec3::ZERO];
        let touching = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let _ = r
            .resolve(&touching, &rad, &v, &w, &model(), 1.0e-4)
            .expect("valid");
        assert_eq!(r.active_contacts(), 1);
        // Move them well apart; the spring must be dropped.
        let apart = [Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)];
        let out = r
            .resolve(&apart, &rad, &v, &w, &model(), 1.0e-4)
            .expect("valid");
        assert_eq!(out.contact_count(), 0);
        assert_eq!(r.active_contacts(), 0);
        assert!(r.springs(0, 1).is_none());
    }

    #[test]
    fn clear_forgets_all_history() {
        let mut r = RollingContactResolver::new();
        let p = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let rad = [1.0_f32, 1.0];
        let v = [Vec3::ZERO, Vec3::new(0.0, 0.01, 0.0)];
        let w = [Vec3::ZERO, Vec3::ZERO];
        let _ = r
            .resolve(&p, &rad, &v, &w, &model(), 1.0e-4)
            .expect("valid");
        assert_eq!(r.active_contacts(), 1);
        r.clear();
        assert_eq!(r.active_contacts(), 0);
        assert!(r.springs(0, 1).is_none());
    }

    #[test]
    fn three_grain_chain_finds_both_contacts() {
        let mut r = RollingContactResolver::new();
        // Collinear, each neighbouring pair overlapping but the ends not.
        let p = [
            Vec3::ZERO,
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let rad = [1.0_f32, 1.0, 1.0];
        let v = [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        let w = [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        let out = r
            .resolve(&p, &rad, &v, &w, &model(), 1.0e-4)
            .expect("valid");
        assert_eq!(out.contact_count(), 2);
        assert_eq!((out.contacts[0].a, out.contacts[0].b), (0, 1));
        assert_eq!((out.contacts[1].a, out.contacts[1].b), (1, 2));
        // The ends (0,2) are 3.0 apart, radius sum 2.0 — no contact.
        assert!(r.springs(0, 2).is_none());
    }

    #[test]
    fn deterministic_contact_ordering() {
        let mut r = RollingContactResolver::new();
        // A small cluster; every pair overlaps.
        let p = [
            Vec3::ZERO,
            Vec3::new(1.2, 0.0, 0.0),
            Vec3::new(0.6, 1.0, 0.0),
        ];
        let rad = [1.0_f32, 1.0, 1.0];
        let v = [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        let w = [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        let out = r
            .resolve(&p, &rad, &v, &w, &model(), 1.0e-4)
            .expect("valid");
        let keys: Vec<(u32, u32)> = out.contacts.iter().map(|c| (c.a, c.b)).collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted, "contacts reported in sorted (a,b) order");
        for c in &out.contacts {
            assert!(c.a < c.b, "lower index first");
        }
    }

    #[test]
    fn rolling_resistance_builds_a_resisting_torque() {
        // Two grains rolling over one another (opposite spins, no slide) must
        // feel a rolling resistance torque that opposes the relative rotation.
        let mut r = RollingContactResolver::new();
        let p = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let rad = [1.0_f32, 1.0];
        let v = [Vec3::ZERO, Vec3::ZERO];
        let w = [Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)];
        let out = r
            .resolve(&p, &rad, &v, &w, &model(), 1.0e-4)
            .expect("valid");
        assert_eq!(out.contact_count(), 1);
        assert!(out.max_rolling_torque() > 0.0);
        // Relative rotation ω_a−ω_b = +2 ẑ, so the couple on a is −ẑ.
        assert!(out.torques[0].z < 0.0, "couple opposes a's relative spin");
        assert!(out.torques[1].z > 0.0, "equal and opposite couple on b");
    }
}
