//! Broad-phase accumulation of Cundall–Strack tangential-history contacts with
//! persistent per-contact friction springs.
//!
//! The soft-sphere and Hertz resolvers
//! ([`bonded_particle_contact_resolver`](super::bonded_particle_contact_resolver),
//! [`hertz_contact_resolver`](super::hertz_contact_resolver)) are *stateless*:
//! each call recomputes a packing's contact forces from scratch, so their
//! friction term is purely viscous and vanishes at zero sliding velocity. A
//! packing of grains resolved that way can never come to rest on a slope — it
//! creeps indefinitely.
//!
//! This resolver is the *stateful* counterpart built on the
//! [`tangential_history_contact`](super::tangential_history_contact)
//! (Cundall–Strack) law. It owns a map of tangential displacement springs keyed
//! by the unordered particle pair `(a, b)`. Each [`resolve`](TangentialHistoryResolver::resolve)
//! call advances every live contact's spring and — crucially — **drops the
//! spring of any pair that is no longer in contact**, so a reopened contact
//! starts fresh. The persistent elastic memory is what lets a pile of grains
//! sustain a static friction force at zero velocity and stand at a true angle
//! of repose.
//!
//! # Acceleration
//!
//! The broad phase is identical to the stateless resolvers: particle centres
//! are binned into a uniform spatial-hash grid whose cell size is the largest
//! possible contact distance `2·R_max`, and each particle tests only the 27
//! cells of its `3×3×3` neighbourhood. Each unordered pair is evaluated once
//! with the lower index first, and contacts are reported in a deterministic
//! `(a, b)` order independent of grid-bucket iteration order.
//!
//! # Forces and momentum
//!
//! For each contacting pair the resolver calls
//! [`tangential_history_between`](super::tangential_history_contact::tangential_history_between),
//! which returns the force on `b`; the equal and opposite force acts on `a`.
//! The per-particle `forces` therefore sum to zero (up to floating-point
//! error), so a resolved packing conserves linear momentum exactly.

use crate::collider::tangential_history_contact::{tangential_history_between, CundallStrackModel};
use glam::Vec3;
use std::collections::{HashMap, HashSet};

/// A single resolved Cundall–Strack contact between two particles of a packing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TangentialContact {
    /// Lower particle index of the contacting pair.
    pub a: u32,
    /// Higher particle index of the contacting pair.
    pub b: u32,
    /// Overlap `δ > 0` resolved by this contact.
    pub overlap: f32,
    /// Normal force magnitude `F_n ≥ 0`.
    pub normal_magnitude: f32,
    /// Tangential (friction) force magnitude actually applied.
    pub tangential_magnitude: f32,
    /// Whether the tangential force reached the Coulomb limit (sliding).
    pub sliding: bool,
}

/// Result of resolving every Cundall–Strack contact in a particle packing.
#[derive(Clone, Debug, PartialEq)]
pub struct TangentialHistoryResolution {
    /// Net contact force on each particle, indexed in parallel with the input
    /// `positions`; always the same length as the packing.
    pub forces: Vec<Vec3>,
    /// The resolved contacts, in deterministic `(a, b)` order.
    pub contacts: Vec<TangentialContact>,
}

impl TangentialHistoryResolution {
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

    /// Sum of all per-particle forces. The contact forces are internal and
    /// equal-and-opposite, so this is zero up to floating-point error; it is
    /// exposed as a conservation diagnostic.
    #[must_use]
    pub fn total_force(&self) -> Vec3 {
        self.forces.iter().copied().sum()
    }

    /// Number of contacts whose tangential force reached the Coulomb limit.
    #[must_use]
    pub fn sliding_count(&self) -> usize {
        self.contacts.iter().filter(|c| c.sliding).count()
    }
}

/// Validates that the packing is well formed: equal-length `positions`,
/// `radii`, and `velocities`; every coordinate finite; every radius finite and
/// strictly positive; every velocity component finite.
fn is_valid(positions: &[Vec3], radii: &[f32], velocities: &[Vec3]) -> bool {
    if positions.len() != radii.len() || positions.len() != velocities.len() {
        return false;
    }
    let finite_vec = |v: &Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    positions.iter().all(finite_vec)
        && velocities.iter().all(finite_vec)
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

/// A stateful resolver that carries persistent Cundall–Strack friction springs
/// across time steps.
///
/// Create one with [`TangentialHistoryResolver::new`] and call
/// [`resolve`](TangentialHistoryResolver::resolve) once per step with the
/// current packing state; the resolver advances the stored springs and prunes
/// contacts that have separated. The same resolver instance (and therefore the
/// same spring map) must be reused across steps for a given packing — a fresh
/// resolver has no friction history and behaves like the stateless law on its
/// first step.
#[derive(Clone, Debug, Default)]
pub struct TangentialHistoryResolver {
    /// Tangential displacement spring per unordered pair `(a, b)` with `a < b`.
    springs: HashMap<(u32, u32), Vec3>,
}

impl TangentialHistoryResolver {
    /// Builds an empty resolver with no stored friction history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of live contact springs currently stored.
    #[must_use]
    pub fn active_contacts(&self) -> usize {
        self.springs.len()
    }

    /// The stored tangential spring displacement for the unordered pair, or
    /// `None` when no live contact exists between those particles. The indices
    /// may be supplied in either order.
    #[must_use]
    pub fn spring(&self, i: u32, j: u32) -> Option<Vec3> {
        let key = if i <= j { (i, j) } else { (j, i) };
        self.springs.get(&key).copied()
    }

    /// Forgets all stored friction history, resetting the resolver to its
    /// freshly constructed state.
    pub fn clear(&mut self) {
        self.springs.clear();
    }

    /// Resolves every Cundall–Strack contact in the packing described by
    /// `positions`, `radii`, and `velocities` under the given `model`, advancing
    /// the stored springs over the time step `dt`.
    ///
    /// Returns `None` when the three slices disagree in length, when any
    /// coordinate or velocity component is non-finite, when any radius is not
    /// finite and strictly positive, or when `dt` is not finite and strictly
    /// positive. On success the returned [`TangentialHistoryResolution`] carries
    /// the net force on each particle (parallel to `positions`) and the list of
    /// resolved contacts in deterministic order; the springs of pairs that are
    /// no longer in contact are dropped.
    #[must_use]
    pub fn resolve(
        &mut self,
        positions: &[Vec3],
        radii: &[f32],
        velocities: &[Vec3],
        model: &CundallStrackModel,
        dt: f32,
    ) -> Option<TangentialHistoryResolution> {
        if !is_valid(positions, radii, velocities) {
            return None;
        }
        if !(dt.is_finite() && dt > 0.0) {
            return None;
        }

        let n = positions.len();
        let mut forces = vec![Vec3::ZERO; n];
        let mut contacts = Vec::new();
        if n == 0 {
            self.springs.clear();
            return Some(TangentialHistoryResolution { forces, contacts });
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
                            let mut spring = self.springs.get(&key).copied().unwrap_or(Vec3::ZERO);
                            let Some(contact) = tangential_history_between(
                                model,
                                (pi, positions[j]),
                                (radii[i], radii[j]),
                                (velocities[i], velocities[j]),
                                &mut spring,
                                dt,
                            ) else {
                                continue;
                            };
                            self.springs.insert(key, spring);
                            touched.insert(key);
                            forces[j] += contact.force_on_b;
                            forces[i] -= contact.force_on_b;
                            contacts.push(TangentialContact {
                                a: i as u32,
                                b: j as u32,
                                overlap: contact.overlap,
                                normal_magnitude: contact.normal_magnitude,
                                tangential_magnitude: contact.tangential_magnitude,
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
        Some(TangentialHistoryResolution { forces, contacts })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CundallStrackModel {
        // kₙ, γₙ, k_t, γ_t, μ.
        CundallStrackModel::new(1.0e5, 0.0, 1.0e5, 0.0, 0.5).unwrap()
    }

    #[test]
    fn empty_packing_resolves_to_nothing() {
        let mut r = TangentialHistoryResolver::new();
        let out = r
            .resolve(&[], &[], &[], &model(), 1.0e-3)
            .expect("empty is valid");
        assert_eq!(out.contact_count(), 0);
        assert!(out.forces.is_empty());
        assert_eq!(r.active_contacts(), 0);
    }

    #[test]
    fn mismatched_lengths_and_bad_dt_return_none() {
        let mut r = TangentialHistoryResolver::new();
        let p = [Vec3::ZERO, Vec3::X];
        let rad = [1.0_f32];
        let v = [Vec3::ZERO, Vec3::ZERO];
        assert!(r.resolve(&p, &rad, &v, &model(), 1.0e-3).is_none());
        let rad2 = [1.0_f32, 1.0];
        assert!(r.resolve(&p, &rad2, &v, &model(), 0.0).is_none());
        assert!(r.resolve(&p, &rad2, &v, &model(), f32::NAN).is_none());
        let bad = [Vec3::new(f32::NAN, 0.0, 0.0), Vec3::X];
        assert!(r.resolve(&bad, &rad2, &v, &model(), 1.0e-3).is_none());
    }

    #[test]
    fn overlapping_pair_conserves_momentum() {
        let mut r = TangentialHistoryResolver::new();
        let p = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let rad = [1.0_f32, 1.0];
        // b slides tangentially so friction is non-zero.
        let v = [Vec3::ZERO, Vec3::new(0.0, 0.01, 0.0)];
        let out = r.resolve(&p, &rad, &v, &model(), 1.0e-4).expect("valid");
        assert_eq!(out.contact_count(), 1);
        assert_eq!(out.contacts[0].a, 0);
        assert_eq!(out.contacts[0].b, 1);
        assert!(out.contacts[0].normal_magnitude > 0.0);
        assert!(out.contacts[0].tangential_magnitude > 0.0);
        assert!(out.total_force().length() < 1e-2, "internal forces cancel");
        // Equal and opposite.
        assert!((out.forces[0] + out.forces[1]).length() < 1e-2);
        assert_eq!(r.active_contacts(), 1);
    }

    #[test]
    fn persistent_spring_grows_across_steps_while_sticking() {
        let mut r = TangentialHistoryResolver::new();
        let p = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let rad = [1.0_f32, 1.0];
        let v = [Vec3::ZERO, Vec3::new(0.0, 0.01, 0.0)];
        let m = model();
        let mut last = 0.0_f32;
        for step in 1..=5 {
            let out = r.resolve(&p, &rad, &v, &m, 1.0e-4).expect("valid");
            assert_eq!(out.sliding_count(), 0, "slow slip must stick, step {step}");
            let t = out.contacts[0].tangential_magnitude;
            assert!(t > last, "static friction accumulates across steps");
            last = t;
        }
        // The spring stored for the pair must have grown from zero.
        let spring = r.spring(0, 1).expect("live contact");
        assert!(spring.length() > 0.0);
        // Index order is irrelevant to the query.
        assert_eq!(r.spring(1, 0), r.spring(0, 1));
    }

    #[test]
    fn losing_contact_drops_the_spring() {
        let mut r = TangentialHistoryResolver::new();
        let rad = [1.0_f32, 1.0];
        let v = [Vec3::ZERO, Vec3::new(0.0, 0.01, 0.0)];
        let m = model();
        // Step 1: overlapping, builds a spring.
        let close = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        r.resolve(&close, &rad, &v, &m, 1.0e-4).expect("valid");
        assert_eq!(r.active_contacts(), 1);
        // Step 2: pulled apart, the spring for the lost contact is pruned.
        let far = [Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)];
        let out = r.resolve(&far, &rad, &v, &m, 1.0e-4).expect("valid");
        assert_eq!(out.contact_count(), 0);
        assert_eq!(r.active_contacts(), 0);
        assert!(r.spring(0, 1).is_none());
    }

    #[test]
    fn reopened_contact_starts_from_zero_spring() {
        let mut r = TangentialHistoryResolver::new();
        let rad = [1.0_f32, 1.0];
        let v = [Vec3::ZERO, Vec3::new(0.0, 0.01, 0.0)];
        let m = model();
        let close = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        // Accumulate several steps of history.
        for _ in 0..5 {
            r.resolve(&close, &rad, &v, &m, 1.0e-4).expect("valid");
        }
        let loaded = r.spring(0, 1).expect("live").length();
        assert!(loaded > 0.0);
        // Separate (prunes), then touch again: a single step of fresh history.
        let far = [Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)];
        r.resolve(&far, &rad, &v, &m, 1.0e-4).expect("valid");
        r.resolve(&close, &rad, &v, &m, 1.0e-4).expect("valid");
        let reopened = r.spring(0, 1).expect("live").length();
        assert!(reopened > 0.0);
        assert!(
            reopened < loaded,
            "reopened contact must not keep old history"
        );
    }

    #[test]
    fn fast_slip_counts_as_sliding() {
        let mut r = TangentialHistoryResolver::new();
        let p = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let rad = [1.0_f32, 1.0];
        // Large tangential slip saturates the Coulomb cone immediately.
        let v = [Vec3::ZERO, Vec3::new(0.0, 1000.0, 0.0)];
        let out = r.resolve(&p, &rad, &v, &model(), 1.0e-3).expect("valid");
        assert_eq!(out.sliding_count(), 1);
        let cap = 0.5 * 1.0e5 * 0.5; // μ·kₙ·δ.
        assert!((out.max_normal_force() - 1.0e5 * 0.5).abs() < 1.0);
        assert!((out.contacts[0].tangential_magnitude - cap).abs() < cap * 1e-3);
    }

    #[test]
    fn contacts_are_reported_in_deterministic_order() {
        let mut r = TangentialHistoryResolver::new();
        // Three spheres in a row, each overlapping its neighbour.
        let p = [
            Vec3::ZERO,
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let rad = [1.0_f32, 1.0, 1.0];
        let v = [Vec3::ZERO; 3];
        let out = r.resolve(&p, &rad, &v, &model(), 1.0e-3).expect("valid");
        // Pairs (0,1) and (1,2) overlap; (0,2) does not.
        assert_eq!(out.contact_count(), 2);
        assert_eq!((out.contacts[0].a, out.contacts[0].b), (0, 1));
        assert_eq!((out.contacts[1].a, out.contacts[1].b), (1, 2));
        assert_eq!(r.active_contacts(), 2);
    }

    #[test]
    fn clear_forgets_all_history() {
        let mut r = TangentialHistoryResolver::new();
        let p = [Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let rad = [1.0_f32, 1.0];
        let v = [Vec3::ZERO, Vec3::new(0.0, 0.01, 0.0)];
        r.resolve(&p, &rad, &v, &model(), 1.0e-4).expect("valid");
        assert_eq!(r.active_contacts(), 1);
        r.clear();
        assert_eq!(r.active_contacts(), 0);
        assert!(r.spring(0, 1).is_none());
    }
}
