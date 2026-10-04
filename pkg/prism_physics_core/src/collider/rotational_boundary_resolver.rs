//! Stateful resolver for grain-versus-boundary rotational contacts.
//!
//! [`rotational_boundary_contact`](super::rotational_boundary_contact) resolves
//! one grain against one [`HalfSpace`]. A real scene confines a whole packing
//! with several boundaries at once — a floor plus the four walls of a hopper,
//! or the curved shell of a tumbler approximated by planes. This resolver loops
//! every grain against every boundary, carries the persistent friction and
//! rolling springs across steps keyed by `(grain, boundary)`, prunes contacts
//! that have separated, and accumulates the net force and torque on each grain.
//!
//! Unlike the grain-grain resolver, boundary forces are *external*: the wall is
//! immovable and absorbs the reaction, so [`BoundaryContactResolution::total_force`]
//! is the net boundary force on the packing rather than a conservation
//! diagnostic.

use super::rotational_boundary_contact::{grain_boundary_contact, HalfSpace};
use super::rotational_contact::{ContactSprings, RollingContactModel};
use glam::Vec3;
use std::collections::HashMap;

/// A single resolved contact between a grain and a boundary plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryPairContact {
    /// Index of the grain in the packing.
    pub grain: u32,
    /// Index of the boundary plane.
    pub plane: u32,
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

/// Result of resolving every grain-versus-boundary contact in a packing.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryContactResolution {
    /// Net boundary force on each grain, indexed in parallel with the input
    /// `positions`; always the same length as the packing.
    pub forces: Vec<Vec3>,
    /// Net boundary torque on each grain, indexed in parallel with the input
    /// `positions`; always the same length as the packing.
    pub torques: Vec<Vec3>,
    /// The resolved contacts, in deterministic `(grain, plane)` order.
    pub contacts: Vec<BoundaryPairContact>,
}

impl BoundaryContactResolution {
    /// Number of resolved boundary contacts.
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

    /// Net boundary force on the whole packing. Boundary forces are external
    /// (the wall absorbs the reaction), so this is the resultant the boundaries
    /// exert, not a conservation diagnostic.
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
/// `radii`, `velocities`, and `angular`; every coordinate, velocity, and
/// angular-velocity component finite; every radius finite and strictly
/// positive.
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

/// A stateful resolver that carries persistent grain-versus-boundary springs
/// across time steps.
///
/// Create one with [`BoundaryContactResolver::new`] and call
/// [`resolve`](BoundaryContactResolver::resolve) once per step with the current
/// packing state and the set of boundary planes; the resolver advances the
/// stored springs and prunes contacts that have separated. The same resolver
/// instance (and therefore the same spring map) must be reused across steps for
/// a given packing so that friction and rolling history persist.
#[derive(Clone, Debug, Default)]
pub struct BoundaryContactResolver {
    /// Persistent springs per `(grain, plane)` contact.
    springs: HashMap<(u32, u32), ContactSprings>,
}

impl BoundaryContactResolver {
    /// Builds an empty resolver with no stored contact history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of live boundary-contact springs currently stored.
    #[must_use]
    pub fn active_contacts(&self) -> usize {
        self.springs.len()
    }

    /// The stored springs for the `(grain, plane)` contact, or `None` when no
    /// live contact exists.
    #[must_use]
    pub fn springs(&self, grain: u32, plane: u32) -> Option<ContactSprings> {
        self.springs.get(&(grain, plane)).copied()
    }

    /// Forgets all stored contact history, resetting the resolver to its
    /// freshly constructed state.
    pub fn clear(&mut self) {
        self.springs.clear();
    }

    /// Resolves every grain-versus-boundary contact for the packing described
    /// by `positions`, `radii`, `velocities`, and `angular` against the set of
    /// `planes` under the given `model`, advancing the stored springs over the
    /// time step `dt`.
    ///
    /// Returns `None` when the four packing slices disagree in length, when any
    /// coordinate, velocity, or angular-velocity component is non-finite, when
    /// any radius is not finite and strictly positive, or when `dt` is not
    /// finite and strictly positive. On success the returned
    /// [`BoundaryContactResolution`] carries the net boundary force and torque
    /// on each grain (parallel to `positions`) and the list of resolved
    /// contacts in deterministic `(grain, plane)` order; the springs of
    /// contacts that have separated are dropped. A packing with more than
    /// `u32::MAX` grains or a plane set larger than `u32::MAX` is rejected.
    #[must_use]
    pub fn resolve(
        &mut self,
        positions: &[Vec3],
        radii: &[f32],
        velocities: &[Vec3],
        angular: &[Vec3],
        planes: &[HalfSpace],
        model: &RollingContactModel,
        dt: f32,
    ) -> Option<BoundaryContactResolution> {
        if !is_valid(positions, radii, velocities, angular) {
            return None;
        }
        if !(dt.is_finite() && dt > 0.0) {
            return None;
        }
        if positions.len() > u32::MAX as usize || planes.len() > u32::MAX as usize {
            return None;
        }

        let n = positions.len();
        let mut forces = vec![Vec3::ZERO; n];
        let mut torques = vec![Vec3::ZERO; n];
        let mut contacts = Vec::new();
        let mut live = HashMap::new();

        for grain in 0..n {
            for (plane_idx, plane) in planes.iter().enumerate() {
                let key = (grain as u32, plane_idx as u32);
                let mut springs = self.springs.get(&key).copied().unwrap_or_default();
                let contact = grain_boundary_contact(
                    model,
                    plane,
                    positions[grain],
                    radii[grain],
                    (velocities[grain], angular[grain]),
                    &mut springs,
                    dt,
                );
                if let Some(c) = contact {
                    forces[grain] += c.force;
                    torques[grain] += c.torque;
                    contacts.push(BoundaryPairContact {
                        grain: key.0,
                        plane: key.1,
                        overlap: c.overlap,
                        normal_magnitude: c.normal_magnitude,
                        tangential_magnitude: c.tangential_magnitude,
                        rolling_magnitude: c.rolling_magnitude,
                        sliding: c.sliding,
                    });
                    live.insert(key, springs);
                }
            }
        }

        self.springs = live;
        Some(BoundaryContactResolution {
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
        RollingContactModel::new((1.0e5, 0.0), (1.0e5, 0.0, 0.5), (1.0e4, 0.0, 0.3)).unwrap()
    }

    fn floor() -> HalfSpace {
        HalfSpace::new(Vec3::ZERO, Vec3::Z).unwrap()
    }

    #[test]
    fn empty_packing_resolves_to_nothing() {
        let mut r = BoundaryContactResolver::new();
        let res = r
            .resolve(&[], &[], &[], &[], &[floor()], &model(), 1.0e-3)
            .unwrap();
        assert_eq!(res.contact_count(), 0);
        assert_eq!(res.forces.len(), 0);
        assert_eq!(r.active_contacts(), 0);
    }

    #[test]
    fn mismatched_lengths_and_bad_dt_return_none() {
        let mut r = BoundaryContactResolver::new();
        let m = model();
        let planes = [floor()];
        assert!(r
            .resolve(
                &[Vec3::ZERO],
                &[1.0, 2.0],
                &[Vec3::ZERO],
                &[Vec3::ZERO],
                &planes,
                &m,
                1.0e-3,
            )
            .is_none());
        assert!(r
            .resolve(
                &[Vec3::ZERO],
                &[1.0],
                &[Vec3::ZERO],
                &[Vec3::ZERO],
                &planes,
                &m,
                0.0,
            )
            .is_none());
    }

    #[test]
    fn resting_grain_gets_upward_force_only() {
        let mut r = BoundaryContactResolver::new();
        let radius = 1.0;
        let depth = 0.01;
        let positions = [Vec3::new(0.0, 0.0, radius - depth)];
        let res = r
            .resolve(
                &positions,
                &[radius],
                &[Vec3::ZERO],
                &[Vec3::ZERO],
                &[floor()],
                &model(),
                1.0e-3,
            )
            .unwrap();
        assert_eq!(res.contact_count(), 1);
        assert_eq!(r.active_contacts(), 1);
        let f = res.forces[0];
        assert!(f.z > 0.0 && f.x.abs() < 1.0e-4 && f.y.abs() < 1.0e-4);
        assert!((res.total_force() - f).length() < 1.0e-6);
        assert_eq!(res.sliding_count(), 0);
    }

    #[test]
    fn separated_grain_prunes_its_spring() {
        let mut r = BoundaryContactResolver::new();
        let radius = 1.0;
        // First step: grain in contact, builds a spring.
        let touching = [Vec3::new(0.0, 0.0, radius - 0.01)];
        let _ = r
            .resolve(
                &touching,
                &[radius],
                &[Vec3::new(0.2, 0.0, 0.0)],
                &[Vec3::ZERO],
                &[floor()],
                &model(),
                1.0e-3,
            )
            .unwrap();
        assert_eq!(r.active_contacts(), 1);
        assert!(r.springs(0, 0).is_some());
        // Second step: grain lifted away, spring is pruned.
        let lifted = [Vec3::new(0.0, 0.0, radius + 1.0)];
        let res = r
            .resolve(
                &lifted,
                &[radius],
                &[Vec3::ZERO],
                &[Vec3::ZERO],
                &[floor()],
                &model(),
                1.0e-3,
            )
            .unwrap();
        assert_eq!(res.contact_count(), 0);
        assert_eq!(r.active_contacts(), 0);
        assert!(r.springs(0, 0).is_none());
    }

    #[test]
    fn corner_grain_touches_two_planes() {
        let mut r = BoundaryContactResolver::new();
        let radius = 1.0;
        let depth = 0.02;
        let floor = HalfSpace::new(Vec3::ZERO, Vec3::Z).unwrap();
        let wall = HalfSpace::new(Vec3::ZERO, Vec3::X).unwrap();
        // Grain pressed into both the floor (z) and the wall (x).
        let center = Vec3::new(radius - depth, 5.0, radius - depth);
        let res = r
            .resolve(
                &[center],
                &[radius],
                &[Vec3::ZERO],
                &[Vec3::ZERO],
                &[floor, wall],
                &model(),
                1.0e-3,
            )
            .unwrap();
        assert_eq!(res.contact_count(), 2);
        assert_eq!(r.active_contacts(), 2);
        // Net force has positive x (from wall) and positive z (from floor).
        let f = res.forces[0];
        assert!(f.x > 0.0 && f.z > 0.0);
        // Contacts are reported in (grain, plane) order.
        assert_eq!(res.contacts[0].plane, 0);
        assert_eq!(res.contacts[1].plane, 1);
    }

    #[test]
    fn diagnostics_track_largest_values() {
        let mut r = BoundaryContactResolver::new();
        let radius = 1.0;
        let positions = [
            Vec3::new(0.0, 0.0, radius - 0.01),
            Vec3::new(5.0, 0.0, radius - 0.03),
        ];
        let res = r
            .resolve(
                &positions,
                &[radius, radius],
                &[Vec3::ZERO, Vec3::ZERO],
                &[Vec3::ZERO, Vec3::ZERO],
                &[floor()],
                &model(),
                1.0e-3,
            )
            .unwrap();
        assert_eq!(res.contact_count(), 2);
        assert!((res.max_overlap() - 0.03).abs() < 1.0e-6);
        // Deeper grain carries the larger normal force.
        assert!((res.max_normal_force() - 1.0e5 * 0.03).abs() < 1.0e-1);
    }

    #[test]
    fn clear_forgets_history() {
        let mut r = BoundaryContactResolver::new();
        let radius = 1.0;
        let positions = [Vec3::new(0.0, 0.0, radius - 0.01)];
        let _ = r
            .resolve(
                &positions,
                &[radius],
                &[Vec3::ZERO],
                &[Vec3::ZERO],
                &[floor()],
                &model(),
                1.0e-3,
            )
            .unwrap();
        assert_eq!(r.active_contacts(), 1);
        r.clear();
        assert_eq!(r.active_contacts(), 0);
        assert!(r.springs(0, 0).is_none());
    }

    #[test]
    fn sliding_grain_is_counted() {
        let mut r = BoundaryContactResolver::new();
        let radius = 1.0;
        let positions = [Vec3::new(0.0, 0.0, radius - 0.01)];
        let res = r
            .resolve(
                &positions,
                &[radius],
                &[Vec3::new(10.0, 0.0, 0.0)],
                &[Vec3::ZERO],
                &[floor()],
                &model(),
                1.0e-3,
            )
            .unwrap();
        assert_eq!(res.sliding_count(), 1);
        assert!(res.contacts[0].sliding);
    }
}
