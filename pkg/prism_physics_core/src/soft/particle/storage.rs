//! Structure-of-Arrays storage for soft-body / cloth / rope particles.
//!
//! Every deformable primitive in the unified XPBD kernel is expressed as a set
//! of point-mass *particles* coupled by *constraints*. This store keeps the
//! per-particle state in parallel columns (positions, previous positions,
//! velocities, inverse masses) so the solver can stream over them with good
//! cache behaviour and, later, hand raw slices to a data-parallel backend.
//!
//! A particle with an inverse mass of zero is *pinned*: it has effectively
//! infinite mass and is never moved by constraint projection, which is how
//! cloth corners, rope ends, and attachment points are anchored.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! Structure-of-Arrays particle layout and the inverse-mass (`w = 1/m`, with
//! `w = 0` for pinned points) convention are standard, publicly documented
//! position-based-dynamics techniques (Müller et al.).

use glam::Vec3;

use crate::math::scalar::Real;

use super::handle::ParticleHandle;

/// A Structure-of-Arrays store of point-mass particles.
///
/// The four columns are index-aligned: for a valid slot `i`,
/// [`positions`](Self::positions)`[i]` is the current position,
/// [`prev_positions`](Self::prev_positions)`[i]` is the position at the start
/// of the current substep (used by XPBD to recover velocities),
/// [`velocities`](Self::velocities)`[i]` is the current velocity, and
/// [`inverse_masses`](Self::inverse_masses)`[i]` is `1/m` (or `0` when pinned).
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ParticleStorage {
    positions: Vec<Vec3>,
    prev_positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    inverse_masses: Vec<Real>,
}

impl ParticleStorage {
    /// Creates an empty particle store.
    #[must_use]
    pub const fn new() -> ParticleStorage {
        ParticleStorage {
            positions: Vec::new(),
            prev_positions: Vec::new(),
            velocities: Vec::new(),
            inverse_masses: Vec::new(),
        }
    }

    /// Creates an empty store pre-allocated for at least `capacity` particles.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> ParticleStorage {
        ParticleStorage {
            positions: Vec::with_capacity(capacity),
            prev_positions: Vec::with_capacity(capacity),
            velocities: Vec::with_capacity(capacity),
            inverse_masses: Vec::with_capacity(capacity),
        }
    }

    /// Spawns a dynamic particle at `position` with the given (finite,
    /// positive) `mass`, returning its handle.
    ///
    /// The stored inverse mass is `1/mass`. A non-positive or non-finite
    /// `mass` is treated as *pinned* (inverse mass `0`), which is the safe
    /// interpretation of "no finite mass to move".
    pub fn spawn(&mut self, position: Vec3, mass: Real) -> ParticleHandle {
        let inverse_mass = if mass > 0.0 && mass.is_finite() {
            1.0 / mass
        } else {
            0.0
        };
        self.push(position, inverse_mass)
    }

    /// Spawns a *pinned* particle at `position` (inverse mass `0`), returning
    /// its handle. Pinned particles are immovable anchors.
    pub fn spawn_pinned(&mut self, position: Vec3) -> ParticleHandle {
        self.push(position, 0.0)
    }

    fn push(&mut self, position: Vec3, inverse_mass: Real) -> ParticleHandle {
        let index = self.positions.len() as u32;
        self.positions.push(position);
        self.prev_positions.push(position);
        self.velocities.push(Vec3::ZERO);
        self.inverse_masses.push(inverse_mass);
        ParticleHandle::from_index(index)
    }

    /// Returns the number of particles in the store.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns `true` when the store holds no particles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Returns `true` when `handle` refers to a live slot in this store.
    #[must_use]
    pub fn contains(&self, handle: ParticleHandle) -> bool {
        handle.index() < self.positions.len()
    }

    /// Returns the current position of `handle`, or `None` if it is out of
    /// range.
    #[must_use]
    pub fn position(&self, handle: ParticleHandle) -> Option<Vec3> {
        self.positions.get(handle.index()).copied()
    }

    /// Returns the current velocity of `handle`, or `None` if it is out of
    /// range.
    #[must_use]
    pub fn velocity(&self, handle: ParticleHandle) -> Option<Vec3> {
        self.velocities.get(handle.index()).copied()
    }

    /// Returns the inverse mass of `handle`, or `None` if it is out of range.
    #[must_use]
    pub fn inverse_mass(&self, handle: ParticleHandle) -> Option<Real> {
        self.inverse_masses.get(handle.index()).copied()
    }

    /// Returns `true` when `handle` refers to a pinned particle (inverse mass
    /// `0`). Out-of-range handles report `false`.
    #[must_use]
    pub fn is_pinned(&self, handle: ParticleHandle) -> bool {
        self.inverse_masses
            .get(handle.index())
            .is_some_and(|&w| w == 0.0)
    }

    /// Overwrites the current position of `handle`. Does nothing for an
    /// out-of-range handle.
    pub fn set_position(&mut self, handle: ParticleHandle, position: Vec3) {
        if let Some(slot) = self.positions.get_mut(handle.index()) {
            *slot = position;
        }
    }

    /// Overwrites the current velocity of `handle`. Does nothing for an
    /// out-of-range handle.
    pub fn set_velocity(&mut self, handle: ParticleHandle, velocity: Vec3) {
        if let Some(slot) = self.velocities.get_mut(handle.index()) {
            *slot = velocity;
        }
    }

    /// Sets the mass of `handle` (recomputing its inverse mass). A non-positive
    /// or non-finite mass pins the particle. Does nothing for an out-of-range
    /// handle.
    pub fn set_mass(&mut self, handle: ParticleHandle, mass: Real) {
        if let Some(slot) = self.inverse_masses.get_mut(handle.index()) {
            *slot = if mass > 0.0 && mass.is_finite() {
                1.0 / mass
            } else {
                0.0
            };
        }
    }

    /// Pins `handle` (sets its inverse mass to `0`), making it an immovable
    /// anchor. Does nothing for an out-of-range handle.
    pub fn pin(&mut self, handle: ParticleHandle) {
        if let Some(slot) = self.inverse_masses.get_mut(handle.index()) {
            *slot = 0.0;
        }
    }

    /// Unpins `handle` by assigning it the given (finite, positive) `mass`.
    /// A non-positive or non-finite `mass` leaves the particle pinned. Does
    /// nothing for an out-of-range handle.
    pub fn unpin(&mut self, handle: ParticleHandle, mass: Real) {
        self.set_mass(handle, mass);
    }

    /// Returns the position column as a shared slice.
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Returns the position column as a mutable slice, for the solver to write
    /// projected positions in bulk.
    #[must_use]
    pub fn positions_mut(&mut self) -> &mut [Vec3] {
        &mut self.positions
    }

    /// Returns the previous-position column as a shared slice.
    #[must_use]
    pub fn prev_positions(&self) -> &[Vec3] {
        &self.prev_positions
    }

    /// Returns the previous-position column as a mutable slice, for the solver
    /// to snapshot substep-start positions.
    #[must_use]
    pub fn prev_positions_mut(&mut self) -> &mut [Vec3] {
        &mut self.prev_positions
    }

    /// Returns the velocity column as a shared slice.
    #[must_use]
    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }

    /// Returns the velocity column as a mutable slice, for the solver to write
    /// recovered velocities in bulk.
    #[must_use]
    pub fn velocities_mut(&mut self) -> &mut [Vec3] {
        &mut self.velocities
    }

    /// Returns the inverse-mass column as a shared slice.
    #[must_use]
    pub fn inverse_masses(&self) -> &[Real] {
        &self.inverse_masses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_store_is_empty() {
        let s = ParticleStorage::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn spawn_assigns_sequential_handles_and_inverse_mass() {
        let mut s = ParticleStorage::new();
        let a = s.spawn(Vec3::ZERO, 2.0);
        let b = s.spawn(Vec3::X, 4.0);
        assert_eq!(a.index(), 0);
        assert_eq!(b.index(), 1);
        assert_eq!(s.len(), 2);
        assert_eq!(s.inverse_mass(a), Some(0.5));
        assert_eq!(s.inverse_mass(b), Some(0.25));
    }

    #[test]
    fn spawn_seeds_prev_position_to_position_and_zero_velocity() {
        let mut s = ParticleStorage::new();
        let h = s.spawn(Vec3::new(1.0, 2.0, 3.0), 1.0);
        assert_eq!(s.position(h), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(s.prev_positions()[h.index()], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(s.velocity(h), Some(Vec3::ZERO));
    }

    #[test]
    fn pinned_particle_has_zero_inverse_mass() {
        let mut s = ParticleStorage::new();
        let h = s.spawn_pinned(Vec3::ZERO);
        assert!(s.is_pinned(h));
        assert_eq!(s.inverse_mass(h), Some(0.0));
    }

    #[test]
    fn non_positive_mass_pins_particle() {
        let mut s = ParticleStorage::new();
        let zero = s.spawn(Vec3::ZERO, 0.0);
        let neg = s.spawn(Vec3::ZERO, -1.0);
        let inf = s.spawn(Vec3::ZERO, Real::INFINITY);
        assert!(s.is_pinned(zero));
        assert!(s.is_pinned(neg));
        assert!(s.is_pinned(inf));
    }

    #[test]
    fn pin_and_unpin_toggle_inverse_mass() {
        let mut s = ParticleStorage::new();
        let h = s.spawn(Vec3::ZERO, 2.0);
        s.pin(h);
        assert!(s.is_pinned(h));
        s.unpin(h, 4.0);
        assert_eq!(s.inverse_mass(h), Some(0.25));
    }

    #[test]
    fn setters_update_columns() {
        let mut s = ParticleStorage::new();
        let h = s.spawn(Vec3::ZERO, 1.0);
        s.set_position(h, Vec3::Y);
        s.set_velocity(h, Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(s.position(h), Some(Vec3::Y));
        assert_eq!(s.velocity(h), Some(Vec3::new(0.0, -1.0, 0.0)));
    }

    #[test]
    fn contains_rejects_out_of_range() {
        let mut s = ParticleStorage::new();
        let h = s.spawn(Vec3::ZERO, 1.0);
        assert!(s.contains(h));
        assert!(!s.contains(ParticleHandle::from_index(5)));
        assert!(!s.contains(ParticleHandle::INVALID));
    }

    #[test]
    fn out_of_range_accessors_return_none() {
        let s = ParticleStorage::new();
        let bad = ParticleHandle::from_index(0);
        assert_eq!(s.position(bad), None);
        assert_eq!(s.velocity(bad), None);
        assert_eq!(s.inverse_mass(bad), None);
        assert!(!s.is_pinned(bad));
    }
}
