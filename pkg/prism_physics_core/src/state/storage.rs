//! Structure-of-Arrays (`SoA`) rigid-body storage.
//!
//! [`BodyStorage`] stores each body attribute in its own parallel `Vec`, which
//! keeps hot integration data (positions, velocities) contiguous and
//! cache-friendly. Slots are recycled through a free list, and each slot keeps
//! a generation counter so that stale [`BodyHandle`]s are rejected after a slot
//! is reused.

use crate::state::body::{BodyDesc, BodyKind, MassProperties};
use crate::state::handle::BodyHandle;
use glam::{Quat, Vec3};

/// Structure-of-Arrays storage for rigid bodies.
///
/// Every field vector is indexed by the same slot index. The `active` and
/// `generations` vectors track slot occupancy and handle validity; freed slot
/// indices are pushed onto `free_list` for reuse.
#[derive(Clone, Debug, Default)]
pub struct BodyStorage {
    positions: Vec<Vec3>,
    orientations: Vec<Quat>,
    linear_velocities: Vec<Vec3>,
    angular_velocities: Vec<Vec3>,
    mass_props: Vec<MassProperties>,
    kinds: Vec<BodyKind>,
    linear_damping: Vec<f32>,
    angular_damping: Vec<f32>,
    generations: Vec<u32>,
    active: Vec<bool>,
    free_list: Vec<u32>,
    live: usize,
}

impl BodyStorage {
    /// Creates an empty storage.
    #[must_use]
    pub fn new() -> BodyStorage {
        BodyStorage::default()
    }

    /// Creates an empty storage with capacity pre-reserved for `capacity`
    /// bodies.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> BodyStorage {
        BodyStorage {
            positions: Vec::with_capacity(capacity),
            orientations: Vec::with_capacity(capacity),
            linear_velocities: Vec::with_capacity(capacity),
            angular_velocities: Vec::with_capacity(capacity),
            mass_props: Vec::with_capacity(capacity),
            kinds: Vec::with_capacity(capacity),
            linear_damping: Vec::with_capacity(capacity),
            angular_damping: Vec::with_capacity(capacity),
            generations: Vec::with_capacity(capacity),
            active: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            live: 0,
        }
    }

    /// Inserts a body described by `desc`, returning a handle to it.
    ///
    /// A freed slot is reused when one is available; otherwise a new slot is
    /// appended.
    pub fn insert(&mut self, desc: BodyDesc) -> BodyHandle {
        self.live += 1;
        if let Some(index) = self.free_list.pop() {
            let i = index as usize;
            self.positions[i] = desc.position;
            self.orientations[i] = desc.orientation;
            self.linear_velocities[i] = desc.linear_velocity;
            self.angular_velocities[i] = desc.angular_velocity;
            self.mass_props[i] = desc.mass_properties;
            self.kinds[i] = desc.kind;
            self.linear_damping[i] = desc.linear_damping;
            self.angular_damping[i] = desc.angular_damping;
            self.active[i] = true;
            BodyHandle::new(index, self.generations[i])
        } else {
            let index = self.positions.len() as u32;
            self.positions.push(desc.position);
            self.orientations.push(desc.orientation);
            self.linear_velocities.push(desc.linear_velocity);
            self.angular_velocities.push(desc.angular_velocity);
            self.mass_props.push(desc.mass_properties);
            self.kinds.push(desc.kind);
            self.linear_damping.push(desc.linear_damping);
            self.angular_damping.push(desc.angular_damping);
            self.generations.push(0);
            self.active.push(true);
            BodyHandle::new(index, 0)
        }
    }

    /// Removes the body referenced by `handle`.
    ///
    /// Returns `true` if a live body was removed, or `false` if the handle was
    /// stale or invalid. The slot's generation is incremented so that any other
    /// copy of the removed handle stops validating.
    pub fn remove(&mut self, handle: BodyHandle) -> bool {
        if !self.contains(handle) {
            return false;
        }
        let i = handle.index() as usize;
        self.active[i] = false;
        self.generations[i] = self.generations[i].wrapping_add(1);
        self.free_list.push(handle.index());
        self.live -= 1;
        true
    }

    /// Returns `true` if `handle` refers to a live body in this storage.
    #[must_use]
    pub fn contains(&self, handle: BodyHandle) -> bool {
        let i = handle.index() as usize;
        i < self.active.len() && self.active[i] && self.generations[i] == handle.generation()
    }

    /// Returns the number of live bodies.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live
    }

    /// Returns `true` if there are no live bodies.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Returns the position of the body, or `None` if the handle is invalid.
    #[must_use]
    pub fn position(&self, handle: BodyHandle) -> Option<Vec3> {
        self.contains(handle)
            .then(|| self.positions[handle.index() as usize])
    }

    /// Sets the position of the body. Returns `true` on success.
    pub fn set_position(&mut self, handle: BodyHandle, value: Vec3) -> bool {
        if self.contains(handle) {
            self.positions[handle.index() as usize] = value;
            true
        } else {
            false
        }
    }

    /// Returns the orientation of the body, or `None` if the handle is invalid.
    #[must_use]
    pub fn orientation(&self, handle: BodyHandle) -> Option<Quat> {
        self.contains(handle)
            .then(|| self.orientations[handle.index() as usize])
    }

    /// Sets the orientation of the body. Returns `true` on success.
    pub fn set_orientation(&mut self, handle: BodyHandle, value: Quat) -> bool {
        if self.contains(handle) {
            self.orientations[handle.index() as usize] = value;
            true
        } else {
            false
        }
    }

    /// Returns the linear velocity, or `None` if the handle is invalid.
    #[must_use]
    pub fn linear_velocity(&self, handle: BodyHandle) -> Option<Vec3> {
        self.contains(handle)
            .then(|| self.linear_velocities[handle.index() as usize])
    }

    /// Sets the linear velocity. Returns `true` on success.
    pub fn set_linear_velocity(&mut self, handle: BodyHandle, value: Vec3) -> bool {
        if self.contains(handle) {
            self.linear_velocities[handle.index() as usize] = value;
            true
        } else {
            false
        }
    }

    /// Returns the angular velocity, or `None` if the handle is invalid.
    #[must_use]
    pub fn angular_velocity(&self, handle: BodyHandle) -> Option<Vec3> {
        self.contains(handle)
            .then(|| self.angular_velocities[handle.index() as usize])
    }

    /// Sets the angular velocity. Returns `true` on success.
    pub fn set_angular_velocity(&mut self, handle: BodyHandle, value: Vec3) -> bool {
        if self.contains(handle) {
            self.angular_velocities[handle.index() as usize] = value;
            true
        } else {
            false
        }
    }

    /// Returns the mass properties, or `None` if the handle is invalid.
    #[must_use]
    pub fn mass_properties(&self, handle: BodyHandle) -> Option<MassProperties> {
        self.contains(handle)
            .then(|| self.mass_props[handle.index() as usize])
    }

    /// Returns the body kind, or `None` if the handle is invalid.
    #[must_use]
    pub fn kind(&self, handle: BodyHandle) -> Option<BodyKind> {
        self.contains(handle)
            .then(|| self.kinds[handle.index() as usize])
    }

    /// Invokes `f` for every live dynamic body, giving mutable access to the
    /// `SoA` columns the integrator needs.
    ///
    /// The closure receives, in order: position, orientation, linear velocity,
    /// angular velocity, the (immutable) mass properties, the linear damping
    /// coefficient, and the angular damping coefficient. Static and kinematic
    /// bodies are skipped.
    pub fn for_each_dynamic_mut(
        &mut self,
        mut f: impl FnMut(&mut Vec3, &mut Quat, &mut Vec3, &mut Vec3, &MassProperties, f32, f32),
    ) {
        for i in 0..self.kinds.len() {
            if !self.active[i] || self.kinds[i] != BodyKind::Dynamic {
                continue;
            }
            f(
                &mut self.positions[i],
                &mut self.orientations[i],
                &mut self.linear_velocities[i],
                &mut self.angular_velocities[i],
                &self.mass_props[i],
                self.linear_damping[i],
                self.angular_damping[i],
            );
        }
    }

    /// Returns the number of allocated slots, including freed ones.
    ///
    /// This is exposed for tests and diagnostics that assert on the internal
    /// `SoA` length invariant; it is not the count of live bodies (use
    /// [`len`](BodyStorage::len) for that).
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.positions.len()
    }

    /// Debug assertion helper: verifies that every `SoA` column has the same
    /// length. Returns `true` when the invariant holds.
    #[must_use]
    pub fn columns_consistent(&self) -> bool {
        let n = self.positions.len();
        self.orientations.len() == n
            && self.linear_velocities.len() == n
            && self.angular_velocities.len() == n
            && self.mass_props.len() == n
            && self.kinds.len() == n
            && self.linear_damping.len() == n
            && self.angular_damping.len() == n
            && self.generations.len() == n
            && self.active.len() == n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_accessors() {
        let mut s = BodyStorage::new();
        assert!(s.is_empty());
        let h = s.insert(BodyDesc::dynamic_at(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(s.len(), 1);
        assert_eq!(s.position(h), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(s.kind(h), Some(BodyKind::Dynamic));
        assert!(s.set_position(h, Vec3::ZERO));
        assert_eq!(s.position(h), Some(Vec3::ZERO));
        assert!(s.columns_consistent());
    }

    #[test]
    fn remove_invalidates_handle_generationally() {
        let mut s = BodyStorage::new();
        let h = s.insert(BodyDesc::dynamic_at(Vec3::X));
        assert!(s.contains(h));
        assert!(s.remove(h));
        assert!(!s.contains(h));
        assert!(!s.remove(h));
        assert_eq!(s.position(h), None);
        assert!(s.is_empty());
    }

    #[test]
    fn slot_reuse_bumps_generation_and_rejects_stale() {
        let mut s = BodyStorage::new();
        let h0 = s.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        assert!(s.remove(h0));
        // Reuse should recycle the same slot index with a new generation.
        let h1 = s.insert(BodyDesc::dynamic_at(Vec3::ONE));
        assert_eq!(h0.index(), h1.index());
        assert_ne!(h0.generation(), h1.generation());
        assert!(!s.contains(h0));
        assert!(s.contains(h1));
        assert_eq!(s.slot_count(), 1);
    }

    #[test]
    fn churn_keeps_other_handles_valid() {
        let mut s = BodyStorage::new();
        let a = s.insert(BodyDesc::dynamic_at(Vec3::new(1.0, 0.0, 0.0)));
        let b = s.insert(BodyDesc::dynamic_at(Vec3::new(2.0, 0.0, 0.0)));
        let c = s.insert(BodyDesc::dynamic_at(Vec3::new(3.0, 0.0, 0.0)));
        assert!(s.remove(b));
        // a and c must remain valid and untouched after removing b.
        assert_eq!(s.position(a), Some(Vec3::new(1.0, 0.0, 0.0)));
        assert_eq!(s.position(c), Some(Vec3::new(3.0, 0.0, 0.0)));
        assert_eq!(s.len(), 2);
        // Insert reuses b's slot; a and c stay valid.
        let d = s.insert(BodyDesc::dynamic_at(Vec3::new(4.0, 0.0, 0.0)));
        assert_eq!(s.position(a), Some(Vec3::new(1.0, 0.0, 0.0)));
        assert_eq!(s.position(c), Some(Vec3::new(3.0, 0.0, 0.0)));
        assert_eq!(s.position(d), Some(Vec3::new(4.0, 0.0, 0.0)));
        assert!(s.columns_consistent());
    }
}
