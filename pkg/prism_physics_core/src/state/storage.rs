//! Structure-of-Arrays (`SoA`) rigid-body storage.
//!
//! [`BodyStorage`] stores each body attribute in its own parallel `Vec`, which
//! keeps hot integration data (positions, velocities) contiguous and
//! cache-friendly. Slots are recycled through a free list, and each slot keeps
//! a generation counter so that stale [`BodyHandle`]s are rejected after a slot
//! is reused.

use crate::collider::{ColliderHandle, PhysicsMaterial};
use crate::state::body::{BodyDesc, BodyKind, MassProperties, SolverMassProperties};
use crate::state::handle::BodyHandle;
use crate::state::view::BodySolverView;
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
    prev_positions: Vec<Vec3>,
    prev_orientations: Vec<Quat>,
    linear_velocities: Vec<Vec3>,
    angular_velocities: Vec<Vec3>,
    gravity_overrides: Vec<Option<Vec3>>,
    mass_props: Vec<MassProperties>,
    /// Per-substep effective properties consumed by XPBD. The authored
    /// properties above remain immutable simulation input; this cache is
    /// refreshed by the world adapter for effects such as added mass.
    solver_mass_props: Vec<SolverMassProperties>,
    kinds: Vec<BodyKind>,
    colliders: Vec<Option<ColliderHandle>>,
    materials: Vec<PhysicsMaterial>,
    is_sensor: Vec<bool>,
    ccd: Vec<bool>,
    linear_damping: Vec<f32>,
    angular_damping: Vec<f32>,
    sleeping: Vec<bool>,
    sleep_timer: Vec<f32>,
    generations: Vec<u32>,
    active: Vec<bool>,
    free_list: Vec<u32>,
    live: usize,
}

/// Allocation order for a replacement prepared under an exclusive world borrow.
/// Only the removed members and future handles are copied, not the storage or
/// its free list. The caller must not mutate storage between prepare and commit.
pub(crate) struct BodyReplacementPlan {
    removed: Vec<BodyHandle>,
    inserted: Vec<BodyHandle>,
}

impl BodyReplacementPlan {
    pub(crate) fn handles(&self) -> &[BodyHandle] {
        &self.inserted
    }
}

impl BodyStorage {
    pub(crate) fn prepare_replacement(
        &self,
        removed: &[BodyHandle],
        insert_count: usize,
    ) -> Option<BodyReplacementPlan> {
        if removed
            .iter()
            .enumerate()
            .any(|(i, body)| !self.contains(*body) || removed[..i].contains(body))
        {
            return None;
        }
        let mut inserted = Vec::with_capacity(insert_count);
        let mut old_free = self.free_list.iter().rev();
        let mut newly_free = removed.iter().rev();
        let mut next_slot = self.positions.len();
        for _ in 0..insert_count {
            let handle = if let Some(body) = newly_free.next() {
                BodyHandle::new(body.index(), body.generation().wrapping_add(1))
            } else if let Some(&slot) = old_free.next() {
                BodyHandle::new(slot, self.generations[slot as usize])
            } else {
                let slot = u32::try_from(next_slot).ok()?;
                next_slot = next_slot.checked_add(1)?;
                BodyHandle::new(slot, 0)
            };
            inserted.push(handle);
        }
        Some(BodyReplacementPlan {
            removed: removed.to_vec(),
            inserted,
        })
    }

    pub(crate) fn commit_replacement(
        &mut self,
        plan: BodyReplacementPlan,
        descriptors: &[BodyDesc],
    ) -> Vec<BodyHandle> {
        assert_eq!(plan.inserted.len(), descriptors.len());
        for body in plan.removed {
            assert!(
                self.remove(body),
                "replacement plan must be committed unchanged"
            );
        }
        for (&expected, &desc) in plan.inserted.iter().zip(descriptors) {
            assert_eq!(
                self.insert(desc),
                expected,
                "replacement allocation changed"
            );
        }
        plan.inserted
    }

    /// Immutable descriptor capture for transaction preparation.
    pub(crate) fn descriptor(&self, handle: BodyHandle) -> Option<BodyDesc> {
        if !self.contains(handle) {
            return None;
        }
        let i = handle.index() as usize;
        Some(BodyDesc {
            kind: self.kinds[i],
            position: self.positions[i],
            orientation: self.orientations[i],
            linear_velocity: self.linear_velocities[i],
            angular_velocity: self.angular_velocities[i],
            mass_properties: self.mass_props[i],
            linear_damping: self.linear_damping[i],
            angular_damping: self.angular_damping[i],
            collider: self.colliders[i],
            material: self.materials[i],
            is_sensor: self.is_sensor[i],
            ccd: self.ccd[i],
        })
    }

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
            prev_positions: Vec::with_capacity(capacity),
            prev_orientations: Vec::with_capacity(capacity),
            linear_velocities: Vec::with_capacity(capacity),
            angular_velocities: Vec::with_capacity(capacity),
            gravity_overrides: Vec::with_capacity(capacity),
            mass_props: Vec::with_capacity(capacity),
            solver_mass_props: Vec::with_capacity(capacity),
            kinds: Vec::with_capacity(capacity),
            colliders: Vec::with_capacity(capacity),
            materials: Vec::with_capacity(capacity),
            is_sensor: Vec::with_capacity(capacity),
            ccd: Vec::with_capacity(capacity),
            linear_damping: Vec::with_capacity(capacity),
            angular_damping: Vec::with_capacity(capacity),
            sleeping: Vec::with_capacity(capacity),
            sleep_timer: Vec::with_capacity(capacity),
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
            self.prev_positions[i] = desc.position;
            self.prev_orientations[i] = desc.orientation;
            self.linear_velocities[i] = desc.linear_velocity;
            self.angular_velocities[i] = desc.angular_velocity;
            self.gravity_overrides[i] = None;
            self.mass_props[i] = desc.mass_properties;
            self.solver_mass_props[i] = desc.mass_properties.into();
            self.kinds[i] = desc.kind;
            self.colliders[i] = desc.collider;
            self.materials[i] = desc.material;
            self.is_sensor[i] = desc.is_sensor;
            self.ccd[i] = desc.ccd;
            self.linear_damping[i] = desc.linear_damping;
            self.angular_damping[i] = desc.angular_damping;
            self.sleeping[i] = false;
            self.sleep_timer[i] = 0.0;
            self.active[i] = true;
            BodyHandle::new(index, self.generations[i])
        } else {
            let index = self.positions.len() as u32;
            self.positions.push(desc.position);
            self.orientations.push(desc.orientation);
            self.prev_positions.push(desc.position);
            self.prev_orientations.push(desc.orientation);
            self.linear_velocities.push(desc.linear_velocity);
            self.angular_velocities.push(desc.angular_velocity);
            self.gravity_overrides.push(None);
            self.mass_props.push(desc.mass_properties);
            self.solver_mass_props.push(desc.mass_properties.into());
            self.kinds.push(desc.kind);
            self.colliders.push(desc.collider);
            self.materials.push(desc.material);
            self.is_sensor.push(desc.is_sensor);
            self.ccd.push(desc.ccd);
            self.linear_damping.push(desc.linear_damping);
            self.angular_damping.push(desc.angular_damping);
            self.sleeping.push(false);
            self.sleep_timer.push(0.0);
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
        self.gravity_overrides[i] = None;
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

    /// Returns the body's world-space gravity policy. `Some(None)` follows
    /// world gravity; `Some(Some(g))` uses the finite acceleration `g` instead.
    /// An invalid or stale handle returns the outer `None`.
    #[must_use]
    pub fn gravity_override(&self, handle: BodyHandle) -> Option<Option<Vec3>> {
        self.contains(handle)
            .then(|| self.gravity_overrides[handle.index() as usize])
    }

    /// Sets a persistent world-space gravity acceleration, or clears it with
    /// `None` to follow world gravity. Invalid handles and non-finite vectors
    /// are rejected without changing any state. A changed policy wakes the
    /// body and resets its sleep timer; an unchanged policy is a no-op.
    pub fn set_gravity_override(&mut self, handle: BodyHandle, value: Option<Vec3>) -> bool {
        if !self.contains(handle) || value.is_some_and(|gravity| !gravity.is_finite()) {
            return false;
        }
        let i = handle.index() as usize;
        if self.gravity_overrides[i] != value {
            self.gravity_overrides[i] = value;
            self.sleeping[i] = false;
            self.sleep_timer[i] = 0.0;
        }
        true
    }

    /// Returns the mass properties, or `None` if the handle is invalid.
    #[must_use]
    pub fn mass_properties(&self, handle: BodyHandle) -> Option<MassProperties> {
        self.contains(handle)
            .then(|| self.mass_props[handle.index() as usize])
    }

    /// Overrides the solver-only effective mass properties for one live body.
    /// Authored mass properties are left unchanged.
    pub fn set_solver_mass_properties(
        &mut self,
        handle: BodyHandle,
        properties: SolverMassProperties,
    ) -> bool {
        if self.contains(handle) {
            self.solver_mass_props[handle.index() as usize] = properties;
            true
        } else {
            false
        }
    }

    /// Restores every solver slot to its authored mass properties.
    pub fn reset_solver_mass_properties(&mut self) {
        for (solver, authored) in self
            .solver_mass_props
            .iter_mut()
            .zip(self.mass_props.iter().copied().map(Into::into))
        {
            *solver = authored;
        }
    }

    /// Returns the body kind, or `None` if the handle is invalid.
    #[must_use]
    pub fn kind(&self, handle: BodyHandle) -> Option<BodyKind> {
        self.contains(handle)
            .then(|| self.kinds[handle.index() as usize])
    }

    /// Returns the previous-step position of the body, or `None` if the handle
    /// is invalid.
    ///
    /// The previous-step columns are maintained by position-based solvers (such
    /// as XPBD) that need the pre-integration pose to recover velocities.
    #[must_use]
    pub fn prev_position(&self, handle: BodyHandle) -> Option<Vec3> {
        self.contains(handle)
            .then(|| self.prev_positions[handle.index() as usize])
    }

    /// Sets the previous-step position of the body. Returns `true` on success.
    pub fn set_prev_position(&mut self, handle: BodyHandle, value: Vec3) -> bool {
        if self.contains(handle) {
            self.prev_positions[handle.index() as usize] = value;
            true
        } else {
            false
        }
    }

    /// Returns the previous-step orientation of the body, or `None` if the
    /// handle is invalid.
    #[must_use]
    pub fn prev_orientation(&self, handle: BodyHandle) -> Option<Quat> {
        self.contains(handle)
            .then(|| self.prev_orientations[handle.index() as usize])
    }

    /// Sets the previous-step orientation of the body. Returns `true` on
    /// success.
    pub fn set_prev_orientation(&mut self, handle: BodyHandle, value: Quat) -> bool {
        if self.contains(handle) {
            self.prev_orientations[handle.index() as usize] = value;
            true
        } else {
            false
        }
    }

    /// Returns the collider handle attached to the body, or `None` if the body
    /// has no collider or the handle is invalid.
    #[must_use]
    pub fn collider(&self, handle: BodyHandle) -> Option<ColliderHandle> {
        if self.contains(handle) {
            self.colliders[handle.index() as usize]
        } else {
            None
        }
    }

    /// Sets the collider handle attached to the body. Returns `true` on success.
    pub fn set_collider(&mut self, handle: BodyHandle, collider: Option<ColliderHandle>) -> bool {
        if self.contains(handle) {
            self.colliders[handle.index() as usize] = collider;
            true
        } else {
            false
        }
    }

    /// Returns the contact material of the body, or `None` if the handle is
    /// invalid.
    #[must_use]
    pub fn material(&self, handle: BodyHandle) -> Option<PhysicsMaterial> {
        self.contains(handle)
            .then(|| self.materials[handle.index() as usize])
    }

    /// Sets the contact material of the body. Returns `true` on success.
    pub fn set_material(&mut self, handle: BodyHandle, material: PhysicsMaterial) -> bool {
        if self.contains(handle) {
            self.materials[handle.index() as usize] = material;
            true
        } else {
            false
        }
    }

    /// Returns a Structure-of-Arrays mutable view over the columns a
    /// position-based solver consumes.
    ///
    /// The view borrows the hot columns (positions, orientations, their
    /// previous-step copies, and velocities) mutably and the descriptive
    /// columns (mass properties, kinds, colliders, materials, occupancy)
    /// immutably. Solvers index every slice by the same slot index; use
    /// [`BodySolverView::is_active`] to skip freed slots.
    #[must_use]
    pub fn solver_view_mut(&mut self) -> BodySolverView<'_> {
        BodySolverView {
            positions: &mut self.positions,
            orientations: &mut self.orientations,
            prev_positions: &mut self.prev_positions,
            prev_orientations: &mut self.prev_orientations,
            linear_velocities: &mut self.linear_velocities,
            angular_velocities: &mut self.angular_velocities,
            gravity_overrides: &self.gravity_overrides,
            mass_props: &self.solver_mass_props,
            kinds: &self.kinds,
            colliders: &self.colliders,
            materials: &self.materials,
            is_sensor: &self.is_sensor,
            linear_damping: &self.linear_damping,
            angular_damping: &self.angular_damping,
            sleeping: &mut self.sleeping,
            sleep_timers: &mut self.sleep_timer,
            active: &self.active,
        }
    }

    /// Returns whether the body is a sensor (trigger volume), or `None` if the
    /// handle is invalid.
    ///
    /// A sensor participates in overlap detection and emits trigger events but
    /// is skipped by the contact solver, so it never pushes other bodies.
    #[must_use]
    pub fn is_sensor(&self, handle: BodyHandle) -> Option<bool> {
        self.contains(handle)
            .then(|| self.is_sensor[handle.index() as usize])
    }

    /// Sets whether the body is a sensor (trigger volume). Returns `true` on
    /// success.
    pub fn set_sensor(&mut self, handle: BodyHandle, is_sensor: bool) -> bool {
        if self.contains(handle) {
            self.is_sensor[handle.index() as usize] = is_sensor;
            true
        } else {
            false
        }
    }

    /// Returns whether continuous collision detection is enabled for the body,
    /// or `None` if the handle is invalid.
    ///
    /// CCD bodies are swept against the world each sub-step (see
    /// [`crate::ccd`]) so that fast motion cannot tunnel through thin geometry.
    #[must_use]
    pub fn ccd(&self, handle: BodyHandle) -> Option<bool> {
        self.contains(handle)
            .then(|| self.ccd[handle.index() as usize])
    }

    /// Enables or disables continuous collision detection for the body. Returns
    /// `true` on success.
    pub fn set_ccd(&mut self, handle: BodyHandle, ccd: bool) -> bool {
        if self.contains(handle) {
            self.ccd[handle.index() as usize] = ccd;
            true
        } else {
            false
        }
    }

    /// Returns whether the body is currently sleeping, or `None` if the handle
    /// is invalid.
    ///
    /// A sleeping body is skipped by the solver's prediction and constraint
    /// phases until it is woken. Newly inserted bodies always start awake.
    #[must_use]
    pub fn is_sleeping(&self, handle: BodyHandle) -> Option<bool> {
        self.contains(handle)
            .then(|| self.sleeping[handle.index() as usize])
    }

    /// Returns the accumulated idle time (seconds) for the body, or `None` if
    /// the handle is invalid.
    ///
    /// The timer grows while the body stays below the sleep velocity thresholds
    /// and resets to zero as soon as it moves faster than a threshold.
    #[must_use]
    pub fn sleep_timer(&self, handle: BodyHandle) -> Option<f32> {
        self.contains(handle)
            .then(|| self.sleep_timer[handle.index() as usize])
    }

    /// Wakes the body: clears its sleeping flag and resets its idle timer.
    ///
    /// Returns `true` on success, or `false` if the handle is stale or invalid.
    /// This is the hook game code and command application call after teleporting
    /// a body, changing its velocity, or applying an impulse.
    pub fn wake(&mut self, handle: BodyHandle) -> bool {
        if self.contains(handle) {
            let i = handle.index() as usize;
            self.sleeping[i] = false;
            self.sleep_timer[i] = 0.0;
            true
        } else {
            false
        }
    }

    /// Wakes every live body, clearing all sleeping flags and idle timers.
    ///
    /// Useful after a global change (such as altering gravity) that could
    /// invalidate the resting assumption for all sleeping bodies.
    pub fn wake_all(&mut self) {
        for i in 0..self.active.len() {
            if self.active[i] {
                self.sleeping[i] = false;
                self.sleep_timer[i] = 0.0;
            }
        }
    }

    /// Returns the live [`BodyHandle`] occupying `slot`, or `None` when the slot
    /// is empty or out of range.
    ///
    /// The narrow-phase pipeline addresses bodies by slot index (the same index
    /// used by [`BodySolverView`]) and uses this to stamp contact manifolds with
    /// stable handles.
    #[must_use]
    pub fn handle_at_slot(&self, slot: usize) -> Option<BodyHandle> {
        if slot < self.active.len() && self.active[slot] {
            Some(BodyHandle::new(slot as u32, self.generations[slot]))
        } else {
            None
        }
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
            && self.prev_positions.len() == n
            && self.prev_orientations.len() == n
            && self.linear_velocities.len() == n
            && self.angular_velocities.len() == n
            && self.gravity_overrides.len() == n
            && self.mass_props.len() == n
            && self.solver_mass_props.len() == n
            && self.kinds.len() == n
            && self.colliders.len() == n
            && self.materials.len() == n
            && self.is_sensor.len() == n
            && self.ccd.len() == n
            && self.linear_damping.len() == n
            && self.angular_damping.len() == n
            && self.sleeping.len() == n
            && self.sleep_timer.len() == n
            && self.generations.len() == n
            && self.active.len() == n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_plan_preserves_allocator_order_and_wrapping_generation() {
        let mut storage = BodyStorage::new();
        let a = storage.insert(BodyDesc::default());
        let b = storage.insert(BodyDesc::default());
        let spare = storage.insert(BodyDesc::default());
        assert!(storage.remove(spare));
        storage.generations[b.index() as usize] = u32::MAX;
        let b = BodyHandle::new(b.index(), u32::MAX);
        let before = format!("{storage:?}");
        assert!(storage.prepare_replacement(&[a, a], 2).is_none());
        assert!(storage
            .prepare_replacement(&[BodyHandle::INVALID], 2)
            .is_none());
        let plan = storage.prepare_replacement(&[a, b], 4).unwrap();
        assert_eq!(format!("{storage:?}"), before);
        assert_eq!(plan.handles()[0], BodyHandle::new(b.index(), 0));
        let mut legacy = storage.clone();
        assert!(legacy.remove(a));
        assert!(legacy.remove(b));
        let descriptors = [BodyDesc::default(); 4];
        let expected: Vec<_> = descriptors
            .iter()
            .map(|&desc| legacy.insert(desc))
            .collect();
        assert_eq!(storage.commit_replacement(plan, &descriptors), expected);
        assert_eq!(format!("{storage:?}"), format!("{legacy:?}"));
        assert_eq!(
            storage.insert(BodyDesc::default()),
            legacy.insert(BodyDesc::default())
        );
        assert!(storage.columns_consistent());
    }

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
