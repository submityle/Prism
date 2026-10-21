//! Deterministic external-force accumulation and rigid-body impulse algebra.
//!
//! The accumulator is intentionally independent of a solver.  Commands and
//! field sampling write into it at a simulation boundary; a backend consumes
//! it once for its chosen sub-step.  This keeps ordering explicit and makes the
//! same force stream usable by the free-body integrator and an XPBD adapter.

use crate::state::body::MassProperties;
use crate::state::handle::BodyHandle;
use crate::state::storage::BodyStorage;
use glam::{Mat3, Quat, Vec3};

const EPS: f32 = 1.0e-8;

/// A world-space force and torque accumulated for one body.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExternalForce {
    pub force: Vec3,
    pub torque: Vec3,
}

/// A world-space force/torque pair returned by actuators such as [`GrabDrive`](crate::dynamics::GrabDrive).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ForceTorque {
    pub force: Vec3,
    pub torque: Vec3,
}

/// Errors returned when an external-force request cannot be accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForceError {
    InvalidBody,
    NonFiniteInput,
}

/// A deterministic sparse force accumulator keyed by generational body handles.
///
/// Entries remain sorted by `(slot, generation)`. Lookup is logarithmic and
/// ascending field batches append without shifting existing entries. The
/// consumption order never depends on hash iteration or thread scheduling.
#[derive(Debug, Default, PartialEq)]
pub struct ForceAccumulator {
    entries: Vec<(BodyHandle, ExternalForce)>,
}

/// A validated sparse edit. It is committed under the same exclusive world
/// operation that prepared it, so unrelated queued loads never need copying.
#[derive(Default)]
pub(crate) struct PreparedForceRemap {
    removed: Vec<BodyHandle>,
    written: Vec<(BodyHandle, ExternalForce)>,
}

fn handle_key(body: &BodyHandle) -> (u32, u32) {
    (body.index(), body.generation())
}

fn sorted_unique_handles(
    handles: impl Iterator<Item = BodyHandle>,
) -> Result<Vec<BodyHandle>, ForceError> {
    let mut sorted: Vec<_> = handles.collect();
    sorted.sort_unstable_by_key(handle_key);
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(ForceError::NonFiniteInput);
    }
    Ok(sorted)
}

impl Clone for ForceAccumulator {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.entries.clone_from(&source.entries);
    }
}

impl ForceAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn pending(&self, body: BodyHandle) -> Option<ExternalForce> {
        self.entries
            .binary_search_by_key(&(body.index(), body.generation()), |(handle, _)| {
                (handle.index(), handle.generation())
            })
            .ok()
            .map(|index| self.entries[index].1)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Moves all pending loads from several bodies to one compound center.
    /// Force is summed and each source torque is transported with `r x F`,
    /// preserving the wrench about the new center.
    pub fn merge_bodies(
        &mut self,
        sources: &[(BodyHandle, Vec3)],
        target: BodyHandle,
        target_center: Vec3,
    ) -> Option<ExternalForce> {
        let (edit, combined) = self
            .prepare_merge_bodies(sources, target, target_center)
            .ok()?;
        self.commit_remap(edit);
        combined
    }

    /// Stages remapping without changing pending loads. Absence is a successful
    /// no-op, while overflow must reject the enclosing body transaction.
    pub(crate) fn prepare_merge_bodies(
        &self,
        sources: &[(BodyHandle, Vec3)],
        target: BodyHandle,
        target_center: Vec3,
    ) -> Result<(PreparedForceRemap, Option<ExternalForce>), ForceError> {
        if !target_center.is_finite()
            || sources.is_empty()
            || sources.iter().any(|(_, center)| !center.is_finite())
        {
            return Err(ForceError::NonFiniteInput);
        }
        let removed = sorted_unique_handles(sources.iter().map(|(body, _)| *body))?;
        let mut combined = ExternalForce::default();
        let mut found = false;
        for (body, center) in sources {
            if let Some(pending) = self.pending(*body) {
                found = true;
                combined.force += pending.force;
                combined.torque += pending.torque + (*center - target_center).cross(pending.force);
            }
        }
        if !found {
            return Ok((PreparedForceRemap::default(), None));
        }
        // A target may already have a load (including when it is a source).
        // Source loads are removed first, exactly as in the physical merge.
        let existing = if removed
            .binary_search_by_key(&handle_key(&target), handle_key)
            .is_ok()
        {
            ExternalForce::default()
        } else {
            self.pending(target).unwrap_or_default()
        };
        let value = checked_sum(existing, combined)?;
        Ok((
            PreparedForceRemap {
                removed,
                written: vec![(target, value)],
            },
            Some(combined),
        ))
    }

    /// Distributes a pending compound wrench to restored bodies. The force is
    /// split by mass and the torque correction keeps the total wrench about
    /// the compound center exactly unchanged.
    pub fn split_body(
        &mut self,
        source: BodyHandle,
        target_center: Vec3,
        targets: &[(BodyHandle, f32, Vec3)],
    ) -> bool {
        let Ok((edit, found)) = self.prepare_split_body(source, target_center, targets) else {
            return false;
        };
        self.commit_remap(edit);
        found
    }

    pub(crate) fn prepare_split_body(
        &self,
        source: BodyHandle,
        target_center: Vec3,
        targets: &[(BodyHandle, f32, Vec3)],
    ) -> Result<(PreparedForceRemap, bool), ForceError> {
        if !target_center.is_finite() || targets.is_empty() {
            return Err(ForceError::NonFiniteInput);
        }
        let Some(pending) = self.pending(source) else {
            return Ok((PreparedForceRemap::default(), false));
        };
        let total_mass: f32 = targets.iter().map(|(_, mass, _)| *mass).sum();
        if !total_mass.is_finite()
            || total_mass <= EPS
            || targets
                .iter()
                .any(|(_, mass, center)| !mass.is_finite() || *mass <= 0.0 || !center.is_finite())
        {
            return Err(ForceError::NonFiniteInput);
        }
        let _ = sorted_unique_handles(targets.iter().map(|(body, _, _)| *body))?;
        let mut written = Vec::with_capacity(targets.len());
        for (body, mass, center) in targets {
            let ratio = *mass / total_mass;
            let force = pending.force * ratio;
            let torque = pending.torque * ratio - (*center - target_center).cross(force);
            let existing = if *body == source {
                ExternalForce::default()
            } else {
                self.pending(*body).unwrap_or_default()
            };
            written.push((
                *body,
                checked_sum(existing, ExternalForce { force, torque })?,
            ));
        }
        Ok((
            PreparedForceRemap {
                removed: vec![source],
                written,
            },
            true,
        ))
    }

    /// No recoverable errors remain after preparation. Reuse the accumulator's
    /// allocation, and leave unrelated values and generational ordering intact.
    pub(crate) fn commit_remap(&mut self, mut edit: PreparedForceRemap) {
        if edit.removed.is_empty() && edit.written.is_empty() {
            return;
        }
        let mut removed_index = 0;
        self.entries.retain(|(body, _)| {
            while removed_index < edit.removed.len()
                && handle_key(&edit.removed[removed_index]) < handle_key(body)
            {
                removed_index += 1;
            }
            edit.removed.get(removed_index) != Some(body)
        });
        edit.written
            .sort_unstable_by_key(|(body, _)| handle_key(body));
        let additional = edit
            .written
            .iter()
            .filter(|(body, _)| self.pending(*body).is_none())
            .count();
        let mut remaining = self.entries.len();
        self.entries.resize(
            remaining + additional,
            (BodyHandle::INVALID, ExternalForce::default()),
        );
        let mut destination = self.entries.len();
        // Merge backwards in place. Repeated Vec::insert would shift every
        // unrelated load once per restored member (quadratic split traffic).
        while let Some(&(body, value)) = edit.written.last() {
            destination -= 1;
            if remaining > 0 && handle_key(&self.entries[remaining - 1].0) > handle_key(&body) {
                remaining -= 1;
                self.entries[destination] = self.entries[remaining];
            } else {
                if remaining > 0 && self.entries[remaining - 1].0 == body {
                    remaining -= 1;
                }
                self.entries[destination] = (body, value);
                edit.written.pop();
            }
        }
    }

    pub fn accumulate_force(&mut self, body: BodyHandle, force: Vec3) -> Result<(), ForceError> {
        self.accumulate(
            body,
            ForceTorque {
                force,
                torque: Vec3::ZERO,
            },
        )
    }

    pub fn accumulate_torque(&mut self, body: BodyHandle, torque: Vec3) -> Result<(), ForceError> {
        self.accumulate(
            body,
            ForceTorque {
                force: Vec3::ZERO,
                torque,
            },
        )
    }

    pub fn accumulate_at_point(
        &mut self,
        body: BodyHandle,
        force: Vec3,
        point: Vec3,
        center_of_mass: Vec3,
    ) -> Result<(), ForceError> {
        if !finite(force) || !finite(point) || !finite(center_of_mass) {
            return Err(ForceError::NonFiniteInput);
        }
        self.accumulate(
            body,
            ForceTorque {
                force,
                torque: (point - center_of_mass).cross(force),
            },
        )
    }

    pub(crate) fn accumulate(
        &mut self,
        body: BodyHandle,
        value: ForceTorque,
    ) -> Result<(), ForceError> {
        if !finite(value.force) || !finite(value.torque) {
            return Err(ForceError::NonFiniteInput);
        }
        let position = self
            .entries
            .binary_search_by_key(&(body.index(), body.generation()), |(handle, _)| {
                (handle.index(), handle.generation())
            });
        if let Ok(index) = position {
            let pending = &mut self.entries[index].1;
            let force = pending.force + value.force;
            let torque = pending.torque + value.torque;
            if !force.is_finite() || !torque.is_finite() {
                return Err(ForceError::NonFiniteInput);
            }
            pending.force = force;
            pending.torque = torque;
        } else {
            self.entries.insert(
                position.expect_err("new body"),
                (
                    body,
                    ExternalForce {
                        force: value.force,
                        torque: value.torque,
                    },
                ),
            );
        }
        Ok(())
    }

    /// Applies an impulse at a world-space point immediately and wakes the
    /// body.  The pending force accumulator is not consumed by this method.
    pub fn apply_impulse(
        bodies: &mut BodyStorage,
        body: BodyHandle,
        impulse: Vec3,
        point: Vec3,
    ) -> Result<(), ForceError> {
        if !finite(impulse) || !finite(point) {
            return Err(ForceError::NonFiniteInput);
        }
        let Some(position) = bodies.position(body) else {
            return Err(ForceError::InvalidBody);
        };
        let Some(mass) = bodies.mass_properties(body) else {
            return Err(ForceError::InvalidBody);
        };
        let linear = bodies
            .linear_velocity(body)
            .ok_or(ForceError::InvalidBody)?
            + impulse * mass.inv_mass.max(0.0);
        let angular_impulse = (point - position).cross(impulse);
        let angular = bodies
            .angular_velocity(body)
            .ok_or(ForceError::InvalidBody)?
            + inverse_inertia_world(bodies.orientation(body).unwrap_or(Quat::IDENTITY), mass)
                * angular_impulse;
        bodies.set_linear_velocity(body, linear);
        bodies.set_angular_velocity(body, angular);
        bodies.wake(body);
        Ok(())
    }

    /// Applies a pure angular impulse immediately.
    pub fn apply_torque_impulse(
        bodies: &mut BodyStorage,
        body: BodyHandle,
        impulse: Vec3,
    ) -> Result<(), ForceError> {
        if !finite(impulse) {
            return Err(ForceError::NonFiniteInput);
        }
        let Some(mass) = bodies.mass_properties(body) else {
            return Err(ForceError::InvalidBody);
        };
        let orientation = bodies.orientation(body).unwrap_or(Quat::IDENTITY);
        let velocity = bodies
            .angular_velocity(body)
            .ok_or(ForceError::InvalidBody)?
            + inverse_inertia_world(orientation, mass) * impulse;
        bodies.set_angular_velocity(body, velocity);
        bodies.wake(body);
        Ok(())
    }

    /// Consumes all pending forces for one sub-step.  Returns the number of
    /// valid live bodies that received an acceleration update.
    pub fn consume(&mut self, bodies: &mut BodyStorage, dt: f32) -> usize {
        if !dt.is_finite() || dt <= 0.0 {
            return 0;
        }
        let mut applied = 0;
        for (body, external) in self.entries.drain(..) {
            let Some(mass) = bodies.mass_properties(body) else {
                continue;
            };
            let Some(linear) = bodies.linear_velocity(body) else {
                continue;
            };
            let Some(angular) = bodies.angular_velocity(body) else {
                continue;
            };
            let orientation = bodies.orientation(body).unwrap_or(Quat::IDENTITY);
            let next_linear = linear + external.force * mass.inv_mass.max(0.0) * dt;
            let next_angular =
                angular + inverse_inertia_world(orientation, mass) * (external.torque * dt);
            bodies.set_linear_velocity(body, next_linear);
            bodies.set_angular_velocity(body, next_angular);
            bodies.wake(body);
            applied += 1;
        }
        applied
    }
}

fn checked_sum(a: ExternalForce, b: ExternalForce) -> Result<ExternalForce, ForceError> {
    let value = ExternalForce {
        force: a.force + b.force,
        torque: a.torque + b.torque,
    };
    if value.force.is_finite() && value.torque.is_finite() {
        Ok(value)
    } else {
        Err(ForceError::NonFiniteInput)
    }
}

/// Adapter for a backend that owns a different body store.
pub trait ExternalForceTarget {
    fn consume_external_forces(&mut self, accumulator: &mut ForceAccumulator, dt: f32) -> usize;
}

impl ExternalForceTarget for BodyStorage {
    fn consume_external_forces(&mut self, accumulator: &mut ForceAccumulator, dt: f32) -> usize {
        accumulator.consume(self, dt)
    }
}

#[must_use]
pub fn inverse_inertia_world(orientation: Quat, mass: MassProperties) -> Mat3 {
    let rotation = Mat3::from_quat(safe_quat(orientation));
    rotation * Mat3::from_diagonal(mass.inv_inertia.max(Vec3::ZERO)) * rotation.transpose()
}

#[must_use]
pub fn inertia_world(orientation: Quat, mass: MassProperties) -> Mat3 {
    let inv = mass.inv_inertia.max(Vec3::ZERO);
    let diagonal = Vec3::new(
        if inv.x > EPS { 1.0 / inv.x } else { 0.0 },
        if inv.y > EPS { 1.0 / inv.y } else { 0.0 },
        if inv.z > EPS { 1.0 / inv.z } else { 0.0 },
    );
    let rotation = Mat3::from_quat(safe_quat(orientation));
    rotation * Mat3::from_diagonal(diagonal) * rotation.transpose()
}

#[must_use]
fn finite(value: Vec3) -> bool {
    value.is_finite()
}

#[must_use]
fn safe_quat(value: Quat) -> Quat {
    let components = value.to_array();
    if !components.iter().all(|component| component.is_finite()) {
        return Quat::IDENTITY;
    }
    let length_squared = value.length_squared();
    if length_squared.is_finite() && length_squared > EPS * EPS {
        value.normalize()
    } else {
        Quat::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::body::{BodyDesc, MassProperties};

    #[test]
    fn sparse_remap_matches_individual_edits_with_aliases_and_generations() {
        let handles: Vec<_> = (0..32)
            .flat_map(|slot| [BodyHandle::new(slot, 2), BodyHandle::new(slot, 7)])
            .collect();
        let mut seed = ForceAccumulator::new();
        for (i, body) in handles.iter().enumerate() {
            seed.accumulate(
                *body,
                ForceTorque {
                    force: Vec3::X * i as f32,
                    torque: Vec3::Z,
                },
            )
            .unwrap();
        }
        // Existing target, target overlapping a source, and a new generation.
        for target in [
            handles[4],
            handles[6],
            BodyHandle::new(0, 3),
            BodyHandle::new(40, 1),
        ] {
            let sources = [
                (handles[60], Vec3::X),
                (handles[6], Vec3::Y),
                (handles[1], -Vec3::Z),
            ];
            let mut expected = seed.clone();
            let mut combined = ExternalForce::default();
            for (body, center) in sources {
                let load = seed.pending(body).unwrap();
                combined.force += load.force;
                combined.torque += load.torque + center.cross(load.force);
            }
            expected
                .entries
                .retain(|(body, _)| !sources.iter().any(|(source, _)| source == body));
            expected
                .accumulate(
                    target,
                    ForceTorque {
                        force: combined.force,
                        torque: combined.torque,
                    },
                )
                .unwrap();
            let mut actual = seed.clone();
            let pointer = actual.entries.as_ptr();
            let (edit, reported) = actual
                .prepare_merge_bodies(&sources, target, Vec3::ZERO)
                .unwrap();
            assert_eq!(edit.removed.len(), 3);
            assert_eq!(edit.written.len(), 1);
            assert_eq!(reported, Some(combined));
            assert_eq!(actual, seed, "preparation must be read-only");
            actual.commit_remap(edit);
            assert_eq!(actual, expected);
            assert_eq!(actual.entries.as_ptr(), pointer);

            // Unsorted restored handles include existing loads, the source
            // itself, a new generation between live entries, and an appended slot.
            let existing_target = if target == handles[4] {
                handles[8]
            } else {
                handles[4]
            };
            let targets = [
                (BodyHandle::new(45, 0), 1.0, Vec3::X),
                (existing_target, 2.0, -Vec3::Y),
                (target, 3.0, Vec3::Z),
                (BodyHandle::new(0, 5), 2.0, -Vec3::X),
            ];
            let pending = actual.pending(target).unwrap();
            let mut expected = actual.clone();
            expected.entries.retain(|(body, _)| *body != target);
            for (body, mass, center) in targets {
                let force = pending.force * (mass / 8.0);
                let torque = pending.torque * (mass / 8.0) - center.cross(force);
                expected
                    .accumulate(body, ForceTorque { force, torque })
                    .unwrap();
            }
            assert!(actual.split_body(target, Vec3::ZERO, &targets));
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn sparse_remap_target_overflow_or_duplicate_rejects_without_writes() {
        let a = BodyHandle::new(1, 2);
        let b = BodyHandle::new(1, 3);
        let mut loads = ForceAccumulator::new();
        loads
            .accumulate_force(a, Vec3::X * (f32::MAX * 0.75))
            .unwrap();
        loads
            .accumulate_force(b, Vec3::X * (f32::MAX * 0.75))
            .unwrap();
        let before = loads.clone();
        assert!(loads
            .merge_bodies(&[(a, Vec3::ZERO)], b, Vec3::ZERO)
            .is_none());
        assert_eq!(loads, before);
        assert!(!loads.split_body(a, Vec3::ZERO, &[(b, 1.0, Vec3::ZERO)]));
        assert_eq!(loads, before);
        assert!(loads
            .merge_bodies(&[(a, Vec3::ZERO), (a, Vec3::ZERO)], b, Vec3::ZERO)
            .is_none());
        assert_eq!(loads, before);
        assert!(!loads.split_body(a, Vec3::ZERO, &[(b, 1.0, Vec3::ZERO), (b, 1.0, Vec3::ZERO)]));
        assert_eq!(loads, before);
    }

    #[test]
    fn checkpoint_clone_from_retains_capacity_and_generational_order() {
        let mut source = ForceAccumulator::new();
        for slot in (0..32).rev() {
            source
                .accumulate_force(BodyHandle::new(slot, slot % 3), Vec3::X * slot as f32)
                .unwrap();
        }
        let mut checkpoint = ForceAccumulator::new();
        checkpoint.clone_from(&source);
        let pointer = checkpoint.entries.as_ptr();
        let capacity = checkpoint.entries.capacity();
        for count in [16, 0, 8, 32, 1, 0] {
            source.clear();
            for slot in (0..count).rev() {
                source
                    .accumulate_force(BodyHandle::new(slot, 7), Vec3::Y * slot as f32)
                    .unwrap();
            }
            checkpoint.clone_from(&source);
            assert_eq!(checkpoint, source);
            assert_eq!(checkpoint.entries.as_ptr(), pointer);
            assert_eq!(checkpoint.entries.capacity(), capacity);
            assert!(checkpoint.entries.windows(2).all(|pair| {
                (pair[0].0.index(), pair[0].0.generation())
                    < (pair[1].0.index(), pair[1].0.generation())
            }));
            if count > 1 {
                assert_eq!(
                    checkpoint.pending(BodyHandle::new(1, 7)).unwrap().force,
                    Vec3::Y
                );
                assert_eq!(checkpoint.pending(BodyHandle::new(1, 0)), None);
            }
        }
    }

    #[test]
    fn off_center_force_accumulates_torque() {
        let mut a = ForceAccumulator::new();
        let body = BodyHandle::INVALID;
        a.accumulate_at_point(body, Vec3::Y, Vec3::X, Vec3::ZERO)
            .unwrap();
        assert_eq!(a.pending(body).unwrap().torque, Vec3::Z);
    }

    #[test]
    fn impulse_updates_linear_and_angular_velocity() {
        let mut bodies = BodyStorage::new();
        let body = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO).with_mass_properties(
            MassProperties {
                inv_mass: 0.5,
                inv_inertia: Vec3::ONE,
            },
        ));
        ForceAccumulator::apply_impulse(&mut bodies, body, Vec3::Y, Vec3::X).unwrap();
        assert_eq!(bodies.linear_velocity(body), Some(Vec3::new(0.0, 0.5, 0.0)));
        assert_eq!(bodies.angular_velocity(body), Some(Vec3::Z));
    }

    #[test]
    fn consumption_is_deterministic_and_clears_entries() {
        let mut bodies = BodyStorage::new();
        let a = bodies.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let b = bodies.insert(BodyDesc::dynamic_at(Vec3::X));
        let mut f = ForceAccumulator::new();
        f.accumulate_force(b, Vec3::Y).unwrap();
        f.accumulate_force(a, Vec3::X).unwrap();
        assert_eq!(f.consume(&mut bodies, 0.5), 2);
        assert!(f.is_empty());
        assert_eq!(bodies.linear_velocity(a), Some(Vec3::new(0.5, 0.0, 0.0)));
        assert_eq!(bodies.linear_velocity(b), Some(Vec3::new(0.0, 0.5, 0.0)));
    }
}
