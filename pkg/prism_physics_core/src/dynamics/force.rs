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
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ForceAccumulator {
    entries: Vec<(BodyHandle, ExternalForce)>,
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
        if !target_center.is_finite()
            || sources.is_empty()
            || sources.iter().enumerate().any(|(index, (body, center))| {
                !center.is_finite() || sources[..index].iter().any(|(other, _)| other == body)
            })
        {
            return None;
        }
        let mut combined = ExternalForce::default();
        let mut found = false;
        for (body, center) in sources {
            if !center.is_finite() {
                return None;
            }
            if let Some((_, pending)) = self.entries.iter().find(|(candidate, _)| candidate == body)
            {
                found = true;
                combined.force += pending.force;
                combined.torque += pending.torque + (*center - target_center).cross(pending.force);
            }
        }
        if !found {
            return None;
        }
        let mut candidate = self.clone();
        candidate
            .entries
            .retain(|(body, _)| !sources.iter().any(|(source, _)| source == body));
        candidate
            .accumulate(
                target,
                ForceTorque {
                    force: combined.force,
                    torque: combined.torque,
                },
            )
            .ok()?;
        *self = candidate;
        Some(combined)
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
        if !target_center.is_finite() || targets.is_empty() {
            return false;
        }
        let Some((_, pending)) = self
            .entries
            .iter()
            .find(|(body, _)| *body == source)
            .copied()
        else {
            return false;
        };
        let total_mass: f32 = targets.iter().map(|(_, mass, _)| *mass).sum();
        if !total_mass.is_finite()
            || total_mass <= EPS
            || targets
                .iter()
                .any(|(_, mass, center)| !mass.is_finite() || *mass <= 0.0 || !center.is_finite())
            || targets.iter().enumerate().any(|(index, (body, _, _))| {
                targets[..index].iter().any(|(other, _, _)| other == body)
            })
        {
            return false;
        }
        let mut candidate = self.clone();
        candidate.entries.retain(|(body, _)| *body != source);
        for (body, mass, center) in targets {
            let ratio = *mass / total_mass;
            let force = pending.force * ratio;
            let torque = pending.torque * ratio - (*center - target_center).cross(force);
            if candidate
                .accumulate(*body, ForceTorque { force, torque })
                .is_err()
            {
                return false;
            }
        }
        *self = candidate;
        true
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
