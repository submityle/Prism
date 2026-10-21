//! Joints: constraints that bind pairs of bodies together.
//!
//! This module provides the joint data model and storage. A [`Joint`] pairs two
//! [`JointAnchor`]s with a [`JointKind`]; a [`JointStorage`] owns joints behind
//! generational [`JointHandle`]s (the same recycling scheme as
//! [`BodyStorage`](crate::state::storage::BodyStorage)). The actual constraint
//! projection lives in the solver
//! ([`solver::xpbd::joint_constraint`](crate::solver::xpbd::joint_constraint)),
//! which reads the active joints each sub-step and corrects the bodies in place.
//!
//! # Provenance
//!
//! Original joint data model and storage. The solver's projection follows
//! Müller et al., *Detailed Rigid Body Simulation with Extended Position Based
//! Dynamics* (2020). This module contains **no Unreal Engine source or derived
//! code**.

pub mod anchor;
pub mod bounded_pd;
pub mod desc;
pub mod drive;
pub mod gear;
pub mod kind;
pub mod motor;
pub mod ragdoll;
pub mod revolute;
pub mod swing_twist;

pub use anchor::JointAnchor;
pub use bounded_pd::{BoundedPd, BoundedPdError, BoundedPdFailure};
pub use desc::JointDesc;
pub use drive::{ActuationBlend, AngularFriction, PoseDrive, SoftLimit};
pub use gear::GearJointState;
pub use kind::{
    DistanceJoint, FixedJoint, GearJoint, GearJointError, JointKind, PrismaticJoint, RevoluteJoint,
    SphericalJoint,
};
pub use motor::{AngleLimit, LinearLimit, Motor, MotorTarget};
pub use ragdoll::{
    HumanoidPart, HumanoidPreset, MassRatioPolicy, MassRatioWarning, RagdollBuildError,
    RagdollBuilder, RagdollDefinition, RagdollDiagnostics, RagdollLink, RagdollPart, RagdollPartId,
    SelfCollisionFilter, SelfCollisionPolicy,
};
pub use revolute::RevoluteJointState;
pub use swing_twist::{
    AngularJointInput, AngularJointResponse, JointFriction, SwingTwistConfig,
    SwingTwistConfigError, SwingTwistJoint, SwingTwistMeasurement,
};

/// A stable, generational reference to a joint stored in a [`JointStorage`].
///
/// Handles are cheap to copy and compare and stop validating once the target
/// slot is freed and its generation advances.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct JointHandle {
    index: u32,
    generation: u32,
}

impl JointHandle {
    /// A sentinel handle that never refers to a live joint.
    pub const INVALID: JointHandle = JointHandle {
        index: u32::MAX,
        generation: u32::MAX,
    };

    /// Creates a handle from raw parts (storage-internal).
    #[must_use]
    const fn new(index: u32, generation: u32) -> JointHandle {
        JointHandle { index, generation }
    }

    /// Returns the slot index this handle refers to.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// Returns the generation this handle was minted with.
    #[must_use]
    pub const fn generation(&self) -> u32 {
        self.generation
    }
}

impl Default for JointHandle {
    fn default() -> Self {
        JointHandle::INVALID
    }
}

/// A live joint: two anchors, a constraint kind, and an enabled flag.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Joint {
    /// Attachment on the first body.
    pub anchor_a: JointAnchor,
    /// Attachment on the second body.
    pub anchor_b: JointAnchor,
    /// The constraint family and its parameters.
    pub kind: JointKind,
    /// Whether the joint is solved this step.
    pub enabled: bool,
    /// Explicit scalar force/torque servo, supported on revolute/prismatic
    /// joints only. Do not combine with the family's XPBD `Motor`.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub bounded_pd: Option<BoundedPd>,
    /// Last enabled actuator sample failure. Cleared after a successful
    /// sample, including a zero effort. A failed sample applies no impulse.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub bounded_pd_failure: Option<BoundedPdFailure>,
    /// World-owned continuous angular history and substep XPBD multiplier.
    /// Cloned by full snapshots; never shared with another joint or solver.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub(crate) gear_state: GearJointState,
    /// Continuous hinge history, including while disabled or unrestricted.
    #[cfg_attr(feature = "serialize", serde(default))]
    pub(crate) revolute_state: RevoluteJointState,
}

impl Joint {
    /// Builds a joint from a description.
    #[must_use]
    fn from_desc(desc: JointDesc) -> Joint {
        Joint {
            anchor_a: desc.anchor_a,
            anchor_b: desc.anchor_b,
            kind: desc.kind,
            enabled: desc.enabled,
            bounded_pd: desc.bounded_pd,
            bounded_pd_failure: None,
            gear_state: GearJointState::default(),
            revolute_state: RevoluteJointState::default(),
        }
    }

    /// Continuous shaft coordinates, including all completed revolutions.
    #[must_use]
    pub fn gear_state(&self) -> &GearJointState {
        &self.gear_state
    }

    /// Rebases the continuous coordinates on the next observation. Use after
    /// teleporting a shaft by an ambiguous (>= pi) angular displacement.
    pub fn reset_gear_state(&mut self) {
        self.gear_state = GearJointState::default();
    }

    /// Last observed continuous hinge coordinate and its world-owned history.
    /// Limits and position motors use this unwrapped coordinate, not modulo tau.
    #[must_use]
    pub fn revolute_state(&self) -> &RevoluteJointState {
        &self.revolute_state
    }

    /// Binds the current pose to an explicit continuous coordinate in radians.
    /// Pass endpoint A/B **body world orientations**, not anchor orientations.
    /// Use for reduced-coordinate handoff or after an ambiguous teleport.
    /// The supplied coordinate defines the branch/zero at this pose; no wrap
    /// or limit clamp is applied. Nonfinite coordinates, non-hinges, invalid
    /// frames/axes and undefined twist return false without changing history.
    #[must_use]
    pub fn set_revolute_coordinate(
        &mut self,
        angle: f64,
        body_orientations: [glam::Quat; 2],
    ) -> bool {
        let JointKind::Revolute(params) = self.kind else {
            return false;
        };
        self.revolute_state.seed(
            angle,
            params.axis,
            self.anchor_a,
            self.anchor_b,
            body_orientations,
        )
    }

    /// Rebase to the principal twist on the next observation. This deliberately
    /// discards completed turns; ordinary enabling/retuning must not call it.
    pub fn reset_revolute_state(&mut self) {
        self.revolute_state = RevoluteJointState::default();
    }
}

/// Generational storage for joints.
///
/// Each slot keeps the joint (when live), a generation counter, and an active
/// flag; freed slots are recycled through a free list, mirroring
/// [`BodyStorage`](crate::state::storage::BodyStorage).
#[derive(Clone, Debug, Default)]
pub struct JointStorage {
    joints: Vec<Joint>,
    generations: Vec<u32>,
    active: Vec<bool>,
    free_list: Vec<u32>,
    live: usize,
}

impl JointStorage {
    /// Creates an empty storage.
    #[must_use]
    pub fn new() -> JointStorage {
        JointStorage::default()
    }

    /// Creates an empty storage with room reserved for `capacity` joints.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> JointStorage {
        JointStorage {
            joints: Vec::with_capacity(capacity),
            generations: Vec::with_capacity(capacity),
            active: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            live: 0,
        }
    }

    /// Inserts a joint described by `desc`, returning a handle to it.
    pub fn insert(&mut self, desc: JointDesc) -> JointHandle {
        let joint = Joint::from_desc(desc);
        self.live += 1;
        if let Some(index) = self.free_list.pop() {
            let i = index as usize;
            self.joints[i] = joint;
            self.active[i] = true;
            JointHandle::new(index, self.generations[i])
        } else {
            let index = self.joints.len() as u32;
            self.joints.push(joint);
            self.generations.push(0);
            self.active.push(true);
            JointHandle::new(index, 0)
        }
    }

    /// Removes the joint referenced by `handle`.
    ///
    /// Returns `true` if a live joint was removed, or `false` if the handle was
    /// stale or invalid.
    pub fn remove(&mut self, handle: JointHandle) -> bool {
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

    /// Returns `true` if `handle` refers to a live joint.
    #[must_use]
    pub fn contains(&self, handle: JointHandle) -> bool {
        let i = handle.index() as usize;
        i < self.active.len() && self.active[i] && self.generations[i] == handle.generation()
    }

    /// Returns a shared reference to the joint behind `handle`, if live.
    #[must_use]
    pub fn get(&self, handle: JointHandle) -> Option<&Joint> {
        if self.contains(handle) {
            Some(&self.joints[handle.index() as usize])
        } else {
            None
        }
    }

    /// Returns a mutable reference to the joint behind `handle`, if live.
    #[must_use]
    pub fn get_mut(&mut self, handle: JointHandle) -> Option<&mut Joint> {
        if self.contains(handle) {
            Some(&mut self.joints[handle.index() as usize])
        } else {
            None
        }
    }

    /// Returns the number of live joints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live
    }

    /// Returns `true` if there are no live joints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Returns the number of allocated slots, including freed ones.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.joints.len()
    }

    /// Iterates over every live, enabled joint.
    pub fn active_joints(&self) -> impl Iterator<Item = &Joint> {
        self.joints
            .iter()
            .enumerate()
            .filter(|(i, joint)| self.active[*i] && joint.enabled)
            .map(|(_, joint)| joint)
    }

    /// Iterates over every live, enabled joint together with its generational
    /// handle. Solvers that need to report measured constraint loads must use
    /// this rather than reconstructing handles from slot indices: a recycled
    /// slot has a different generation and must never receive an old load.
    pub fn active_entries(&self) -> impl Iterator<Item = (JointHandle, &Joint)> {
        self.joints
            .iter()
            .enumerate()
            .filter(move |(i, joint)| self.active[*i] && joint.enabled)
            .map(move |(i, joint)| (JointHandle::new(i as u32, self.generations[i]), joint))
    }

    /// Mutable active entries, including world-owned constraint history.
    pub fn active_entries_mut(&mut self) -> impl Iterator<Item = (JointHandle, &mut Joint)> {
        let active = &self.active;
        let generations = &self.generations;
        self.joints
            .iter_mut()
            .enumerate()
            .filter(move |(i, joint)| active[*i] && joint.enabled)
            .map(move |(i, joint)| (JointHandle::new(i as u32, generations[i]), joint))
    }

    /// Visits live joints even when disabled, so a temporarily disengaged gear
    /// can retain the shaft winding numbers until it is re-engaged.
    pub(crate) fn live_joints_mut(&mut self) -> impl Iterator<Item = &mut Joint> {
        let active = &self.active;
        self.joints
            .iter_mut()
            .enumerate()
            .filter(move |(i, _)| active[*i])
            .map(|(_, joint)| joint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::handle::BodyHandle;

    fn sample_desc() -> JointDesc {
        JointDesc::fixed(
            JointAnchor::at_center(BodyHandle::INVALID),
            JointAnchor::at_center(BodyHandle::INVALID),
        )
    }

    #[test]
    fn insert_get_remove_roundtrip() {
        let mut s = JointStorage::new();
        let h = s.insert(sample_desc());
        assert_eq!(s.len(), 1);
        assert!(s.get(h).is_some());
        assert!(s.remove(h));
        assert!(!s.contains(h));
        assert_eq!(s.len(), 0);
        assert!(s.get(h).is_none());
    }

    #[test]
    fn freed_slot_is_recycled_with_new_generation() {
        let mut s = JointStorage::new();
        let h0 = s.insert(sample_desc());
        assert!(s.remove(h0));
        let h1 = s.insert(sample_desc());
        // Same slot index, but the stale handle no longer validates.
        assert_eq!(h0.index(), h1.index());
        assert_ne!(h0.generation(), h1.generation());
        assert!(!s.contains(h0));
        assert!(s.contains(h1));
    }

    #[test]
    fn active_joints_skips_disabled() {
        let mut s = JointStorage::new();
        s.insert(sample_desc());
        s.insert(sample_desc().with_enabled(false));
        assert_eq!(s.active_joints().count(), 1);
    }
}
