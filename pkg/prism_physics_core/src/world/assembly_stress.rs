//! Incremental member-load reactions for merged, rigid welded trees.
//!
//! This is rigid-body load recovery, not material stress/FEA. Each tree cut has
//! a unique reaction: sum the members' inertial wrench minus their applied
//! wrench, then transport the result to that connection's anchor. A cyclic or
//! compliant structure needs a constitutive load-sharing model and is rejected.

use super::*;
use crate::dynamics::assembly::JointLoad;
use crate::dynamics::force::ForceTorque;
use crate::joint::JointKind;
use glam::{DMat3, DQuat, DVec3};

const MAX_MEMBERS: usize = 256;
const MAX_CONTRIBUTIONS: usize = 4096;

/// A one-substep world-space load on an original member of a live merged proxy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssemblyMemberWrench {
    /// Exact pre-merge generational handle, qualified by the API's live proxy.
    pub member: BodyHandle,
    /// World-space force in newtons, applied at this member's COM.
    pub force: Vec3,
    /// Pure torque about this member's COM, not about the proxy COM.
    pub torque: Vec3,
}

/// Recovered reaction on the joint's original side A, in N and Nm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssemblyConnectionReaction {
    /// Live generational handle of the disabled internal joint.
    pub joint: JointHandle,
    /// World-space force on the original side A, in newtons.
    pub force: Vec3,
    /// Pure couple at the joint anchor; lever-arm torque is removed.
    pub torque: Vec3,
}

/// Rejection leaves force queues, body state and fracture policies unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssemblyStressError {
    /// The handle is not a live merged proxy of this world.
    InvalidProxy,
    /// A member handle does not match this merge's original member generation.
    UnknownMember,
    /// The merge is outside 2..=256 members or would exceed 4096 queued loads.
    CapacityExceeded,
    /// Internal connections do not form a connected tree with unique joints.
    UnsupportedTopology,
    /// A connection is not an unchanged, merge-disabled rigid fixed joint.
    UnsupportedJoint,
    /// Mass, inertia, pose or fixed-joint anchor/frame geometry is inconsistent.
    InvalidGeometry,
    /// A load is nonfinite or its accumulated wrench/reaction overflows.
    NonFiniteLoad,
    /// A rejected fracture boundary must be repaired/retried first.
    BoundaryPending,
}

#[derive(Clone, Debug, Default)]
pub(super) struct AssemblyStressState {
    pending: BTreeMap<u64, PendingMemberStress>,
    pub(super) measured: Vec<SolverJointLoad>,
}

#[derive(Clone, Debug)]
struct PendingMemberStress {
    proxy: BodyHandle,
    contributions: Vec<AssemblyMemberWrench>,
    forces: ForceAccumulator,
    reactions: Vec<AssemblyConnectionReaction>,
}

impl AssemblyStressState {
    pub(super) fn has_pending(&self, proxy: BodyHandle) -> bool {
        self.pending.contains_key(&body_sort_key(&proxy))
    }
}

impl PhysicsWorld {
    /// Queues real member-resolved loads, applying the resultant to the proxy
    /// exactly once at the next external-force consumption. Reactions are
    /// recovered from individual masses/full inertias, not the proxy net load.
    /// Subsequent calls accumulate; the returned reactions cover the entire
    /// pending batch in canonical member/value order.
    ///
    /// Supported: 2..=256 dynamic members connected by an enabled, zero-
    /// compliance FixedJoint tree before merge (the live joints are disabled
    /// by merge). At most 4096 contributions may be queued before consumption.
    /// Geometry is sampled when queued, like `apply_force`'s world-space point.
    /// These are the incremental reactions due to these explicit loads only:
    /// contact/field/aero loads, impulses, centrifugal/gyroscopic preload and
    /// damping are NOT inferred from the proxy's motion. Apply this method once
    /// per desired loaded substep; the ordinary XPBD boundary handles fracture,
    /// fatigue, atomic split and retry. Free-body `step_external_dynamics` also
    /// completes that boundary. A custom backend using low-level force
    /// consumption must finish with `submit_joint_loads(dt, solver_loads)`.
    ///
    /// The queue is world-owned (including full snapshots), separate from the
    /// public unassigned proxy-force accumulator. A manual split is rejected
    /// while this queue is pending; consume it or explicitly cancel it first.
    /// Do not mutate raw body/joint geometry between queuing and consumption.
    /// A successful fracture restores individual members; remerging surviving
    /// connected components is an explicit caller operation, not automatic.
    ///
    /// Returns world-space reactions sorted by joint index/generation. An empty
    /// input validates and reports the current pending batch without queuing or
    /// waking the proxy. All errors leave the world unchanged.
    ///
    /// # Errors
    ///
    /// Rejects stale proxy/member identities, unsupported topology/joints,
    /// inconsistent geometry, capacity overflow and nonfinite arithmetic. A
    /// retained failed automatic fracture boundary must be repaired and retried
    /// before more member loads may be queued.
    pub fn apply_assembly_member_wrenches(
        &mut self,
        proxy: BodyHandle,
        wrenches: &[AssemblyMemberWrench],
    ) -> Result<Vec<AssemblyConnectionReaction>, AssemblyStressError> {
        if self.joint_boundary_failure.is_some() {
            return Err(AssemblyStressError::BoundaryPending);
        }
        let key = body_sort_key(&proxy);
        let state = self
            .assembly_merges
            .get(&key)
            .filter(|state| state.proxy == proxy && self.bodies.contains(proxy))
            .ok_or(AssemblyStressError::InvalidProxy)?;
        let previous = self.assembly_stress.pending.get(&key);
        let count = previous.map_or(0, |value| value.contributions.len());
        if wrenches.len() > MAX_CONTRIBUTIONS.saturating_sub(count) {
            return Err(AssemblyStressError::CapacityExceeded);
        }
        let mut contributions = previous
            .map(|value| value.contributions.clone())
            .unwrap_or_default();
        contributions.extend_from_slice(wrenches);
        if contributions
            .iter()
            .any(|load| !load.force.is_finite() || !load.torque.is_finite())
        {
            return Err(AssemblyStressError::NonFiniteLoad);
        }
        contributions.sort_by_key(|load| {
            (
                body_sort_key(&load.member),
                load.force.to_array().map(f32::to_bits),
                load.torque.to_array().map(f32::to_bits),
            )
        });
        let (wrench, reactions) = recover_reactions(self, state, &contributions)?;
        // All numeric, topology and generation checks precede any mutation.
        let mut forces = ForceAccumulator::new();
        forces
            .accumulate(proxy, wrench)
            .map_err(|_| AssemblyStressError::NonFiniteLoad)?;
        if !wrenches.is_empty() {
            self.assembly_stress.pending.insert(
                key,
                PendingMemberStress {
                    proxy,
                    contributions,
                    forces,
                    reactions: reactions.clone(),
                },
            );
            self.bodies.wake(proxy);
        }
        Ok(reactions)
    }

    /// Cancels only not-yet-consumed member loads. It cannot erase a measured
    /// fracture boundary or unrelated ordinary proxy loads.
    /// Returns whether a pending batch existed for this exact proxy generation.
    pub fn cancel_assembly_member_wrenches(&mut self, proxy: BodyHandle) -> bool {
        self.assembly_stress
            .pending
            .remove(&body_sort_key(&proxy))
            .is_some()
    }

    pub(super) fn consume_assembly_member_wrenches(&mut self, dt: Real) -> usize {
        if !dt.is_finite() || dt <= 0.0 {
            return 0;
        }
        let mut additional = 0;
        for (_, mut pending) in std::mem::take(&mut self.assembly_stress.pending) {
            let unassigned = self.external_forces.pending(pending.proxy).is_some();
            let consumed = pending.forces.consume(&mut self.bodies, dt);
            if consumed == 0 {
                // The public raw body store may have retired the proxy. Never
                // apply an old generation's loads to a recycled slot.
                continue;
            }
            if !unassigned {
                additional += 1;
            }
            self.assembly_stress
                .measured
                .extend(pending.reactions.into_iter().map(|value| SolverJointLoad {
                    joint: value.joint,
                    load: JointLoad {
                        linear_impulse: value.force * dt,
                        angular_impulse: value.torque * dt,
                        substep_dt: dt,
                    },
                }));
        }
        additional
    }
}

struct Edge {
    a: usize,
    b: usize,
    joint: JointHandle,
    /// Endpoint carrying the original joint's side A (graph order may differ).
    side_a: usize,
    anchor: DVec3,
}

fn finite_vec(value: DVec3) -> Result<Vec3, AssemblyStressError> {
    let result = value.as_vec3();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(AssemblyStressError::NonFiniteLoad)
    }
}

fn recover_reactions(
    world: &PhysicsWorld,
    state: &AssemblyMergeState,
    contributions: &[AssemblyMemberWrench],
) -> Result<(ForceTorque, Vec<AssemblyConnectionReaction>), AssemblyStressError> {
    let parts = &state.compound.parts;
    let n = parts.len();
    if !(2..=MAX_MEMBERS).contains(&n) {
        return Err(AssemblyStressError::CapacityExceeded);
    }
    if state.connections.len() != n - 1 {
        return Err(AssemblyStressError::UnsupportedTopology);
    }
    let index = |body: BodyHandle| {
        parts
            .binary_search_by_key(&body_sort_key(&body), |part| body_sort_key(&part.body))
            .map_err(|_| AssemblyStressError::UnknownMember)
    };
    let desc = world
        .bodies
        .descriptor(state.proxy)
        .ok_or(AssemblyStressError::InvalidProxy)?;
    let (_, inverse_inertia) = state.compound.principal_mass_properties();
    if desc.kind != crate::state::body::BodyKind::Dynamic
        || desc.mass_properties.inv_mass != 1.0 / state.compound.properties.mass
        || desc.mass_properties.inv_inertia != inverse_inertia
        || !desc.position.is_finite()
        || !desc.orientation.is_finite()
        || !desc.orientation.is_normalized()
    {
        return Err(AssemblyStressError::InvalidGeometry);
    }
    let rotation = DQuat::from_xyzw(
        desc.orientation.x as f64,
        desc.orientation.y as f64,
        desc.orientation.z as f64,
        desc.orientation.w as f64,
    )
    .normalize();
    let offsets: Vec<_> = parts
        .iter()
        .map(|part| rotation * part.relative_pose.position.as_dvec3())
        .collect();
    let mut inertias = Vec::with_capacity(n);
    let mut inertia = DMat3::ZERO;
    let mut mass = 0.0;
    for (part, &r) in parts.iter().zip(&offsets) {
        let q = part.relative_pose.orientation;
        let q =
            rotation * DQuat::from_xyzw(q.x as f64, q.y as f64, q.z as f64, q.w as f64).normalize();
        let basis = DMat3::from_quat(q);
        let member_inertia = basis
            * DMat3::from_diagonal(part.mass_properties.inv_inertia.as_dvec3().recip())
            * basis.transpose();
        inertia += member_inertia
            + (DMat3::from_diagonal(DVec3::splat(r.length_squared()))
                - DMat3::from_cols(r * r.x, r * r.y, r * r.z))
                * part.mass as f64;
        mass += part.mass as f64;
        inertias.push(member_inertia);
    }
    if !inertia.is_finite() || !inertia.determinant().is_finite() || inertia.determinant() <= 0.0 {
        return Err(AssemblyStressError::InvalidGeometry);
    }
    let mut edges = Vec::with_capacity(n - 1);
    let mut adjacency = vec![Vec::new(); n];
    let mut used_joints = Vec::with_capacity(n - 1);
    for entry in &state.connections {
        let connection = entry.connection;
        let a = index(connection.a)?;
        let b = index(connection.b)?;
        let handle = connection
            .joint
            .ok_or(AssemblyStressError::UnsupportedJoint)?;
        let joint = entry.joint.ok_or(AssemblyStressError::UnsupportedJoint)?;
        let live = world
            .joints
            .get(handle)
            .ok_or(AssemblyStressError::UnsupportedJoint)?;
        if !connection.enabled
            || !joint.enabled
            || live.enabled
            || !matches!(joint.kind, JointKind::Fixed(fixed) if fixed.compliance == 0.0)
            || connection.kind != joint.kind
            || live.kind != joint.kind
            || joint.bounded_pd.is_some()
            || live.bounded_pd.is_some()
            || live.anchor_a.body != state.proxy
            || live.anchor_b.body != state.proxy
            || live.anchor_a.local_point != joint.anchor_a.local_point
            || live.anchor_b.local_point != joint.anchor_b.local_point
            || live.anchor_a.local_frame != joint.anchor_a.local_frame
            || live.anchor_b.local_frame != joint.anchor_b.local_frame
        {
            return Err(AssemblyStressError::UnsupportedJoint);
        }
        if a == b || used_joints.contains(&handle) {
            return Err(AssemblyStressError::UnsupportedTopology);
        }
        used_joints.push(handle);
        let side_a = index(joint.anchor_a.body)?;
        let side_b = index(joint.anchor_b.body)?;
        if !((side_a == a && side_b == b) || (side_a == b && side_b == a)) {
            return Err(AssemblyStressError::UnsupportedJoint);
        }
        let anchor_offset = |i: usize, local: Vec3| {
            let q = parts[i].relative_pose.orientation;
            offsets[i] + rotation * (q * local).as_dvec3()
        };
        let anchor = anchor_offset(side_a, joint.anchor_a.local_point);
        let other = anchor_offset(side_b, joint.anchor_b.local_point);
        let frame_a = parts[side_a].relative_pose.orientation * joint.anchor_a.local_frame;
        let frame_b = parts[side_b].relative_pose.orientation * joint.anchor_b.local_frame;
        // A collapsed rigid connection must have coincident anchors. No
        // preload/deformation law is fabricated for an inconsistent weld.
        let scale = 1.0 + anchor.length().max(other.length());
        if !anchor.is_finite()
            || !other.is_finite()
            || (anchor - other).length() > 1.0e-4 * scale
            || !frame_a.is_finite()
            || !frame_b.is_finite()
            || !frame_a.is_normalized()
            || !frame_b.is_normalized()
            || frame_a.dot(frame_b).abs() < 1.0 - 1.0e-5
        {
            return Err(AssemblyStressError::InvalidGeometry);
        }
        adjacency[a].push((b, edges.len()));
        adjacency[b].push((a, edges.len()));
        edges.push(Edge {
            a,
            b,
            joint: handle,
            side_a,
            anchor,
        });
    }
    // Root at the lowest original generational member; postorder cuts recover
    // all connection wrenches in O(members + connections) after validation.
    let mut parents = vec![None; n];
    let mut order = vec![0];
    parents[0] = Some((0, usize::MAX));
    let mut cursor = 0;
    while cursor < order.len() {
        let i = order[cursor];
        for &(next, edge) in &adjacency[i] {
            if parents[next].is_none() {
                parents[next] = Some((i, edge));
                order.push(next);
            } else if parents[i].is_none_or(|(parent, _)| parent != next) {
                return Err(AssemblyStressError::UnsupportedTopology);
            }
        }
        cursor += 1;
    }
    if order.len() != n {
        return Err(AssemblyStressError::UnsupportedTopology);
    }
    let mut forces = vec![DVec3::ZERO; n];
    let mut torques = vec![DVec3::ZERO; n];
    for value in contributions {
        let i = index(value.member)?;
        forces[i] += value.force.as_dvec3();
        torques[i] += value.torque.as_dvec3() + offsets[i].cross(value.force.as_dvec3());
    }
    let force: DVec3 = forces.iter().copied().sum();
    let torque: DVec3 = torques.iter().copied().sum();
    let acceleration = force / mass;
    let alpha = inertia.inverse() * torque;
    for i in 0..n {
        let required_force = (acceleration + alpha.cross(offsets[i])) * parts[i].mass as f64;
        forces[i] = required_force - forces[i];
        torques[i] = inertias[i] * alpha + offsets[i].cross(required_force) - torques[i];
    }
    let mut reactions = Vec::with_capacity(edges.len());
    for &i in order.iter().skip(1).rev() {
        let (parent, edge_index) = parents[i].ok_or(AssemblyStressError::UnsupportedTopology)?;
        let edge = &edges[edge_index];
        debug_assert!((edge.a == i && edge.b == parent) || (edge.b == i && edge.a == parent));
        let sign = if edge.side_a == i { 1.0 } else { -1.0 };
        reactions.push(AssemblyConnectionReaction {
            joint: edge.joint,
            force: finite_vec(forces[i] * sign)?,
            torque: finite_vec((torques[i] - edge.anchor.cross(forces[i])) * sign)?,
        });
        let subtree_force = forces[i];
        let subtree_torque = torques[i];
        forces[parent] += subtree_force;
        torques[parent] += subtree_torque;
    }
    reactions.sort_by_key(|value| (value.joint.index(), value.joint.generation()));
    Ok((
        ForceTorque {
            force: finite_vec(force)?,
            torque: finite_vec(torque)?,
        },
        reactions,
    ))
}
