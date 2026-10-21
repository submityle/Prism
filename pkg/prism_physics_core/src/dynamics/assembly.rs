//! Runtime assembly graphs, conserved compound-body state, break policies and
//! compliant grab drives.

use crate::dynamics::force::{inertia_world, ForceTorque};
use crate::joint::{JointHandle, JointKind, JointStorage};
use crate::state::body::{BodyDesc, MassProperties};
use crate::state::handle::BodyHandle;
use glam::{Mat3, Quat, Vec3};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const EPS: f32 = 1.0e-7;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub position: Vec3,
    pub orientation: Quat,
}

impl Pose {
    #[must_use]
    pub fn identity() -> Self {
        Self {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyState {
    pub handle: BodyHandle,
    pub pose: Pose,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub mass_properties: MassProperties,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodySnapshot {
    pub handle: BodyHandle,
    pub desc: BodyDesc,
    pub previous_pose: Pose,
    pub sleeping: bool,
    pub sleep_timer: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreakLimit {
    pub max_force: f32,
    pub max_torque: f32,
    pub fatigue_rate: f32,
}

impl BreakLimit {
    #[must_use]
    pub fn new(max_force: f32, max_torque: f32) -> Self {
        Self {
            max_force,
            max_torque,
            fatigue_rate: 0.0,
        }
    }
    #[must_use]
    pub fn with_fatigue(mut self, rate: f32) -> Self {
        self.fatigue_rate = rate.max(0.0);
        self
    }
    #[must_use]
    fn valid(&self) -> bool {
        self.max_force.is_finite()
            && self.max_torque.is_finite()
            && self.max_force > 0.0
            && self.max_torque > 0.0
            && self.fatigue_rate.is_finite()
            && self.fatigue_rate >= 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointLoad {
    pub linear_impulse: Vec3,
    pub angular_impulse: Vec3,
    pub substep_dt: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverJointLoad {
    pub joint: JointHandle,
    pub load: JointLoad,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryLoadError {
    InvalidDt,
    NonFiniteLoad,
    MismatchedDt,
    UnknownJoint,
    /// A collapsed assembly cannot currently be restored. No policies,
    /// topology, allocator state, or queued forces were committed.
    AssemblySplit {
        proxy: BodyHandle,
        error: crate::world::AssemblyWorldError,
    },
}

/// Deterministic collection of all constraint impulses produced during one
/// solver sub-step. Entries may be recorded in any order, including from
/// parallel constraint workers; submission emits a stable joint order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JointLoadAccumulator {
    loads: BTreeMap<u64, (JointHandle, Vec<JointLoad>)>,
    substep_dt: Option<f32>,
}

impl JointLoadAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.loads.clear();
        self.substep_dt = None;
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.loads.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.loads.len()
    }

    /// Records a world-space reaction impulse on side A of a joint. Do not
    /// also record the equal and opposite side-B reaction.
    pub fn record(&mut self, joint: JointHandle, load: JointLoad) -> Result<(), BoundaryLoadError> {
        validate_joint_load(load)?;
        if let Some(dt) = self.substep_dt {
            if dt != load.substep_dt {
                return Err(BoundaryLoadError::MismatchedDt);
            }
        }
        let key = joint_key(joint);
        self.loads
            .entry(key)
            .or_insert_with(|| (joint, Vec::new()))
            .1
            .push(load);
        self.substep_dt.get_or_insert(load.substep_dt);
        Ok(())
    }

    pub fn record_impulse(
        &mut self,
        joint: JointHandle,
        linear_impulse: Vec3,
        angular_impulse: Vec3,
        substep_dt: f32,
    ) -> Result<(), BoundaryLoadError> {
        self.record(
            joint,
            JointLoad {
                linear_impulse,
                angular_impulse,
                substep_dt,
            },
        )
    }

    /// Converts XPBD position-level multipliers to impulses (`lambda / h`).
    /// Angular lambda is the pure rotational constraint reaction at the joint,
    /// not the lever-arm torque already represented by the linear reaction.
    pub fn record_projection(
        &mut self,
        joint: JointHandle,
        linear_lambda: Vec3,
        angular_lambda: Vec3,
        substep_dt: f32,
    ) -> Result<(), BoundaryLoadError> {
        if !substep_dt.is_finite() || substep_dt <= 0.0 {
            return Err(BoundaryLoadError::InvalidDt);
        }
        self.record_impulse(
            joint,
            linear_lambda / substep_dt,
            angular_lambda / substep_dt,
            substep_dt,
        )
    }

    /// Joins worker-local contributions without changing their reduction
    /// order. A failed merge leaves the receiver unchanged.
    pub fn merge(&mut self, other: &Self) -> Result<(), BoundaryLoadError> {
        if let (Some(left), Some(right)) = (self.substep_dt, other.substep_dt) {
            if left != right {
                return Err(BoundaryLoadError::MismatchedDt);
            }
        }
        for (key, (joint, loads)) in &other.loads {
            self.loads
                .entry(*key)
                .or_insert_with(|| (*joint, Vec::new()))
                .1
                .extend_from_slice(loads);
        }
        self.substep_dt = self.substep_dt.or(other.substep_dt);
        Ok(())
    }

    pub fn loads(&self) -> Result<Vec<SolverJointLoad>, BoundaryLoadError> {
        let mut result = Vec::with_capacity(self.loads.len());
        for (joint, contributions) in self.loads.values() {
            let mut ordered = contributions.clone();
            ordered.sort_by(|left, right| {
                cmp_vec3_total(left.linear_impulse, right.linear_impulse)
                    .then_with(|| cmp_vec3_total(left.angular_impulse, right.angular_impulse))
            });
            // Numeric total ordering keeps negative and positive contributions
            // in canonical order while remaining deterministic for signed zero.
            let mut linear = [0.0_f64; 3];
            let mut angular = [0.0_f64; 3];
            for load in &ordered {
                let l = load.linear_impulse.to_array();
                let a = load.angular_impulse.to_array();
                for axis in 0..3 {
                    linear[axis] += f64::from(l[axis]);
                    angular[axis] += f64::from(a[axis]);
                }
            }
            let load = JointLoad {
                linear_impulse: Vec3::from_array(linear.map(|v| v as f32)),
                angular_impulse: Vec3::from_array(angular.map(|v| v as f32)),
                substep_dt: self
                    .substep_dt
                    .expect("nonempty contributions have a substep dt"),
            };
            validate_joint_load(load)?;
            result.push(SolverJointLoad {
                joint: *joint,
                load,
            });
        }
        Ok(result)
    }

    pub fn submit(
        &mut self,
        dt: f32,
        policies: &mut BreakableJointSet,
        joints: &mut JointStorage,
    ) -> Result<Vec<BreakEvent>, BoundaryLoadError> {
        if self
            .substep_dt
            .is_some_and(|recorded| !same_dt(recorded, dt))
        {
            return Err(BoundaryLoadError::MismatchedDt);
        }
        let events = policies.submit_solver_boundary(dt, &self.loads()?, joints)?;
        self.clear();
        Ok(events)
    }
}

fn cmp_vec3_total(left: Vec3, right: Vec3) -> core::cmp::Ordering {
    left.x
        .total_cmp(&right.x)
        .then_with(|| left.y.total_cmp(&right.y))
        .then_with(|| left.z.total_cmp(&right.z))
}

fn same_dt(left: f32, right: f32) -> bool {
    (left - right).abs() <= left.abs().max(right.abs()) * 1.0e-5
}

fn validate_joint_load(load: JointLoad) -> Result<(), BoundaryLoadError> {
    if !load.linear_impulse.is_finite()
        || !load.angular_impulse.is_finite()
        || !load.substep_dt.is_finite()
    {
        return Err(BoundaryLoadError::NonFiniteLoad);
    }
    if load.substep_dt <= 0.0 {
        return Err(BoundaryLoadError::InvalidDt);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreakEvent {
    pub joint: JointHandle,
    pub force: f32,
    pub torque: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BreakState {
    joint: JointHandle,
    limit: BreakLimit,
    impulse: Vec3,
    angular_impulse: Vec3,
    damage: f32,
    broken: bool,
}

/// A deterministic policy store for all joint kinds.  The XPBD solver can
/// report its measured constraint impulses at a sub-step boundary; this type
/// converts them to force/torque, accumulates fatigue and disables the joint
/// only at the boundary.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BreakableJointSet {
    policies: BTreeMap<u64, BreakState>,
}

impl BreakableJointSet {
    pub fn register(&mut self, joint: JointHandle, limit: BreakLimit) -> Result<(), BreakLimit> {
        if !limit.valid() || self.policies.contains_key(&joint_key(joint)) {
            return Err(limit);
        }
        self.insert_validated(joint, limit);
        Ok(())
    }

    // Only called after validating the limit and policy uniqueness while
    // holding exclusive access. There is no recoverable failure after this.
    fn insert_validated(&mut self, joint: JointHandle, limit: BreakLimit) {
        self.policies.insert(
            joint_key(joint),
            BreakState {
                joint,
                limit,
                impulse: Vec3::ZERO,
                angular_impulse: Vec3::ZERO,
                damage: 0.0,
                broken: false,
            },
        );
    }

    #[must_use]
    pub fn contains(&self, joint: JointHandle) -> bool {
        self.policies.contains_key(&joint_key(joint))
    }

    pub fn unregister(&mut self, joint: JointHandle) -> bool {
        self.policies.remove(&joint_key(joint)).is_some()
    }

    pub fn begin_substep(&mut self) {
        for state in self.policies.values_mut() {
            state.impulse = Vec3::ZERO;
            state.angular_impulse = Vec3::ZERO;
        }
    }

    pub fn record_load(&mut self, joint: JointHandle, load: JointLoad) -> bool {
        if !load.linear_impulse.is_finite()
            || !load.angular_impulse.is_finite()
            || !load.substep_dt.is_finite()
            || load.substep_dt <= 0.0
        {
            return false;
        }
        let Some(state) = self.policies.get_mut(&joint_key(joint)) else {
            return false;
        };
        if state.broken {
            return false;
        }
        let impulse = state.impulse + load.linear_impulse;
        let angular_impulse = state.angular_impulse + load.angular_impulse;
        if !impulse.is_finite() || !angular_impulse.is_finite() {
            return false;
        }
        state.impulse = impulse;
        state.angular_impulse = angular_impulse;
        true
    }

    /// Commits all constraint loads produced by one solver sub-step. The
    /// input is copied and canonically ordered before accumulation, so worker
    /// completion order cannot change the result. Evaluation and joint
    /// disabling happen at the same boundary.
    pub fn submit_solver_boundary(
        &mut self,
        dt: f32,
        loads: &[SolverJointLoad],
        joints: &mut JointStorage,
    ) -> Result<Vec<BreakEvent>, BoundaryLoadError> {
        let (candidate, events) = self.prepare_solver_boundary(dt, loads, joints)?;
        candidate.disable_broken(joints, &events);
        *self = candidate;
        Ok(events)
    }

    /// Computes policy and fatigue changes without disabling any live joint.
    /// World-level topology transactions commit this only after every affected
    /// assembly has been validated against the same replacement allocation.
    pub(crate) fn prepare_solver_boundary(
        &self,
        dt: f32,
        loads: &[SolverJointLoad],
        joints: &JointStorage,
    ) -> Result<(Self, Vec<BreakEvent>), BoundaryLoadError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(BoundaryLoadError::InvalidDt);
        }
        let mut ordered = loads.to_vec();
        ordered.sort_by(|a, b| {
            joint_key(a.joint)
                .cmp(&joint_key(b.joint))
                .then_with(|| {
                    a.load
                        .linear_impulse
                        .to_array()
                        .map(f32::to_bits)
                        .cmp(&b.load.linear_impulse.to_array().map(f32::to_bits))
                })
                .then_with(|| {
                    a.load
                        .angular_impulse
                        .to_array()
                        .map(f32::to_bits)
                        .cmp(&b.load.angular_impulse.to_array().map(f32::to_bits))
                })
        });
        for item in &ordered {
            let load = item.load;
            validate_joint_load(load)?;
            if !same_dt(load.substep_dt, dt) {
                return Err(BoundaryLoadError::MismatchedDt);
            }
            if !self.policies.contains_key(&joint_key(item.joint)) || !joints.contains(item.joint) {
                return Err(BoundaryLoadError::UnknownJoint);
            }
        }
        let mut candidate = self.clone();
        candidate.begin_substep();
        for item in ordered {
            if !candidate.policies[&joint_key(item.joint)].broken
                && !candidate.record_load(item.joint, item.load)
            {
                return Err(BoundaryLoadError::NonFiniteLoad);
            }
        }
        let events = candidate.evaluate(dt);
        Ok((candidate, events))
    }

    /// Evaluates all policies in stable handle order and returns newly broken
    /// joints.  The returned events are side-effect free with respect to the
    /// joint storage; call [`disable_broken`] after the sub-step.
    pub fn evaluate(&mut self, dt: f32) -> Vec<BreakEvent> {
        if !dt.is_finite() || dt <= 0.0 {
            return Vec::new();
        }
        let mut events = Vec::new();
        for state in self.policies.values_mut() {
            if state.broken {
                continue;
            }
            let force = state.impulse.length() / dt;
            let torque = state.angular_impulse.length() / dt;
            let normalized = (force / state.limit.max_force).max(torque / state.limit.max_torque);
            state.damage =
                (state.damage + normalized * state.limit.fatigue_rate * dt).clamp(0.0, 0.99);
            let force_limit = state.limit.max_force * (1.0 - state.damage);
            let torque_limit = state.limit.max_torque * (1.0 - state.damage);
            if force > force_limit || torque > torque_limit {
                state.broken = true;
                events.push(BreakEvent {
                    joint: state.joint,
                    force,
                    torque,
                });
            }
        }
        events
    }

    pub fn disable_broken(&self, joints: &mut JointStorage, events: &[BreakEvent]) {
        for event in events {
            if let Some(joint) = joints.get_mut(event.joint) {
                joint.enabled = false;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct BodyKey(u64);

fn body_key(body: BodyHandle) -> BodyKey {
    BodyKey((u64::from(body.index()) << 32) | u64::from(body.generation()))
}
fn joint_key(joint: JointHandle) -> u64 {
    (u64::from(joint.index()) << 32) | u64::from(joint.generation())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConnectionId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Connection {
    pub id: ConnectionId,
    pub a: BodyHandle,
    pub b: BodyHandle,
    pub joint: Option<JointHandle>,
    pub kind: JointKind,
    pub break_limit: Option<BreakLimit>,
    pub enabled: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssemblyError {
    SameBody,
    InvalidLimit,
    DuplicateConnection,
    MissingBody,
    ProxyConflict,
    DuplicatePolicy,
    IdExhausted,
}

/// The source-of-truth graph for runtime assembly.  Components are computed
/// with deterministic breadth-first traversal; no hash iteration participates
/// in the result.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AssemblyGraph {
    nodes: BTreeMap<BodyKey, BodyHandle>,
    connections: BTreeMap<ConnectionId, Connection>,
    next_id: u64,
}

impl AssemblyGraph {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_body(&mut self, body: BodyHandle) {
        self.nodes.insert(body_key(body), body);
    }
    pub fn remove_body(&mut self, body: BodyHandle) -> bool {
        let key = body_key(body);
        if self.nodes.remove(&key).is_none() {
            return false;
        }
        let ids: Vec<ConnectionId> = self
            .connections
            .values()
            .filter(|c| c.a == body || c.b == body)
            .map(|c| c.id)
            .collect();
        for id in ids {
            self.connections.remove(&id);
        }
        true
    }
    pub fn connect(
        &mut self,
        a: BodyHandle,
        b: BodyHandle,
        joint: Option<JointHandle>,
        kind: JointKind,
        break_limit: Option<BreakLimit>,
    ) -> Result<ConnectionId, AssemblyError> {
        let connection = self.prepare_connection(a, b, joint, kind, break_limit)?;
        Ok(self.commit_connection(connection))
    }

    // Preparation is read-only, including ID exhaustion. Keeping graph
    // validation separate lets the policy transaction share it without
    // cloning unrelated graph nodes, edges or accumulated fatigue state.
    fn prepare_connection(
        &self,
        a: BodyHandle,
        b: BodyHandle,
        joint: Option<JointHandle>,
        kind: JointKind,
        break_limit: Option<BreakLimit>,
    ) -> Result<Connection, AssemblyError> {
        if a == b {
            return Err(AssemblyError::SameBody);
        }
        if let Some(limit) = break_limit {
            if !limit.valid() {
                return Err(AssemblyError::InvalidLimit);
            }
        }
        if self
            .connections
            .values()
            .any(|c| c.enabled && ((c.a == a && c.b == b) || (c.a == b && c.b == a)))
        {
            return Err(AssemblyError::DuplicateConnection);
        }
        let id = self
            .next_id
            .checked_add(1)
            .ok_or(AssemblyError::IdExhausted)?;
        Ok(Connection {
            id: ConnectionId(id),
            a,
            b,
            joint,
            kind,
            break_limit,
            enabled: true,
        })
    }

    fn commit_connection(&mut self, connection: Connection) -> ConnectionId {
        self.add_body(connection.a);
        self.add_body(connection.b);
        self.next_id = connection.id.0;
        self.connections.insert(connection.id, connection);
        connection.id
    }

    /// Atomically connects an edge and registers its break policy. Both
    /// stores are committed only after every validation succeeds. Validation
    /// scans existing edges for duplicates, but does not clone either store.
    pub fn connect_with_policy(
        &mut self,
        policies: &mut BreakableJointSet,
        a: BodyHandle,
        b: BodyHandle,
        joint: Option<JointHandle>,
        kind: JointKind,
        break_limit: Option<BreakLimit>,
    ) -> Result<ConnectionId, AssemblyError> {
        let connection = self.prepare_connection(a, b, joint, kind, break_limit)?;
        if let (Some(joint), Some(_)) = (joint, break_limit) {
            if policies.contains(joint) {
                return Err(AssemblyError::DuplicatePolicy);
            }
        }
        // Exclusive borrows keep all validated inputs unchanged between
        // preparation and commit. Allocation failure is process-fatal, not
        // a recoverable transaction error.
        let id = self.commit_connection(connection);
        if let (Some(joint), Some(limit)) = (joint, break_limit) {
            policies.insert_validated(joint, limit);
        }
        Ok(id)
    }
    pub fn disconnect(&mut self, id: ConnectionId) -> Option<Connection> {
        self.connections.remove(&id)
    }

    /// Replaces all members of a component by one proxy in one graph
    /// transaction. Internal connections are removed and external connection
    /// endpoints are rewritten to the proxy while retaining their IDs.
    pub fn collapse_component(
        &mut self,
        component: &[BodyHandle],
        proxy: BodyHandle,
    ) -> Result<Vec<Connection>, AssemblyError> {
        if component.is_empty()
            || component
                .iter()
                .any(|body| !self.nodes.contains_key(&body_key(*body)))
        {
            return Err(AssemblyError::MissingBody);
        }
        if self.nodes.contains_key(&body_key(proxy)) || component.contains(&proxy) {
            return Err(AssemblyError::ProxyConflict);
        }
        let members: BTreeSet<BodyKey> = component.iter().copied().map(body_key).collect();
        let mut replacement = self.connections.clone();
        let mut affected = Vec::new();
        for (id, connection) in &self.connections {
            let a_member = members.contains(&body_key(connection.a));
            let b_member = members.contains(&body_key(connection.b));
            if !a_member && !b_member {
                continue;
            }
            affected.push(*connection);
            if a_member && b_member {
                replacement.remove(id);
            } else {
                let mut rewritten = *connection;
                if a_member {
                    rewritten.a = proxy;
                }
                if b_member {
                    rewritten.b = proxy;
                }
                if rewritten.a == rewritten.b {
                    return Err(AssemblyError::DuplicateConnection);
                }
                replacement.insert(*id, rewritten);
            }
        }
        let values: Vec<Connection> = replacement.values().copied().collect();
        for (i, left) in values.iter().enumerate() {
            if !left.enabled {
                continue;
            }
            for right in values.iter().skip(i + 1) {
                if right.enabled
                    && ((left.a == right.a && left.b == right.b)
                        || (left.a == right.b && left.b == right.a))
                {
                    return Err(AssemblyError::DuplicateConnection);
                }
            }
        }
        self.connections = replacement;
        for body in component {
            self.nodes.remove(&body_key(*body));
        }
        self.nodes.insert(body_key(proxy), proxy);
        affected.sort_by_key(|connection| connection.id);
        Ok(affected)
    }

    /// Restores the original nodes and connection records after a proxy has
    /// been removed. The caller supplies remapped connections when body slots
    /// received new generations during the split.
    pub fn restore_component(
        &mut self,
        proxy: BodyHandle,
        component: &[BodyHandle],
        original_connections: &[Connection],
    ) -> Result<(), AssemblyError> {
        if !self.nodes.contains_key(&body_key(proxy)) {
            return Err(AssemblyError::ProxyConflict);
        }
        if component.is_empty() {
            return Err(AssemblyError::MissingBody);
        }
        if component.contains(&proxy) {
            return Err(AssemblyError::ProxyConflict);
        }
        let mut candidate = self.clone();
        candidate.nodes.remove(&body_key(proxy));
        let proxy_ids: Vec<ConnectionId> = candidate
            .connections
            .values()
            .filter(|connection| connection.a == proxy || connection.b == proxy)
            .map(|connection| connection.id)
            .collect();
        for id in proxy_ids {
            candidate.connections.remove(&id);
        }
        let mut component_keys = BTreeSet::new();
        for body in component {
            if !component_keys.insert(body_key(*body))
                || candidate.nodes.contains_key(&body_key(*body))
            {
                return Err(AssemblyError::DuplicateConnection);
            }
        }
        for body in component {
            candidate.nodes.insert(body_key(*body), *body);
        }
        let existing_ids: BTreeSet<ConnectionId> = candidate.connections.keys().copied().collect();
        let mut inserted_ids = BTreeSet::new();
        for connection in original_connections {
            if connection.a == connection.b {
                return Err(AssemblyError::SameBody);
            }
            if connection.break_limit.is_some_and(|limit| !limit.valid()) {
                return Err(AssemblyError::InvalidLimit);
            }
            if !candidate.nodes.contains_key(&body_key(connection.a))
                || !candidate.nodes.contains_key(&body_key(connection.b))
            {
                return Err(AssemblyError::MissingBody);
            }
            if !inserted_ids.insert(connection.id) || existing_ids.contains(&connection.id) {
                return Err(AssemblyError::DuplicateConnection);
            }
            candidate.connections.insert(connection.id, *connection);
            candidate.next_id = candidate.next_id.max(connection.id.0);
        }
        let values: Vec<Connection> = candidate.connections.values().copied().collect();
        for (index, left) in values.iter().enumerate() {
            if !left.enabled {
                continue;
            }
            for right in values.iter().skip(index + 1) {
                if right.enabled
                    && ((left.a == right.a && left.b == right.b)
                        || (left.a == right.b && left.b == right.a))
                {
                    return Err(AssemblyError::DuplicateConnection);
                }
            }
        }
        *self = candidate;
        Ok(())
    }

    /// Detaches exactly the supplied nodes and returns their connections in
    /// stable id order. Callers can later reconnect them after replacing the
    /// physical representation of the component.
    pub fn detach_component(&mut self, component: &[BodyHandle]) -> Vec<Connection> {
        let members: BTreeSet<BodyKey> = component.iter().copied().map(body_key).collect();
        let ids: Vec<ConnectionId> = self
            .connections
            .values()
            .filter(|connection| {
                members.contains(&body_key(connection.a))
                    || members.contains(&body_key(connection.b))
            })
            .map(|connection| connection.id)
            .collect();
        let mut detached = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(connection) = self.connections.remove(&id) {
                detached.push(connection);
            }
        }
        for body in component {
            self.nodes.remove(&body_key(*body));
        }
        detached
    }
    #[must_use]
    pub fn connections(&self) -> impl Iterator<Item = &Connection> {
        self.connections.values()
    }
    #[must_use]
    pub fn connected_components(&self) -> Vec<Vec<BodyHandle>> {
        let mut adjacency: BTreeMap<BodyKey, Vec<BodyKey>> = self
            .nodes
            .keys()
            .copied()
            .map(|key| (key, Vec::new()))
            .collect();
        for connection in self.connections.values().filter(|c| c.enabled) {
            adjacency
                .entry(body_key(connection.a))
                .or_default()
                .push(body_key(connection.b));
            adjacency
                .entry(body_key(connection.b))
                .or_default()
                .push(body_key(connection.a));
        }
        for values in adjacency.values_mut() {
            values.sort();
        }
        let mut visited = BTreeSet::new();
        let mut result = Vec::new();
        for (&root, _) in &adjacency {
            if !visited.insert(root) {
                continue;
            }
            let mut queue = VecDeque::from([root]);
            let mut component = Vec::new();
            while let Some(key) = queue.pop_front() {
                component.push(self.nodes[&key]);
                for &next in &adjacency[&key] {
                    if visited.insert(next) {
                        queue.push_back(next);
                    }
                }
            }
            component.sort_by_key(|body| body_key(*body));
            result.push(component);
        }
        result
    }
    #[must_use]
    pub fn component_of(&self, body: BodyHandle) -> Option<Vec<BodyHandle>> {
        self.connected_components()
            .into_iter()
            .find(|component| component.contains(&body))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssemblyPart {
    pub body: BodyHandle,
    pub relative_pose: Pose,
    pub mass: f32,
    pub mass_properties: MassProperties,
    pub inertia_world: Mat3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompoundMassProperties {
    pub mass: f32,
    pub center_of_mass: Vec3,
    pub inertia_world: Mat3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompoundBody {
    pub parts: Vec<AssemblyPart>,
    pub properties: CompoundMassProperties,
    pub reference_orientation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompoundPartRuntime {
    pub body: BodyHandle,
    pub pose: Pose,
    pub previous_pose: Pose,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompoundError {
    Empty,
    NonFinite,
    ImmovablePart,
    SingularInertia,
    InvalidHistory,
}

impl CompoundBody {
    pub fn from_states(states: &[BodyState]) -> Result<Self, CompoundError> {
        if states.is_empty() {
            return Err(CompoundError::Empty);
        }
        if states.iter().any(|state| {
            !state.pose.position.is_finite()
                || !state.pose.orientation.is_finite()
                || !state.pose.orientation.is_normalized()
                || !state.linear_velocity.is_finite()
                || !state.angular_velocity.is_finite()
                || !state.mass_properties.inv_mass.is_finite()
                || !state.mass_properties.inv_inertia.is_finite()
        }) {
            return Err(CompoundError::NonFinite);
        }
        if states
            .iter()
            .any(|state| state.mass_properties.inv_mass <= 0.0)
        {
            return Err(CompoundError::ImmovablePart);
        }
        let masses: Vec<f32> = states
            .iter()
            .map(|state| 1.0 / state.mass_properties.inv_mass)
            .collect();
        let total_mass: f32 = masses.iter().sum();
        if !total_mass.is_finite() || total_mass <= EPS {
            return Err(CompoundError::SingularInertia);
        }
        let center = states
            .iter()
            .zip(&masses)
            .fold(Vec3::ZERO, |sum, (state, mass)| {
                sum + state.pose.position * *mass
            })
            / total_mass;
        let momentum = states
            .iter()
            .zip(&masses)
            .fold(Vec3::ZERO, |sum, (state, mass)| {
                sum + state.linear_velocity * *mass
            });
        let mut inertia = Mat3::ZERO;
        let mut angular_momentum = Vec3::ZERO;
        let mut parts = Vec::with_capacity(states.len());
        for (state, mass) in states.iter().zip(&masses) {
            let offset = state.pose.position - center;
            let i_world = inertia_world(state.pose.orientation, state.mass_properties);
            let parallel = Mat3::from_diagonal(Vec3::splat(offset.length_squared()))
                - Mat3::from_cols(offset * offset.x, offset * offset.y, offset * offset.z);
            inertia += i_world + parallel * *mass;
            angular_momentum +=
                i_world * state.angular_velocity + offset.cross(state.linear_velocity * *mass);
            parts.push(AssemblyPart {
                body: state.handle,
                relative_pose: Pose {
                    position: offset,
                    orientation: state.pose.orientation,
                },
                mass: *mass,
                mass_properties: state.mass_properties,
                inertia_world: i_world,
            });
        }
        if !center.is_finite()
            || !momentum.is_finite()
            || !inertia.is_finite()
            || !angular_momentum.is_finite()
            || !inertia.determinant().is_finite()
            || inertia.determinant() <= EPS
            || states
                .iter()
                .any(|state| state.mass_properties.inv_inertia.min_element() <= 0.0)
        {
            return Err(CompoundError::SingularInertia);
        }
        let angular_velocity = inertia.inverse() * angular_momentum;
        if !angular_velocity.is_finite() {
            return Err(CompoundError::SingularInertia);
        }
        let (reference_orientation, _) = principal_axes(inertia);
        for part in &mut parts {
            part.relative_pose.position =
                reference_orientation.inverse() * part.relative_pose.position;
            part.relative_pose.orientation =
                (reference_orientation.inverse() * part.relative_pose.orientation).normalize();
        }
        Ok(Self {
            parts,
            properties: CompoundMassProperties {
                mass: total_mass,
                center_of_mass: center,
                inertia_world: inertia,
            },
            reference_orientation,
            linear_velocity: momentum / total_mass,
            angular_velocity,
        })
    }

    #[must_use]
    pub fn split_velocities(
        &self,
        linear_velocity: Vec3,
        angular_velocity: Vec3,
    ) -> Vec<(BodyHandle, Vec3, Vec3)> {
        self.split_velocities_at(
            self.reference_orientation,
            linear_velocity,
            angular_velocity,
        )
    }

    #[must_use]
    pub fn split_velocities_at(
        &self,
        proxy_orientation: Quat,
        linear_velocity: Vec3,
        angular_velocity: Vec3,
    ) -> Vec<(BodyHandle, Vec3, Vec3)> {
        self.parts
            .iter()
            .map(|part| {
                let world_offset = proxy_orientation * part.relative_pose.position;
                (
                    part.body,
                    linear_velocity + angular_velocity.cross(world_offset),
                    angular_velocity,
                )
            })
            .collect()
    }

    #[must_use]
    pub fn part_pose(&self, proxy_pose: Pose, part: &AssemblyPart) -> Pose {
        Pose {
            position: proxy_pose.position + proxy_pose.orientation * part.relative_pose.position,
            orientation: (proxy_pose.orientation * part.relative_pose.orientation).normalize(),
        }
    }

    /// Projects member histories onto the rigid proxy frame. For an already
    /// rigid assembly this exactly recovers its previous center and basis;
    /// otherwise orientations are mass-weighted in a common quaternion hemisphere.
    pub fn proxy_previous_pose(&self, snapshots: &[BodySnapshot]) -> Result<Pose, CompoundError> {
        if snapshots.len() != self.parts.len() || self.parts.is_empty() {
            return Err(CompoundError::InvalidHistory);
        }
        let mut center = Vec3::ZERO;
        let mut orientation_sum = glam::Vec4::ZERO;
        let mut reference = None;
        for (part, snapshot) in self.parts.iter().zip(snapshots) {
            let previous = snapshot.previous_pose;
            if snapshot.handle != part.body
                || !previous.position.is_finite()
                || !previous.orientation.is_finite()
                || !previous.orientation.is_normalized()
            {
                return Err(CompoundError::InvalidHistory);
            }
            center += previous.position * (part.mass / self.properties.mass);
            let orientation =
                (previous.orientation * part.relative_pose.orientation.inverse()).normalize();
            let reference = *reference.get_or_insert(orientation);
            let sign = if reference.dot(orientation) < 0.0 {
                -1.0
            } else {
                1.0
            };
            orientation_sum += glam::Vec4::from_array(orientation.to_array())
                * (sign * part.mass / self.properties.mass);
        }
        if !center.is_finite()
            || !orientation_sum.is_finite()
            || orientation_sum.length_squared() <= EPS * EPS
        {
            return Err(CompoundError::InvalidHistory);
        }
        Ok(Pose {
            position: center,
            orientation: Quat::from_array(orientation_sum.normalize().to_array()),
        })
    }

    #[must_use]
    pub fn reconstruct_runtime(
        &self,
        proxy_pose: Pose,
        previous_proxy_pose: Pose,
        linear_velocity: Vec3,
        angular_velocity: Vec3,
    ) -> Vec<CompoundPartRuntime> {
        let velocities =
            self.split_velocities_at(proxy_pose.orientation, linear_velocity, angular_velocity);
        self.parts
            .iter()
            .zip(velocities)
            .map(
                |(part, (body, linear_velocity, angular_velocity))| CompoundPartRuntime {
                    body,
                    pose: self.part_pose(proxy_pose, part),
                    previous_pose: self.part_pose(previous_proxy_pose, part),
                    linear_velocity,
                    angular_velocity,
                },
            )
            .collect()
    }

    #[must_use]
    pub fn conserved_momentum(&self) -> (Vec3, Vec3) {
        let linear = self.linear_velocity * self.properties.mass;
        let angular = self.properties.inertia_world * self.angular_velocity;
        (linear, angular)
    }

    #[must_use]
    pub fn principal_mass_properties(&self) -> (Quat, Vec3) {
        let (basis, eigenvalues) = principal_axes(self.properties.inertia_world);
        let reciprocal = |value: f32| {
            if value.is_finite() && value > EPS {
                1.0 / value
            } else {
                0.0
            }
        };
        (
            basis,
            Vec3::new(
                reciprocal(eigenvalues.x),
                reciprocal(eigenvalues.y),
                reciprocal(eigenvalues.z),
            ),
        )
    }
}

fn principal_axes(input: Mat3) -> (Quat, Vec3) {
    let mut a = [[0.0_f32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            a[row][col] = input.col(col)[row];
        }
    }
    let mut v = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        v[i][i] = 1.0;
    }
    for _ in 0..24 {
        let (mut p, mut q) = (0, 1);
        for (candidate_p, candidate_q) in [(0, 1), (0, 2), (1, 2)] {
            if a[candidate_p][candidate_q].abs() > a[p][q].abs() {
                p = candidate_p;
                q = candidate_q;
            }
        }
        if a[p][q].abs() <= 1.0e-6 {
            break;
        }
        let theta = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (c, s) = (theta.cos(), theta.sin());
        for k in 0..3 {
            let (apk, aqk) = (a[p][k], a[q][k]);
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for k in 0..3 {
            let (akp, akq) = (a[k][p], a[k][q]);
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..3 {
            let (vkp, vkq) = (v[k][p], v[k][q]);
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }
    let mut eigen = [a[0][0], a[1][1], a[2][2]];
    for i in 0..2 {
        for j in (i + 1)..3 {
            if eigen[j] < eigen[i] {
                eigen.swap(i, j);
                for row in &mut v {
                    row.swap(i, j);
                }
            }
        }
    }
    let mut basis = Mat3::from_cols(
        Vec3::new(v[0][0], v[1][0], v[2][0]),
        Vec3::new(v[0][1], v[1][1], v[2][1]),
        Vec3::new(v[0][2], v[1][2], v[2][2]),
    );
    if basis.determinant() < 0.0 {
        basis.x_axis = -basis.x_axis;
    }
    (Quat::from_mat3(&basis).normalize(), Vec3::from_array(eigen))
}

/// Extension boundary for external body stores. `PhysicsWorld` provides the
/// concrete implementation; this trait remains useful for backends that own
/// a compatible store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompoundAdapterError {
    Rejected,
    InvalidState,
}

pub trait CompoundWorldAdapter {
    fn merge_component(
        &mut self,
        component: &[BodyHandle],
        compound: &CompoundBody,
    ) -> Result<BodyHandle, CompoundAdapterError>;
    fn split_component(
        &mut self,
        compound: &CompoundBody,
        linear_velocity: Vec3,
        angular_velocity: Vec3,
    ) -> Result<Vec<BodyHandle>, CompoundAdapterError>;
}

use glam::{DQuat, DVec3};

/// A compliant world-space pose drive, sampled without mutating body state.
/// Compliance must be finite and nonnegative. Magnitude caps may be zero or
/// positive infinity (unbounded); negative and NaN caps are invalid. A zero
/// speed cap requests rest (so it can still brake); a zero force/torque cap
/// disables that wrench component. Even unbounded drives return finite loads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrabDrive {
    pub target: BodyHandle,
    pub target_pose: Pose,
    pub linear_compliance: f32,
    pub angular_compliance: f32,
    pub max_force: f32,
    pub max_torque: f32,
    pub max_linear_speed: f32,
    pub max_angular_speed: f32,
}

impl GrabDrive {
    #[must_use]
    pub fn soft(target: BodyHandle, target_pose: Pose) -> Self {
        Self {
            target,
            target_pose,
            linear_compliance: 1.0e-3,
            angular_compliance: 1.0e-3,
            max_force: 1.0e4,
            max_torque: 1.0e4,
            max_linear_speed: 50.0,
            max_angular_speed: 30.0,
        }
    }
    /// Returns `None` for mismatched bodies, malformed/nonunit poses, invalid
    /// parameters, non-finite velocities, or invalid inverse mass/inertia.
    /// Zero inverse inertia locks that principal axis. Computation uses f64
    /// intermediates before magnitude limiting; an unbounded result outside
    /// finite f32 representation is rejected rather than silently saturated.
    pub fn sample(&self, state: &BodyState, dt: f32) -> Option<ForceTorque> {
        if state.handle != self.target
            || !dt.is_finite()
            || dt <= 0.0
            || !grab_valid_pose(self.target_pose)
            || !grab_valid_pose(state.pose)
            || !state.linear_velocity.is_finite()
            || !state.angular_velocity.is_finite()
            || !state.mass_properties.inv_mass.is_finite()
            || state.mass_properties.inv_mass <= 0.0
            || !state.mass_properties.inv_inertia.is_finite()
            || state.mass_properties.inv_inertia.min_element() < 0.0
            || [self.linear_compliance, self.angular_compliance]
                .into_iter()
                .any(|value| !value.is_finite() || value < 0.0)
            || [
                self.max_force,
                self.max_torque,
                self.max_linear_speed,
                self.max_angular_speed,
            ]
            .into_iter()
            .any(|value| value.is_nan() || value < 0.0)
        {
            return None;
        }
        // All authored inputs are f32: even their most extreme finite values
        // and positive subnormal dt/mass fit these f64 intermediate products.
        let dt = f64::from(dt);
        let dt_squared = dt * dt;
        let desired_linear = grab_limit(
            (self.target_pose.position.as_dvec3() - state.pose.position.as_dvec3()) / dt,
            self.max_linear_speed,
        );
        let linear_gain = dt_squared / (dt_squared + f64::from(self.linear_compliance));
        let force = (desired_linear - state.linear_velocity.as_dvec3()) * linear_gain
            / dt
            / f64::from(state.mass_properties.inv_mass);

        let orientation = grab_quat(state.pose.orientation);
        let mut delta =
            (grab_quat(self.target_pose.orientation) * orientation.conjugate()).normalize();
        // q and -q encode the same orientation. Canonicalize the hemisphere,
        // including a deterministic tie at exactly pi, before taking the log.
        let first_nonzero = [delta.w, delta.x, delta.y, delta.z]
            .into_iter()
            .find(|value| *value != 0.0)
            .unwrap_or(1.0);
        if first_nonzero < 0.0 {
            delta = -delta;
        }
        let imaginary = DVec3::new(delta.x, delta.y, delta.z);
        let sine = imaginary.length();
        let rotation = if sine == 0.0 {
            DVec3::ZERO
        } else {
            imaginary * (2.0 * sine.atan2(delta.w) / sine)
        };
        let desired_angular = grab_limit(rotation / dt, self.max_angular_speed);
        let angular_gain = dt_squared / (dt_squared + f64::from(self.angular_compliance));
        let acceleration =
            (desired_angular - state.angular_velocity.as_dvec3()) * angular_gain / dt;
        let local_acceleration = orientation.conjugate() * acceleration;
        let inverse = state.mass_properties.inv_inertia.as_dvec3();
        let local_torque = DVec3::new(
            if inverse.x > 0.0 {
                local_acceleration.x / inverse.x
            } else {
                0.0
            },
            if inverse.y > 0.0 {
                local_acceleration.y / inverse.y
            } else {
                0.0
            },
            if inverse.z > 0.0 {
                local_acceleration.z / inverse.z
            } else {
                0.0
            },
        );
        let torque = orientation * local_torque;
        Some(ForceTorque {
            force: grab_wrench_component(force, self.max_force)?,
            torque: grab_wrench_component(torque, self.max_torque)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnapResult {
    pub pose: Pose,
    pub score: f32,
}

/// Aligns the local anchor normal opposite the target normal. Poses must be
/// finite with unit orientation; normal magnitudes are irrelevant but must
/// be finite and nonzero. Returns `None` if the solved position is not finite
/// in f32. No malformed input is silently replaced with an identity pose.
#[must_use]
pub fn solve_snap(
    current: Pose,
    local_position: Vec3,
    local_normal: Vec3,
    target_position: Vec3,
    target_normal: Vec3,
) -> Option<SnapResult> {
    if !grab_valid_pose(current)
        || !local_position.is_finite()
        || !target_position.is_finite()
        || !local_normal.is_finite()
        || !target_normal.is_finite()
        || local_normal == Vec3::ZERO
        || target_normal == Vec3::ZERO
    {
        return None;
    }
    let current_orientation = grab_quat(current.orientation);
    let world_normal = (current_orientation * local_normal.as_dvec3().normalize()).normalize();
    let desired_normal = -target_normal.as_dvec3().normalize();
    let cross = world_normal.cross(desired_normal);
    let sine = cross.length();
    let cosine = world_normal.dot(desired_normal).clamp(-1.0, 1.0);
    let align = if sine > 0.0 {
        DQuat::from_axis_angle(cross / sine, sine.atan2(cosine))
    } else if cosine < 0.0 {
        DQuat::from_rotation_arc(world_normal, desired_normal)
    } else {
        DQuat::IDENTITY
    };
    let orientation = (align * current_orientation).normalize();
    let position = (target_position.as_dvec3() - orientation * local_position.as_dvec3()).as_vec3();
    if !position.is_finite() {
        return None;
    }
    Some(SnapResult {
        pose: Pose {
            position,
            orientation: Quat::from_xyzw(
                orientation.x as f32,
                orientation.y as f32,
                orientation.z as f32,
                orientation.w as f32,
            )
            .normalize(),
        },
        score: cosine as f32,
    })
}

fn grab_valid_pose(pose: Pose) -> bool {
    pose.position.is_finite() && pose.orientation.is_finite() && pose.orientation.is_normalized()
}

fn grab_quat(value: Quat) -> DQuat {
    DQuat::from_xyzw(
        f64::from(value.x),
        f64::from(value.y),
        f64::from(value.z),
        f64::from(value.w),
    )
    .normalize()
}

fn grab_limit(value: DVec3, max: f32) -> DVec3 {
    // Squaring an extreme but finite f64 force can itself overflow. Compare
    // lengths in a scaled frame instead, preserving direction when limiting.
    let scale = value.abs().max_element();
    if max == 0.0 {
        DVec3::ZERO
    } else if scale == 0.0 || !value.is_finite() {
        value
    } else {
        let scaled = value / scale;
        let length = scaled.length();
        if scale > f64::from(max) / length {
            scaled * (f64::from(max) / length)
        } else {
            value
        }
    }
}

fn grab_wrench_component(value: DVec3, max: f32) -> Option<Vec3> {
    let value = grab_limit(value, max);
    if !value.is_finite() || value.abs().max_element() > f64::from(f32::MAX) {
        return None;
    }
    let mut result = value.as_vec3();
    if !result.is_finite() {
        return None;
    }
    // Component rounding must not push a finite cap's magnitude above its
    // bound. Leave one f32 relative ULP of room when correction is needed.
    let rounded_length = result.as_dvec3().length();
    if max.is_finite() && rounded_length > f64::from(max) {
        result = (result.as_dvec3()
            * (f64::from(max) * (1.0 - f64::from(f32::EPSILON)) / rounded_length))
            .as_vec3();
        // Subnormal rounding can still exceed a cap smaller than one ULP.
        if result.as_dvec3().length() > f64::from(max) {
            result = Vec3::ZERO;
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::{JointAnchor, JointDesc};

    fn assert_vec(actual: Vec3, expected: Vec3) {
        assert!(
            (actual - expected).length() < 2.0e-4,
            "{actual:?} != {expected:?}"
        );
    }

    fn assert_pose(actual: Pose, expected: Pose) {
        assert_vec(actual.position, expected.position);
        assert!(actual.orientation.dot(expected.orientation).abs() > 1.0 - 2.0e-5);
    }

    fn joint(joints: &mut JointStorage) -> JointHandle {
        joints.insert(JointDesc::fixed(
            JointAnchor::at_center(BodyHandle::INVALID),
            JointAnchor::at_center(BodyHandle::INVALID),
        ))
    }

    fn handles(count: usize) -> Vec<BodyHandle> {
        let mut storage = crate::state::storage::BodyStorage::new();
        (0..count)
            .map(|_| storage.insert(BodyDesc::dynamic_at(Vec3::ZERO)))
            .collect()
    }

    #[test]
    fn joint_load_accumulator_reduces_workers_in_canonical_order() {
        let mut joints = JointStorage::new();
        let a = joint(&mut joints);
        let b = joint(&mut joints);
        let values = [1.0e10, 3.0, -1.0e10];
        let mut forward = JointLoadAccumulator::new();
        let mut reverse = JointLoadAccumulator::new();
        for value in values {
            forward
                .record_impulse(a, Vec3::X * value, Vec3::ZERO, 0.1)
                .unwrap();
        }
        for value in values.into_iter().rev() {
            let mut worker = JointLoadAccumulator::new();
            worker
                .record_impulse(a, Vec3::X * value, Vec3::ZERO, 0.1)
                .unwrap();
            reverse.merge(&worker).unwrap();
        }
        forward.record_impulse(b, Vec3::ZERO, Vec3::Y, 0.1).unwrap();
        reverse.record_impulse(b, Vec3::ZERO, Vec3::Y, 0.1).unwrap();
        let loads = forward.loads().unwrap();
        assert_eq!(loads, reverse.loads().unwrap());
        assert_eq!(loads[0].joint, a);
        assert_eq!(loads[0].load.linear_impulse, Vec3::X * 3.0);
        assert_eq!(loads[1].joint, b);
    }

    #[test]
    fn projection_measurement_breaks_once_and_consumes_at_boundary() {
        let mut joints = JointStorage::new();
        let handle = joint(&mut joints);
        let mut policies = BreakableJointSet::default();
        policies
            .register(handle, BreakLimit::new(5.0, 20.0))
            .unwrap();
        let mut measured = JointLoadAccumulator::new();
        measured
            .record_projection(handle, Vec3::X * 0.06, Vec3::Z * 0.02, 0.1)
            .unwrap();
        assert!(joints.get(handle).unwrap().enabled);
        let events = measured.submit(0.1, &mut policies, &mut joints).unwrap();
        assert_eq!(events.len(), 1);
        assert!((events[0].force - 6.0).abs() < 1.0e-5);
        assert!((events[0].torque - 2.0).abs() < 1.0e-5);
        assert!(!joints.get(handle).unwrap().enabled);
        assert!(measured.is_empty());
        assert!(measured
            .submit(0.1, &mut policies, &mut joints)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn failed_load_record_merge_and_submission_are_transactional() {
        let mut joints = JointStorage::new();
        let handle = joint(&mut joints);
        let mut measured = JointLoadAccumulator::new();
        measured
            .record_impulse(handle, Vec3::X, Vec3::ZERO, 1.0e-6)
            .unwrap();
        let before = measured.clone();
        assert_eq!(
            measured.record_impulse(handle, Vec3::X, Vec3::ZERO, 2.0e-6),
            Err(BoundaryLoadError::MismatchedDt)
        );
        assert_eq!(
            measured.record_impulse(handle, Vec3::splat(f32::NAN), Vec3::ZERO, 1.0e-6),
            Err(BoundaryLoadError::NonFiniteLoad)
        );
        let mut worker = JointLoadAccumulator::new();
        worker
            .record_impulse(handle, Vec3::X, Vec3::ZERO, 0.1)
            .unwrap();
        assert_eq!(
            measured.merge(&worker),
            Err(BoundaryLoadError::MismatchedDt)
        );
        let mut policies = BreakableJointSet::default();
        let policies_before = policies.clone();
        assert_eq!(
            measured.submit(1.0e-6, &mut policies, &mut joints),
            Err(BoundaryLoadError::UnknownJoint)
        );
        assert_eq!(measured, before);
        assert_eq!(policies, policies_before);
        assert!(joints.get(handle).unwrap().enabled);
    }

    #[test]
    fn boundary_rejects_overflow_and_stale_handles_without_policy_mutation() {
        let mut joints = JointStorage::new();
        let handle = joint(&mut joints);
        let mut policies = BreakableJointSet::default();
        policies
            .register(handle, BreakLimit::new(10.0, 10.0))
            .unwrap();
        let before = policies.clone();
        let load = SolverJointLoad {
            joint: handle,
            load: JointLoad {
                linear_impulse: Vec3::splat(f32::MAX),
                angular_impulse: Vec3::ZERO,
                substep_dt: 0.1,
            },
        };
        assert_eq!(
            policies.submit_solver_boundary(0.1, &[load, load], &mut joints),
            Err(BoundaryLoadError::NonFiniteLoad)
        );
        assert_eq!(policies, before);
        joints.remove(handle);
        assert_eq!(
            policies.submit_solver_boundary(0.1, &[load], &mut joints),
            Err(BoundaryLoadError::UnknownJoint)
        );
        assert_eq!(policies, before);
    }

    #[test]
    fn exhausted_connection_ids_preserve_graph_and_policy() {
        let bodies = handles(2);
        let mut joints = JointStorage::new();
        let handle = joint(&mut joints);
        let kind = joints.get(handle).unwrap().kind;
        let mut graph = AssemblyGraph::new();
        graph.next_id = u64::MAX;
        let mut policies = BreakableJointSet::default();
        let before = graph.clone();
        let policies_before = policies.clone();
        let limit = BreakLimit::new(10.0, 10.0);
        assert_eq!(
            graph.connect_with_policy(
                &mut policies,
                bodies[0],
                bodies[1],
                Some(handle),
                kind,
                Some(limit)
            ),
            Err(AssemblyError::IdExhausted)
        );
        assert_eq!(graph, before);
        assert_eq!(policies, policies_before);
        assert_eq!(
            graph.connect(bodies[0], bodies[1], None, kind, None),
            Err(AssemblyError::IdExhausted)
        );
        assert_eq!(graph, before);
    }

    #[test]
    fn connection_and_policy_registration_commit_together() {
        let bodies = handles(3);
        let mut joints = JointStorage::new();
        let handle = joint(&mut joints);
        let kind = joints.get(handle).unwrap().kind;
        let mut graph = AssemblyGraph::new();
        let mut policies = BreakableJointSet::default();
        let limit = BreakLimit::new(10.0, 10.0);
        let id = graph
            .connect_with_policy(
                &mut policies,
                bodies[0],
                bodies[1],
                Some(handle),
                kind,
                Some(limit),
            )
            .unwrap();
        assert!(policies.contains(handle));
        let graph_before = graph.clone();
        let policies_before = policies.clone();
        assert_eq!(
            graph.connect_with_policy(
                &mut policies,
                bodies[1],
                bodies[2],
                Some(handle),
                kind,
                Some(limit)
            ),
            Err(AssemblyError::DuplicatePolicy)
        );
        assert_eq!(graph, graph_before);
        assert_eq!(policies, policies_before);
        assert!(policies.register(handle, limit).is_err());
        assert_eq!(policies, policies_before);
        assert_eq!(graph.connections().next().unwrap().id, id);
        assert_eq!(
            graph.connect_with_policy(
                &mut policies,
                bodies[1],
                bodies[2],
                None,
                kind,
                Some(BreakLimit::new(-1.0, 1.0))
            ),
            Err(AssemblyError::InvalidLimit)
        );
        assert_eq!(graph, graph_before);
    }

    #[test]
    fn restore_component_validates_every_record_before_committing() {
        let bodies = handles(5);
        let (a, b, proxy, external, missing) =
            (bodies[0], bodies[1], bodies[2], bodies[3], bodies[4]);
        let mut joints = JointStorage::new();
        let handle = joint(&mut joints);
        let kind = joints.get(handle).unwrap().kind;
        let mut graph = AssemblyGraph::new();
        graph.connect(a, b, None, kind, None).unwrap();
        graph.connect(b, external, None, kind, None).unwrap();
        let original = graph.clone();
        let saved = graph.collapse_component(&[a, b], proxy).unwrap();
        let collapsed = graph.clone();
        let mut invalid = saved.clone();
        invalid[1].b = missing;
        assert_eq!(
            graph.restore_component(proxy, &[a, b], &invalid),
            Err(AssemblyError::MissingBody)
        );
        assert_eq!(graph, collapsed);
        assert_eq!(
            graph.restore_component(proxy, &[a, a], &saved),
            Err(AssemblyError::DuplicateConnection)
        );
        assert_eq!(graph, collapsed);
        let mut invalid = saved.clone();
        invalid[1].id = invalid[0].id;
        assert_eq!(
            graph.restore_component(proxy, &[a, b], &invalid),
            Err(AssemblyError::DuplicateConnection)
        );
        assert_eq!(graph, collapsed);
        let mut invalid = saved.clone();
        invalid[1].a = a;
        invalid[1].b = b;
        assert_eq!(
            graph.restore_component(proxy, &[a, b], &invalid),
            Err(AssemblyError::DuplicateConnection)
        );
        assert_eq!(graph, collapsed);
        graph.restore_component(proxy, &[a, b], &saved).unwrap();
        assert_eq!(graph, original);
    }

    #[test]
    fn runtime_reconstruction_uses_current_and_previous_proxy_frames() {
        let bodies = handles(2);
        let orientation = Quat::from_euler(glam::EulerRot::XYZ, 0.4, -0.7, 0.9);
        let mut states = [
            state(bodies[0], Vec3::new(-1.0, 2.0, 0.2), Vec3::ZERO),
            state(bodies[1], Vec3::new(2.0, -1.0, 1.0), Vec3::ZERO),
        ];
        for state in &mut states {
            state.pose.orientation = orientation;
            state.mass_properties.inv_inertia = Vec3::new(0.5, 0.25, 1.0);
        }
        let compound = CompoundBody::from_states(&states).unwrap();
        let reference = Pose {
            position: compound.properties.center_of_mass,
            orientation: compound.reference_orientation,
        };
        for (part, state) in compound.parts.iter().zip(&states) {
            assert_pose(compound.part_pose(reference, part), state.pose);
        }
        let now = Pose {
            position: Vec3::new(4.0, -2.0, 1.0),
            orientation: Quat::from_rotation_y(1.3) * reference.orientation,
        };
        let previous = Pose {
            position: now.position - Vec3::X * 0.1,
            orientation: Quat::from_rotation_z(-0.2) * now.orientation,
        };
        let linear = Vec3::new(1.0, 2.0, -1.0);
        let angular = Vec3::new(0.3, -0.4, 2.0);
        let runtime = compound.reconstruct_runtime(now, previous, linear, angular);
        let mut momentum = Vec3::ZERO;
        let mut angular_momentum = Vec3::ZERO;
        for (part, restored) in compound.parts.iter().zip(runtime) {
            let offset = now.orientation * part.relative_pose.position;
            assert_vec(restored.pose.position, now.position + offset);
            assert_vec(restored.linear_velocity, linear + angular.cross(offset));
            assert_vec(restored.angular_velocity, angular);
            assert_pose(restored.previous_pose, compound.part_pose(previous, part));
            momentum += restored.linear_velocity * part.mass;
            angular_momentum += inertia_world(restored.pose.orientation, part.mass_properties)
                * angular
                + offset.cross(restored.linear_velocity * part.mass);
        }
        assert_vec(momentum, linear * compound.properties.mass);
        let rotation = Mat3::from_quat(now.orientation * reference.orientation.inverse());
        assert_vec(
            angular_momentum,
            rotation * compound.properties.inertia_world * rotation.transpose() * angular,
        );
        let snapshots: Vec<BodySnapshot> = compound
            .parts
            .iter()
            .zip(&states)
            .map(|(part, state)| BodySnapshot {
                handle: state.handle,
                desc: BodyDesc::dynamic_at(state.pose.position),
                previous_pose: compound.part_pose(previous, part),
                sleeping: false,
                sleep_timer: 0.0,
            })
            .collect();
        assert_pose(compound.proxy_previous_pose(&snapshots).unwrap(), previous);
        assert_eq!(
            compound.proxy_previous_pose(&[]),
            Err(CompoundError::InvalidHistory)
        );
    }

    fn state(handle: BodyHandle, position: Vec3, velocity: Vec3) -> BodyState {
        BodyState {
            handle,
            pose: Pose {
                position,
                orientation: Quat::IDENTITY,
            },
            linear_velocity: velocity,
            angular_velocity: Vec3::ZERO,
            mass_properties: MassProperties {
                inv_mass: 1.0,
                inv_inertia: Vec3::ONE,
            },
        }
    }

    #[test]
    fn graph_split_is_deterministic() {
        let mut storage = crate::state::storage::BodyStorage::new();
        let a = storage.insert(crate::state::body::BodyDesc::dynamic_at(Vec3::ZERO));
        let b = storage.insert(crate::state::body::BodyDesc::dynamic_at(Vec3::X));
        let mut graph = AssemblyGraph::new();
        graph.add_body(a);
        graph.add_body(b);
        assert_eq!(graph.connected_components().len(), 2);
    }

    #[test]
    fn compound_conserves_linear_momentum() {
        let states = [
            state(BodyHandle::INVALID, Vec3::ZERO, Vec3::X),
            state(BodyHandle::INVALID, Vec3::X, Vec3::NEG_X),
        ];
        let compound = CompoundBody::from_states(&states).unwrap();
        let (momentum, _) = compound.conserved_momentum();
        assert!(momentum.length() < 1e-6);
        assert_eq!(compound.parts.len(), 2);
    }

    #[test]
    fn grab_drive_is_force_limited() {
        let body = BodyHandle::INVALID;
        let drive = GrabDrive {
            target: body,
            target_pose: Pose {
                position: Vec3::X * 100.0,
                orientation: Quat::IDENTITY,
            },
            max_force: 2.0,
            max_torque: 3.0,
            ..GrabDrive::soft(body, Pose::identity())
        };
        let force = drive
            .sample(&state(body, Vec3::ZERO, Vec3::ZERO), 0.1)
            .unwrap();
        assert!((force.force.length() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn break_policy_disables_joint_at_boundary() {
        let mut joints = JointStorage::new();
        let desc = JointDesc::fixed(
            JointAnchor::at_center(BodyHandle::INVALID),
            JointAnchor::at_center(BodyHandle::INVALID),
        );
        let handle = joints.insert(desc);
        let mut policies = BreakableJointSet::default();
        policies
            .register(handle, BreakLimit::new(1.0, 1.0))
            .unwrap();
        policies.begin_substep();
        policies.record_load(
            handle,
            JointLoad {
                linear_impulse: Vec3::X * 2.0,
                angular_impulse: Vec3::ZERO,
                substep_dt: 1.0,
            },
        );
        let events = policies.evaluate(1.0);
        policies.disable_broken(&mut joints, &events);
        assert_eq!(events.len(), 1);
        assert!(!joints.get(handle).unwrap().enabled);
    }

    #[test]
    fn snap_aligns_anchor_and_reverses_normal() {
        let result = solve_snap(Pose::identity(), Vec3::ZERO, Vec3::Y, Vec3::X, Vec3::Y).unwrap();
        assert_eq!(result.pose.position, Vec3::X);
        assert!((result.score - -1.0).abs() < 1e-5);
    }
}
