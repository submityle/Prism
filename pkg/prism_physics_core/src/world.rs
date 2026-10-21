//! The top-level physics world container.
//!
//! [`PhysicsWorld`] bundles the body storage, the shared collider registry, and
//! the world configuration. It is deliberately backend-agnostic: solvers and
//! backends operate on it but are not owned by it.

use crate::aero::{AeroBodyConfig, AeroBodyState, AeroWrench, WindField};
use crate::collider::compound::{CompoundLeafPolicy, PreparedCompound, MAX_COMPOUND_CHILDREN};
use crate::collider::{CompoundChild, ShapeRegistry};
use crate::collision_filter::{
    CollisionExclusionError, CollisionExclusionGroupId, CollisionExclusions,
};
use crate::config::WorldConfig;
use crate::dynamics::assembly::{
    AssemblyError, AssemblyGraph, BodySnapshot, BodyState, BoundaryLoadError, BreakEvent,
    BreakLimit, BreakableJointSet, CompoundBody, CompoundError, Connection, ConnectionId,
    GrabDrive, Pose, SolverJointLoad,
};
use crate::dynamics::field::{
    FieldBatchError, FieldBatchSampler, FieldBody, FieldRegistry, FieldTargetMask,
};
use crate::dynamics::force::{ForceAccumulator, ForceError, PreparedForceRemap};
use crate::dynamics::Integrator;
use crate::events::{ContactEventTracker, PhysicsEvent};
use crate::joint::{Joint, JointDesc, JointHandle, JointStorage};
use crate::math::scalar::Real;
use crate::math::transform::Isometry;
use crate::pipeline::detect_contacts;
use crate::reduced::{
    Articulation, ArticulationColliderDesc, ArticulationColliderError, ArticulationColliderId,
    ArticulationColliderSet, ArticulationDegradationEvent, ArticulationHandle, ArticulationStorage,
    ArticulationWorldStepReport, LinkContactImpulse,
};
use crate::soft::rigid_coupling::{
    couple_cloth_to_rigid, ClothRigidCouplingConfig, CouplingReport,
};
use crate::soft::solver::{SoftContacts, SoftSolver};
use crate::soft::topology::TopologyTearUpdate;
use crate::soft::{SoftBody, SoftBodyHandle, SoftBodyStorage, TopologyTearEvent};
use crate::state::body::{BodyDesc, SolverMassProperties};
use crate::state::handle::BodyHandle;
use crate::state::storage::{BodyReplacementPlan, BodyStorage};
use glam::Vec3;
use std::collections::BTreeMap;

mod assembly_break;
pub use assembly_break::{AssemblyMemberRemap, AssemblySplitReport, JointBoundaryFailure};

#[derive(Clone, Debug)]
struct AssemblyJointSnapshot {
    connection: Connection,
    joint: Option<Joint>,
}

#[derive(Clone, Debug)]
struct AssemblyMergeState {
    proxy: BodyHandle,
    component: Vec<BodySnapshot>,
    compound: CompoundBody,
    connections: Vec<AssemblyJointSnapshot>,
}

// These plans never escape the public operation's exclusive world borrow.
// Every recoverable failure happens while preparing; commit has no Result.
struct PreparedAssemblyMerge {
    allocation: BodyReplacementPlan,
    descriptor: BodyDesc,
    collider: Option<PreparedCompound>,
    gravity: Option<Vec3>,
    graph: AssemblyGraph,
    forces: PreparedForceRemap,
    joints: Vec<(JointHandle, Joint)>,
    recovery: AssemblyMergeState,
}

struct PreparedAssemblySplit {
    allocation: BodyReplacementPlan,
    descriptors: Vec<BodyDesc>,
    parts: Vec<PreparedAssemblySplitPart>,
    graph: AssemblyGraph,
}

struct PreparedAssemblySplitPart {
    proxy: BodyHandle,
    range: std::ops::Range<usize>,
    proxy_snapshot: BodySnapshot,
    gravity: Option<Vec3>,
    forces: PreparedForceRemap,
    joints: Vec<(JointHandle, Joint)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssemblyWorldError {
    InvalidBody,
    NotConnectedComponent,
    DuplicateBody,
    ImmovableBody,
    UnsupportedCollider,
    Compound(CompoundError),
    Graph(AssemblyError),
    ActiveJointMissing,
    InvalidProxy,
    NotMerged,
    /// Collapsing or replacing a registered body would lose its exact policy.
    CollisionExclusionConflict,
    /// One proxy cannot preserve distinct per-member gravity policies.
    /// World-following and explicit overrides remain distinct even when their
    /// current accelerations happen to match.
    GravityPolicyConflict,
    /// A member's damping coefficient is negative or non-finite.
    InvalidDamping,
    /// Pending wrench remapping failed, including finite-input overflow.
    Force(ForceError),
}

/// Error returned when aerodynamic data cannot be registered for a body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AeroRegistrationError {
    /// The generational body handle is not live in this world.
    InvalidBody,
}

/// A registered aerodynamic configuration and its body handle.
#[derive(Clone, Debug, PartialEq)]
pub struct AeroBodyRegistration {
    /// Live rigid body receiving the aerodynamic contribution.
    pub body: BodyHandle,
    /// Authoring data evaluated at each fixed sub-step.
    pub config: AeroBodyConfig,
}

/// The result of evaluating one registered body for a fixed sub-step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AeroBodySample {
    /// Live rigid body sampled by the pass.
    pub body: BodyHandle,
    /// Sanitized world-space body state used for evaluation.
    pub state: AeroBodyState,
    /// Resulting world-space force and torque.
    pub wrench: AeroWrench,
}

/// Result of queuing one body's aerodynamic wrench into the external-force
/// accumulator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AeroApplyReport {
    /// Live rigid body targeted by the request.
    pub body: BodyHandle,
    /// World-space force and torque produced by evaluation.
    pub wrench: AeroWrench,
    /// Whether the wrench was accepted by the external-force accumulator.
    pub queued: bool,
}

/// Deterministic aggregate for one fixed aerodynamic pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AeroStepReport {
    /// Number of aerodynamic body registrations visited by the pass.
    pub registered: usize,
    /// Number of registrations with complete live rigid-body state.
    pub sampled: usize,
    /// Number of sampled wrenches accepted by the force accumulator.
    pub applied: usize,
    /// Deterministic sum of all sampled world-space wrenches.
    pub wrench: AeroWrench,
}

/// A physics world: all body state, shared shapes, and configuration.
#[derive(Clone, Debug, Default)]
pub struct PhysicsWorld {
    /// Structure-of-Arrays storage for all bodies.
    pub bodies: BodyStorage,
    /// Registry of shared collider shapes.
    pub shapes: ShapeRegistry,
    /// Storage for all joints binding pairs of bodies.
    pub joints: JointStorage,
    /// Global simulation configuration.
    pub config: WorldConfig,
    /// Frame-to-frame contact/trigger event tracker. Updated by
    /// [`PhysicsWorld::drain_contact_events`].
    pub contact_events: ContactEventTracker,
    /// Opt-in configuration for the cloth-rigid two-way coupling stage.
    /// Defaults to disabled so rigid-only worlds stay bit-identical.
    pub cloth_coupling: ClothRigidCouplingConfig,
    /// World-owned deformable state. Entries are advanced by
    /// [`crate::solver::XpbdSolver`] in deterministic slot order.
    pub soft_bodies: SoftBodyStorage,
    /// Topology edits emitted by world-owned soft bodies during fixed steps.
    pub topology_events: Vec<TopologyTearEvent>,
    /// Complete versioned topology updates, including their owning body
    /// handle and render/collision replacement data.
    pub topology_updates: Vec<WorldTopologyTearUpdate>,
    /// M10 external force/impulse state consumed at an explicit sub-step
    /// boundary.
    pub external_forces: ForceAccumulator,
    /// M11 deterministic analytic force fields.
    pub fields: FieldRegistry,
    /// M10 runtime connection graph.
    pub assemblies: AssemblyGraph,
    /// Boundary-owned policies for measured breakable-joint loads.
    pub breakable_joints: BreakableJointSet,
    /// Registered rigid-body aerodynamic authoring data, kept in insertion
    /// order so fixed-step reductions remain deterministic.
    pub aero_bodies: Vec<AeroBodyRegistration>,
    /// Simulation time used by the convenience fixed-substep API.
    pub aero_time: Real,
    /// Shared deterministic wind/flow field sampled by registered M12 bodies.
    pub aero_field: WindField,
    /// World-owned reduced-coordinate articulations. They are advanced by
    /// [`Self::step_articulations_substep`] at a fixed-step boundary.
    pub articulations: ArticulationStorage,
    /// Validated collision attachments, independent of rigid-body proxies.
    pub articulation_colliders: ArticulationColliderSet,
    /// Maximum-coordinate handoff events emitted by articulation breakage.
    pub articulation_degradations: Vec<ArticulationDegradationEvent>,
    collision_exclusions: CollisionExclusions,
    assembly_merges: BTreeMap<u64, AssemblyMergeState>,
    assembly_split_reports: Vec<AssemblySplitReport>,
    joint_boundary_failure: Option<JointBoundaryFailure>,
}

impl PhysicsWorld {
    /// Creates a world with the given configuration and empty storage.
    #[must_use]
    pub fn new(config: WorldConfig) -> PhysicsWorld {
        PhysicsWorld {
            bodies: BodyStorage::new(),
            shapes: ShapeRegistry::new(),
            joints: JointStorage::new(),
            config,
            contact_events: ContactEventTracker::new(),
            cloth_coupling: ClothRigidCouplingConfig::disabled(),
            soft_bodies: SoftBodyStorage::new(),
            topology_events: Vec::new(),
            topology_updates: Vec::new(),
            external_forces: ForceAccumulator::new(),
            fields: FieldRegistry::new(),
            assemblies: AssemblyGraph::new(),
            breakable_joints: BreakableJointSet::default(),
            aero_bodies: Vec::new(),
            aero_time: 0.0,
            aero_field: WindField::default(),
            articulations: ArticulationStorage::new(),
            articulation_colliders: ArticulationColliderSet::default(),
            articulation_degradations: Vec::new(),
            collision_exclusions: CollisionExclusions::default(),
            assembly_merges: BTreeMap::new(),
            assembly_split_reports: Vec::new(),
            joint_boundary_failure: None,
        }
    }

    /// Creates a world with the given gravity and default configuration
    /// otherwise.
    #[must_use]
    pub fn with_gravity(gravity: Vec3) -> PhysicsWorld {
        PhysicsWorld::new(WorldConfig::with_gravity(gravity))
    }

    /// Spawns a body described by `desc`, returning its handle.
    pub fn spawn(&mut self, desc: BodyDesc) -> BodyHandle {
        let handle = self.bodies.insert(desc);
        self.assemblies.add_body(handle);
        handle
    }

    /// Installs one persistent group covering all original articulation bodies,
    /// including branches disconnected during handoff. Each body may belong to
    /// only one group. A same-group pair collides only if both flags are true;
    /// bodies outside the group are unaffected. No joint adjacency is stored.
    ///
    /// Validation is atomic: empty, duplicate, stale or already-grouped members
    /// leave the policy unchanged. Membership lasts until body removal.
    pub fn install_collision_exclusion_group(
        &mut self,
        members: &[(BodyHandle, bool)],
    ) -> Result<CollisionExclusionGroupId, CollisionExclusionError> {
        if let Some(&(body, _)) = members
            .iter()
            .find(|(body, _)| self.assembly_merges.contains_key(&body_sort_key(body)))
        {
            return Err(CollisionExclusionError::MergedBody(body));
        }
        self.collision_exclusions.install(&self.bodies, members)
    }

    /// Immutable inspection of the world-owned exclusion policy.
    #[must_use]
    pub fn collision_exclusions(&self) -> &CollisionExclusions {
        &self.collision_exclusions
    }

    /// Shared rigid-pair gate for discrete, mixed-articulation and CCD paths.
    /// Callers enumerate live bodies; this checks policy, not handle liveness.
    #[must_use]
    pub fn collision_pair_allowed(&self, a: BodyHandle, b: BodyHandle) -> bool {
        self.collision_exclusions.allows_pair(a, b)
    }

    /// Removes a rigid body and all world-owned assembly bookkeeping referring
    /// to it. Bodies participating in a merged proxy must be split first.
    pub fn remove_body(&mut self, body: BodyHandle) -> bool {
        if self.assembly_merges.values().any(|state| {
            state.proxy == body
                || state
                    .component
                    .iter()
                    .any(|snapshot| snapshot.handle == body)
        }) {
            return false;
        }
        let removed = self.bodies.remove(body);
        if removed {
            self.collision_exclusions.remove(body);
            self.assemblies.remove_body(body);
            self.unregister_aero_body(body);
        }
        removed
    }

    /// Inserts a project-independent soft body into world-owned storage.
    ///
    /// The body may represent cloth, rope, hair, a volumetric body, or any
    /// other particle/constraint authoring. Its state is then advanced by the
    /// world solver at the same fixed substep boundary as rigid bodies.
    pub fn spawn_soft_body(&mut self, body: SoftBody) -> SoftBodyHandle {
        self.soft_bodies.insert(body)
    }

    /// Removes a world-owned soft body and invalidates its handle.
    pub fn remove_soft_body(&mut self, handle: SoftBodyHandle) -> bool {
        self.soft_bodies.remove(handle)
    }

    /// Inserts a reduced-coordinate articulation into world-owned storage.
    pub fn spawn_articulation(&mut self, articulation: Articulation) -> ArticulationHandle {
        self.articulations.insert(articulation)
    }

    /// Removes a world-owned articulation and invalidates its handle.
    pub fn remove_articulation(&mut self, handle: ArticulationHandle) -> bool {
        let removed = self.articulations.remove(handle);
        if removed {
            self.articulation_colliders
                .retain(|desc| desc.articulation != handle);
        }
        removed
    }

    /// Registers real collision geometry on a live articulation body.
    /// Rejection is atomic; neither identities nor registry entries change.
    pub fn attach_articulation_collider(
        &mut self,
        desc: ArticulationColliderDesc,
    ) -> Result<ArticulationColliderId, ArticulationColliderError> {
        use ArticulationColliderError as E;
        let model = self
            .articulations
            .get(desc.articulation)
            .ok_or(E::InvalidArticulation)?;
        if model.broken_joint().is_some() {
            return Err(E::InvalidArticulation);
        }
        if desc.link.is_some_and(|link| link >= model.link_count()) {
            return Err(E::InvalidLink);
        }
        if self.shapes.get(desc.shape).is_none() {
            return Err(E::InvalidShape);
        }
        if !desc.local_pose.translation.is_finite()
            || !desc.local_pose.rotation.is_finite()
            || (desc.local_pose.rotation.length_squared() - 1.0).abs() > 1e-4
        {
            return Err(E::InvalidPose);
        }
        if !desc.material.friction.is_finite()
            || desc.material.friction < 0.0
            || !desc.material.restitution.is_finite()
            || !(0.0..=1.0).contains(&desc.material.restitution)
        {
            return Err(E::InvalidMaterial);
        }
        self.articulation_colliders.insert(desc)
    }

    /// Queues a link-frame contact impulse for the next fixed sub-step.
    ///
    /// Queuing preserves the fixed-step boundary: callers may gather contacts
    /// from any adapter without mutating ABA state during broad-phase or
    /// render-thread work. Multiple impulses for one articulation are batched
    /// into one articulated inverse-mass solve.
    pub fn queue_articulation_contact_impulse(
        &mut self,
        handle: ArticulationHandle,
        impulse: LinkContactImpulse,
    ) -> bool {
        self.articulations.queue_contact_impulse(handle, impulse)
    }

    /// Queues a joint-break handoff for the next fixed sub-step.
    pub fn request_articulation_break(&mut self, handle: ArticulationHandle, link: usize) -> bool {
        self.articulations.request_break(handle, link)
    }

    /// Advances all world-owned articulations by one fixed sub-step.
    ///
    /// The deterministic order is: consume queued contact impulses, apply a
    /// queued break request and emit its complete handoff snapshot, then run
    /// ABA/semi-implicit integration. The method is intentionally public so a
    /// backend can call it exactly once from its fixed-substep schedule after
    /// contact generation and before publishing the next world snapshot.
    pub fn step_articulations_substep(&mut self, dt: Real) -> ArticulationWorldStepReport {
        let mut report = ArticulationWorldStepReport::default();
        if !dt.is_finite() || dt <= 0.0 {
            return report;
        }
        let handles: Vec<ArticulationHandle> = self.articulations.handles().collect();
        for handle in handles {
            let impulses = self.articulations.take_contact_impulses(handle);
            report.contact_impulses += impulses.len();
            if let Some(articulation) = self.articulations.get_mut(handle) {
                let _ = articulation.apply_contact_impulses(&impulses);
            }

            let break_request = self.articulations.take_break_request(handle);
            let degradation = if let (Some(link), Some(articulation)) =
                (break_request, self.articulations.get_mut(handle))
            {
                articulation.break_joint(link)
            } else {
                None
            };
            if let Some(degradation) = degradation {
                self.articulation_degradations
                    .push(ArticulationDegradationEvent {
                        articulation: handle,
                        degradation,
                    });
                report.degradations += 1;
            }

            if let Some(articulation) = self.articulations.get_mut(handle) {
                let step = articulation.step(dt);
                report.reports.push((handle, step));
                report.stepped += 1;
            }
        }
        report
    }

    /// Drains articulation handoff events emitted by fixed sub-steps.
    pub fn drain_articulation_degradations(&mut self) -> Vec<ArticulationDegradationEvent> {
        core::mem::take(&mut self.articulation_degradations)
    }

    /// Spawns a joint described by `desc`, returning its handle.
    pub fn spawn_joint(&mut self, desc: JointDesc) -> JointHandle {
        let a = desc.anchor_a.body;
        let b = desc.anchor_b.body;
        let kind = desc.kind;
        let handle = self.joints.insert(desc);
        let _ = self.assemblies.connect(a, b, Some(handle), kind, None);
        handle
    }

    /// Atomically registers an assembly edge and its explicit break policy.
    /// Rejection preserves both stores, including connection IDs and fatigue.
    pub fn connect_assembly(
        &mut self,
        a: BodyHandle,
        b: BodyHandle,
        joint: Option<JointHandle>,
        kind: crate::joint::JointKind,
        break_limit: Option<BreakLimit>,
    ) -> Result<ConnectionId, AssemblyError> {
        self.assemblies.connect_with_policy(
            &mut self.breakable_joints,
            a,
            b,
            joint,
            kind,
            break_limit,
        )
    }

    /// Commits measured solver constraint loads at one fixed-step boundary.
    /// This is the only world entry point that may disable a breakable joint.
    pub fn submit_joint_loads(
        &mut self,
        dt: Real,
        loads: &[SolverJointLoad],
    ) -> Result<Vec<BreakEvent>, BoundaryLoadError> {
        // Solvers report every joint they project. Policy evaluation is opt-in,
        // so keep the strict low-level boundary API while accepting the full
        // solver report at this world integration boundary.
        let breakable_loads: Vec<SolverJointLoad> = loads
            .iter()
            .copied()
            .filter(|item| self.breakable_joints.contains(item.joint))
            .collect();
        let (policies, events) =
            self.breakable_joints
                .prepare_solver_boundary(dt, &breakable_loads, &self.joints)?;
        if events.is_empty() {
            self.breakable_joints = policies;
            return Ok(events);
        }

        // A collapsed assembly keeps its internal connection snapshots outside
        // the live graph. Restore every affected proxy before disconnecting
        // broken edges so the graph and physical bodies observe one atomic
        // fixed-step break boundary.
        let mut proxies = Vec::new();
        for state in self.assembly_merges.values() {
            if state.connections.iter().any(|entry| {
                entry
                    .connection
                    .joint
                    .is_some_and(|joint| events.iter().any(|event| event.joint == joint))
            }) {
                proxies.push(state.proxy);
            }
        }
        proxies.sort_by_key(body_sort_key);
        proxies.dedup();
        if !proxies.is_empty() {
            // One allocator plan covers the entire boundary: independently
            // preparing each proxy would predict overlapping free-list slots.
            let prepared = self
                .prepare_assembly_splits(&proxies, None)
                .map_err(|(proxy, error)| BoundaryLoadError::AssemblySplit { proxy, error })?;
            let reports: Vec<_> = prepared
                .parts
                .iter()
                .map(|part| {
                    let state = &self.assembly_merges[&body_sort_key(&part.proxy)];
                    AssemblySplitReport {
                        proxy: part.proxy,
                        members: state
                            .component
                            .iter()
                            .zip(&prepared.allocation.handles()[part.range.clone()])
                            .map(|(snapshot, &current)| AssemblyMemberRemap {
                                original: snapshot.handle,
                                current,
                            })
                            .collect(),
                    }
                })
                .collect();
            self.commit_assembly_split(prepared);
            self.assembly_split_reports.extend(reports);
        }

        // Splitting restores the pre-merge joint snapshots, so re-apply the
        // policy outcome before exposing the post-break graph.
        self.breakable_joints = policies;
        self.breakable_joints
            .disable_broken(&mut self.joints, &events);

        let mut disconnected: Vec<ConnectionId> = self
            .assemblies
            .connections()
            .filter(|connection| {
                connection
                    .joint
                    .is_some_and(|joint| events.iter().any(|event| event.joint == joint))
            })
            .map(|connection| connection.id)
            .collect();
        disconnected.sort();
        disconnected.dedup();
        for id in disconnected {
            let _ = self.assemblies.disconnect(id);
        }
        Ok(events)
    }

    /// Drains ownership handoffs generated by automatic break splits.
    ///
    /// Explicit ECS-requested splits do not enqueue a report because their
    /// caller already owns the split transaction and receives the handles
    /// directly.
    pub fn drain_assembly_split_reports(&mut self) -> Vec<AssemblySplitReport> {
        std::mem::take(&mut self.assembly_split_reports)
    }

    /// Replaces one connected runtime assembly by a single physical proxy.
    /// The operation is deterministic, conserves rigid-body momentum, keeps
    /// a conservative compound convex collider, and stores enough state for a
    /// lossless split at a later fixed-step boundary.
    pub fn merge_assembly(
        &mut self,
        component: &[BodyHandle],
    ) -> Result<BodyHandle, AssemblyWorldError> {
        let prepared = self.prepare_assembly_merge(component, None)?;
        Ok(self.commit_assembly_merge(prepared))
    }

    /// Atomically merges with one policy per expanded collider leaf, ordered
    /// by sorted original body handle and then that collider's leaf order.
    /// Callers can resolve member material inheritance before merging. Geometry
    /// and policies are validated together before any world state is changed.
    pub fn merge_assembly_with_leaf_policies(
        &mut self,
        component: &[BodyHandle],
        policies: &[CompoundLeafPolicy],
    ) -> Result<BodyHandle, AssemblyWorldError> {
        let prepared = self.prepare_assembly_merge(component, Some(policies))?;
        Ok(self.commit_assembly_merge(prepared))
    }

    fn prepare_assembly_merge(
        &self,
        component: &[BodyHandle],
        policies: Option<&[CompoundLeafPolicy]>,
    ) -> Result<PreparedAssemblyMerge, AssemblyWorldError> {
        let mut members = component.to_vec();
        members.sort_by_key(body_sort_key);
        members.dedup();
        if members.is_empty() {
            return Err(AssemblyWorldError::InvalidBody);
        }
        if members.iter().any(|body| !self.bodies.contains(*body)) {
            return Err(AssemblyWorldError::InvalidBody);
        }
        let gravity_override = self
            .bodies
            .gravity_override(members[0])
            .expect("live member");
        if members
            .iter()
            .any(|body| self.bodies.gravity_override(*body) != Some(gravity_override))
        {
            return Err(AssemblyWorldError::GravityPolicyConflict);
        }
        // A single proxy cannot encode per-member flags or exclusions against
        // members left outside this component. Reject before any mutation.
        if members
            .iter()
            .any(|body| self.collision_exclusions.membership(*body).is_some())
        {
            return Err(AssemblyWorldError::CollisionExclusionConflict);
        }
        let Some(graph_component) = self.assemblies.component_of(members[0]) else {
            return Err(AssemblyWorldError::NotConnectedComponent);
        };
        if graph_component != members {
            return Err(AssemblyWorldError::NotConnectedComponent);
        }
        if members.iter().any(|body| {
            self.assembly_merges
                .values()
                .any(|state| state.proxy == *body)
        }) {
            return Err(AssemblyWorldError::InvalidProxy);
        }
        let snapshots: Vec<BodySnapshot> = members
            .iter()
            .copied()
            .map(|body| {
                self.capture_body_snapshot(body)
                    .ok_or(AssemblyWorldError::InvalidBody)
            })
            .collect::<Result<_, _>>()?;
        if snapshots
            .iter()
            .any(|snapshot| snapshot.desc.kind != crate::state::body::BodyKind::Dynamic)
        {
            return Err(AssemblyWorldError::ImmovableBody);
        }
        let states: Vec<BodyState> = snapshots
            .iter()
            .map(|snapshot| BodyState {
                handle: snapshot.handle,
                pose: Pose {
                    position: snapshot.desc.position,
                    orientation: snapshot.desc.orientation,
                },
                linear_velocity: snapshot.desc.linear_velocity,
                angular_velocity: snapshot.desc.angular_velocity,
                mass_properties: snapshot.desc.mass_properties,
            })
            .collect();
        let compound = CompoundBody::from_states(&states).map_err(AssemblyWorldError::Compound)?;
        let allocation = self
            .bodies
            .prepare_replacement(&members, 1)
            .ok_or(AssemblyWorldError::InvalidBody)?;
        let proxy = allocation.handles()[0];
        let mut graph = self.assemblies.clone();
        let affected = graph
            .collapse_component(&members, proxy)
            .map_err(AssemblyWorldError::Graph)?;
        let mut joint_snapshots = Vec::with_capacity(affected.len());
        for connection in &affected {
            let joint = connection
                .joint
                .map(|handle| {
                    self.joints
                        .get(handle)
                        .copied()
                        .ok_or(AssemblyWorldError::ActiveJointMissing)
                })
                .transpose()?;
            joint_snapshots.push(AssemblyJointSnapshot {
                connection: *connection,
                joint,
            });
        }
        let collider = self.prepare_assembly_collider(&snapshots, &compound, policies)?;
        let (proxy_orientation, inverse_inertia) = compound.principal_mass_properties();
        let first = snapshots[0].desc;
        let total_mass = compound.properties.mass;
        if snapshots.iter().any(|snapshot| {
            !snapshot.desc.linear_damping.is_finite()
                || snapshot.desc.linear_damping < 0.0
                || !snapshot.desc.angular_damping.is_finite()
                || snapshot.desc.angular_damping < 0.0
        }) {
            return Err(AssemblyWorldError::InvalidDamping);
        }
        // A finite weighted mean must not overflow just because its f32
        // numerator does. Use the same f64 mass sum for both coefficients;
        // normalizing by the rounded compound mass can also exceed f32::MAX.
        let damping_mass = snapshots
            .iter()
            .map(|snapshot| f64::from(snapshot.desc.mass_properties.mass()))
            .sum::<f64>();
        let linear_damping = snapshots
            .iter()
            .map(|snapshot| {
                f64::from(snapshot.desc.linear_damping)
                    * f64::from(snapshot.desc.mass_properties.mass())
            })
            .sum::<f64>()
            / damping_mass;
        let angular_damping = snapshots
            .iter()
            .map(|snapshot| {
                f64::from(snapshot.desc.angular_damping)
                    * f64::from(snapshot.desc.mass_properties.mass())
            })
            .sum::<f64>()
            / damping_mass;
        let proxy_desc = BodyDesc {
            kind: crate::state::body::BodyKind::Dynamic,
            position: compound.properties.center_of_mass,
            orientation: proxy_orientation,
            linear_velocity: compound.linear_velocity,
            angular_velocity: compound.angular_velocity,
            mass_properties: crate::state::body::MassProperties {
                inv_mass: 1.0 / total_mass,
                inv_inertia: inverse_inertia,
            },
            linear_damping: linear_damping as f32,
            angular_damping: angular_damping as f32,
            collider: None,
            material: first.material,
            is_sensor: false,
            ccd: snapshots.iter().any(|snapshot| snapshot.desc.ccd),
        };
        let source_centers: Vec<(BodyHandle, Vec3)> = snapshots
            .iter()
            .map(|snapshot| (snapshot.handle, snapshot.desc.position))
            .collect();

        let (forces, _) = self
            .external_forces
            .prepare_merge_bodies(&source_centers, proxy, compound.properties.center_of_mass)
            .map_err(AssemblyWorldError::Force)?;
        let mut joints = Vec::new();
        for entry in &joint_snapshots {
            if let Some(handle) = entry.connection.joint {
                if let Some(mut joint) = entry.joint {
                    if members.contains(&joint.anchor_a.body) {
                        joint.anchor_a.body = proxy;
                    }
                    if members.contains(&joint.anchor_b.body) {
                        joint.anchor_b.body = proxy;
                    }
                    if members.contains(&entry.connection.a)
                        && members.contains(&entry.connection.b)
                    {
                        joint.enabled = false;
                    }
                    joints.push((handle, joint));
                }
            }
        }
        Ok(PreparedAssemblyMerge {
            allocation,
            descriptor: proxy_desc,
            collider,
            gravity: gravity_override,
            graph,
            forces,
            joints,
            recovery: AssemblyMergeState {
                proxy,
                component: snapshots,
                compound,
                connections: joint_snapshots,
            },
        })
    }

    fn commit_assembly_merge(&mut self, mut prepared: PreparedAssemblyMerge) -> BodyHandle {
        prepared.descriptor.collider = prepared
            .collider
            .map(|collider| self.shapes.insert_prepared_compound(collider));
        let handles = self
            .bodies
            .commit_replacement(prepared.allocation, &[prepared.descriptor]);
        let proxy = handles[0];
        assert!(self.bodies.set_gravity_override(proxy, prepared.gravity));
        self.assemblies = prepared.graph;
        self.external_forces.commit_remap(prepared.forces);
        for (handle, joint) in prepared.joints {
            *self.joints.get_mut(handle).expect("prepared live joint") = joint;
        }
        self.assembly_merges
            .insert(body_sort_key(&proxy), prepared.recovery);
        proxy
    }

    /// Splits a previously merged proxy and restores all member bodies,
    /// collision state, joint endpoints, graph edges and pending wrench.
    pub fn split_assembly(
        &mut self,
        proxy: BodyHandle,
    ) -> Result<Vec<BodyHandle>, AssemblyWorldError> {
        let prepared = self.prepare_assembly_split(proxy, None)?;
        Ok(self.commit_assembly_split(prepared))
    }

    /// Splits only if the recovery record matches the caller's original
    /// generational member handles in exact order. A mismatch leaves the world
    /// unchanged, just like graph, allocation, or force preparation failure.
    pub fn split_assembly_checked(
        &mut self,
        proxy: BodyHandle,
        expected_original_members: &[BodyHandle],
    ) -> Result<Vec<BodyHandle>, AssemblyWorldError> {
        let prepared = self.prepare_assembly_split(proxy, Some(expected_original_members))?;
        Ok(self.commit_assembly_split(prepared))
    }

    fn prepare_assembly_split(
        &self,
        proxy: BodyHandle,
        expected_original_members: Option<&[BodyHandle]>,
    ) -> Result<PreparedAssemblySplit, AssemblyWorldError> {
        self.prepare_assembly_splits(&[proxy], expected_original_members)
            .map_err(|(_, error)| error)
    }

    fn prepare_assembly_splits(
        &self,
        proxies: &[BodyHandle],
        expected_original_members: Option<&[BodyHandle]>,
    ) -> Result<PreparedAssemblySplit, (BodyHandle, AssemblyWorldError)> {
        let mut count = 0usize;
        for &proxy in proxies {
            let state = self
                .assembly_merges
                .get(&body_sort_key(&proxy))
                .ok_or((proxy, AssemblyWorldError::NotMerged))?;
            if state.proxy != proxy {
                return Err((proxy, AssemblyWorldError::InvalidProxy));
            }
            count = count
                .checked_add(state.component.len())
                .ok_or((proxy, AssemblyWorldError::InvalidProxy))?;
        }
        let allocation = self
            .bodies
            .prepare_replacement(proxies, count)
            .ok_or((proxies[0], AssemblyWorldError::InvalidProxy))?;
        let mut descriptors = Vec::with_capacity(count);
        let mut parts = Vec::with_capacity(proxies.len());
        let mut graph = self.assemblies.clone();
        for &proxy in proxies {
            let state = &self.assembly_merges[&body_sort_key(&proxy)];
            let start = descriptors.len();
            let range = start..start + state.component.len();
            let part = self
                .prepare_assembly_split_part(
                    state,
                    expected_original_members,
                    &allocation.handles()[range.clone()],
                    &mut descriptors,
                    &mut graph,
                    range,
                )
                .map_err(|error| (proxy, error))?;
            parts.push(part);
        }
        Ok(PreparedAssemblySplit {
            allocation,
            descriptors,
            parts,
            graph,
        })
    }

    fn prepare_assembly_split_part(
        &self,
        state: &AssemblyMergeState,
        expected_original_members: Option<&[BodyHandle]>,
        new_handles: &[BodyHandle],
        descriptors: &mut Vec<BodyDesc>,
        graph: &mut AssemblyGraph,
        range: std::ops::Range<usize>,
    ) -> Result<PreparedAssemblySplitPart, AssemblyWorldError> {
        let proxy = state.proxy;
        if !self.bodies.contains(proxy) {
            return Err(AssemblyWorldError::InvalidProxy);
        }
        if expected_original_members.is_some_and(|expected| {
            !expected
                .iter()
                .copied()
                .eq(state.component.iter().map(|snapshot| snapshot.handle))
        }) || state.component.len() != state.compound.parts.len()
        {
            return Err(AssemblyWorldError::InvalidProxy);
        }
        if self.collision_exclusions.membership(proxy).is_some() {
            return Err(AssemblyWorldError::CollisionExclusionConflict);
        }
        let proxy_snapshot = self
            .capture_body_snapshot(proxy)
            .ok_or(AssemblyWorldError::InvalidProxy)?;
        let gravity_override = self.bodies.gravity_override(proxy).expect("live proxy");
        let mut mapping = BTreeMap::new();
        for ((snapshot, part), &handle) in state
            .component
            .iter()
            .zip(&state.compound.parts)
            .zip(new_handles)
        {
            let world_radius = proxy_snapshot.desc.orientation * part.relative_pose.position;
            let position = proxy_snapshot.desc.position + world_radius;
            let orientation =
                (proxy_snapshot.desc.orientation * part.relative_pose.orientation).normalize();
            let linear_velocity = proxy_snapshot.desc.linear_velocity
                + proxy_snapshot.desc.angular_velocity.cross(world_radius);
            let mut desc = snapshot.desc;
            desc.position = position;
            desc.orientation = orientation;
            desc.linear_velocity = linear_velocity;
            desc.angular_velocity = proxy_snapshot.desc.angular_velocity;
            if !position.is_finite()
                || !orientation.is_finite()
                || !orientation.is_normalized()
                || !linear_velocity.is_finite()
                || !desc.angular_velocity.is_finite()
            {
                return Err(AssemblyWorldError::Compound(CompoundError::NonFinite));
            }
            descriptors.push(desc);
            mapping.insert(body_sort_key(&snapshot.handle), handle);
        }
        let remapped_connections: Vec<Connection> = state
            .connections
            .iter()
            .map(|entry| remap_connection(entry.connection, &mapping))
            .collect();
        graph
            .restore_component(proxy, new_handles, &remapped_connections)
            .map_err(AssemblyWorldError::Graph)?;
        let mut joints = Vec::new();
        for entry in &state.connections {
            if let Some(handle) = entry.connection.joint {
                if let (Some(_), Some(original)) = (self.joints.get(handle), entry.joint) {
                    joints.push((handle, remap_joint(original, &mapping)));
                }
            }
        }
        let targets: Vec<(BodyHandle, f32, Vec3)> = state
            .component
            .iter()
            .zip(new_handles)
            .zip(&state.compound.parts)
            .map(|((_snapshot, handle), part)| {
                (
                    *handle,
                    part.mass,
                    proxy_snapshot.desc.position
                        + proxy_snapshot.desc.orientation * part.relative_pose.position,
                )
            })
            .collect();
        let (forces, _) = self
            .external_forces
            .prepare_split_body(proxy, proxy_snapshot.desc.position, &targets)
            .map_err(AssemblyWorldError::Force)?;
        Ok(PreparedAssemblySplitPart {
            proxy,
            range,
            proxy_snapshot,
            gravity: gravity_override,
            forces,
            joints,
        })
    }

    fn commit_assembly_split(&mut self, prepared: PreparedAssemblySplit) -> Vec<BodyHandle> {
        let handles = self
            .bodies
            .commit_replacement(prepared.allocation, &prepared.descriptors);
        for part in prepared.parts {
            for (&handle, desc) in handles[part.range.clone()]
                .iter()
                .zip(&prepared.descriptors[part.range.clone()])
            {
                // The live proxy owns current policy and lifecycle state.
                assert!(self.bodies.set_gravity_override(handle, part.gravity));
                self.restore_body_runtime(
                    handle,
                    &part.proxy_snapshot,
                    desc.position,
                    desc.orientation,
                );
            }
            self.external_forces.commit_remap(part.forces);
            for (handle, joint) in part.joints {
                *self.joints.get_mut(handle).expect("prepared live joint") = joint;
            }
            self.assembly_merges.remove(&body_sort_key(&part.proxy));
        }
        self.assemblies = prepared.graph;
        handles
    }

    fn capture_body_snapshot(&self, body: BodyHandle) -> Option<BodySnapshot> {
        Some(BodySnapshot {
            handle: body,
            desc: self.bodies.descriptor(body)?,
            previous_pose: Pose {
                position: self.bodies.prev_position(body)?,
                orientation: self.bodies.prev_orientation(body)?,
            },
            sleeping: self.bodies.is_sleeping(body)?,
            sleep_timer: self.bodies.sleep_timer(body)?,
        })
    }

    fn restore_body_runtime(
        &mut self,
        body: BodyHandle,
        snapshot: &BodySnapshot,
        position: Vec3,
        orientation: glam::Quat,
    ) {
        // Insertion already installed the current proxy's velocity field.
        // Restore lifecycle state only, not the member's pre-merge motion.
        let index = body.index() as usize;
        let view = self.bodies.solver_view_mut();
        if view.is_active(index) {
            view.prev_positions[index] = position;
            view.prev_orientations[index] = orientation;
            view.sleeping[index] = snapshot.sleeping;
            view.sleep_timers[index] = snapshot.sleep_timer;
        }
    }

    fn prepare_assembly_collider(
        &self,
        snapshots: &[BodySnapshot],
        compound: &CompoundBody,
        policies: Option<&[CompoundLeafPolicy]>,
    ) -> Result<Option<PreparedCompound>, AssemblyWorldError> {
        let proxy_pose = Isometry::new(
            compound.properties.center_of_mass,
            compound.principal_mass_properties().0,
        );
        let world_to_proxy = proxy_pose.inverse();
        let roots: Vec<_> = snapshots
            .iter()
            .filter_map(|snapshot| {
                Some(CompoundChild {
                    collider: snapshot.desc.collider?,
                    local_pose: world_to_proxy.mul(&Isometry::new(
                        snapshot.desc.position,
                        snapshot.desc.orientation,
                    )),
                })
            })
            .collect();
        if roots.is_empty() {
            return if policies.is_some_and(|policies| !policies.is_empty()) {
                Err(AssemblyWorldError::UnsupportedCollider)
            } else {
                Ok(None)
            };
        }
        // Validate the original compound handles and their inline caches too;
        // flattening them ourselves first would hide forged cache metadata.
        let inherited = self
            .shapes
            .prepare_compound(&roots)
            .map_err(|_| AssemblyWorldError::UnsupportedCollider)?;
        let Some(policies) = policies else {
            return Ok(Some(inherited));
        };
        let mut children = Vec::new();
        for root in roots {
            let collider = root.collider;
            let local = root.local_pose;
            if let Some(crate::collider::ColliderShape::Compound { compound, .. }) =
                self.shapes.get(collider)
            {
                let data = self
                    .shapes
                    .compound(*compound)
                    .ok_or(AssemblyWorldError::UnsupportedCollider)?;
                if children.len() + data.children().len() > MAX_COMPOUND_CHILDREN {
                    return Err(AssemblyWorldError::UnsupportedCollider);
                }
                for child in data.children() {
                    children.push(CompoundChild {
                        collider: child.collider,
                        local_pose: local.mul(&child.local_pose),
                    });
                }
            } else {
                if children.len() == MAX_COMPOUND_CHILDREN {
                    return Err(AssemblyWorldError::UnsupportedCollider);
                }
                children.push(CompoundChild {
                    collider,
                    local_pose: local,
                });
            }
        }
        if policies.len() != children.len() {
            return Err(AssemblyWorldError::UnsupportedCollider);
        }
        self.shapes
            .prepare_compound_with_policies(&children, policies)
            .map(Some)
            .map_err(|_| AssemblyWorldError::UnsupportedCollider)
    }

    /// Replaces the shared deterministic wind/flow field used by M12 bodies.
    pub fn set_aero_field(&mut self, field: WindField) {
        self.aero_field = field;
    }

    /// Returns the shared rigid-body wind/flow field.
    #[must_use]
    pub fn aero_field(&self) -> &WindField {
        &self.aero_field
    }

    /// Returns mutable access to the shared rigid-body wind/flow field.
    pub fn aero_field_mut(&mut self) -> &mut WindField {
        &mut self.aero_field
    }

    /// Registers or replaces all aerodynamic elements attached to `body`.
    /// Replacement preserves the original registration position and therefore
    /// the deterministic reduction order.
    pub fn register_aero_body(
        &mut self,
        body: BodyHandle,
        config: AeroBodyConfig,
    ) -> Result<(), AeroRegistrationError> {
        if !self.bodies.contains(body) {
            return Err(AeroRegistrationError::InvalidBody);
        }
        if let Some(existing) = self.aero_bodies.iter_mut().find(|entry| entry.body == body) {
            existing.config = config;
        } else {
            self.aero_bodies.push(AeroBodyRegistration { body, config });
        }
        Ok(())
    }

    /// Removes a body's aerodynamic registration.
    pub fn unregister_aero_body(&mut self, body: BodyHandle) -> bool {
        let Some(index) = self.aero_bodies.iter().position(|entry| entry.body == body) else {
            return false;
        };
        self.aero_bodies.remove(index);
        true
    }

    /// Returns the registered configuration for `body`, if any.
    #[must_use]
    pub fn aero_body(&self, body: BodyHandle) -> Option<&AeroBodyConfig> {
        self.aero_bodies
            .iter()
            .find(|entry| entry.body == body)
            .map(|entry| &entry.config)
    }

    /// Samples one registered body without mutating the world at an explicit
    /// flow-field time. This inspection API does not consume fixed-step
    /// velocity history, so its acceleration fields are zero.
    #[must_use]
    pub fn sample_aero_body(&self, body: BodyHandle, time: Real) -> Option<AeroBodySample> {
        let entry = self.aero_bodies.iter().find(|entry| entry.body == body)?;
        self.sample_aero_entry(entry, time)
    }

    fn sample_aero_entry(
        &self,
        entry: &AeroBodyRegistration,
        time: Real,
    ) -> Option<AeroBodySample> {
        let state = self.aero_body_state(entry.body)?;
        let wrench = entry.config.compute_wrench(&state, &self.aero_field, time);
        Some(AeroBodySample {
            body: entry.body,
            state,
            wrench,
        })
    }

    /// Samples and queues one registered body's world-space force and torque
    /// for consumption at the next fixed prediction boundary.
    pub fn apply_aero_body(&mut self, body: BodyHandle, time: Real) -> Option<AeroApplyReport> {
        let sample = self.sample_aero_body(body, time)?;
        let queued = self.apply_aero_wrench(body, sample.wrench, sample.state.position);
        Some(AeroApplyReport {
            body,
            wrench: sample.wrench,
            queued,
        })
    }

    /// Evaluates every registered body once and queues all M12 wrenches for
    /// one fixed sub-step. `dt` is the positive sub-step duration; the current
    /// [`aero_time`](Self::aero_time) is passed to the flow field, then the
    /// clock advances after the pass. Already queued non-aerodynamic loads are
    /// included in the implicit added-mass solve, but are not queued again.
    pub fn step_aero_substep(&mut self, dt: Real) -> AeroStepReport {
        if !dt.is_finite() || dt <= 0.0 {
            return AeroStepReport {
                registered: self.aero_bodies.len(),
                ..AeroStepReport::default()
            };
        }
        let time = self.aero_time;
        // Publish all registered propulsor wakes into one immutable field
        // snapshot before evaluating any body. This makes the result
        // independent of registration order and lets downstream panels sample
        // every active thruster wake during the same fixed sub-step.
        let mut flow = self.aero_field.clone();
        for entry in &self.aero_bodies {
            let Some(state) = self.aero_body_state(entry.body) else {
                continue;
            };
            for thruster in &entry.config.thrusters {
                if let Some(wake) = thruster.world_slipstream(&state) {
                    flow.slipstreams.push(wake);
                }
            }
        }
        let mut report = AeroStepReport {
            registered: self.aero_bodies.len(),
            ..AeroStepReport::default()
        };
        for index in 0..self.aero_bodies.len() {
            let body = self.aero_bodies[index].body;
            let Some(state) = self.aero_body_state(body) else {
                continue;
            };
            let Some(mass) = self.bodies.mass_properties(body) else {
                continue;
            };
            let pending = self.external_forces.pending(body).unwrap_or_default();
            let wrench = self.aero_bodies[index].config.compute_substep_wrench(
                &state,
                &flow,
                time,
                dt,
                mass.inv_mass,
                mass.inv_inertia,
                AeroWrench {
                    force: pending.force,
                    torque: pending.torque,
                },
                self.bodies
                    .gravity_override(body)
                    .flatten()
                    .unwrap_or(self.config.gravity),
            );
            report.sampled += 1;
            report.wrench.force += wrench.force;
            report.wrench.torque += wrench.torque;
            if self.apply_aero_wrench(body, wrench, state.position) {
                report.applied += 1;
            }
        }
        self.aero_time += dt;
        report
    }

    /// Accumulates a world-space force at a world-space point. The induced
    /// torque is computed about the body's current center of mass.
    pub fn apply_force(&mut self, body: BodyHandle, force: Vec3, point: Vec3) -> bool {
        let Some(center) = self.bodies.position(body) else {
            return false;
        };
        if self
            .external_forces
            .accumulate_at_point(body, force, point, center)
            .is_err()
        {
            return false;
        }
        self.bodies.wake(body);
        true
    }

    /// Accumulates a pure world-space torque for the next consumed sub-step.
    pub fn apply_torque(&mut self, body: BodyHandle, torque: Vec3) -> bool {
        if self
            .external_forces
            .accumulate_torque(body, torque)
            .is_err()
        {
            return false;
        }
        self.bodies.wake(body);
        true
    }

    /// Applies an instantaneous world-space impulse at a world-space point.
    pub fn apply_impulse(&mut self, body: BodyHandle, impulse: Vec3, point: Vec3) -> bool {
        ForceAccumulator::apply_impulse(&mut self.bodies, body, impulse, point).is_ok()
    }

    /// Applies a pure angular impulse immediately.
    pub fn apply_torque_impulse(&mut self, body: BodyHandle, impulse: Vec3) -> bool {
        ForceAccumulator::apply_torque_impulse(&mut self.bodies, body, impulse).is_ok()
    }

    /// Samples all registered fields and accumulates their world-space forces.
    /// Body layers are not present in the current SoA, so the explicit world
    /// adapter uses the ALL mask; an ECS adapter can filter before insertion.
    pub fn sample_fields(&mut self) -> usize {
        let mut samples = Vec::new();
        for slot in 0..self.bodies.slot_count() {
            let Some(body) = self.bodies.handle_at_slot(slot) else {
                continue;
            };
            if self.bodies.kind(body) != Some(crate::state::body::BodyKind::Dynamic) {
                continue;
            }
            let Some(position) = self.bodies.position(body) else {
                continue;
            };
            let Some(velocity) = self.bodies.linear_velocity(body) else {
                continue;
            };
            let mass = self
                .bodies
                .mass_properties(body)
                .map(|properties| properties.mass())
                .unwrap_or(0.0);
            let sample = self
                .fields
                .sample(position, velocity, mass, FieldTargetMask::ALL);
            if sample.force != Vec3::ZERO || sample.torque != Vec3::ZERO {
                samples.push((body, position, sample));
            }
        }
        let count = samples.len();
        for (body, position, sample) in samples {
            if sample.force != Vec3::ZERO {
                let _ = self.apply_force(body, sample.force, position);
            }
            if sample.torque != Vec3::ZERO {
                let _ = self.apply_torque(body, sample.torque);
            }
        }
        count
    }

    /// Samples a complete field batch using an external authority. The batch
    /// contains only live dynamic bodies, including sleepers (fields may wake
    /// them), in slot order. World-space force/torque is applied at the COM.
    /// No force, sleep, or simulation state changes if the provider fails or
    /// returns malformed/nonfinite/overflowing output. Providers live outside
    /// the world, so snapshots and clones never capture GPU device objects.
    pub fn sample_fields_with<S: FieldBatchSampler + ?Sized>(
        &mut self,
        sampler: &mut S,
    ) -> Result<usize, FieldBatchError<S::Error>> {
        let mut handles = Vec::new();
        let mut inputs = Vec::new();
        for slot in 0..self.bodies.slot_count() {
            let Some(body) = self.bodies.handle_at_slot(slot) else {
                continue;
            };
            if self.bodies.kind(body) != Some(crate::state::body::BodyKind::Dynamic) {
                continue;
            }
            handles.push(body);
            inputs.push(FieldBody {
                position: self.bodies.position(body).expect("live body"),
                velocity: self.bodies.linear_velocity(body).expect("live body"),
                mass: self.bodies.mass_properties(body).expect("live body").mass(),
                target_mask: FieldTargetMask::ALL,
            });
        }
        let samples = sampler
            .sample_batch(&self.fields, &inputs)
            .map_err(FieldBatchError::Provider)?;
        if samples.len() != handles.len() {
            return Err(FieldBatchError::SampleCount {
                expected: handles.len(),
                actual: samples.len(),
            });
        }
        // Validate sums as well as samples before touching the accumulator.
        for (index, (&body, sample)) in handles.iter().zip(&samples).enumerate() {
            if !sample.force.is_finite() || !sample.torque.is_finite() {
                return Err(FieldBatchError::NonFiniteSample { index });
            }
            let pending = self.external_forces.pending(body).unwrap_or_default();
            if !(pending.force + sample.force).is_finite()
                || !(pending.torque + sample.torque).is_finite()
            {
                return Err(FieldBatchError::AccumulationOverflow { index });
            }
        }
        let mut applied = 0;
        for (body, sample) in handles.into_iter().zip(samples) {
            if sample.force == Vec3::ZERO && sample.torque == Vec3::ZERO {
                continue;
            }
            self.external_forces
                .accumulate(
                    body,
                    crate::dynamics::force::ForceTorque {
                        force: sample.force,
                        torque: sample.torque,
                    },
                )
                .expect("all unique body sums were validated before commit");
            self.bodies.wake(body);
            applied += 1;
        }
        Ok(applied)
    }

    /// Consumes accumulated external forces into velocities for one sub-step.
    /// Backends call this immediately before their prediction phase.
    pub fn consume_external_forces(&mut self, dt: Real) -> usize {
        self.external_forces.consume(&mut self.bodies, dt)
    }

    /// Complete free-body M10/M11 step: field sampling, force consumption,
    /// then gravity and damping integration.
    pub fn step_external_dynamics(&mut self, dt: Real) -> usize {
        self.collision_exclusions.retain_live(&self.bodies);
        self.sample_fields();
        let _ = self.step_aero_substep(dt);
        let count = self.consume_external_forces(dt);
        Integrator::integrate(&mut self.bodies, self.config.gravity, dt);
        count
    }

    /// Prepares one fixed XPBD sub-step by applying queued external wrenches.
    /// Gravity and damping remain owned by the XPBD predictor.
    pub fn prepare_external_substep(&mut self, dt: Real) -> usize {
        if !dt.is_finite() || dt <= 0.0 {
            return 0;
        }
        self.collision_exclusions.retain_live(&self.bodies);
        self.sample_fields();
        self.finish_external_substep(dt)
    }

    /// The same fixed-substep preparation as the CPU default, with field
    /// evaluation delegated to a caller-owned compute authority. CPU analytic
    /// fields are not sampled again. Invalid duration does not call the provider.
    pub fn prepare_external_substep_with<S: FieldBatchSampler + ?Sized>(
        &mut self,
        dt: Real,
        sampler: &mut S,
    ) -> Result<usize, FieldBatchError<S::Error>> {
        if !dt.is_finite() || dt <= 0.0 {
            return Ok(0);
        }
        self.sample_fields_with(sampler)?;
        self.collision_exclusions.retain_live(&self.bodies);
        Ok(self.finish_external_substep(dt))
    }

    fn finish_external_substep(&mut self, dt: Real) -> usize {
        let _ = self.step_aero_substep(dt);
        let count = self.consume_external_forces(dt);
        self.refresh_solver_mass_properties();
        count
    }

    fn refresh_solver_mass_properties(&mut self) {
        self.bodies.reset_solver_mass_properties();
        for index in 0..self.aero_bodies.len() {
            let body = self.aero_bodies[index].body;
            let Some(base) = self.bodies.mass_properties(body) else {
                continue;
            };
            let Some(position) = self.bodies.position(body) else {
                continue;
            };
            let config = &self.aero_bodies[index].config;
            let (inv_mass, inv_inertia) = config.implicit_inverse_properties(
                Vec3::splat(base.inv_mass),
                base.inv_inertia,
                config.added_mass_immersion(position),
            );
            let _ = self.bodies.set_solver_mass_properties(
                body,
                SolverMassProperties {
                    inv_mass,
                    inv_inertia,
                },
            );
        }
    }

    /// Converts a grab drive into an accumulated force/torque pair.
    pub fn apply_grab_drive(&mut self, drive: &GrabDrive, dt: Real) -> bool {
        let Some(state) = self.body_state(drive.target) else {
            return false;
        };
        let Some(force) = drive.sample(&state, dt) else {
            return false;
        };
        if force.force == Vec3::ZERO && force.torque == Vec3::ZERO {
            return true;
        }
        if self
            .external_forces
            .accumulate(drive.target, force)
            .is_err()
        {
            return false;
        }
        self.bodies.wake(drive.target);
        true
    }

    fn body_state(&self, body: BodyHandle) -> Option<BodyState> {
        Some(BodyState {
            handle: body,
            pose: Pose {
                position: self.bodies.position(body)?,
                orientation: self.bodies.orientation(body)?,
            },
            linear_velocity: self.bodies.linear_velocity(body)?,
            angular_velocity: self.bodies.angular_velocity(body)?,
            mass_properties: self.bodies.mass_properties(body)?,
        })
    }

    fn aero_body_state(&self, body: BodyHandle) -> Option<AeroBodyState> {
        Some(AeroBodyState {
            position: self.bodies.position(body)?,
            orientation: self.bodies.orientation(body)?,
            linear_velocity: self.bodies.linear_velocity(body)?,
            angular_velocity: self.bodies.angular_velocity(body)?,
            ..AeroBodyState::default()
        })
    }

    fn apply_aero_wrench(&mut self, body: BodyHandle, wrench: AeroWrench, point: Vec3) -> bool {
        let Some(center) = self.bodies.position(body) else {
            return false;
        };
        if self
            .external_forces
            .accumulate_at_point(body, wrench.force, point, center)
            .is_err()
        {
            return false;
        }
        if self
            .external_forces
            .accumulate_torque(body, wrench.torque)
            .is_err()
        {
            return false;
        }
        self.bodies.wake(body);
        true
    }

    /// Runs one substep of the opt-in cloth↔rigid two-way coupling stage.
    ///
    /// `positions` is a soft body's particle position column (corrected in
    /// place) and `inverse_masses` its index-aligned inverse-mass column; `dt`
    /// is the substep. The soft particles and the rigid proxies they rest
    /// against exchange a mass-weighted push, and each rigid body receives the
    /// linear reaction impulse the particles exerted on it. When
    /// [`cloth_coupling`](Self::cloth_coupling) is disabled (the default) this
    /// is a no-op and the rigid-only path is untouched, so existing goldens stay
    /// bit-identical.
    ///
    /// Returns a [`CouplingReport`] summarising the pass (participating proxy
    /// count, bodies written back, and the total applied impulse).
    pub fn resolve_cloth_coupling(
        &mut self,
        positions: &mut [Vec3],
        inverse_masses: &[Real],
        dt: Real,
    ) -> CouplingReport {
        couple_cloth_to_rigid(self, positions, inverse_masses, dt)
    }

    /// Runs the existing two-way coupling driver against one world-owned soft
    /// body. The body is temporarily detached from storage only to satisfy the
    /// borrow boundary between particle state and rigid-world writeback; its
    /// handle and deterministic slot are unchanged.
    pub fn resolve_soft_body_coupling(
        &mut self,
        handle: SoftBodyHandle,
        dt: Real,
    ) -> Option<CouplingReport> {
        let mut body = self.soft_bodies.take(handle)?;
        let (positions, inverse_masses) = body.particles.positions_and_inverse_masses_mut();
        let report = self.resolve_cloth_coupling(positions, inverse_masses, dt);
        self.soft_bodies.put(handle, body);
        Some(report)
    }

    /// Advances one fixed substep for every world-owned soft body.
    ///
    /// The order is fixed: soft prediction and internal projection, proxy
    /// gather/two-way coupling/reaction writeback when enabled, then particle
    /// velocity recovery. Empty storage is a no-op. With coupling disabled the
    /// rigid-only path is not entered and existing rigid worlds remain
    /// bit-identical.
    pub(crate) fn step_soft_bodies_substep(&mut self, h: Real) {
        if self.soft_bodies.is_empty() {
            return;
        }
        let handles: Vec<SoftBodyHandle> = self.soft_bodies.handles().collect();
        let solver = SoftSolver::new();
        for handle in handles {
            let Some(mut body) = self.soft_bodies.take(handle) else {
                continue;
            };
            solver.step_substep_before_velocity_recovery(
                &mut body.particles,
                &mut body.constraints,
                &body.config,
                &SoftContacts::EMPTY,
                h,
            );
            self.soft_bodies.put(handle, body);

            if self.cloth_coupling.is_enabled() {
                let _ = self.resolve_soft_body_coupling(handle, h);
            }

            let Some(mut body) = self.soft_bodies.take(handle) else {
                continue;
            };
            let events = body.advance_topology_tearing();
            self.topology_events.extend(events);
            if let Some(update) = body.take_topology_update() {
                self.topology_updates.push(WorldTopologyTearUpdate {
                    body: handle,
                    update,
                });
            }
            solver.recover_substep_velocities(&mut body.particles, h);
            self.soft_bodies.put(handle, body);
        }
    }

    /// Drains topology edits generated by soft-body tearing since the last
    /// call. Consumers use these events to update render meshes, collision
    /// acceleration structures, effects, and replay records without polling
    /// or duplicating the topology decision.
    pub fn drain_topology_events(&mut self) -> Vec<TopologyTearEvent> {
        core::mem::take(&mut self.topology_events)
    }

    /// Drains complete topology updates in deterministic soft-body order.
    pub fn drain_topology_updates(&mut self) -> Vec<WorldTopologyTearUpdate> {
        core::mem::take(&mut self.topology_updates)
    }

    /// Detects the current contacts and diffs them against the previous call to
    /// produce collision and trigger events.
    ///
    /// This should be called once per rendered/simulated frame, after the
    /// solver has advanced the world, so that events reflect the settled poses.
    /// Pairs where at least one body is a sensor
    /// ([`BodyStorage::is_sensor`](crate::state::storage::BodyStorage::is_sensor))
    /// yield trigger enter/exit events; all other pairs yield collision
    /// start/end events. The returned events can be forwarded to an
    /// [`ObserverRegistry`](crate::events::ObserverRegistry).
    #[must_use]
    pub fn drain_contact_events(&mut self) -> Vec<PhysicsEvent> {
        let manifolds = detect_contacts(self);
        let bodies = &self.bodies;
        self.contact_events.record(&manifolds, |handle| {
            bodies.is_sensor(handle).unwrap_or(false)
        })
    }
}

fn body_sort_key(body: &BodyHandle) -> u64 {
    (u64::from(body.index()) << 32) | u64::from(body.generation())
}

fn remap_body(body: BodyHandle, mapping: &BTreeMap<u64, BodyHandle>) -> BodyHandle {
    mapping.get(&body_sort_key(&body)).copied().unwrap_or(body)
}

fn remap_connection(mut connection: Connection, mapping: &BTreeMap<u64, BodyHandle>) -> Connection {
    connection.a = remap_body(connection.a, mapping);
    connection.b = remap_body(connection.b, mapping);
    connection
}

fn remap_joint(mut joint: Joint, mapping: &BTreeMap<u64, BodyHandle>) -> Joint {
    joint.anchor_a.body = remap_body(joint.anchor_a.body, mapping);
    joint.anchor_b.body = remap_body(joint.anchor_b.body, mapping);
    joint
        .gear_state
        .remap_bodies(joint.anchor_a.body, joint.anchor_b.body);
    joint
        .revolute_state
        .remap_bodies(joint.anchor_a.body, joint.anchor_b.body);
    joint
}

/// A topology update with its world-owned soft-body source.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldTopologyTearUpdate {
    /// Soft body whose particle and triangle streams changed.
    pub body: SoftBodyHandle,
    /// Complete atomic replacement payload.
    pub update: TopologyTearUpdate,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::assembly::{JointLoad, SolverJointLoad};
    use crate::joint::{DistanceJoint, JointKind};
    use crate::reduced::{
        ArticulationBase, ArticulationBuilder, ArticulationJoint, SpatialInertia, SpatialTransform,
    };
    use crate::solver::{Solver, XpbdSolver};

    #[test]
    fn spawn_adds_body() {
        let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
        let h = world.spawn(BodyDesc::dynamic_at(Vec3::Y));
        assert_eq!(world.bodies.len(), 1);
        assert_eq!(world.bodies.position(h), Some(Vec3::Y));
        assert_eq!(world.config.gravity.y, -9.81);
    }

    fn breakable_distance_world(
        limit: BreakLimit,
    ) -> (
        PhysicsWorld,
        BodyHandle,
        BodyHandle,
        JointHandle,
        ConnectionId,
    ) {
        let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
        let a = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
        let b = world.spawn(BodyDesc::dynamic_at(Vec3::new(3.0, 0.0, 0.0)));
        let kind = JointKind::Distance(DistanceJoint::rigid(1.0));
        let joint = world.joints.insert(JointDesc::distance(
            a,
            Vec3::ZERO,
            b,
            Vec3::ZERO,
            DistanceJoint::rigid(1.0),
        ));
        let connection = world
            .connect_assembly(a, b, Some(joint), kind, Some(limit))
            .expect("valid breakable distance assembly");
        (world, a, b, joint, connection)
    }

    #[test]
    fn xpbd_projection_breaks_joint_and_disconnects_assembly() {
        let (mut world, _a, _b, joint, connection) =
            breakable_distance_world(BreakLimit::new(0.01, 0.01));

        let mut solver = XpbdSolver::new();
        solver.step(&mut world, 1.0 / 60.0, 1);

        assert!(!world.joints.get(joint).expect("live joint").enabled);
        assert!(world
            .assemblies
            .connections()
            .all(|entry| entry.id != connection));
        assert_eq!(world.assemblies.connected_components().len(), 2);
    }

    #[test]
    fn subthreshold_projection_keeps_joint_and_assembly_connected() {
        let (mut world, _a, _b, joint, connection) =
            breakable_distance_world(BreakLimit::new(1.0e9, 1.0e9));

        let mut solver = XpbdSolver::new();
        solver.step(&mut world, 1.0 / 60.0, 1);

        assert!(world.joints.get(joint).expect("live joint").enabled);
        assert!(world
            .assemblies
            .connections()
            .any(|entry| entry.id == connection));
        assert_eq!(world.assemblies.connected_components().len(), 1);
    }

    #[test]
    fn stale_joint_generation_cannot_break_a_recycled_slot() {
        let (mut world, a, b, stale, _connection) =
            breakable_distance_world(BreakLimit::new(0.5, 0.5));
        assert!(world.joints.remove(stale));
        let replacement = world.joints.insert(JointDesc::distance(
            a,
            Vec3::ZERO,
            b,
            Vec3::ZERO,
            DistanceJoint::rigid(1.0),
        ));
        assert_eq!(replacement.index(), stale.index());
        assert_ne!(replacement.generation(), stale.generation());

        let stale_load = SolverJointLoad {
            joint: stale,
            load: JointLoad {
                linear_impulse: Vec3::X,
                angular_impulse: Vec3::ZERO,
                substep_dt: 1.0,
            },
        };
        assert_eq!(
            world.submit_joint_loads(1.0, &[stale_load]),
            Err(BoundaryLoadError::UnknownJoint)
        );
        assert!(
            world
                .joints
                .get(replacement)
                .expect("recycled slot is live")
                .enabled
        );
    }

    #[test]
    fn break_of_collapsed_assembly_splits_proxy_then_disconnects_edge() {
        let (mut world, a, b, joint, connection) =
            breakable_distance_world(BreakLimit::new(0.5, 0.5));
        let proxy = world
            .merge_assembly(&[a, b])
            .expect("connected dynamic assembly can merge");
        assert!(world.bodies.contains(proxy));

        let events = world
            .submit_joint_loads(
                1.0,
                &[SolverJointLoad {
                    joint,
                    load: JointLoad {
                        linear_impulse: Vec3::X,
                        angular_impulse: Vec3::ZERO,
                        substep_dt: 1.0,
                    },
                }],
            )
            .expect("valid policy load");

        assert_eq!(events.len(), 1);
        assert!(!world.bodies.contains(proxy));
        assert!(!world.joints.get(joint).expect("live joint").enabled);
        assert!(world
            .assemblies
            .connections()
            .all(|entry| entry.id != connection));
        assert_eq!(world.assemblies.connected_components().len(), 2);
    }

    fn articulation_for_world() -> Articulation {
        let mut builder =
            ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default());
        builder
            .add_link(
                None,
                SpatialTransform::IDENTITY,
                SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::splat(0.2)),
                ArticulationJoint::revolute(Vec3::Z),
            )
            .expect("valid articulation link");
        builder.build().expect("valid articulation")
    }

    #[test]
    fn articulation_fixed_step_consumes_contact_impulse_once() {
        let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
        let handle = world.spawn_articulation(articulation_for_world());
        assert!(world.queue_articulation_contact_impulse(
            handle,
            LinkContactImpulse {
                link: 0,
                point: Vec3::X,
                impulse: Vec3::Y,
            },
        ));
        let first = world.step_articulations_substep(1.0 / 60.0);
        assert_eq!(first.stepped, 1);
        assert_eq!(first.contact_impulses, 1);
        assert!(world.articulations.get(handle).unwrap().velocities()[0].abs() > 1e-5);
        let second = world.step_articulations_substep(1.0 / 60.0);
        assert_eq!(second.contact_impulses, 0);
    }

    #[test]
    fn articulation_break_is_emitted_and_stops_aba_progression() {
        let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
        let handle = world.spawn_articulation(articulation_for_world());
        assert!(world.request_articulation_break(handle, 0));
        let report = world.step_articulations_substep(1.0 / 60.0);
        assert_eq!(report.degradations, 1);
        assert!(report.reports[0].1.degraded);
        let events = world.drain_articulation_degradations();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].articulation, handle);
        assert_eq!(events[0].degradation.broken_link, 0);
        assert!(world.drain_articulation_degradations().is_empty());
    }

    #[test]
    fn xpbd_fixed_substep_dispatches_world_articulation_bridge() {
        let mut world = PhysicsWorld::new(WorldConfig::with_gravity(Vec3::ZERO));
        let handle = world.spawn_articulation(articulation_for_world());
        assert!(world.queue_articulation_contact_impulse(
            handle,
            LinkContactImpulse {
                link: 0,
                point: Vec3::X,
                impulse: Vec3::Y,
            },
        ));

        let mut solver = XpbdSolver::new();
        solver.step(&mut world, 1.0 / 60.0, 1);

        let articulation = world.articulations.get(handle).expect("live articulation");
        assert!(articulation.velocities()[0].abs() > 1.0e-5);
    }
}
