//! Native ECS ownership for world-owned reduced-coordinate articulations.
//!
//! Add `(PrismPhysicsPlugin, articulation::PrismArticulationPlugin)` to the app
//! using this crate's public `articulation` module.
//! Alternatively call `init(world)` during the physics plugin's build and put
//! `sync` in PreSync before Sync and `writeback` in PostWriteback after Writeback
//! in the fixed schedule.
//! Use exactly one registration path. XPBD already advances articulations once
//! per physical substep; this module must never step them a second time.
//!
//! Authoring uses physics-space numerical types, not entity Transform/ChildOf.
//! Edits replace the complete runtime with the authored initial state. Unchanged
//! authoring preserves live resource-side edits, queued impulses and breakage.
//! Cached ECS change ticks skip deep authoring clones/comparisons on unchanged
//! ticks; equal-value authoring writes preserve the current runtime as well.
//! Removing and readding authoring between ticks starts a new lifetime. All
//! mappings and output are reconciled at fixed boundaries, not on ECS mutation.
//! Checked publication counters retire exhausted handles; fixed-tick exhaustion
//! stops this bridge permanently. Neither counter wraps or republishes zero.
//! Public resources are reconstructible views, not ownership/counter authority.
//! While the shared world is absent, output/identity views are cleared but private
//! ownership is retained for cleanup or resumption when that same world returns.
//! Observer cleanup uses a finite entry workset and attempts each snapshot removal
//! at most once per cleanup. Residual output is untrusted and
//! reports LifecycleDeferred; publication stops and managed runtimes are retired
//! (or queued for retirement while the shared world is absent). A later boundary
//! retries cleanup. Callbacks must themselves return; recursive user callbacks
//! cannot be bounded by this bridge. Missing public views are reconstructed from
//! private state, including diagnostics, before the next bridge-owned snapshot
//! mutation.
//! Direct world mutations must retain the world's generational storage: replacing
//! the entire storage/world with one that aliases existing handles is unsupported.
//!
//! Broken trees are handed to the shared maximal-coordinate solver at Writeback.
//! Their independent, unparented body entities use the ordinary rigid-body input,
//! contact and Transform/Velocity output paths in the COM/principal-inertia frame.
//! Each mapping also owns an authored_entity child with immutable local transform
//! inverse(link_from_body). Its GlobalTransform preserves the original authored
//! link/base world frame for visuals and attachments, not physical inputs.
//! Removing/reparenting/editing this child frame, or adding RigidBody/Collider to
//! it, retires the group; put user-authored offsets on children of that frame.
//! Root authoring is frozen after handoff. Core requires per-articulation gravity
//! to match world gravity before
//! conversion; a mismatch is diagnosed without converting the source tree.
//! Maximal bodies then use the shared world's gravity/force policy.
//! Root initial generalized positions/velocities are no longer input. Remove and
//! readd root authoring to start a new lifetime. Removing a generated member
//! or changing its immutable RigidBody policy retires the entire owned group.
//! Assembly merge/split of generated members is unsupported: it replaces their
//! body generations without remapping this bridge's private ownership records.
//! Reduced snapshots disappear on success; no fictitious generalized state is
//! published for a maximal tree. Core snapshots cover the resulting bodies and
//! joints, but do not cover ECS topology/ownership. Rolling back across a handoff
//! requires a paired ECS+core checkpoint, not a core-only snapshot restore.

use std::collections::{BTreeMap, BTreeSet};

use glam::{Quat, Vec3};
use prism_app::{App, FixedUpdate, Plugin};
use prism_ecs::prelude::*;
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::reduced::{
    Articulation, ArticulationBase, ArticulationBuildError, ArticulationBuilder,
    ArticulationHandle, ArticulationLink, SpatialInertia, SpatialMotion,
};

use crate::{PhysicsDiagnostics, PhysicsSystems, PhysicsWorldResource};

mod handoff;
pub use handoff::{ArticulationBodyEntity, ArticulationHandoff};

/// Editable model and initial state. No simulated output is written here.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct ArticulationBinding {
    /// Fixed or floating base.
    pub base: ArticulationBase,
    /// Base-frame inertia; a floating base must have usable positive mass.
    pub base_inertia: SpatialInertia,
    /// World-space initial base frame, including for a fixed base.
    pub base_pose: Isometry,
    /// Initial floating-base velocity, expressed in the base frame. Must be
    /// zero for a fixed base.
    pub base_velocity: SpatialMotion,
    /// Per-articulation world gravity; independent of rigid-body world gravity.
    pub gravity: Vec3,
    /// Parent-before-child links. Indices have no implicit gameplay meaning.
    pub links: Vec<ArticulationLink>,
    /// Initial generalized positions. Empty means all zeros; otherwise the
    /// length must equal the sum of the links' joint DOFs.
    pub positions: Vec<f32>,
    /// Initial generalized velocities, with the same sizing rule as positions.
    pub velocities: Vec<f32>,
}

impl Default for ArticulationBinding {
    fn default() -> Self {
        Self::new(ArticulationBase::Fixed, SpatialInertia::default())
    }
}

impl ArticulationBinding {
    /// Starts an empty tree at the origin with zero gravity and velocity.
    #[must_use]
    pub fn new(base: ArticulationBase, base_inertia: SpatialInertia) -> Self {
        Self {
            base,
            base_inertia,
            base_pose: Isometry::IDENTITY,
            base_velocity: SpatialMotion::default(),
            gravity: Vec3::ZERO,
            links: Vec::new(),
            positions: Vec::new(),
            velocities: Vec::new(),
        }
    }

    /// Validates authoring and builds a fresh core model, without mutating ECS
    /// or world storage. Builder errors retain their core link/parent details.
    pub fn build(&self) -> Result<Articulation, ArticulationIssue> {
        let invalid = |field, link| ArticulationIssue::InvalidAuthoring { field, link };
        if !valid_pose(self.base_pose) {
            return Err(invalid("base_pose", None));
        }
        if !valid_motion(self.base_velocity)
            || (self.base == ArticulationBase::Fixed
                && self.base_velocity != SpatialMotion::default())
        {
            return Err(invalid("base_velocity", None));
        }
        if !self.gravity.is_finite() {
            return Err(invalid("gravity", None));
        }
        if !self.base_inertia.mass.is_finite()
            || self.base_inertia.mass < 0.0
            || !self.base_inertia.center_of_mass.is_finite()
            || !self
                .base_inertia
                .inertia_com
                .iter()
                .flatten()
                .all(|x| x.is_finite())
        {
            return Err(invalid("base_inertia", None));
        }
        let mut builder = ArticulationBuilder::new(self.base, self.base_inertia)
            .with_base_pose(self.base_pose)
            .with_gravity(self.gravity);
        for (index, link) in self.links.iter().enumerate() {
            if !valid_rotation(link.x_tree.rotation) || !link.x_tree.translation.is_finite() {
                return Err(invalid("x_tree", Some(index)));
            }
            let joint = link.joint;
            if !joint.damping.is_finite()
                || joint.damping < 0.0
                || !joint.armature.is_finite()
                || joint.armature < 0.0
            {
                return Err(invalid("joint_dynamics", Some(index)));
            }
            if let Some(motor) = joint.motor {
                if !motor.target_position.is_finite()
                    || !motor.target_velocity.is_finite()
                    || !motor.stiffness.is_finite()
                    || motor.stiffness < 0.0
                    || !motor.damping.is_finite()
                    || motor.damping < 0.0
                    || motor
                        .max_force
                        .to_array()
                        .iter()
                        .any(|x| x.is_nan() || *x < 0.0)
                {
                    return Err(invalid("motor", Some(index)));
                }
            }
            builder
                .add_link(link.parent, link.x_tree, link.inertia, joint)
                .map_err(ArticulationIssue::BuildFailed)?;
        }
        let mut model = builder.build().map_err(ArticulationIssue::BuildFailed)?;
        for (field, values) in [
            ("positions", &self.positions),
            ("velocities", &self.velocities),
        ] {
            if (!values.is_empty() && values.len() != model.dof_count())
                || !values.iter().all(|x| x.is_finite())
            {
                return Err(invalid(field, None));
            }
        }
        for (dof, &value) in self.positions.iter().enumerate() {
            model.set_position(dof, value);
        }
        for (dof, &value) in self.velocities.iter().enumerate() {
            model.set_velocity(dof, value);
        }
        if self.base == ArticulationBase::Floating {
            model.set_base_velocity(self.base_velocity);
        }
        // Catch finite inputs whose derived transforms/velocities overflow.
        if ArticulationSnapshot::capture(ArticulationHandle::INVALID, &model, 0, 0, 0.0).is_none() {
            return Err(invalid("initial_state", None));
        }
        Ok(model)
    }
}

/// Versioned output; link columns follow authored link order. This component
/// is bridge-owned and is removed when its binding is retired or invalidated.
/// Observer interference may defer physical removal: `LifecycleDeferred` and
/// empty public ownership maps mark residual components as untrusted output.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct ArticulationSnapshot {
    /// Full generational identity; versions restart when this handle changes.
    pub handle: ArticulationHandle,
    /// Publication sequence for this handle. Exhaustion retires the handle
    /// rather than publishing a wrapped version. Reserved values may be skipped
    /// when observer interference cancels a buffered publication.
    pub version: u64,
    /// Bridge fixed tick that published this state.
    pub tick: u64,
    /// Accepted physics time this tick; zero when the shared step was rejected.
    pub simulated_dt: f32,
    /// World base frame.
    pub base_pose: Isometry,
    /// Base-frame spatial velocity.
    pub base_velocity: SpatialMotion,
    /// Joint generalized positions.
    pub positions: Vec<f32>,
    /// Joint generalized velocities.
    pub velocities: Vec<f32>,
    /// World link frames.
    pub link_poses: Vec<Isometry>,
    /// Link-frame spatial velocities (not world-space velocities).
    pub link_velocities: Vec<SpatialMotion>,
    /// Broken joint, if ABA has stopped and automatic handoff has not succeeded.
    pub broken_joint: Option<usize>,
}

impl ArticulationSnapshot {
    fn capture(
        handle: ArticulationHandle,
        model: &Articulation,
        version: u64,
        tick: u64,
        dt: f32,
    ) -> Option<Self> {
        // Reject corrupt state before kinematics attempts quaternion operations
        // that may assert on NaN inputs when glam assertions are enabled.
        if !valid_pose(model.base_pose())
            || !valid_motion(model.base_velocity())
            || !model
                .positions()
                .iter()
                .chain(model.velocities())
                .all(|value| value.is_finite())
        {
            return None;
        }
        let snapshot = Self {
            handle,
            version,
            tick,
            simulated_dt: dt,
            base_pose: model.base_pose(),
            base_velocity: model.base_velocity(),
            positions: model.positions().to_vec(),
            velocities: model.velocities().to_vec(),
            link_poses: model.forward_kinematics(),
            link_velocities: model.link_velocities(),
            broken_joint: model.broken_joint(),
        };
        snapshot.valid().then_some(snapshot)
    }

    fn valid(&self) -> bool {
        valid_pose(self.base_pose)
            && valid_motion(self.base_velocity)
            && self.positions.len() == self.velocities.len()
            && self
                .positions
                .iter()
                .chain(&self.velocities)
                .all(|x| x.is_finite())
            && self.link_poses.len() == self.link_velocities.len()
            && self.link_poses.iter().copied().all(valid_pose)
            && self.link_velocities.iter().copied().all(valid_motion)
            && self.link_velocities.iter().copied().all(valid_motion)
    }
}

fn valid_rotation(rotation: Quat) -> bool {
    rotation.is_finite() && (rotation.length_squared() - 1.0).abs() <= 1.0e-4
}

fn valid_pose(pose: Isometry) -> bool {
    pose.translation.is_finite() && valid_rotation(pose.rotation)
}

fn valid_motion(motion: SpatialMotion) -> bool {
    motion.angular.is_finite() && motion.linear.is_finite()
}

struct RuntimeBinding {
    handle: ArticulationHandle,
    authored: ArticulationBinding,
    authored_tick: Tick,
    version: u64,
    // Remains true even after member invalidation: never resurrect ABA from
    // unchanged authoring (or edits) within this authoring lifetime.
    maximal: bool,
    handoff: Option<handoff::OwnedHandoff>,
}

// Public view removal/replacement must not orphan core models, reset versions,
// or revive an exhausted bridge. Only this private resource owns those facts.
#[derive(Resource, Default)]
struct ArticulationRuntime {
    bindings: BTreeMap<Entity, RuntimeBinding>,
    tick: u64,
    tick_exhausted: bool,
    publish_pending: bool,
    publication_blocked: bool,
    diagnostics: ArticulationDiagnostics,
    retired_handoffs: Vec<handoff::OwnedHandoff>,
    despawn_pending: BTreeSet<Entity>,
    handoff_issues: Vec<ArticulationDiagnostic>,
}

/// Bidirectional identity, including both ECS and physics generations. Lookup
/// is a last-fixed-boundary view; it does not validate external world mutations.
/// Removal/replacement is repaired at the next bridge boundary, without resetting
/// the privately owned runtimes or their publication sequences.
#[derive(Resource, Default)]
pub struct ArticulationBindings {
    bindings: BTreeMap<Entity, ArticulationHandle>,
    entities: BTreeMap<(u32, u32), Entity>,
    handoffs: BTreeMap<Entity, ArticulationHandoff>,
    tick: u64,
}

impl ArticulationBindings {
    /// Live maximal-coordinate ownership after a successful handoff. The source
    /// articulation handle is lineage only, not a live reduced-coordinate handle.
    #[must_use]
    pub fn handoff(&self, entity: Entity) -> Option<&ArticulationHandoff> {
        self.handoffs.get(&entity)
    }

    /// Last synchronized articulation for the complete ECS entity identity.
    #[must_use]
    pub fn articulation(&self, entity: Entity) -> Option<ArticulationHandle> {
        self.bindings.get(&entity).copied()
    }

    /// Managed ECS owner, never an unmanaged articulation reusing a slot.
    #[must_use]
    pub fn entity(&self, handle: ArticulationHandle) -> Option<Entity> {
        self.entities
            .get(&(handle.index(), handle.generation()))
            .copied()
    }
}

/// Explicit authoring, recovery and simulation failure classifications.
#[derive(Clone, Debug, PartialEq)]
pub enum ArticulationIssue {
    /// Bad numerical input or state sizing. `link` identifies a link-local field.
    InvalidAuthoring {
        /// Authored field that failed validation.
        field: &'static str,
        /// Authored link index, or None for articulation-wide input.
        link: Option<usize>,
    },
    /// Core model validation failure, retaining its exact cause.
    BuildFailed(
        /// Core topology, inertia, or joint validation error.
        ArticulationBuildError,
    ),
    /// Externally removed handle; sync recreates it from authored initial state.
    StaleHandle(
        /// Complete generational identity that is no longer live.
        ArticulationHandle,
    ),
    /// Non-finite/invalid runtime output; the bridge retires this runtime rather
    /// than publishing it. The next sync retries from validated authoring.
    InvalidOutput(
        /// Retired runtime whose state could not be published safely.
        ArticulationHandle,
    ),
    /// ABA stopped by core breakage. Automatic conversion is attempted without
    /// draining world-global degradation events. Reduced output remains only
    /// when conversion fails; successful conversion publishes a handoff view.
    Degraded {
        /// Source articulation being handed off.
        handle: ArticulationHandle,
        /// Broken joint's authored link index.
        link: usize,
    },
    /// Atomic core conversion was rejected; the stopped reduced model and its
    /// attachments are retained and conversion is retried at a later boundary.
    HandoffFailed(
        /// Exact core policy/state error.
        prism_physics_core::reduced::articulation::handoff::ArticulationHandoffError,
    ),
    /// A generated member was removed, replaced, or changed immutable policy.
    /// Its whole maximal group is retired; root remove/readd is required to restart.
    HandoffInvalidated,
    /// Shared physics rejected this tick's step; snapshots contain unstepped state.
    StepRejected,
    /// The base physics plugin/resources have not been installed.
    MissingPhysics,
    /// A snapshot observer removed/replaced a required resource. Public bridge views are
    /// reconstructed; missing shared physics cancels publication for this boundary.
    ObserverResourceRemoved {
        /// Resource type removed during a bridge-owned snapshot mutation.
        resource: &'static str,
    },
    /// Observer interference or an absent shared world deferred lifecycle cleanup.
    /// Residual snapshots are not trusted output: publication is blocked and
    /// managed runtimes are retired, or queued until the shared world returns.
    LifecycleDeferred,
    /// A checked counter cannot advance. With an entity, its publication
    /// sequence is exhausted and the runtime is retired; sync may rebuild it
    /// under a new handle. Without an entity, the fixed tick is exhausted and
    /// this bridge permanently stops managing/publishing articulations.
    SequenceExhausted,
}

/// One failure/recovery entry for the latest fixed tick.
#[derive(Clone, Debug, PartialEq)]
pub struct ArticulationDiagnostic {
    /// None for shared-world configuration/integration failures.
    pub entity: Option<Entity>,
    /// Detailed cause.
    pub issue: ArticulationIssue,
}

/// Latest fixed tick only; no unbounded diagnostic history.
#[derive(Resource, Default, Clone)]
pub struct ArticulationDiagnostics {
    /// Attempted bridge sync count. Stops at u64::MAX; further attempts fail
    /// closed with `SequenceExhausted` instead of reusing a tick.
    /// This is output: editing/removing it does not reset the private counter.
    pub tick: u64,
    /// Accepted time from the shared physics step on this tick.
    pub simulated_dt: f32,
    /// Failures and recoveries during sync/writeback.
    pub issues: Vec<ArticulationDiagnostic>,
}

impl ArticulationDiagnostics {
    fn issue(&mut self, entity: Option<Entity>, issue: ArticulationIssue) {
        let entry = ArticulationDiagnostic { entity, issue };
        if !self.issues.contains(&entry) {
            self.issues.push(entry);
        }
    }
}

#[derive(Resource, Default)]
struct RemovedAuthoring(BTreeSet<Entity>);

/// Initializes resources and the remove/readd observer exactly once. Does not
/// initialize/replace the shared physics world or register any step system.
pub(crate) fn init(world: &mut World) {
    world.init_resource::<ArticulationRuntime>();
    world.init_resource::<ArticulationBindings>();
    world.init_resource::<ArticulationDiagnostics>();
    if !world.contains_resource::<RemovedAuthoring>() {
        world.init_resource::<RemovedAuthoring>();
        world.observe::<ArticulationBinding, _>(LifecycleEvent::Remove, |ctx| {
            ctx.world
                .resource_mut::<RemovedAuthoring>()
                .0
                .insert(ctx.entity);
        });
    }
}

fn retire(physics: &mut PhysicsWorldResource, runtime: &mut ArticulationRuntime, entity: Entity) {
    if let Some(binding) = runtime.bindings.remove(&entity) {
        // Core removal checks the entire generation, so a reused foreign slot
        // cannot be removed through this stale ownership record.
        if let Some(owned) = binding.handoff {
            runtime.retired_handoffs.push(owned);
        } else if !binding.maximal {
            physics.remove_articulation(binding.handle);
        }
    }
}

fn restore_views(
    world: &mut World,
    mut runtime: ArticulationRuntime,
    mut diagnostics: ArticulationDiagnostics,
    available: bool,
) {
    for entry in std::mem::take(&mut runtime.handoff_issues) {
        diagnostics.issue(entry.entity, entry.issue);
    }
    let mut bindings = world
        .remove_resource::<ArticulationBindings>()
        .unwrap_or_default();
    bindings.bindings.clear();
    bindings.entities.clear();
    bindings.handoffs.clear();
    bindings.tick = runtime.tick;
    if available && !runtime.tick_exhausted && !runtime.publication_blocked {
        for (&entity, binding) in &runtime.bindings {
            if binding.maximal {
                if let Some(owned) = &binding.handoff {
                    bindings.handoffs.insert(entity, owned.view.clone());
                }
                continue;
            }
            bindings.bindings.insert(entity, binding.handle);
            bindings.entities.insert(
                (binding.handle.index(), binding.handle.generation()),
                entity,
            );
        }
    }
    diagnostics.tick = runtime.tick;
    runtime.diagnostics = diagnostics.clone();
    world.insert_resource(runtime);
    world.insert_resource(bindings);
    world.insert_resource(diagnostics);
}

// Diagnostics are also an output view. An observer deleting/replacing that view
// must not erase the failures already recorded during the current boundary.
fn take_diagnostics(
    world: &mut World,
    runtime: &mut ArticulationRuntime,
) -> ArticulationDiagnostics {
    world.remove_resource::<ArticulationDiagnostics>();
    std::mem::take(&mut runtime.diagnostics)
}

fn observer_resources(world: &World) -> [(&'static str, bool); 4] {
    [
        (
            "PhysicsWorldResource",
            world.contains_resource::<PhysicsWorldResource>(),
        ),
        (
            "PhysicsDiagnostics",
            world.contains_resource::<PhysicsDiagnostics>(),
        ),
        (
            "ArticulationBindings",
            world.contains_resource::<ArticulationBindings>(),
        ),
        (
            "ArticulationDiagnostics",
            world.contains_resource::<ArticulationDiagnostics>(),
        ),
    ]
}

fn after_snapshot_observers(world: &mut World, before: [(&'static str, bool); 4]) {
    let after = observer_resources(world);
    let tick = world.resource::<ArticulationRuntime>().tick;
    let bindings_intact = world
        .get_resource::<ArticulationBindings>()
        .is_some_and(|view| view.tick == tick);
    let diagnostics_intact = world
        .get_resource::<ArticulationDiagnostics>()
        .is_some_and(|view| view.tick == tick);
    // The common publication path must remain O(1) per callback, not rebuild
    // every public mapping for each of N output components (O(N^2)).
    if before == after && bindings_intact && diagnostics_intact {
        return;
    }
    let mut runtime = world.remove_resource::<ArticulationRuntime>().unwrap();
    let mut diagnostics = take_diagnostics(world, &mut runtime);
    for ((resource, was_present), (_, present)) in before.into_iter().zip(after) {
        if was_present && !present {
            diagnostics.issue(
                None,
                ArticulationIssue::ObserverResourceRemoved { resource },
            );
        }
    }
    for (resource, intact) in [
        ("ArticulationBindings", bindings_intact),
        ("ArticulationDiagnostics", diagnostics_intact),
    ] {
        if !intact {
            diagnostics.issue(
                None,
                ArticulationIssue::ObserverResourceRemoved { resource },
            );
        }
    }
    let available = world.contains_resource::<PhysicsWorldResource>()
        && world.contains_resource::<PhysicsDiagnostics>();
    if !available {
        runtime.publish_pending = false;
        runtime.publication_blocked = true;
        diagnostics.simulated_dt = 0.0;
        diagnostics.issue(None, ArticulationIssue::MissingPhysics);
    }
    restore_views(world, runtime, diagnostics, available);
}

fn snapshot_entities(world: &mut World) -> Vec<Entity> {
    world
        .query::<(Entity, &ArticulationSnapshot)>()
        .iter(world)
        .map(|(entity, _)| entity)
        .collect()
}

fn retire_removed(
    world: &mut World,
    physics: &mut PhysicsWorldResource,
    runtime: &mut ArticulationRuntime,
) -> BTreeSet<Entity> {
    let removed = std::mem::take(&mut world.resource_mut::<RemovedAuthoring>().0);
    let entities: Vec<_> = runtime.bindings.keys().copied().collect();
    let mut clear = BTreeSet::new();
    for entity in entities {
        if runtime.tick_exhausted
            || removed.contains(&entity)
            || world.get::<ArticulationBinding>(entity).is_none()
        {
            retire(physics, runtime, entity);
            clear.insert(entity);
        }
    }
    handoff::reconcile(world, physics, runtime);
    clear
}

/// Reconciles complete authoring and lifecycle before the existing XPBD step.
pub(crate) fn sync(world: &mut World) {
    let mut runtime = world.remove_resource::<ArticulationRuntime>().unwrap();
    let mut diagnostics = take_diagnostics(world, &mut runtime);
    runtime.publish_pending = false;
    runtime.publication_blocked = false;
    diagnostics.simulated_dt = 0.0;
    diagnostics.issues.clear();
    if !runtime.tick_exhausted {
        if let Some(tick) = runtime.tick.checked_add(1) {
            runtime.tick = tick;
        } else {
            runtime.tick_exhausted = true;
        }
    }
    if runtime.tick_exhausted {
        diagnostics.issue(None, ArticulationIssue::SequenceExhausted);
    }
    let Some(mut physics) = world.remove_resource::<PhysicsWorldResource>() else {
        runtime.publication_blocked = true;
        diagnostics.issue(None, ArticulationIssue::MissingPhysics);
        restore_views(world, runtime, diagnostics, false);
        settle_removals(world, BTreeSet::new());
        return;
    };
    let mut clear = retire_removed(world, &mut physics, &mut runtime);
    let mut authored: Vec<_> = if runtime.tick_exhausted {
        Vec::new()
    } else {
        world
            .query::<(Entity, Ref<ArticulationBinding>)>()
            .iter(world)
            .map(|(entity, config)| (entity, config.changed_tick()))
            .collect()
    };
    authored.sort_by_key(|(entity, _)| *entity);
    for (entity, authored_tick) in authored {
        if let Some(binding) = runtime.bindings.get_mut(&entity) {
            if binding.maximal {
                continue;
            }
            if binding.version == u64::MAX {
                // Retire before XPBD can advance a runtime that can no longer
                // publish. Rebuild on the following sync, not in this tick.
                diagnostics.issue(Some(entity), ArticulationIssue::SequenceExhausted);
                retire(&mut physics, &mut runtime, entity);
                clear.insert(entity);
                continue;
            }
            let stale = !physics.articulations.contains(binding.handle);
            if stale {
                diagnostics.issue(Some(entity), ArticulationIssue::StaleHandle(binding.handle));
            } else {
                binding.authored_tick.check_tick(world.change_tick());
                // Unchanged ticks never clone or compare deep authoring state.
                // Equal-value writes still preserve the live runtime.
                if binding.authored_tick == authored_tick {
                    continue;
                }
                if &binding.authored == world.get::<ArticulationBinding>(entity).unwrap() {
                    binding.authored_tick = authored_tick;
                    continue;
                }
            }
        }
        let config = world.get::<ArticulationBinding>(entity).unwrap().clone();
        // Invalid edits retire the previous simulation too: it must not keep
        // running invisibly behind rejected authoring.
        let candidate = config.build();
        retire(&mut physics, &mut runtime, entity);
        clear.insert(entity);
        match candidate {
            Ok(model) => {
                let handle = physics.spawn_articulation(model);
                runtime.bindings.insert(
                    entity,
                    RuntimeBinding {
                        handle,
                        authored: config,
                        authored_tick,
                        version: 0,
                        maximal: false,
                        handoff: None,
                    },
                );
            }
            Err(error) => diagnostics.issue(Some(entity), error),
        }
    }
    // Also clear orphaned output, including output left by deleted authoring
    // before its very first synchronization.
    for (entity, _) in world.query::<(Entity, &ArticulationSnapshot)>().iter(world) {
        if !runtime.bindings.contains_key(&entity) {
            clear.insert(entity);
        }
    }
    runtime.publish_pending = !runtime.tick_exhausted;
    world.insert_resource(physics);
    restore_views(world, runtime, diagnostics, true);
    settle_removals(world, clear);
}

/// Publishes completed shared-world state; never advances simulation or drains
/// world-global articulation degradation events owned by other consumers.
pub(crate) fn writeback(world: &mut World) {
    let mut runtime = world.remove_resource::<ArticulationRuntime>().unwrap();
    let mut diagnostics = take_diagnostics(world, &mut runtime);
    diagnostics.tick = runtime.tick;
    let dt = world
        .get_resource::<PhysicsDiagnostics>()
        .map(|d| d.simulated_dt);
    let Some(mut physics) = world.remove_resource::<PhysicsWorldResource>() else {
        runtime.publish_pending = false;
        runtime.publication_blocked = true;
        diagnostics.simulated_dt = 0.0;
        diagnostics.issue(None, ArticulationIssue::MissingPhysics);
        restore_views(world, runtime, diagnostics, false);
        settle_removals(world, BTreeSet::new());
        return;
    };
    // Lifecycle is reconciled even when there is no accepted/shared step or
    // sync could not run. An inactive runtime must not outlive its ECS owner.
    let mut clear = retire_removed(world, &mut physics, &mut runtime);
    let publish = std::mem::take(&mut runtime.publish_pending)
        && !runtime.tick_exhausted
        && !runtime.publication_blocked;
    if runtime.tick_exhausted
        && !diagnostics.issues.iter().any(|entry| {
            entry.entity.is_none() && entry.issue == ArticulationIssue::SequenceExhausted
        })
    {
        diagnostics.issue(None, ArticulationIssue::SequenceExhausted);
    }
    if dt.is_none() {
        runtime.publication_blocked = true;
        diagnostics.simulated_dt = 0.0;
        diagnostics.issue(None, ArticulationIssue::MissingPhysics);
    }
    if !publish || dt.is_none() {
        if dt.is_none() || runtime.tick_exhausted {
            clear.extend(snapshot_entities(world));
        }
        world.insert_resource(physics);
        restore_views(world, runtime, diagnostics, true);
        settle_removals(world, clear);
        return;
    }
    let raw_dt = dt.unwrap();
    let dt = if raw_dt.is_finite() && raw_dt > 0.0 {
        raw_dt
    } else {
        diagnostics.issue(None, ArticulationIssue::StepRejected);
        0.0
    };
    diagnostics.simulated_dt = dt;
    let mut output = Vec::new();
    let entities: Vec<_> = runtime.bindings.keys().copied().collect();
    for entity in entities {
        let binding = runtime.bindings.get_mut(&entity).unwrap();
        if binding.maximal {
            continue;
        }
        let handle = binding.handle;
        let Some(version) = binding.version.checked_add(1) else {
            diagnostics.issue(Some(entity), ArticulationIssue::SequenceExhausted);
            retire(&mut physics, &mut runtime, entity);
            clear.insert(entity);
            continue;
        };
        let Some(model) = physics.articulations.get(handle) else {
            diagnostics.issue(Some(entity), ArticulationIssue::StaleHandle(handle));
            retire(&mut physics, &mut runtime, entity);
            clear.insert(entity);
            continue;
        };
        let snapshot = ArticulationSnapshot::capture(handle, model, version, diagnostics.tick, dt);
        let Some(snapshot) = snapshot else {
            diagnostics.issue(Some(entity), ArticulationIssue::InvalidOutput(handle));
            retire(&mut physics, &mut runtime, entity);
            clear.insert(entity);
            continue;
        };
        if let Some(link) = snapshot.broken_joint {
            diagnostics.issue(Some(entity), ArticulationIssue::Degraded { handle, link });
            match handoff::convert(world, &mut physics, entity, handle) {
                Ok(owned) => {
                    binding.maximal = true;
                    binding.handoff = Some(owned);
                    clear.insert(entity);
                    continue;
                }
                Err(error) => {
                    diagnostics.issue(Some(entity), ArticulationIssue::HandoffFailed(error))
                }
            }
        }
        binding.version = snapshot.version;
        output.push((entity, snapshot));
    }
    world.insert_resource(physics);
    restore_views(world, runtime, diagnostics, true);
    handoff::materialize(world);
    settle_removals(world, clear);
    for (entity, snapshot) in output {
        if world.resource::<ArticulationRuntime>().publication_blocked {
            break;
        }
        // A preceding snapshot observer may despawn/remove/readd another
        // entity. Presence alone does not establish the same authoring lifetime.
        if world.get::<ArticulationBinding>(entity).is_some()
            && !world.resource::<RemovedAuthoring>().0.contains(&entity)
            && world
                .get_resource::<PhysicsWorldResource>()
                .is_some_and(|physics| physics.articulations.contains(snapshot.handle))
            && world
                .resource::<ArticulationRuntime>()
                .bindings
                .get(&entity)
                .is_some_and(|binding| !binding.maximal && binding.handle == snapshot.handle)
        {
            let before = observer_resources(world);
            world.insert(entity, (snapshot,));
            after_snapshot_observers(world, before);
        }
    }
    settle_removals(world, BTreeSet::new());
}

fn invalid_snapshot(world: &World, entity: Entity) -> bool {
    let runtime = world.resource::<ArticulationRuntime>();
    runtime.publication_blocked
        || runtime.tick_exhausted
        || world.resource::<RemovedAuthoring>().0.contains(&entity)
        || world.get::<ArticulationBinding>(entity).is_none()
        || world
            .get::<ArticulationSnapshot>(entity)
            .is_some_and(|snapshot| {
                !runtime
                    .bindings
                    .get(&entity)
                    .is_some_and(|binding| !binding.maximal && binding.handle == snapshot.handle)
            })
}

// No new private owners are created here: N+1 passes allow ordinary owner-removal
// cascades to drain. Regardless of observer behavior, that loop bound is fixed.
// The immutable entry workset and attempted set additionally cap snapshot removal
// attempts at M, even if an observer keeps creating/reintroducing orphan output.
// Residual work fails closed without further callbacks. Resource recovery has no
// component callbacks. None of these bounds can force a user callback to return.
fn settle_removals(world: &mut World, forced: BTreeSet<Entity>) {
    let owners = world.resource::<ArticulationRuntime>().bindings.len();
    let mut candidates: BTreeSet<_> = snapshot_entities(world).into_iter().collect();
    candidates.extend(
        world
            .resource::<ArticulationRuntime>()
            .bindings
            .keys()
            .copied(),
    );
    candidates.extend(forced.iter().copied());
    let mut attempted = BTreeSet::new();
    for _ in 0..=owners {
        let mut runtime = world.remove_resource::<ArticulationRuntime>().unwrap();
        let mut diagnostics = take_diagnostics(world, &mut runtime);
        let available = world.contains_resource::<PhysicsWorldResource>()
            && world.contains_resource::<PhysicsDiagnostics>();
        if !available {
            runtime.publish_pending = false;
            runtime.publication_blocked = true;
            diagnostics.simulated_dt = 0.0;
            diagnostics.issue(None, ArticulationIssue::MissingPhysics);
        }
        if let Some(mut physics) = world.remove_resource::<PhysicsWorldResource>() {
            retire_removed(world, &mut physics, &mut runtime);
            world.insert_resource(physics);
        }
        restore_views(world, runtime, diagnostics, available);
        let clear: Vec<_> = candidates
            .iter()
            .copied()
            .filter(|entity| {
                !attempted.contains(entity)
                    && (forced.contains(entity) || invalid_snapshot(world, *entity))
            })
            .collect();
        for entity in clear {
            attempted.insert(entity);
            let before = observer_resources(world);
            world.remove::<ArticulationSnapshot>(entity);
            after_snapshot_observers(world, before);
        }
        handoff::despawn_retired(world);
        if !world.contains_resource::<PhysicsWorldResource>()
            || world.resource::<RemovedAuthoring>().0.is_empty()
        {
            // A callback can make additional entry snapshots invalid after the
            // clear list was captured. The final bounded sweep handles those.
            break;
        }
    }
    // Resource disappearance may invalidate earlier output with no authoring
    // removal. Do not retry anything already attempted or chase new entities.
    for entity in candidates {
        if !attempted.contains(&entity) && invalid_snapshot(world, entity) {
            attempted.insert(entity);
            let before = observer_resources(world);
            world.remove::<ArticulationSnapshot>(entity);
            after_snapshot_observers(world, before);
        }
    }
    let residual: Vec<_> = snapshot_entities(world)
        .into_iter()
        .filter(|entity| forced.contains(entity) || invalid_snapshot(world, *entity))
        .collect();
    let pending = !world.resource::<RemovedAuthoring>().0.is_empty();
    if !residual.is_empty() || pending {
        let mut runtime = world.remove_resource::<ArticulationRuntime>().unwrap();
        let mut diagnostics = take_diagnostics(world, &mut runtime);
        diagnostics.issue(None, ArticulationIssue::LifecycleDeferred);
        // Fail closed without invoking any more component observers. If the
        // world is absent these records remain queued, never silently dropped.
        runtime.publish_pending = false;
        runtime.publication_blocked = true;
        diagnostics.simulated_dt = 0.0;
        let entities: Vec<_> = runtime.bindings.keys().copied().collect();
        if let Some(mut physics) = world.remove_resource::<PhysicsWorldResource>() {
            for entity in entities {
                retire(&mut physics, &mut runtime, entity);
            }
            handoff::reconcile(world, &mut physics, &mut runtime);
            world.resource_mut::<RemovedAuthoring>().0.clear();
            world.insert_resource(physics);
        } else {
            world.resource_mut::<RemovedAuthoring>().0.extend(entities);
        }
        restore_views(world, runtime, diagnostics, false);
    }
    handoff::despawn_retired(world);
}

/// Optional registration alongside [`crate::PrismPhysicsPlugin`]. Shares its
/// fixed clock, XPBD solver/substep policy and public ordering anchors.
#[derive(Default)]
pub struct PrismArticulationPlugin;

impl Plugin for PrismArticulationPlugin {
    fn build(&self, app: &mut App) {
        init(app.world_mut());
        app.add_systems(
            FixedUpdate,
            (
                sync.in_set(PhysicsSystems::PreSync)
                    .before(PhysicsSystems::Sync)
                    .before(PhysicsSystems::Step)
                    .before(PhysicsSystems::Writeback),
                writeback
                    .in_set(PhysicsSystems::PostWriteback)
                    .after(PhysicsSystems::Sync)
                    .after(PhysicsSystems::Writeback)
                    .after(PhysicsSystems::Step),
            ),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> (World, Entity) {
        let mut world = World::new();
        world.init_resource::<PhysicsWorldResource>();
        world.insert_resource(PhysicsDiagnostics {
            simulated_dt: 0.01,
            ..Default::default()
        });
        init(&mut world);
        let entity = world.spawn((ArticulationBinding::default(),));
        sync(&mut world);
        writeback(&mut world);
        (world, entity)
    }

    #[test]
    fn publication_exhaustion_retires_only_its_handle_and_recovers_with_new_identity() {
        let (mut world, entity) = world();
        let other = world.spawn((ArticulationBinding::default(),));
        let old = world
            .resource::<ArticulationBindings>()
            .articulation(entity)
            .unwrap();
        world
            .resource_mut::<ArticulationRuntime>()
            .bindings
            .get_mut(&entity)
            .unwrap()
            .version = u64::MAX - 1;
        sync(&mut world);
        writeback(&mut world);
        assert_eq!(
            world.get::<ArticulationSnapshot>(entity).unwrap().version,
            u64::MAX
        );
        let other_handle = world
            .resource::<ArticulationBindings>()
            .articulation(other)
            .unwrap();

        world.observe::<ArticulationSnapshot, _>(LifecycleEvent::Remove, |ctx| {
            assert!(ctx.world.contains_resource::<PhysicsWorldResource>());
            assert!(ctx.world.contains_resource::<ArticulationBindings>());
            assert!(ctx.world.contains_resource::<ArticulationDiagnostics>());
        });
        sync(&mut world);
        assert!(
            !world
                .resource::<PhysicsWorldResource>()
                .articulations
                .contains(old),
            "exhausted runtime must retire before the shared step"
        );
        writeback(&mut world);
        let bindings = world.resource::<ArticulationBindings>();
        assert_eq!(bindings.articulation(entity), None);
        assert_eq!(bindings.entity(old), None);
        assert_eq!(bindings.articulation(other), Some(other_handle));
        assert!(world.get::<ArticulationSnapshot>(entity).is_none());
        assert!(!world
            .resource::<PhysicsWorldResource>()
            .articulations
            .contains(old));
        assert_eq!(world.get::<ArticulationSnapshot>(other).unwrap().version, 2);
        assert_eq!(
            world.resource::<ArticulationDiagnostics>().issues,
            vec![ArticulationDiagnostic {
                entity: Some(entity),
                issue: ArticulationIssue::SequenceExhausted,
            }]
        );

        sync(&mut world);
        writeback(&mut world);
        let snapshot = world.get::<ArticulationSnapshot>(entity).unwrap();
        assert_ne!(snapshot.handle, old);
        assert_eq!(snapshot.version, 1);
        assert_eq!(snapshot.tick, 4);
        assert_eq!(world.get::<ArticulationSnapshot>(other).unwrap().version, 3);
        assert!(world
            .resource::<ArticulationDiagnostics>()
            .issues
            .is_empty());
    }

    #[test]
    fn remove_readd_is_not_lost_while_the_shared_world_is_absent() {
        let (mut world, entity) = world();
        let old = world
            .resource::<ArticulationBindings>()
            .articulation(entity)
            .unwrap();
        let physics = world.remove_resource::<PhysicsWorldResource>().unwrap();
        assert!(world.remove::<ArticulationBinding>(entity));
        world.insert(entity, (ArticulationBinding::default(),));
        sync(&mut world);
        assert!(world.resource::<RemovedAuthoring>().0.contains(&entity));
        assert!(world.get::<ArticulationSnapshot>(entity).is_none());
        assert_eq!(
            world
                .resource::<ArticulationBindings>()
                .articulation(entity),
            None
        );
        assert_eq!(world.resource::<ArticulationBindings>().entity(old), None);
        world.insert_resource(physics);
        sync(&mut world);
        writeback(&mut world);
        let snapshot = world.get::<ArticulationSnapshot>(entity).unwrap();
        assert_ne!(snapshot.handle, old);
        assert_eq!(snapshot.version, 1);
        assert!(!world
            .resource::<PhysicsWorldResource>()
            .articulations
            .contains(old));
    }

    #[test]
    fn fixed_tick_exhaustion_survives_public_view_recreation_and_deferred_cleanup() {
        let (mut world, entity) = world();
        let managed = world
            .resource::<ArticulationBindings>()
            .articulation(entity)
            .unwrap();
        let foreign = world
            .resource_mut::<PhysicsWorldResource>()
            .spawn_articulation(ArticulationBinding::default().build().unwrap());
        world.resource_mut::<ArticulationRuntime>().tick = u64::MAX - 1;
        sync(&mut world);
        writeback(&mut world);
        assert_eq!(
            world.get::<ArticulationSnapshot>(entity).unwrap().tick,
            u64::MAX
        );
        assert_eq!(
            world.get::<ArticulationSnapshot>(entity).unwrap().version,
            2
        );

        let physics = world.remove_resource::<PhysicsWorldResource>().unwrap();
        sync(&mut world);
        writeback(&mut world);
        assert!(world.resource::<ArticulationRuntime>().tick_exhausted);
        assert!(world.get::<ArticulationSnapshot>(entity).is_none());
        assert_eq!(
            world
                .resource::<ArticulationBindings>()
                .articulation(entity),
            None
        );
        assert!(
            physics.articulations.contains(managed),
            "cleanup waits for the same world"
        );

        world.remove_resource::<ArticulationBindings>();
        world.remove_resource::<ArticulationDiagnostics>();
        world.insert_resource(physics);
        // Even writeback without another sync must clean up exhausted state.
        writeback(&mut world);
        assert!(!world
            .resource::<PhysicsWorldResource>()
            .articulations
            .contains(managed));
        assert!(world
            .resource::<PhysicsWorldResource>()
            .articulations
            .contains(foreign));
        assert_eq!(world.resource::<ArticulationDiagnostics>().tick, u64::MAX);

        assert!(world.remove::<ArticulationBinding>(entity));
        world.insert(entity, (ArticulationBinding::default(),));
        let newcomer = world.spawn((ArticulationBinding::default(),));
        for _ in 0..2 {
            world.insert_resource(ArticulationBindings::default());
            world.insert_resource(ArticulationDiagnostics::default());
            sync(&mut world);
            writeback(&mut world);
            let diagnostics = world.resource::<ArticulationDiagnostics>();
            assert_eq!(diagnostics.tick, u64::MAX);
            assert_eq!(diagnostics.simulated_dt, 0.0);
            assert_eq!(
                diagnostics.issues,
                vec![ArticulationDiagnostic {
                    entity: None,
                    issue: ArticulationIssue::SequenceExhausted,
                }]
            );
            for owner in [entity, newcomer] {
                assert_eq!(
                    world.resource::<ArticulationBindings>().articulation(owner),
                    None
                );
                assert!(world.get::<ArticulationSnapshot>(owner).is_none());
            }
            assert_eq!(
                world.resource::<PhysicsWorldResource>().articulations.len(),
                1
            );
        }
    }

    #[test]
    fn unchanged_runtime_resumes_after_same_shared_world_returns() {
        let (mut world, entity) = world();
        let before = world.get::<ArticulationSnapshot>(entity).unwrap().clone();
        let physics = world.remove_resource::<PhysicsWorldResource>().unwrap();
        sync(&mut world);
        writeback(&mut world);
        assert!(world.get::<ArticulationSnapshot>(entity).is_none());
        assert_eq!(
            world
                .resource::<ArticulationBindings>()
                .entity(before.handle),
            None
        );
        world.insert_resource(physics);
        sync(&mut world);
        writeback(&mut world);
        let after = world.get::<ArticulationSnapshot>(entity).unwrap();
        assert_eq!(after.handle, before.handle);
        assert_eq!((after.tick, after.version), (3, 2));
        assert!(world
            .resource::<ArticulationDiagnostics>()
            .issues
            .is_empty());
    }

    #[test]
    fn inactive_owner_cleanup_does_not_remove_reused_foreign_generation() {
        let (mut world, entity) = world();
        let old = world.get::<ArticulationSnapshot>(entity).unwrap().handle;
        let mut physics = world.remove_resource::<PhysicsWorldResource>().unwrap();
        assert!(physics.remove_articulation(old));
        let foreign = physics.spawn_articulation(ArticulationBinding::default().build().unwrap());
        assert_eq!(old.index(), foreign.index());
        assert_ne!(old.generation(), foreign.generation());
        assert!(world.despawn(entity));
        sync(&mut world);
        world.remove_resource::<ArticulationBindings>();
        world.remove_resource::<ArticulationDiagnostics>();
        world.insert_resource(physics);
        // Missing step diagnostics may suppress output, but not retirement.
        world.remove_resource::<PhysicsDiagnostics>();
        writeback(&mut world);
        assert!(world.resource::<ArticulationRuntime>().bindings.is_empty());
        assert_eq!(world.resource::<ArticulationBindings>().entity(old), None);
        assert_eq!(
            world.resource::<ArticulationBindings>().entity(foreign),
            None
        );
        let physics = world.resource::<PhysicsWorldResource>();
        assert!(physics.articulations.contains(foreign));
        assert_eq!(physics.articulations.len(), 1);
    }

    #[test]
    fn writeback_reconstructs_views_and_publishes_at_most_once_per_sync() {
        let (mut world, entity) = world();
        let old = world.get::<ArticulationSnapshot>(entity).unwrap().handle;
        sync(&mut world);
        world.remove_resource::<ArticulationBindings>();
        world.remove_resource::<ArticulationDiagnostics>();
        writeback(&mut world);
        let published = world.get::<ArticulationSnapshot>(entity).unwrap().clone();
        assert_eq!(published.handle, old);
        assert_eq!((published.tick, published.version), (2, 2));
        assert_eq!(
            world.resource::<ArticulationBindings>().entity(old),
            Some(entity)
        );
        writeback(&mut world);
        assert_eq!(world.get::<ArticulationSnapshot>(entity), Some(&published));

        assert!(world.remove::<ArticulationBinding>(entity));
        world.insert(entity, (ArticulationBinding::default(),));
        // No pending publication is not a reason to leave the old lifetime live.
        writeback(&mut world);
        assert!(world.get::<ArticulationSnapshot>(entity).is_none());
        assert!(!world
            .resource::<PhysicsWorldResource>()
            .articulations
            .contains(old));
        sync(&mut world);
        writeback(&mut world);
        let fresh = world.get::<ArticulationSnapshot>(entity).unwrap();
        assert_ne!(fresh.handle, old);
        assert_eq!((fresh.tick, fresh.version), (3, 1));
    }

    #[test]
    fn invalid_step_duration_is_never_published_as_accepted_time() {
        let (mut world, entity) = world();
        for raw_dt in [f32::NAN, f32::INFINITY, -0.01, 0.0] {
            world.resource_mut::<PhysicsDiagnostics>().simulated_dt = raw_dt;
            sync(&mut world);
            writeback(&mut world);
            assert_eq!(
                world
                    .get::<ArticulationSnapshot>(entity)
                    .unwrap()
                    .simulated_dt,
                0.0
            );
            let diagnostics = world.resource::<ArticulationDiagnostics>();
            assert_eq!(diagnostics.simulated_dt, 0.0);
            assert!(diagnostics.issues.contains(&ArticulationDiagnostic {
                entity: None,
                issue: ArticulationIssue::StepRejected,
            }));
        }
    }

    #[test]
    fn snapshot_removal_observer_retires_other_owner_before_the_shared_step() {
        let (mut world, first) = world();
        let later = world.spawn((ArticulationBinding::default(),));
        sync(&mut world);
        writeback(&mut world);
        let old = world.get::<ArticulationSnapshot>(later).unwrap().handle;
        world.observe::<ArticulationSnapshot, _>(LifecycleEvent::Remove, move |ctx| {
            if ctx.entity == first {
                assert!(ctx.world.contains_resource::<PhysicsWorldResource>());
                assert!(ctx.world.contains_resource::<ArticulationBindings>());
                assert!(ctx.world.contains_resource::<ArticulationDiagnostics>());
                assert!(ctx.world.remove::<ArticulationBinding>(later));
                ctx.world.insert(later, (ArticulationBinding::default(),));
            }
        });
        assert!(world.remove::<ArticulationBinding>(first));
        world.insert(first, (ArticulationBinding::default(),));
        sync(&mut world);
        assert!(!world
            .resource::<PhysicsWorldResource>()
            .articulations
            .contains(old));
        assert_eq!(
            world.resource::<ArticulationBindings>().articulation(later),
            None
        );
        writeback(&mut world);
        assert!(world.get::<ArticulationSnapshot>(later).is_none());
        sync(&mut world);
        writeback(&mut world);
        let fresh = world.get::<ArticulationSnapshot>(later).unwrap();
        assert_ne!(fresh.handle, old);
        assert_eq!(fresh.version, 1);
    }

    #[test]
    fn bounded_cleanup_drains_a_finite_owner_cascade_without_deferral() {
        let (mut world, first) = world();
        let mut chain = vec![first];
        for _ in 0..7 {
            chain.push(world.spawn((ArticulationBinding::default(),)));
        }
        sync(&mut world);
        writeback(&mut world);
        let foreign = world
            .resource_mut::<PhysicsWorldResource>()
            .spawn_articulation(ArticulationBinding::default().build().unwrap());
        let watched = chain.clone();
        world.observe::<ArticulationSnapshot, _>(LifecycleEvent::Remove, move |ctx| {
            let index = watched
                .iter()
                .position(|entity| *entity == ctx.entity)
                .unwrap();
            if let Some(&next) = watched.get(index + 1) {
                assert!(ctx.world.remove::<ArticulationBinding>(next));
            }
        });
        assert!(world.remove::<ArticulationBinding>(first));
        settle_removals(&mut world, BTreeSet::new());
        for entity in chain {
            assert!(world.get::<ArticulationSnapshot>(entity).is_none());
            assert_eq!(
                world
                    .resource::<ArticulationBindings>()
                    .articulation(entity),
                None
            );
        }
        assert!(world.resource::<ArticulationRuntime>().bindings.is_empty());
        assert!(world.resource::<RemovedAuthoring>().0.is_empty());
        assert!(world
            .resource::<ArticulationDiagnostics>()
            .issues
            .is_empty());
        let physics = world.resource::<PhysicsWorldResource>();
        assert!(physics.articulations.contains(foreign));
        assert_eq!(physics.articulations.len(), 1);
    }

    #[test]
    fn cleanup_does_not_chase_observer_generated_orphans_and_reports_deferral() {
        #[derive(Resource, Default)]
        struct Generated(Vec<Entity>);
        let (mut world, first) = world();
        let survivor = world.spawn((ArticulationBinding::default(),));
        sync(&mut world);
        writeback(&mut world);
        let survivor_handle = world.get::<ArticulationSnapshot>(survivor).unwrap().handle;
        let foreign = world
            .resource_mut::<PhysicsWorldResource>()
            .spawn_articulation(ArticulationBinding::default().build().unwrap());
        world.init_resource::<Generated>();
        let template = world.get::<ArticulationSnapshot>(first).unwrap().clone();
        let old = template.handle;
        world.observe::<ArticulationSnapshot, _>(LifecycleEvent::Remove, move |ctx| {
            if ctx.entity == survivor {
                return;
            }
            // Each callback returns, but every removal manufactures fresh work.
            // A cleanup that chases new snapshots would never converge.
            let orphan = ctx
                .world
                .spawn((ArticulationBinding::default(), template.clone()));
            assert!(ctx.world.remove::<ArticulationBinding>(orphan));
            ctx.world.resource_mut::<Generated>().0.push(orphan);
        });
        assert!(world.remove::<ArticulationBinding>(first));
        for expected in 1..=3 {
            settle_removals(&mut world, BTreeSet::new());
            assert_eq!(world.resource::<Generated>().0.len(), expected);
            // Fail-closed retirement itself must not invoke more observers.
            // The survivor's old snapshot can therefore remain until the next
            // cleanup, but it has no live model or public ownership authority.
            assert_eq!(
                snapshot_entities(&mut world).len(),
                if expected == 1 { 2 } else { 1 }
            );
            assert!(world.resource::<ArticulationRuntime>().publication_blocked);
            assert!(world.resource::<ArticulationRuntime>().bindings.is_empty());
            assert_eq!(world.resource::<ArticulationBindings>().entity(old), None);
            assert_eq!(
                world
                    .resource::<ArticulationBindings>()
                    .entity(survivor_handle),
                None
            );
            let physics = world.resource::<PhysicsWorldResource>();
            assert!(!physics.articulations.contains(survivor_handle));
            assert!(physics.articulations.contains(foreign));
            assert_eq!(physics.articulations.len(), 1);
            assert!(world.resource::<ArticulationDiagnostics>().issues.contains(
                &ArticulationDiagnostic {
                    entity: None,
                    issue: ArticulationIssue::LifecycleDeferred
                }
            ));
        }
        assert!(!world
            .resource::<PhysicsWorldResource>()
            .articulations
            .contains(old));
    }

    #[test]
    fn publication_restores_public_views_and_preserves_prior_diagnostics() {
        #[derive(Resource, Default)]
        struct Calls(usize);
        let (mut world, first) = world();
        let second = world.spawn((ArticulationBinding::default(),));
        let mut invalid = ArticulationBinding::default();
        invalid.gravity.x = f32::NAN;
        let rejected = world.spawn((invalid,));
        world.init_resource::<Calls>();
        world.observe::<ArticulationSnapshot, _>(LifecycleEvent::Insert, |ctx| {
            assert!(ctx.world.contains_resource::<ArticulationBindings>());
            assert!(ctx
                .world
                .resource::<ArticulationDiagnostics>()
                .issues
                .iter()
                .any(|entry| matches!(
                    entry.issue,
                    ArticulationIssue::InvalidAuthoring {
                        field: "gravity",
                        ..
                    }
                )));
            ctx.world.remove_resource::<ArticulationBindings>().unwrap();
            ctx.world
                .remove_resource::<ArticulationDiagnostics>()
                .unwrap();
            ctx.world.resource_mut::<Calls>().0 += 1;
        });
        sync(&mut world);
        writeback(&mut world);
        assert_eq!(world.resource::<Calls>().0, 2);
        for entity in [first, second] {
            let snapshot = world.get::<ArticulationSnapshot>(entity).unwrap();
            assert_eq!(
                world
                    .resource::<ArticulationBindings>()
                    .entity(snapshot.handle),
                Some(entity)
            );
        }
        let issues = &world.resource::<ArticulationDiagnostics>().issues;
        assert!(issues.contains(&ArticulationDiagnostic {
            entity: Some(rejected),
            issue: ArticulationIssue::InvalidAuthoring {
                field: "gravity",
                link: None
            },
        }));
        for resource in ["ArticulationBindings", "ArticulationDiagnostics"] {
            assert!(issues.contains(&ArticulationDiagnostic {
                entity: None,
                issue: ArticulationIssue::ObserverResourceRemoved { resource },
            }));
        }
        assert!(!issues
            .iter()
            .any(|entry| entry.issue == ArticulationIssue::LifecycleDeferred));
    }

    #[test]
    fn publication_world_removal_clears_output_and_defers_removed_owner_cleanup() {
        #[derive(Resource, Default)]
        struct Stashed {
            physics: Option<PhysicsWorldResource>,
            fired: bool,
        }
        let (mut world, a) = world();
        let b = world.spawn((ArticulationBinding::default(),));
        sync(&mut world);
        writeback(&mut world);
        let (first, later) = if a < b { (a, b) } else { (b, a) };
        let old: Vec<_> = [first, later]
            .into_iter()
            .map(|entity| world.get::<ArticulationSnapshot>(entity).unwrap().handle)
            .collect();
        let foreign = world
            .resource_mut::<PhysicsWorldResource>()
            .spawn_articulation(ArticulationBinding::default().build().unwrap());
        world.init_resource::<Stashed>();
        world.observe::<ArticulationSnapshot, _>(LifecycleEvent::Insert, move |ctx| {
            if ctx.entity == first && !ctx.world.resource::<Stashed>().fired {
                let physics = ctx.world.remove_resource::<PhysicsWorldResource>().unwrap();
                ctx.world.resource_mut::<Stashed>().physics = Some(physics);
                ctx.world.resource_mut::<Stashed>().fired = true;
                ctx.world.remove_resource::<ArticulationBindings>();
                ctx.world.remove_resource::<ArticulationDiagnostics>();
                assert!(ctx.world.remove::<ArticulationBinding>(first));
                ctx.world.insert(first, (ArticulationBinding::default(),));
                assert!(ctx.world.remove::<ArticulationBinding>(later));
                ctx.world.insert(later, (ArticulationBinding::default(),));
            }
        });
        sync(&mut world);
        writeback(&mut world);
        for entity in [first, later] {
            assert!(world.get::<ArticulationSnapshot>(entity).is_none());
            assert_eq!(
                world
                    .resource::<ArticulationBindings>()
                    .articulation(entity),
                None
            );
            assert!(world.resource::<RemovedAuthoring>().0.contains(&entity));
        }
        let diagnostics = world.resource::<ArticulationDiagnostics>();
        assert_eq!(diagnostics.simulated_dt, 0.0);
        for issue in [
            ArticulationIssue::MissingPhysics,
            ArticulationIssue::LifecycleDeferred,
            ArticulationIssue::ObserverResourceRemoved {
                resource: "PhysicsWorldResource",
            },
        ] {
            assert!(diagnostics.issues.contains(&ArticulationDiagnostic {
                entity: None,
                issue
            }));
        }
        let physics = world.resource_mut::<Stashed>().physics.take().unwrap();
        assert!(old
            .iter()
            .all(|handle| physics.articulations.contains(*handle)));
        world.insert_resource(physics);
        sync(&mut world);
        assert!(old.iter().all(|handle| !world
            .resource::<PhysicsWorldResource>()
            .articulations
            .contains(*handle)));
        writeback(&mut world);
        for entity in [first, later] {
            let snapshot = world.get::<ArticulationSnapshot>(entity).unwrap();
            assert!(!old.contains(&snapshot.handle));
            assert_eq!(snapshot.version, 1);
        }
        assert!(world
            .resource::<PhysicsWorldResource>()
            .articulations
            .contains(foreign));
        assert!(world
            .resource::<ArticulationDiagnostics>()
            .issues
            .is_empty());
    }

    #[test]
    fn publication_shared_resource_removal_cancels_output_without_resetting_handles() {
        #[derive(Resource, Default)]
        struct Interference {
            calls: usize,
            physics: Option<PhysicsWorldResource>,
        }
        for remove_world in [false, true] {
            let (mut world, a) = world();
            let b = world.spawn((ArticulationBinding::default(),));
            sync(&mut world);
            writeback(&mut world);
            let old: Vec<_> = [a, b]
                .into_iter()
                .map(|entity| {
                    (
                        entity,
                        world.get::<ArticulationSnapshot>(entity).unwrap().clone(),
                    )
                })
                .collect();
            world.init_resource::<Interference>();
            world.observe::<ArticulationSnapshot, _>(LifecycleEvent::Insert, move |ctx| {
                ctx.world.resource_mut::<Interference>().calls += 1;
                if ctx.world.resource::<Interference>().calls != 1 {
                    return;
                }
                if remove_world {
                    let physics = ctx.world.remove_resource::<PhysicsWorldResource>().unwrap();
                    ctx.world.resource_mut::<Interference>().physics = Some(physics);
                } else {
                    ctx.world.remove_resource::<PhysicsDiagnostics>().unwrap();
                }
            });
            sync(&mut world);
            writeback(&mut world);
            assert_eq!(world.resource::<Interference>().calls, 1);
            for (entity, prior) in &old {
                assert!(world.get::<ArticulationSnapshot>(*entity).is_none());
                assert_eq!(
                    world
                        .resource::<ArticulationBindings>()
                        .articulation(*entity),
                    None
                );
                assert_eq!(
                    world.resource::<ArticulationRuntime>().bindings[entity].handle,
                    prior.handle
                );
            }
            let diagnostics = world.resource::<ArticulationDiagnostics>();
            assert_eq!(diagnostics.simulated_dt, 0.0);
            assert!(diagnostics.issues.contains(&ArticulationDiagnostic {
                entity: None,
                issue: ArticulationIssue::ObserverResourceRemoved {
                    resource: if remove_world {
                        "PhysicsWorldResource"
                    } else {
                        "PhysicsDiagnostics"
                    },
                },
            }));
            assert!(diagnostics.issues.contains(&ArticulationDiagnostic {
                entity: None,
                issue: ArticulationIssue::MissingPhysics,
            }));
            assert!(!diagnostics
                .issues
                .iter()
                .any(|entry| entry.issue == ArticulationIssue::LifecycleDeferred));
            if remove_world {
                let physics = world.resource_mut::<Interference>().physics.take().unwrap();
                world.insert_resource(physics);
            } else {
                world.insert_resource(PhysicsDiagnostics {
                    simulated_dt: 0.01,
                    ..Default::default()
                });
            }
            sync(&mut world);
            writeback(&mut world);
            assert_eq!(world.resource::<Interference>().calls, 3);
            for (entity, prior) in old {
                let resumed = world.get::<ArticulationSnapshot>(entity).unwrap();
                assert_eq!(resumed.handle, prior.handle);
                assert_eq!(resumed.tick, prior.tick + 2);
                assert_eq!(resumed.version, prior.version + 2);
            }
        }
    }
}
