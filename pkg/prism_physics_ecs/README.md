# Native Prism physics ECS bridge

Headless `prism_app` / `prism_ecs` / `prism_transform` integration for
`prism_physics_core` and the backend-selectable MPM runtime. No Bevy, renderer,
window, or gameplay dependency. XPBD rigid/soft bodies use the CPU core; MPM can
select the CPU reference or resident wgpu backend without exposing GPU handles
to ECS consumers. Analytic-field evaluation can separately use an explicitly
installed GPU runtime; its synchronous readback still feeds CPU XPBD, not a
GPU-resident rigid/soft-body simulation.

```rust
use std::time::Duration;
use prism_app::{App, TimeUpdateStrategy};
use prism_physics_ecs::{
    Collider, ExternalForce, PhysicsBindings, PhysicsWorldResource,
    PrismPhysicsPlugin, RigidBody, Vec3, Velocity,
};
use prism_transform::Transform;

let mut app = App::new();
app.set_fixed_timestep(Duration::from_millis(10))
    .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)))
    .add_plugins(PrismPhysicsPlugin);
let entity = app.world_mut().spawn((
    RigidBody::dynamic(), Transform::from_xyz(0.0, 3.0, 0.0),
    Collider::sphere(0.5), Velocity::default(),
    ExternalForce { force: Vec3::X, torque: Vec3::ZERO },
));
app.update();
let handle = app.world().resource::<PhysicsBindings>().body(entity).unwrap();
let position = app.world().resource::<PhysicsWorldResource>()
    .bodies.position(handle).unwrap();
assert!(position.y < 3.0);
```

## Schedule and ownership

The plugin installs native transform tracking and orders `FixedUpdate` as
`PhysicsSystems::PreSync -> Sync -> Step -> Writeback -> PostWriteback`.
The optional articulation plugin uses PreSync and PostWriteback. Sync chains
ordinary body reconciliation, assembly requests, ragdoll reconciliation, field
reconciliation, and MPM synchronization, in that order. Step and Writeback each
run the shared-world path before the independent MPM path.

Ordinary authoring/force systems should run `.before(PhysicsSystems::Sync)`;
articulation authoring must run before PreSync. Ordinary consumers can run after
Writeback; articulation snapshot/handoff consumers must run after PostWriteback
or in `FixedPostUpdate`. Mutate physics components before their reconciliation
boundary, not during simulation/writeback. Engine fixed-clock
accumulation determines whether a frame runs zero, one, or multiple ticks.
Physics never advances in the variable-rate `Update` schedule.

The world resource, solver, maps, and diagnostics are initialized only if absent.
Configure `PhysicsWorldResource.config` and `PhysicsSolver.0.config` before a
tick. World `default_substeps` must be 1 through 8; every substep gets exactly
`EngineClocks.fixed().delta_secs() / default_substeps`. Invalid fixed dt,
gravity, substep count, position-iteration count, contact compliance, or
restitution threshold reports `InvalidConfiguration` with zero advanced time.
Reconciliation, contact sampling, and writeback still run on a rejected tick.
Other low-level solver settings retain the core's own preconditions.

`PhysicsDiagnostics` retains only the latest fixed tick's issues, tick number,
and accepted simulated dt (only completed XPBD substeps on field failure).
Inspect it in a fixed-step consumer if every
rejection matters; several ticks can occur in one frame.

## Rigid-body lifecycle and transforms

- `RigidBody + Transform` creates a body, with an optional `Collider`, `Velocity`,
  and persistent `ExternalForce`. Mass/inertia are explicitly authored, never
  inferred from shape dimensions. The dynamic default has unit mass/inertia.
- Editing kind/mass/damping/CCD rebuilds the body and rotates its generational
  handle. Live pose and velocity survive unless ECS explicitly changed them.
  Handle-dependent registrations are **not** automatically migrated.
- Collider shape/material/sensor edits update the existing body; removing the
  collider leaves a body without collision geometry. Inline analytic shapes
  are validated and deduplicated in the append-only registry. Compound, convex,
  and triangle-mesh shapes must be inserted into the core's arenas/registry
  first, then referenced with `Collider::registered`. Their internal validity
  remains the caller's responsibility. No automatic mesh cooking is provided.
- Invalid authoring removes the managed body and reports the rejection; repair
  recreates it. A nonfinite force alone is ignored and diagnosed for that tick.
- Removing/readding `RigidBody` or `Transform` between ticks is a new lifecycle,
  even when values compare equal. Despawning removes the binding and body at
  the next Sync. Full entity and body generations are used in both directions.
- A root, unit-scale `Transform` is required. Missing Transform, any `ChildOf`
  parent, nonunit scale, nonfinite pose, or nonunit quaternion is explicitly
  rejected. This applies to static and kinematic bodies too. A parent added to
  a live body also removes it. Bake scale into geometry and keep the physics
  entity a root; nonphysics children are supported and native global transform
  propagation updates them after writeback. There is no silent local/world
  conversion for unsupported parented bodies. A malformed hierarchy is reported
  as `TransformPropagation`; physics locals may already have been written.
- Dynamic pose/optional velocity are written back after simulation. ECS pose
  changes are teleports: both current and previous core pose are reset, and the
  body is woken. Static and kinematic poses are externally authored; velocities
  do not integrate their positions and teleport does not synthesize velocity.
  A kinematic velocity can still participate in core contact velocity algebra.
- `Velocity` is optional. Removing it explicitly clears linear/angular velocity;
  writeback does not reinsert it. Removing/readding it overrides resource-side
  velocity edits even if the new value equals the previous output.

Transform numerics use `prism_math`; physics numerics use the core's `glam`
version, reexported here as `Vec3` and `Quat`. Conversion is explicit in the
bridge. Units are metres, seconds, kilograms, radians; forces/torques are world
space. Large-coordinate precision remains limited by the core's f32 state.

## Forces, queries, and events

`ExternalForce` applies a world-space centre-of-mass force and torque on **every
completed substep**, so total impulse is force times the full fixed dt when all
substeps complete. Remove or zero it
to stop applying it; a zero wrench does not wake a sleeping body. One-shot
impulses can use `PhysicsWorldResource.apply_impulse` or `apply_torque_impulse`.
Direct `apply_force` queues a core pending force for its next substep only; it
does not have the persistent component's whole-tick semantics.

The resource exposes the actual `PhysicsWorld` API through deref, including
raycasts, shape queries, fields, particle state, and solver settings. For a
raycast use `geometry::Ray` and `core::QueryFilter`; convert `hit.body` to an ECS
entity via `PhysicsBindings::entity`. Unmanaged bodies legitimately map to None.
Maps reconcile on fixed ticks, not immediately when ECS components change.

`Events<PhysicsContactEvent>` carries collision start/end and sensor enter/exit,
including the fixed tick and both generational handles/optional entity IDs.
The first endpoint of a trigger is its sensor (the lower-keyed body when both
are sensors). Swapping sensor roles emits the old exit then the new enter.
Core broad-phase semantics are preserved: pairs with no dynamic participant
are filtered, including two static sensors. Overlap transitions are sampled
before solving and after each substep; a transient pair can enter and exit in
one tick. This is not an impact-impulse stream or a swept-hit notification.
Within each sample, ends precede starts and handles are sorted stably.

Native event buffers are frame-maintained, not fixed-tick-cleared, so a frame's
multiple fixed ticks remain readable. Use an `EventCursor` as usual. Exit events
retain the original entity identity even after despawn/reused body slots; check
entity liveness before querying its components. The plugin owns draining core
contact transitions; do not independently drain/reset the core contact tracker.

## Aerodynamics and force fields

### Authoring, coordinates, and composition

`RigidBodyAero(AeroBodyConfig)` remains the body-wide authoring option. On the
same rigid-body entity, `AeroFunctionalComponents` can add vectors of surfaces,
thrusters, and buoyancy volumes. Independent source entities can instead carry
`AeroSurfaceComponent { target, surface }`, `AeroThrusterComponent { target,
thruster }`, or `AeroBuoyancyComponent { target, volume }`; neither aggregate
component is required. Each source can be edited, removed, despawned, or
retargeted independently. Use multiple source entities for multiple elements of
one family; different families may share a source entity.

`target` is the full generational Entity of an ordinary managed `RigidBody`,
not a raw core handle, assembly alias, or articulation link. Binding is explicit,
never inferred from a parent; a source may also target itself. Sources need no
RigidBody, Collider, or Transform. Element positions, directions, and volume
geometry are **target body-local**: source Transform/hierarchy does not move or
rotate them. Body-wide model gravity and water-plane parameters retain their
core world-space semantics.

Each target gets one combined core aero registration. Within each element
family, composition is stable and additive:

1. `RigidBodyAero` elements, preserving vector order.
2. `AeroFunctionalComponents` elements, preserving vector order.
3. Independent components in full source Entity order (index, then generation),
   not query/archetype or insertion order.

Only `RigidBodyAero` supplies the body-wide `AeroModel`; without it the core
default model is used. There are no implicit overrides or value deduplication:
equal elements from two sources intentionally contribute twice. Removing one
aggregate removes only its contribution (and resets the model to default when
the base is removed); independent sources remain. Removing the last source
unregisters aero, subject to the atomic-rejection rules below.

### Fixed-step lifecycle and rejection

Authoring changes commit at fixed `PhysicsSystems::Sync`, without rebuilding a
body solely for aero edits. A zero-fixed-tick frame commits nothing. Body
rebuild/external removal rebinds valid current authoring to the new BodyHandle;
neither recycled entity slots nor body slots inherit old registrations. Removing
the target RigidBody or despawning the target retires its registration. Sources
can remain authored, are diagnosed while unavailable, and retry on subsequent
Syncs after repair or explicit retargeting. ECS owns registrations for its
targets: direct core registration edits are reconciled at Sync; unrelated
unmanaged registrations are preserved.

Aero replacements are validated before Step, not silently sanitized or partly
applied. Parameters must satisfy the ECS checks, including finiteness,
nonnegative dimensions/densities/drag, nondegenerate geometric directions,
buoyancy quadrature 1 through 8, and control effectiveness 0 through 1. Signed
thrust, spin, lift slopes, control deflection, and Magnus coefficients are
supported; zero dimensions/coefficients can intentionally disable effects.

An invalid source rejects the **entire target replacement** and retains its
last accepted configuration only on the exact same live BodyHandle. This also
defers otherwise valid edits and removals until the replacement is repaired.
With no accepted same-handle configuration, no aero is registered; stale
fallback never crosses a rebuild. Targets connected by source retargeting
commit or roll back together, including pending last-source removals and chained
moves, so a retained old registration and a new one cannot apply a moved source
twice. Unrelated targets can still update.

`AeroDiagnostics.issues` is published at Sync and contains source Entity,
target Entity, `AeroSource`, and `AeroIssue`. Issues distinguish `MissingTarget`,
`TargetNotRigidBody`, `TargetUnavailable`, `InvalidParameter { parameter, index }`,
`ReplacementBlocked { rejected_target }`, and `RegistrationRejected`.
Only the first invalid parameter per source is reported; `index` identifies
that source's vector element (zero for an independent component), or is None for
model parameters. Diagnostics are sorted by target, source kind, and source
Entity, replaced each fixed Sync, and may not exist before the first Sync.
Inspect them in a fixed-step consumer if every rejection matters.

The core evaluates accepted aero and samples force fields on **every XPBD
substep**, using current simulation state. They add to `ExternalForce`; do not
manually replay the same contribution with `apply_force`. A constant force
produces force times the full fixed dt, not once per rendered frame or multiplied
by the substep count. State-dependent forces are reevaluated each substep, so
different substep counts need not produce identical trajectories.

With `app` configured as in the opening example, create independently owned
sources and optionally mix them with aggregate authoring:

```rust
use prism_physics_ecs::{
    core, AeroBuoyancyComponent, AeroDiagnostics, AeroFunctionalComponents,
    AeroSurfaceComponent, AeroThrusterComponent, RigidBodyAero,
};

let target = app.world_mut().spawn((
    RigidBody::dynamic(), Transform::from_xyz(0.0, 2.0, 0.0),
    Velocity::default(), RigidBodyAero(core::AeroBodyConfig::default()),
    AeroFunctionalComponents {
        thrusters: vec![core::Thruster::new(Vec3::ZERO, Vec3::X, 5.0)],
        ..Default::default()
    },
));
let panel = app.world_mut().spawn(AeroSurfaceComponent {
    target,
    surface: core::AeroSurface::new(Vec3::ZERO, Vec3::Y, Vec3::X, 1.0),
});
let engine = app.world_mut().spawn(AeroThrusterComponent {
    target, thruster: core::Thruster::new(Vec3::ZERO, Vec3::X, 10.0),
});
let volume = app.world_mut().spawn(AeroBuoyancyComponent {
    target,
    volume: core::BuoyancyVolume::box_volume(
        Vec3::ZERO, Vec3::splat(0.5), core::aero::FluidKind::Water,
    ),
});
app.update(); // A fixed tick commits all sources before stepping.
let body = app.world().resource::<PhysicsBindings>().body(target).unwrap();
let config = app.world().resource::<PhysicsWorldResource>().aero_body(body).unwrap();
assert_eq!(config.surfaces.len(), 1);
assert_eq!(config.thrusters.len(), 2); // Aggregate + independent, not an override.
assert_eq!(config.buoyancy.len(), 1);
assert!(app.world().resource::<AeroDiagnostics>().issues.is_empty());

app.world_mut().get_mut::<AeroThrusterComponent>(engine).unwrap().thruster.thrust = 20.0;
app.world_mut().remove::<AeroSurfaceComponent>(panel);
app.world_mut().despawn(volume);
app.update(); // Only the two thrusters remain; the target body is not rebuilt.
for issue in &app.world().resource::<AeroDiagnostics>().issues {
    eprintln!("{issue:?}");
}
```

The volume above demonstrates registration; configure the base model's
`water_plane` for a water surface and partial-immersion buoyancy.

### Analytic force-field sources

`ForceField(core::dynamics::field::Field)` is a separate source component, not an
aero element or per-Entity target binding. Its shape and `FieldTargetMask` use
the core field contract. The current world sampler supplies live dynamic rigid
bodies, including sleepers, at their world centres of mass with recipient mask
`FieldTargetMask::ALL`; this is not a per-ECS-category filter or automatic
soft-particle/MPM field coupling. Shape/effect positions and directions
are world-space; source Transform/hierarchy is ignored. Multiple fields are
additive, and core registry reduction order is FieldId order.

Sources reconcile in full Entity order at fixed Sync. The authored FieldId is
ignored: `FieldBindings::field(source)` returns the assigned identity, retained
across edits while registered. Removing the component or despawning its entity
retires the field at the next Sync; edits do not accumulate duplicate fields.
Removal/readdition without an intervening Sync need not create a new identity.
Sync also repairs direct removal or edits of an ECS-owned registry entry.
The aero atomic-validation/diagnostic policy above does **not** extend to
ForceField: field values retain the core API's preconditions and sanitization.

Continuing with the same `app`:

```rust
use prism_physics_ecs::{FieldBindings, ForceField};
use prism_physics_ecs::core::dynamics::field::{Field, FieldId};

// FieldId(0) requests core ID allocation; ECS ignores any authored ID.
let field_source = app.world_mut().spawn(ForceField(
    Field::constant(FieldId(0), Vec3::X * 2.0),
));
app.update();
let id = app.world().resource::<FieldBindings>().field(field_source).unwrap();
assert!(app.world().resource::<PhysicsWorldResource>()
    .fields.fields().iter().any(|field| field.id == id));
app.world_mut().remove::<ForceField>(field_source);
app.update();
assert!(app.world().resource::<FieldBindings>().field(field_source).is_none());
```

### Explicit GPU field runtime and rejected substeps

Without `FieldRuntimeResource`, fields use the core CPU sampler. To opt in,
the caller acquires/configures a `prism_physics_gpu::GpuContext`, constructs
`field_runtime::GpuFieldRuntime::new(context)?`, and inserts
`FieldRuntimeResource(runtime)` before a fixed tick. The plugin does not acquire
a device or create this resource. Removing it returns to CPU sampling without
rebuilding bodies, field identities, or the solver. This authority is separate
from MPM backend selection. Runtime construction is fallible: shader/layout/
pipeline failures are returned as `FieldGpuError`, without installing an
uncaptured-error handler or silently selecting a CPU implementation.

Each XPBD substep batches dynamic bodies through the GPU field evaluator,
including membership, falloff, effects, and accumulation. The runtime waits for
GPU completion and reads results back synchronously. Validated forces then feed
CPU aero/dynamics and CPU XPBD; CPU fields are not sampled a second time. This
is a hybrid path, not asynchronous or zero-copy simulation.

`FieldRuntimeDiagnostics` is reset on every attempted fixed tick, including CPU
and invalid-configuration ticks. It exposes `tick`, `gpu_authority`,
`requested_substeps`, `completed_substeps`, and
`error: Option<FieldStepError<FieldGpuError>>`. GPU authority records resource
presence, not successful completion. Core errors distinguish invalid time step
from sampling failure; a sampling error carries the whole tick's completed
substep count and a `FieldBatchError` source: provider error, wrong sample count,
nonfinite sample, or force-accumulation overflow. Provider errors include GPU
capacity/device/workspace, scoped GPU validation/internal/out-of-memory,
polling, and mapping failures. Allocation, upload, encoding, submission, and
readback have separate operation diagnostics. Failed buffer growth retains
the previous capacity; subsequent valid batches can reuse it.

A failed batch is rejected before field-force/wake mutation or solver advance.
The ECS bridge restores the pre-attempt pending-force queue and the sleep flags
**and idle timers** changed by its persistent force/torque application. It stops
remaining XPBD substeps, reports `PhysicsIssue::FieldSamplingFailed`, and retains
the GPU resource without silently falling back to CPU. Earlier completed
substeps stay committed: `PhysicsDiagnostics.simulated_dt` is their count times
the substep dt (the full fixed dt when all complete). Writeback and contact
publication still run for the initial sample and committed prefix. Invalid
configuration instead reports `InvalidConfiguration`, completes zero substeps,
and applies no persistent ECS wrench. This is not whole-tick rollback: the
separately scheduled MPM path is not rolled back with XPBD.

## Reduced-coordinate articulations

Add `articulation::PrismArticulationPlugin` alongside `PrismPhysicsPlugin` to
enable `articulation::ArticulationBinding`. It authors a fixed/floating base,
base inertia and world pose, base-frame spatial velocity, independent world
gravity, parent-before-child links, and initial generalized positions/velocities.
Empty generalized arrays mean zero state; otherwise lengths must match joint
DOFs. Fixed-base velocity must be zero; constructor gravity defaults to zero.
Authoring uses physics-space types, not entity Transform/ChildOf. `build()`
validates numerical data and the core topology contract.

PreSync creates/reconciles world-owned trees; the shared CPU XPBD solver advances
them once per physical substep, never a second articulation-plugin step.
Unchanged or equal-value authoring preserves runtime edits, impulses, and
breakage. Actual edits replace the complete runtime from authored initial state
and rotate its handle. Invalid edits retire the old runtime rather than leaving
stale simulation active; repair retries. Removal/readdition starts a new lifetime,
and despawn/stale external removal use generational cleanup and repair.

`ArticulationBindings` and `ArticulationSnapshot` are reconstructible public views,
not ownership authority. PostWriteback publishes the handle, checked version,
tick, accepted `simulated_dt`, world base/link poses, generalized state, and
spatial velocities in each base/link's local frame. Rejected shared steps can
publish zero-dt output with `StepRejected`; publication alone does not prove
advance. Invalid output retires the runtime. Counters fail closed instead of
wrapping. Observer-driven cleanup is bounded; `LifecycleDeferred` means residual
output is untrusted and publication is blocked pending cleanup.

Broken trees attempt atomic maximal-coordinate handoff at PostWriteback.
`Degraded`/`HandoffFailed` diagnostics distinguish the transition and rejection;
failure retains the stopped reduced tree and attachments for retry. Conversion
requires articulation gravity to match world gravity and can reject unsupported
policies. Success adopts the exact core-generated bodies, colliders, and surviving
joints, removes reduced snapshots, and exposes
`ArticulationBindings::handoff(owner)` with the report and body/entity mappings.
The original articulation handle is lineage, not a live reduced tree.

Each `ArticulationBodyEntity` maps a base/link to an unparented physical `entity`
in the COM/principal-inertia frame and an owned `authored_entity` child preserving
the original authored frame via inverse(link_from_body). Physical entities use
ordinary pose/Velocity/force/collider paths. Attach visuals and user offsets below
the authored child; editing/reparenting/removing that frame or adding RigidBody
or Collider to it retires the group. Root authoring is frozen after handoff;
remove/readd it to restart. Removing a generated member or changing its immutable
RigidBody policy also retires the whole group. Assembly merge/split of these
members is unsupported. Core-only snapshots do not cover ECS topology/ownership;
rollback across handoff requires a paired ECS-plus-core checkpoint.

## Core-owned assemblies

The base plugin installs `AssemblyBody { handle }`, `AssemblyRequestQueue`, and
`AssemblyDiagnostics`. AssemblyBody aliases an already world-owned core body;
it neither spawns a body nor participates in ordinary RigidBody ownership or
Transform/Velocity writeback. The caller supplies core bodies and their state.

Queue `merge(entities)` or `split(entity)` for processing in insertion order at
Sync. Merge canonicalizes members and rejects duplicate/missing/stale members,
ordinary RigidBody-owned bodies, already managed members, and core-invalid sets
(for example, disconnected or nondynamic bodies). Success remaps every member
alias to the shared proxy; `PhysicsBindings::entity(proxy)` returns the minimum
full Entity deterministically. Split uses the core remap report to restore
distinct member handles.

Losing/despawning/detaching a member triggers automatic split at the next Sync.
Surviving aliases are updated; released bodies without aliases remain world-owned
and unmanaged, not deleted. A rejected automatic split reports
`MemberLostSplitRejected` and forgets the managed mapping rather than claiming
successful cleanup. Diagnostics contain the latest Sync's errors; requests are
consumed, so retrying a rejected explicit request requires enqueueing it again.

## Ragdolls over ordinary rigid bodies

The base plugin reconciles `EcsRagdoll { definition, parts }` at Sync, after body
and assembly mapping. `EcsRagdoll::new` sorts stable `RagdollPartId`/Entity pairs;
the complete unique mapping must refer to existing ordinary managed RigidBody
entities. This authors constraints, not a skeleton inference or body-spawning
pipeline. `EcsRagdollBindings::{ragdoll,body}` exposes active generational bindings.

Unchanged definitions and handles preserve bindings; body rebuilds refresh them.
Invalid/missing/stale mappings remove the active binding, report through
`EcsRagdollDiagnostics`, and retry after repair. Removing/despawning the owner
removes its constraints, not the part bodies. Each Sync rebuilds the solver's
aggregate ECS ragdoll constraint list while preserving its configuration;
externally installed ragdoll lists are not merged into that ECS-owned list.

## Soft bodies

`SoftBodyBinding(SoftBody)` owns an authored template, not a mirrored simulation
snapshot. It creates a world-owned soft body; template edits/removal/readd,
entity despawn, and stale external removal are reconciled generationally.
Template comparison does not include the evolving live simulation, so particles
are not reset every tick. `PhysicsBindings::{soft_body,soft_entity}` locate live
state in `PhysicsWorldResource.soft_bodies`.

Particles are **world space**. The ECS Transform neither moves nor receives
particle state. Soft bodies use the world's substep dt/count, their own gravity,
damping and iteration settings, and core soft/rigid coupling. Their standalone
`config.substeps` does not add a second nested substep loop. The bridge validates
finite particle state, nonnegative inverse masses/damping, and finite gravity;
constraint indices, membrane material, tearing, topology and collision settings
must satisfy the underlying core APIs. It does not refer to individual tearing
threshold fields.

`M15TopologyBridgePlugin` is an opt-in `FixedPostUpdate` route for versioned
topology replacements. Install a downstream `M15TopologyAdapter` through
`M15TopologyBridge::set_adapter`. Its `register` receives complete initial state;
`apply` receives the producer update and validated candidate consumer, including
conservative swept triangle bounds. Render/collision/CCD changes must commit
atomically; the local revision advances only after adapter acceptance.

The core is the sole topology producer. This route drains its update stream;
do not also drain it independently. Missing adapters, invalid updates, and adapter
rejection retain the update and subsequent updates for ordered retry. Inspect
`consumer`, `revision`, `pending_len`, and `diagnostics()` on the bridge for
committed state, cumulative applied updates, pending count, live streams, and
last error. Retiring a managed soft body removes its consumer, pending payloads,
and downstream adapter state. Late installation baselines the available stream
from retained history or current topology; stable streams do not re-register
and clone the authored mesh every tick. The bridge is renderer-independent: it
does not itself stitch a render mesh, compact GPU topology, or rebuild a
production CCD/BVH consumer.

## Material-point simulations

`MpmBinding` owns authored initial particles, grid, material, numerical config,
backend policy, per-advance work budget, and optional surface settings. It is
reconciled at Sync into an ECS-owned `MpmRuntime` outside PhysicsWorldResource;
CPU and GPU execution share the same fixed-step boundary, not the world's XPBD
substeps or automatic rigid/soft coupling. The runtime dt must match engine fixed
dt; each tick requests one runtime step, bounded by the per-advance budget.
Changing authored solver data, backend policy, or budget replaces the live runtime
from authored initial state, clearing old output even if construction fails.
This is not live-state backend migration. Equal authoring preserves simulation;
remove/readd starts a new lifetime, and removal/despawn cleans runtime and output.
The authored component is never overwritten with evolving simulation state.

Writeback publishes a backend-neutral `MpmParticleSnapshot` only after a
successful step and snapshot. Its version advances once per publication and
its tick identifies the fixed schedule that produced it; `epoch` distinguishes
runtime generations. Arc-backed, index-aligned columns contain world-space
positions/velocities, APIC affine matrices, deformation gradients, and plastic
determinants, independently of entity Transform. A fixed-dt mismatch,
runtime construction error, invalid output, or GPU readback error is diagnosed
without publishing fabricated state. In particular, a failed GPU snapshot does
not silently resume from the older CPU shadow. The last published same-generation
snapshot may remain; inspect epoch/version/tick for freshness. An authored backend
policy edit rebuilds from initial state, not that last live snapshot.

`MpmDiagnostics.simulated_dt` is one fixed dt if any runtime advanced in the
latest tick, even if snapshot publication then failed; it is not summed across
runtimes or proof of fresh output. `steps` is aggregate runtime work. Preferred
GPU construction either reports GPU authority or an explicit CPU fallback
reason. `MpmRuntimeResource` exposes backend authority/fallback and snapshot info.

`MpmBinding::with_surface_reconstruction(MpmSurfaceConfig)` enables bounded CPU
scalar-field construction and marching-cubes reconstruction. Settings include
kernel radius, grid spacing, isovalue, padding, and node/cell/sample-visit/mesh
budgets. `MpmSurfaceSnapshot` publishes an Arc-backed world-space mesh with the
particle epoch, source version/tick, and a separate mesh revision. Surface-only
edits do not reset particles: they can reconstruct from the latest valid snapshot
even when the current step is rejected, advancing mesh revision without particle
version. Disabling/invalidating settings or reconstruction/budget/allocation
failure clears old surface output and reports the corresponding MpmIssue;
particle publication remains independent. This crate does not install render
extraction or zero-copy render-buffer sharing.

## Direct world mutation contract and broader gaps

Direct pose, velocity, force/impulse edits are supported. Unchanged ECS output
does not overwrite resource-side pose/velocity before stepping. ECS-authored
mass/kind/shape/material/sensor remain authoritative at Sync. External removal
of a managed body is diagnosed and recreates it from the latest ECS state;
cleanup never removes an unmanaged body that reused its slot with a different
generation. Prefer world removal APIs over raw arena mutation.

Do not replace/reset/restore the entire world or maps independently while live
bindings exist: a new world can reuse identical handles. Build a new App for a
new world. Direct low-level changes must respect all core invariants.

Ordinary RigidBody-managed handles and articulation handoff members must **not**
participate in external assembly merge/split: these ownership paths do not
remap their private records across that operation. Use the explicit AssemblyBody
request route for eligible core-owned bodies instead; it consumes core remap
reports without implying ordinary RigidBody integration.
When a live body's removal is refused by core, its binding is retained inactive
with `RemovalRefused` and retried, rather than silently dropped.

Body removal cleans attached joints created through `PhysicsWorld::spawn_joint`
(including disabled joints) and active raw-storage joints. Disabled joints
inserted directly into raw storage without assembly edges remain caller-owned:
core currently has no complete public generational joint iterator. Rebuilds
invalidate direct core joints, grabs, ragdolls, and other handle-based
registrations instead of migrating them. ECS-authored ragdolls rebind as described
above. Valid aero authoring through `RigidBodyAero`,
`AeroFunctionalComponents`, or independent aero components is reconciled to the
replacement handle at the next Sync, without replaying a stale rejected config.
There is no general-purpose standalone ECS joint authoring component; the
specialized articulation, assembly, and ragdoll routes above have distinct
ownership and lifecycle contracts.

There is no automatic streamed-terrain collider binding, render extraction,
interpolation, rollback/replication, scene serialization, arbitrary parent/scale
support, or zero-copy render sharing or performance
guarantee. Reconciliation scans managed authoring and clones soft templates;
this is a correctness-first bridge, not a claim that the full engine physics
specification is complete.

## Verification

`tests/app_lifecycle.rs` uses actual `App::update()` fixed schedules, not direct
system calls. It covers accumulator/order, spawn/despawn and component edits,
remove/readd, stale generations, teleport and transform descendants, explicit
hierarchy/scale rejection, solid collision response, sensor role edits and
despawn exits, queries, persistent force/torque and sleep, static/kinematic
contract, shape reuse/invalid repair, disabled-joint cleanup, soft-body lifecycle,
and configuration rejection/recovery.

These are test-source coverage descriptions, not a claim that tests have passed.
Verification execution/results belong to the main run; this README update does
not run Cargo or tests.

Workspace command: `cargo test -p prism_physics_ecs --test app_lifecycle`.
Aero/field lifecycle coverage: `tests/field_lifecycle.rs` exercises stable
composition, independent source edits/removal/retargeting, entity/body generation
changes, atomic rejection/recovery, and fixed-Sync/substep behavior. A serial
selection is `cargo test -p prism_physics_ecs -j 1 --test field_lifecycle -- --test-threads=1`.
`tests/field_runtime.rs` covers explicit GPU authority changes, CPU parity,
sampling failure, pending-force/sleep-timer recovery, partial completion, and
invalid configuration. Articulation lifecycle/public-view repair and exactly-once
stepping are covered in `tests/articulation_ecs_lifecycle.rs`; handoff ownership
and frames are covered in `tests/articulation_handoff.rs` and
`tests/articulation_handoff_frames.rs`. `tests/assembly_lifecycle.rs` covers
merge/split remapping and lost-member cleanup; `tests/ragdoll_ecs.rs` covers
constraint authoring, invalid mappings, and handle refresh.
`tests/mpm_bridge.rs` covers runtime/snapshot lifecycle and bounded surfaces,
including surface-only edits; `tests/m15_bridge.rs` covers atomic routing,
ordered retry, retirement, and late baselines. Corresponding reference commands
are `cargo test -p prism_physics_ecs --test mpm_bridge` and
`cargo test -p prism_physics_ecs --test m15_bridge`.
