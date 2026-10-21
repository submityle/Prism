# Native Prism physics ECS bridge

Headless `prism_app` / `prism_ecs` / `prism_transform` integration for
`prism_physics_core` and the backend-selectable MPM runtime. No Bevy, renderer,
window, or gameplay dependency. XPBD rigid/soft bodies use the CPU core; MPM can
select the CPU reference or resident wgpu backend without exposing GPU handles
to ECS consumers.

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

The plugin installs native transform tracking and chains three systems in
`FixedUpdate`: `PhysicsSystems::Sync -> Step -> Writeback`. Authoring/force
systems should run `.before(PhysicsSystems::Sync)`; consumers can run
`.after(PhysicsSystems::Writeback)` or in `FixedPostUpdate`. Mutate physics
components before Sync, not between Sync and Writeback. Engine fixed-clock
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
and accepted simulated dt. Inspect it in a fixed-step consumer if every
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
substep**, so total impulse is force times the full fixed dt. Remove or zero it
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

`RigidBodyAero(AeroBodyConfig)` optionally registers core rigid-body aerodynamics
for the entity's managed body. Registration is reconciled during `Sync`, and
the core evaluates the config at XPBD substeps. Editing the config replaces the
registration without rebuilding the rigid body; removing the component or its
body unregisters it. Rebuilding a body rotates its generational handle, so the
bridge rebinds aero on the next fixed tick. Aero authoring on an entity without
a valid active `RigidBody` remains dormant until one is available. The component
exposes the core config as-is; it does not provide separate ECS components for
surfaces, fields, or runtime GPU sampling.

Native event buffers are frame-maintained, not fixed-tick-cleared, so a frame's
multiple fixed ticks remain readable. Use an `EventCursor` as usual. Exit events
retain the original entity identity even after despawn/reused body slots; check
entity liveness before querying its components. The plugin owns draining core
contact transitions; do not independently drain/reset the core contact tracker.

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
topology replacements. Its adapter receives the complete initial state and each
validated render/collision/CCD revision as an atomic unit. Adapter rejection
retains the update for ordered retry; retiring the ECS soft body unregisters its
downstream state. The bridge is renderer-independent: it does not itself stitch
a render mesh, compact GPU topology, or rebuild a production CCD/BVH consumer.

## Material-point simulations

`MpmBinding` owns authored initial particles, grid, material, numerical config,
backend policy, and per-advance work budget. It is reconciled at Sync into an
ECS-owned `MpmRuntime`; CPU and GPU execution share the same fixed-step boundary.
Changing any authored solver field, backend policy, or budget replaces the live
runtime. The authored component is never overwritten with evolving simulation
state.

Writeback publishes a backend-neutral `MpmParticleSnapshot` only after a
successful step and snapshot. Its version advances once per publication and
its tick identifies the fixed schedule that produced it. A fixed-dt mismatch,
runtime construction error, invalid output, or GPU readback error is diagnosed
without publishing fabricated state. In particular, a failed GPU snapshot does
not silently resume from the older CPU shadow; an explicit backend switch must
first synchronize successfully.

`MpmDiagnostics.simulated_dt` is the fixed simulation time accepted by the
latest tick, while `steps` is aggregate work across all MPM runtimes. Preferred
GPU construction either reports GPU authority or an explicit CPU fallback
reason. MPM particle output is available to renderer integrations, but this
crate does not install render extraction, surface reconstruction, or zero-copy
render-buffer sharing.

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

Managed handles must **not** participate in external assembly merge/split:
core hides the merged-member snapshots and split remapping, so the bridge
cannot distinguish a collapsed member from a removed body. Fully supporting
this requires a public merged-membership query and body-remap report from core.
When a live body's removal is refused by core, its binding is retained inactive
with `RemovalRefused` and retried, rather than silently dropped.

Body removal cleans attached joints created through `PhysicsWorld::spawn_joint`
(including disabled joints) and active raw-storage joints. Disabled joints
inserted directly into raw storage without assembly edges remain caller-owned:
core currently has no complete public generational joint iterator. Rebuilds
invalidate joints, grab/ragdoll, and other handle-based registrations instead
of migrating them. Aero registrations authored through `RigidBodyAero` are
reconciled to the replacement handle at the next Sync. No authored ECS joint,
articulation, assembly, or ragdoll component is supplied; these features need
explicit core resource orchestration with lifetime management by the caller.

There is no automatic streamed-terrain collider binding, render extraction,
interpolation, rollback/replication, scene serialization, arbitrary parent/scale
support, MPM surface reconstruction/zero-copy render sharing, or performance
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

Workspace command: `cargo test -p prism_physics_ecs --test app_lifecycle`.
MPM bridge coverage: `cargo test -p prism_physics_ecs --test mpm_bridge`.
M15 topology bridge coverage: `cargo test -p prism_physics_ecs --test m15_bridge`.
When other agents own the workspace lock/target, run the same test source from
an external path-dependency harness with an isolated target and lock instead.
