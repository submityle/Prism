//! Native, renderer-independent fixed-step physics. See the crate README for
//! ownership, coordinate-space, lifecycle and external-world mutation contracts.

pub mod articulation;
pub mod assembly;
pub mod field;
pub mod field_runtime;
mod m15_bridge;
mod mpm;
pub mod ragdoll;
mod sync;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Deref, DerefMut};

use prism_app::{App, FixedUpdate, Plugin};
use prism_ecs::prelude::*;
use prism_physics_core::{
    AeroBodyConfig, BodyDesc, BodyHandle, BodyKind, ColliderHandle, ColliderShape, MassProperties,
    PhysicsMaterial, PhysicsWorld, SoftBody, SoftBodyHandle, XpbdSolver,
};
use prism_transform::Transform;

pub use assembly::{
    AssemblyBody, AssemblyDiagnostic, AssemblyDiagnostics, AssemblyIssue, AssemblyRequestQueue,
    AssemblyRequestResult,
};
pub use field::{
    AeroBuoyancyComponent, AeroDiagnostic, AeroDiagnostics, AeroFunctionalComponents, AeroIssue,
    AeroSource, AeroSurfaceComponent, AeroThrusterComponent, FieldBindings, ForceField,
};
pub use field_runtime::{FieldRuntimeDiagnostics, FieldRuntimeResource};
pub use m15_bridge::{
    M15TopologyAdapter, M15TopologyAdapterError, M15TopologyBridge, M15TopologyBridgeError,
    M15TopologyBridgePlugin, M15TopologyDiagnostics,
};
pub use mpm::{
    MpmAuthority, MpmBinding, MpmDiagnostic, MpmDiagnostics, MpmGpuHost, MpmIssue,
    MpmParticleSnapshot, MpmResidentSurfaceConfig, MpmResidentSurfaceSnapshot, MpmRuntimeInfo,
    MpmRuntimeResource, MpmSurfaceConfig, MpmSurfaceSnapshot,
};
pub use ragdoll::{
    EcsRagdoll, EcsRagdollBindings, EcsRagdollDiagnostic, EcsRagdollDiagnostics,
    EcsRagdollIssue,
};
pub use prism_physics_gpu::{MpmBackendPolicy, MpmFallbackReason, MpmRuntimeError};

/// Physics numerical types (distinct from `prism_math` transform types).
pub use glam::{Quat, Vec3};
/// The engine-independent physics API, including query filters and soft bodies.
pub use prism_physics_core as core;
/// Geometry queries use physics numerical types.
pub use prism_physics_geometry as geometry;

/// Authored mass/motion policy. Changing any field rebuilds the body and handle.
/// Pose lives in `Transform`; velocity lives in [`Velocity`].
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct RigidBody {
    /// Static, externally posed kinematic, or simulated dynamic.
    pub kind: BodyKind,
    /// Explicit inverse mass and local principal inverse inertia; not shape-derived.
    pub mass: MassProperties,
    /// Nonnegative per-second damping.
    pub linear_damping: f32,
    /// Nonnegative per-second angular damping.
    pub angular_damping: f32,
    /// Continuous collision detection.
    pub ccd: bool,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self::dynamic()
    }
}

impl RigidBody {
    /// A unit-mass dynamic body with unit principal inertia and no damping.
    pub fn dynamic() -> Self {
        Self::from_desc(BodyDesc::dynamic_at(Vec3::ZERO))
    }

    /// An immovable static body.
    pub fn fixed() -> Self {
        Self::from_desc(BodyDesc::static_at(Vec3::ZERO))
    }

    /// An externally posed body; velocity does not move it automatically.
    pub fn kinematic() -> Self {
        Self {
            kind: BodyKind::Kinematic,
            ..Self::fixed()
        }
    }

    fn from_desc(desc: BodyDesc) -> Self {
        Self {
            kind: desc.kind,
            mass: desc.mass_properties,
            linear_damping: desc.linear_damping,
            angular_damping: desc.angular_damping,
            ccd: desc.ccd,
        }
    }
}

/// World-space velocity, both authored input and simulation output. Optional;
/// removing it explicitly clears both velocities, without automatically readding it.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct Velocity {
    /// Metres per second.
    pub linear: Vec3,
    /// Axis-angle radians per second in world space.
    pub angular: Vec3,
}

/// Inline analytic geometry or a shape registered in [`PhysicsWorldResource`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CollisionShape {
    /// Validated sphere/cuboid/capsule/plane. Arena-backed shapes must use Registered.
    Analytic(ColliderShape),
    /// Immutable geometry in this resource's shape registry, including meshes/compounds.
    Registered(ColliderHandle),
}

/// Optional collision state for the rigid body on the same entity. Removing
/// this component leaves a simulated body with no collision shape.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Collider {
    /// Geometry in body-local coordinates, with no implicit transform scaling.
    pub shape: CollisionShape,
    /// Friction and restitution.
    pub material: PhysicsMaterial,
    /// Sensors emit overlap transitions but do not resolve contacts.
    pub sensor: bool,
}

impl Collider {
    /// A solid sphere. Invalid dimensions are reported at the next fixed sync.
    pub fn sphere(radius: f32) -> Self {
        Self::analytic(ColliderShape::Sphere { radius })
    }

    /// An inline analytic shape.
    pub fn analytic(shape: ColliderShape) -> Self {
        Self {
            shape: CollisionShape::Analytic(shape),
            material: PhysicsMaterial::default(),
            sensor: false,
        }
    }

    /// A shape inserted into the owning physics world's registry.
    pub fn registered(handle: ColliderHandle) -> Self {
        Self {
            shape: CollisionShape::Registered(handle),
            material: PhysicsMaterial::default(),
            sensor: false,
        }
    }
}

/// Persistent world-space wrench applied at the centre of mass on **every**
/// substep. Remove/zero it to stop; one-shot impulses use the world resource API.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct ExternalForce {
    /// Newtons, world space.
    pub force: Vec3,
    /// Newton-metres, world space.
    pub torque: Vec3,
}

/// Rigid-body aerodynamic authoring data evaluated by the core during every
/// XPBD substep. Removing this component unregisters the body from the aero
/// pass; editing it replaces the registration without rebuilding the body.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct RigidBodyAero(pub AeroBodyConfig);

/// Authored soft-body template. Particles are world-space, independent of the
/// entity's Transform. Editing the template replaces the simulated body; live
/// particle output is read through the resource and [`PhysicsBindings`].
#[derive(Component, Clone, Debug, PartialEq)]
pub struct SoftBodyBinding(pub SoftBody);

/// Engine-independent simulation/query API. Direct mutations must follow the
/// ownership contract in README; managed body state is reconciled on fixed ticks.
#[derive(Resource, Default)]
pub struct PhysicsWorldResource(pub PhysicsWorld);

impl Deref for PhysicsWorldResource {
    type Target = PhysicsWorld;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for PhysicsWorldResource {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Replace/configure the default XPBD solver before a fixed tick.
#[derive(Resource, Default)]
pub struct PhysicsSolver(pub XpbdSolver);

#[derive(Clone)]
struct BodyBinding {
    handle: BodyHandle,
    authored: RigidBody,
    pose: Transform,
    velocity: Option<Velocity>,
    active: bool,
}

struct SoftBinding {
    handle: SoftBodyHandle,
    authored: SoftBody,
}

/// Generational identity for synchronizer-owned bodies and explicit assembly
/// aliases. Never treat a body slot index as identity. Maps are reconciled at
/// the next fixed tick, not immediately on ECS mutation.
#[derive(Resource, Default)]
pub struct PhysicsBindings {
    bodies: BTreeMap<Entity, BodyBinding>,
    entities: BTreeMap<(u32, u32), Entity>,
    retired: BTreeMap<(u32, u32), Entity>,
    soft: BTreeMap<Entity, SoftBinding>,
    soft_entities: BTreeMap<(u32, u32), Entity>,
    aero: BTreeMap<Entity, (BodyHandle, AeroBodyConfig)>,
    trigger_roles: BTreeMap<(u64, u64), (BodyHandle, BodyHandle)>,
    assembly_bodies: BTreeMap<Entity, BodyHandle>,
    assembly_entities: BTreeMap<(u32, u32), Entity>,
}

impl PhysicsBindings {
    /// Current body handle, including its generation. During an assembly merge,
    /// every member entity resolves to the shared proxy handle.
    pub fn body(&self, entity: Entity) -> Option<BodyHandle> {
        self.bodies
            .get(&entity)
            .map(|b| b.handle)
            .or_else(|| self.assembly_bodies.get(&entity).copied())
    }
    /// Entity identity for a current body. For a shared assembly proxy, this is
    /// the deterministic canonical member (the minimum generational Entity).
    pub fn entity(&self, body: BodyHandle) -> Option<Entity> {
        self.entities
            .get(&(body.index(), body.generation()))
            .copied()
            .or_else(|| {
                self.assembly_entities
                    .get(&(body.index(), body.generation()))
                    .copied()
            })
    }
    /// Current soft-body handle.
    pub fn soft_body(&self, entity: Entity) -> Option<SoftBodyHandle> {
        self.soft.get(&entity).map(|b| b.handle)
    }
    /// Entity identity for a current managed soft body.
    pub fn soft_entity(&self, body: SoftBodyHandle) -> Option<Entity> {
        self.soft_entities
            .get(&(body.index(), body.generation()))
            .copied()
    }
}

/// Endpoint retains the original handle and optional managed ECS identity.
/// Exit events may refer to an already-despawned entity (including its generation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContactEndpoint {
    /// Physics generational identity.
    pub body: BodyHandle,
    /// None for bodies created directly through the physics world resource.
    pub entity: Option<Entity>,
}

/// Contact transition sampled before solving and after each substep. A fixed
/// tick can contain both enter and exit (transient overlap). Changing the sensor
/// endpoint of an overlapping pair emits an exit followed by a new enter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Event)]
pub struct PhysicsContactEvent {
    /// Monotonic fixed tick, wrapping at u64::MAX.
    pub tick: u64,
    /// Transition family. For triggers, `first` is the sensor.
    pub kind: ContactEventKind,
    /// First endpoint.
    pub first: ContactEndpoint,
    /// Second endpoint.
    pub second: ContactEndpoint,
}

/// Mapped counterpart of the core collision and sensor transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContactEventKind {
    /// Solid contact began.
    CollisionStarted,
    /// Solid contact ended.
    CollisionEnded,
    /// Sensor overlap began.
    TriggerEntered,
    /// Sensor overlap ended.
    TriggerExited,
}

/// Explicit per-tick rejection/recovery reason; no invalid state is silently accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicsIssue {
    /// Rigid bodies require a local Transform.
    MissingTransform,
    /// Physics entities must be roots; nonphysics children are supported.
    ParentedBody,
    /// Physics Transform scale must be exactly one.
    NonUnitScale,
    /// Pose, mass, velocity, damping, collider or force is invalid/nonfinite.
    InvalidAuthoring,
    /// Core arena-backed geometry must be explicitly Registered.
    UnsupportedShape,
    /// Registered geometry no longer exists in the physics registry.
    MissingShape,
    /// A direct world mutation removed a managed body; a fresh body was created.
    StaleBody,
    /// A direct world mutation removed a managed soft body; it was rebound.
    StaleSoftBody,
    /// Core refused removal (e.g. an active merged assembly); split it externally.
    RemovalRefused,
    /// Invalid fixed dt/substep count or XPBD settings; simulation did not advance.
    InvalidConfiguration,
    /// Explicit GPU field sampling failed; earlier substeps remain committed.
    /// See [`FieldRuntimeDiagnostics`] for the source and completed count.
    FieldSamplingFailed,
    /// Native hierarchy propagation rejected the graph; globals were not written.
    TransformPropagation,
}

/// One diagnostic tied to an ECS entity, or to the whole simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicsDiagnostic {
    /// None for configuration/global hierarchy errors.
    pub entity: Option<Entity>,
    /// Failure or recovery classification.
    pub issue: PhysicsIssue,
}

/// Latest fixed tick's diagnostics and counters. No unbounded history retained.
#[derive(Resource, Default)]
pub struct PhysicsDiagnostics {
    /// Number of attempted fixed ticks.
    pub tick: u64,
    /// Accepted time on the latest tick, including only completed substeps on
    /// field sampling failure (zero on invalid settings).
    pub simulated_dt: f32,
    /// Diagnostics from the most recent fixed tick.
    pub issues: Vec<PhysicsDiagnostic>,
}

#[derive(Resource, Default)]
struct RemovalSignals {
    rigid: BTreeSet<Entity>,
    soft: BTreeSet<Entity>,
    velocity: BTreeSet<Entity>,
}

/// Ordering anchors within `FixedUpdate`. Add force/authoring systems before
/// Sync; consume completed output after Writeback or in FixedPostUpdate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, SystemSet)]
pub enum PhysicsSystems {
    /// Reconcile ownership that must precede ordinary body authoring sync.
    PreSync,
    /// Reconcile authored state and lifecycle.
    Sync,
    /// Advance XPBD and sample contact transitions.
    Step,
    /// Write simulated poses/velocities and propagate native world transforms.
    Writeback,
    /// Publish compound ownership transitions after ordinary body writeback.
    PostWriteback,
}

/// Installs a native headless fixed-step bridge. Preinserted resources survive.
#[derive(Default)]
pub struct PrismPhysicsPlugin;

impl Plugin for PrismPhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PhysicsWorldResource>()
            .init_resource::<PhysicsSolver>()
            .init_resource::<PhysicsBindings>()
            .init_resource::<PhysicsDiagnostics>()
            .init_resource::<AssemblyRequestQueue>()
            .init_resource::<AssemblyDiagnostics>()
            .init_resource::<FieldBindings>()
            .init_resource::<FieldRuntimeDiagnostics>()
            .init_resource::<MpmRuntimeResource>()
            .init_resource::<MpmDiagnostics>()
            .init_resource::<RemovalSignals>()
            .add_event::<PhysicsContactEvent>();
        let world = app.world_mut();
        mpm::init(world);
        prism_transform::ecs::install(world);
        world.observe::<RigidBody, _>(LifecycleEvent::Remove, |ctx| {
            ctx.world
                .resource_mut::<RemovalSignals>()
                .rigid
                .insert(ctx.entity);
        });
        world.observe::<Transform, _>(LifecycleEvent::Remove, |ctx| {
            ctx.world
                .resource_mut::<RemovalSignals>()
                .rigid
                .insert(ctx.entity);
        });
        world.observe::<Velocity, _>(LifecycleEvent::Remove, |ctx| {
            ctx.world
                .resource_mut::<RemovalSignals>()
                .velocity
                .insert(ctx.entity);
        });
        world.observe::<SoftBodyBinding, _>(LifecycleEvent::Remove, |ctx| {
            ctx.world
                .resource_mut::<RemovalSignals>()
                .soft
                .insert(ctx.entity);
        });
        app.add_systems(
            FixedUpdate,
            (
                (
                    sync::reconcile,
                    assembly::process,
                    ragdoll::reconcile,
                    field::reconcile,
                    mpm::sync,
                )
                    .chain()
                    .in_set(PhysicsSystems::Sync)
                    .before(PhysicsSystems::Step)
                    .before(PhysicsSystems::Writeback),
                (sync::step, mpm::step)
                    .chain()
                    .in_set(PhysicsSystems::Step)
                    .after(PhysicsSystems::Sync)
                    .before(PhysicsSystems::Writeback),
                (sync::writeback, mpm::writeback)
                    .chain()
                    .in_set(PhysicsSystems::Writeback)
                    .after(PhysicsSystems::Sync)
                    .after(PhysicsSystems::Step),
            )
                .chain()
                .after(PhysicsSystems::PreSync)
                .before(PhysicsSystems::PostWriteback),
        );
    }
}
