//! `Module` / `Force` / `DataInterface` plugin system (design §8).
//!
//! An authored particle system is assembled from small, composable plugins.
//! Following production stacks (Unreal `Niagara`, Unity `VFX Graph`,
//! `PopcornFX`), Ember splits those plugins into two contracts:
//!
//! * an [`EmberModule`] contributes a step of per-particle behaviour to one
//!   simulation [`StageKind`] (spawn, initialise, force, collide, constrain,
//!   age, or emit an event), declaring which attribute channels it needs and
//!   which `GPU` resources it binds; and
//! * a [`DataInterface`] exposes an external data source (a mesh, a texture, a
//!   `SDF`/`VDB` volume, the depth buffer, an audio spectrum, ...) as a set of
//!   `sample_*` functions the modules can call from generated shader code.
//!
//! This module owns the *`CPU`-verifiable* half of that contract: the plugin
//! traits, the built-in plugin registries with their stage/attribute/resource
//! metadata, and a small [`CodegenCtx`] that accumulates inline `WESL`
//! fragments and de-duplicates binding declarations. Actually running the
//! generated `WESL`/`WGSL` on a `GPU` is out of scope for this layer, so the
//! codegen hooks only assemble deterministic source fragments that tests can
//! assert on.
//!
//! The [`StageKind`] defined here is the *plugin-level* categorisation from
//! design §8; it is a coarser, authoring-facing grouping than the compiled
//! iteration domains in [`super::stages`] and the compiled stage graph in
//! [`super::graph`], and [`StageKind::default_domain`] maps each plugin stage to
//! the [`IterationDomain`] its kernels dispatch under.

use alloc::string::String;
use alloc::vec::Vec;

use super::attributes::{AttributeFormat, AttributeSemantic};
use super::IterationDomain;

/// The plugin-level simulation stage a [`EmberModule`] contributes to
/// (design §8.1).
///
/// This is the authoring-facing grouping that decides *when* in the per-frame
/// pipeline a module runs. It is intentionally coarser than the compiled
/// iteration domains in [`super::stages`]; use [`StageKind::default_domain`] to
/// recover the [`IterationDomain`] a stage dispatches under.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StageKind {
    /// Spawns new particles into free pool slots (rate / burst / distribution).
    Emit,
    /// Initialises freshly spawned particles (set / random-range / inherit).
    Init,
    /// A generic per-particle update step (integration and bookkeeping).
    Update,
    /// Accumulates a force into velocity (gravity, drag, attractors, ...).
    Force,
    /// Resolves collisions against planes, depth, `SDF` volumes, or rays.
    Collision,
    /// Solves a position constraint batch (`XPBD` distance / bend / volume).
    Constraint,
    /// Ages particles and applies over-life curves; kills the expired.
    Lifecycle,
    /// Emits events that can spawn into other emitters (death / collision /
    /// spawn / condition).
    Event,
}

impl StageKind {
    /// Every stage kind in stable declaration order.
    pub const ALL: [StageKind; 8] = [
        StageKind::Emit,
        StageKind::Init,
        StageKind::Update,
        StageKind::Force,
        StageKind::Collision,
        StageKind::Constraint,
        StageKind::Lifecycle,
        StageKind::Event,
    ];

    /// The stable ordinal used for serialisation and registry indexing.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        self as u32
    }

    /// Reconstructs a stage kind from its [`StageKind::ordinal`]; returns `None`
    /// for an out-of-range value so the mapping round-trips exactly.
    #[must_use]
    pub const fn from_ordinal(value: u32) -> Option<StageKind> {
        let index = value as usize;
        if index < Self::ALL.len() {
            Some(Self::ALL[index])
        } else {
            None
        }
    }

    /// The [`IterationDomain`] this plugin stage dispatches under (design §7).
    ///
    /// Emission, initialisation, forces, collisions, and lifecycle all run one
    /// invocation per live particle. Constraint solving runs per constraint
    /// batch, and event stages run per buffered event.
    #[must_use]
    pub const fn default_domain(self) -> IterationDomain {
        match self {
            StageKind::Emit
            | StageKind::Init
            | StageKind::Update
            | StageKind::Force
            | StageKind::Collision
            | StageKind::Lifecycle => IterationDomain::PerParticle,
            StageKind::Constraint => IterationDomain::PerConstraint,
            StageKind::Event => IterationDomain::PerEvent,
        }
    }
}

/// The kind of `GPU` resource a plugin declares (design §8.1).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceKind {
    /// A small constant block of parameters (a `WGSL` `uniform`).
    Uniform,
    /// A one-dimensional look-up table sampled by normalised age (`LUT`).
    Curve,
    /// A two-dimensional texture (depth buffer, scene normals, 2D fields).
    Texture2d,
    /// A three-dimensional texture (vector fields, density, `SDF` volumes).
    Texture3d,
    /// A shader storage buffer (meshes, point caches, acceleration structures).
    StorageBuffer,
}

/// Whether a plugin binds a resource read-only or read-write (design §8.1).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceAccess {
    /// The resource is only read by generated code.
    ReadOnly,
    /// The resource is both read and written by generated code.
    ReadWrite,
}

/// A single `GPU` resource declaration: a named binding slot, its kind, its
/// access, the element/texel format its samples produce, and the
/// `group`/`binding` slot it occupies (design §8.1).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ResourceDecl {
    /// The `WGSL` identifier the binding is exposed under.
    pub name: &'static str,
    /// The resource kind (uniform / curve / texture / buffer).
    pub kind: ResourceKind,
    /// Whether generated code writes the resource.
    pub access: ResourceAccess,
    /// The element or texel format a sample of the resource yields.
    pub format: AttributeFormat,
    /// The `WGSL` bind-group index.
    pub group: u32,
    /// The `WGSL` binding index within the group.
    pub binding: u32,
}

impl ResourceDecl {
    /// A fully specified resource declaration.
    #[must_use]
    pub const fn new(
        name: &'static str,
        kind: ResourceKind,
        access: ResourceAccess,
        format: AttributeFormat,
        group: u32,
        binding: u32,
    ) -> Self {
        Self {
            name,
            kind,
            access,
            format,
            group,
            binding,
        }
    }

    /// A read-only uniform parameter block, sampled as a four-lane vector.
    #[must_use]
    pub const fn uniform(name: &'static str, group: u32, binding: u32) -> Self {
        Self::new(
            name,
            ResourceKind::Uniform,
            ResourceAccess::ReadOnly,
            AttributeFormat::Vec4,
            group,
            binding,
        )
    }

    /// A read-only over-life curve look-up table, sampled as a four-lane vector.
    #[must_use]
    pub const fn curve(name: &'static str, group: u32, binding: u32) -> Self {
        Self::new(
            name,
            ResourceKind::Curve,
            ResourceAccess::ReadOnly,
            AttributeFormat::Vec4,
            group,
            binding,
        )
    }

    /// A read-only two-dimensional texture, sampled as a four-lane vector.
    #[must_use]
    pub const fn texture2d(name: &'static str, group: u32, binding: u32) -> Self {
        Self::new(
            name,
            ResourceKind::Texture2d,
            ResourceAccess::ReadOnly,
            AttributeFormat::Vec4,
            group,
            binding,
        )
    }

    /// A read-only three-dimensional texture, sampled as a four-lane vector.
    #[must_use]
    pub const fn texture3d(name: &'static str, group: u32, binding: u32) -> Self {
        Self::new(
            name,
            ResourceKind::Texture3d,
            ResourceAccess::ReadOnly,
            AttributeFormat::Vec4,
            group,
            binding,
        )
    }

    /// A read-only shader storage buffer, addressed as raw 32-bit words.
    #[must_use]
    pub const fn buffer(name: &'static str, group: u32, binding: u32) -> Self {
        Self::new(
            name,
            ResourceKind::StorageBuffer,
            ResourceAccess::ReadOnly,
            AttributeFormat::U32,
            group,
            binding,
        )
    }

    /// A read-write shader storage buffer, addressed as raw 32-bit words.
    #[must_use]
    pub const fn storage_rw(name: &'static str, group: u32, binding: u32) -> Self {
        Self::new(
            name,
            ResourceKind::StorageBuffer,
            ResourceAccess::ReadWrite,
            AttributeFormat::U32,
            group,
            binding,
        )
    }

    /// Whether generated code writes this resource.
    #[must_use]
    pub const fn is_writable(self) -> bool {
        matches!(self.access, ResourceAccess::ReadWrite)
    }
}

/// The minimal `WESL` code-generation context a plugin emits into (design §8.1).
///
/// It is deliberately not a shader compiler: it only accumulates inline source
/// fragments in emission order and records the set of resource bindings the
/// assembled kernel needs, de-duplicating repeated declarations by name. The
/// backend consumes the fragments and bindings to build the final `WESL`
/// module; tests assert on the accumulated strings without touching a `GPU`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CodegenCtx {
    fragments: Vec<String>,
    bindings: Vec<ResourceDecl>,
}

impl CodegenCtx {
    /// An empty context with no fragments and no bindings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether nothing has been emitted yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty() && self.bindings.is_empty()
    }

    /// Appends an inline source fragment in emission order.
    pub fn push_fragment(&mut self, code: impl Into<String>) {
        self.fragments.push(code.into());
    }

    /// Records a resource binding, de-duplicating by [`ResourceDecl::name`].
    ///
    /// Returns `true` if the binding was newly added, or `false` if a binding
    /// with the same name was already declared (so plugins that share a data
    /// source declare it only once in the assembled kernel).
    pub fn declare_binding(&mut self, decl: ResourceDecl) -> bool {
        if self
            .bindings
            .iter()
            .any(|existing| existing.name == decl.name)
        {
            return false;
        }
        self.bindings.push(decl);
        true
    }

    /// The accumulated source fragments in emission order.
    #[must_use]
    pub fn fragments(&self) -> &[String] {
        &self.fragments
    }

    /// The de-duplicated resource bindings the assembled kernel needs.
    #[must_use]
    pub fn bindings(&self) -> &[ResourceDecl] {
        &self.bindings
    }

    /// The number of accumulated fragments.
    #[must_use]
    pub fn fragment_count(&self) -> usize {
        self.fragments.len()
    }

    /// The number of distinct bindings.
    #[must_use]
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// The fragments joined into a single newline-separated source string.
    #[must_use]
    pub fn assembled_source(&self) -> String {
        let mut out = String::new();
        for (index, fragment) in self.fragments.iter().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            out.push_str(fragment);
        }
        out
    }
}

/// The unified plugin contract every module implements (design §8.1).
///
/// A module reports the stage it runs in, the attribute channels it touches
/// (so the layout planner in [`super::attributes`] allocates them on demand),
/// the `GPU` resources it binds, and a hook that emits its inline `WESL` into a
/// [`CodegenCtx`].
pub trait EmberModule {
    /// The simulation stage this module contributes to.
    fn stage(&self) -> StageKind;
    /// The attribute channels this module reads or writes (allocated on demand).
    fn required_attributes(&self) -> &[AttributeSemantic];
    /// The `GPU` resources this module binds.
    fn resources(&self) -> &[ResourceDecl];
    /// Emits this module's inline `WESL` fragment and binding declarations.
    fn emit_wesl(&self, ctx: &mut CodegenCtx);
}

/// A data source exposed to modules as callable `sample_*` functions
/// (design §8.1, §8.3).
pub trait DataInterface {
    /// The `GPU` resources this interface binds.
    fn bindings(&self) -> &[ResourceDecl];
    /// Emits the `sample_*` function declarations and binding declarations this
    /// interface provides into a [`CodegenCtx`].
    fn emit_wesl_functions(&self, ctx: &mut CodegenCtx);
}

// --- Built-in module metadata tables (design §8.2) -------------------------

/// The immutable metadata backing one [`BuiltinModule`].
struct ModuleMeta {
    stage: StageKind,
    attributes: &'static [AttributeSemantic],
    resources: &'static [ResourceDecl],
    kernel: &'static str,
}

const A_EMPTY: &[AttributeSemantic] = &[];
const A_SPAWN_BASE: &[AttributeSemantic] = &[
    AttributeSemantic::Alive,
    AttributeSemantic::ParticleId,
    AttributeSemantic::Age,
    AttributeSemantic::Lifetime,
];
const A_SPAWN_SHAPED: &[AttributeSemantic] = &[
    AttributeSemantic::Position,
    AttributeSemantic::Alive,
    AttributeSemantic::ParticleId,
    AttributeSemantic::Age,
    AttributeSemantic::Lifetime,
];
const A_POSITION: &[AttributeSemantic] = &[AttributeSemantic::Position];
const A_VELOCITY: &[AttributeSemantic] = &[AttributeSemantic::Velocity];
const A_POS_VEL: &[AttributeSemantic] = &[AttributeSemantic::Position, AttributeSemantic::Velocity];
const A_COLOR: &[AttributeSemantic] = &[AttributeSemantic::Color];
const A_SIZE: &[AttributeSemantic] = &[AttributeSemantic::Size];
const A_ROTATION: &[AttributeSemantic] = &[AttributeSemantic::Rotation];
const A_LIFETIME_AGE: &[AttributeSemantic] = &[AttributeSemantic::Lifetime, AttributeSemantic::Age];
const A_AGE: &[AttributeSemantic] = &[AttributeSemantic::Age];
const A_AGEKILL: &[AttributeSemantic] = &[
    AttributeSemantic::Age,
    AttributeSemantic::Lifetime,
    AttributeSemantic::Alive,
];
const A_COLOR_LIFE: &[AttributeSemantic] = &[
    AttributeSemantic::Age,
    AttributeSemantic::Lifetime,
    AttributeSemantic::Color,
];
const A_SIZE_LIFE: &[AttributeSemantic] = &[
    AttributeSemantic::Age,
    AttributeSemantic::Lifetime,
    AttributeSemantic::Size,
];
const A_VEL_LIFE: &[AttributeSemantic] = &[
    AttributeSemantic::Age,
    AttributeSemantic::Lifetime,
    AttributeSemantic::Velocity,
];
const A_DEATH: &[AttributeSemantic] = &[AttributeSemantic::Alive, AttributeSemantic::Position];
const A_PARTICLE_ID: &[AttributeSemantic] = &[AttributeSemantic::ParticleId];

const R_EMPTY: &[ResourceDecl] = &[];
const R_GRAVITY: &[ResourceDecl] = &[ResourceDecl::uniform("gravity", 2, 0)];
const R_DRAG: &[ResourceDecl] = &[ResourceDecl::uniform("drag", 2, 0)];
const R_VORTEX: &[ResourceDecl] = &[ResourceDecl::uniform("vortex", 2, 0)];
const R_POINT_ATTRACTOR: &[ResourceDecl] = &[ResourceDecl::uniform("point_attractor", 2, 0)];
const R_LINE_ATTRACTOR: &[ResourceDecl] = &[ResourceDecl::uniform("line_attractor", 2, 0)];
const R_CURL_NOISE: &[ResourceDecl] = &[ResourceDecl::uniform("curl_noise", 2, 0)];
const R_WIND: &[ResourceDecl] = &[ResourceDecl::uniform("wind", 2, 0)];
const R_VECTOR_FIELD: &[ResourceDecl] = &[ResourceDecl::texture3d("vector_field", 2, 0)];
const R_COLLISION_PLANE: &[ResourceDecl] = &[ResourceDecl::uniform("collision_plane", 2, 0)];
const R_DEPTH: &[ResourceDecl] = &[ResourceDecl::texture2d("depth_buffer", 2, 0)];
const R_COLLISION_SDF: &[ResourceDecl] = &[ResourceDecl::texture3d("collision_sdf", 2, 0)];
const R_RAYTRACE: &[ResourceDecl] = &[ResourceDecl::buffer("raytrace_scene", 2, 0)];
const R_BOUNCE: &[ResourceDecl] = &[ResourceDecl::uniform("bounce_params", 2, 0)];
const R_COLOR_CURVE: &[ResourceDecl] = &[ResourceDecl::curve("color_over_life", 2, 0)];
const R_SIZE_CURVE: &[ResourceDecl] = &[ResourceDecl::curve("size_over_life", 2, 0)];
const R_VEL_CURVE: &[ResourceDecl] = &[ResourceDecl::curve("velocity_over_life", 2, 0)];
const R_RANDOM_RANGE: &[ResourceDecl] = &[ResourceDecl::uniform("random_range", 2, 0)];
const R_SPAWN_RATE: &[ResourceDecl] = &[ResourceDecl::uniform("spawn_rate", 2, 0)];
const R_SPAWN_BURST: &[ResourceDecl] = &[ResourceDecl::uniform("spawn_burst", 2, 0)];
const R_SPAWN_SHAPE: &[ResourceDecl] = &[ResourceDecl::uniform("spawn_shape", 2, 0)];

/// The first-ship built-in module library (design §8.2).
///
/// Every variant maps deterministically to a [`StageKind`], the set of
/// attribute semantics it reads or writes, the `GPU` resources it binds, and a
/// stable `WESL` kernel identifier. The enum is fieldless: authored parameter
/// values live in the graph's authoring data, while this registry provides the
/// static compile-time contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BuiltinModule {
    /// Spawn a steady number of particles per second.
    SpawnRate,
    /// Spawn a fixed count at one instant.
    SpawnBurst,
    /// Spawn distributed at a point.
    SpawnPoint,
    /// Spawn distributed over a sphere.
    SpawnSphere,
    /// Spawn distributed inside a box.
    SpawnBox,
    /// Spawn distributed inside a cone.
    SpawnCone,
    /// Spawn distributed over a mesh surface.
    SpawnMeshSurface,
    /// Spawn distributed over a skeletal mesh.
    SpawnSkeleton,
    /// Spawn distributed over a `SDF` surface.
    SpawnSdfSurface,
    /// Initialise position.
    SetPosition,
    /// Initialise velocity.
    SetVelocity,
    /// Initialise colour.
    SetColor,
    /// Initialise size.
    SetSize,
    /// Initialise lifetime (and reset age).
    SetLifetime,
    /// Initialise rotation.
    SetRotation,
    /// Assign a uniformly random value inside an authored range.
    RandomRange,
    /// Inherit the emitter's velocity at spawn.
    InheritVelocity,
    /// Constant gravitational acceleration.
    Gravity,
    /// Velocity-proportional drag.
    Drag,
    /// Swirling vortex force about an axis.
    Vortex,
    /// Attraction toward a point.
    PointAttractor,
    /// Attraction toward a line.
    LineAttractor,
    /// Divergence-free curl-noise turbulence.
    CurlNoise,
    /// Directional wind force.
    Wind,
    /// A force sampled from a 3D vector-field texture.
    VectorField,
    /// Collision against an infinite plane.
    CollisionPlane,
    /// Collision against the scene depth buffer.
    CollisionDepth,
    /// Collision against a `SDF` volume.
    CollisionSdf,
    /// Collision resolved by tracing rays against the scene.
    CollisionRaytrace,
    /// Bounce and friction response after a collision.
    CollisionBounce,
    /// `XPBD` distance constraint.
    DistanceConstraint,
    /// `XPBD` bend constraint.
    BendConstraint,
    /// `XPBD` volume-preservation constraint.
    VolumeConstraint,
    /// Kill particles whose age has reached their lifetime.
    AgeKill,
    /// Drive colour from an over-life curve.
    ColorOverLife,
    /// Drive size from an over-life curve.
    SizeOverLife,
    /// Drive velocity from an over-life curve.
    VelocityOverLife,
    /// Emit an event when a particle dies.
    EmitOnDeath,
    /// Emit an event when a particle collides.
    EmitOnCollision,
    /// Emit an event when a particle spawns.
    EmitOnSpawn,
    /// Emit an event when an authored condition passes.
    EmitOnCondition,
}

impl BuiltinModule {
    /// Every built-in module in stable declaration order.
    pub const ALL: [BuiltinModule; 41] = [
        BuiltinModule::SpawnRate,
        BuiltinModule::SpawnBurst,
        BuiltinModule::SpawnPoint,
        BuiltinModule::SpawnSphere,
        BuiltinModule::SpawnBox,
        BuiltinModule::SpawnCone,
        BuiltinModule::SpawnMeshSurface,
        BuiltinModule::SpawnSkeleton,
        BuiltinModule::SpawnSdfSurface,
        BuiltinModule::SetPosition,
        BuiltinModule::SetVelocity,
        BuiltinModule::SetColor,
        BuiltinModule::SetSize,
        BuiltinModule::SetLifetime,
        BuiltinModule::SetRotation,
        BuiltinModule::RandomRange,
        BuiltinModule::InheritVelocity,
        BuiltinModule::Gravity,
        BuiltinModule::Drag,
        BuiltinModule::Vortex,
        BuiltinModule::PointAttractor,
        BuiltinModule::LineAttractor,
        BuiltinModule::CurlNoise,
        BuiltinModule::Wind,
        BuiltinModule::VectorField,
        BuiltinModule::CollisionPlane,
        BuiltinModule::CollisionDepth,
        BuiltinModule::CollisionSdf,
        BuiltinModule::CollisionRaytrace,
        BuiltinModule::CollisionBounce,
        BuiltinModule::DistanceConstraint,
        BuiltinModule::BendConstraint,
        BuiltinModule::VolumeConstraint,
        BuiltinModule::AgeKill,
        BuiltinModule::ColorOverLife,
        BuiltinModule::SizeOverLife,
        BuiltinModule::VelocityOverLife,
        BuiltinModule::EmitOnDeath,
        BuiltinModule::EmitOnCollision,
        BuiltinModule::EmitOnSpawn,
        BuiltinModule::EmitOnCondition,
    ];

    /// The static metadata backing this module.
    const fn meta(self) -> ModuleMeta {
        match self {
            BuiltinModule::SpawnRate => ModuleMeta {
                stage: StageKind::Emit,
                attributes: A_SPAWN_BASE,
                resources: R_SPAWN_RATE,
                kernel: "spawn_rate",
            },
            BuiltinModule::SpawnBurst => ModuleMeta {
                stage: StageKind::Emit,
                attributes: A_SPAWN_BASE,
                resources: R_SPAWN_BURST,
                kernel: "spawn_burst",
            },
            BuiltinModule::SpawnPoint => ModuleMeta {
                stage: StageKind::Emit,
                attributes: A_SPAWN_SHAPED,
                resources: R_SPAWN_SHAPE,
                kernel: "spawn_point",
            },
            BuiltinModule::SpawnSphere => ModuleMeta {
                stage: StageKind::Emit,
                attributes: A_SPAWN_SHAPED,
                resources: R_SPAWN_SHAPE,
                kernel: "spawn_sphere",
            },
            BuiltinModule::SpawnBox => ModuleMeta {
                stage: StageKind::Emit,
                attributes: A_SPAWN_SHAPED,
                resources: R_SPAWN_SHAPE,
                kernel: "spawn_box",
            },
            BuiltinModule::SpawnCone => ModuleMeta {
                stage: StageKind::Emit,
                attributes: A_SPAWN_SHAPED,
                resources: R_SPAWN_SHAPE,
                kernel: "spawn_cone",
            },
            BuiltinModule::SpawnMeshSurface => ModuleMeta {
                stage: StageKind::Emit,
                attributes: A_SPAWN_SHAPED,
                resources: R_SPAWN_SHAPE,
                kernel: "spawn_mesh_surface",
            },
            BuiltinModule::SpawnSkeleton => ModuleMeta {
                stage: StageKind::Emit,
                attributes: A_SPAWN_SHAPED,
                resources: R_SPAWN_SHAPE,
                kernel: "spawn_skeleton",
            },
            BuiltinModule::SpawnSdfSurface => ModuleMeta {
                stage: StageKind::Emit,
                attributes: A_SPAWN_SHAPED,
                resources: R_SPAWN_SHAPE,
                kernel: "spawn_sdf_surface",
            },
            BuiltinModule::SetPosition => ModuleMeta {
                stage: StageKind::Init,
                attributes: A_POSITION,
                resources: R_EMPTY,
                kernel: "set_position",
            },
            BuiltinModule::SetVelocity => ModuleMeta {
                stage: StageKind::Init,
                attributes: A_VELOCITY,
                resources: R_EMPTY,
                kernel: "set_velocity",
            },
            BuiltinModule::SetColor => ModuleMeta {
                stage: StageKind::Init,
                attributes: A_COLOR,
                resources: R_EMPTY,
                kernel: "set_color",
            },
            BuiltinModule::SetSize => ModuleMeta {
                stage: StageKind::Init,
                attributes: A_SIZE,
                resources: R_EMPTY,
                kernel: "set_size",
            },
            BuiltinModule::SetLifetime => ModuleMeta {
                stage: StageKind::Init,
                attributes: A_LIFETIME_AGE,
                resources: R_EMPTY,
                kernel: "set_lifetime",
            },
            BuiltinModule::SetRotation => ModuleMeta {
                stage: StageKind::Init,
                attributes: A_ROTATION,
                resources: R_EMPTY,
                kernel: "set_rotation",
            },
            BuiltinModule::RandomRange => ModuleMeta {
                stage: StageKind::Init,
                attributes: A_EMPTY,
                resources: R_RANDOM_RANGE,
                kernel: "random_range",
            },
            BuiltinModule::InheritVelocity => ModuleMeta {
                stage: StageKind::Init,
                attributes: A_VELOCITY,
                resources: R_EMPTY,
                kernel: "inherit_velocity",
            },
            BuiltinModule::Gravity => ModuleMeta {
                stage: StageKind::Force,
                attributes: A_VELOCITY,
                resources: R_GRAVITY,
                kernel: "gravity",
            },
            BuiltinModule::Drag => ModuleMeta {
                stage: StageKind::Force,
                attributes: A_VELOCITY,
                resources: R_DRAG,
                kernel: "drag",
            },
            BuiltinModule::Vortex => ModuleMeta {
                stage: StageKind::Force,
                attributes: A_POS_VEL,
                resources: R_VORTEX,
                kernel: "vortex",
            },
            BuiltinModule::PointAttractor => ModuleMeta {
                stage: StageKind::Force,
                attributes: A_POS_VEL,
                resources: R_POINT_ATTRACTOR,
                kernel: "point_attractor",
            },
            BuiltinModule::LineAttractor => ModuleMeta {
                stage: StageKind::Force,
                attributes: A_POS_VEL,
                resources: R_LINE_ATTRACTOR,
                kernel: "line_attractor",
            },
            BuiltinModule::CurlNoise => ModuleMeta {
                stage: StageKind::Force,
                attributes: A_POS_VEL,
                resources: R_CURL_NOISE,
                kernel: "curl_noise",
            },
            BuiltinModule::Wind => ModuleMeta {
                stage: StageKind::Force,
                attributes: A_VELOCITY,
                resources: R_WIND,
                kernel: "wind",
            },
            BuiltinModule::VectorField => ModuleMeta {
                stage: StageKind::Force,
                attributes: A_POS_VEL,
                resources: R_VECTOR_FIELD,
                kernel: "vector_field",
            },
            BuiltinModule::CollisionPlane => ModuleMeta {
                stage: StageKind::Collision,
                attributes: A_POS_VEL,
                resources: R_COLLISION_PLANE,
                kernel: "collision_plane",
            },
            BuiltinModule::CollisionDepth => ModuleMeta {
                stage: StageKind::Collision,
                attributes: A_POS_VEL,
                resources: R_DEPTH,
                kernel: "collision_depth",
            },
            BuiltinModule::CollisionSdf => ModuleMeta {
                stage: StageKind::Collision,
                attributes: A_POS_VEL,
                resources: R_COLLISION_SDF,
                kernel: "collision_sdf",
            },
            BuiltinModule::CollisionRaytrace => ModuleMeta {
                stage: StageKind::Collision,
                attributes: A_POS_VEL,
                resources: R_RAYTRACE,
                kernel: "collision_raytrace",
            },
            BuiltinModule::CollisionBounce => ModuleMeta {
                stage: StageKind::Collision,
                attributes: A_POS_VEL,
                resources: R_BOUNCE,
                kernel: "collision_bounce",
            },
            BuiltinModule::DistanceConstraint => ModuleMeta {
                stage: StageKind::Constraint,
                attributes: A_POSITION,
                resources: R_EMPTY,
                kernel: "distance_constraint",
            },
            BuiltinModule::BendConstraint => ModuleMeta {
                stage: StageKind::Constraint,
                attributes: A_POSITION,
                resources: R_EMPTY,
                kernel: "bend_constraint",
            },
            BuiltinModule::VolumeConstraint => ModuleMeta {
                stage: StageKind::Constraint,
                attributes: A_POSITION,
                resources: R_EMPTY,
                kernel: "volume_constraint",
            },
            BuiltinModule::AgeKill => ModuleMeta {
                stage: StageKind::Lifecycle,
                attributes: A_AGEKILL,
                resources: R_EMPTY,
                kernel: "age_kill",
            },
            BuiltinModule::ColorOverLife => ModuleMeta {
                stage: StageKind::Lifecycle,
                attributes: A_COLOR_LIFE,
                resources: R_COLOR_CURVE,
                kernel: "color_over_life",
            },
            BuiltinModule::SizeOverLife => ModuleMeta {
                stage: StageKind::Lifecycle,
                attributes: A_SIZE_LIFE,
                resources: R_SIZE_CURVE,
                kernel: "size_over_life",
            },
            BuiltinModule::VelocityOverLife => ModuleMeta {
                stage: StageKind::Lifecycle,
                attributes: A_VEL_LIFE,
                resources: R_VEL_CURVE,
                kernel: "velocity_over_life",
            },
            BuiltinModule::EmitOnDeath => ModuleMeta {
                stage: StageKind::Event,
                attributes: A_DEATH,
                resources: R_EMPTY,
                kernel: "emit_on_death",
            },
            BuiltinModule::EmitOnCollision => ModuleMeta {
                stage: StageKind::Event,
                attributes: A_POS_VEL,
                resources: R_EMPTY,
                kernel: "emit_on_collision",
            },
            BuiltinModule::EmitOnSpawn => ModuleMeta {
                stage: StageKind::Event,
                attributes: A_PARTICLE_ID,
                resources: R_EMPTY,
                kernel: "emit_on_spawn",
            },
            BuiltinModule::EmitOnCondition => ModuleMeta {
                stage: StageKind::Event,
                attributes: A_AGE,
                resources: R_EMPTY,
                kernel: "emit_on_condition",
            },
        }
    }

    /// The simulation stage this module runs in.
    #[must_use]
    pub const fn stage(self) -> StageKind {
        self.meta().stage
    }

    /// The attribute channels this module reads or writes.
    #[must_use]
    pub const fn required_attributes(self) -> &'static [AttributeSemantic] {
        self.meta().attributes
    }

    /// The `GPU` resources this module binds.
    #[must_use]
    pub const fn resources(self) -> &'static [ResourceDecl] {
        self.meta().resources
    }

    /// The stable `WESL` kernel identifier generated code calls for this module.
    #[must_use]
    pub const fn kernel_name(self) -> &'static str {
        self.meta().kernel
    }

    /// The stable ordinal used for serialisation and registry indexing.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        self as u32
    }

    /// Reconstructs a module from its [`BuiltinModule::ordinal`]; returns `None`
    /// for an out-of-range value so the mapping round-trips exactly.
    #[must_use]
    pub const fn from_ordinal(value: u32) -> Option<BuiltinModule> {
        let index = value as usize;
        if index < Self::ALL.len() {
            Some(Self::ALL[index])
        } else {
            None
        }
    }
}

impl EmberModule for BuiltinModule {
    fn stage(&self) -> StageKind {
        self.meta().stage
    }

    fn required_attributes(&self) -> &[AttributeSemantic] {
        self.meta().attributes
    }

    fn resources(&self) -> &[ResourceDecl] {
        self.meta().resources
    }

    fn emit_wesl(&self, ctx: &mut CodegenCtx) {
        let meta = self.meta();
        for &decl in meta.resources {
            let _ = ctx.declare_binding(decl);
        }
        let mut call = String::from(meta.kernel);
        call.push_str("(particle_index);");
        ctx.push_fragment(call);
    }
}

// --- Built-in data-interface catalog (design §8.3) -------------------------

/// The category a [`BuiltinDataInterface`] belongs to (design §8.3).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BindingClass {
    /// Geometry sources: meshes, skeletal meshes, splines.
    Geometry,
    /// Field sources: textures, `SDF`/`VDB` volumes, point caches.
    Field,
    /// Scene sources: camera, depth, normals, lights, ray-tracing, `GI` probes.
    Scene,
    /// Signal sources: audio spectra, curves, and global time/frame state.
    Signal,
    /// Interaction sources: user parameters and custom grids/neighbour lists.
    Interaction,
}

impl BindingClass {
    /// Every binding class in stable declaration order.
    pub const ALL: [BindingClass; 5] = [
        BindingClass::Geometry,
        BindingClass::Field,
        BindingClass::Scene,
        BindingClass::Signal,
        BindingClass::Interaction,
    ];

    /// The stable ordinal used for serialisation.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        self as u32
    }

    /// Reconstructs a binding class from its ordinal; returns `None` when out of
    /// range so the mapping round-trips exactly.
    #[must_use]
    pub const fn from_ordinal(value: u32) -> Option<BindingClass> {
        let index = value as usize;
        if index < Self::ALL.len() {
            Some(Self::ALL[index])
        } else {
            None
        }
    }
}

/// The immutable metadata backing one [`BuiltinDataInterface`].
struct DataInterfaceMeta {
    class: BindingClass,
    bindings: &'static [ResourceDecl],
    functions: &'static [&'static str],
}

const B_MESH: &[ResourceDecl] = &[
    ResourceDecl::buffer("mesh_vertices", 3, 0),
    ResourceDecl::buffer("mesh_indices", 3, 1),
];
const B_SKELETAL: &[ResourceDecl] = &[
    ResourceDecl::buffer("skeletal_vertices", 3, 0),
    ResourceDecl::buffer("skeletal_bones", 3, 1),
];
const B_SPLINE: &[ResourceDecl] = &[ResourceDecl::buffer("spline_control_points", 3, 0)];
const B_TEX2D: &[ResourceDecl] = &[ResourceDecl::texture2d("field_texture_2d", 3, 0)];
const B_TEX3D: &[ResourceDecl] = &[ResourceDecl::texture3d("field_texture_3d", 3, 0)];
const B_SDF: &[ResourceDecl] = &[ResourceDecl::texture3d("sdf_volume", 3, 0)];
const B_VDB: &[ResourceDecl] = &[ResourceDecl::buffer("vdb_tree", 3, 0)];
const B_POINT_CACHE: &[ResourceDecl] = &[ResourceDecl::buffer("point_cache", 3, 0)];
const B_CAMERA: &[ResourceDecl] = &[ResourceDecl::uniform("camera", 3, 0)];
const B_DEPTH: &[ResourceDecl] = &[ResourceDecl::texture2d("scene_depth", 3, 0)];
const B_NORMAL: &[ResourceDecl] = &[ResourceDecl::texture2d("scene_normal", 3, 0)];
const B_LIGHT: &[ResourceDecl] = &[ResourceDecl::buffer("clustered_lights", 3, 0)];
const B_RAYTRACE: &[ResourceDecl] = &[ResourceDecl::buffer("raytrace_tlas", 3, 0)];
const B_GI: &[ResourceDecl] = &[ResourceDecl::texture3d("gi_irradiance", 3, 0)];
const B_AUDIO: &[ResourceDecl] = &[ResourceDecl::buffer("audio_spectrum", 3, 0)];
const B_CURVE: &[ResourceDecl] = &[ResourceDecl::curve("signal_curve", 3, 0)];
const B_GLOBAL: &[ResourceDecl] = &[ResourceDecl::uniform("globals", 3, 0)];
const B_USER_PARAM: &[ResourceDecl] = &[ResourceDecl::uniform("user_params", 3, 0)];
const B_GRID: &[ResourceDecl] = &[ResourceDecl::buffer("interaction_grid", 3, 0)];
const B_NEIGHBOR: &[ResourceDecl] = &[ResourceDecl::buffer("neighbor_list", 3, 0)];

const F_MESH: &[&str] = &[
    "sample_mesh_position",
    "sample_mesh_normal",
    "sample_mesh_barycentric",
];
const F_SKELETAL: &[&str] = &["sample_skeletal_position", "sample_skeletal_velocity"];
const F_SPLINE: &[&str] = &["sample_spline_point", "sample_spline_tangent"];
const F_TEX2D: &[&str] = &["sample_texture_2d"];
const F_TEX3D: &[&str] = &["sample_texture_3d"];
const F_SDF: &[&str] = &["sample_sdf_distance", "sample_sdf_gradient"];
const F_VDB: &[&str] = &["sample_vdb_density", "sample_vdb_gradient"];
const F_POINT_CACHE: &[&str] = &["sample_point_cache"];
const F_CAMERA: &[&str] = &["sample_camera_position", "sample_camera_view_projection"];
const F_DEPTH: &[&str] = &["sample_scene_depth"];
const F_NORMAL: &[&str] = &["sample_scene_normal"];
const F_LIGHT: &[&str] = &["sample_clustered_light"];
const F_RAYTRACE: &[&str] = &["trace_ray_closest", "trace_ray_any"];
const F_GI: &[&str] = &["sample_gi_irradiance"];
const F_AUDIO: &[&str] = &["sample_audio_band"];
const F_CURVE: &[&str] = &["sample_curve"];
const F_GLOBAL: &[&str] = &[
    "global_time",
    "global_delta_time",
    "global_frame",
    "random_stream",
];
const F_USER_PARAM: &[&str] = &["user_param"];
const F_GRID: &[&str] = &["grid_lookup"];
const F_NEIGHBOR: &[&str] = &["neighbor_begin", "neighbor_next"];

/// The first-ship built-in `DataInterface` catalog (design §8.3).
///
/// Every variant declares its [`BindingClass`], the `GPU` resources it binds,
/// and the set of `sample_*` function names it exposes to modules. The enum is
/// fieldless: concrete resource handles are bound at runtime, while this
/// registry provides the static compile-time contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BuiltinDataInterface {
    /// Sample a static mesh surface (position / normal / barycentric).
    SampleMesh,
    /// Sample a skinned skeletal mesh (position / velocity).
    SampleSkeletalMesh,
    /// Sample a spline or curve (point / tangent).
    SampleSpline,
    /// Sample a 2D field texture.
    SampleTexture2d,
    /// Sample a 3D field texture.
    SampleTexture3d,
    /// Sample a `SDF` volume (distance / gradient normal).
    SampleSdf,
    /// Sample a `VDB` volume (density / gradient).
    SampleVdb,
    /// Sample a baked point-cache cloud.
    SamplePointCache,
    /// Query the camera (position / view-projection).
    SampleCamera,
    /// Sample the scene depth buffer.
    SampleDepth,
    /// Sample the scene normal buffer.
    SampleNormal,
    /// Query clustered scene lights.
    SampleLight,
    /// Trace rays against the scene acceleration structure (`TLAS`).
    SampleRayTraceScene,
    /// Sample the global-illumination irradiance probes (`GI`).
    SampleGiProbe,
    /// Sample an audio spectrum band.
    SampleAudioSpectrum,
    /// Sample a curve / gradient look-up table.
    SampleCurve,
    /// Read global state (time / delta-time / frame / random stream).
    SampleGlobal,
    /// Read an authored user parameter (Property Binder).
    SampleUserParam,
    /// Look up a custom acceleration grid.
    SampleGrid,
    /// Iterate a neighbour list.
    SampleNeighbor,
}

impl BuiltinDataInterface {
    /// Every built-in data interface in stable declaration order.
    pub const ALL: [BuiltinDataInterface; 20] = [
        BuiltinDataInterface::SampleMesh,
        BuiltinDataInterface::SampleSkeletalMesh,
        BuiltinDataInterface::SampleSpline,
        BuiltinDataInterface::SampleTexture2d,
        BuiltinDataInterface::SampleTexture3d,
        BuiltinDataInterface::SampleSdf,
        BuiltinDataInterface::SampleVdb,
        BuiltinDataInterface::SamplePointCache,
        BuiltinDataInterface::SampleCamera,
        BuiltinDataInterface::SampleDepth,
        BuiltinDataInterface::SampleNormal,
        BuiltinDataInterface::SampleLight,
        BuiltinDataInterface::SampleRayTraceScene,
        BuiltinDataInterface::SampleGiProbe,
        BuiltinDataInterface::SampleAudioSpectrum,
        BuiltinDataInterface::SampleCurve,
        BuiltinDataInterface::SampleGlobal,
        BuiltinDataInterface::SampleUserParam,
        BuiltinDataInterface::SampleGrid,
        BuiltinDataInterface::SampleNeighbor,
    ];

    /// The static metadata backing this interface.
    const fn meta(self) -> DataInterfaceMeta {
        match self {
            BuiltinDataInterface::SampleMesh => DataInterfaceMeta {
                class: BindingClass::Geometry,
                bindings: B_MESH,
                functions: F_MESH,
            },
            BuiltinDataInterface::SampleSkeletalMesh => DataInterfaceMeta {
                class: BindingClass::Geometry,
                bindings: B_SKELETAL,
                functions: F_SKELETAL,
            },
            BuiltinDataInterface::SampleSpline => DataInterfaceMeta {
                class: BindingClass::Geometry,
                bindings: B_SPLINE,
                functions: F_SPLINE,
            },
            BuiltinDataInterface::SampleTexture2d => DataInterfaceMeta {
                class: BindingClass::Field,
                bindings: B_TEX2D,
                functions: F_TEX2D,
            },
            BuiltinDataInterface::SampleTexture3d => DataInterfaceMeta {
                class: BindingClass::Field,
                bindings: B_TEX3D,
                functions: F_TEX3D,
            },
            BuiltinDataInterface::SampleSdf => DataInterfaceMeta {
                class: BindingClass::Field,
                bindings: B_SDF,
                functions: F_SDF,
            },
            BuiltinDataInterface::SampleVdb => DataInterfaceMeta {
                class: BindingClass::Field,
                bindings: B_VDB,
                functions: F_VDB,
            },
            BuiltinDataInterface::SamplePointCache => DataInterfaceMeta {
                class: BindingClass::Field,
                bindings: B_POINT_CACHE,
                functions: F_POINT_CACHE,
            },
            BuiltinDataInterface::SampleCamera => DataInterfaceMeta {
                class: BindingClass::Scene,
                bindings: B_CAMERA,
                functions: F_CAMERA,
            },
            BuiltinDataInterface::SampleDepth => DataInterfaceMeta {
                class: BindingClass::Scene,
                bindings: B_DEPTH,
                functions: F_DEPTH,
            },
            BuiltinDataInterface::SampleNormal => DataInterfaceMeta {
                class: BindingClass::Scene,
                bindings: B_NORMAL,
                functions: F_NORMAL,
            },
            BuiltinDataInterface::SampleLight => DataInterfaceMeta {
                class: BindingClass::Scene,
                bindings: B_LIGHT,
                functions: F_LIGHT,
            },
            BuiltinDataInterface::SampleRayTraceScene => DataInterfaceMeta {
                class: BindingClass::Scene,
                bindings: B_RAYTRACE,
                functions: F_RAYTRACE,
            },
            BuiltinDataInterface::SampleGiProbe => DataInterfaceMeta {
                class: BindingClass::Scene,
                bindings: B_GI,
                functions: F_GI,
            },
            BuiltinDataInterface::SampleAudioSpectrum => DataInterfaceMeta {
                class: BindingClass::Signal,
                bindings: B_AUDIO,
                functions: F_AUDIO,
            },
            BuiltinDataInterface::SampleCurve => DataInterfaceMeta {
                class: BindingClass::Signal,
                bindings: B_CURVE,
                functions: F_CURVE,
            },
            BuiltinDataInterface::SampleGlobal => DataInterfaceMeta {
                class: BindingClass::Signal,
                bindings: B_GLOBAL,
                functions: F_GLOBAL,
            },
            BuiltinDataInterface::SampleUserParam => DataInterfaceMeta {
                class: BindingClass::Interaction,
                bindings: B_USER_PARAM,
                functions: F_USER_PARAM,
            },
            BuiltinDataInterface::SampleGrid => DataInterfaceMeta {
                class: BindingClass::Interaction,
                bindings: B_GRID,
                functions: F_GRID,
            },
            BuiltinDataInterface::SampleNeighbor => DataInterfaceMeta {
                class: BindingClass::Interaction,
                bindings: B_NEIGHBOR,
                functions: F_NEIGHBOR,
            },
        }
    }

    /// The category this interface belongs to.
    #[must_use]
    pub const fn class(self) -> BindingClass {
        self.meta().class
    }

    /// The `GPU` resources this interface binds.
    #[must_use]
    pub const fn binding_decls(self) -> &'static [ResourceDecl] {
        self.meta().bindings
    }

    /// The `sample_*` function names this interface exposes to modules.
    #[must_use]
    pub const fn sample_functions(self) -> &'static [&'static str] {
        self.meta().functions
    }

    /// The stable ordinal used for serialisation and registry indexing.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        self as u32
    }

    /// Reconstructs an interface from its [`BuiltinDataInterface::ordinal`];
    /// returns `None` for an out-of-range value so the mapping round-trips.
    #[must_use]
    pub const fn from_ordinal(value: u32) -> Option<BuiltinDataInterface> {
        let index = value as usize;
        if index < Self::ALL.len() {
            Some(Self::ALL[index])
        } else {
            None
        }
    }
}

impl DataInterface for BuiltinDataInterface {
    fn bindings(&self) -> &[ResourceDecl] {
        self.meta().bindings
    }

    fn emit_wesl_functions(&self, ctx: &mut CodegenCtx) {
        let meta = self.meta();
        for &decl in meta.bindings {
            let _ = ctx.declare_binding(decl);
        }
        for &name in meta.functions {
            let mut decl = String::from("fn ");
            decl.push_str(name);
            decl.push_str("(id: u32) -> vec4<f32>;");
            ctx.push_fragment(decl);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn stage_kind_ordinals_round_trip() {
        for (index, kind) in StageKind::ALL.iter().enumerate() {
            let ordinal = kind.ordinal();
            assert_eq!(ordinal as usize, index);
            assert_eq!(StageKind::from_ordinal(ordinal), Some(*kind));
        }
        assert_eq!(StageKind::from_ordinal(StageKind::ALL.len() as u32), None);
    }

    #[test]
    fn stage_kind_default_domains() {
        assert_eq!(
            StageKind::Force.default_domain(),
            IterationDomain::PerParticle
        );
        assert_eq!(
            StageKind::Constraint.default_domain(),
            IterationDomain::PerConstraint
        );
        assert_eq!(StageKind::Event.default_domain(), IterationDomain::PerEvent);
    }

    #[test]
    fn module_stage_mapping_is_correct() {
        assert_eq!(BuiltinModule::SpawnRate.stage(), StageKind::Emit);
        assert_eq!(BuiltinModule::SetPosition.stage(), StageKind::Init);
        assert_eq!(BuiltinModule::Gravity.stage(), StageKind::Force);
        assert_eq!(BuiltinModule::CollisionPlane.stage(), StageKind::Collision);
        assert_eq!(
            BuiltinModule::DistanceConstraint.stage(),
            StageKind::Constraint
        );
        assert_eq!(BuiltinModule::AgeKill.stage(), StageKind::Lifecycle);
        assert_eq!(BuiltinModule::EmitOnDeath.stage(), StageKind::Event);
    }

    #[test]
    fn module_required_attributes_are_correct() {
        assert_eq!(
            BuiltinModule::Gravity.required_attributes(),
            &[AttributeSemantic::Velocity]
        );
        assert_eq!(
            BuiltinModule::SetPosition.required_attributes(),
            &[AttributeSemantic::Position]
        );
        assert!(BuiltinModule::AgeKill
            .required_attributes()
            .contains(&AttributeSemantic::Alive));
        assert!(BuiltinModule::VectorField
            .required_attributes()
            .contains(&AttributeSemantic::Position));
    }

    #[test]
    fn random_range_is_an_attribute_free_boundary() {
        assert!(BuiltinModule::RandomRange.required_attributes().is_empty());
        // It still binds its range uniform, so it is not resource-free.
        assert_eq!(BuiltinModule::RandomRange.resources().len(), 1);
    }

    #[test]
    fn module_registry_is_fully_covered_and_stable() {
        assert_eq!(BuiltinModule::ALL.len(), 41);
        let mut kernels: Vec<&str> = Vec::new();
        for (index, module) in BuiltinModule::ALL.iter().enumerate() {
            assert_eq!(module.ordinal() as usize, index);
            assert_eq!(BuiltinModule::from_ordinal(module.ordinal()), Some(*module));
            assert!(!module.kernel_name().is_empty());
            assert!(
                !kernels.contains(&module.kernel_name()),
                "duplicate kernel name"
            );
            kernels.push(module.kernel_name());
        }
        assert_eq!(
            BuiltinModule::from_ordinal(BuiltinModule::ALL.len() as u32),
            None
        );
    }

    #[test]
    fn module_metadata_is_deterministic() {
        for module in BuiltinModule::ALL {
            assert_eq!(module.stage(), module.stage());
            assert_eq!(module.required_attributes(), module.required_attributes());
            assert_eq!(module.kernel_name(), module.kernel_name());
        }
    }

    #[test]
    fn data_interface_classification_is_correct() {
        assert_eq!(
            BuiltinDataInterface::SampleMesh.class(),
            BindingClass::Geometry
        );
        assert_eq!(BuiltinDataInterface::SampleSdf.class(), BindingClass::Field);
        assert_eq!(
            BuiltinDataInterface::SampleCamera.class(),
            BindingClass::Scene
        );
        assert_eq!(
            BuiltinDataInterface::SampleGlobal.class(),
            BindingClass::Signal
        );
        assert_eq!(
            BuiltinDataInterface::SampleGrid.class(),
            BindingClass::Interaction
        );
    }

    #[test]
    fn data_interface_registry_round_trips() {
        assert_eq!(BuiltinDataInterface::ALL.len(), 20);
        for (index, interface) in BuiltinDataInterface::ALL.iter().enumerate() {
            assert_eq!(interface.ordinal() as usize, index);
            assert_eq!(
                BuiltinDataInterface::from_ordinal(interface.ordinal()),
                Some(*interface)
            );
            assert!(!interface.sample_functions().is_empty());
            assert!(!interface.binding_decls().is_empty());
        }
        assert_eq!(
            BuiltinDataInterface::from_ordinal(BuiltinDataInterface::ALL.len() as u32),
            None
        );
    }

    #[test]
    fn binding_class_round_trips() {
        for (index, class) in BindingClass::ALL.iter().enumerate() {
            assert_eq!(class.ordinal() as usize, index);
            assert_eq!(BindingClass::from_ordinal(class.ordinal()), Some(*class));
        }
        assert_eq!(
            BindingClass::from_ordinal(BindingClass::ALL.len() as u32),
            None
        );
    }

    #[test]
    fn codegen_ctx_accumulates_fragments_and_bindings() {
        let mut ctx = CodegenCtx::new();
        assert!(ctx.is_empty());
        BuiltinModule::Gravity.emit_wesl(&mut ctx);
        BuiltinModule::Drag.emit_wesl(&mut ctx);
        assert_eq!(ctx.fragment_count(), 2);
        assert_eq!(ctx.binding_count(), 2);
        assert_eq!(ctx.fragments()[0], "gravity(particle_index);");
        assert_eq!(ctx.bindings()[0].name, "gravity");
        let source = ctx.assembled_source();
        assert_eq!(source, "gravity(particle_index);\ndrag(particle_index);");
    }

    #[test]
    fn codegen_ctx_dedups_shared_bindings() {
        let mut ctx = CodegenCtx::new();
        // Both spawn distributions share the `spawn_shape` uniform.
        BuiltinModule::SpawnPoint.emit_wesl(&mut ctx);
        BuiltinModule::SpawnSphere.emit_wesl(&mut ctx);
        assert_eq!(ctx.fragment_count(), 2);
        assert_eq!(ctx.binding_count(), 1);
        assert_eq!(ctx.bindings()[0].name, "spawn_shape");
    }

    #[test]
    fn declare_binding_reports_novelty() {
        let mut ctx = CodegenCtx::new();
        let decl = ResourceDecl::uniform("gravity", 2, 0);
        assert!(ctx.declare_binding(decl));
        assert!(!ctx.declare_binding(decl));
        assert_eq!(ctx.binding_count(), 1);
    }

    #[test]
    fn resource_free_module_declares_no_bindings() {
        let mut ctx = CodegenCtx::new();
        BuiltinModule::SetPosition.emit_wesl(&mut ctx);
        assert_eq!(ctx.fragment_count(), 1);
        assert_eq!(ctx.binding_count(), 0);
        assert_eq!(ctx.fragments()[0], "set_position(particle_index);");
    }

    #[test]
    fn data_interface_emits_functions_and_bindings() {
        let mut ctx = CodegenCtx::new();
        BuiltinDataInterface::SampleGlobal.emit_wesl_functions(&mut ctx);
        assert_eq!(ctx.binding_count(), 1);
        assert_eq!(ctx.bindings()[0].name, "globals");
        assert_eq!(ctx.fragment_count(), 4);
        assert_eq!(ctx.fragments()[0], "fn global_time(id: u32) -> vec4<f32>;");
        assert!(ctx
            .assembled_source()
            .contains("fn random_stream(id: u32) -> vec4<f32>;"));
    }

    #[test]
    fn traits_dispatch_dynamically() {
        let module: &dyn EmberModule = &BuiltinModule::Gravity;
        assert_eq!(module.stage(), StageKind::Force);
        assert_eq!(module.required_attributes(), &[AttributeSemantic::Velocity]);
        assert_eq!(module.resources().len(), 1);

        let interface: &dyn DataInterface = &BuiltinDataInterface::SampleMesh;
        assert_eq!(interface.bindings().len(), 2);
    }

    #[test]
    fn resource_access_is_reported() {
        assert!(!ResourceDecl::uniform("u", 0, 0).is_writable());
        assert!(!ResourceDecl::buffer("b", 0, 0).is_writable());
        assert!(ResourceDecl::storage_rw("s", 0, 0).is_writable());
    }

    #[test]
    fn empty_codegen_ctx_is_a_no_op() {
        let ctx = CodegenCtx::new();
        assert!(ctx.is_empty());
        assert_eq!(ctx.fragment_count(), 0);
        assert_eq!(ctx.binding_count(), 0);
        assert_eq!(ctx.assembled_source(), "");
    }
}
