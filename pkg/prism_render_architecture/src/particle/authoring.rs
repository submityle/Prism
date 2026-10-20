//! `.ember` asset format and authoring workflow contracts (design §30).
//!
//! An authored effect ships as an `.ember` asset: the compiled stage graph
//! (design §7, see [`super::graph`]), the exposed parameter table, the renderer
//! configuration, and the [`super::EmberShadingModel`] settings, all under a
//! versioned, deterministically ordered serialization. This module owns the
//! *`CPU`-verifiable* half of that workflow: the asset model, the
//! template/inheritance override algebra (a `Niagara`-style *Parent Emitter*),
//! the exposed-parameter binding table, the point-cache / attribute-map driving
//! references, the per-module version + migration chain, and the hot-reload
//! state machine.
//!
//! The codec is deliberately dependency-free and lives beside the asset model:
//! shipped assets use a versioned little-endian binary representation, while
//! The format has a fixed binary schema and does not depend on an external
//! serialization library. Runtime binding to `gameplay`, materials, or the
//! timeline remains outside this asset container.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use super::EmberShadingModel;
use super::ShadingBasis;

/// Absolute tolerance for comparing authored `f32` parameters when diffing an
/// override against its base (design §30). Two scalars closer than this are
/// treated as "unchanged" so a serialization round-trip does not
/// manufacture spurious overrides.
pub const PARAM_EPS: f32 = 1e-6;

/// Returns `true` when two authored scalars are equal within [`PARAM_EPS`].
///
/// Used instead of a bare `f32` equality so the diff/patch contract never
/// relies on exact bit equality of re-serialized values.
#[must_use]
pub fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= PARAM_EPS
}

/// Identifies one `.ember` asset instance in the asset database (design §30).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EmberAssetHandle(pub u32);

/// References the compiled stage graph an `.ember` asset owns (design §7).
///
/// The graph itself is modelled by [`super::graph`]; the asset only stores a
/// stable reference so the two can be versioned and hot-reloaded independently.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GraphAssetRef(pub u32);

/// Identifies one renderer configuration block within an asset (design §16).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RendererConfigId(pub u32);

/// A stable handle to an external baked resource (a point cache or an attribute
/// texture) referenced by the asset (design §30).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceHandle(pub u32);

/// A stable handle to an exposed parameter within an asset (design §30).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ParamHandle(pub u32);

/// Identifies a reusable emitter template that other emitters inherit from
/// (a `Niagara`-style *Parent Emitter*, design §30).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EmitterTemplateHandle(pub u32);

/// The on-disk container format an `.ember` asset is serialized to (design §30).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SerializationFormat {
    /// A compact little-endian binary form, preferred for shipped builds where
    /// load time and size matter more than readability.
    Binary,
}

/// What an asset is being serialized *for*, used to pick a default
/// [`SerializationFormat`] (design §30).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AssetPurpose {
    /// Cooking a shipped build: prefer the compact binary form.
    Shipping,
}

/// Picks the recommended [`SerializationFormat`] for an [`AssetPurpose`].
///
/// Shipping uses [`SerializationFormat::Binary`] (compact).
#[must_use]
pub fn recommended_format(purpose: AssetPurpose) -> SerializationFormat {
    match purpose {
        AssetPurpose::Shipping => SerializationFormat::Binary,
    }
}

/// The canonical field order every `.ember` codec must emit, in order
/// (design §30).
///
/// The binary writer walks fields in exactly this order and prefixes each field
/// with a tag. The decoder checks every tag, so reordered fields cannot be
/// accepted as a different asset.
pub const EMBER_FIELD_ORDER: [&str; 6] = [
    "format_version",
    "graph",
    "params",
    "renderer_config",
    "shading",
    "format",
];

/// Eight-byte signature for the dependency-free binary `.ember` container.
pub const EMBER_BINARY_MAGIC: [u8; 8] = *b"PREMBER1";
/// Version of the binary container layout, independent of the asset schema.
pub const EMBER_BINARY_VERSION: u32 = 1;
/// Maximum number of exposed parameters accepted by the decoder.
pub const MAX_EMBER_PARAMS: u32 = 1_000_000;
/// Maximum UTF-8 byte length accepted for one exposed parameter name.
pub const MAX_EMBER_PARAM_NAME_BYTES: u32 = 1 << 20;

/// The kind of primitive a renderer configuration draws (design §16).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RendererKind {
    /// Camera-facing (or velocity-aligned) sprite quads.
    Sprite,
    /// Connected ribbons / trails threaded through particles.
    Ribbon,
    /// Instanced meshes, one per particle.
    Mesh,
    /// Light-emitting particles feeding the clustered light list.
    Light,
    /// Projected decals stamped onto surfaces.
    Decal,
    /// A raymarched volumetric renderer (design §21).
    Volume,
}

/// The renderer configuration block of an `.ember` asset (design §16).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RendererConfig {
    /// Stable identifier for this configuration within the asset.
    pub id: RendererConfigId,
    /// The primitive this renderer draws.
    pub kind: RendererKind,
    /// Whether the renderer writes motion vectors for temporal passes.
    pub writes_motion_vectors: bool,
}

/// The `.ember` shading settings block wrapping [`super::EmberShadingModel`]
/// (design §16).
///
/// Holds an `f32` (the hybrid blend weight lives inside the model), so it is
/// `PartialEq` but deliberately not `Eq`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadingSettings {
    /// The shading response selected for the effect.
    pub model: EmberShadingModel,
    /// Whether particles receive shadows from the scene.
    pub receive_shadows: bool,
    /// Whether particles cast shadows into the scene.
    pub cast_shadows: bool,
}

/// A typed value that can back an exposed parameter's constant default
/// (design §30).
///
/// Some variants carry `f32`, so the value is `PartialEq` but not `Eq`; compare
/// scalar payloads through [`approx_eq`] rather than a bare `f32` equality.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamValue {
    /// A single scalar.
    Scalar(f32),
    /// A three-component vector `[x, y, z]`.
    Vec3([f32; 3]),
    /// A linear `RGBA` colour `[r, g, b, a]`.
    Color([f32; 4]),
    /// A signed integer.
    Int(i32),
    /// A boolean flag.
    Bool(bool),
}

/// The declared type of an exposed parameter (design §30).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ParamType {
    /// A single scalar.
    Scalar,
    /// A three-component vector.
    Vec3,
    /// A linear `RGBA` colour.
    Color,
    /// A signed integer.
    Int,
    /// A boolean flag.
    Bool,
}

/// Where an exposed parameter's runtime value comes from (design §30).
///
/// A [`BindingSource::Constant`] is baked into the asset; every other variant
/// is driven externally at runtime and carries an opaque channel index the host
/// resolves against `gameplay` state, a material instance, or a timeline track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BindingSource {
    /// A fixed value stored in the asset (not externally driven).
    Constant(ParamValue),
    /// Driven by `gameplay` code through the identified channel.
    Gameplay(u32),
    /// Driven by a material-instance parameter through the identified channel.
    Material(u32),
    /// Driven by a sequencer / timeline track through the identified channel.
    Timeline(u32),
    /// Driven by a named authored curve evaluated over time.
    Curve(u32),
}

impl BindingSource {
    /// Returns `true` when the value is supplied at runtime rather than baked in.
    #[must_use]
    pub fn is_external(self) -> bool {
        !matches!(self, BindingSource::Constant(_))
    }

    /// Returns the external channel index for a runtime-driven source, or
    /// `None` for a [`BindingSource::Constant`].
    #[must_use]
    pub fn channel(self) -> Option<u32> {
        match self {
            BindingSource::Constant(_) => None,
            BindingSource::Gameplay(c)
            | BindingSource::Material(c)
            | BindingSource::Timeline(c)
            | BindingSource::Curve(c) => Some(c),
        }
    }
}

/// One author-exposed parameter: a value the author marks as externally
/// drivable so `gameplay`, materials, or the timeline can steer the effect
/// (design §30).
#[derive(Clone, Debug, PartialEq)]
pub struct ExposedParam {
    /// Stable handle used to look this parameter up at runtime.
    pub handle: ParamHandle,
    /// The author-facing display name.
    pub name: String,
    /// The declared value type.
    pub ty: ParamType,
    /// Where the runtime value comes from.
    pub binding: BindingSource,
}

impl ExposedParam {
    /// Returns `true` when this parameter is driven by an external source.
    #[must_use]
    pub fn is_externally_driven(&self) -> bool {
        self.binding.is_external()
    }
}

/// A logical channel of a baked point cache or attribute map that can drive a
/// particle's initial state (design §30).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PointCacheChannel {
    /// Initial position.
    Position,
    /// Initial velocity.
    Velocity,
    /// Initial colour.
    Color,
    /// Initial size / radius.
    Size,
    /// Initial normalized age.
    Age,
    /// A user-defined channel identified by an index.
    Custom(u32),
}

/// Maps one logical [`PointCacheChannel`] to a source column (point cache) or
/// texture channel (attribute map) index in the baked resource (design §30).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ChannelMapping {
    /// The logical channel being driven.
    pub channel: PointCacheChannel,
    /// The zero-based source index in the baked resource.
    pub source_index: u32,
}

/// A reference to a baked point cloud whose points seed the initial particle
/// state (design §30).
///
/// The baked point data lives in an external resource identified by
/// [`ResourceHandle`]; only the reference and its channel mapping are part of
/// the asset contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PointCacheBinding {
    /// The baked point-cache resource.
    pub resource: ResourceHandle,
    /// The number of points in the cache.
    pub point_count: u32,
    /// Per-channel source mappings.
    pub mappings: Vec<ChannelMapping>,
}

impl PointCacheBinding {
    /// Returns the source index feeding a given channel, or `None` when the
    /// channel is not mapped.
    #[must_use]
    pub fn source_for(&self, channel: PointCacheChannel) -> Option<u32> {
        self.mappings
            .iter()
            .find(|m| m.channel == channel)
            .map(|m| m.source_index)
    }
}

/// A reference to a baked attribute texture whose texels drive per-particle
/// initial state (design §30).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttributeMapBinding {
    /// The baked attribute-texture resource.
    pub texture: ResourceHandle,
    /// Texture width in texels.
    pub width: u32,
    /// Texture height in texels.
    pub height: u32,
    /// Per-channel source mappings (texture channel indices).
    pub mappings: Vec<ChannelMapping>,
}

impl AttributeMapBinding {
    /// The number of texels the attribute map exposes.
    #[must_use]
    pub fn texel_count(&self) -> u32 {
        self.width * self.height
    }

    /// Returns the texture channel index feeding a given channel, or `None`.
    #[must_use]
    pub fn source_for(&self, channel: PointCacheChannel) -> Option<u32> {
        self.mappings
            .iter()
            .find(|m| m.channel == channel)
            .map(|m| m.source_index)
    }
}

/// The fully resolved configuration of a single emitter after inheritance
/// (design §30).
///
/// Carries an `f32`, so it is `PartialEq` but not `Eq`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmitterConfig {
    /// Particles spawned per second.
    pub spawn_rate: f32,
    /// Base particle lifetime in seconds.
    pub lifetime: f32,
    /// Fixed pool capacity for the emitter.
    pub capacity: u32,
    /// The shading model the emitter draws with.
    pub shading: EmberShadingModel,
}

/// A field-level override layered on top of a parent [`EmitterConfig`]
/// (design §30).
///
/// Every field is an [`Option`]: `None` means *inherit the parent's value* and
/// `Some` means *override it*. Editing the parent template therefore
/// re-propagates through every un-overridden field of every child at once.
/// Carries an `f32` inside its options, so it is `PartialEq` but not `Eq`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EmitterOverride {
    /// Overridden spawn rate, or `None` to inherit.
    pub spawn_rate: Option<f32>,
    /// Overridden lifetime, or `None` to inherit.
    pub lifetime: Option<f32>,
    /// Overridden capacity, or `None` to inherit.
    pub capacity: Option<u32>,
    /// Overridden shading model, or `None` to inherit.
    pub shading: Option<EmberShadingModel>,
}

/// A reusable emitter template that child emitters inherit from (a `Niagara`
/// *Parent Emitter*, design §30).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmitterTemplate {
    /// Stable handle used by children to reference this template.
    pub handle: EmitterTemplateHandle,
    /// The fully specified base configuration children start from.
    pub base: EmitterConfig,
}

/// Applies a child [`EmitterOverride`] onto a parent [`EmitterConfig`],
/// producing the resolved configuration (design §30).
///
/// Un-overridden (`None`) fields inherit the parent value; overridden (`Some`)
/// fields replace it.
#[must_use]
pub fn apply_override(base: &EmitterConfig, ov: &EmitterOverride) -> EmitterConfig {
    EmitterConfig {
        spawn_rate: ov.spawn_rate.unwrap_or(base.spawn_rate),
        lifetime: ov.lifetime.unwrap_or(base.lifetime),
        capacity: ov.capacity.unwrap_or(base.capacity),
        shading: ov.shading.unwrap_or(base.shading),
    }
}

/// Composes a parent override with a child override, child taking precedence
/// (design §30).
///
/// For each field the child's `Some` wins; otherwise the parent's value (which
/// may itself be `None`, meaning "still inherit from the base") is kept. This is
/// how a `System`-level override layers on top of an `Emitter`-level one.
#[must_use]
pub fn compose_overrides(parent: &EmitterOverride, child: &EmitterOverride) -> EmitterOverride {
    EmitterOverride {
        spawn_rate: child.spawn_rate.or(parent.spawn_rate),
        lifetime: child.lifetime.or(parent.lifetime),
        capacity: child.capacity.or(parent.capacity),
        shading: child.shading.or(parent.shading),
    }
}

/// Computes the field-level override that turns `base` into `modified`
/// (design §30).
///
/// Only fields that actually differ are captured as `Some`; `f32` fields are
/// compared through [`approx_eq`] so a re-serialized value does not register as
/// a spurious override. This is the diff half of the diff/patch contract, the
/// inverse of [`apply_override`].
#[must_use]
pub fn diff_override(base: &EmitterConfig, modified: &EmitterConfig) -> EmitterOverride {
    EmitterOverride {
        spawn_rate: if approx_eq(base.spawn_rate, modified.spawn_rate) {
            None
        } else {
            Some(modified.spawn_rate)
        },
        lifetime: if approx_eq(base.lifetime, modified.lifetime) {
            None
        } else {
            Some(modified.lifetime)
        },
        capacity: if base.capacity == modified.capacity {
            None
        } else {
            Some(modified.capacity)
        },
        shading: if base.shading == modified.shading {
            None
        } else {
            Some(modified.shading)
        },
    }
}

/// An opaque, versioned payload for one authored module instance (design §30).
///
/// The `blob` is codec-agnostic bytes; migrations rewrite it as the module's
/// schema evolves. Pure integer/byte data, so this is `Eq`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleData {
    /// Identifies the module *type* whose schema this payload follows.
    pub type_id: u32,
    /// The schema version the payload was authored/serialized at.
    pub version: u32,
    /// The opaque serialized payload.
    pub blob: Vec<u8>,
}

/// The signature of a single-step module migration: it rewrites a payload from
/// one schema version to the next (design §30).
///
/// The function transforms the payload bytes only; the driving
/// [`MigrationChain`] is responsible for stamping the new [`ModuleData::version`]
/// so the step is guaranteed to make forward progress. No real deserialization
/// crate is involved at this contract layer.
pub type ModuleMigrationFn = fn(ModuleData) -> ModuleData;

/// One versioned migration step: rewrite `from_version` payloads into
/// `to_version` payloads (design §30).
///
/// Holds a function pointer, so it is `Copy` but intentionally not `PartialEq`.
#[derive(Clone, Copy, Debug)]
pub struct Migration {
    /// The schema version this step upgrades *from*.
    pub from_version: u32,
    /// The schema version this step upgrades *to* (must exceed `from_version`).
    pub to_version: u32,
    /// The payload rewrite applied by this step.
    pub apply: ModuleMigrationFn,
}

/// Why a migration failed (design §30).
///
/// Pure integer data, so this is `Eq`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MigrationError {
    /// The data version is newer than the running code can understand.
    VersionTooNew {
        /// The version found in the data.
        found: u32,
        /// The current (target) version the code supports.
        current: u32,
    },
    /// No migration step starts at this version, so the chain cannot advance.
    NoMigrationFrom(u32),
    /// A step declared `to_version <= from_version`, so it cannot make progress.
    NonMonotonicStep {
        /// The offending step's source version.
        from_version: u32,
        /// The offending step's (invalid) target version.
        to_version: u32,
    },
}

/// The ordered set of migration steps for one module type, upgrading old
/// payloads to `current_version` (design §30).
///
/// Applying the chain walks steps from the data's version up to
/// `current_version`, so shipping a new engine build never breaks assets saved
/// by an older one. Holds [`Migration`] (no `PartialEq`), so it derives only
/// `Clone`/`Debug`.
#[derive(Clone, Debug)]
pub struct MigrationChain {
    /// The module type this chain migrates.
    pub type_id: u32,
    /// The version all data is brought up to.
    pub current_version: u32,
    /// The available steps; each starts at a distinct `from_version`.
    pub steps: Vec<Migration>,
}

impl MigrationChain {
    /// Returns `true` when every step advances the version (`to_version` strictly
    /// greater than `from_version`), the precondition for termination.
    #[must_use]
    pub fn is_monotonic(&self) -> bool {
        self.steps.iter().all(|s| s.to_version > s.from_version)
    }

    /// Migrates `data` up to [`MigrationChain::current_version`] by applying each
    /// matching step in turn (design §30).
    ///
    /// Returns [`MigrationError::VersionTooNew`] when the data is ahead of the
    /// code, [`MigrationError::NoMigrationFrom`] when the chain has a gap, and
    /// [`MigrationError::NonMonotonicStep`] when a step would not advance the
    /// version. Data already at the current version is returned unchanged.
    pub fn migrate(&self, mut data: ModuleData) -> Result<ModuleData, MigrationError> {
        if data.version > self.current_version {
            return Err(MigrationError::VersionTooNew {
                found: data.version,
                current: self.current_version,
            });
        }
        while data.version < self.current_version {
            let Some(step) = self.steps.iter().find(|s| s.from_version == data.version) else {
                return Err(MigrationError::NoMigrationFrom(data.version));
            };
            if step.to_version <= step.from_version {
                return Err(MigrationError::NonMonotonicStep {
                    from_version: step.from_version,
                    to_version: step.to_version,
                });
            }
            let target = step.to_version;
            data = (step.apply)(data);
            data.version = target;
        }
        Ok(data)
    }
}

/// The lifecycle state of an in-flight hot reload (design §30).
///
/// Editing the graph or the shading recompiles a `GPU` kernel and swaps the
/// live pipeline through the `pipeline_cache`; this enum tracks that lifecycle.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum ReloadState {
    /// No reload is active; the live pipeline is current.
    #[default]
    Idle,
    /// A kernel recompile is in progress after an edit.
    Recompiling,
    /// A freshly compiled pipeline is being swapped into the `pipeline_cache`.
    Swapping,
    /// The last recompile failed; the previous pipeline is still live.
    Failed,
}

impl ReloadState {
    /// Returns `true` while a reload is actively working (recompiling or
    /// swapping), when new edits should coalesce rather than restart.
    #[must_use]
    pub fn is_busy(self) -> bool {
        matches!(self, ReloadState::Recompiling | ReloadState::Swapping)
    }
}

/// An event that can drive the [`ReloadState`] machine (design §30).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReloadEvent {
    /// The author edited the stage graph.
    GraphEdited,
    /// The author edited the shading model / settings.
    ShadingEdited,
    /// The `GPU` kernel recompile succeeded.
    CompileSucceeded,
    /// The `GPU` kernel recompile failed.
    CompileFailed,
    /// The `pipeline_cache` finished swapping in the new pipeline.
    SwapCompleted,
    /// The author asked to retry after a failure.
    Retry,
    /// The author cancelled / reset back to a clean state.
    Reset,
}

/// Computes the next [`ReloadState`] for a `(state, event)` pair (design §30).
///
/// Unhandled pairs leave the state unchanged, so spurious or duplicated events
/// are harmless. This is the pure transition function; [`HotReload::on_event`]
/// wraps it and bumps the pipeline generation on a successful swap.
#[must_use]
pub fn transition(state: ReloadState, event: ReloadEvent) -> ReloadState {
    match (state, event) {
        (_, ReloadEvent::Reset) | (ReloadState::Swapping, ReloadEvent::SwapCompleted) => {
            ReloadState::Idle
        }
        (ReloadState::Idle, ReloadEvent::GraphEdited | ReloadEvent::ShadingEdited)
        | (ReloadState::Failed, ReloadEvent::Retry) => ReloadState::Recompiling,
        (ReloadState::Recompiling, ReloadEvent::CompileSucceeded) => ReloadState::Swapping,
        (ReloadState::Recompiling, ReloadEvent::CompileFailed) => ReloadState::Failed,
        _ => state,
    }
}

/// The hot-reload driver: a [`ReloadState`] plus the live pipeline generation
/// counter (design §30).
///
/// The generation increments each time a swap completes, giving the
/// `pipeline_cache` and any cached bindings a monotonic tag to invalidate
/// against without comparing pipeline contents.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct HotReload {
    /// The current reload state.
    pub state: ReloadState,
    /// The live pipeline generation, bumped on each completed swap.
    pub generation: u32,
}

impl HotReload {
    /// Feeds an event through the machine, updating the state and bumping the
    /// generation when a swap completes; returns the resulting state.
    pub fn on_event(&mut self, event: ReloadEvent) -> ReloadState {
        let next = transition(self.state, event);
        if self.state == ReloadState::Swapping && next == ReloadState::Idle {
            self.generation = self.generation.wrapping_add(1);
        }
        self.state = next;
        next
    }
}

/// The top-level `.ember` asset (design §30).
///
/// Bundles the compiled graph reference, the exposed parameter table, the
/// renderer configuration, and the shading settings under a versioned,
/// deterministically ordered serialization (see [`EMBER_FIELD_ORDER`]). Carries
/// `f32` through its shading settings, so it is `PartialEq` but not `Eq`.
#[derive(Clone, Debug, PartialEq)]
pub struct EmberAsset {
    /// The asset schema version, used to gate whole-asset migrations.
    pub format_version: u32,
    /// Reference to the compiled stage graph.
    pub graph: GraphAssetRef,
    /// The author-exposed parameter table.
    pub params: Vec<ExposedParam>,
    /// The renderer configuration block.
    pub renderer_config: RendererConfig,
    /// The shading settings block.
    pub shading: ShadingSettings,
    /// The serialization format the asset is stored in.
    pub format: SerializationFormat,
}

impl EmberAsset {
    /// Looks up an exposed parameter by handle.
    #[must_use]
    pub fn param(&self, handle: ParamHandle) -> Option<&ExposedParam> {
        self.params.iter().find(|p| p.handle == handle)
    }

    /// Returns the number of externally driven (non-constant) exposed
    /// parameters.
    #[must_use]
    pub fn externally_driven_count(&self) -> usize {
        self.params
            .iter()
            .filter(|p| p.is_externally_driven())
            .count()
    }
}

/// Errors produced by the `.ember` codec.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmberCodecError {
    /// The container does not begin with [`EMBER_BINARY_MAGIC`].
    InvalidMagic,
    /// The binary container layout is not supported by this decoder.
    UnsupportedBinaryVersion(u32),
    /// A field tag is not the next tag in [`EMBER_FIELD_ORDER`].
    InvalidFieldOrder { expected: u8, found: u8 },
    /// The input ended before a complete field was read.
    Truncated,
    /// Bytes remained after the final field.
    TrailingBytes,
    /// A bounded count exceeded the decoder's allocation limit.
    CountTooLarge { field: &'static str, value: u32 },
    /// A bounded byte length exceeded the decoder's allocation limit.
    LengthTooLarge { field: &'static str, value: u32 },
    /// An enum discriminant is not part of the wire contract.
    InvalidEnum { field: &'static str, value: u8 },
    /// A boolean was not encoded as zero or one.
    InvalidBoolean { field: &'static str, value: u8 },
    /// A parameter name was not valid UTF-8.
    InvalidUtf8,
    /// A serialized float was NaN or infinite.
    NonFiniteFloat { field: &'static str },
    /// A hybrid shading weight was outside its documented interval.
    InvalidFloatRange { field: &'static str },
    /// A parameter handle occurred more than once.
    DuplicateParamHandle(u32),
    /// A constant parameter value did not match its declared parameter type.
    ParamValueTypeMismatch { handle: u32 },
}

impl fmt::Display for EmberCodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => formatter.write_str("invalid ember binary magic"),
            Self::UnsupportedBinaryVersion(version) => {
                write!(formatter, "unsupported ember binary version {version}")
            }
            Self::InvalidFieldOrder { expected, found } => write!(
                formatter,
                "invalid ember field order: expected tag {expected}, found {found}"
            ),
            Self::Truncated => formatter.write_str("truncated ember asset"),
            Self::TrailingBytes => formatter.write_str("trailing bytes after ember asset"),
            Self::CountTooLarge { field, value } => {
                write!(formatter, "ember {field} count {value} is too large")
            }
            Self::LengthTooLarge { field, value } => {
                write!(formatter, "ember {field} length {value} is too large")
            }
            Self::InvalidEnum { field, value } => {
                write!(formatter, "invalid ember {field} enum value {value}")
            }
            Self::InvalidBoolean { field, value } => {
                write!(formatter, "invalid ember {field} boolean value {value}")
            }
            Self::InvalidUtf8 => formatter.write_str("invalid UTF-8 in ember asset"),
            Self::NonFiniteFloat { field } => {
                write!(formatter, "non-finite ember float in {field}")
            }
            Self::InvalidFloatRange { field } => {
                write!(
                    formatter,
                    "ember float in {field} is outside its valid range"
                )
            }
            Self::DuplicateParamHandle(handle) => {
                write!(formatter, "duplicate ember parameter handle {handle}")
            }
            Self::ParamValueTypeMismatch { handle } => {
                write!(formatter, "ember parameter {handle} constant type mismatch")
            }
        }
    }
}

/// Serializes an asset using the requested on-disk format.
pub fn serialize_asset(
    asset: &EmberAsset,
    format: SerializationFormat,
) -> Result<Vec<u8>, EmberCodecError> {
    match format {
        SerializationFormat::Binary => encode_binary(asset),
    }
}

/// Deserializes an asset using the requested on-disk format.
pub fn deserialize_asset(
    bytes: &[u8],
    format: SerializationFormat,
) -> Result<EmberAsset, EmberCodecError> {
    match format {
        SerializationFormat::Binary => decode_binary(bytes),
    }
}

/// Encodes an `.ember` asset as the deterministic little-endian binary format.
pub fn encode_binary(asset: &EmberAsset) -> Result<Vec<u8>, EmberCodecError> {
    validate_asset(asset)?;

    let mut writer = BinaryWriter::default();
    writer.bytes(&EMBER_BINARY_MAGIC);
    writer.u32(EMBER_BINARY_VERSION);

    writer.field(1);
    writer.u32(asset.format_version);

    writer.field(2);
    writer.u32(asset.graph.0);

    writer.field(3);
    writer.count(asset.params.len(), "params")?;
    for param in &asset.params {
        writer.u32(param.handle.0);
        writer.string(&param.name, "parameter name")?;
        write_param_type(&mut writer, param.ty);
        write_binding(&mut writer, param.binding)?;
    }

    writer.field(4);
    writer.u32(asset.renderer_config.id.0);
    write_renderer_kind(&mut writer, asset.renderer_config.kind);
    writer.bool(asset.renderer_config.writes_motion_vectors);

    writer.field(5);
    write_shading_model(&mut writer, asset.shading.model)?;
    writer.bool(asset.shading.receive_shadows);
    writer.bool(asset.shading.cast_shadows);

    writer.field(6);
    write_serialization_format(&mut writer, asset.format);

    Ok(writer.into_bytes())
}

/// Decodes and validates a deterministic little-endian binary `.ember` asset.
pub fn decode_binary(bytes: &[u8]) -> Result<EmberAsset, EmberCodecError> {
    let mut reader = BinaryReader::new(bytes);
    if reader.take(EMBER_BINARY_MAGIC.len())? != EMBER_BINARY_MAGIC {
        return Err(EmberCodecError::InvalidMagic);
    }
    let version = reader.u32()?;
    if version != EMBER_BINARY_VERSION {
        return Err(EmberCodecError::UnsupportedBinaryVersion(version));
    }

    reader.field(1)?;
    let format_version = reader.u32()?;

    reader.field(2)?;
    let graph = GraphAssetRef(reader.u32()?);

    reader.field(3)?;
    let param_count = reader.count(MAX_EMBER_PARAMS, "params")?;
    let mut params = Vec::new();
    let mut handles = BTreeSet::new();
    for _ in 0..param_count {
        let handle = ParamHandle(reader.u32()?);
        if !handles.insert(handle.0) {
            return Err(EmberCodecError::DuplicateParamHandle(handle.0));
        }
        let name = reader.string(MAX_EMBER_PARAM_NAME_BYTES, "parameter name")?;
        let ty = read_param_type(&mut reader)?;
        let binding = read_binding(&mut reader)?;
        if let BindingSource::Constant(value) = binding {
            validate_param_value(ty, value, handle.0)?;
        }
        params.push(ExposedParam {
            handle,
            name,
            ty,
            binding,
        });
    }

    reader.field(4)?;
    let renderer_config = RendererConfig {
        id: RendererConfigId(reader.u32()?),
        kind: read_renderer_kind(&mut reader)?,
        writes_motion_vectors: reader.bool("writes_motion_vectors")?,
    };

    reader.field(5)?;
    let shading = ShadingSettings {
        model: read_shading_model(&mut reader)?,
        receive_shadows: reader.bool("receive_shadows")?,
        cast_shadows: reader.bool("cast_shadows")?,
    };

    reader.field(6)?;
    let format = read_serialization_format(&mut reader)?;
    let _ = format;
    let asset = EmberAsset {
        format_version,
        graph,
        params,
        renderer_config,
        shading,
        format,
    };
    if !reader.is_empty() {
        return Err(EmberCodecError::TrailingBytes);
    }
    validate_asset(&asset)?;
    Ok(asset)
}

fn validate_asset(asset: &EmberAsset) -> Result<(), EmberCodecError> {
    let mut handles = BTreeSet::new();
    for param in &asset.params {
        if !handles.insert(param.handle.0) {
            return Err(EmberCodecError::DuplicateParamHandle(param.handle.0));
        }
        if let BindingSource::Constant(value) = param.binding {
            validate_param_value(param.ty, value, param.handle.0)?;
        }
    }
    validate_shading_model(asset.shading.model)
}

fn validate_param_value(
    ty: ParamType,
    value: ParamValue,
    handle: u32,
) -> Result<(), EmberCodecError> {
    let matches = match (ty, value) {
        (ParamType::Scalar, ParamValue::Scalar(value)) => {
            validate_f32(value, "parameter scalar")?;
            true
        }
        (ParamType::Vec3, ParamValue::Vec3(values)) => {
            validate_f32s(&values, "parameter vec3")?;
            true
        }
        (ParamType::Color, ParamValue::Color(values)) => {
            validate_f32s(&values, "parameter color")?;
            true
        }
        (ParamType::Int, ParamValue::Int(_)) | (ParamType::Bool, ParamValue::Bool(_)) => true,
        _ => false,
    };
    if matches {
        Ok(())
    } else {
        Err(EmberCodecError::ParamValueTypeMismatch { handle })
    }
}

fn validate_shading_model(model: EmberShadingModel) -> Result<(), EmberCodecError> {
    if let EmberShadingModel::Hybrid { weight, .. } = model {
        validate_f32(weight, "hybrid weight")?;
        if !(0.0..=1.0).contains(&weight) {
            return Err(EmberCodecError::InvalidFloatRange {
                field: "hybrid weight",
            });
        }
    }
    Ok(())
}

fn validate_f32(value: f32, field: &'static str) -> Result<(), EmberCodecError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EmberCodecError::NonFiniteFloat { field })
    }
}

fn validate_f32s(values: &[f32], field: &'static str) -> Result<(), EmberCodecError> {
    for value in values {
        validate_f32(*value, field)?;
    }
    Ok(())
}

fn write_param_type(writer: &mut BinaryWriter, value: ParamType) {
    writer.u8(match value {
        ParamType::Scalar => 0,
        ParamType::Vec3 => 1,
        ParamType::Color => 2,
        ParamType::Int => 3,
        ParamType::Bool => 4,
    });
}

fn read_param_type(reader: &mut BinaryReader<'_>) -> Result<ParamType, EmberCodecError> {
    match reader.u8()? {
        0 => Ok(ParamType::Scalar),
        1 => Ok(ParamType::Vec3),
        2 => Ok(ParamType::Color),
        3 => Ok(ParamType::Int),
        4 => Ok(ParamType::Bool),
        value => Err(EmberCodecError::InvalidEnum {
            field: "parameter type",
            value,
        }),
    }
}

fn write_binding(writer: &mut BinaryWriter, binding: BindingSource) -> Result<(), EmberCodecError> {
    match binding {
        BindingSource::Constant(value) => {
            writer.u8(0);
            write_param_value(writer, value)?;
        }
        BindingSource::Gameplay(channel) => {
            writer.u8(1);
            writer.u32(channel);
        }
        BindingSource::Material(channel) => {
            writer.u8(2);
            writer.u32(channel);
        }
        BindingSource::Timeline(channel) => {
            writer.u8(3);
            writer.u32(channel);
        }
        BindingSource::Curve(channel) => {
            writer.u8(4);
            writer.u32(channel);
        }
    }
    Ok(())
}

fn read_binding(reader: &mut BinaryReader<'_>) -> Result<BindingSource, EmberCodecError> {
    match reader.u8()? {
        0 => Ok(BindingSource::Constant(read_param_value(reader)?)),
        1 => Ok(BindingSource::Gameplay(reader.u32()?)),
        2 => Ok(BindingSource::Material(reader.u32()?)),
        3 => Ok(BindingSource::Timeline(reader.u32()?)),
        4 => Ok(BindingSource::Curve(reader.u32()?)),
        value => Err(EmberCodecError::InvalidEnum {
            field: "binding source",
            value,
        }),
    }
}

fn write_param_value(writer: &mut BinaryWriter, value: ParamValue) -> Result<(), EmberCodecError> {
    match value {
        ParamValue::Scalar(value) => {
            writer.u8(0);
            writer.f32(value, "parameter scalar")?;
        }
        ParamValue::Vec3(values) => {
            writer.u8(1);
            for value in values {
                writer.f32(value, "parameter vec3")?;
            }
        }
        ParamValue::Color(values) => {
            writer.u8(2);
            for value in values {
                writer.f32(value, "parameter color")?;
            }
        }
        ParamValue::Int(value) => {
            writer.u8(3);
            writer.i32(value);
        }
        ParamValue::Bool(value) => {
            writer.u8(4);
            writer.bool(value);
        }
    }
    Ok(())
}

fn read_param_value(reader: &mut BinaryReader<'_>) -> Result<ParamValue, EmberCodecError> {
    match reader.u8()? {
        0 => Ok(ParamValue::Scalar(reader.f32("parameter scalar")?)),
        1 => Ok(ParamValue::Vec3([
            reader.f32("parameter vec3")?,
            reader.f32("parameter vec3")?,
            reader.f32("parameter vec3")?,
        ])),
        2 => Ok(ParamValue::Color([
            reader.f32("parameter color")?,
            reader.f32("parameter color")?,
            reader.f32("parameter color")?,
            reader.f32("parameter color")?,
        ])),
        3 => Ok(ParamValue::Int(reader.i32()?)),
        4 => Ok(ParamValue::Bool(reader.bool("parameter bool")?)),
        value => Err(EmberCodecError::InvalidEnum {
            field: "parameter value",
            value,
        }),
    }
}

fn write_renderer_kind(writer: &mut BinaryWriter, value: RendererKind) {
    writer.u8(match value {
        RendererKind::Sprite => 0,
        RendererKind::Ribbon => 1,
        RendererKind::Mesh => 2,
        RendererKind::Light => 3,
        RendererKind::Decal => 4,
        RendererKind::Volume => 5,
    });
}

fn read_renderer_kind(reader: &mut BinaryReader<'_>) -> Result<RendererKind, EmberCodecError> {
    match reader.u8()? {
        0 => Ok(RendererKind::Sprite),
        1 => Ok(RendererKind::Ribbon),
        2 => Ok(RendererKind::Mesh),
        3 => Ok(RendererKind::Light),
        4 => Ok(RendererKind::Decal),
        5 => Ok(RendererKind::Volume),
        value => Err(EmberCodecError::InvalidEnum {
            field: "renderer kind",
            value,
        }),
    }
}

fn write_shading_model(
    writer: &mut BinaryWriter,
    value: EmberShadingModel,
) -> Result<(), EmberCodecError> {
    validate_shading_model(value)?;
    match value {
        EmberShadingModel::Unlit => writer.u8(0),
        EmberShadingModel::Pbr => writer.u8(1),
        EmberShadingModel::Npr => writer.u8(2),
        EmberShadingModel::Custom(id) => {
            writer.u8(3);
            writer.u32(id);
        }
        EmberShadingModel::Hybrid {
            base,
            overlay,
            weight,
        } => {
            writer.u8(4);
            write_shading_basis(writer, base);
            write_shading_basis(writer, overlay);
            writer.f32(weight, "hybrid weight")?;
        }
    }
    Ok(())
}

fn read_shading_model(reader: &mut BinaryReader<'_>) -> Result<EmberShadingModel, EmberCodecError> {
    let model = match reader.u8()? {
        0 => EmberShadingModel::Unlit,
        1 => EmberShadingModel::Pbr,
        2 => EmberShadingModel::Npr,
        3 => EmberShadingModel::Custom(reader.u32()?),
        4 => EmberShadingModel::Hybrid {
            base: read_shading_basis(reader)?,
            overlay: read_shading_basis(reader)?,
            weight: reader.f32("hybrid weight")?,
        },
        value => {
            return Err(EmberCodecError::InvalidEnum {
                field: "shading model",
                value,
            });
        }
    };
    validate_shading_model(model)?;
    Ok(model)
}

fn write_shading_basis(writer: &mut BinaryWriter, value: ShadingBasis) {
    match value {
        ShadingBasis::Unlit => writer.u8(0),
        ShadingBasis::Pbr => writer.u8(1),
        ShadingBasis::Npr => writer.u8(2),
        ShadingBasis::Custom(id) => {
            writer.u8(3);
            writer.u32(id);
        }
    }
}

fn read_shading_basis(reader: &mut BinaryReader<'_>) -> Result<ShadingBasis, EmberCodecError> {
    match reader.u8()? {
        0 => Ok(ShadingBasis::Unlit),
        1 => Ok(ShadingBasis::Pbr),
        2 => Ok(ShadingBasis::Npr),
        3 => Ok(ShadingBasis::Custom(reader.u32()?)),
        value => Err(EmberCodecError::InvalidEnum {
            field: "shading basis",
            value,
        }),
    }
}

fn write_serialization_format(writer: &mut BinaryWriter, value: SerializationFormat) {
    writer.u8(match value {
        SerializationFormat::Binary => 0,
    });
}

fn read_serialization_format(
    reader: &mut BinaryReader<'_>,
) -> Result<SerializationFormat, EmberCodecError> {
    match reader.u8()? {
        0 => Ok(SerializationFormat::Binary),
        value => Err(EmberCodecError::InvalidEnum {
            field: "serialization format",
            value,
        }),
    }
}

#[derive(Default)]
struct BinaryWriter {
    bytes: Vec<u8>,
}

impl BinaryWriter {
    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn i32(&mut self, value: i32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, value: f32, field: &'static str) -> Result<(), EmberCodecError> {
        validate_f32(value, field)?;
        self.u32(value.to_bits());
        Ok(())
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn bytes(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn field(&mut self, tag: u8) {
        self.u8(tag);
    }

    fn count(&mut self, value: usize, field: &'static str) -> Result<(), EmberCodecError> {
        let value = u32::try_from(value).map_err(|_| EmberCodecError::CountTooLarge {
            field,
            value: u32::MAX,
        })?;
        if value > MAX_EMBER_PARAMS {
            return Err(EmberCodecError::CountTooLarge { field, value });
        }
        self.u32(value);
        Ok(())
    }

    fn string(&mut self, value: &str, field: &'static str) -> Result<(), EmberCodecError> {
        let length = u32::try_from(value.len()).map_err(|_| EmberCodecError::LengthTooLarge {
            field,
            value: u32::MAX,
        })?;
        if length > MAX_EMBER_PARAM_NAME_BYTES {
            return Err(EmberCodecError::LengthTooLarge {
                field,
                value: length,
            });
        }
        self.u32(length);
        self.bytes(value.as_bytes());
        Ok(())
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

struct BinaryReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], EmberCodecError> {
        let end = self
            .cursor
            .checked_add(length)
            .ok_or(EmberCodecError::Truncated)?;
        let value = self
            .bytes
            .get(self.cursor..end)
            .ok_or(EmberCodecError::Truncated)?;
        self.cursor = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, EmberCodecError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, EmberCodecError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| EmberCodecError::Truncated)?,
        ))
    }

    fn i32(&mut self) -> Result<i32, EmberCodecError> {
        Ok(i32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| EmberCodecError::Truncated)?,
        ))
    }

    fn f32(&mut self, field: &'static str) -> Result<f32, EmberCodecError> {
        let value = f32::from_bits(self.u32()?);
        validate_f32(value, field)?;
        Ok(value)
    }

    fn bool(&mut self, field: &'static str) -> Result<bool, EmberCodecError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(EmberCodecError::InvalidBoolean { field, value }),
        }
    }

    fn string(&mut self, max_length: u32, field: &'static str) -> Result<String, EmberCodecError> {
        let length = self.u32()?;
        if length > max_length {
            return Err(EmberCodecError::LengthTooLarge {
                field,
                value: length,
            });
        }
        String::from_utf8(self.take(length as usize)?.to_vec())
            .map_err(|_| EmberCodecError::InvalidUtf8)
    }

    fn count(&mut self, max_count: u32, field: &'static str) -> Result<u32, EmberCodecError> {
        let count = self.u32()?;
        if count > max_count {
            return Err(EmberCodecError::CountTooLarge {
                field,
                value: count,
            });
        }
        Ok(count)
    }

    fn field(&mut self, expected: u8) -> Result<(), EmberCodecError> {
        let found = self.u8()?;
        if found == expected {
            Ok(())
        } else {
            Err(EmberCodecError::InvalidFieldOrder { expected, found })
        }
    }

    fn is_empty(&self) -> bool {
        self.cursor == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn base_config() -> EmitterConfig {
        EmitterConfig {
            spawn_rate: 100.0,
            lifetime: 2.0,
            capacity: 4096,
            shading: EmberShadingModel::Unlit,
        }
    }

    #[test]
    fn serialization_format_selection_by_purpose() {
        assert_eq!(
            recommended_format(AssetPurpose::Shipping),
            SerializationFormat::Binary
        );
    }

    #[test]
    fn stable_field_order_is_fixed() {
        assert_eq!(EMBER_FIELD_ORDER.len(), 6);
        assert_eq!(EMBER_FIELD_ORDER[0], "format_version");
        assert_eq!(EMBER_FIELD_ORDER[5], "format");
    }

    fn codec_asset() -> EmberAsset {
        EmberAsset {
            format_version: 7,
            graph: GraphAssetRef(0x1234_5678),
            params: vec![
                ExposedParam {
                    handle: ParamHandle(10),
                    name: String::from("intensity"),
                    ty: ParamType::Scalar,
                    binding: BindingSource::Constant(ParamValue::Scalar(0.75)),
                },
                ExposedParam {
                    handle: ParamHandle(11),
                    name: String::from("tint/火"),
                    ty: ParamType::Color,
                    binding: BindingSource::Material(3),
                },
                ExposedParam {
                    handle: ParamHandle(12),
                    name: String::from("enabled"),
                    ty: ParamType::Bool,
                    binding: BindingSource::Timeline(9),
                },
            ],
            renderer_config: RendererConfig {
                id: RendererConfigId(4),
                kind: RendererKind::Ribbon,
                writes_motion_vectors: true,
            },
            shading: ShadingSettings {
                model: EmberShadingModel::Hybrid {
                    base: ShadingBasis::Pbr,
                    overlay: ShadingBasis::Custom(0xAABB_CCDD),
                    weight: 0.375,
                },
                receive_shadows: true,
                cast_shadows: false,
            },
            format: SerializationFormat::Binary,
        }
    }

    #[test]
    fn binary_codec_round_trips_all_asset_fields() {
        let asset = codec_asset();
        let encoded = encode_binary(&asset).expect("valid asset encodes");
        let decoded = decode_binary(&encoded).expect("encoded asset decodes");
        assert_eq!(decoded, asset);
        assert_eq!(
            deserialize_asset(&encoded, SerializationFormat::Binary).unwrap(),
            asset
        );
    }

    #[test]
    fn binary_codec_has_stable_canonical_bytes() {
        let asset = codec_asset();
        let first = encode_binary(&asset).unwrap();
        let second = encode_binary(&asset).unwrap();
        assert_eq!(first, second);
        assert_eq!(&first[..8], &EMBER_BINARY_MAGIC);
        assert_eq!(&first[8..12], &EMBER_BINARY_VERSION.to_le_bytes());
        assert_eq!(&first[12..14], &[1, 7]);
        assert_eq!(&first[17..22], &[2, 0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn binary_codec_rejects_wrong_magic_version_field_order_and_trailing_data() {
        let encoded = encode_binary(&codec_asset()).unwrap();
        let mut bad_magic = encoded.clone();
        bad_magic[0] ^= 1;
        assert_eq!(
            decode_binary(&bad_magic),
            Err(EmberCodecError::InvalidMagic)
        );

        let mut bad_version = encoded.clone();
        bad_version[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            decode_binary(&bad_version),
            Err(EmberCodecError::UnsupportedBinaryVersion(2))
        );

        let mut bad_order = encoded.clone();
        bad_order[12] = 2;
        assert_eq!(
            decode_binary(&bad_order),
            Err(EmberCodecError::InvalidFieldOrder {
                expected: 1,
                found: 2,
            })
        );

        let mut trailing = encoded;
        trailing.push(0);
        assert_eq!(
            decode_binary(&trailing),
            Err(EmberCodecError::TrailingBytes)
        );
    }

    #[test]
    fn binary_codec_rejects_every_truncation() {
        let encoded = encode_binary(&codec_asset()).unwrap();
        for end in 0..encoded.len() {
            assert!(
                decode_binary(&encoded[..end]).is_err(),
                "accepted truncated prefix of length {end}"
            );
        }
    }

    #[test]
    fn binary_codec_rejects_invalid_enum_boolean_and_non_finite_float() {
        let mut invalid_type = encode_binary(&codec_asset()).unwrap();
        // Header (12), fields 1+4 and 2+4, param count (field 3+4), then
        // handle (4) and name length+UTF-8 bytes (4+9).
        invalid_type[12 + 5 + 5 + 5 + 4 + 4 + 9] = 99;
        assert!(matches!(
            decode_binary(&invalid_type),
            Err(EmberCodecError::InvalidEnum {
                field: "parameter type",
                value: 99
            })
        ));

        let mut marker = encode_binary(&codec_asset()).unwrap();
        let bool_index = marker.len() - 3;
        marker[bool_index] = 2;
        assert!(matches!(
            decode_binary(&marker),
            Err(EmberCodecError::InvalidBoolean { .. })
        ));

        let mut nan_asset = codec_asset();
        nan_asset.params[0].binding = BindingSource::Constant(ParamValue::Scalar(f32::NAN));
        assert_eq!(
            encode_binary(&nan_asset),
            Err(EmberCodecError::NonFiniteFloat {
                field: "parameter scalar"
            })
        );

        let mut encoded_nan = encode_binary(&codec_asset()).unwrap();
        let scalar_value = 12 + 5 + 5 + 5 + 4 + 4 + "intensity".len() + 1 + 1 + 1;
        encoded_nan[scalar_value..scalar_value + 4]
            .copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        assert_eq!(
            decode_binary(&encoded_nan),
            Err(EmberCodecError::NonFiniteFloat {
                field: "parameter scalar"
            })
        );
    }

    #[test]
    fn binary_codec_rejects_invalid_utf8_duplicate_handles_and_mismatched_values() {
        let mut invalid_utf8 = encode_binary(&codec_asset()).unwrap();
        // First name begins after the fixed header, first three field tags and
        // values, parameter count, handle, and name-length word.
        let first_name = 12 + 5 + 5 + 5 + 4 + 4 + 4;
        invalid_utf8[first_name] = 0xFF;
        assert_eq!(
            decode_binary(&invalid_utf8),
            Err(EmberCodecError::InvalidUtf8)
        );

        let mut duplicate = codec_asset();
        duplicate.params[1].handle = duplicate.params[0].handle;
        assert_eq!(
            encode_binary(&duplicate),
            Err(EmberCodecError::DuplicateParamHandle(10))
        );

        let mut mismatch = codec_asset();
        mismatch.params[0].ty = ParamType::Vec3;
        assert_eq!(
            encode_binary(&mismatch),
            Err(EmberCodecError::ParamValueTypeMismatch { handle: 10 })
        );
    }

    #[test]
    fn binary_codec_rejects_invalid_hybrid_weight_and_preserves_format_contract() {
        let mut invalid_weight = codec_asset();
        invalid_weight.shading.model = EmberShadingModel::Hybrid {
            base: ShadingBasis::Unlit,
            overlay: ShadingBasis::Pbr,
            weight: 1.01,
        };
        assert_eq!(
            encode_binary(&invalid_weight),
            Err(EmberCodecError::InvalidFloatRange {
                field: "hybrid weight"
            })
        );

        let encoded = serialize_asset(&codec_asset(), SerializationFormat::Binary).unwrap();
        assert_eq!(
            deserialize_asset(&encoded, SerializationFormat::Binary).unwrap(),
            codec_asset()
        );
    }

    #[test]
    fn override_inherits_none_fields_and_replaces_some() {
        let base = base_config();
        let ov = EmitterOverride {
            spawn_rate: Some(250.0),
            shading: Some(EmberShadingModel::Pbr),
            ..EmitterOverride::default()
        };
        let resolved = apply_override(&base, &ov);
        assert_eq!(resolved.spawn_rate, 250.0);
        assert_eq!(resolved.shading, EmberShadingModel::Pbr);
        // Un-overridden fields inherit the base.
        assert_eq!(resolved.lifetime, 2.0);
        assert_eq!(resolved.capacity, 4096);
    }

    #[test]
    fn editing_template_repropagates_to_uninherited_fields() {
        let ov = EmitterOverride {
            spawn_rate: Some(10.0),
            ..EmitterOverride::default()
        };
        let first = apply_override(&base_config(), &ov);
        assert_eq!(first.lifetime, 2.0);

        // The author edits the parent template's lifetime; the child never
        // overrode lifetime, so the change flows through.
        let mut edited = base_config();
        edited.lifetime = 5.0;
        let second = apply_override(&edited, &ov);
        assert_eq!(second.lifetime, 5.0);
        // The child's own override still wins.
        assert_eq!(second.spawn_rate, 10.0);
    }

    #[test]
    fn compose_overrides_child_wins() {
        let parent = EmitterOverride {
            spawn_rate: Some(10.0),
            lifetime: Some(3.0),
            ..EmitterOverride::default()
        };
        let child = EmitterOverride {
            spawn_rate: Some(20.0),
            capacity: Some(512),
            ..EmitterOverride::default()
        };
        let merged = compose_overrides(&parent, &child);
        assert_eq!(merged.spawn_rate, Some(20.0));
        assert_eq!(merged.lifetime, Some(3.0));
        assert_eq!(merged.capacity, Some(512));
        assert_eq!(merged.shading, None);
    }

    #[test]
    fn diff_then_apply_round_trips() {
        let base = base_config();
        let mut modified = base;
        modified.capacity = 8192;
        modified.lifetime = 4.0;

        let ov = diff_override(&base, &modified);
        assert_eq!(ov.spawn_rate, None);
        assert_eq!(ov.capacity, Some(8192));
        assert_eq!(ov.lifetime, Some(4.0));

        let restored = apply_override(&base, &ov);
        assert_eq!(restored, modified);
    }

    #[test]
    fn diff_ignores_sub_epsilon_scalar_drift() {
        let base = base_config();
        let mut modified = base;
        modified.spawn_rate += PARAM_EPS / 2.0;
        let ov = diff_override(&base, &modified);
        assert_eq!(ov.spawn_rate, None);
    }

    #[test]
    fn exposed_param_binding_classification() {
        let constant = ExposedParam {
            handle: ParamHandle(1),
            name: String::from("intensity"),
            ty: ParamType::Scalar,
            binding: BindingSource::Constant(ParamValue::Scalar(1.0)),
        };
        let driven = ExposedParam {
            handle: ParamHandle(2),
            name: String::from("tint"),
            ty: ParamType::Color,
            binding: BindingSource::Material(7),
        };
        assert!(!constant.is_externally_driven());
        assert!(driven.is_externally_driven());
        assert_eq!(constant.binding.channel(), None);
        assert_eq!(driven.binding.channel(), Some(7));
    }

    #[test]
    fn asset_param_lookup_and_counts() {
        let asset = EmberAsset {
            format_version: 1,
            graph: GraphAssetRef(3),
            params: vec![
                ExposedParam {
                    handle: ParamHandle(1),
                    name: String::from("a"),
                    ty: ParamType::Scalar,
                    binding: BindingSource::Constant(ParamValue::Scalar(0.0)),
                },
                ExposedParam {
                    handle: ParamHandle(2),
                    name: String::from("b"),
                    ty: ParamType::Vec3,
                    binding: BindingSource::Gameplay(4),
                },
                ExposedParam {
                    handle: ParamHandle(3),
                    name: String::from("c"),
                    ty: ParamType::Bool,
                    binding: BindingSource::Timeline(9),
                },
            ],
            renderer_config: RendererConfig {
                id: RendererConfigId(0),
                kind: RendererKind::Sprite,
                writes_motion_vectors: true,
            },
            shading: ShadingSettings {
                model: EmberShadingModel::Pbr,
                receive_shadows: true,
                cast_shadows: false,
            },
            format: SerializationFormat::Binary,
        };
        assert_eq!(asset.externally_driven_count(), 2);
        assert!(asset.param(ParamHandle(2)).is_some());
        assert!(asset.param(ParamHandle(99)).is_none());
    }

    #[test]
    fn point_cache_and_attribute_map_channel_lookup() {
        let cache = PointCacheBinding {
            resource: ResourceHandle(11),
            point_count: 1024,
            mappings: vec![
                ChannelMapping {
                    channel: PointCacheChannel::Position,
                    source_index: 0,
                },
                ChannelMapping {
                    channel: PointCacheChannel::Velocity,
                    source_index: 3,
                },
            ],
        };
        assert_eq!(cache.source_for(PointCacheChannel::Position), Some(0));
        assert_eq!(cache.source_for(PointCacheChannel::Velocity), Some(3));
        assert_eq!(cache.source_for(PointCacheChannel::Color), None);

        let map = AttributeMapBinding {
            texture: ResourceHandle(12),
            width: 64,
            height: 32,
            mappings: vec![ChannelMapping {
                channel: PointCacheChannel::Color,
                source_index: 2,
            }],
        };
        assert_eq!(map.texel_count(), 2048);
        assert_eq!(map.source_for(PointCacheChannel::Color), Some(2));
        assert_eq!(map.source_for(PointCacheChannel::Age), None);
    }

    fn bump_only(mut data: ModuleData) -> ModuleData {
        data.blob.push(0xAB);
        data
    }

    #[test]
    fn migration_chain_applies_each_step_in_order() {
        let chain = MigrationChain {
            type_id: 42,
            current_version: 3,
            steps: vec![
                Migration {
                    from_version: 1,
                    to_version: 2,
                    apply: bump_only,
                },
                Migration {
                    from_version: 2,
                    to_version: 3,
                    apply: bump_only,
                },
            ],
        };
        assert!(chain.is_monotonic());
        let old = ModuleData {
            type_id: 42,
            version: 1,
            blob: Vec::new(),
        };
        let migrated = chain.migrate(old).expect("migration should succeed");
        assert_eq!(migrated.version, 3);
        // Two steps ran, each appending one byte.
        assert_eq!(migrated.blob.len(), 2);
    }

    #[test]
    fn migration_current_version_is_noop() {
        let chain = MigrationChain {
            type_id: 42,
            current_version: 2,
            steps: Vec::new(),
        };
        let data = ModuleData {
            type_id: 42,
            version: 2,
            blob: Vec::new(),
        };
        let out = chain.migrate(data.clone()).expect("no-op should succeed");
        assert_eq!(out, data);
    }

    #[test]
    fn migration_rejects_future_and_gapped_versions() {
        let chain = MigrationChain {
            type_id: 42,
            current_version: 2,
            steps: vec![Migration {
                from_version: 1,
                to_version: 2,
                apply: bump_only,
            }],
        };
        let newer = ModuleData {
            type_id: 42,
            version: 5,
            blob: Vec::new(),
        };
        assert_eq!(
            chain.migrate(newer),
            Err(MigrationError::VersionTooNew {
                found: 5,
                current: 2,
            })
        );

        let gapped_chain = MigrationChain {
            type_id: 42,
            current_version: 3,
            steps: vec![Migration {
                from_version: 1,
                to_version: 2,
                apply: bump_only,
            }],
        };
        let data = ModuleData {
            type_id: 42,
            version: 1,
            blob: Vec::new(),
        };
        assert_eq!(
            gapped_chain.migrate(data),
            Err(MigrationError::NoMigrationFrom(2))
        );
    }

    #[test]
    fn hot_reload_happy_path_bumps_generation() {
        let mut hr = HotReload::default();
        assert_eq!(hr.state, ReloadState::Idle);
        assert_eq!(
            hr.on_event(ReloadEvent::GraphEdited),
            ReloadState::Recompiling
        );
        assert!(hr.state.is_busy());
        assert_eq!(
            hr.on_event(ReloadEvent::CompileSucceeded),
            ReloadState::Swapping
        );
        assert_eq!(hr.on_event(ReloadEvent::SwapCompleted), ReloadState::Idle);
        assert_eq!(hr.generation, 1);
    }

    #[test]
    fn hot_reload_failure_then_retry() {
        let mut hr = HotReload::default();
        hr.on_event(ReloadEvent::ShadingEdited);
        assert_eq!(hr.on_event(ReloadEvent::CompileFailed), ReloadState::Failed);
        // A failed compile does not advance the live generation.
        assert_eq!(hr.generation, 0);
        assert_eq!(hr.on_event(ReloadEvent::Retry), ReloadState::Recompiling);
        assert_eq!(
            hr.on_event(ReloadEvent::CompileSucceeded),
            ReloadState::Swapping
        );
        assert_eq!(hr.on_event(ReloadEvent::SwapCompleted), ReloadState::Idle);
        assert_eq!(hr.generation, 1);
    }

    #[test]
    fn hot_reload_reset_from_any_state_and_ignores_spurious_events() {
        let mut hr = HotReload::default();
        hr.on_event(ReloadEvent::GraphEdited);
        // A swap-completed while recompiling is spurious and ignored.
        assert_eq!(
            hr.on_event(ReloadEvent::SwapCompleted),
            ReloadState::Recompiling
        );
        assert_eq!(hr.on_event(ReloadEvent::Reset), ReloadState::Idle);
        assert_eq!(hr.generation, 0);
    }

    #[test]
    fn transition_is_pure_and_matches_machine() {
        assert_eq!(
            transition(ReloadState::Idle, ReloadEvent::GraphEdited),
            ReloadState::Recompiling
        );
        assert_eq!(
            transition(ReloadState::Failed, ReloadEvent::Retry),
            ReloadState::Recompiling
        );
        assert_eq!(
            transition(ReloadState::Swapping, ReloadEvent::Reset),
            ReloadState::Idle
        );
    }
}
