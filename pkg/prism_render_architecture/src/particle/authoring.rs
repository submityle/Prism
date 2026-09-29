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
//! Nothing here links a real reflection or serialization crate: `bevy_reflect`,
//! `serde`, and a `RON`/binary writer are all out of scope for this contracts
//! layer. Instead the format is expressed with contract types, enums, version
//! numbers, and migration function *signatures*; the stable field order that a
//! real `RON` or binary codec must honour is documented on
//! [`EMBER_FIELD_ORDER`] rather than implemented. Runtime binding to
//! `gameplay`, materials, or the timeline, and the `GPU` kernel recompile the
//! hot-reload machine drives, live behind these contracts and are documented as
//! pending the `GPU` backend.

use alloc::string::String;
use alloc::vec::Vec;

use super::EmberShadingModel;

/// Absolute tolerance for comparing authored `f32` parameters when diffing an
/// override against its base (design §30). Two scalars closer than this are
/// treated as "unchanged" so a round-trip through a text codec does not
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
///
/// The choice is a codec decision only; the *field order* is fixed for both
/// formats (see [`EMBER_FIELD_ORDER`]) so a text diff and a binary hash both
/// stay stable across saves. No real codec (`serde`, a `RON` writer) is linked
/// at this layer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SerializationFormat {
    /// A human-diffable text form (`RON`), preferred while authoring so version
    /// control shows meaningful field-level diffs.
    Ron,
    /// A compact little-endian binary form, preferred for shipped builds where
    /// load time and size matter more than readability.
    Binary,
}

impl SerializationFormat {
    /// Returns `true` for the text (`RON`) form and `false` for the binary form.
    #[must_use]
    pub fn is_text(self) -> bool {
        matches!(self, SerializationFormat::Ron)
    }
}

/// What an asset is being serialized *for*, used to pick a default
/// [`SerializationFormat`] (design §30).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AssetPurpose {
    /// Iterating in the editor: prefer a diffable text form.
    Authoring,
    /// Cooking a shipped build: prefer the compact binary form.
    Shipping,
}

/// Picks the recommended [`SerializationFormat`] for an [`AssetPurpose`].
///
/// Authoring maps to [`SerializationFormat::Ron`] (diffable), shipping maps to
/// [`SerializationFormat::Binary`] (compact).
#[must_use]
pub fn recommended_format(purpose: AssetPurpose) -> SerializationFormat {
    match purpose {
        AssetPurpose::Authoring => SerializationFormat::Ron,
        AssetPurpose::Shipping => SerializationFormat::Binary,
    }
}

/// The canonical field order every `.ember` codec must emit, in order
/// (design §30).
///
/// Both the `RON` and binary writers walk fields in exactly this order so a
/// text diff stays readable and a content hash stays stable regardless of the
/// in-memory layout. This is the *documented* contract for a future `serde` /
/// `bevy_reflect` implementation, not a live codec.
pub const EMBER_FIELD_ORDER: [&str; 6] = [
    "format_version",
    "graph",
    "params",
    "renderer_config",
    "shading",
    "format",
];

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
    /// The data is newer than the running code can understand.
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
    /// Nothing pending; the live pipeline is current.
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
            recommended_format(AssetPurpose::Authoring),
            SerializationFormat::Ron
        );
        assert_eq!(
            recommended_format(AssetPurpose::Shipping),
            SerializationFormat::Binary
        );
        assert!(SerializationFormat::Ron.is_text());
        assert!(!SerializationFormat::Binary.is_text());
    }

    #[test]
    fn stable_field_order_is_fixed() {
        assert_eq!(EMBER_FIELD_ORDER.len(), 6);
        assert_eq!(EMBER_FIELD_ORDER[0], "format_version");
        assert_eq!(EMBER_FIELD_ORDER[5], "format");
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
            format: SerializationFormat::Ron,
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
        let future = ModuleData {
            type_id: 42,
            version: 5,
            blob: Vec::new(),
        };
        assert_eq!(
            chain.migrate(future),
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
