//! Self-describing PCM material schema — the single declarative source of truth
//! for the über-BSDF parameter layout (design doc §17.5).
//!
//! # Why
//!
//! The lobe field layout, bit masks, defaults and valid ranges are consumed by
//! Rust (`surface.rs` / `record.rs`), the WESL unpack/sample path, the CPU
//! golden in `prism_render_shading`, the doc field tables (§4.4), and the
//! compliance enum (§4.5). Adding one lobe must not let those consumers drift
//! into different CPU/GPU ABI or compliance contracts.
//!
//! This crate holds the declarative schema and the deterministic, pure layout
//! derivation consumed by the S2 codegen. The generator emits the committed
//! Rust layout facts and WESL unpack twin; freshness gates keep those products
//! byte-identical to regeneration. The S1 acceptance gate remains in
//! `prism_render_material` and asserts, byte-for-byte, that the schema-derived
//! layout equals the authored typed surface image for every lobe subset.
//!
//! # Design boundary (§17.5.5)
//!
//! The schema owns *layout and contract* (field name/type/bit-region/pack
//! offset/consumed-by step/default/valid predicate/energy policy/view
//! dependence). It deliberately does **not** express
//! behavioural math; that stays in typed hooks. The schema is not Turing
//! complete by design.

#![forbid(unsafe_code)]

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use core::fmt;

use serde::Deserialize;

pub mod codegen;
/// Schema-derived bake-time specialization and CPU sample/golden metadata.
///
/// This module is generated from `schema/surface.toml`; keeping it compiled
/// here makes S3/S4 products part of the crate contract instead of an
/// uncompiled file that can silently drift.
#[allow(missing_docs)]
pub mod generated_contract;
mod response;
pub use response::*;

/// The canonical surface schema, embedded at build time from
/// `schema/surface.toml`. This is the committed single source of truth.
pub const SURFACE_SCHEMA_TOML: &str = include_str!("../schema/surface.toml");

/// Parse and validate the embedded canonical surface schema.
///
/// # Panics
///
/// Panics if the committed `schema/surface.toml` fails to parse or validate.
/// The schema is an authored, version-controlled asset, so a failure here is a
/// build-time programming error, not a runtime condition.
#[must_use]
pub fn surface_schema() -> SurfaceSchema {
    load_surface_schema().expect("embedded surface.toml must parse and validate")
}

/// Parse and validate the embedded canonical surface schema, returning any
/// parse or validation error instead of panicking.
pub fn load_surface_schema() -> Result<SurfaceSchema, SchemaError> {
    SurfaceSchema::parse(SURFACE_SCHEMA_TOML)
}

/// A fully parsed, validated surface schema.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceSchema {
    /// Monotonic schema format version (independent of `abi_version`).
    pub schema_version: u32,
    /// The runtime `MATERIAL_ABI_VERSION` this layout targets.
    pub abi_version: u32,
    /// Bit-level specialization identity layout used by bake-time products.
    /// This is metadata for code generation, not a runtime header field.
    pub specialization: SpecializationSchema,
    /// The always-present über-BSDF core record.
    pub core: Record,
    /// Optional lobes, declared in any order (`registry_slot` fixes identity
    /// and pack order, not declaration order).
    #[serde(default, rename = "lobe")]
    pub lobes: Vec<Lobe>,
    /// Runtime dynamic parameter channels (§17.1). Not lobes: a channel carries
    /// no packed block, it owns one of the header's `dynamic_*` words.
    #[serde(default, rename = "channel")]
    pub channels: Vec<Channel>,
    /// Response program identities and their five consumption contracts.
    #[serde(default, rename = "response_program")]
    pub response_programs: Vec<ResponseProgram>,
    /// Object-space cache auxiliary declarations (§14.3).
    #[serde(default, rename = "cache_aux")]
    pub cache_aux: Vec<CacheAux>,
    /// Bake/pass-time extensions that resolve before the runtime and therefore
    /// hold no header-digest slot (§5.2).
    #[serde(default, rename = "identity_extension")]
    pub identity_extensions: Vec<IdentityExtension>,
}

/// Declarative bit layout for the material specialization identity.
///
/// The current runtime uses execution-table bindings rather than carrying this
/// identity in `GpuMaterialHeader`. The layout remains schema-owned because
/// bake-time permutation keys, bin keys, and diagnostics still need one stable
/// packing contract. Codegen emits a pure value type without reviving the
/// removed runtime API.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SpecializationSchema {
    /// Width of the serialized identity.
    pub storage_bits: u8,
    /// Illumination axis bit range.
    pub illumination: BitRange,
    /// Live surface-lobe mask bit range.
    pub lobe_mask: BitRange,
    /// Raster/blend class bit range.
    pub render_class: BitRange,
    /// Reserved high bits.
    pub reserved: BitRange,
}

/// A half-open bit range `[shift, shift + bits)`.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BitRange {
    /// First bit occupied by this range.
    pub shift: u8,
    /// Number of bits occupied by this range.
    pub bits: u8,
}

/// A runtime dynamic parameter channel (§17.1).
///
/// The value itself is a per-frame scalar in the material header; the schema
/// declares its identity, which header word carries it, and its neutral default
/// (a channel is at its identity value there, so a material that never drives it
/// is bit-for-bit unchanged).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Channel {
    /// Human-facing channel id (display only).
    pub id: String,
    /// Immutable global identity, shared with the PCM registry row.
    pub uid: String,
    /// Mutable governance namespace (§5.4).
    pub namespace: Namespace,
    /// Mutable vendor tag (§5.4).
    pub vendor: String,
    /// Mutable stability level (§5.4).
    pub stability: Stability,
    /// Header-digest slot (the PCM row's address; §5.2).
    ///
    /// This is distinct from `header_word`: the digest slot is the extension
    /// identity's address, while `header_word` is the index of the per-material
    /// dynamic scalar inside `GpuMaterialHeader`.
    pub registry_slot: u32,
    /// Which of the header's `dynamic_*` words carries this channel's value.
    pub header_word: u32,
    /// Neutral default: the value at which the channel is an exact identity.
    pub default: DefaultValue,
    /// Validity predicate over the runtime value.
    #[serde(default)]
    pub valid: Option<String>,
    /// Platform capabilities required beyond the baseline.
    #[serde(default)]
    pub requires_caps: Vec<String>,
    /// Whether the channel's effect changes with the view direction.
    pub view_dependent: bool,
    /// Whether a plan may use this scalar as a mask, inheriting capabilities.
    pub usable_as_mask: bool,
}

/// A scalar validity interval declared by a field or runtime channel.
///
/// The schema currently uses the compact `lower..=upper` spelling (with an
/// optional exclusive upper bound via `lower..upper`). Keeping the parsed form
/// here makes the range contract executable instead of leaving it as an
/// unchecked comment/string that each consumer would have to reinterpret.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValidRange {
    /// Lower endpoint.
    pub min: f32,
    /// Upper endpoint.
    pub max: f32,
    /// Whether the lower endpoint is included. Rust-style ranges always include it.
    pub min_inclusive: bool,
    /// Whether the upper endpoint is included (`..=` versus `..`).
    pub max_inclusive: bool,
}

impl ValidRange {
    /// Parse the schema's scalar interval spelling.
    pub fn parse(expression: &str) -> Result<Self, String> {
        let expression = expression.trim();
        let (min_text, max_text) = expression
            .split_once("..")
            .ok_or_else(|| format!("expected a scalar range, got {expression:?}"))?;
        let max_inclusive = max_text.starts_with('=');
        let max_text = max_text.strip_prefix('=').unwrap_or(max_text).trim();
        let min = parse_range_endpoint(min_text.trim())?;
        let max = parse_range_endpoint(max_text)?;
        if min.is_nan() || max.is_nan() {
            return Err(format!("range {expression:?} cannot contain NaN"));
        }
        if min > max || (min == max && !max_inclusive) {
            return Err(format!(
                "range {expression:?} has inverted or empty endpoints"
            ));
        }
        Ok(Self {
            min,
            max,
            min_inclusive: true,
            max_inclusive,
        })
    }

    /// Whether a value satisfies this interval.
    #[must_use]
    pub fn contains(self, value: f32) -> bool {
        if value.is_nan() {
            return false;
        }
        let lower = if self.min_inclusive {
            value >= self.min
        } else {
            value > self.min
        };
        let upper = if self.max_inclusive {
            value <= self.max
        } else {
            value < self.max
        };
        lower && upper
    }
}

/// Parse a validity endpoint, accepting the finite literals used by the schema
/// plus `inf`/`-inf` for unbounded OpenPBR quantities such as luminance.
pub fn parse_range_endpoint(text: &str) -> Result<f32, String> {
    match text {
        "inf" | "+inf" | "infinity" | "+infinity" => Ok(f32::INFINITY),
        "-inf" | "-infinity" => Ok(f32::NEG_INFINITY),
        _ => text
            .parse::<f32>()
            .map_err(|error| format!("invalid range endpoint {text:?}: {error}")),
    }
}

/// An object-space cache auxiliary output (§14.3).
///
/// A cache-aux row is header-resident and has a registry slot, but carries no
/// per-material packed block or dynamic word of its own: the runtime reads its
/// identity from the header digest.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheAux {
    /// Human-facing cache-aux id.
    pub id: String,
    /// Immutable global identity, shared with the PCM registry row.
    pub uid: String,
    /// Header-digest slot (immutable; §5.4).
    pub registry_slot: u32,
    /// Mutable governance namespace (§5.4).
    pub namespace: Namespace,
    /// Mutable vendor tag (§5.4).
    pub vendor: String,
    /// Mutable stability level (§5.4).
    pub stability: Stability,
    /// Platform capabilities required beyond the baseline.
    #[serde(default)]
    pub requires_caps: Vec<String>,
    /// Whether the cached quantity changes with the view direction.
    pub view_dependent: bool,
}

/// A bake/pass-time extension with no header-digest address (§5.2).
///
/// These rows are registered, promoted and negotiated exactly like
/// header-resident extensions, but they evaporate into the bake-time
/// permutation key or the pass graph before the runtime. The schema still owns
/// their identity, governance, capability and view-dependence facts so the
/// hand-written PCM catalogue cannot drift from the manifest.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdentityExtension {
    /// Human-facing extension id.
    pub id: String,
    /// Immutable global identity, shared with the PCM registry row.
    pub uid: String,
    /// Which bake/pass-time kind this row registers.
    pub kind: IdentityKind,
    /// Mutable governance namespace (§5.4).
    pub namespace: Namespace,
    /// Mutable vendor tag (§5.4).
    pub vendor: String,
    /// Mutable stability level (§5.4).
    pub stability: Stability,
    /// Platform capabilities required beyond the baseline.
    #[serde(default)]
    pub requires_caps: Vec<String>,
    /// Whether the produced result changes with the view direction.
    pub view_dependent: bool,
}

/// A record is a fixed-width group of `u32` words tiled by its fields.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    /// Number of `u32` words the record occupies.
    pub words: u32,
    /// Fields, which must tile `[0, words)` exactly with no gaps or overlaps.
    #[serde(default, rename = "field")]
    pub fields: Vec<Field>,
}

/// An optional lobe: a variable-length word record plus its manifest identity.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lobe {
    /// Derived, human-facing display id (§5.4 — not an identity anchor).
    pub id: String,
    /// Immutable global identity (§5.4). Never reused once allocated.
    pub uid: String,
    /// Immutable bit index in the surface `LobeMask`; also the canonical
    /// low-bit-first pack order. Never recycled (§5.4 bit-region isolation).
    pub registry_slot: u32,
    /// Name of the matching `LobeMask` associated constant (e.g. `EMISSION`).
    pub mask_const: String,
    /// Mutable governance namespace (§5.4). `KHR` auto-checks against `OpenPBR`.
    pub namespace: Namespace,
    /// Mutable vendor tag (§5.4).
    pub vendor: String,
    /// Mutable stability level (§5.4 promotion flips mutable fields only).
    pub stability: Stability,
    /// Platform capabilities this lobe needs beyond the shared über path
    /// (§5.3): e.g. `object_space_cache`, `virtual_texturing`, `compute_scatter`.
    /// Empty means "runs on the baseline über path". The capability negotiator
    /// checks the union of these for a material's **required** extensions.
    #[serde(default)]
    pub requires_caps: Vec<String>,
    /// Authoring `ClosureKind` implementation family used by this lobe.
    ///
    /// This is a build-time relationship, not a runtime bucket or program id.
    /// Several lobes may augment the same implementation. Codegen emits typed
    /// references, so a family absent from the material IR fails compilation.
    #[serde(default)]
    pub closure: Option<String>,
    /// Number of `u32` words the lobe occupies. Lobes are **variable length**
    /// (§4.4): the pack cursor advances by each present lobe's own `words`, and
    /// the codegen emits a per-lobe footprint (`SURFACE_LOBE_WORDS_BY_SLOT` /
    /// `PRISM_LOBE_WORDS_*`). Today every lobe happens to be 4 words; growing a
    /// lobe to its full `OpenPBR` field set changes only this number.
    pub words: u32,
    /// Fields, which must tile `[0, words)` exactly with no gaps or overlaps.
    #[serde(default, rename = "field")]
    pub fields: Vec<Field>,
}

/// A single declared field: the field-level completion of the manifest.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Field {
    /// Field name, unique within its record.
    pub name: String,
    /// Storage type; determines the word footprint.
    #[serde(rename = "type")]
    pub ty: FieldType,
    /// Bit-region partition key, equal to the owning namespace (§5.4).
    pub bit_region: Namespace,
    /// First `u32` word of this field within its record.
    pub pack_slot: u32,
    /// Which pipeline steps consume this field (§8.5).
    #[serde(default)]
    pub consumed_by: Vec<ConsumedStep>,
    /// Exhaustive response semantics. Missing entries are errors, not ignores.
    #[serde(default)]
    pub responses: BTreeMap<String, FieldResponse>,
    /// Neutral default used when the owning lobe is absent.
    pub default: DefaultValue,
    /// Human/machine-checkable validity predicate over the field value.
    #[serde(default)]
    pub valid: Option<String>,
    /// How the field participates in the layered energy budget (§17.2).
    #[serde(default)]
    pub energy_policy: EnergyPolicy,
    /// View dependence (§14.3): `invariant` fields may enter object-space cache.
    #[serde(default)]
    pub view: ViewDependence,
    /// Optional debug-view binding name (§4.6).
    #[serde(default)]
    pub debug_view: Option<String>,
    /// Target field name in the decoded `PrismSurfaceParameters` /
    /// `GpuSurfaceParameters` fat struct, when it differs from [`Self::name`].
    /// Lobe-local names are terse (e.g. the face lobe's `softness`), but the
    /// fat struct namespaces them (`npr_face_softness`); codegen (§17.5 S2) uses
    /// this to emit the correct assignment target. Defaults to `name`.
    #[serde(default)]
    pub surface_field: Option<String>,
}

impl Field {
    /// Target field name in the decoded fat surface struct. Falls back to
    /// [`Self::name`] when no explicit [`Self::surface_field`] override is set.
    #[must_use]
    pub fn surface_field(&self) -> &str {
        self.surface_field.as_deref().unwrap_or(&self.name)
    }
}

/// Storage type of a field; fixes its word footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    /// 32-bit float, one word.
    F32,
    /// Four packed 32-bit floats, four words (vec4<f32>).
    Vec4f,
    /// 16-bit float occupying a half-word slot (reserved for ABI v5 packing).
    F16,
    /// Explicit reserved padding word carrying no authored value.
    Pad,
}

impl FieldType {
    /// Canonical schema spelling used by generated field metadata.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Vec4f => "vec4f",
            Self::F16 => "f16",
            Self::Pad => "pad",
        }
    }

    /// Number of `u32` words this type occupies in a packed record.
    #[must_use]
    pub const fn words(self) -> u32 {
        match self {
            Self::F32 | Self::Pad | Self::F16 => 1,
            Self::Vec4f => 4,
        }
    }

    /// Number of scalar components a default literal must supply.
    #[must_use]
    pub const fn scalar_components(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Pad => 1,
            Self::Vec4f => 4,
        }
    }
}

/// Governance namespace / bit-region partition key (§5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Namespace {
    /// Khronos `OpenPBR` specification core.
    Khr,
    /// Ratified cross-vendor extension.
    Ext,
    /// Prism first-party extension.
    Prism,
    /// Private studio/vendor extension (`X_studio_*`).
    #[serde(rename = "X_STUDIO")]
    XStudio,
}

impl Namespace {
    /// Canonical schema spelling of this governance namespace.
    ///
    /// This is identity metadata (§5.4): the value never changes on promotion
    /// (`X_studio → EXT → KHR` only flips this string, not the `uid`/slot), and
    /// it is the bit-region partition key that keeps OpenPBR-spec fields from
    /// colliding with Prism-private ones (§4.5 constraint 2).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Khr => "KHR",
            Self::Ext => "EXT",
            Self::Prism => "PRISM",
            Self::XStudio => "X_STUDIO",
        }
    }

    /// Whether this namespace belongs to the `OpenPBR` specification core, i.e.
    /// the canonical bit region a pure-`OpenPBR` material may occupy.
    ///
    /// Only `KHR` is spec core: `EXT` is a ratified *cross-vendor* extension
    /// that still leaves the `OpenPBR` baseline, so it lives in the extension
    /// region alongside `PRISM`/`X_STUDIO` (§4.5 constraint 2).
    #[must_use]
    pub const fn is_openpbr_core(self) -> bool {
        matches!(self, Self::Khr)
    }
}

/// Stability level of a capability (§5.4).
///
/// The three levels are exactly the ones the PCM registry speaks
/// (`prism_material_pcm::Stability`): the schema and the registry must not
/// drift into two vocabularies for the same governance axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stability {
    /// Under active design; layout or response may still change.
    Experimental,
    /// Layout frozen, response settling; a promotion candidate.
    Candidate,
    /// Frozen; covered by compliance parity.
    Stable,
}

impl Stability {
    /// The schema spelling, emitted verbatim by codegen and parsed back by the
    /// PCM registry.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Experimental => "experimental",
            Self::Candidate => "candidate",
            Self::Stable => "stable",
        }
    }
}

/// A PCM kind that resolves at bake/pass time and therefore holds no header
/// slot (§5.2's identity-only half).
///
/// The enum is the schema-side vocabulary for the kinds `provider`, `transfer`,
/// `downgrade`, `author_node` and `composite`; the runtime resolves the
/// canonical spelling through its own [`Kind`](https://docs.rs/prism_material_pcm)
/// and a mismatch fails the PCM/schema parity gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityKind {
    /// Produces a page/atlas (SDF, LUT, measured BTF).
    Provider,
    /// A transfer remap applied between response and composition.
    Transfer,
    /// A distance/quality downgrade policy hook.
    Downgrade,
    /// An authoring-layer node type (§4, editor front-end).
    AuthorNode,
    /// A composite pass recorded into the final step.
    Composite,
}

impl IdentityKind {
    /// Canonical lowercase spelling used in the schema and PCM catalogue.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Provider => "provider",
            Self::Transfer => "transfer",
            Self::Downgrade => "downgrade",
            Self::AuthorNode => "author_node",
            Self::Composite => "composite",
        }
    }
}

/// A pipeline step that consumes a field (§8.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsumedStep {
    /// Word-heap unpack (`material_unpack`).
    Unpack,
    /// BSDF sampling / evaluation (`material_sample`).
    Sample,
    /// CPU golden mirror (`prism_render_shading`).
    Golden,
    /// Built-in debug view (§4.6).
    Debug,
    /// Doc field table (§4.4).
    Doc,
    /// Compliance predicate (§4.5).
    Compliance,
    /// Lit response semantics (not merely unpacking).
    Lit,
    /// NPR response semantics.
    Npr,
    /// Unlit response semantics.
    Unlit,
}

impl ConsumedStep {
    /// Canonical schema spelling used by generated field metadata.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unpack => "unpack",
            Self::Sample => "sample",
            Self::Golden => "golden",
            Self::Debug => "debug",
            Self::Doc => "doc",
            Self::Compliance => "compliance",
            Self::Lit => "lit",
            Self::Npr => "npr",
            Self::Unlit => "unlit",
        }
    }
}

/// How a field participates in the layered energy budget (§17.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyPolicy {
    /// Shares a coupled energy budget with other coupled lobes/fields.
    Coupled,
    /// Adds energy independently (e.g. emission).
    Independent,
    /// Not an energy-carrying parameter.
    #[default]
    None,
}

/// View dependence of a field (§14.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewDependence {
    /// View-independent; may be cached in object space.
    #[default]
    Invariant,
    /// View-dependent; must be evaluated per view.
    Dependent,
}

/// A neutral default literal: either a scalar or a fixed vector of scalars.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum DefaultValue {
    /// Single scalar default.
    Scalar(f64),
    /// Component-wise vector default.
    Vector(Vec<f64>),
}

impl DefaultValue {
    /// The default as a fixed-length component vector, replicating a scalar to
    /// a single component.
    #[must_use]
    pub fn components(&self) -> Vec<f32> {
        match self {
            Self::Scalar(v) => vec![*v as f32],
            Self::Vector(v) => v.iter().map(|c| *c as f32).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Derived layout
// ---------------------------------------------------------------------------

/// The absolute placement of one field inside a packed surface block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldPlacement {
    /// Owning lobe id, or `None` for a core field.
    pub lobe: Option<String>,
    /// Field name.
    pub name: String,
    /// Absolute `u32` word offset from the start of the packed block.
    pub word_offset: u32,
    /// Word footprint of the field.
    pub words: u32,
    /// Whether this field is reserved padding (no authored value).
    pub is_pad: bool,
}

impl SurfaceSchema {
    /// Parse a schema from TOML and run full structural validation.
    pub fn parse(toml_src: &str) -> Result<Self, SchemaError> {
        let schema: Self =
            toml::from_str(toml_src).map_err(|e| SchemaError::Parse(e.to_string()))?;
        schema.validate()?;
        Ok(schema)
    }

    /// Lobes in canonical pack order (ascending `registry_slot`).
    #[must_use]
    pub fn canonical_lobes(&self) -> Vec<&Lobe> {
        let mut lobes: Vec<&Lobe> = self.lobes.iter().collect();
        lobes.sort_by_key(|l| l.registry_slot);
        lobes
    }

    /// Number of defined lobes (equivalent to `LobeMask::COUNT`).
    #[must_use]
    pub fn lobe_count(&self) -> u32 {
        self.lobes.len() as u32
    }

    /// Packed word count for a given lobe-mask bit set: core words plus the
    /// footprint of every present lobe. Mirrors
    /// `SurfaceParameterBlock::packed_len_words`.
    #[must_use]
    pub fn packed_len_words(&self, mask_bits: u32) -> u32 {
        let mut words = self.core.words;
        for lobe in &self.lobes {
            if mask_bits & (1 << lobe.registry_slot) != 0 {
                words += lobe.words;
            }
        }
        words
    }

    /// Resolve the absolute placement of every field present under `mask_bits`,
    /// in packed order (core fields, then present lobes low-bit-first). This is
    /// the deterministic pure layout a future codegen emits.
    #[must_use]
    pub fn placements(&self, mask_bits: u32) -> Vec<FieldPlacement> {
        let mut out = Vec::new();
        for field in &self.core.fields {
            out.push(FieldPlacement {
                lobe: None,
                name: field.name.clone(),
                word_offset: field.pack_slot,
                words: field.ty.words(),
                is_pad: field.ty == FieldType::Pad,
            });
        }
        let mut cursor = self.core.words;
        for lobe in self.canonical_lobes() {
            if mask_bits & (1 << lobe.registry_slot) == 0 {
                continue;
            }
            for field in &lobe.fields {
                out.push(FieldPlacement {
                    lobe: Some(lobe.id.clone()),
                    name: field.name.clone(),
                    word_offset: cursor + field.pack_slot,
                    words: field.ty.words(),
                    is_pad: field.ty == FieldType::Pad,
                });
            }
            cursor += lobe.words;
        }
        out
    }

    /// Absolute word offset of a specific lobe field under `mask_bits`, or
    /// `None` if the lobe is absent or the field is unknown.
    #[must_use]
    pub fn lobe_field_offset(
        &self,
        mask_bits: u32,
        lobe_id: &str,
        field_name: &str,
    ) -> Option<u32> {
        let mut cursor = self.core.words;
        for lobe in self.canonical_lobes() {
            if mask_bits & (1 << lobe.registry_slot) == 0 {
                continue;
            }
            if lobe.id == lobe_id {
                return lobe
                    .fields
                    .iter()
                    .find(|f| f.name == field_name)
                    .map(|f| cursor + f.pack_slot);
            }
            cursor += lobe.words;
        }
        None
    }

    /// Absolute word offset of a core field, or `None` if unknown.
    #[must_use]
    pub fn core_field_offset(&self, field_name: &str) -> Option<u32> {
        self.core
            .fields
            .iter()
            .find(|f| f.name == field_name)
            .map(|f| f.pack_slot)
    }

    /// Validate structural invariants that any valid schema must satisfy.
    pub fn validate(&self) -> Result<(), SchemaError> {
        validate_specialization(&self.specialization)?;
        validate_record("core", &self.core)?;

        // The decoded fat surface is the CPU/GPU mirror target. Its field
        // names must be globally unique even though packed records are split
        // into core/lobes; otherwise two schema inputs would project onto one
        // ABI member and one value would silently overwrite the other.
        let mut surface_fields = BTreeSet::new();
        for field in &self.core.fields {
            if field.ty != FieldType::Pad && !surface_fields.insert(field.surface_field()) {
                return Err(SchemaError::DuplicateSurfaceFieldName(
                    field.surface_field().to_string(),
                ));
            }
        }

        let mut slots = BTreeSet::new();
        let mut uids = BTreeSet::new();
        for lobe in &self.lobes {
            if !slots.insert(lobe.registry_slot) {
                return Err(SchemaError::DuplicateRegistrySlot(lobe.registry_slot));
            }
            if lobe.registry_slot >= self.specialization.lobe_mask.bits as u32 {
                return Err(SchemaError::InvalidSpecializationLayout(format!(
                    "lobe `{}` registry_slot {} does not fit the specialization lobe_mask width {}",
                    lobe.id, lobe.registry_slot, self.specialization.lobe_mask.bits
                )));
            }
            if !uids.insert(lobe.uid.clone()) {
                return Err(SchemaError::DuplicateUid(lobe.uid.clone()));
            }
            validate_record(&lobe.id, &record_view(lobe))?;
            for field in &lobe.fields {
                if field.ty != FieldType::Pad && !surface_fields.insert(field.surface_field()) {
                    return Err(SchemaError::DuplicateSurfaceFieldName(
                        field.surface_field().to_string(),
                    ));
                }
            }
            // A KHR / PRISM lobe's fields must all live in the owning bit region.
            for field in &lobe.fields {
                if field.bit_region != lobe.namespace {
                    return Err(SchemaError::FieldBitRegionMismatch {
                        lobe: lobe.id.clone(),
                        field: field.name.clone(),
                    });
                }
            }
        }

        // §4.5 constraint 2 (bit-region isolation): the `LobeMask` layout is two
        // dense segments. Every OpenPBR `KHR` lobe occupies the canonical region
        // `0..k`; every extension lobe (PRISM/EXT/X_STUDIO) occupies the
        // extension region `EXTENSION_SLOT_BASE..`. The gap between them is
        // reserved so new specification lobes can be appended to the canonical
        // region (and new private lobes to the extension region) without either
        // ever aliasing the other's slots.
        let mut canonical: Vec<u32> = self
            .lobes
            .iter()
            .filter(|lobe| lobe.namespace.is_openpbr_core())
            .map(|lobe| lobe.registry_slot)
            .collect();
        let mut extension: Vec<u32> = self
            .lobes
            .iter()
            .filter(|lobe| !lobe.namespace.is_openpbr_core())
            .map(|lobe| lobe.registry_slot)
            .collect();
        canonical.sort_unstable();
        extension.sort_unstable();
        for (index, slot) in canonical.iter().enumerate() {
            if *slot != index as u32 {
                return Err(SchemaError::NonContiguousRegistrySlots);
            }
        }
        for (index, slot) in extension.iter().enumerate() {
            if *slot != EXTENSION_SLOT_BASE + index as u32 {
                return Err(SchemaError::NonContiguousRegistrySlots);
            }
        }

        // Runtime dynamic channels (§17.1): each owns exactly one header word,
        // the words tile `0..channels.len()`, the default is a scalar, and the
        // identity is unique across the whole catalogue (lobes included) so a
        // channel can never alias a lobe's uid.
        for lobe in &self.lobes {
            if let Some(closure) = &lobe.closure {
                if closure.is_empty() {
                    return Err(SchemaError::EmptyClosureName(lobe.id.clone()));
                }
            }
        }

        // All header-resident kinds share the lobe slot and identity space.
        let mut header_slots: BTreeSet<u32> =
            self.lobes.iter().map(|lobe| lobe.registry_slot).collect();
        for program in &self.response_programs {
            if !uids.insert(program.uid.clone()) {
                return Err(SchemaError::DuplicateUid(program.uid.clone()));
            }
            if !header_slots.insert(program.registry_slot) {
                return Err(SchemaError::DuplicateHeaderSlot(program.registry_slot));
            }
        }

        let mut channel_words = BTreeSet::new();
        for channel in &self.channels {
            if !channel_words.insert(channel.header_word) {
                return Err(SchemaError::DuplicateChannelWord(channel.header_word));
            }
            if !header_slots.insert(channel.registry_slot) {
                return Err(SchemaError::DuplicateHeaderSlot(channel.registry_slot));
            }
            if !uids.insert(channel.uid.clone()) {
                return Err(SchemaError::DuplicateUid(channel.uid.clone()));
            }
            if channel.default.components().len() != 1 {
                return Err(SchemaError::ChannelDefaultNotScalar(channel.id.clone()));
            }
            let range = channel
                .valid
                .as_deref()
                .ok_or_else(|| SchemaError::MissingValidityPredicate {
                    record: format!("channel {}", channel.id),
                    field: "value".to_string(),
                })
                .and_then(|expression| {
                    ValidRange::parse(expression).map_err(|reason| {
                        SchemaError::InvalidValidityPredicate {
                            record: format!("channel {}", channel.id),
                            field: "value".to_string(),
                            expression: expression.to_string(),
                            reason,
                        }
                    })
                })?;
            let default = channel.default.components()[0];
            if !range.contains(default) {
                return Err(SchemaError::DefaultOutOfRange {
                    record: format!("channel {}", channel.id),
                    field: "value".to_string(),
                    value: default.to_string(),
                    expression: channel.valid.clone().unwrap_or_default(),
                });
            }
        }
        for (index, word) in channel_words.iter().enumerate() {
            if *word != index as u32 {
                return Err(SchemaError::NonContiguousChannelWords);
            }
        }

        // Object-space cache auxiliaries: like channels and response programs they
        // are header-resident, but their payload is a cached render quantity,
        // not a per-material word. They still share the same slot space.
        for aux in &self.cache_aux {
            if !uids.insert(aux.uid.clone()) {
                return Err(SchemaError::DuplicateUid(aux.uid.clone()));
            }
            if !header_slots.insert(aux.registry_slot) {
                return Err(SchemaError::DuplicateHeaderSlot(aux.registry_slot));
            }
        }

        // Bake/pass-time identity rows share the same global uid space but no
        // header slot: their identity is resolved by the registry before the
        // runtime, so there is no per-material address to allocate.
        for extension in &self.identity_extensions {
            if !uids.insert(extension.uid.clone()) {
                return Err(SchemaError::DuplicateUid(extension.uid.clone()));
            }
        }
        self.validate_responses()
    }
}

/// First `registry_slot` of the Prism extension region (§4.5 constraint 2).
///
/// The canonical `OpenPBR` (`KHR`) bit region grows densely from slot `0`; every
/// non-`KHR` lobe occupies the extension region that starts here, leaving the
/// slots between as a reserved gap so the two regions can grow independently
/// without ever colliding.
pub const EXTENSION_SLOT_BASE: u32 = 16;

fn validate_specialization(layout: &SpecializationSchema) -> Result<(), SchemaError> {
    if layout.storage_bits == 0 || layout.storage_bits > 64 {
        return Err(SchemaError::InvalidSpecializationLayout(format!(
            "storage_bits must be in 1..=64, got {}",
            layout.storage_bits
        )));
    }
    let ranges = [
        ("illumination", layout.illumination),
        ("lobe_mask", layout.lobe_mask),
        ("render_class", layout.render_class),
        ("reserved", layout.reserved),
    ];
    for (name, range) in ranges {
        let end = u16::from(range.shift) + u16::from(range.bits);
        if range.bits == 0 || end > u16::from(layout.storage_bits) {
            return Err(SchemaError::InvalidSpecializationLayout(format!(
                "{name} range [{}..{}) is outside {} bits",
                range.shift, end, layout.storage_bits
            )));
        }
    }
    for (name, range) in [
        ("illumination", layout.illumination),
        ("lobe_mask", layout.lobe_mask),
        ("render_class", layout.render_class),
    ] {
        if range.bits > 32 {
            return Err(SchemaError::InvalidSpecializationLayout(format!(
                "{name} range is projected into a u32 component but declares {} bits",
                range.bits
            )));
        }
    }
    for (index, (left_name, left)) in ranges.iter().enumerate() {
        for (right_name, right) in ranges.iter().skip(index + 1) {
            let left_end = u16::from(left.shift) + u16::from(left.bits);
            let right_end = u16::from(right.shift) + u16::from(right.bits);
            if u16::from(left.shift) < right_end && u16::from(right.shift) < left_end {
                return Err(SchemaError::InvalidSpecializationLayout(format!(
                    "{left_name} overlaps {right_name}"
                )));
            }
        }
    }
    Ok(())
}

/// A borrowed record view so core and lobe records share one validator.
fn record_view(lobe: &Lobe) -> Record {
    Record {
        words: lobe.words,
        fields: lobe.fields.clone(),
    }
}

/// Validate that a record's fields tile `[0, words)` exactly and that every
/// default literal matches its field type's component count.
fn validate_record(name: &str, record: &Record) -> Result<(), SchemaError> {
    let mut covered = vec![false; record.words as usize];
    let mut names = BTreeSet::new();
    for field in &record.fields {
        if !names.insert(field.name.clone()) {
            return Err(SchemaError::DuplicateFieldName {
                record: name.to_string(),
                field: field.name.clone(),
            });
        }
        let start = field.pack_slot as usize;
        let end = start + field.ty.words() as usize;
        if end > record.words as usize {
            return Err(SchemaError::FieldOutOfBounds {
                record: name.to_string(),
                field: field.name.clone(),
            });
        }
        for word in &mut covered[start..end] {
            if *word {
                return Err(SchemaError::OverlappingFields {
                    record: name.to_string(),
                    field: field.name.clone(),
                });
            }
            *word = true;
        }
        if field.default.components().len() != field.ty.scalar_components() {
            return Err(SchemaError::DefaultArityMismatch {
                record: name.to_string(),
                field: field.name.clone(),
            });
        }
    }
    if covered.iter().any(|c| !c) {
        return Err(SchemaError::IncompleteTiling {
            record: name.to_string(),
        });
    }

    for field in &record.fields {
        if field.ty == FieldType::Pad {
            continue;
        }
        let expression =
            field
                .valid
                .as_deref()
                .ok_or_else(|| SchemaError::MissingValidityPredicate {
                    record: name.to_string(),
                    field: field.name.clone(),
                })?;
        let range = ValidRange::parse(expression).map_err(|reason| {
            SchemaError::InvalidValidityPredicate {
                record: name.to_string(),
                field: field.name.clone(),
                expression: expression.to_string(),
                reason,
            }
        })?;
        for value in field.default.components() {
            if !range.contains(value) {
                return Err(SchemaError::DefaultOutOfRange {
                    record: name.to_string(),
                    field: field.name.clone(),
                    value: value.to_string(),
                    expression: expression.to_string(),
                });
            }
        }
    }
    Ok(())
}

/// An error produced while parsing or validating a schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    /// The TOML failed to deserialize into the schema model.
    Parse(String),
    /// The specialization ranges do not form a valid, non-overlapping layout.
    InvalidSpecializationLayout(String),
    /// Two lobes declared the same `registry_slot`.
    DuplicateRegistrySlot(u32),
    /// Two lobes declared the same `uid`.
    DuplicateUid(String),
    /// `registry_slot` values are not a contiguous `0..n` range.
    NonContiguousRegistrySlots,
    /// A lobe declared an empty closure implementation name.
    EmptyClosureName(String),
    /// A response declaration is incomplete or inconsistent.
    ResponseContract(String),
    /// Two runtime channels claim the same header word.
    DuplicateChannelWord(u32),
    /// Two header-resident non-lobe records claim the same registry slot.
    DuplicateHeaderSlot(u32),
    /// Runtime channel words do not tile `0..channels.len()`.
    NonContiguousChannelWords,
    /// A runtime channel's neutral default is not a single scalar.
    ChannelDefaultNotScalar(String),
    /// A record declared the same field name twice.
    DuplicateFieldName {
        /// Owning record (core or lobe id).
        record: String,
        /// Offending field name.
        field: String,
    },
    /// A field extends past its record's declared word count.
    FieldOutOfBounds {
        /// Owning record (core or lobe id).
        record: String,
        /// Offending field name.
        field: String,
    },
    /// Two fields claim the same word.
    OverlappingFields {
        /// Owning record (core or lobe id).
        record: String,
        /// Offending field name.
        field: String,
    },
    /// A record's fields leave at least one word uncovered.
    IncompleteTiling {
        /// Owning record (core or lobe id).
        record: String,
    },
    /// A default literal's component count did not match its field type.
    DefaultArityMismatch {
        /// Owning record (core or lobe id).
        record: String,
        /// Offending field name.
        field: String,
    },
    /// Two packed fields project onto the same decoded fat-surface member.
    DuplicateSurfaceFieldName(String),
    /// A semantic field or channel omitted its executable validity range.
    MissingValidityPredicate {
        /// Owning record or channel id.
        record: String,
        /// Field name, or `value` for a channel scalar.
        field: String,
    },
    /// A validity range could not be parsed as a scalar interval.
    InvalidValidityPredicate {
        /// Owning record or channel id.
        record: String,
        /// Field name, or `value` for a channel scalar.
        field: String,
        /// Original schema expression.
        expression: String,
        /// Parser explanation.
        reason: String,
    },
    /// A declared neutral default lies outside its declared validity interval.
    DefaultOutOfRange {
        /// Owning record or channel id.
        record: String,
        /// Field name, or `value` for a channel scalar.
        field: String,
        /// Default value rendered for diagnostics.
        value: String,
        /// Original schema expression.
        expression: String,
    },
    /// A lobe field's bit region differs from the lobe's namespace.
    FieldBitRegionMismatch {
        /// Owning lobe id.
        lobe: String,
        /// Offending field name.
        field: String,
    },
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "schema parse error: {e}"),
            Self::InvalidSpecializationLayout(message) => {
                write!(f, "invalid specialization layout: {message}")
            }
            Self::EmptyClosureName(id) => {
                write!(f, "lobe {id:?} declares an empty closure name")
            }
            Self::ResponseContract(message) => write!(f, "response contract: {message}"),
            Self::DuplicateChannelWord(word) => {
                write!(f, "two runtime channels claim header word {word}")
            }
            Self::DuplicateHeaderSlot(slot) => {
                write!(f, "two header-resident records claim registry slot {slot}")
            }
            Self::NonContiguousChannelWords => {
                write!(
                    f,
                    "runtime channel header words must tile 0..channels.len()"
                )
            }
            Self::ChannelDefaultNotScalar(id) => {
                write!(f, "runtime channel {id:?} needs a scalar default")
            }
            Self::DuplicateRegistrySlot(slot) => {
                write!(f, "duplicate registry_slot {slot}")
            }
            Self::DuplicateUid(uid) => write!(f, "duplicate uid {uid}"),
            Self::NonContiguousRegistrySlots => {
                write!(f, "registry_slot values are not contiguous 0..n")
            }
            Self::DuplicateFieldName { record, field } => {
                write!(f, "record `{record}` declares field `{field}` twice")
            }
            Self::FieldOutOfBounds { record, field } => {
                write!(f, "field `{record}.{field}` extends past the record words")
            }
            Self::OverlappingFields { record, field } => {
                write!(f, "field `{record}.{field}` overlaps another field")
            }
            Self::IncompleteTiling { record } => {
                write!(f, "record `{record}` leaves words uncovered")
            }
            Self::DefaultArityMismatch { record, field } => {
                write!(f, "field `{record}.{field}` default arity != type")
            }
            Self::DuplicateSurfaceFieldName(name) => {
                write!(f, "duplicate decoded surface field {name:?}")
            }
            Self::MissingValidityPredicate { record, field } => {
                write!(f, "field `{record}.{field}` has no validity predicate")
            }
            Self::InvalidValidityPredicate {
                record,
                field,
                expression,
                reason,
            } => write!(
                f,
                "field `{record}.{field}` validity {expression:?} is invalid: {reason}"
            ),
            Self::DefaultOutOfRange {
                record,
                field,
                value,
                expression,
            } => write!(
                f,
                "field `{record}.{field}` default {value} is outside validity {expression:?}"
            ),
            Self::FieldBitRegionMismatch { lobe, field } => {
                write!(f, "field `{lobe}.{field}` bit_region != lobe namespace")
            }
        }
    }
}

impl std::error::Error for SchemaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_schema_parses_and_validates() {
        let schema = surface_schema();
        assert_eq!(schema.schema_version, 3);
        assert_eq!(schema.core.words, 20);
        assert_eq!(schema.lobe_count(), 10);
        // The open-vocabulary glint lobe (§17.6) is a first-class member of the
        // same table as the specification core: it just lives in the extension
        // region with its own uid and a 4-word block.
        let glint = schema
            .lobes
            .iter()
            .find(|lobe| lobe.id == "glint")
            .expect("glint lobe declared");
        assert_eq!(glint.uid, "org.prism.lobe.glint");
        assert_eq!(glint.registry_slot, EXTENSION_SLOT_BASE + 1);
        assert_eq!(glint.mask_const, "GLINT");
        assert_eq!(glint.namespace, Namespace::Prism);
        assert_eq!(glint.words, 4);
        assert!(glint.requires_caps.is_empty());

        assert_eq!(schema.response_programs.len(), 3);
        let lit = schema
            .response_programs
            .iter()
            .find(|program| program.id == "lit")
            .expect("lit response program declared");
        assert_eq!(lit.uid, "org.prism.response_program.lit");
        assert_eq!(lit.registry_slot, 23);
        assert_eq!(lit.namespace, Namespace::Prism);
        assert_eq!(lit.stability, Stability::Candidate);
        assert!(lit.requires_caps.is_empty());
        assert!(lit.view_dependent);

        let wetness = schema
            .channels
            .iter()
            .find(|channel| channel.id == "wetness")
            .expect("wetness channel declared");
        assert_eq!(wetness.registry_slot, 19);
        assert_eq!(wetness.header_word, 0);
        assert_eq!(wetness.requires_caps, ["object_space_cache"]);
        assert!(!wetness.view_dependent);

        assert_eq!(schema.cache_aux.len(), 1);
        let surfel = &schema.cache_aux[0];
        assert_eq!(surfel.uid, "org.prism.cache_aux.surfel");
        assert_eq!(surfel.registry_slot, 26);
        assert_eq!(surfel.requires_caps, ["object_space_cache"]);
        assert!(!surfel.view_dependent);

        assert_eq!(schema.identity_extensions.len(), 5);
        let measured = schema
            .identity_extensions
            .iter()
            .find(|extension| extension.id == "measured_brdf")
            .expect("measured-BRDF provider declared");
        assert_eq!(measured.kind, IdentityKind::Provider);
        assert_eq!(measured.uid, "org.prism.provider.measured_brdf");
        assert_eq!(measured.requires_caps, ["heavy_sampling"]);
        assert!(measured.view_dependent);
    }

    #[test]
    fn specialization_has_no_implicit_layout_fallback() {
        let (prefix, rest) = SURFACE_SCHEMA_TOML
            .split_once("[specialization]\n")
            .unwrap();
        let (_, suffix) = rest.split_once("\n[core]").unwrap();
        let missing = format!("{prefix}\n[core]{suffix}");
        assert!(SurfaceSchema::parse(&missing).is_err());
        for declaration in [
            "storage_bits = 64\n",
            "illumination = { shift = 0, bits = 8 }\n",
            "lobe_mask = { shift = 8, bits = 32 }\n",
            "render_class = { shift = 40, bits = 8 }\n",
            "reserved = { shift = 48, bits = 16 }\n",
        ] {
            assert!(SURFACE_SCHEMA_TOML.contains(declaration));
            let incomplete = SURFACE_SCHEMA_TOML.replacen(declaration, "", 1);
            assert!(SurfaceSchema::parse(&incomplete).is_err(), "{declaration}");
        }
    }

    #[test]
    fn specialization_rejects_invalid_storage_ranges_and_overlaps() {
        let original = surface_schema().specialization;
        for width in [0, 65] {
            let mut layout = original.clone();
            layout.storage_bits = width;
            assert!(validate_specialization(&layout).is_err());
        }
        for range in [
            BitRange { shift: 0, bits: 0 },
            BitRange { shift: 63, bits: 2 },
        ] {
            let mut layout = original.clone();
            layout.reserved = range;
            assert!(validate_specialization(&layout).is_err());
        }
        let mut layout = original;
        layout.render_class.shift = layout.lobe_mask.shift;
        assert!(matches!(
            validate_specialization(&layout),
            Err(SchemaError::InvalidSpecializationLayout(message)) if message.contains("overlaps")
        ));
    }

    #[test]
    fn specialization_components_are_bounded_by_generated_u32_parts() {
        let mut schema = surface_schema();
        schema.specialization.illumination.bits = 33;
        assert!(matches!(
            schema.validate(),
            Err(SchemaError::InvalidSpecializationLayout(message))
                if message.contains("illumination") && message.contains("u32")
        ));
    }

    #[test]
    fn lobe_registry_slots_must_fit_the_specialization_lobe_axis() {
        let mut schema = surface_schema();
        schema.specialization.lobe_mask.bits = 1;
        assert!(matches!(
            schema.validate(),
            Err(SchemaError::InvalidSpecializationLayout(message))
                if message.contains("registry_slot") && message.contains("lobe_mask")
        ));
    }

    #[test]
    fn defaults_and_validity_ranges_are_executable_schema_contracts() {
        let schema = surface_schema();
        let mut semantic_fields = 0usize;
        let mut check = |record: &str, field: &Field| {
            if field.ty == FieldType::Pad {
                return;
            }
            semantic_fields += 1;
            let expression = field
                .valid
                .as_deref()
                .expect("every semantic field must declare a validity range");
            let range = ValidRange::parse(expression).expect("schema range must parse");
            assert!(
                field
                    .default
                    .components()
                    .into_iter()
                    .all(|value| range.contains(value)),
                "{record}.{} default is outside {expression}",
                field.name
            );
        };
        for field in &schema.core.fields {
            check("core", field);
        }
        for lobe in &schema.lobes {
            for field in &lobe.fields {
                check(&lobe.id, field);
            }
        }
        assert_eq!(semantic_fields, 53);

        for channel in &schema.channels {
            let expression = channel
                .valid
                .as_deref()
                .expect("every runtime channel must declare a validity range");
            let range = ValidRange::parse(expression).expect("channel range must parse");
            assert!(range.contains(channel.default.components()[0]));
        }
    }

    #[test]
    fn validity_range_parser_rejects_invalid_or_inverted_ranges() {
        let inclusive = ValidRange::parse("0.0..=1.0").unwrap();
        assert!(inclusive.contains(1.0));
        let exclusive = ValidRange::parse("0.0..1.0").unwrap();
        assert!(!exclusive.contains(1.0));
        assert!(ValidRange::parse("not-a-range").is_err());
        assert!(ValidRange::parse("2.0..=1.0").is_err());
    }

    #[test]
    fn schema_validation_rejects_missing_or_out_of_range_defaults() {
        let mut missing = surface_schema();
        missing.core.fields[0].valid = None;
        assert!(matches!(
            missing.validate(),
            Err(SchemaError::MissingValidityPredicate { .. })
        ));

        let mut outside = surface_schema();
        outside.core.fields[0].valid = Some("0.9..=1.0".into());
        assert!(matches!(
            outside.validate(),
            Err(SchemaError::DefaultOutOfRange { .. })
        ));
    }

    #[test]
    fn canonical_lobe_order_is_by_registry_slot() {
        let schema = surface_schema();
        let order: Vec<u32> = schema
            .canonical_lobes()
            .iter()
            .map(|l| l.registry_slot)
            .collect();
        // Canonical slots 0..6 then the extension region from 16 (§4.5
        // constraint 2): the private lobes are dense from the extension base.
        assert_eq!(order, vec![0, 1, 2, 3, 4, 5, 6, 16, 17, 18]);
    }

    #[test]
    fn packed_len_matches_core_plus_present_lobes() {
        let schema = surface_schema();
        assert_eq!(schema.packed_len_words(0), 20);
        // The lobe mask is two dense segments (§4.5 constraint 2), so "every
        // lobe" is the OR of the declared slots rather than a contiguous range.
        let all_mask: u32 = schema
            .lobes
            .iter()
            .fold(0u32, |acc, lobe| acc | (1 << lobe.registry_slot));
        let all: u32 = 20 + schema.lobes.iter().map(|lobe| lobe.words).sum::<u32>();
        assert_eq!(schema.packed_len_words(all_mask), all);
        // Only clearcoat (bit 1) present: core + the coat lobe's own footprint
        // (16 words: the OpenPBR coat group plus trailing pads).
        assert_eq!(schema.packed_len_words(0b10), 20 + 16);
        // Only sheen (bit 3) present: core + the fuzz lobe's own footprint
        // (8 words: sheen, fuzz_roughness, two pads, fuzz_color).
        assert_eq!(schema.packed_len_words(0b1000), 20 + 8);
    }

    #[test]
    fn transmission_fields_land_at_expected_offsets() {
        let schema = surface_schema();
        let all: u32 = schema
            .lobes
            .iter()
            .fold(0u32, |acc, lobe| acc | (1 << lobe.registry_slot));
        // Core (20) + emission (8, the OpenPBR emission group) + clearcoat (16)
        // + anisotropy (4) + sheen (8) + subsurface (16) = 72 is the transmission
        // base; the transmission lobe then packs its fields at 72..88.
        assert_eq!(
            schema.lobe_field_offset(all, "transmission", "transmission_weight"),
            Some(72)
        );
        assert_eq!(
            schema.lobe_field_offset(all, "transmission", "index_of_refraction"),
            Some(74)
        );
    }

    #[test]
    fn rejects_overlapping_fields() {
        let bad = r#"
schema_version = 1
abi_version = 4
[core]
words = 2
[[core.field]]
name = "a"
type = "f32"
bit_region = "KHR"
pack_slot = 0
default = 0.0
valid = "0.0..=1.0"
[[core.field]]
name = "b"
type = "f32"
bit_region = "KHR"
pack_slot = 0
default = 0.0
valid = "0.0..=1.0"
"#;
        assert!(matches!(
            SurfaceSchema::parse(&with_specialization(bad)),
            Err(SchemaError::OverlappingFields { .. })
        ));
    }

    #[test]
    fn rejects_incomplete_tiling() {
        let bad = r#"
schema_version = 1
abi_version = 4
[core]
words = 4
[[core.field]]
name = "a"
type = "f32"
bit_region = "KHR"
pack_slot = 0
default = 0.0
valid = "0.0..=1.0"
"#;
        assert!(matches!(
            SurfaceSchema::parse(&with_specialization(bad)),
            Err(SchemaError::IncompleteTiling { .. })
        ));
    }

    fn with_specialization(source: &str) -> String {
        let (_, rest) = SURFACE_SCHEMA_TOML
            .split_once("[specialization]\n")
            .unwrap();
        let (declaration, _) = rest.split_once("\n[core]").unwrap();
        source.replacen(
            "[core]",
            &format!("[specialization]\n{declaration}\n[core]"),
            1,
        )
    }

    fn schema_with_records(records: &str) -> String {
        with_specialization(&format!(
            r#"
schema_version = 1
abi_version = 5
[core]
words = 1
[[core.field]]
name = "value"
type = "f32"
bit_region = "KHR"
pack_slot = 0
default = 0.0
valid = "0.0..=1.0"
{records}
"#
        ))
    }

    #[test]
    fn rejects_an_empty_closure_name() {
        let bad = schema_with_records(
            r#"
[[lobe]]
id = "a"
uid = "org.khronos.openpbr.a"
closure = ""
registry_slot = 0
mask_const = "A"
namespace = "KHR"
vendor = "KHR"
stability = "stable"
words = 0
"#,
        );
        assert_eq!(
            SurfaceSchema::parse(&bad).unwrap_err(),
            SchemaError::EmptyClosureName(String::from("a"))
        );
    }

    #[test]
    fn lobes_can_share_a_closure_implementation() {
        let shared = schema_with_records(
            r#"
[[lobe]]
id = "a"
uid = "org.khronos.openpbr.a"
closure = "shared"
registry_slot = 0
mask_const = "A"
namespace = "KHR"
vendor = "KHR"
stability = "stable"
words = 0

[[lobe]]
id = "b"
uid = "org.khronos.openpbr.b"
closure = "shared"
registry_slot = 1
mask_const = "B"
namespace = "KHR"
vendor = "KHR"
stability = "stable"
words = 0
"#,
        );
        let schema = SurfaceSchema::parse(&shared).unwrap();
        assert_eq!(schema.lobes[0].closure, schema.lobes[1].closure);
    }

    #[test]
    fn rejects_two_response_programs_claiming_one_slot() {
        let mut schema = surface_schema();
        schema.response_programs[1].registry_slot = schema.response_programs[0].registry_slot;
        assert!(matches!(
            schema.validate(),
            Err(SchemaError::DuplicateHeaderSlot(_))
        ));
    }

    #[test]
    fn response_program_cannot_alias_a_lobe_slot() {
        let mut schema = surface_schema();
        schema.response_programs[0].registry_slot = schema.lobes[0].registry_slot;
        assert!(matches!(
            schema.validate(),
            Err(SchemaError::DuplicateHeaderSlot(_))
        ));
    }

    #[test]
    fn rejects_duplicate_builtin_programs() {
        let mut schema = surface_schema();
        schema.response_programs[1].builtin = schema.response_programs[0].builtin;
        assert!(matches!(
            schema.validate(),
            Err(SchemaError::ResponseContract(_))
        ));
    }

    #[test]
    fn adding_a_field_requires_an_explicit_response_declaration() {
        let mut schema = surface_schema();
        let mut field = schema.core.fields[0].clone();
        field.name = "new_surface_parameter".into();
        field.pack_slot = schema.core.words;
        field.responses.clear();
        schema.core.words += field.ty.words();
        schema.core.fields.push(field);
        assert!(matches!(
            schema.validate(),
            Err(SchemaError::ResponseContract(_))
        ));
    }

    #[test]
    fn deleting_a_mapping_or_its_consumer_is_rejected() {
        let mut schema = surface_schema();
        schema.core.fields[0].responses.remove("lit");
        assert!(matches!(
            schema.validate(),
            Err(SchemaError::ResponseContract(_))
        ));
        let mut schema = surface_schema();
        schema.core.fields[0]
            .consumed_by
            .retain(|s| *s != ConsumedStep::Lit);
        assert!(matches!(
            schema.validate(),
            Err(SchemaError::ResponseContract(_))
        ));
    }

    #[test]
    fn padding_is_not_a_semantic_response_input() {
        let schema = surface_schema();
        assert!(schema
            .response_fields()
            .iter()
            .all(|(_, f)| f.ty != FieldType::Pad));
    }

    #[test]
    fn rejects_an_unknown_identity_extension_kind() {
        let bad = schema_with_records(
            r#"
[[identity_extension]]
id = "watercolor"
uid = "org.prism.provider.watercolor"
kind = "watercolor"
namespace = "PRISM"
vendor = "prism"
stability = "experimental"
requires_caps = []
view_dependent = false
"#,
        );
        assert!(matches!(
            SurfaceSchema::parse(&bad),
            Err(SchemaError::Parse(_))
        ));
    }
}
