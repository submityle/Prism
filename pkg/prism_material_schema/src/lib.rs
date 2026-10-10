//! Self-describing PCM material schema — the single declarative source of truth
//! for the über-BSDF parameter layout (design doc §17.5).
//!
//! # Why
//!
//! The lobe field layout, bit masks, specialization keys, defaults and valid
//! ranges are currently hand-written in six places: Rust (`surface.rs` /
//! `record.rs`), the WESL twins (`material_unpack` / `material_classification`
//! / `material_sample`), the CPU golden in `prism_render_shading`, the doc
//! field tables (§4.4), and the compliance enum (§4.5). Adding one lobe means
//! editing all six, and any drift is a silent CPU/GPU parity or compliance bug.
//!
//! This crate holds the declarative schema and the deterministic, pure layout
//! derivation that a future codegen (stages S2–S4) will consume to emit all six
//! products. In the current landing stage **S1** (§17.5.8) the schema is only
//! *declared and parity-checked* against the hand-written layout — nothing is
//! generated yet. The S1 acceptance gate lives in `prism_render_material` and
//! asserts, byte-for-byte, that the schema-derived layout equals the authored
//! layout in `surface.rs` for every lobe subset.
//!
//! # Design boundary (§17.5.5)
//!
//! The schema owns *layout and contract* (field name/type/bit-region/pack
//! offset/specialization subfield/consumed-by step/default/valid predicate/
//! energy policy/view dependence). It deliberately does **not** express
//! behavioural math; that stays in typed hooks. The schema is not Turing
//! complete by design.

#![forbid(unsafe_code)]

extern crate alloc;

use alloc::collections::BTreeSet;
use core::fmt;

use serde::Deserialize;

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
    /// The always-present über-BSDF core record.
    pub core: Record,
    /// Optional lobes, declared in any order (`registry_slot` fixes identity
    /// and pack order, not declaration order).
    #[serde(default, rename = "lobe")]
    pub lobes: Vec<Lobe>,
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

/// An optional lobe: a uniform 4-word quantum plus its manifest identity.
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
    /// Number of `u32` words the lobe occupies (currently always 4).
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
    /// Which `SpecializationId` subfield this contributes to (§8.1).
    #[serde(default)]
    pub spec_subfield: SpecSubfield,
    /// Which pipeline steps consume this field (§8.5).
    #[serde(default)]
    pub consumed_by: Vec<ConsumedStep>,
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

/// Stability level of a capability (§5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stability {
    /// Frozen; covered by compliance parity.
    Stable,
    /// Usable but may change before stabilization.
    Experimental,
    /// Early draft, no compatibility promise.
    Draft,
    /// Superseded; retained only for bit-region stability.
    Deprecated,
}

/// Which `SpecializationId` subfield a field contributes to (§8.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecSubfield {
    /// Contributes to no specialization subfield.
    #[default]
    None,
    /// Contributes to the open lobe closure mask.
    ClosureMask,
    /// Contributes to the illumination axis.
    Illumination,
    /// Contributes to the render-class axis.
    RenderClass,
}

/// A pipeline step that consumes a field (§8.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsumedStep {
    /// Word-heap unpack (`material_unpack`).
    Unpack,
    /// Response-mode classification (`material_classification`).
    Classify,
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
        let schema: Self = toml::from_str(toml_src).map_err(|e| SchemaError::Parse(e.to_string()))?;
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
    pub fn lobe_field_offset(&self, mask_bits: u32, lobe_id: &str, field_name: &str) -> Option<u32> {
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
        validate_record("core", &self.core)?;

        let mut slots = BTreeSet::new();
        let mut uids = BTreeSet::new();
        for lobe in &self.lobes {
            if !slots.insert(lobe.registry_slot) {
                return Err(SchemaError::DuplicateRegistrySlot(lobe.registry_slot));
            }
            if !uids.insert(lobe.uid.clone()) {
                return Err(SchemaError::DuplicateUid(lobe.uid.clone()));
            }
            validate_record(&lobe.id, &record_view(lobe))?;
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

        // `registry_slot` values must be contiguous `0..lobe_count` so the mask
        // bit layout has no reserved holes at S1.
        for expected in 0..self.lobe_count() {
            if !slots.contains(&expected) {
                return Err(SchemaError::NonContiguousRegistrySlots);
            }
        }
        Ok(())
    }
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
    Ok(())
}

/// An error produced while parsing or validating a schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    /// The TOML failed to deserialize into the schema model.
    Parse(String),
    /// Two lobes declared the same `registry_slot`.
    DuplicateRegistrySlot(u32),
    /// Two lobes declared the same `uid`.
    DuplicateUid(String),
    /// `registry_slot` values are not a contiguous `0..n` range.
    NonContiguousRegistrySlots,
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
        assert_eq!(schema.schema_version, 1);
        assert_eq!(schema.core.words, 12);
        assert_eq!(schema.lobe_count(), 7);
    }

    #[test]
    fn canonical_lobe_order_is_by_registry_slot() {
        let schema = surface_schema();
        let order: Vec<u32> = schema
            .canonical_lobes()
            .iter()
            .map(|l| l.registry_slot)
            .collect();
        assert_eq!(order, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn packed_len_matches_core_plus_present_lobes() {
        let schema = surface_schema();
        assert_eq!(schema.packed_len_words(0), 12);
        assert_eq!(schema.packed_len_words(0b111_1111), 12 + 7 * 4);
        // Only clearcoat (bit 1) present: core + one lobe.
        assert_eq!(schema.packed_len_words(0b10), 16);
    }

    #[test]
    fn transmission_fields_land_at_expected_offsets() {
        let schema = surface_schema();
        let all = 0b111_1111;
        // Core (12) + emission (4) + clearcoat (4) + anisotropy (4)
        // + sheen (4) + subsurface (4) = 32 is the transmission base.
        assert_eq!(
            schema.lobe_field_offset(all, "transmission", "transmission"),
            Some(32)
        );
        assert_eq!(
            schema.lobe_field_offset(all, "transmission", "index_of_refraction"),
            Some(34)
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
[[core.field]]
name = "b"
type = "f32"
bit_region = "KHR"
pack_slot = 0
default = 0.0
"#;
        assert!(matches!(
            SurfaceSchema::parse(bad),
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
"#;
        assert!(matches!(
            SurfaceSchema::parse(bad),
            Err(SchemaError::IncompleteTiling { .. })
        ));
    }
}
