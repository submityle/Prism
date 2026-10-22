//! Core identifiers, schema definitions, validation metadata, and values.
use alloc::{boxed::Box, collections::BTreeMap, string::String, vec::Vec};
use core::fmt;

/// Conservative default element cap for arrays whose definition is inferred
/// from a non-empty default value.
pub const DEFAULT_MAX_ARRAY_LEN: usize = 4096;

/// Maximum number of nested array shapes accepted by a settings schema.
///
/// This bounds validation stack use for schemas assembled from untrusted or
/// generated metadata while remaining far beyond practical configuration
/// nesting requirements.
pub const MAX_VALUE_SHAPE_DEPTH: usize = 32;

/// Stable, dot-separated persisted key (for example `audio.master_volume`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SettingKey(String);

impl SettingKey {
    /// Creates a key. Syntax is checked when its definition is registered.
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }
    /// Returns the canonical key text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for SettingKey {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}
impl From<String> for SettingKey {
    fn from(value: String) -> Self {
        Self(value)
    }
}
impl fmt::Display for SettingKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Dense registry-local setting identifier assigned during [`RegistryBuilder::finish`](crate::RegistryBuilder::finish).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SettingId(pub u32);
/// Registry-local namespace identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NamespaceId(pub u32);
/// Stable registry-local enum identifier derived from an enum option's name.
///
/// Persistent codecs resolve this ID to the option's stable string name using
/// the setting schema; the numeric ID is never a persisted ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EnumId(pub u32);

impl EnumId {
    /// Derives a deterministic runtime ID from a stable enum/variant key.
    ///
    /// IDs use FNV-1a over UTF-8 bytes and are independent of declaration
    /// order, process hash seeds, and platform endianness. Callers must reject
    /// collisions among the options registered for one setting.
    pub fn from_stable_name(name: &str) -> Self {
        let mut hash = 0x811c_9dc5u32;
        for byte in name.as_bytes() {
            hash ^= u32::from(*byte);
            hash = hash.wrapping_mul(0x0100_0193);
        }
        Self(hash)
    }
}

/// Stable identity of an enum type, independent of its declaration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EnumTypeId(pub u32);

impl EnumTypeId {
    /// Derives a deterministic runtime ID from a stable enum type name.
    pub fn from_stable_name(name: &str) -> Self {
        let mut hash = 0x811c_9dc5u32;
        for byte in name.as_bytes() {
            hash ^= u32::from(*byte);
            hash = hash.wrapping_mul(0x0100_0193);
        }
        Self(hash)
    }
}

/// Stable schema for one enum type and all of its variants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumSchema {
    /// Stable enum type identity.
    pub id: EnumTypeId,
    /// Stable persisted enum type name.
    pub stable_name: String,
    /// All variants in declaration-independent stable order.
    pub variants: Vec<EnumVariantSchema>,
}

/// Stable schema for one enum variant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumVariantSchema {
    /// Runtime variant identity.
    pub id: EnumId,
    /// Stable persisted variant name.
    pub stable_name: String,
    /// Payload shape for tuple/struct variants; `None` means unit.
    pub payload: Option<EnumPayloadShape>,
}

/// A stable, user-facing name paired with its deterministic runtime enum ID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumOption {
    /// Runtime identity used in [`Value::Enum`].
    pub id: EnumId,
    /// Stable persisted name; never a declaration ordinal.
    pub stable_name: String,
}

impl EnumOption {
    /// Derives an option ID from its fully-qualified stable name.
    pub fn from_stable_name(stable_name: impl Into<String>) -> Self {
        let stable_name = stable_name.into();
        let id = EnumId::from_stable_name(&stable_name);
        Self { id, stable_name }
    }
}

/// Recursive schema for the payload of one non-unit enum option.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnumPayloadShape {
    /// Positional tuple fields, validated in declaration order.
    Tuple(Vec<ValueShape>),
    /// Named struct fields, requiring exactly the declared keys.
    Struct(BTreeMap<String, ValueShape>),
}

/// Payload schema associated with a stable enum option identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumOptionPayload {
    /// Stable option identity. Must also appear in `SettingDef::enum_options`.
    pub id: EnumId,
    /// Payload layout required by this option.
    pub shape: EnumPayloadShape,
}

/// Runtime payload data for a non-unit enum option.
#[derive(Clone, Debug, PartialEq)]
pub enum EnumPayload {
    /// Positional tuple fields.
    Tuple(Vec<Value>),
    /// Named struct fields.
    Struct(BTreeMap<String, Value>),
}
/// Stable identity for byte payloads carried by a [`Value`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BytesRef(pub u64);
/// Opaque secret handle. Secret material is never represented in the core value tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SecretRef(pub u64);

/// Closed, deterministic setting value tree.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Boolean value.
    Bool(bool),
    /// Signed integer value.
    Int(i64),
    /// Floating point value; non-finite values are rejected by validation.
    Float(f64),
    /// UTF-8 string value.
    String(String),
    /// Registry-local enum identity.
    Enum(EnumId),
    /// Type-qualified enum identity.
    EnumTyped {
        /// Enum type identity.
        enum_type: EnumTypeId,
        /// Variant identity.
        id: EnumId,
    },
    /// Registry-local enum identity and its tuple or struct payload.
    EnumPayload {
        /// Stable enum option identity.
        id: EnumId,
        /// Variant payload.
        payload: crate::EnumPayload,
    },
    /// Type-qualified enum identity and payload.
    EnumPayloadTyped {
        /// Enum type identity.
        enum_type: EnumTypeId,
        /// Variant identity.
        id: EnumId,
        /// Variant payload.
        payload: crate::EnumPayload,
    },
    /// Ordered sequence.
    Array(Vec<Value>),
    /// Key-sorted structured value.
    Table(BTreeMap<String, Value>),
    /// Opaque bytes reference.
    Bytes(BytesRef),
    /// Secret handle, never secret material.
    Secret(SecretRef),
}

/// Runtime type discriminator for [`Value`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueType {
    /// Boolean.
    Bool,
    /// Integer.
    Int,
    /// Float.
    Float,
    /// String.
    String,
    /// Enum.
    Enum,
    /// Array.
    Array,
    /// Table.
    Table,
    /// Bytes reference.
    Bytes,
    /// Secret reference.
    Secret,
}

/// Recursive schema shape for a setting value.
///
/// Scalar leaves carry their canonical [`ValueType`]. Arrays and dynamic maps
/// are homogeneous and bounded. Fixed tables describe an exact set of named
/// fields, which makes reflected structs and persisted table values fail
/// closed on missing or unexpected keys.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueShape {
    /// A scalar leaf. `Array` and `Table` are not valid scalar leaves.
    Scalar(ValueType),
    /// A type-qualified enum leaf.
    Enum(EnumTypeId),
    /// A homogeneous, recursively shaped array with inclusive length bounds.
    Array {
        /// Shape required for every element.
        element: Box<ValueShape>,
        /// Inclusive minimum element count.
        min_len: usize,
        /// Inclusive maximum element count.
        max_len: usize,
    },
    /// A fixed set of named fields with recursively validated shapes.
    Table(BTreeMap<String, ValueShape>),
    /// A string-keyed map with a homogeneous value shape and bounded entries.
    Map {
        /// Shape required for every map value.
        value: Box<ValueShape>,
        /// Maximum number of entries.
        max_len: usize,
    },
    /// A bounded set stored as a canonically sorted array with unique values.
    Set {
        /// Shape required for every set member.
        element: Box<ValueShape>,
        /// Maximum number of members.
        max_len: usize,
    },
}

impl ValueShape {
    /// Creates a scalar shape.
    #[must_use]
    pub const fn scalar(value_type: ValueType) -> Self {
        Self::Scalar(value_type)
    }

    /// Creates a bounded homogeneous array shape.
    #[must_use]
    pub fn array(element: Self, min_len: usize, max_len: usize) -> Self {
        Self::Array {
            element: Box::new(element),
            min_len,
            max_len,
        }
    }

    /// Creates an array shape with no required elements and a finite maximum.
    #[must_use]
    pub fn bounded_array(element: Self, max_len: usize) -> Self {
        Self::array(element, 0, max_len)
    }

    /// Creates an exact-length homogeneous array shape.
    #[must_use]
    pub fn fixed_array(element: Self, len: usize) -> Self {
        Self::array(element, len, len)
    }

    /// Creates a fixed table whose keys and value shapes are declared by schema.
    #[must_use]
    pub fn table(fields: BTreeMap<String, ValueShape>) -> Self {
        Self::Table(fields)
    }

    /// Creates a string-keyed map with homogeneous bounded values.
    #[must_use]
    pub fn map(value: Self, max_len: usize) -> Self {
        Self::Map {
            value: Box::new(value),
            max_len,
        }
    }

    /// Creates a bounded set shape. The serialized value is a sorted array.
    #[must_use]
    pub fn set(element: Self, max_len: usize) -> Self {
        Self::Set {
            element: Box::new(element),
            max_len,
        }
    }

    /// Infers a conservative exact shape from a non-container value tree.
    ///
    /// An empty array has no element type and therefore returns `None`; its
    /// schema must be declared explicitly with [`Self::array`].
    #[must_use]
    pub fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Array(values) => {
                let first = values.first()?;
                let element = Self::from_value(first)?;
                if values
                    .iter()
                    .skip(1)
                    .any(|value| Self::from_value(value).as_ref() != Some(&element))
                {
                    return None;
                }
                Some(Self::bounded_array(element, DEFAULT_MAX_ARRAY_LEN))
            }
            Value::Table(values) => values
                .iter()
                .map(|(key, value)| Some((key.clone(), Self::from_value(value)?)))
                .collect::<Option<BTreeMap<_, _>>>()
                .map(Self::Table),
            Value::EnumTyped { enum_type, .. } | Value::EnumPayloadTyped { enum_type, .. } => {
                Some(Self::Enum(*enum_type))
            }
            _ => Some(Self::scalar(value.value_type())),
        }
    }

    /// Returns the root [`ValueType`] represented by this shape.
    #[must_use]
    pub const fn value_type(&self) -> ValueType {
        match self {
            Self::Scalar(value_type) => *value_type,
            Self::Enum(_) => ValueType::Enum,
            Self::Array { .. } => ValueType::Array,
            Self::Table(_) | Self::Map { .. } => ValueType::Table,
            Self::Set { .. } => ValueType::Array,
        }
    }

    /// Returns whether this shape is supported and internally consistent.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.is_valid_at_depth(0)
    }

    fn is_valid_at_depth(&self, depth: usize) -> bool {
        if depth > MAX_VALUE_SHAPE_DEPTH {
            return false;
        }
        match self {
            Self::Scalar(value_type) => !matches!(value_type, ValueType::Array | ValueType::Table),
            Self::Enum(_) => true,
            Self::Array {
                element,
                min_len,
                max_len,
            } => min_len <= max_len && element.is_valid_at_depth(depth + 1),
            Self::Table(fields) => fields
                .values()
                .all(|shape| shape.is_valid_at_depth(depth + 1)),
            Self::Map { value, .. } => value.is_valid_at_depth(depth + 1),
            Self::Set { element, .. } => element.is_valid_at_depth(depth + 1),
        }
    }

    /// Returns whether a value satisfies this shape without applying scalar
    /// range or enum-option constraints.
    #[must_use]
    pub fn matches(&self, value: &Value) -> bool {
        self.matches_at_depth(value, 0)
    }

    fn matches_at_depth(&self, value: &Value, depth: usize) -> bool {
        if depth > MAX_VALUE_SHAPE_DEPTH {
            return false;
        }
        match self {
            Self::Scalar(value_type) => value.value_type() == *value_type,
            Self::Enum(enum_type) => matches!(
                value,
                Value::EnumTyped { enum_type: actual, .. }
                    | Value::EnumPayloadTyped { enum_type: actual, .. }
                    if actual == enum_type
            ),
            Self::Array {
                element,
                min_len,
                max_len,
            } => match value {
                Value::Array(values) if (*min_len..=*max_len).contains(&values.len()) => values
                    .iter()
                    .all(|value| element.matches_at_depth(value, depth + 1)),
                _ => false,
            },
            Self::Table(fields) => match value {
                Value::Table(values) if values.len() == fields.len() => {
                    fields.iter().all(|(key, shape)| {
                        values
                            .get(key)
                            .is_some_and(|value| shape.matches_at_depth(value, depth + 1))
                    })
                }
                _ => false,
            },
            Self::Map {
                value: shape,
                max_len,
            } => match value {
                Value::Table(values) if values.len() <= *max_len => values
                    .values()
                    .all(|value| shape.matches_at_depth(value, depth + 1)),
                _ => false,
            },
            Self::Set {
                element: shape,
                max_len,
            } => match value {
                Value::Array(values) if values.len() <= *max_len => {
                    values.windows(2).all(|pair| pair[0] != pair[1])
                        && values
                            .iter()
                            .all(|value| shape.matches_at_depth(value, depth + 1))
                }
                _ => false,
            },
        }
    }
}

#[cfg(test)]
mod value_shape_tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn nested_shapes_are_bounded_and_match_fails_closed_past_the_limit() {
        let mut shape = ValueShape::scalar(ValueType::Int);
        let mut value = Value::Int(1);
        for _ in 0..=MAX_VALUE_SHAPE_DEPTH {
            shape = ValueShape::bounded_array(shape, 1);
            value = Value::Array(vec![value]);
        }

        assert!(!shape.is_valid());
        assert!(!shape.matches(&value));
    }
}

impl Value {
    /// Returns the variant type without allocation.
    pub const fn value_type(&self) -> ValueType {
        match self {
            Self::Bool(_) => ValueType::Bool,
            Self::Int(_) => ValueType::Int,
            Self::Float(_) => ValueType::Float,
            Self::String(_) => ValueType::String,
            Self::Enum(_) => ValueType::Enum,
            Self::EnumTyped { .. } => ValueType::Enum,
            Self::EnumPayload { .. } => ValueType::Enum,
            Self::EnumPayloadTyped { .. } => ValueType::Enum,
            Self::Array(_) => ValueType::Array,
            Self::Table(_) => ValueType::Table,
            Self::Bytes(_) => ValueType::Bytes,
            Self::Secret(_) => ValueType::Secret,
        }
    }
    /// True only for `Bool(true)`; used by the built-in cross-setting rules.
    pub const fn is_enabled(&self) -> bool {
        matches!(self, Self::Bool(true))
    }
}

/// Product lifecycle domain to which a definition is visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Domain {
    /// Runtime/game application.
    Runtime,
    /// Editor preferences and project editing.
    Editor,
    /// Machine/installation settings.
    System,
    /// Headless build and CI.
    Build,
}

/// Domain visibility bit set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DomainMask(u8);
impl DomainMask {
    /// No visible domains.
    pub const NONE: Self = Self(0);
    /// Runtime domain.
    pub const RUNTIME: Self = Self(1);
    /// Editor domain.
    pub const EDITOR: Self = Self(2);
    /// System domain.
    pub const SYSTEM: Self = Self(4);
    /// Build domain.
    pub const BUILD: Self = Self(8);
    /// All four design domains.
    pub const ALL: Self = Self(15);
    /// Combines two masks.
    pub const fn union(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
    /// Whether this mask includes a domain.
    pub const fn contains(self, domain: Domain) -> bool {
        self.0
            & match domain {
                Domain::Runtime => 1,
                Domain::Editor => 2,
                Domain::System => 4,
                Domain::Build => 8,
            }
            != 0
    }
    /// Whether this mask is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Layer identity. Domain-specific legal stacks are exposed by [`Domain::layers`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayerKind {
    /// Compiled engine baseline.
    EngineDefault,
    /// Editor baseline.
    EditorDefault,
    /// Project configuration.
    Project,
    /// Platform configuration.
    Platform,
    /// Administrator recommendation.
    Recommended,
    /// Remote cached authority.
    Remote,
    /// User preference.
    User,
    /// Machine-level setting.
    Machine,
    /// Device-specific setting.
    Device,
    /// Editor user setting.
    EditorUser,
    /// Editor-local machine setting.
    EditorMachine,
    /// Project-local untracked override.
    ProjectLocal,
    /// Build profile.
    BuildProfile,
    /// Session override.
    Session,
    /// Runtime console/temporary override.
    Runtime,
    /// Play-in-editor runtime overlay.
    PieRuntime,
    /// Enforced policy.
    Mandatory,
}

impl Domain {
    /// Returns the design-defined low-to-high stack for this domain.
    pub const fn layers(self) -> &'static [LayerKind] {
        use LayerKind::*;
        match self {
            Self::Runtime => &[
                EngineDefault,
                Project,
                Platform,
                Recommended,
                Remote,
                User,
                Device,
                Session,
                Runtime,
                Mandatory,
            ],
            Self::Editor => &[
                EngineDefault,
                EditorDefault,
                Machine,
                Recommended,
                EditorUser,
                Project,
                ProjectLocal,
                Session,
                PieRuntime,
                Mandatory,
            ],
            Self::System => &[EngineDefault, Machine, Recommended, Mandatory],
            Self::Build => &[
                EngineDefault,
                Project,
                BuildProfile,
                Recommended,
                Session,
                Mandatory,
            ],
        }
    }
}

/// Trust assigned to a candidate source. It is metadata, not an OS sandbox.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    /// Engine or verified policy source.
    Trusted,
    /// User-owned configuration.
    User,
    /// Project/VCS data, never executable.
    Project,
    /// Unknown/downloaded source with restricted behavior.
    Restricted,
}
/// Default write scope metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Machine-wide.
    Machine,
    /// Per user.
    User,
    /// Per project.
    Project,
}
/// Granularity at which a value may be overridden.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverrideScope {
    /// One global value.
    Global,
    /// Machine-specific.
    Machine,
    /// Project-specific.
    Project,
    /// Resource sidecar, reserved for later milestone.
    Resource,
    /// Language sidecar, reserved for later milestone.
    Language,
}
/// Ordering policy for selected source classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecedenceProfile {
    /// User wins over remote authority.
    UserChoiceWins,
    /// Remote authority wins over user choice.
    ServerWins,
    /// Device-specific value wins over user preference.
    DeviceScoped,
    /// Only default and mandatory values participate.
    Locked,
}
/// Whether an invalid value is clamped or rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViolationPolicy {
    /// Clamp supported scalar bounds.
    Clamp,
    /// Reject the candidate.
    Reject,
}

/// Definition flags used by the M0 resolver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct DefFlags {
    /// Candidate should be eligible during bootstrap.
    pub bootstrap: bool,
    /// Key represents an action/capability and must only come from Trusted sources.
    pub execution_class: bool,
    /// Ordinary non-policy writes are not permitted.
    pub read_only: bool,
    /// Project scope may override the setting.
    pub project_overridable: bool,
    /// A non-default value must be supplied by at least one accepted source.
    pub required: bool,
}

/// Scalar and collection limits validated by the core.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Constraints {
    /// Inclusive lower bound for numeric scalar values and homogeneous array leaves.
    pub min: Option<f64>,
    /// Inclusive upper bound for numeric scalar values and homogeneous array leaves.
    pub max: Option<f64>,
    /// Maximum UTF-8 byte length (M0 avoids a regex engine dependency).
    pub max_string_bytes: Option<usize>,
    /// Maximum element count enforced at every array depth.
    pub max_array_len: Option<usize>,
    /// Permitted enum identities; empty means unrestricted.
    pub enum_values: &'static [EnumId],
}
/// Cross-setting rule evaluated on effective values.
#[derive(Clone, Debug, PartialEq)]
pub enum CrossRule {
    /// If this definition is true, the referenced key must also be true.
    Requires {
        /// Required setting key.
        key: SettingKey,
        /// Whether this definition must equal true or false to activate the rule.
        when: bool,
    },
    /// If this definition is true, referenced key must be false.
    Conflicts {
        /// Conflicting setting key.
        key: SettingKey,
        /// Whether this definition must equal true or false to activate the rule.
        when: bool,
    },
}

/// Complete M0 schema definition. IDs are assigned at registry freeze.
#[derive(Clone, Debug, PartialEq)]
pub struct SettingDef {
    /// Canonical stable key.
    pub key: SettingKey,
    /// Owning namespace prefix, such as `render` or `plugin.example`.
    pub namespace: String,
    /// Human-readable description inherited from schema documentation.
    pub description: Option<String>,
    /// Optional presentation category for inspectors and console tooling.
    pub category: Option<String>,
    /// Default value.
    pub default: Value,
    /// Named stable options when the value type is `Enum`.
    pub enum_options: Vec<EnumOption>,
    /// Tuple/struct payload schemas for non-unit enum options. Options omitted
    /// here remain unit variants, preserving the original enum API.
    pub enum_payloads: Vec<EnumOptionPayload>,
    /// Type-qualified enum catalogs used by typed enum values.
    pub enum_schemas: Vec<EnumSchema>,
    /// Value variant required by the schema.
    pub ty: ValueType,
    /// Recursive value shape, including homogeneous array element and length
    /// constraints. `ty` is retained as the inexpensive root discriminator.
    pub shape: ValueShape,
    /// Validation limits.
    pub constraints: Constraints,
    /// Violation behavior for numeric bounds, including numeric array leaves.
    pub on_violation: ViolationPolicy,
    /// Domains where the key is visible.
    pub domains: DomainMask,
    /// Default persistence target metadata.
    pub scope: Scope,
    /// Override granularity metadata.
    pub override_scope: OverrideScope,
    /// Resolution precedence.
    pub precedence: PrecedenceProfile,
    /// M0 behavior flags.
    pub flags: DefFlags,
    /// Cross-setting invariants.
    pub cross_rules: Vec<CrossRule>,
    /// Alternative stable keys.
    pub aliases: Vec<SettingKey>,
}

impl SettingDef {
    /// Starts a definition with conservative defaults and inferred value type.
    pub fn new(key: impl Into<SettingKey>, namespace: impl Into<String>, default: Value) -> Self {
        let key = key.into();
        let ty = default.value_type();
        let shape = ValueShape::from_value(&default).unwrap_or_else(|| ValueShape::scalar(ty));
        Self {
            key: key.clone(),
            namespace: namespace.into(),
            description: None,
            category: None,
            default,
            enum_options: Vec::new(),
            enum_payloads: Vec::new(),
            enum_schemas: Vec::new(),
            ty,
            shape,
            constraints: Constraints::default(),
            on_violation: ViolationPolicy::Reject,
            domains: DomainMask::ALL,
            scope: Scope::User,
            override_scope: OverrideScope::Global,
            precedence: PrecedenceProfile::UserChoiceWins,
            flags: DefFlags {
                project_overridable: !key.as_str().starts_with("editor.")
                    && !key.as_str().starts_with("system."),
                ..DefFlags::default()
            },
            cross_rules: Vec::new(),
            aliases: Vec::new(),
        }
    }

    /// Replaces this definition's recursive shape and synchronizes its root
    /// [`ValueType`].
    #[must_use]
    pub fn with_shape(mut self, shape: ValueShape) -> Self {
        self.set_shape(shape);
        self
    }

    /// Replaces this definition's recursive shape and synchronizes its root
    /// [`ValueType`].
    pub fn set_shape(&mut self, shape: ValueShape) {
        self.ty = shape.value_type();
        self.shape = shape;
    }
}
