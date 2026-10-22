//! Core identifiers, schema definitions, validation metadata, and values.
use alloc::{collections::BTreeMap, string::String, vec::Vec};
use core::fmt;

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
/// Stable registry-local enum identifier; persistent codecs should store a stable string instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EnumId(pub u32);
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

impl Value {
    /// Returns the variant type without allocation.
    pub const fn value_type(&self) -> ValueType {
        match self {
            Self::Bool(_) => ValueType::Bool,
            Self::Int(_) => ValueType::Int,
            Self::Float(_) => ValueType::Float,
            Self::String(_) => ValueType::String,
            Self::Enum(_) => ValueType::Enum,
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
}

/// Scalar and collection limits validated by the core.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Constraints {
    /// Inclusive numeric lower bound.
    pub min: Option<f64>,
    /// Inclusive numeric upper bound.
    pub max: Option<f64>,
    /// Maximum UTF-8 byte length (M0 avoids a regex engine dependency).
    pub max_string_bytes: Option<usize>,
    /// Maximum array element count.
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
    /// Value variant required by the schema.
    pub ty: ValueType,
    /// Validation limits.
    pub constraints: Constraints,
    /// Violation behavior for scalar bounds.
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
        Self {
            key: key.clone(),
            namespace: namespace.into(),
            description: None,
            category: None,
            default,
            ty,
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
}
