//! Deterministic, schema-checked build profiles and declared field dependencies.
//!
//! Capability values are supplied by a caller/platform adapter. This module
//! performs no hardware probing, environment reads, or platform-name inference.
//! Hash protocol v1 uses BLAKE3, type tags, little-endian numbers and u64 byte
//! lengths. Names and strings are exact UTF-8 (no implicit Unicode folding).
//! Provenance is diagnostic only: equivalent effective values have equal keys.

use std::collections::{BTreeMap, BTreeSet};

/// A domain-separated profile, projection, field, or selection fingerprint.
pub type ProfileHash = [u8; 32];

/// The only supported layers, in increasing precedence.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(u8)]
pub enum ProfileLayer {
    /// Engine-wide defaults.
    EngineBaseline,
    /// Project defaults.
    ProjectDefaults,
    /// Explicit capabilities supplied by the target adapter.
    PlatformCapabilities,
    /// Device or quality tier policy.
    DeviceQualityTier,
    /// Development, shipping, or another build configuration.
    BuildConfiguration,
    /// Policy for a group of assets.
    AssetGroupOverride,
    /// Policy for one asset.
    AssetOverride,
}

impl ProfileLayer {
    /// Fixed merge order; callers cannot reorder precedence.
    pub const ALL: [Self; 7] = [
        Self::EngineBaseline,
        Self::ProjectDefaults,
        Self::PlatformCapabilities,
        Self::DeviceQualityTier,
        Self::BuildConfiguration,
        Self::AssetGroupOverride,
        Self::AssetOverride,
    ];
}

/// Explicit value types; numbers are never implicitly coerced.
#[derive(Clone, Debug, PartialEq)]
pub enum ProfileValue {
    /// A boolean.
    Bool(bool),
    /// A signed 64-bit integer.
    Integer(i64),
    /// An unsigned 64-bit integer.
    Unsigned(u64),
    /// A finite double. Negative zero is normalized to positive zero.
    Float(f64),
    /// An exact UTF-8 string.
    String(String),
    /// An unordered set of exact UTF-8 strings.
    StringSet(BTreeSet<String>),
}

impl ProfileValue {
    /// Constructs a set, removing duplicate members and input ordering.
    pub fn string_set(values: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::StringSet(values.into_iter().map(Into::into).collect())
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::Integer(_) => "integer",
            Self::Unsigned(_) => "unsigned",
            Self::Float(_) => "float",
            Self::String(_) => "string",
            Self::StringSet(_) => "string_set",
        }
    }

    fn normalized(&self) -> Self {
        match self {
            Self::Float(value) if *value == 0.0 => Self::Float(0.0),
            _ => self.clone(),
        }
    }
}

/// Field type and inclusive constraints. String lengths count UTF-8 bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum ProfileFieldType {
    /// Boolean with no coercions.
    Bool,
    /// Signed inclusive bounds.
    Integer { min: i64, max: i64 },
    /// Unsigned inclusive bounds.
    Unsigned { min: u64, max: u64 },
    /// Finite inclusive bounds; NaN and infinity are always illegal.
    Float { min: f64, max: f64 },
    /// Byte-length bounds and optional enumeration (Some(empty) permits none).
    String {
        min_bytes: usize,
        max_bytes: usize,
        allowed_values: Option<BTreeSet<String>>,
    },
    /// Cardinality bounds and optional member enumeration.
    StringSet {
        min_items: usize,
        max_items: usize,
        allowed_values: Option<BTreeSet<String>>,
    },
}

impl ProfileFieldType {
    fn kind(&self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Integer { .. } => "integer",
            Self::Unsigned { .. } => "unsigned",
            Self::Float { .. } => "float",
            Self::String { .. } => "string",
            Self::StringSet { .. } => "string_set",
        }
    }

    fn validate_schema(&self) -> bool {
        match self {
            Self::Bool => true,
            Self::Integer { min, max } => min <= max,
            Self::Unsigned { min, max } => min <= max,
            Self::Float { min, max } => min.is_finite() && max.is_finite() && min <= max,
            Self::String {
                min_bytes,
                max_bytes,
                allowed_values,
            } => {
                min_bytes <= max_bytes
                    && allowed_values.as_ref().is_none_or(|values| {
                        values
                            .iter()
                            .all(|v| (*min_bytes..=*max_bytes).contains(&v.len()))
                    })
            }
            Self::StringSet {
                min_items,
                max_items,
                ..
            } => min_items <= max_items,
        }
    }

    fn validate_value(&self, field: &str, value: &ProfileValue) -> Result<(), ProfileError> {
        if self.kind() != value.kind() {
            return Err(ProfileError::TypeMismatch {
                field: field.into(),
                expected: self.kind(),
                actual: value.kind(),
            });
        }
        let valid = match (self, value) {
            (Self::Bool, ProfileValue::Bool(_)) => true,
            (Self::Integer { min, max }, ProfileValue::Integer(value)) => {
                (*min..=*max).contains(value)
            }
            (Self::Unsigned { min, max }, ProfileValue::Unsigned(value)) => {
                (*min..=*max).contains(value)
            }
            (Self::Float { min, max }, ProfileValue::Float(value)) => {
                value.is_finite() && (*min..=*max).contains(value)
            }
            (
                Self::String {
                    min_bytes,
                    max_bytes,
                    allowed_values,
                },
                ProfileValue::String(value),
            ) => {
                (*min_bytes..=*max_bytes).contains(&value.len())
                    && allowed_values
                        .as_ref()
                        .is_none_or(|allowed| allowed.contains(value))
            }
            (
                Self::StringSet {
                    min_items,
                    max_items,
                    allowed_values,
                },
                ProfileValue::StringSet(value),
            ) => {
                (*min_items..=*max_items).contains(&value.len())
                    && allowed_values
                        .as_ref()
                        .is_none_or(|allowed| value.is_subset(allowed))
            }
            _ => false,
        };
        if valid {
            Ok(())
        } else {
            Err(ProfileError::ConstraintViolation {
                field: field.into(),
                value: value.clone(),
            })
        }
    }
}

/// Merge semantics declared by the schema, never by an individual override.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfileMergePolicy {
    /// The highest supplied layer wins. Legal only for scalar fields.
    ScalarOverride,
    /// Accumulate unique set members. Bounds apply to inputs AND merged results.
    SetUnion,
    /// The highest supplied set replaces all previous members, including empty.
    SetReplace,
}

/// Controls whether a later authorized layer may change an established value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfileOverridePolicy {
    /// Later authorized writes may change the value.
    Allow,
    /// Repeating the same value is legal; changing it is an error.
    ForbidChange,
}

/// One field's explicit contract. Defaults are supplied by an engine layer.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileField {
    field_type: ProfileFieldType,
    merge: ProfileMergePolicy,
    required: bool,
    allowed_layers: BTreeSet<ProfileLayer>,
    override_policy: ProfileOverridePolicy,
}

impl ProfileField {
    /// Declares a type and merge policy; the schema constructor checks legality.
    pub fn new(field_type: ProfileFieldType, merge: ProfileMergePolicy) -> Self {
        Self {
            field_type,
            merge,
            required: false,
            allowed_layers: ProfileLayer::ALL.into_iter().collect(),
            override_policy: ProfileOverridePolicy::Allow,
        }
    }

    /// Requires a value after all seven layers have been merged.
    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Restricts ALL writes, including first assignments and equal-value writes.
    pub fn allowed_layers(mut self, layers: impl IntoIterator<Item = ProfileLayer>) -> Self {
        self.allowed_layers = layers.into_iter().collect();
        self
    }

    /// Restricts subsequent changes independently of allowed writer layers.
    pub fn override_policy(mut self, policy: ProfileOverridePolicy) -> Self {
        self.override_policy = policy;
        self
    }

    /// The field's type and constraints.
    pub fn field_type(&self) -> &ProfileFieldType {
        &self.field_type
    }
    /// Declared merge semantics.
    pub fn merge_policy(&self) -> ProfileMergePolicy {
        self.merge
    }
    /// Whether the final value is required.
    pub fn is_required(&self) -> bool {
        self.required
    }
    /// Layers authorized to supply this field.
    pub fn writable_layers(&self) -> &BTreeSet<ProfileLayer> {
        &self.allowed_layers
    }
    /// Policy governing changes to a previously assigned value.
    pub fn change_policy(&self) -> ProfileOverridePolicy {
        self.override_policy
    }
}

/// Values from one layer; duplicate field assignments are rejected.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileLayerValues {
    layer: ProfileLayer,
    fields: BTreeMap<String, ProfileValue>,
}

impl ProfileLayerValues {
    /// Creates an empty layer (omitted layers have identical semantics).
    pub fn new(layer: ProfileLayer) -> Self {
        Self {
            layer,
            fields: BTreeMap::new(),
        }
    }

    /// Adds a field without silently overwriting an earlier assignment.
    pub fn set(
        &mut self,
        field: impl Into<String>,
        value: ProfileValue,
    ) -> Result<&mut Self, ProfileError> {
        let field = field.into();
        validate_name(&field)?;
        if self.fields.contains_key(&field) {
            return Err(ProfileError::DuplicateField(field));
        }
        self.fields.insert(field, value);
        Ok(self)
    }

    /// This layer's fixed precedence identity.
    pub fn layer(&self) -> ProfileLayer {
        self.layer
    }
    /// Explicitly supplied fields, in canonical name order.
    pub fn fields(&self) -> &BTreeMap<String, ProfileValue> {
        &self.fields
    }
}

/// Stable schema namespace and caller-managed semantic version.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfileSchemaIdentity {
    /// Exact schema namespace.
    pub id: String,
    /// Nonzero semantic version. A change invalidates every projection.
    pub version: u32,
}

/// Validated immutable schema. Unknown fields and duplicate layers are errors.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileSchema {
    identity: ProfileSchemaIdentity,
    fields: BTreeMap<String, ProfileField>,
}

impl ProfileSchema {
    /// Constructs and checks all types, bounds, policies, and unique field names.
    pub fn new<N, I>(id: impl Into<String>, version: u32, fields: I) -> Result<Self, ProfileError>
    where
        N: Into<String>,
        I: IntoIterator<Item = (N, ProfileField)>,
    {
        let id = id.into();
        validate_name(&id)?;
        if version == 0 {
            return Err(ProfileError::InvalidSchema(
                "schema version must be nonzero".into(),
            ));
        }
        let mut checked = BTreeMap::new();
        for (name, mut field) in fields {
            let name = name.into();
            validate_name(&name)?;
            if checked.contains_key(&name) {
                return Err(ProfileError::DuplicateField(name));
            }
            let is_set = matches!(field.field_type, ProfileFieldType::StringSet { .. });
            if !field.field_type.validate_schema()
                || is_set
                    != matches!(
                        field.merge,
                        ProfileMergePolicy::SetUnion | ProfileMergePolicy::SetReplace
                    )
                || field.allowed_layers.is_empty()
            {
                return Err(ProfileError::InvalidSchema(name));
            }
            if let ProfileFieldType::Float { min, max } = &mut field.field_type {
                if *min == 0.0 {
                    *min = 0.0;
                }
                if *max == 0.0 {
                    *max = 0.0;
                }
            }
            checked.insert(name, field);
        }
        Ok(Self {
            identity: ProfileSchemaIdentity { id, version },
            fields: checked,
        })
    }

    /// Schema namespace and version.
    pub fn identity(&self) -> &ProfileSchemaIdentity {
        &self.identity
    }
    /// Field contracts in canonical order.
    pub fn fields(&self) -> &BTreeMap<String, ProfileField> {
        &self.fields
    }

    /// Merges by fixed precedence, independently of caller ordering.
    ///
    /// Every assignment is checked even if a later value would hide it. A
    /// repeated layer is ambiguous and rejected, even when the layer is empty.
    pub fn resolve(&self, layers: &[ProfileLayerValues]) -> Result<EffectiveProfile, ProfileError> {
        let mut ordered = BTreeMap::new();
        for layer in layers {
            if ordered.insert(layer.layer, layer).is_some() {
                return Err(ProfileError::DuplicateLayer(layer.layer));
            }
        }
        let mut fields: BTreeMap<String, ResolvedField> = self
            .fields
            .iter()
            .map(|(name, definition)| {
                (
                    name.clone(),
                    ResolvedField {
                        definition: definition.clone(),
                        value: None,
                        history: Vec::new(),
                    },
                )
            })
            .collect();
        for (layer, input) in ordered {
            for (name, supplied) in &input.fields {
                let field = fields
                    .get_mut(name)
                    .ok_or_else(|| ProfileError::UnknownField(name.clone()))?;
                if !field.definition.allowed_layers.contains(&layer) {
                    return Err(ProfileError::IllegalLayer {
                        field: name.clone(),
                        layer,
                    });
                }
                field.definition.field_type.validate_value(name, supplied)?;
                let supplied = supplied.normalized();
                let result = match (&field.value, &supplied, field.definition.merge) {
                    (
                        Some(ProfileValue::StringSet(previous)),
                        ProfileValue::StringSet(next),
                        ProfileMergePolicy::SetUnion,
                    ) => ProfileValue::StringSet(previous.union(next).cloned().collect()),
                    _ => supplied.clone(),
                };
                field.definition.field_type.validate_value(name, &result)?;
                if field.definition.override_policy == ProfileOverridePolicy::ForbidChange
                    && field
                        .value
                        .as_ref()
                        .is_some_and(|previous| previous != &result)
                {
                    return Err(ProfileError::IllegalOverride {
                        field: name.clone(),
                        layer,
                    });
                }
                field.history.push(ProfileContribution {
                    layer,
                    supplied,
                    result: result.clone(),
                });
                field.value = Some(result);
            }
        }
        for (name, field) in &fields {
            if field.definition.required && field.value.is_none() {
                return Err(ProfileError::MissingRequiredField(name.clone()));
            }
        }
        Ok(EffectiveProfile {
            data: ProfileData {
                identity: self.identity.clone(),
                fields,
            },
        })
    }
}

/// An accepted assignment and its result, including no-op assignments.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileContribution {
    /// Writer layer.
    pub layer: ProfileLayer,
    /// Validated, normalized supplied value.
    pub supplied: ProfileValue,
    /// Effective value immediately after this assignment.
    pub result: ProfileValue,
}

#[derive(Clone, Debug, PartialEq)]
struct ResolvedField {
    definition: ProfileField,
    value: Option<ProfileValue>,
    history: Vec<ProfileContribution>,
}

#[derive(Clone, Debug, PartialEq)]
struct ProfileData {
    identity: ProfileSchemaIdentity,
    fields: BTreeMap<String, ResolvedField>,
}

impl ProfileData {
    fn field(&self, name: &str) -> Result<&ResolvedField, ProfileError> {
        self.fields
            .get(name)
            .ok_or_else(|| ProfileError::UnknownField(name.into()))
    }

    fn canonical_bytes(&self, domain: &[u8]) -> Vec<u8> {
        let mut bytes = domain.to_vec();
        put_bytes(&mut bytes, self.identity.id.as_bytes());
        bytes.extend_from_slice(&self.identity.version.to_le_bytes());
        put_len(&mut bytes, self.fields.len());
        for (name, field) in &self.fields {
            encode_field(&mut bytes, name, field);
        }
        bytes
    }

    fn field_hash(&self, name: &str) -> Result<ProfileHash, ProfileError> {
        let field = self.field(name)?;
        let mut data = Self {
            identity: self.identity.clone(),
            fields: BTreeMap::new(),
        };
        data.fields.insert(name.into(), field.clone());
        Ok(hash_bytes(
            &data.canonical_bytes(b"Prism.AssetBuild.ProfileField\0v1\0"),
        ))
    }

    fn project<I, S>(&self, fields: I) -> Result<ProfileProjection, ProfileError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut selected = BTreeMap::new();
        for name in fields {
            let name = name.as_ref();
            selected.insert(name.into(), self.field(name)?.clone());
        }
        Ok(ProfileProjection {
            data: Self {
                identity: self.identity.clone(),
                fields: selected,
            },
        })
    }

    fn difference(&self, other: &Self, domain: &[u8]) -> ProfileDifference {
        let names: BTreeSet<_> = self.fields.keys().chain(other.fields.keys()).collect();
        let mut fields = Vec::new();
        for name in names {
            let before = self.fields.get(name);
            let after = other.fields.get(name);
            if before.map(|f| (&f.definition, &f.value)) != after.map(|f| (&f.definition, &f.value))
            {
                fields.push(ProfileFieldDifference {
                    field: name.clone(),
                    before: before.and_then(|f| f.value.clone()),
                    after: after.and_then(|f| f.value.clone()),
                    before_definition: before.map(|f| f.definition.clone()),
                    after_definition: after.map(|f| f.definition.clone()),
                    before_history: before.map_or_else(Vec::new, |f| f.history.clone()),
                    after_history: after.map_or_else(Vec::new, |f| f.history.clone()),
                    before_hash: self.field_hash(name).ok(),
                    after_hash: other.field_hash(name).ok(),
                });
            }
        }
        ProfileDifference {
            before_hash: hash_bytes(&self.canonical_bytes(domain)),
            after_hash: hash_bytes(&other.canonical_bytes(domain)),
            before_schema: self.identity.clone(),
            after_schema: other.identity.clone(),
            fields,
        }
    }
}

const PROFILE_DOMAIN: &[u8] = b"Prism.AssetBuild.EffectiveProfile\0v1\0";
const PROJECTION_DOMAIN: &[u8] = b"Prism.AssetBuild.ProfileProjection\0v1\0";

/// A fully validated, immutable effective profile.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectiveProfile {
    data: ProfileData,
}

/// Backwards-compatible public name for the resolved build profile.
pub type BuildProfile = EffectiveProfile;

impl EffectiveProfile {
    /// Schema namespace and version used to resolve this profile.
    pub fn schema_identity(&self) -> &ProfileSchemaIdentity {
        &self.data.identity
    }
    /// Reads a field; absent optional fields return None, unknown names error.
    pub fn get(&self, field: &str) -> Result<Option<&ProfileValue>, ProfileError> {
        Ok(self.data.field(field)?.value.as_ref())
    }
    /// Returns the fixed-order accepted assignments for a field.
    pub fn explain(&self, field: &str) -> Result<&[ProfileContribution], ProfileError> {
        Ok(&self.data.field(field)?.history)
    }
    /// Canonical full-profile protocol bytes, excluding assignment provenance.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.data.canonical_bytes(PROFILE_DOMAIN)
    }
    /// Fingerprint of all declared fields, including absent optional fields.
    pub fn hash(&self) -> ProfileHash {
        hash_bytes(&self.canonical_bytes())
    }
    /// Fingerprint of one field's name, contract, presence, value and schema identity.
    pub fn field_hash(&self, field: &str) -> Result<ProfileHash, ProfileError> {
        self.data.field_hash(field)
    }
    /// Selects declared dependencies; order and duplicate names have no effect.
    /// Unselected fields and their contracts never enter the projection hash.
    pub fn project<I, S>(&self, fields: I) -> Result<ProfileProjection, ProfileError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.data.project(fields)
    }
    /// Compares semantic fields and includes before/after merge explanations.
    pub fn differences(&self, other: &Self) -> ProfileDifference {
        self.data.difference(&other.data, PROFILE_DOMAIN)
    }
    /// Starts a reader recording every successful read, including absent values.
    pub fn reader(&self) -> ProfileReader<'_> {
        ProfileReader {
            profile: self,
            reads: BTreeSet::new(),
        }
    }

    /// Evaluates explicit capability requirements, without probing the platform.
    /// Highest priority wins; equal priorities use ascending candidate ID.
    /// All candidates are validated/evaluated, including losing alternatives.
    pub fn select_capability(
        &self,
        candidates: &[CapabilityCandidate],
    ) -> Result<CapabilitySelection, ProfileError> {
        let mut ids = BTreeSet::new();
        let mut dependencies = BTreeSet::new();
        let mut ordered: Vec<_> = candidates.iter().collect();
        ordered.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.id.cmp(&b.id)));
        let mut evaluations = Vec::new();
        for candidate in ordered {
            validate_name(&candidate.id)?;
            if !ids.insert(&candidate.id) {
                return Err(ProfileError::DuplicateCandidate(candidate.id.clone()));
            }
            let mut requirements: Vec<_> = candidate.requirements.iter().collect();
            requirements.sort_by_key(|requirement| requirement.canonical_bytes());
            requirements.dedup_by(|a, b| a.canonical_bytes() == b.canonical_bytes());
            let mut checks = Vec::new();
            for requirement in requirements {
                let name = requirement.field();
                let field = self.data.field(name)?;
                dependencies.insert(name);
                let satisfied = requirement.evaluate(name, field)?;
                checks.push(CapabilityRequirementEvaluation {
                    requirement: requirement.clone(),
                    actual: field.value.clone(),
                    satisfied,
                });
            }
            evaluations.push(CapabilityCandidateEvaluation {
                id: candidate.id.clone(),
                priority: candidate.priority,
                requirements: checks,
            });
        }
        let selected = evaluations
            .iter()
            .find(|candidate| candidate.is_supported())
            .map(|c| c.id.clone());
        Ok(CapabilitySelection {
            selected,
            evaluations,
            dependency_projection: self.project(dependencies)?,
        })
    }
}

/// Immutable subset suitable for a processor's ArtifactSpec.profile_hash.
/// The caller must declare every semantic profile dependency it consumes.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileProjection {
    data: ProfileData,
}

impl ProfileProjection {
    /// Schema namespace and version used by this projection.
    pub fn schema_identity(&self) -> &ProfileSchemaIdentity {
        &self.data.identity
    }

    /// Selected fields in canonical order, including declared absent fields.
    pub fn fields(&self) -> impl Iterator<Item = &str> {
        self.data.fields.keys().map(String::as_str)
    }
    /// Reads a selected field; undeclared dependencies are errors.
    pub fn get(&self, field: &str) -> Result<Option<&ProfileValue>, ProfileError> {
        Ok(self.data.field(field)?.value.as_ref())
    }
    /// Merge history for a selected field.
    pub fn explain(&self, field: &str) -> Result<&[ProfileContribution], ProfileError> {
        Ok(&self.data.field(field)?.history)
    }
    /// Canonical bytes including schema identity and only selected contracts/values.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.data.canonical_bytes(PROJECTION_DOMAIN)
    }
    /// Fingerprint for fine-grained cache keys.
    pub fn hash(&self) -> ProfileHash {
        hash_bytes(&self.canonical_bytes())
    }
    /// Single-field fingerprint, identical to the effective profile's field hash.
    pub fn field_hash(&self, field: &str) -> Result<ProfileHash, ProfileError> {
        self.data.field_hash(field)
    }
    /// Explains semantic changes within the selected dependencies.
    pub fn differences(&self, other: &Self) -> ProfileDifference {
        self.data.difference(&other.data, PROJECTION_DOMAIN)
    }
}

/// Scoped read recorder. It records only reads made through this object.
pub struct ProfileReader<'a> {
    profile: &'a EffectiveProfile,
    reads: BTreeSet<String>,
}

impl<'a> ProfileReader<'a> {
    /// Reads and records a declared field, including an absent optional value.
    pub fn get(&mut self, field: &str) -> Result<Option<&'a ProfileValue>, ProfileError> {
        let value = self.profile.get(field)?;
        self.reads.insert(field.into());
        Ok(value)
    }
    /// Consumes the reader and freezes all recorded dependencies.
    pub fn finish(self) -> ProfileProjection {
        // Names enter reads only after successful schema lookup.
        let fields = self
            .reads
            .into_iter()
            .map(|name| {
                let field = self.profile.data.fields[&name].clone();
                (name, field)
            })
            .collect();
        ProfileProjection {
            data: ProfileData {
                identity: self.profile.data.identity.clone(),
                fields,
            },
        }
    }
}

/// Explainable full-profile or projection comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileDifference {
    /// Previous semantic fingerprint.
    pub before_hash: ProfileHash,
    /// New semantic fingerprint.
    pub after_hash: ProfileHash,
    /// Previous schema namespace/version.
    pub before_schema: ProfileSchemaIdentity,
    /// New schema namespace/version.
    pub after_schema: ProfileSchemaIdentity,
    /// Sorted changed fields. Provenance-only changes are omitted.
    pub fields: Vec<ProfileFieldDifference>,
}

impl ProfileDifference {
    /// Whether this dependency set needs a different cache key.
    pub fn invalidates_cache(&self) -> bool {
        self.before_hash != self.after_hash
    }
    /// Global namespace/version changes invalidate even an empty projection.
    pub fn schema_identity_changed(&self) -> bool {
        self.before_schema != self.after_schema
    }
}

/// Before/after values, contracts, hashes and provenance for a changed field.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileFieldDifference {
    /// Field name.
    pub field: String,
    /// Old value (None also covers absence of the field in a projection).
    pub before: Option<ProfileValue>,
    /// New value.
    pub after: Option<ProfileValue>,
    /// Old contract; distinguishes unselected/removed fields from absent values.
    pub before_definition: Option<ProfileField>,
    /// New contract.
    pub after_definition: Option<ProfileField>,
    /// Previous accepted assignments.
    pub before_history: Vec<ProfileContribution>,
    /// New accepted assignments.
    pub after_history: Vec<ProfileContribution>,
    /// Old field hash, or None if not declared/selected.
    pub before_hash: Option<ProfileHash>,
    /// New field hash, or None if not declared/selected.
    pub after_hash: Option<ProfileHash>,
}

/// A typed predicate over one explicitly supplied capability field.
#[derive(Clone, Debug, PartialEq)]
pub enum CapabilityRequirement {
    /// Requires a present, true boolean.
    Enabled(String),
    /// Requires exact equality, with no numeric conversions.
    Equals { field: String, value: ProfileValue },
    /// Requires a string member in a set.
    SetContains { field: String, member: String },
    /// Requires a present unsigned value at least this large.
    MinimumUnsigned { field: String, minimum: u64 },
}

impl CapabilityRequirement {
    /// Field that this predicate reads.
    pub fn field(&self) -> &str {
        match self {
            Self::Enabled(field)
            | Self::Equals { field, .. }
            | Self::SetContains { field, .. }
            | Self::MinimumUnsigned { field, .. } => field,
        }
    }

    fn evaluate(&self, name: &str, field: &ResolvedField) -> Result<bool, ProfileError> {
        let expected = match self {
            Self::Enabled(_) => "bool",
            Self::Equals { value, .. } => value.kind(),
            Self::SetContains { .. } => "string_set",
            Self::MinimumUnsigned { .. } => "unsigned",
        };
        if field.definition.field_type.kind() != expected {
            return Err(ProfileError::TypeMismatch {
                field: name.into(),
                expected,
                actual: field.definition.field_type.kind(),
            });
        }
        if let Self::Equals { value, .. } = self {
            field.definition.field_type.validate_value(name, value)?;
        }
        Ok(match (self, &field.value) {
            (Self::Enabled(_), Some(ProfileValue::Bool(value))) => *value,
            (Self::Equals { value, .. }, Some(actual)) => value == actual,
            (Self::SetContains { member, .. }, Some(ProfileValue::StringSet(value))) => {
                value.contains(member)
            }
            (Self::MinimumUnsigned { minimum, .. }, Some(ProfileValue::Unsigned(value))) => {
                value >= minimum
            }
            _ => false,
        })
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        put_bytes(&mut bytes, self.field().as_bytes());
        match self {
            Self::Enabled(_) => bytes.push(0),
            Self::Equals { value, .. } => {
                bytes.push(1);
                encode_value(&mut bytes, value);
            }
            Self::SetContains { member, .. } => {
                bytes.push(2);
                put_bytes(&mut bytes, member.as_bytes());
            }
            Self::MinimumUnsigned { minimum, .. } => {
                bytes.push(3);
                bytes.extend_from_slice(&minimum.to_le_bytes());
            }
        }
        bytes
    }
}

/// An implementation alternative; candidates without requirements are fallbacks.
#[derive(Clone, Debug, PartialEq)]
pub struct CapabilityCandidate {
    /// Unique stable implementation ID, with no platform-name semantics.
    pub id: String,
    /// Higher wins; ties use ascending ID (input order is irrelevant).
    pub priority: i32,
    /// Conjunction of requirements. Order and duplicates have no semantic effect.
    pub requirements: Vec<CapabilityRequirement>,
}

/// Explanation for one tested predicate.
#[derive(Clone, Debug, PartialEq)]
pub struct CapabilityRequirementEvaluation {
    /// Predicate as declared by the candidate.
    pub requirement: CapabilityRequirement,
    /// Effective supplied value, or absence.
    pub actual: Option<ProfileValue>,
    /// Whether this predicate accepted the value.
    pub satisfied: bool,
}

/// Explanation for one alternative, in deterministic selection order.
#[derive(Clone, Debug, PartialEq)]
pub struct CapabilityCandidateEvaluation {
    /// Stable implementation ID.
    pub id: String,
    /// Declared priority.
    pub priority: i32,
    /// Every predicate's result, including failures.
    pub requirements: Vec<CapabilityRequirementEvaluation>,
}

impl CapabilityCandidateEvaluation {
    /// Whether all requirements are satisfied (true for an explicit fallback).
    pub fn is_supported(&self) -> bool {
        self.requirements.iter().all(|r| r.satisfied)
    }
}

/// Capability choice, explanations, and all potential profile dependencies.
#[derive(Clone, Debug, PartialEq)]
pub struct CapabilitySelection {
    /// Winning ID; None explicitly means there is no supported alternative.
    pub selected: Option<String>,
    /// Every alternative, sorted by descending priority then ascending ID.
    pub evaluations: Vec<CapabilityCandidateEvaluation>,
    /// Union of ALL candidate reads, including rejected candidates. Cache users
    /// must include these fields so a newly supported candidate can invalidate.
    pub dependency_projection: ProfileProjection,
}

impl CapabilitySelection {
    /// Canonical choice key material: candidate rules, priority, winner and reads.
    /// Candidate-definition changes are therefore included as well as values.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"Prism.AssetBuild.CapabilitySelection\0v1\0".to_vec();
        put_len(&mut bytes, self.evaluations.len());
        for candidate in &self.evaluations {
            put_bytes(&mut bytes, candidate.id.as_bytes());
            bytes.extend_from_slice(&candidate.priority.to_le_bytes());
            put_len(&mut bytes, candidate.requirements.len());
            for requirement in &candidate.requirements {
                put_bytes(&mut bytes, &requirement.requirement.canonical_bytes());
            }
        }
        match &self.selected {
            None => bytes.push(0),
            Some(id) => {
                bytes.push(1);
                put_bytes(&mut bytes, id.as_bytes());
            }
        }
        put_bytes(&mut bytes, &self.dependency_projection.canonical_bytes());
        bytes
    }
    /// Fingerprint covering selection rules AND relevant effective profile data.
    pub fn hash(&self) -> ProfileHash {
        hash_bytes(&self.canonical_bytes())
    }
}

/// Structured configuration/selection failures, with no partial profile output.
#[derive(Clone, Debug, PartialEq)]
pub enum ProfileError {
    /// Empty names or names containing whitespace/control characters are illegal.
    InvalidName(String),
    /// Invalid bounds, incompatible merge policy, empty writers, or zero version.
    InvalidSchema(String),
    /// Duplicate schema field or duplicate assignment within a layer.
    DuplicateField(String),
    /// The same precedence slot was supplied more than once.
    DuplicateLayer(ProfileLayer),
    /// Undeclared field or dependency name.
    UnknownField(String),
    /// A field is missing after the full merge.
    MissingRequiredField(String),
    /// Value/predicate type does not match its contract.
    TypeMismatch {
        field: String,
        expected: &'static str,
        actual: &'static str,
    },
    /// A typed value violates bounds, enumeration, cardinality, or finiteness.
    ConstraintViolation { field: String, value: ProfileValue },
    /// A layer is not permitted to write a field, even if the value is unchanged.
    IllegalLayer { field: String, layer: ProfileLayer },
    /// A later layer tried to change an immutable field.
    IllegalOverride { field: String, layer: ProfileLayer },
    /// Two capability alternatives use the same ID.
    DuplicateCandidate(String),
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidName(name) => write!(f, "invalid profile name {name:?}"),
            Self::InvalidSchema(field) => write!(f, "invalid profile schema: {field}"),
            Self::DuplicateField(field) => write!(f, "duplicate profile field: {field}"),
            Self::DuplicateLayer(layer) => write!(f, "duplicate profile layer: {layer:?}"),
            Self::UnknownField(field) => write!(f, "unknown profile field: {field}"),
            Self::MissingRequiredField(field) => {
                write!(f, "missing required profile field: {field}")
            }
            Self::TypeMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "profile field {field}: expected {expected}, got {actual}"
            ),
            Self::ConstraintViolation { field, value } => {
                write!(f, "profile field {field} violates constraints: {value:?}")
            }
            Self::IllegalLayer { field, layer } => {
                write!(f, "profile layer {layer:?} cannot write {field}")
            }
            Self::IllegalOverride { field, layer } => write!(
                f,
                "profile layer {layer:?} cannot change immutable field {field}"
            ),
            Self::DuplicateCandidate(id) => write!(f, "duplicate capability candidate: {id}"),
        }
    }
}

impl std::error::Error for ProfileError {}

fn validate_name(name: &str) -> Result<(), ProfileError> {
    if name.is_empty() || name.chars().any(|c| c.is_whitespace() || c.is_control()) {
        Err(ProfileError::InvalidName(name.into()))
    } else {
        Ok(())
    }
}

fn hash_bytes(bytes: &[u8]) -> ProfileHash {
    *blake3::hash(bytes).as_bytes()
}

fn put_len(bytes: &mut Vec<u8>, len: usize) {
    // Rust targets have pointer widths <= 64 bits; allocated lengths fit u64.
    bytes.extend_from_slice(&(len as u64).to_le_bytes());
}

fn put_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    put_len(bytes, value.len());
    bytes.extend_from_slice(value);
}

fn encode_strings(bytes: &mut Vec<u8>, values: &BTreeSet<String>) {
    put_len(bytes, values.len());
    for value in values {
        put_bytes(bytes, value.as_bytes());
    }
}

fn encode_allowed(bytes: &mut Vec<u8>, values: &Option<BTreeSet<String>>) {
    match values {
        None => bytes.push(0),
        Some(values) => {
            bytes.push(1);
            encode_strings(bytes, values);
        }
    }
}

fn encode_float(bytes: &mut Vec<u8>, value: f64) {
    let value = if value == 0.0 { 0.0 } else { value };
    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn encode_value(bytes: &mut Vec<u8>, value: &ProfileValue) {
    match value {
        ProfileValue::Bool(value) => {
            bytes.push(0);
            bytes.push(u8::from(*value));
        }
        ProfileValue::Integer(value) => {
            bytes.push(1);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        ProfileValue::Unsigned(value) => {
            bytes.push(2);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        ProfileValue::Float(value) => {
            bytes.push(3);
            encode_float(bytes, *value);
        }
        ProfileValue::String(value) => {
            bytes.push(4);
            put_bytes(bytes, value.as_bytes());
        }
        ProfileValue::StringSet(values) => {
            bytes.push(5);
            encode_strings(bytes, values);
        }
    }
}

fn encode_type(bytes: &mut Vec<u8>, field_type: &ProfileFieldType) {
    match field_type {
        ProfileFieldType::Bool => bytes.push(0),
        ProfileFieldType::Integer { min, max } => {
            bytes.push(1);
            bytes.extend_from_slice(&min.to_le_bytes());
            bytes.extend_from_slice(&max.to_le_bytes());
        }
        ProfileFieldType::Unsigned { min, max } => {
            bytes.push(2);
            bytes.extend_from_slice(&min.to_le_bytes());
            bytes.extend_from_slice(&max.to_le_bytes());
        }
        ProfileFieldType::Float { min, max } => {
            bytes.push(3);
            encode_float(bytes, *min);
            encode_float(bytes, *max);
        }
        ProfileFieldType::String {
            min_bytes,
            max_bytes,
            allowed_values,
        } => {
            bytes.push(4);
            put_len(bytes, *min_bytes);
            put_len(bytes, *max_bytes);
            encode_allowed(bytes, allowed_values);
        }
        ProfileFieldType::StringSet {
            min_items,
            max_items,
            allowed_values,
        } => {
            bytes.push(5);
            put_len(bytes, *min_items);
            put_len(bytes, *max_items);
            encode_allowed(bytes, allowed_values);
        }
    }
}

fn encode_field(bytes: &mut Vec<u8>, name: &str, field: &ResolvedField) {
    put_bytes(bytes, name.as_bytes());
    let definition = &field.definition;
    encode_type(bytes, &definition.field_type);
    bytes.push(match definition.merge {
        ProfileMergePolicy::ScalarOverride => 0,
        ProfileMergePolicy::SetUnion => 1,
        ProfileMergePolicy::SetReplace => 2,
    });
    bytes.push(u8::from(definition.required));
    put_len(bytes, definition.allowed_layers.len());
    for layer in &definition.allowed_layers {
        bytes.push(*layer as u8);
    }
    bytes.push(match definition.override_policy {
        ProfileOverridePolicy::Allow => 0,
        ProfileOverridePolicy::ForbidChange => 1,
    });
    match &field.value {
        None => bytes.push(0),
        Some(value) => {
            bytes.push(1);
            encode_value(bytes, value);
        }
    }
}
