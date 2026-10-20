//! Canonical typed settings and explicit, validated schema upgrades.
//!
//! Settings use the reflection PRVB envelope and canonical PRB1 body. Policies
//! and migrations are trusted application code; byte/tree limits constrain
//! serialization and decoding, not arbitrary allocations made by callbacks.

use crate::EditableAsset;
use prism_reflect::schema::{validate, SchemaRegistry, TypeMetadata, TypeSchema};
use prism_reflect::{
    from_binary_with_limits, to_canonical_binary_with_schema, CanonicalError, CanonicalLimits,
    DeserializeError, DeserializeLimits, DynamicStruct, FromReflect, Reflect, ReflectRef, TypeInfo,
    TypeRegistry, Typed,
};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};

const MAGIC: &[u8; 4] = b"PRVB";
const ENVELOPE_BYTES: usize = 8;
const MAX_BYTES: usize = 1024 * 1024;
const MAX_DEPTH: usize = 64;
// The reflection decoder's dynamic map/set insertion is quadratic. Keep its
// per-collection admission substantially below the generic serializer limit.
const MAX_COLLECTION_LEN: usize = 4096;
const MAX_NODES: usize = 65_536;
const MAX_MIGRATIONS: u32 = 64;

/// An asset kind's concrete settings and semantic normalization policy.
///
/// Use a stable project-owned logical name across Rust type/schema renames.
/// `canonicalize` must be deterministic and idempotent, and fill defaults and
/// normalize only fields whose semantics require it (for example, paths).
/// `validate` checks kind-specific constraints after normalization; reflection
/// metadata ranges are also checked, including nested structs.
pub trait EditableSettings: Reflect + Typed + FromReflect {
    const LOGICAL_TYPE: &'static str;

    fn canonicalize(&mut self) -> Result<(), String>;
    fn validate(&self) -> Result<(), String>;
}

/// A settings operation failed before changing the asset's saved bytes.
#[derive(Debug)]
#[non_exhaustive]
pub enum EditableSettingsError {
    InvalidLimits,
    ByteLimit { actual: usize, limit: usize },
    BadEnvelope,
    Registration(String),
    FutureVersion { found: u32, current: u32 },
    UnknownVersion(u32),
    MigrationLimit { from: u32, current: u32 },
    Migration(String),
    Canonical(CanonicalError),
    Decode(DeserializeError),
    NonCanonical,
    Metadata { path: String, errors: String },
    Policy(String),
    CallbackPanic(&'static str),
    Reconstruction(&'static str),
}

impl fmt::Display for EditableSettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => {
                write!(f, "settings limits exceed policy or cannot hold a header")
            }
            Self::ByteLimit { actual, limit } => {
                write!(f, "settings bytes {actual} exceed {limit}")
            }
            Self::BadEnvelope => write!(f, "invalid settings PRVB envelope"),
            Self::Registration(error) => write!(f, "invalid settings registration: {error}"),
            Self::FutureVersion { found, current } => {
                write!(f, "settings version {found} is newer than {current}")
            }
            Self::UnknownVersion(version) => write!(f, "unknown settings version {version}"),
            Self::MigrationLimit { from, current } => write!(
                f,
                "settings migration {from} -> {current} exceeds {MAX_MIGRATIONS} steps"
            ),
            Self::Migration(error) => write!(f, "settings migration failed: {error}"),
            Self::Canonical(error) => error.fmt(f),
            Self::Decode(error) => error.fmt(f),
            Self::NonCanonical => write!(f, "settings are not canonical"),
            Self::Metadata { path, errors } => write!(f, "settings metadata at {path}: {errors}"),
            Self::Policy(error) => write!(f, "settings policy: {error}"),
            Self::CallbackPanic(stage) => write!(f, "settings callback panicked during {stage}"),
            Self::Reconstruction(name) => write!(f, "cannot reconstruct settings {name}"),
        }
    }
}

impl std::error::Error for EditableSettingsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Canonical(error) => Some(error),
            Self::Decode(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CanonicalError> for EditableSettingsError {
    fn from(error: CanonicalError) -> Self {
        Self::Canonical(error)
    }
}
impl From<DeserializeError> for EditableSettingsError {
    fn from(error: DeserializeError) -> Self {
        Self::Decode(error)
    }
}

/// An owned current value plus the exact bytes to save after an upgrade.
pub struct DecodedEditableSettings<T> {
    pub value: T,
    pub source_version: u32,
    pub current_version: u32,
    pub canonical_bytes: Vec<u8>,
}

/// Borrowed registries shared by authoring, migrations, and Cook.
///
/// Register every schema root and every nonprimitive child type. The same
/// logical name must resolve to the same concrete shape in both registries.
#[derive(Clone, Copy)]
pub struct EditableSettingsCodec<'a> {
    registry: &'a TypeRegistry,
    schemas: &'a SchemaRegistry,
    limits: CanonicalLimits,
}

impl<'a> EditableSettingsCodec<'a> {
    pub fn new(registry: &'a TypeRegistry, schemas: &'a SchemaRegistry) -> Self {
        Self {
            registry,
            schemas,
            limits: CanonicalLimits {
                max_bytes: MAX_BYTES,
                max_depth: MAX_DEPTH,
                max_collection_len: MAX_COLLECTION_LEN,
                max_nodes: MAX_NODES,
            },
        }
    }

    /// Tighten the settings limits. `max_bytes` includes the PRVB envelope.
    /// Limits cannot exceed 1 MiB, 64 levels, 4096 children, or 65536 nodes.
    pub fn with_limits(mut self, limits: CanonicalLimits) -> Result<Self, EditableSettingsError> {
        if !(ENVELOPE_BYTES + 13..=MAX_BYTES).contains(&limits.max_bytes)
            || !(1..=MAX_DEPTH).contains(&limits.max_depth)
            || limits.max_collection_len > MAX_COLLECTION_LEN
            || !(1..=MAX_NODES).contains(&limits.max_nodes)
        {
            return Err(EditableSettingsError::InvalidLimits);
        }
        self.limits = limits;
        Ok(self)
    }

    pub fn limits(&self) -> CanonicalLimits {
        self.limits
    }

    /// Consume and normalize a value without cloning its reflected tree.
    pub fn encode<T: EditableSettings>(
        &self,
        mut value: T,
    ) -> Result<Vec<u8>, EditableSettingsError> {
        let schema = self.current_schema::<T>()?;
        self.normalize(&mut value)?;
        self.encode_current(&value, schema)
    }

    /// Strictly read saved bytes and migrate through every registered version.
    ///
    /// Every step must produce the exact target shape and satisfy metadata.
    /// The final value is rebuilt as `T`, normalized, semantically validated,
    /// and serialized with its current schema. A current-version input must
    /// already equal those canonical bytes; old-version upgrades are explicit
    /// in the returned `canonical_bytes`, with no implicit write to an asset.
    pub fn decode<T: EditableSettings>(
        &self,
        bytes: &[u8],
    ) -> Result<DecodedEditableSettings<T>, EditableSettingsError> {
        let current = self.current_schema::<T>()?;
        if bytes.len() > self.limits.max_bytes {
            return Err(EditableSettingsError::ByteLimit {
                actual: bytes.len(),
                limit: self.limits.max_bytes,
            });
        }
        if bytes.len() < ENVELOPE_BYTES || &bytes[..4] != MAGIC {
            return Err(EditableSettingsError::BadEnvelope);
        }
        let source_version = u32::from_le_bytes(bytes[4..8].try_into().expect("checked envelope"));
        if source_version > current.version() {
            return Err(EditableSettingsError::FutureVersion {
                found: source_version,
                current: current.version(),
            });
        }
        if current.version() - source_version > MAX_MIGRATIONS {
            return Err(EditableSettingsError::MigrationLimit {
                from: source_version,
                current: current.version(),
            });
        }
        let source = self
            .schemas
            .schema(T::LOGICAL_TYPE, source_version)
            .ok_or(EditableSettingsError::UnknownVersion(source_version))?;
        self.check_schema(source)?;
        let mut value = self.decode_body(&bytes[ENVELOPE_BYTES..], source)?;
        if self.body(value.as_ref(), source)? != bytes[ENVELOPE_BYTES..] {
            return Err(EditableSettingsError::NonCanonical);
        }
        self.check_metadata(value.as_ref(), source)?;
        let chain = self
            .schemas
            .chain(T::LOGICAL_TYPE, source_version)
            .map_err(|e| EditableSettingsError::Migration(e.to_string()))?;
        for step in chain {
            let target = self
                .schemas
                .schema(T::LOGICAL_TYPE, step.to_version())
                .ok_or(EditableSettingsError::UnknownVersion(step.to_version()))?;
            self.check_schema(target)?;
            let dynamic = value
                .as_any_mut()
                .downcast_mut::<DynamicStruct>()
                .ok_or(EditableSettingsError::Reconstruction(source.type_name()))?;
            catch_unwind(AssertUnwindSafe(|| step.apply(dynamic)))
                .map_err(|_| EditableSettingsError::CallbackPanic("migration"))?
                .map_err(|e| EditableSettingsError::Migration(e.to_string()))?;
            // The schema-aware writer follows names throughout nested trees,
            // so migration insertion order cannot swap equal-typed fields.
            let body = self.body(value.as_ref(), target)?;
            value = self.decode_body(&body, target)?;
            self.check_metadata(value.as_ref(), target)?;
        }
        let mut value = T::from_reflect(value.as_ref())
            .ok_or(EditableSettingsError::Reconstruction(current.type_name()))?;
        self.normalize(&mut value)?;
        let canonical_bytes = self.encode_current(&value, current)?;
        if source_version == current.version() && canonical_bytes != bytes {
            return Err(EditableSettingsError::NonCanonical);
        }
        Ok(DecodedEditableSettings {
            value,
            source_version,
            current_version: current.version(),
            canonical_bytes,
        })
    }

    fn normalize<T: EditableSettings>(&self, value: &mut T) -> Result<(), EditableSettingsError> {
        catch_unwind(AssertUnwindSafe(|| value.canonicalize()))
            .map_err(|_| EditableSettingsError::CallbackPanic("normalization"))?
            .map_err(EditableSettingsError::Policy)?;
        catch_unwind(AssertUnwindSafe(|| value.validate()))
            .map_err(|_| EditableSettingsError::CallbackPanic("validation"))?
            .map_err(EditableSettingsError::Policy)
    }

    fn current_schema<T: EditableSettings>(&self) -> Result<&'a TypeSchema, EditableSettingsError> {
        if T::LOGICAL_TYPE.is_empty() || T::LOGICAL_TYPE.contains('\0') {
            return Err(EditableSettingsError::Registration(
                "empty or NUL logical type".into(),
            ));
        }
        let schema = self
            .schemas
            .current_schema(T::LOGICAL_TYPE)
            .ok_or_else(|| EditableSettingsError::Registration(T::LOGICAL_TYPE.into()))?;
        if schema.info() != <T as Typed>::type_info() {
            return Err(EditableSettingsError::Registration(format!(
                "{} is not the current concrete settings type",
                schema.type_name()
            )));
        }
        self.check_schema(schema)?;
        Ok(schema)
    }

    fn check_schema(&self, schema: &TypeSchema) -> Result<(), EditableSettingsError> {
        let registered = self
            .registry
            .get_with_name(schema.type_name())
            .ok_or_else(|| EditableSettingsError::Registration(schema.type_name().into()))?;
        if schema.type_name() != schema.info().type_name()
            || registered.type_info() != schema.info()
            || !matches!(schema.info(), TypeInfo::Struct(_))
        {
            return Err(EditableSettingsError::Registration(format!(
                "{} has inconsistent or non-struct schema",
                schema.type_name()
            )));
        }
        Ok(())
    }

    fn body(
        &self,
        value: &dyn Reflect,
        schema: &TypeSchema,
    ) -> Result<Vec<u8>, EditableSettingsError> {
        let limits = CanonicalLimits {
            max_bytes: self.limits.max_bytes - ENVELOPE_BYTES,
            ..self.limits
        };
        Ok(to_canonical_binary_with_schema(
            value,
            schema.info(),
            self.registry,
            limits,
        )?)
    }

    fn decode_body(
        &self,
        body: &[u8],
        schema: &TypeSchema,
    ) -> Result<Box<dyn Reflect>, EditableSettingsError> {
        Ok(from_binary_with_limits(
            body,
            self.registry,
            schema.info(),
            &DeserializeLimits::new(self.limits.max_depth, self.limits.max_collection_len),
        )?)
    }

    fn encode_current(
        &self,
        value: &dyn Reflect,
        schema: &TypeSchema,
    ) -> Result<Vec<u8>, EditableSettingsError> {
        let body = self.body(value, schema)?;
        self.check_metadata(value, schema)?;
        let mut bytes = Vec::with_capacity(ENVELOPE_BYTES + body.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&schema.version().to_le_bytes());
        bytes.extend_from_slice(&body);
        Ok(bytes)
    }

    fn check_metadata(
        &self,
        value: &dyn Reflect,
        schema: &TypeSchema,
    ) -> Result<(), EditableSettingsError> {
        let metadata = self
            .registry
            .get_with_name(schema.type_name())
            .and_then(|r| r.data::<TypeMetadata>());
        validate(value, schema, metadata).map_err(|errors| EditableSettingsError::Metadata {
            path: schema.type_name().into(),
            errors: format!("{errors:?}"),
        })?;
        self.check_nested_metadata(value)
    }

    // Called only after bounded canonical shape validation. No unbounded
    // visitor walks an arbitrary caller-supplied reflection implementation.
    fn check_nested_metadata(&self, value: &dyn Reflect) -> Result<(), EditableSettingsError> {
        match value.reflect_ref() {
            ReflectRef::Struct(s) => {
                if let Some(registration) = self.registry.get_with_name(value.type_name()) {
                    let shape = TypeSchema::new(value.type_name(), 0, registration.type_info());
                    validate(value, &shape, registration.data::<TypeMetadata>()).map_err(
                        |errors| EditableSettingsError::Metadata {
                            path: value.type_name().into(),
                            errors: format!("{errors:?}"),
                        },
                    )?;
                }
                for index in 0..s.field_count() {
                    self.check_nested_metadata(s.field_at(index).expect("validated shape"))?;
                }
            }
            ReflectRef::TupleStruct(s) => {
                for index in 0..s.field_count() {
                    self.check_nested_metadata(s.field(index).expect("validated shape"))?;
                }
            }
            ReflectRef::Enum(e) => {
                for index in 0..e.field_count() {
                    self.check_nested_metadata(e.field_at(index).expect("validated shape"))?;
                }
            }
            ReflectRef::List(list) => {
                for index in 0..list.len() {
                    self.check_nested_metadata(list.get(index).expect("validated shape"))?;
                }
            }
            ReflectRef::Array(array) => {
                for index in 0..array.len() {
                    self.check_nested_metadata(array.get(index).expect("validated shape"))?;
                }
            }
            ReflectRef::Map(map) => {
                for (key, value) in map.iter_reflect() {
                    self.check_nested_metadata(key)?;
                    self.check_nested_metadata(value)?;
                }
            }
            ReflectRef::Set(set) => {
                for element in set.iter_reflect() {
                    self.check_nested_metadata(element)?;
                }
            }
            ReflectRef::Value(_) => {}
            _ => {
                return Err(EditableSettingsError::Registration(format!(
                    "unsupported reflected settings kind {}",
                    value.type_name()
                )));
            }
        }
        Ok(())
    }
}

impl EditableAsset {
    /// Replace settings only after normalization, encoding, and validation.
    pub fn set_typed_settings<T: EditableSettings>(
        &mut self,
        value: T,
        codec: &EditableSettingsCodec<'_>,
    ) -> Result<(), EditableSettingsError> {
        let bytes = codec.encode(value)?;
        self.canonical_settings = bytes;
        Ok(())
    }

    pub fn typed_settings<T: EditableSettings>(
        &self,
        codec: &EditableSettingsCodec<'_>,
    ) -> Result<DecodedEditableSettings<T>, EditableSettingsError> {
        codec.decode(&self.canonical_settings)
    }

    /// Explicitly adopt current bytes; workspace persistence remains a save.
    pub fn migrate_typed_settings<T: EditableSettings>(
        &mut self,
        codec: &EditableSettingsCodec<'_>,
    ) -> Result<T, EditableSettingsError> {
        let decoded = self.typed_settings::<T>(codec)?;
        self.canonical_settings = decoded.canonical_bytes;
        Ok(decoded.value)
    }
}

/// Normalize an explicitly path-valued settings field to a relative `/` path.
///
/// Collapses repeated separators and `.`/`..`, rejects root escape, absolute
/// paths, drive/scheme colons, NUL, empty results, and inputs over 4096 bytes.
/// This is lexical normalization, not a filesystem containment check.
pub fn canonical_settings_path(path: &str) -> Result<String, EditableSettingsError> {
    if path.len() > 4096 || path.contains(['\0', ':']) {
        return Err(EditableSettingsError::Policy(
            "invalid relative settings path".into(),
        ));
    }
    let normalized = path.replace('\\', "/");
    if normalized.starts_with('/') {
        return Err(EditableSettingsError::Policy(
            "absolute settings path".into(),
        ));
    }
    let mut segments = Vec::new();
    for segment in normalized.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                if segments.pop().is_none() {
                    return Err(EditableSettingsError::Policy(
                        "settings path escapes root".into(),
                    ));
                }
            }
            name => segments.push(name),
        }
    }
    if segments.is_empty() {
        return Err(EditableSettingsError::Policy("empty settings path".into()));
    }
    Ok(segments.join("/"))
}
