//! Host-owned typed state migration for native module reloads.
//!
//! The native ABI remains byte-oriented. This layer gives those bytes a host
//! meaning without placing Rust types, trait objects, allocators, or drop glue
//! across the ABI: each schema version supplies a host decoder/encoder and the
//! migration chain operates on `DynamicStruct` values owned by the host.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use prism_reflect::schema::{validate, TypeMetadata, TypeSchema};
use prism_reflect::{
    from_binary_with_limits, to_binary, DeserializeLimits, DynamicStruct, FromReflect,
    GetTypeRegistration, Reflect, TypeInfo, TypeRegistry, Typed,
};

/// A host-side failure while decoding, validating, migrating, or re-encoding
/// reflected module state. The native candidate is not activated on any error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReflectStateError {
    message: String,
}

impl ReflectStateError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ReflectStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ReflectStateError {}

type Decode = Arc<dyn Fn(&[u8]) -> Result<Box<dyn Reflect>, String> + Send + Sync>;
type Rebuild = Arc<dyn Fn(&dyn Reflect) -> Result<Box<dyn Reflect>, String> + Send + Sync>;
type Encode = Arc<dyn Fn(&dyn Reflect) -> Result<Vec<u8>, String> + Send + Sync>;
type Step = Arc<dyn Fn(&mut DynamicStruct) -> Result<(), String> + Send + Sync>;

struct Version {
    schema: TypeSchema,
    info: &'static TypeInfo,
    metadata: Option<TypeMetadata>,
    decode: Decode,
    rebuild: Rebuild,
    encode: Encode,
}

/// A complete host contract for one logical module state type.
///
/// Versions are keyed independently from the module ABI's opaque schema
/// numbers. A contract must register every source and candidate version needed
/// by a reload, plus consecutive migration steps. The highest registered
/// version is not implicitly selected: reload asks for the candidate's exact
/// schema, so a future module cannot accidentally migrate through a newer
/// host-only version.
pub struct ReflectStateContract {
    logical: String,
    registry: TypeRegistry,
    versions: BTreeMap<u64, Version>,
    migrations: BTreeMap<u64, Step>,
    limits: DeserializeLimits,
}

impl ReflectStateContract {
    /// Start a contract for a logical state name.
    #[must_use]
    pub fn new(logical: impl Into<String>) -> Self {
        Self {
            logical: logical.into(),
            registry: TypeRegistry::new(),
            versions: BTreeMap::new(),
            migrations: BTreeMap::new(),
            limits: DeserializeLimits::DEFAULT,
        }
    }

    /// Change the bounds applied to the host's reflected binary round trips.
    #[must_use]
    pub fn with_limits(mut self, limits: DeserializeLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Register a nested reflected type needed by a version's binary shape.
    pub fn register_type<T>(&mut self)
    where
        T: GetTypeRegistration,
    {
        self.registry.register::<T>();
    }

    /// Register a schema version with typed host transport adapters.
    ///
    /// `decode` and `encode` translate the module's stable ABI payload to and
    /// from a host-owned `T`; the reflection serializer then verifies the
    /// typed shape before migration. `T` itself never crosses the ABI.
    pub fn register<T, D, E>(
        &mut self,
        version: u64,
        decode: D,
        encode: E,
    ) -> Result<(), ReflectStateError>
    where
        T: Reflect + Typed + FromReflect + GetTypeRegistration + 'static,
        D: Fn(&[u8]) -> Result<T, String> + Send + Sync + 'static,
        E: Fn(&T) -> Result<Vec<u8>, String> + Send + Sync + 'static,
    {
        self.register_with_metadata::<T, D, E>(version, decode, encode, None)
    }

    /// Register a schema version and its optional reflect validation metadata.
    pub fn register_with_metadata<T, D, E>(
        &mut self,
        version: u64,
        decode: D,
        encode: E,
        metadata: Option<TypeMetadata>,
    ) -> Result<(), ReflectStateError>
    where
        T: Reflect + Typed + FromReflect + GetTypeRegistration + 'static,
        D: Fn(&[u8]) -> Result<T, String> + Send + Sync + 'static,
        E: Fn(&T) -> Result<Vec<u8>, String> + Send + Sync + 'static,
    {
        let version32 = u32::try_from(version)
            .map_err(|_| ReflectStateError::new("reflect schema exceeds u32"))?;
        if self.versions.contains_key(&version) {
            return Err(ReflectStateError::new(format!(
                "duplicate reflect schema version {version}"
            )));
        }
        self.registry.register::<T>();
        let type_name = <T as Typed>::type_info().type_name();
        let schema = TypeSchema::new(type_name, version32, <T as Typed>::type_info());
        let decode = Arc::new(move |bytes: &[u8]| {
            decode(bytes).map(|value| Box::new(value) as Box<dyn Reflect>)
        });
        let rebuild: Rebuild = Arc::new(|value| {
            T::from_reflect(value)
                .map(|value| Box::new(value) as Box<dyn Reflect>)
                .ok_or_else(|| "target reflected value has the wrong shape or leaf types".into())
        });
        let encode = Arc::new(move |value: &dyn Reflect| {
            let value = value
                .downcast_ref::<T>()
                .ok_or_else(|| "reflected value did not rebuild as the target type".to_owned())?;
            encode(value)
        });
        self.versions.insert(
            version,
            Version {
                schema,
                info: <T as Typed>::type_info(),
                metadata,
                decode,
                rebuild,
                encode,
            },
        );
        Ok(())
    }

    /// Add one consecutive schema migration (`from -> from + 1`).
    pub fn migration<F>(
        &mut self,
        from: u64,
        to: u64,
        step: F,
    ) -> Result<(), ReflectStateError>
    where
        F: Fn(&mut DynamicStruct) -> Result<(), String> + Send + Sync + 'static,
    {
        if to != from.saturating_add(1) {
            return Err(ReflectStateError::new(
                "reflect migrations must advance exactly one schema version",
            ));
        }
        if self.migrations.insert(from, Arc::new(step)).is_some() {
            return Err(ReflectStateError::new(format!(
                "duplicate reflect migration from schema {from}"
            )));
        }
        Ok(())
    }

    /// Prepare old module bytes for a candidate schema without invoking the
    /// candidate library. The returned fingerprint is a host canonical binary
    /// used to detect a candidate that silently loses migrated fields.
    pub fn prepare(
        &self,
        from: u64,
        to: u64,
        bytes: &[u8],
    ) -> Result<PreparedState, ReflectStateError> {
        let source = self.version(from)?;
        let target = self.version(to)?;
        if from > to {
            return Err(ReflectStateError::new(format!(
                "cannot downgrade reflected state from schema {from} to {to}"
            )));
        }
        let typed = (source.decode)(bytes).map_err(ReflectStateError::new)?;
        if typed.type_name() != source.schema.type_name() {
            return Err(ReflectStateError::new("source decoder returned the wrong type"));
        }
        let canonical = to_binary(&*typed).map_err(|e| ReflectStateError::new(e.to_string()))?;
        let mut dynamic = from_binary_with_limits(
            &canonical,
            &self.registry,
            source.info,
            &self.limits,
        )
        .map_err(|e| ReflectStateError::new(e.to_string()))?
        .downcast::<DynamicStruct>()
        .map_err(|_| ReflectStateError::new("reflect binary did not produce a struct"))?;
        self.validate(&*dynamic, source)?;
        let mut version = from;
        while version < to {
            let step = self.migrations.get(&version).ok_or_else(|| {
                ReflectStateError::new(format!(
                    "no reflect migration path from schema {version} to {to}"
                ))
            })?;
            step(&mut dynamic).map_err(ReflectStateError::new)?;
            version = version.saturating_add(1);
        }
        self.validate(&*dynamic, target)?;
        let typed_target = (target.rebuild)(&*dynamic).map_err(ReflectStateError::new)?;
        let target_canonical = to_binary(&*typed_target)
            .map_err(|e| ReflectStateError::new(e.to_string()))?;
        let checked = from_binary_with_limits(
            &target_canonical,
            &self.registry,
            target.info,
            &self.limits,
        )
        .map_err(|e| ReflectStateError::new(e.to_string()))?;
        if checked.type_name() != target.schema.type_name() {
            return Err(ReflectStateError::new("target reflect round trip changed type"));
        }
        let payload = (target.encode)(&*typed_target).map_err(ReflectStateError::new)?;
        Ok(PreparedState {
            schema: to,
            payload,
            fingerprint: target_canonical,
        })
    }

    /// Validate and fingerprint a candidate's export after it imports a
    /// prepared state. This rejects silent field loss before activation.
    pub fn validate_import(
        &self,
        schema: u64,
        bytes: &[u8],
        expected_fingerprint: &[u8],
    ) -> Result<(), ReflectStateError> {
        let version = self.version(schema)?;
        let typed = (version.decode)(bytes).map_err(ReflectStateError::new)?;
        let canonical = to_binary(&*typed).map_err(|e| ReflectStateError::new(e.to_string()))?;
        if canonical != expected_fingerprint {
            return Err(ReflectStateError::new(
                "candidate export does not preserve the migrated reflected state",
            ));
        }
        let decoded = from_binary_with_limits(
            &canonical,
            &self.registry,
            version.info,
            &self.limits,
        )
        .map_err(|e| ReflectStateError::new(e.to_string()))?;
        self.validate(&*decoded, version)
    }

    fn version(&self, version: u64) -> Result<&Version, ReflectStateError> {
        self.versions.get(&version).ok_or_else(|| {
            ReflectStateError::new(format!(
                "reflect contract `{}` has no schema version {version}",
                self.logical
            ))
        })
    }

    fn validate(&self, value: &dyn Reflect, version: &Version) -> Result<(), ReflectStateError> {
        validate(value, &version.schema, version.metadata.as_ref()).map_err(|errors| {
            ReflectStateError::new(
                errors
                    .into_iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<_>>()
                    .join("; "),
            )
        })
    }
}

/// Candidate-native bytes and a host canonical representation prepared before
/// any candidate state is created.
pub struct PreparedState {
    pub(crate) schema: u64,
    pub(crate) payload: Vec<u8>,
    pub(crate) fingerprint: Vec<u8>,
}
