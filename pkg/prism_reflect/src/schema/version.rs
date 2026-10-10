//! Schema versioning: per-type version shapes and the [`SchemaRegistry`].
//!
//! Every reflectable type carries a *schema version*. A [`TypeSchema`] records
//! the shape (static [`TypeInfo`]) and required fields of one version of a
//! *logical* type, while the [`SchemaRegistry`] catalogues every known version
//! of every logical type plus the ordered [`Migration`] steps between them.
//!
//! Logical types are keyed by a caller-chosen name (not a Rust `TypeId`) so a
//! type can evolve across versions that are backed by *different* Rust types
//! (e.g. `MonsterV1`, `MonsterV2`, `Monster`) while sharing one migration
//! chain. [`SchemaRegistry::locate`] maps a concrete type name back to the
//! `(logical, version)` it was registered under, which the versioned
//! serializer uses to stamp an outgoing payload with its version.

use crate::schema::migration::{MigrateError, Migration};
use crate::type_info::TypeInfo;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use std::collections::HashMap;

/// A schema version number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SchemaVersion(u32);

impl SchemaVersion {
    /// Wrap a raw version number.
    #[must_use]
    pub fn new(version: u32) -> Self {
        Self(version)
    }

    /// The raw version number.
    #[must_use]
    pub fn value(self) -> u32 {
        self.0
    }
}

impl From<u32> for SchemaVersion {
    fn from(version: u32) -> Self {
        Self(version)
    }
}

/// The shape of one version of a logical type.
///
/// Carries the static [`TypeInfo`] used to decode a payload of this version and
/// the set of field names schema validation treats as required.
#[derive(Debug, Clone)]
pub struct TypeSchema {
    type_name: &'static str,
    version: u32,
    info: &'static TypeInfo,
    required_fields: Vec<&'static str>,
}

impl TypeSchema {
    /// Describe version `version` of a type whose shape is `info`.
    ///
    /// `type_name` is the concrete Rust type name backing this version (used to
    /// decode a payload); it may differ across versions of the same logical
    /// type.
    #[must_use]
    pub fn new(type_name: &'static str, version: u32, info: &'static TypeInfo) -> Self {
        Self {
            type_name,
            version,
            info,
            required_fields: Vec::new(),
        }
    }

    /// Mark the named fields as required for schema validation.
    #[must_use]
    pub fn with_required_fields(mut self, fields: &[&'static str]) -> Self {
        self.required_fields = fields.to_vec();
        self
    }

    /// The concrete Rust type name backing this version.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// This version's number.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The static shape used to decode a payload of this version.
    #[must_use]
    pub fn info(&self) -> &'static TypeInfo {
        self.info
    }

    /// The field names required by validation.
    #[must_use]
    pub fn required_fields(&self) -> &[&'static str] {
        &self.required_fields
    }
}

/// All known versions and migrations of one logical type.
#[derive(Debug, Default)]
struct VersionedType {
    current: u32,
    versions: BTreeMap<u32, TypeSchema>,
    migrations: BTreeMap<u32, Migration>,
}

/// A runtime catalogue of schema versions and their migration chains.
///
/// Register each version's shape with [`register_version`](Self::register_version)
/// and each `vN -> vN+1` step with
/// [`register_migration`](Self::register_migration); the newest registered
/// version becomes [`current_version`](Self::current_version).
#[derive(Debug, Default)]
pub struct SchemaRegistry {
    types: HashMap<String, VersionedType>,
    by_type_name: HashMap<String, (String, u32)>,
}

impl SchemaRegistry {
    /// Build an empty schema registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the shape of one version of the logical type `logical`.
    ///
    /// The highest version seen so far becomes the current version.
    pub fn register_version(&mut self, logical: impl Into<String>, schema: TypeSchema) {
        let logical = logical.into();
        let version = schema.version();
        let concrete = schema.type_name();
        let entry = self.types.entry(logical.clone()).or_default();
        entry.current = entry.current.max(version);
        entry.versions.insert(version, schema);
        self.by_type_name
            .insert(String::from(concrete), (logical, version));
    }

    /// Register an ordered `vN -> vN+1` migration step for `logical`.
    ///
    /// # Errors
    /// Returns [`MigrateError::NonContiguous`] if the step advances more than
    /// one version.
    pub fn register_migration(
        &mut self,
        logical: impl Into<String>,
        migration: Migration,
    ) -> Result<(), MigrateError> {
        if migration.to_version() != migration.from_version() + 1 {
            return Err(MigrateError::NonContiguous {
                from: migration.from_version(),
                to: migration.to_version(),
            });
        }
        let entry = self.types.entry(logical.into()).or_default();
        entry.migrations.insert(migration.from_version(), migration);
        Ok(())
    }

    /// The current (newest registered) version of `logical`, if any.
    #[must_use]
    pub fn current_version(&self, logical: &str) -> Option<SchemaVersion> {
        self.types
            .get(logical)
            .map(|t| SchemaVersion::new(t.current))
    }

    /// The schema shape for a specific version of `logical`.
    #[must_use]
    pub fn schema(&self, logical: &str, version: u32) -> Option<&TypeSchema> {
        self.types.get(logical)?.versions.get(&version)
    }

    /// The current (newest) schema shape for `logical`.
    #[must_use]
    pub fn current_schema(&self, logical: &str) -> Option<&TypeSchema> {
        let entry = self.types.get(logical)?;
        entry.versions.get(&entry.current)
    }

    /// Resolve which `(logical, version)` a concrete type name was registered
    /// under.
    #[must_use]
    pub fn locate(&self, type_name: &str) -> Option<(&str, u32)> {
        self.by_type_name
            .get(type_name)
            .map(|(logical, version)| (logical.as_str(), *version))
    }

    /// Compose the ordered migration chain upgrading `from` to the current
    /// version of `logical`.
    ///
    /// Returns an empty slice of steps when `from` is already current.
    ///
    /// # Errors
    /// - [`MigrateError::UnknownType`] when `logical` is unregistered.
    /// - [`MigrateError::VersionTooNew`] when `from` exceeds the current version.
    /// - [`MigrateError::NoMigrationPath`] when a step between `from` and
    ///   current is missing.
    pub fn chain(&self, logical: &str, from: u32) -> Result<Vec<&Migration>, MigrateError> {
        let entry = self
            .types
            .get(logical)
            .ok_or_else(|| MigrateError::UnknownType(String::from(logical)))?;
        let current = entry.current;
        if from > current {
            return Err(MigrateError::VersionTooNew {
                found: from,
                current,
            });
        }
        let mut steps = Vec::new();
        let mut version = from;
        while version < current {
            let step = entry
                .migrations
                .get(&version)
                .ok_or(MigrateError::NoMigrationPath {
                    from: version,
                    current,
                })?;
            steps.push(step);
            version = step.to_version();
        }
        Ok(steps)
    }
}
