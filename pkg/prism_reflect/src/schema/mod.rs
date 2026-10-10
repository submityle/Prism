//! Attribute metadata, schema versioning, migration chains, and validation
//! (design §10, §11, §22 — milestone **M4**).
//!
//! This module is the "schema" half of the type-truth layer. On top of the M3
//! reflection serializer it adds the pieces an archive/network/editor pipeline
//! needs to stay compatible as types evolve:
//!
//! - **Attribute metadata** ([`metadata`]): [`TypeMetadata`]/[`FieldMetadata`]
//!   attach typed intent (ranges, docs, categories, read-only/hidden flags,
//!   defaults, custom key/values) to a reflected type. Metadata is a
//!   [`TypeData`](crate::TypeData), so it is attached to and queried from the
//!   existing [`TypeRegistry`](crate::TypeRegistry).
//! - **Schema versioning** ([`version`]): every reflectable type carries a
//!   schema version; a [`SchemaRegistry`] records each version's shape
//!   ([`TypeSchema`]) and the current version per logical type.
//! - **Migration chains** ([`migration`]): ordered [`Migration`] steps
//!   (`vN -> vN+1`) compose into a chain that upgrades an old payload decoded by
//!   the M3 serializer, one version at a time, to the current shape.
//! - **Validation** ([`validate`]): check a reflected value against its schema
//!   and metadata (required fields, numeric ranges) and a payload version
//!   against the registry (not newer than current, reachable by migration).
//! - **Versioned serialization** ([`serde_integration`]): thin wrappers over
//!   the M3 [`to_binary`](crate::to_binary)/[`to_ron`](crate::to_ron) streams
//!   that stamp and recover a schema version and run the migration chain, so a
//!   versioned payload round-trips and an old-version payload migrates
//!   correctly.

pub mod clamp;
pub mod metadata;
pub mod migration;
pub mod serde_integration;
pub mod validate;
pub mod version;

pub use clamp::{clamp, Clamped};
pub use metadata::{AttributeValue, FieldMetadata, TypeMetadata};
pub use migration::{MigrateError, Migration, MigrationFn};
pub use serde_integration::{
    from_versioned_binary, from_versioned_ron, to_versioned_binary, to_versioned_ron,
};
pub use validate::{validate, validate_version, ValidationError};
pub use version::{SchemaRegistry, SchemaVersion, TypeSchema};
