//! Versioned (de)serialization built on the M3 reflection serializer.
//!
//! These helpers wrap the format-level [`to_binary`]/[`from_binary`] and
//! [`to_ron`]/[`from_ron`] round-trips with a schema version so that:
//!
//! - an outgoing payload is stamped with its type's current schema version
//!   ([`to_versioned_binary`]/[`to_versioned_ron`]); and
//! - an incoming payload is decoded against the shape it was written with, then
//!   walked forward through the registered [`Migration`](crate::schema::Migration)
//!   chain to the current shape before [`FromReflect`](crate::FromReflect)
//!   rebuilds the concrete value
//!   ([`from_versioned_binary`]/[`from_versioned_ron`]).
//!
//! The envelope is deliberately thin — a small header carrying the version,
//! followed by the untouched M3 stream — so the on-disk M3 format is unchanged
//! and a current-version payload round-trips with zero migration work.

use crate::dynamic::DynamicStruct;
use crate::reflect::Reflect;
use crate::schema::migration::MigrateError;
use crate::schema::version::SchemaRegistry;
use crate::ser::{from_binary, from_ron, to_binary, to_ron};
use crate::TypeRegistry;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// The 4-byte magic opening a versioned binary envelope.
const BINARY_MAGIC: [u8; 4] = *b"PRVB";

/// The textual prefix opening a versioned RON envelope (`#prism-schema vN\n`).
const RON_PREFIX: &str = "#prism-schema v";

/// Serialize `value` to the versioned binary envelope.
///
/// The type is located in `schema` to stamp the payload with its registered
/// version; the body is the ordinary M3 binary stream.
///
/// # Errors
/// - [`MigrateError::UnknownType`] when `value`'s type is not registered in
///   `schema`.
/// - [`MigrateError::Serialize`] when the M3 serializer rejects a leaf.
pub fn to_versioned_binary(
    value: &dyn Reflect,
    schema: &SchemaRegistry,
) -> Result<Vec<u8>, MigrateError> {
    let (_, version) = schema
        .locate(value.type_name())
        .ok_or_else(|| MigrateError::UnknownType(String::from(value.type_name())))?;
    let body = to_binary(value)?;
    let mut out = Vec::with_capacity(8 + body.len());
    out.extend_from_slice(&BINARY_MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Deserialize a versioned binary payload for logical type `logical`, migrating
/// it to the current schema version.
///
/// Decoding is guided by the shape registered for the payload's recorded
/// version; the result is then walked through the migration chain to current.
/// The returned value is a [`DynamicStruct`] (or the raw decoded value when the
/// payload is already current) ready for
/// [`FromReflect`](crate::FromReflect)::`from_reflect`.
///
/// # Errors
/// A [`MigrateError`] for a bad envelope, an unknown type/version, a
/// deserialization failure, a missing migration step, or a step that fails.
pub fn from_versioned_binary(
    bytes: &[u8],
    registry: &TypeRegistry,
    schema: &SchemaRegistry,
    logical: &str,
) -> Result<Box<dyn Reflect>, MigrateError> {
    if bytes.len() < 8 || bytes[..4] != BINARY_MAGIC {
        return Err(MigrateError::BadEnvelope);
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let body = &bytes[8..];

    let shape = schema
        .schema(logical, version)
        .ok_or(MigrateError::UnknownVersion { found: version })?;
    let decoded = from_binary(body, registry, shape.info())?;
    migrate_decoded(decoded, registry, schema, logical, version)
}

/// Serialize `value` to the versioned RON envelope.
///
/// # Errors
/// As [`to_versioned_binary`], but for the RON back-end.
pub fn to_versioned_ron(
    value: &dyn Reflect,
    schema: &SchemaRegistry,
) -> Result<String, MigrateError> {
    let (_, version) = schema
        .locate(value.type_name())
        .ok_or_else(|| MigrateError::UnknownType(String::from(value.type_name())))?;
    let body = to_ron(value)?;
    let mut out = String::with_capacity(RON_PREFIX.len() + body.len() + 4);
    out.push_str(RON_PREFIX);
    out.push_str(&version.to_string());
    out.push('\n');
    out.push_str(&body);
    Ok(out)
}

/// Deserialize a versioned RON payload for logical type `logical`, migrating it
/// to the current schema version.
///
/// # Errors
/// As [`from_versioned_binary`], but for the RON back-end.
pub fn from_versioned_ron(
    text: &str,
    registry: &TypeRegistry,
    schema: &SchemaRegistry,
    logical: &str,
) -> Result<Box<dyn Reflect>, MigrateError> {
    let rest = text
        .strip_prefix(RON_PREFIX)
        .ok_or(MigrateError::BadEnvelope)?;
    let newline = rest.find('\n').ok_or(MigrateError::BadEnvelope)?;
    let version: u32 = rest[..newline]
        .trim()
        .parse()
        .map_err(|_| MigrateError::BadEnvelope)?;
    let body = &rest[newline + 1..];

    let shape = schema
        .schema(logical, version)
        .ok_or(MigrateError::UnknownVersion { found: version })?;
    let decoded = from_ron(body, registry, shape.info())?;
    migrate_decoded(decoded, registry, schema, logical, version)
}

/// Walk a freshly-decoded payload through the migration chain to current.
fn migrate_decoded(
    decoded: Box<dyn Reflect>,
    _registry: &TypeRegistry,
    schema: &SchemaRegistry,
    logical: &str,
    version: u32,
) -> Result<Box<dyn Reflect>, MigrateError> {
    let chain = schema.chain(logical, version)?;
    if chain.is_empty() {
        return Ok(decoded);
    }
    let mut dynamic = decoded
        .into_any()
        .downcast::<DynamicStruct>()
        .map_err(|_| MigrateError::NotAStruct)?;
    for step in chain {
        step.apply(&mut dynamic)?;
    }
    Ok(dynamic)
}
