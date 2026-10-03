//! Schema resolution shared by the binary and RON deserializers.
//!
//! Reconstruction is driven by the target type's static [`TypeInfo`], resolved
//! through the [`TypeRegistry`] by fully-qualified type name. A child type is
//! either a built-in leaf [`Primitive`] (which needs no registration) or a
//! composite whose [`TypeInfo`] must be registered; [`resolve`] turns a type
//! name into the [`Schema`] the decoders walk.

use crate::ser::error::DeserializeError;
use crate::ser::primitive::{Primitive, leaf_primitive};
use crate::type_info::TypeInfo;
use crate::TypeRegistry;

/// The decoding plan for a single value position.
pub enum Schema<'a> {
    /// A built-in leaf primitive (no registry lookup required).
    Primitive(Primitive),
    /// A composite type described by its static [`TypeInfo`].
    Info(&'a TypeInfo),
}

/// Resolve a child type name into a [`Schema`].
///
/// Leaf primitives resolve directly; any other name must be registered in
/// `registry` so its [`TypeInfo`] can guide reconstruction.
///
/// # Errors
/// Returns [`DeserializeError::UnregisteredType`] when a non-primitive type
/// name has no registration.
pub fn resolve<'a>(
    registry: &'a TypeRegistry,
    type_name: &str,
) -> Result<Schema<'a>, DeserializeError> {
    if let Some(primitive) = leaf_primitive(type_name) {
        return Ok(Schema::Primitive(primitive));
    }
    match registry.get_with_name(type_name) {
        Some(registration) => Ok(Schema::Info(registration.type_info())),
        None => Err(DeserializeError::UnregisteredType(type_name.into())),
    }
}

/// Build the root [`Schema`] directly from the caller-supplied target info.
///
/// A leaf target becomes [`Schema::Primitive`]; everything else is wrapped as
/// [`Schema::Info`] without a registry round-trip.
#[must_use]
pub fn root_schema(target: &TypeInfo) -> Schema<'_> {
    if let TypeInfo::Value(info) = target {
        if let Some(primitive) = leaf_primitive(info.type_name()) {
            return Schema::Primitive(primitive);
        }
    }
    Schema::Info(target)
}
