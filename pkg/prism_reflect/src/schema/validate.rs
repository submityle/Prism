//! Validating a reflected value against its schema and metadata (design §11).
//!
//! [`validate`] checks a reflected *struct* value against a [`TypeSchema`]
//! (required fields present) and, when supplied, a [`TypeMetadata`] record
//! (numeric range constraints, required-field flags). All violations are
//! collected so a caller sees every problem at once. [`validate_version`]
//! separately checks that a payload's recorded version is compatible with a
//! [`SchemaRegistry`] (not newer than current, and reachable by a migration
//! chain).

use crate::reflect::Reflect;
use crate::schema::metadata::TypeMetadata;
use crate::schema::version::{SchemaRegistry, TypeSchema};
use crate::ReflectRef;
use std::string::String;
use std::vec::Vec;

/// A single schema/metadata validation failure.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ValidationError {
    /// The value was not a struct, so field-level rules cannot apply.
    NotAStruct,
    /// A required field was missing from the value.
    MissingRequiredField {
        /// The absent field's name.
        field: String,
    },
    /// A numeric field fell outside its metadata range.
    OutOfRange {
        /// The offending field's name.
        field: String,
        /// The field's actual value (widened to `f64`).
        value: f64,
        /// The inclusive lower bound.
        min: f64,
        /// The inclusive upper bound.
        max: f64,
    },
    /// A range-constrained field was not a numeric leaf.
    NotNumeric {
        /// The offending field's name.
        field: String,
    },
    /// The payload's version is newer than the current schema.
    VersionTooNew {
        /// The version read from the payload.
        found: u32,
        /// The current schema version.
        current: u32,
    },
    /// No migration chain reaches the current version from the payload's.
    NoMigrationPath {
        /// The payload's version.
        from: u32,
        /// The current schema version.
        current: u32,
    },
}

impl ::core::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        match self {
            ValidationError::NotAStruct => write!(f, "value is not a struct"),
            ValidationError::MissingRequiredField { field } => {
                write!(f, "required field `{field}` is missing")
            }
            ValidationError::OutOfRange {
                field,
                value,
                min,
                max,
            } => write!(
                f,
                "field `{field}` value {value} is outside range {min}..={max}"
            ),
            ValidationError::NotNumeric { field } => {
                write!(f, "range-constrained field `{field}` is not numeric")
            }
            ValidationError::VersionTooNew { found, current } => write!(
                f,
                "payload version v{found} is newer than current schema v{current}"
            ),
            ValidationError::NoMigrationPath { from, current } => {
                write!(f, "no migration path from v{from} to current v{current}")
            }
        }
    }
}

impl ::std::error::Error for ValidationError {}

/// Validate a reflected struct value against its schema and optional metadata.
///
/// Checks performed:
/// - every field named in [`TypeSchema::required_fields`] is present;
/// - every field flagged required in `metadata` is present;
/// - every range-constrained field in `metadata` holds a numeric value within
///   its inclusive bounds.
///
/// # Errors
/// Returns every [`ValidationError`] found (never an empty `Vec`), or `Ok` when
/// the value satisfies all rules.
pub fn validate(
    value: &dyn Reflect,
    schema: &TypeSchema,
    metadata: Option<&TypeMetadata>,
) -> Result<(), Vec<ValidationError>> {
    let ReflectRef::Struct(source) = value.reflect_ref() else {
        return Err(std::vec![ValidationError::NotAStruct]);
    };

    let mut errors = Vec::new();

    for required in schema.required_fields() {
        if source.field(required).is_none() {
            errors.push(ValidationError::MissingRequiredField {
                field: String::from(*required),
            });
        }
    }

    if let Some(metadata) = metadata {
        for field in metadata.fields() {
            let present = source.field(field.name());
            if field.is_required() && present.is_none() {
                errors.push(ValidationError::MissingRequiredField {
                    field: String::from(field.name()),
                });
            }
            if let (Some((min, max)), Some(value)) = (field.range(), present) {
                match as_f64(value) {
                    Some(number) => {
                        if number < min || number > max {
                            errors.push(ValidationError::OutOfRange {
                                field: String::from(field.name()),
                                value: number,
                                min,
                                max,
                            });
                        }
                    }
                    None => errors.push(ValidationError::NotNumeric {
                        field: String::from(field.name()),
                    }),
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Validate that a payload's recorded version is compatible with the registry.
///
/// Succeeds when `found` is reachable by a migration chain to the current
/// version of `logical` (including when `found` already is current).
///
/// # Errors
/// - [`ValidationError::VersionTooNew`] when `found` exceeds the current version.
/// - [`ValidationError::NoMigrationPath`] when no chain reaches current.
pub fn validate_version(
    found: u32,
    schema: &SchemaRegistry,
    logical: &str,
) -> Result<(), ValidationError> {
    let current = schema
        .current_version(logical)
        .map(|v| v.value())
        .unwrap_or(found);
    if found > current {
        return Err(ValidationError::VersionTooNew { found, current });
    }
    match schema.chain(logical, found) {
        Ok(_) => Ok(()),
        Err(_) => Err(ValidationError::NoMigrationPath { from: found, current }),
    }
}

/// Widen any numeric leaf value to `f64` for range checks.
///
/// Returns `None` for non-numeric leaves (bool, char, string) and composites.
fn as_f64(value: &dyn Reflect) -> Option<f64> {
    let any = value.as_any();
    if let Some(v) = any.downcast_ref::<i8>() {
        return Some(f64::from(*v));
    }
    if let Some(v) = any.downcast_ref::<i16>() {
        return Some(f64::from(*v));
    }
    if let Some(v) = any.downcast_ref::<i32>() {
        return Some(f64::from(*v));
    }
    if let Some(v) = any.downcast_ref::<i64>() {
        return Some(*v as f64);
    }
    if let Some(v) = any.downcast_ref::<i128>() {
        return Some(*v as f64);
    }
    if let Some(v) = any.downcast_ref::<isize>() {
        return Some(*v as f64);
    }
    if let Some(v) = any.downcast_ref::<u8>() {
        return Some(f64::from(*v));
    }
    if let Some(v) = any.downcast_ref::<u16>() {
        return Some(f64::from(*v));
    }
    if let Some(v) = any.downcast_ref::<u32>() {
        return Some(f64::from(*v));
    }
    if let Some(v) = any.downcast_ref::<u64>() {
        return Some(*v as f64);
    }
    if let Some(v) = any.downcast_ref::<u128>() {
        return Some(*v as f64);
    }
    if let Some(v) = any.downcast_ref::<usize>() {
        return Some(*v as f64);
    }
    if let Some(v) = any.downcast_ref::<f32>() {
        return Some(f64::from(*v));
    }
    if let Some(v) = any.downcast_ref::<f64>() {
        return Some(*v);
    }
    None
}
