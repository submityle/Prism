//! Reflected configuration: schema validation + range clamping (design §25.3).
//!
//! The design (§25.3) delegates configuration / cvar rules — range clamping,
//! defaults, renames, and rejecting invalid input — to the reflection layer
//! rather than re-deriving them inside `prism_app`. This module is that seam: it
//! applies [`prism_reflect`]'s [`clamp`] (repair out-of-range numeric fields,
//! reporting each) and [`validate`] (reject structurally invalid input) to a
//! reflected configuration value against the type's schema + metadata.
//!
//! Invalid input is *reported* as [`ValidationError`], never panicked on, which
//! is the safety boundary the design asks for (§24.8). Defaults / renames ride
//! on the same metadata ([`FieldMetadata::default_value`](prism_reflect::schema::FieldMetadata))
//! and feed [`App::register_reflected_cvars`](crate::App::register_reflected_cvars)
//! through [`CvarSpec`] into the settings cascade.

use std::any::type_name as rust_type_name;
use std::error::Error;
use std::fmt;

use prism_reflect::schema::{
    clamp, validate, AttributeValue, Clamped, TypeMetadata, TypeSchema, ValidationError,
};
use prism_reflect::{GetTypeRegistration, Reflect, TypeInfo, TypeRegistration};

use crate::cvar::{CvarBounds, CvarCategory, CvarError, CvarFlags, CvarSpec};
use crate::settings::SettingValue;

/// Failure while deriving cvar declarations from reflected field metadata.
///
/// Attribute syntax is checked by `#[derive(Reflect)]`; these errors cover the
/// semantic boundary that only the consumer can validate, such as applying a
/// numeric range to a string default or representing an unsigned default that
/// does not fit the settings store's `i64` integer type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReflectConfigError {
    /// The reflected type has no [`TypeMetadata`] attached to its registration.
    MissingMetadata {
        /// Fully-qualified reflected type name.
        type_name: String,
    },
    /// Cvar field metadata was attached to a non-struct reflected type.
    UnsupportedReflectedType {
        /// Fully-qualified reflected type name.
        type_name: String,
        /// Reflected shape that was found.
        found: &'static str,
    },
    /// Metadata declares a cvar for a field absent from the reflected struct.
    UnknownField {
        /// Fully-qualified reflected type name.
        type_name: String,
        /// Metadata field name.
        field: String,
    },
    /// The reflected field type cannot be represented by [`SettingValue`].
    UnsupportedFieldType {
        /// Source Rust field name.
        field: String,
        /// Fully-qualified Rust field type.
        field_type: String,
    },
    /// The metadata default kind is incompatible with the reflected field type.
    DefaultTypeMismatch {
        /// Source Rust field name.
        field: String,
        /// Fully-qualified Rust field type.
        field_type: String,
        /// Metadata default kind.
        default_kind: &'static str,
    },
    /// The metadata default is outside the reflected field type's value domain.
    DefaultOutOfRange {
        /// Source Rust field name.
        field: String,
        /// Fully-qualified Rust field type.
        field_type: String,
    },
    /// A `rename` attribute exists but is not a non-empty string.
    InvalidRename {
        /// Source Rust field name.
        field: String,
    },
    /// A default cannot be represented by [`SettingValue`].
    UnsupportedDefault {
        /// Source Rust field name.
        field: String,
        /// Stable description of the unsupported value.
        reason: &'static str,
    },
    /// A numeric range is non-finite, inverted, or contains no representable
    /// integer for an integer default.
    InvalidRange {
        /// Source Rust field name.
        field: String,
    },
    /// A numeric range was attached to a non-numeric default.
    RangeTypeMismatch {
        /// Source Rust field name.
        field: String,
        /// Default value kind.
        default_kind: &'static str,
    },
    /// The derived declaration conflicts with the existing cvar registry.
    Cvar(CvarError),
}

impl fmt::Display for ReflectConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingMetadata { type_name } => {
                write!(f, "reflected type `{type_name}` has no attribute metadata")
            }
            Self::UnsupportedReflectedType { type_name, found } => write!(
                f,
                "reflected type `{type_name}` is a {found}; reflected cvars require a named-field struct"
            ),
            Self::UnknownField { type_name, field } => write!(
                f,
                "reflected metadata for `{type_name}` names unknown field `{field}`"
            ),
            Self::UnsupportedFieldType { field, field_type } => write!(
                f,
                "reflected field `{field}` has unsupported cvar type `{field_type}`"
            ),
            Self::DefaultTypeMismatch {
                field,
                field_type,
                default_kind,
            } => write!(
                f,
                "reflected field `{field}` of type `{field_type}` cannot use a {default_kind} default"
            ),
            Self::DefaultOutOfRange { field, field_type } => write!(
                f,
                "reflected field `{field}` has a default outside `{field_type}`'s value range"
            ),
            Self::InvalidRename { field } => write!(
                f,
                "reflected field `{field}` has a rename that is not a non-empty string"
            ),
            Self::UnsupportedDefault { field, reason } => {
                write!(
                    f,
                    "reflected field `{field}` has an unsupported default: {reason}"
                )
            }
            Self::InvalidRange { field } => {
                write!(f, "reflected field `{field}` has an invalid numeric range")
            }
            Self::RangeTypeMismatch {
                field,
                default_kind,
            } => write!(
                f,
                "reflected field `{field}` applies a numeric range to a {default_kind} default"
            ),
            Self::Cvar(error) => error.fmt(f),
        }
    }
}

impl Error for ReflectConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Cvar(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CvarError> for ReflectConfigError {
    fn from(value: CvarError) -> Self {
        Self::Cvar(value)
    }
}

/// Derive cvar declarations from `T`'s `#[reflect(...)]` field metadata.
///
/// Every field carrying `default = ...` becomes one cvar. `clamp`/`range`
/// becomes its inclusive bounds, `rename` becomes the canonical cvar name and
/// the Rust field name remains accepted as a compatibility alias. Fields with
/// metadata but no default remain ordinary reflected configuration fields and
/// are not exposed as cvars.
///
/// # Errors
/// Returns [`ReflectConfigError`] when metadata is absent or semantically
/// incompatible with the settings/cvar value model.
pub fn cvar_specs_for<T: GetTypeRegistration>() -> Result<Vec<CvarSpec>, ReflectConfigError> {
    let registration = T::get_type_registration();
    cvar_specs_from_registration(&registration)
}

/// Derive cvar declarations from an existing reflected type registration.
///
/// This form consumes metadata attached manually through a [`TypeRegistry`](prism_reflect::TypeRegistry)
/// as well as metadata emitted by `#[derive(Reflect)]`.
///
/// # Errors
/// Returns [`ReflectConfigError`] when metadata is absent or invalid.
pub fn cvar_specs_from_registration(
    registration: &TypeRegistration,
) -> Result<Vec<CvarSpec>, ReflectConfigError> {
    let metadata =
        registration
            .data::<TypeMetadata>()
            .ok_or_else(|| ReflectConfigError::MissingMetadata {
                type_name: registration.type_name().to_owned(),
            })?;
    let TypeInfo::Struct(type_info) = registration.type_info() else {
        return Err(ReflectConfigError::UnsupportedReflectedType {
            type_name: registration.type_name().to_owned(),
            found: type_info_kind(registration.type_info()),
        });
    };
    cvar_specs_from_metadata_impl(metadata, Some((registration.type_name(), type_info)))
}

/// Derive cvar declarations directly from reflected type metadata.
///
/// Conversion is deterministic and preserves metadata field order. It performs
/// no registry mutation; callers can validate the complete batch before making
/// it visible to the settings cascade. This metadata-only form infers each cvar
/// kind from its default; use [`cvar_specs_from_registration`] or
/// [`cvar_specs_for`] to also check the reflected Rust field types.
///
/// # Errors
/// Returns [`ReflectConfigError`] for malformed rename/default/range metadata.
pub fn cvar_specs_from_metadata(
    metadata: &TypeMetadata,
) -> Result<Vec<CvarSpec>, ReflectConfigError> {
    cvar_specs_from_metadata_impl(metadata, None)
}

fn cvar_specs_from_metadata_impl(
    metadata: &TypeMetadata,
    reflected: Option<(&str, &prism_reflect::StructInfo)>,
) -> Result<Vec<CvarSpec>, ReflectConfigError> {
    let mut specs = Vec::new();
    for field in metadata.fields() {
        if matches!(field.custom("skip"), Some(AttributeValue::Bool(true))) {
            continue;
        }
        let Some(default) = field.default_value() else {
            continue;
        };

        let field_type = if let Some((type_name, type_info)) = reflected {
            Some(
                type_info
                    .field(field.name())
                    .ok_or_else(|| ReflectConfigError::UnknownField {
                        type_name: type_name.to_owned(),
                        field: field.name().to_owned(),
                    })?
                    .type_name(),
            )
        } else {
            None
        };

        let canonical = match field.custom("rename") {
            None => field.name(),
            Some(AttributeValue::Text(name)) if !name.trim().is_empty() => name.as_str(),
            Some(_) => {
                return Err(ReflectConfigError::InvalidRename {
                    field: field.name().to_owned(),
                });
            }
        };
        let default = setting_value(field.name(), field_type, default)?;
        let bounds = cvar_bounds(field.name(), field_type, field.range(), &default)?;

        let mut spec = CvarSpec::new(canonical, default)
            .bounds(bounds)
            .category(category(field.category()))
            .description(field.docs().unwrap_or_default());
        if canonical != field.name() {
            spec = spec.alias(field.name());
        }
        if field.is_readonly() {
            spec = spec.flag(CvarFlags::READ_ONLY);
        }
        specs.push(spec);
    }
    Ok(specs)
}

#[derive(Clone, Copy)]
enum ReflectedScalar {
    Bool,
    SignedInt { min: i128, max: i128 },
    UnsignedInt { max: u128 },
    Float { max: f64 },
    String,
}

fn reflected_scalar(type_name: &str) -> Option<ReflectedScalar> {
    if type_name == rust_type_name::<bool>() {
        Some(ReflectedScalar::Bool)
    } else if type_name == rust_type_name::<i8>() {
        Some(ReflectedScalar::SignedInt {
            min: i128::from(i8::MIN),
            max: i128::from(i8::MAX),
        })
    } else if type_name == rust_type_name::<i16>() {
        Some(ReflectedScalar::SignedInt {
            min: i128::from(i16::MIN),
            max: i128::from(i16::MAX),
        })
    } else if type_name == rust_type_name::<i32>() {
        Some(ReflectedScalar::SignedInt {
            min: i128::from(i32::MIN),
            max: i128::from(i32::MAX),
        })
    } else if type_name == rust_type_name::<i64>() {
        Some(ReflectedScalar::SignedInt {
            min: i128::from(i64::MIN),
            max: i128::from(i64::MAX),
        })
    } else if type_name == rust_type_name::<i128>() {
        Some(ReflectedScalar::SignedInt {
            min: i128::MIN,
            max: i128::MAX,
        })
    } else if type_name == rust_type_name::<isize>() {
        Some(ReflectedScalar::SignedInt {
            min: isize::MIN as i128,
            max: isize::MAX as i128,
        })
    } else if type_name == rust_type_name::<u8>() {
        Some(ReflectedScalar::UnsignedInt {
            max: u128::from(u8::MAX),
        })
    } else if type_name == rust_type_name::<u16>() {
        Some(ReflectedScalar::UnsignedInt {
            max: u128::from(u16::MAX),
        })
    } else if type_name == rust_type_name::<u32>() {
        Some(ReflectedScalar::UnsignedInt {
            max: u128::from(u32::MAX),
        })
    } else if type_name == rust_type_name::<u64>() {
        Some(ReflectedScalar::UnsignedInt {
            max: u128::from(u64::MAX),
        })
    } else if type_name == rust_type_name::<u128>() {
        Some(ReflectedScalar::UnsignedInt { max: u128::MAX })
    } else if type_name == rust_type_name::<usize>() {
        Some(ReflectedScalar::UnsignedInt {
            max: usize::MAX as u128,
        })
    } else if type_name == rust_type_name::<f32>() {
        Some(ReflectedScalar::Float {
            max: f64::from(f32::MAX),
        })
    } else if type_name == rust_type_name::<f64>() {
        Some(ReflectedScalar::Float { max: f64::MAX })
    } else if type_name == rust_type_name::<String>() {
        Some(ReflectedScalar::String)
    } else {
        None
    }
}

fn setting_value(
    field: &str,
    field_type: Option<&str>,
    value: &AttributeValue,
) -> Result<SettingValue, ReflectConfigError> {
    let Some(field_type) = field_type else {
        return untyped_setting_value(field, value);
    };
    let scalar =
        reflected_scalar(field_type).ok_or_else(|| ReflectConfigError::UnsupportedFieldType {
            field: field.to_owned(),
            field_type: field_type.to_owned(),
        })?;

    match (scalar, value) {
        (ReflectedScalar::Bool, AttributeValue::Bool(value)) => Ok(SettingValue::Bool(*value)),
        (ReflectedScalar::String, AttributeValue::Text(value)) => {
            Ok(SettingValue::Str(value.clone()))
        }
        (ReflectedScalar::SignedInt { min, max }, AttributeValue::Int(value))
            if (min..=max).contains(&i128::from(*value)) =>
        {
            Ok(SettingValue::Int(*value))
        }
        (ReflectedScalar::SignedInt { max, .. }, AttributeValue::UInt(value))
            if i128::from(*value) <= max =>
        {
            i64::try_from(*value).map(SettingValue::Int).map_err(|_| {
                ReflectConfigError::UnsupportedDefault {
                    field: field.to_owned(),
                    reason: "unsigned integer does not fit i64",
                }
            })
        }
        (ReflectedScalar::UnsignedInt { max }, AttributeValue::Int(value))
            if *value >= 0 && (*value as u128) <= max =>
        {
            Ok(SettingValue::Int(*value))
        }
        (ReflectedScalar::UnsignedInt { max }, AttributeValue::UInt(value))
            if u128::from(*value) <= max =>
        {
            i64::try_from(*value).map(SettingValue::Int).map_err(|_| {
                ReflectConfigError::UnsupportedDefault {
                    field: field.to_owned(),
                    reason: "unsigned integer does not fit i64",
                }
            })
        }
        (ReflectedScalar::Float { max }, AttributeValue::Float(value))
            if value.is_finite() && value.abs() <= max =>
        {
            Ok(SettingValue::Float(*value))
        }
        (ReflectedScalar::Float { max }, AttributeValue::Int(value))
            if (*value as f64).abs() <= max =>
        {
            Ok(SettingValue::Float(*value as f64))
        }
        (ReflectedScalar::Float { max }, AttributeValue::UInt(value)) if (*value as f64) <= max => {
            Ok(SettingValue::Float(*value as f64))
        }
        (ReflectedScalar::SignedInt { .. }, AttributeValue::Int(_) | AttributeValue::UInt(_))
        | (ReflectedScalar::UnsignedInt { .. }, AttributeValue::Int(_) | AttributeValue::UInt(_))
        | (
            ReflectedScalar::Float { .. },
            AttributeValue::Int(_) | AttributeValue::UInt(_) | AttributeValue::Float(_),
        ) => Err(ReflectConfigError::DefaultOutOfRange {
            field: field.to_owned(),
            field_type: field_type.to_owned(),
        }),
        _ => Err(ReflectConfigError::DefaultTypeMismatch {
            field: field.to_owned(),
            field_type: field_type.to_owned(),
            default_kind: attribute_kind(value),
        }),
    }
}

fn untyped_setting_value(
    field: &str,
    value: &AttributeValue,
) -> Result<SettingValue, ReflectConfigError> {
    match value {
        AttributeValue::Bool(value) => Ok(SettingValue::Bool(*value)),
        AttributeValue::Int(value) => Ok(SettingValue::Int(*value)),
        AttributeValue::UInt(value) => i64::try_from(*value).map(SettingValue::Int).map_err(|_| {
            ReflectConfigError::UnsupportedDefault {
                field: field.to_owned(),
                reason: "unsigned integer does not fit i64",
            }
        }),
        AttributeValue::Float(value) if value.is_finite() => Ok(SettingValue::Float(*value)),
        AttributeValue::Float(_) => Err(ReflectConfigError::UnsupportedDefault {
            field: field.to_owned(),
            reason: "floating-point default is not finite",
        }),
        AttributeValue::Text(value) => Ok(SettingValue::Str(value.clone())),
        _ => Err(ReflectConfigError::UnsupportedDefault {
            field: field.to_owned(),
            reason: "attribute value kind is not supported by settings",
        }),
    }
}

fn attribute_kind(value: &AttributeValue) -> &'static str {
    match value {
        AttributeValue::Bool(_) => "bool",
        AttributeValue::Int(_) => "signed integer",
        AttributeValue::UInt(_) => "unsigned integer",
        AttributeValue::Float(_) => "float",
        AttributeValue::Text(_) => "string",
        _ => "unsupported",
    }
}

fn cvar_bounds(
    field: &str,
    field_type: Option<&str>,
    range: Option<(f64, f64)>,
    default: &SettingValue,
) -> Result<CvarBounds, ReflectConfigError> {
    let Some((min, max)) = range else {
        return Ok(CvarBounds::None);
    };
    if !min.is_finite() || !max.is_finite() || min > max {
        return Err(ReflectConfigError::InvalidRange {
            field: field.to_owned(),
        });
    }
    let bounds = match default {
        SettingValue::Int(_) => {
            let min = min.ceil();
            let max = max.floor();
            // i64::MAX rounds up to 2^63 as f64. Reject that endpoint rather
            // than silently saturating an out-of-domain range during the cast.
            if min > max || min < i64::MIN as f64 || max >= -(i64::MIN as f64) {
                return Err(ReflectConfigError::InvalidRange {
                    field: field.to_owned(),
                });
            }
            CvarBounds::Int(min as i64, max as i64)
        }
        SettingValue::Float(_) => CvarBounds::Float(min, max),
        SettingValue::Bool(_) => {
            return Err(ReflectConfigError::RangeTypeMismatch {
                field: field.to_owned(),
                default_kind: "bool",
            });
        }
        SettingValue::Str(_) => {
            return Err(ReflectConfigError::RangeTypeMismatch {
                field: field.to_owned(),
                default_kind: "string",
            });
        }
    };

    if let Some(field_type) = field_type {
        let valid_for_field = match (reflected_scalar(field_type), bounds) {
            (Some(ReflectedScalar::SignedInt { min, max }), CvarBounds::Int(lo, hi)) => {
                i128::from(lo) >= min && i128::from(hi) <= max
            }
            (Some(ReflectedScalar::UnsignedInt { max }), CvarBounds::Int(lo, hi)) => {
                lo >= 0 && (hi as u128) <= max
            }
            (Some(ReflectedScalar::Float { max }), CvarBounds::Float(lo, hi)) => {
                lo >= -max && hi <= max
            }
            _ => false,
        };
        if !valid_for_field {
            return Err(ReflectConfigError::InvalidRange {
                field: field.to_owned(),
            });
        }
    }
    Ok(bounds)
}

fn type_info_kind(type_info: &TypeInfo) -> &'static str {
    match type_info {
        TypeInfo::Struct(_) => "struct",
        TypeInfo::TupleStruct(_) => "tuple struct",
        TypeInfo::Enum(_) => "enum",
        TypeInfo::List(_) => "list",
        TypeInfo::Array(_) => "array",
        TypeInfo::Map(_) => "map",
        TypeInfo::Set(_) => "set",
        TypeInfo::Value(_) => "value",
        _ => "unknown reflected shape",
    }
}

fn category(value: Option<&str>) -> CvarCategory {
    match value {
        Some(value) if value.eq_ignore_ascii_case("render") => CvarCategory::Render,
        Some(value) if value.eq_ignore_ascii_case("audio") => CvarCategory::Audio,
        Some(value) if value.eq_ignore_ascii_case("network") => CvarCategory::Network,
        Some(value) if value.eq_ignore_ascii_case("input") => CvarCategory::Input,
        Some(value) if value.eq_ignore_ascii_case("physics") => CvarCategory::Physics,
        Some(value) if value.eq_ignore_ascii_case("debug") => CvarCategory::Debug,
        Some(value) if value.eq_ignore_ascii_case("gameplay") => CvarCategory::Gameplay,
        Some(value) if value.eq_ignore_ascii_case("system") => CvarCategory::System,
        _ => CvarCategory::General,
    }
}

/// Build the [`TypeSchema`] describing `value`'s reflected type at `version`.
///
/// The static [`TypeInfo`](prism_reflect::TypeInfo) is read from the value
/// itself ([`Reflect::type_info`]), so callers do not need the `Typed` derive.
#[must_use]
pub fn schema_for(value: &dyn Reflect, name: &'static str, version: u32) -> TypeSchema {
    TypeSchema::new(name, version, value.type_info())
}

/// Clamp every range-constrained field of `value` into its declared range, then
/// validate the result against `schema` and `metadata`.
///
/// Returns the fields that were clamped (empty when none were), or every
/// [`ValidationError`] found. Out-of-range numbers are *repaired* (clamped);
/// structural problems — a missing required field, a range-constrained field
/// that is not numeric, or a non-struct value — are *rejected*. Never panics.
///
/// # Errors
/// Every [`ValidationError`] collected by `validate` (never an empty `Vec`).
pub fn clamp_and_validate(
    value: &mut dyn Reflect,
    schema: &TypeSchema,
    metadata: &TypeMetadata,
) -> Result<Vec<Clamped>, Vec<ValidationError>> {
    let clamped = clamp(value, metadata);
    validate(&*value, schema, Some(metadata))?;
    Ok(clamped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs::event::Events;
    use prism_reflect::schema::FieldMetadata;

    use crate::app::App;
    use crate::cvar::{CvarChanged, CvarError, CvarRegistry};
    use crate::settings::{SettingChanged, Settings};

    #[derive(prism_reflect::Reflect, Default)]
    struct Render {
        exposure: f32,
        bounces: i32,
    }

    #[derive(prism_reflect::Reflect, Default)]
    struct ReflectedCvars {
        /// Camera exposure compensation.
        #[reflect(
            rename = "r.exposure",
            default = 5.0,
            clamp(0.0..=2.0),
            category = "render"
        )]
        exposure: f64,

        #[reflect(default = 99, clamp(0..=8))]
        bounces: i32,

        #[reflect(default = true, readonly)]
        vsync: bool,

        #[reflect(default = "balanced")]
        profile: String,

        #[reflect(category = "render")]
        ordinary_config: u32,

        #[reflect(default = 1, skip)]
        skipped: i32,
    }

    #[derive(prism_reflect::Reflect, Default)]
    struct UnannotatedConfig {
        value: i32,
    }

    #[derive(prism_reflect::Reflect, Default)]
    struct CollidingCvars {
        #[reflect(rename = "new.name", default = 1)]
        fresh: i32,

        #[reflect(rename = "existing", default = 2)]
        conflicts: i32,
    }

    #[derive(prism_reflect::Reflect, Default)]
    struct AliasCanonicalCollision {
        #[reflect(rename = "second", default = 1)]
        first: i32,

        #[reflect(default = 2)]
        second: i32,
    }

    #[derive(prism_reflect::Reflect, Default)]
    struct WrongDefaultType {
        #[reflect(default = "many")]
        count: i32,
    }

    #[derive(prism_reflect::Reflect, Default)]
    struct OutOfRangeDefault {
        #[reflect(default = 300)]
        count: i8,
    }

    #[derive(prism_reflect::Reflect, Default)]
    struct UnrepresentableUnsignedDefault {
        #[reflect(default = 18446744073709551615)]
        count: u64,
    }

    #[derive(prism_reflect::Reflect, Default)]
    struct UnsupportedDefaultType {
        #[reflect(default = 1)]
        values: Vec<i32>,
    }

    #[derive(prism_reflect::Reflect, Default)]
    struct OutOfRangeBounds {
        #[reflect(default = 1, range(-200..=200))]
        count: i8,
    }

    #[derive(prism_reflect::Reflect, Default)]
    struct NegativeDefaults {
        #[reflect(default = -3, range(-8..=8))]
        offset: i32,

        #[reflect(default = -9223372036854775808)]
        lower_limit: i64,

        #[reflect(default = -0.25, clamp(-1.0..=1.0))]
        bias: f32,
    }

    fn metadata() -> TypeMetadata {
        TypeMetadata::new()
            .with_field(FieldMetadata::new("exposure").with_range(0.0, 2.0))
            .with_field(FieldMetadata::new("bounces").with_range(0.0, 8.0))
    }

    #[test]
    fn clamps_out_of_range_fields_and_reports_them() {
        let mut render = Render {
            exposure: 5.0,
            bounces: -3,
        };
        let schema = schema_for(&render, "Render", 1);
        let clamped = clamp_and_validate(&mut render, &schema, &metadata()).expect("valid");
        assert_eq!(clamped.len(), 2);
        assert_eq!(render.exposure, 2.0); // clamped up to the declared max
        assert_eq!(render.bounces, 0);
    }

    #[test]
    fn in_range_values_are_untouched() {
        let mut render = Render {
            exposure: 1.0,
            bounces: 3,
        };
        let schema = schema_for(&render, "Render", 1);
        let clamped = clamp_and_validate(&mut render, &schema, &metadata()).expect("valid");
        assert!(clamped.is_empty());
        assert_eq!(render.exposure, 1.0);
        assert_eq!(render.bounces, 3);
    }

    #[test]
    fn rejects_a_missing_required_field() {
        let mut render = Render::default();
        let schema = schema_for(&render, "Render", 1).with_required_fields(&["absent"]);
        let errors = clamp_and_validate(&mut render, &schema, &metadata()).unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::MissingRequiredField { field } if field.as_str() == "absent"
        )));
    }

    #[test]
    fn rejects_a_non_struct_value() {
        let mut value = 7u32;
        let schema = schema_for(&value, "u32", 1);
        let errors = clamp_and_validate(&mut value, &schema, &TypeMetadata::new()).unwrap_err();
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::NotAStruct)));
    }

    #[test]
    fn derives_and_registers_cvars_from_reflect_metadata() {
        let mut app = App::new();
        app.register_reflected_cvars::<ReflectedCvars>()
            .expect("metadata should produce valid cvars");

        let registry = app.world().resource::<CvarRegistry>();
        assert_eq!(registry.len(), 4);
        assert_eq!(registry.resolve("exposure"), "r.exposure");
        assert!(!registry.contains("ordinary_config"));
        assert!(!registry.contains("skipped"));

        let exposure = registry.get("r.exposure").expect("renamed cvar");
        assert_eq!(exposure.default_value(), &SettingValue::Float(2.0));
        assert_eq!(exposure.bounds(), CvarBounds::Float(0.0, 2.0));
        assert_eq!(exposure.category(), CvarCategory::Render);
        assert_eq!(exposure.description(), "Camera exposure compensation.");

        let bounces = registry.get("bounces").expect("integer cvar");
        assert_eq!(bounces.default_value(), &SettingValue::Int(8));
        assert_eq!(bounces.bounds(), CvarBounds::Int(0, 8));

        let vsync = registry.get("vsync").expect("boolean cvar");
        assert!(vsync.flags().contains(CvarFlags::READ_ONLY));

        let settings = app.world().resource::<Settings>();
        assert_eq!(settings.get("r.exposure"), Some(&SettingValue::Float(2.0)));
        assert_eq!(settings.get("bounces"), Some(&SettingValue::Int(8)));
        assert_eq!(settings.get("vsync"), Some(&SettingValue::Bool(true)));
        assert_eq!(
            settings.get("profile"),
            Some(&SettingValue::Str(String::from("balanced")))
        );
        assert!(settings.get("exposure").is_none());
    }

    #[test]
    fn rename_alias_resolves_writes_to_the_canonical_key() {
        let mut app = App::new();
        app.register_reflected_cvars::<ReflectedCvars>()
            .expect("metadata should produce valid cvars");

        let outcome = app
            .set_cvar("exposure", -3.0)
            .expect("the source field name should remain a valid alias");
        assert!(outcome.clamped);
        assert_eq!(outcome.resolved, SettingValue::Float(0.0));

        let settings = app.world().resource::<Settings>();
        assert_eq!(settings.get("r.exposure"), Some(&SettingValue::Float(0.0)));
        assert!(settings.get("exposure").is_none());
        assert_eq!(app.cvar_float("exposure"), Some(0.0));
        assert_eq!(app.cvar("exposure"), app.cvar("r.exposure"));
    }

    #[test]
    fn negative_numeric_defaults_reach_the_settings_cascade() {
        let mut app = App::new();
        app.register_reflected_cvars::<NegativeDefaults>()
            .expect("negative defaults should be representable");

        assert_eq!(app.cvar_int("offset"), Some(-3));
        assert_eq!(app.cvar_int("lower_limit"), Some(i64::MIN));
        assert_eq!(app.cvar_float("bias"), Some(-0.25));
        assert_eq!(
            app.world()
                .resource::<CvarRegistry>()
                .get("offset")
                .unwrap()
                .bounds(),
            CvarBounds::Int(-8, 8)
        );
    }

    #[test]
    fn wrong_type_alias_write_leaves_settings_and_events_unchanged() {
        let mut app = App::new();
        app.register_reflected_cvars::<ReflectedCvars>().unwrap();
        let event_count = app.world().resource::<Events<SettingChanged>>().len();

        assert_eq!(
            app.set_cvar("exposure", "invalid").unwrap_err(),
            CvarError::TypeMismatch {
                name: String::from("r.exposure"),
                expected: "float",
                found: "string",
            }
        );
        assert_eq!(app.cvar_float("exposure"), Some(2.0));
        assert!(app.world().resource::<Settings>().get("exposure").is_none());
        assert_eq!(
            app.world().resource::<Events<SettingChanged>>().len(),
            event_count
        );
    }

    #[test]
    fn rejects_semantically_invalid_cvar_metadata() {
        let string_range = TypeMetadata::new().with_field(
            FieldMetadata::new("mode")
                .with_default(AttributeValue::Text(String::from("fast")))
                .with_range(0.0, 1.0),
        );
        assert_eq!(
            cvar_specs_from_metadata(&string_range).unwrap_err(),
            ReflectConfigError::RangeTypeMismatch {
                field: String::from("mode"),
                default_kind: "string",
            }
        );

        let inverted = TypeMetadata::new().with_field(
            FieldMetadata::new("quality")
                .with_default(AttributeValue::Int(2))
                .with_range(4.0, 1.0),
        );
        assert_eq!(
            cvar_specs_from_metadata(&inverted).unwrap_err(),
            ReflectConfigError::InvalidRange {
                field: String::from("quality"),
            }
        );

        let invalid_rename = TypeMetadata::new().with_field(
            FieldMetadata::new("quality")
                .with_default(AttributeValue::Int(2))
                .with_custom("rename", AttributeValue::Bool(true)),
        );
        assert_eq!(
            cvar_specs_from_metadata(&invalid_rename).unwrap_err(),
            ReflectConfigError::InvalidRename {
                field: String::from("quality"),
            }
        );

        let oversized_unsigned = TypeMetadata::new()
            .with_field(FieldMetadata::new("budget").with_default(AttributeValue::UInt(u64::MAX)));
        assert_eq!(
            cvar_specs_from_metadata(&oversized_unsigned).unwrap_err(),
            ReflectConfigError::UnsupportedDefault {
                field: String::from("budget"),
                reason: "unsigned integer does not fit i64",
            }
        );
    }

    #[test]
    fn reports_missing_attribute_metadata() {
        let error = cvar_specs_for::<UnannotatedConfig>().unwrap_err();
        assert_eq!(
            error,
            ReflectConfigError::MissingMetadata {
                type_name: std::any::type_name::<UnannotatedConfig>().to_owned(),
            }
        );
    }

    #[test]
    fn integer_range_conversion_rejects_saturation_and_empty_intervals() {
        for (min, max) in [
            (0.0, -(i64::MIN as f64)),
            (i64::MIN as f64 - 2048.0, 0.0),
            (0.1, 0.9),
            (f64::NAN, 1.0),
            (0.0, f64::INFINITY),
        ] {
            let metadata = TypeMetadata::new().with_field(
                FieldMetadata::new("count")
                    .with_default(AttributeValue::Int(0))
                    .with_range(min, max),
            );
            assert_eq!(
                cvar_specs_from_metadata(&metadata).unwrap_err(),
                ReflectConfigError::InvalidRange {
                    field: String::from("count")
                }
            );
        }

        let metadata = TypeMetadata::new().with_field(
            FieldMetadata::new("count")
                .with_default(AttributeValue::Int(i64::MIN))
                .with_range(i64::MIN as f64, i64::MIN as f64),
        );
        let mut registry = CvarRegistry::new();
        registry
            .register(cvar_specs_from_metadata(&metadata).unwrap().remove(0))
            .unwrap();
        assert_eq!(
            registry.get("count").unwrap().bounds(),
            CvarBounds::Int(i64::MIN, i64::MIN)
        );
    }

    #[test]
    fn rejects_metadata_that_disagrees_with_the_reflected_field_type() {
        assert_eq!(
            cvar_specs_for::<WrongDefaultType>().unwrap_err(),
            ReflectConfigError::DefaultTypeMismatch {
                field: String::from("count"),
                field_type: std::any::type_name::<i32>().to_owned(),
                default_kind: "string",
            }
        );
        assert_eq!(
            cvar_specs_for::<OutOfRangeDefault>().unwrap_err(),
            ReflectConfigError::DefaultOutOfRange {
                field: String::from("count"),
                field_type: std::any::type_name::<i8>().to_owned(),
            }
        );
        assert_eq!(
            cvar_specs_for::<UnrepresentableUnsignedDefault>().unwrap_err(),
            ReflectConfigError::UnsupportedDefault {
                field: String::from("count"),
                reason: "unsigned integer does not fit i64",
            }
        );
        assert_eq!(
            cvar_specs_for::<UnsupportedDefaultType>().unwrap_err(),
            ReflectConfigError::UnsupportedFieldType {
                field: String::from("values"),
                field_type: std::any::type_name::<Vec<i32>>().to_owned(),
            }
        );
        assert_eq!(
            cvar_specs_for::<OutOfRangeBounds>().unwrap_err(),
            ReflectConfigError::InvalidRange {
                field: String::from("count"),
            }
        );
    }

    #[test]
    fn rejects_metadata_for_a_field_absent_from_the_reflected_struct() {
        let mut registration = <Render as GetTypeRegistration>::get_type_registration();
        registration.insert(
            TypeMetadata::new()
                .with_field(FieldMetadata::new("missing").with_default(AttributeValue::Int(1))),
        );

        assert_eq!(
            cvar_specs_from_registration(&registration).unwrap_err(),
            ReflectConfigError::UnknownField {
                type_name: std::any::type_name::<Render>().to_owned(),
                field: String::from("missing"),
            }
        );
    }

    #[test]
    fn reflected_registration_is_atomic_on_name_collision() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("existing", 7_i64))
            .expect("baseline cvar registration");
        let event_count = app.world().resource::<Events<SettingChanged>>().len();

        let error = app
            .register_reflected_cvars::<CollidingCvars>()
            .unwrap_err();
        assert_eq!(
            error,
            ReflectConfigError::Cvar(CvarError::Redeclared(String::from("existing")))
        );

        let registry = app.world().resource::<CvarRegistry>();
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("existing"));
        assert!(!registry.contains("new.name"));
        assert!(!registry.contains("fresh"));
        assert!(!registry.contains("conflicts"));

        let settings = app.world().resource::<Settings>();
        assert_eq!(settings.get("existing"), Some(&SettingValue::Int(7)));
        assert!(settings.get("new.name").is_none());
        assert!(settings.get("fresh").is_none());
        assert_eq!(
            app.world().resource::<Events<SettingChanged>>().len(),
            event_count
        );
    }

    #[test]
    fn rejected_reflected_registration_does_not_install_resources_or_events() {
        let mut app = App::new();
        assert!(app
            .register_reflected_cvars::<AliasCanonicalCollision>()
            .is_err());
        assert!(app.register_reflected_cvars::<WrongDefaultType>().is_err());

        assert!(app.world().get_resource::<CvarRegistry>().is_none());
        assert!(app.world().get_resource::<Settings>().is_none());
        assert!(app
            .world()
            .get_resource::<Events<SettingChanged>>()
            .is_none());
        assert!(app.world().get_resource::<Events<CvarChanged>>().is_none());
    }

    #[test]
    fn reflected_registration_is_atomic_on_alias_canonical_collision() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("baseline", 7_i64))
            .expect("baseline cvar registration");

        let error = app
            .register_reflected_cvars::<AliasCanonicalCollision>()
            .unwrap_err();
        assert_eq!(
            error,
            ReflectConfigError::Cvar(CvarError::Redeclared(String::from("second")))
        );

        let registry = app.world().resource::<CvarRegistry>();
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("baseline"));
        assert!(!registry.contains("first"));
        assert!(!registry.contains("second"));

        let settings = app.world().resource::<Settings>();
        assert_eq!(settings.get("baseline"), Some(&SettingValue::Int(7)));
        assert!(settings.get("first").is_none());
        assert!(settings.get("second").is_none());
    }
}
