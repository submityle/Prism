//! Derive the canonical settings schema from reflected configuration structs.
//!
//! This opt-in adapter keeps `prism_reflect` metadata as the authoring input and
//! `prism_settings` as the only runtime schema. Unsupported fields are rejected
//! explicitly rather than silently omitted.

use prism_reflect::{
    schema::{AttributeValue, TypeMetadata},
    Enum, GetTypeRegistration, Map, Reflect, ReflectRef, Struct, TypeInfo, Typed, VariantKind,
    VariantType,
};
use prism_settings::{
    canonical_value_cmp, Constraints, Domain, DomainMask, OverrideScope, PrecedenceProfile,
    RegistryBuilder, Scope, SettingDef, SettingKey, SettingsError, SettingsRegistry, Value,
    ValueShape,
};
use std::{collections::BTreeMap, error::Error, fmt};

const MAX_REFLECT_SETTING_COLLECTION_LEN: usize = 1024;

type EnumOption = prism_settings::EnumOption;
type EnumPayloadSchema = prism_settings::EnumOptionPayload;
type EnumSchema = prism_settings::EnumSchema;
type EnumVariantSchema = prism_settings::EnumVariantSchema;
type Derived = (
    Value,
    ValueShape,
    Vec<EnumOption>,
    Vec<EnumPayloadSchema>,
    Vec<EnumSchema>,
);

/// Failure while deriving a canonical settings registry.
#[derive(Debug)]
#[non_exhaustive]
pub enum ReflectSettingsError {
    /// The input is not a reflected named-field struct.
    NotNamedStruct(&'static str),
    /// A default instance did not expose a field present in static type info.
    MissingDefaultField {
        /// Fully-qualified reflected type name.
        type_name: String,
        /// Missing field name.
        field: String,
    },
    /// A field type cannot be represented by canonical settings values.
    UnsupportedValue {
        /// Field name.
        field: String,
        /// Fully-qualified Rust field type name.
        type_name: String,
    },
    /// A reflected literal default does not match a canonical value type.
    MetadataTypeMismatch {
        /// Field name whose metadata default conflicts with its Rust type.
        field: String,
    },
    /// The resulting schema failed canonical registry validation.
    InvalidSchema(Vec<SettingsError>),
}

impl fmt::Display for ReflectSettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotNamedStruct(name) => write!(
                f,
                "reflected settings type `{name}` is not a named-field struct"
            ),
            Self::MissingDefaultField { type_name, field } => {
                write!(f, "default `{type_name}` has no reflected field `{field}`")
            }
            Self::UnsupportedValue { field, type_name } => write!(
                f,
                "field `{field}` has unsupported canonical settings type `{type_name}`"
            ),
            Self::MetadataTypeMismatch { field } => {
                write!(f, "metadata default for `{field}` has an incompatible type")
            }
            Self::InvalidSchema(errors) => {
                write!(f, "derived settings schema is invalid: {errors:?}")
            }
        }
    }
}

impl Error for ReflectSettingsError {}

/// Derive a frozen settings registry from `T`'s reflected schema and defaults.
///
/// Each key is `<namespace>.<field>`. Only fields explicitly marked with
/// `#[reflect(setting)]` are considered; unmarked, `skip`, and hidden fields
/// are excluded.
/// docs, categories, ranges, defaults, and read-only attributes are mapped
/// into canonical definitions. Scalar defaults support bool, signed/unsigned
/// integers fitting `i64`, `f32`, `f64`, and `String`. Explicit settings may
/// also be recursively homogeneous `Vec<T>` or `[T; N]` values, named
/// reflected structs, and string-keyed homogeneous maps. Every dynamic
/// collection is bounded by [`MAX_REFLECT_SETTING_COLLECTION_LEN`]. Empty
/// collections are accepted only when reflected type metadata identifies a
/// supported scalar value type; nested empty collections fail closed because
/// type-name strings are not parsed to guess child schemas. Other types fail
/// closed.
pub fn derive_settings_registry<T>(
    namespace: &str,
    owner: &str,
) -> Result<SettingsRegistry, ReflectSettingsError>
where
    T: Reflect + Typed + GetTypeRegistration + Default,
{
    let registration = T::get_type_registration();
    let metadata = registration.data::<TypeMetadata>();
    let default = T::default();
    let ReflectRef::Struct(fields) = default.reflect_ref() else {
        return Err(ReflectSettingsError::NotNamedStruct(default.type_name()));
    };
    let TypeInfo::Struct(info) = <T as Typed>::type_info() else {
        return Err(ReflectSettingsError::NotNamedStruct(default.type_name()));
    };

    let mut builder = RegistryBuilder::new();
    builder
        .namespace(namespace, owner)
        .map_err(|error| ReflectSettingsError::InvalidSchema(vec![error]))?;
    for descriptor in info.fields() {
        let name = descriptor.name();
        let meta = metadata.and_then(|metadata| metadata.field(name));
        if !meta.is_some_and(|meta| meta.is_setting()) {
            continue;
        }
        if meta.is_some_and(|meta| {
            meta.is_hidden() || matches!(meta.custom("skip"), Some(AttributeValue::Bool(true)))
        }) {
            continue;
        }
        let reflected =
            fields
                .field(name)
                .ok_or_else(|| ReflectSettingsError::MissingDefaultField {
                    type_name: default.type_name().to_owned(),
                    field: name.to_owned(),
                })?;
        let (rust_default, shape, enum_options, enum_payloads, enum_schemas) = match reflected
            .reflect_ref()
        {
            ReflectRef::Enum(active) => {
                let TypeInfo::Enum(enum_info) = reflected.type_info() else {
                    return Err(ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    });
                };
                from_reflect_enum(active, enum_info, 0, name, descriptor.type_name())?
            }
            ReflectRef::List(list) => {
                let TypeInfo::List(info) = reflected.type_info() else {
                    return Err(ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    });
                };
                from_reflect_sequence(
                    list.iter_reflect(),
                    list.len(),
                    info.item_type_name(),
                    None,
                    0,
                    name,
                    descriptor.type_name(),
                )?
            }
            ReflectRef::Array(array) => {
                let TypeInfo::Array(info) = reflected.type_info() else {
                    return Err(ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    });
                };
                from_reflect_sequence(
                    array.iter_reflect(),
                    array.len(),
                    info.item_type_name(),
                    Some(info.capacity()),
                    0,
                    name,
                    descriptor.type_name(),
                )?
            }
            ReflectRef::Struct(fields) => {
                let TypeInfo::Struct(info) = reflected.type_info() else {
                    return Err(ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    });
                };
                from_reflect_struct(fields, info, 0, name, descriptor.type_name())?
            }
            ReflectRef::Map(map) => {
                let TypeInfo::Map(info) = reflected.type_info() else {
                    return Err(ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    });
                };
                from_reflect_map(
                    map,
                    info.key_type_name(),
                    info.value_type_name(),
                    0,
                    name,
                    descriptor.type_name(),
                )?
            }
            ReflectRef::Set(set) => {
                let TypeInfo::Set(info) = reflected.type_info() else {
                    return Err(ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    });
                };
                let mut values = set
                    .iter_reflect()
                    .map(|value| from_reflect_value(value, 1, name, descriptor.type_name()))
                    .collect::<Result<Vec<_>, _>>()?;
                values.sort_by(|left, right| canonical_value_cmp(&left.0, &right.0));
                if values.len() > MAX_REFLECT_SETTING_COLLECTION_LEN
                    || values.windows(2).any(|pair| pair[0].0 == pair[1].0)
                {
                    return Err(ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    });
                }
                let element_shape = values
                    .first()
                    .map(|item| item.1.clone())
                    .or_else(|| {
                        scalar_value_type_from_name(info.value_type_name()).map(ValueShape::scalar)
                    })
                    .ok_or_else(|| ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    })?;
                if values.iter().any(|item| item.1 != element_shape) {
                    return Err(ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    });
                }
                let options = values
                    .iter()
                    .find(|item| !item.2.is_empty())
                    .map(|item| item.2.clone())
                    .unwrap_or_default();
                let payloads = values
                    .iter()
                    .flat_map(|item| item.3.clone())
                    .collect::<Vec<_>>();
                let schemas = values
                    .iter()
                    .flat_map(|item| item.4.clone())
                    .collect::<Vec<_>>();
                (
                    Value::Array(values.into_iter().map(|item| item.0).collect()),
                    ValueShape::set(element_shape, MAX_REFLECT_SETTING_COLLECTION_LEN),
                    options,
                    payloads,
                    schemas,
                )
            }
            _ => {
                let value = from_reflect_scalar(reflected).ok_or_else(|| {
                    ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    }
                })?;
                let shape = ValueShape::scalar(value.value_type());
                (value, shape, Vec::new(), Vec::new(), Vec::new())
            }
        };
        let value = match meta.and_then(|meta| meta.default_value()) {
            Some(value) => {
                let mut value = from_attribute(value).ok_or_else(|| {
                    ReflectSettingsError::MetadataTypeMismatch {
                        field: name.to_owned(),
                    }
                })?;
                value = resolve_enum_names(value, &enum_options).ok_or_else(|| {
                    ReflectSettingsError::MetadataTypeMismatch {
                        field: name.to_owned(),
                    }
                })?;
                if !shape.matches(&value) {
                    return Err(ReflectSettingsError::MetadataTypeMismatch {
                        field: name.to_owned(),
                    });
                }
                value
            }
            None => rust_default,
        };
        let mut definition = SettingDef::new(format!("{namespace}.{name}"), namespace, value);
        definition.set_shape(shape);
        definition.enum_options = enum_options;
        definition.enum_payloads = enum_payloads;
        definition.enum_schemas = enum_schemas;
        if let Some(meta) = meta {
            if let Some(AttributeValue::Text(display_name)) = meta.custom("rename") {
                if display_name.trim().is_empty() {
                    return Err(ReflectSettingsError::MetadataTypeMismatch {
                        field: name.to_owned(),
                    });
                }
                definition.display_name = Some(display_name.clone());
            }
            definition.description = meta.docs().map(str::to_owned);
            definition.category = meta.category().map(str::to_owned);
            definition.flags.read_only = meta.is_readonly();
            definition.flags.required = meta.is_required();
            if let Some((min, max)) = meta.range() {
                definition.constraints = Constraints {
                    min: Some(min),
                    max: Some(max),
                    ..Constraints::default()
                };
            }
            if let Some(AttributeValue::Text(policy)) = meta.custom("on_violation") {
                definition.on_violation = match policy.as_str() {
                    "clamp" => prism_settings::ViolationPolicy::Clamp,
                    "reject" => prism_settings::ViolationPolicy::Reject,
                    _ => {
                        return Err(ReflectSettingsError::MetadataTypeMismatch {
                            field: name.to_owned(),
                        })
                    }
                };
            }
            if let Some(value) = meta.custom("scope") {
                definition.scope = match value {
                    AttributeValue::Text(value) => match value.as_str() {
                        "machine" => Scope::Machine,
                        "user" => Scope::User,
                        "project" => Scope::Project,
                        _ => {
                            return Err(ReflectSettingsError::MetadataTypeMismatch {
                                field: name.to_owned(),
                            })
                        }
                    },
                    _ => {
                        return Err(ReflectSettingsError::MetadataTypeMismatch {
                            field: name.to_owned(),
                        })
                    }
                };
            }
            if let Some(value) = meta.custom("override_scope") {
                definition.override_scope = match value {
                    AttributeValue::Text(value) => match value.as_str() {
                        "global" => OverrideScope::Global,
                        "machine" => OverrideScope::Machine,
                        "project" => OverrideScope::Project,
                        "resource" => OverrideScope::Resource,
                        "language" => OverrideScope::Language,
                        _ => {
                            return Err(ReflectSettingsError::MetadataTypeMismatch {
                                field: name.to_owned(),
                            })
                        }
                    },
                    _ => {
                        return Err(ReflectSettingsError::MetadataTypeMismatch {
                            field: name.to_owned(),
                        })
                    }
                };
            }
            if let Some(value) = meta.custom("precedence") {
                definition.precedence = match value {
                    AttributeValue::Text(value) => match value.as_str() {
                        "user_choice_wins" => PrecedenceProfile::UserChoiceWins,
                        "server_wins" => PrecedenceProfile::ServerWins,
                        "device_scoped" => PrecedenceProfile::DeviceScoped,
                        "locked" => PrecedenceProfile::Locked,
                        _ => {
                            return Err(ReflectSettingsError::MetadataTypeMismatch {
                                field: name.to_owned(),
                            })
                        }
                    },
                    _ => {
                        return Err(ReflectSettingsError::MetadataTypeMismatch {
                            field: name.to_owned(),
                        })
                    }
                };
            }
            if let Some(value) = meta.custom("domains") {
                let AttributeValue::Text(value) = value else {
                    return Err(ReflectSettingsError::MetadataTypeMismatch {
                        field: name.to_owned(),
                    });
                };
                let mut domains = DomainMask::NONE;
                let mut found = false;
                for domain in value.split(',').map(str::trim) {
                    let bit = match domain {
                        "runtime" => DomainMask::RUNTIME,
                        "editor" => DomainMask::EDITOR,
                        "system" => DomainMask::SYSTEM,
                        "build" => DomainMask::BUILD,
                        _ => {
                            return Err(ReflectSettingsError::MetadataTypeMismatch {
                                field: name.to_owned(),
                            })
                        }
                    };
                    domains = domains.union(bit);
                    found = true;
                }
                if !found {
                    return Err(ReflectSettingsError::MetadataTypeMismatch {
                        field: name.to_owned(),
                    });
                }
                definition.domains = domains;
            }
            if let Some(value) = meta.custom("aliases") {
                let AttributeValue::Text(value) = value else {
                    return Err(ReflectSettingsError::MetadataTypeMismatch {
                        field: name.to_owned(),
                    });
                };
                let aliases = value
                    .split(',')
                    .map(str::trim)
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                if aliases.is_empty() || aliases.iter().any(String::is_empty) {
                    return Err(ReflectSettingsError::MetadataTypeMismatch {
                        field: name.to_owned(),
                    });
                }
                definition.aliases = aliases.into_iter().map(SettingKey::from).collect();
            }
            for (key, slot) in [
                (
                    "project_overridable",
                    &mut definition.flags.project_overridable,
                ),
                ("bootstrap", &mut definition.flags.bootstrap),
                ("execution_class", &mut definition.flags.execution_class),
            ] {
                if let Some(value) = meta.custom(key) {
                    let AttributeValue::Bool(value) = value else {
                        return Err(ReflectSettingsError::MetadataTypeMismatch {
                            field: name.to_owned(),
                        });
                    };
                    *slot = *value;
                }
            }
            for (key, slot) in [
                (
                    "max_string_bytes",
                    &mut definition.constraints.max_string_bytes,
                ),
                ("max_array_len", &mut definition.constraints.max_array_len),
            ] {
                if let Some(value) = meta.custom(key) {
                    let AttributeValue::UInt(value) = value else {
                        return Err(ReflectSettingsError::MetadataTypeMismatch {
                            field: name.to_owned(),
                        });
                    };
                    *slot = Some(usize::try_from(*value).map_err(|_| {
                        ReflectSettingsError::MetadataTypeMismatch {
                            field: name.to_owned(),
                        }
                    })?);
                }
            }
        }
        builder.register(definition);
    }
    builder
        .finish()
        .map_err(ReflectSettingsError::InvalidSchema)
}

fn from_attribute(value: &AttributeValue) -> Option<Value> {
    match value {
        AttributeValue::Bool(value) => Some(Value::Bool(*value)),
        AttributeValue::Int(value) => Some(Value::Int(*value)),
        AttributeValue::UInt(value) => i64::try_from(*value).ok().map(Value::Int),
        AttributeValue::Float(value) => Some(Value::Float(*value)),
        AttributeValue::Text(value) => Some(Value::String(value.clone())),
        AttributeValue::Array(values) => values
            .iter()
            .map(from_attribute)
            .collect::<Option<Vec<_>>>()
            .map(Value::Array),
        _ => None,
    }
}

fn resolve_enum_names(value: Value, options: &[prism_settings::EnumOption]) -> Option<Value> {
    match value {
        Value::String(name) if !options.is_empty() => options
            .iter()
            .find(|option| option.stable_name == name)
            .map(|option| Value::Enum(option.id)),
        Value::Array(values) => values
            .into_iter()
            .map(|value| resolve_enum_names(value, options))
            .collect::<Option<Vec<_>>>()
            .map(Value::Array),
        value => Some(value),
    }
}

fn from_reflect_scalar(value: &dyn Reflect) -> Option<Value> {
    let any = value.as_any();
    if let Some(value) = any.downcast_ref::<bool>() {
        return Some(Value::Bool(*value));
    }
    if let Some(value) = any.downcast_ref::<i8>() {
        return Some(Value::Int(i64::from(*value)));
    }
    if let Some(value) = any.downcast_ref::<i16>() {
        return Some(Value::Int(i64::from(*value)));
    }
    if let Some(value) = any.downcast_ref::<i32>() {
        return Some(Value::Int(i64::from(*value)));
    }
    if let Some(value) = any.downcast_ref::<i64>() {
        return Some(Value::Int(*value));
    }
    if let Some(value) = any.downcast_ref::<i128>() {
        return i64::try_from(*value).ok().map(Value::Int);
    }
    if let Some(value) = any.downcast_ref::<isize>() {
        return i64::try_from(*value).ok().map(Value::Int);
    }
    if let Some(value) = any.downcast_ref::<u8>() {
        return Some(Value::Int(i64::from(*value)));
    }
    if let Some(value) = any.downcast_ref::<u16>() {
        return Some(Value::Int(i64::from(*value)));
    }
    if let Some(value) = any.downcast_ref::<u32>() {
        return Some(Value::Int(i64::from(*value)));
    }
    if let Some(value) = any.downcast_ref::<u64>() {
        return i64::try_from(*value).ok().map(Value::Int);
    }
    if let Some(value) = any.downcast_ref::<u128>() {
        return i64::try_from(*value).ok().map(Value::Int);
    }
    if let Some(value) = any.downcast_ref::<usize>() {
        return i64::try_from(*value).ok().map(Value::Int);
    }
    if let Some(value) = any.downcast_ref::<f32>() {
        return Some(Value::Float(f64::from(*value)));
    }
    if let Some(value) = any.downcast_ref::<f64>() {
        return Some(Value::Float(*value));
    }
    any.downcast_ref::<String>().cloned().map(Value::String)
}

fn from_reflect_sequence<'a, I>(
    items: I,
    len: usize,
    item_type_name: &'static str,
    fixed_len: Option<usize>,
    depth: usize,
    field: &str,
    type_name: &str,
) -> Result<Derived, ReflectSettingsError>
where
    I: IntoIterator<Item = &'a dyn Reflect>,
{
    if depth > prism_settings::MAX_VALUE_SHAPE_DEPTH
        || len > MAX_REFLECT_SETTING_COLLECTION_LEN
        || fixed_len.is_some_and(|expected| expected != len)
    {
        return Err(ReflectSettingsError::UnsupportedValue {
            field: field.to_owned(),
            type_name: type_name.to_owned(),
        });
    }

    let mut values = Vec::with_capacity(len);
    let mut element_type = None;
    let mut enum_options = Vec::new();
    let mut enum_payloads = Vec::new();
    let mut enum_schemas = Vec::new();
    for item in items {
        let (value, item_shape, item_enum_options, item_payloads, item_schemas) =
            from_reflect_value(item, depth + 1, field, type_name)?;
        if let Some(expected) = &element_type {
            if *expected != item_shape {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: type_name.to_owned(),
                });
            }
        } else {
            element_type = Some(item_shape);
        }
        if !item_enum_options.is_empty() {
            merge_enum_options(&mut enum_options, item_enum_options, field, type_name)?;
        }
        merge_payload_schemas(&mut enum_payloads, item_payloads, field, type_name)?;
        merge_enum_schemas(&mut enum_schemas, item_schemas, field, type_name)?;
        values.push(value);
    }

    let element_shape = element_type
        .or_else(|| scalar_value_type_from_name(item_type_name).map(ValueShape::scalar))
        .ok_or_else(|| ReflectSettingsError::UnsupportedValue {
            field: field.to_owned(),
            type_name: type_name.to_owned(),
        })?;
    let shape = match fixed_len {
        Some(length) => ValueShape::fixed_array(element_shape, length),
        None => ValueShape::bounded_array(element_shape, MAX_REFLECT_SETTING_COLLECTION_LEN),
    };
    Ok((
        Value::Array(values),
        shape,
        enum_options,
        enum_payloads,
        enum_schemas,
    ))
}

fn from_reflect_struct(
    fields: &dyn Struct,
    info: &prism_reflect::StructInfo,
    depth: usize,
    field: &str,
    type_name: &str,
) -> Result<Derived, ReflectSettingsError> {
    if depth > prism_settings::MAX_VALUE_SHAPE_DEPTH
        || info.field_count() > MAX_REFLECT_SETTING_COLLECTION_LEN
    {
        return Err(ReflectSettingsError::UnsupportedValue {
            field: field.to_owned(),
            type_name: type_name.to_owned(),
        });
    }
    let mut values = BTreeMap::new();
    let mut shapes = BTreeMap::new();
    let mut enum_options = Vec::new();
    let mut enum_payloads = Vec::new();
    let mut enum_schemas = Vec::new();
    for descriptor in info.fields() {
        let reflected_value = fields.field(descriptor.name()).ok_or_else(|| {
            ReflectSettingsError::MissingDefaultField {
                type_name: type_name.to_owned(),
                field: descriptor.name().to_owned(),
            }
        })?;
        let (value, shape, options, payloads, schemas) =
            from_reflect_value(reflected_value, depth + 1, field, type_name)?;
        merge_enum_options(&mut enum_options, options, field, type_name)?;
        merge_payload_schemas(&mut enum_payloads, payloads, field, type_name)?;
        merge_enum_schemas(&mut enum_schemas, schemas, field, type_name)?;
        values.insert(descriptor.name().to_owned(), value);
        shapes.insert(descriptor.name().to_owned(), shape);
    }
    Ok((
        Value::Table(values),
        ValueShape::table(shapes),
        enum_options,
        enum_payloads,
        enum_schemas,
    ))
}

fn from_reflect_map(
    map: &dyn Map,
    key_type_name: &str,
    value_type_name: &str,
    depth: usize,
    field: &str,
    type_name: &str,
) -> Result<Derived, ReflectSettingsError> {
    if depth > prism_settings::MAX_VALUE_SHAPE_DEPTH
        || map.len() > MAX_REFLECT_SETTING_COLLECTION_LEN
        || !matches!(
            key_type_name,
            "alloc::string::String" | "std::string::String"
        )
    {
        return Err(ReflectSettingsError::UnsupportedValue {
            field: field.to_owned(),
            type_name: type_name.to_owned(),
        });
    }
    let mut values = BTreeMap::new();
    let mut value_shape = None;
    let mut enum_options = Vec::new();
    let mut enum_payloads = Vec::new();
    let mut enum_schemas = Vec::new();
    for (key, reflected_value) in map.iter_reflect() {
        let Some(Value::String(key)) = from_reflect_scalar(key) else {
            return Err(ReflectSettingsError::UnsupportedValue {
                field: field.to_owned(),
                type_name: type_name.to_owned(),
            });
        };
        let (value, shape, options, payloads, schemas) =
            from_reflect_value(reflected_value, depth + 1, field, type_name)?;
        if let Some(expected) = &value_shape {
            if *expected != shape {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: type_name.to_owned(),
                });
            }
        } else {
            value_shape = Some(shape);
        }
        merge_enum_options(&mut enum_options, options, field, type_name)?;
        merge_payload_schemas(&mut enum_payloads, payloads, field, type_name)?;
        merge_enum_schemas(&mut enum_schemas, schemas, field, type_name)?;
        values.insert(key, value);
    }
    let value_shape = value_shape
        .or_else(|| scalar_value_type_from_name(value_type_name).map(ValueShape::scalar))
        .ok_or_else(|| ReflectSettingsError::UnsupportedValue {
            field: field.to_owned(),
            type_name: type_name.to_owned(),
        })?;
    Ok((
        Value::Table(values),
        ValueShape::map(value_shape, MAX_REFLECT_SETTING_COLLECTION_LEN),
        enum_options,
        enum_payloads,
        enum_schemas,
    ))
}

fn from_reflect_value(
    value: &dyn Reflect,
    depth: usize,
    field: &str,
    field_type_name: &str,
) -> Result<Derived, ReflectSettingsError> {
    if depth > prism_settings::MAX_VALUE_SHAPE_DEPTH {
        return Err(ReflectSettingsError::UnsupportedValue {
            field: field.to_owned(),
            type_name: field_type_name.to_owned(),
        });
    }
    match value.reflect_ref() {
        ReflectRef::List(list) => {
            let TypeInfo::List(info) = value.type_info() else {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: field_type_name.to_owned(),
                });
            };
            from_reflect_sequence(
                list.iter_reflect(),
                list.len(),
                info.item_type_name(),
                None,
                depth,
                field,
                field_type_name,
            )
        }
        ReflectRef::Array(array) => {
            let TypeInfo::Array(info) = value.type_info() else {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: field_type_name.to_owned(),
                });
            };
            from_reflect_sequence(
                array.iter_reflect(),
                array.len(),
                info.item_type_name(),
                Some(info.capacity()),
                depth,
                field,
                field_type_name,
            )
        }
        ReflectRef::Struct(fields) => {
            let TypeInfo::Struct(info) = value.type_info() else {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: field_type_name.to_owned(),
                });
            };
            from_reflect_struct(fields, info, depth, field, field_type_name)
        }
        ReflectRef::Map(map) => {
            let TypeInfo::Map(info) = value.type_info() else {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: field_type_name.to_owned(),
                });
            };
            from_reflect_map(
                map,
                info.key_type_name(),
                info.value_type_name(),
                depth,
                field,
                field_type_name,
            )
        }
        ReflectRef::Enum(active) => {
            let TypeInfo::Enum(enum_info) = value.type_info() else {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: field_type_name.to_owned(),
                });
            };
            from_reflect_enum(active, enum_info, depth, field, field_type_name)
        }
        _ => {
            let scalar = from_reflect_scalar(value).ok_or_else(|| {
                ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: field_type_name.to_owned(),
                }
            })?;
            Ok((
                scalar.clone(),
                ValueShape::scalar(scalar.value_type()),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ))
        }
    }
}

fn from_reflect_enum(
    active: &dyn Enum,
    info: &prism_reflect::EnumInfo,
    depth: usize,
    field: &str,
    type_name: &str,
) -> Result<Derived, ReflectSettingsError> {
    if depth > prism_settings::MAX_VALUE_SHAPE_DEPTH {
        return Err(ReflectSettingsError::UnsupportedValue {
            field: field.to_owned(),
            type_name: type_name.to_owned(),
        });
    }
    if info
        .variants()
        .iter()
        .all(|variant| matches!(variant.kind(), VariantKind::Unit))
    {
        let options = info
            .variants()
            .iter()
            .map(|variant| EnumOption::from_stable_name(variant.name()))
            .collect::<Vec<_>>();
        let option = options
            .iter()
            .find(|option| option.stable_name == active.variant_name())
            .ok_or_else(|| ReflectSettingsError::UnsupportedValue {
                field: field.to_owned(),
                type_name: type_name.to_owned(),
            })?;
        return Ok((
            Value::Enum(option.id),
            ValueShape::scalar(prism_settings::ValueType::Enum),
            options,
            Vec::new(),
            Vec::new(),
        ));
    }
    let (all_options, payloads, schemas) = enum_schema(info, depth, field, type_name)?;
    let enum_type = prism_settings::EnumTypeId::from_stable_name(info.type_name());
    let active_option =
        EnumOption::from_stable_name(&typed_variant_name(info, active.variant_name()));
    let value = match active.variant_type() {
        VariantType::Unit => Value::EnumTyped {
            enum_type,
            id: active_option.id,
        },
        VariantType::Tuple => Value::EnumPayloadTyped {
            enum_type,
            id: active_option.id,
            payload: prism_settings::EnumPayload::Tuple(
                (0..active.field_count())
                    .map(|index| {
                        active
                            .field_at(index)
                            .ok_or_else(|| ReflectSettingsError::UnsupportedValue {
                                field: field.to_owned(),
                                type_name: type_name.to_owned(),
                            })
                            .and_then(|child| {
                                from_reflect_value(child, depth + 1, field, type_name)
                                    .map(|item| item.0)
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        },
        VariantType::Struct => {
            let VariantKind::Struct(fields) = info
                .variant(active.variant_name())
                .ok_or_else(|| ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: type_name.to_owned(),
                })?
                .kind()
            else {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: type_name.to_owned(),
                });
            };
            let mut values = BTreeMap::new();
            for descriptor in fields {
                let child = active.field(descriptor.name()).ok_or_else(|| {
                    ReflectSettingsError::UnsupportedValue {
                        field: field.to_owned(),
                        type_name: type_name.to_owned(),
                    }
                })?;
                values.insert(
                    descriptor.name().to_owned(),
                    from_reflect_value(child, depth + 1, field, type_name)?.0,
                );
            }
            Value::EnumPayloadTyped {
                enum_type,
                id: active_option.id,
                payload: prism_settings::EnumPayload::Struct(values),
            }
        }
        _ => {
            return Err(ReflectSettingsError::UnsupportedValue {
                field: field.to_owned(),
                type_name: type_name.to_owned(),
            });
        }
    };
    Ok((
        value,
        ValueShape::Enum(enum_type),
        all_options,
        payloads,
        schemas,
    ))
}

fn enum_schema(
    info: &prism_reflect::EnumInfo,
    depth: usize,
    field: &str,
    type_name: &str,
) -> Result<(Vec<EnumOption>, Vec<EnumPayloadSchema>, Vec<EnumSchema>), ReflectSettingsError> {
    let mut options = info
        .variants()
        .iter()
        .map(|variant| EnumOption::from_stable_name(&typed_variant_name(info, variant.name())))
        .collect::<Vec<_>>();
    let mut payloads = Vec::new();
    let mut schemas = Vec::new();
    for variant in info.variants() {
        match variant.kind() {
            VariantKind::Unit => {}
            VariantKind::Tuple(fields) => {
                let mut shapes = Vec::with_capacity(fields.len());
                for descriptor in fields {
                    let (shape, nested_options, nested_payloads, nested_schemas) =
                        shape_from_type_info(
                            descriptor.type_info(),
                            descriptor.type_name(),
                            depth + 1,
                            field,
                            type_name,
                        )?;
                    shapes.push(shape);
                    merge_enum_options(&mut options, nested_options, field, type_name)?;
                    merge_payload_schemas(&mut payloads, nested_payloads, field, type_name)?;
                    merge_enum_schemas(&mut schemas, nested_schemas, field, type_name)?;
                }
                payloads.push(EnumPayloadSchema {
                    id: EnumOption::from_stable_name(&typed_variant_name(info, variant.name())).id,
                    shape: prism_settings::EnumPayloadShape::Tuple(shapes),
                });
            }
            VariantKind::Struct(fields) => {
                let mut shapes = BTreeMap::new();
                for descriptor in fields {
                    let (shape, nested_options, nested_payloads, nested_schemas) =
                        shape_from_type_info(
                            descriptor.type_info(),
                            descriptor.type_name(),
                            depth + 1,
                            field,
                            type_name,
                        )?;
                    shapes.insert(descriptor.name().to_owned(), shape);
                    merge_enum_options(&mut options, nested_options, field, type_name)?;
                    merge_payload_schemas(&mut payloads, nested_payloads, field, type_name)?;
                    merge_enum_schemas(&mut schemas, nested_schemas, field, type_name)?;
                }
                payloads.push(EnumPayloadSchema {
                    id: EnumOption::from_stable_name(&typed_variant_name(info, variant.name())).id,
                    shape: prism_settings::EnumPayloadShape::Struct(shapes),
                });
            }
            _ => {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: type_name.to_owned(),
                });
            }
        }
    }
    schemas.push(EnumSchema {
        id: prism_settings::EnumTypeId::from_stable_name(info.type_name()),
        stable_name: info.type_name().to_owned(),
        variants: info
            .variants()
            .iter()
            .map(|variant| EnumVariantSchema {
                id: EnumOption::from_stable_name(&typed_variant_name(info, variant.name())).id,
                stable_name: typed_variant_name(info, variant.name()),
                payload: payloads
                    .iter()
                    .find(|payload| {
                        payload.id
                            == EnumOption::from_stable_name(&typed_variant_name(
                                info,
                                variant.name(),
                            ))
                            .id
                    })
                    .map(|payload| payload.shape.clone()),
            })
            .collect(),
    });
    Ok((options, payloads, schemas))
}

fn typed_variant_name(info: &prism_reflect::EnumInfo, variant: &str) -> String {
    format!("{}::{}", info.type_name(), variant)
}

fn shape_from_type_info(
    info: Option<&'static TypeInfo>,
    type_name: &'static str,
    depth: usize,
    field: &str,
    owner_type: &str,
) -> Result<
    (
        ValueShape,
        Vec<EnumOption>,
        Vec<EnumPayloadSchema>,
        Vec<EnumSchema>,
    ),
    ReflectSettingsError,
> {
    if depth > prism_settings::MAX_VALUE_SHAPE_DEPTH {
        return Err(ReflectSettingsError::UnsupportedValue {
            field: field.to_owned(),
            type_name: owner_type.to_owned(),
        });
    }
    let Some(info) = info else {
        return scalar_value_type_from_name(type_name)
            .map(|ty| (ValueShape::scalar(ty), Vec::new(), Vec::new(), Vec::new()))
            .ok_or_else(|| ReflectSettingsError::UnsupportedValue {
                field: field.to_owned(),
                type_name: owner_type.to_owned(),
            });
    };
    match info {
        TypeInfo::Value(_) => scalar_value_type_from_name(type_name)
            .map(|ty| (ValueShape::scalar(ty), Vec::new(), Vec::new(), Vec::new()))
            .ok_or_else(|| ReflectSettingsError::UnsupportedValue {
                field: field.to_owned(),
                type_name: owner_type.to_owned(),
            }),
        TypeInfo::Enum(enum_info) => {
            if enum_info
                .variants()
                .iter()
                .all(|variant| matches!(variant.kind(), VariantKind::Unit))
            {
                let options = enum_info
                    .variants()
                    .iter()
                    .map(|variant| EnumOption::from_stable_name(variant.name()))
                    .collect::<Vec<_>>();
                return Ok((
                    ValueShape::scalar(prism_settings::ValueType::Enum),
                    options,
                    Vec::new(),
                    Vec::new(),
                ));
            }
            let (options, payloads, schemas) = enum_schema(enum_info, depth, field, owner_type)?;
            Ok((
                ValueShape::Enum(prism_settings::EnumTypeId::from_stable_name(
                    enum_info.type_name(),
                )),
                options,
                payloads,
                schemas,
            ))
        }
        TypeInfo::List(list) => {
            let (element, options, payloads, schemas) = shape_from_type_info(
                list.item_type_info(),
                list.item_type_name(),
                depth + 1,
                field,
                owner_type,
            )?;
            Ok((
                ValueShape::bounded_array(element, MAX_REFLECT_SETTING_COLLECTION_LEN),
                options,
                payloads,
                schemas,
            ))
        }
        TypeInfo::Array(array) => {
            let (element, options, payloads, schemas) = shape_from_type_info(
                array.item_type_info(),
                array.item_type_name(),
                depth + 1,
                field,
                owner_type,
            )?;
            Ok((
                ValueShape::fixed_array(element, array.capacity()),
                options,
                payloads,
                schemas,
            ))
        }
        TypeInfo::Struct(struct_info) => {
            let mut fields = BTreeMap::new();
            let mut options = Vec::new();
            let mut payloads = Vec::new();
            let mut schemas = Vec::new();
            for descriptor in struct_info.fields() {
                let (shape, child_options, child_payloads, child_schemas) = shape_from_type_info(
                    descriptor.type_info(),
                    descriptor.type_name(),
                    depth + 1,
                    field,
                    owner_type,
                )?;
                fields.insert(descriptor.name().to_owned(), shape);
                merge_enum_options(&mut options, child_options, field, owner_type)?;
                merge_payload_schemas(&mut payloads, child_payloads, field, owner_type)?;
                merge_enum_schemas(&mut schemas, child_schemas, field, owner_type)?;
            }
            Ok((ValueShape::table(fields), options, payloads, schemas))
        }
        TypeInfo::TupleStruct(tuple_info) => {
            let _ = tuple_info;
            Err(ReflectSettingsError::UnsupportedValue {
                field: field.to_owned(),
                type_name: owner_type.to_owned(),
            })
        }
        TypeInfo::Map(map)
            if map.key_type_name() == "alloc::string::String"
                || map.key_type_name() == "std::string::String" =>
        {
            let (shape, options, payloads, schemas) = shape_from_type_info(
                map.value_type_info(),
                map.value_type_name(),
                depth + 1,
                field,
                owner_type,
            )?;
            Ok((
                ValueShape::map(shape, MAX_REFLECT_SETTING_COLLECTION_LEN),
                options,
                payloads,
                schemas,
            ))
        }
        TypeInfo::Set(set) => {
            let (shape, options, payloads, schemas) = shape_from_type_info(
                set.value_type_info(),
                set.value_type_name(),
                depth + 1,
                field,
                owner_type,
            )?;
            Ok((
                ValueShape::set(shape, MAX_REFLECT_SETTING_COLLECTION_LEN),
                options,
                payloads,
                schemas,
            ))
        }
        _ => Err(ReflectSettingsError::UnsupportedValue {
            field: field.to_owned(),
            type_name: owner_type.to_owned(),
        }),
    }
}

fn merge_enum_options(
    target: &mut Vec<EnumOption>,
    incoming: Vec<EnumOption>,
    field: &str,
    type_name: &str,
) -> Result<(), ReflectSettingsError> {
    for option in incoming {
        if let Some(existing) = target.iter().find(|existing| existing.id == option.id) {
            if existing.stable_name != option.stable_name {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: type_name.to_owned(),
                });
            }
        } else {
            target.push(option);
        }
    }
    Ok(())
}

fn merge_payload_schemas(
    target: &mut Vec<EnumPayloadSchema>,
    incoming: Vec<EnumPayloadSchema>,
    field: &str,
    type_name: &str,
) -> Result<(), ReflectSettingsError> {
    for schema in incoming {
        if let Some(existing) = target.iter().find(|existing| existing.id == schema.id) {
            if existing.shape != schema.shape {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: type_name.to_owned(),
                });
            }
        } else {
            target.push(schema);
        }
    }
    Ok(())
}

fn merge_enum_schemas(
    target: &mut Vec<EnumSchema>,
    incoming: Vec<EnumSchema>,
    field: &str,
    type_name: &str,
) -> Result<(), ReflectSettingsError> {
    for schema in incoming {
        if let Some(existing) = target.iter().find(|existing| existing.id == schema.id) {
            if existing != &schema {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: type_name.to_owned(),
                });
            }
        } else {
            target.push(schema);
        }
    }
    Ok(())
}

fn scalar_value_type_from_name(type_name: &str) -> Option<prism_settings::ValueType> {
    match type_name {
        "bool" => Some(prism_settings::ValueType::Bool),
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64" | "u128"
        | "usize" => Some(prism_settings::ValueType::Int),
        "f32" | "f64" => Some(prism_settings::ValueType::Float),
        "alloc::string::String" | "std::string::String" => Some(prism_settings::ValueType::String),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_reflect::Reflect;
    use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

    #[derive(Reflect, Default)]
    struct RenderConfig {
        /// Resolution multiplier.
        #[reflect(setting, rename = "Resolution Scale", range(0.25..=2.0), default = 1.0, category = "Quality")]
        scale: f32,
        #[reflect(setting, default = 60, readonly)]
        max_fps: u32,
        #[reflect(skip)]
        cache: u32,
        #[reflect(hidden = true)]
        internal: bool,
    }

    #[test]
    fn derives_metadata_defaults_and_filters_hidden_fields() {
        let schema = derive_settings_registry::<RenderConfig>("render", "prism_app").unwrap();
        assert_eq!(schema.len(), 2);
        let (_, scale) = schema.get(&"render.scale".into()).unwrap();
        assert_eq!(scale.default, Value::Float(1.0));
        assert_eq!(scale.display_name.as_deref(), Some("Resolution Scale"));
        assert_eq!(scale.key.as_str(), "render.scale");
        assert_eq!(scale.constraints.min, Some(0.25));
        assert_eq!(scale.constraints.max, Some(2.0));
        assert_eq!(scale.description.as_deref(), Some("Resolution multiplier."));
        assert_eq!(scale.category.as_deref(), Some("Quality"));
        let (_, fps) = schema.get(&"render.max_fps".into()).unwrap();
        assert_eq!(fps.default, Value::Int(60));
        assert!(fps.flags.read_only);
        assert!(schema.get(&"render.cache".into()).is_none());
        assert!(schema.get(&"render.internal".into()).is_none());
    }

    #[test]
    fn derives_on_violation_policy_and_registry_applies_it() {
        #[derive(Reflect, Default)]
        struct PolicyConfig {
            #[reflect(setting, range(0.0..=1.0), on_violation = "clamp")]
            clamped: f32,
            #[reflect(setting, range(0.0..=1.0), on_violation = "reject")]
            rejected: f32,
        }

        let schema = derive_settings_registry::<PolicyConfig>("policy", "prism_app").unwrap();
        let (clamped_id, clamped) = schema.get(&"policy.clamped".into()).unwrap();
        let (rejected_id, rejected) = schema.get(&"policy.rejected".into()).unwrap();
        assert_eq!(clamped.on_violation, prism_settings::ViolationPolicy::Clamp);
        assert_eq!(
            rejected.on_violation,
            prism_settings::ViolationPolicy::Reject
        );
        assert_eq!(
            schema
                .validate_value(clamped_id, Value::Float(2.0))
                .unwrap(),
            Value::Float(1.0)
        );
        assert!(matches!(
            schema.validate_value(rejected_id, Value::Float(2.0)),
            Err(SettingsError::OutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_metadata_type_mismatch_and_supports_data_enums() {
        #[derive(Reflect, Default)]
        struct WrongDefault {
            #[reflect(setting, default = "bad")]
            value: i32,
        }
        assert!(matches!(
            derive_settings_registry::<WrongDefault>("test", "test"),
            Err(ReflectSettingsError::MetadataTypeMismatch { .. })
        ));

        #[derive(Reflect, Default)]
        struct Unsupported {
            #[reflect(setting)]
            value: Option<u32>,
        }
        let schema = derive_settings_registry::<Unsupported>("test", "test").unwrap();
        let (_, definition) = schema.get(&"test.value".into()).unwrap();
        assert_eq!(definition.ty, prism_settings::ValueType::Enum);
        assert_eq!(definition.enum_schemas.len(), 1);
        assert!(matches!(definition.default, Value::EnumTyped { .. }));
    }

    #[test]
    fn includes_only_explicitly_marked_settings_and_allows_empty_schema() {
        #[derive(Reflect, Default)]
        struct MixedConfig {
            #[reflect(setting)]
            enabled: bool,
            ordinary_reflected_scalar: u32,
            #[reflect(category = "Editor only")]
            editor_state: String,
        }

        let schema = derive_settings_registry::<MixedConfig>("mixed", "prism_app").unwrap();
        assert_eq!(schema.len(), 1);
        assert!(schema.get(&"mixed.enabled".into()).is_some());
        assert!(schema
            .get(&"mixed.ordinary_reflected_scalar".into())
            .is_none());
        assert!(schema.get(&"mixed.editor_state".into()).is_none());

        #[derive(Reflect, Default)]
        struct NoSettings {
            ordinary: u64,
        }
        let empty = derive_settings_registry::<NoSettings>("empty", "prism_app").unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn derives_unit_enum_variants_by_stable_name_not_ordinal() {
        #[derive(Reflect, Default)]
        enum Quality {
            Low,
            #[default]
            Medium,
            High,
        }

        #[derive(Reflect, Default)]
        struct GraphicsSettings {
            #[reflect(setting, default = "High")]
            quality: Quality,
        }

        let schema = derive_settings_registry::<GraphicsSettings>("graphics", "prism_app").unwrap();
        let (_, definition) = schema.get(&"graphics.quality".into()).unwrap();
        assert_eq!(definition.ty, prism_settings::ValueType::Enum);
        assert_eq!(definition.enum_options.len(), 3);
        let high = definition
            .enum_options
            .iter()
            .find(|option| option.stable_name == "High")
            .unwrap();
        assert_eq!(definition.default, Value::Enum(high.id));
        assert_eq!(high.id, prism_settings::EnumId::from_stable_name("High"));
    }

    #[test]
    fn derives_bounded_vec_and_fixed_array_settings_with_registry_validation() {
        #[derive(Reflect, Default)]
        struct CollectionSettings {
            #[reflect(setting)]
            recent: Vec<i32>,
            #[reflect(setting)]
            offsets: [f32; 2],
        }

        let schema =
            derive_settings_registry::<CollectionSettings>("collections", "prism_app").unwrap();
        let (recent_id, recent) = schema.get(&"collections.recent".into()).unwrap();
        assert!(matches!(
            recent.shape,
            ValueShape::Array {
                min_len: 0,
                max_len: MAX_REFLECT_SETTING_COLLECTION_LEN,
                ..
            }
        ));
        assert_eq!(
            schema
                .validate_value(recent_id, Value::Array(vec![Value::Int(3), Value::Int(5)]))
                .unwrap(),
            Value::Array(vec![Value::Int(3), Value::Int(5)])
        );
        assert!(matches!(
            schema.validate_value(
                recent_id,
                Value::Array(vec![Value::Int(3), Value::Bool(true)])
            ),
            Err(SettingsError::TypeMismatch { .. })
        ));
        assert!(matches!(
            schema.validate_value(
                recent_id,
                Value::Array(vec![Value::Int(0); MAX_REFLECT_SETTING_COLLECTION_LEN + 1])
            ),
            Err(SettingsError::TooLong { .. })
        ));

        let (offsets_id, offsets) = schema.get(&"collections.offsets".into()).unwrap();
        assert_eq!(
            offsets.shape,
            ValueShape::fixed_array(ValueShape::scalar(prism_settings::ValueType::Float), 2)
        );
        assert!(schema
            .validate_value(offsets_id, Value::Array(vec![Value::Float(1.0)]))
            .is_err());
        assert!(schema
            .validate_value(
                offsets_id,
                Value::Array(vec![Value::Float(1.0), Value::Int(2)])
            )
            .is_err());
    }

    #[test]
    fn derives_nested_unit_enum_collections_with_stable_options() {
        #[derive(Reflect, Default)]
        enum Quality {
            Low,
            #[default]
            Medium,
            High,
        }

        #[derive(Reflect)]
        struct GraphicsSettings {
            #[reflect(setting)]
            history: Vec<Vec<Quality>>,
        }

        impl Default for GraphicsSettings {
            fn default() -> Self {
                Self {
                    history: vec![vec![Quality::High]],
                }
            }
        }

        let schema = derive_settings_registry::<GraphicsSettings>("graphics", "prism_app").unwrap();
        let (id, definition) = schema.get(&"graphics.history".into()).unwrap();
        assert_eq!(definition.enum_options.len(), 3);
        assert!(matches!(
            &definition.shape,
            ValueShape::Array { element, .. }
                if matches!(element.as_ref(), ValueShape::Array { element, .. }
                    if element.as_ref() == &ValueShape::scalar(prism_settings::ValueType::Enum))
        ));
        assert_eq!(
            schema
                .validate_value(
                    id,
                    Value::Array(vec![Value::Array(vec![Value::Enum(
                        prism_settings::EnumId::from_stable_name("High")
                    )])]),
                )
                .unwrap(),
            Value::Array(vec![Value::Array(vec![Value::Enum(
                prism_settings::EnumId::from_stable_name("High")
            )])]),
        );
    }

    #[test]
    fn nested_reflected_collections_preserve_recursive_shape_and_fail_closed_when_empty() {
        #[derive(Reflect)]
        struct NestedSettings {
            #[reflect(setting)]
            matrix: Vec<Vec<i32>>,
            #[reflect(setting)]
            fixed: [[i32; 2]; 2],
        }

        impl Default for NestedSettings {
            fn default() -> Self {
                Self {
                    matrix: vec![vec![1, 2], vec![3, 4]],
                    fixed: [[5, 6], [7, 8]],
                }
            }
        }

        let schema = derive_settings_registry::<NestedSettings>("nested", "prism_app").unwrap();
        let (matrix_id, matrix) = schema.get(&"nested.matrix".into()).unwrap();
        assert!(matches!(
            &matrix.shape,
            ValueShape::Array { element, .. }
                if matches!(**element, ValueShape::Array { .. })
        ));
        let candidate = Value::Array(vec![
            Value::Array(vec![Value::Int(10)]),
            Value::Array(vec![Value::Int(11)]),
        ]);
        assert_eq!(
            schema.validate_value(matrix_id, candidate.clone()).unwrap(),
            candidate
        );
        assert!(schema
            .validate_value(
                matrix_id,
                Value::Array(vec![
                    Value::Array(vec![Value::Int(1)]),
                    Value::Array(vec![Value::Bool(true)])
                ]),
            )
            .is_err());

        let (fixed_id, fixed) = schema.get(&"nested.fixed".into()).unwrap();
        assert!(matches!(fixed.shape, ValueShape::Array { .. }));
        assert!(schema
            .validate_value(
                fixed_id,
                Value::Array(vec![Value::Array(vec![Value::Int(1)])])
            )
            .is_err());

        #[derive(Reflect, Default)]
        struct EmptyNested {
            #[reflect(setting)]
            matrix: Vec<Vec<i32>>,
        }
        assert!(matches!(
            derive_settings_registry::<EmptyNested>("nested", "prism_app"),
            Err(ReflectSettingsError::UnsupportedValue { field, .. }) if field == "matrix"
        ));
    }

    #[test]
    fn recursive_array_attribute_values_convert_without_losing_shape() {
        let attribute = AttributeValue::array(vec![
            AttributeValue::array(vec![AttributeValue::Int(1), AttributeValue::Int(2)]),
            AttributeValue::array(vec![AttributeValue::Int(3)]),
        ]);
        assert_eq!(
            from_attribute(&attribute),
            Some(Value::Array(vec![
                Value::Array(vec![Value::Int(1), Value::Int(2)]),
                Value::Array(vec![Value::Int(3)]),
            ]))
        );
    }

    #[test]
    fn derives_named_structs_and_string_keyed_maps_with_stable_table_values() {
        #[derive(Reflect, Default)]
        struct Color {
            red: u8,
            green: u8,
        }

        #[derive(Reflect)]
        struct MapSettings {
            #[reflect(setting)]
            color: Color,
            #[reflect(setting)]
            ordered: BTreeMap<String, i32>,
            #[reflect(setting)]
            hashed: HashMap<String, f32>,
        }

        impl Default for MapSettings {
            fn default() -> Self {
                Self {
                    color: Color { red: 10, green: 20 },
                    ordered: BTreeMap::from([("zeta".to_owned(), 2), ("alpha".to_owned(), 1)]),
                    hashed: HashMap::from([("gpu".to_owned(), 1.5)]),
                }
            }
        }

        let schema = derive_settings_registry::<MapSettings>("maps", "prism_app").unwrap();

        let (color_id, color) = schema.get(&"maps.color".into()).unwrap();
        assert_eq!(
            color.default,
            Value::Table(BTreeMap::from([
                ("green".to_owned(), Value::Int(20)),
                ("red".to_owned(), Value::Int(10)),
            ]))
        );
        assert!(matches!(color.shape, ValueShape::Table(_)));
        assert!(schema
            .validate_value(
                color_id,
                Value::Table(BTreeMap::from([
                    ("red".to_owned(), Value::Int(1)),
                    ("green".to_owned(), Value::Int(2)),
                ])),
            )
            .is_ok());

        let (ordered_id, ordered) = schema.get(&"maps.ordered".into()).unwrap();
        assert!(matches!(ordered.shape, ValueShape::Map { .. }));
        assert!(schema
            .validate_value(
                ordered_id,
                Value::Table(BTreeMap::from([
                    ("alpha".to_owned(), Value::Int(3)),
                    ("beta".to_owned(), Value::Int(4)),
                ])),
            )
            .is_ok());

        let (hashed_id, hashed) = schema.get(&"maps.hashed".into()).unwrap();
        assert!(matches!(hashed.shape, ValueShape::Map { .. }));
        assert!(schema
            .validate_value(
                hashed_id,
                Value::Table(BTreeMap::from([("gpu".to_owned(), Value::Float(2.0))])),
            )
            .is_ok());
    }

    #[test]
    fn map_derivation_rejects_non_string_keys_and_non_homogeneous_values() {
        #[derive(Reflect, Default)]
        struct NonStringKey {
            #[reflect(setting)]
            values: BTreeMap<i32, i32>,
        }
        assert!(matches!(
            derive_settings_registry::<NonStringKey>("maps", "prism_app"),
            Err(ReflectSettingsError::UnsupportedValue { field, .. }) if field == "values"
        ));

        #[derive(Reflect, Default)]
        struct EmptyNestedMap {
            #[reflect(setting)]
            values: BTreeMap<String, Vec<i32>>,
        }
        assert!(matches!(
            derive_settings_registry::<EmptyNestedMap>("maps", "prism_app"),
            Err(ReflectSettingsError::UnsupportedValue { field, .. }) if field == "values"
        ));
    }

    #[test]
    fn derives_sets_with_value_order_not_hash_iteration_or_debug_text() {
        #[derive(Reflect)]
        struct SetSettings {
            #[reflect(setting)]
            ordered: BTreeSet<i32>,
            #[reflect(setting)]
            hashed: HashSet<String>,
        }
        impl Default for SetSettings {
            fn default() -> Self {
                Self {
                    ordered: [9, 2, 4].into_iter().collect(),
                    hashed: ["z".to_owned(), "a".to_owned()].into_iter().collect(),
                }
            }
        }
        let schema = derive_settings_registry::<SetSettings>("sets", "prism_app").unwrap();
        let (_, ordered) = schema.get(&"sets.ordered".into()).unwrap();
        assert_eq!(
            ordered.default,
            Value::Array(vec![Value::Int(2), Value::Int(4), Value::Int(9)])
        );
        assert!(matches!(ordered.shape, ValueShape::Set { .. }));
        let (_, hashed) = schema.get(&"sets.hashed".into()).unwrap();
        assert_eq!(
            hashed.default,
            Value::Array(vec![Value::String("a".into()), Value::String("z".into())])
        );
    }

    #[test]
    fn derives_required_flag_from_reflect_metadata() {
        #[derive(Reflect, Default)]
        struct RequiredSetting {
            #[reflect(setting, required)]
            value: i32,
        }
        let schema = derive_settings_registry::<RequiredSetting>("required", "prism_app").unwrap();
        let (_, definition) = schema.get(&"required.value".into()).unwrap();
        assert!(definition.flags.required);
    }

    #[test]
    fn derives_settings_source_visibility_and_resolver_policies() {
        #[derive(Reflect, Default)]
        struct PolicySetting {
            #[reflect(
                setting,
                scope = "machine",
                override_scope = "project",
                precedence = "locked",
                domains = "editor, build",
                project_overridable = false,
                bootstrap,
                execution_class = true
            )]
            value: bool,
        }

        let schema = derive_settings_registry::<PolicySetting>("policy", "prism_app").unwrap();
        let (_, definition) = schema.get(&"policy.value".into()).unwrap();
        assert_eq!(definition.scope, Scope::Machine);
        assert_eq!(definition.override_scope, OverrideScope::Project);
        assert_eq!(definition.precedence, PrecedenceProfile::Locked);
        assert!(definition.domains.contains(Domain::Editor));
        assert!(definition.domains.contains(Domain::Build));
        assert!(!definition.domains.contains(Domain::Runtime));
        assert!(!definition.flags.project_overridable);
        assert!(definition.flags.bootstrap);
        assert!(definition.flags.execution_class);
    }

    #[test]
    fn rejects_unknown_settings_policy_values() {
        #[derive(Reflect, Default)]
        struct InvalidPolicy {
            #[reflect(setting, precedence = "sometimes")]
            value: bool,
        }

        assert!(matches!(
            derive_settings_registry::<InvalidPolicy>("invalid", "prism_app"),
            Err(ReflectSettingsError::MetadataTypeMismatch { field }) if field == "value"
        ));
    }

    #[test]
    fn derives_string_and_collection_size_constraints() {
        #[derive(Reflect, Default)]
        struct BoundedSettings {
            #[reflect(setting, max_string_bytes = 24)]
            label: String,
            #[reflect(setting, max_array_len = 3)]
            samples: Vec<i32>,
        }

        let schema = derive_settings_registry::<BoundedSettings>("bounded", "prism_app").unwrap();
        let (label_id, label) = schema.get(&"bounded.label".into()).unwrap();
        assert_eq!(label.constraints.max_string_bytes, Some(24));
        assert!(schema
            .validate_value(label_id, Value::String("x".repeat(25)))
            .is_err());
        let (samples_id, samples) = schema.get(&"bounded.samples".into()).unwrap();
        assert_eq!(samples.constraints.max_array_len, Some(3));
        assert!(schema
            .validate_value(
                samples_id,
                Value::Array(vec![
                    Value::Int(1),
                    Value::Int(2),
                    Value::Int(3),
                    Value::Int(4)
                ])
            )
            .is_err());
    }

    #[test]
    fn derives_deprecated_key_aliases_and_rejects_empty_aliases() {
        #[derive(Reflect, Default)]
        struct RenamedSetting {
            #[reflect(setting, aliases = "legacy.audio.volume,old.audio.volume")]
            volume: f32,
        }

        let schema = derive_settings_registry::<RenamedSetting>("audio", "prism_app").unwrap();
        let canonical = schema.id(&"audio.volume".into()).unwrap();
        assert_eq!(schema.id(&"legacy.audio.volume".into()), Some(canonical));
        assert_eq!(schema.id(&"old.audio.volume".into()), Some(canonical));

        #[derive(Reflect, Default)]
        struct EmptyAlias {
            #[reflect(setting, aliases = "legacy.value,")]
            value: bool,
        }
        assert!(matches!(
            derive_settings_registry::<EmptyAlias>("invalid", "prism_app"),
            Err(ReflectSettingsError::MetadataTypeMismatch { field }) if field == "value"
        ));
    }
}
