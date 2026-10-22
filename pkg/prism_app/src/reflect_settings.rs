//! Derive the canonical settings schema from reflected configuration structs.
//!
//! This opt-in adapter keeps `prism_reflect` metadata as the authoring input and
//! `prism_settings` as the only runtime schema. Unsupported fields are rejected
//! explicitly rather than silently omitted.

use prism_reflect::{
    schema::{AttributeValue, TypeMetadata},
    GetTypeRegistration, Map, Reflect, ReflectRef, Struct, TypeInfo, Typed, VariantKind,
};
use prism_settings::{
    Constraints, RegistryBuilder, SettingDef, SettingsError, SettingsRegistry, Value, ValueShape,
};
use std::{collections::BTreeMap, error::Error, fmt};

const MAX_REFLECT_SETTING_COLLECTION_LEN: usize = 1024;

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
        let (rust_default, shape, enum_options) = match reflected.reflect_ref() {
            ReflectRef::Enum(active) => {
                let TypeInfo::Enum(enum_info) = reflected.type_info() else {
                    return Err(ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    });
                };
                if enum_info
                    .variants()
                    .iter()
                    .any(|variant| !matches!(variant.kind(), VariantKind::Unit))
                {
                    return Err(ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    });
                }
                let enum_options = enum_info
                    .variants()
                    .iter()
                    .map(|variant| prism_settings::EnumOption::from_stable_name(variant.name()))
                    .collect::<Vec<_>>();
                let default = enum_options
                    .iter()
                    .find(|option| option.stable_name == active.variant_name())
                    .ok_or_else(|| ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    })?;
                (
                    Value::Enum(default.id),
                    ValueShape::scalar(prism_settings::ValueType::Enum),
                    enum_options,
                )
            }
            ReflectRef::List(list) => {
                let TypeInfo::List(info) = reflected.type_info() else {
                    return Err(ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    });
                };
                let (value, shape, enum_options) = from_reflect_sequence(
                    list.iter_reflect(),
                    list.len(),
                    info.item_type_name(),
                    None,
                    0,
                    name,
                    descriptor.type_name(),
                )?;
                (value, shape, enum_options)
            }
            ReflectRef::Array(array) => {
                let TypeInfo::Array(info) = reflected.type_info() else {
                    return Err(ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    });
                };
                let (value, shape, enum_options) = from_reflect_sequence(
                    array.iter_reflect(),
                    array.len(),
                    info.item_type_name(),
                    Some(info.capacity()),
                    0,
                    name,
                    descriptor.type_name(),
                )?;
                (value, shape, enum_options)
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
            _ => {
                let value = from_reflect_scalar(reflected).ok_or_else(|| {
                    ReflectSettingsError::UnsupportedValue {
                        field: name.to_owned(),
                        type_name: descriptor.type_name().to_owned(),
                    }
                })?;
                let shape = ValueShape::scalar(value.value_type());
                (value, shape, Vec::new())
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
        if let Some(meta) = meta {
            definition.description = meta.docs().map(str::to_owned);
            definition.category = meta.category().map(str::to_owned);
            definition.flags.read_only = meta.is_readonly();
            if let Some((min, max)) = meta.range() {
                definition.constraints = Constraints {
                    min: Some(min),
                    max: Some(max),
                    ..Constraints::default()
                };
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
) -> Result<(Value, ValueShape, Vec<prism_settings::EnumOption>), ReflectSettingsError>
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
    for item in items {
        let (value, item_shape, item_enum_options) =
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
            if !enum_options.is_empty() && enum_options != item_enum_options {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: type_name.to_owned(),
                });
            }
            enum_options = item_enum_options;
        }
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
    Ok((Value::Array(values), shape, enum_options))
}

fn from_reflect_struct(
    fields: &dyn Struct,
    info: &prism_reflect::StructInfo,
    depth: usize,
    field: &str,
    type_name: &str,
) -> Result<(Value, ValueShape, Vec<prism_settings::EnumOption>), ReflectSettingsError> {
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
    for descriptor in info.fields() {
        let reflected_value = fields.field(descriptor.name()).ok_or_else(|| {
            ReflectSettingsError::MissingDefaultField {
                type_name: type_name.to_owned(),
                field: descriptor.name().to_owned(),
            }
        })?;
        let (value, shape, options) =
            from_reflect_value(reflected_value, depth + 1, field, type_name)?;
        if !options.is_empty() {
            if !enum_options.is_empty() && enum_options != options {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: type_name.to_owned(),
                });
            }
            enum_options = options;
        }
        values.insert(descriptor.name().to_owned(), value);
        shapes.insert(descriptor.name().to_owned(), shape);
    }
    Ok((
        Value::Table(values),
        ValueShape::table(shapes),
        enum_options,
    ))
}

fn from_reflect_map(
    map: &dyn Map,
    key_type_name: &str,
    value_type_name: &str,
    depth: usize,
    field: &str,
    type_name: &str,
) -> Result<(Value, ValueShape, Vec<prism_settings::EnumOption>), ReflectSettingsError> {
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
    for (key, reflected_value) in map.iter_reflect() {
        let Some(Value::String(key)) = from_reflect_scalar(key) else {
            return Err(ReflectSettingsError::UnsupportedValue {
                field: field.to_owned(),
                type_name: type_name.to_owned(),
            });
        };
        let (value, shape, options) =
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
        if !options.is_empty() {
            if !enum_options.is_empty() && enum_options != options {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: type_name.to_owned(),
                });
            }
            enum_options = options;
        }
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
    ))
}

fn from_reflect_value(
    value: &dyn Reflect,
    depth: usize,
    field: &str,
    field_type_name: &str,
) -> Result<(Value, ValueShape, Vec<prism_settings::EnumOption>), ReflectSettingsError> {
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
            if enum_info
                .variants()
                .iter()
                .any(|variant| !matches!(variant.kind(), VariantKind::Unit))
            {
                return Err(ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: field_type_name.to_owned(),
                });
            }
            let enum_options = enum_info
                .variants()
                .iter()
                .map(|variant| prism_settings::EnumOption::from_stable_name(variant.name()))
                .collect::<Vec<_>>();
            let option = enum_options
                .iter()
                .find(|option| option.stable_name == active.variant_name())
                .ok_or_else(|| ReflectSettingsError::UnsupportedValue {
                    field: field.to_owned(),
                    type_name: field_type_name.to_owned(),
                })?;
            Ok((
                Value::Enum(option.id),
                ValueShape::scalar(prism_settings::ValueType::Enum),
                enum_options,
            ))
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
            ))
        }
    }
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
    use std::collections::{BTreeMap, HashMap};

    #[derive(Reflect, Default)]
    struct RenderConfig {
        /// Resolution multiplier.
        #[reflect(setting, range(0.25..=2.0), default = 1.0, category = "Quality")]
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
    fn rejects_metadata_type_mismatch_and_unsupported_fields() {
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
        assert!(matches!(
            derive_settings_registry::<Unsupported>("test", "test"),
            Err(ReflectSettingsError::UnsupportedValue { .. })
        ));
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
}
