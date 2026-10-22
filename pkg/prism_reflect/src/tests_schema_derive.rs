//! Tests for the design §24.7 `#[reflect(...)]` attribute parsing in the
//! `#[derive(Reflect)]` macro.
//!
//! Each assertion checks that the derive-generated `GetTypeRegistration`
//! attaches a [`TypeMetadata`](crate::schema::TypeMetadata) payload matching the
//! attributes written on the type and its fields. Oracles are the literal
//! attribute values spelled out in the type definitions below, cross-checked
//! against the metadata queried back through the registry after
//! `register::<T>()` (no manual `register_type_data` call).

use crate::schema::{AttributeValue, TypeMetadata};
use crate::{Reflect, TypeRegistry};
use alloc::string::String;
use core::any::type_name;

// A light source exercising every supported field-level `#[reflect(...)]` key
// plus type-level docs and a type-level custom attribute.
#[derive(Reflect, Default)]
#[reflect(docs = "A configurable point light.", icon = "light")]
struct Light {
    /// Luminous intensity in candela.
    #[reflect(category = "Appearance", range(0.0..=100000.0))]
    intensity: f32,

    #[reflect(tooltip = "Clamp between dim and bright.", clamp(-1.0..=1.0))]
    bias: f32,

    #[reflect(rename = "uuid", readonly)]
    id: u64,

    #[reflect(hidden = true, skip)]
    cache: u32,

    #[reflect(required, default = 42)]
    priority: i32,

    #[reflect(default = -3)]
    offset: i32,

    #[reflect(default = -9223372036854775808)]
    signed_limit: i64,

    #[reflect(default = -0.25)]
    negative_bias: f64,

    #[reflect(default = 18446744073709551615)]
    unsigned_limit: u64,

    #[reflect(default = "none")]
    tag: String,
}

// A field with no attributes must not appear in the metadata.
#[derive(Reflect, Default)]
struct Mixed {
    annotated: f32,
    plain: f32,
}

fn metadata_for<T>(registry: &TypeRegistry) -> &TypeMetadata
where
    T: 'static,
{
    registry
        .get_with_name(type_name::<T>())
        .and_then(|r| r.data::<TypeMetadata>())
        .expect("derived type should carry TypeMetadata")
}

#[test]
fn field_range_maps_to_inclusive_bounds() {
    let mut registry = TypeRegistry::new();
    registry.register::<Light>();
    let meta = metadata_for::<Light>(&registry);

    assert_eq!(
        meta.field("intensity").unwrap().range(),
        Some((0.0, 100_000.0))
    );
    // `clamp(...)` is an alias for `range(...)`, including negative bounds.
    assert_eq!(meta.field("bias").unwrap().range(), Some((-1.0, 1.0)));
}

#[test]
fn field_docs_category_and_flags() {
    let mut registry = TypeRegistry::new();
    registry.register::<Light>();
    let meta = metadata_for::<Light>(&registry);

    let intensity = meta.field("intensity").unwrap();
    assert_eq!(intensity.docs(), Some("Luminous intensity in candela."));
    assert_eq!(intensity.category(), Some("Appearance"));

    let bias = meta.field("bias").unwrap();
    assert_eq!(bias.docs(), Some("Clamp between dim and bright."));

    let id = meta.field("id").unwrap();
    assert!(id.is_readonly());
    assert_eq!(
        id.custom("rename"),
        Some(&AttributeValue::Text(String::from("uuid")))
    );

    let cache = meta.field("cache").unwrap();
    assert!(cache.is_hidden());
    assert_eq!(cache.custom("skip"), Some(&AttributeValue::Bool(true)));
}

#[test]
fn field_required_and_defaults() {
    let mut registry = TypeRegistry::new();
    registry.register::<Light>();
    let meta = metadata_for::<Light>(&registry);

    let priority = meta.field("priority").unwrap();
    assert!(priority.is_required());
    assert_eq!(priority.default_value(), Some(&AttributeValue::Int(42)));

    assert_eq!(
        meta.field("offset").unwrap().default_value(),
        Some(&AttributeValue::Int(-3))
    );
    assert_eq!(
        meta.field("signed_limit").unwrap().default_value(),
        Some(&AttributeValue::Int(i64::MIN))
    );
    assert_eq!(
        meta.field("negative_bias").unwrap().default_value(),
        Some(&AttributeValue::Float(-0.25))
    );
    assert_eq!(
        meta.field("unsigned_limit").unwrap().default_value(),
        Some(&AttributeValue::UInt(u64::MAX))
    );

    let tag = meta.field("tag").unwrap();
    assert_eq!(
        tag.default_value(),
        Some(&AttributeValue::Text(String::from("none")))
    );
}

#[test]
fn type_level_docs_and_custom() {
    let mut registry = TypeRegistry::new();
    registry.register::<Light>();
    let meta = metadata_for::<Light>(&registry);

    assert_eq!(meta.docs(), Some("A configurable point light."));
    assert_eq!(
        meta.custom("icon"),
        Some(&AttributeValue::Text(String::from("light")))
    );
}

#[test]
fn only_annotated_fields_are_recorded() {
    let mut registry = TypeRegistry::new();
    registry.register::<Mixed>();
    // `Mixed` has no metadata-bearing attributes at all, so no TypeMetadata is
    // attached and the registration carries none.
    let registration = registry.get_with_name(type_name::<Mixed>()).unwrap();
    assert!(registration.data::<TypeMetadata>().is_none());
}

#[derive(Reflect, Default)]
struct SettingsOptIn {
    #[reflect(setting)]
    enabled: bool,
    plain: bool,
}

#[test]
fn setting_opt_in_is_explicit_metadata() {
    let mut registry = TypeRegistry::new();
    registry.register::<SettingsOptIn>();
    let meta = metadata_for::<SettingsOptIn>(&registry);
    assert!(meta.field("enabled").unwrap().is_setting());
    assert!(meta.field("plain").is_none());
}

// A tuple struct carries type-level docs but no per-field metadata.
#[derive(Reflect, Default)]
#[reflect(docs = "A 2D screen offset.")]
struct Offset(f32, f32);

#[test]
fn tuple_struct_records_type_level_docs_only() {
    let mut registry = TypeRegistry::new();
    registry.register::<Offset>();
    let meta = metadata_for::<Offset>(&registry);
    assert_eq!(meta.docs(), Some("A 2D screen offset."));
    assert!(meta.fields().is_empty());
}

// Only the annotated field of a partially annotated struct is present.
#[derive(Reflect, Default)]
struct Partial {
    #[reflect(category = "Core")]
    kept: f32,
    dropped: f32,
}

#[test]
fn partial_struct_records_only_annotated_field() {
    let mut registry = TypeRegistry::new();
    registry.register::<Partial>();
    let meta = metadata_for::<Partial>(&registry);

    assert!(meta.field("kept").is_some());
    assert_eq!(meta.field("kept").unwrap().category(), Some("Core"));
    assert!(meta.field("dropped").is_none());
}

/// A doc comment on the type folds into type-level metadata docs even without
/// an explicit `#[reflect(docs = ...)]`.
#[derive(Reflect, Default)]
struct Documented {
    value: f32,
}

#[test]
fn type_doc_comment_folds_into_metadata() {
    let mut registry = TypeRegistry::new();
    registry.register::<Documented>();
    let meta = metadata_for::<Documented>(&registry);
    assert_eq!(
        meta.docs(),
        Some(
            "A doc comment on the type folds into type-level metadata docs even without\nan explicit `#[reflect(docs = ...)]`."
        )
    );
}
