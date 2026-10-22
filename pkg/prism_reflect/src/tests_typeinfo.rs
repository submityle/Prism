//! Tests for the design §24.1 static `TypeInfo` layer.
//!
//! Each assertion is oracle-checked against an independent source of truth:
//! field offsets against [`core::mem::offset_of!`], type names against
//! [`core::any::type_name`], and stable ids against
//! [`StableTypeId::of_type`](crate::StableTypeId::of_type).

use crate::{
    Reflect, StableTypeId, StaticField, StaticKind, StaticTyped, TypeInfo, Typed, VariantKind,
};
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Reflect, StaticTyped)]
struct Translation {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Reflect, StaticTyped)]
struct Health {
    current: i32,
    max: u32,
    label: String,
}

#[derive(Reflect)]
struct NestedField {
    value: u32,
}

#[derive(Reflect)]
enum PayloadFields {
    Tuple(NestedField, Vec<NestedField>),
    Struct {
        nested: NestedField,
        values: Vec<NestedField>,
    },
}

#[derive(Reflect)]
enum RecursiveFields {
    End,
    Branch(Vec<RecursiveFields>),
}

#[test]
fn enum_variant_fields_expose_recursive_static_type_info() {
    let TypeInfo::Enum(info) = <PayloadFields as Typed>::type_info() else {
        panic!("expected enum TypeInfo");
    };
    let VariantKind::Tuple(tuple) = info.variant("Tuple").unwrap().kind() else {
        panic!("expected tuple variant");
    };
    assert!(matches!(tuple[0].type_info(), Some(TypeInfo::Struct(_))));
    assert!(matches!(tuple[1].type_info(), Some(TypeInfo::List(_))));

    let VariantKind::Struct(named) = info.variant("Struct").unwrap().kind() else {
        panic!("expected struct variant");
    };
    assert!(matches!(named[0].type_info(), Some(TypeInfo::Struct(_))));
    assert!(matches!(named[1].type_info(), Some(TypeInfo::List(_))));
}

#[test]
fn recursive_enum_type_info_does_not_reenter_initialization() {
    let TypeInfo::Enum(info) = <RecursiveFields as Typed>::type_info() else {
        panic!("expected enum TypeInfo");
    };
    let VariantKind::Tuple(fields) = info.variant("Branch").unwrap().kind() else {
        panic!("expected tuple variant");
    };
    let Some(TypeInfo::List(list)) = fields[0].type_info() else {
        panic!("expected list TypeInfo");
    };
    assert_eq!(
        list.item_type_name(),
        core::any::type_name::<RecursiveFields>()
    );
    assert!(core::ptr::eq(
        <RecursiveFields as Typed>::type_info(),
        <RecursiveFields as Typed>::type_info()
    ));
}

#[test]
fn legacy_runtime_field_descriptors_remain_untyped() {
    let named = crate::NamedField::new("value", "u32");
    let unnamed = crate::UnnamedField::new(0, "u32");
    assert_eq!(named.type_info(), None);
    assert_eq!(unnamed.type_info(), None);
}

#[test]
fn static_info_reports_kind_and_type_name() {
    let info = Translation::static_type_info();
    assert_eq!(info.kind(), StaticKind::Struct);
    assert_eq!(info.type_name(), core::any::type_name::<Translation>());
    assert_eq!(
        info.stable_type_id(),
        StableTypeId::of_type::<Translation>()
    );
}

#[test]
fn static_info_field_names_and_order() {
    let info = Health::static_type_info();
    assert_eq!(info.field_count(), 3);
    let names: Vec<_> = info.fields().iter().map(StaticField::name).collect();
    assert_eq!(names, ["current", "max", "label"]);
}

#[test]
fn static_info_offsets_match_offset_of_oracle() {
    let info = Health::static_type_info();
    assert_eq!(
        info.field("current").unwrap().offset(),
        core::mem::offset_of!(Health, current)
    );
    assert_eq!(
        info.field("max").unwrap().offset(),
        core::mem::offset_of!(Health, max)
    );
    assert_eq!(
        info.field("label").unwrap().offset(),
        core::mem::offset_of!(Health, label)
    );
}

#[test]
fn static_info_field_type_metadata_matches_oracle() {
    let info = Health::static_type_info();
    let current = info.field("current").unwrap();
    assert_eq!(current.type_name(), core::any::type_name::<i32>());
    assert_eq!(current.stable_type_id(), StableTypeId::of_type::<i32>());

    let label = info.field("label").unwrap();
    assert_eq!(label.type_name(), core::any::type_name::<String>());
    assert_eq!(label.stable_type_id(), StableTypeId::of_type::<String>());
}

#[test]
fn static_info_field_lookup_and_index() {
    let info = Translation::static_type_info();
    assert_eq!(info.field_index("y"), Some(1));
    assert_eq!(info.field_index("missing"), None);
    assert!(info.field("z").is_some());
    assert!(info.field("missing").is_none());
    // Resolving an index then reading `fields()[idx]` is the O(1) reuse path.
    let idx = info.field_index("z").unwrap();
    assert_eq!(info.fields()[idx].name(), "z");
}

#[test]
fn static_info_is_a_stable_zero_registration_pointer() {
    // No OnceLock / allocation: the descriptor is a `const`, so every call
    // hands back the very same `&'static` address.
    let a = core::ptr::from_ref(Translation::static_type_info());
    let b = core::ptr::from_ref(Translation::static_type_info());
    assert!(core::ptr::eq(a, b));
    // The associated const and the borrowed static agree.
    assert_eq!(
        Translation::static_type_info(),
        &Translation::STATIC_TYPE_INFO
    );
}

#[test]
fn distinct_types_have_distinct_stable_ids() {
    assert_ne!(
        Translation::static_type_info().stable_type_id(),
        Health::static_type_info().stable_type_id()
    );
}
