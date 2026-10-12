//! Tests for the design §24.2 hot-path accessor layer.
//!
//! The typed [`FieldAccess`](crate::FieldAccess) helpers and the cached
//! [`AccessPlan`](crate::AccessPlan) are oracle-checked against direct field
//! access and against [`core::mem::offset_of!`].

use crate::{AccessError, AccessPlan, Accessors, FieldAccess, FieldAccessor, Reflect};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Reflect, Accessors, Debug, PartialEq)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Reflect, Accessors, Debug, PartialEq)]
struct Transform {
    translation: Vec3,
    scale: Vec3,
}

#[derive(Reflect, Accessors, Debug, PartialEq)]
struct Entity {
    id: u32,
    transform: Transform,
}

#[derive(Reflect, Accessors, Debug, PartialEq)]
struct Health {
    current: i32,
    label: String,
}

fn sample_entity() -> Entity {
    Entity {
        id: 7,
        transform: Transform {
            translation: Vec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            scale: Vec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
        },
    }
}

#[test]
fn accessor_table_names_and_order() {
    let names: Vec<_> = Vec3::accessors().iter().map(FieldAccessor::name).collect();
    assert_eq!(names, ["x", "y", "z"]);
    assert_eq!(Vec3::accessors().len(), 3);
}

#[test]
fn accessor_offsets_match_offset_of_oracle() {
    let by_name = |name: &str| Vec3::accessor(name).unwrap().offset();
    assert_eq!(by_name("x"), core::mem::offset_of!(Vec3, x));
    assert_eq!(by_name("y"), core::mem::offset_of!(Vec3, y));
    assert_eq!(by_name("z"), core::mem::offset_of!(Vec3, z));
}

#[test]
fn accessor_index_resolves_once() {
    assert_eq!(Transform::accessor_index("translation"), Some(0));
    assert_eq!(Transform::accessor_index("scale"), Some(1));
    assert_eq!(Transform::accessor_index("missing"), None);
}

#[test]
fn field_and_get_field_read_through_generated_getter() {
    let v = Vec3 {
        x: 10.0,
        y: 20.0,
        z: 30.0,
    };
    let y = v.field("y").unwrap();
    assert_eq!(y.downcast_ref::<f32>(), Some(&20.0));
    assert_eq!(v.get_field::<f32>("z"), Some(&30.0));
    assert_eq!(v.get_field::<f32>("missing"), None);
    // Reading a field at the wrong concrete type returns None, not garbage.
    assert_eq!(v.get_field::<i32>("x"), None);
}

#[test]
fn field_mut_and_get_field_mut_write_through_generated_getter() {
    let mut v = Vec3 {
        x: 1.0,
        y: 2.0,
        z: 3.0,
    };
    *v.get_field_mut::<f32>("x").unwrap() = 99.0;
    assert_eq!(v.x, 99.0);

    let z = v.field_mut("z").unwrap();
    *z.downcast_mut::<f32>().unwrap() = -1.0;
    assert_eq!(v.z, -1.0);
}

#[test]
fn set_field_applies_compatible_value() {
    let mut h = Health {
        current: 50,
        label: "ok".to_string(),
    };
    h.set_field("current", &100_i32).unwrap();
    assert_eq!(h.current, 100);
    h.set_field("label", &"hurt".to_string()).unwrap();
    assert_eq!(h.label, "hurt");
}

#[test]
fn set_field_rejects_unknown_field() {
    let mut v = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let err = v.set_field("missing", &1.0_f32).unwrap_err();
    match err {
        AccessError::UnknownField { field, .. } => assert_eq!(field, "missing"),
        other => panic!("expected UnknownField, got {other:?}"),
    }
}

#[test]
fn set_field_rejects_type_mismatch() {
    let mut h = Health {
        current: 1,
        label: "x".to_string(),
    };
    // A String cannot be applied onto an i32 field.
    let err = h.set_field("current", &"nope".to_string()).unwrap_err();
    assert!(matches!(err, AccessError::Apply(_)));
    // The field was left untouched.
    assert_eq!(h.current, 1);
}

#[test]
fn access_plan_reads_nested_path() {
    let entity = sample_entity();
    let plan = AccessPlan::compile(&entity, "transform.translation.y").unwrap();
    assert_eq!(plan.path(), "transform.translation.y");
    assert_eq!(plan.steps(), &[1, 0, 1]);
    let value = plan.get(&entity).unwrap();
    assert_eq!(value.downcast_ref::<f32>(), Some(&2.0));
}

#[test]
fn access_plan_reuses_one_plan_across_values() {
    let plan = AccessPlan::compile(&sample_entity(), "transform.translation.x").unwrap();
    for expected in [1.0_f32, 4.0, 7.5] {
        let mut entity = sample_entity();
        entity.transform.translation.x = expected;
        let value = plan.get(&entity).unwrap();
        assert_eq!(value.downcast_ref::<f32>(), Some(&expected));
    }
}

#[test]
fn access_plan_writes_nested_path() {
    let mut entity = sample_entity();
    let plan = AccessPlan::compile(&entity, "transform.scale.z").unwrap();
    {
        let target = plan.get_mut(&mut entity).unwrap();
        *target.downcast_mut::<f32>().unwrap() = 2.5;
    }
    assert_eq!(entity.transform.scale.z, 2.5);
}

#[test]
fn access_plan_set_applies_value() {
    let mut entity = sample_entity();
    let plan = AccessPlan::compile(&entity, "id").unwrap();
    plan.set(&mut entity, &42_u32).unwrap();
    assert_eq!(entity.id, 42);
}

#[test]
fn access_plan_rejects_empty_path() {
    let entity = sample_entity();
    assert!(matches!(
        AccessPlan::compile(&entity, ""),
        Err(AccessError::EmptyPath)
    ));
}

#[test]
fn access_plan_rejects_unknown_field() {
    let entity = sample_entity();
    let err = AccessPlan::compile(&entity, "transform.missing").unwrap_err();
    match err {
        AccessError::UnknownField { field, .. } => assert_eq!(field, "missing"),
        other => panic!("expected UnknownField, got {other:?}"),
    }
}

#[test]
fn access_plan_rejects_descent_into_non_struct() {
    let entity = sample_entity();
    // `id` is a `u32` leaf, so it cannot be descended into.
    let err = AccessPlan::compile(&entity, "id.nope").unwrap_err();
    match err {
        AccessError::NotAStruct { segment, .. } => assert_eq!(segment, "nope"),
        other => panic!("expected NotAStruct, got {other:?}"),
    }
}

#[test]
fn access_plan_on_wrong_shape_returns_none() {
    // A plan compiled for `Entity` replayed on a differently shaped struct
    // fails the index lookup instead of misreading memory.
    let plan = AccessPlan::compile(&sample_entity(), "transform.translation.x").unwrap();
    let wrong = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    assert!(plan.get(&wrong).is_none());
}
