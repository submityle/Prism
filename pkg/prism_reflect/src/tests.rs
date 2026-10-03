//! M0 reflection smoke tests: derive, field traversal, kinds, downcasting.

use crate::{Reflect, ReflectRef, Struct, TupleStruct, TypeInfo, Typed};

#[derive(Reflect)]
struct Player {
    health: i32,
    name: String,
    speed: f32,
}

#[derive(Reflect)]
struct Wrapper(i32, bool);

#[test]
fn struct_type_info_is_struct_kind() {
    let info = <Player as Typed>::type_info();
    match info {
        TypeInfo::Struct(s) => {
            assert_eq!(s.field_count(), 3);
            assert_eq!(s.fields()[0].name(), "health");
            assert_eq!(s.fields()[1].name(), "name");
            assert_eq!(s.fields()[2].name(), "speed");
            assert!(s.field("name").is_some());
            assert!(s.field("missing").is_none());
        }
        other => panic!("expected Struct, got {other:?}"),
    }
    // Static caching: same &'static returned twice.
    let again = <Player as Typed>::type_info();
    assert!(core::ptr::eq(info, again));
}

#[test]
fn struct_field_traversal_by_name_and_index() {
    let mut p = Player {
        health: 100,
        name: "aria".to_string(),
        speed: 2.5,
    };

    let health = p.field("health").unwrap();
    assert_eq!(health.downcast_ref::<i32>(), Some(&100));

    assert_eq!(p.name_at(1), Some("name"));
    let name = p.field_at(1).unwrap();
    assert_eq!(name.downcast_ref::<String>().unwrap(), "aria");

    // Mutate through reflection.
    let speed = p.field_mut("speed").unwrap();
    *speed.downcast_mut::<f32>().unwrap() = 9.0;
    assert_eq!(p.speed, 9.0);

    assert_eq!(p.field_count(), 3);
}

#[test]
fn reflect_ref_reports_struct_variant() {
    let p = Player {
        health: 1,
        name: String::new(),
        speed: 0.0,
    };
    match p.reflect_ref() {
        ReflectRef::Struct(s) => assert_eq!(s.field_count(), 3),
        _ => panic!("expected Struct view"),
    }
}

#[test]
fn tuple_struct_info_and_traversal() {
    let info = <Wrapper as Typed>::type_info();
    match info {
        TypeInfo::TupleStruct(ts) => {
            assert_eq!(ts.field_count(), 2);
            assert_eq!(ts.field_at(0).unwrap().index(), 0);
        }
        other => panic!("expected TupleStruct, got {other:?}"),
    }

    let w = Wrapper(42, true);
    let f0 = TupleStruct::field(&w, 0).unwrap();
    assert_eq!(f0.downcast_ref::<i32>(), Some(&42));
    let f1 = TupleStruct::field(&w, 1).unwrap();
    assert_eq!(f1.downcast_ref::<bool>(), Some(&true));
    assert!(TupleStruct::field(&w, 2).is_none());
}

#[test]
fn value_leaf_impls_report_value_kind() {
    let x: i32 = 7;
    assert!(matches!(x.type_info(), TypeInfo::Value(_)));
    let r: &dyn Reflect = &x;
    assert!(r.is::<i32>());
    assert_eq!(r.downcast_ref::<i32>(), Some(&7));

    let s = String::from("hi");
    assert!(matches!(s.type_info(), TypeInfo::Value(_)));
}
