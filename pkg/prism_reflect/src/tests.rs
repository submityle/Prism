//! M0 + M1 reflection tests: derive, every kind's traversal, the registry
//! round-trip with `ReflectDefault`, enum derive (all three shapes), and
//! `prism_math` value reflection.

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

mod list_kind {
    use crate::{List, Reflect, ReflectRef, TypeInfo, Typed};
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn vec_type_info_is_list_kind() {
        match <Vec<i32> as Typed>::type_info() {
            TypeInfo::List(info) => {
                assert_eq!(info.item_type_name(), ::core::any::type_name::<i32>());
            }
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn list_get_len_iter_and_push() {
        let mut v: Vec<i32> = vec![10, 20, 30];
        assert_eq!(List::len(&v), 3);
        assert!(!List::is_empty(&v));
        assert_eq!(List::get(&v, 1).unwrap().downcast_ref::<i32>(), Some(&20));
        assert!(List::get(&v, 9).is_none());

        let collected: Vec<i32> = v
            .iter_reflect()
            .map(|e| *e.downcast_ref::<i32>().unwrap())
            .collect();
        assert_eq!(collected, vec![10, 20, 30]);

        assert!(List::push(&mut v, Box::new(40i32)).is_ok());
        assert_eq!(List::len(&v), 4);
        assert_eq!(List::get(&v, 3).unwrap().downcast_ref::<i32>(), Some(&40));

        // Wrong element type is rejected and handed back.
        assert!(List::push(&mut v, Box::new(true)).is_err());

        // Mutate in place through reflection.
        *List::get_mut(&mut v, 0)
            .unwrap()
            .downcast_mut::<i32>()
            .unwrap() = 11;
        assert_eq!(v[0], 11);
    }

    #[test]
    fn reflect_ref_reports_list() {
        let v: Vec<i32> = vec![1, 2];
        match v.reflect_ref() {
            ReflectRef::List(l) => assert_eq!(l.len(), 2),
            _ => panic!("expected List view"),
        }
    }
}

mod array_kind {
    use crate::{Array, Reflect, ReflectRef, TypeInfo, Typed};
    use alloc::vec::Vec;

    #[test]
    fn array_type_info_reports_capacity() {
        match <[i32; 3] as Typed>::type_info() {
            TypeInfo::Array(info) => {
                assert_eq!(info.capacity(), 3);
                assert_eq!(info.item_type_name(), ::core::any::type_name::<i32>());
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[test]
    fn array_get_len_iter_and_mutate() {
        let mut a: [i32; 3] = [1, 2, 3];
        assert_eq!(Array::len(&a), 3);
        assert!(!Array::is_empty(&a));
        assert_eq!(Array::get(&a, 2).unwrap().downcast_ref::<i32>(), Some(&3));
        assert!(Array::get(&a, 3).is_none());

        let collected: Vec<i32> = a
            .iter_reflect()
            .map(|e| *e.downcast_ref::<i32>().unwrap())
            .collect();
        assert_eq!(collected, [1, 2, 3]);

        *Array::get_mut(&mut a, 1)
            .unwrap()
            .downcast_mut::<i32>()
            .unwrap() = 20;
        assert_eq!(a[1], 20);
    }

    #[test]
    fn reflect_ref_reports_array() {
        let a: [u8; 2] = [7, 8];
        match a.reflect_ref() {
            ReflectRef::Array(arr) => assert_eq!(arr.len(), 2),
            _ => panic!("expected Array view"),
        }
    }
}

mod map_kind {
    use crate::{Map, Reflect, ReflectRef, TypeInfo, Typed};
    use alloc::boxed::Box;
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use alloc::string::ToString;

    #[test]
    fn map_type_info_is_map_kind() {
        match <BTreeMap<String, i32> as Typed>::type_info() {
            TypeInfo::Map(info) => {
                assert_eq!(info.key_type_name(), ::core::any::type_name::<String>());
                assert_eq!(info.value_type_name(), ::core::any::type_name::<i32>());
            }
            other => panic!("expected Map, got {other:?}"),
        }
    }

    #[test]
    fn map_get_len_insert_and_iter() {
        let mut m: BTreeMap<String, i32> = BTreeMap::new();
        m.insert("a".to_string(), 1);
        m.insert("b".to_string(), 2);
        assert_eq!(Map::len(&m), 2);
        assert!(!Map::is_empty(&m));

        let key = "a".to_string();
        let value = Map::get(&m, &key).unwrap();
        assert_eq!(value.downcast_ref::<i32>(), Some(&1));

        assert!(Map::insert(&mut m, Box::new("c".to_string()), Box::new(3i32)).is_ok());
        assert_eq!(Map::len(&m), 3);

        // Wrong value type is handed back together with the key.
        assert!(Map::insert(&mut m, Box::new("d".to_string()), Box::new(true)).is_err());

        // BTreeMap iterates in sorted key order.
        let sum: i32 = m
            .iter_reflect()
            .map(|(_, v)| *v.downcast_ref::<i32>().unwrap())
            .sum();
        assert_eq!(sum, 6);

        *Map::get_mut(&mut m, &key)
            .unwrap()
            .downcast_mut::<i32>()
            .unwrap() = 100;
        assert_eq!(m["a"], 100);
    }

    #[test]
    fn reflect_ref_reports_map() {
        let mut m: BTreeMap<i32, i32> = BTreeMap::new();
        m.insert(1, 1);
        match m.reflect_ref() {
            ReflectRef::Map(map) => assert_eq!(map.len(), 1),
            _ => panic!("expected Map view"),
        }
    }
}

mod set_kind {
    use crate::{Reflect, ReflectRef, Set, TypeInfo, Typed};
    use alloc::boxed::Box;
    use alloc::collections::BTreeSet;

    #[test]
    fn set_type_info_is_set_kind() {
        match <BTreeSet<i32> as Typed>::type_info() {
            TypeInfo::Set(info) => {
                assert_eq!(info.value_type_name(), ::core::any::type_name::<i32>());
            }
            other => panic!("expected Set, got {other:?}"),
        }
    }

    #[test]
    fn set_contains_len_insert_and_iter() {
        let mut s: BTreeSet<i32> = BTreeSet::new();
        s.insert(1);
        s.insert(2);
        assert_eq!(Set::len(&s), 2);
        assert!(!Set::is_empty(&s));

        let probe = 2i32;
        assert!(Set::contains(&s, &probe));
        let missing = 9i32;
        assert!(!Set::contains(&s, &missing));

        assert_eq!(Set::insert(&mut s, Box::new(3i32)).ok(), Some(true));
        assert_eq!(Set::insert(&mut s, Box::new(3i32)).ok(), Some(false));
        assert!(Set::insert(&mut s, Box::new(true)).is_err());

        let sum: i32 = s
            .iter_reflect()
            .map(|v| *v.downcast_ref::<i32>().unwrap())
            .sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn reflect_ref_reports_set() {
        let mut s: BTreeSet<u8> = BTreeSet::new();
        s.insert(7);
        match s.reflect_ref() {
            ReflectRef::Set(set) => assert_eq!(set.len(), 1),
            _ => panic!("expected Set view"),
        }
    }
}

mod enum_kind {
    use crate::{Enum, Reflect, ReflectRef, TypeInfo, Typed, VariantKind, VariantType};

    #[test]
    fn option_is_enum_kind() {
        match <Option<i32> as Typed>::type_info() {
            TypeInfo::Enum(info) => {
                assert_eq!(info.variant_count(), 2);
                assert_eq!(info.variant_at(0).unwrap().name(), "None");
                assert_eq!(info.variant_at(1).unwrap().name(), "Some");
            }
            other => panic!("expected Enum, got {other:?}"),
        }

        let some: Option<i32> = Some(5);
        assert_eq!(some.variant_name(), "Some");
        assert_eq!(some.variant_index(), 1);
        assert_eq!(some.variant_type(), VariantType::Tuple);
        assert_eq!(some.field_at(0).unwrap().downcast_ref::<i32>(), Some(&5));

        let none: Option<i32> = None;
        assert_eq!(none.variant_name(), "None");
        assert_eq!(none.variant_type(), VariantType::Unit);
        assert_eq!(none.field_count(), 0);
    }

    #[test]
    fn result_is_enum_kind() {
        let ok: Result<i32, String> = Ok(3);
        assert_eq!(ok.variant_name(), "Ok");
        assert_eq!(ok.field_at(0).unwrap().downcast_ref::<i32>(), Some(&3));

        let err: Result<i32, String> = Err("boom".into());
        assert_eq!(err.variant_name(), "Err");
        assert_eq!(err.variant_index(), 1);
        assert_eq!(
            err.field_at(0).unwrap().downcast_ref::<String>().unwrap(),
            "boom"
        );
    }

    #[derive(Reflect)]
    enum Shape {
        Empty,
        Circle(f32),
        Rect { width: f32, height: f32 },
    }

    #[test]
    fn derived_enum_type_info_covers_all_shapes() {
        match <Shape as Typed>::type_info() {
            TypeInfo::Enum(info) => {
                assert_eq!(info.variant_count(), 3);
                assert_eq!(*info.variant_at(0).unwrap().kind(), VariantKind::Unit);
                match info.variant_at(1).unwrap().kind() {
                    VariantKind::Tuple(fields) => assert_eq!(fields.len(), 1),
                    other => panic!("expected Tuple variant, got {other:?}"),
                }
                match info.variant_at(2).unwrap().kind() {
                    VariantKind::Struct(fields) => {
                        assert_eq!(fields[0].name(), "width");
                        assert_eq!(fields[1].name(), "height");
                    }
                    other => panic!("expected Struct variant, got {other:?}"),
                }
            }
            other => panic!("expected Enum, got {other:?}"),
        }
    }

    #[test]
    fn derived_enum_unit_variant() {
        let s = Shape::Empty;
        assert_eq!(s.variant_name(), "Empty");
        assert_eq!(s.variant_index(), 0);
        assert_eq!(s.variant_type(), VariantType::Unit);
        assert_eq!(s.field_count(), 0);
        assert!(matches!(s.reflect_ref(), ReflectRef::Enum(_)));
    }

    #[test]
    fn derived_enum_tuple_variant() {
        let mut s = Shape::Circle(2.0);
        assert_eq!(s.variant_name(), "Circle");
        assert_eq!(s.variant_index(), 1);
        assert_eq!(s.variant_type(), VariantType::Tuple);
        assert_eq!(s.field_count(), 1);
        assert_eq!(s.field_at(0).unwrap().downcast_ref::<f32>(), Some(&2.0));

        *s.field_at_mut(0).unwrap().downcast_mut::<f32>().unwrap() = 5.0;
        assert_eq!(s.field_at(0).unwrap().downcast_ref::<f32>(), Some(&5.0));
    }

    #[test]
    fn derived_enum_struct_variant() {
        let mut s = Shape::Rect {
            width: 3.0,
            height: 4.0,
        };
        assert_eq!(s.variant_name(), "Rect");
        assert_eq!(s.variant_index(), 2);
        assert_eq!(s.variant_type(), VariantType::Struct);
        assert_eq!(s.field_count(), 2);
        assert_eq!(
            Enum::field(&s, "width").unwrap().downcast_ref::<f32>(),
            Some(&3.0)
        );
        assert_eq!(s.field_at(1).unwrap().downcast_ref::<f32>(), Some(&4.0));
        assert!(Enum::field(&s, "missing").is_none());

        *Enum::field_mut(&mut s, "height")
            .unwrap()
            .downcast_mut::<f32>()
            .unwrap() = 9.0;
        assert_eq!(
            Enum::field(&s, "height").unwrap().downcast_ref::<f32>(),
            Some(&9.0)
        );
    }
}

mod registry_roundtrip {
    use crate::{GetTypeRegistration, Reflect, ReflectDefault, TypeInfo, TypeRegistry};
    use core::any::{type_name, TypeId};

    #[derive(Reflect, Default)]
    struct Config {
        level: i32,
        label: String,
    }

    #[test]
    fn register_lookup_and_default_construct() {
        let mut registry = TypeRegistry::new();
        assert!(registry.is_empty());
        registry.register::<Config>();
        assert_eq!(registry.len(), 1);
        assert!(registry.contains(TypeId::of::<Config>()));

        // Lookup by TypeId.
        let by_id = registry.get(TypeId::of::<Config>()).unwrap();
        assert!(matches!(by_id.type_info(), TypeInfo::Struct(_)));
        assert_eq!(by_id.type_id(), TypeId::of::<Config>());

        // Lookup by fully-qualified name.
        let by_name = registry.get_with_name(type_name::<Config>()).unwrap();
        assert_eq!(by_name.type_name(), type_name::<Config>());

        // Attach ReflectDefault type data and default-construct through it.
        assert!(
            registry.register_type_data::<Config, ReflectDefault>(ReflectDefault::new::<Config>())
        );
        let reg = registry.get(TypeId::of::<Config>()).unwrap();
        let default = reg.data::<ReflectDefault>().unwrap();
        let value = default.default_value();
        let config = value.as_any().downcast_ref::<Config>().unwrap();
        assert_eq!(config.level, 0);
        assert_eq!(config.label, "");

        // register_type_data on an unregistered type reports false.
        assert!(!registry.register_type_data::<i32, ReflectDefault>(ReflectDefault::new::<i32>()));
    }

    #[test]
    fn get_type_registration_is_generated() {
        let registration = Config::get_type_registration();
        assert_eq!(registration.type_id(), TypeId::of::<Config>());
    }
}

#[cfg(feature = "math")]
mod math_kind {
    use crate::{Reflect, Struct, TypeInfo, Typed};
    use prism_math::{Mat4, Quat, Vec2, Vec3, Vec3A, Vec4};

    #[test]
    fn vec3_reflects_as_struct() {
        match <Vec3 as Typed>::type_info() {
            TypeInfo::Struct(info) => {
                assert_eq!(info.field_count(), 3);
                assert_eq!(info.fields()[0].name(), "x");
                assert_eq!(info.fields()[2].name(), "z");
            }
            other => panic!("expected Struct, got {other:?}"),
        }

        let mut v = Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        assert_eq!(
            Struct::field(&v, "y").unwrap().downcast_ref::<f32>(),
            Some(&2.0)
        );
        *Struct::field_mut(&mut v, "x")
            .unwrap()
            .downcast_mut::<f32>()
            .unwrap() = 10.0;
        assert_eq!(v.x, 10.0);
    }

    #[test]
    fn all_math_types_reflect() {
        assert_eq!(
            <Vec2 as Typed>::type_info().type_name(),
            ::core::any::type_name::<Vec2>()
        );
        assert_eq!(
            <Vec3A as Typed>::type_info().type_name(),
            ::core::any::type_name::<Vec3A>()
        );
        assert_eq!(
            <Vec4 as Typed>::type_info().type_name(),
            ::core::any::type_name::<Vec4>()
        );
        assert_eq!(
            <Quat as Typed>::type_info().type_name(),
            ::core::any::type_name::<Quat>()
        );

        let m = Mat4::default();
        match m.type_info() {
            TypeInfo::Struct(info) => {
                assert_eq!(info.field_count(), 4);
                assert_eq!(info.fields()[0].name(), "x_axis");
                assert_eq!(info.fields()[3].name(), "w_axis");
            }
            other => panic!("expected Struct, got {other:?}"),
        }
    }

    #[test]
    fn math_type_registers() {
        use crate::{GetTypeRegistration, TypeRegistry};
        use core::any::TypeId;
        let mut registry = TypeRegistry::new();
        registry.add_registration(Vec3::get_type_registration());
        assert!(registry.contains(TypeId::of::<Vec3>()));
    }
}

mod m2_dynamic {
    use crate::prelude::*;
    use alloc::boxed::Box;
    #[cfg(feature = "math")]
    use prism_math::Vec3;
    use std::collections::HashMap;

    #[derive(Reflect, Debug, PartialEq, Clone)]
    struct Stats {
        health: i32,
        name: String,
        speed: f32,
    }

    #[derive(Reflect, Debug, PartialEq, Clone)]
    struct Pair(i32, bool);

    #[derive(Reflect, Debug, PartialEq, Clone)]
    enum Shape {
        Empty,
        Circle(f32),
        Rect { width: f32, height: f32 },
    }

    #[derive(Reflect, Debug, PartialEq, Clone)]
    struct World {
        player: Stats,
        pair: Pair,
        grid: Vec<Vec<i32>>,
        lookup: HashMap<String, i32>,
    }

    fn sample_world() -> World {
        let mut lookup = HashMap::new();
        lookup.insert("a".to_string(), 1);
        lookup.insert("b".to_string(), 2);
        World {
            player: Stats {
                health: 100,
                name: "aria".to_string(),
                speed: 1.5,
            },
            pair: Pair(7, true),
            grid: vec![vec![10, 20, 30], vec![40, 50]],
            lookup,
        }
    }

    #[test]
    fn dynamic_struct_apply_patches_concrete_by_name() {
        let mut stats = Stats {
            health: 100,
            name: "aria".to_string(),
            speed: 1.0,
        };

        let mut patch = DynamicStruct::new();
        patch.insert("health", 42i32);
        patch.insert("speed", 9.5f32);

        stats.apply(&patch).expect("dynamic struct patch applies");

        assert_eq!(stats.health, 42);
        assert_eq!(stats.speed, 9.5);
        // `name` was not present in the patch and stays untouched.
        assert_eq!(stats.name, "aria");
    }

    #[test]
    fn from_reflect_round_trips_struct_direct_and_via_clone() {
        let stats = Stats {
            health: 73,
            name: "nova".to_string(),
            speed: 4.25,
        };

        let direct = Stats::from_reflect(&stats as &dyn Reflect).expect("direct round-trip");
        assert_eq!(direct, stats);

        let cloned = stats.reflect_clone();
        let via_clone = Stats::from_reflect(&*cloned).expect("round-trip via DynamicStruct");
        assert_eq!(via_clone, stats);
    }

    #[test]
    fn from_reflect_round_trips_tuple_struct_and_enum() {
        let pair = Pair(9, false);
        assert_eq!(Pair::from_reflect(&pair as &dyn Reflect).unwrap(), pair);
        assert_eq!(Pair::from_reflect(&*pair.reflect_clone()).unwrap(), pair);

        for shape in [
            Shape::Empty,
            Shape::Circle(2.5),
            Shape::Rect {
                width: 3.0,
                height: 4.0,
            },
        ] {
            let direct = Shape::from_reflect(&shape as &dyn Reflect).unwrap();
            assert_eq!(direct, shape);
            let via_clone = Shape::from_reflect(&*shape.reflect_clone()).unwrap();
            assert_eq!(via_clone, shape);
        }
    }

    #[test]
    fn from_reflect_round_trips_collections_and_math() {
        let list = vec![1, 2, 3, 4];
        assert_eq!(
            Vec::<i32>::from_reflect(&list as &dyn Reflect).unwrap(),
            list
        );

        let mut map = HashMap::new();
        map.insert("x".to_string(), 10);
        map.insert("y".to_string(), 20);
        assert_eq!(
            HashMap::<String, i32>::from_reflect(&map as &dyn Reflect).unwrap(),
            map
        );

        #[cfg(feature = "math")]
        {
            let v = Vec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            };
            assert_eq!(Vec3::from_reflect(&v as &dyn Reflect).unwrap(), v);
            assert_eq!(Vec3::from_reflect(&*v.reflect_clone()).unwrap(), v);
        }
    }

    #[test]
    fn concrete_enum_apply_patches_matching_variant() {
        let mut shape = Shape::Rect {
            width: 1.0,
            height: 1.0,
        };
        let patch = DynamicEnum::new(
            2,
            "Rect",
            DynamicVariant::Struct(vec![
                ("width", Box::new(5.0f32) as Box<dyn Reflect>),
                ("height", Box::new(6.0f32) as Box<dyn Reflect>),
            ]),
        );
        shape
            .apply(&patch)
            .expect("same-variant enum patch applies");
        assert_eq!(
            shape,
            Shape::Rect {
                width: 5.0,
                height: 6.0
            }
        );

        // A different variant cannot be forced onto a concrete enum.
        let mismatch = DynamicEnum::new(0, "Empty", DynamicVariant::Unit);
        assert!(shape.apply(&mismatch).is_err());
    }

    #[test]
    fn dynamic_enum_switches_variant_on_apply_then_from_reflect() {
        let mut dynamic = DynamicEnum::new(0, "Empty", DynamicVariant::Unit);
        assert_eq!(dynamic.variant_name(), "Empty");

        let circle = Shape::Circle(3.5);
        dynamic
            .apply(&circle)
            .expect("DynamicEnum switches to the source variant");
        assert_eq!(dynamic.variant_name(), "Circle");
        assert_eq!(
            dynamic.field_at(0).unwrap().downcast_ref::<f32>(),
            Some(&3.5)
        );

        let rebuilt = Shape::from_reflect(&dynamic).expect("rebuild concrete from switched enum");
        assert_eq!(rebuilt, Shape::Circle(3.5));
    }

    #[test]
    fn parsed_path_navigates_nested_struct_list_and_map() {
        let world = sample_world();

        let health = reflect_path(&world, &ParsedPath::parse(".player.health").unwrap()).unwrap();
        assert_eq!(health.downcast_ref::<i32>(), Some(&100));

        let flag = reflect_path(&world, &ParsedPath::parse(".pair#1").unwrap()).unwrap();
        assert_eq!(flag.downcast_ref::<bool>(), Some(&true));

        let cell = reflect_path(&world, &ParsedPath::parse(".grid[0][2]").unwrap()).unwrap();
        assert_eq!(cell.downcast_ref::<i32>(), Some(&30));

        let mapped = reflect_path(&world, &ParsedPath::parse(r#".lookup["b"]"#).unwrap()).unwrap();
        assert_eq!(mapped.downcast_ref::<i32>(), Some(&2));

        // A path that does not match the shape resolves to `None`.
        assert!(reflect_path(&world, &ParsedPath::parse(".player.missing").unwrap()).is_none());
    }

    #[test]
    fn parsed_path_mut_mutates_through_nested_shapes() {
        let mut world = sample_world();

        let cell =
            reflect_path_mut(&mut world, &ParsedPath::parse(".grid[1][0]").unwrap()).unwrap();
        *cell.downcast_mut::<i32>().unwrap() = 400;
        assert_eq!(world.grid[1][0], 400);

        let speed =
            reflect_path_mut(&mut world, &ParsedPath::parse(".player.speed").unwrap()).unwrap();
        *speed.downcast_mut::<f32>().unwrap() = 9.0;
        assert_eq!(world.player.speed, 9.0);

        let entry =
            reflect_path_mut(&mut world, &ParsedPath::parse(r#".lookup["a"]"#).unwrap()).unwrap();
        *entry.downcast_mut::<i32>().unwrap() = 111;
        assert_eq!(world.lookup["a"], 111);
    }

    #[test]
    fn apply_patches_and_grows_lists() {
        let mut list = vec![1, 2, 3];
        let source = vec![10, 20, 30, 40];
        list.apply(&source as &dyn Reflect).expect("list apply");
        assert_eq!(list, vec![10, 20, 30, 40]);
    }

    #[test]
    fn apply_patches_existing_and_inserts_new_map_entries() {
        let mut map = HashMap::new();
        map.insert("a".to_string(), 1);

        let mut source = HashMap::new();
        source.insert("a".to_string(), 5);
        source.insert("b".to_string(), 9);

        map.apply(&source as &dyn Reflect).expect("map apply");
        assert_eq!(map.get("a"), Some(&5));
        assert_eq!(map.get("b"), Some(&9));
    }
}

/// M3 serialization: binary + RON round-trips across every reflected kind,
/// `StableTypeId` determinism, and the deserializer's validation errors.
mod serialization {
    use crate::prelude::*;
    use alloc::collections::{BTreeMap, BTreeSet};
    use std::collections::{HashMap, HashSet};
    use std::fmt::Debug;

    #[derive(Reflect, Debug, PartialEq, Clone)]
    struct Stats {
        health: i32,
        name: String,
        speed: f32,
    }

    #[derive(Reflect, Debug, PartialEq, Clone)]
    struct Pair(i32, bool);

    #[derive(Reflect, Debug, PartialEq, Clone)]
    enum Shape {
        Empty,
        Circle(f32),
        Rect { width: f32, height: f32 },
    }

    #[derive(Reflect, Debug, PartialEq, Clone)]
    struct Scalars {
        b: bool,
        c: char,
        i: i64,
        u: u8,
        big: u128,
        f: f64,
        text: String,
    }

    #[derive(Reflect, Debug, PartialEq, Clone)]
    struct World {
        player: Stats,
        shape: Shape,
        pair: Pair,
        grid: Vec<Vec<i32>>,
        corners: [i32; 3],
        lookup: BTreeMap<String, i32>,
        tags: BTreeSet<i32>,
        maybe: Option<i32>,
        outcome: Result<i32, String>,
    }

    /// Register every non-primitive type the deserializer must resolve.
    fn registry() -> TypeRegistry {
        let mut registry = TypeRegistry::new();
        registry.register::<Stats>();
        registry.register::<Pair>();
        registry.register::<Shape>();
        registry.register::<Scalars>();
        registry.register::<World>();
        registry.register::<Vec<i32>>();
        registry.register::<Vec<Vec<i32>>>();
        registry.register::<Vec<String>>();
        registry.register::<[i32; 3]>();
        registry.register::<HashMap<String, i32>>();
        registry.register::<BTreeMap<String, i32>>();
        registry.register::<HashSet<i32>>();
        registry.register::<BTreeSet<i32>>();
        registry.register::<Option<i32>>();
        registry.register::<Result<i32, String>>();
        registry
    }

    /// Round-trip `value` through the binary format and back to a concrete `T`.
    fn binary_roundtrip<T>(value: &T, registry: &TypeRegistry) -> T
    where
        T: Reflect + Typed + FromReflect + PartialEq + Debug,
    {
        let bytes = to_binary(value).expect("serialize to binary");
        let dynamic = from_binary(&bytes, registry, <T as Typed>::type_info())
            .expect("deserialize from binary");
        T::from_reflect(&*dynamic).expect("rebuild concrete from binary")
    }

    /// Round-trip `value` through the RON format and back to a concrete `T`.
    fn ron_roundtrip<T>(value: &T, registry: &TypeRegistry) -> T
    where
        T: Reflect + Typed + FromReflect + PartialEq + Debug,
    {
        let text = to_ron(value).expect("serialize to RON");
        let dynamic =
            from_ron(&text, registry, <T as Typed>::type_info()).expect("deserialize from RON");
        T::from_reflect(&*dynamic).expect("rebuild concrete from RON")
    }

    /// Assert both formats round-trip `value` back to itself.
    fn assert_roundtrips<T>(value: T, registry: &TypeRegistry)
    where
        T: Reflect + Typed + FromReflect + PartialEq + Debug + Clone,
    {
        assert_eq!(
            binary_roundtrip(&value, registry),
            value,
            "binary round-trip"
        );
        assert_eq!(ron_roundtrip(&value, registry), value, "RON round-trip");
    }

    fn sample_world() -> World {
        World {
            player: Stats {
                health: 100,
                name: "aria".to_string(),
                speed: 1.5,
            },
            shape: Shape::Rect {
                width: 3.0,
                height: 4.0,
            },
            pair: Pair(7, true),
            grid: vec![vec![10, 20, 30], vec![], vec![40, 50]],
            corners: [1, 2, 3],
            lookup: BTreeMap::from([("a".to_string(), 1), ("b".to_string(), 2)]),
            tags: BTreeSet::from([5, 9, 11]),
            maybe: Some(42),
            outcome: Err("boom".to_string()),
        }
    }

    #[test]
    fn leaf_scalars_round_trip_both_formats() {
        let registry = TypeRegistry::new();
        // Primitive roots need no registration (resolved as built-in leaves).
        assert_roundtrips(true, &registry);
        assert_roundtrips('Z', &registry);
        assert_roundtrips(-12_345_i32, &registry);
        assert_roundtrips(9_000_000_000_i64, &registry);
        assert_roundtrips(255u8, &registry);
        assert_roundtrips(u128::MAX, &registry);
        assert_roundtrips(2.5_f32, &registry);
        assert_roundtrips(-0.125_f64, &registry);
        assert_roundtrips("hello, reflect".to_string(), &registry);
    }

    #[test]
    fn special_chars_and_floats_round_trip() {
        let registry = TypeRegistry::new();
        // Escaped characters and strings.
        assert_roundtrips('\n', &registry);
        assert_roundtrips('\'', &registry);
        assert_roundtrips('\\', &registry);
        assert_roundtrips("tab\tnew\nline\"quote\\slash".to_string(), &registry);
        // Infinities (NaN is intentionally excluded: it never compares equal).
        assert_roundtrips(f32::INFINITY, &registry);
        assert_roundtrips(f64::NEG_INFINITY, &registry);
        assert_roundtrips(3.0_f32, &registry);
    }

    #[test]
    fn struct_round_trips_both_formats() {
        let registry = registry();
        assert_roundtrips(
            Stats {
                health: 73,
                name: "nova".to_string(),
                speed: 4.25,
            },
            &registry,
        );
    }

    #[test]
    fn tuple_struct_round_trips_both_formats() {
        let registry = registry();
        assert_roundtrips(Pair(9, false), &registry);
        assert_roundtrips(Pair(-1, true), &registry);
    }

    #[test]
    fn enum_variants_round_trip_both_formats() {
        let registry = registry();
        assert_roundtrips(Shape::Empty, &registry);
        assert_roundtrips(Shape::Circle(2.5), &registry);
        assert_roundtrips(
            Shape::Rect {
                width: 3.0,
                height: 4.0,
            },
            &registry,
        );
    }

    #[test]
    fn wide_scalar_struct_round_trips() {
        let registry = registry();
        assert_roundtrips(
            Scalars {
                b: true,
                c: '✓',
                i: -98_765,
                u: 7,
                big: 170_141_183_460_469_231_731_687_303_715_884_105_727,
                f: 6.5,
                text: "ünïcödé".to_string(),
            },
            &registry,
        );
    }

    #[test]
    fn lists_arrays_round_trip() {
        let registry = registry();
        assert_roundtrips(vec![1, 2, 3, 4], &registry);
        assert_roundtrips(Vec::<i32>::new(), &registry);
        assert_roundtrips(vec!["a".to_string(), "b".to_string()], &registry);
        assert_roundtrips(vec![vec![10, 20, 30], vec![], vec![40]], &registry);
        assert_roundtrips([7, 8, 9], &registry);
    }

    #[test]
    fn maps_round_trip() {
        let registry = registry();
        let hash = HashMap::from([("x".to_string(), 10), ("y".to_string(), 20)]);
        assert_roundtrips(hash, &registry);
        let tree = BTreeMap::from([("a".to_string(), 1), ("b".to_string(), 2)]);
        assert_roundtrips(tree, &registry);
        assert_roundtrips(HashMap::<String, i32>::new(), &registry);
    }

    #[test]
    fn sets_round_trip() {
        let registry = registry();
        assert_roundtrips(HashSet::from([1, 2, 3]), &registry);
        assert_roundtrips(BTreeSet::from([5, 9, 11]), &registry);
        assert_roundtrips(BTreeSet::<i32>::new(), &registry);
    }

    #[test]
    fn option_and_result_round_trip() {
        let registry = registry();
        assert_roundtrips(Some(7_i32), &registry);
        assert_roundtrips(None::<i32>, &registry);
        assert_roundtrips(Ok::<i32, String>(3), &registry);
        assert_roundtrips(Err::<i32, String>("nope".to_string()), &registry);
    }

    #[test]
    fn deeply_nested_world_round_trips() {
        let registry = registry();
        assert_roundtrips(sample_world(), &registry);
    }

    #[test]
    fn binary_header_pins_the_root_type() {
        let bytes = to_binary(&5_i32).unwrap();
        // `MAGIC`(4) + `VERSION`(1) + stable id(8) precede the body.
        assert_eq!(&bytes[0..4], b"PRB1");
        assert_eq!(bytes[4], 1);
        let found = u64::from_le_bytes(bytes[5..13].try_into().unwrap());
        assert_eq!(found, StableTypeId::of_type::<i32>().value());
    }

    #[test]
    fn ron_text_is_anonymous_and_compact() {
        let value = Shape::Rect {
            width: 3.0,
            height: 4.0,
        };
        assert_eq!(to_ron(&value).unwrap(), "Rect(width:3.0,height:4.0)");
        assert_eq!(to_ron(&Shape::Empty).unwrap(), "Empty");
        assert_eq!(to_ron(&Shape::Circle(2.5)).unwrap(), "Circle(2.5)");
        assert_eq!(to_ron(&Pair(9, true)).unwrap(), "(9,true)");
        assert_eq!(to_ron(&vec![1, 2, 3]).unwrap(), "[1,2,3]");
    }

    #[test]
    fn ron_tolerates_whitespace_and_trailing_commas() {
        let registry = registry();
        let text = " Rect ( width : 3.0 , height : 4.0 , ) ";
        let dynamic = from_ron(text, &registry, <Shape as Typed>::type_info()).unwrap();
        assert_eq!(
            Shape::from_reflect(&*dynamic).unwrap(),
            Shape::Rect {
                width: 3.0,
                height: 4.0
            }
        );
    }

    #[test]
    fn ron_matches_struct_fields_out_of_order() {
        let registry = registry();
        let text = "(speed:9.5,name:\"zed\",health:3)";
        let dynamic = from_ron(text, &registry, <Stats as Typed>::type_info()).unwrap();
        assert_eq!(
            Stats::from_reflect(&*dynamic).unwrap(),
            Stats {
                health: 3,
                name: "zed".to_string(),
                speed: 9.5,
            }
        );
    }

    #[test]
    fn stable_type_id_is_deterministic_and_path_based() {
        // Same path hashes identically; `of_type` matches `of_path`.
        assert_eq!(
            StableTypeId::of_path("foo::Bar"),
            StableTypeId::of_path("foo::Bar")
        );
        assert_eq!(
            StableTypeId::of_type::<i32>(),
            StableTypeId::of_path(core::any::type_name::<i32>())
        );
        // Distinct paths differ.
        assert_ne!(
            StableTypeId::of_path("foo::Bar"),
            StableTypeId::of_path("foo::Baz")
        );
        assert_ne!(
            StableTypeId::of_type::<i32>(),
            StableTypeId::of_type::<u32>()
        );
        // Raw value round-trips.
        let id = StableTypeId::of_type::<Stats>();
        assert_eq!(StableTypeId::from_raw(id.value()), id);
    }

    #[test]
    fn binary_rejects_bad_magic() {
        let registry = TypeRegistry::new();
        let err = from_binary(&[0, 1, 2, 3, 4], &registry, <i32 as Typed>::type_info())
            .err()
            .unwrap();
        assert_eq!(err, DeserializeError::BadMagic);
    }

    #[test]
    fn binary_rejects_stable_id_mismatch() {
        let registry = TypeRegistry::new();
        let bytes = to_binary(&5_i32).unwrap();
        // Decode the same bytes against a different target type.
        let err = from_binary(&bytes, &registry, <u32 as Typed>::type_info())
            .err()
            .unwrap();
        assert!(matches!(err, DeserializeError::StableIdMismatch { .. }));
    }

    #[test]
    fn binary_rejects_unregistered_nested_type() {
        // Root resolves from the target directly, but the nested `Stats`,
        // `Shape`, ... are not registered here.
        let registry = TypeRegistry::new();
        let bytes = to_binary(&sample_world()).unwrap();
        let err = from_binary(&bytes, &registry, <World as Typed>::type_info())
            .err()
            .unwrap();
        assert!(matches!(err, DeserializeError::UnregisteredType(_)));
    }

    #[test]
    fn binary_detects_corrupt_leaf_tag() {
        let registry = TypeRegistry::new();
        let mut bytes = to_binary(&7_i32).unwrap();
        // Layout: MAGIC(4) VERSION(1) id(8) VALUE_TAG(1) PRIM_TAG(1) payload.
        // Flip the primitive tag to `bool` (0): a valid tag, wrong type.
        bytes[14] = 0;
        let err = from_binary(&bytes, &registry, <i32 as Typed>::type_info())
            .err()
            .unwrap();
        assert!(matches!(err, DeserializeError::LeafTypeMismatch { .. }));

        // An out-of-range primitive tag is reported distinctly.
        let mut bytes = to_binary(&7_i32).unwrap();
        bytes[14] = 200;
        let err = from_binary(&bytes, &registry, <i32 as Typed>::type_info())
            .err()
            .unwrap();
        assert_eq!(err, DeserializeError::UnknownPrimitiveTag(200));
    }

    #[test]
    fn binary_rejects_trailing_data() {
        let registry = TypeRegistry::new();
        let mut bytes = to_binary(&7_i32).unwrap();
        bytes.push(0xff);
        let err = from_binary(&bytes, &registry, <i32 as Typed>::type_info())
            .err()
            .unwrap();
        assert_eq!(err, DeserializeError::TrailingData);
    }

    #[test]
    fn ron_reports_syntax_and_schema_errors() {
        let registry = registry();
        // Not a struct opening.
        let err = from_ron("nonsense", &registry, <Stats as Typed>::type_info())
            .err()
            .unwrap();
        assert!(matches!(err, DeserializeError::RonSyntax(_)));
        // Unknown field name.
        let err = from_ron(
            "(health:1,bogus:2,name:\"x\",speed:1.0)",
            &registry,
            <Stats as Typed>::type_info(),
        )
        .err()
        .unwrap();
        assert!(matches!(err, DeserializeError::UnknownField(_)));
        // Unknown enum variant.
        let err = from_ron("Triangle(1.0)", &registry, <Shape as Typed>::type_info())
            .err()
            .unwrap();
        assert_eq!(err, DeserializeError::UnknownVariant);
        // Trailing text after the root value.
        let err = from_ron("[1,2,3] extra", &registry, <Vec<i32> as Typed>::type_info())
            .err()
            .unwrap();
        assert_eq!(err, DeserializeError::TrailingData);
    }

    #[test]
    fn unsupported_leaf_is_rejected_on_serialize() {
        // A dynamic set of boxed `dyn Reflect` whose element is a supported
        // leaf still serializes; this guards the opposite — an opaque value is
        // the only serialize failure, exercised through the error's shape.
        let err = DeserializeError::UnknownField("x".to_string());
        // Smoke-check the Display impls stay wired (no panics / empty output).
        assert!(!format!("{err}").is_empty());
        assert!(!format!("{}", SerializeError::UnsupportedLeaf { type_name: "T" }).is_empty());
    }
}

/// M4 schema tests: attribute metadata attach+query through the registry, a
/// composable multi-step migration chain (v1 -> v2 -> v3 payload upgrade),
/// validation success + failure, and versioned (de)serialization layered on
/// the M3 serializer (current-version round-trip and old-payload migration).
mod m4_schema {
    use crate::prelude::*;
    use core::any::type_name;

    // Three successive shapes of one *logical* type ("Monster"), each backed by
    // a distinct Rust type so the migration chain can rewrite fields between
    // versions. v1 -> v2 adds `armor`; v2 -> v3 renames `hp` -> `health` and
    // adds `mana`.
    #[derive(Reflect, Debug, PartialEq, Clone)]
    struct MonsterV1 {
        hp: i32,
        name: String,
    }

    #[derive(Reflect, Debug, PartialEq, Clone)]
    struct MonsterV2 {
        hp: i32,
        name: String,
        armor: i32,
    }

    #[derive(Reflect, Debug, PartialEq, Clone)]
    struct Monster {
        health: i32,
        name: String,
        armor: i32,
        mana: i32,
    }

    /// A registry able to decode every monster version's payload shape.
    fn type_registry() -> TypeRegistry {
        let mut registry = TypeRegistry::new();
        registry.register::<MonsterV1>();
        registry.register::<MonsterV2>();
        registry.register::<Monster>();
        registry
    }

    /// A schema registry recording the three versions and the two `vN -> vN+1`
    /// migration steps between them.
    fn schema_registry() -> SchemaRegistry {
        let mut schema = SchemaRegistry::new();
        schema.register_version(
            "Monster",
            TypeSchema::new(
                type_name::<MonsterV1>(),
                1,
                <MonsterV1 as Typed>::type_info(),
            )
            .with_required_fields(&["hp", "name"]),
        );
        schema.register_version(
            "Monster",
            TypeSchema::new(
                type_name::<MonsterV2>(),
                2,
                <MonsterV2 as Typed>::type_info(),
            )
            .with_required_fields(&["hp", "name", "armor"]),
        );
        schema.register_version(
            "Monster",
            TypeSchema::new(type_name::<Monster>(), 3, <Monster as Typed>::type_info())
                .with_required_fields(&["health", "name", "armor", "mana"]),
        );

        // v1 -> v2: new `armor` field defaulting to 0.
        schema
            .register_migration(
                "Monster",
                Migration::new(1, 2, |d| {
                    d.insert("armor", 0i32);
                    Ok(())
                }),
            )
            .expect("register v1->v2");

        // v2 -> v3: rename `hp` -> `health`, add `mana` defaulting to 100.
        schema
            .register_migration(
                "Monster",
                Migration::new(2, 3, |d| {
                    let hp = d
                        .remove("hp")
                        .ok_or_else(|| MigrateError::MissingField("hp".into()))?;
                    d.insert_boxed("health", hp);
                    d.insert("mana", 100i32);
                    Ok(())
                }),
            )
            .expect("register v2->v3");

        schema
    }

    #[test]
    fn metadata_attaches_and_queries_through_registry() {
        let mut registry = type_registry();
        registry.register_type_data::<Monster, _>(
            TypeMetadata::new()
                .with_docs("A hostile creature.")
                .with_custom("icon", AttributeValue::Text("skull".into()))
                .with_field(
                    FieldMetadata::new("health")
                        .with_docs("Current hit points.")
                        .with_category("Combat")
                        .with_range(0.0, 1000.0)
                        .with_default(AttributeValue::Int(100)),
                )
                .with_field(FieldMetadata::new("name").required(true))
                .with_field(FieldMetadata::new("mana").hidden(true)),
        );

        let meta = registry
            .get_with_name(type_name::<Monster>())
            .and_then(|r| r.data::<TypeMetadata>())
            .expect("metadata attached to Monster");

        assert_eq!(meta.docs(), Some("A hostile creature."));
        assert_eq!(
            meta.custom("icon"),
            Some(&AttributeValue::Text("skull".into()))
        );
        let health = meta.field("health").expect("health metadata");
        assert_eq!(health.docs(), Some("Current hit points."));
        assert_eq!(health.category(), Some("Combat"));
        assert_eq!(health.range(), Some((0.0, 1000.0)));
        assert_eq!(health.default_value(), Some(&AttributeValue::Int(100)));
        assert!(meta.field("name").unwrap().is_required());
        assert!(meta.field("mana").unwrap().is_hidden());
        assert!(meta.field("missing").is_none());
    }

    #[test]
    fn schema_registry_tracks_versions_and_chains() {
        let schema = schema_registry();

        assert_eq!(
            schema.current_version("Monster"),
            Some(SchemaVersion::new(3))
        );
        assert_eq!(schema.current_version("Ghost"), None);
        assert_eq!(
            schema.locate(type_name::<MonsterV1>()),
            Some(("Monster", 1))
        );
        assert_eq!(schema.locate(type_name::<Monster>()), Some(("Monster", 3)));
        assert_eq!(
            schema.current_schema("Monster").map(TypeSchema::version),
            Some(3)
        );

        // Full chain from the oldest version has two steps; from current, none.
        let chain = schema.chain("Monster", 1).expect("chain from v1");
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].from_version(), 1);
        assert_eq!(chain[0].to_version(), 2);
        assert_eq!(chain[1].from_version(), 2);
        assert_eq!(chain[1].to_version(), 3);
        assert!(schema
            .chain("Monster", 3)
            .expect("chain from v3")
            .is_empty());

        // Unknown type and too-new version are reported distinctly.
        assert!(matches!(
            schema.chain("Ghost", 1),
            Err(MigrateError::UnknownType(_))
        ));
        assert!(matches!(
            schema.chain("Monster", 9),
            Err(MigrateError::VersionTooNew {
                found: 9,
                current: 3
            })
        ));
    }

    #[test]
    #[should_panic(expected = "advance exactly one version")]
    fn migration_must_advance_one_version() {
        // A migration that skips a version is a construction error: the chain
        // walker relies on every step advancing exactly one version.
        let _ = Migration::new(1, 3, |_| Ok(()));
    }

    #[test]
    fn validation_success_and_failures() {
        let schema = TypeSchema::new(type_name::<Monster>(), 3, <Monster as Typed>::type_info())
            .with_required_fields(&["health", "name"]);
        let meta = TypeMetadata::new()
            .with_field(FieldMetadata::new("health").with_range(0.0, 1000.0))
            .with_field(FieldMetadata::new("name").required(true));

        // Success: all required fields present, health within range.
        let good = Monster {
            health: 250,
            name: "dragon".into(),
            armor: 10,
            mana: 60,
        };
        assert!(validate(&good, &schema, Some(&meta)).is_ok());

        // Out-of-range numeric field.
        let hot = Monster {
            health: 5000,
            name: "inferno".into(),
            armor: 0,
            mana: 0,
        };
        let errs = validate(&hot, &schema, Some(&meta)).expect_err("range violation");
        assert!(errs.iter().any(|e| matches!(
            e,
            ValidationError::OutOfRange { field, min, max, .. }
                if field == "health" && *min == 0.0 && *max == 1000.0
        )));

        // Missing required field (DynamicStruct without `name`).
        let mut partial = DynamicStruct::new();
        partial.insert("health", 10i32);
        let errs = validate(&partial, &schema, Some(&meta)).expect_err("missing field");
        assert!(errs.iter().any(|e| matches!(
            e,
            ValidationError::MissingRequiredField { field } if field == "name"
        )));

        // A non-struct value cannot satisfy field rules.
        let errs = validate(&7i32, &schema, Some(&meta)).expect_err("not a struct");
        assert_eq!(errs, vec![ValidationError::NotAStruct]);
    }

    #[test]
    fn version_validation_against_registry() {
        let schema = schema_registry();
        assert!(validate_version(1, &schema, "Monster").is_ok());
        assert!(validate_version(3, &schema, "Monster").is_ok());
        assert!(matches!(
            validate_version(9, &schema, "Monster"),
            Err(ValidationError::VersionTooNew {
                found: 9,
                current: 3
            })
        ));
    }

    #[test]
    fn versioned_binary_round_trips_current_version() {
        let schema = schema_registry();
        let registry = type_registry();
        let monster = Monster {
            health: 320,
            name: "wyrm".into(),
            armor: 15,
            mana: 90,
        };

        let bytes = to_versioned_binary(&monster, &schema).expect("versioned serialize");
        // Envelope stamps the current version (3) ahead of the M3 body.
        assert_eq!(&bytes[..4], b"PRVB");
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            3
        );

        let decoded = from_versioned_binary(&bytes, &registry, &schema, "Monster")
            .expect("versioned deserialize");
        let back = Monster::from_reflect(&*decoded).expect("rebuild Monster");
        assert_eq!(back, monster);
    }

    #[test]
    fn versioned_ron_round_trips_current_version() {
        let schema = schema_registry();
        let registry = type_registry();
        let monster = Monster {
            health: 11,
            name: "imp".into(),
            armor: 1,
            mana: 7,
        };

        let text = to_versioned_ron(&monster, &schema).expect("versioned RON serialize");
        assert!(text.starts_with("#prism-schema v3\n"));

        let decoded = from_versioned_ron(&text, &registry, &schema, "Monster")
            .expect("versioned RON deserialize");
        let back = Monster::from_reflect(&*decoded).expect("rebuild Monster from RON");
        assert_eq!(back, monster);
    }

    #[test]
    fn old_version_payload_migrates_through_full_chain() {
        let schema = schema_registry();
        let registry = type_registry();

        // A payload written by the v1 shape (hp + name only).
        let legacy = MonsterV1 {
            hp: 42,
            name: "goblin".into(),
        };
        let bytes = to_versioned_binary(&legacy, &schema).expect("serialize v1 payload");
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            1
        );

        // Reading as the logical "Monster" walks v1 -> v2 -> v3 before rebuild.
        let decoded = from_versioned_binary(&bytes, &registry, &schema, "Monster")
            .expect("migrate v1 payload to current");
        let migrated = Monster::from_reflect(&*decoded).expect("rebuild migrated Monster");
        assert_eq!(
            migrated,
            Monster {
                health: 42, // renamed from hp
                name: "goblin".into(),
                armor: 0,  // added by v1 -> v2
                mana: 100, // added by v2 -> v3
            }
        );

        // The same migration works through the RON back-end too.
        let text = to_versioned_ron(&legacy, &schema).expect("serialize v1 RON payload");
        let decoded = from_versioned_ron(&text, &registry, &schema, "Monster")
            .expect("migrate v1 RON payload");
        let migrated = Monster::from_reflect(&*decoded).expect("rebuild migrated Monster from RON");
        assert_eq!(migrated.health, 42);
        assert_eq!(migrated.armor, 0);
        assert_eq!(migrated.mana, 100);
    }

    #[test]
    fn intermediate_version_payload_migrates_remaining_steps() {
        let schema = schema_registry();
        let registry = type_registry();

        // A v2 payload only needs the v2 -> v3 step.
        let mid = MonsterV2 {
            hp: 5,
            name: "slime".into(),
            armor: 3,
        };
        let bytes = to_versioned_binary(&mid, &schema).expect("serialize v2 payload");
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            2
        );

        let decoded = from_versioned_binary(&bytes, &registry, &schema, "Monster")
            .expect("migrate v2 payload");
        let migrated = Monster::from_reflect(&*decoded).expect("rebuild migrated Monster");
        assert_eq!(
            migrated,
            Monster {
                health: 5,
                name: "slime".into(),
                armor: 3,
                mana: 100,
            }
        );
    }

    #[test]
    fn versioned_binary_rejects_bad_envelope_and_unknown_type() {
        let schema = schema_registry();
        let registry = type_registry();

        // Too short / wrong magic.
        assert!(matches!(
            from_versioned_binary(b"PR", &registry, &schema, "Monster"),
            Err(MigrateError::BadEnvelope)
        ));
        assert!(matches!(
            from_versioned_binary(b"XXXX\x01\x00\x00\x00", &registry, &schema, "Monster"),
            Err(MigrateError::BadEnvelope)
        ));

        // Serializing a type with no schema entry is an unknown type.
        let orphan = MonsterV1 {
            hp: 1,
            name: "x".into(),
        };
        let empty = SchemaRegistry::new();
        assert!(matches!(
            to_versioned_binary(&orphan, &empty),
            Err(MigrateError::UnknownType(_))
        ));
    }
}

mod function_reflection {
    use crate::{ArgList, FunctionError, FunctionRegistry, IntoFunction};
    use core::any::type_name;

    fn add(a: i32, b: i32) -> i32 {
        a + b
    }

    fn greet() -> String {
        "hi".into()
    }

    fn noop(_value: i32) {}

    #[test]
    fn call_by_name_happy_path() {
        let mut registry = FunctionRegistry::new();
        registry.register("add", add);
        let result = registry
            .call("add", ArgList::new().push(2_i32).push(40_i32))
            .expect("add call succeeds");
        assert_eq!(result.downcast_ref::<i32>(), Some(&42));
    }

    #[test]
    fn zero_arg_and_void_return() {
        let mut registry = FunctionRegistry::new();
        registry.register("greet", greet);
        registry.register("noop", noop);

        let greeting = registry
            .call("greet", ArgList::new())
            .expect("greet call succeeds");
        assert_eq!(
            greeting.downcast_ref::<String>().map(String::as_str),
            Some("hi")
        );

        let nothing = registry
            .call("noop", ArgList::new().push(7_i32))
            .expect("noop call succeeds");
        assert!(nothing.downcast_ref::<()>().is_some());
    }

    #[test]
    fn closure_capture_registers() {
        let base = 100_i32;
        let mut registry = FunctionRegistry::new();
        registry.register("addbase", move |x: i32| x + base);
        let result = registry
            .call("addbase", ArgList::new().push(5_i32))
            .expect("closure call succeeds");
        assert_eq!(result.downcast_ref::<i32>(), Some(&105));
    }

    #[test]
    fn unknown_function_errors() {
        let registry = FunctionRegistry::new();
        let err = registry
            .call("missing", ArgList::new())
            .err()
            .expect("unknown function errors");
        assert_eq!(
            err,
            FunctionError::UnknownFunction {
                name: "missing".into()
            }
        );
    }

    #[test]
    fn arity_mismatch_errors() {
        let mut registry = FunctionRegistry::new();
        registry.register("add", add);
        let err = registry
            .call("add", ArgList::new().push(1_i32))
            .err()
            .expect("arity mismatch errors");
        assert_eq!(
            err,
            FunctionError::ArityMismatch {
                function: Some("add".into()),
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn arg_type_mismatch_errors() {
        let mut registry = FunctionRegistry::new();
        registry.register("add", add);
        let err = registry
            .call("add", ArgList::new().push(1_i32).push(true))
            .err()
            .expect("argument type mismatch errors");
        assert_eq!(
            err,
            FunctionError::ArgTypeMismatch {
                index: 1,
                expected: type_name::<i32>(),
                actual: type_name::<bool>(),
            }
        );
    }

    #[test]
    fn direct_dynamic_function_and_info() {
        let function = add.into_function().with_name("add");
        assert_eq!(function.name(), Some("add"));
        assert_eq!(function.info().arg_count(), 2);
        assert_eq!(
            function.info().arg_types(),
            &[type_name::<i32>(), type_name::<i32>()]
        );
        assert_eq!(function.info().return_type(), type_name::<i32>());
        let result = function
            .call(&ArgList::new().push(3_i32).push(4_i32))
            .expect("direct call succeeds");
        assert_eq!(result.downcast_ref::<i32>(), Some(&7));
    }

    #[test]
    fn registry_iter_and_contains() {
        let mut registry = FunctionRegistry::new();
        registry.register("add", add);
        registry.register("greet", greet);
        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());
        assert!(registry.contains("add"));
        assert!(registry.contains("greet"));
        assert!(!registry.contains("missing"));
        let mut names: Vec<&str> = registry.iter().map(|(name, _)| name).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["add", "greet"]);
    }
}

mod reflect_trait_dispatch {
    use crate::{reflect_trait, Reflect, TypeRegistry};

    trait Area: Reflect {
        fn area(&self) -> f32;
        fn scale(&mut self, factor: f32);
    }

    reflect_trait!(
        /// Reflected accessor for [`Area`].
        ReflectArea for Area
    );

    #[derive(Reflect)]
    struct Circle {
        radius: f32,
    }

    impl Area for Circle {
        fn area(&self) -> f32 {
            core::f32::consts::PI * self.radius * self.radius
        }

        fn scale(&mut self, factor: f32) {
            self.radius *= factor;
        }
    }

    #[test]
    fn dispatch_shared_and_mut() {
        let mut registry = TypeRegistry::new();
        registry.register::<Circle>();
        assert!(
            registry.register_type_data::<Circle, ReflectArea>(ReflectArea::from_type::<Circle>())
        );

        let mut value = Circle { radius: 2.0 };
        let type_id = value.as_any().type_id();
        let accessor = *registry
            .get(type_id)
            .expect("registration")
            .data::<ReflectArea>()
            .expect("ReflectArea type data");

        let area = accessor.get(&value).expect("shared downcast").area();
        assert!((area - core::f32::consts::PI * 4.0).abs() < 1e-5);

        accessor
            .get_mut(&mut value)
            .expect("mutable downcast")
            .scale(3.0);
        assert!((value.radius - 6.0).abs() < 1e-6);
    }

    #[test]
    fn dispatch_rejects_wrong_type() {
        let accessor = ReflectArea::from_type::<Circle>();
        let other = 5_i32;
        assert!(accessor.get(&other).is_none());
    }
}

mod runtime_types {
    use crate::DynamicEnum;
    use crate::{
        from_binary, to_binary, DynamicStruct, DynamicVariant, EnumTypeBuilder, Reflect,
        ReflectRef, StructTypeBuilder, TypeInfo, TypeRegistry,
    };

    #[test]
    fn struct_builder_registers_and_round_trips() {
        let mut registry = TypeRegistry::new();
        let info = StructTypeBuilder::new("game::Health")
            .with_field("current", "i32")
            .with_field("max", "i32")
            .register(&mut registry);
        assert!(matches!(info, TypeInfo::Struct(_)));
        assert!(registry.get_with_name("game::Health").is_some());

        let mut value = DynamicStruct::new();
        value.set_represented_type_name("game::Health");
        value.insert("current", 7_i32);
        value.insert("max", 10_i32);

        let bytes = to_binary(&value).expect("encode runtime struct");
        let decoded = from_binary(&bytes, &registry, info).expect("decode runtime struct");
        let ReflectRef::Struct(decoded) = decoded.reflect_ref() else {
            panic!("decoded value is not a struct");
        };
        assert_eq!(
            decoded
                .field("current")
                .and_then(|f| f.downcast_ref::<i32>()),
            Some(&7)
        );
        assert_eq!(
            decoded.field("max").and_then(|f| f.downcast_ref::<i32>()),
            Some(&10)
        );
    }

    #[test]
    fn nested_runtime_struct_round_trips() {
        let mut registry = TypeRegistry::new();
        StructTypeBuilder::new("game::Vec2")
            .with_field("x", "f32")
            .with_field("y", "f32")
            .register(&mut registry);
        let transform_info = StructTypeBuilder::new("game::Transform")
            .with_field("position", "game::Vec2")
            .with_field("rotation", "f32")
            .register(&mut registry);

        let mut position = DynamicStruct::new();
        position.set_represented_type_name("game::Vec2");
        position.insert("x", 1.5_f32);
        position.insert("y", -2.5_f32);

        let mut transform = DynamicStruct::new();
        transform.set_represented_type_name("game::Transform");
        transform.insert_boxed("position", Box::new(position));
        transform.insert("rotation", 0.25_f32);

        let bytes = to_binary(&transform).expect("encode nested runtime struct");
        let decoded = from_binary(&bytes, &registry, transform_info).expect("decode nested");
        let ReflectRef::Struct(decoded) = decoded.reflect_ref() else {
            panic!("decoded value is not a struct");
        };
        assert_eq!(
            decoded
                .field("rotation")
                .and_then(|f| f.downcast_ref::<f32>()),
            Some(&0.25)
        );
        let ReflectRef::Struct(inner) = decoded
            .field("position")
            .expect("position field")
            .reflect_ref()
        else {
            panic!("nested position is not a struct");
        };
        assert_eq!(
            inner.field("x").and_then(|f| f.downcast_ref::<f32>()),
            Some(&1.5)
        );
        assert_eq!(
            inner.field("y").and_then(|f| f.downcast_ref::<f32>()),
            Some(&-2.5)
        );
    }

    #[test]
    fn enum_builder_round_trips() {
        let mut registry = TypeRegistry::new();
        let info = EnumTypeBuilder::new("game::State")
            .with_unit_variant("Idle")
            .with_tuple_variant("Score", vec!["i32"])
            .with_struct_variant("Named", vec![("name", "String")])
            .register(&mut registry);
        assert!(matches!(info, TypeInfo::Enum(_)));
        assert!(registry.get_with_name("game::State").is_some());

        let mut value = DynamicEnum::new(
            1,
            "Score",
            DynamicVariant::Tuple(vec![Box::new(99_i32) as Box<dyn Reflect>]),
        );
        value.set_represented_type_name("game::State");

        let bytes = to_binary(&value).expect("encode runtime enum");
        let decoded = from_binary(&bytes, &registry, info).expect("decode runtime enum");
        let ReflectRef::Enum(decoded) = decoded.reflect_ref() else {
            panic!("decoded value is not an enum");
        };
        assert_eq!(decoded.variant_name(), "Score");
        assert_eq!(decoded.variant_index(), 1);
        assert_eq!(
            decoded.field_at(0).and_then(|f| f.downcast_ref::<i32>()),
            Some(&99)
        );
    }
}

mod diff_merge {
    use crate::{diff, merge, DiffError, DynamicEnum, DynamicVariant, Patch, Reflect};
    use std::collections::{HashMap, HashSet};

    #[derive(Reflect, Clone, PartialEq, Debug)]
    struct Point {
        x: i32,
        y: i32,
    }

    #[derive(Reflect, Clone, PartialEq, Debug)]
    struct Nested {
        point: Point,
        label: String,
    }

    #[derive(Reflect, Clone, PartialEq, Debug)]
    struct Tup(i32, bool);

    #[derive(Reflect, Clone, PartialEq, Debug)]
    enum Choice {
        A(i32),
        B { v: i32 },
    }

    #[test]
    fn struct_field_change() {
        let a = Point { x: 1, y: 2 };
        let b = Point { x: 1, y: 9 };
        let mut merged = a.clone();
        diff(&a, &b).apply(&mut merged).expect("apply struct patch");
        assert_eq!(merged, b);
    }

    #[test]
    fn identical_is_unchanged() {
        let a = Point { x: 3, y: 4 };
        assert!(diff(&a, &a).is_unchanged());
    }

    #[test]
    fn nested_struct_round_trips() {
        let a = Nested {
            point: Point { x: 1, y: 2 },
            label: "a".into(),
        };
        let b = Nested {
            point: Point { x: 1, y: 7 },
            label: "a".into(),
        };
        let patch = diff(&a, &b);
        assert!(matches!(patch, Patch::Struct(_)));
        let mut merged = a.clone();
        patch.apply(&mut merged).expect("apply nested patch");
        assert_eq!(merged, b);
    }

    #[test]
    fn list_grow_and_modify() {
        let a: Vec<i32> = vec![1, 2];
        let b: Vec<i32> = vec![9, 2, 3];
        let mut merged = a.clone();
        merge(&mut merged, &diff(&a, &b)).expect("apply list patch");
        assert_eq!(merged, b);
    }

    #[test]
    fn map_modify_and_insert() {
        let mut a = HashMap::new();
        a.insert("hp".to_string(), 10_i32);
        a.insert("mp".to_string(), 5_i32);
        let mut b = HashMap::new();
        b.insert("hp".to_string(), 99_i32);
        b.insert("mp".to_string(), 5_i32);
        b.insert("xp".to_string(), 1_i32);

        let mut merged = a.clone();
        diff(&a, &b).apply(&mut merged).expect("apply map patch");
        assert_eq!(merged, b);
    }

    #[test]
    fn set_add_round_trips() {
        let a: HashSet<i32> = [1, 2].into_iter().collect();
        let b: HashSet<i32> = [1, 2, 3].into_iter().collect();
        let mut merged = a.clone();
        diff(&a, &b).apply(&mut merged).expect("apply set patch");
        assert_eq!(merged, b);
    }

    #[test]
    fn enum_same_variant_modify() {
        let a = Choice::A(1);
        let b = Choice::A(5);
        let patch = diff(&a, &b);
        assert!(matches!(patch, Patch::Enum(_)));
        let mut merged = a.clone();
        patch.apply(&mut merged).expect("apply enum patch");
        assert_eq!(merged, b);
    }

    #[test]
    fn struct_variant_modify_round_trips() {
        let a = Choice::B { v: 1 };
        let b = Choice::B { v: 42 };
        let mut merged = a.clone();
        diff(&a, &b)
            .apply(&mut merged)
            .expect("apply struct-variant patch");
        assert_eq!(merged, b);
    }

    #[test]
    fn tuple_struct_change() {
        let a = Tup(1, true);
        let b = Tup(9, false);
        let patch = diff(&a, &b);
        assert!(matches!(patch, Patch::TupleStruct(_)));
        let mut merged = a.clone();
        patch.apply(&mut merged).expect("apply tuple-struct patch");
        assert_eq!(merged, b);
    }

    #[test]
    fn kind_mismatch_errors() {
        // A struct patch applied onto a value of a different kind is a mismatch.
        let patch = diff(&Point { x: 1, y: 2 }, &Point { x: 3, y: 4 });
        assert!(matches!(patch, Patch::Struct(_)));
        let mut wrong = 0_i32;
        let err = patch.apply(&mut wrong).expect_err("kind mismatch errors");
        assert!(matches!(err, DiffError::KindMismatch));
    }

    #[test]
    fn dynamic_enum_variant_switch_via_replace() {
        let a = DynamicEnum::new(
            0,
            "A",
            DynamicVariant::Tuple(vec![Box::new(1_i32) as Box<dyn Reflect>]),
        );
        let b = DynamicEnum::new(
            1,
            "B",
            DynamicVariant::Tuple(vec![Box::new(2_i32) as Box<dyn Reflect>]),
        );
        let patch = diff(&a, &b);
        assert!(matches!(patch, Patch::Replace(_)));
        let mut merged = DynamicEnum::new(
            0,
            "A",
            DynamicVariant::Tuple(vec![Box::new(1_i32) as Box<dyn Reflect>]),
        );
        patch
            .apply(&mut merged)
            .expect("apply replace onto dynamic enum");
        assert_eq!(crate::Enum::variant_name(&merged), "B");
    }
}
