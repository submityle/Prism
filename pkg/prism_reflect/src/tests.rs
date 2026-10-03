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
    use std::boxed::Box;
    use std::vec;
    use std::vec::Vec;

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
    use std::vec::Vec;

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
    use std::boxed::Box;
    use std::collections::BTreeMap;
    use std::string::String;
    use std::string::ToString;

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
    use std::boxed::Box;
    use std::collections::BTreeSet;

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
    use crate::{
        GetTypeRegistration, Reflect, ReflectDefault, TypeInfo, TypeRegistry,
    };
    use core::any::{TypeId, type_name};

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
        assert!(registry.register_type_data::<Config, ReflectDefault>(
            ReflectDefault::new::<Config>()
        ));
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
        assert_eq!(<Vec2 as Typed>::type_info().type_name(), ::core::any::type_name::<Vec2>());
        assert_eq!(<Vec3A as Typed>::type_info().type_name(), ::core::any::type_name::<Vec3A>());
        assert_eq!(<Vec4 as Typed>::type_info().type_name(), ::core::any::type_name::<Vec4>());
        assert_eq!(<Quat as Typed>::type_info().type_name(), ::core::any::type_name::<Quat>());

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
    use prism_math::Vec3;
    use std::boxed::Box;
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
        assert_eq!(Vec::<i32>::from_reflect(&list as &dyn Reflect).unwrap(), list);

        let mut map = HashMap::new();
        map.insert("x".to_string(), 10);
        map.insert("y".to_string(), 20);
        assert_eq!(
            HashMap::<String, i32>::from_reflect(&map as &dyn Reflect).unwrap(),
            map
        );

        let v = Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        assert_eq!(Vec3::from_reflect(&v as &dyn Reflect).unwrap(), v);
        assert_eq!(Vec3::from_reflect(&*v.reflect_clone()).unwrap(), v);
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
        shape.apply(&patch).expect("same-variant enum patch applies");
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
        assert_eq!(dynamic.field_at(0).unwrap().downcast_ref::<f32>(), Some(&3.5));

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

        let cell = reflect_path_mut(&mut world, &ParsedPath::parse(".grid[1][0]").unwrap()).unwrap();
        *cell.downcast_mut::<i32>().unwrap() = 400;
        assert_eq!(world.grid[1][0], 400);

        let speed = reflect_path_mut(&mut world, &ParsedPath::parse(".player.speed").unwrap()).unwrap();
        *speed.downcast_mut::<f32>().unwrap() = 9.0;
        assert_eq!(world.player.speed, 9.0);

        let entry = reflect_path_mut(&mut world, &ParsedPath::parse(r#".lookup["a"]"#).unwrap()).unwrap();
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
