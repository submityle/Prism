//! Oracle tests for the §24.6 script/editor property bridge (`property_bridge`).
//!
//! The central oracle is *get/set round-trip identity*: writing a value through
//! the bridge and reading it back returns the value just written (clamped into
//! range when the field is range-constrained), both for typed `set` and for the
//! dynamic `AttributeValue` scalar exchange. The suite also pins read-only
//! rejection, range clamping, nested-path access, lossless scalar coercion
//! (float-into-int and out-of-range-int are rejected, never truncated), and the
//! `properties` enumeration's metadata (hidden omission, kind, range, docs,
//! category, default).

use crate::property_bridge::{properties, PropertyBridge, PropertyError, PropertyKind};
use crate::schema::{AttributeValue, FieldMetadata};
use crate::{Reflect, TypeMetadata, TypeRegistry};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Reflect, Debug, PartialEq, Clone)]
struct Volume {
    master: f32,
    music: f32,
}

#[derive(Reflect, Debug, PartialEq, Clone)]
struct Settings {
    name: String,
    difficulty: i32,
    volume: Volume,
    fullscreen: bool,
    build_id: u64,
    internal_seed: i64,
}

fn volume_metadata() -> TypeMetadata {
    TypeMetadata::new()
        .with_field(FieldMetadata::new("master").with_range(0.0, 1.0))
        .with_field(FieldMetadata::new("music").with_range(0.0, 1.0))
}

fn settings_metadata() -> TypeMetadata {
    TypeMetadata::new()
        .with_field(
            FieldMetadata::new("difficulty")
                .with_docs("Game difficulty tier")
                .with_category("Gameplay")
                .with_range(1.0, 5.0)
                .with_default(AttributeValue::Int(3)),
        )
        .with_field(FieldMetadata::new("build_id").readonly(true))
        .with_field(FieldMetadata::new("internal_seed").hidden(true))
}

fn registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    registry.register::<Volume>();
    registry.register::<Settings>();
    registry.register_type_data::<Volume, _>(volume_metadata());
    registry.register_type_data::<Settings, _>(settings_metadata());
    registry
}

fn sample() -> Settings {
    Settings {
        name: "default".to_string(),
        difficulty: 3,
        volume: Volume {
            master: 0.5,
            music: 0.4,
        },
        fullscreen: false,
        build_id: 42,
        internal_seed: 7,
    }
}

#[test]
fn typed_get_set_round_trip() {
    let registry = registry();
    let bridge = PropertyBridge::new(&registry);
    let mut settings = sample();

    bridge.set(&mut settings, "difficulty", &4_i32).expect("set");
    let got = bridge.get_as::<i32>(&settings, "difficulty").expect("get");
    assert_eq!(*got, 4);

    bridge
        .set(&mut settings, "name", &"renamed".to_string())
        .expect("set string");
    assert_eq!(settings.name, "renamed");
}

#[test]
fn get_as_reports_type_mismatch() {
    let registry = registry();
    let bridge = PropertyBridge::new(&registry);
    let settings = sample();

    let err = bridge
        .get_as::<String>(&settings, "difficulty")
        .expect_err("difficulty is not a String");
    match err {
        PropertyError::TypeMismatch { path, .. } => assert_eq!(path, "difficulty"),
        other => panic!("expected TypeMismatch, got {other:?}"),
    }
}

#[test]
fn unknown_path_and_bad_syntax_are_distinct_errors() {
    let registry = registry();
    let bridge = PropertyBridge::new(&registry);
    let settings = sample();

    let missing = bridge.get(&settings, "no_such_field");
    assert!(matches!(missing, Err(PropertyError::NoSuchProperty(_))));

    let bad = bridge.get(&settings, "volume..master");
    assert!(matches!(bad, Err(PropertyError::Path(_))));
}

#[test]
fn readonly_field_rejects_writes() {
    let registry = registry();
    let bridge = PropertyBridge::new(&registry);
    let mut settings = sample();

    let err = bridge
        .set(&mut settings, "build_id", &99_u64)
        .expect_err("build_id is read-only");
    assert!(matches!(err, PropertyError::ReadOnly(_)));
    assert_eq!(settings.build_id, 42, "value untouched after rejection");

    let err = bridge
        .set_scalar(&mut settings, "build_id", &AttributeValue::Int(99))
        .expect_err("scalar write to read-only field");
    assert!(matches!(err, PropertyError::ReadOnly(_)));
    assert_eq!(settings.build_id, 42);
}

#[test]
fn range_constrained_write_is_clamped_into_bounds() {
    let registry = registry();
    let bridge = PropertyBridge::new(&registry);
    let mut settings = sample();

    // Above the max clamps down to 5.
    bridge
        .set_scalar(&mut settings, "difficulty", &AttributeValue::Int(99))
        .expect("set high");
    assert_eq!(settings.difficulty, 5);

    // Below the min clamps up to 1.
    bridge
        .set_scalar(&mut settings, "difficulty", &AttributeValue::Int(-10))
        .expect("set low");
    assert_eq!(settings.difficulty, 1);

    // In range is left exactly as written.
    bridge
        .set_scalar(&mut settings, "difficulty", &AttributeValue::Int(2))
        .expect("set mid");
    assert_eq!(settings.difficulty, 2);
}

#[test]
fn nested_path_read_write_and_clamp() {
    let registry = registry();
    let bridge = PropertyBridge::new(&registry);
    let mut settings = sample();

    // Write a nested leaf by path.
    bridge
        .set_scalar(&mut settings, "volume.master", &AttributeValue::Float(0.25))
        .expect("set nested");
    let got = bridge
        .read_scalar(&settings, "volume.master")
        .expect("read nested");
    assert_eq!(got, AttributeValue::Float(0.25));

    // Out-of-range nested write clamps into [0, 1].
    bridge
        .set_scalar(&mut settings, "volume.master", &AttributeValue::Float(3.0))
        .expect("set over max");
    assert_eq!(settings.volume.master, 1.0);
}

#[test]
fn scalar_round_trip_is_a_no_op() {
    let registry = registry();
    let bridge = PropertyBridge::new(&registry);
    let mut settings = sample();

    for path in ["name", "difficulty", "fullscreen", "volume.music"] {
        let before = bridge.read_scalar(&settings, path).expect("read");
        bridge.set_scalar(&mut settings, path, &before).expect("write back");
        let after = bridge.read_scalar(&settings, path).expect("read back");
        assert_eq!(before, after, "round-trip of `{path}` must be a no-op");
    }
}

#[test]
fn read_scalar_classifies_each_leaf_kind() {
    let registry = registry();
    let bridge = PropertyBridge::new(&registry);
    let settings = sample();

    assert_eq!(
        bridge.read_scalar(&settings, "name").unwrap(),
        AttributeValue::Text("default".to_string())
    );
    assert_eq!(
        bridge.read_scalar(&settings, "difficulty").unwrap(),
        AttributeValue::Int(3)
    );
    assert_eq!(
        bridge.read_scalar(&settings, "fullscreen").unwrap(),
        AttributeValue::Bool(false)
    );
    assert_eq!(
        bridge.read_scalar(&settings, "build_id").unwrap(),
        AttributeValue::Int(42)
    );
    assert_eq!(
        bridge.read_scalar(&settings, "volume.master").unwrap(),
        AttributeValue::Float(0.5)
    );
}

#[test]
fn read_scalar_on_a_struct_is_not_scalar() {
    let registry = registry();
    let bridge = PropertyBridge::new(&registry);
    let settings = sample();

    let err = bridge
        .read_scalar(&settings, "volume")
        .expect_err("a struct is not a scalar leaf");
    assert!(matches!(err, PropertyError::NotScalar { .. }));
}

#[test]
fn float_into_integer_leaf_is_rejected() {
    let registry = registry();
    let bridge = PropertyBridge::new(&registry);
    let mut settings = sample();

    let err = bridge
        .set_scalar(&mut settings, "difficulty", &AttributeValue::Float(3.0))
        .expect_err("float must not silently truncate into an integer leaf");
    assert!(matches!(err, PropertyError::TypeMismatch { .. }));
    assert_eq!(settings.difficulty, 3, "value untouched after rejection");
}

#[test]
fn out_of_range_integer_is_rejected() {
    let registry = registry();
    let bridge = PropertyBridge::new(&registry);
    let mut settings = sample();

    // 5_000_000_000 does not fit an i32; reject rather than wrap/truncate.
    let err = bridge
        .set_scalar(
            &mut settings,
            "difficulty",
            &AttributeValue::Int(5_000_000_000),
        )
        .expect_err("out-of-range integer");
    assert!(matches!(err, PropertyError::TypeMismatch { .. }));
    assert_eq!(settings.difficulty, 3);
}

#[test]
fn properties_enumerates_metadata_and_omits_hidden() {
    let registry = registry();
    let settings = sample();
    let descriptors = properties(&settings, &registry);

    let names: Vec<&str> = descriptors.iter().map(|d| d.name.as_str()).collect();
    assert!(
        !names.contains(&"internal_seed"),
        "hidden field must be omitted"
    );
    assert_eq!(names, ["name", "difficulty", "volume", "fullscreen", "build_id"]);

    let difficulty = descriptors
        .iter()
        .find(|d| d.name == "difficulty")
        .expect("difficulty present");
    assert_eq!(difficulty.kind, PropertyKind::Integer);
    assert_eq!(difficulty.range, Some((1.0, 5.0)));
    assert_eq!(difficulty.docs.as_deref(), Some("Game difficulty tier"));
    assert_eq!(difficulty.category.as_deref(), Some("Gameplay"));
    assert_eq!(difficulty.default, Some(AttributeValue::Int(3)));
    assert!(!difficulty.readonly);
    assert_eq!(difficulty.path, "difficulty");

    let build_id = descriptors
        .iter()
        .find(|d| d.name == "build_id")
        .expect("build_id present");
    assert!(build_id.readonly, "read-only metadata surfaced");
    assert_eq!(build_id.kind, PropertyKind::Integer);

    let volume = descriptors
        .iter()
        .find(|d| d.name == "volume")
        .expect("volume present");
    assert_eq!(volume.kind, PropertyKind::Struct);

    let name = descriptors.iter().find(|d| d.name == "name").unwrap();
    assert_eq!(name.kind, PropertyKind::Text);
    let fullscreen = descriptors.iter().find(|d| d.name == "fullscreen").unwrap();
    assert_eq!(fullscreen.kind, PropertyKind::Bool);

    assert!(PropertyKind::Integer.is_scalar());
    assert!(!PropertyKind::Struct.is_scalar());
}

#[test]
fn properties_of_non_struct_is_empty() {
    let registry = registry();
    assert!(properties(&5_i32, &registry).is_empty());
}

#[test]
fn bridge_properties_matches_free_function() {
    let registry = registry();
    let bridge = PropertyBridge::new(&registry);
    let settings = sample();
    assert_eq!(bridge.properties(&settings), properties(&settings, &registry));
}
