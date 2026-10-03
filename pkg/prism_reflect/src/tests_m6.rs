//! M6 integration-layer tests: scene snapshots (binary + text round-trips and
//! error paths), the editor inspector model (nested traversal, metadata hints,
//! hidden-field omission), field-level replication (opt-in/opt-out planning,
//! minimal delta, apply, and shape guards), the script bridge (path get/set and
//! call-by-name plus their typed errors), and the `bevy_reflect` compat prelude.

use crate::integration::inspector::{InspectorKind, inspect};
use crate::integration::replication::{
    ReplicationPlan, ReplicationPolicy, apply_replicated, replicated_diff,
};
use crate::integration::scene::{DynamicScene, SceneError};
use crate::integration::script::{ScriptBridge, ScriptError};
use crate::schema::{AttributeValue, FieldMetadata};
use crate::{
    ArgList, FromReflect, FunctionError, FunctionRegistry, Patch, Reflect, TypeMetadata,
    TypeRegistry, Typed,
};
use alloc::vec::Vec;
use core::any::type_name;

#[derive(Reflect, Debug, PartialEq, Clone)]
struct Transform {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Reflect, Debug, PartialEq, Clone)]
struct Health {
    current: i32,
    max: i32,
}

#[derive(Reflect, Debug, PartialEq, Clone)]
struct Actor {
    name: String,
    transform: Transform,
    health: Health,
    secret: i32,
}

fn registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    registry.register::<Transform>();
    registry.register::<Health>();
    registry.register::<Actor>();
    registry
}

fn sample_actor() -> Actor {
    Actor {
        name: "hero".to_string(),
        transform: Transform { x: 1.0, y: 2.0, z: 3.0 },
        health: Health { current: 70, max: 100 },
        secret: 42,
    }
}

// ----------------------------------------------------------------------------
// Scene snapshots
// ----------------------------------------------------------------------------

#[test]
fn scene_binary_round_trip_rebuilds_every_entry() {
    let registry = registry();
    let actor = sample_actor();
    let transform = Transform { x: -4.0, y: 5.5, z: 6.25 };

    let mut scene = DynamicScene::new();
    scene.push_value(actor.clone());
    scene.push_value(transform.clone());
    assert_eq!(scene.len(), 2);
    assert!(!scene.is_empty());

    let bytes = scene.to_binary().expect("serialize scene to binary");
    let restored = DynamicScene::from_binary(&bytes, &registry).expect("deserialize scene");
    assert_eq!(restored.len(), 2);

    let entries = restored.entries();
    assert_eq!(entries[0].type_name(), type_name::<Actor>());
    assert_eq!(entries[1].type_name(), type_name::<Transform>());

    let rebuilt_actor = Actor::from_reflect(entries[0].value()).expect("rebuild actor");
    assert_eq!(rebuilt_actor, actor);
    let rebuilt_transform = Transform::from_reflect(entries[1].value()).expect("rebuild transform");
    assert_eq!(rebuilt_transform, transform);
}

#[test]
fn scene_text_round_trip_rebuilds_every_entry() {
    let registry = registry();
    let actor = sample_actor();
    let health = Health { current: 1, max: 9 };

    let mut scene = DynamicScene::new();
    scene.push_value(actor.clone());
    scene.push_value(health.clone());

    let text = scene.to_text().expect("serialize scene to text");
    assert!(text.starts_with("PSCN-RON v1\n"));
    let restored = DynamicScene::from_text(&text, &registry).expect("deserialize scene text");
    assert_eq!(restored.len(), 2);

    let entries = restored.entries();
    let rebuilt_actor = Actor::from_reflect(entries[0].value()).expect("rebuild actor");
    assert_eq!(rebuilt_actor, actor);
    let rebuilt_health = Health::from_reflect(entries[1].value()).expect("rebuild health");
    assert_eq!(rebuilt_health, health);
}

#[test]
fn scene_stable_id_matches_type_name() {
    let mut scene = DynamicScene::new();
    scene.push_value(sample_actor());
    let entry = &scene.entries()[0];
    let expected = crate::ser::StableTypeId::of_path(type_name::<Actor>());
    assert_eq!(entry.stable_id(), expected);
}

#[test]
fn scene_binary_rejects_unregistered_type() {
    // Serialize against a full registry, then decode against an empty one.
    let full = registry();
    let mut scene = DynamicScene::new();
    scene.push_value(sample_actor());
    let bytes = scene.to_binary().expect("serialize");

    let empty = TypeRegistry::new();
    let err = DynamicScene::from_binary(&bytes, &empty).expect_err("unknown type must fail");
    match err {
        SceneError::UnknownType(name) => assert_eq!(name, type_name::<Actor>()),
        other => panic!("expected UnknownType, got {other:?}"),
    }
    // The error renders a helpful message.
    let _ = full; // keep registry construction exercised
}

#[test]
fn scene_binary_rejects_bad_magic_and_truncation() {
    let registry = registry();
    let mut scene = DynamicScene::new();
    scene.push_value(Transform { x: 0.0, y: 0.0, z: 0.0 });
    let bytes = scene.to_binary().expect("serialize");

    // Flip the magic.
    let mut corrupt = bytes.clone();
    corrupt[0] ^= 0xFF;
    assert!(matches!(
        DynamicScene::from_binary(&corrupt, &registry),
        Err(SceneError::BadHeader)
    ));

    // Chop the payload mid-stream.
    let truncated = &bytes[..bytes.len() - 2];
    assert!(matches!(
        DynamicScene::from_binary(truncated, &registry),
        Err(SceneError::Truncated)
    ));
}

#[test]
fn scene_empty_round_trips_both_formats() {
    let registry = registry();
    let scene = DynamicScene::new();
    assert!(scene.is_empty());

    let bytes = scene.to_binary().expect("serialize empty");
    assert!(DynamicScene::from_binary(&bytes, &registry)
        .expect("deserialize empty")
        .is_empty());

    let text = scene.to_text().expect("serialize empty text");
    assert!(DynamicScene::from_text(&text, &registry)
        .expect("deserialize empty text")
        .is_empty());
}

// ----------------------------------------------------------------------------
// Inspector model
// ----------------------------------------------------------------------------

fn actor_metadata() -> TypeMetadata {
    TypeMetadata::new()
        .with_docs("A controllable actor.")
        .with_field(
            FieldMetadata::new("name")
                .with_docs("Display name.")
                .with_category("Identity"),
        )
        .with_field(
            FieldMetadata::new("health")
                .with_category("Combat")
                .readonly(true),
        )
        .with_field(FieldMetadata::new("secret").hidden(true))
}

#[test]
fn inspector_builds_nested_tree_with_hints_and_hidden_fields() {
    let mut registry = registry();
    registry.register_type_data::<Actor, _>(actor_metadata());

    let actor = sample_actor();
    let node = inspect(&actor, &registry);

    assert_eq!(node.label, "root");
    assert_eq!(node.type_name, type_name::<Actor>());
    assert_eq!(node.kind, InspectorKind::Struct);

    // `secret` is hidden → only name/transform/health survive.
    assert_eq!(node.children.len(), 3);
    let labels: Vec<&str> = node.children.iter().map(|c| c.label.as_str()).collect();
    assert_eq!(labels, ["name", "transform", "health"]);

    let name = &node.children[0];
    assert_eq!(name.kind, InspectorKind::Value);
    assert_eq!(name.value.as_deref(), Some("\"hero\""));
    assert_eq!(name.hints.docs.as_deref(), Some("Display name."));
    assert_eq!(name.hints.category.as_deref(), Some("Identity"));
    assert!(!name.hints.readonly);

    let transform = &node.children[1];
    assert_eq!(transform.kind, InspectorKind::Struct);
    assert_eq!(transform.children.len(), 3);
    assert_eq!(transform.children[0].label, "x");
    assert_eq!(transform.children[0].value.as_deref(), Some("1.0"));

    let health = &node.children[2];
    assert_eq!(health.kind, InspectorKind::Struct);
    assert!(health.hints.readonly);
    assert_eq!(health.hints.category.as_deref(), Some("Combat"));
    assert_eq!(health.children[0].label, "current");
    assert_eq!(health.children[0].value.as_deref(), Some("70"));
}

#[test]
fn inspector_without_metadata_shows_all_fields_unhinted() {
    let registry = registry();
    let actor = sample_actor();
    let node = inspect(&actor, &registry);
    // No metadata registered → nothing hidden, no hints.
    assert_eq!(node.children.len(), 4);
    assert!(node.children.iter().all(|c| c.hints.docs.is_none()));
}

// ----------------------------------------------------------------------------
// Field-level replication
// ----------------------------------------------------------------------------

#[derive(Reflect, Debug, PartialEq, Clone)]
struct NetState {
    position: i32,
    velocity: i32,
    debug_counter: i32,
}

fn opt_in_metadata() -> TypeMetadata {
    TypeMetadata::new()
        .with_field(
            FieldMetadata::new("position").with_custom("replicate", AttributeValue::Bool(true)),
        )
        .with_field(
            FieldMetadata::new("velocity").with_custom("replicate", AttributeValue::Bool(true)),
        )
    // `debug_counter` intentionally unflagged.
}

fn opt_out_metadata() -> TypeMetadata {
    TypeMetadata::new().with_field(
        FieldMetadata::new("debug_counter").with_custom("no_replicate", AttributeValue::Bool(true)),
    )
}

#[test]
fn replication_opt_in_plans_only_flagged_fields() {
    let info = <NetState as Typed>::type_info();
    let meta = opt_in_metadata();
    let plan = ReplicationPlan::from_type(info, Some(&meta), ReplicationPolicy::OptIn);
    assert_eq!(plan.fields(), ["position", "velocity"]);
}

#[test]
fn replication_opt_out_plans_all_but_excluded() {
    let info = <NetState as Typed>::type_info();
    let meta = opt_out_metadata();
    let plan = ReplicationPlan::from_type(info, Some(&meta), ReplicationPolicy::OptOut);
    assert_eq!(plan.fields(), ["position", "velocity"]);
}

#[test]
fn replicated_diff_carries_only_changed_selected_fields_and_applies() {
    let info = <NetState as Typed>::type_info();
    let meta = opt_in_metadata();
    let plan = ReplicationPlan::from_type(info, Some(&meta), ReplicationPolicy::OptIn);

    let old = NetState { position: 1, velocity: 2, debug_counter: 100 };
    // Change position (replicated) and debug_counter (not replicated).
    let new = NetState { position: 9, velocity: 2, debug_counter: 999 };

    let patch = replicated_diff(&old, &new, &plan).expect("diff succeeds");
    match &patch {
        Patch::Struct(fields) => {
            // velocity unchanged → dropped; debug_counter not selected → dropped.
            assert_eq!(fields.len(), 1);
            assert_eq!(fields[0].0, "position");
        }
        other => panic!("expected a struct patch, got {other:?}"),
    }

    let mut target = old.clone();
    apply_replicated(&mut target, &patch).expect("apply replicated patch");
    // Only the replicated change landed; debug_counter stayed at the old value.
    assert_eq!(target, NetState { position: 9, velocity: 2, debug_counter: 100 });
}

#[test]
fn replicated_diff_reports_unchanged_when_nothing_selected_moved() {
    let info = <NetState as Typed>::type_info();
    let meta = opt_in_metadata();
    let plan = ReplicationPlan::from_type(info, Some(&meta), ReplicationPolicy::OptIn);

    let old = NetState { position: 1, velocity: 2, debug_counter: 1 };
    let new = NetState { position: 1, velocity: 2, debug_counter: 777 };
    let patch = replicated_diff(&old, &new, &plan).expect("diff succeeds");
    assert!(patch.is_unchanged());
}

#[test]
fn replicated_diff_rejects_non_structs() {
    let plan = ReplicationPlan::new(["position"]);
    let err = replicated_diff(&1_i32, &2_i32, &plan).expect_err("scalars are not structs");
    assert_eq!(err, crate::integration::replication::ReplicationError::NotAStruct);
}

// ----------------------------------------------------------------------------
// Script bridge
// ----------------------------------------------------------------------------

fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[test]
fn script_bridge_reads_nested_path() {
    let functions = FunctionRegistry::new();
    let bridge = ScriptBridge::new(&functions);
    let actor = sample_actor();

    let x = bridge.get(&actor, "transform.x").expect("resolve path");
    assert_eq!(x.downcast_ref::<f32>(), Some(&1.0));

    let name = bridge.get(&actor, "name").expect("resolve name");
    assert_eq!(name.downcast_ref::<String>().map(String::as_str), Some("hero"));
}

#[test]
fn script_bridge_writes_nested_path() {
    let functions = FunctionRegistry::new();
    let bridge = ScriptBridge::new(&functions);
    let mut actor = sample_actor();

    bridge
        .set(&mut actor, "health.current", &55_i32)
        .expect("assign through path");
    assert_eq!(actor.health.current, 55);

    // Confirm the mutable accessor sees the same slot.
    let slot = bridge.get_mut(&mut actor, "transform.z").expect("mut path");
    *slot.downcast_mut::<f32>().unwrap() = -1.0;
    assert_eq!(actor.transform.z, -1.0);
}

#[test]
fn script_bridge_calls_registered_function() {
    let mut functions = FunctionRegistry::new();
    functions.register("add", add);
    let bridge = ScriptBridge::new(&functions);

    assert!(bridge.has_function("add"));
    let result = bridge
        .call("add", ArgList::new().push(2_i32).push(40_i32))
        .expect("call add");
    assert_eq!(result.downcast_ref::<i32>(), Some(&42));
}

#[test]
fn script_bridge_surfaces_no_such_path_and_unknown_function() {
    let functions = FunctionRegistry::new();
    let bridge = ScriptBridge::new(&functions);
    let actor = sample_actor();

    let err = bridge.get(&actor, "transform.w").err().expect("missing field");
    match err {
        ScriptError::NoSuchPath(path) => assert_eq!(path, "transform.w"),
        other => panic!("expected NoSuchPath, got {other:?}"),
    }

    let call_err = bridge
        .call("missing", ArgList::new())
        .err()
        .expect("unknown function");
    assert!(matches!(
        call_err,
        ScriptError::Function(FunctionError::UnknownFunction { .. })
    ));
}

#[test]
fn script_bridge_set_rejects_type_mismatch() {
    let functions = FunctionRegistry::new();
    let bridge = ScriptBridge::new(&functions);
    let mut actor = sample_actor();
    // `transform.x` is an f32; assigning an i32 must fail at apply.
    let err = bridge
        .set(&mut actor, "transform.x", &7_i32)
        .expect_err("type mismatch");
    assert!(matches!(err, ScriptError::Apply(_)));
}

// ----------------------------------------------------------------------------
// bevy_reflect compat prelude
// ----------------------------------------------------------------------------

#[cfg(feature = "compat-bevy")]
#[test]
fn compat_bevy_prelude_aliases_resolve() {
    use crate::compat_bevy::prelude::*;

    // `AppTypeRegistry` is Prism's `TypeRegistry` under a familiar spelling.
    let mut registry: AppTypeRegistry = AppTypeRegistry::new();
    registry.register::<Transform>();
    assert!(registry.get_with_name(type_name::<Transform>()).is_some());

    // The reflection traits are re-exported, so trait methods are in scope.
    let t = Transform { x: 1.0, y: 2.0, z: 3.0 };
    let as_struct: &dyn Struct = &t;
    assert_eq!(as_struct.field_count(), 3);
}
