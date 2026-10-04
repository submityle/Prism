//! Oracle tests for the §24.5 field-level network delta (`net_delta`).
//!
//! The central oracle is *full-snapshot parity*: for any pair of snapshots, the
//! value a remote replica reaches by decoding and applying the compact dirty
//! delta must equal the value it would reach from a full snapshot (`new`
//! itself). The suite also pins dirty-mask semantics, idempotent re-application,
//! plan filtering, the stateful `ReplicationState` baseline tracker, the
//! `FieldDelta` wire round-trip, and the malformed-frame guards.

use crate::integration::{ReplicationPlan, ReplicationPolicy};
use crate::net_delta::{
    apply_delta, decode_and_apply, decode_delta, encode_delta, DeltaError, DirtyMask,
    ReplicationState,
};
use crate::schema::{AttributeValue, FieldMetadata};
use crate::{Reflect, TypeMetadata, TypeRegistry, Typed};
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

#[derive(Reflect, Debug, PartialEq, Clone)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Reflect, Debug, PartialEq, Clone)]
struct Entity {
    name: String,
    hp: i32,
    position: Vec3,
    tags: Vec<String>,
    alive: bool,
}

fn registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    registry.register::<Vec3>();
    registry.register::<Entity>();
    registry
}

fn sample() -> Entity {
    Entity {
        name: "hero".to_string(),
        hp: 100,
        position: Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        },
        tags: vec!["player".to_string()],
        alive: true,
    }
}

/// The core oracle: apply the dirty delta of `new` relative to `old` onto a
/// clone of `old` and assert it equals the full snapshot `new`.
fn assert_delta_matches_snapshot(registry: &TypeRegistry, old: &Entity, new: &Entity) -> Vec<u8> {
    let mask = DirtyMask::changed(old, new).expect("same-shape structs");
    let bytes = encode_delta(new, &mask).expect("encode delta");

    let mut replica = old.clone();
    decode_and_apply(&mut replica, &bytes, registry).expect("apply delta");
    assert_eq!(&replica, new, "delta path must match full snapshot");

    // Idempotent: replaying the same delta leaves the replica unchanged.
    decode_and_apply(&mut replica, &bytes, registry).expect("re-apply delta");
    assert_eq!(&replica, new, "re-applying a delta is a no-op");

    bytes
}

#[test]
fn dirty_mask_bit_operations() {
    let mut mask = DirtyMask::new();
    assert!(mask.is_empty());
    assert_eq!(mask.count(), 0);

    mask.mark(0);
    mask.mark(65);
    mask.mark(130);
    assert!(mask.is_marked(0));
    assert!(mask.is_marked(65));
    assert!(mask.is_marked(130));
    assert!(!mask.is_marked(1));
    assert_eq!(mask.count(), 3);
    assert_eq!(mask.iter().collect::<Vec<_>>(), vec![0, 65, 130]);

    mask.unmark(65);
    assert!(!mask.is_marked(65));
    assert_eq!(mask.iter().collect::<Vec<_>>(), vec![0, 130]);

    let mut other = DirtyMask::with_field_capacity(8);
    other.mark(1);
    other.mark(130);
    mask.union(&other);
    assert_eq!(mask.iter().collect::<Vec<_>>(), vec![0, 1, 130]);

    mask.clear();
    assert!(mask.is_empty());
    assert_eq!(mask.count(), 0);
}

#[test]
fn changed_marks_exactly_the_moved_fields() {
    let old = sample();
    let mut new = old.clone();
    new.hp = 80; // field index 1
    new.position.y = 9.0; // field index 2 (nested change)

    let mask = DirtyMask::changed(&old, &new).expect("same shape");
    assert_eq!(mask.iter().collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(mask.count(), 2);
}

#[test]
fn changed_on_equal_values_is_empty() {
    let old = sample();
    let new = old.clone();
    let mask = DirtyMask::changed(&old, &new).expect("same shape");
    assert!(mask.is_empty());
}

#[test]
fn changed_rejects_non_structs() {
    let err = DirtyMask::changed(&1_i32, &2_i32).expect_err("scalars are not structs");
    assert!(matches!(err, DeltaError::NotAStruct));
}

#[test]
fn full_snapshot_parity_for_leaf_string_nested_and_bool_changes() {
    let registry = registry();
    let old = sample();
    let mut new = old.clone();
    new.name = "villain".to_string();
    new.hp = 42;
    new.position = Vec3 {
        x: -1.0,
        y: -2.0,
        z: -3.0,
    };
    new.alive = false;

    assert_delta_matches_snapshot(&registry, &old, &new);
}

#[test]
fn full_snapshot_parity_for_grow_only_vec() {
    let registry = registry();
    let old = sample();
    let mut new = old.clone();
    new.tags = vec![
        "player".to_string(),
        "boss".to_string(),
        "elite".to_string(),
    ];

    let bytes = assert_delta_matches_snapshot(&registry, &old, &new);

    // Only the `tags` field (index 3) travels in the delta.
    let delta = decode_delta(&bytes).expect("decode");
    assert_eq!(delta.len(), 1);
    assert_eq!(delta.entries()[0].0, 3);
}

#[test]
fn empty_delta_when_nothing_changed() {
    let registry = registry();
    let old = sample();
    let new = old.clone();

    let mask = DirtyMask::changed(&old, &new).expect("same shape");
    let bytes = encode_delta(&new, &mask).expect("encode");
    assert_eq!(bytes, vec![0]); // a single count=0 varint
    let delta = decode_delta(&bytes).expect("decode");
    assert!(delta.is_empty());

    let mut replica = old.clone();
    decode_and_apply(&mut replica, &bytes, &registry).expect("apply empty");
    assert_eq!(replica, new);
}

#[test]
fn explicit_mark_encodes_requested_fields() {
    let registry = registry();
    let old = sample();
    let mut new = old.clone();
    new.hp = 7;
    new.alive = false;

    // Mark only `hp` (index 1), ignoring the `alive` change on purpose.
    let mut mask = DirtyMask::new();
    mask.mark(1);
    let bytes = encode_delta(&new, &mask).expect("encode");

    let mut replica = old.clone();
    decode_and_apply(&mut replica, &bytes, &registry).expect("apply");
    assert_eq!(replica.hp, 7, "marked field replicated");
    assert!(replica.alive, "unmarked field stayed at the old value");
}

#[test]
fn field_delta_to_bytes_round_trips() {
    let old = sample();
    let mut new = old.clone();
    new.name = "renamed".to_string();
    new.hp = 1;

    let mask = DirtyMask::changed(&old, &new).expect("same shape");
    let bytes = encode_delta(&new, &mask).expect("encode");

    let delta = decode_delta(&bytes).expect("decode");
    assert_eq!(delta.to_bytes(), bytes, "FieldDelta re-serialises identically");

    let delta2 = decode_delta(&delta.to_bytes()).expect("decode round-trip");
    assert_eq!(delta, delta2);
}

#[test]
fn decode_rejects_truncated_and_trailing_frames() {
    let old = sample();
    let mut new = old.clone();
    new.hp = 3;
    let mask = DirtyMask::changed(&old, &new).expect("same shape");
    let bytes = encode_delta(&new, &mask).expect("encode");

    // Truncating the final value byte must be rejected, not silently accepted.
    let truncated = &bytes[..bytes.len() - 1];
    assert!(matches!(
        decode_delta(truncated),
        Err(DeltaError::Malformed)
    ));

    // Trailing garbage after a well-formed frame is rejected too.
    let mut trailing = bytes.clone();
    trailing.push(0xff);
    assert!(matches!(
        decode_delta(&trailing),
        Err(DeltaError::Malformed)
    ));
}

#[test]
fn apply_delta_rejects_out_of_range_field_index() {
    let registry = registry();
    // Encode a legitimate single-field delta, then rewrite its wire frame to
    // name field index 99 (Entity has 5 fields) with the same value bytes.
    let mut mark = DirtyMask::new();
    mark.mark(1);
    let good = encode_delta(&sample(), &mark).expect("encode");
    let decoded = decode_delta(&good).expect("decode");
    let value = decoded.entries()[0].1.clone();

    let mut frame = Vec::new();
    frame.push(1); // count = 1
    frame.push(99); // field index = 99
    frame.push(value.len() as u8);
    frame.extend_from_slice(&value);

    let delta = decode_delta(&frame).expect("decode hand-built frame");
    let mut target = sample();
    let err = apply_delta(&mut target, &delta, &registry).expect_err("index out of range");
    assert!(matches!(
        err,
        DeltaError::FieldIndexOutOfRange {
            index: 99,
            field_count: 5
        }
    ));
}

#[test]
fn replication_state_tracks_baseline_across_ticks() {
    let registry = registry();
    let mut state = ReplicationState::new(&sample());

    // Tick 1: change hp only.
    let mut tick1 = sample();
    tick1.hp = 90;
    let bytes1 = state.encode(&tick1).expect("encode tick 1");
    let mut replica = sample();
    decode_and_apply(&mut replica, &bytes1, &registry).expect("apply tick 1");
    assert_eq!(replica, tick1);
    state.commit(&tick1);

    // Tick 2: a delta against the committed baseline carries only the new move.
    let mut tick2 = tick1.clone();
    tick2.position.z = 42.0;
    let bytes2 = state.encode(&tick2).expect("encode tick 2");
    let delta2 = decode_delta(&bytes2).expect("decode tick 2");
    assert_eq!(delta2.len(), 1, "only position moved since the baseline");
    decode_and_apply(&mut replica, &bytes2, &registry).expect("apply tick 2");
    assert_eq!(replica, tick2);
}

#[derive(Reflect, Debug, PartialEq, Clone)]
struct PlannedState {
    position: i32,
    velocity: i32,
    debug_counter: i32,
}

fn planned_registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    registry.register::<PlannedState>();
    registry.register_type_data::<PlannedState, _>(
        TypeMetadata::new()
            .with_field(
                FieldMetadata::new("debug_counter")
                    .with_custom("no_replicate", AttributeValue::Bool(true)),
            ),
    );
    registry
}

#[test]
fn changed_in_plan_filters_non_replicated_fields() {
    let registry = planned_registry();
    let info = <PlannedState as Typed>::type_info();
    let meta = registry
        .get_with_name(core::any::type_name::<PlannedState>())
        .and_then(|reg| reg.data::<TypeMetadata>());
    let plan = ReplicationPlan::from_type(info, meta, ReplicationPolicy::OptOut);
    assert_eq!(plan.fields(), ["position", "velocity"]);

    let old = PlannedState {
        position: 0,
        velocity: 0,
        debug_counter: 0,
    };
    let new = PlannedState {
        position: 5,
        velocity: 7,
        debug_counter: 999, // moved but excluded by the plan
    };

    let mask = DirtyMask::changed_in_plan(&old, &new, &plan).expect("same shape");
    assert_eq!(mask.iter().collect::<Vec<_>>(), vec![0, 1]);

    let bytes = encode_delta(&new, &mask).expect("encode");
    let mut replica = old.clone();
    decode_and_apply(&mut replica, &bytes, &registry).expect("apply");
    assert_eq!(replica.position, 5);
    assert_eq!(replica.velocity, 7);
    assert_eq!(replica.debug_counter, 0, "excluded field never replicated");
}

#[test]
fn replication_state_with_plan_omits_excluded_fields() {
    let registry = planned_registry();
    let info = <PlannedState as Typed>::type_info();
    let meta = registry
        .get_with_name(core::any::type_name::<PlannedState>())
        .and_then(|reg| reg.data::<TypeMetadata>());
    let plan = ReplicationPlan::from_type(info, meta, ReplicationPolicy::OptOut);

    let baseline = PlannedState {
        position: 0,
        velocity: 0,
        debug_counter: 0,
    };
    let state = ReplicationState::with_plan(&baseline, plan);

    let mut current = baseline.clone();
    current.debug_counter = 123; // only the excluded field moved
    let bytes = state.encode(&current).expect("encode");
    assert_eq!(bytes, vec![0], "nothing replicable changed");
}
