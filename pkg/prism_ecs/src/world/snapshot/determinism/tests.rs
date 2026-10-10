//! Dual-run determinism auditing tests (design §14 / §24.4).
//!
//! Divergence localization is validated against the invariant it promises:
//! `locate_divergence(a, b).is_none() == a.structurally_eq(b)`, and each
//! structural / value coordinate is pinpointed exactly. Entity allocation and
//! spawn tick stamping are deterministic, so two worlds built with the same
//! operation sequence agree bit-for-bit, which is what lets a desync audit
//! compare two independent runs captured at the same tick.

use crate::component::{Component, StorageType};
use crate::world::World;

use super::{FrameHashLog, SnapshotDivergence, TickDivergence};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Pos {
    x: i32,
    y: i32,
}
impl Component for Pos {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Vel(i32);
impl Component for Vel {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Charge(u32);
impl Component for Charge {
    const STORAGE: StorageType = StorageType::SparseSet;
}

/// A fresh world with snapshot + hash glue registered in a fixed order, so two
/// such worlds assign identical [`ComponentId`](crate::component::ComponentId)s.
fn world() -> World {
    let mut w = World::new();
    w.register_snapshot_component_hashable::<Pos>();
    w.register_snapshot_component_hashable::<Vel>();
    w.register_snapshot_component_hashable::<Charge>();
    w
}

#[test]
fn identical_runs_have_no_divergence() {
    let mut a = world();
    a.spawn((Pos { x: 1, y: 2 }, Vel(3)));
    let mut b = world();
    b.spawn((Pos { x: 1, y: 2 }, Vel(3)));

    let sa = a.snapshot();
    let sb = b.snapshot();
    assert!(sa.structurally_eq(&sb));
    assert!(sa.locate_divergence(&sb).is_none());
    // The invariant that ties the two together.
    assert_eq!(sa.locate_divergence(&sb).is_none(), sa.structurally_eq(&sb));
}

#[test]
fn value_divergence_pinpoints_entity_and_component() {
    let mut a = world();
    let ea = a.spawn((Pos { x: 1, y: 2 }, Vel(3)));
    let mut b = world();
    let eb = b.spawn((Pos { x: 1, y: 999 }, Vel(3)));
    assert_eq!(ea, eb, "first spawn must be deterministic across worlds");

    let sa = a.snapshot();
    let sb = b.snapshot();
    assert!(!sa.structurally_eq(&sb));

    let pos_id = a.components().id_of::<Pos>().unwrap();
    match sa.locate_divergence(&sb) {
        Some(SnapshotDivergence::Value { component, entity }) => {
            assert_eq!(component, pos_id);
            assert_eq!(entity, ea);
        }
        other => panic!("expected Value divergence, got {other:?}"),
    }
}

#[test]
fn change_tick_divergence_is_reported_first() {
    let mut w = world();
    w.spawn(Pos { x: 1, y: 2 });
    let before = w.snapshot();
    w.increment_change_tick();
    let after = w.snapshot();

    match before.locate_divergence(&after) {
        Some(SnapshotDivergence::ChangeTick { left, right }) => {
            assert_eq!(left, 1);
            assert_eq!(right, 2);
        }
        other => panic!("expected ChangeTick divergence, got {other:?}"),
    }
}

#[test]
fn baseline_tick_divergence() {
    let mut a = world();
    a.spawn(Pos { x: 1, y: 2 });
    let mut b = world();
    b.spawn(Pos { x: 1, y: 2 });
    b.set_last_change_tick(crate::change::Tick::new(7));

    match a.snapshot().locate_divergence(&b.snapshot()) {
        Some(SnapshotDivergence::BaselineTick { left, right }) => {
            assert_eq!(left, 0);
            assert_eq!(right, 7);
        }
        other => panic!("expected BaselineTick divergence, got {other:?}"),
    }
}

#[test]
fn live_entity_count_divergence() {
    let mut a = world();
    a.spawn(Pos { x: 1, y: 2 });
    let mut b = world();
    b.spawn(Pos { x: 1, y: 2 });
    b.spawn(Pos { x: 3, y: 4 });

    match a.snapshot().locate_divergence(&b.snapshot()) {
        Some(SnapshotDivergence::LiveEntityCount { left, right }) => {
            assert_eq!(left, 1);
            assert_eq!(right, 2);
        }
        other => panic!("expected LiveEntityCount divergence, got {other:?}"),
    }
}

#[test]
fn entity_list_divergence_on_extra_holder() {
    // Both worlds keep the same live count (2), but B's second entity holds a
    // component (so it enters the captured entity list) while A's does not.
    let mut a = world();
    let a0 = a.spawn(Pos { x: 1, y: 2 });
    a.spawn(()); // component-less: live, but absent from the entity list.
    let mut b = world();
    let b0 = b.spawn(Pos { x: 1, y: 2 });
    let b1 = b.spawn(Pos { x: 3, y: 4 });
    assert_eq!(a0, b0);

    let sa = a.snapshot();
    let sb = b.snapshot();
    assert_eq!(sa.live_entity_count(), sb.live_entity_count());
    match sa.locate_divergence(&sb) {
        Some(SnapshotDivergence::EntityList { index, left, right }) => {
            assert_eq!(index, 1);
            assert_eq!(left, None);
            assert_eq!(right, Some(b1));
        }
        other => panic!("expected EntityList divergence, got {other:?}"),
    }
}

#[test]
fn column_set_divergence_on_extra_component() {
    let mut a = world();
    let a0 = a.spawn(Pos { x: 1, y: 2 });
    let mut b = world();
    let b0 = b.spawn(Pos { x: 1, y: 2 });
    b.insert(b0, Charge(5));
    assert_eq!(a0, b0);

    let charge_id = b.components().id_of::<Charge>().unwrap();
    match a.snapshot().locate_divergence(&b.snapshot()) {
        Some(SnapshotDivergence::ColumnSet { index, left, right }) => {
            assert_eq!(index, 1);
            assert_eq!(left, None);
            assert_eq!(right, Some(charge_id));
        }
        other => panic!("expected ColumnSet divergence, got {other:?}"),
    }
}

#[test]
fn holder_divergence_when_component_sits_on_different_entity() {
    // Same entity list, same Pos values, but Vel lives on a different member.
    let mut a = world();
    let a0 = a.spawn(Pos { x: 1, y: 2 });
    let a1 = a.spawn(Pos { x: 3, y: 4 });
    a.insert(a0, Vel(9));
    let mut b = world();
    let b0 = b.spawn(Pos { x: 1, y: 2 });
    let b1 = b.spawn(Pos { x: 3, y: 4 });
    b.insert(b1, Vel(9));
    assert_eq!(a0, b0);
    assert_eq!(a1, b1);

    let vel_id = a.components().id_of::<Vel>().unwrap();
    match a.snapshot().locate_divergence(&b.snapshot()) {
        Some(SnapshotDivergence::Holder {
            component,
            index,
            left,
            right,
        }) => {
            assert_eq!(component, vel_id);
            assert_eq!(index, 0);
            assert_eq!(left, Some(a0));
            assert_eq!(right, Some(b1));
        }
        other => panic!("expected Holder divergence, got {other:?}"),
    }
}

#[test]
fn frame_hash_log_matches_identical_runs() {
    let mut left = FrameHashLog::new();
    let mut right = FrameHashLog::with_capacity(3);
    for tick in 0..3 {
        left.record(tick, tick * 10);
        right.record(tick, tick * 10);
    }
    assert_eq!(left.len(), 3);
    assert!(!left.is_empty());
    assert_eq!(left.first_divergence(&right), None);
}

#[test]
fn frame_hash_log_finds_first_hash_divergence() {
    let mut left = FrameHashLog::new();
    let mut right = FrameHashLog::new();
    left.record(0, 100);
    right.record(0, 100);
    left.record(1, 200);
    right.record(1, 999); // desync here
    left.record(2, 300);
    right.record(2, 300);

    match left.first_divergence(&right) {
        Some(TickDivergence::Hash { tick, left, right }) => {
            assert_eq!(tick, 1);
            assert_eq!(left, 200);
            assert_eq!(right, 999);
        }
        other => panic!("expected Hash divergence, got {other:?}"),
    }
}

#[test]
fn frame_hash_log_detects_cadence_and_length_drift() {
    let mut left = FrameHashLog::new();
    let mut right = FrameHashLog::new();
    left.record(0, 1);
    right.record(0, 1);
    left.record(1, 2);
    right.record(2, 2); // tick cadence drifted at index 1
    match left.first_divergence(&right) {
        Some(TickDivergence::Tick { index, left, right }) => {
            assert_eq!(index, 1);
            assert_eq!(left, 1);
            assert_eq!(right, 2);
        }
        other => panic!("expected Tick divergence, got {other:?}"),
    }

    let mut short = FrameHashLog::new();
    let mut long = FrameHashLog::new();
    short.record(0, 1);
    long.record(0, 1);
    long.record(1, 2);
    match short.first_divergence(&long) {
        Some(TickDivergence::Length {
            index,
            left_len,
            right_len,
        }) => {
            assert_eq!(index, 1);
            assert_eq!(left_len, 1);
            assert_eq!(right_len, 2);
        }
        other => panic!("expected Length divergence, got {other:?}"),
    }
}

#[test]
fn resource_divergence() {
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Score(u32);
    impl crate::resource::Resource for Score {}

    let mut a = world();
    a.register_snapshot_resource_hashable::<Score>();
    a.spawn(Pos { x: 1, y: 2 });
    a.insert_resource(Score(1));
    let mut b = world();
    b.register_snapshot_resource_hashable::<Score>();
    b.spawn(Pos { x: 1, y: 2 });
    b.insert_resource(Score(2));

    match a.snapshot().locate_divergence(&b.snapshot()) {
        Some(SnapshotDivergence::Resource { index }) => assert_eq!(index, 0),
        other => panic!("expected Resource divergence, got {other:?}"),
    }
}
