//! Snapshot / delta / rollback tests (design §14, §16.5, §20 roundtrip 等价).

use alloc::vec::Vec;

use crate::component::{Component, StorageType};
use crate::entity::Entity;
use crate::world::World;

use super::SnapshotRing;

// --- Test components: a table pair, two sparse components, a ZST tag. ---

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
struct Tag;
impl Component for Tag {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Charge(u32);
impl Component for Charge {
    const STORAGE: StorageType = StorageType::SparseSet;
}

/// A world with the full snapshot glue registered for every test component.
fn world_with_glue() -> World {
    let mut w = World::new();
    w.register_snapshot_component_hashable::<Pos>();
    w.register_snapshot_component_hashable::<Vel>();
    w.register_snapshot_component_hashable::<Tag>();
    w.register_snapshot_component_hashable::<Charge>();
    w
}

/// A representative world: a table-only entity, a mixed table+sparse entity, a
/// sparse-only entity, and a component-less entity.
fn populate() -> (World, Entity, Entity, Entity, Entity) {
    let mut w = world_with_glue();
    let e0 = w.spawn((Pos { x: 1, y: 2 }, Vel(3)));
    let e1 = w.spawn((Pos { x: 4, y: 5 }, Tag, Charge(7)));
    let e2 = w.spawn((Tag, Charge(9)));
    let e3 = w.spawn(());
    (w, e0, e1, e2, e3)
}

#[test]
fn snapshot_is_byte_deterministic() {
    let (w, ..) = populate();
    let a = w.snapshot();
    let b = w.snapshot();
    assert!(a.structurally_eq(&b));
    assert_eq!(a.state_hash(), b.state_hash());
    assert_eq!(a.entity_count(), 3); // e0, e1, e2 hold components; e3 does not
    assert_eq!(a.live_entity_count(), 4); // all four are live in the allocator
}

#[test]
fn restore_roundtrips_after_mutation() {
    let (mut w, e0, e1, e2, e3) = populate();
    let snap = w.snapshot();

    // Diverge the world in every structural dimension.
    w.get_mut::<Pos>(e0).unwrap().x = 999;
    w.get_mut::<Charge>(e2).unwrap().0 = 123;
    assert!(w.despawn(e1));
    let _late = w.spawn((Pos { x: 7, y: 7 }, Vel(7)));

    w.restore(&snap);

    // Values, liveness, and generations are all back.
    assert_eq!(w.get::<Pos>(e0), Some(&Pos { x: 1, y: 2 }));
    assert_eq!(w.get::<Vel>(e0), Some(&Vel(3)));
    assert_eq!(w.get::<Pos>(e1), Some(&Pos { x: 4, y: 5 }));
    assert!(w.has::<Tag>(e1));
    assert_eq!(w.get::<Charge>(e1), Some(&Charge(7)));
    assert_eq!(w.get::<Charge>(e2), Some(&Charge(9)));
    assert!(w.contains(e3));

    // A fresh capture is byte+tick identical to the original.
    let after = w.snapshot();
    assert!(snap.structurally_eq(&after));
    assert_eq!(snap.state_hash(), after.state_hash());
}

#[test]
fn restore_can_replay_the_same_snapshot_repeatedly() {
    let (mut w, e0, ..) = populate();
    let snap = w.snapshot();
    for i in 0..4 {
        w.get_mut::<Pos>(e0).unwrap().x = 1000 + i;
        w.restore(&snap);
        assert_eq!(w.get::<Pos>(e0), Some(&Pos { x: 1, y: 2 }));
        let again = w.snapshot();
        assert!(snap.structurally_eq(&again));
    }
}

#[test]
fn state_hash_detects_divergence() {
    let (mut w, e0, ..) = populate();
    let base = w.snapshot();
    w.get_mut::<Pos>(e0).unwrap().y = -1;
    let changed = w.snapshot();
    assert_ne!(base.state_hash(), changed.state_hash());
    assert!(!base.structurally_eq(&changed));
}

#[test]
fn delta_apply_reconstructs_target_exactly() {
    let (mut w, e0, _e1, e2, _e3) = populate();
    let base = w.snapshot();

    // A value change, a sparse change, a despawn, and a fresh spawn.
    w.get_mut::<Pos>(e0).unwrap().x = 50;
    w.get_mut::<Charge>(e2).unwrap().0 = 50;
    let _n = w.spawn((Vel(42),));
    let target = w.snapshot();

    let delta = base.diff(&target);
    let rebuilt = delta.apply(&base);

    assert!(rebuilt.structurally_eq(&target));
    assert_eq!(rebuilt.state_hash(), target.state_hash());

    // The delta owns strictly less than the whole target: unchanged cells (e.g.
    // e0's Vel, e1's Pos) are reused from the base, only changed/added are fresh.
    assert!(delta.reused_cell_count() > 0);
    assert!(delta.changed_cell_count() > 0);
    assert!(delta.changed_cell_count() < delta.changed_cell_count() + delta.reused_cell_count());
}

#[test]
fn delta_handles_added_and_removed_components() {
    let mut w = world_with_glue();
    let e = w.spawn((Pos { x: 1, y: 1 },));
    let base = w.snapshot();

    // Add a brand-new component column (Vel) and remove an existing one is not
    // directly supported by `insert`/remove here; instead cover the "component
    // the base lacks" (added) and "component the target lacks" (removed) paths
    // by diffing in both directions.
    w.insert(e, Vel(5));
    let target = w.snapshot();

    // Forward: Vel is a column the base lacks -> every Vel cell is fresh.
    let fwd = base.diff(&target);
    assert!(fwd.apply(&base).structurally_eq(&target));

    // Backward: Vel is a column the target (base) lacks -> it simply vanishes.
    let back = target.diff(&base);
    assert!(back.apply(&target).structurally_eq(&base));
}

#[test]
fn delta_with_no_changes_is_all_reuse() {
    let (w, ..) = populate();
    let a = w.snapshot();
    let b = w.snapshot();
    let delta = a.diff(&b);
    assert_eq!(delta.changed_cell_count(), 0);
    assert!(delta.reused_cell_count() > 0);
    assert!(delta.apply(&a).structurally_eq(&b));
}

#[test]
fn empty_world_snapshots_and_restores() {
    let mut w = world_with_glue();
    let snap = w.snapshot();
    assert_eq!(snap.entity_count(), 0);
    assert_eq!(snap.column_count(), 0);
    let e = w.spawn((Pos { x: 1, y: 2 },));
    w.restore(&snap);
    // The entity spawned after capture is gone; the slot is free again.
    assert!(!w.contains(e));
    assert_eq!(w.entity_count(), 0);
    assert!(snap.structurally_eq(&w.snapshot()));
}

#[test]
fn component_less_entities_survive_restore() {
    let mut w = world_with_glue();
    let a = w.spawn(());
    let b = w.spawn(());
    let snap = w.snapshot();
    assert_eq!(snap.entity_count(), 0); // no component holders
    assert_eq!(snap.live_entity_count(), 2);
    w.despawn(a);
    w.restore(&snap);
    assert!(w.contains(a));
    assert!(w.contains(b));
}

#[test]
fn try_snapshot_reports_missing_clone_glue() {
    let mut w = World::new();
    w.register_snapshot_component::<Pos>();
    // Vel has no glue registered.
    let _e = w.spawn((Pos { x: 1, y: 2 }, Vel(3)));
    let err = match w.try_snapshot() {
        Ok(_) => panic!("expected try_snapshot to refuse: Vel has no clone glue"),
        Err(missing) => missing,
    };
    let vel_id = w.components().id_of::<Vel>().unwrap();
    assert!(err.contains(&vel_id));
}

#[test]
fn rollback_ring_bounds_memory_and_restores_frames() {
    let (mut w, e0, ..) = populate();
    let mut ring = SnapshotRing::new(3);
    assert!(ring.is_empty());

    // Record 5 frames into a capacity-3 ring; only the newest 3 survive.
    let mut hashes: Vec<(u64, u64)> = Vec::new();
    for frame in 0..5u64 {
        w.get_mut::<Pos>(e0).unwrap().x = frame as i32;
        let snap = w.snapshot();
        hashes.push((frame, snap.state_hash()));
        ring.push(frame, snap);
    }
    assert_eq!(ring.len(), 3);
    assert_eq!(ring.capacity(), 3);
    assert!(ring.get(0).is_none());
    assert!(ring.get(1).is_none());
    assert!(ring.contains(4));
    assert_eq!(ring.oldest().map(|(f, _)| f), Some(2));
    assert_eq!(ring.latest().map(|(f, _)| f), Some(4));
    let kept: Vec<u64> = ring.frames().collect();
    assert_eq!(kept, Vec::from([2, 3, 4]));

    // Rolling back to a retained frame reproduces that frame exactly.
    let want = ring.get(3).unwrap();
    let want_hash = want.state_hash();
    w.restore(want);
    assert_eq!(w.get::<Pos>(e0), Some(&Pos { x: 3, y: 2 }));
    assert_eq!(w.snapshot().state_hash(), want_hash);
    assert_eq!(want_hash, hashes[3].1);
}

#[test]
fn rollback_ring_overwrites_reconfirmed_frame() {
    let (w, ..) = populate();
    let mut ring = SnapshotRing::new(4);
    ring.push(10, w.snapshot());
    ring.push(11, w.snapshot());
    // Re-confirming frame 10 overwrites in place without changing order/length.
    ring.push(10, w.snapshot());
    assert_eq!(ring.len(), 2);
    assert_eq!(ring.oldest().map(|(f, _)| f), Some(10));
    assert_eq!(ring.latest().map(|(f, _)| f), Some(11));
}

#[test]
fn restore_rebuilds_owning_group_membership() {
    // A `restore` replaces archetype/sparse storage wholesale; the owning-group
    // registry survives as declarations but its packed prefix must be rebuilt
    // against the re-materialised entities (design §6 / §14).
    let mut w = world_with_glue();
    let pos = w.register_component::<Pos>();
    let vel = w.register_component::<Vel>();
    let group = w.register_owning_group(&[pos, vel]).unwrap();

    let member = w.spawn((Pos { x: 1, y: 2 }, Vel(3)));
    let _partial = w.spawn(Pos { x: 9, y: 9 });
    assert!(w.owning_group(group).unwrap().contains(member));
    assert_eq!(w.owning_group(group).unwrap().len(), 1);

    let snap = w.snapshot();

    // Diverge: despawn the member and add a post-snapshot full member so the
    // live group no longer matches the snapshot.
    w.despawn(member);
    let other = w.spawn((Pos { x: 5, y: 5 }, Vel(6)));
    assert!(!w.owning_group(group).unwrap().contains(member));
    assert!(w.owning_group(group).unwrap().contains(other));

    // Restore: storage and the allocator revert exactly, so the rebuilt group
    // must contain the restored `member` and nothing else (`other`'s slot is
    // no longer live).
    w.restore(&snap);
    let g = w.owning_group(group).unwrap();
    assert_eq!(g.len(), 1);
    assert!(g.contains(member));
    assert!(!g.contains(other));
}
