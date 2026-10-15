//! Owning, padded, fixed-point and opaque values across real snapshot deltas.

use prism_ecs::prelude::*;
use prism_math::Fixed;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[repr(C)]
#[derive(Component, Clone, Debug, PartialEq, Eq)]
struct Owned {
    flag: u8,
    count: u64,
    name: String,
    samples: Vec<u32>,
}

fn owned(count: u64) -> Owned {
    Owned {
        flag: 1,
        count,
        name: "owning value".into(),
        samples: vec![2, 3, 5],
    }
}

#[test]
fn independently_allocated_string_vec_and_padding_snapshots_compare_equal() {
    let mut world = World::new();
    world.register_snapshot_component_equatable::<Owned>();
    world.spawn((owned(7),));
    let a = world.snapshot();
    let b = world.snapshot();
    assert!(a.structurally_eq(&b));
    let delta = a.diff(&b);
    assert_eq!(delta.changed_cell_count(), 0);
    assert_eq!(delta.reused_cell_count(), 1);
    assert!(delta.apply(&a).structurally_eq(&b));
}

#[test]
fn owning_delta_keeps_fresh_and_reused_values_after_sources_are_dropped() {
    let mut world = World::new();
    world.register_snapshot_component_equatable::<Owned>();
    let a = world.spawn((owned(7),));
    let b = world.spawn((owned(8),));
    let base = world.snapshot();
    world
        .get_mut::<Owned>(a)
        .unwrap()
        .name
        .push_str(" corrected");
    let target = world.snapshot();
    let delta = base.diff(&target);
    assert_eq!(delta.changed_cell_count(), 1);
    assert_eq!(delta.reused_cell_count(), 1);
    drop(target);
    world.despawn(a);
    world.despawn(b);
    let rebuilt = delta.apply(&base);
    drop(delta);
    drop(base);
    world.restore(&rebuilt);
    assert_eq!(
        world.get::<Owned>(a).unwrap().name,
        "owning value corrected"
    );
    assert_eq!(world.get::<Owned>(b), Some(&owned(8)));
    assert!(rebuilt.structurally_eq(&world.snapshot()));
}

#[derive(Component, Clone)]
struct Opaque(Vec<String>);

#[test]
fn clone_only_values_remain_capturable_restorable_and_conservatively_fresh() {
    let mut world = World::new();
    world.register_snapshot_component::<Opaque>();
    let entity = world.spawn((Opaque(vec!["kept".into()]),));
    let base = world.snapshot();
    let target = world.snapshot();
    assert!(!base.structurally_eq(&target));
    let delta = base.diff(&target);
    assert_eq!(delta.changed_cell_count(), 1);
    assert_eq!(delta.reused_cell_count(), 0);
    drop(target);
    let rebuilt = delta.apply(&base);
    drop(base);
    drop(delta);
    world.get_mut::<Opaque>(entity).unwrap().0.clear();
    world.restore(&rebuilt);
    assert_eq!(world.get::<Opaque>(entity).unwrap().0, ["kept"]);
}

#[derive(Resource, Clone)]
struct OpaqueResource(Vec<String>);

#[test]
fn clone_only_resource_delta_owns_target_value_without_claiming_equality() {
    let mut world = World::new();
    world.register_snapshot_resource::<OpaqueResource>();
    world.insert_resource(OpaqueResource(vec!["before".into()]));
    let base = world.snapshot();
    world.resource_mut::<OpaqueResource>().0 = vec!["after".into()];
    let target = world.snapshot();
    assert!(!base.structurally_eq(&target));
    let delta = base.diff(&target);
    drop(target);
    let rebuilt = delta.apply(&base);
    drop((base, delta));
    world.resource_mut::<OpaqueResource>().0.clear();
    world.restore(&rebuilt);
    assert_eq!(world.resource::<OpaqueResource>().0, ["after"]);
    world
        .resource_mut::<OpaqueResource>()
        .0
        .push("later".into());
    world.restore(&rebuilt);
    assert_eq!(world.resource::<OpaqueResource>().0, ["after"]);
}

#[derive(Component, Clone, Debug, PartialEq, Eq)]
#[component(storage = "SparseSet")]
struct SparseOwned(Vec<String>);

#[derive(Component, Clone, Debug, PartialEq, Eq, Hash)]
#[component(storage = "Shared")]
struct SharedOwned(String);

#[test]
fn sparse_and_shared_owning_values_survive_delta_and_repeated_restore() {
    let mut world = World::new();
    world.register_snapshot_component_equatable::<SparseOwned>();
    world.register_snapshot_component_equatable::<SharedOwned>();
    let first = world.spawn((
        SparseOwned(vec!["before".into()]),
        SharedOwned("shared".into()),
    ));
    let second = world.spawn((SharedOwned("shared".into()),));
    let base = world.snapshot();
    let same = world.snapshot();
    assert!(base.structurally_eq(&same));
    assert_eq!(base.diff(&same).reused_cell_count(), 3);
    world.get_mut::<SparseOwned>(first).unwrap().0 = vec!["after".into()];
    assert!(world.insert(first, (SharedOwned("changed".into()),)));
    let target = world.snapshot();
    let delta = base.diff(&target);
    assert_eq!(delta.changed_cell_count(), 2);
    assert_eq!(delta.reused_cell_count(), 1);
    drop((target, same));
    let rebuilt = delta.apply(&base);
    drop((base, delta));
    for _ in 0..2 {
        world.despawn(first);
        world.restore(&rebuilt);
        assert_eq!(world.get::<SparseOwned>(first).unwrap().0, ["after"]);
        assert_eq!(world.get::<SharedOwned>(first).unwrap().0, "changed");
        assert_eq!(world.get::<SharedOwned>(second).unwrap().0, "shared");
        assert!(rebuilt.structurally_eq(&world.snapshot()));
    }
}

#[derive(Component, Clone, Debug, PartialEq, Eq, Hash)]
struct FixedPosition(Fixed);

#[test]
fn fixed_point_comparison_and_delta_roundtrip_preserve_raw_integer_values() {
    let mut world = World::new();
    world.register_snapshot_component_hashable::<FixedPosition>();
    world.register_snapshot_component_equatable::<FixedPosition>();
    let entity = world.spawn((FixedPosition(Fixed::from_bits(i64::MIN + 17)),));
    let base = world.snapshot();
    let same = world.snapshot();
    assert!(base.structurally_eq(&same));
    assert_eq!(base.state_hash(), same.state_hash());
    assert_eq!(base.diff(&same).reused_cell_count(), 1);
    world.get_mut::<FixedPosition>(entity).unwrap().0 = Fixed::from_bits(i64::MAX - 17);
    let target = world.snapshot();
    let rebuilt = base.diff(&target).apply(&base);
    assert!(rebuilt.structurally_eq(&target));
    assert_ne!(base.state_hash(), target.state_hash());
    world.restore(&rebuilt);
    assert_eq!(
        world.get::<FixedPosition>(entity).unwrap().0.to_bits(),
        i64::MAX - 17
    );
}

#[derive(Component, Resource, Clone, PartialEq, Eq)]
struct Collision(String);
impl Hash for Collision {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u8(0);
    }
}

#[test]
fn equal_component_hashes_do_not_reuse_different_values() {
    let mut world = World::new();
    world.register_snapshot_component_hashable::<Collision>();
    world.register_snapshot_component_equatable::<Collision>();
    let entity = world.spawn((Collision("before".into()),));
    let base = world.snapshot();
    world.get_mut::<Collision>(entity).unwrap().0 = "after".into();
    let target = world.snapshot();
    assert!(!base.structurally_eq(&target));
    let delta = base.diff(&target);
    assert_eq!(delta.changed_cell_count(), 1);
    assert_eq!(delta.reused_cell_count(), 0);
    assert!(delta.apply(&base).structurally_eq(&target));
}

#[test]
fn equal_resource_hashes_do_not_establish_snapshot_equality() {
    let mut world = World::new();
    world.register_snapshot_resource_hashable::<Collision>();
    world.register_snapshot_resource_equatable::<Collision>();
    world.insert_resource(Collision("before".into()));
    let base = world.snapshot();
    world.resource_mut::<Collision>().0 = "after".into();
    let target = world.snapshot();
    assert_eq!(base.state_hash(), target.state_hash());
    assert!(!base.structurally_eq(&target));
    let delta = base.diff(&target);
    drop(target);
    let rebuilt = delta.apply(&base);
    drop(base);
    drop(delta);
    world.restore(&rebuilt);
    assert_eq!(world.resource::<Collision>().0, "after");
    assert!(rebuilt.structurally_eq(&world.snapshot()));
}

#[derive(Component)]
struct Tracked {
    data: String,
    live: Arc<AtomicUsize>,
}
impl Clone for Tracked {
    fn clone(&self) -> Self {
        let data = self.data.clone();
        self.live.fetch_add(1, Ordering::SeqCst);
        Self {
            data,
            live: self.live.clone(),
        }
    }
}
impl PartialEq for Tracked {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}
impl Drop for Tracked {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

#[test]
fn delta_owns_all_clones_after_live_world_and_target_are_dropped() {
    let live = Arc::new(AtomicUsize::new(1));
    {
        let mut world = World::new();
        world.register_snapshot_component_equatable::<Tracked>();
        let entity = world.spawn((Tracked {
            data: "old".into(),
            live: live.clone(),
        },));
        let base = world.snapshot();
        world.get_mut::<Tracked>(entity).unwrap().data = "new".into();
        let target = world.snapshot();
        let delta = base.diff(&target);
        let expected = delta.apply(&base);
        drop(target);
        drop(world);
        let rebuilt = delta.apply(&base);
        drop(base);
        drop(delta);
        assert_eq!(live.load(Ordering::SeqCst), 2);
        assert!(rebuilt.structurally_eq(&expected));
    }
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[cfg(feature = "determinism")]
#[test]
fn divergence_locator_uses_semantic_values_for_owning_components() {
    use prism_ecs::world::snapshot::{locate_divergence, SnapshotDivergence};
    let mut world = World::new();
    world.register_snapshot_component_equatable::<Owned>();
    let entity = world.spawn((owned(7),));
    let before = world.snapshot();
    let same = world.snapshot();
    assert!(locate_divergence(&before, &same).is_none());
    world.get_mut::<Owned>(entity).unwrap().samples.push(11);
    let after = world.snapshot();
    assert!(
        matches!(locate_divergence(&before, &after), Some(SnapshotDivergence::Value { entity: e, .. }) if e == entity)
    );
}
