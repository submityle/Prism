//! Snapshot completeness and exact delta baseline regression tests.

use super::*;
use crate::relation::{CleanupPolicy, RelationKind};

struct Link;
impl Component for Link {}
struct FutureLink;
impl Component for FutureLink {}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Value(u32);
impl Component for Value {}

fn world() -> World {
    let mut world = World::new();
    world.register_snapshot_component_hashable::<Value>();
    world.register_snapshot_component_equatable::<Value>();
    world
}

#[test]
fn restore_rebuilds_fragmenting_pairs_before_owning_groups() {
    let mut world = world();
    world.register_relation::<Link>(RelationKind::new().with_fragmenting(true));
    let value_id = world.components().id_of::<Value>().unwrap();
    let group = world.register_owning_group(&[value_id]).unwrap();
    let target = world.spawn(());
    let member = world.spawn(Value(17));
    let plain = world.spawn(Value(23));
    let pair_only = world.spawn(());
    world.add_relation::<Link>(member, target);
    world.add_relation::<Link>(pair_only, target);
    let link = world.components().id_of::<Link>().unwrap();
    let ticks = world.get_ticks::<Value>(member).unwrap();
    let snapshot = world.snapshot();

    world.remove_relation::<Link>(member, target);
    world.despawn(pair_only);
    world.despawn(plain);
    world.get_mut::<Value>(member).unwrap().0 = 99;
    world.restore(&snapshot);

    for entity in [member, pair_only] {
        let location = world.entities().location(entity).unwrap();
        let archetype = world.archetypes().get(location.archetype_id).unwrap();
        assert_eq!(archetype.relation_pairs(), &[(link, target)]);
        assert_eq!(archetype.table().entities()[location.row as usize], entity);
    }
    let location = world.entities().location(plain).unwrap();
    assert!(world
        .archetypes()
        .get(location.archetype_id)
        .unwrap()
        .relation_pairs()
        .is_empty());
    let packed = world.owning_group(group).unwrap();
    assert_eq!(packed.len(), 2);
    assert!(packed.contains(member));
    assert!(packed.contains(plain));
    assert!(!packed.contains(pair_only));
    assert_eq!(world.get::<Value>(member).unwrap().0, 17);
    assert_eq!(world.get::<Value>(plain).unwrap().0, 23);
    let restored_ticks = world.get_ticks::<Value>(member).unwrap();
    assert_eq!(restored_ticks.added, ticks.added);
    assert_eq!(restored_ticks.changed, ticks.changed);
    assert!(snapshot.structurally_eq(&world.snapshot()));
}

#[test]
fn relations_roundtrip_policies_cycles_generations_and_ordered_indexes() {
    let mut world = world();
    let kind = RelationKind::new().with_transitive(true);
    world.register_relation::<Link>(kind);
    let a = world.spawn(());
    let b = world.spawn((Value(2),));
    let c = world.spawn(());
    world.add_relation::<Link>(a, c);
    world.add_relation::<Link>(b, c);
    world.add_relation::<Link>(a, b);
    world.add_relation::<Link>(c, a);
    let snapshot = world.snapshot();
    assert_eq!(snapshot.relation_count(), 4);
    let generation = world.archetype_generation();

    world.register_relation::<Link>(RelationKind::new().with_exclusive(true));
    world.register_relation::<FutureLink>(RelationKind::new());
    world.despawn(c);
    let replacement = world.spawn(());
    assert_eq!(replacement.index(), c.index());
    assert_ne!(replacement, c);
    world.add_relation::<FutureLink>(a, replacement);

    world.restore(&snapshot);
    assert_ne!(generation, world.archetype_generation());
    assert!(world.contains(c));
    assert!(!world.contains(replacement));
    assert_eq!(world.relation_targets::<Link>(a), &[c, b]);
    assert_eq!(world.relation_sources::<Link>(c), &[a, b]);
    assert_eq!(world.relation_targets::<Link>(c), &[a]);
    let link = world.components().id_of::<Link>().unwrap();
    assert_eq!(world.relations().kind(link), Some(&kind));
    let future = world.components().id_of::<FutureLink>().unwrap();
    assert!(!world.relations().is_registered(future));
    assert!(world.relation_targets::<FutureLink>(a).is_empty());
    let restored = world.snapshot();
    assert!(snapshot.structurally_eq(&restored));
    assert_eq!(snapshot.state_hash(), restored.state_hash());
}

#[test]
fn restored_exclusive_and_cascade_policies_control_live_operations() {
    let mut world = world();
    let kind = RelationKind::new()
        .with_exclusive(true)
        .with_on_delete_target(CleanupPolicy::Delete);
    world.register_relation::<Link>(kind);
    let parent = world.spawn(());
    let child = world.spawn(());
    let alternate = world.spawn(());
    world.add_relation::<Link>(child, parent);
    let snapshot = world.snapshot();
    world.register_relation::<Link>(RelationKind::new());
    world.add_relation::<Link>(child, alternate);
    world.restore(&snapshot);
    assert_eq!(world.add_relation::<Link>(child, alternate), Some(parent));
    assert_eq!(world.relation_targets::<Link>(child), &[alternate]);
    assert!(world.relation_sources::<Link>(parent).is_empty());
    world.despawn(alternate);
    assert!(!world.contains(child));
    assert!(world.contains(parent));
}

#[test]
fn relation_only_delta_carries_target_policies_and_both_indexes() {
    let mut world = World::new();
    world.register_relation::<Link>(RelationKind::new());
    let a = world.spawn(());
    let b = world.spawn(());
    let c = world.spawn(());
    world.add_relation::<Link>(a, b);
    let base = world.snapshot();
    world.remove_relation::<Link>(a, b);
    world.add_relation::<Link>(c, b);
    world.add_relation::<Link>(a, c);
    world.register_relation::<Link>(RelationKind::new().with_transitive(true));
    let target = world.snapshot();
    let delta = base.diff(&target);
    assert_eq!(delta.changed_cell_count(), 0);
    assert_eq!(delta.reused_cell_count(), 0);
    assert_ne!(base.state_hash(), target.state_hash());
    let rebuilt = delta.apply(&base);
    drop(target);
    world.restore(&rebuilt);
    assert!(rebuilt.structurally_eq(&world.snapshot()));
    assert_eq!(world.relation_sources::<Link>(b), &[c]);
    assert_eq!(world.relation_targets::<Link>(a), &[c]);
}

#[test]
fn every_relation_policy_field_enters_hash_and_comparison() {
    let mut world = World::new();
    world.register_relation::<Link>(RelationKind::new());
    let base = world.snapshot();
    for kind in [
        RelationKind::new().with_fragmenting(true),
        RelationKind::new().with_transitive(true),
        RelationKind::new().with_exclusive(true),
        RelationKind::new().with_on_delete(CleanupPolicy::Delete),
        RelationKind::new().with_on_delete(CleanupPolicy::Panic),
        RelationKind::new().with_on_delete_target(CleanupPolicy::Delete),
        RelationKind::new().with_on_delete_target(CleanupPolicy::Panic),
    ] {
        world.register_relation::<Link>(kind);
        let changed = world.snapshot();
        assert_ne!(base.state_hash(), changed.state_hash());
        assert!(!base.structurally_eq(&changed));
        #[cfg(feature = "determinism")]
        assert_eq!(
            base.locate_divergence(&changed),
            Some(SnapshotDivergence::Relations)
        );
    }
}

#[test]
fn relation_hash_is_bucket_independent_but_preserves_observable_neighbor_order() {
    fn populate(reverse: bool) -> World {
        let mut world = World::new();
        world.register_relation::<Link>(RelationKind::new());
        world.register_relation::<FutureLink>(RelationKind::new());
        let a = world.spawn(());
        let b = world.spawn(());
        let c = world.spawn(());
        let d = world.spawn(());
        // Independent buckets may be inserted in either order.
        if reverse {
            world.add_relation::<Link>(c, d);
            world.add_relation::<Link>(a, b);
        } else {
            world.add_relation::<Link>(a, b);
            world.add_relation::<Link>(c, d);
        }
        world
    }
    let a = populate(false).snapshot();
    let b = populate(true).snapshot();
    assert_eq!(a.state_hash(), b.state_hash());
    assert!(a.structurally_eq(&b));

    let mut world = World::new();
    let x = world.spawn(());
    let y = world.spawn(());
    let z = world.spawn(());
    world.add_relation::<Link>(x, z);
    world.add_relation::<Link>(y, z);
    let base = world.snapshot();
    world.remove_relation::<Link>(x, z);
    world.add_relation::<Link>(x, z);
    let reordered_reverse = world.snapshot();
    assert_ne!(base.state_hash(), reordered_reverse.state_hash());
    assert!(!base.structurally_eq(&reordered_reverse));

    world.add_relation::<Link>(x, y);
    let base = world.snapshot();
    world.remove_relation::<Link>(x, z);
    world.add_relation::<Link>(x, z);
    let reordered_forward = world.snapshot();
    assert_ne!(base.state_hash(), reordered_forward.state_hash());
    assert!(!base.structurally_eq(&reordered_forward));
}

#[test]
fn hash_includes_both_tick_cursors_and_each_component_tick() {
    let mut world = world();
    world.spawn((Value(1),));
    let baseline = world.snapshot();
    for field in 0..4 {
        let mut changed = world.snapshot();
        match field {
            0 => changed.change_tick = Tick::new(100),
            1 => changed.last_change_tick = Tick::new(100),
            2 => changed.columns[0].added[0] = Tick::new(100),
            _ => changed.columns[0].changed[0] = Tick::new(100),
        }
        assert_ne!(baseline.state_hash(), changed.state_hash());
        assert!(!baseline.structurally_eq(&changed));
        #[cfg(feature = "determinism")]
        assert!(baseline.locate_divergence(&changed).is_some());
    }
}

#[test]
fn hash_includes_dead_slot_generations_and_free_list_order() {
    let mut a = World::new();
    let mut b = World::new();
    let dead = a.spawn(());
    a.despawn(dead);
    let dead = b.spawn(());
    b.despawn(dead);
    let baseline = a.snapshot();
    assert_eq!(baseline.state_hash(), b.snapshot().state_hash());
    let reused = b.spawn(());
    b.despawn(reused);
    assert_ne!(baseline.state_hash(), b.snapshot().state_hash());

    let mut a = World::new();
    let mut b = World::new();
    let a0 = a.spawn(());
    let a1 = a.spawn(());
    let b0 = b.spawn(());
    let b1 = b.spawn(());
    a.despawn(a0);
    a.despawn(a1);
    b.despawn(b1);
    b.despawn(b0);
    let left = a.snapshot();
    let right = b.snapshot();
    assert_eq!(left.live_entity_count(), 0);
    assert_eq!(right.live_entity_count(), 0);
    assert_ne!(left.state_hash(), right.state_hash());
    assert!(!left.structurally_eq(&right));
    #[cfg(feature = "determinism")]
    assert_eq!(
        left.locate_divergence(&right),
        Some(SnapshotDivergence::AllocatorState)
    );
    assert_ne!(a.spawn(()), b.spawn(()));
}

#[test]
fn delta_reconstructions_retain_target_lineage_for_chained_application() {
    let mut world = world();
    let entity = world.spawn((Value(1),));
    let a = world.snapshot();
    world.get_mut::<Value>(entity).unwrap().0 = 2;
    let b = world.snapshot();
    world.spawn((Value(3),));
    let c = world.snapshot();
    let ab = a.diff(&b);
    let bc = b.diff(&c);
    drop(b);
    let rebuilt_b = ab.apply(&a);
    let rebuilt_c = bc.apply(&rebuilt_b);
    assert!(rebuilt_c.structurally_eq(&c));
    assert_eq!(rebuilt_c.state_hash(), c.state_hash());
    assert!(bc.apply(&ab.apply(&a)).structurally_eq(&c));
}

#[cfg(feature = "std")]
#[test]
fn delta_rejects_independent_same_world_baselines_before_any_value_clone() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    struct Probe(Arc<AtomicUsize>);
    impl Component for Probe {}
    impl Clone for Probe {
        fn clone(&self) -> Self {
            self.0.fetch_add(1, Ordering::SeqCst);
            Self(self.0.clone())
        }
    }
    let mut world = World::new();
    world.register_snapshot_component::<Probe>();
    let count = Arc::new(AtomicUsize::new(0));
    let empty = world.snapshot();
    world.spawn((Probe(count.clone()),));
    let base = world.snapshot();
    let target = world.snapshot();
    let independent = world.snapshot();
    assert_eq!(base.state_hash(), independent.state_hash());
    let delta = base.diff(&target);
    let before = count.load(Ordering::SeqCst);
    for wrong in [&empty, &independent, &target] {
        let error = catch_unwind(AssertUnwindSafe(|| delta.apply(wrong)));
        assert!(error.is_err());
        assert_eq!(count.load(Ordering::SeqCst), before);
    }
    let _valid = delta.apply(&base);
    assert_eq!(count.load(Ordering::SeqCst), before + 1);
}

#[test]
fn value_hash_coverage_reports_clone_only_components_and_resources() {
    #[derive(Clone)]
    struct Opaque;
    impl Component for Opaque {}
    #[derive(Clone)]
    struct State;
    impl crate::resource::Resource for State {}
    let mut world = world();
    world.spawn((Value(1),));
    assert!(world.snapshot().value_hash_complete());
    world.register_snapshot_resource::<State>();
    world.insert_resource(State);
    assert!(!world.snapshot().value_hash_complete());
    world.remove_resource::<State>();
    assert!(world.snapshot().value_hash_complete());
    world.register_snapshot_component::<Opaque>();
    world.spawn((Opaque,));
    assert!(!world.snapshot().value_hash_complete());
}

#[cfg(feature = "determinism")]
#[test]
fn unavailable_comparison_is_not_reported_as_proven_value_drift() {
    #[derive(Clone)]
    struct Opaque;
    impl Component for Opaque {}
    impl crate::resource::Resource for Opaque {}
    let mut world = World::new();
    world.register_snapshot_component::<Opaque>();
    let entity = world.spawn((Opaque,));
    let a = world.snapshot();
    let b = world.snapshot();
    assert!(!a.structurally_eq(&b));
    assert_eq!(a.state_hash(), b.state_hash());
    assert_eq!(
        a.locate_divergence(&b),
        Some(SnapshotDivergence::ValueIncomparable {
            component: world.components().id_of::<Opaque>().unwrap(),
            entity,
        })
    );
    assert_eq!(a.diff(&b).reused_cell_count(), 0);

    world.despawn(entity);
    world.register_snapshot_resource::<Opaque>();
    world.insert_resource(Opaque);
    let a = world.snapshot();
    let b = world.snapshot();
    assert!(!a.structurally_eq(&b));
    assert_eq!(
        a.locate_divergence(&b),
        Some(SnapshotDivergence::ResourceIncomparable { index: 0 })
    );
}

#[test]
#[should_panic(expected = "snapshot requires deferred entity reservations")]
fn snapshot_refuses_unflushed_entity_reservations_in_every_build_mode() {
    let world = World::new();
    let _reserved = world.entities.reserve_entity();
    let _ = world.try_snapshot();
}

#[test]
fn delta_tick_only_change_is_fresh_and_replays_exact_ticks() {
    let mut world = world();
    world.spawn((Value(7),));
    let base = world.snapshot();
    let mut target = world.snapshot();
    target.columns[0].changed[0] = Tick::new(19);
    let delta = base.diff(&target);
    assert_eq!(delta.reused_cell_count(), 0);
    assert_eq!(delta.changed_cell_count(), 1);
    let rebuilt = delta.apply(&base);
    assert!(rebuilt.structurally_eq(&target));
    world.restore(&rebuilt);
    assert_eq!(world.snapshot().columns[0].changed[0], Tick::new(19));
}
