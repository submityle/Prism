//! Fragmenting pairs are real archetype keys, not marker columns.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::archetype::{ArchetypeId, RelationPair};
use crate::change::ComponentTicks;
use crate::command::CommandQueue;
use crate::component::{Component, ComponentId, Components, StorageType};
use crate::component_hooks::{ComponentHooks, HookContext};
use crate::entity::Entity;
use crate::hierarchy::ChildOf;
#[cfg(feature = "std")]
use crate::relation::CleanupPolicy;
use crate::relation::{RelationKind, RelationTarget};
use crate::world::World;

struct Link;
impl Component for Link {}
struct Other;
impl Component for Other {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Value(i32);
impl Component for Value {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Sparse(i32);
impl Component for Sparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Batch(u32);
impl Component for Batch {
    const STORAGE: StorageType = StorageType::Shared;

    fn install_storage_glue(components: &mut Components, id: ComponentId) {
        components.set_shared_box(id, crate::component::shared_box_of::<Self>());
    }
}

struct Owned(Arc<AtomicUsize>);
impl Component for Owned {}
impl Drop for Owned {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn archetype(world: &World, entity: Entity) -> ArchetypeId {
    world.entities().location(entity).unwrap().archetype_id
}

fn pairs(world: &World, entity: Entity) -> Vec<RelationPair> {
    world
        .archetypes()
        .get(archetype(world, entity))
        .unwrap()
        .relation_pairs()
        .to_vec()
}

fn assert_ticks(actual: ComponentTicks, expected: ComponentTicks) {
    assert_eq!(actual.added, expected.added);
    assert_eq!(actual.changed, expected.changed);
}

fn fragmenting() -> RelationKind {
    RelationKind::new().with_fragmenting(true)
}

#[test]
fn fragmenting_pairs_split_and_merge_real_archetypes() {
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    let (a, b, c) = (
        world.spawn(Value(1)),
        world.spawn(Value(2)),
        world.spawn(Value(3)),
    );
    let (t1, t2) = (world.spawn(()), world.spawn(()));
    let base = archetype(&world, a);
    world.add_relation::<Link>(a, t1);
    world.add_relation::<Link>(b, t1);
    world.add_relation::<Link>(c, t2);
    assert_eq!(archetype(&world, a), archetype(&world, b));
    assert_ne!(archetype(&world, a), base);
    assert_ne!(archetype(&world, a), archetype(&world, c));
    let rid = world.components().id_of::<Link>().unwrap();
    assert_eq!(pairs(&world, a), vec![(rid, t1)]);
    let arch = world.archetypes().get(archetype(&world, a)).unwrap();
    assert_eq!(arch.len(), 2);
    assert!(!arch.components().contains(rid));
    assert!(arch.table().column(rid).is_none());
    assert_eq!(
        world.query_pair::<Link>(RelationTarget::Entity(t2)),
        vec![(c, t2)]
    );
    assert_eq!(world.query_pair::<Link>(RelationTarget::Wildcard).len(), 3);
    assert!(world.remove_relation::<Link>(a, t1));
    assert_eq!(archetype(&world, a), base);
    assert_eq!(world.get::<Value>(b), Some(&Value(2)));
    assert_eq!(world.get::<Value>(c), Some(&Value(3)));
}

#[test]
fn shared_value_eviction_releases_fragmenting_relation_identity_without_id_reuse() {
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    let (source, target) = (world.spawn(Batch(9)), world.spawn(()));
    world.add_relation::<Link>(source, target);

    let evicted_id = archetype(&world, source);
    let relation_id = world.components().id_of::<Link>().unwrap();
    let batch_id = world.components().id_of::<Batch>().unwrap();
    let before = world.archetypes().get(evicted_id).unwrap();
    assert_eq!(before.relation_pairs(), &[(relation_id, target)]);
    assert!(before.shared_binding(batch_id).is_some());
    assert!(before.shared_arc(batch_id).is_some());
    assert!(!before.is_evicted());

    assert!(world.despawn(source));
    assert!(world.archetypes().get(evicted_id).is_none());

    let replacement = world.spawn(Batch(9));
    let replacement_id = archetype(&world, replacement);
    assert_ne!(
        replacement_id, evicted_id,
        "stable tombstone IDs are not reused"
    );
    assert!(world.archetypes().get(evicted_id).is_none());
    assert!(!world.archetypes().get(replacement_id).unwrap().is_evicted());
    assert!(world
        .archetypes()
        .get(replacement_id)
        .unwrap()
        .relation_pairs()
        .is_empty());
}

#[test]
fn nonfragmenting_edges_never_move_rows_or_create_archetypes() {
    let mut world = World::new();
    let source = world.spawn(Value(1));
    let (a, b) = (world.spawn(()), world.spawn(()));
    let loc = world.entities().location(source).unwrap();
    let count = world.archetypes().len();
    world.add_relation::<Link>(source, a);
    world.add_relation::<Link>(source, b);
    world.remove_relation::<Link>(source, a);
    assert_eq!(world.entities().location(source), Some(loc));
    assert_eq!(world.archetypes().len(), count);
    assert!(pairs(&world, source).is_empty());
    assert_eq!(world.relation_targets::<Link>(source), &[b]);
}

#[test]
fn pair_identity_is_canonical_across_kinds_and_insertion_order() {
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    world.register_relation::<Other>(fragmenting());
    let (a, b, t1, t2) = (
        world.spawn(()),
        world.spawn(()),
        world.spawn(()),
        world.spawn(()),
    );
    world.add_relation::<Link>(a, t1);
    world.add_relation::<Other>(a, t2);
    world.add_relation::<Link>(a, t2);
    world.add_relation::<Link>(b, t2);
    world.add_relation::<Other>(b, t2);
    world.add_relation::<Link>(b, t1);
    assert_eq!(archetype(&world, a), archetype(&world, b));
    let count = world.archetypes().len();
    world.add_relation::<Link>(a, t1);
    assert_eq!(world.archetypes().len(), count);
    assert_eq!(pairs(&world, a).len(), 3);
    assert_eq!(world.relation_targets::<Link>(a), &[t1, t2]);
    assert_eq!(world.relation_targets::<Link>(b), &[t2, t1]);
}

#[test]
fn exclusive_replacement_rekeys_once_and_cleans_reverse_index() {
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting().with_exclusive(true));
    let (source, a, b) = (world.spawn(Value(7)), world.spawn(()), world.spawn(()));
    let base = archetype(&world, source);
    world.add_relation::<Link>(source, a);
    let first = archetype(&world, source);
    let count = world.archetypes().len();
    assert_eq!(world.add_relation::<Link>(source, b), Some(a));
    assert_eq!(world.archetypes().len(), count + 1);
    assert_eq!(pairs(&world, source).len(), 1);
    assert!(world.relation_sources::<Link>(a).is_empty());
    assert_eq!(world.relation_sources::<Link>(b), &[source]);
    assert_eq!(world.add_relation::<Link>(source, a), Some(b));
    assert_eq!(archetype(&world, source), first);
    assert_eq!(world.add_relation::<Link>(source, a), None);
    assert!(!world.remove_relation::<Link>(source, b));
    assert!(world.remove_relation::<Link>(source, a));
    assert_eq!(archetype(&world, source), base);
}

#[test]
fn pair_moves_preserve_ownership_ticks_sparse_shared_and_groups() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    let source = world.spawn((Owned(drops.clone()), Value(1), Sparse(2), Batch(3)));
    let neighbor = world.spawn((Owned(drops.clone()), Value(4), Sparse(5), Batch(3)));
    let target = world.spawn(());
    let vid = world.components().id_of::<Value>().unwrap();
    let sid = world.components().id_of::<Sparse>().unwrap();
    let bid = world.components().id_of::<Batch>().unwrap();
    let group = world.register_owning_group(&[vid, sid]).unwrap();
    let original_group = world.owning_group(group).unwrap().packed().to_vec();
    let original_sparse = world.sparse_sets.get(sid).unwrap().entities().to_vec();
    let table_ticks = world.get_ticks::<Value>(source).unwrap();
    let sparse_ticks = world.get_ticks::<Sparse>(source).unwrap();
    let src = world.archetypes().get(archetype(&world, source)).unwrap();
    let binding = src.shared_binding(bid).unwrap();
    let arc = src.shared_arc(bid).unwrap().clone();
    world.increment_change_tick();
    for _ in 0..3 {
        world.add_relation::<Link>(source, target);
        assert!(world.remove_relation::<Link>(source, target));
    }
    world.add_relation::<Link>(source, target);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_ticks(world.get_ticks::<Value>(source).unwrap(), table_ticks);
    assert_ticks(world.get_ticks::<Sparse>(source).unwrap(), sparse_ticks);
    assert_eq!(world.get::<Value>(source), Some(&Value(1)));
    assert_eq!(world.get::<Value>(neighbor), Some(&Value(4)));
    assert_eq!(world.get::<Sparse>(source), Some(&Sparse(2)));
    assert_eq!(world.get::<Batch>(source), Some(&Batch(3)));
    assert_eq!(world.owning_group(group).unwrap().packed(), original_group);
    assert_eq!(
        world.sparse_sets.get(sid).unwrap().entities(),
        original_sparse
    );
    assert_eq!(world.shared_components.refcount(bid, binding), 2);
    let dst = world.archetypes().get(archetype(&world, source)).unwrap();
    assert!(Arc::ptr_eq(dst.shared_arc(bid).unwrap(), &arc));
    world.despawn(source);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(world.shared_components.refcount(bid, binding), 1);
    drop(world);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn component_and_shared_structural_changes_preserve_pairs() {
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    let (source, target) = (world.spawn(()), world.spawn(()));
    world.add_relation::<Link>(source, target);
    let expected = pairs(&world, source);
    assert!(world.insert(source, Value(1)));
    assert_eq!(pairs(&world, source), expected);
    assert!(world.insert(source, Batch(2)));
    assert_eq!(pairs(&world, source), expected);
    assert!(world.insert(source, Batch(3)));
    assert_eq!(pairs(&world, source), expected);
    assert!(world.remove::<Value>(source));
    assert_eq!(pairs(&world, source), expected);
    assert_eq!(world.get::<Batch>(source), Some(&Batch(3)));
    assert!(world.remove::<Batch>(source));
    assert_eq!(pairs(&world, source), expected);
    assert!(world.remove_relation::<Link>(source, target));
    assert_eq!(archetype(&world, source), ArchetypeId::EMPTY);
}

#[test]
fn coalesced_commands_preserve_fragmenting_identity() {
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    let (source, target) = (world.spawn(Value(1)), world.spawn(()));
    world.add_relation::<Link>(source, target);
    let expected = pairs(&world, source);
    let mut queue = CommandQueue::new();
    queue
        .commands(world.entities())
        .entity(source)
        .insert(Sparse(2))
        .remove::<Value>();
    queue.apply(&mut world);
    assert_eq!(pairs(&world, source), expected);
    assert_eq!(world.get::<Sparse>(source), Some(&Sparse(2)));
    assert!(world.get::<Value>(source).is_none());
}

#[test]
fn recycled_targets_do_not_reuse_stale_pair_archetypes() {
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    let (source, old) = (world.spawn(Value(1)), world.spawn(()));
    world.add_relation::<Link>(source, old);
    let previous = archetype(&world, source);
    world.despawn(old);
    assert!(pairs(&world, source).is_empty());
    let new = world.spawn(());
    assert_eq!(new.index(), old.index());
    assert_ne!(new.generation(), old.generation());
    world.add_relation::<Link>(source, new);
    assert_ne!(archetype(&world, source), previous);
    assert!(world.relation_sources::<Link>(old).is_empty());
    assert!(!world.remove_relation::<Link>(source, old));
    assert_eq!(world.relation_targets::<Link>(source), &[new]);
}

#[test]
fn metadata_changes_rekey_existing_holders_and_normalize_exclusivity() {
    let mut world = World::new();
    let (source, a, b) = (world.spawn(Value(1)), world.spawn(()), world.spawn(()));
    let base = archetype(&world, source);
    world.add_relation::<Link>(source, a);
    world.add_relation::<Link>(source, b);
    world.register_relation::<Link>(fragmenting());
    assert_ne!(archetype(&world, source), base);
    assert_eq!(pairs(&world, source).len(), 2);
    world.register_relation::<Link>(fragmenting().with_exclusive(true));
    assert_eq!(world.relation_targets::<Link>(source), &[b]);
    assert_eq!(pairs(&world, source).len(), 1);
    assert!(world.relation_sources::<Link>(a).is_empty());
    world.register_relation::<Link>(RelationKind::new());
    assert_eq!(archetype(&world, source), base);
    assert_eq!(world.relation_targets::<Link>(source), &[b]);
}

#[test]
fn bulk_registry_replacement_and_forgotten_edits_remain_consistent() {
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting().with_exclusive(true));
    let (source, a, b) = (world.spawn(()), world.spawn(()), world.spawn(()));
    let rid = world.components().id_of::<Link>().unwrap();
    world.add_relation::<Link>(source, a);
    let original = world.relations().clone();
    {
        let mut edit = world.relations_mut();
        edit.index_mut().add(rid, source, b, false);
    }
    assert_eq!(world.relation_targets::<Link>(source), &[b]);
    assert_eq!(pairs(&world, source), vec![(rid, b)]);
    {
        let mut edit = world.relations_mut();
        edit.remove(rid, source, b);
        core::mem::forget(edit);
    }
    assert_eq!(world.relation_targets::<Link>(source), &[b]);
    assert_eq!(pairs(&world, source), vec![(rid, b)]);
    *world.relations_mut() = original;
    assert_eq!(pairs(&world, source), vec![(rid, a)]);
    *world.relations_mut() = Default::default();
    assert!(pairs(&world, source).is_empty());
    assert_eq!(archetype(&world, source), ArchetypeId::EMPTY);
}

#[test]
fn bulk_edits_discard_dead_and_reserved_endpoints() {
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    let (source, dead) = (world.spawn(()), world.spawn(()));
    world.despawn(dead);
    let reserved = world.entities().reserve_entity();
    let rid = world.components().id_of::<Link>().unwrap();
    {
        let mut edit = world.relations_mut();
        edit.add(rid, source, dead);
        edit.add(rid, source, reserved);
    }
    assert!(pairs(&world, source).is_empty());
    assert!(world.relations().index().is_empty());
    world.flush_reserved();
    assert!(world.despawn(reserved));
}

#[test]
fn bulk_edits_filter_stale_generations_without_discarding_live_edges() {
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    let old_source = world.spawn(());
    let old_target = world.spawn(());
    world.despawn(old_source);
    world.despawn(old_target);
    let target = world.spawn(());
    let source = world.spawn(Value(9));
    assert_eq!(source.index(), old_source.index());
    assert_eq!(target.index(), old_target.index());
    let rid = world.components().id_of::<Link>().unwrap();
    {
        let mut edit = world.relations_mut();
        edit.add(rid, old_source, target);
        edit.add(rid, source, old_target);
        edit.add(rid, source, target);
    }
    assert_eq!(world.relations().index().edge_count(), 1);
    assert_eq!(world.relation_targets::<Link>(source), &[target]);
    assert_eq!(world.relation_sources::<Link>(target), &[source]);
    assert!(world.relation_targets::<Link>(old_source).is_empty());
    assert!(world.relation_sources::<Link>(old_target).is_empty());
    assert_eq!(pairs(&world, source), vec![(rid, target)]);
    world.despawn(target);
    assert!(pairs(&world, source).is_empty());
    assert_eq!(world.get::<Value>(source), Some(&Value(9)));
}

#[test]
fn cascade_remove_rekeys_survivors_without_losing_other_pairs() {
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    world.register_relation::<Other>(fragmenting());
    let (source, a, b) = (world.spawn(Value(1)), world.spawn(()), world.spawn(()));
    world.add_relation::<Link>(source, a);
    world.add_relation::<Other>(source, b);
    world.despawn(a);
    let oid = world.components().id_of::<Other>().unwrap();
    assert_eq!(pairs(&world, source), vec![(oid, b)]);
    assert_eq!(world.get::<Value>(source), Some(&Value(1)));
    assert!(world.relation_targets::<Link>(source).is_empty());
    world.despawn(b);
    assert!(pairs(&world, source).is_empty());
}

#[test]
fn fragmenting_delete_cascade_handles_cycles_and_shared_tables() {
    let mut world = World::new();
    world.register_child_of();
    let (root, a, b) = (
        world.spawn(Value(1)),
        world.spawn(Value(2)),
        world.spawn(Value(3)),
    );
    let survivor = world.spawn(Value(4));
    world.add_relation::<ChildOf>(a, root);
    world.add_relation::<ChildOf>(b, a);
    world.add_relation::<ChildOf>(root, b);
    world.despawn(root);
    assert!(!world.contains(root));
    assert!(!world.contains(a));
    assert!(!world.contains(b));
    assert_eq!(world.get::<Value>(survivor), Some(&Value(4)));
    assert!(world.relations().index().is_empty());
}

#[test]
fn lifecycle_hook_added_incoming_pair_is_cleaned_before_free() {
    struct Attach(Entity);
    impl Component for Attach {}
    fn attach(ctx: HookContext<'_>) {
        let holder = ctx.world.get::<Attach>(ctx.entity).unwrap().0;
        ctx.world.add_relation::<Link>(holder, ctx.entity);
    }
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    world.register_component_hooks::<Attach>(ComponentHooks::new().with_on_remove(attach));
    let holder = world.spawn(Value(1));
    let target = world.spawn(Attach(holder));
    world.despawn(target);
    assert!(world.contains(holder));
    assert!(pairs(&world, holder).is_empty());
    assert!(world.relations().index().is_empty());
}

#[test]
fn pair_migration_does_not_invoke_component_hooks() {
    fn forbidden(_: HookContext<'_>) {
        panic!("pair migration must not run value hooks");
    }
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    let (source, target) = (world.spawn(Value(7)), world.spawn(()));
    world.register_component_hooks::<Value>(
        ComponentHooks::new()
            .with_on_add(forbidden)
            .with_on_insert(forbidden)
            .with_on_replace(forbidden)
            .with_on_remove(forbidden),
    );
    world.add_relation::<Link>(source, target);
    world.remove_relation::<Link>(source, target);
    assert_eq!(world.get::<Value>(source), Some(&Value(7)));
}

#[test]
fn pair_moves_support_zero_size_and_high_alignment_columns() {
    struct Zst;
    impl Component for Zst {}
    #[repr(align(256))]
    struct Aligned(u64);
    impl Component for Aligned {}
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    let (source, target) = (world.spawn((Zst, Aligned(19))), world.spawn(()));
    for _ in 0..4 {
        world.add_relation::<Link>(source, target);
        let value = world.get::<Aligned>(source).unwrap();
        assert_eq!((value as *const Aligned as usize) % 256, 0);
        assert_eq!(value.0, 19);
        assert!(world.get::<Zst>(source).is_some());
        world.remove_relation::<Link>(source, target);
    }
}

#[test]
fn snapshot_restores_fragmenting_identity_for_mixed_and_pair_only_holders() {
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    world.register_snapshot_component_hashable::<Value>();
    world.register_snapshot_component_hashable::<Sparse>();
    world.register_snapshot_component_hashable::<Batch>();
    world.register_snapshot_component_equatable::<Value>();
    world.register_snapshot_component_equatable::<Sparse>();
    world.register_snapshot_component_equatable::<Batch>();
    let target = world.spawn(());
    let mixed = world.spawn((Value(1), Sparse(2), Batch(3)));
    let bare = world.spawn(());
    world.add_relation::<Link>(mixed, target);
    world.add_relation::<Link>(bare, target);
    let rid = world.components().id_of::<Link>().unwrap();
    let ticks = world.get_ticks::<Value>(mixed).unwrap();
    let state = world.query::<&Value>();
    let snapshot = world.snapshot();
    world.register_relation::<Link>(RelationKind::new());
    world.insert(mixed, (Value(4), Batch(5)));
    world.remove_relation::<Link>(bare, target);
    world.despawn(target);
    world.restore(&snapshot);
    assert_eq!(pairs(&world, mixed), vec![(rid, target)]);
    assert_eq!(pairs(&world, bare), vec![(rid, target)]);
    assert_ne!(archetype(&world, bare), ArchetypeId::EMPTY);
    assert_eq!(world.get::<Value>(mixed), Some(&Value(1)));
    assert_eq!(world.get::<Sparse>(mixed), Some(&Sparse(2)));
    assert_eq!(world.get::<Batch>(mixed), Some(&Batch(3)));
    assert_ticks(world.get_ticks::<Value>(mixed).unwrap(), ticks);
    assert_eq!(state.iter(&world).map(|v| v.0).collect::<Vec<_>>(), vec![1]);
    let after = world.snapshot();
    assert!(
        snapshot.structurally_eq(&after),
        "restore reproduces relation/fragments exactly"
    );
    assert_eq!(snapshot.state_hash(), after.state_hash());
    world.despawn(target);
    assert!(world.contains(bare));
    assert_eq!(archetype(&world, bare), ArchetypeId::EMPTY);
    assert!(pairs(&world, mixed).is_empty());
}

#[cfg(feature = "std")]
#[test]
fn panic_cleanup_policy_rejects_without_partial_edges_rows_or_groups() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting().with_on_delete_target(CleanupPolicy::Panic));
    world.register_relation::<Other>(fragmenting().with_on_delete_target(CleanupPolicy::Delete));
    let (holder, target, child) = (
        world.spawn(Value(1)),
        world.spawn(Value(2)),
        world.spawn(Value(3)),
    );
    world.add_relation::<Link>(holder, target);
    world.add_relation::<Other>(child, target);
    let vid = world.components().id_of::<Value>().unwrap();
    let group = world.register_owning_group(&[vid]).unwrap();
    let before_group = world.owning_group(group).unwrap().packed().to_vec();
    let before_pairs = pairs(&world, holder);
    let before_loc = world.entities().location(target);
    assert!(catch_unwind(AssertUnwindSafe(|| world.despawn(target))).is_err());
    assert_eq!(world.owning_group(group).unwrap().packed(), before_group);
    assert_eq!(world.entities().location(target), before_loc);
    assert_eq!(pairs(&world, holder), before_pairs);
    assert!(world.has_relation::<Other>(child, target));
    assert!(world.contains(child));
    assert_eq!(world.get::<Value>(target), Some(&Value(2)));
}

#[cfg(feature = "std")]
#[test]
fn bulk_guard_unwind_publishes_pairs_without_dropping_values() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let drops = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    let (source, target) = (world.spawn(Owned(drops.clone())), world.spawn(()));
    let rid = world.components().id_of::<Link>().unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let mut edit = world.relations_mut();
        edit.add(rid, source, target);
        panic!("bulk editor failed after an accepted edit");
    }))
    .is_err());
    assert_eq!(pairs(&world, source), vec![(rid, target)]);
    assert!(world.has_relation::<Link>(source, target));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(world);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[cfg(feature = "std")]
#[test]
fn typed_add_rejects_dead_endpoints_before_changing_archetypes() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let mut world = World::new();
    world.register_relation::<Link>(fragmenting());
    let (source, dead) = (world.spawn(()), world.spawn(()));
    world.despawn(dead);
    let count = world.archetypes().len();
    assert!(catch_unwind(AssertUnwindSafe(|| world.add_relation::<Link>(source, dead))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| world.add_relation::<Link>(dead, source))).is_err());
    assert_eq!(world.archetypes().len(), count);
    assert!(world.relations().index().is_empty());
}
