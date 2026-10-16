//! Work-count and semantic regressions for the source-delta hierarchy cache.
#![cfg(feature = "ecs")]

use prism_ecs::component::Component;
use prism_ecs::entity::Entity;
use prism_ecs::hierarchy::ChildOf;
use prism_ecs::observer::LifecycleEvent;
use prism_ecs::relation::{
    CleanupPolicy, Relation, RelationIndex, RelationKind, RELATION_CHANGE_CAPACITY,
};
use prism_ecs::world::snapshot::WorldSnapshot;
use prism_ecs::world::World;
use prism_transform::ecs::{install, propagate_transforms, uninstall, EcsPropagationStats};
use prism_transform::{GlobalTransform, Transform};

std::thread_local! {
    // Snapshot payloads are not Send/Sync. Keep test-only restore inputs on the
    // calling thread without enabling ECS's optional non-send resource API.
    static RESTORE_INPUT: std::cell::RefCell<Option<WorldSnapshot>> = const {
        std::cell::RefCell::new(None)
    };
}

fn restore_input(world: &mut World) {
    let snapshot = RESTORE_INPUT.with(|input| input.borrow_mut().take().unwrap());
    world.restore(&snapshot);
}

fn setup() -> World {
    let mut world = World::new();
    install(&mut world);
    world
}

fn x(world: &World, entity: Entity) -> f32 {
    world
        .get::<GlobalTransform>(entity)
        .unwrap()
        .translation()
        .x
}

fn scan(stats: EcsPropagationStats, _legacy_edge_count: usize) {
    assert_eq!(stats.relation_scans, 1);
    // Transform topology refreshes are source-indexed. It must not enumerate
    // or sort the global relation graph, which may contain unrelated kinds.
    assert_eq!((stats.relation_edges_scanned, stats.relation_sorts), (0, 0));
}

fn no_scan(stats: EcsPropagationStats) {
    assert_eq!(
        (
            stats.relation_scans,
            stats.relation_edges_scanned,
            stats.relation_sorts
        ),
        (0, 0, 0)
    );
    assert_eq!(stats.relation_checks, 0);
}

#[test]
fn static_frames_and_local_writes_never_collect_or_sort_edges() {
    let mut world = setup();
    let root = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let mut parent = root;
    for _ in 0..128 {
        let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
        world.add_relation::<ChildOf>(child, parent);
        parent = child;
    }
    scan(propagate_transforms(&mut world).unwrap(), 128);
    assert_eq!(x(&world, parent), 138.0);
    for _ in 0..8 {
        let stats = propagate_transforms(&mut world).unwrap();
        no_scan(stats);
        assert_eq!(
            (stats.local_reads, stats.recomputed, stats.written),
            (0, 0, 0)
        );
    }
    world.get_mut::<Transform>(root).unwrap().translation.x = 20.0;
    let stats = propagate_transforms(&mut world).unwrap();
    no_scan(stats);
    assert_eq!(
        (stats.local_reads, stats.recomputed, stats.written),
        (1, 129, 129)
    );
    assert!(stats.change_rows_scanned > 0);
    assert_eq!(x(&world, parent), 148.0);
}

#[test]
fn typed_edits_invalidate_once_and_typed_noops_keep_cache() {
    let mut world = setup();
    let a = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let b = world.spawn(Transform::from_xyz(20.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    let leaf = world.spawn(Transform::from_xyz(2.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(leaf, child);
    scan(propagate_transforms(&mut world).unwrap(), 1);
    world.add_relation::<ChildOf>(leaf, child);
    assert!(!world.remove_relation::<ChildOf>(child, a));
    no_scan(propagate_transforms(&mut world).unwrap());
    world.add_relation::<ChildOf>(child, a);
    scan(propagate_transforms(&mut world).unwrap(), 2);
    assert_eq!(x(&world, leaf), 13.0);
    world.add_relation::<ChildOf>(child, b);
    let stats = propagate_transforms(&mut world).unwrap();
    scan(stats, 2);
    assert_eq!((stats.recomputed, stats.written), (2, 2));
    assert_eq!(x(&world, leaf), 23.0);
    assert!(world.remove_relation::<ChildOf>(child, b));
    scan(propagate_transforms(&mut world).unwrap(), 1);
    assert_eq!(x(&world, leaf), 3.0);
    no_scan(propagate_transforms(&mut world).unwrap());
}

#[test]
fn raw_index_add_remove_and_clear_refresh_canonical_edges() {
    let mut world = setup();
    let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    scan(propagate_transforms(&mut world).unwrap(), 0);
    let relation = world.components().id_of::<ChildOf>().unwrap();
    world
        .relations_mut()
        .index_mut()
        .add(relation, child, parent, true);
    scan(propagate_transforms(&mut world).unwrap(), 1);
    assert_eq!(x(&world, child), 11.0);
    assert!(world
        .relations_mut()
        .index_mut()
        .remove(relation, child, parent));
    scan(propagate_transforms(&mut world).unwrap(), 0);
    assert_eq!(x(&world, child), 1.0);
    world
        .relations_mut()
        .index_mut()
        .add(relation, child, parent, true);
    scan(propagate_transforms(&mut world).unwrap(), 1);
    world.relations_mut().index_mut().clear();
    scan(propagate_transforms(&mut world).unwrap(), 0);
    assert_eq!(x(&world, child), 1.0);
    no_scan(propagate_transforms(&mut world).unwrap());
}

#[test]
fn cloned_registry_and_index_replacements_publish_authoritative_source_diffs() {
    let mut world = setup();
    let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(child, parent);
    scan(propagate_transforms(&mut world).unwrap(), 1);
    let original = world.relations().index().version();
    let registry = world.relations().clone();
    assert_ne!(registry.index().version().identity(), original.identity());
    *world.relations_mut() = registry;
    let stats = propagate_transforms(&mut world).unwrap();
    no_scan(stats); // Equal graph: clone identity is not a source mutation.
    assert_eq!(stats.recomputed, 0);
    let saved = world.relations().index().clone();
    *world.relations_mut().index_mut() = RelationIndex::new();
    scan(propagate_transforms(&mut world).unwrap(), 0);
    assert_eq!(x(&world, child), 1.0);
    *world.relations_mut().index_mut() = saved;
    scan(propagate_transforms(&mut world).unwrap(), 1);
    assert_eq!(x(&world, child), 11.0);
    no_scan(propagate_transforms(&mut world).unwrap());
}

#[test]
fn restore_invalidates_edges_even_when_revision_and_contents_match() {
    let mut world = setup();
    world.register_snapshot_component::<Transform>();
    world.register_snapshot_component::<GlobalTransform>();
    let a = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let b = world.spawn(Transform::from_xyz(20.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(child, a);
    scan(propagate_transforms(&mut world).unwrap(), 1);
    let snapshot = world.snapshot();
    let before = world.relations().index().version();
    world.restore(&snapshot);
    assert_ne!(
        world.relations().index().version().identity(),
        before.identity()
    );
    scan(propagate_transforms(&mut world).unwrap(), 1);
    no_scan(propagate_transforms(&mut world).unwrap());
    world.add_relation::<ChildOf>(child, b);
    scan(propagate_transforms(&mut world).unwrap(), 1);
    assert_eq!(x(&world, child), 21.0);
    world.restore(&snapshot);
    scan(propagate_transforms(&mut world).unwrap(), 1);
    assert_eq!(x(&world, child), 11.0);
    no_scan(propagate_transforms(&mut world).unwrap());
}

#[test]
fn unrelated_edges_never_refresh_transform_sources() {
    struct Link;
    impl Component for Link {}
    let mut world = setup();
    let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(child, parent);
    scan(propagate_transforms(&mut world).unwrap(), 1);
    world.add_relation::<Link>(parent, child);
    let stats = propagate_transforms(&mut world).unwrap();
    no_scan(stats);
    assert_eq!((stats.relation_checks, stats.recomputed), (0, 0));
    assert_eq!(x(&world, child), 11.0);
    no_scan(propagate_transforms(&mut world).unwrap());

    let link = world.components().id_of::<Link>().unwrap();
    world
        .relations_mut()
        .index_mut()
        .remove(link, parent, child);
    let stats = propagate_transforms(&mut world).unwrap();
    no_scan(stats);
    assert_eq!(
        (
            stats.membership_scans,
            stats.relation_checks,
            stats.recomputed,
            stats.written
        ),
        (0, 0, 0, 0)
    );
    no_scan(propagate_transforms(&mut world).unwrap());

    world
        .relations_mut()
        .index_mut()
        .add(link, parent, child, false);
    let stats = propagate_transforms(&mut world).unwrap();
    no_scan(stats);
    assert_eq!(
        (
            stats.membership_scans,
            stats.relation_checks,
            stats.recomputed,
            stats.written
        ),
        (0, 0, 0, 0)
    );
    no_scan(propagate_transforms(&mut world).unwrap());
}

#[test]
fn rejected_cycle_does_not_commit_version_and_repair_retries_cleanly() {
    let mut world = setup();
    let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(child, parent);
    scan(propagate_transforms(&mut world).unwrap(), 1);
    let relation = world.components().id_of::<ChildOf>().unwrap();
    world
        .relations_mut()
        .index_mut()
        .add(relation, parent, child, true);
    for _ in 0..2 {
        assert!(propagate_transforms(&mut world).is_err());
        assert_eq!(x(&world, child), 11.0);
    }
    world
        .relations_mut()
        .index_mut()
        .remove(relation, parent, child);
    let stats = propagate_transforms(&mut world).unwrap();
    scan(stats, 1);
    assert_eq!(stats.relation_checks, 1); // Only the repaired source is revalidated.
    no_scan(propagate_transforms(&mut world).unwrap());
}

#[test]
fn deletion_and_reused_generations_do_not_alias_cached_parent() {
    let mut world = setup();
    world.register_relation::<ChildOf>(RelationKind {
        on_delete_target: CleanupPolicy::Remove,
        ..<ChildOf as Relation>::KIND
    });
    let old = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(child, old);
    scan(propagate_transforms(&mut world).unwrap(), 1);
    world.despawn(old);
    let replacement = world.spawn(Transform::from_xyz(20.0, 0.0, 0.0));
    assert_eq!(old.index(), replacement.index());
    assert_ne!(old.generation(), replacement.generation());
    scan(propagate_transforms(&mut world).unwrap(), 0);
    assert_eq!(x(&world, child), 1.0);
    // Raw publication rejects a stale handle rather than connecting the reused slot.
    let relation = world.components().id_of::<ChildOf>().unwrap();
    world
        .relations_mut()
        .index_mut()
        .add(relation, child, old, true);
    no_scan(propagate_transforms(&mut world).unwrap());
    assert!(world.relation_targets::<ChildOf>(child).is_empty());
    world.add_relation::<ChildOf>(child, replacement);
    scan(propagate_transforms(&mut world).unwrap(), 1);
    assert_eq!(x(&world, child), 21.0);
    no_scan(propagate_transforms(&mut world).unwrap());
}

// Both branches exercise the same ownership transitions: normal return and
// unwinding must leave the current installation, never the outgoing cache.
fn interrupted_propagation(world: &mut World, panics: bool) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        propagate_transforms(world).unwrap()
    }));
    if panics {
        let payload = result.expect_err("callback must panic");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied());
        assert!(
            message.is_some_and(|message| message.starts_with("panic after")),
            "unexpected panic instead of the callback's intentional panic: {message:?}"
        );
    } else {
        result.unwrap();
    }
}

#[test]
fn callback_reinstall_keeps_new_registration_cache_with_and_without_panic() {
    for panics in [false, true] {
        let mut world = setup();
        let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
        let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
        world.add_relation::<ChildOf>(child, parent);
        let mut armed = true;
        world.observe::<GlobalTransform, _>(LifecycleEvent::Insert, move |ctx| {
            if !armed {
                return;
            }
            armed = false;
            assert_eq!(ctx.entity, parent);
            let _removals = uninstall(ctx.world);
            install(ctx.world);
            ctx.world
                .get_mut::<Transform>(parent)
                .unwrap()
                .translation
                .x = 30.0;
            assert!(!panics, "panic after reinstall");
        });

        interrupted_propagation(&mut world, panics);
        // The superseded pass must not continue publishing its child matrix.
        assert!(!world.has::<GlobalTransform>(child));
        let stats = propagate_transforms(&mut world).unwrap();
        // A fresh cache must perform its initial discovery. Restoring the old
        // guard's cache would incorrectly skip this scan, even after a panic.
        scan(stats, 1);
        assert_eq!(stats.membership_scans, 1);
        assert_eq!(x(&world, parent), 30.0);
        assert_eq!(x(&world, child), 31.0);
        no_scan(propagate_transforms(&mut world).unwrap());
    }
}

#[test]
fn callback_uninstall_stops_publication_and_does_not_restore_cache() {
    for panics in [false, true] {
        let mut world = setup();
        let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
        let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
        world.add_relation::<ChildOf>(child, parent);
        let mut armed = true;
        world.observe::<GlobalTransform, _>(LifecycleEvent::Insert, move |ctx| {
            if !armed {
                return;
            }
            armed = false;
            let _removals = uninstall(ctx.world);
            assert!(!panics, "panic after uninstall");
        });
        interrupted_propagation(&mut world, panics);
        assert!(!world.has::<GlobalTransform>(child));
        scan(propagate_transforms(&mut world).unwrap(), 1);
        assert_eq!(x(&world, child), 11.0);
        no_scan(propagate_transforms(&mut world).unwrap());
    }
}

#[test]
fn callback_world_replacement_discards_old_cache() {
    world_replacement(false);
}

#[test]
fn callback_world_replacement_panic_discards_old_cache() {
    world_replacement(true);
}

fn world_replacement(panics: bool) {
    for populated in [false, true] {
        for installed in [false, true] {
            let mut world = setup();
            let old_world = world.id();
            let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
            let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
            world.add_relation::<ChildOf>(child, parent);
            world.observe::<GlobalTransform, _>(LifecycleEvent::Insert, move |ctx| {
                let mut replacement = World::new();
                if installed {
                    install(&mut replacement);
                }
                if populated {
                    let new_parent = replacement.spawn(Transform::from_xyz(40.0, 0.0, 0.0));
                    let new_child = replacement.spawn(Transform::from_xyz(2.0, 0.0, 0.0));
                    assert_eq!((new_parent, new_child), (parent, child));
                    replacement.register_child_of();
                    replacement.add_relation::<ChildOf>(new_child, new_parent);
                }
                *ctx.world = replacement;
                assert!(!panics, "panic after world replacement");
            });

            interrupted_propagation(&mut world, panics);
            assert_ne!(world.id(), old_world);
            assert!(!world.has::<GlobalTransform>(parent));
            assert!(!world.has::<GlobalTransform>(child));
            let stats = propagate_transforms(&mut world).unwrap();
            scan(stats, usize::from(populated));
            assert_eq!(stats.membership_scans, 1);
            if populated {
                assert_eq!(x(&world, parent), 40.0);
                assert_eq!(x(&world, child), 42.0);
            } else {
                assert_eq!((stats.recomputed, stats.written), (0, 0));
            }
            no_scan(propagate_transforms(&mut world).unwrap());
        }
    }
}

#[test]
fn callback_reinstall_during_removal_keeps_new_cache_and_remaining_globals() {
    for panics in [false, true] {
        let mut world = setup();
        let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
        let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
        world.add_relation::<ChildOf>(child, parent);
        propagate_transforms(&mut world).unwrap();
        world.remove::<Transform>(parent);
        world.remove::<Transform>(child);
        let mut armed = true;
        world.observe::<GlobalTransform, _>(LifecycleEvent::Remove, move |ctx| {
            if !armed {
                return;
            }
            armed = false;
            let _removals = uninstall(ctx.world);
            install(ctx.world);
            ctx.world
                .insert(parent, Transform::from_xyz(30.0, 0.0, 0.0));
            ctx.world.insert(child, Transform::from_xyz(2.0, 0.0, 0.0));
            assert!(!panics, "panic after removal reinstall");
        });
        interrupted_propagation(&mut world, panics);
        // Only the in-flight removal may finish; the abandoned pass must not
        // remove the child's global after the new installation has adopted it.
        assert_eq!(x(&world, child), 11.0);
        let stats = propagate_transforms(&mut world).unwrap();
        scan(stats, 1);
        assert_eq!(stats.membership_scans, 1);
        assert_eq!(x(&world, parent), 30.0);
        assert_eq!(x(&world, child), 32.0);
        no_scan(propagate_transforms(&mut world).unwrap());
    }
}

#[test]
fn callback_restore_invalidates_publication_and_retries_with_and_without_panic() {
    for panics in [false, true] {
        let mut world = setup();
        world.register_snapshot_component::<Transform>();
        world.register_snapshot_component::<GlobalTransform>();
        let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
        let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
        world.add_relation::<ChildOf>(child, parent);
        let mut armed = true;
        world.observe::<GlobalTransform, _>(LifecycleEvent::Insert, move |ctx| {
            if !armed {
                return;
            }
            armed = false;
            restore_input(ctx.world);
            assert!(!panics, "panic after restore");
        });
        RESTORE_INPUT.with(|input| *input.borrow_mut() = Some(world.snapshot()));
        world.get_mut::<Transform>(parent).unwrap().translation.x = 100.0;
        interrupted_propagation(&mut world, panics);
        assert!(!world.has::<GlobalTransform>(parent));
        assert!(!world.has::<GlobalTransform>(child));
        let stats = propagate_transforms(&mut world).unwrap();
        scan(stats, 1);
        assert_eq!(stats.membership_scans, 1);
        assert_eq!(x(&world, parent), 10.0);
        assert_eq!(x(&world, child), 11.0);
        no_scan(propagate_transforms(&mut world).unwrap());
    }
}

#[test]
fn callback_restore_during_removal_abandons_remaining_old_removals() {
    for panics in [false, true] {
        let mut world = setup();
        world.register_snapshot_component::<Transform>();
        world.register_snapshot_component::<GlobalTransform>();
        let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
        let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
        world.add_relation::<ChildOf>(child, parent);
        propagate_transforms(&mut world).unwrap();
        let mut armed = true;
        world.observe::<GlobalTransform, _>(LifecycleEvent::Remove, move |ctx| {
            if !armed {
                return;
            }
            armed = false;
            restore_input(ctx.world);
            assert!(!panics, "panic after removal restore");
        });
        RESTORE_INPUT.with(|input| *input.borrow_mut() = Some(world.snapshot()));
        world.remove::<Transform>(parent);
        world.remove::<Transform>(child);
        interrupted_propagation(&mut world, panics);
        assert_eq!(x(&world, child), 11.0);
        let stats = propagate_transforms(&mut world).unwrap();
        scan(stats, 1);
        assert_eq!(stats.membership_scans, 1);
        assert_eq!(x(&world, parent), 10.0);
        assert_eq!(x(&world, child), 11.0);
        no_scan(propagate_transforms(&mut world).unwrap());
    }
}

#[test]
fn relation_edit_during_publication_is_discovered_on_the_next_pass() {
    let mut world = setup();
    let a = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let b = world.spawn(Transform::from_xyz(20.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(child, a);
    let mut armed = true;
    world.observe::<GlobalTransform, _>(LifecycleEvent::Insert, move |ctx| {
        if armed {
            armed = false;
            ctx.world.add_relation::<ChildOf>(child, b);
        }
    });
    scan(propagate_transforms(&mut world).unwrap(), 1);
    scan(propagate_transforms(&mut world).unwrap(), 1);
    assert_eq!(x(&world, child), 21.0);
    no_scan(propagate_transforms(&mut world).unwrap());
}

#[test]
fn large_graph_clean_and_single_leaf_preparation_have_size_independent_counts() {
    for size in [64, 8192] {
        let mut world = setup();
        let root = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
        let mut leaf = root;
        for _ in 0..size {
            leaf = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
            world.add_relation::<ChildOf>(leaf, root);
        }
        let initial = propagate_transforms(&mut world).unwrap();
        assert_eq!(initial.membership_baseline_entities, size + 1);
        assert_eq!(initial.relation_checks, size + 1);
        // Consume our initial GlobalTransform publications. Their changed
        // chunks/signals may conservatively be inspected on this next pass.
        let follow_up = propagate_transforms(&mut world).unwrap();
        no_scan(follow_up);
        assert_eq!(follow_up.dirty_nodes_examined, 0);
        assert_eq!(follow_up.publications_examined, 0);
        for _ in 0..4 {
            let clean = propagate_transforms(&mut world).unwrap();
            no_scan(clean);
            assert_eq!(clean.membership_scans, 0);
            assert_eq!(clean.membership_baseline_entities, 0);
            assert_eq!(clean.membership_entities, 0);
            assert_eq!(clean.relation_journal_entries, 0);
            assert!(clean.relation_journal_lookups <= 12);
            assert_eq!(clean.relation_source_candidates, 0);
            assert_eq!(clean.relation_dependents_examined, 0);
            assert_eq!(clean.topology_checks, 0);
            assert_eq!(clean.dirty_nodes_examined, 0);
            assert_eq!(clean.ancestor_checks, 0);
            assert_eq!(clean.change_rows_scanned, 0);
            assert_eq!(clean.local_reads, 0);
            assert_eq!(clean.recomputed, 0);
            assert_eq!(clean.publications_examined, 0);
        }
        for value in [2.0, 3.0] {
            world.get_mut::<Transform>(leaf).unwrap().translation.x = value;
        }
        let dirty = propagate_transforms(&mut world).unwrap();
        no_scan(dirty);
        assert_eq!(dirty.membership_entities, 0);
        assert_eq!(dirty.relation_source_candidates, 0);
        assert_eq!(dirty.topology_checks, 0);
        assert_eq!(dirty.dirty_nodes_examined, 1);
        assert_eq!(dirty.ancestor_checks, 1);
        assert_eq!(dirty.local_reads, 1);
        assert_eq!(dirty.recomputed, 1);
        assert_eq!(dirty.publications_examined, 1);
        // Tick filtering is chunk-conservative, not an exact row journal.
        // Even at 8192 siblings it must not scan the entire local column.
        assert!(dirty.change_rows_scanned > 0);
        if size == 8192 {
            assert!(dirty.change_rows_scanned < size / 2, "{dirty:?}");
        }
        assert_eq!(x(&world, leaf), 13.0);
    }
}

#[test]
fn membership_deltas_check_only_the_member_and_its_dependents() {
    let mut world = setup();
    let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let leaves: Vec<_> = (0..2048)
        .map(|_| {
            let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
            world.add_relation::<ChildOf>(child, parent);
            child
        })
        .collect();
    propagate_transforms(&mut world).unwrap();
    propagate_transforms(&mut world).unwrap();
    let leaf = leaves[0];
    world.remove::<Transform>(leaf);
    let removed = propagate_transforms(&mut world).unwrap();
    assert_eq!(removed.membership_scans, 0);
    assert_eq!(removed.membership_entities, 1);
    assert_eq!(removed.relation_source_candidates, 1);
    assert_eq!(removed.relation_dependents_examined, 0);
    assert_eq!(removed.relation_checks, 0);
    assert_eq!(removed.removed, 1);
    assert_eq!(removed.recomputed, 0);
    assert!(!world.has::<GlobalTransform>(leaf));

    world.insert(leaf, Transform::from_xyz(2.0, 0.0, 0.0));
    let added = propagate_transforms(&mut world).unwrap();
    assert_eq!(added.membership_scans, 0);
    assert_eq!(added.membership_entities, 1);
    assert_eq!(added.relation_checks, 1);
    assert_eq!(added.relation_source_candidates, 1);
    assert_eq!(added.recomputed, 1);
    assert_eq!(x(&world, leaf), 12.0);

    world.remove::<Transform>(parent);
    let boundary = propagate_transforms(&mut world).unwrap();
    assert_eq!(boundary.membership_scans, 0);
    assert_eq!(boundary.membership_entities, 1);
    assert_eq!(boundary.relation_source_candidates, leaves.len() + 1);
    assert_eq!(boundary.relation_dependents_examined, 2 * leaves.len());
    assert_eq!(boundary.relation_checks, leaves.len());
    assert_eq!(boundary.recomputed, leaves.len());
    assert_eq!(x(&world, leaf), 2.0);
    assert_eq!(x(&world, leaves[1]), 1.0);

    world.insert(parent, Transform::from_xyz(20.0, 0.0, 0.0));
    let restored_member = propagate_transforms(&mut world).unwrap();
    assert_eq!(restored_member.membership_scans, 0);
    assert_eq!(restored_member.membership_entities, 1);
    assert_eq!(restored_member.relation_dependents_examined, leaves.len());
    assert_eq!(restored_member.relation_checks, leaves.len() + 1);
    assert_eq!(restored_member.recomputed, leaves.len() + 1);
    assert_eq!(x(&world, leaf), 22.0);
    no_scan(propagate_transforms(&mut world).unwrap());
}

#[test]
fn relevant_journal_gap_refreshes_parents_without_rescanning_membership() {
    let mut world = setup();
    world.register_snapshot_component::<Transform>();
    world.register_snapshot_component::<GlobalTransform>();
    let a = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let b = world.spawn(Transform::from_xyz(20.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(child, a);
    for _ in 0..256 {
        world.spawn(Transform::IDENTITY);
    }
    propagate_transforms(&mut world).unwrap();
    let snapshot = world.snapshot();
    let before = world.relations().index().change_cursor();
    for i in 0..=RELATION_CHANGE_CAPACITY {
        world.add_relation::<ChildOf>(child, if i % 2 == 0 { b } else { a });
    }
    let kind = world.components().id_of::<ChildOf>().unwrap();
    assert!(world
        .relations()
        .index()
        .changed_sources_since(kind, before)
        .is_none());
    let gap = propagate_transforms(&mut world).unwrap();
    assert_eq!(gap.relation_fallbacks, 1);
    assert_eq!(gap.membership_scans, 0);
    assert_eq!(gap.membership_entities, 0);
    assert_eq!(gap.relation_checks, 259);
    assert_eq!(gap.relation_source_candidates, 259);
    assert_eq!(gap.recomputed, 1);
    assert_eq!(x(&world, child), 21.0);
    let clean = propagate_transforms(&mut world).unwrap();
    no_scan(clean);
    assert_eq!(clean.relation_fallbacks, 0);

    world.restore(&snapshot);
    let restored = propagate_transforms(&mut world).unwrap();
    assert_eq!(restored.membership_scans, 1);
    assert_eq!(restored.membership_baseline_entities, 259);
    assert_eq!(restored.relation_fallbacks, 0);
    assert_eq!(x(&world, child), 11.0);
    no_scan(propagate_transforms(&mut world).unwrap());
}

#[test]
fn oversized_single_batch_clear_cannot_masquerade_as_a_partial_delta() {
    let mut world = setup();
    let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let children: Vec<_> = (0..RELATION_CHANGE_CAPACITY + 1)
        .map(|_| {
            let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
            world.add_relation::<ChildOf>(child, parent);
            child
        })
        .collect();
    propagate_transforms(&mut world).unwrap();
    world.relations_mut().index_mut().clear();
    let cleared = propagate_transforms(&mut world).unwrap();
    assert_eq!(cleared.relation_fallbacks, 1);
    assert_eq!(cleared.membership_scans, 0);
    assert_eq!(cleared.relation_checks, children.len() + 1);
    assert_eq!(cleared.recomputed, children.len());
    for child in children {
        assert_eq!(x(&world, child), 1.0);
    }
    no_scan(propagate_transforms(&mut world).unwrap());
}

#[test]
fn unrelated_journal_overflow_does_not_evict_child_of_history() {
    struct Link;
    impl Component for Link {}
    let mut world = setup();
    let a = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let b = world.spawn(Transform::from_xyz(20.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(child, a);
    world.register_relation::<Link>(RelationKind {
        exclusive: true,
        ..RelationKind::default()
    });
    propagate_transforms(&mut world).unwrap();
    for i in 0..=RELATION_CHANGE_CAPACITY {
        world.add_relation::<Link>(child, if i % 2 == 0 { b } else { a });
    }
    let unrelated = propagate_transforms(&mut world).unwrap();
    no_scan(unrelated);
    assert_eq!(unrelated.relation_fallbacks, 0);
    assert_eq!(unrelated.relation_source_candidates, 0);
    world.add_relation::<ChildOf>(child, b);
    for i in 0..=RELATION_CHANGE_CAPACITY {
        world.add_relation::<Link>(a, if i % 2 == 0 { b } else { child });
    }
    let relevant = propagate_transforms(&mut world).unwrap();
    assert_eq!(relevant.relation_fallbacks, 0);
    assert_eq!(relevant.relation_journal_entries, 1);
    assert_eq!(relevant.relation_checks, 1);
    assert_eq!(relevant.relation_source_candidates, 1);
    assert_eq!(relevant.recomputed, 1);
    assert_eq!(x(&world, child), 21.0);
}

#[test]
fn final_batch_edge_reversal_is_validated_before_detaching_and_attaching() {
    for raw in [false, true] {
        let mut world = setup();
        let a = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
        let b = world.spawn(Transform::from_xyz(2.0, 0.0, 0.0));
        let leaf = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
        world.add_relation::<ChildOf>(b, a);
        world.add_relation::<ChildOf>(leaf, b);
        propagate_transforms(&mut world).unwrap();
        if raw {
            let kind = world.components().id_of::<ChildOf>().unwrap();
            let mut relations = world.relations_mut();
            relations.index_mut().add(kind, a, b, true);
            relations.index_mut().remove(kind, b, a);
        } else {
            // The intermediate graph is cyclic; the final batch is a forest.
            world.add_relation::<ChildOf>(a, b);
            world.remove_relation::<ChildOf>(b, a);
        }
        let reversed = propagate_transforms(&mut world).unwrap();
        assert_eq!(reversed.membership_scans, 0);
        assert_eq!(reversed.relation_checks, 2);
        assert_eq!(reversed.relation_source_candidates, 2);
        assert_eq!(reversed.recomputed, 3);
        assert_eq!(x(&world, a), 12.0);
        assert_eq!(x(&world, b), 2.0);
        assert_eq!(x(&world, leaf), 3.0);
        no_scan(propagate_transforms(&mut world).unwrap());
    }
}

#[test]
fn callback_member_changes_stop_stale_publication_and_retry_the_final_membership() {
    for action in 0..4 {
        for panics in [false, true] {
            let mut world = setup();
            let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
            let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
            world.add_relation::<ChildOf>(child, parent);
            let mut armed = true;
            world.observe::<GlobalTransform, _>(LifecycleEvent::Insert, move |ctx| {
                if !armed {
                    return;
                }
                armed = false;
                assert_eq!(ctx.entity, parent);
                match action {
                    0 => {
                        ctx.world.remove::<Transform>(child);
                    }
                    1 => {
                        ctx.world.despawn(child);
                    }
                    2 => {
                        ctx.world.remove::<Transform>(child);
                        ctx.world.insert(child, Transform::from_xyz(5.0, 0.0, 0.0));
                    }
                    3 => {
                        ctx.world.remove::<Transform>(parent);
                    }
                    _ => unreachable!(),
                }
                assert!(!panics, "panic after member mutation");
            });
            interrupted_propagation(&mut world, panics);
            assert!(!world.has::<GlobalTransform>(child));
            let retry = propagate_transforms(&mut world).unwrap();
            if !panics {
                assert_eq!(retry.membership_scans, 0);
                assert_eq!(retry.membership_entities, 1);
            }
            match action {
                0 | 1 => assert!(!world.has::<GlobalTransform>(child)),
                2 => assert_eq!(x(&world, child), 15.0),
                3 => {
                    assert!(!world.has::<GlobalTransform>(parent));
                    assert_eq!(x(&world, child), 1.0);
                }
                _ => unreachable!(),
            }
            no_scan(propagate_transforms(&mut world).unwrap());
        }
    }
}

#[test]
fn unrelated_callback_relation_edit_retains_unfinished_publications() {
    struct Link;
    impl Component for Link {}
    let mut world = setup();
    let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(child, parent);
    let mut armed = true;
    world.observe::<GlobalTransform, _>(LifecycleEvent::Insert, move |ctx| {
        if armed {
            armed = false;
            ctx.world.add_relation::<Link>(parent, child);
        }
    });
    let interrupted = propagate_transforms(&mut world).unwrap();
    assert_eq!(interrupted.written, 1);
    assert!(!world.has::<GlobalTransform>(child));
    let retry = propagate_transforms(&mut world).unwrap();
    no_scan(retry);
    assert_eq!(retry.membership_scans, 0);
    assert_eq!(retry.recomputed, 0);
    assert_eq!(retry.publications_examined, 1);
    assert_eq!(retry.written, 1);
    assert_eq!(x(&world, child), 11.0);
}

#[test]
fn reparent_before_cascade_preserves_moved_subtree_and_restore_rebuilds_it() {
    let mut world = setup();
    world.register_snapshot_component::<Transform>();
    world.register_snapshot_component::<GlobalTransform>();
    let safe = world.spawn(Transform::from_xyz(100.0, 0.0, 0.0));
    let doomed = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let moved = world.spawn(Transform::from_xyz(2.0, 0.0, 0.0));
    let leaf = world.spawn(Transform::from_xyz(3.0, 0.0, 0.0));
    let deleted = world.spawn(Transform::from_xyz(5.0, 0.0, 0.0));
    let deleted_leaf = world.spawn(Transform::from_xyz(7.0, 0.0, 0.0));
    for (child, parent) in [
        (moved, doomed),
        (leaf, moved),
        (deleted, doomed),
        (deleted_leaf, deleted),
    ] {
        world.add_relation::<ChildOf>(child, parent);
    }
    let unrelated: Vec<_> = (0..2048)
        .map(|_| world.spawn(Transform::IDENTITY))
        .collect();
    propagate_transforms(&mut world).unwrap();
    let snapshot = world.snapshot();
    world.add_relation::<ChildOf>(moved, safe);
    world.despawn(doomed);
    for entity in [doomed, deleted, deleted_leaf] {
        assert!(!world.contains(entity));
    }
    let delta = propagate_transforms(&mut world).unwrap();
    assert_eq!(delta.membership_scans, 0);
    assert_eq!(delta.membership_entities, 3);
    assert_eq!(delta.relation_checks, 1);
    assert_eq!(delta.recomputed, 2);
    assert_eq!(x(&world, moved), 102.0);
    assert_eq!(x(&world, leaf), 105.0);
    for entity in &unrelated {
        assert_eq!(
            world.get::<GlobalTransform>(*entity),
            Some(&GlobalTransform::IDENTITY)
        );
    }
    no_scan(propagate_transforms(&mut world).unwrap());
    world.restore(&snapshot);
    let restored = propagate_transforms(&mut world).unwrap();
    assert_eq!(restored.membership_scans, 1);
    assert_eq!(restored.membership_baseline_entities, unrelated.len() + 6);
    assert_eq!(x(&world, moved), 12.0);
    assert_eq!(x(&world, leaf), 15.0);
    assert_eq!(x(&world, deleted_leaf), 22.0);
    no_scan(propagate_transforms(&mut world).unwrap());
}

// Independent oracle: read authoritative locals/edges each time, walk to the
// boundary, and compose root-to-leaf without consulting the incremental cache.
fn assert_full_recompute(world: &World, entities: &[Entity], seed: u64, step: usize) {
    for &entity in entities {
        if !world.has::<Transform>(entity) {
            assert!(
                !world.has::<GlobalTransform>(entity),
                "seed {seed}, step {step}, {entity:?}"
            );
            continue;
        }
        let mut chain = Vec::new();
        let mut current = entity;
        while let Some(local) = world.get::<Transform>(current) {
            assert!(chain.len() < entities.len(), "oracle cycle");
            chain.push(*local);
            let targets = world.relation_targets::<ChildOf>(current);
            assert!(targets.len() <= 1);
            let Some(&parent) = targets.first() else {
                break;
            };
            current = parent;
        }
        let mut expected = GlobalTransform::from_transform(&chain.pop().unwrap());
        while let Some(local) = chain.pop() {
            expected = expected.mul_transform(&local);
        }
        assert_eq!(
            world.get::<GlobalTransform>(entity),
            Some(&expected),
            "seed {seed}, step {step}, {entity:?}"
        );
    }
}

#[test]
fn restore_then_remove_before_first_pass_cleans_restored_global() {
    let mut world = setup();
    world.register_snapshot_component::<Transform>();
    world.register_snapshot_component::<GlobalTransform>();
    let entity = world.spawn(Transform::from_xyz(4.0, 0.0, 0.0));
    propagate_transforms(&mut world).unwrap();
    let checkpoint = world.snapshot();
    world.remove::<Transform>(entity);
    propagate_transforms(&mut world).unwrap();
    assert!(!world.has::<GlobalTransform>(entity));

    world.restore(&checkpoint);
    world.remove::<Transform>(entity);
    let restored = propagate_transforms(&mut world).unwrap();
    assert_eq!(restored.membership_scans, 1);
    assert_eq!(restored.removed, 1);
    assert!(!world.has::<GlobalTransform>(entity));
    no_scan(propagate_transforms(&mut world).unwrap());
}

#[test]
fn restore_discards_old_membership_events_but_keeps_new_timeline_removals() {
    let mut world = setup();
    world.register_snapshot_component::<Transform>();
    world.register_snapshot_component::<GlobalTransform>();
    let standalone = world.spawn(GlobalTransform::from_transform(&Transform::from_xyz(
        8.0, 0.0, 0.0,
    )));
    let member = world.spawn(Transform::from_xyz(4.0, 0.0, 0.0));
    propagate_transforms(&mut world).unwrap();
    let checkpoint = world.snapshot();
    // This adoption/removal belongs only to the abandoned timeline. Its
    // membership signal must not delete the standalone global in checkpoint.
    world.insert(standalone, Transform::IDENTITY);
    world.remove::<Transform>(standalone);
    world.restore(&checkpoint);
    // A global event may be the first observer event on the new timeline.
    world.remove::<GlobalTransform>(member);
    world.insert(member, GlobalTransform::IDENTITY);
    world.remove::<Transform>(member);
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!(stats.removed, 1);
    assert_eq!(x(&world, standalone), 8.0);
    assert!(!world.has::<GlobalTransform>(member));

    // Repeated restores before propagation must also discard the intermediate
    // timeline's removal and leave an authoritative member to be repaired.
    world.restore(&checkpoint);
    world.remove::<Transform>(member);
    world.restore(&checkpoint);
    propagate_transforms(&mut world).unwrap();
    assert_eq!(x(&world, member), 4.0);
    assert_eq!(x(&world, standalone), 8.0);
    no_scan(propagate_transforms(&mut world).unwrap());
}

#[test]
fn local_added_and_removed_before_adoption_cleans_only_its_derived_global() {
    let mut world = setup();
    let standalone = world.spawn(GlobalTransform::IDENTITY);
    propagate_transforms(&mut world).unwrap();
    let transient = world.spawn((Transform::IDENTITY, GlobalTransform::IDENTITY));
    world.remove::<Transform>(transient);
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!(stats.membership_scans, 0);
    assert_eq!(stats.membership_entities, 1);
    assert_eq!(stats.removed, 1);
    assert!(!world.has::<GlobalTransform>(transient));
    assert!(world.has::<GlobalTransform>(standalone));
}

#[test]
fn mixed_batches_match_full_recompute_across_membership_reparent_and_restore() {
    use prism_math::{Quat, Vec3};
    struct Link;
    impl Component for Link {}
    for seed in [1, 0x5e37_2a91, 0x764c_aa83_5bd1_8ee7, u64::MAX] {
        let mut world = setup();
        world.register_snapshot_component::<Transform>();
        world.register_snapshot_component::<GlobalTransform>();
        // Keep survivors on deletion here; the separate cascade test covers
        // Delete policy. Slot order supplies a DAG independently of handles.
        world.register_relation::<ChildOf>(RelationKind {
            on_delete_target: CleanupPolicy::Remove,
            ..<ChildOf as Relation>::KIND
        });
        let mut entities: Vec<_> = (0..32).map(|_| world.spawn(Transform::IDENTITY)).collect();
        propagate_transforms(&mut world).unwrap();
        let mut checkpoint = (world.snapshot(), entities.clone());
        let mut random = seed;
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        for step in 0..160 {
            let mut save = false;
            for _ in 0..1 + next() % 5 {
                let slot = next() as usize % entities.len();
                let entity = entities[slot];
                let parent = entities[if slot == 0 { 0 } else { next() as usize % slot }];
                match next() % 14 {
                    0..=2 => {
                        let value = next() as u32;
                        world.insert(
                            entity,
                            Transform {
                                translation: Vec3::new(
                                    (value % 17) as f32 - 8.0,
                                    (value % 7) as f32,
                                    -1.0,
                                ),
                                rotation: Quat::from_rotation_z((value % 5) as f32 * 0.125),
                                scale: Vec3::new(0.5 + (value % 3) as f32 * 0.5, 1.0, 1.25),
                            },
                        );
                    }
                    3 => {
                        world.remove::<Transform>(entity);
                    }
                    4 if slot != 0 => {
                        world.add_relation::<ChildOf>(entity, parent);
                    }
                    5 => {
                        for target in world.relation_targets::<ChildOf>(entity).to_vec() {
                            world.remove_relation::<ChildOf>(entity, target);
                        }
                    }
                    6 if slot != 0 => {
                        let kind = world.components().id_of::<ChildOf>().unwrap();
                        let mut raw = world.relations_mut();
                        raw.index_mut().add(kind, entity, parent, true);
                    }
                    7 => {
                        world.despawn(entity);
                        entities[slot] =
                            world.spawn(Transform::from_xyz((next() % 9) as f32, 1.0, 0.0));
                    }
                    8 => {
                        world.remove::<GlobalTransform>(entity);
                    }
                    9 if world.has::<Transform>(entity) => {
                        world.insert(
                            entity,
                            GlobalTransform::from_transform(&Transform::from_xyz(-900.0, 0.0, 0.0)),
                        );
                    }
                    10 => save = true,
                    11 => {
                        world.restore(&checkpoint.0);
                        entities.clone_from(&checkpoint.1);
                    }
                    12 => {
                        let cloned = world.relations().clone();
                        *world.relations_mut() = cloned;
                    }
                    _ => {
                        world.add_relation::<Link>(entity, parent);
                    }
                }
            }
            propagate_transforms(&mut world).unwrap();
            assert_full_recompute(&world, &entities, seed, step);
            if save {
                checkpoint = (world.snapshot(), entities.clone());
            }
            let settled = propagate_transforms(&mut world).unwrap();
            no_scan(settled);
            assert_eq!(
                (
                    settled.membership_scans,
                    settled.local_reads,
                    settled.recomputed,
                    settled.written
                ),
                (0, 0, 0, 0)
            );
            let clean = propagate_transforms(&mut world).unwrap();
            assert_eq!(clean.change_rows_scanned, 0);
        }
    }
}
