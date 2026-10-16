//! Work-count and semantic regressions for the index-version hierarchy cache.
#![cfg(feature = "ecs")]

use prism_ecs::component::Component;
use prism_ecs::entity::Entity;
use prism_ecs::hierarchy::ChildOf;
use prism_ecs::observer::LifecycleEvent;
use prism_ecs::relation::{CleanupPolicy, Relation, RelationIndex, RelationKind};
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
fn cloned_registry_and_index_replacements_cannot_reuse_cache_identity() {
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
    scan(stats, 1);
    assert_eq!(stats.relation_checks, 2); // Version replacement still revalidates sources.
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
fn unrelated_edges_refresh_once_without_rebuilding_transform_topology() {
    struct Link;
    impl Component for Link {}
    let mut world = setup();
    let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(child, parent);
    scan(propagate_transforms(&mut world).unwrap(), 1);
    world.add_relation::<Link>(parent, child);
    let stats = propagate_transforms(&mut world).unwrap();
    scan(stats, 2); // Counter measures all visited edges, not only ChildOf.
    assert_eq!((stats.relation_checks, stats.recomputed), (2, 0));
    assert_eq!(x(&world, child), 11.0);
    no_scan(propagate_transforms(&mut world).unwrap());

    let link = world.components().id_of::<Link>().unwrap();
    world
        .relations_mut()
        .index_mut()
        .remove(link, parent, child);
    let stats = propagate_transforms(&mut world).unwrap();
    scan(stats, 1);
    assert_eq!(
        (
            stats.membership_scans,
            stats.relation_checks,
            stats.recomputed,
            stats.written
        ),
        (0, 2, 0, 0)
    );
    no_scan(propagate_transforms(&mut world).unwrap());

    world
        .relations_mut()
        .index_mut()
        .add(link, parent, child, false);
    let stats = propagate_transforms(&mut world).unwrap();
    scan(stats, 2);
    assert_eq!(
        (
            stats.membership_scans,
            stats.relation_checks,
            stats.recomputed,
            stats.written
        ),
        (0, 2, 0, 0)
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
    assert_eq!(stats.relation_checks, 2); // The repaired version is revalidated safely.
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
    scan(propagate_transforms(&mut world).unwrap(), 0);
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
