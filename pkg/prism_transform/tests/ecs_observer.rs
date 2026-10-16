//! Real ECS propagation, change-bus delivery and spatial integration contracts.
#![cfg(feature = "ecs")]

extern crate alloc;

use alloc::sync::Arc;
use prism_ecs::entity::Entity;
use prism_ecs::hierarchy::ChildOf;
use prism_ecs::observer::LifecycleEvent;
use prism_ecs::resource::Resource;
use prism_ecs::schedule::{IntoSystemConfigs, Schedule};
use prism_ecs::world::World;
use prism_math::{Aabb3, Vec3};
use prism_transform::ecs::{
    drain_spatial_updates, install, propagate_transforms, set_spatial_margin, transform_system,
    uninstall, LocalBounds, SpatialUpdate, TransformEcsError,
};
use prism_transform::hierarchy::HierarchyError;
use prism_transform::{GlobalTransform, Transform};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default, Debug, PartialEq)]
struct Deliveries(Vec<(u8, Vec<Entity>)>);
impl Resource for Deliveries {}

fn bounds() -> LocalBounds {
    LocalBounds(Aabb3::new(Vec3::ZERO, Vec3::ONE))
}

fn setup() -> World {
    let mut world = World::new();
    install(&mut world);
    world.insert_resource(Deliveries::default());
    for label in [0, 1] {
        world.observe_changed::<GlobalTransform, _>(move |ctx| {
            let entities = ctx.changed().to_vec();
            ctx.world
                .resource_mut::<Deliveries>()
                .0
                .push((label, entities));
        });
    }
    world
}

fn flush(world: &mut World) -> Vec<SpatialUpdate> {
    propagate_transforms(world).unwrap();
    world.flush_changed_observers();
    drain_spatial_updates(world)
}

fn clear(world: &mut World) {
    world.resource_mut::<Deliveries>().0.clear();
}

fn position(world: &World, entity: Entity) -> Vec3 {
    world.get::<GlobalTransform>(entity).unwrap().translation()
}

#[test]
fn repeated_local_writes_publish_once_and_propagate_descendants() {
    let mut world = setup();
    let parent = world.spawn((Transform::from_xyz(10.0, 0.0, 0.0), bounds()));
    let child = world.spawn((Transform::from_xyz(1.0, 0.0, 0.0), bounds()));
    let clean = world.spawn((Transform::IDENTITY, bounds()));
    world.add_relation::<ChildOf>(child, parent);
    assert_eq!(flush(&mut world).len(), 3);
    clear(&mut world);

    for x in [12.0, 15.0, 20.0] {
        world.get_mut::<Transform>(parent).unwrap().translation.x = x;
    }
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!((stats.recomputed, stats.written), (2, 2));
    assert_eq!((stats.membership_scans, stats.relation_checks), (0, 0));
    assert_eq!(position(&world, parent), Vec3::new(20.0, 0.0, 0.0));
    assert_eq!(position(&world, child), Vec3::new(21.0, 0.0, 0.0));
    assert_eq!(position(&world, clean), Vec3::ZERO);
    // In-place global writes notify through Changed, not Insert lifecycle.
    assert!(drain_spatial_updates(&mut world).is_empty());
    world.flush_changed_observers();
    let edits = drain_spatial_updates(&mut world);
    assert_eq!(edits.len(), 2);
    assert!(matches!(edits[0], SpatialUpdate::Update { entity, new, .. }
        if entity == parent && new.min.x == 20.0));
    assert!(matches!(edits[1], SpatialUpdate::Update { entity, new, .. }
        if entity == child && new.min.x == 21.0));
    assert_eq!(
        world.resource::<Deliveries>().0,
        vec![(0, vec![parent, child]), (1, vec![parent, child])]
    );
}

#[test]
fn static_and_net_noop_frames_do_not_write_or_notify() {
    let mut world = setup();
    let entity = world.spawn((Transform::IDENTITY, bounds()));
    assert_eq!(flush(&mut world).len(), 1); // Identity must still insert.
    clear(&mut world);
    let ticks = world.get_ticks::<GlobalTransform>(entity).unwrap();
    world.get_mut::<Transform>(entity).unwrap().translation.x = 10.0;
    world.get_mut::<Transform>(entity).unwrap().translation.x = 0.0;
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!((stats.recomputed, stats.written), (0, 0));
    assert_eq!(
        world.get_ticks::<GlobalTransform>(entity).unwrap().changed,
        ticks.changed
    );
    world.flush_changed_observers();
    assert!(drain_spatial_updates(&mut world).is_empty());
    assert!(world.resource::<Deliveries>().0.is_empty());
    assert_eq!(propagate_transforms(&mut world).unwrap().recomputed, 0);
}

#[test]
fn multiple_propagations_coalesce_on_ecs_bus_and_spatial_drain() {
    let mut world = setup();
    let entity = world.spawn((Transform::IDENTITY, bounds()));
    flush(&mut world);
    clear(&mut world);
    for x in [1.0, 2.0, 3.0] {
        world.get_mut::<Transform>(entity).unwrap().translation.x = x;
        propagate_transforms(&mut world).unwrap();
    }
    world.flush_changed_observers();
    assert_eq!(
        world.resource::<Deliveries>().0,
        vec![(0, vec![entity]), (1, vec![entity])]
    );
    assert!(matches!(drain_spatial_updates(&mut world)[..],
        [SpatialUpdate::Update { old, new, .. }] if old.min.x == 0.0 && new.min.x == 3.0));
    // Even across separate bus flushes, spatial commands retain the last
    // drained baseline and suppress a move out and back before draining.
    for x in [8.0, 3.0] {
        world.get_mut::<Transform>(entity).unwrap().translation.x = x;
        propagate_transforms(&mut world).unwrap();
        world.flush_changed_observers();
    }
    assert!(drain_spatial_updates(&mut world).is_empty());
}

#[test]
fn bounds_only_edits_and_global_only_entities_use_the_bus() {
    let mut world = setup();
    let entity = world.spawn((GlobalTransform::IDENTITY, bounds()));
    assert_eq!(flush(&mut world).len(), 1);
    clear(&mut world);
    world.get_mut::<LocalBounds>(entity).unwrap().0.max.x = 4.0;
    world.flush_changed_observers();
    assert!(matches!(drain_spatial_updates(&mut world)[..],
        [SpatialUpdate::Update { new, .. }] if new.max.x == 4.0));
    assert!(world.resource::<Deliveries>().0.is_empty());
    *world.get_mut::<GlobalTransform>(entity).unwrap() =
        GlobalTransform::from_transform(&Transform::from_xyz(5.0, 0.0, 0.0));
    world.flush_changed_observers();
    assert!(matches!(drain_spatial_updates(&mut world)[..],
        [SpatialUpdate::Update { new, .. }] if new.min.x == 5.0));
}

#[test]
fn bounds_global_local_removals_and_despawn_remove_proxies() {
    let mut world = setup();
    let a = world.spawn((Transform::IDENTITY, bounds()));
    let b = world.spawn((Transform::IDENTITY, bounds()));
    let c = world.spawn((Transform::IDENTITY, bounds()));
    let d = world.spawn((GlobalTransform::IDENTITY, bounds()));
    flush(&mut world);
    world.remove::<LocalBounds>(a);
    world.remove::<Transform>(b);
    world.despawn(c);
    world.remove::<GlobalTransform>(d);
    let updates = flush(&mut world);
    assert_eq!(
        updates
            .iter()
            .map(SpatialUpdate::entity)
            .collect::<Vec<_>>(),
        vec![a, b, c, d]
    );
    assert!(updates
        .iter()
        .all(|u| matches!(u, SpatialUpdate::Remove { .. })));
    assert!(!world.has::<GlobalTransform>(b));
    assert!(flush(&mut world).is_empty());
    world.insert(a, bounds());
    assert!(matches!(flush(&mut world)[..], [SpatialUpdate::Insert { entity, .. }] if entity == a));
}

#[test]
fn remove_reinsert_and_entity_reuse_preserve_generational_identity() {
    let mut world = setup();
    let old = world.spawn((Transform::IDENTITY, bounds()));
    flush(&mut world);
    world.remove::<LocalBounds>(old);
    world.insert(old, bounds());
    assert!(flush(&mut world).is_empty());
    world.despawn(old);
    let new = world.spawn((Transform::from_xyz(20.0, 0.0, 0.0), bounds()));
    assert_eq!(old.index(), new.index());
    assert_ne!(old.generation(), new.generation());
    assert!(matches!(flush(&mut world)[..],
        [SpatialUpdate::Remove { entity: a, .. }, SpatialUpdate::Insert { entity: b, bounds }]
        if a == old && b == new && bounds.min.x == 20.0));
}

#[test]
fn reparent_detach_and_parent_local_removal_dirty_only_affected_subtrees() {
    let mut world = setup();
    let a = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let b = world.spawn(Transform::from_xyz(20.0, 0.0, 0.0));
    let child = world.spawn((Transform::from_xyz(1.0, 0.0, 0.0), bounds()));
    world.add_relation::<ChildOf>(child, a);
    flush(&mut world);
    world.add_relation::<ChildOf>(child, b);
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!((stats.recomputed, stats.written), (1, 1));
    assert_eq!(position(&world, child).x, 21.0);
    world.remove_relation::<ChildOf>(child, b);
    propagate_transforms(&mut world).unwrap();
    assert_eq!(position(&world, child).x, 1.0);
    world.add_relation::<ChildOf>(child, a);
    propagate_transforms(&mut world).unwrap();
    world.remove::<Transform>(a);
    propagate_transforms(&mut world).unwrap();
    assert_eq!(position(&world, child).x, 1.0);
    assert!(!world.has::<GlobalTransform>(a));
}

#[test]
fn cycle_is_atomic_and_recoverable() {
    let mut world = setup();
    let a = world.spawn((Transform::IDENTITY, bounds()));
    let b = world.spawn((Transform::from_xyz(1.0, 0.0, 0.0), bounds()));
    world.add_relation::<ChildOf>(b, a);
    flush(&mut world);
    clear(&mut world);
    world.add_relation::<ChildOf>(a, b);
    world.get_mut::<Transform>(a).unwrap().translation.x = 5.0;
    assert_eq!(
        propagate_transforms(&mut world),
        Err(TransformEcsError::Hierarchy(HierarchyError::Cycle))
    );
    assert_eq!(position(&world, a).x, 0.0);
    assert_eq!(position(&world, b).x, 1.0);
    world.remove_relation::<ChildOf>(a, b);
    assert_eq!(flush(&mut world).len(), 2);
    assert_eq!(position(&world, b).x, 6.0);
}

#[test]
fn removed_or_overwritten_derived_global_is_repaired() {
    let mut world = setup();
    let entity = world.spawn((Transform::from_xyz(3.0, 0.0, 0.0), bounds()));
    flush(&mut world);
    world.remove::<GlobalTransform>(entity);
    assert_eq!(propagate_transforms(&mut world).unwrap().written, 1);
    assert_eq!(position(&world, entity).x, 3.0);
    *world.get_mut::<GlobalTransform>(entity).unwrap() = GlobalTransform::IDENTITY;
    assert_eq!(propagate_transforms(&mut world).unwrap().written, 1);
    assert_eq!(position(&world, entity).x, 3.0);
    world.flush_changed_observers();
    assert!(drain_spatial_updates(&mut world).is_empty());
}

#[test]
fn fat_margin_suppresses_refits_until_escape() {
    let mut world = setup();
    set_spatial_margin(&mut world, 1.0);
    let entity = world.spawn((Transform::IDENTITY, bounds()));
    flush(&mut world);
    world.get_mut::<Transform>(entity).unwrap().translation.x = 0.5;
    assert!(flush(&mut world).is_empty());
    world.get_mut::<Transform>(entity).unwrap().translation.x = 3.0;
    assert!(matches!(flush(&mut world)[..],
        [SpatialUpdate::Update { new, .. }] if new.min.x == 2.0 && new.max.x == 5.0));
}

#[test]
fn schedule_flushes_notifications_and_spatial_updates_without_manual_bus_calls() {
    let mut world = setup();
    let entity = world.spawn((Transform::IDENTITY, bounds()));
    fn move_local(world: &mut World) {
        let query = world.query::<&mut Transform>();
        for mut local in query.iter_mut(world) {
            local.translation.x += 1.0;
        }
    }
    let mut schedule = Schedule::new();
    schedule.add_systems((move_local, transform_system).chain());
    schedule.run(&mut world);
    assert_eq!(position(&world, entity).x, 1.0);
    assert!(
        matches!(drain_spatial_updates(&mut world)[..], [SpatialUpdate::Insert { bounds, .. }] if bounds.min.x == 1.0)
    );
    clear(&mut world);
    schedule.run(&mut world);
    assert!(
        matches!(drain_spatial_updates(&mut world)[..], [SpatialUpdate::Update { new, .. }] if new.min.x == 2.0)
    );
    assert_eq!(world.resource::<Deliveries>().0.len(), 2);
}

#[test]
fn existing_components_install_idempotently_and_uninstall_removes_subscriptions() {
    let mut world = World::new();
    let entity = world.spawn((GlobalTransform::IDENTITY, bounds()));
    install(&mut world);
    install(&mut world);
    assert!(
        matches!(drain_spatial_updates(&mut world)[..], [SpatialUpdate::Insert { entity: e, .. }] if e == entity)
    );
    assert!(matches!(uninstall(&mut world)[..],
        [SpatialUpdate::Remove { entity: e, old }] if e == entity && old.max.x == 1.0));
    world.get_mut::<LocalBounds>(entity).unwrap().0.max.x = 2.0;
    assert_eq!(world.flush_changed_observers(), 0);
    install(&mut world);
    assert!(
        matches!(drain_spatial_updates(&mut world)[..], [SpatialUpdate::Insert { bounds, .. }] if bounds.max.x == 2.0)
    );
}

#[test]
fn chunk_versions_skip_static_local_reads_and_select_only_written_rows() {
    let mut world = setup();
    let entities: Vec<_> = (0..1024)
        .map(|_| world.spawn(Transform::IDENTITY))
        .collect();
    assert_eq!(propagate_transforms(&mut world).unwrap().local_reads, 1024);
    for _ in 0..3 {
        let stats = propagate_transforms(&mut world).unwrap();
        assert_eq!(
            (stats.local_reads, stats.recomputed, stats.written),
            (0, 0, 0)
        );
        assert_eq!((stats.membership_scans, stats.relation_checks), (0, 0));
    }
    world
        .get_mut::<Transform>(entities[513])
        .unwrap()
        .translation
        .z = 7.0;
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!(
        (stats.local_reads, stats.recomputed, stats.written),
        (1, 1, 1)
    );
    assert_eq!((stats.membership_scans, stats.relation_checks), (0, 0));
    assert_eq!(position(&world, entities[513]).z, 7.0);
    assert_eq!(propagate_transforms(&mut world).unwrap().local_reads, 0);
    // Overwriting an existing local is a value edit, not a membership change.
    world.insert(entities[513], Transform::from_xyz(0.0, 0.0, 9.0));
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!((stats.membership_scans, stats.relation_checks), (0, 0));
    assert_eq!(
        (stats.local_reads, stats.recomputed, stats.written),
        (1, 1, 1)
    );
    assert_eq!(position(&world, entities[513]).z, 9.0);
}

#[test]
fn relation_edge_snapshot_covers_typed_raw_and_replaced_registry_edges() {
    let mut world = setup();
    let a = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let b = world.spawn(Transform::from_xyz(20.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    let leaf = world.spawn(Transform::from_xyz(2.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(leaf, child);
    flush(&mut world);

    world.add_relation::<ChildOf>(child, a);
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!((stats.membership_scans, stats.relation_checks), (0, 1));
    assert_eq!((stats.recomputed, stats.written), (2, 2));
    assert_eq!(position(&world, leaf).x, 13.0);
    let saved = world.relations().clone();
    let relation = world.components().id_of::<ChildOf>().unwrap();
    world
        .relations_mut()
        .index_mut()
        .add(relation, child, b, true);
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!((stats.membership_scans, stats.relation_checks), (0, 1));
    assert_eq!(position(&world, leaf).x, 23.0);

    *world.relations_mut() = saved;
    propagate_transforms(&mut world).unwrap();
    assert_eq!(position(&world, leaf).x, 13.0);
    // Registry replacement publishes the net changed sources, including
    // deleted buckets, rather than invalidating every Transform source.
    *world.relations_mut() = Default::default();
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!(stats.relation_checks, 2);
    assert_eq!(position(&world, child).x, 1.0);
    assert_eq!(position(&world, leaf).x, 2.0);
    assert_eq!(propagate_transforms(&mut world).unwrap().relation_checks, 0);
}

#[test]
fn despawning_transformless_parent_invalidates_cached_topology() {
    let mut world = setup();
    // Exercise edge cleanup while the child survives; canonical ChildOf
    // otherwise deletes the whole subtree when the parent is despawned.
    world.register_relation::<ChildOf>(prism_ecs::relation::RelationKind {
        on_delete_target: prism_ecs::relation::CleanupPolicy::Remove,
        ..<ChildOf as prism_ecs::relation::Relation>::KIND
    });
    let boundary = world.spawn(());
    let child = world.spawn(Transform::from_xyz(1.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(child, boundary);
    flush(&mut world);
    world.despawn(boundary);
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!((stats.membership_scans, stats.relation_checks), (0, 1));
    assert_eq!((stats.recomputed, stats.written), (0, 0));
    assert_eq!(position(&world, child).x, 1.0);
}

#[test]
fn snapshot_restore_reloads_locals_membership_and_spatial_pairs_without_bus_events() {
    let mut world = setup();
    world.register_snapshot_component::<Transform>();
    world.register_snapshot_component::<GlobalTransform>();
    world.register_snapshot_component::<LocalBounds>();
    let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let child = world.spawn((Transform::from_xyz(1.0, 0.0, 0.0), bounds()));
    world.add_relation::<ChildOf>(child, parent);
    let standalone = world.spawn((GlobalTransform::IDENTITY, bounds()));
    let resurrected = world.spawn((Transform::from_xyz(30.0, 0.0, 0.0), bounds()));
    flush(&mut world);
    let snapshot = world.snapshot();

    world.get_mut::<Transform>(parent).unwrap().translation.x = 20.0;
    world.get_mut::<LocalBounds>(standalone).unwrap().0.max.x = 4.0;
    world.remove::<Transform>(resurrected);
    let abandoned = world.spawn((Transform::from_xyz(50.0, 0.0, 0.0), bounds()));
    flush(&mut world);
    clear(&mut world);
    let generation = world.archetype_generation();
    world.restore(&snapshot);
    assert_ne!(world.archetype_generation(), generation);
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!(
        (
            stats.membership_scans,
            stats.relation_checks,
            stats.local_reads
        ),
        (1, 3, 3)
    );
    assert_eq!((stats.recomputed, stats.written), (3, 0));
    assert_eq!(position(&world, child).x, 11.0);
    assert_eq!(position(&world, resurrected).x, 30.0);
    assert!(!world.contains(abandoned));
    // No observer flush: restore itself must invalidate spatial membership and
    // standalone bounds, including a removed and resurrected local/global pair.
    let updates = drain_spatial_updates(&mut world);
    assert_eq!(updates.len(), 4);
    assert!(updates
        .iter()
        .any(|u| matches!(u, SpatialUpdate::Update { entity, new, .. }
        if *entity == child && new.min.x == 11.0)));
    assert!(updates
        .iter()
        .any(|u| matches!(u, SpatialUpdate::Update { entity, new, .. }
        if *entity == standalone && new.max.x == 1.0)));
    assert!(updates
        .iter()
        .any(|u| matches!(u, SpatialUpdate::Insert { entity, .. } if *entity == resurrected)));
    assert!(updates
        .iter()
        .any(|u| matches!(u, SpatialUpdate::Remove { entity, .. } if *entity == abandoned)));
    assert!(drain_spatial_updates(&mut world).is_empty());
    assert_eq!(
        propagate_transforms(&mut world).unwrap().membership_scans,
        0
    );
    assert_eq!(propagate_transforms(&mut world).unwrap().relation_checks, 0);
}

#[test]
fn restore_preserves_standalone_globals_managed_only_on_the_abandoned_timeline() {
    let mut world = setup();
    world.register_snapshot_component::<Transform>();
    world.register_snapshot_component::<GlobalTransform>();
    world.register_snapshot_component::<LocalBounds>();
    let entity = world.spawn((
        GlobalTransform::from_transform(&Transform::from_xyz(8.0, 0.0, 0.0)),
        bounds(),
    ));
    flush(&mut world);
    let snapshot = world.snapshot();
    world.insert(entity, Transform::from_xyz(20.0, 0.0, 0.0));
    flush(&mut world);
    world.restore(&snapshot);
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!(
        (
            stats.local_reads,
            stats.recomputed,
            stats.written,
            stats.removed
        ),
        (0, 0, 0, 0)
    );
    assert!(!world.has::<Transform>(entity));
    assert_eq!(position(&world, entity).x, 8.0);
    assert!(matches!(drain_spatial_updates(&mut world)[..],
        [SpatialUpdate::Update { entity: e, new, .. }] if e == entity && new.min.x == 8.0));
}

#[test]
fn replacing_world_reinstalls_caches_for_reused_entity_handles() {
    let mut world = setup();
    let old_id = world.id();
    let old = world.spawn((Transform::from_xyz(100.0, 0.0, 0.0), bounds()));
    propagate_transforms(&mut world).unwrap();
    world = World::new();
    assert_ne!(world.id(), old_id);
    let entity = world.spawn((Transform::from_xyz(2.0, 0.0, 0.0), bounds()));
    assert_eq!(entity, old);
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!(stats.local_reads, 1);
    assert_eq!(position(&world, entity).x, 2.0);
    assert!(matches!(drain_spatial_updates(&mut world)[..],
        [SpatialUpdate::Insert { entity: e, bounds }] if e == entity && bounds.min.x == 2.0));
}

#[test]
fn insert_observer_panic_preserves_cache_and_retries_spatial_publication() {
    let mut world = World::new();
    let panic_once = Arc::new(AtomicBool::new(true));
    let armed = Arc::clone(&panic_once);
    // Register before the integration so the first panic also prevents its
    // lifecycle subscriber from receiving the initial publication.
    world.observe::<GlobalTransform, _>(LifecycleEvent::Insert, move |_| {
        assert!(
            !armed.swap(false, Ordering::SeqCst),
            "insert observer panic"
        );
    });
    install(&mut world);
    let parent = world.spawn((Transform::from_xyz(10.0, 0.0, 0.0), bounds()));
    let child = world.spawn((Transform::from_xyz(2.0, 0.0, 0.0), bounds()));
    world.add_relation::<ChildOf>(child, parent);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        propagate_transforms(&mut world).unwrap();
    }))
    .is_err());
    assert!(!panic_once.load(Ordering::SeqCst));

    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!(stats.recomputed, 2);
    assert_eq!(position(&world, parent).x, 10.0);
    assert_eq!(position(&world, child).x, 12.0);
    world.flush_changed_observers();
    let updates = drain_spatial_updates(&mut world);
    assert_eq!(updates.len(), 2);
    assert!(updates.iter().any(|u| matches!(u,
        SpatialUpdate::Insert { entity, bounds } if *entity == parent && bounds.min.x == 10.0)));
    assert!(updates.iter().any(|u| matches!(u,
        SpatialUpdate::Insert { entity, bounds } if *entity == child && bounds.min.x == 12.0)));
    assert_eq!(propagate_transforms(&mut world).unwrap().recomputed, 0);
}

#[test]
fn remove_observer_panic_retries_current_and_remaining_global_removals() {
    let mut world = World::new();
    let panic_once = Arc::new(AtomicBool::new(true));
    let armed = Arc::clone(&panic_once);
    world.observe::<GlobalTransform, _>(LifecycleEvent::Remove, move |_| {
        assert!(
            !armed.swap(false, Ordering::SeqCst),
            "remove observer panic"
        );
    });
    install(&mut world);
    let a = world.spawn((Transform::IDENTITY, bounds()));
    let b = world.spawn((Transform::from_xyz(2.0, 0.0, 0.0), bounds()));
    assert_eq!(flush(&mut world).len(), 2);
    world.remove::<Transform>(a);
    world.remove::<Transform>(b);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        propagate_transforms(&mut world).unwrap();
    }))
    .is_err());
    // Remove callbacks run before the component is dropped.
    assert!(world.has::<GlobalTransform>(a));
    assert!(world.has::<GlobalTransform>(b));
    let stats = propagate_transforms(&mut world).unwrap();
    assert_eq!(stats.removed, 2);
    assert!(!world.has::<GlobalTransform>(a));
    assert!(!world.has::<GlobalTransform>(b));
    world.flush_changed_observers();
    let updates = drain_spatial_updates(&mut world);
    assert_eq!(updates.len(), 2);
    assert!(updates.iter().any(|u| matches!(u,
        SpatialUpdate::Remove { entity, .. } if *entity == a)));
    assert!(updates.iter().any(|u| matches!(u,
        SpatialUpdate::Remove { entity, .. } if *entity == b)));
    assert_eq!(propagate_transforms(&mut world).unwrap().removed, 0);
}

#[test]
fn install_preserves_preconfigured_child_of_cleanup_policy() {
    use prism_ecs::relation::{CleanupPolicy, Relation, RelationKind};

    let mut world = World::new();
    let kind = RelationKind {
        on_delete_target: CleanupPolicy::Remove,
        ..<ChildOf as Relation>::KIND
    };
    world.register_relation::<ChildOf>(kind);
    install(&mut world);
    let id = world.components().id_of::<ChildOf>().unwrap();
    assert_eq!(world.relations().kind(id), Some(&kind));
    let parent = world.spawn(Transform::from_xyz(10.0, 0.0, 0.0));
    let child = world.spawn(Transform::from_xyz(2.0, 0.0, 0.0));
    world.add_relation::<ChildOf>(child, parent);
    propagate_transforms(&mut world).unwrap();
    assert_eq!(position(&world, child).x, 12.0);
    world.despawn(parent);
    assert!(world.contains(child));
    propagate_transforms(&mut world).unwrap();
    assert_eq!(position(&world, child).x, 2.0);
    assert!(world.relation_targets::<ChildOf>(child).is_empty());
}

#[test]
fn uninstall_snapshot_keeps_committed_bounds_with_pending_removal_or_move() {
    let mut world = setup();
    let a = world.spawn((Transform::IDENTITY, bounds()));
    let b = world.spawn((Transform::IDENTITY, bounds()));
    let c = world.spawn((Transform::IDENTITY, bounds()));
    flush(&mut world);
    world.despawn(a);
    world.get_mut::<Transform>(b).unwrap().translation.x = 9.0;
    propagate_transforms(&mut world).unwrap();
    world.flush_changed_observers();
    let undrained = world.spawn((GlobalTransform::IDENTITY, bounds()));
    let snapshot = prism_transform::ecs::spatial_snapshot(&world);
    assert_eq!(snapshot.len(), 3);
    assert!(snapshot.iter().all(|&(_, old)| old.min.x == 0.0));
    let removals = uninstall(&mut world);
    assert_eq!(
        removals
            .iter()
            .map(SpatialUpdate::entity)
            .collect::<Vec<_>>(),
        vec![a, b, c]
    );
    assert!(removals
        .iter()
        .all(|u| matches!(u, SpatialUpdate::Remove { old, .. } if old.min.x == 0.0)));
    assert!(uninstall(&mut world).is_empty());
    assert!(prism_transform::ecs::spatial_snapshot(&world).is_empty());
    clear(&mut world);
    world.remove::<LocalBounds>(undrained);
    world
        .get_mut::<GlobalTransform>(b)
        .unwrap()
        .clone_from(&GlobalTransform::IDENTITY);
    // Uninstall owns only its subscriptions; the two external subscribers
    // still receive the pending insertion and subsequent global write.
    assert_eq!(world.flush_changed_observers(), 2);
    let mut changed = vec![b, undrained];
    changed.sort_unstable();
    assert_eq!(
        world.resource::<Deliveries>().0,
        vec![(0, changed.clone()), (1, changed)]
    );
    assert_eq!(world.flush_changed_observers(), 0);
}

#[test]
fn write_and_relation_insertion_order_do_not_change_delivery_or_commands() {
    fn run(reverse: bool) -> (Deliveries, Vec<SpatialUpdate>, Vec<GlobalTransform>) {
        let mut world = setup();
        let root = world.spawn(Transform::IDENTITY);
        let children: Vec<_> = (0..4)
            .map(|_| world.spawn((Transform::IDENTITY, bounds())))
            .collect();
        let mut order = children.clone();
        if reverse {
            order.reverse();
        }
        for &entity in &order {
            world.add_relation::<ChildOf>(entity, root);
        }
        flush(&mut world);
        clear(&mut world);
        for &entity in &order {
            world.get_mut::<Transform>(entity).unwrap().translation.x = entity.index() as f32;
        }
        world.get_mut::<Transform>(root).unwrap().translation.y = 4.0;
        let commands = flush(&mut world);
        let globals = children
            .iter()
            .map(|&entity| *world.get::<GlobalTransform>(entity).unwrap())
            .collect();
        (
            world.remove_resource::<Deliveries>().unwrap(),
            commands,
            globals,
        )
    }
    assert_eq!(run(false), run(true));
}
