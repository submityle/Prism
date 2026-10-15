//! Real backend queries, incremental mutation and observer/uninstall contracts.

use prism_ecs::entity::Entity;
use prism_ecs::world::World;
use prism_math::{Aabb3, Vec3};
use prism_transform::ecs::{self, LocalBounds, SpatialUpdate};
use prism_transform::{GlobalTransform, Transform};
use prism_transform_ecs::{self as adapters, SpatialIndexError, SpatialIndexes};

fn bounds() -> LocalBounds {
    LocalBounds(Aabb3::new(Vec3::ZERO, Vec3::ONE))
}

fn setup() -> World {
    let mut world = World::new();
    adapters::install(&mut world);
    #[cfg(feature = "bvh")]
    assert!(adapters::install_bvh(&mut world));
    #[cfg(feature = "grid")]
    assert!(adapters::install_grid(&mut world, 4.0).unwrap());
    world
}

fn sync(world: &mut World) -> adapters::SpatialIndexBatch {
    ecs::propagate_transforms(world).unwrap();
    world.flush_changed_observers();
    adapters::sync_spatial_indexes(world).unwrap()
}

fn hits(world: &World, x: f32, expected: &[Entity]) {
    let indexes = world.resource::<SpatialIndexes>();
    #[cfg(feature = "bvh")]
    {
        let query = Aabb3::new(Vec3::new(x, -1.0, -1.0), Vec3::new(x + 1.0, 2.0, 2.0));
        assert_eq!(indexes.bvh().unwrap().query_aabb(query), expected);
        assert!(indexes.bvh().unwrap().tree().validate());
    }
    #[cfg(feature = "grid")]
    assert_eq!(
        indexes.grid().unwrap().query_circle(x + 0.5, 0.5, 0.25),
        expected
    );
    let _ = (indexes, x, expected);
}

#[test]
fn repeated_writes_bus_flush_static_and_net_noop_are_incremental() {
    let mut world = setup();
    let entity = world.spawn((Transform::IDENTITY, bounds()));
    assert_eq!(sync(&mut world).stats.inserts, 1);
    hits(&world, 0.0, &[entity]);
    for x in [2.0, 8.0, 12.0] {
        world.get_mut::<Transform>(entity).unwrap().translation.x = x;
    }
    ecs::propagate_transforms(&mut world).unwrap();
    assert!(adapters::sync_spatial_indexes(&mut world)
        .unwrap()
        .updates
        .is_empty());
    hits(&world, 0.0, &[entity]);
    world.flush_changed_observers();
    let batch = adapters::sync_spatial_indexes(&mut world).unwrap();
    assert_eq!(batch.stats.updates, 1);
    #[cfg(feature = "bvh")]
    assert_eq!(batch.stats.bvh_refits, 1);
    #[cfg(feature = "grid")]
    assert_eq!(batch.stats.grid_moves, 1);
    hits(&world, 0.0, &[]);
    hits(&world, 12.0, &[entity]);
    assert_eq!(sync(&mut world), adapters::SpatialIndexBatch::default());
    for x in [25.0, 12.0] {
        world.get_mut::<Transform>(entity).unwrap().translation.x = x;
        ecs::propagate_transforms(&mut world).unwrap();
        world.flush_changed_observers();
    }
    assert_eq!(
        adapters::sync_spatial_indexes(&mut world).unwrap(),
        adapters::SpatialIndexBatch::default()
    );
}

#[test]
fn removals_despawn_local_loss_and_generation_reuse_remove_actual_proxies() {
    let mut world = setup();
    let old = world.spawn((Transform::IDENTITY, bounds()));
    let no_bounds = world.spawn((Transform::from_xyz(4.0, 0.0, 0.0), bounds()));
    let no_local = world.spawn((Transform::from_xyz(8.0, 0.0, 0.0), bounds()));
    let no_global = world.spawn((
        GlobalTransform::from_transform(&Transform::from_xyz(12.0, 0.0, 0.0)),
        bounds(),
    ));
    sync(&mut world);
    world.despawn(old);
    let new = world.spawn((Transform::from_xyz(20.0, 0.0, 0.0), bounds()));
    assert_eq!(new.index(), old.index());
    assert_ne!(new, old);
    world.remove::<LocalBounds>(no_bounds);
    world.remove::<Transform>(no_local);
    world.remove::<GlobalTransform>(no_global);
    let batch = sync(&mut world);
    assert_eq!((batch.stats.inserts, batch.stats.removes), (1, 4));
    for x in [0.0, 4.0, 8.0, 12.0] {
        hits(&world, x, &[]);
    }
    hits(&world, 20.0, &[new]);
    let indexes = world.resource::<SpatialIndexes>();
    assert_eq!(indexes.len(), 1);
    assert!(indexes.bounds(old).is_none());
    #[cfg(feature = "bvh")]
    assert_eq!(indexes.bvh().unwrap().tree().len(), 1);
    #[cfg(feature = "grid")]
    assert_eq!(indexes.grid().unwrap().grid().len(), 1);
}

#[test]
fn bounds_only_global_only_and_backend_specific_suppression() {
    let mut world = setup();
    let entity = world.spawn((GlobalTransform::IDENTITY, bounds()));
    sync(&mut world);
    // Y changes affect BVH bounds, while the grid's XZ point stays still.
    world.get_mut::<LocalBounds>(entity).unwrap().0.max.y = 4.0;
    let batch = sync(&mut world);
    assert_eq!(batch.stats.updates, 1);
    assert_eq!(batch.stats.grid_moves, 0);
    #[cfg(feature = "bvh")]
    assert_eq!(batch.stats.bvh_refits, 1);
    *world.get_mut::<GlobalTransform>(entity).unwrap() =
        GlobalTransform::from_transform(&Transform::from_xyz(0.02, 0.0, 0.0));
    let batch = sync(&mut world);
    assert_eq!(batch.stats.updates, 1);
    // The existing BVH fat margin suppresses a refit, but its adapter's exact
    // committed-box filter still rejects a point that the entity has vacated.
    assert_eq!(batch.stats.bvh_refits, 0);
    #[cfg(feature = "bvh")]
    {
        let indexes = world.resource::<SpatialIndexes>();
        let query = Aabb3::new(Vec3::new(0.0, 0.1, 0.1), Vec3::new(0.01, 0.2, 0.2));
        assert!(indexes.bvh().unwrap().query_aabb(query).is_empty());
    }
    #[cfg(feature = "grid")]
    assert_eq!(
        world
            .resource::<SpatialIndexes>()
            .grid()
            .unwrap()
            .position(entity),
        Some((0.52, 0.5))
    );
}

#[test]
fn invalid_batch_is_atomic_retained_and_repaired_by_new_signals() {
    let mut world = setup();
    let a = world.spawn((Transform::IDENTITY, bounds()));
    let b = world.spawn((Transform::IDENTITY, bounds()));
    sync(&mut world);
    world.get_mut::<Transform>(a).unwrap().translation.x = 10.0;
    world.get_mut::<LocalBounds>(b).unwrap().0.max.x = f32::INFINITY;
    ecs::propagate_transforms(&mut world).unwrap();
    world.flush_changed_observers();
    assert_eq!(
        adapters::sync_spatial_indexes(&mut world),
        Err(SpatialIndexError::InvalidBounds(b))
    );
    // The valid edit is also held until the complete batch can be committed.
    hits(&world, 0.0, &[a, b]);
    assert_eq!(
        adapters::sync_spatial_indexes(&mut world),
        Err(SpatialIndexError::InvalidBounds(b))
    );
    world.remove::<LocalBounds>(b);
    let batch = sync(&mut world);
    assert_eq!((batch.stats.updates, batch.stats.removes), (1, 1));
    hits(&world, 0.0, &[]);
    hits(&world, 10.0, &[a]);
}

#[test]
fn restore_reconciles_real_indexes_without_bus_events_and_preserves_live_proxies() {
    let mut world = setup();
    world.register_snapshot_component::<Transform>();
    world.register_snapshot_component::<GlobalTransform>();
    world.register_snapshot_component::<LocalBounds>();
    let moved = world.spawn((Transform::IDENTITY, bounds()));
    let resurrected = world.spawn((Transform::from_xyz(8.0, 0.0, 0.0), bounds()));
    let standalone = world.spawn((
        GlobalTransform::from_transform(&Transform::from_xyz(24.0, 0.0, 0.0)),
        bounds(),
    ));
    sync(&mut world);
    let snapshot = world.snapshot();
    #[cfg(feature = "bvh")]
    let proxy = world
        .resource::<SpatialIndexes>()
        .bvh()
        .unwrap()
        .proxy(moved);

    world.get_mut::<Transform>(moved).unwrap().translation.x = 12.0;
    world.get_mut::<LocalBounds>(standalone).unwrap().0.max.y = 4.0;
    world.remove::<Transform>(resurrected);
    let abandoned = world.spawn((Transform::from_xyz(20.0, 0.0, 0.0), bounds()));
    let changed = sync(&mut world);
    assert_eq!(
        (
            changed.stats.inserts,
            changed.stats.updates,
            changed.stats.removes
        ),
        (1, 2, 1)
    );

    world.restore(&snapshot);
    let propagation = ecs::propagate_transforms(&mut world).unwrap();
    assert_eq!((propagation.recomputed, propagation.written), (2, 0));
    // Restore bypasses observers. Reconciliation must also discover the
    // standalone bounds edit and the resurrected/abandoned entity pair.
    let batch = adapters::sync_spatial_indexes(&mut world).unwrap();
    assert_eq!(
        (
            batch.stats.inserts,
            batch.stats.updates,
            batch.stats.removes
        ),
        (1, 2, 1)
    );
    assert!(batch.updates.iter().any(|update| matches!(update,
        SpatialUpdate::Remove { entity, .. } if *entity == abandoned)));
    hits(&world, 0.0, &[moved]);
    hits(&world, 8.0, &[resurrected]);
    hits(&world, 12.0, &[]);
    hits(&world, 20.0, &[]);
    hits(&world, 24.0, &[standalone]);
    assert_eq!(
        world
            .resource::<SpatialIndexes>()
            .bounds(standalone)
            .unwrap()
            .max
            .y,
        1.0
    );
    #[cfg(feature = "bvh")]
    assert_eq!(
        world
            .resource::<SpatialIndexes>()
            .bvh()
            .unwrap()
            .proxy(moved),
        proxy
    );

    // Replaying the same snapshot restores exactly the same old component
    // ticks. It must invalidate caches again without publishing equal values.
    world.restore(&snapshot);
    let propagation = ecs::propagate_transforms(&mut world).unwrap();
    assert_eq!(
        (
            propagation.local_reads,
            propagation.recomputed,
            propagation.written
        ),
        (2, 2, 0)
    );
    assert_eq!(
        adapters::sync_spatial_indexes(&mut world).unwrap(),
        adapters::SpatialIndexBatch::default()
    );
    let idle = ecs::propagate_transforms(&mut world).unwrap();
    assert_eq!(
        (
            idle.membership_scans,
            idle.relation_checks,
            idle.local_reads
        ),
        (0, 0, 0)
    );
    assert_eq!(
        adapters::sync_spatial_indexes(&mut world).unwrap(),
        adapters::SpatialIndexBatch::default()
    );
}

#[test]
fn restore_discards_invalid_pending_timeline_and_removes_absent_committed_owners() {
    let mut world = setup();
    world.register_snapshot_component::<Transform>();
    world.register_snapshot_component::<GlobalTransform>();
    world.register_snapshot_component::<LocalBounds>();
    let a = world.spawn((Transform::IDENTITY, bounds()));
    let b = world.spawn((Transform::from_xyz(8.0, 0.0, 0.0), bounds()));
    let removed = world.spawn((Transform::from_xyz(16.0, 0.0, 0.0), bounds()));
    sync(&mut world);
    world.remove::<LocalBounds>(removed);
    let snapshot = world.snapshot();
    world.get_mut::<Transform>(a).unwrap().translation.x = 12.0;
    world.get_mut::<LocalBounds>(b).unwrap().0.max.x = f32::INFINITY;
    let abandoned = world.spawn((
        GlobalTransform::IDENTITY,
        LocalBounds(Aabb3::new(Vec3::ZERO, Vec3::splat(f32::INFINITY))),
    ));
    ecs::propagate_transforms(&mut world).unwrap();
    world.flush_changed_observers();
    assert_eq!(
        adapters::sync_spatial_indexes(&mut world),
        Err(SpatialIndexError::InvalidBounds(b))
    );
    assert!(ecs::spatial_snapshot(&world)
        .iter()
        .all(|(entity, _)| *entity != removed));
    // The base stream already drained the removal, while the invalid batch
    // kept the actual backend proxy. Restore must reconcile from both states.
    hits(&world, 16.0, &[removed]);
    world.restore(&snapshot);
    // No propagation or observer flush is needed for the restored globals.
    let batch = adapters::sync_spatial_indexes(&mut world).unwrap();
    assert_eq!(
        (
            batch.stats.inserts,
            batch.stats.updates,
            batch.stats.removes
        ),
        (0, 0, 1)
    );
    hits(&world, 0.0, &[a]);
    hits(&world, 8.0, &[b]);
    hits(&world, 12.0, &[]);
    hits(&world, 16.0, &[]);
    assert!(world
        .resource::<SpatialIndexes>()
        .bounds(abandoned)
        .is_none());
    assert_eq!(world.resource::<SpatialIndexes>().len(), 2);
    assert_eq!(
        adapters::sync_spatial_indexes(&mut world).unwrap(),
        adapters::SpatialIndexBatch::default()
    );
}

#[test]
fn moving_indexes_to_another_world_cleans_previous_proxies_with_reused_entity_handles() {
    let mut previous = setup();
    let old = previous.spawn((Transform::IDENTITY, bounds()));
    previous.spawn((Transform::from_xyz(8.0, 0.0, 0.0), bounds()));
    sync(&mut previous);
    let indexes = previous.remove_resource::<SpatialIndexes>().unwrap();
    let mut world = World::new();
    let entity = world.spawn((Transform::from_xyz(40.0, 0.0, 0.0), bounds()));
    assert_eq!(entity, old);
    assert_ne!(world.id(), previous.id());
    world.insert_resource(indexes);
    let batch = sync(&mut world);
    assert_eq!(
        (
            batch.stats.inserts,
            batch.stats.updates,
            batch.stats.removes
        ),
        (1, 0, 0)
    );
    assert_eq!(world.resource::<SpatialIndexes>().len(), 1);
    hits(&world, 0.0, &[]);
    hits(&world, 8.0, &[]);
    hits(&world, 40.0, &[entity]);
    #[cfg(feature = "bvh")]
    assert_eq!(
        world
            .resource::<SpatialIndexes>()
            .bvh()
            .unwrap()
            .tree()
            .len(),
        1
    );
    #[cfg(feature = "grid")]
    assert_eq!(
        world
            .resource::<SpatialIndexes>()
            .grid()
            .unwrap()
            .grid()
            .len(),
        1
    );
    assert_eq!(sync(&mut world), adapters::SpatialIndexBatch::default());
}

#[test]
fn uninstall_cleans_committed_proxies_even_with_invalid_or_unflushed_pending_edits() {
    let mut world = setup();
    let a = world.spawn((Transform::IDENTITY, bounds()));
    let b = world.spawn((Transform::IDENTITY, bounds()));
    sync(&mut world);
    world.get_mut::<LocalBounds>(b).unwrap().0.max.x = f32::INFINITY;
    world.flush_changed_observers();
    assert!(adapters::sync_spatial_indexes(&mut world).is_err());
    world.despawn(a);
    world.insert(b, bounds());
    world.get_mut::<Transform>(b).unwrap().translation.x = 15.0;
    let (removals, indexes) = adapters::uninstall(&mut world);
    assert_eq!(
        removals
            .iter()
            .map(SpatialUpdate::entity)
            .collect::<Vec<_>>(),
        vec![a, b]
    );
    let indexes = indexes.unwrap();
    assert!(indexes.is_empty());
    #[cfg(feature = "bvh")]
    {
        assert!(indexes.bvh().unwrap().tree().is_empty());
        assert!(indexes.bvh().unwrap().tree().validate());
        assert!(indexes.bvh().unwrap().proxy(a).is_none());
    }
    #[cfg(feature = "grid")]
    {
        assert!(indexes.grid().unwrap().grid().is_empty());
        assert!(indexes
            .grid()
            .unwrap()
            .query_circle(0.5, 0.5, 100.0)
            .is_empty());
    }
    assert!(!world.contains_resource::<SpatialIndexes>());
    assert_eq!(world.flush_changed_observers(), 0);
    let (updates, indexes) = adapters::uninstall(&mut world);
    assert!(updates.is_empty() && indexes.is_none());
    adapters::install(&mut world);
    #[cfg(feature = "bvh")]
    adapters::install_bvh(&mut world);
    #[cfg(feature = "grid")]
    adapters::install_grid(&mut world, 4.0).unwrap();
    assert_eq!(sync(&mut world).stats.inserts, 1);
    hits(&world, 15.0, &[b]);
}

#[test]
fn late_install_seeds_previously_drained_baseline_and_applies_pending_moves() {
    let mut world = World::new();
    let entity = world.spawn((Transform::IDENTITY, bounds()));
    ecs::propagate_transforms(&mut world).unwrap();
    world.flush_changed_observers();
    assert_eq!(ecs::drain_spatial_updates(&mut world).len(), 1);
    world.get_mut::<Transform>(entity).unwrap().translation.x = 8.0;
    adapters::install(&mut world);
    #[cfg(feature = "bvh")]
    adapters::install_bvh(&mut world);
    #[cfg(feature = "grid")]
    adapters::install_grid(&mut world, 4.0).unwrap();
    assert_eq!(sync(&mut world).stats.inserts, 1);
    hits(&world, 0.0, &[]);
    hits(&world, 8.0, &[entity]);
    #[cfg(feature = "bvh")]
    assert!(!adapters::install_bvh(&mut world));
    #[cfg(feature = "grid")]
    assert!(!adapters::install_grid(&mut world, 8.0).unwrap());
    assert_eq!(sync(&mut world), adapters::SpatialIndexBatch::default());
}

#[test]
fn insertion_write_and_removal_order_do_not_change_results() {
    fn run(reverse: bool) -> (Vec<SpatialUpdate>, Vec<Option<Aabb3>>) {
        let mut world = setup();
        let entities: Vec<_> = (0..10)
            .map(|_| world.spawn((Transform::IDENTITY, bounds())))
            .collect();
        sync(&mut world);
        let mut order = entities.clone();
        if reverse {
            order.reverse();
        }
        for entity in order {
            if entity.index() % 2 == 0 {
                world.remove::<LocalBounds>(entity);
            } else {
                world.get_mut::<Transform>(entity).unwrap().translation.x = 8.0;
            }
        }
        let batch = sync(&mut world);
        let expected: Vec<_> = entities
            .iter()
            .copied()
            .filter(|e| e.index() % 2 != 0)
            .collect();
        hits(&world, 8.0, &expected);
        let indexes = world.resource::<SpatialIndexes>();
        (
            batch.updates,
            entities.into_iter().map(|e| indexes.bounds(e)).collect(),
        )
    }
    assert_eq!(run(false), run(true));
}

#[cfg(feature = "bvh")]
#[test]
fn late_backend_enablement_seeds_committed_entries_without_rebuilding_existing_bvh() {
    let mut world = World::new();
    let entity = world.spawn((Transform::IDENTITY, bounds()));
    adapters::install(&mut world);
    sync(&mut world);
    assert!(adapters::install_bvh(&mut world));
    let proxy = world
        .resource::<SpatialIndexes>()
        .bvh()
        .unwrap()
        .proxy(entity)
        .unwrap();
    assert!(!adapters::install_bvh(&mut world));
    world.get_mut::<Transform>(entity).unwrap().translation.x = 20.0;
    sync(&mut world);
    assert_eq!(
        world
            .resource::<SpatialIndexes>()
            .bvh()
            .unwrap()
            .proxy(entity),
        Some(proxy)
    );
    #[cfg(feature = "grid")]
    {
        assert!(adapters::install_grid(&mut world, 4.0).unwrap());
        assert_eq!(
            world
                .resource::<SpatialIndexes>()
                .grid()
                .unwrap()
                .query_circle(20.5, 0.5, 0.25),
            vec![entity]
        );
    }
}

#[cfg(feature = "grid")]
#[test]
fn grid_configuration_rejects_invalid_size_and_tracks_xz_points() {
    let mut world = World::new();
    assert!(adapters::install_grid(&mut world, 0.0).is_err());
    assert!(adapters::install_grid(&mut world, f32::NAN).is_err());
    assert!(adapters::install_grid(&mut world, 4.0).unwrap());
    let entity = world.spawn((Transform::from_xyz(-8.0, 100.0, -8.0), bounds()));
    sync(&mut world);
    let grid = world.resource::<SpatialIndexes>().grid().unwrap();
    assert_eq!(grid.query_circle(-7.5, -7.5, 0.1), vec![entity]);
    assert!(grid.query_circle(0.0, 0.0, 0.1).is_empty());
    assert!(grid.query_circle(f32::NAN, 0.0, 1.0).is_empty());
}
