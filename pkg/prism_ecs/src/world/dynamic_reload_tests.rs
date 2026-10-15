use core::alloc::Layout;

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::convert::Infallible;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::change::ComponentTicks;
use crate::component::{Component, ComponentId, StorageType};
use crate::component_hooks::{ComponentHooks, HookContext};
use crate::entity::Entity;

use super::{DynamicComponentError, DynamicComponentLayout, DynamicReloadError, World};

// Test-local ABI glue: do not expose the registry's private unsafe helpers.
unsafe fn drop_value<T>(ptr: *mut u8) {
    // SAFETY: test descriptors always use the exact initialized T.
    unsafe { core::ptr::drop_in_place(ptr.cast::<T>()) };
}

unsafe fn clone_value<T: Clone>(src: *const u8, dst: *mut u8) {
    // SAFETY: callers supply a live T and separate uninitialized T storage.
    let value = unsafe { (&*src.cast::<T>()).clone() };
    // SAFETY: clone succeeded, and dst has T's layout.
    unsafe { dst.cast::<T>().write(value) };
}

unsafe fn hash_value<T: core::hash::Hash>(ptr: *const u8, mut hasher: &mut dyn core::hash::Hasher) {
    // SAFETY: test descriptors bind this glue to exactly T.
    core::hash::Hash::hash(unsafe { &*ptr.cast::<T>() }, &mut hasher);
}

fn layout<T: Send + Sync + 'static>() -> DynamicComponentLayout {
    // SAFETY: every glue operation is derived from exactly T.
    unsafe {
        DynamicComponentLayout::new(
            Layout::new::<T>(),
            core::mem::needs_drop::<T>().then_some(drop_value::<T> as crate::component::DropFn),
            None,
            None,
        )
    }
}

fn snapshot_layout<T: Send + Sync + Clone + core::hash::Hash + 'static>() -> DynamicComponentLayout
{
    unsafe {
        DynamicComponentLayout::new(
            Layout::new::<T>(),
            core::mem::needs_drop::<T>().then_some(drop_value::<T> as crate::component::DropFn),
            Some(clone_value::<T>),
            Some(hash_value::<T>),
        )
    }
}

fn insert<T: Send + Sync + 'static>(world: &mut World, entity: Entity, id: ComponentId, value: T) {
    assert!(world
        .insert_dynamic_component(entity, id, |slot| {
            unsafe { slot.write(value) };
            Ok::<_, Infallible>(())
        })
        .unwrap());
}

fn read<T: Copy>(world: &World, entity: Entity, id: ComponentId) -> T {
    unsafe {
        *world
            .get_dynamic_component(entity, id)
            .unwrap()
            .unwrap()
            .as_ref::<T>()
    }
}

fn ticks(world: &World, entity: Entity, id: ComponentId) -> ComponentTicks {
    let info = world.components().info(id).unwrap();
    match info.storage() {
        StorageType::Table => {
            let loc = world.entities.location(entity).unwrap();
            world
                .archetypes
                .get(loc.archetype_id)
                .unwrap()
                .table()
                .column(id)
                .unwrap()
                .component_ticks(loc.row as usize)
        }
        StorageType::SparseSet => world
            .sparse_sets
            .get(id)
            .unwrap()
            .component_ticks(entity)
            .unwrap(),
        StorageType::Shared => unreachable!(),
    }
}

fn register_table_u32(world: &mut World) -> ComponentId {
    world
        .register_dynamic_component("runtime.u32", StorageType::Table, layout::<u32>())
        .unwrap()
}

fn place_u32(world: &mut World, id: ComponentId, value: u32) -> Entity {
    let entity = world.spawn(());
    insert(world, entity, id, value);
    entity
}

fn u64_layout() -> DynamicComponentLayout {
    // SAFETY: the descriptor has no user glue and exactly describes `u64`.
    unsafe { DynamicComponentLayout::new(Layout::new::<u64>(), None, None, None) }
}

#[test]
fn live_table_values_migrate_and_publish_the_new_layout() {
    let mut world = World::new();
    let id = register_table_u32(&mut world);
    let entity = place_u32(&mut world, id, 21);
    let before_generation = world.archetype_generation();

    let report = world
        .reload_dynamic_component(id, u64_layout(), |_, source, output| {
            let value = unsafe { *source.as_ref::<u32>() };
            unsafe { output.write(u64::from(value) * 2) };
            Ok::<_, Infallible>(())
        })
        .expect("dynamic migration succeeds");

    assert_eq!(report.values, 1);
    assert_eq!(report.tables, 1);
    assert_eq!(world.components.schema_generation(), 1);
    assert_ne!(world.archetype_generation(), before_generation);
    let value = world
        .get_dynamic_component(entity, id)
        .expect("dynamic component lookup")
        .expect("migrated value");
    assert_eq!(value.layout(), Layout::new::<u64>());
    assert_eq!(unsafe { *value.as_ref::<u64>() }, 42);
}

#[test]
fn failed_migration_keeps_live_value_layout_and_schema_revision() {
    let mut world = World::new();
    let id = register_table_u32(&mut world);
    let entity = place_u32(&mut world, id, 9);
    let before_generation = world.archetype_generation();

    let result = world.reload_dynamic_component(id, u64_layout(), |_, _, _| {
        Err::<(), _>("migration rejected")
    });
    assert!(matches!(
        result,
        Err(DynamicReloadError::Migration { entity: failed, error: "migration rejected" })
            if failed == entity
    ));
    assert_eq!(world.components.schema_generation(), 0);
    assert_eq!(world.archetype_generation(), before_generation);
    let value = world
        .get_dynamic_component(entity, id)
        .expect("dynamic component lookup")
        .expect("original value remains");
    assert_eq!(value.layout(), Layout::new::<u32>());
    assert_eq!(unsafe { *value.as_ref::<u32>() }, 9);
}

#[test]
fn schema_generation_rejects_old_restore_and_forces_fresh_delta_cells() {
    let mut world = World::new();
    let id = register_table_u32(&mut world);
    let old = world.snapshot();

    let report = world
        .reload_dynamic_component(id, u64_layout(), |_, _, output| {
            unsafe { output.write(0_u64) };
            Ok::<_, Infallible>(())
        })
        .expect("empty dynamic migration succeeds");
    assert_eq!(report.values, 0);
    assert_eq!(report.tables, 0);
    assert_eq!(world.components.schema_generation(), 1);

    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.restore(&old);
    }));
    assert!(rejected.is_err(), "a pre-reload snapshot must not restore");

    let current = world.snapshot();
    assert_eq!(current.schema_generation(), 1);
    let rebuilt = old.diff(&current).apply(&old);
    assert!(rebuilt.structurally_eq(&current));
    world.restore(&current);
}

#[test]
fn restore_rejects_a_snapshot_from_another_world() {
    let source = World::new();
    let snapshot = source.snapshot();
    let mut target = World::new();
    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        target.restore(&snapshot);
    }));
    assert!(rejected.is_err());
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Neighbor(u32);
impl Component for Neighbor {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Batch(u32);
impl Component for Batch {
    const STORAGE: StorageType = StorageType::Shared;
    fn install_storage_glue(components: &mut crate::component::Components, id: ComponentId) {
        components.set_shared_box(id, crate::component::shared_box_of::<Self>());
    }
}

#[derive(Clone)]
struct Tracked {
    value: u32,
    drops: Arc<AtomicUsize>,
    panic_once: Option<Arc<AtomicBool>>,
}
impl Drop for Tracked {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
        if self
            .panic_once
            .as_ref()
            .is_some_and(|flag| flag.swap(false, Ordering::SeqCst))
        {
            panic!("runtime destructor panic");
        }
    }
}
fn tracked(value: u32, drops: &Arc<AtomicUsize>) -> Tracked {
    Tracked {
        value,
        drops: drops.clone(),
        panic_once: None,
    }
}

#[test]
fn public_table_and_sparse_insert_replace_mutate_and_track_changes() {
    for storage in [StorageType::Table, StorageType::SparseSet] {
        let mut world = World::new();
        let id = world
            .register_dynamic_component("runtime.scalar", storage, layout::<u32>())
            .unwrap();
        let first = world.spawn(Neighbor(3));
        let second = world.spawn(Neighbor(4));
        let before = world.entities.location(first).unwrap();
        let group = world
            .register_owning_group(&[id, world.components().id_of::<Neighbor>().unwrap()])
            .unwrap();
        let initial_tick = world.change_tick();
        insert(&mut world, first, id, 7_u32);
        assert_eq!(ticks(&world, first, id), ComponentTicks::new(initial_tick));
        assert!(world.owning_group(group).unwrap().contains(first));
        if storage == StorageType::SparseSet {
            assert_eq!(world.entities.location(first).unwrap(), before);
        }
        world.increment_change_tick();
        let now = world.change_tick();
        {
            let mut value = world.get_dynamic_component_mut(first, id).unwrap().unwrap();
            assert_eq!(unsafe { *value.as_ref::<u32>() }, 7);
            assert_eq!(value.ticks(), ComponentTicks::new(initial_tick));
            unsafe { *value.as_mut::<u32>() += 2 };
            assert_eq!(
                value.ticks(),
                ComponentTicks {
                    added: initial_tick,
                    changed: now
                }
            );
        }
        assert_eq!(read::<u32>(&world, first, id), 9);
        if storage == StorageType::Table {
            let loc = world.entities.location(first).unwrap();
            let column = world
                .archetypes
                .get(loc.archetype_id)
                .unwrap()
                .table()
                .column(id)
                .unwrap();
            assert_eq!(
                column.chunk_version(loc.row as usize / column.rows_per_chunk()),
                now
            );
        }
        world.increment_change_tick();
        insert(&mut world, first, id, 12_u32);
        assert_eq!(
            ticks(&world, first, id),
            ComponentTicks {
                added: initial_tick,
                changed: world.change_tick()
            }
        );
        assert_eq!(world.get::<Neighbor>(first).unwrap().0, 3);
        assert_eq!(world.get::<Neighbor>(second).unwrap().0, 4);
        assert!(world.get_dynamic_component(second, id).unwrap().is_none());
    }
}

#[test]
fn runtime_initializer_error_uninitialized_and_panic_preserve_existing_values() {
    for storage in [StorageType::Table, StorageType::SparseSet] {
        let mut world = World::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let id = world
            .register_dynamic_component("runtime.owned", storage, layout::<Tracked>())
            .unwrap();
        let entity = world.spawn(());
        insert(&mut world, entity, id, tracked(7, &drops));
        let before = ticks(&world, entity, id);
        let generation = world.archetype_generation();
        world.increment_change_tick();
        let result = world.insert_dynamic_component(entity, id, |slot| {
            unsafe { slot.write(tracked(9, &drops)) };
            Err("rejected")
        });
        assert!(matches!(
            result,
            Err(DynamicReloadError::Migration {
                error: "rejected",
                ..
            })
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let result = world.insert_dynamic_component(entity, id, |_| Ok::<_, Infallible>(()));
        assert!(matches!(
            result,
            Err(DynamicReloadError::Uninitialized { .. })
        ));
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = world.insert_dynamic_component(entity, id, |slot| -> Result<(), Infallible> {
                unsafe { slot.write(tracked(11, &drops)) };
                panic!("initializer panicked after transfer");
            });
        }));
        assert!(panic.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert_eq!(
            unsafe {
                world
                    .get_dynamic_component(entity, id)
                    .unwrap()
                    .unwrap()
                    .as_ref::<Tracked>()
                    .value
            },
            7
        );
        assert_eq!(ticks(&world, entity, id), before);
        assert_eq!(world.archetype_generation(), generation);
        drop(world);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
    }
}

#[test]
fn table_migration_preserves_other_columns_shared_bindings_locations_and_groups() {
    let mut world = World::new();
    let id = register_table_u32(&mut world);
    let a = world.spawn((Neighbor(1), Batch(8)));
    let b = world.spawn((Neighbor(2), Batch(8)));
    let c = world.spawn(Batch(9));
    for (entity, value) in [(a, 10_u32), (b, 20), (c, 30)] {
        insert(&mut world, entity, id, value);
    }
    let neighbor_ticks = world.get_ticks::<Neighbor>(a).unwrap();
    let old_ticks = ticks(&world, a, id);
    let locations: Vec<_> = [a, b, c]
        .map(|e| world.entities.location(e).unwrap())
        .into();
    let batch = world.components().id_of::<Batch>().unwrap();
    let bindings: Vec<_> = locations
        .iter()
        .map(|l| {
            world
                .archetypes
                .get(l.archetype_id)
                .unwrap()
                .shared_binding(batch)
        })
        .collect();
    let group = world.register_owning_group(&[id]).unwrap();
    let members = world.owning_group(group).unwrap().packed().to_vec();
    let neighbors = world.query::<&Neighbor>();
    assert_eq!(neighbors.iter(&world).map(|n| n.0).sum::<u32>(), 3);
    world.increment_change_tick();
    let report = world
        .reload_dynamic_component(id, layout::<u64>(), |_, input, output| {
            unsafe { output.write(u64::from(*input.as_ref::<u32>()) + 1) };
            Ok::<_, Infallible>(())
        })
        .unwrap();
    assert_eq!(report.values, 3);
    assert!(report.tables >= 2);
    for (i, entity) in [a, b, c].into_iter().enumerate() {
        assert_eq!(world.entities.location(entity).unwrap(), locations[i]);
        assert_eq!(
            world
                .archetypes
                .get(locations[i].archetype_id)
                .unwrap()
                .shared_binding(batch),
            bindings[i]
        );
        assert_eq!(read::<u64>(&world, entity, id), (i as u64 + 1) * 10 + 1);
    }
    assert_eq!(world.get::<Batch>(a).unwrap().0, 8);
    assert_eq!(world.get::<Batch>(c).unwrap().0, 9);
    assert_eq!(world.get::<Neighbor>(a).unwrap().0, 1);
    assert_eq!(world.get::<Neighbor>(b).unwrap().0, 2);
    // The query was warmed against the retired physical tables, and must
    // resolve the replacement columns rather than retain old chunk pointers.
    assert_eq!(neighbors.iter(&world).map(|n| n.0).sum::<u32>(), 3);
    assert_eq!(world.get_ticks::<Neighbor>(a).unwrap(), neighbor_ticks);
    assert_eq!(
        ticks(&world, a, id),
        ComponentTicks {
            added: old_ticks.added,
            changed: world.change_tick()
        }
    );
    assert_eq!(world.owning_group(group).unwrap().packed(), members);
    assert_eq!(
        world
            .shared_components
            .refcount(batch, bindings[0].unwrap()),
        2
    );
}

#[test]
fn sparse_migration_preserves_dense_order_ticks_and_table_locations() {
    let mut world = World::new();
    let id = world
        .register_dynamic_component("runtime.sparse", StorageType::SparseSet, layout::<u32>())
        .unwrap();
    let entities = [
        world.spawn(Neighbor(1)),
        world.spawn(Neighbor(2)),
        world.spawn(Neighbor(3)),
    ];
    // Dense order deliberately differs from entity index order.
    for entity in [entities[2], entities[0], entities[1]] {
        insert(&mut world, entity, id, entity.index());
        world.increment_change_tick();
    }
    let order = world.sparse_sets.get(id).unwrap().entities().to_vec();
    let old_ticks: Vec<_> = order.iter().map(|&e| ticks(&world, e, id)).collect();
    let locations: Vec<_> = order
        .iter()
        .map(|&e| world.entities.location(e).unwrap())
        .collect();
    let mut calls = Vec::new();
    let report = world
        .reload_dynamic_component(id, layout::<u64>(), |entity, input, output| {
            calls.push(entity);
            unsafe { output.write(u64::from(*input.as_ref::<u32>()) + 100) };
            Ok::<_, Infallible>(())
        })
        .unwrap();
    assert_eq!(report.values, 3);
    assert_eq!(report.tables, 0);
    assert_eq!(calls, order);
    assert_eq!(world.sparse_sets.get(id).unwrap().entities(), order);
    for (i, &entity) in order.iter().enumerate() {
        assert_eq!(world.entities.location(entity).unwrap(), locations[i]);
        assert_eq!(
            ticks(&world, entity, id),
            ComponentTicks {
                added: old_ticks[i].added,
                changed: world.change_tick()
            }
        );
        assert_eq!(
            read::<u64>(&world, entity, id),
            u64::from(entity.index()) + 100
        );
    }
}

#[test]
fn migration_error_missing_output_and_panic_drop_all_staged_values_once() {
    for storage in [StorageType::Table, StorageType::SparseSet] {
        for failure in 0..3 {
            let mut world = World::new();
            let id = world
                .register_dynamic_component("runtime.old", storage, layout::<u32>())
                .unwrap();
            let a = world.spawn(());
            let b = world.spawn(());
            insert(&mut world, a, id, 3_u32);
            insert(&mut world, b, id, 5_u32);
            let before = [ticks(&world, a, id), ticks(&world, b, id)];
            let generation = world.archetype_generation();
            let drops = Arc::new(AtomicUsize::new(0));
            let mut calls = 0;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                world.reload_dynamic_component(id, layout::<Tracked>(), |_, input, output| {
                    calls += 1;
                    if calls == 2 && failure == 1 {
                        return Ok(());
                    }
                    unsafe { output.write(tracked(*input.as_ref::<u32>(), &drops)) };
                    if calls == 2 {
                        if failure == 2 {
                            panic!("conversion panic after initialization");
                        }
                        return Err("conversion rejected");
                    }
                    Ok(())
                })
            }));
            match failure {
                0 => assert!(matches!(
                    result,
                    Ok(Err(DynamicReloadError::Migration { .. }))
                )),
                1 => assert!(matches!(
                    result,
                    Ok(Err(DynamicReloadError::Uninitialized { .. }))
                )),
                2 => assert!(result.is_err()),
                _ => unreachable!(),
            }
            assert_eq!(calls, 2);
            assert_eq!(
                drops.load(Ordering::SeqCst),
                if failure == 1 { 1 } else { 2 }
            );
            assert_eq!(world.components().schema_generation(), 0);
            assert_eq!(world.archetype_generation(), generation);
            assert_eq!(
                world.components().info(id).unwrap().layout(),
                Layout::new::<u32>()
            );
            assert_eq!(
                [read::<u32>(&world, a, id), read::<u32>(&world, b, id)],
                [3, 5]
            );
            assert_eq!([ticks(&world, a, id), ticks(&world, b, id)], before);
        }
    }
}

#[test]
fn same_layout_live_migration_invalidates_old_snapshots_and_installs_new_glue() {
    for storage in [StorageType::Table, StorageType::SparseSet] {
        let mut world = World::new();
        let id = world
            .register_dynamic_component("runtime.snapshot", storage, snapshot_layout::<u32>())
            .unwrap();
        let entity = world.spawn(());
        insert(&mut world, entity, id, 17_u32);
        let old = world.snapshot();
        let old_hash = old.state_hash();
        world
            .reload_dynamic_component(id, snapshot_layout::<u32>(), |_, input, output| {
                unsafe { output.write(*input.as_ref::<u32>() + 1) };
                Ok::<_, Infallible>(())
            })
            .unwrap();
        let rejected =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| world.restore(&old)));
        assert!(rejected.is_err());
        assert_eq!(read::<u32>(&world, entity, id), 18);
        let new = world.snapshot();
        assert_ne!(old_hash, new.state_hash());
        assert_eq!(
            old.state_hash(),
            old_hash,
            "old snapshot retains its own ABI/glue"
        );
        insert(&mut world, entity, id, 99_u32);
        world.restore(&new);
        assert_eq!(read::<u32>(&world, entity, id), 18);
        let rebuilt = old.diff(&new).apply(&old);
        assert_eq!(rebuilt.schema_generation(), new.schema_generation());
        assert_eq!(rebuilt.state_hash(), new.state_hash());
        assert_eq!(rebuilt.entity_count(), new.entity_count());
        assert_eq!(rebuilt.column_count(), new.column_count());
    }
}

#[test]
fn old_destructor_panic_keeps_published_world_and_drops_remaining_retired_values() {
    for storage in [StorageType::Table, StorageType::SparseSet] {
        let mut world = World::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let panic_once = Arc::new(AtomicBool::new(true));
        let id = world
            .register_dynamic_component("runtime.retired", storage, layout::<Tracked>())
            .unwrap();
        let entities = [world.spawn(()), world.spawn(())];
        for entity in entities {
            let mut value = tracked(entity.index(), &drops);
            value.panic_once = Some(panic_once.clone());
            insert(&mut world, entity, id, value);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.reload_dynamic_component(id, layout::<u64>(), |_, input, output| {
                unsafe { output.write(u64::from(input.as_ref::<Tracked>().value)) };
                Ok::<_, Infallible>(())
            })
        }));
        assert!(result.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert_eq!(world.components().schema_generation(), 1);
        for entity in entities {
            assert_eq!(read::<u64>(&world, entity, id), u64::from(entity.index()));
        }
        insert(&mut world, entities[0], id, 88_u64);
        assert_eq!(read::<u64>(&world, entities[0], id), 88);
    }
}

#[test]
fn replacement_destructor_panic_publishes_new_value_and_membership() {
    for storage in [StorageType::Table, StorageType::SparseSet] {
        let mut world = World::new();
        let old_drops = Arc::new(AtomicUsize::new(0));
        let new_drops = Arc::new(AtomicUsize::new(0));
        let id = world
            .register_dynamic_component("runtime.replace", storage, layout::<Tracked>())
            .unwrap();
        let entity = world.spawn(());
        let group = world.register_owning_group(&[id]).unwrap();
        let mut old = tracked(1, &old_drops);
        old.panic_once = Some(Arc::new(AtomicBool::new(true)));
        insert(&mut world, entity, id, old);
        let added = ticks(&world, entity, id).added;
        world.increment_change_tick();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            insert(&mut world, entity, id, tracked(2, &new_drops));
        }));
        assert!(panic.is_err());
        assert_eq!(old_drops.load(Ordering::SeqCst), 1);
        assert_eq!(new_drops.load(Ordering::SeqCst), 0);
        assert!(world.owning_group(group).unwrap().contains(entity));
        assert_eq!(
            unsafe {
                world
                    .get_dynamic_component(entity, id)
                    .unwrap()
                    .unwrap()
                    .as_ref::<Tracked>()
                    .value
            },
            2
        );
        assert_eq!(
            ticks(&world, entity, id),
            ComponentTicks {
                added,
                changed: world.change_tick()
            }
        );
        drop(world);
        assert_eq!(old_drops.load(Ordering::SeqCst), 1);
        assert_eq!(new_drops.load(Ordering::SeqCst), 1);
    }
}

#[repr(align(256))]
#[derive(Clone, Copy)]
struct Aligned([u64; 33]);
#[repr(align(512))]
struct AlignedZst;

#[test]
fn runtime_slots_and_table_sparse_storage_support_high_alignment_and_zsts() {
    for storage in [StorageType::Table, StorageType::SparseSet] {
        let mut world = World::new();
        let id = world
            .register_dynamic_component("runtime.aligned", storage, layout::<Aligned>())
            .unwrap();
        let entities: Vec<_> = (0..80).map(|_| world.spawn(())).collect();
        for (i, &entity) in entities.iter().enumerate() {
            insert(&mut world, entity, id, Aligned([i as u64; 33]));
        }
        world
            .reload_dynamic_component(id, layout::<Aligned>(), |_, input, output| {
                assert_eq!(input.as_ptr() as usize % 256, 0);
                assert_eq!(output.as_mut_ptr() as usize % 256, 0);
                let mut value = unsafe { *input.as_ref::<Aligned>() };
                value.0[32] += 1;
                unsafe { output.write(value) };
                Ok::<_, Infallible>(())
            })
            .unwrap();
        for (i, &entity) in entities.iter().enumerate() {
            let value = read::<Aligned>(&world, entity, id);
            assert_eq!(value.0[0], i as u64);
            assert_eq!(value.0[32], i as u64 + 1);
        }
        world
            .reload_dynamic_component(id, layout::<AlignedZst>(), |_, _, output| {
                assert_eq!(output.as_mut_ptr() as usize % 512, 0);
                unsafe { output.write(AlignedZst) };
                Ok::<_, Infallible>(())
            })
            .unwrap();
        for entity in entities {
            let value = world.get_dynamic_component(entity, id).unwrap().unwrap();
            assert_eq!(value.layout(), Layout::new::<AlignedZst>());
            assert_eq!(value.as_ptr() as usize % 512, 0);
            let _ = unsafe { value.as_ref::<AlignedZst>() };
            assert!(world.remove_dynamic_component(entity, id).unwrap());
        }
    }
}

#[test]
fn invalid_descriptors_typed_shared_unknown_and_dead_entities_are_rejected() {
    let mut world = World::new();
    let count = world.components().len();
    let unpadded = unsafe {
        DynamicComponentLayout::new(Layout::from_size_align(1, 8).unwrap(), None, None, None)
    };
    assert_eq!(
        world.register_dynamic_component("invalid", StorageType::Table, unpadded),
        Err(DynamicComponentError::UnpaddedLayout)
    );
    assert_eq!(
        world.register_dynamic_component("shared", StorageType::Shared, layout::<u32>()),
        Err(DynamicComponentError::SharedStorage)
    );
    assert_eq!(world.components().len(), count);
    let entity = world.spawn(Neighbor(1));
    let typed = world.components().id_of::<Neighbor>().unwrap();
    assert!(
        matches!(world.get_dynamic_component(entity, typed), Err(DynamicComponentError::RustType(id)) if id == typed)
    );
    let shared = world.register_component::<Batch>();
    assert!(matches!(
        world.get_dynamic_component(entity, shared),
        Err(DynamicComponentError::RustType(_))
    ));
    let raw_shared = world.components.register_dynamic(
        "legacy.shared",
        Layout::new::<u32>(),
        StorageType::Shared,
        None,
    );
    assert!(matches!(
        world.reload_dynamic_component(raw_shared, layout::<u32>(), |_, _, _| Ok::<_, Infallible>(
            ()
        )),
        Err(DynamicReloadError::Component(
            DynamicComponentError::SharedStorage
        ))
    ));
    let unknown = ComponentId::new(u32::MAX);
    assert!(
        matches!(world.get_dynamic_component(entity, unknown), Err(DynamicComponentError::UnknownComponent(id)) if id == unknown)
    );
    let id = register_table_u32(&mut world);
    assert!(world.despawn(entity));
    assert!(!world
        .insert_dynamic_component(entity, id, |_| -> Result<(), Infallible> {
            panic!("dead entity must not initialize")
        })
        .unwrap());
    assert!(world
        .get_dynamic_component_mut(entity, id)
        .unwrap()
        .is_none());
}

#[test]
fn removal_preserves_neighbors_shared_bindings_ticks_and_survivor_indices() {
    for storage in [StorageType::Table, StorageType::SparseSet] {
        let mut world = World::new();
        let id = world
            .register_dynamic_component("remove", storage, layout::<u32>())
            .unwrap();
        let entities = [
            world.spawn((Neighbor(1), Batch(7))),
            world.spawn((Neighbor(2), Batch(7))),
        ];
        for entity in entities {
            insert(&mut world, entity, id, entity.index());
            world.increment_change_tick();
        }
        let survivor_ticks = ticks(&world, entities[1], id);
        let neighbor_ticks = world.get_ticks::<Neighbor>(entities[0]).unwrap();
        let group = world.register_owning_group(&[id]).unwrap();
        assert!(world.remove_dynamic_component(entities[0], id).unwrap());
        assert!(!world.remove_dynamic_component(entities[0], id).unwrap());
        assert!(world
            .get_dynamic_component(entities[0], id)
            .unwrap()
            .is_none());
        assert!(!world.owning_group(group).unwrap().contains(entities[0]));
        assert!(world.owning_group(group).unwrap().contains(entities[1]));
        assert_eq!(read::<u32>(&world, entities[1], id), entities[1].index());
        assert_eq!(ticks(&world, entities[1], id), survivor_ticks);
        assert_eq!(
            world.get_ticks::<Neighbor>(entities[0]).unwrap(),
            neighbor_ticks
        );
        assert_eq!(world.get::<Neighbor>(entities[0]).unwrap().0, 1);
        assert_eq!(world.get::<Batch>(entities[0]).unwrap().0, 7);
        let batch = world.components().id_of::<Batch>().unwrap();
        let location = world.entities.location(entities[0]).unwrap();
        let binding = world
            .archetypes
            .get(location.archetype_id)
            .unwrap()
            .shared_binding(batch)
            .unwrap();
        assert_eq!(world.shared_components.refcount(batch, binding), 2);
        insert(&mut world, entities[0], id, 99_u32);
        assert_eq!(read::<u32>(&world, entities[0], id), 99);
        assert!(world.despawn(entities[0]));
        assert!(!world.remove_dynamic_component(entities[0], id).unwrap());
        assert!(matches!(
            world.remove_dynamic_component(entities[1], batch),
            Err(DynamicComponentError::RustType(_))
        ));
        assert!(matches!(
            world.remove_dynamic_component(entities[1], ComponentId::new(u32::MAX)),
            Err(DynamicComponentError::UnknownComponent(_))
        ));
    }
}

#[test]
fn removal_drop_panic_commits_membership_and_keeps_sparse_ticks_consistent() {
    for storage in [StorageType::Table, StorageType::SparseSet] {
        let mut world = World::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let id = world
            .register_dynamic_component("drop.remove", storage, layout::<Tracked>())
            .unwrap();
        let entities = [
            world.spawn(Neighbor(1)),
            world.spawn(Neighbor(2)),
            world.spawn(Neighbor(3)),
        ];
        for (i, entity) in entities.into_iter().enumerate() {
            let mut value = tracked(i as u32, &drops);
            if i == 0 {
                value.panic_once = Some(Arc::new(AtomicBool::new(true)));
            }
            insert(&mut world, entity, id, value);
            world.increment_change_tick();
        }
        let survivor_ticks = ticks(&world, entities[2], id);
        let group = world.register_owning_group(&[id]).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.remove_dynamic_component(entities[0], id)
        }));
        assert!(result.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(!world.owning_group(group).unwrap().contains(entities[0]));
        assert!(world
            .get_dynamic_component(entities[0], id)
            .unwrap()
            .is_none());
        assert_eq!(ticks(&world, entities[2], id), survivor_ticks);
        assert_eq!(
            unsafe {
                world
                    .get_dynamic_component(entities[2], id)
                    .unwrap()
                    .unwrap()
                    .as_ref::<Tracked>()
                    .value
            },
            2
        );
        assert!(world.remove_dynamic_component(entities[2], id).unwrap());
        insert(&mut world, entities[0], id, tracked(9, &drops));
        drop(world);
        assert_eq!(drops.load(Ordering::SeqCst), 4);
    }
}

#[derive(Default)]
struct LifecycleLog(Vec<&'static str>);
impl crate::resource::Resource for LifecycleLog {}

fn record_add(ctx: HookContext<'_>) {
    ctx.world.resource_mut::<LifecycleLog>().0.push("add");
}
fn record_insert(ctx: HookContext<'_>) {
    ctx.world.resource_mut::<LifecycleLog>().0.push("insert");
}
fn record_replace(ctx: HookContext<'_>) {
    assert!(ctx
        .world
        .get_dynamic_component(ctx.entity, ctx.component)
        .unwrap()
        .is_some());
    ctx.world.resource_mut::<LifecycleLog>().0.push("replace");
}
fn record_remove(ctx: HookContext<'_>) {
    assert!(ctx
        .world
        .get_dynamic_component(ctx.entity, ctx.component)
        .unwrap()
        .is_some());
    ctx.world.resource_mut::<LifecycleLog>().0.push("remove");
}

fn required_u32(value: u32) -> crate::component::RequiredCtor {
    Arc::new(move |out| {
        let mut value = core::mem::ManuallyDrop::new(value);
        out((&mut *value as *mut u32).cast());
    })
}

#[test]
fn lifecycle_order_and_transitive_requirements_match_typed_insertion_and_removal() {
    for storage in [StorageType::Table, StorageType::SparseSet] {
        let mut world = World::new();
        world.insert_resource(LifecycleLog::default());
        let id = world
            .register_dynamic_component("root", storage, layout::<u32>())
            .unwrap();
        let required = world
            .register_dynamic_component("required", StorageType::Table, layout::<u32>())
            .unwrap();
        let transitive = world
            .register_dynamic_component("transitive", StorageType::SparseSet, layout::<u32>())
            .unwrap();
        unsafe {
            world
                .register_dynamic_required_component(id, required, required_u32(10))
                .unwrap();
            world
                .register_dynamic_required_component(required, transitive, required_u32(20))
                .unwrap();
        }
        world
            .register_dynamic_component_hooks(
                id,
                ComponentHooks::new()
                    .with_on_add(record_add)
                    .with_on_insert(record_insert)
                    .with_on_replace(record_replace)
                    .with_on_remove(record_remove),
            )
            .unwrap();
        let entity = world.spawn(());
        insert(&mut world, entity, required, 77_u32);
        insert(&mut world, entity, id, 1_u32);
        assert_eq!(read::<u32>(&world, entity, required), 77);
        assert_eq!(read::<u32>(&world, entity, transitive), 20);
        insert(&mut world, entity, id, 2_u32);
        assert!(world.remove_dynamic_component(entity, id).unwrap());
        assert_eq!(
            world.resource::<LifecycleLog>().0,
            ["add", "insert", "replace", "insert", "replace", "remove"]
        );
        assert_eq!(read::<u32>(&world, entity, required), 77);
        assert_eq!(read::<u32>(&world, entity, transitive), 20);
        assert!(matches!(
            world.reload_dynamic_component(
                id,
                layout::<u64>(),
                |_, _, _| -> Result<(), Infallible> { panic!("must reject hooks first") }
            ),
            Err(DynamicReloadError::Component(
                DynamicComponentError::HasHooks(_)
            ))
        ));
        assert!(matches!(
            world.reload_dynamic_component(
                required,
                layout::<u64>(),
                |_, _, _| -> Result<(), Infallible> {
                    panic!("must reject required constructors first")
                }
            ),
            Err(DynamicReloadError::Component(
                DynamicComponentError::RequiredConstructor(_)
            ))
        ));
    }
}

#[test]
fn required_constructor_panic_after_yield_discards_all_staged_values() {
    for storage in [StorageType::Table, StorageType::SparseSet] {
        let mut world = World::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let id = world
            .register_dynamic_component("root", storage, layout::<Tracked>())
            .unwrap();
        let required = world
            .register_dynamic_component("required", storage, layout::<Tracked>())
            .unwrap();
        let ctor_drops = drops.clone();
        unsafe {
            world
                .register_dynamic_required_component(
                    id,
                    required,
                    Arc::new(move |out| {
                        let mut value = core::mem::ManuallyDrop::new(tracked(10, &ctor_drops));
                        out((&mut *value as *mut Tracked).cast());
                        panic!("valid owned yield followed by panic");
                    }),
                )
                .unwrap();
        }
        let entity = world.spawn(Neighbor(5));
        let location = world.entities.location(entity).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            insert(&mut world, entity, id, tracked(1, &drops))
        }));
        assert!(result.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert_eq!(world.entities.location(entity).unwrap(), location);
        assert!(world.get_dynamic_component(entity, id).unwrap().is_none());
        assert!(world
            .get_dynamic_component(entity, required)
            .unwrap()
            .is_none());
        assert_eq!(world.get::<Neighbor>(entity).unwrap().0, 5);
        drop(world);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn staged_output_drop_panic_on_returned_error_cleans_other_outputs_without_commit() {
    for storage in [StorageType::Table, StorageType::SparseSet] {
        let mut world = World::new();
        let id = world
            .register_dynamic_component("staged", storage, layout::<u32>())
            .unwrap();
        let entities = [world.spawn(()), world.spawn(()), world.spawn(())];
        for entity in entities {
            insert(&mut world, entity, id, 7_u32);
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let mut calls = 0;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.reload_dynamic_component(id, layout::<Tracked>(), |_, _, out| {
                calls += 1;
                if calls == 3 {
                    return Err("reject");
                }
                let mut value = tracked(calls, &drops);
                if calls == 1 {
                    value.panic_once = Some(Arc::new(AtomicBool::new(true)));
                }
                unsafe { out.write(value) };
                Ok(())
            })
        }));
        assert!(result.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert_eq!(world.components().schema_generation(), 0);
        for entity in entities {
            assert_eq!(read::<u32>(&world, entity, id), 7);
        }
    }
}

#[test]
fn removal_pre_hook_panic_keeps_value_and_post_insert_hook_panic_keeps_publication() {
    fn panic_hook(_: HookContext<'_>) {
        panic!("lifecycle panic");
    }
    for storage in [StorageType::Table, StorageType::SparseSet] {
        let mut world = World::new();
        let id = world
            .register_dynamic_component("hooks", storage, layout::<u32>())
            .unwrap();
        let entity = world.spawn(());
        insert(&mut world, entity, id, 10_u32);
        let before = ticks(&world, entity, id);
        world
            .register_dynamic_component_hooks(id, ComponentHooks::new().with_on_remove(panic_hook))
            .unwrap();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || world.remove_dynamic_component(entity, id)
        ))
        .is_err());
        assert_eq!(read::<u32>(&world, entity, id), 10);
        assert_eq!(ticks(&world, entity, id), before);
        world
            .register_dynamic_component_hooks(id, ComponentHooks::new().with_on_insert(panic_hook))
            .unwrap();
        world.increment_change_tick();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| insert(
                &mut world, entity, id, 11_u32
            )))
            .is_err()
        );
        assert_eq!(read::<u32>(&world, entity, id), 11);
        assert_eq!(ticks(&world, entity, id).added, before.added);
        assert_eq!(ticks(&world, entity, id).changed, world.change_tick());
    }
}

#[test]
fn pre_hook_world_replacement_aborts_without_using_old_schema_or_entity() {
    fn replace_world(ctx: HookContext<'_>) {
        *ctx.world = World::new();
    }
    for storage in [StorageType::Table, StorageType::SparseSet] {
        for remove in [false, true] {
            let mut world = World::new();
            let drops = Arc::new(AtomicUsize::new(0));
            let id = world
                .register_dynamic_component("context", storage, layout::<Tracked>())
                .unwrap();
            let entity = world.spawn(());
            insert(&mut world, entity, id, tracked(1, &drops));
            let old_world = world.id();
            world
                .register_dynamic_component_hooks(
                    id,
                    ComponentHooks::new().with_on_replace(replace_world),
                )
                .unwrap();
            if remove {
                assert_eq!(
                    world.remove_dynamic_component(entity, id),
                    Err(DynamicComponentError::ContextChanged)
                );
            } else {
                let result = world.insert_dynamic_component(entity, id, |out| {
                    unsafe { out.write(tracked(2, &drops)) };
                    Ok::<_, Infallible>(())
                });
                assert_eq!(
                    result,
                    Err(DynamicReloadError::Component(
                        DynamicComponentError::ContextChanged
                    ))
                );
            }
            assert_ne!(world.id(), old_world);
            assert_eq!(world.components().len(), 0);
            assert_eq!(drops.load(Ordering::SeqCst), if remove { 1 } else { 2 });
            let fresh = world.spawn(Neighbor(42));
            assert_eq!(world.get::<Neighbor>(fresh).unwrap().0, 42);
        }
    }
}

#[test]
fn removal_revalidates_location_after_pre_hook_structural_changes() {
    fn move_entity(ctx: HookContext<'_>) {
        assert!(ctx.world.insert(ctx.entity, Neighbor(99)));
    }
    fn despawn_entity(ctx: HookContext<'_>) {
        // Disable this hook before despawning the same component recursively.
        ctx.world
            .register_dynamic_component_hooks(ctx.component, ComponentHooks::new())
            .unwrap();
        assert!(ctx.world.despawn(ctx.entity));
    }
    for storage in [StorageType::Table, StorageType::SparseSet] {
        for despawn in [false, true] {
            let mut world = World::new();
            let id = world
                .register_dynamic_component("reentry", storage, layout::<u32>())
                .unwrap();
            let entity = world.spawn(());
            insert(&mut world, entity, id, 10_u32);
            world
                .register_dynamic_component_hooks(
                    id,
                    ComponentHooks::new().with_on_replace(if despawn {
                        despawn_entity
                    } else {
                        move_entity
                    }),
                )
                .unwrap();
            assert_eq!(
                world.remove_dynamic_component(entity, id).unwrap(),
                !despawn
            );
            assert!(world.get_dynamic_component(entity, id).unwrap().is_none());
            if despawn {
                assert!(!world.contains(entity));
            } else {
                assert_eq!(world.get::<Neighbor>(entity).unwrap().0, 99);
            }
        }
    }
}
