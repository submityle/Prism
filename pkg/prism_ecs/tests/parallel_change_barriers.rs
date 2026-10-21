//! Parallel table and sparse change-barrier acceptance tests.

#![cfg(feature = "multi_thread")]

use prism_ecs::component::StorageType;
use prism_ecs::prelude::*;
use prism_tasks::TaskPool;
use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Component, Clone)]
struct TableValue {
    ordinal: usize,
    value: u32,
    // A small fixture still needs more than one physical table chunk.
    _padding: [u64; 256],
}

#[derive(Clone, Copy)]
struct SparseValue(u32);

impl Component for SparseValue {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[derive(Component, Clone, Copy)]
struct Marker;

#[derive(Resource)]
struct Workers(TaskPool);

fn seed(count: usize) -> (World, Vec<Entity>) {
    let mut world = World::new();
    let entities = (0..count)
        .map(|ordinal| {
            world.spawn((
                TableValue {
                    ordinal,
                    value: ordinal as u32,
                    _padding: [0; 256],
                },
                SparseValue(ordinal as u32),
            ))
        })
        .collect();
    (world, entities)
}

fn ordered(mut entities: Vec<Entity>) -> Vec<Entity> {
    entities.sort_unstable_by_key(|entity| (entity.index(), entity.generation()));
    entities
}

fn changed<T: Component>(world: &mut World, last: Tick, this: Tick) -> Vec<Entity> {
    let query = world.query_filtered::<Entity, Changed<T>>();
    ordered(query.iter_with_ticks(world, last, this).collect())
}

fn dirty_windows(world: &mut World, entities: &[Entity]) -> Vec<DirtyChunk> {
    let mut expected = Vec::new();
    for &entity in entities {
        let location = world.entities().location(entity).expect("live entity");
        let archetype = world
            .archetypes()
            .get(location.archetype_id)
            .expect("live archetype");
        let rows_per_chunk = archetype.table().rows_per_chunk();
        let start = location.row as usize / rows_per_chunk * rows_per_chunk;
        expected.push(DirtyChunk {
            archetype: location.archetype_id,
            start,
            end: (start + rows_per_chunk).min(archetype.table().len()),
        });
    }
    expected.sort_unstable_by_key(|chunk| (chunk.archetype, chunk.start));
    expected.dedup();
    expected
}

fn advance(world: &mut World) -> (Tick, Tick) {
    let last = world.change_tick();
    world.set_last_change_tick(last);
    let this = world.increment_change_tick();
    (last, this)
}

fn install_observers(
    world: &mut World,
) -> (Arc<Mutex<Vec<Vec<Entity>>>>, Arc<Mutex<Vec<Vec<Entity>>>>) {
    let table_batches = Arc::new(Mutex::new(Vec::new()));
    let sparse_batches = Arc::new(Mutex::new(Vec::new()));

    let table_log = Arc::clone(&table_batches);
    world.observe_changed::<TableValue, _>(move |context| {
        table_log
            .lock()
            .unwrap()
            .push(ordered(context.changed().to_vec()));
    });
    let sparse_log = Arc::clone(&sparse_batches);
    world.observe_changed::<SparseValue, _>(move |context| {
        sparse_log
            .lock()
            .unwrap()
            .push(ordered(context.changed().to_vec()));
    });

    (table_batches, sparse_batches)
}

fn selected_sparse(ordinal: usize) -> bool {
    ordinal % 3 == 0 || ordinal == 17
}

fn snapshot_values(world: &World, entities: &[Entity]) -> Vec<(Entity, u32, u32)> {
    entities
        .iter()
        .map(|&entity| {
            (
                entity,
                world.get::<TableValue>(entity).unwrap().value,
                world.get::<SparseValue>(entity).unwrap().0,
            )
        })
        .collect()
}

fn scheduled_writer(
    mut query: Query<(Entity, &mut TableValue, &mut SparseValue)>,
    workers: Res<Workers>,
) {
    query.par_for_each_mut(&workers.0, 1, |(_entity, mut table, mut sparse)| {
        table.value += 1;
        sparse.0 += 1;
    });
}

#[test]
fn parallel_table_and_sparse_writes_match_serial_change_barriers() {
    const COUNT: usize = 20;

    fn run(
        parallel: bool,
    ) -> (
        Vec<(Entity, u32, u32)>,
        Vec<Entity>,
        Vec<Entity>,
        Vec<DirtyChunk>,
        Vec<DirtyChunk>,
        Vec<Vec<Entity>>,
        Vec<Vec<Entity>>,
        usize,
        usize,
    ) {
        let (mut world, entities) = seed(COUNT);
        let table_query = world.query::<&TableValue>();
        let sparse_query = world.query::<&SparseValue>();
        let baseline = world.change_tick();
        let _ = table_query.dirty_chunks_with_stats(&world, baseline, baseline);
        let _ = sparse_query.dirty_chunks_with_stats(&world, baseline, baseline);
        let (table_log, sparse_log) = install_observers(&mut world);
        let (last, this) = advance(&mut world);
        let table_work = Arc::new(AtomicUsize::new(0));
        let sparse_work = Arc::new(AtomicUsize::new(0));
        let pool = TaskPool::with_threads(4);

        if parallel {
            let table_work_ref = Arc::clone(&table_work);
            let table_state = world.query::<&mut TableValue>();
            table_state.par_chunks_mut(&mut world, &pool, 1, move |slice| {
                table_work_ref.fetch_add(1, Ordering::Relaxed);
                for value in slice {
                    value.value += 100;
                }
            });
            let sparse_work_ref = Arc::clone(&sparse_work);
            let sparse_state = world.query::<(&TableValue, &mut SparseValue)>();
            sparse_state.par_for_each_mut(&mut world, &pool, 1, move |(table, mut sparse)| {
                if selected_sparse(table.ordinal) {
                    sparse.0 += 1000;
                    sparse_work_ref.fetch_add(1, Ordering::Relaxed);
                }
            });
        } else {
            let table_state = world.query::<&mut TableValue>();
            for mut value in table_state.iter_mut(&mut world) {
                value.value += 100;
            }
            let sparse_state = world.query::<(&TableValue, &mut SparseValue)>();
            for (table, mut sparse) in sparse_state.iter_mut(&mut world) {
                if selected_sparse(table.ordinal) {
                    sparse.0 += 1000;
                    sparse_work.fetch_add(1, Ordering::Relaxed);
                }
            }
            table_work.store(1, Ordering::Relaxed);
        }

        let changed_table = changed::<TableValue>(&mut world, last, this);
        let changed_sparse = changed::<SparseValue>(&mut world, last, this);
        let table_report = table_query.dirty_chunks(&world, last, this);
        let sparse_report = sparse_query.dirty_chunks(&world, last, this);
        assert_eq!(world.flush_changed_observers(), 2);
        let table_batches = table_log.lock().unwrap().clone();
        let sparse_batches = sparse_log.lock().unwrap().clone();

        (
            snapshot_values(&world, &entities),
            changed_table,
            changed_sparse,
            table_report,
            sparse_report,
            table_batches,
            sparse_batches,
            table_work.load(Ordering::Relaxed),
            sparse_work.load(Ordering::Relaxed),
        )
    }

    let serial = run(false);
    let parallel = run(true);
    assert_eq!(serial.0, parallel.0, "parallel writes preserve values");
    assert_eq!(serial.1, parallel.1, "table Changed oracle is stable");
    assert_eq!(serial.2, parallel.2, "sparse Changed oracle is stable");
    assert_eq!(serial.3, parallel.3, "table dirty windows are stable");
    assert_eq!(serial.4, parallel.4, "sparse dirty windows are stable");
    assert_eq!(serial.5, parallel.5, "table observer delivery is stable");
    assert_eq!(serial.6, parallel.6, "sparse observer delivery is stable");

    let (
        entity_values,
        _,
        _,
        table_report,
        _,
        table_batches,
        sparse_batches,
        table_work,
        sparse_work,
    ) = parallel;
    assert_eq!(
        table_work,
        table_report.len(),
        "one work item per physical table chunk"
    );
    assert!(
        table_work > 1,
        "fixture spans multiple physical table chunks"
    );
    assert_eq!(
        sparse_work, 8,
        "one row work item per selected sparse write"
    );
    assert_eq!(
        table_batches,
        vec![ordered(
            entity_values.iter().map(|(entity, _, _)| *entity).collect(),
        )]
    );
    assert_eq!(
        sparse_batches,
        vec![ordered(
            entity_values
                .iter()
                .enumerate()
                .filter_map(|(ordinal, (entity, _, _))| selected_sparse(ordinal).then_some(*entity))
                .collect(),
        )]
    );
}

#[test]
fn structural_migration_and_swap_remove_keep_parallel_oracles_exact() {
    let (mut world, entities) = seed(14);
    world.insert(entities[0], Marker);
    world.insert(entities[1], Marker);
    assert!(world.get::<Marker>(entities[0]).is_some());
    assert!(world.get::<Marker>(entities[1]).is_some());
    assert!(world.remove::<Marker>(entities[0]));
    assert!(world.get::<Marker>(entities[0]).is_none());
    assert!(world.get::<Marker>(entities[1]).is_some());

    let table_query = world.query::<&TableValue>();
    let sparse_query = world.query::<&SparseValue>();
    let baseline = world.change_tick();
    let _ = table_query.dirty_chunks_with_stats(&world, baseline, baseline);
    let _ = sparse_query.dirty_chunks_with_stats(&world, baseline, baseline);
    let (table_log, sparse_log) = install_observers(&mut world);
    let (last, this) = advance(&mut world);
    let selected = [entities[1], entities[4], entities[9]];
    let pool = TaskPool::with_threads(3);
    let state = world.query::<(Entity, &mut TableValue, &mut SparseValue)>();
    state.par_for_each_mut(&mut world, &pool, 1, |(entity, mut table, mut sparse)| {
        if selected.contains(&entity) {
            table.value += 7;
            sparse.0 += 11;
        }
    });

    assert_eq!(
        changed::<TableValue>(&mut world, last, this),
        selected.to_vec()
    );
    assert_eq!(
        changed::<SparseValue>(&mut world, last, this),
        selected.to_vec()
    );
    assert_eq!(
        table_query.dirty_chunks(&world, last, this),
        dirty_windows(&mut world, &selected),
    );
    assert_eq!(
        sparse_query.dirty_chunks(&world, last, this),
        dirty_windows(&mut world, &selected),
    );
    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(*table_log.lock().unwrap(), vec![selected.to_vec()]);
    assert_eq!(*sparse_log.lock().unwrap(), vec![selected.to_vec()]);
}

#[test]
fn schedule_run_and_parallel_run_preserve_parallel_barriers() {
    fn run(parallel: bool) -> (Vec<(Entity, u32, u32)>, Vec<Vec<Entity>>, Vec<Vec<Entity>>) {
        let (mut world, entities) = seed(10);
        let table_query = world.query::<&TableValue>();
        let sparse_query = world.query::<&SparseValue>();
        let tick = world.change_tick();
        let _ = table_query.dirty_chunks_with_stats(&world, tick, tick);
        let _ = sparse_query.dirty_chunks_with_stats(&world, tick, tick);
        let (table_log, sparse_log) = install_observers(&mut world);
        world.insert_resource(Workers(TaskPool::with_threads(if parallel {
            4
        } else {
            1
        })));

        let mut schedule = Schedule::new();
        schedule.add_systems(scheduled_writer);
        if parallel {
            let pool = TaskPool::with_threads(4);
            schedule.run_parallel(&pool, &mut world);
        } else {
            schedule.run(&mut world);
        }

        (
            snapshot_values(&world, &entities),
            table_log.lock().unwrap().clone(),
            sparse_log.lock().unwrap().clone(),
        )
    }

    let serial = run(false);
    let parallel = run(true);
    assert_eq!(
        serial, parallel,
        "schedule executor preserves write barriers"
    );
    let expected = ordered(parallel.0.iter().map(|(entity, _, _)| *entity).collect());
    assert_eq!(parallel.1, vec![expected]);
    assert_eq!(parallel.2, parallel.1);
}

#[test]
fn parallel_row_panic_unwinds_and_the_barrier_can_be_reused() {
    let (mut world, entities) = seed(12);
    let query = world.query::<&TableValue>();
    let baseline = world.change_tick();
    let _ = query.dirty_chunks_with_stats(&world, baseline, baseline);
    let (last, this) = advance(&mut world);
    let completed = Arc::new(Mutex::new(Vec::new()));
    let panic_entity = entities[3];
    let pool = TaskPool::with_threads(4);

    let completed_ref = Arc::clone(&completed);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = world.query::<(Entity, &mut TableValue)>();
        state.par_for_each_mut(&mut world, &pool, 1, move |(entity, mut value)| {
            if entity == panic_entity {
                panic!("parallel row acceptance panic");
            }
            value.value += 1;
            completed_ref.lock().unwrap().push(entity);
        });
    }));
    assert!(result.is_err());

    let expected = ordered(completed.lock().unwrap().clone());
    let unique: BTreeSet<_> = expected.iter().copied().collect();
    assert_eq!(unique.len(), expected.len(), "each row is one work item");
    assert_eq!(changed::<TableValue>(&mut world, last, this), expected);
    assert_eq!(completed.lock().unwrap().len(), expected.len());

    let (retry_last, retry_this) = advance(&mut world);
    let work = Arc::new(AtomicUsize::new(0));
    let work_ref = Arc::clone(&work);
    let retry_state = world.query::<&mut TableValue>();
    retry_state.par_for_each_mut(&mut world, &pool, 1, move |mut value| {
        value.value += 1;
        work_ref.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(work.load(Ordering::Relaxed), entities.len());
    assert_eq!(
        changed::<TableValue>(&mut world, retry_last, retry_this),
        ordered(entities.clone())
    );
    assert!(
        world.insert(entities[0], Marker),
        "world remains structurally usable"
    );
}

#[test]
fn parallel_chunk_panic_reports_stamped_slices_and_recovers() {
    let (mut world, entities) = seed(12);
    let query = world.query::<&TableValue>();
    let baseline = world.change_tick();
    let _ = query.dirty_chunks_with_stats(&world, baseline, baseline);
    let (last, this) = advance(&mut world);
    let stamped = Arc::new(Mutex::new(Vec::new()));
    let panic_ordinal = 3;
    let pool = TaskPool::with_threads(4);

    let stamped_ref = Arc::clone(&stamped);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = world.query::<&mut TableValue>();
        state.par_chunks_mut(&mut world, &pool, 1, move |slice| {
            stamped_ref
                .lock()
                .unwrap()
                .push(slice.iter().map(|value| value.ordinal).collect::<Vec<_>>());
            for value in slice {
                value.value += 1;
                if value.ordinal == panic_ordinal {
                    panic!("parallel chunk acceptance panic");
                }
            }
        });
    }));
    assert!(result.is_err());

    let expected_ordinals: BTreeSet<_> =
        stamped.lock().unwrap().iter().flatten().copied().collect();
    let expected: Vec<_> = entities
        .iter()
        .enumerate()
        .filter_map(|(ordinal, &entity)| expected_ordinals.contains(&ordinal).then_some(entity))
        .collect();
    assert_eq!(changed::<TableValue>(&mut world, last, this), expected);

    let (retry_last, retry_this) = advance(&mut world);
    let work = Arc::new(AtomicUsize::new(0));
    let work_ref = Arc::clone(&work);
    let retry_state = world.query::<&mut TableValue>();
    retry_state.par_chunks_mut(&mut world, &pool, 1, move |slice| {
        work_ref.fetch_add(1, Ordering::Relaxed);
        for value in slice {
            value.value += 1;
        }
    });
    assert_eq!(
        work.load(Ordering::Relaxed),
        query.dirty_chunks(&world, retry_last, retry_this).len()
    );
    assert_eq!(
        changed::<TableValue>(&mut world, retry_last, retry_this),
        ordered(entities)
    );
}

#[test]
fn one_multi_chunk_task_does_not_stamp_chunks_after_a_callback_panic() {
    let mut world = World::new();
    let mut entities = Vec::new();
    let first = world.spawn(TableValue {
        ordinal: 0,
        value: 0,
        _padding: [0; 256],
    });
    entities.push(first);
    let location = world.entities().location(first).unwrap();
    let rows_per_chunk = world
        .archetypes()
        .get(location.archetype_id)
        .unwrap()
        .table()
        .rows_per_chunk()
        .max(1);
    let count = rows_per_chunk * 3;
    for ordinal in 1..count {
        entities.push(world.spawn(TableValue {
            ordinal,
            value: ordinal as u32,
            _padding: [0; 256],
        }));
    }
    let query = world.query::<&TableValue>();
    let baseline = world.change_tick();
    let _ = query.dirty_chunks_with_stats(&world, baseline, baseline);
    let (last, this) = advance(&mut world);
    let pool = TaskPool::with_threads(4);
    let panic_ordinal = rows_per_chunk;

    let result = catch_unwind(AssertUnwindSafe(|| {
        let state = world.query::<&mut TableValue>();
        state.par_chunks_mut(&mut world, &pool, rows_per_chunk * 3, |slice| {
            assert!(slice.len() <= rows_per_chunk);
            for value in slice {
                if value.ordinal == panic_ordinal {
                    panic!("panic before the later physical chunk is yielded");
                }
                value.value += 1;
            }
        });
    }));
    assert!(result.is_err());

    let changed_entities = changed::<TableValue>(&mut world, last, this);
    let expected: Vec<_> = (0..rows_per_chunk * 2)
        .map(|ordinal| entities[ordinal])
        .collect();
    assert_eq!(changed_entities, expected);
    let dirty = query.dirty_chunks(&world, last, this);
    assert_eq!(dirty.len(), 2);
    assert!(dirty.iter().all(|window| window.start < rows_per_chunk * 2));
}

#[test]
fn public_tick_wrap_window_reaches_changed_and_dirty_chunk_oracles() {
    let (mut world, entities) = seed(6);
    let query = world.query::<&mut TableValue>();
    let clean = world.change_tick();
    let _ = world
        .query::<&TableValue>()
        .dirty_chunks_with_stats(&world, clean, clean);

    // Move the seeded values outside the wrapped window. The public world tick
    // API intentionally leaves the current tick at 1 here, so without this
    // normalization the initial spawn at tick 1 would also be a valid change.
    for &entity in &entities {
        world
            .get_mut::<TableValue>(entity)
            .expect("seeded table value")
            .set_last_changed(Tick::MAX);
    }

    // Public tick controls let the acceptance test exercise the same
    // `(MAX, 0/1]` comparison without a billion-iteration warm-up.
    world.set_last_change_tick(Tick::MAX);
    let last = world.last_change_tick();
    let this = world.change_tick();
    let selected = vec![entities[1], entities[4]];
    let pool = TaskPool::with_threads(2);
    let state = world.query::<(Entity, &mut TableValue)>();
    state.par_for_each_mut(&mut world, &pool, 1, |(entity, mut value)| {
        if selected.contains(&entity) {
            value.value += 5;
        }
    });

    assert_eq!(changed::<TableValue>(&mut world, last, this), selected);
    let report = query.dirty_chunks(&world, last, this);
    assert_eq!(report, dirty_windows(&mut world, &selected));
}
