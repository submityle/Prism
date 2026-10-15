//! Scheduling regressions: count filter cursor initialisations, not callbacks.
//! The planner resolves one cursor per nonempty matched archetype; every
//! spawned task resolves exactly one more, even if it produces no rows.
#![cfg(feature = "multi_thread")]

use core::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use prism_ecs::archetype::Archetype;
use prism_ecs::change::Tick;
use prism_ecs::component::{Component, Components, StorageType};
use prism_ecs::entity::Entity;
use prism_ecs::query::{Access, Changed, Or, QueryFilter, With, Without};
use prism_ecs::storage::SparseSets;
use prism_ecs::world::World;
use prism_tasks::TaskPool;

struct Big {
    value: usize,
    _padding: [u64; 511],
}
impl Component for Big {}

struct Sparse(usize);
impl Component for Sparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

struct Marker;
impl Component for Marker {}

static COUNTERS: Mutex<()> = Mutex::new(());
static FETCHES: AtomicUsize = AtomicUsize::new(0);
static CHUNK_TESTS: AtomicUsize = AtomicUsize::new(0);

struct CountTasks<F>(PhantomData<fn() -> F>);

// SAFETY: matching, metadata access and live-row contracts are delegated to F.
// The wrapper only counts cursor creation and coarse predicate invocations.
unsafe impl<F: QueryFilter> QueryFilter for CountTasks<F> {
    type State = F::State;
    type Fetch<'w> = F::Fetch<'w>;

    fn init_state(components: &mut Components) -> Self::State {
        F::init_state(components)
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        F::matches(state, archetype)
    }

    fn update_access(state: &Self::State, access: &mut Access) {
        F::update_access(state, access);
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        sparse: &'w SparseSets,
        last: Tick,
        now: Tick,
    ) -> Self::Fetch<'w> {
        FETCHES.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarded matching archetype and observer window.
        unsafe { F::init_fetch(state, archetype, sparse, last, now) }
    }

    fn filter_chunk<'w>(fetch: Self::Fetch<'w>, chunk: usize) -> bool {
        CHUNK_TESTS.fetch_add(1, Ordering::Relaxed);
        F::filter_chunk(fetch, chunk)
    }

    unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> bool {
        // SAFETY: forwarded live row of the same archetype.
        unsafe { F::filter_fetch(fetch, entity, row) }
    }
}

fn fixture() -> (World, Vec<Entity>, usize) {
    let mut world = World::new();
    let first = world.spawn(Big {
        value: 0,
        _padding: [0; 511],
    });
    let location = world.entities().location(first).unwrap();
    let rpc = world
        .archetypes()
        .get(location.archetype_id)
        .unwrap()
        .table()
        .rows_per_chunk();
    assert!(rpc > 1);
    let mut entities = vec![first];
    for value in 1..(5 * rpc + 1) {
        entities.push(world.spawn(Big {
            value,
            _padding: [0; 511],
        }));
    }
    (world, entities, rpc)
}

fn advance(world: &mut World) {
    world.set_last_change_tick(world.change_tick());
    world.increment_change_tick();
}

fn reset_counts() {
    FETCHES.store(0, Ordering::Relaxed);
    CHUNK_TESTS.store(0, Ordering::Relaxed);
}

fn task_count() -> usize {
    // Counted queries below have exactly one nonempty matching archetype.
    FETCHES.load(Ordering::Relaxed).checked_sub(1).unwrap()
}

fn expected_tasks(len: usize, rpc: usize, batch_size: usize, hot: &[usize]) -> usize {
    let batch_size = batch_size.max(1);
    let mut tasks = 0;
    let mut start = 0;
    while start < len {
        let end = start + batch_size.min(len - start);
        tasks += usize::from((start..end).any(|row| hot.contains(&(row / rpc))));
        start = end;
    }
    tasks
}

fn sorted(mut entities: Vec<Entity>) -> Vec<Entity> {
    entities.sort_unstable();
    entities
}

#[test]
fn planner_dispatches_only_batches_intersecting_hot_chunks() {
    let _guard = COUNTERS.lock().unwrap();
    let (mut world, entities, rpc) = fixture();
    let query = world.query_filtered::<(Entity, &Big), CountTasks<Changed<Big>>>();
    let chunks = entities.len().div_ceil(rpc);
    let all_chunks = (0..chunks).collect::<Vec<_>>();
    for threads in [1, 4] {
        let pool = TaskPool::with_threads(threads);
        // Initial writes make every chunk hot. Batch grid and row multiplicity
        // must remain unchanged, including zero and overflow-sized requests.
        for batch_size in [0, 1, rpc - 1, rpc + 1, 3 * rpc + 1, usize::MAX] {
            let output = Mutex::new(Vec::new());
            reset_counts();
            query.par_for_each(&world, &pool, batch_size, |(entity, _)| {
                output.lock().unwrap().push(entity);
            });
            assert_eq!(
                sorted(output.into_inner().unwrap()),
                sorted(entities.clone())
            );
            assert_eq!(
                task_count(),
                expected_tasks(entities.len(), rpc, batch_size, &all_chunks)
            );
        }
    }

    advance(&mut world);
    for threads in [1, 4] {
        let pool = TaskPool::with_threads(threads);
        for batch_size in [0, 1, rpc - 1, rpc + 1, usize::MAX] {
            reset_counts();
            query.par_for_each(&world, &pool, batch_size, |_| panic!("cold row"));
            assert_eq!(task_count(), 0, "cold frame must not spawn tasks");
            assert_eq!(CHUNK_TESTS.load(Ordering::Relaxed), chunks);
        }
    }

    let changed = [rpc + 1, 3 * rpc, 5 * rpc];
    for &row in &changed {
        world.get_mut::<Big>(entities[row]).unwrap().value += 1;
    }
    let expected = changed.map(|row| entities[row]).to_vec();
    for threads in [1, 4] {
        let pool = TaskPool::with_threads(threads);
        for batch_size in [0, 1, rpc - 1, rpc + 1, 3 * rpc + 1, usize::MAX] {
            // Repeating a pass cannot alter planning or duplicate callbacks.
            for _ in 0..2 {
                let output = Mutex::new(Vec::new());
                reset_counts();
                query.par_for_each(&world, &pool, batch_size, |(entity, _)| {
                    output.lock().unwrap().push(entity);
                });
                assert_eq!(
                    sorted(output.into_inner().unwrap()),
                    sorted(expected.clone())
                );
                assert_eq!(
                    task_count(),
                    expected_tasks(entities.len(), rpc, batch_size, &[1, 3, 5]),
                    "threads={threads}, batch_size={batch_size}"
                );
            }
        }
    }
}

#[test]
fn mutable_changed_planning_does_not_touch_cold_rows() {
    let _guard = COUNTERS.lock().unwrap();
    let (mut world, entities, rpc) = fixture();
    advance(&mut world);
    let pool = TaskPool::with_threads(4);
    let query = world.query_filtered::<(Entity, &mut Big), CountTasks<Changed<Big>>>();
    let before = entities
        .iter()
        .map(|&entity| world.get_ticks::<Big>(entity).unwrap().changed)
        .collect::<Vec<_>>();
    reset_counts();
    query.par_for_each_mut(&mut world, &pool, 1, |_| panic!("cold mutable row"));
    assert_eq!(task_count(), 0);
    assert_eq!(
        before,
        entities
            .iter()
            .map(|&entity| world.get_ticks::<Big>(entity).unwrap().changed)
            .collect::<Vec<_>>()
    );

    let row = rpc + 1;
    world.get_mut::<Big>(entities[row]).unwrap().value += 1;
    let output = Mutex::new(Vec::new());
    reset_counts();
    query.par_for_each_mut(&mut world, &pool, rpc - 1, |(entity, mut value)| {
        value.value += 10;
        output.lock().unwrap().push(entity);
    });
    assert_eq!(output.into_inner().unwrap(), [entities[row]]);
    assert_eq!(
        task_count(),
        expected_tasks(entities.len(), rpc, rpc - 1, &[1])
    );
    for (row_index, &entity) in entities.iter().enumerate() {
        assert_eq!(
            world.get::<Big>(entity).unwrap().value,
            row_index + if row_index == row { 11 } else { 0 }
        );
    }
}

#[test]
fn sparse_filters_remain_conservative_and_sparse_data_keeps_its_row_gate() {
    let _guard = COUNTERS.lock().unwrap();
    let (mut world, entities, rpc) = fixture();
    advance(&mut world);
    let pool = TaskPool::with_threads(4);
    let sparse = world.query_filtered::<(Entity, &Big), CountTasks<Changed<Sparse>>>();
    reset_counts();
    sparse.par_for_each(&world, &pool, 1, |_| panic!("absent sparse set"));
    assert_eq!(task_count(), 0);

    world.insert(entities[2 * rpc], Sparse(7));
    let output = Mutex::new(Vec::new());
    reset_counts();
    sparse.par_for_each(&world, &pool, rpc + 1, |(entity, _)| {
        output.lock().unwrap().push(entity);
    });
    assert_eq!(output.into_inner().unwrap(), [entities[2 * rpc]]);
    assert_eq!(task_count(), entities.len().div_ceil(rpc + 1));

    // An allocated but clean sparse set is not sufficient to reject a table
    // chunk: per-entity filtering remains authoritative.
    advance(&mut world);
    reset_counts();
    sparse.par_for_each(&world, &pool, rpc + 1, |_| panic!("clean sparse row"));
    assert_eq!(task_count(), entities.len().div_ceil(rpc + 1));

    world.get_mut::<Big>(entities[rpc]).unwrap().value += 1;
    world.get_mut::<Big>(entities[2 * rpc]).unwrap().value += 1;
    let data = world.query_filtered::<(Entity, &Sparse), CountTasks<Changed<Big>>>();
    let output = Mutex::new(Vec::new());
    reset_counts();
    data.par_for_each(&world, &pool, rpc + 1, |(entity, sparse)| {
        assert_eq!(sparse.0, 7);
        output.lock().unwrap().push(entity);
    });
    assert_eq!(output.into_inner().unwrap(), [entities[2 * rpc]]);
    assert_eq!(
        task_count(),
        expected_tasks(entities.len(), rpc, rpc + 1, &[1, 2])
    );
}

#[test]
fn absent_or_branch_does_not_keep_cold_batches_alive() {
    let _guard = COUNTERS.lock().unwrap();
    let (mut world, entities, rpc) = fixture();
    advance(&mut world);
    let pool = TaskPool::with_threads(4);
    let query =
        world.query_filtered::<(Entity, &Big), CountTasks<Or<(Changed<Big>, With<Marker>)>>>();
    reset_counts();
    query.par_for_each(&world, &pool, 1, |_| panic!("cold Or row"));
    assert_eq!(task_count(), 0);

    world.get_mut::<Big>(entities[3 * rpc]).unwrap().value += 1;
    for batch_size in [1, rpc - 1, rpc + 1, usize::MAX] {
        let output = Mutex::new(Vec::new());
        reset_counts();
        query.par_for_each(&world, &pool, batch_size, |(entity, _)| {
            output.lock().unwrap().push(entity);
        });
        assert_eq!(output.into_inner().unwrap(), [entities[3 * rpc]]);
        assert_eq!(
            task_count(),
            expected_tasks(entities.len(), rpc, batch_size, &[3])
        );
    }
}

#[test]
fn empty_archetypes_do_not_create_filter_cursors_or_tasks() {
    let _guard = COUNTERS.lock().unwrap();
    let mut world = World::new();
    let query = world.query_filtered::<Entity, CountTasks<OddRows>>();
    let pool = TaskPool::with_threads(1);
    reset_counts();
    query.par_for_each(&world, &pool, 1, |_| panic!("empty world row"));
    assert_eq!(FETCHES.load(Ordering::Relaxed), 0);
    assert_eq!(CHUNK_TESTS.load(Ordering::Relaxed), 0);
}

#[test]
fn nested_or_and_absent_branches_match_serial_across_archetypes() {
    let (mut world, entities, rpc) = fixture();
    let marker = world.spawn(Marker);
    let empty = world.spawn(());
    world.insert(entities[rpc], Sparse(0));
    advance(&mut world);
    world.get_mut::<Big>(entities[rpc + 1]).unwrap().value += 1;
    world.get_mut::<Sparse>(entities[rpc]).unwrap().0 += 1;
    let query = world.query_filtered::<Entity, Or<(
        (Changed<Big>, Without<Sparse>),
        (Changed<Sparse>, With<Big>),
        With<Marker>,
    )>>();
    let expected = sorted(vec![entities[rpc], entities[rpc + 1], marker]);
    assert_eq!(sorted(query.iter(&world).collect()), expected);
    for threads in [1, 4] {
        let pool = TaskPool::with_threads(threads);
        for batch_size in [0, 1, rpc - 1, rpc + 1, usize::MAX] {
            let output = Mutex::new(Vec::new());
            query.par_for_each(&world, &pool, batch_size, |entity| {
                output.lock().unwrap().push(entity);
            });
            assert_eq!(sorted(output.into_inner().unwrap()), expected);
        }
    }
    // Newly allocated/removed sparse membership must be resolved on each pass,
    // not retained in the planner or confused with absent Or table branches.
    world.insert(empty, Sparse(1));
    let query = world.query_filtered::<Entity, Or<(Changed<Big>, Changed<Sparse>)>>();
    let pool = TaskPool::with_threads(4);
    for remove in [false, true] {
        if remove {
            world.remove::<Sparse>(empty);
        }
        let output = Mutex::new(Vec::new());
        query.par_for_each(&world, &pool, rpc + 1, |entity| {
            output.lock().unwrap().push(entity);
        });
        assert_eq!(
            sorted(output.into_inner().unwrap()),
            sorted(query.iter(&world).collect())
        );
    }
}

struct OddRows;

// SAFETY: this filter reads no storage and accepts only live odd row indexes.
// It deliberately retains QueryFilter's default conservative chunk predicate.
unsafe impl QueryFilter for OddRows {
    type State = ();
    type Fetch<'w> = ();

    fn init_state(_: &mut Components) {}
    fn matches(_: &(), _: &Archetype) -> bool {
        true
    }
    fn update_access(_: &(), _: &mut Access) {}
    unsafe fn init_fetch<'w>(_: &(), _: &'w Archetype, _: &'w SparseSets, _: Tick, _: Tick) {}
    unsafe fn filter_fetch<'w>(_: (), _: Entity, row: usize) -> bool {
        row % 2 == 1
    }
}

#[test]
fn custom_row_only_filter_retains_all_batches() {
    let _guard = COUNTERS.lock().unwrap();
    let (mut world, entities, rpc) = fixture();
    advance(&mut world);
    let query = world.query_filtered::<(Entity, &Big), CountTasks<OddRows>>();
    let expected = sorted(
        entities
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(row, entity)| (row % 2 == 1).then_some(entity))
            .collect(),
    );
    let pool = TaskPool::with_threads(4);
    for batch_size in [0, rpc - 1, rpc + 1, usize::MAX] {
        let output = Mutex::new(Vec::new());
        reset_counts();
        query.par_for_each(&world, &pool, batch_size, |(entity, _)| {
            output.lock().unwrap().push(entity);
        });
        assert_eq!(sorted(output.into_inner().unwrap()), expected);
        assert_eq!(task_count(), entities.len().div_ceil(batch_size.max(1)));
    }
}
