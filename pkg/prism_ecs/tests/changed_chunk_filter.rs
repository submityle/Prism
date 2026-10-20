//! Operation-count and row-oracle regressions for ordinary Changed queries.

use core::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use prism_ecs::archetype::Archetype;
use prism_ecs::change::Tick;
use prism_ecs::component::{Component, Components, StorageType};
use prism_ecs::entity::Entity;
use prism_ecs::query::{
    Access, Added, Changed, Or, QueryData, QueryFilter, ReadOnlyQueryData, With, Without,
};
use prism_ecs::storage::SparseSets;
use prism_ecs::world::World;

#[derive(Clone, Copy)]
struct Big {
    value: u64,
    _padding: [u64; 511],
}
impl Component for Big {}

struct Sparse(u64);
impl Component for Sparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

struct MissingSparse;
impl Component for MissingSparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

struct Marker;
impl Component for Marker {}

fn advance(world: &mut World) {
    world.set_last_change_tick(world.change_tick());
    world.increment_change_tick();
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
    for value in 1..(3 * rpc + 1) {
        entities.push(world.spawn(Big {
            value: value as u64,
            _padding: [0; 511],
        }));
    }
    (world, entities, rpc)
}

fn changed<T: Component>(world: &World, entity: Entity, last: Tick, now: Tick) -> bool {
    world
        .get_ticks::<T>(entity)
        .is_some_and(|ticks| ticks.changed.is_newer_than(last, now))
}

fn sorted(mut entities: Vec<Entity>) -> Vec<Entity> {
    entities.sort_unstable();
    entities
}

// Tests using these counters hold COUNTERS for their whole lifetime. Counting
// the outer predicate detects even calls rejected by the wrapped row filter.
static COUNTERS: Mutex<()> = Mutex::new(());
static ROWS: AtomicUsize = AtomicUsize::new(0);
static DATA_CHUNKS: AtomicUsize = AtomicUsize::new(0);

fn reset_counts() {
    ROWS.store(0, Ordering::Relaxed);
    DATA_CHUNKS.store(0, Ordering::Relaxed);
}

struct CountRows<F>(PhantomData<fn() -> F>);

// SAFETY: all state, matching and access contracts are delegated to F; the
// wrapper only counts row-predicate invocations.
unsafe impl<F: QueryFilter> QueryFilter for CountRows<F> {
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
        // SAFETY: forwarded matching archetype and observer window.
        unsafe { F::init_fetch(state, archetype, sparse, last, now) }
    }

    fn filter_chunk<'w>(fetch: Self::Fetch<'w>, chunk: usize) -> bool {
        F::filter_chunk(fetch, chunk)
    }

    unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> bool {
        ROWS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarded live row of the same archetype.
        unsafe { F::filter_fetch(fetch, entity, row) }
    }
}

struct CountEntity;

// SAFETY: returns only the entity supplied by the driver and reads no storage.
unsafe impl QueryData for CountEntity {
    type Item<'w> = Entity;
    type State = ();
    type Fetch<'w> = ();

    fn init_state(_: &mut Components) {}
    fn matches(_: &(), _: &Archetype) -> bool {
        true
    }
    fn update_access(_: &(), _: &mut Access) {}

    unsafe fn init_fetch<'w>(_: &(), _: &'w Archetype, _: &'w SparseSets, _: Tick, _: Tick) {
        // The default init_fetch_for_chunk delegates here once per cursor.
        DATA_CHUNKS.fetch_add(1, Ordering::Relaxed);
    }

    unsafe fn fetch<'w>(_: Self::Fetch<'w>, entity: Entity, _: usize) -> Self::Item<'w> {
        entity
    }
}

// SAFETY: CountEntity never reads or mutates component storage.
unsafe impl ReadOnlyQueryData for CountEntity {}

#[test]
fn serial_next_fold_and_partial_fold_skip_cold_chunks_before_data_fetch() {
    let _guard = COUNTERS.lock().unwrap();
    let (mut world, entities, rpc) = fixture();
    advance(&mut world);
    let query = world.query_filtered::<CountEntity, CountRows<Changed<Big>>>();
    reset_counts();
    assert_eq!(query.iter(&world).next(), None);
    assert_eq!(ROWS.load(Ordering::Relaxed), 0);
    assert_eq!(DATA_CHUNKS.load(Ordering::Relaxed), 0);

    world.get_mut::<Big>(entities[rpc + 1]).unwrap().value += 1;
    world.get_mut::<Big>(entities[3 * rpc]).unwrap().value += 1;
    let expected = vec![entities[rpc + 1], entities[3 * rpc]];

    reset_counts();
    let iter = query.iter(&world);
    let mut via_next = Vec::new();
    for entity in iter {
        via_next.push(entity);
    }
    assert_eq!(via_next, expected);
    assert_eq!(ROWS.load(Ordering::Relaxed), rpc + 1);
    assert_eq!(DATA_CHUNKS.load(Ordering::Relaxed), 2);

    reset_counts();
    let via_fold = query.iter(&world).fold(Vec::new(), |mut acc, entity| {
        acc.push(entity);
        acc
    });
    assert_eq!(via_fold, expected);
    assert_eq!(ROWS.load(Ordering::Relaxed), rpc + 1);
    assert_eq!(DATA_CHUNKS.load(Ordering::Relaxed), 2);

    reset_counts();
    let mut iter = query.iter(&world);
    assert_eq!(iter.next(), Some(expected[0]));
    assert_eq!(ROWS.load(Ordering::Relaxed), 2);
    assert_eq!(DATA_CHUNKS.load(Ordering::Relaxed), 1);
    let remaining = iter.fold(Vec::new(), |mut acc, entity| {
        acc.push(entity);
        acc
    });
    assert_eq!(remaining, expected[1..]);
    assert_eq!(ROWS.load(Ordering::Relaxed), rpc + 1);
    assert_eq!(DATA_CHUNKS.load(Ordering::Relaxed), 2);

    advance(&mut world);
    reset_counts();
    assert_eq!(query.iter(&world).count(), 0);
    assert_eq!(ROWS.load(Ordering::Relaxed), 0);
    assert_eq!(DATA_CHUNKS.load(Ordering::Relaxed), 0);
}

#[test]
fn historical_and_wrapping_windows_match_row_tick_oracle() {
    let (mut world, entities, rpc) = fixture();
    let spawn = world.change_tick();
    advance(&mut world);
    let earlier = world.change_tick();
    world.get_mut::<Big>(entities[rpc]).unwrap().value += 1;
    advance(&mut world);
    let later = world.change_tick();
    // A later write in the same chunk must not hide an earlier row when this
    // consumer queries a historical window.
    world.get_mut::<Big>(entities[rpc + 1]).unwrap().value += 1;
    world.get_mut::<Big>(entities[3 * rpc]).unwrap().value += 1;
    let query = world.query_filtered::<Entity, Changed<Big>>();
    for (last, now) in [
        (Tick::ZERO, spawn),
        (spawn, earlier),
        (spawn, later),
        (earlier, later),
        (later, later),
        (Tick::new(u32::MAX - 2), later),
        (Tick::MAX, Tick::ZERO),
        (Tick::ZERO, Tick::MAX),
    ] {
        let expected = entities
            .iter()
            .copied()
            .filter(|&entity| changed::<Big>(&world, entity, last, now))
            .collect();
        assert_eq!(
            sorted(query.iter_with_ticks(&world, last, now).collect()),
            sorted(expected),
            "observer window ({last:?}, {now:?}]"
        );
    }
}

#[test]
fn dirty_query_matches_row_oracle_for_ticks_straddling_u32_wrap() {
    let (mut world, entities, rpc) = fixture();
    let component = world.components().id_of::<Big>().unwrap();
    let query = world.query_filtered::<Entity, Changed<Big>>();
    let rows = [
        (0, Tick::new(u32::MAX - 1)),
        (rpc, Tick::MAX),
        (2 * rpc, Tick::ZERO),
        (3 * rpc, Tick::new(1)),
        (rpc + 1, Tick::new(Tick::MAX_CHANGE_AGE)),
    ];

    // Exercise the real table chunk summaries and journal while positioning
    // component ticks on both sides of the u32 wrap boundary.
    for &(row, tick) in &rows {
        let location = world.entities().location(entities[row]).unwrap();
        let column = world
            .archetypes()
            .get(location.archetype_id)
            .unwrap()
            .table()
            .column(component)
            .unwrap();
        // SAFETY: this test has exclusive access to `world`, and each row is
        // written exactly once before any query iterator borrows the column.
        unsafe {
            *column.changed_tick_ptr(location.row as usize) = tick;
            (*column.chunk_changed_ptr(location.row as usize)).record(tick);
        }
    }

    let boundaries = [
        Tick::ZERO,
        Tick::new(1),
        Tick::new(2),
        Tick::new(Tick::MAX_CHANGE_AGE),
        Tick::new(Tick::MAX_CHANGE_AGE + 1),
        Tick::new(1 << 31),
        Tick::new(u32::MAX - 1),
        Tick::MAX,
    ];
    for &last in &boundaries {
        for &now in &boundaries {
            let expected = entities
                .iter()
                .copied()
                .filter(|&entity| changed::<Big>(&world, entity, last, now))
                .collect();
            assert_eq!(
                sorted(query.iter_with_ticks(&world, last, now).collect()),
                sorted(expected),
                "observer window ({last:?}, {now:?}]",
            );
        }
    }
}

#[test]
fn sparse_and_nested_or_and_filters_match_row_oracles() {
    let (mut world, mut entities, rpc) = fixture();
    for &index in &[0, rpc + 1, 2 * rpc] {
        assert!(world.insert(entities[index], Sparse(0)));
    }
    let marker = world.spawn(Marker);
    let empty = world.spawn(());
    entities.extend([marker, empty]);
    advance(&mut world);
    world.get_mut::<Big>(entities[rpc + 1]).unwrap().value += 1;
    world.get_mut::<Sparse>(entities[2 * rpc]).unwrap().0 += 1;
    let last = world.last_change_tick();
    let now = world.change_tick();
    let big = |world: &World, e| changed::<Big>(world, e, last, now);
    let sparse = |world: &World, e| changed::<Sparse>(world, e, last, now);

    let query = world.query_filtered::<Entity, Or<(Changed<Big>, Changed<Sparse>)>>();
    assert_eq!(
        sorted(query.iter(&world).collect()),
        sorted(
            entities
                .iter()
                .copied()
                .filter(|&e| big(&world, e) || sparse(&world, e))
                .collect()
        )
    );
    let query = world.query_filtered::<Entity, (Changed<Big>, With<Sparse>)>();
    assert_eq!(
        sorted(query.iter(&world).collect()),
        sorted(
            entities
                .iter()
                .copied()
                .filter(|&e| big(&world, e) && world.get::<Sparse>(e).is_some())
                .collect()
        )
    );
    let query = world.query_filtered::<Entity, Or<(
        (Changed<Big>, Without<Sparse>),
        (Changed<Sparse>, With<Big>),
        With<Marker>,
    )>>();
    let expected = entities
        .iter()
        .copied()
        .filter(|&e| {
            (big(&world, e) && world.get::<Sparse>(e).is_none())
                || (sparse(&world, e) && world.get::<Big>(e).is_some())
                || e == marker
        })
        .collect();
    assert_eq!(sorted(query.iter(&world).collect()), sorted(expected));

    #[cfg(feature = "multi_thread")]
    {
        let pool = prism_tasks::TaskPool::with_threads(4);
        for batch_size in [1, rpc - 1, rpc + 1] {
            let output = Mutex::new(Vec::new());
            query.par_for_each(&world, &pool, batch_size, |entity| {
                output.lock().unwrap().push(entity);
            });
            assert_eq!(
                sorted(output.into_inner().unwrap()),
                sorted(query.iter(&world).collect())
            );
        }
    }

    let query = world.query_filtered::<Entity, Or<(Changed<Big>, Without<Big>)>>();
    let expected = entities
        .iter()
        .copied()
        .filter(|&e| big(&world, e) || world.get::<Big>(e).is_none())
        .collect();
    assert_eq!(sorted(query.iter(&world).collect()), sorted(expected));
    assert_eq!(
        world
            .query_filtered::<Entity, Changed<MissingSparse>>()
            .iter(&world)
            .count(),
        0
    );

    // Allocating a previously absent sparse set must work with the same query.
    let missing = world.query_filtered::<Entity, Or<(Changed<Big>, Added<MissingSparse>)>>();
    assert!(world.insert(empty, MissingSparse));
    let mut expected = vec![entities[rpc + 1], empty];
    assert_eq!(
        sorted(missing.iter(&world).collect()),
        sorted(expected.clone())
    );
    world.remove::<MissingSparse>(empty);
    expected.pop();
    assert_eq!(sorted(missing.iter(&world).collect()), sorted(expected));
}

// Deliberately implements only the pre-existing row API.
struct OddRows;
// SAFETY: reads no component values or metadata; only narrows row membership.
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

struct Never;
// SAFETY: never admits an archetype or row and reads no storage.
unsafe impl QueryFilter for Never {
    type State = ();
    type Fetch<'w> = ();
    fn init_state(_: &mut Components) {}
    fn matches(_: &(), _: &Archetype) -> bool {
        false
    }
    fn update_access(_: &(), _: &mut Access) {}
    unsafe fn init_fetch<'w>(_: &(), _: &'w Archetype, _: &'w SparseSets, _: Tick, _: Tick) {
        panic!("nonmatching Or branch must not be initialised");
    }
    unsafe fn filter_fetch<'w>(_: (), _: Entity, _: usize) -> bool {
        panic!("nonmatching Or branch must not be evaluated");
    }
}

#[test]
fn custom_filter_default_and_nonmatching_or_branches_are_conservative() {
    let _guard = COUNTERS.lock().unwrap();
    let (mut world, entities, _) = fixture();
    advance(&mut world);
    let expected = entities
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(row, e)| (row % 2 == 1).then_some(e))
        .collect::<Vec<_>>();
    let query = world.query_filtered::<Entity, OddRows>();
    assert_eq!(query.iter(&world).collect::<Vec<_>>(), expected);
    let query = world.query_filtered::<Entity, Or<(Never, OddRows)>>();
    assert_eq!(query.iter(&world).collect::<Vec<_>>(), expected);

    let query = world.query_filtered::<CountEntity, CountRows<Or<(Never, Changed<Big>)>>>();
    reset_counts();
    assert_eq!(query.iter(&world).count(), 0);
    assert_eq!(ROWS.load(Ordering::Relaxed), 0);
    assert_eq!(DATA_CHUNKS.load(Ordering::Relaxed), 0);
}

#[test]
fn changed_filter_allows_mutable_data_for_the_same_component() {
    let (mut world, entities, rpc) = fixture();
    advance(&mut world);
    world.get_mut::<Big>(entities[rpc + 1]).unwrap().value += 1;
    let query = world.query_filtered::<(Entity, &mut Big), Changed<Big>>();
    let visited = query
        .iter_mut(&mut world)
        .map(|(entity, mut value)| {
            value.value += 1;
            entity
        })
        .collect::<Vec<_>>();
    assert_eq!(visited, [entities[rpc + 1]]);
    assert_eq!(
        world.get::<Big>(entities[rpc + 1]).unwrap().value,
        (rpc + 3) as u64
    );
}

#[cfg(feature = "multi_thread")]
#[test]
fn parallel_partial_batches_skip_cold_chunks_before_data_fetch() {
    let _guard = COUNTERS.lock().unwrap();
    let (mut world, entities, rpc) = fixture();
    advance(&mut world);
    let pool = prism_tasks::TaskPool::with_threads(4);
    let query = world.query_filtered::<CountEntity, CountRows<Changed<Big>>>();
    reset_counts();
    query.par_for_each(&world, &pool, 1, |_| panic!("clean frame yielded a row"));
    assert_eq!(ROWS.load(Ordering::Relaxed), 0);
    assert_eq!(DATA_CHUNKS.load(Ordering::Relaxed), 0);

    world.get_mut::<Big>(entities[rpc + 1]).unwrap().value += 1;
    world.get_mut::<Big>(entities[3 * rpc]).unwrap().value += 1;
    for batch_size in [1, rpc - 1, rpc + 1, 2 * rpc + 1] {
        let output = Mutex::new(Vec::new());
        reset_counts();
        query.par_for_each(&world, &pool, batch_size, |entity| {
            output.lock().unwrap().push(entity);
        });
        assert_eq!(
            sorted(output.into_inner().unwrap()),
            sorted(vec![entities[rpc + 1], entities[3 * rpc]])
        );
        assert_eq!(ROWS.load(Ordering::Relaxed), rpc + 1);
        // Each batch/chunk intersection needs one data cursor iff the chunk
        // is hot; partial intersections must not duplicate or drop any rows.
        let mut expected_fetches = 0;
        for start in (0..entities.len()).step_by(batch_size) {
            let end = (start + batch_size).min(entities.len());
            for chunk in (start / rpc)..=((end - 1) / rpc) {
                expected_fetches += usize::from(chunk == 1 || chunk == 3);
            }
        }
        assert_eq!(
            DATA_CHUNKS.load(Ordering::Relaxed),
            expected_fetches,
            "batch_size={batch_size}"
        );
    }

    let query = world.query_filtered::<(Entity, &mut Big), Changed<Big>>();
    let output = Mutex::new(Vec::new());
    query.par_for_each_mut(&mut world, &pool, rpc - 1, |(entity, mut value)| {
        value.value += 1;
        output.lock().unwrap().push(entity);
    });
    assert_eq!(
        sorted(output.into_inner().unwrap()),
        sorted(vec![entities[rpc + 1], entities[3 * rpc]])
    );
}
