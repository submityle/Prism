//! External-crate coverage of the public column-slice extension and barriers.
//! Sequential cases deliberately run without the `multi_thread` feature.

use prism_ecs::archetype::Archetype;
use prism_ecs::component::Components;
use prism_ecs::prelude::*;
use prism_ecs::query::{Access, ColumnSliceChanges, ColumnSliceData, QueryData};
use prism_ecs::storage::SparseSets;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Component)]
struct Read(u32);

#[derive(Component)]
struct Write {
    value: u32,
    // Keep the fixture small while exercising multiple physical chunks.
    _padding: [u64; 256],
}

#[derive(Component)]
struct Marker;

#[derive(Component)]
struct Write2 {
    value: u32,
    _padding: [u64; 256],
}

type Tuple = (Entity, &'static Read, &'static mut Write);

/// An external query term with exactly the built-in tuple's access and layout.
struct ForwardedTuple;

// SAFETY: every associated type and operation is forwarded to the same built-in
// tuple. In particular, matching, access registration, presence filtering and
// fetch all agree; no additional borrow or lifetime extension is introduced.
unsafe impl QueryData for ForwardedTuple {
    type Item<'w> = <Tuple as QueryData>::Item<'w>;
    type State = <Tuple as QueryData>::State;
    type Fetch<'w> = <Tuple as QueryData>::Fetch<'w>;

    fn init_state(components: &mut Components) -> Self::State {
        <Tuple as QueryData>::init_state(components)
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        <Tuple as QueryData>::matches(state, archetype)
    }

    fn update_access(state: &Self::State, access: &mut Access) {
        <Tuple as QueryData>::update_access(state, access);
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        sparse_sets: &'w SparseSets,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w> {
        // SAFETY: the forwarded `matches` is exactly the caller's precondition.
        unsafe {
            <Tuple as QueryData>::init_fetch(state, archetype, sparse_sets, last_run, this_run)
        }
    }

    unsafe fn init_fetch_for_chunk<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        sparse_sets: &'w SparseSets,
        last_run: Tick,
        this_run: Tick,
        chunk: usize,
    ) -> Self::Fetch<'w> {
        // SAFETY: same matched archetype, state and chunk as the caller supplied.
        unsafe {
            <Tuple as QueryData>::init_fetch_for_chunk(
                state,
                archetype,
                sparse_sets,
                last_run,
                this_run,
                chunk,
            )
        }
    }

    unsafe fn filter_fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> bool {
        // SAFETY: the caller supplies an in-bounds row for this forwarded fetch.
        unsafe { <Tuple as QueryData>::filter_fetch(fetch, entity, row) }
    }

    unsafe fn fetch<'w>(fetch: Self::Fetch<'w>, entity: Entity, row: usize) -> Self::Item<'w> {
        // SAFETY: the caller guarantees presence, bounds and unique row access;
        // this wrapper neither duplicates the fetch nor changes its row.
        unsafe { <Tuple as QueryData>::fetch(fetch, entity, row) }
    }
}

// SAFETY: sliceability and slice construction delegate to the identical tuple
// used by QueryData, preserving its table-only restriction, access set, exact
// row range, unique mutable slice and complete change barriers. The borrowed
// change context is forwarded, never retained or given a longer lifetime.
unsafe impl ColumnSliceData for ForwardedTuple {
    type Slice<'w> = <Tuple as ColumnSliceData>::Slice<'w>;

    fn assert_sliceable() {
        <Tuple as ColumnSliceData>::assert_sliceable();
    }

    unsafe fn column_slice<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        start: usize,
        end: usize,
        changes: ColumnSliceChanges<'_>,
        this_run: Tick,
    ) -> Self::Slice<'w> {
        // Exercise the publicly named context and its public recording method.
        // An empty batch must be inert; the delegated mutable term records the
        // actual rows, ticks and coarse windows before returning its slice.
        changes.record_entities(state.2, &[], this_run);
        // SAFETY: all caller guarantees, including a same-world change context
        // and a contiguous, exclusively accessible range, are forwarded intact.
        unsafe {
            <Tuple as ColumnSliceData>::column_slice(
                state, archetype, start, end, changes, this_run,
            )
        }
    }
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
struct Deliveries {
    read_observers: Vec<Vec<Entity>>,
    write_observers: Vec<Vec<Entity>>,
    read_reactions: Vec<Vec<Entity>>,
    write_reactions: Vec<Vec<Entity>>,
}

fn ordered(mut entities: Vec<Entity>) -> Vec<Entity> {
    entities.sort_unstable_by_key(|entity| (entity.index(), entity.generation()));
    entities
}

fn fixture() -> (World, Vec<Entity>, Tick, Tick) {
    let mut world = World::new();
    let entities: Vec<_> = (0..35)
        .map(|index| {
            let bundle = (
                Read(index),
                Write {
                    value: index,
                    _padding: [0; 256],
                },
            );
            if index % 2 == 0 {
                world.spawn(bundle)
            } else {
                world.spawn((bundle, Marker))
            }
        })
        .collect();
    assert!(dirty_windows(&world, &entities).len() >= 4);

    // Establish the journals' baseline without relying on their event counts.
    let baseline = world.change_tick();
    let _ = world
        .query::<&Write>()
        .dirty_chunks(&world, baseline, baseline);
    let _ = world
        .query::<&Read>()
        .dirty_chunks(&world, baseline, baseline);
    world.insert_resource(Deliveries::default());
    world.observe_changed::<Read, _>(|ctx| {
        let batch = ctx.changed().to_vec();
        ctx.world
            .resource_mut::<Deliveries>()
            .read_observers
            .push(batch);
    });
    world.observe_changed::<Write, _>(|ctx| {
        let batch = ctx.changed().to_vec();
        ctx.world
            .resource_mut::<Deliveries>()
            .write_observers
            .push(batch);
    });
    let read_source = world.reaction_source::<Read>();
    let write_source = world.reaction_source::<Write>();
    world.add_reaction(&[read_source], move |world| {
        let batch = world.reaction_changes(read_source).to_vec();
        world
            .resource_mut::<Deliveries>()
            .read_reactions
            .push(batch);
    });
    world.add_reaction(&[write_source], move |world| {
        let batch = world.reaction_changes(write_source).to_vec();
        world
            .resource_mut::<Deliveries>()
            .write_reactions
            .push(batch);
    });
    let last = world.change_tick();
    world.set_last_change_tick(last);
    let this = world.increment_change_tick();
    (world, entities, last, this)
}

fn dirty_windows(world: &World, entities: &[Entity]) -> Vec<DirtyChunk> {
    let mut windows = Vec::new();
    for &entity in entities {
        let location = world
            .entities()
            .location(entity)
            .expect("live fixture entity");
        let archetype = world.archetypes().get(location.archetype_id).unwrap();
        let width = archetype.table().rows_per_chunk();
        let start = location.row as usize / width * width;
        windows.push(DirtyChunk {
            archetype: location.archetype_id,
            start,
            end: (start + width).min(archetype.len()),
        });
    }
    windows.sort_unstable_by_key(|window| (window.archetype, window.start));
    windows.dedup();
    windows
}

fn changed<T: Component>(world: &mut World, last: Tick, this: Tick) -> Vec<Entity> {
    let query = world.query_filtered::<Entity, Changed<T>>();
    ordered(query.iter_with_ticks(world, last, this).collect())
}

fn assert_idle(world: &mut World) {
    let before = world.resource::<Deliveries>().clone();
    assert_eq!(world.flush_changed_observers(), 0);
    assert_eq!(world.flush_reactions(), 0);
    assert_eq!(world.resource::<Deliveries>(), &before);
}

fn assert_barriers(world: &mut World, last: Tick, this: Tick, handed_out: &[Entity]) {
    let expected = ordered(handed_out.to_vec());
    assert!(!expected.is_empty());
    assert!(
        expected.windows(2).all(|pair| pair[0] != pair[1]),
        "overlapping slices"
    );
    assert_eq!(changed::<Write>(world, last, this), expected);
    assert!(changed::<Read>(world, last, this).is_empty());
    let expected_windows = dirty_windows(world, &expected);
    assert_eq!(
        world.query::<&Write>().dirty_chunks(world, last, this),
        expected_windows
    );
    assert!(world
        .query::<&Read>()
        .dirty_chunks(world, last, this)
        .is_empty());

    // Collect the component bus first; flush_reactions alone does not ingest
    // source changes. One explicit observer and one source observer should run.
    assert_eq!(world.flush_changed_observers(), 2);
    let log = world.resource::<Deliveries>();
    assert!(log.read_observers.is_empty());
    assert!(log.read_reactions.is_empty());
    assert_eq!(log.write_observers, vec![expected.clone()]);
    assert_eq!(log.write_reactions, vec![expected]);
    assert_idle(world);
}

fn check_tuple(ids: &[Entity], reads: &[Read], writes: &[Write]) {
    assert!(!ids.is_empty());
    assert_eq!(ids.len(), reads.len());
    assert_eq!(ids.len(), writes.len());
    for (read, write) in reads.iter().zip(writes) {
        assert_eq!(
            read.0, write.value,
            "tuple columns must describe the same row"
        );
    }
}

fn assert_values(world: &World, entities: &[Entity], modified: Option<Entity>) {
    for (index, &entity) in entities.iter().enumerate() {
        assert_eq!(world.get::<Read>(entity).unwrap().0, index as u32);
        assert_eq!(
            world.get::<Write>(entity).unwrap().value,
            index as u32 + if Some(entity) == modified { 100 } else { 0 },
        );
    }
}

#[test]
fn external_term_can_name_changes_and_forward_public_column_slice_contract() {
    let (mut world, entities, last, this) = fixture();
    let query = world.query::<ForwardedTuple>();
    let mut handed_out = Vec::new();
    for (ids, reads, writes) in query.simd_iter_mut::<3>(&mut world) {
        check_tuple(ids, reads, writes);
        handed_out.extend_from_slice(ids);
        // No mutable dereference is needed to trigger the slice barrier.
    }
    assert_eq!(ordered(handed_out.clone()), ordered(entities.clone()));
    assert_values(&world, &entities, None);
    assert_barriers(&mut world, last, this, &handed_out);
}

#[test]
fn sequential_noop_tuple_stamps_only_yielded_rows_not_the_unvisited_tail() {
    let (mut world, entities, last, this) = fixture();
    let query = world.query::<Tuple>();
    // Merely constructing and dropping a lazy iterator must not mark anything.
    drop(query.simd_iter_mut::<3>(&mut world));
    assert!(changed::<Write>(&mut world, last, this).is_empty());
    assert!(world
        .query::<&Write>()
        .dirty_chunks(&world, last, this)
        .is_empty());
    assert_idle(&mut world);

    // Even an idle observer flush advances the world's tick boundary. Capture
    // a fresh window for the slices instead of reusing the pre-flush `this`.
    let last = world.change_tick();
    world.set_last_change_tick(last);
    let this = world.increment_change_tick();

    let handed_out = {
        let mut slices = query.simd_iter_mut::<3>(&mut world);
        let (ids, reads, writes) = slices.next().expect("nonempty fixture");
        check_tuple(ids, reads, writes);
        assert_eq!(ids.len(), 3);
        ids.to_vec()
    };
    assert!(handed_out.len() < entities.len());
    assert_values(&world, &entities, None);
    assert_barriers(&mut world, last, this, &handed_out);
}

#[test]
fn sequential_tuple_panic_reports_the_entire_yielded_slice_before_any_retry() {
    let (mut world, entities, last, this) = fixture();
    let query = world.query::<Tuple>();
    let mut handed_out = Vec::new();
    let mut modified = None;
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut slices = query.simd_iter_mut::<3>(&mut world);
        let (ids, reads, writes) = slices.next().unwrap();
        check_tuple(ids, reads, writes);
        assert_eq!(ids.len(), 3);
        handed_out.extend_from_slice(ids);
        modified = Some(ids[0]);
        writes[0].value += 100;
        panic!("sequential slice callback panic before writing the tail");
    }));
    assert!(result.is_err());
    assert_eq!(handed_out.len(), 3);
    assert_values(&world, &entities, modified);
    assert_barriers(&mut world, last, this, &handed_out);
}

#[test]
fn two_mutable_columns_preserve_entity_and_payload_alignment() {
    let mut world = World::new();
    let entities: Vec<_> = (0..17)
        .map(|index| {
            world.spawn((
                Write {
                    value: index,
                    _padding: [0; 256],
                },
                Write2 {
                    value: index + 1000,
                    _padding: [0; 256],
                },
            ))
        })
        .collect();
    let last = world.change_tick();
    world.set_last_change_tick(last);
    let this = world.increment_change_tick();
    let query = world.query::<(Entity, &'static mut Write, &'static mut Write2)>();
    let mut handed_out = Vec::new();
    for (ids, writes, writes2) in query.simd_iter_mut::<3>(&mut world) {
        assert_eq!(ids.len(), writes.len());
        assert_eq!(ids.len(), writes2.len());
        for ((&entity, write), write2) in ids.iter().zip(writes.iter()).zip(writes2.iter()) {
            let ordinal = entities
                .iter()
                .position(|&candidate| candidate == entity)
                .unwrap();
            assert_eq!(write.value, ordinal as u32);
            assert_eq!(write2.value, ordinal as u32 + 1000);
        }
        handed_out.extend_from_slice(ids);
    }
    assert_eq!(ordered(handed_out), ordered(entities.clone()));
    assert_eq!(
        changed::<Write>(&mut world, last, this),
        ordered(entities.clone())
    );
    assert_eq!(changed::<Write2>(&mut world, last, this), ordered(entities));
}

#[cfg(feature = "multi_thread")]
mod parallel {
    use super::*;
    use prism_tasks::TaskPool;
    use std::sync::Mutex;

    #[test]
    fn noop_tuple_stamps_every_disjoint_slice_for_one_and_multiple_workers() {
        for threads in [1, 4] {
            let (mut world, entities, last, this) = fixture();
            let query = world.query::<Tuple>();
            let handed_out = Mutex::new(Vec::new());
            let pool = TaskPool::with_threads(threads);
            query.par_chunks_mut(&mut world, &pool, 1, |(ids, reads, writes)| {
                check_tuple(ids, reads, writes);
                handed_out.lock().unwrap().extend_from_slice(ids);
            });
            let handed_out = handed_out.into_inner().unwrap();
            assert_eq!(ordered(handed_out.clone()), ordered(entities.clone()));
            assert_values(&world, &entities, None);
            assert_barriers(&mut world, last, this, &handed_out);
        }
    }

    #[test]
    fn tuple_panic_reports_exactly_the_slices_handed_out_including_unwritten_tail() {
        for threads in [1, 4] {
            let (mut world, entities, last, this) = fixture();
            let query = world.query::<Tuple>();
            let handed_out = Mutex::new(Vec::new());
            let target = entities[0];
            let pool = TaskPool::with_threads(threads);
            let result = catch_unwind(AssertUnwindSafe(|| {
                query.par_chunks_mut(&mut world, &pool, 1, |(ids, reads, writes)| {
                    check_tuple(ids, reads, writes);
                    // Release the mutex before the intentional panic. Expected
                    // rows come from actual callbacks, not worker scheduling.
                    handed_out.lock().unwrap().extend_from_slice(ids);
                    if let Some(index) = ids.iter().position(|&entity| entity == target) {
                        assert!(ids.len() > 1, "panic must leave an unwritten tail");
                        assert_eq!(index, 0);
                        writes[index].value += 100;
                        panic!("parallel slice callback panic before writing the tail");
                    }
                });
            }));
            assert!(result.is_err());
            let handed_out = handed_out.into_inner().unwrap();
            assert!(handed_out.contains(&target));
            assert_values(&world, &entities, Some(target));
            assert_barriers(&mut world, last, this, &handed_out);
        }
    }
}
