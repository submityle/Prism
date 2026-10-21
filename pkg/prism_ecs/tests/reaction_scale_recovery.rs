//! Public Query -> observe_changed -> Reaction regression tests.
//! Work assertions count query operations; no test depends on elapsed time.

use prism_ecs::archetype::COLUMN_CHANGE_LOG_CAPACITY;
use prism_ecs::component::StorageType;
use prism_ecs::prelude::*;
use std::collections::BTreeMap;

#[derive(Component, Clone, Copy)]
struct Value(u32);

#[derive(Clone, Copy)]
struct SparseValue(u32);
impl Component for SparseValue {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[derive(Component, Clone, Copy)]
struct Ordinal(usize);

#[derive(Component, Clone, Copy)]
struct Other(u32);

#[derive(Component, Clone, Copy)]
struct Marker;

// A bounded 4 MiB fixture at 16K entities spans hundreds of physical chunks,
// so a sparse entity oracle cannot accidentally pass as a coarse chunk oracle.
#[derive(Component, Clone)]
struct Padding {
    _words: [u64; 32],
}

trait Reading: Component {
    fn value(&self) -> u32;
}
impl Reading for Value {
    fn value(&self) -> u32 {
        self.0
    }
}
impl Reading for SparseValue {
    fn value(&self) -> u32 {
        self.0
    }
}

type Batch = Vec<(Entity, u32)>;

#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
struct Deliveries {
    observers: Vec<(usize, Batch)>,
    effects: Vec<Batch>,
    joins: Vec<u64>,
}

fn populate(world: &mut World, count: usize) -> Vec<Entity> {
    (0..count)
        .map(|index| {
            let bundle = (
                Value(index as u32),
                SparseValue(2 * index as u32),
                Ordinal(index),
                Other(0),
                Padding { _words: [0; 32] },
            );
            if index % 2 == 0 {
                world.spawn(bundle)
            } else {
                world.spawn((bundle, Marker))
            }
        })
        .collect()
}

fn values<T: Reading>(world: &World, entities: &[Entity]) -> Batch {
    entities
        .iter()
        .map(|&entity| {
            (
                entity,
                world.get::<T>(entity).expect("readable delivery").value(),
            )
        })
        .collect()
}

fn sorted(mut batch: Batch) -> Batch {
    batch.sort_unstable_by_key(|(entity, _)| (entity.index(), entity.generation()));
    batch
}

fn changed<T: Reading>(world: &mut World, last: Tick, this: Tick) -> Batch {
    let query = world.query_filtered::<(Entity, &T), Changed<T>>();
    sorted(
        query
            .iter_with_ticks(world, last, this)
            .map(|(entity, value)| (entity, value.value()))
            .collect(),
    )
}

fn install<T: Reading>(
    world: &mut World,
    observer_count: usize,
    effect_count: usize,
    clean_nodes: usize,
) -> ReactionId {
    world.insert_resource(Deliveries::default());
    for index in 0..observer_count {
        world.observe_changed::<T, _>(move |ctx| {
            let batch = values::<T>(ctx.world, ctx.changed());
            ctx.world
                .resource_mut::<Deliveries>()
                .observers
                .push((index, batch));
        });
    }
    let source = world.reaction_source::<T>();
    let dependencies: Vec<_> = (0..effect_count)
        .map(|_| {
            world.add_reaction(&[source], move |world| {
                let batch = values::<T>(world, world.reaction_changes(source));
                world.resource_mut::<Deliveries>().effects.push(batch);
            })
        })
        .collect();
    // Read every sibling's materialized result. A join running too early, or
    // more than once per batch, changes this checksum or the exact log below.
    world.add_reaction(&dependencies, move |world| {
        let sum = world
            .resource::<Deliveries>()
            .effects
            .iter()
            .rev()
            .take(effect_count)
            .flatten()
            .map(|(_, value)| u64::from(*value))
            .sum();
        world.resource_mut::<Deliveries>().joins.push(sum);
    });
    for _ in 0..clean_nodes {
        world.add_reaction(&[], |_| panic!("unrelated clean effect ran"));
    }
    source
}

fn oracle(batches: &[Batch], observers: usize, effects: usize) -> Deliveries {
    let mut log = Deliveries::default();
    for batch in batches {
        log.observers
            .extend((0..observers).map(|index| (index, batch.clone())));
        log.effects.extend((0..effects).map(|_| batch.clone()));
        log.joins.push(
            batch
                .iter()
                .map(|(_, value)| u64::from(*value))
                .sum::<u64>()
                * effects as u64,
        );
    }
    log
}

fn advance(world: &mut World) -> (Tick, Tick) {
    let last = world.change_tick();
    world.set_last_change_tick(last);
    (last, world.increment_change_tick())
}

fn assert_idle(world: &mut World, source: ReactionId) {
    let before = world.resource::<Deliveries>().clone();
    for _ in 0..3 {
        assert_eq!(world.flush_changed_observers(), 0);
        assert_eq!(world.flush_reactions(), 0);
        assert!(world.reaction_changes(source).is_empty());
        assert_eq!(*world.resource::<Deliveries>(), before);
    }
}

// Derive the expected coarse windows from explicitly chosen entities and the
// public location/layout contract, independently of the query's change index.
fn windows(world: &World, batch: &Batch) -> Vec<DirtyChunk> {
    let mut expected = BTreeMap::new();
    for &(entity, _) in batch {
        let location = world.entities().location(entity).unwrap();
        let table = world
            .archetypes()
            .get(location.archetype_id)
            .unwrap()
            .table();
        let width = table.rows_per_chunk();
        let start = location.row as usize / width * width;
        expected.insert(
            (location.archetype_id, start),
            DirtyChunk {
                archetype: location.archetype_id,
                start,
                end: (start + width).min(table.len()),
            },
        );
    }
    expected.into_values().collect()
}

#[test]
fn high_cardinality_query_observer_reaction_has_exact_batches_and_idle_work() {
    const ENTITIES: usize = 16_384;
    const OBSERVERS: usize = 128;
    const EFFECTS: usize = 64;
    let mut world = World::new();
    let entities = populate(&mut world, ENTITIES);
    let source = install::<Value>(&mut world, OBSERVERS, EFFECTS, 512);
    let cached_queries: Vec<_> = (0..64).map(|_| world.query::<&Value>()).collect();
    for query in &cached_queries {
        assert_eq!(query.iter(&world).count(), ENTITIES);
    }
    assert_idle(&mut world, source);

    let query = world.query::<&Value>();
    let tick = world.change_tick();
    let baseline = query.dirty_chunks_with_stats(&world, tick, tick);
    assert!(baseline.stats.journal_gap);
    assert!(baseline.chunks.is_empty());
    let writes = world.query::<(&Ordinal, &mut Value, &mut Other)>();
    let (last, this) = advance(&mut world);
    for (ordinal, mut value, mut other) in writes.iter_mut(&mut world) {
        if ordinal.0 % 31 == 0 || ordinal.0 == ENTITIES - 1 {
            value.0 += 10_000;
            value.0 += 7; // Repeated writes must produce one entity delivery.
        }
        if ordinal.0 % 17 == 0 {
            other.0 += 1; // Unobserved writes must not enter Value's report.
        }
    }
    let expected = sorted(
        entities
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 31 == 0 || *index == ENTITIES - 1)
            .map(|(index, &entity)| (entity, index as u32 + 10_007))
            .collect(),
    );
    assert_eq!(changed::<Value>(&mut world, last, this), expected);
    let report = query.dirty_chunks_with_stats(&world, last, this);
    assert_eq!(report.chunks, windows(&world, &expected));
    assert!(report.chunks.len() > 8, "exercise many candidate chunks");
    assert!(
        report.stats.reported_rows > expected.len(),
        "coarse windows are not exact deliveries"
    );
    assert!(!report.stats.journal_gap);
    assert_eq!(report.stats.journal_events, 2 * expected.len());
    assert_eq!(report.stats.journal_records_scanned, 2 * expected.len());
    assert_eq!(report.stats.candidate_archetype_checks, 2);
    assert_eq!(report.stats.summaries_tested, 0);
    assert_eq!(report.stats.archetype_index_entries_scanned, 0);

    // The public return counts observer callbacks, including the source's
    // internal observer; graph effects are checked separately by exact output.
    assert_eq!(world.flush_changed_observers(), OBSERVERS + 1);
    assert_eq!(
        *world.resource::<Deliveries>(),
        oracle(&[expected], OBSERVERS, EFFECTS)
    );
    for _ in 0..3 {
        let last = world.change_tick();
        assert_idle(&mut world, source);
        let idle = query.dirty_chunks_with_stats(&world, last, world.change_tick());
        assert!(idle.chunks.is_empty());
        assert!(!idle.stats.journal_gap);
        assert_eq!(idle.stats.journal_records_scanned, 0);
        assert_eq!(idle.stats.candidate_archetype_checks, 0);
        assert_eq!(idle.stats.summaries_tested, 0);
    }
}

#[test]
fn sparse_historical_replay_preserves_forward_delivery_and_exact_values() {
    let mut world = World::new();
    let entities = populate(&mut world, 2048);
    let source = install::<SparseValue>(&mut world, 1, 2, 64);
    assert_idle(&mut world, source);
    let query = world.query::<&SparseValue>();
    let writer = world.query::<&mut SparseValue>();
    let tick = world.change_tick();
    assert!(query
        .dirty_chunks_with_stats(&world, tick, tick)
        .chunks
        .is_empty());
    let (first_last, first_this) = advance(&mut world);
    let first = sorted(
        [7, 711, 2031]
            .map(|index| {
                writer.get_mut(&mut world, entities[index]).unwrap().0 += 10_000;
                (entities[index], 2 * index as u32 + 10_000)
            })
            .to_vec(),
    );
    let initial = query.dirty_chunks_with_stats(&world, first_last, first_this);
    assert_eq!(initial.chunks, windows(&world, &first));
    assert_eq!(initial.stats.sparse_index_events, first.len());
    assert_eq!(initial.stats.sparse_fallback_scans, 0);
    assert_eq!(
        changed::<SparseValue>(&mut world, first_last, first_this),
        first
    );
    assert_eq!(world.flush_changed_observers(), 2);
    let delivered = oracle(&[first.clone()], 1, 2);
    assert_eq!(*world.resource::<Deliveries>(), delivered);

    let (second_last, second_this) = advance(&mut world);
    let second = sorted(
        [129, 1000, 1900]
            .map(|index| {
                writer.get_mut(&mut world, entities[index]).unwrap().0 += 20_000;
                (entities[index], 2 * index as u32 + 20_000)
            })
            .to_vec(),
    );
    let replay = query.dirty_chunks_with_stats(&world, first_last, first_this);
    assert_eq!(replay.chunks, windows(&world, &first));
    assert_eq!(replay.stats.sparse_replay_sources, 1);
    assert_eq!(replay.stats.sparse_index_events, first.len());
    assert_eq!(replay.stats.sparse_index_gaps, 0);
    assert_eq!(replay.stats.sparse_fallback_scans, 0);
    assert_eq!(replay.stats.sparse_entities_checked, 0);
    assert_eq!(
        changed::<SparseValue>(&mut world, first_last, first_this),
        first
    );
    assert_eq!(world.flush_reactions(), 0);
    assert_eq!(*world.resource::<Deliveries>(), delivered);

    let forward = query.dirty_chunks_with_stats(&world, second_last, second_this);
    assert_eq!(forward.chunks, windows(&world, &second));
    assert_eq!(forward.stats.sparse_index_events, second.len());
    assert_eq!(forward.stats.sparse_replay_sources, 0);
    assert_eq!(forward.stats.sparse_fallback_scans, 0);
    assert_eq!(world.flush_changed_observers(), 2);
    assert_eq!(
        *world.resource::<Deliveries>(),
        oracle(&[first, second], 1, 2)
    );
    assert_idle(&mut world, source);
}

#[test]
fn lost_table_and_exact_entity_journals_recover_live_values_once() {
    // Real writes exceed both the public 8192-record column journal and the
    // exact entity index's 65536-event bound. Sparse stats confirm index loss;
    // the test never injects records or changes an internal retention limit.
    const WRITES: u32 = 70_000;
    assert!(WRITES as usize > COLUMN_CHANGE_LOG_CAPACITY);
    let mut world = World::new();
    let entities = populate(&mut world, 256);
    let source = install::<Value>(&mut world, 2, 3, 32);
    assert_idle(&mut world, source);
    let table = world.query::<&Value>();
    let sparse = world.query::<&SparseValue>();
    let tick = world.change_tick();
    assert!(table
        .dirty_chunks_with_stats(&world, tick, tick)
        .chunks
        .is_empty());
    assert!(sparse
        .dirty_chunks_with_stats(&world, tick, tick)
        .chunks
        .is_empty());
    let cursor = world.archetypes().column_changes_since(None).cursor;
    let (last, this) = advance(&mut world);
    world.get_mut::<Value>(entities[0]).unwrap().0 = 1_000_000;
    world.get_mut::<SparseValue>(entities[0]).unwrap().0 = 1_100_000;
    world.get_mut::<Value>(entities[2]).unwrap().0 = 9_000_000;
    world.get_mut::<SparseValue>(entities[2]).unwrap().0 = 9_100_000;
    for _ in 0..WRITES {
        world.get_mut::<Value>(entities[1]).unwrap().0 += 1;
        world.get_mut::<SparseValue>(entities[1]).unwrap().0 += 1;
    }
    assert!(world.despawn(entities[2]));
    let replacement = world.spawn((Value(3_000_000), SparseValue(3_100_000)));
    world.get_mut::<Value>(entities[255]).unwrap().0 = 2_000_000;
    world.get_mut::<SparseValue>(entities[255]).unwrap().0 = 2_100_000;
    let expected = sorted(vec![
        (entities[0], 1_000_000),
        (entities[1], 1 + WRITES),
        (entities[255], 2_000_000),
        (replacement, 3_000_000),
    ]);
    let expected_sparse = sorted(vec![
        (entities[0], 1_100_000),
        (entities[1], 2 + WRITES),
        (entities[255], 2_100_000),
        (replacement, 3_100_000),
    ]);
    assert!(world.archetypes().column_changes_since(Some(cursor)).gap);
    assert_eq!(changed::<Value>(&mut world, last, this), expected);
    assert_eq!(
        changed::<SparseValue>(&mut world, last, this),
        expected_sparse
    );
    let recovered = table.dirty_chunks_with_stats(&world, last, this);
    assert!(recovered.stats.journal_gap);
    assert_eq!(recovered.stats.journal_records_scanned, 0);
    assert!(!recovered.chunks.is_empty());
    for window in windows(&world, &expected) {
        assert!(recovered.chunks.contains(&window));
    }
    let sparse_recovered = sparse.dirty_chunks_with_stats(&world, last, this);
    assert_eq!(sparse_recovered.stats.sparse_index_gaps, 1);
    assert_eq!(
        sparse_recovered.stats.reported_rows,
        world.entity_count() as usize
    );
    assert_eq!(world.flush_changed_observers(), 3);
    assert_eq!(*world.resource::<Deliveries>(), oracle(&[expected], 2, 3));
    assert_idle(&mut world, source);
    let idle = table.dirty_chunks_with_stats(&world, this, world.change_tick());
    assert!(!idle.stats.journal_gap);
    assert!(idle.chunks.is_empty());
    let idle_sparse = sparse.dirty_chunks_with_stats(&world, this, world.change_tick());
    assert_eq!(idle_sparse.stats.sparse_index_gaps, 0);
    assert_eq!(idle_sparse.stats.sparse_fallback_scans, 0);
    assert!(idle_sparse.chunks.is_empty());
}

#[test]
fn repeated_restore_keeps_sources_but_discards_abandoned_timeline_changes() {
    let mut world = World::new();
    world.register_snapshot_component::<Value>();
    world.register_snapshot_component::<SparseValue>();
    world.register_snapshot_component::<Ordinal>();
    world.register_snapshot_component::<Other>();
    world.register_snapshot_component::<Marker>();
    world.register_snapshot_component::<Padding>();
    let entities = populate(&mut world, 1024);
    let source = install::<Value>(&mut world, 2, 2, 32);
    assert_idle(&mut world, source);
    let query = world.query::<&Value>();
    let writer = world.query::<&mut Value>();
    assert_eq!(query.iter(&world).count(), entities.len());
    let tick = world.change_tick();
    assert!(query
        .dirty_chunks_with_stats(&world, tick, tick)
        .chunks
        .is_empty());
    let snapshot = world.snapshot();
    let generation = world.archetype_generation();
    advance(&mut world);
    writer.get_mut(&mut world, entities[0]).unwrap().0 = 99_999;
    let abandoned = world.spawn((Value(88_888), Marker));
    assert_eq!(query.iter(&world).count(), entities.len() + 1);
    world.restore(&snapshot);
    assert_ne!(world.archetype_generation(), generation);
    assert!(!world.contains(abandoned));
    assert_eq!(world.reaction_source::<Value>(), source);
    let restored: Batch = sorted(
        world
            .query::<(Entity, &Value)>()
            .iter(&world)
            .map(|(entity, value)| (entity, value.0))
            .collect(),
    );
    let base: Batch = entities
        .iter()
        .enumerate()
        .map(|(index, &entity)| (entity, index as u32))
        .collect();
    assert_eq!(restored, base);
    assert_eq!(query.iter(&world).count(), entities.len());
    let reset = query.dirty_chunks_with_stats(&world, tick, tick);
    assert!(reset.stats.journal_gap);
    assert!(reset.chunks.is_empty());
    assert_idle(&mut world, source);

    let mut expected_batches = Vec::new();
    for (index, value) in [(1023, 555_555), (17, 777_777)] {
        world.restore(&snapshot);
        assert_eq!(world.change_tick(), tick);
        assert_eq!(world.reaction_source::<Value>(), source);
        // No flush or tick advance intervenes: this write has the snapshot's
        // own tick, while restored values themselves must remain undelivered.
        writer.get_mut(&mut world, entities[index]).unwrap().0 = value;
        assert_eq!(world.change_tick(), tick);
        let last = Tick::new(tick.get().wrapping_sub(1));
        let expected = vec![(entities[index], value)];
        assert_eq!(changed::<Value>(&mut world, last, tick), expected);
        let report = query.dirty_chunks_with_stats(&world, last, tick);
        // Restore establishes a fresh journal epoch. The first write after
        // restore is therefore a normal forward event, not a stale-history
        // gap, while the abandoned pre-restore write remains discarded.
        assert!(!report.stats.journal_gap);
        assert_eq!(report.chunks, windows(&world, &expected));
        assert_eq!(world.flush_changed_observers(), 3);
        expected_batches.push(expected);
        assert_eq!(
            *world.resource::<Deliveries>(),
            oracle(&expected_batches, 2, 2)
        );
        assert_idle(&mut world, source);
    }
}
