//! Public-API observer regressions, independent of dirty-query row discovery.
//!
//! All windows are short, maintained tick windows. Delivery means a readable
//! current-live component, not a removal/despawn event or an unbounded history.
//! Workers only use the exclusive, scope-joined query APIs; structural edits
//! and flushes always happen after those calls return. No unsafe concurrency.

use prism_ecs::component::StorageType;
use prism_ecs::prelude::*;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Component, Clone)]
struct TableValue {
    ordinal: usize,
    value: u32,
    // Multiple physical chunks, without thousands of entities.
    _padding: [u64; 256],
}

#[derive(Clone)]
struct SparseValue(u32);
impl Component for SparseValue {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[derive(Component, Clone)]
struct Marker;

// Keep genuinely ancient values resident while the other rows are written.
#[derive(Component)]
struct Unwritten;

trait Observed: Component {
    fn value(&self) -> u32;
}
impl Observed for TableValue {
    fn value(&self) -> u32 {
        self.value
    }
}
impl Observed for SparseValue {
    fn value(&self) -> u32 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Sample {
    entity: Entity,
    value: u32,
    added: Tick,
    changed: Tick,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Delivery {
    component: u8,
    rows: Vec<Sample>,
}
type Log = Arc<Mutex<Vec<Delivery>>>;

fn spawn(world: &mut World, ordinal: usize) -> Entity {
    world.spawn((
        TableValue {
            ordinal,
            value: ordinal as u32,
            _padding: [0; 256],
        },
        SparseValue(ordinal as u32),
    ))
}

fn warm_to(world: &mut World, target: u32) {
    // Exercise the production idle-clock API; skipped ticks contain no writes.
    let distance = target.wrapping_sub(world.change_tick().get());
    world.advance_change_tick_by(distance);
    assert_eq!(world.change_tick(), Tick::new(target));
}

fn sample<T: Observed>(world: &World, entity: Entity) -> Option<Sample> {
    let value = world.get::<T>(entity)?;
    let ticks = world
        .get_ticks::<T>(entity)
        .expect("live component has ticks");
    Some(Sample {
        entity,
        value: value.value(),
        added: ticks.added,
        changed: ticks.changed,
    })
}

fn samples<T: Observed>(world: &World, entities: &[Entity]) -> BTreeMap<Entity, Sample> {
    entities
        .iter()
        .filter_map(|&e| sample::<T>(world, e).map(|row| (e, row)))
        .collect()
}

fn assert_preserved<T: Observed>(
    world: &World,
    before: BTreeMap<Entity, Sample>,
    except: &[Entity],
) {
    for (entity, row) in before {
        if !except.contains(&entity) {
            assert_eq!(sample::<T>(world, entity), Some(row));
        }
    }
}

fn observe<T: Observed>(world: &mut World, component: u8, log: &Log) {
    let log = Arc::clone(log);
    world.observe_changed::<T, _>(move |ctx| {
        assert!(!ctx.is_empty());
        assert_eq!(ctx.len(), ctx.changed().len());
        assert!(ctx.changed().windows(2).all(|pair| {
            (pair[0].index(), pair[0].generation()) < (pair[1].index(), pair[1].generation())
        }));
        let rows = ctx
            .changed()
            .iter()
            .map(|&entity| {
                sample::<T>(ctx.world, entity)
                    .expect("observer must only deliver current-live components")
            })
            .collect();
        log.lock().unwrap().push(Delivery { component, rows });
    });
}

fn register(world: &mut World) -> Log {
    let log = Arc::new(Mutex::new(Vec::new()));
    observe::<TableValue>(world, 0, &log);
    observe::<SparseValue>(world, 1, &log);
    // Registration excludes existing values and advances the tick itself.
    assert_eq!(world.flush_changed_observers(), 0);
    assert!(log.lock().unwrap().is_empty());
    log
}

// No observer, dirty query, index traversal or Tick::is_newer_than in the oracle.
fn recent(stamp: Tick, last: Tick, now: Tick) -> bool {
    let width = now.get().wrapping_sub(last.get()).min(Tick::MAX_CHANGE_AGE);
    let age = now
        .get()
        .wrapping_sub(stamp.get())
        .min(Tick::MAX_CHANGE_AGE);
    age < width
}

fn expected<T: Observed>(world: &World, entities: &[Entity], last: Tick, now: Tick) -> Vec<Sample> {
    let mut rows: Vec<_> = samples::<T>(world, entities)
        .into_values()
        .filter(|row| recent(row.changed, last, now))
        .collect();
    rows.sort_unstable_by_key(|row| (row.entity.index(), row.entity.generation()));
    rows
}

fn flush_checked(world: &mut World, entities: &[Entity], last: Tick, log: &Log) -> Vec<Delivery> {
    let now = world.change_tick();
    assert!(now.get().wrapping_sub(last.get()) <= Tick::MAX_CHANGE_AGE);
    let expected: Vec<_> = [
        (0, expected::<TableValue>(world, entities, last, now)),
        (1, expected::<SparseValue>(world, entities, last, now)),
    ]
    .into_iter()
    .filter(|(_, rows)| !rows.is_empty())
    .map(|(component, rows)| Delivery { component, rows })
    .collect();
    assert!(log.lock().unwrap().is_empty());
    assert_eq!(world.flush_changed_observers(), expected.len());
    let delivered = std::mem::take(&mut *log.lock().unwrap());
    assert_eq!(delivered, expected, "observer window ({last:?}, {now:?}]");
    // A second, idle flush must neither replay successes nor invent changes.
    assert_eq!(world.flush_changed_observers(), 0);
    assert!(log.lock().unwrap().is_empty());
    delivered
}

fn write_rows(world: &mut World, workers: usize, phase: usize) {
    let query =
        world.query_filtered::<(&mut TableValue, Option<&mut SparseValue>), Without<Unwritten>>();
    let write = |(mut table, sparse): (Mut<'_, TableValue>, Option<Mut<'_, SparseValue>>)| {
        if table.ordinal % 3 == phase % 3 {
            // Repeated writes coalesce to one entity delivery.
            table.value += 7;
            table.value += 3;
        }
        if table.ordinal % 4 == phase % 4 {
            if let Some(mut sparse) = sparse {
                sparse.0 += 5;
                sparse.0 += 15;
            }
        }
    };
    #[cfg(feature = "multi_thread")]
    if workers != 0 {
        let pool = prism_tasks::TaskPool::with_threads(workers);
        query.par_for_each_mut(world, &pool, 1, write);
        return; // The public call has already joined every worker.
    }
    assert_eq!(workers, 0, "workers require multi_thread");
    query.iter_mut(world).for_each(write);
}

fn checked_row_writes(world: &mut World, entities: &[Entity], workers: usize, phase: usize) {
    let table = samples::<TableValue>(world, entities);
    let sparse = samples::<SparseValue>(world, entities);
    let ordinals: BTreeMap<_, _> = table
        .keys()
        .filter(|&&e| world.get::<Unwritten>(e).is_none())
        .map(|&e| (e, world.get::<TableValue>(e).unwrap().ordinal))
        .collect();
    let now = world.change_tick();
    write_rows(world, workers, phase);
    assert_eq!(world.change_tick(), now);
    // Independent intended-write checks prevent a missing stamp AND event
    // from making both the observer and a tick-based reference silently agree.
    for (entity, mut before) in table {
        if ordinals.get(&entity).is_some_and(|&n| n % 3 == phase % 3) {
            before.value += 10;
            before.changed = now;
        }
        assert_eq!(sample::<TableValue>(world, entity), Some(before));
    }
    for (entity, mut before) in sparse {
        if ordinals.get(&entity).is_some_and(|&n| n % 4 == phase % 4) {
            before.value += 20;
            before.changed = now;
        }
        assert_eq!(sample::<SparseValue>(world, entity), Some(before));
    }
}

fn checked_chunk_writes(world: &mut World, entities: &[Entity], workers: usize) {
    let table = samples::<TableValue>(world, entities);
    let sparse = samples::<SparseValue>(world, entities);
    let now = world.change_tick();
    let query = world.query::<&mut TableValue>();
    #[cfg(feature = "multi_thread")]
    if workers != 0 {
        let pool = prism_tasks::TaskPool::with_threads(workers);
        // Bare mutable slices stamp EVERY handed-out row, even untouched ones.
        query.par_chunks_mut(world, &pool, 1, |slice| {
            for value in slice {
                if value.ordinal % 2 == 0 {
                    value.value += 1;
                }
            }
        });
    }
    if workers == 0 {
        for mut value in query.iter_mut(world) {
            value.set_changed(); // Match the conservative whole-slice barrier.
            if value.ordinal % 2 == 0 {
                value.value += 1;
            }
        }
    }
    #[cfg(not(feature = "multi_thread"))]
    assert_eq!(workers, 0);
    for (entity, mut before) in table {
        if world.get::<TableValue>(entity).unwrap().ordinal % 2 == 0 {
            before.value += 1;
        }
        before.changed = now;
        assert_eq!(sample::<TableValue>(world, entity), Some(before));
    }
    assert_eq!(samples::<SparseValue>(world, entities), sparse);
}

fn structural_matrix(workers: usize, warm_target: Option<u32>) -> Vec<Delivery> {
    let mut world = World::new();
    if let Some(target) = warm_target {
        warm_to(&mut world, target);
    }
    let mut entities: Vec<_> = (0..73).map(|n| spawn(&mut world, n)).collect();
    let table_only = spawn(&mut world, 100);
    assert!(world.remove::<SparseValue>(table_only));
    let sparse_only = spawn(&mut world, 101);
    assert!(world.remove::<TableValue>(sparse_only));
    entities.extend([table_only, sparse_only]);
    let log = register(&mut world);
    let baseline = world.change_tick();
    if let Some(target) = warm_target {
        assert_eq!(baseline.get(), target.wrapping_add(3));
    }
    let mut saw_zero = false;
    let mut deliveries = Vec::new();
    for phase in 0..4 {
        assert_eq!(
            world.change_tick().get(),
            baseline.get().wrapping_add(4 * phase as u32)
        );
        saw_zero |= world.change_tick() == Tick::new(0);
        let last = Tick::new(world.change_tick().get().wrapping_sub(1));
        checked_row_writes(&mut world, &entities, workers, phase);
        let table_before = samples::<TableValue>(&world, &entities);
        let sparse_before = samples::<SparseValue>(&world, &entities);
        let now = world.change_tick();
        let start = phase * 12;
        // All structural edits occur after joined row writes, before delivery.
        assert!(world.insert(entities[start], Marker));
        if phase != 0 {
            assert!(world.remove::<Marker>(entities[start - 12]));
        }
        assert!(world.remove::<SparseValue>(entities[start + 1]));
        let reinserted = entities[start + 2];
        assert!(world.remove::<SparseValue>(reinserted));
        assert!(world.insert(reinserted, SparseValue(700 + phase as u32)));
        let sparse = sample::<SparseValue>(&world, reinserted).unwrap();
        assert_eq!((sparse.added, sparse.changed), (now, now));
        assert_eq!(sparse.value, 700 + phase as u32);
        let replaced = entities[start + 3];
        assert!(world.remove::<TableValue>(replaced));
        assert!(world.insert(
            replaced,
            TableValue {
                ordinal: start + 3,
                value: 900 + phase as u32,
                _padding: [0; 256],
            }
        ));
        let table = sample::<TableValue>(&world, replaced).unwrap();
        assert_eq!((table.added, table.changed), (now, now));
        assert_eq!(table.value, 900 + phase as u32);
        assert!(world.remove::<TableValue>(entities[start + 4]));
        let dead = entities[start + 5];
        assert!(world.despawn(dead));
        let replacement = spawn(&mut world, 200 + phase);
        assert_eq!(replacement.index(), dead.index());
        assert_ne!(replacement.generation(), dead.generation());
        assert!(sample::<TableValue>(&world, dead).is_none());
        assert!(sample::<SparseValue>(&world, dead).is_none());
        for row in [
            sample::<TableValue>(&world, replacement).unwrap(),
            sample::<SparseValue>(&world, replacement).unwrap(),
        ] {
            assert_eq!((row.added, row.changed), (now, now));
            assert_eq!(row.value, (200 + phase) as u32);
        }
        assert!(sample::<SparseValue>(&world, entities[start + 1]).is_none());
        assert!(sample::<TableValue>(&world, entities[start + 4]).is_none());
        // Migrations/swap-removes must not stamp surviving sibling columns
        // or whichever unrelated entity was relocated into a vacated row.
        assert_preserved::<TableValue>(
            &world,
            table_before,
            &[replaced, entities[start + 4], dead],
        );
        assert_preserved::<SparseValue>(
            &world,
            sparse_before,
            &[entities[start + 1], reinserted, dead],
        );
        entities.push(replacement); // Keep stale handles in the reference too.
        deliveries.extend(flush_checked(&mut world, &entities, last, &log));

        assert_eq!(
            world.change_tick().get(),
            baseline.get().wrapping_add(4 * phase as u32 + 2)
        );
        saw_zero |= world.change_tick() == Tick::new(0);
        let last = Tick::new(world.change_tick().get().wrapping_sub(1));
        checked_chunk_writes(&mut world, &entities, workers);
        let table_before = samples::<TableValue>(&world, &entities);
        let sparse_before = samples::<SparseValue>(&world, &entities);
        // Migrate/swap rows AFTER the chunk barrier and BEFORE its flush too.
        assert!(world.insert(entities[start + 6], Marker));
        assert_preserved::<TableValue>(&world, table_before, &[]);
        assert_preserved::<SparseValue>(&world, sparse_before, &[]);
        deliveries.extend(flush_checked(&mut world, &entities, last, &log));
    }
    if warm_target.is_some() {
        assert!(
            saw_zero,
            "row or chunk barrier and structural edits must run at actual tick 0"
        );
    }
    deliveries
}

fn restore_matrix(workers: usize, warm_target: Option<u32>) -> Vec<Delivery> {
    let mut world = World::new();
    world.register_snapshot_component::<TableValue>();
    world.register_snapshot_component::<SparseValue>();
    world.register_snapshot_component::<Marker>();
    if let Some(target) = warm_target {
        warm_to(&mut world, target);
    }
    let mut entities: Vec<_> = (0..17).map(|n| spawn(&mut world, n)).collect();
    let log = register(&mut world);
    let tick = world.change_tick();
    if let Some(target) = warm_target {
        assert_eq!(tick.get(), target.wrapping_add(3));
    }
    let table = samples::<TableValue>(&world, &entities);
    let sparse = samples::<SparseValue>(&world, &entities);
    let snapshot = world.snapshot();

    world.increment_change_tick();
    checked_row_writes(&mut world, &entities, workers, 0);
    assert!(world.remove::<SparseValue>(entities[1]));
    assert!(world.insert(entities[2], Marker));
    assert!(world.despawn(entities[3]));
    let abandoned = spawn(&mut world, 999);
    assert_eq!(abandoned.index(), entities[3].index());
    assert_ne!(abandoned.generation(), entities[3].generation());
    entities.push(abandoned);
    if warm_target.is_some() {
        // For snapshot MAX-1, these abandoned chunk writes happen at 0;
        // for snapshot 0, the restored same-tick writes below happen at 0.
        world.increment_change_tick();
        assert_eq!(world.change_tick().get(), tick.get().wrapping_add(2));
        checked_chunk_writes(&mut world, &entities, workers);
        if tick != Tick::new(0) {
            assert_eq!(world.change_tick(), Tick::new(0));
        }
    }
    // Leave indexed writes unflushed on the abandoned timeline.
    world.restore(&snapshot);
    assert_eq!(world.change_tick(), tick);
    assert_eq!(samples::<TableValue>(&world, &entities), table);
    assert_eq!(samples::<SparseValue>(&world, &entities), sparse);
    assert!(sample::<TableValue>(&world, abandoned).is_none());
    assert!(sample::<SparseValue>(&world, abandoned).is_none());
    assert!(world.get::<Marker>(entities[2]).is_none());
    assert_eq!(world.flush_changed_observers(), 0);
    assert!(log.lock().unwrap().is_empty());

    // Restore a second time and write IMMEDIATELY at the exact snapshot tick.
    world.restore(&snapshot);
    checked_row_writes(&mut world, &entities, workers, 1);
    assert_eq!(world.change_tick(), tick);
    let mut delivered = flush_checked(
        &mut world,
        &entities,
        Tick::new(tick.get().wrapping_sub(1)),
        &log,
    );
    let last = Tick::new(world.change_tick().get().wrapping_sub(1));
    checked_chunk_writes(&mut world, &entities, workers);
    delivered.extend(flush_checked(&mut world, &entities, last, &log));
    delivered
}

#[test]
fn serial_observers_cover_joined_barriers_and_structural_churn() {
    assert!(!structural_matrix(0, None).is_empty());
}

#[test]
fn serial_observers_restore_exact_stamps_and_discard_abandoned_writes() {
    assert!(!restore_matrix(0, None).is_empty());
}

#[cfg(feature = "multi_thread")]
#[test]
fn joined_one_and_eight_worker_observers_match_serial_delivery() {
    let structural = structural_matrix(0, None);
    let restored = restore_matrix(0, None);
    for workers in [1, 8] {
        assert_eq!(
            structural_matrix(workers, None),
            structural,
            "{workers} workers"
        );
        assert_eq!(restore_matrix(workers, None), restored, "{workers} workers");
    }
}

#[test]
fn actual_rollover_observers_cover_row_and_chunk_barriers_with_structural_churn() {
    // First chunk barrier at 0, then a separate world with first row barrier
    // at 0. Compare complete values/ticks/order, not merely callback counts.
    for target in [u32::MAX - 4, u32::MAX - 2] {
        let serial = structural_matrix(0, Some(target));
        assert!(!serial.is_empty());
        #[cfg(feature = "multi_thread")]
        for workers in [1, 8] {
            assert_eq!(
                structural_matrix(workers, Some(target)),
                serial,
                "{workers} workers, warm {target}"
            );
        }
    }
}

#[test]
fn actual_rollover_observers_restore_abandoned_writes_and_accept_same_tick_writes() {
    for target in [u32::MAX - 4, u32::MAX - 2] {
        let serial = restore_matrix(0, Some(target));
        assert!(!serial.is_empty());
        #[cfg(feature = "multi_thread")]
        for workers in [1, 8] {
            assert_eq!(
                restore_matrix(workers, Some(target)),
                serial,
                "{workers} workers, warm {target}"
            );
        }
    }
}

fn retained_gap_matrix(single_tick: bool, rollover_restore: bool) {
    let mut world = World::new();
    world.register_snapshot_component::<TableValue>();
    world.register_snapshot_component::<SparseValue>();
    if rollover_restore {
        warm_to(&mut world, u32::MAX - 4);
    }
    let mut entities: Vec<_> = (0..8).map(|n| spawn(&mut world, n)).collect();
    assert!(world.remove::<SparseValue>(entities[5])); // table-only fallback row
    assert!(world.remove::<TableValue>(entities[6])); // sparse-only fallback row
    let log = register(&mut world);
    // This independent public probe proves an actual shared sparse-log gap;
    // it does not consume the observer's cursor or form the delivery oracle.
    let probe = world.query::<&SparseValue>();
    let last = world.change_tick();
    let saved_table = samples::<TableValue>(&world, &entities);
    let saved_sparse = samples::<SparseValue>(&world, &entities);
    let snapshot = world.snapshot();
    if rollover_restore {
        assert_eq!(last, Tick::new(u32::MAX - 1));
    }
    let baseline = probe.dirty_chunks_with_stats(&world, last, last);
    assert!(baseline.chunks.is_empty());
    assert_eq!(baseline.stats.sparse_fallback_scans, 0);
    // A window starting before activation intentionally scans current members
    // instead of reading the retained log, so it cannot demonstrate log loss.
    // Start after this acknowledged baseline and write in the following tick.
    world.increment_change_tick();
    let early = world.change_tick();
    for &entity in &entities[..3] {
        world.get_mut::<TableValue>(entity).unwrap().value += 100;
        world.get_mut::<SparseValue>(entity).unwrap().0 += 100;
    }
    world.get_mut::<TableValue>(entities[5]).unwrap().value += 100;
    world.get_mut::<SparseValue>(entities[6]).unwrap().0 += 100;
    assert!(world.remove::<TableValue>(entities[1]));
    assert!(world.remove::<SparseValue>(entities[1]));
    let dead = entities[2];
    assert!(world.despawn(dead));
    let replacement = spawn(&mut world, 999);
    assert_eq!(replacement.index(), dead.index());
    assert_ne!(replacement.generation(), dead.generation());
    entities.push(replacement);

    if rollover_restore && single_tick {
        // Keep early live writes at MAX, but overflow a single batch at 0.
        world.increment_change_tick();
        assert_eq!(world.change_tick(), Tick::new(0));
    }

    // Current bounds are 4096 batches / 65_536 events per component log.
    // Exceed each via ordinary public writes, not private event injection.
    // If retention grows, the explicit gap assertion demands updating this
    // stress size rather than silently pretending the fallback was exercised.
    let writes = if single_tick { 65_568 } else { 4_128 };
    for _ in 0..writes {
        if !single_tick {
            world.increment_change_tick();
        }
        world.get_mut::<TableValue>(entities[3]).unwrap().value += 1;
        world.get_mut::<SparseValue>(entities[3]).unwrap().0 += 1;
    }
    let now = world.change_tick();
    if rollover_restore {
        assert_eq!(early, Tick::new(u32::MAX));
        assert_eq!(now.get(), if single_tick { 0 } else { writes - 1 });
    }
    let old = sample::<TableValue>(&world, entities[0]).unwrap();
    assert_eq!((old.value, old.changed), (100, early));
    let old = sample::<SparseValue>(&world, entities[0]).unwrap();
    assert_eq!((old.value, old.changed), (100, early));
    for row in [
        sample::<TableValue>(&world, entities[3]).unwrap(),
        sample::<SparseValue>(&world, entities[3]).unwrap(),
    ] {
        assert_eq!(row.changed, now);
        assert_eq!(row.value, 3 + writes);
    }
    let gap = probe.dirty_chunks_with_stats(&world, last, now);
    assert_eq!(gap.stats.sparse_index_gaps, 1);
    assert_eq!(gap.stats.sparse_fallback_scans, 1);
    let delivered = flush_checked(&mut world, &entities, last, &log);
    assert_eq!(delivered.len(), 2);
    for delivery in &delivered {
        assert!(delivery.rows.iter().any(|row| row.entity == entities[0]));
        assert!(delivery.rows.iter().any(|row| row.entity == replacement));
        assert!(!delivery.rows.iter().any(|row| row.entity == dead));
        assert!(!delivery.rows.iter().any(|row| row.entity == entities[1]));
        assert!(!delivery.rows.iter().any(|row| row.entity == entities[4]));
    }
    // Recovery re-establishes a forward cursor: later writes are delivered
    // exactly once, including writes after a single-tick overflow discard.
    let recovery_last = Tick::new(world.change_tick().get().wrapping_sub(1));
    world.get_mut::<TableValue>(entities[4]).unwrap().value += 1;
    world.get_mut::<SparseValue>(entities[4]).unwrap().0 += 1;
    let delivered = flush_checked(&mut world, &entities, recovery_last, &log);
    assert_eq!(delivered.len(), 2);
    for delivery in delivered {
        assert_eq!(delivery.rows.len(), 1);
        assert_eq!(delivery.rows[0].entity, entities[4]);
    }
    if rollover_restore {
        // Recovering from a real gap must not prevent the SAME observers from
        // dropping that whole timeline on restore, including new unread writes.
        world.get_mut::<TableValue>(replacement).unwrap().value += 1;
        world.get_mut::<SparseValue>(replacement).unwrap().0 += 1;
        for _ in 0..2 {
            world.restore(&snapshot);
            assert_eq!(world.change_tick(), last);
            assert_eq!(samples::<TableValue>(&world, &entities), saved_table);
            assert_eq!(samples::<SparseValue>(&world, &entities), saved_sparse);
            assert!(world.entities().location(replacement).is_none());
            assert_eq!(world.flush_changed_observers(), 0);
            assert!(log.lock().unwrap().is_empty());

            world.restore(&snapshot);
            let entity = entities[4];
            world.get_mut::<TableValue>(entity).unwrap().value += 7;
            world.get_mut::<SparseValue>(entity).unwrap().0 += 9;
            let table = sample::<TableValue>(&world, entity).unwrap();
            let sparse = sample::<SparseValue>(&world, entity).unwrap();
            assert_eq!(
                (table.value, table.added, table.changed),
                (11, saved_table[&entity].added, last)
            );
            assert_eq!(
                (sparse.value, sparse.added, sparse.changed),
                (13, saved_sparse[&entity].added, last)
            );
            let delivered = flush_checked(
                &mut world,
                &entities,
                Tick::new(last.get().wrapping_sub(1)),
                &log,
            );
            assert_eq!(delivered.len(), 2);
            assert_eq!(delivered[0].rows, vec![table]);
            assert_eq!(delivered[1].rows, vec![sparse]);
            // MAX-1 delivery plus the idle flush crosses the real clock to 0.
            assert_eq!(world.change_tick(), Tick::new(0));
            let prior = Tick::new(u32::MAX);
            world.get_mut::<TableValue>(entities[0]).unwrap().value += 3;
            world.get_mut::<SparseValue>(entities[0]).unwrap().0 += 5;
            assert_eq!(sample::<TableValue>(&world, entities[0]).unwrap().value, 3);
            assert_eq!(sample::<SparseValue>(&world, entities[0]).unwrap().value, 5);
            assert_eq!(
                sample::<TableValue>(&world, entities[0]).unwrap().changed,
                Tick::new(0)
            );
            assert_eq!(
                sample::<SparseValue>(&world, entities[0]).unwrap().changed,
                Tick::new(0)
            );
            let delivered = flush_checked(&mut world, &entities, prior, &log);
            assert_eq!(delivered.len(), 2);
            for delivery in delivered {
                assert_eq!(delivery.rows.len(), 1);
                assert_eq!(delivery.rows[0].entity, entities[0]);
            }
        }
    }
}

#[test]
fn retained_batch_gap_rebuilds_current_live_observer_delivery() {
    retained_gap_matrix(false, false);
}

#[test]
fn single_tick_event_overflow_rebuilds_current_live_observer_delivery() {
    retained_gap_matrix(true, false);
}

#[test]
fn actual_rollover_retained_batch_gap_then_restore_discards_abandoned_delivery() {
    retained_gap_matrix(false, true);
}

#[test]
fn actual_rollover_single_tick_overflow_then_restore_discards_abandoned_delivery() {
    retained_gap_matrix(true, true);
}

#[test]
fn maintained_world_rollover_delivers_joined_writes_without_resurrecting_ancient_rows() {
    let mut world = World::new();
    let ancient = spawn(&mut world, 1000);
    assert!(world.insert(ancient, Unwritten));
    // Advance the actual public World clock, including periodic maintenance.
    // A synthetic Tick or setting last_change_tick does not test World rollover.
    let target = u32::MAX - 4;
    let distance = target.wrapping_sub(world.change_tick().get());
    world.advance_change_tick_by(distance);
    let mut entities: Vec<_> = (0..25).map(|n| spawn(&mut world, n)).collect();
    entities.push(ancient);
    let log = register(&mut world);
    // Each registration and the initial flush advances the public clock.
    assert_eq!(world.change_tick(), Tick::new(u32::MAX - 1));
    #[cfg(feature = "multi_thread")]
    let worker_counts = [0, 1, 8, 0, 1, 8];
    #[cfg(not(feature = "multi_thread"))]
    let worker_counts = [0; 6];
    let mut saw_zero = false;
    for (phase, workers) in worker_counts.into_iter().enumerate() {
        let now = world.change_tick();
        saw_zero |= now == Tick::new(0);
        let last = Tick::new(now.get().wrapping_sub(1));
        // Both old components must remain present, not merely disappear from
        // delivery because they were removed. Without<Unwritten> excludes the
        // old row before fetching mutable values or installing write barriers.
        let ancient_table = sample::<TableValue>(&world, ancient).unwrap();
        let ancient_sparse = sample::<SparseValue>(&world, ancient).unwrap();
        assert_eq!((ancient_table.value, ancient_sparse.value), (1000, 1000));
        assert!(!recent(ancient_table.changed, last, now));
        assert!(!recent(ancient_sparse.changed, last, now));
        checked_row_writes(&mut world, &entities, workers, phase);
        let dead = entities[phase];
        assert!(world.despawn(dead));
        let replacement = spawn(&mut world, 200 + phase);
        assert_eq!(replacement.index(), dead.index());
        assert_ne!(replacement.generation(), dead.generation());
        entities.push(replacement);
        assert_eq!(sample::<TableValue>(&world, ancient), Some(ancient_table));
        assert_eq!(sample::<SparseValue>(&world, ancient), Some(ancient_sparse));
        let delivered = flush_checked(&mut world, &entities, last, &log);
        assert_eq!(delivered.len(), 2);
        for delivery in delivered {
            assert!(!delivery.rows.iter().any(|row| row.entity == dead));
            assert!(!delivery.rows.iter().any(|row| row.entity == ancient));
            assert!(delivery.rows.iter().any(|row| row.entity == replacement));
        }
        world.check_change_ticks();
    }
    assert!(
        saw_zero,
        "joined writes must occur at the actual wrapped tick 0"
    );
}
