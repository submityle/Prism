//! Public-API dirty-window regressions, independent of the private index tests.
//!
//! Exact rows use clamped tick ages; dirty chunks are only a conservative
//! superset. Strict interval interpretation is limited to MAX_CHANGE_AGE and
//! requires periodic tick maintenance. In particular an over-age/full-cycle
//! observer window is NOT a promise to recover every write in that interval.

use prism_ecs::component::StorageType;
use prism_ecs::prelude::*;
use std::collections::BTreeSet;

#[derive(Component, Clone)]
struct TableValue {
    ordinal: usize,
    value: u32,
    // Force several physical chunks without a large entity count.
    _padding: [u64; 256],
}

#[derive(Clone, Copy)]
struct SparseValue(u32);

impl Component for SparseValue {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[derive(Component, Clone)]
struct Marker;

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

fn advance(world: &mut World) -> Tick {
    world.set_last_change_tick(world.change_tick());
    world.increment_change_tick()
}

fn warm_to(world: &mut World, target: u32) {
    let distance = target.wrapping_sub(world.change_tick().get());
    world.advance_change_tick_by(distance);
    assert_eq!(world.change_tick(), Tick::new(target));
}

// Deliberately independent of Tick::is_newer_than and query/index traversal.
fn recent(stamp: Tick, last: Tick, now: Tick) -> bool {
    let width = now.get().wrapping_sub(last.get()).min(Tick::MAX_CHANGE_AGE);
    let age = now
        .get()
        .wrapping_sub(stamp.get())
        .min(Tick::MAX_CHANGE_AGE);
    age < width
}

fn reference<T: Component>(
    world: &World,
    entities: &[Entity],
    last: Tick,
    now: Tick,
    added: bool,
) -> BTreeSet<Entity> {
    entities
        .iter()
        .copied()
        .filter(|&entity| {
            world.get_ticks::<T>(entity).is_some_and(|ticks| {
                recent(if added { ticks.added } else { ticks.changed }, last, now)
            })
        })
        .collect()
}

fn assert_exact<T: Component>(world: &mut World, entities: &[Entity], last: Tick, now: Tick) {
    let changed = world.query_filtered::<Entity, Changed<T>>();
    let rows: Vec<_> = changed.iter_with_ticks(world, last, now).collect();
    let actual: BTreeSet<_> = rows.iter().copied().collect();
    assert_eq!(actual.len(), rows.len(), "duplicate Changed row");
    assert_eq!(
        actual,
        reference::<T>(world, entities, last, now, false),
        "Changed<{}> ({last:?}, {now:?}]",
        std::any::type_name::<T>()
    );
    let added = world.query_filtered::<Entity, Added<T>>();
    let rows: Vec<_> = added.iter_with_ticks(world, last, now).collect();
    let actual: BTreeSet<_> = rows.iter().copied().collect();
    assert_eq!(actual.len(), rows.len(), "duplicate Added row");
    assert_eq!(
        actual,
        reference::<T>(world, entities, last, now, true),
        "Added<{}> ({last:?}, {now:?}]",
        std::any::type_name::<T>()
    );
}

#[track_caller]
fn assert_covers(world: &World, chunks: &[DirtyChunk], expected: &BTreeSet<Entity>) {
    let mut unique = BTreeSet::new();
    for chunk in chunks {
        let table = world.archetypes().get(chunk.archetype).unwrap().table();
        assert!(
            chunk.start < chunk.end && chunk.end <= table.len(),
            "{chunk:?}"
        );
        assert_eq!(chunk.start % table.rows_per_chunk(), 0);
        assert_eq!(
            chunk.end,
            (chunk.start + table.rows_per_chunk()).min(table.len())
        );
        assert!(
            unique.insert((chunk.archetype, chunk.start)),
            "duplicate {chunk:?}"
        );
    }
    for &entity in expected {
        let location = world
            .entities()
            .location(entity)
            .expect("reference is live");
        assert!(
            chunks
                .iter()
                .any(|chunk| chunk.archetype == location.archetype_id
                    && chunk.start <= location.row as usize
                    && (location.row as usize) < chunk.end),
            "dirty chunks missed {entity:?} at {location:?}: {chunks:?}"
        );
    }
}

#[test]
fn actual_world_rollover_and_sparse_indexed_window_matrix() {
    let mut world = World::new();
    let ancient = spawn(&mut world, 1000);

    // There is no public current-tick setter. Exercise actual increments, not
    // a private-layout cast or set_last_change_tick (which only sets baseline).
    // This bounded warm-up is intentionally slower in unoptimized test builds.
    let target = u32::MAX - 3;
    let increments = target.wrapping_sub(world.change_tick().get());
    let started = std::time::Instant::now();
    world.advance_change_tick_by(increments);
    assert_eq!(world.change_tick(), Tick::new(target));
    eprintln!(
        "actual World warm-up: {increments} public increments, tick={}, elapsed={:?}",
        world.change_tick().get(),
        started.elapsed()
    );
    let activation = world.change_tick();
    let mut entities: Vec<_> = (0..24).map(|i| spawn(&mut world, i)).collect();
    entities.push(ancient);
    let table = world.query::<&TableValue>();
    let sparse = world.query::<&SparseValue>();
    table.dirty_chunks(&world, activation, activation);
    sparse.dirty_chunks(&world, activation, activation);

    let mut boundaries = vec![activation];
    for (phase, raw) in [u32::MAX - 2, u32::MAX - 1, u32::MAX, 0, 1, 2]
        .into_iter()
        .enumerate()
    {
        let tick = advance(&mut world);
        assert_eq!(
            tick,
            Tick::new(raw),
            "actual World clock must cross MAX -> 0"
        );
        for &entity in &entities[phase * 3..phase * 3 + 3] {
            world.get_mut::<TableValue>(entity).unwrap().value += 1;
            world.get_mut::<SparseValue>(entity).unwrap().0 += 1;
        }
        // Later writes must not make a historical coarse report lose other
        // earlier rows in the same chunk; exact rows use their latest stamp.
        world.get_mut::<SparseValue>(entities[23]).unwrap().0 += 1;
        boundaries.push(tick);
        let last = boundaries[phase];
        assert_exact::<TableValue>(&mut world, &entities, last, tick);
        assert_exact::<SparseValue>(&mut world, &entities, last, tick);
        assert_covers(
            &world,
            &table.dirty_chunks(&world, last, tick),
            &reference::<TableValue>(&world, &entities, last, tick, false),
        );
        let report = sparse.dirty_chunks_with_stats(&world, last, tick);
        assert_eq!(report.stats.sparse_fallback_scans, 0);
        assert_eq!(report.stats.sparse_index_gaps, 0);
        assert!(
            report.stats.sparse_index_events >= 4,
            "must exercise the index: {:?}",
            report.stats
        );
        assert_covers(
            &world,
            &report.chunks,
            &reference::<SparseValue>(&world, &entities, last, tick, false),
        );
    }
    eprintln!("actual World rollover verified: MAX-3 -> MAX-2 -> MAX-1 -> MAX -> 0 -> 1 -> 2");

    // Every chronological pair, including empty, historical, overlapping and
    // repeated windows across MAX/0, reusing the SAME consumer after every
    // rewind. No timing/implementation-cost assertions.
    for (i, &last) in boundaries.iter().enumerate() {
        for &now in &boundaries[i..] {
            assert_exact::<TableValue>(&mut world, &entities, last, now);
            assert_exact::<SparseValue>(&mut world, &entities, last, now);
            for _ in 0..2 {
                let report = sparse.dirty_chunks_with_stats(&world, last, now);
                assert_eq!(report.stats.sparse_fallback_scans, 0, "({last:?}, {now:?}]");
                assert_eq!(report.stats.sparse_index_gaps, 0);
                if now != world.change_tick() {
                    assert_eq!(report.stats.sparse_replay_sources, 1);
                }
                assert_covers(
                    &world,
                    &report.chunks,
                    &reference::<SparseValue>(&world, &entities, last, now, false),
                );
                assert_covers(
                    &world,
                    &table.dirty_chunks(&world, last, now),
                    &reference::<TableValue>(&world, &entities, last, now, false),
                );
            }
        }
    }

    // Age boundary and almost/full-cycle windows: row truth is CLAMPED age,
    // not the unbounded mathematical interval. Do not require coarse filtering
    // or full historical recovery outside the supported age horizon.
    let now = world.change_tick();
    for width in [
        Tick::MAX_CHANGE_AGE - 1,
        Tick::MAX_CHANGE_AGE,
        Tick::MAX_CHANGE_AGE + 1,
        u32::MAX,
        0,
    ] {
        let last = Tick::new(now.get().wrapping_sub(width));
        assert_exact::<TableValue>(&mut world, &entities, last, now);
        assert_exact::<SparseValue>(&mut world, &entities, last, now);
        let report = sparse.dirty_chunks_with_stats(&world, last, now);
        let expected = if width <= Tick::MAX_CHANGE_AGE {
            reference::<SparseValue>(&world, &entities, last, now, false)
        } else {
            BTreeSet::new() // Validate geometry only, not over-age recovery.
        };
        assert_covers(&world, &report.chunks, &expected);
    }
    world.check_change_ticks();
    for ticks in [
        world.get_ticks::<TableValue>(ancient).unwrap(),
        world.get_ticks::<SparseValue>(ancient).unwrap(),
    ] {
        assert!(ticks.changed.age_since(now) <= Tick::MAX_CHANGE_AGE);
        assert!(
            !recent(ticks.changed, Tick::MAX, now),
            "ancient row aliased a recent write"
        );
    }
}

fn historical_empty_window_preserves_coverage<T: Component>() {
    let mut world = World::new();
    let entity = spawn(&mut world, 0);
    let query = world.query::<&T>();
    let last = world.change_tick();
    query.dirty_chunks(&world, last, last);
    let now = advance(&mut world);
    world.get_mut::<T>(entity).unwrap().set_changed();
    let expected = reference::<T>(&world, &[entity], last, now, false);
    assert_eq!(expected, BTreeSet::from([entity]));
    assert_covers(&world, &query.dirty_chunks(&world, last, now), &expected);

    // All three requests have width <= 1. A historical empty read must not
    // erase the fact that the forward cursor already consumed (last, now].
    query.dirty_chunks(&world, last, last);
    let report = query.dirty_chunks_with_stats(&world, last, now);
    assert_covers(&world, &report.chunks, &expected);
}

#[test]
fn sparse_historical_empty_window_preserves_forward_coverage() {
    historical_empty_window_preserves_coverage::<SparseValue>();
}

#[test]
fn table_historical_empty_window_preserves_forward_coverage() {
    historical_empty_window_preserves_coverage::<TableValue>();
}

fn historical_read_preserves_unread_forward_changes<T: Component>() {
    let mut world = World::new();
    let entity = spawn(&mut world, 0);
    let query = world.query::<&T>();
    let last = world.change_tick();
    query.dirty_chunks(&world, last, last);
    let now = advance(&mut world);
    world.get_mut::<T>(entity).unwrap().set_changed();

    // A historical replay consumes no forward events. It must neither erase
    // the established boundary nor unnecessarily advance it to the world tick.
    query.dirty_chunks(&world, last, last);
    let report = query.dirty_chunks_with_stats(&world, last, now);
    assert_eq!(report.stats.sparse_replay_sources, 0);
    assert_covers(&world, &report.chunks, &BTreeSet::from([entity]));

    // A CURRENT empty window, in contrast, acknowledges all available cursor
    // records despite filtering their ticks out. A wider same-tick request
    // must recover those records from history rather than an exhausted cursor.
    query.dirty_chunks(&world, now, now);
    let report = query.dirty_chunks_with_stats(&world, last, now);
    assert_covers(&world, &report.chunks, &BTreeSet::from([entity]));
}

#[test]
fn sparse_historical_read_preserves_unread_forward_changes() {
    historical_read_preserves_unread_forward_changes::<SparseValue>();
}

#[test]
fn table_historical_read_preserves_unread_forward_changes() {
    historical_read_preserves_unread_forward_changes::<TableValue>();
}

fn structural_matrix(workers: usize, rollover: bool) -> Vec<(Entity, u32, Option<u32>)> {
    let mut world = World::new();
    if rollover {
        warm_to(&mut world, u32::MAX - 2);
    }
    let mut entities: Vec<_> = (0..97).map(|i| spawn(&mut world, i)).collect();
    let table = world.query::<&TableValue>();
    let sparse = world.query::<&SparseValue>();
    let with_sparse = world.query_filtered::<&TableValue, With<SparseValue>>();
    let without_sparse = world.query_filtered::<&TableValue, Without<SparseValue>>();
    let baseline = world.change_tick();
    table.dirty_chunks(&world, baseline, baseline);
    sparse.dirty_chunks(&world, baseline, baseline);
    with_sparse.dirty_chunks(&world, baseline, baseline);
    without_sparse.dirty_chunks(&world, baseline, baseline);
    let mut windows = Vec::new();
    #[cfg(feature = "multi_thread")]
    let pool = prism_tasks::TaskPool::with_threads(workers);
    #[cfg(not(feature = "multi_thread"))]
    assert_eq!(workers, 0);

    for phase in 0..4 {
        let last = world.change_tick();
        let now = advance(&mut world);
        if rollover {
            assert_eq!(now.get(), (u32::MAX - 1).wrapping_add(phase as u32));
        }
        let before: Vec<_> = entities
            .iter()
            .filter_map(|&entity| {
                let table = world.get::<TableValue>(entity)?;
                Some((
                    entity,
                    table.ordinal,
                    table.value,
                    world.get_ticks::<TableValue>(entity).unwrap(),
                    world
                        .get::<SparseValue>(entity)
                        .map(|s| (s.0, world.get_ticks::<SparseValue>(entity).unwrap())),
                ))
            })
            .collect();
        let writer = world.query::<(&mut TableValue, Option<&mut SparseValue>)>();
        let write = |(mut table, sparse): (Mut<'_, TableValue>, Option<Mut<'_, SparseValue>>)| {
            if table.ordinal % 3 == phase % 3 {
                table.value += 10;
            }
            if table.ordinal % 4 == phase {
                if let Some(mut sparse) = sparse {
                    sparse.0 += 20;
                }
            }
        };
        #[cfg(feature = "multi_thread")]
        if workers != 0 {
            writer.par_for_each_mut(&mut world, &pool, 1, write);
        } else {
            writer.iter_mut(&mut world).for_each(write);
        }
        #[cfg(not(feature = "multi_thread"))]
        writer.iter_mut(&mut world).for_each(write);

        // Query/reference agreement alone would not catch a writer that forgot
        // BOTH its row stamp and index event. Check stamps against the intended
        // write selection before structural operations add their own changes.
        for (entity, ordinal, value, table_ticks, sparse_before) in before {
            let table = world.get::<TableValue>(entity).unwrap();
            let written = ordinal % 3 == phase % 3;
            assert_eq!(
                table.value,
                value + if written { 10 } else { 0 },
                "table row value: {entity:?}"
            );
            let ticks = world.get_ticks::<TableValue>(entity).unwrap();
            assert_eq!(ticks.added, table_ticks.added);
            assert_eq!(
                ticks.changed,
                if written { now } else { table_ticks.changed },
                "table row write barrier: {entity:?}"
            );
            if let Some((value, before_ticks)) = sparse_before {
                let written = ordinal % 4 == phase;
                assert_eq!(
                    world.get::<SparseValue>(entity).unwrap().0,
                    value + if written { 20 } else { 0 }
                );
                let ticks = world.get_ticks::<SparseValue>(entity).unwrap();
                assert_eq!(ticks.added, before_ticks.added);
                assert_eq!(
                    ticks.changed,
                    if written { now } else { before_ticks.changed },
                    "sparse row write barrier: {entity:?}"
                );
            }
        }

        // Mutations happen AFTER the joined worker barrier. These operations
        // invalidate cached archetype/row locations, swap-remove live rows,
        // churn sparse membership and reuse an entity index with a new generation.
        let start = phase * 10;
        for &entity in &entities[start..start + 5] {
            assert!(world.insert(entity, Marker));
        }
        if phase != 0 {
            for &entity in &entities[start - 10..start - 5] {
                assert!(world.remove::<Marker>(entity));
            }
        }
        let removed = entities[start + 5];
        assert!(world.remove::<SparseValue>(removed));
        let reinserted = entities[start + 6];
        assert!(world.remove::<SparseValue>(reinserted));
        assert!(world.insert(reinserted, SparseValue(700 + phase as u32)));
        let replaced = entities[start + 7];
        assert!(world.remove::<TableValue>(replaced));
        assert!(world.insert(
            replaced,
            TableValue {
                ordinal: start + 7,
                value: 900 + phase as u32,
                _padding: [0; 256],
            }
        ));
        let dead = entities[start + 8];
        assert!(world.despawn(dead));
        let replacement = spawn(&mut world, 200 + phase);
        assert_eq!(replacement.index(), dead.index());
        assert_ne!(replacement.generation(), dead.generation());
        entities.push(replacement);

        // Also exercise the chunk-slice write barrier on the migrated table.
        let before: Vec<_> = entities
            .iter()
            .filter_map(|&entity| {
                let table = world.get::<TableValue>(entity)?;
                Some((
                    entity,
                    table.value,
                    world.get_ticks::<TableValue>(entity).unwrap(),
                ))
            })
            .collect();
        let slices = world.query_filtered::<&mut TableValue, With<Marker>>();
        #[cfg(feature = "multi_thread")]
        if workers != 0 {
            slices.par_chunks_mut(&mut world, &pool, 1, |slice| {
                for table in slice {
                    table.value += 1;
                }
            });
        } else {
            for mut table in slices.iter_mut(&mut world) {
                table.value += 1;
            }
        }
        #[cfg(not(feature = "multi_thread"))]
        for mut table in slices.iter_mut(&mut world) {
            table.value += 1;
        }
        for (entity, value, before_ticks) in before {
            let marked = world.get::<Marker>(entity).is_some();
            assert_eq!(
                world.get::<TableValue>(entity).unwrap().value,
                value + u32::from(marked)
            );
            let ticks = world.get_ticks::<TableValue>(entity).unwrap();
            assert_eq!(ticks.added, before_ticks.added);
            assert_eq!(
                ticks.changed,
                if marked { now } else { before_ticks.changed }
            );
        }

        windows.push((last, now));
        for &(last, now) in windows.iter().rev() {
            assert_exact::<TableValue>(&mut world, &entities, last, now);
            assert_exact::<SparseValue>(&mut world, &entities, last, now);
            let table_rows = reference::<TableValue>(&world, &entities, last, now, false);
            let sparse_rows = reference::<SparseValue>(&world, &entities, last, now, false);
            for _ in 0..2 {
                assert_covers(&world, &table.dirty_chunks(&world, last, now), &table_rows);
                assert_covers(
                    &world,
                    &sparse.dirty_chunks(&world, last, now),
                    &sparse_rows,
                );
                let with_rows = table_rows
                    .union(&sparse_rows)
                    .copied()
                    .filter(|&e| world.get::<SparseValue>(e).is_some())
                    .collect();
                assert_covers(
                    &world,
                    &with_sparse.dirty_chunks(&world, last, now),
                    &with_rows,
                );
                let mut without_rows: BTreeSet<_> = table_rows
                    .iter()
                    .copied()
                    .filter(|&e| world.get::<SparseValue>(e).is_none())
                    .collect();
                if now == world.change_tick() {
                    without_rows.insert(removed);
                }
                assert_covers(
                    &world,
                    &without_sparse.dirty_chunks(&world, last, now),
                    &without_rows,
                );
            }
        }
    }
    entities
        .iter()
        .copied()
        .filter_map(|entity| {
            world.get::<TableValue>(entity).map(|table| {
                (
                    entity,
                    table.value,
                    world.get::<SparseValue>(entity).map(|sparse| sparse.0),
                )
            })
        })
        .collect()
}

#[test]
fn serial_barriers_and_structural_changes_match_entity_reference() {
    assert_eq!(structural_matrix(0, false).len(), 97);
}

#[cfg(feature = "multi_thread")]
#[test]
fn multiworker_barriers_then_structural_changes_match_entity_reference() {
    let serial = structural_matrix(0, false);
    for workers in [1, 4] {
        assert_eq!(
            structural_matrix(workers, false),
            serial,
            "{workers}-worker values differ"
        );
    }
}

#[test]
fn rollover_barriers_and_structural_changes_match_entity_reference() {
    let serial = structural_matrix(0, true);
    assert_eq!(serial.len(), 97);
    #[cfg(feature = "multi_thread")]
    for workers in [1, 4] {
        assert_eq!(
            structural_matrix(workers, true),
            serial,
            "{workers} workers"
        );
    }
}

fn assert_cached_exact<T: Component>(
    world: &World,
    entities: &[Entity],
    last: Tick,
    now: Tick,
    changed: &prism_ecs::query::QueryState<Entity, Changed<T>>,
    added: &prism_ecs::query::QueryState<Entity, Added<T>>,
) {
    for (rows, is_added) in [
        (
            changed
                .iter_with_ticks(world, last, now)
                .collect::<Vec<_>>(),
            false,
        ),
        (added.iter_with_ticks(world, last, now).collect(), true),
    ] {
        let unique: BTreeSet<_> = rows.iter().copied().collect();
        assert_eq!(rows.len(), unique.len(), "duplicate cached row");
        assert_eq!(unique, reference::<T>(world, entities, last, now, is_added));
    }
}

fn cached_queries_across_restore(rollover: bool, consume_abandoned: bool) {
    let mut world = World::new();
    world.register_snapshot_component::<TableValue>();
    world.register_snapshot_component::<SparseValue>();
    world.register_snapshot_component::<Marker>();
    if rollover {
        warm_to(&mut world, u32::MAX - 1);
    }
    let mut entities: Vec<_> = (0..65).map(|i| spawn(&mut world, i)).collect();
    for (i, &entity) in entities.iter().enumerate() {
        if i % 5 == 0 {
            assert!(world.remove::<SparseValue>(entity));
        }
        if i % 7 == 0 {
            assert!(world.insert(entity, Marker));
        }
    }

    // ALL consumers predate the snapshot/restore. Recreating Changed/Added
    // queries inside the checker would miss stale archetype-cache bugs.
    let table = world.query::<&TableValue>();
    let sparse = world.query::<&SparseValue>();
    let table_changed = world.query_filtered::<Entity, Changed<TableValue>>();
    let table_added = world.query_filtered::<Entity, Added<TableValue>>();
    let sparse_changed = world.query_filtered::<Entity, Changed<SparseValue>>();
    let sparse_added = world.query_filtered::<Entity, Added<SparseValue>>();
    let marked = world.query_filtered::<Entity, With<Marker>>();
    let with_sparse = world.query_filtered::<Entity, With<SparseValue>>();
    let without_sparse = world.query_filtered::<Entity, Without<SparseValue>>();

    let check_cached = |world: &World, entities: &[Entity], last: Tick, now: Tick| {
        assert_cached_exact::<TableValue>(world, entities, last, now, &table_changed, &table_added);
        assert_cached_exact::<SparseValue>(
            world,
            entities,
            last,
            now,
            &sparse_changed,
            &sparse_added,
        );
        assert_covers(
            world,
            &table.dirty_chunks(world, last, now),
            &reference::<TableValue>(world, entities, last, now, false),
        );
        assert_covers(
            world,
            &sparse.dirty_chunks(world, last, now),
            &reference::<SparseValue>(world, entities, last, now, false),
        );
        for (rows, expected) in [
            (
                marked.iter(world).collect::<Vec<_>>(),
                entities
                    .iter()
                    .copied()
                    .filter(|&e| world.get::<Marker>(e).is_some())
                    .collect::<BTreeSet<_>>(),
            ),
            (
                with_sparse.iter(world).collect(),
                entities
                    .iter()
                    .copied()
                    .filter(|&e| world.get::<SparseValue>(e).is_some())
                    .collect(),
            ),
            (
                without_sparse.iter(world).collect(),
                entities
                    .iter()
                    .copied()
                    .filter(|&e| {
                        world.entities().location(e).is_some()
                            && world.get::<SparseValue>(e).is_none()
                    })
                    .collect(),
            ),
        ] {
            let unique: BTreeSet<_> = rows.iter().copied().collect();
            assert_eq!(rows.len(), unique.len());
            assert_eq!(
                unique, expected,
                "cached current membership after time travel"
            );
        }
    };

    let born = world.change_tick();
    check_cached(
        &world,
        &entities,
        Tick::new(born.get().wrapping_sub(1)),
        born,
    );
    let restored_tick = advance(&mut world);
    if rollover {
        assert_eq!(restored_tick, Tick::new(u32::MAX));
    }
    check_cached(&world, &entities, born, restored_tick);
    let saved: Vec<_> = entities
        .iter()
        .map(|&e| {
            (
                e,
                world.get::<TableValue>(e).unwrap().value,
                world.get_ticks::<TableValue>(e).unwrap(),
                world
                    .get::<SparseValue>(e)
                    .map(|s| (s.0, world.get_ticks::<SparseValue>(e).unwrap())),
                world.get::<Marker>(e).is_some(),
            )
        })
        .collect();
    let snapshot = world.snapshot();

    let abandoned_start = advance(&mut world);
    if rollover {
        assert_eq!(abandoned_start, Tick::new(0));
    }
    world.get_mut::<TableValue>(entities[1]).unwrap().value += 100;
    world.get_mut::<SparseValue>(entities[1]).unwrap().0 += 200;
    assert!(world.remove::<SparseValue>(entities[2]));
    assert!(world.remove::<TableValue>(entities[3]));
    assert!(world.insert(entities[4], Marker));
    assert!(world.remove::<Marker>(entities[7]));
    assert!(world.despawn(entities[8]));
    let abandoned = spawn(&mut world, 999);
    assert_eq!(abandoned.index(), entities[8].index());
    assert_ne!(abandoned.generation(), entities[8].generation());
    entities.push(abandoned); // Keep both generations in the independent universe.
    let abandoned_end = advance(&mut world);
    world.get_mut::<TableValue>(entities[9]).unwrap().value += 300;
    world.get_mut::<SparseValue>(entities[9]).unwrap().0 += 400;
    if consume_abandoned {
        check_cached(&world, &entities, restored_tick, abandoned_end);
        check_cached(&world, &entities, abandoned_end, abandoned_end); // Cursor acknowledged here.
    }

    // Repeat restore after both a consumed and an unread post-restore write.
    // The very same consumers must invalidate caches/cursors each time.
    for pass in 0..2 {
        world.restore(&snapshot);
        assert_eq!(world.change_tick(), restored_tick);
        assert!(world.get::<TableValue>(abandoned).is_none());
        assert!(world.get::<SparseValue>(abandoned).is_none());
        assert!(world.entities().location(abandoned).is_none());
        for &(entity, value, ticks, sparse_before, marker) in &saved {
            assert_eq!(world.get::<TableValue>(entity).unwrap().value, value);
            assert_eq!(world.get_ticks::<TableValue>(entity), Some(ticks));
            assert_eq!(
                world
                    .get::<SparseValue>(entity)
                    .map(|s| (s.0, world.get_ticks::<SparseValue>(entity).unwrap())),
                sparse_before
            );
            assert_eq!(world.get::<Marker>(entity).is_some(), marker);
        }
        // No restored row was written at the snapshot tick: an exact empty
        // result here cannot be satisfied by replaying the abandoned events.
        assert!(reference::<TableValue>(&world, &entities, born, restored_tick, false).is_empty());
        assert!(reference::<SparseValue>(&world, &entities, born, restored_tick, false).is_empty());
        check_cached(&world, &entities, born, restored_tick);
        check_cached(&world, &entities, restored_tick, restored_tick);

        // A same-tick write must survive an empty-window acknowledgement and
        // later wider/history reads, without restoring the abandoned values.
        let entity = entities[11 + pass];
        let table_ticks = world.get_ticks::<TableValue>(entity).unwrap();
        let sparse_ticks = world.get_ticks::<SparseValue>(entity).unwrap();
        world.get_mut::<TableValue>(entity).unwrap().value += 7;
        world.get_mut::<SparseValue>(entity).unwrap().0 += 9;
        assert_eq!(
            world.get::<TableValue>(entity).unwrap().value,
            (11 + pass) as u32 + 7
        );
        assert_eq!(
            world.get::<SparseValue>(entity).unwrap().0,
            (11 + pass) as u32 + 9
        );
        assert_eq!(
            world.get_ticks::<TableValue>(entity).unwrap().added,
            table_ticks.added
        );
        assert_eq!(
            world.get_ticks::<SparseValue>(entity).unwrap().added,
            sparse_ticks.added
        );
        assert_eq!(
            world.get_ticks::<TableValue>(entity).unwrap().changed,
            restored_tick
        );
        assert_eq!(
            world.get_ticks::<SparseValue>(entity).unwrap().changed,
            restored_tick
        );
        assert_eq!(
            reference::<TableValue>(&world, &entities, born, restored_tick, false),
            BTreeSet::from([entity])
        );
        assert_eq!(
            reference::<SparseValue>(&world, &entities, born, restored_tick, false),
            BTreeSet::from([entity])
        );
        check_cached(&world, &entities, born, born); // Historical empty read cannot eat new writes.
        check_cached(&world, &entities, restored_tick, restored_tick); // Current empty ack can.
        check_cached(&world, &entities, born, restored_tick); // Recover from retained history.
        check_cached(&world, &entities, born, restored_tick); // Repeated window stays stable.

        let next = advance(&mut world);
        if rollover {
            assert_eq!(next, Tick::new(0));
        }
        let forward = entities[13];
        world.get_mut::<TableValue>(forward).unwrap().value += 13;
        world.get_mut::<SparseValue>(forward).unwrap().0 += 17;
        assert_eq!(world.get::<TableValue>(forward).unwrap().value, 26);
        assert_eq!(world.get::<SparseValue>(forward).unwrap().0, 30);
        assert_eq!(
            world.get_ticks::<TableValue>(forward).unwrap().changed,
            next
        );
        assert_eq!(
            world.get_ticks::<SparseValue>(forward).unwrap().changed,
            next
        );
        check_cached(&world, &entities, restored_tick, next);
        check_cached(&world, &entities, born, restored_tick); // Rewind after a forward read.
        check_cached(&world, &entities, born, next);
        // Deliberately unread when the next restore drops this timeline.
        world.get_mut::<SparseValue>(entities[14]).unwrap().0 += 31;
    }
}

#[test]
fn same_query_states_survive_restore_abandoned_timelines_and_same_tick_writes() {
    for consume_abandoned in [false, true] {
        cached_queries_across_restore(false, consume_abandoned);
    }
}

#[test]
fn same_query_states_survive_restore_across_actual_rollover() {
    for consume_abandoned in [false, true] {
        cached_queries_across_restore(true, consume_abandoned);
    }
}
