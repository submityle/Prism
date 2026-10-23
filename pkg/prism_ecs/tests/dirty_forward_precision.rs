//! A retained forward window must not quietly degrade into an all-chunk scan.
//! Historical/gap reports remain conservative; this test deliberately keeps a
//! complete journal and makes no exactness claim for those recovery paths.

use prism_ecs::archetype::ArchetypeId;
use prism_ecs::component::StorageType;
use prism_ecs::prelude::*;
use std::collections::BTreeSet;

#[derive(Component)]
struct TableValue {
    ordinal: usize,
    value: u32,
    _padding: [u64; 63],
}

struct SparseValue(u32);

impl Component for SparseValue {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[derive(Component)]
struct Unrelated(u32);

fn expected_chunks(world: &World, entities: &[Entity]) -> BTreeSet<(ArchetypeId, usize, usize)> {
    entities
        .iter()
        .map(|&entity| {
            let location = world.entities().location(entity).unwrap();
            let table = world
                .archetypes()
                .get(location.archetype_id)
                .unwrap()
                .table();
            let start = location.row as usize / table.rows_per_chunk() * table.rows_per_chunk();
            (
                location.archetype_id,
                start,
                (start + table.rows_per_chunk()).min(table.len()),
            )
        })
        .collect()
}

#[track_caller]
fn assert_precise(world: &World, report: DirtyChunkReport, changed: &[Entity]) {
    let actual: BTreeSet<_> = report
        .chunks
        .iter()
        .map(|chunk| (chunk.archetype, chunk.start, chunk.end))
        .collect();
    assert_eq!(actual.len(), report.chunks.len(), "duplicate chunk");
    assert_eq!(
        actual,
        expected_chunks(world, changed),
        "{:?}",
        report.stats
    );
    assert!(!report.stats.journal_gap, "{:?}", report.stats);
    assert_eq!(report.stats.sparse_index_gaps, 0);
    assert_eq!(report.stats.sparse_fallback_scans, 0);
    assert_eq!(report.stats.sparse_replay_sources, 0);
    assert_eq!(report.stats.archetype_index_entries_scanned, 0);
    assert_eq!(report.stats.summaries_tested, 0);
    assert_eq!(report.stats.reported_chunks, actual.len());
    assert_eq!(report.stats.merged_candidates, actual.len());
    assert_eq!(
        report.stats.reported_rows,
        actual
            .iter()
            .map(|(_, start, end)| end - start)
            .sum::<usize>()
    );
    // Output order is part of the deterministic coarse consumer contract.
    assert_eq!(
        report
            .chunks
            .iter()
            .map(|chunk| (chunk.archetype, chunk.start, chunk.end))
            .collect::<Vec<_>>(),
        actual.into_iter().collect::<Vec<_>>()
    );
}

fn run(workers: usize) -> Vec<(u32, u32)> {
    let mut world = World::new();
    let entities: Vec<_> = (0..4096)
        .map(|ordinal| {
            world.spawn((
                TableValue {
                    ordinal,
                    value: 0,
                    _padding: [0; 63],
                },
                SparseValue(0),
                Unrelated(0),
            ))
        })
        .collect();
    let table = world.query::<&TableValue>();
    let sparse = world.query::<&SparseValue>();
    let both = world.query::<(&TableValue, &SparseValue)>();
    let activation = world.change_tick();
    table.dirty_chunks(&world, activation, activation);
    sparse.dirty_chunks(&world, activation, activation);
    both.dirty_chunks(&world, activation, activation);
    #[cfg(feature = "multi_thread")]
    let pool = prism_tasks::TaskPool::with_threads(workers.max(1));
    #[cfg(not(feature = "multi_thread"))]
    assert_eq!(workers, 0);

    for phase in 0..5 {
        let last = world.change_tick();
        world.set_last_change_tick(last);
        let now = world.increment_change_tick();
        // Sparse/scattered, clustered, fully idle, unrelated-only and a final
        // scattered window. The final window tests recovery after idle acks.
        let table_ordinals: Vec<_> = match phase {
            0 | 4 => (0..16).map(|i| i * 193 + phase).collect(),
            1 => (95..112).collect(),
            _ => Vec::new(),
        };
        let sparse_ordinals: Vec<_> = match phase {
            0 | 4 => (0..16).map(|i| i * 197 + 50 + phase).collect(),
            1 => (101..118).collect(),
            _ => Vec::new(),
        };
        let writer = world.query::<(&mut TableValue, &mut SparseValue)>();
        let write = |(mut table, mut sparse): (Mut<'_, TableValue>, Mut<'_, SparseValue>)| {
            if table_ordinals.contains(&table.ordinal) {
                for _ in 0..17 {
                    table.value += 1;
                }
            }
            if sparse_ordinals.contains(&table.ordinal) {
                for _ in 0..19 {
                    sparse.0 += 1;
                }
            }
        };
        #[cfg(feature = "multi_thread")]
        if workers != 0 {
            writer.par_for_each_mut(&mut world, &pool, 32, write);
        } else {
            writer.iter_mut(&mut world).for_each(write);
        }
        #[cfg(not(feature = "multi_thread"))]
        writer.iter_mut(&mut world).for_each(write);

        if phase != 2 {
            for &entity in &entities[3500..3600] {
                world.get_mut::<Unrelated>(entity).unwrap().0 += 1;
            }
        }
        let table_changed: Vec<_> = table_ordinals.iter().map(|&i| entities[i]).collect();
        let sparse_changed: Vec<_> = sparse_ordinals.iter().map(|&i| entities[i]).collect();
        let mut union = table_changed.clone();
        union.extend_from_slice(&sparse_changed);
        assert_precise(
            &world,
            table.dirty_chunks_with_stats(&world, last, now),
            &table_changed,
        );
        assert_precise(
            &world,
            sparse.dirty_chunks_with_stats(&world, last, now),
            &sparse_changed,
        );
        assert_precise(
            &world,
            both.dirty_chunks_with_stats(&world, last, now),
            &union,
        );
        // A new current empty window acknowledges the cursor. It must return
        // nothing despite non-empty reports earlier in the same world tick.
        assert_precise(&world, table.dirty_chunks_with_stats(&world, now, now), &[]);
        assert_precise(
            &world,
            sparse.dirty_chunks_with_stats(&world, now, now),
            &[],
        );
        assert_precise(&world, both.dirty_chunks_with_stats(&world, now, now), &[]);
    }
    entities
        .iter()
        .map(|&entity| {
            (
                world.get::<TableValue>(entity).unwrap().value,
                world.get::<SparseValue>(entity).unwrap().0,
            )
        })
        .collect()
}

#[test]
fn retained_forward_reports_have_exact_touched_chunk_union() {
    let result = run(0);
    assert!(result
        .iter()
        .any(|&(table, sparse)| table != 0 && sparse != 0));
}

#[cfg(feature = "multi_thread")]
#[test]
fn parallel_retained_forward_reports_do_not_expand_to_clean_chunks() {
    let serial = run(0);
    for workers in [1, 4] {
        assert_eq!(run(workers), serial, "{workers} workers");
    }
}
