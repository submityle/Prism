//! Behavioural tests for the dirty-chunk accessor
//! ([`QueryState::dirty_chunks`](crate::query::QueryState::dirty_chunks)), the
//! coarse, chunk-granular half of double-layer change detection (design §7/§10).
//!
//! The accessor must be a correct *superset* of the per-row
//! [`Changed<T>`](crate::query::Changed) scan: every row a `Changed<T>` filter
//! would yield lies inside some reported window, and no window is reported for a
//! chunk whose rows are all older than `last_run`.

use alloc::vec::Vec;

use crate::change::Tick;
use crate::component::Component;
use crate::entity::Entity;
use crate::query::{Changed, DirtyChunk, Or, With, Without};
use crate::storage::rows_per_chunk;
use crate::world::World;

/// A component large enough that only a handful of rows fit per logical chunk,
/// so a modest spawn count spans several chunks (keeps the test fast + legible).
/// 4 KiB per row => `rows_per_chunk == 16 KiB / 4 KiB == 4`.
#[derive(Clone, Copy)]
struct Big {
    tag: u64,
    _pad: [u64; 511],
}
impl Big {
    fn new(tag: u64) -> Self {
        Self {
            tag,
            _pad: [0; 511],
        }
    }
}
impl Component for Big {}

struct SparseDirty(u32);
impl Component for SparseDirty {
    const STORAGE: crate::component::StorageType = crate::component::StorageType::SparseSet;
}

#[derive(Clone, Copy)]
struct Tag;
impl Component for Tag {}

/// Advance the world as if a system had just finished a run: the next window
/// starts where this one ended, and the frame tick advances by one.
fn advance_frame(w: &mut World) {
    w.set_last_change_tick(w.change_tick());
    w.increment_change_tick();
}

/// Whether `row` of `archetype`-equivalent window set is covered by any reported
/// dirty chunk (archetype equality is checked by the caller when needed).
fn row_is_covered(dirty: &[DirtyChunk], row: usize) -> bool {
    dirty.iter().any(|d| row >= d.start && row < d.end)
}

#[test]
fn rpc_is_four_for_a_four_kib_row() {
    // Sanity-check the fixture math the rest of the tests rely on.
    assert_eq!(size_of::<Big>(), 4096);
    assert_eq!(rows_per_chunk(size_of::<Big>()), 4);
}

#[test]
fn only_touched_chunks_are_reported() {
    let mut w = World::new();
    // 10 rows => chunks [0,4), [4,8), [8,10) at rpc = 4.
    let entities: Vec<Entity> = (0..10).map(|i| w.spawn(Big::new(i))).collect();

    // Move past the spawn frame so the spawn-time writes fall out of the window.
    advance_frame(&mut w);

    // Touch one row in chunk 0 and one row in chunk 2; leave chunk 1 untouched.
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    w.get_mut::<Big>(entities[1]).unwrap().tag += 100; // chunk 0
    w.get_mut::<Big>(entities[9]).unwrap().tag += 100; // chunk 2

    let state = w.query::<&mut Big>();
    let dirty = state.dirty_chunks(&w, last_run, this_run);

    // Exactly the two touched windows, nothing from chunk 1.
    assert_eq!(dirty.len(), 2, "only the two touched chunks are reported");
    assert!(
        dirty.iter().any(|d| d.start == 0 && d.end == 4),
        "chunk 0 window [0,4) reported: {dirty:?}"
    );
    assert!(
        dirty.iter().any(|d| d.start == 8 && d.end == 10),
        "last partial chunk window [8,10) reported: {dirty:?}"
    );
    assert!(
        !dirty.iter().any(|d| d.start == 4),
        "untouched middle chunk [4,8) is NOT reported: {dirty:?}"
    );
}

#[test]
fn dirty_chunks_superset_of_per_row_changed() {
    let mut w = World::new();
    let entities: Vec<Entity> = (0..10).map(|i| w.spawn(Big::new(i))).collect();
    advance_frame(&mut w);

    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    // Touch a scattered subset spanning two chunks.
    for &idx in &[0usize, 2, 9] {
        w.get_mut::<Big>(entities[idx]).unwrap().tag += 1;
    }

    // Rows yielded by the fine, per-row Changed filter.
    let changed_state = w.query_filtered::<Entity, Changed<Big>>();
    let changed: Vec<Entity> = changed_state.iter(&w).collect();
    assert_eq!(changed.len(), 3, "three rows were written this window");

    let state = w.query::<&Big>();
    let dirty = state.dirty_chunks(&w, last_run, this_run);

    // Superset: every per-row Changed entity's row lies inside a reported window
    // of its own archetype.
    for e in &changed {
        let loc = w.entities().location(*e).unwrap();
        let covered = dirty
            .iter()
            .any(|d| d.archetype == loc.archetype_id && row_is_covered(&[*d], loc.row as usize));
        assert!(
            covered,
            "changed entity {e:?} at row {} not covered",
            loc.row
        );
    }
}

#[test]
fn spawn_frame_marks_every_chunk_dirty() {
    let mut w = World::new();
    for i in 0..10 {
        w.spawn(Big::new(i));
    }
    // Observe from the very first frame: last_run = ZERO, this_run = spawn tick.
    let last_run = Tick::ZERO;
    let this_run = w.change_tick();

    let state = w.query::<&Big>();
    let dirty = state.dirty_chunks(&w, last_run, this_run);

    // All three chunks (two full + one partial) are freshly written.
    assert_eq!(
        dirty.len(),
        3,
        "every chunk is dirty on the spawn frame: {dirty:?}"
    );
    let total: usize = dirty.iter().map(DirtyChunk::len).sum();
    assert_eq!(total, 10, "windows cover all 10 rows exactly");
    assert!(dirty.iter().all(|d| !d.is_empty()));
}

#[test]
fn no_writes_reports_no_dirty_chunks() {
    let mut w = World::new();
    for i in 0..10 {
        w.spawn(Big::new(i));
    }
    // Advance so the spawn writes age out of the window, then observe a frame
    // with no writes at all.
    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();

    let state = w.query::<&Big>();
    let dirty = state.dirty_chunks(&w, last_run, this_run);
    assert!(
        dirty.is_empty(),
        "a quiescent frame reports no dirty chunks: {dirty:?}"
    );
}

#[test]
fn empty_window_uses_only_the_root_summary() {
    let mut w = World::new();
    for i in 0..10 {
        w.spawn(Big::new(i));
    }
    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();

    let state = w.query::<&Big>();
    let dirty = state.dirty_chunks(&w, last_run, this_run);
    assert!(dirty.is_empty());
}

#[test]
fn forward_dirty_window_uses_column_journal_without_scanning_all_archetypes() {
    let mut w = World::new();
    let entities: Vec<Entity> = (0..10).map(|i| w.spawn(Big::new(i))).collect();
    advance_frame(&mut w);

    // The first request establishes the journal baseline. A new query has no
    // cursor yet, so it must conservatively use the indexed fallback once.
    let state = w.query::<&Big>();
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    let baseline = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert!(baseline.stats.journal_gap);

    // The next normal forward request is discovered from the retained table
    // journal. It should not visit any column interval summaries.
    let last_run = w.change_tick();
    w.increment_change_tick();
    let this_run = w.change_tick();
    for _ in 0..3 {
        w.get_mut::<Big>(entities[9]).unwrap().tag += 1;
    }
    let report = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert!(!report.stats.journal_gap);
    assert_eq!(report.stats.journal_events, 3);
    assert_eq!(report.stats.candidate_archetype_checks, 1);
    assert_eq!(report.stats.summaries_tested, 0);
    assert_eq!(report.stats.column_candidates, 0);
    assert_eq!(report.stats.matched_archetypes, 1);
    assert_eq!(report.chunks.len(), 1);
}

#[test]
fn repeated_writes_deduplicate_dirty_chunk_candidates_before_sorting() {
    let mut w = World::new();
    let entity = w.spawn(Big::new(10));
    advance_frame(&mut w);

    let state = w.query::<&Big>();
    let last_run = w.change_tick();
    let this_run = w.change_tick();
    let baseline = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert!(baseline.stats.journal_gap);

    let last_run = w.change_tick();
    w.increment_change_tick();
    let this_run = w.change_tick();
    const WRITES: usize = 4096;
    for _ in 0..WRITES {
        w.get_mut::<Big>(entity).unwrap().tag += 1;
    }

    let report = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert!(!report.stats.journal_gap);
    assert_eq!(report.stats.journal_events, WRITES);
    assert_eq!(report.stats.candidate_chunk_marks, WRITES);
    assert_eq!(report.stats.merged_candidates, 1);
    assert_eq!(report.chunks.len(), 1);
}

#[cfg(feature = "multi_thread")]
#[test]
fn parallel_query_writes_reach_attached_column_journal_without_losing_chunks() {
    use prism_tasks::TaskPool;

    const ENTITIES: usize = 512;
    let mut w = World::new();
    for i in 0..ENTITIES {
        w.spawn(Big::new(i as u64));
    }
    advance_frame(&mut w);

    // Establish the query's journal baseline after the structural spawn batch.
    let state = w.query::<&mut Big>();
    let baseline = state.dirty_chunks_with_stats(&w, w.last_change_tick(), w.change_tick());
    assert!(baseline.stats.journal_gap);

    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    let pool = TaskPool::with_threads(4);
    state.par_for_each_mut(&mut w, &pool, 3, |mut value| {
        value.tag += 1;
    });

    // `Table` attached each physical column's ChunkChange cells to the shared
    // archetype journal. This verifies the real parallel write path, not only
    // the journal and chunk-summary primitives in isolation.
    let report = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert!(!report.stats.journal_gap);
    assert_eq!(report.stats.journal_events, ENTITIES);
    assert_eq!(report.chunks.len(), ENTITIES.div_ceil(4));
    assert_eq!(
        report.chunks.iter().map(|chunk| chunk.len()).sum::<usize>(),
        ENTITIES
    );
}

#[test]
fn high_cardinality_dirty_query_work_tracks_distinct_journal_candidates() {
    const CHUNKS: usize = 1024;
    let mut w = World::new();
    let entities: Vec<_> = (0..CHUNKS * 4)
        .map(|i| w.spawn(Big::new(i as u64)))
        .collect();
    advance_frame(&mut w);

    let state = w.query::<&Big>();
    let baseline = state.dirty_chunks_with_stats(&w, w.last_change_tick(), w.change_tick());
    assert!(baseline.stats.journal_gap);

    let last_run = w.change_tick();
    w.increment_change_tick();
    let this_run = w.change_tick();
    // One write in each physical chunk forces DirtyChunkCandidates through
    // its promoted high-cardinality set path rather than its <=8 inline path.
    for chunk in 0..CHUNKS {
        w.get_mut::<Big>(entities[chunk * 4]).unwrap().tag += 1;
    }

    let report = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert!(!report.stats.journal_gap);
    assert_eq!(report.stats.journal_records_scanned, CHUNKS);
    assert_eq!(report.stats.journal_events, CHUNKS);
    assert_eq!(report.stats.candidate_chunk_marks, CHUNKS);
    assert_eq!(report.stats.merged_candidates, CHUNKS);
    assert_eq!(report.stats.reported_chunks, CHUNKS);
    assert_eq!(report.chunks.len(), CHUNKS);
    assert_eq!(report.stats.reported_rows, CHUNKS * 4);
}

#[test]
fn dirty_chunks_follow_an_archetype_migration() {
    #[derive(Clone, Copy)]
    struct Tag;
    impl Component for Tag {}

    let mut w = World::new();
    let entity = w.spawn(Big::new(1));
    advance_frame(&mut w);
    let state = w.query::<&Big>();

    let last_run = w.change_tick();
    w.increment_change_tick();
    assert!(w.insert(entity, Tag));
    w.get_mut::<Big>(entity).unwrap().tag = 99;
    let this_run = w.change_tick();

    let location = w.entities().location(entity).unwrap();
    let dirty = state.dirty_chunks(&w, last_run, this_run);
    assert_eq!(dirty.len(), 1, "migrated row should have one dirty window");
    assert_eq!(dirty[0].archetype, location.archetype_id);
    assert_eq!((dirty[0].start, dirty[0].end), (0, 1));
}

#[test]
fn write_set_component_marks_chunk_dirty_for_read_query() {
    // A query reading A over an archetype {A, B}: writing B must still flag the
    // chunk, because dirty_chunks unions the query's read AND write sets, and B
    // shares the chunk window with A.
    #[derive(Clone, Copy)]
    struct Marker(u32);
    impl Component for Marker {}

    let mut w = World::new();
    let entities: Vec<Entity> = (0..6).map(|i| w.spawn((Big::new(i), Marker(0)))).collect();
    advance_frame(&mut w);

    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    // Write only Marker on one row of chunk 0.
    w.get_mut::<Marker>(entities[0]).unwrap().0 = 7;

    // Query reads Big only; its access set includes Big (read). Marker's write
    // bumps Marker's chunk version, not Big's, so a Big-only read query should
    // NOT see it — confirming per-column granularity.
    let big_only = w.query::<&Big>();
    let dirty_big = big_only.dirty_chunks(&w, last_run, this_run);
    assert!(
        dirty_big.is_empty(),
        "writing Marker does not dirty a Big-only query: {dirty_big:?}"
    );

    // A query that reads Marker does see it.
    let marker_only = w.query::<&Marker>();
    let dirty_marker = marker_only.dirty_chunks(&w, last_run, this_run);
    assert_eq!(
        dirty_marker.len(),
        1,
        "Marker write flags its chunk: {dirty_marker:?}"
    );
    // {Big, Marker} share one rpc from the summed row bytes
    // (4096 + 4 = 4100 => 16 KiB / 4100 = 3 rows per chunk), so chunk 0 is [0,3).
    assert_eq!((dirty_marker[0].start, dirty_marker[0].end), (0, 3));
}

#[test]
fn multiple_access_columns_merge_the_same_chunk_once() {
    #[derive(Clone, Copy)]
    struct Marker(u32);
    impl Component for Marker {}

    let mut w = World::new();
    let entities: Vec<Entity> = (0..6).map(|i| w.spawn((Big::new(i), Marker(0)))).collect();
    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    w.get_mut::<Big>(entities[0]).unwrap().tag += 1;
    w.get_mut::<Marker>(entities[1]).unwrap().0 += 1;

    let state = w.query::<(&Big, &Marker)>();
    let report = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(report.chunks.len(), 1, "both columns share one chunk");
    assert_eq!(report.stats.table_columns, 2);
    assert_eq!(report.stats.column_candidates, 2);
    assert_eq!(report.stats.merged_candidates, 1);
    assert_eq!(report.stats.reported_chunks, 1);
}

#[test]
fn sorted_chunk_merge_scales_with_entries_not_sources_per_output() {
    use super::dirty::merge_sorted_chunks;

    const SOURCES: usize = 48;
    const CHUNKS: usize = 512;

    let overlapping: Vec<Vec<usize>> = (0..SOURCES).map(|_| (0..CHUNKS).collect()).collect();
    let mut merged = Vec::new();
    let operations = merge_sorted_chunks(&overlapping, |chunk| merged.push(chunk));
    assert_eq!(merged, (0..CHUNKS).collect::<Vec<_>>());
    assert_eq!(operations, 2 * SOURCES * CHUNKS);

    let disjoint: Vec<Vec<usize>> = (0..SOURCES)
        .map(|source| (source * CHUNKS..(source + 1) * CHUNKS).collect())
        .collect();
    merged.clear();
    let operations = merge_sorted_chunks(&disjoint, |chunk| merged.push(chunk));
    assert_eq!(merged, (0..SOURCES * CHUNKS).collect::<Vec<_>>());
    assert_eq!(operations, 2 * SOURCES * CHUNKS);
}

#[test]
fn filter_chunk_rejects_a_changed_column_that_is_not_in_the_filter_window() {
    #[derive(Clone, Copy)]
    struct NeverChanged;
    impl Component for NeverChanged {}

    let mut w = World::new();
    let entities: Vec<Entity> = (0..10)
        .map(|i| w.spawn((Big::new(i), NeverChanged)))
        .collect();
    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    w.get_mut::<Big>(entities[1]).unwrap().tag += 1;

    // The access column is dirty, but the independent Changed filter is clean
    // for a different component present on every row. It must conservatively
    // reject the merged chunk instead of manufacturing a dirty result.
    let state = w.query_filtered::<&Big, (With<Big>, Changed<NeverChanged>)>();
    let report = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert!(report.chunks.is_empty());
    assert_eq!(report.stats.merged_candidates, 1);
    assert_eq!(report.stats.filter_rejected, 1);
}

#[test]
fn sparse_changes_fallback_to_changed_entities_without_scanning_table_rows() {
    let mut w = World::new();
    let entities: Vec<Entity> = (0..5).map(|i| w.spawn(SparseDirty(i))).collect();
    let state = w.query::<&SparseDirty>();
    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    w.get_mut::<SparseDirty>(entities[3]).unwrap().0 += 1;

    let report = state.dirty_chunks_with_stats(&w, last_run, this_run);

    assert_eq!(report.chunks.len(), 1, "one changed sparse row window");
    assert_eq!(report.stats.sparse_entities_checked, 0);
    assert_eq!(report.stats.sparse_fallback_scans, 0);
    assert_eq!(report.stats.sparse_index_events, 1);
    assert_eq!(report.stats.sparse_chunk_candidates, 1);
    assert_eq!(report.stats.table_columns, 0);
    assert!(report.stats.non_table_columns > 0);
    let location = w.entities().location(entities[3]).unwrap();
    assert_eq!(report.chunks[0].archetype, location.archetype_id);
    assert!(location.row as usize >= report.chunks[0].start);
    assert!((location.row as usize) < report.chunks[0].end);
}

#[test]
fn sparse_dirty_query_uses_live_fallback_only_for_pre_activation_windows() {
    let mut w = World::new();
    let entities: Vec<Entity> = (0..32).map(|i| w.spawn(SparseDirty(i))).collect();
    advance_frame(&mut w);

    // This write predates the query, so an older observer window cannot rely
    // on a cursor that did not exist yet. The bounded fallback preserves the
    // historical query semantics and is visible in the report.
    let this_run = w.change_tick();
    w.get_mut::<SparseDirty>(entities[17]).unwrap().0 += 1;
    let state = w.query::<&SparseDirty>();
    let report = state.dirty_chunks_with_stats(&w, Tick::ZERO, this_run);

    assert_eq!(report.chunks.len(), 1);
    assert_eq!(report.stats.sparse_index_events, 0);
    assert_eq!(report.stats.sparse_fallback_scans, 1);
    assert_eq!(report.stats.sparse_entities_checked, entities.len());
}

#[test]
fn sparse_dirty_query_falls_back_for_window_entirely_before_activation() {
    let mut w = World::new();
    let entity = w.spawn(SparseDirty(1));
    let historical_end = w.change_tick();
    w.increment_change_tick();
    let state = w.query::<&SparseDirty>();

    let report = state.dirty_chunks_with_stats(&w, Tick::ZERO, historical_end);

    assert_eq!(report.chunks.len(), 1);
    assert_eq!(report.stats.sparse_index_events, 0);
    assert_eq!(report.stats.sparse_fallback_scans, 1);
    assert_eq!(report.stats.sparse_entities_checked, 1);
    let location = w.entities().location(entity).unwrap();
    assert_eq!(report.chunks[0].archetype, location.archetype_id);
}

#[test]
fn sparse_dirty_query_rewinds_to_live_fallback_after_index_consumption() {
    let mut w = World::new();
    let entity = w.spawn(SparseDirty(1));
    let state = w.query::<&SparseDirty>();
    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    w.get_mut::<SparseDirty>(entity).unwrap().0 += 1;

    let first = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(first.stats.sparse_index_events, 1);
    assert_eq!(first.stats.sparse_fallback_scans, 0);

    let repeated = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(repeated.stats.sparse_index_events, 0);
    assert_eq!(repeated.stats.sparse_fallback_scans, 1);
    assert_eq!(repeated.stats.sparse_entities_checked, 1);
    assert_eq!(repeated.chunks, first.chunks);
}

#[test]
fn sparse_dirty_fallback_is_empty_for_a_clean_window() {
    let mut w = World::new();
    for i in 0..5 {
        w.spawn(SparseDirty(i));
    }
    advance_frame(&mut w);
    let report = w.query::<&SparseDirty>().dirty_chunks_with_stats(
        &w,
        w.last_change_tick(),
        w.change_tick(),
    );
    assert!(report.chunks.is_empty());
    assert_eq!(report.stats.sparse_entities_checked, 5);
    assert_eq!(report.stats.sparse_fallback_scans, 1);
    assert_eq!(report.stats.sparse_index_events, 0);
    assert_eq!(report.stats.sparse_chunk_candidates, 0);
}

#[test]
fn sparse_membership_filters_feed_dirty_candidates_across_archetypes() {
    let mut w = World::new();
    let plain = [w.spawn(Big::new(0)), w.spawn(Big::new(1))];
    let tagged = [w.spawn((Big::new(2), Tag)), w.spawn((Big::new(3), Tag))];
    let with = w.query_filtered::<Entity, With<SparseDirty>>();
    let without = w.query_filtered::<Entity, Without<SparseDirty>>();

    // Prime both match caches before sparse membership changes. The sparse
    // component does not create an archetype, so the new dirty source must be
    // responsible for discovering both current row windows.
    assert!(with.iter(&w).next().is_none());
    assert_eq!(without.iter(&w).count(), 4);
    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();

    assert!(w.insert(plain[0], SparseDirty(10)));
    assert!(w.insert(tagged[0], SparseDirty(20)));

    let with_report = with.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(with_report.stats.filter_sources, 1);
    assert_eq!(with_report.stats.access_columns, 0);
    assert_eq!(with_report.stats.sparse_entities_checked, 0);
    assert_eq!(with_report.stats.sparse_fallback_scans, 0);
    assert_eq!(with_report.stats.sparse_index_events, 2);
    assert_eq!(with_report.stats.sparse_chunk_candidates, 2);
    assert_eq!(with_report.chunks.len(), 2);
    assert!(with_report
        .chunks
        .iter()
        .any(|chunk| chunk.archetype == w.entities().location(plain[0]).unwrap().archetype_id));
    assert!(with_report
        .chunks
        .iter()
        .any(|chunk| chunk.archetype == w.entities().location(tagged[0]).unwrap().archetype_id));

    // Without<SparseDirty> shares the same candidate source. Its coarse result
    // is intentionally conservative: the changed chunks are still reported,
    // while exact row admission remains the normal per-row filter's job.
    let without_report = without.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(without_report.stats.filter_sources, 1);
    assert_eq!(without_report.stats.sparse_index_events, 2);
    assert_eq!(without_report.chunks.len(), 2);
}

#[test]
fn sparse_filter_source_merges_with_dirty_table_columns() {
    let mut w = World::new();
    let entity = w.spawn((Big::new(0), Tag));
    let query = w.query_filtered::<&Big, With<SparseDirty>>();
    assert!(query.iter(&w).next().is_none());

    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    assert!(w.insert(entity, SparseDirty(7)));
    w.get_mut::<Big>(entity).unwrap().tag += 1;

    let report = query.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(report.stats.access_columns, 1);
    assert_eq!(report.stats.filter_sources, 1);
    assert_eq!(report.stats.sparse_entities_checked, 0);
    assert_eq!(report.stats.sparse_fallback_scans, 0);
    assert_eq!(report.stats.sparse_index_events, 1);
    assert_eq!(report.stats.sparse_chunk_candidates, 1);
    assert_eq!(report.stats.column_candidates, 1);
    assert_eq!(report.stats.merged_candidates, 1);
    assert_eq!(report.chunks.len(), 1);
}

#[test]
fn sparse_membership_removal_is_reported_for_live_entity() {
    let mut w = World::new();
    let entity = w.spawn(Big::new(0));
    assert!(w.insert(entity, SparseDirty(7)));
    let with = w.query_filtered::<Entity, With<SparseDirty>>();
    let without = w.query_filtered::<Entity, Without<SparseDirty>>();
    assert_eq!(with.iter(&w).count(), 1);
    assert_eq!(without.iter(&w).count(), 0);

    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    assert!(w.remove::<SparseDirty>(entity));

    let with_report = with.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(with_report.stats.sparse_index_events, 1);
    assert_eq!(with_report.chunks.len(), 1);
    assert!(with_report.chunks[0].start <= w.entities().location(entity).unwrap().row as usize);

    let without_report = without.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(without_report.stats.sparse_index_events, 1);
    assert_eq!(without_report.chunks.len(), 1);
}

#[test]
fn sparse_filter_source_propagates_through_or() {
    let mut w = World::new();
    let entity = w.spawn(Big::new(0));
    let query = w.query_filtered::<Entity, Or<(With<SparseDirty>, With<Tag>)>>();
    assert!(query.iter(&w).next().is_none());

    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    assert!(w.insert(entity, SparseDirty(9)));

    let report = query.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(report.stats.filter_sources, 1);
    assert_eq!(report.stats.sparse_entities_checked, 0);
    assert_eq!(report.stats.sparse_fallback_scans, 0);
    assert_eq!(report.stats.sparse_index_events, 1);
    assert_eq!(report.stats.sparse_chunk_candidates, 1);
    assert_eq!(report.chunks.len(), 1);
}
