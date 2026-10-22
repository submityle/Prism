//! Behavioural tests for the dirty-chunk accessor
//! ([`QueryState::dirty_chunks`](crate::query::QueryState::dirty_chunks)), the
//! coarse, chunk-granular half of double-layer change detection (design §7/§10).
//!
//! The accessor must be a correct *superset* of the per-row
//! [`Changed<T>`](crate::query::Changed) scan: every row a `Changed<T>` filter
//! would yield lies inside some reported window, and no window is reported for a
//! chunk whose rows are all older than `last_run`.

use alloc::vec::Vec;
use core::cell::Cell;

use crate::change::Tick;
use crate::component::{Component, Components};
use crate::entity::Entity;
use crate::query::access::Access;
use crate::query::filter::QueryFilter;
use crate::query::{Changed, DirtyChunk, Or, With, Without};
use crate::storage::rows_per_chunk;
use crate::storage::SparseSets;
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

std::thread_local! {
    static PANIC_NEXT_CHUNK_FILTER: Cell<bool> = const { Cell::new(false) };
}

struct PanicOnceChunkFilter;

unsafe impl QueryFilter for PanicOnceChunkFilter {
    type State = ();
    type Fetch<'w> = ();

    fn init_state(_components: &mut Components) -> Self::State {}

    fn matches(_state: &Self::State, _archetype: &crate::archetype::Archetype) -> bool {
        true
    }

    fn update_access(_state: &Self::State, _access: &mut Access) {}

    unsafe fn init_fetch<'w>(
        _state: &Self::State,
        _archetype: &'w crate::archetype::Archetype,
        _sparse_sets: &'w SparseSets,
        _last_run: Tick,
        _this_run: Tick,
    ) -> Self::Fetch<'w> {
    }

    fn filter_chunk<'w>(_fetch: Self::Fetch<'w>, _chunk: usize) -> bool {
        if PANIC_NEXT_CHUNK_FILTER.with(|panic_next| panic_next.replace(false)) {
            panic!("injected one-shot dirty-chunk filter failure");
        }
        true
    }

    unsafe fn filter_fetch<'w>(_fetch: Self::Fetch<'w>, _entity: Entity, _row: usize) -> bool {
        true
    }
}

macro_rules! noise_components {
    ($($name:ident),+ $(,)?) => {
        $(
            #[derive(Clone, Copy)]
            struct $name;
            impl Component for $name {}
        )+
    };
}

noise_components!(Noise0, Noise1, Noise2, Noise3, Noise4, Noise5, Noise6, Noise7, Noise8);

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
fn dirty_journal_cursor_is_acknowledged_only_after_filter_success() {
    let mut world = World::new();
    let entity = world.spawn((Big::new(1), SparseDirty(0)));
    let state = world.query_filtered::<&Big, (With<SparseDirty>, PanicOnceChunkFilter)>();

    let baseline = world.change_tick();
    assert!(state
        .dirty_chunks_with_stats(&world, baseline, baseline)
        .chunks
        .is_empty());

    advance_frame(&mut world);
    let last_run = world.last_change_tick();
    let this_run = world.change_tick();
    world.get_mut::<Big>(entity).unwrap().tag += 1;
    world.get_mut::<SparseDirty>(entity).unwrap().0 += 1;

    PANIC_NEXT_CHUNK_FILTER.with(|panic_next| panic_next.set(true));
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.dirty_chunks_with_stats(&world, last_run, this_run)
    }));
    assert!(panic.is_err(), "the injected filter panic must be observed");

    let retry = state.dirty_chunks_with_stats(&world, last_run, this_run);
    assert_eq!(retry.chunks.len(), 1);
    assert_eq!(retry.stats.journal_events, 1);
    assert_eq!(retry.stats.sparse_index_events, 1);
}

#[test]
fn dirty_fallback_window_is_retryable_after_filter_panic() {
    let mut world = World::new();
    let entity = world.spawn((Big::new(1), SparseDirty(0)));
    let state = world.query_filtered::<&Big, (With<SparseDirty>, PanicOnceChunkFilter)>();

    let initial_tick = world.change_tick();
    assert!(state
        .dirty_chunks_with_stats(&world, initial_tick, initial_tick)
        .chunks
        .is_empty());

    advance_frame(&mut world);
    let first_last_run = world.last_change_tick();
    let first_this_run = world.change_tick();
    world.get_mut::<Big>(entity).unwrap().tag += 1;
    world.get_mut::<SparseDirty>(entity).unwrap().0 += 1;
    assert_eq!(
        state
            .dirty_chunks_with_stats(&world, first_last_run, first_this_run)
            .chunks
            .len(),
        1
    );

    advance_frame(&mut world);
    world.get_mut::<Big>(entity).unwrap().tag += 1;
    world.get_mut::<SparseDirty>(entity).unwrap().0 += 1;
    let fallback_this_run = world.change_tick();

    // Rewind the lower bound across the previous request. The retained sparse
    // event history must make this window replayable without scanning the set.
    PANIC_NEXT_CHUNK_FILTER.with(|panic_next| panic_next.set(true));
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.dirty_chunks_with_stats(&world, initial_tick, fallback_this_run)
    }));
    assert!(panic.is_err(), "the injected filter panic must be observed");

    let retry = state.dirty_chunks_with_stats(&world, initial_tick, fallback_this_run);
    assert_eq!(retry.chunks.len(), 1);
    assert_eq!(retry.stats.sparse_fallback_scans, 0);
    assert_eq!(retry.stats.sparse_replay_sources, 1);
    assert_eq!(retry.stats.sparse_index_events, 2);
}

#[test]
fn forward_dirty_query_checks_only_changed_archetypes_in_large_world() {
    let mut w = World::new();
    for mask in 0_u8..(1 << 6) {
        let entity = w.spawn(());
        if mask & (1 << 0) != 0 {
            w.insert(entity, Noise0);
        }
        if mask & (1 << 1) != 0 {
            w.insert(entity, Noise1);
        }
        if mask & (1 << 2) != 0 {
            w.insert(entity, Noise2);
        }
        if mask & (1 << 3) != 0 {
            w.insert(entity, Noise3);
        }
        if mask & (1 << 4) != 0 {
            w.insert(entity, Noise4);
        }
        if mask & (1 << 5) != 0 {
            w.insert(entity, Noise5);
        }
    }
    let target = w.spawn(Big::new(0));
    assert_eq!(w.archetypes().len(), 65, "64 noise archetypes + target");

    let state = w.query::<&Big>();
    let baseline_tick = w.change_tick();
    let baseline = state.dirty_chunks_with_stats(&w, baseline_tick, baseline_tick);
    assert!(baseline.chunks.is_empty());
    assert!(
        baseline.stats.journal_gap,
        "first call establishes baseline"
    );

    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    w.get_mut::<Big>(target).unwrap().tag += 1;

    let report = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert!(!report.stats.journal_gap);
    assert_eq!(report.stats.journal_records_scanned, 1);
    assert_eq!(report.stats.candidate_archetype_checks, 1);
    assert_eq!(report.stats.matched_archetypes, 1);
    assert_eq!(report.stats.summaries_tested, 0);
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
    assert_eq!(report.stats.journal_records_scanned, WRITES);
    assert_eq!(report.stats.journal_runs_materialized, 1);
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
    assert_eq!(report.stats.journal_runs_materialized, CHUNKS);
    assert_eq!(report.stats.journal_events, CHUNKS);
    assert_eq!(report.stats.candidate_chunk_marks, CHUNKS);
    assert_eq!(report.stats.merged_candidates, CHUNKS);
    assert_eq!(report.stats.reported_chunks, CHUNKS);
    assert_eq!(report.chunks.len(), CHUNKS);
    assert_eq!(report.stats.reported_rows, CHUNKS * 4);
}

#[test]
fn journal_overflow_falls_back_without_missing_chunks_or_replaying_them() {
    use crate::archetype::COLUMN_CHANGE_LOG_CAPACITY;

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
    // Overflow the retained event window while touching every chunk. The
    // journal must request a conservative indexed rebuild rather than lose
    // early events that have been evicted.
    for _round in 0..(COLUMN_CHANGE_LOG_CAPACITY / CHUNKS + 1) {
        for chunk in 0..CHUNKS {
            w.get_mut::<Big>(entities[chunk * 4]).unwrap().tag += 1;
        }
    }

    let report = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert!(report.stats.journal_gap);
    assert_eq!(report.stats.journal_records_scanned, 0);
    assert_eq!(report.stats.column_candidates, CHUNKS);
    assert_eq!(report.chunks.len(), CHUNKS);
    assert_eq!(report.stats.reported_rows, CHUNKS * 4);

    // The fallback covered this current tick window and advanced the journal
    // cursor. A later clean forward window must not replay those same writes.
    advance_frame(&mut w);
    let next = state.dirty_chunks_with_stats(&w, this_run, w.change_tick());
    assert!(!next.stats.journal_gap);
    assert!(next.chunks.is_empty());
}

#[test]
fn dirty_fallback_uses_table_component_index_for_optional_queries() {
    use crate::archetype::COLUMN_CHANGE_LOG_CAPACITY;

    let mut w = World::new();
    for mask in 0_u8..(1 << 6) {
        let entity = w.spawn(());
        if mask & (1 << 0) != 0 {
            w.insert(entity, Noise0);
        }
        if mask & (1 << 1) != 0 {
            w.insert(entity, Noise1);
        }
        if mask & (1 << 2) != 0 {
            w.insert(entity, Noise2);
        }
        if mask & (1 << 3) != 0 {
            w.insert(entity, Noise3);
        }
        if mask & (1 << 4) != 0 {
            w.insert(entity, Noise4);
        }
        if mask & (1 << 5) != 0 {
            w.insert(entity, Noise5);
        }
    }
    let target = w.spawn(Big::new(0));
    assert_eq!(w.archetypes().len(), 65);

    // `Option<&Big>` matches every archetype, including all 64 unrelated
    // variants. Only one archetype actually owns the accessed table column.
    let state = w.query::<Option<&Big>>();
    let baseline = state.dirty_chunks_with_stats(&w, w.change_tick(), w.change_tick());
    assert!(baseline.stats.journal_gap);

    let last_run = w.change_tick();
    w.increment_change_tick();
    let this_run = w.change_tick();
    for _ in 0..(COLUMN_CHANGE_LOG_CAPACITY + 1) {
        w.get_mut::<Big>(target).unwrap().tag += 1;
    }

    let report = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert!(report.stats.journal_gap);
    assert_eq!(report.stats.archetype_index_entries_scanned, 1);
    assert_eq!(report.stats.candidate_archetype_checks, 1);
    assert_eq!(report.stats.matched_archetypes, 1);
    assert_eq!(report.stats.column_candidates, 1);
    assert_eq!(report.stats.reported_chunks, 1);
    assert_eq!(report.chunks.len(), 1);
}

#[test]
fn dirty_fallback_merges_overlapping_component_postings_once() {
    use crate::archetype::COLUMN_CHANGE_LOG_CAPACITY;

    let mut w = World::new();
    let mut target = None;
    for mask in 0_u16..(1 << 9) {
        let entity = w.spawn(());
        if mask & (1 << 0) != 0 {
            w.insert(entity, Noise0);
        }
        if mask & (1 << 1) != 0 {
            w.insert(entity, Noise1);
        }
        if mask & (1 << 2) != 0 {
            w.insert(entity, Noise2);
        }
        if mask & (1 << 3) != 0 {
            w.insert(entity, Noise3);
        }
        if mask & (1 << 4) != 0 {
            w.insert(entity, Noise4);
        }
        if mask & (1 << 5) != 0 {
            w.insert(entity, Noise5);
        }
        if mask & (1 << 6) != 0 {
            w.insert(entity, Noise6);
        }
        if mask & (1 << 7) != 0 {
            w.insert(entity, Noise7);
        }
        if mask & (1 << 8) != 0 {
            w.insert(entity, Noise8);
        }
        if mask == (1 << 9) - 1 {
            target = Some(entity);
        }
    }
    let target = target.expect("the full component-set archetype was spawned");

    let state = w.query_filtered::<(
        Option<&Noise0>,
        Option<&Noise1>,
        Option<&Noise2>,
        Option<&Noise3>,
        Option<&Noise4>,
        Option<&Noise5>,
        Option<&Noise6>,
        Option<&Noise7>,
        Option<&Noise8>,
    ), With<Noise0>>();
    let baseline = state.dirty_chunks_with_stats(&w, w.change_tick(), w.change_tick());
    assert!(baseline.stats.journal_gap);

    let last_run = w.change_tick();
    w.increment_change_tick();
    let this_run = w.change_tick();
    for _ in 0..(COLUMN_CHANGE_LOG_CAPACITY + 1) {
        *w.get_mut::<Noise0>(target).unwrap() = Noise0;
    }

    let report = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert!(report.stats.journal_gap);
    // Each of nine columns is present in 256 component-set archetypes, forcing
    // every posting past the 128-item inline limit and through chunked storage.
    // Their union is the 511 non-empty combinations, with overlaps removed
    // once while the nine sorted posting sources are merged.
    assert_eq!(w.archetypes().len(), 512);
    assert_eq!(report.stats.archetype_index_entries_scanned, 9 * 256);
    assert_eq!(report.stats.candidate_archetype_checks, 511);
    // Candidate IDs include every non-empty component combination, while the
    // With<Noise0> query matches only half; the sorted intersection must skip
    // non-matches without disturbing later matching candidates.
    assert_eq!(report.stats.matched_archetypes, 256);
    assert_eq!(report.stats.reported_chunks, 1);
    assert_eq!(report.chunks.len(), 1);
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
    let overlapping_sources: Vec<&[usize]> = overlapping.iter().map(Vec::as_slice).collect();
    let mut merged = Vec::new();
    let operations = merge_sorted_chunks(&overlapping_sources, |chunk| merged.push(chunk));
    assert_eq!(merged, (0..CHUNKS).collect::<Vec<_>>());
    assert_eq!(operations, 2 * SOURCES * CHUNKS);

    let disjoint: Vec<Vec<usize>> = (0..SOURCES)
        .map(|source| (source * CHUNKS..(source + 1) * CHUNKS).collect())
        .collect();
    let disjoint_sources: Vec<&[usize]> = disjoint.iter().map(Vec::as_slice).collect();
    merged.clear();
    let operations = merge_sorted_chunks(&disjoint_sources, |chunk| merged.push(chunk));
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
fn sparse_dirty_query_replays_overlapping_window_without_live_set_scan() {
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
    assert_eq!(repeated.stats.sparse_replay_sources, 1);
    assert_eq!(repeated.stats.sparse_index_batches_scanned, 1);
    assert_eq!(repeated.stats.sparse_index_events, 1);
    assert_eq!(repeated.stats.sparse_fallback_scans, 0);
    assert_eq!(repeated.stats.sparse_entities_checked, 0);
    assert_eq!(repeated.chunks, first.chunks);
}

#[test]
fn sparse_retained_replay_matches_historical_tick_oracle() {
    let mut w = World::new();
    let entities: Vec<_> = (0..8)
        .map(|i| w.spawn((Big::new(i), SparseDirty(i as u32))))
        .collect();
    let state = w.query::<&SparseDirty>();
    let component = w.components().id_of::<SparseDirty>().unwrap();
    let activation = w.change_tick();
    assert!(state
        .dirty_chunks_with_stats(&w, activation, activation)
        .chunks
        .is_empty());

    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    w.get_mut::<SparseDirty>(entities[0]).unwrap().0 += 10;
    let this_run = w.change_tick();
    let first = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(first.stats.sparse_index_events, 1);

    // Advance the world and create a later event which must not leak into the
    // historical replay of the first window.
    advance_frame(&mut w);
    w.get_mut::<SparseDirty>(entities[7]).unwrap().0 += 10;
    let replay = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(replay.stats.sparse_replay_sources, 1);
    assert_eq!(replay.stats.sparse_fallback_scans, 0);
    assert_eq!(replay.stats.sparse_index_events, 1);
    assert_eq!(replay.stats.sparse_entities_checked, 0);

    let changed_rows: Vec<_> = entities
        .iter()
        .filter_map(|&entity| {
            let changed = w
                .sparse_sets()
                .get(component)
                .unwrap()
                .changed_tick(entity)
                .unwrap();
            changed
                .is_newer_than(last_run, this_run)
                .then(|| w.entities().location(entity).unwrap().row as usize)
        })
        .collect();
    assert_eq!(changed_rows.len(), 1);
    for &row in &changed_rows {
        assert!(row_is_covered(&replay.chunks, row));
    }
    for chunk in &replay.chunks {
        assert!(changed_rows
            .iter()
            .any(|&row| row >= chunk.start && row < chunk.end));
    }
}

#[test]
fn sparse_historical_window_selects_only_its_tick_from_full_retained_history() {
    let mut w = World::new();
    let entity = w.spawn((Big::new(0), SparseDirty(0)));
    let state = w.query::<&SparseDirty>();
    let activation = w.change_tick();
    let _ = state.dirty_chunks_with_stats(&w, activation, activation);
    let mut target = (Tick::ZERO, Tick::ZERO);
    for index in 0..crate::change_index::MAX_RETAINED_BATCHES {
        advance_frame(&mut w);
        w.get_mut::<SparseDirty>(entity).unwrap().0 += 1;
        if index == 2047 {
            target = (w.last_change_tick(), w.change_tick());
        }
    }
    let replay = state.dirty_chunks_with_stats(&w, target.0, target.1);
    assert_eq!(replay.stats.sparse_replay_sources, 1);
    assert_eq!(replay.stats.sparse_index_batches_scanned, 1);
    assert_eq!(replay.stats.sparse_index_events, 1);
    assert_eq!(replay.stats.sparse_index_gaps, 0);
    assert_eq!(replay.stats.sparse_fallback_scans, 0);
    assert_eq!(replay.stats.sparse_entities_checked, 0);
    assert_eq!(replay.chunks.len(), 1);
    assert!(row_is_covered(&replay.chunks, 0));
    assert_eq!(
        replay.chunks,
        state.dirty_chunks_with_stats(&w, target.0, target.1).chunks
    );
}

#[test]
fn sparse_replay_gap_rebaselines_transactionally_for_next_forward_window() {
    let mut w = World::new();
    let entity = w.spawn((Big::new(0), SparseDirty(0)));
    let state = w.query::<&SparseDirty>();
    let activation = w.change_tick();
    assert!(state
        .dirty_chunks_with_stats(&w, activation, activation)
        .chunks
        .is_empty());

    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    w.get_mut::<SparseDirty>(entity).unwrap().0 += 1;
    let this_run = w.change_tick();
    let first = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(first.stats.sparse_index_events, 1);

    // A restore changes the tracker epoch and invalidates activation history.
    // The replay must fall back, then transactionally publish a current cursor.
    w.change_tracker.reset();
    let recovered = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(recovered.stats.sparse_index_gaps, 1);
    assert_eq!(recovered.stats.sparse_fallback_scans, 1);
    assert_eq!(recovered.stats.reported_chunks, 1);

    advance_frame(&mut w);
    let next_last = w.last_change_tick();
    w.get_mut::<SparseDirty>(entity).unwrap().0 += 1;
    let next = state.dirty_chunks_with_stats(&w, next_last, w.change_tick());
    assert_eq!(next.stats.sparse_index_gaps, 0);
    assert_eq!(next.stats.sparse_fallback_scans, 0);
    assert_eq!(next.stats.sparse_index_events, 1);
    assert_eq!(next.stats.sparse_index_batches_scanned, 1);
    assert_eq!(next.stats.reported_chunks, 1);
    assert!(next.stats.reported_rows > 0);
}

#[test]
fn sparse_historical_gap_recovery_preserves_current_cursor_coverage() {
    let mut w = World::new();
    let entity = w.spawn((Big::new(0), SparseDirty(0)));
    let state = w.query::<&SparseDirty>();
    let activation = w.change_tick();
    state.dirty_chunks(&w, activation, activation);
    advance_frame(&mut w);
    let last_run = w.change_tick();
    w.get_mut::<SparseDirty>(entity).unwrap().0 += 1;
    state.dirty_chunks(&w, activation, last_run);
    advance_frame(&mut w);
    w.get_mut::<SparseDirty>(entity).unwrap().0 += 1;
    let current = w.change_tick();

    // Restore invalidates retained history. Even an empty historical report
    // must rebase its sparse cursor through CURRENT, not the old upper bound.
    w.change_tracker.reset();
    let recovered = state.dirty_chunks_with_stats(&w, activation, activation);
    assert_eq!(recovered.stats.sparse_index_gaps, 1);
    assert_eq!(state.dirty_cursor_tick(), Some(current));
    let repeated = state.dirty_chunks_with_stats(&w, last_run, current);
    assert_eq!(repeated.stats.sparse_replay_sources, 1);
    assert_eq!(repeated.stats.sparse_index_gaps, 1);
    let location = w.entities().location(entity).unwrap();
    assert!(repeated.chunks.iter().any(|chunk| {
        chunk.archetype == location.archetype_id
            && row_is_covered(core::slice::from_ref(chunk), location.row as usize)
    }));

    // After that recovery, a disjoint forward window uses the new cursor.
    advance_frame(&mut w);
    w.get_mut::<SparseDirty>(entity).unwrap().0 += 1;
    let forward = state.dirty_chunks_with_stats(&w, current, w.change_tick());
    assert_eq!(forward.stats.sparse_replay_sources, 0);
    assert_eq!(forward.stats.sparse_index_gaps, 0);
    assert_eq!(forward.stats.sparse_index_events, 1);
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
    assert_eq!(report.stats.sparse_replay_sources, 0);
    assert_eq!(report.stats.sparse_chunk_candidates, 0);
}

#[test]
fn sparse_log_gap_falls_back_to_all_matching_table_chunks() {
    let mut w = World::new();
    let entities: Vec<_> = (0..4)
        .map(|i| w.spawn((Big::new(i), SparseDirty(i as u32))))
        .collect();
    advance_frame(&mut w);
    let state = w.query_filtered::<&Big, With<SparseDirty>>();
    let baseline = w.change_tick();
    let _ = state.dirty_chunks(&w, baseline, baseline);
    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();

    // Force the bounded source log to lose this tick's sparse records, then
    // remove the last current member. The fallback must still conservatively
    // include the table rows because current SparseSet membership cannot
    // reconstruct a lost removal event.
    for _ in 0..=crate::change_index::MAX_RETAINED_EVENTS {
        w.get_mut::<SparseDirty>(entities[0]).unwrap().0 += 1;
    }
    assert!(w.remove::<SparseDirty>(entities[3]));

    let report = state.dirty_chunks_with_stats(&w, last_run, this_run);
    assert_eq!(report.stats.sparse_index_gaps, 1);
    assert_eq!(report.stats.reported_rows, 4);
    assert_eq!(report.chunks.len(), 1);
}

#[test]
fn sparse_dirty_fallback_matches_tick_oracle_across_wrap() {
    let mut w = World::new();
    let entities: Vec<Entity> = (0..8).map(|i| w.spawn(SparseDirty(i))).collect();
    let state = w.query::<&SparseDirty>();
    let component = w.components().id_of::<SparseDirty>().unwrap();
    let ticks = [
        Tick::new(u32::MAX - 1),
        Tick::MAX,
        Tick::ZERO,
        Tick::new(1),
        Tick::new(Tick::MAX_CHANGE_AGE),
        Tick::new(u32::MAX - 3),
        Tick::new(2),
        Tick::new(3),
    ];
    for (&entity, tick) in entities.iter().zip(ticks) {
        // SAFETY: the test is single-threaded and no component borrow is live.
        let changed_tick = unsafe {
            w.sparse_sets()
                .get(component)
                .unwrap()
                .changed_tick_ptr(entity)
        }
        .unwrap();
        // SAFETY: the pointer came from the live sparse row above, which this
        // test exclusively owns while installing synthetic boundary ticks.
        unsafe { *changed_tick = tick };
    }

    let last_run = Tick::new(u32::MAX - 2);
    let this_run = Tick::new(1);
    let report = state.dirty_chunks_with_stats(&w, last_run, this_run);

    // This synthetic historical window does not end at World's current tick,
    // so it must use the live-set fallback rather than claim to test the sparse
    // change index. Compare each reported row window against the exact tick
    // oracle while allowing chunk-granularity false positives within a window.
    assert_eq!(report.stats.sparse_index_events, 0);
    assert_eq!(report.stats.sparse_fallback_scans, 1);
    assert_eq!(report.stats.sparse_entities_checked, entities.len());
    let changed_rows: Vec<_> = entities
        .iter()
        .filter_map(|&entity| {
            let changed = w
                .sparse_sets()
                .get(component)
                .unwrap()
                .changed_tick(entity)
                .unwrap();
            changed
                .is_newer_than(last_run, this_run)
                .then(|| w.entities().location(entity).unwrap().row as usize)
        })
        .collect();
    assert!(!changed_rows.is_empty());
    for &row in &changed_rows {
        assert!(row_is_covered(&report.chunks, row), "changed row {row}");
    }
    for chunk in &report.chunks {
        assert!(changed_rows
            .iter()
            .any(|&row| row >= chunk.start && row < chunk.end));
    }
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
