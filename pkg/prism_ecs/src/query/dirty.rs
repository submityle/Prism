//! The dirty-chunk accessor: the coarse, chunk-granular half of double-layer
//! change detection (design §7 脏块访问器, §10).
//!
//! Where [`Changed<T>`](crate::query::Changed) filters row-by-row, the
//! dirty-chunk accessor answers the *coarse* question "which row windows of
//! which archetypes contain anything this query touched since `last_run`?" by
//! consuming retained column-change events on the normal forward path and
//! traversing a hierarchical [`ChunkVersions`](crate::storage::ChunkVersions)
//! interval index on conservative paths. The fallback prunes archetypes through
//! table-component postings before reading chunk intervals, and prunes clean
//! interval subtrees within candidate columns. A GPU-resident uploader (design
//! §15) or a push-reaction pass walks the returned [`DirtyChunk`]s to upload /
//! react to only the changed windows; the per-row ticks remain available to
//! refine within a flagged window.
//!
//! # Correctness: a superset, never a miss
//!
//! A chunk's version upper-bounds the newest changed-tick of any row in its
//! window (the invariant maintained in [`ChunkVersions`]). So if any row in a
//! chunk changed within `(last_run, this_run]`, the chunk version also lies in
//! (or after) that window and the chunk is reported. The result is therefore a
//! correct *superset* of the rows a per-row [`Changed<T>`] scan would yield: it
//! never misses a changed row, and may occasionally include a window whose only
//! recently-changed row has since been swap-removed. The design explicitly
//! accepts this over-approximation for the coarse layer.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::iter::Peekable;
use hashbrown::HashSet;

use crate::archetype::ArchetypeId;
use crate::change::Tick;
use crate::component::StorageType;
use crate::query::fetch::QueryData;
use crate::query::filter::QueryFilter;
use crate::query::state::QueryState;
use crate::world::World;

/// Counters for one [`QueryState::dirty_chunks_with_stats`] pass.
///
/// These counters describe the actual coarse-path work. They are intentionally
/// metadata counts rather than wall-clock measurements, so tests can assert
/// the shape of the traversal without making machine-dependent timing claims.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirtyChunkStats {
    /// Number of archetypes that pass query/filter matching and are inspected.
    pub matched_archetypes: usize,
    /// Number of distinct component ids in the query's read/write access.
    pub access_columns: usize,
    /// Number of distinct component ids contributed by sparse membership
    /// filters as entity-indexed dirty sources.
    pub filter_sources: usize,
    /// Number of accessed ids resolved to table columns in the current
    /// archetype. Sparse/shared-only access has no table row-window source.
    pub table_columns: usize,
    /// Number of accessed ids with no table column in the current archetype.
    /// This includes sparse-backed sources, whose changes are entity-indexed.
    pub non_table_columns: usize,
    /// Number of candidate chunk leaves returned across all accessed columns,
    /// before cross-column deduplication.
    pub column_candidates: usize,
    /// Number of unique candidate chunks after the multi-column merge.
    pub merged_candidates: usize,
    /// Number of merged candidates rejected by `F::filter_chunk`.
    pub filter_rejected: usize,
    /// Number of row windows returned to the caller.
    pub reported_chunks: usize,
    /// Sum of rows covered by the returned windows.
    pub reported_rows: usize,
    /// Number of hierarchical chunk summaries inspected by the column indexes.
    pub summaries_tested: usize,
    /// Number of entities inspected in sparse-set change sources.
    pub sparse_entities_checked: usize,
    /// Number of sparse sets scanned because the requested window predates the
    /// query's change-index activation boundary.
    pub sparse_fallback_scans: usize,
    /// Number of sparse-derived chunk candidates after row-to-chunk deduplication.
    pub sparse_chunk_candidates: usize,
    /// Number of sparse writes/removals delivered from the exact change index.
    pub sparse_index_events: usize,
    /// Tick batches inspected by the forward sparse cursor or retained-history
    /// replay, including batches outside the requested tick window.
    pub sparse_index_batches_scanned: usize,
    /// Number of component logs queried through the non-advancing retained
    /// history replay path.
    pub sparse_replay_sources: usize,
    /// Number of source logs that reported lost retained history and forced a
    /// conservative all-matched-chunk fallback.
    pub sparse_index_gaps: usize,
    /// Number of matching table journal events consumed by the strict
    /// incremental path.
    pub journal_events: usize,
    /// Number of component-indexed journal event records selected/materialized
    /// for merge by the filtered read, including structural records and before
    /// archetype or chunk validation. Binary-search probes and unrelated writes
    /// are excluded. A conservative gap reports zero because no delta records
    /// are consumed.
    pub journal_records_scanned: usize,
    /// Number of distinct candidate archetypes checked against query data and
    /// filters after the journal/sparse candidate union or fallback index lookup.
    pub candidate_archetype_checks: usize,
    /// Number of table-component posting-list entries visited while building
    /// the conservative fallback's candidate archetype set. This can exceed
    /// the number of unique candidate archetypes when accessed columns share
    /// archetypes.
    pub archetype_index_entries_scanned: usize,
    /// Number of chunk candidates offered before per-archetype deduplication.
    pub candidate_chunk_marks: usize,
    /// Whether this request had to use the conservative path because the
    /// retained table journal could not establish a complete baseline.
    pub journal_gap: bool,
}

/// Dirty row windows plus measurable traversal counters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirtyChunkReport {
    /// Conservative, chunk-aligned row windows.
    pub chunks: Vec<DirtyChunk>,
    /// Metadata work and output volume for this pass.
    pub stats: DirtyChunkStats,
}

#[derive(Default)]
struct DirtyChunkCandidates {
    chunks: Vec<usize>,
    chunk_index: Option<HashSet<usize>>,
    journal_events: usize,
}

impl DirtyChunkCandidates {
    // Most dirty windows contain only a few chunks. Keep that common path
    // allocation-light, then promote to an indexed set for high-cardinality
    // updates where repeated linear membership checks would dominate.
    const INLINE_CHUNKS: usize = 8;

    fn insert_chunk(&mut self, chunk: usize) {
        if let Some(index) = &mut self.chunk_index {
            if index.insert(chunk) {
                self.chunks.push(chunk);
            }
            return;
        }

        if self.chunks.contains(&chunk) {
            return;
        }
        if self.chunks.len() < Self::INLINE_CHUNKS {
            self.chunks.push(chunk);
            return;
        }

        let mut index = HashSet::with_capacity(self.chunks.len() * 2);
        index.extend(self.chunks.iter().copied());
        if index.insert(chunk) {
            self.chunks.push(chunk);
        }
        self.chunk_index = Some(index);
    }
}

/// Visit the sorted unique union of sorted sources using a min-heap.
/// A small-source linear path avoids heap allocation for common low-arity
/// queries. Returns heap pushes plus pops (zero for the linear path).
pub(super) fn merge_sorted_chunks<T: Copy + Ord>(
    sources: &[&[T]],
    mut visit: impl FnMut(T),
) -> usize {
    if sources.len() <= 4 {
        let mut cursors = [0usize; 4];
        loop {
            let next = sources
                .iter()
                .enumerate()
                .filter_map(|(index, chunks)| chunks.get(cursors[index]).copied())
                .min();
            let Some(chunk) = next else {
                return 0;
            };
            for (index, chunks) in sources.iter().enumerate() {
                let cursor = &mut cursors[index];
                while chunks.get(*cursor).copied() == Some(chunk) {
                    *cursor += 1;
                }
            }
            visit(chunk);
        }
    }

    let mut cursors = vec![0usize; sources.len()];
    let mut heap = BinaryHeap::new();
    let mut heap_operations = 0;
    for (source, chunks) in sources.iter().enumerate() {
        if let Some(&chunk) = chunks.first() {
            heap.push(Reverse((chunk, source)));
            heap_operations += 1;
        }
    }

    while let Some(Reverse((chunk, _))) = heap.peek().copied() {
        while heap.peek().is_some_and(|Reverse((next, _))| *next == chunk) {
            let Reverse((_, source)) = heap.pop().expect("peeked heap item");
            heap_operations += 1;
            let chunks = &sources[source];
            let cursor = &mut cursors[source];
            *cursor += 1;
            while chunks.get(*cursor).is_some_and(|&next| next == chunk) {
                *cursor += 1;
            }
            if let Some(&next) = chunks.get(*cursor) {
                heap.push(Reverse((next, source)));
                heap_operations += 1;
            }
        }
        visit(chunk);
    }
    heap_operations
}

/// Merge sorted archetype postings as logical sources rather than exposing
/// every bounded posting block as a separate source. This keeps heap width
/// proportional to queried columns (plus sparse candidates), independent of
/// archetype count.
fn merge_sorted_archetype_iterators<'a>(
    sources: &mut [Peekable<Box<dyn Iterator<Item = ArchetypeId> + 'a>>],
    mut visit: impl FnMut(ArchetypeId),
) -> usize {
    if sources.len() <= 4 {
        let mut last = None;
        loop {
            let next = sources
                .iter_mut()
                .filter_map(|source| source.peek().copied())
                .min();
            let Some(id) = next else { return 0 };
            if last != Some(id) {
                visit(id);
                last = Some(id);
            }
            for source in sources.iter_mut() {
                while source.peek().copied() == Some(id) {
                    source.next();
                }
            }
        }
    }

    let mut heap = BinaryHeap::new();
    let mut heap_operations = 0;
    for (source_index, source) in sources.iter_mut().enumerate() {
        if let Some(&id) = source.peek() {
            heap.push(Reverse((id, source_index)));
            heap_operations += 1;
        }
    }
    let mut last = None;
    while let Some(Reverse((id, source_index))) = heap.pop() {
        heap_operations += 1;
        if last != Some(id) {
            visit(id);
            last = Some(id);
        }
        let source = &mut sources[source_index];
        while source.peek().copied() == Some(id) {
            source.next();
        }
        if let Some(&next) = source.peek() {
            heap.push(Reverse((next, source_index)));
            heap_operations += 1;
        }
    }
    heap_operations
}

/// A contiguous window of rows in one archetype whose chunk-version indicates a
/// change within a query's observed tick window.
///
/// The window is the half-open row range `[start, end)` of the archetype's
/// table — the same window across every column, since all columns of a table
/// share one `rows_per_chunk` (design §7). Row indices are stable only while no
/// structural change occurs, i.e. for the duration of a single read pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DirtyChunk {
    /// The archetype whose table the window belongs to.
    pub archetype: ArchetypeId,
    /// First row of the window (inclusive).
    pub start: usize,
    /// One past the last row of the window (exclusive).
    pub end: usize,
}

impl DirtyChunk {
    /// Number of rows in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the window is empty (never true for a reported chunk).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

impl<D: QueryData, F: QueryFilter> QueryState<D, F> {
    fn dirty_chunks_from_journal(
        &self,
        world: &World,
        last_run: Tick,
        this_run: Tick,
        access_ids: &[crate::component::ComponentId],
        sparse_chunks: &BTreeMap<ArchetypeId, Vec<usize>>,
        report: &mut DirtyChunkReport,
    ) -> bool {
        // Journal events do not carry simulation ticks. They are therefore
        // sound only for the normal forward window ending at the world's
        // current tick; rewound/historical requests use the interval index.
        let current = world.change_tick();
        if this_run != current
            || self
                .dirty_window()
                .is_some_and(|(_, previous)| previous.is_newer_than(last_run, current))
        {
            return false;
        }

        let changes = world
            .archetypes()
            .column_changes_since_components(self.dirty_column_cursor(), access_ids);
        report.stats.journal_gap = changes.gap;
        report.stats.journal_records_scanned = changes.records_scanned;
        if changes.gap {
            return false;
        }

        let mut candidates: BTreeMap<ArchetypeId, DirtyChunkCandidates> = BTreeMap::new();
        for (archetype, rows) in sparse_chunks {
            let rows_per_chunk = world
                .archetypes()
                .get(*archetype)
                .map_or(1, |a| a.table().rows_per_chunk());
            let candidate = candidates.entry(*archetype).or_default();
            for &row in rows {
                report.stats.candidate_chunk_marks += 1;
                candidate.insert_chunk(row / rows_per_chunk);
            }
        }
        for change in changes.events {
            if let Some(component) = change.component
                && access_ids.binary_search(&component).is_err()
            {
                continue;
            }
            let Some(archetype) = world.archetypes().get(change.archetype) else {
                continue;
            };
            let Some(chunk) = change.chunk else {
                continue;
            };
            if chunk >= archetype.table().chunk_count() {
                continue;
            }
            let candidate = candidates.entry(change.archetype).or_default();
            report.stats.candidate_chunk_marks += 1;
            candidate.insert_chunk(chunk);
            candidate.journal_events += 1;
        }

        for (archetype_id, candidate) in candidates {
            let Some(archetype) = world.archetypes().get(archetype_id) else {
                continue;
            };
            report.stats.candidate_archetype_checks += 1;
            if !D::matches(self.data_state(), archetype)
                || !F::matches(self.filter_state(), archetype)
            {
                continue;
            }
            report.stats.matched_archetypes += 1;
            report.stats.journal_events += candidate.journal_events;
            let table = archetype.table();
            let filter_fetch = unsafe {
                F::init_fetch(
                    self.filter_state(),
                    archetype,
                    world.sparse_sets(),
                    last_run,
                    this_run,
                )
            };
            let rpc = table.rows_per_chunk();
            let len = table.len();
            // Only unique chunks reach this sort. Hash-set iteration is not
            // observable; the published output remains deterministic.
            let mut chunks = candidate.chunks;
            chunks.sort_unstable();
            for chunk in chunks {
                report.stats.merged_candidates += 1;
                if !F::filter_chunk(filter_fetch, chunk) {
                    report.stats.filter_rejected += 1;
                    continue;
                }
                let start = chunk * rpc;
                let end = ((chunk + 1) * rpc).min(len);
                if start >= end {
                    continue;
                }
                report.chunks.push(DirtyChunk {
                    archetype: archetype_id,
                    start,
                    end,
                });
                report.stats.reported_chunks += 1;
                report.stats.reported_rows += end - start;
            }
        }
        // The cursor is only acknowledged after every candidate has been
        // filtered and the complete report has been built. A custom filter may
        // panic; in that case the caller can retry this same window without
        // losing the journal events.
        self.set_dirty_column_cursor(changes.cursor);
        self.remember_sparse_window(last_run, this_run);
        true
    }

    /// Collect the chunk windows this query touches whose change-version falls in
    /// the observer window `(last_run, this_run]` (design §7/§10).
    ///
    /// A chunk is reported when *any* component this query accesses (its read or
    /// write set) has a chunk-version newer than `last_run` for that window.
    /// This is the coarse layer: the returned windows form a superset of the
    /// rows a per-row [`Changed<T>`](crate::query::Changed) scan would visit (see
    /// the module docs). The forward journal path checks event-derived archetypes;
    /// a conservative tick-index fallback uses table-component postings plus
    /// sparse-change locations to build candidates, then intersects them with
    /// the incremental query-match cache before inspecting column intervals.
    pub fn dirty_chunks(&self, world: &World, last_run: Tick, this_run: Tick) -> Vec<DirtyChunk> {
        self.dirty_chunks_with_stats(world, last_run, this_run)
            .chunks
    }

    /// Collect dirty windows and counters for the coarse change path.
    ///
    /// The implementation first gathers sorted candidate chunk leaves from
    /// every distinct accessed table column, then performs a multi-way merge so
    /// overlapping columns are deduplicated without a global sort of all
    /// candidates. The merged chunk is conservatively retained only when the
    /// query filter's [`QueryFilter::filter_chunk`] accepts it.
    ///
    /// This is still a coarse superset. It does not turn a chunk window into an
    /// exact row set, and sparse-backed changes do not have a table chunk index;
    /// consumers needing exact entities must use `Changed<T>` iteration or the
    /// world change-observer/reaction boundary.
    pub fn dirty_chunks_with_stats(
        &self,
        world: &World,
        last_run: Tick,
        this_run: Tick,
    ) -> DirtyChunkReport {
        self.validate_world(world);
        let _dirty_pass = self.lock_dirty_pass();
        let mut report = DirtyChunkReport {
            chunks: Vec::new(),
            stats: DirtyChunkStats::default(),
        };

        // Components whose chunk-version signals a change relevant to this
        // query: everything it reads plus everything it writes. QueryData can
        // list the same read more than once, so deduplicate ids before touching
        // the columns.
        let access = self.access();
        let mut access_ids = Vec::with_capacity(access.reads().len() + access.writes().len());
        access_ids.extend_from_slice(access.reads());
        access_ids.extend_from_slice(access.writes());
        access_ids.sort_unstable();
        access_ids.dedup();
        report.stats.access_columns = access_ids.len();

        // Sparse membership is not part of QueryData's component access: the
        // filter only gates entities. It still changes the candidate set when
        // an entity gains or loses the sparse component, so merge its explicit
        // dirty sources into the same entity-to-row fallback used by sparse
        // data terms.
        let mut filter_sources = Vec::new();
        F::dirty_sources(self.filter_state(), &mut filter_sources);
        filter_sources.sort_unstable();
        filter_sources.dedup();
        report.stats.filter_sources = filter_sources.len();
        access_ids.extend(filter_sources);
        access_ids.sort_unstable();
        access_ids.dedup();

        // Sparse components do not own table chunks. Resolve their changed
        // entities once, then route each entity through the live location table
        // to the row window of its current archetype. This keeps the fallback
        // proportional to the sparse set's actual membership rather than to
        // every row in every matched archetype.
        let mut sparse_chunks: BTreeMap<ArchetypeId, Vec<usize>> = BTreeMap::new();
        let mut sparse_cursors = self.snapshot_dirty_cursors();
        let mut sparse_fallback = self.sparse_fallback_needed(world, last_run, this_run);
        let sparse_replay =
            !sparse_fallback && self.sparse_replay_needed(world, last_run, this_run);
        let mut indexed_entities = Vec::new();
        if !sparse_fallback {
            let index_stats = if sparse_replay {
                report.stats.sparse_replay_sources = self.dirty_sparse_source_count();
                self.replay_sparse_changes(last_run, this_run, &mut indexed_entities)
            } else {
                self.collect_sparse_changes(
                    last_run,
                    this_run,
                    &mut sparse_cursors,
                    &mut indexed_entities,
                )
            };
            report.stats.sparse_index_batches_scanned = index_stats.batches_scanned;
            if index_stats.gap {
                sparse_fallback = true;
                report.stats.sparse_index_gaps = 1;
                indexed_entities.clear();
                if sparse_replay {
                    // Capture a fresh forward boundary before the conservative
                    // scan. Keep it local: a filter/consumer panic must leave
                    // the old committed cursor intact so the request retries.
                    sparse_cursors = self.rebaseline_dirty_cursors();
                }
            }
        }
        report.stats.sparse_index_events = indexed_entities.len();
        for entity in indexed_entities {
            let Some(location) = world.entities().location(entity) else {
                continue;
            };
            sparse_chunks
                .entry(location.archetype_id)
                .or_default()
                .push(location.row as usize);
        }
        if sparse_fallback {
            for &id in &access_ids {
                let Some(info) = world.components().info(id) else {
                    continue;
                };
                if info.storage() != StorageType::SparseSet {
                    continue;
                }
                let Some(set) = world.sparse_sets().get(id) else {
                    continue;
                };
                report.stats.sparse_fallback_scans += 1;
                for &entity in set.entities() {
                    report.stats.sparse_entities_checked += 1;
                    let Some(ticks) = set.component_ticks(entity) else {
                        continue;
                    };
                    if !ticks.changed.is_newer_than(last_run, this_run) {
                        continue;
                    }
                    let Some(location) = world.entities().location(entity) else {
                        continue;
                    };
                    let chunks = sparse_chunks.entry(location.archetype_id).or_default();
                    chunks.push(location.row as usize);
                }
            }
        }
        for rows in sparse_chunks.values_mut() {
            rows.sort_unstable();
            rows.dedup();
        }

        // A bounded sparse log can lose an entity removal, which cannot be
        // reconstructed by scanning only the current sparse-set members. In
        // that rare case, return every row window in every structurally
        // matching archetype. This is a conservative superset and preserves
        // correctness without retaining unbounded event history.
        if report.stats.sparse_index_gaps != 0 {
            let journal_baseline = (this_run == world.change_tick())
                .then(|| world.archetypes().column_change_cursor());
            let matched = self.matched_archetypes(world);
            report.stats.matched_archetypes = matched.len();
            for &archetype_id in matched.as_ref() {
                let Some(archetype) = world.archetypes().get(archetype_id) else {
                    continue;
                };
                for chunk in archetype.table().iter_chunks() {
                    if chunk.rows == 0 {
                        continue;
                    }
                    let end = chunk.first_row + chunk.rows;
                    report.chunks.push(DirtyChunk {
                        archetype: archetype_id,
                        start: chunk.first_row,
                        end,
                    });
                    report.stats.reported_chunks += 1;
                    report.stats.reported_rows += chunk.rows;
                }
            }
            if let Some(cursor) = journal_baseline {
                self.set_dirty_column_cursor(cursor);
            }
            self.commit_dirty_cursors(sparse_cursors);
            self.remember_sparse_window(last_run, this_run);
            return report;
        }

        if self.dirty_chunks_from_journal(
            world,
            last_run,
            this_run,
            &access_ids,
            &sparse_chunks,
            &mut report,
        ) {
            self.commit_dirty_cursors(sparse_cursors);
            return report;
        }

        // A conservative scan covers the requested ticks but does not consume
        // journal revisions. If this was a current-tick request, remember a
        // revision boundary before scanning so those already-covered writes
        // cannot be replayed as changes in the next forward window. Capturing
        // before the scan keeps concurrent later writes available to the next
        // journal pass; such writes may conservatively overlap this scan.
        let journal_baseline =
            (this_run == world.change_tick()).then(|| world.archetypes().column_change_cursor());

        let matched_archetypes = self.matched_archetypes(world);

        // A conservative tick-index fallback used to walk every matched
        // archetype and probe every query access id for a table column. Build
        // the candidate set from the registry's inverted table-component
        // index instead. Sparse changes contribute their current archetypes
        // separately because sparse values are not table columns. The sorted
        // union keeps output ordering deterministic and does not alter the
        // interval-index superset semantics.
        let archetypes = world.archetypes();
        let mut sparse_archetypes = Vec::new();
        let mut candidate_sources: Vec<Peekable<Box<dyn Iterator<Item = ArchetypeId> + '_>>> =
            Vec::new();
        for &id in &access_ids {
            if !world
                .components()
                .info(id)
                .is_some_and(|info| info.storage() == StorageType::Table)
            {
                continue;
            }
            if let Some(indexed) = archetypes.with_component(id) {
                report.stats.archetype_index_entries_scanned += indexed.len();
                let source: Box<dyn Iterator<Item = ArchetypeId> + '_> = Box::new(indexed.iter());
                candidate_sources.push(source.peekable());
            }
        }
        if !sparse_chunks.is_empty() {
            sparse_archetypes.extend(sparse_chunks.keys().copied());
            sparse_archetypes.sort_unstable();
            let source: Box<dyn Iterator<Item = ArchetypeId> + '_> =
                Box::new(sparse_archetypes.iter().copied());
            candidate_sources.push(source.peekable());
        }
        let mut candidate_archetypes = Vec::new();
        merge_sorted_archetype_iterators(&mut candidate_sources, |id| {
            candidate_archetypes.push(id)
        });

        for archetype_id in candidate_archetypes {
            report.stats.candidate_archetype_checks += 1;
            if matched_archetypes.binary_search(&archetype_id).is_err() {
                continue;
            }
            report.stats.matched_archetypes += 1;
            let Some(archetype) = world.archetypes().get(archetype_id) else {
                continue;
            };
            let table = archetype.table();
            let rpc = table.rows_per_chunk();
            let len = table.len();

            // Each source is sorted by its index visitor. Merge the source
            // heads with a min-heap, so fallback work is
            // O((source entries + sources) * log(sources)), not a scan of all
            // columns for every distinct candidate chunk.
            let mut candidates = Vec::new();
            if let Some(rows) = sparse_chunks.get(&archetype_id)
                && !rows.is_empty()
            {
                let mut sparse_candidates = Vec::with_capacity(rows.len());
                let mut previous = None;
                for &row in rows {
                    let chunk = row / rpc;
                    if previous != Some(chunk) {
                        sparse_candidates.push(chunk);
                        previous = Some(chunk);
                    }
                }
                report.stats.sparse_chunk_candidates += sparse_candidates.len();
                candidates.push(sparse_candidates);
            }
            for &id in &access_ids {
                let Some(column) = table.column(id) else {
                    report.stats.non_table_columns += 1;
                    continue;
                };
                report.stats.table_columns += 1;
                let mut column_candidates = Vec::new();
                let stats = column.visit_changed_chunks(last_run, this_run, |chunk| {
                    column_candidates.push(chunk);
                });
                report.stats.column_candidates += stats.chunks_returned;
                report.stats.summaries_tested += stats.summaries_tested;
                candidates.push(column_candidates);
            }

            if candidates.is_empty() || len == 0 {
                continue;
            }

            // SAFETY: `archetype` is a matched archetype for F, so every
            // filter branch's init_fetch contract is satisfied.
            let filter_fetch = unsafe {
                F::init_fetch(
                    self.filter_state(),
                    archetype,
                    world.sparse_sets(),
                    last_run,
                    this_run,
                )
            };

            let chunk_count = len.div_ceil(rpc);
            let candidate_sources: Vec<&[usize]> = candidates.iter().map(Vec::as_slice).collect();
            merge_sorted_chunks(&candidate_sources, |chunk| {
                report.stats.merged_candidates += 1;
                if chunk >= chunk_count || !F::filter_chunk(filter_fetch, chunk) {
                    report.stats.filter_rejected += 1;
                    return;
                }

                let start = chunk * rpc;
                let end = ((chunk + 1) * rpc).min(len);
                report.chunks.push(DirtyChunk {
                    archetype: archetype_id,
                    start,
                    end,
                });
                report.stats.reported_chunks += 1;
                report.stats.reported_rows += end - start;
            });
        }
        if let Some(cursor) = journal_baseline {
            self.set_dirty_column_cursor(cursor);
        }
        self.commit_dirty_cursors(sparse_cursors);
        self.remember_sparse_window(last_run, this_run);
        report
    }
}
