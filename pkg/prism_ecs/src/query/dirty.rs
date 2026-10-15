//! The dirty-chunk accessor: the coarse, chunk-granular half of double-layer
//! change detection (design §7 脏块访问器, §10).
//!
//! Where [`Changed<T>`](crate::query::Changed) filters row-by-row, the
//! dirty-chunk accessor answers the *coarse* question "which row windows of
//! which archetypes contain anything this query touched since `last_run`?" by
//! traversing a hierarchical [`ChunkVersions`](crate::storage::ChunkVersions)
//! interval index, pruning entire clean subtrees. A GPU-resident uploader (design §15) or
//! a push-reaction pass walks the returned [`DirtyChunk`]s to upload / react to
//! only the changed windows; the per-row ticks remain available to refine
//! within a flagged window.
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

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

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
    /// Number of matched archetypes considered by the pass.
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
    /// Number of sparse-derived chunk candidates after row-to-chunk deduplication.
    pub sparse_chunk_candidates: usize,
    /// Number of sparse writes/removals delivered from the exact change index.
    pub sparse_index_events: usize,
}

/// Dirty row windows plus measurable traversal counters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirtyChunkReport {
    /// Conservative, chunk-aligned row windows.
    pub chunks: Vec<DirtyChunk>,
    /// Metadata work and output volume for this pass.
    pub stats: DirtyChunkStats,
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
    /// Collect the chunk windows this query touches whose change-version falls in
    /// the observer window `(last_run, this_run]` (design §7/§10).
    ///
    /// A chunk is reported when *any* component this query accesses (its read or
    /// write set) has a chunk-version newer than `last_run` for that window.
    /// This is the coarse layer: the returned windows form a superset of the
    /// rows a per-row [`Changed<T>`](crate::query::Changed) scan would visit (see
    /// the module docs). The matched-archetype cache is refreshed first, so only
    /// archetypes this query actually matches are inspected.
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
        let archetypes = self.matched_archetypes(world);
        let mut report = DirtyChunkReport {
            chunks: Vec::new(),
            stats: DirtyChunkStats {
                matched_archetypes: archetypes.len(),
                ..DirtyChunkStats::default()
            },
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
        let mut indexed_entities = Vec::new();
        self.collect_sparse_changes(last_run, this_run, &mut indexed_entities);
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
        for rows in sparse_chunks.values_mut() {
            rows.sort_unstable();
            rows.dedup();
        }

        for archetype_id in archetypes {
            let Some(archetype) = world.archetypes().get(archetype_id) else {
                continue;
            };
            let table = archetype.table();
            let rpc = table.rows_per_chunk();
            let len = table.len();

            // Each source is already sorted by the hierarchical visitor. Keep
            // one cursor per source and merge their heads below. This is
            // O(sum(source candidates) + distinct candidates * column count),
            // rather than O(total candidates * log(total candidates)).
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

            let mut cursors = vec![0usize; candidates.len()];
            let chunk_count = len.div_ceil(rpc);
            loop {
                let next = candidates
                    .iter()
                    .enumerate()
                    .filter_map(|(index, chunks)| chunks.get(cursors[index]).copied())
                    .min();
                let Some(chunk) = next else {
                    break;
                };

                // Advance every source currently pointing at the merged
                // chunk. Since each source is sorted, this also deduplicates
                // repeated access to the same column or chunk.
                for (index, chunks) in candidates.iter().enumerate() {
                    if chunks.get(cursors[index]).copied() == Some(chunk) {
                        cursors[index] += 1;
                    }
                }

                report.stats.merged_candidates += 1;
                if chunk >= chunk_count || !F::filter_chunk(filter_fetch, chunk) {
                    report.stats.filter_rejected += 1;
                    continue;
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
            }
        }
        report
    }
}
