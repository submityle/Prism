//! The dirty-chunk accessor: the coarse, chunk-granular half of double-layer
//! change detection (design §7 脏块访问器, §10).
//!
//! Where [`Changed<T>`](crate::query::Changed) filters row-by-row, the
//! dirty-chunk accessor answers the *coarse* question "which row windows of
//! which archetypes contain anything this query touched since `last_run`?" by
//! consulting one [`ChunkVersions`](crate::storage::ChunkVersions) tick per
//! chunk instead of scanning every row. A GPU-resident uploader (design §15) or
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

use alloc::vec::Vec;

use crate::archetype::ArchetypeId;
use crate::change::Tick;
use crate::query::fetch::QueryData;
use crate::query::filter::QueryFilter;
use crate::query::state::QueryState;
use crate::world::World;

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
        let mut dirty = Vec::new();
        // Components whose chunk-version signals a change relevant to this query:
        // everything it reads plus everything it writes.
        let access = self.access();
        for archetype_id in self.matched_archetypes(world) {
            let Some(archetype) = world.archetypes().get(archetype_id) else {
                continue;
            };
            let table = archetype.table();
            let rpc = table.rows_per_chunk();
            let len = table.len();
            for chunk in 0..table.chunk_count() {
                let is_dirty = access
                    .reads()
                    .iter()
                    .chain(access.writes().iter())
                    .filter_map(|&id| table.column(id))
                    .any(|col| col.chunk_version(chunk).is_newer_than(last_run, this_run));
                if is_dirty {
                    let start = chunk * rpc;
                    let end = ((chunk + 1) * rpc).min(len);
                    dirty.push(DirtyChunk {
                        archetype: archetype_id,
                        start,
                        end,
                    });
                }
            }
        }
        dirty
    }
}
