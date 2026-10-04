//! Chunk-parallel query iteration (design §7 "par_iter：按 chunk 切分投递到
//! `prism_tasks`", §8.3 fiber 作业图 system 内并行).
//!
//! [`QueryState::par_for_each`](crate::query::QueryState::par_for_each) and
//! [`QueryState::par_for_each_mut`](crate::query::QueryState::par_for_each_mut)
//! split a query's matched rows into fixed-size *row batches* (the logical
//! chunk unit of design §5.3 / §10) and dispatch them onto a
//! [`prism_tasks::TaskPool`] work-stealing scope. Each batch is a disjoint
//! half-open row range `[start, end)` of exactly one archetype, so the `&mut T`
//! references two different tasks form can never alias the same column slot —
//! the same per-row fetch discipline as the serial
//! [`QueryIter`](crate::query::QueryIter), just partitioned across threads.
//!
//! # Soundness model
//!
//! * **Disjoint writes.** Batches never overlap: row ranges within one
//!   archetype are contiguous and non-overlapping, and distinct archetypes own
//!   distinct storage. A given `(archetype, row)` is therefore visited by
//!   exactly one task, so a `&mut` term yields a unique reference just as in the
//!   serial iterator.
//! * **Shared world view.** Every task reads the world through a shared
//!   `&World` reconstructed from one raw pointer. `&mut T` terms write through
//!   the interior-mutability column helpers on
//!   [`Column`](crate::storage::Column) (never through a `&mut World`), so no
//!   `&mut World` aliasing is created. The exclusive `&mut World` the caller
//!   passed to `par_for_each_mut` guarantees nothing else touches the world for
//!   the duration of the dispatch; [`TaskPool::scope`] joins every task before
//!   returning, so all borrows stay live for the whole pass.
//! * **`Send`.** The only non-`Send` capture is the world pointer, wrapped in
//!   [`SendWorldPtr`] whose `Send`/`Sync` impls are justified by the disjoint
//!   access above. The resolved query state (`D::State`/`F::State`) is
//!   `Send + Sync` by its trait bound, and the user closure is required to be
//!   `Send + Sync`.
//!
//! Enabled by the `multi_thread` feature.

use alloc::vec::Vec;

use prism_tasks::TaskPool;

use crate::archetype::ArchetypeId;
use crate::change::Tick;
use crate::query::fetch::QueryData;
use crate::query::filter::QueryFilter;
use crate::world::World;

/// A raw world pointer that is safe to move across the task boundary *under the
/// disjoint-access contract of this module*.
///
/// It is `Copy` so each spawned batch task can capture its own copy.
#[derive(Clone, Copy)]
struct SendWorldPtr(*mut World);

// SAFETY: the pointer is only ever dereferenced to form a shared `&World`
// (reads) and to write through the interior-mutability column helpers on
// disjoint rows. The caller of the driver guarantees exclusive world access for
// the whole dispatch (`&mut World` for the mutable entry point, or the
// `ReadOnlyQueryData` bound for the shared one), and the batch partition makes
// every task's touched rows disjoint, so sharing the pointer across threads
// introduces no data race.
unsafe impl Send for SendWorldPtr {}
// SAFETY: see the `Send` justification; `&SendWorldPtr` grants no capability
// beyond the `Copy` of the inner pointer, which is itself `Send` here.
unsafe impl Sync for SendWorldPtr {}

impl SendWorldPtr {
    /// Extract the inner pointer. Taking `self` by value forces a closure that
    /// calls this to capture the whole (`Send`) [`SendWorldPtr`] rather than the
    /// bare `*mut World` field (edition-2024 disjoint closure captures).
    #[inline]
    fn get(self) -> *mut World {
        self.0
    }
}

/// One unit of parallel work: a half-open row range of a single archetype.
#[derive(Clone, Copy)]
struct Batch {
    archetype: ArchetypeId,
    start: usize,
    end: usize,
}

/// Partition the matched archetypes into row batches of at most `batch_size`.
fn build_batches(world: &World, archetypes: &[ArchetypeId], batch_size: usize) -> Vec<Batch> {
    let batch_size = batch_size.max(1);
    let mut batches = Vec::new();
    for &archetype in archetypes {
        let len = world
            .archetypes()
            .get(archetype)
            .expect("matched archetype id must resolve")
            .len();
        let mut start = 0;
        while start < len {
            let end = (start + batch_size).min(len);
            batches.push(Batch {
                archetype,
                start,
                end,
            });
            start = end;
        }
    }
    batches
}

/// Run `func` over the rows of a single batch, applying the full filter/data
/// acceptance gates (mirrors the body of [`QueryIter::next`]).
///
/// # Safety
/// - `world` must be valid for reads for the duration of the call, and the
///   batch's rows must not be touched by any other concurrent access (upheld by
///   the disjoint batch partition plus the caller's exclusive-world contract).
/// - `batch.archetype` must satisfy `D::matches`/`F::matches` for the states,
///   and `batch.end <= archetype.len()`.
unsafe fn run_batch<D, F, Func>(
    world: *mut World,
    data_state: &D::State,
    filter_state: &F::State,
    batch: Batch,
    last_run: Tick,
    this_run: Tick,
    func: &Func,
) where
    D: QueryData,
    F: QueryFilter,
    Func: Fn(D::Item<'_>),
{
    // SAFETY: `world` is valid for reads for this call (caller contract); we
    // only form a shared `&World` to read archetype storage. `&mut` data terms
    // write through interior-mutability column helpers, never through this ref.
    let world: &World = unsafe { &*world };
    let archetype = world
        .archetypes()
        .get(batch.archetype)
        .expect("batch archetype id must resolve");
    let sparse_sets = world.sparse_sets();
    let entities = archetype.table().entities();

    // SAFETY: `batch.archetype` satisfies `D::matches`/`F::matches` for the
    // respective states (caller contract) — the precondition of `init_fetch`.
    let fetch = unsafe { D::init_fetch(data_state, archetype, sparse_sets, last_run, this_run) };
    // SAFETY: same archetype/state match contract for the filter terms.
    let filter_fetch =
        unsafe { F::init_fetch(filter_state, archetype, sparse_sets, last_run, this_run) };

    for (offset, &entity) in entities[batch.start..batch.end].iter().enumerate() {
        let row = batch.start + offset;
        // SAFETY: `row < batch.end <= archetype.len()` and `filter_fetch` was
        // built for this archetype; `F::Fetch` is `Copy`.
        if !unsafe { F::filter_fetch(filter_fetch, entity, row) } {
            continue;
        }
        // SAFETY: same row/archetype invariants; resolves sparse membership for
        // sparse-backed required terms. `D::Fetch` is `Copy`.
        if !unsafe { D::filter_fetch(fetch, entity, row) } {
            continue;
        }
        // SAFETY: `row` lies in this batch's exclusive range, `fetch` was built
        // for this archetype, and no other task visits this `(archetype, row)`,
        // so any `&mut` term yields a unique reference.
        func(unsafe { D::fetch(fetch, entity, row) });
    }
}

/// Dispatch `func` over every matched row in parallel.
///
/// # Safety
/// - `world` must stay valid for the whole call and must not be aliased by any
///   other access that conflicts with `D`'s terms (upheld by the exclusive
///   `&mut World` of the mutable entry point, or the `ReadOnlyQueryData` bound
///   of the shared one).
/// - every id in `archetypes` must satisfy `D::matches`/`F::matches`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn par_for_each_raw<D, F, Func>(
    world: *mut World,
    data_state: &D::State,
    filter_state: &F::State,
    archetypes: &[ArchetypeId],
    last_run: Tick,
    this_run: Tick,
    pool: &TaskPool,
    batch_size: usize,
    func: &Func,
) where
    D: QueryData,
    F: QueryFilter,
    Func: Fn(D::Item<'_>) + Send + Sync,
{
    // SAFETY: forming a shared `&World` to size the batches is sound under the
    // caller's validity contract.
    let batches = build_batches(unsafe { &*world }, archetypes, batch_size);
    if batches.is_empty() {
        return;
    }

    let send_world = SendWorldPtr(world);
    pool.scope(|scope| {
        for batch in &batches {
            let batch = *batch;
            scope.spawn(move || {
                // SAFETY: `send_world.0` is valid for the scope (joined before
                // the enclosing `&mut World`/`&World` borrow ends); `batch` is a
                // disjoint row range of a matched archetype, so this task's
                // accesses do not alias any sibling task's.
                unsafe {
                    run_batch::<D, F, Func>(
                        send_world.get(),
                        data_state,
                        filter_state,
                        batch,
                        last_run,
                        this_run,
                        func,
                    );
                }
            });
        }
    });
}
