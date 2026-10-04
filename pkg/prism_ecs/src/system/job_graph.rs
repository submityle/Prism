//! [`JobGraph`]: the system-internal parallelism param (design §8.3).
//!
//! The design doc's §8.3 ("Fiber 作业图") splits one heavy system's work into
//! chunk-granular sub-jobs that run on the shared task pool while the
//! conflict-graph executor (§8.2) schedules whole systems. This module provides
//! the ergonomic entry point for that: a system takes a [`JobGraph`] param
//! alongside its queries and calls
//! [`par_for_each`](JobGraph::par_for_each) /
//! [`par_for_each_mut`](JobGraph::par_for_each_mut) to fan a query's rows out
//! across the pool, e.g.
//!
//! ```ignore
//! fn skinning(mut q: Query<(&mut Joint, &Pose)>, jobs: JobGraph) {
//!     // one chunk-aligned slice per sub-job — SIMD over the whole run at once
//!     jobs.par_chunks_mut(&mut q, 256, |(joints, poses)| { /* SIMD 蒙皮 */ });
//! }
//! ```
//!
//! The pool itself is **not** owned by the ECS: per design §24.1 the kernel does
//! not build its own thread pool but dispatches onto `prism_tasks`. The pool is
//! stored in the world as the [`ComputeTaskPool`] resource; `JobGraph` is a
//! [`SystemParam`] that reads it. A system that uses `JobGraph` therefore
//! requires `ComputeTaskPool` to be inserted into the world first.
//!
//! # Granularity: rows and slices
//!
//! Two fan-out granularities ship, both over the same shared pool:
//!
//! * **Row-granular** ([`par_for_each`](JobGraph::par_for_each) /
//!   [`par_for_each_mut`](JobGraph::par_for_each_mut)): rows are partitioned
//!   into disjoint batches and `func` is invoked once per row — the general
//!   form that accepts any [`QueryData`]/[`QueryFilter`].
//! * **Slice-granular** ([`par_chunks`](JobGraph::par_chunks) /
//!   [`par_chunks_mut`](JobGraph::par_chunks_mut)): §8.3's headline
//!   `jobs.par_chunks(&mut q, |chunk| ...)`. Each sub-job receives one
//!   **chunk-aligned typed column slice** (`&[T]` / `&mut [T]`, or a tuple), so
//!   the body can run a DOTS-style `IJobChunk` SIMD kernel over the whole slice
//!   in a single pass. This form restricts the query to table-backed, sliceable
//!   data ([`ColumnSliceData`](crate::query::ColumnSliceData)) and an
//!   archetype-uniform filter ([`ArchetypalFilter`](crate::query::ArchetypalFilter));
//!   the per-row change filters and sparse/shared storage are rejected at
//!   compile time (see [`crate::query::slice`]).

use crate::query::{Access, ArchetypalFilter, ColumnSliceData, QueryData, QueryFilter, ReadOnlyQueryData};
use crate::system::job_dag::JobDag;
use crate::resource::{Resource, ResourceId};
use crate::system::param::SystemParam;
use crate::system::query_param::Query;
use crate::system::world_cell::UnsafeWorldCell;
use crate::world::World;

use prism_tasks::TaskPool;

/// The world-global [`prism_tasks::TaskPool`] that ECS system-internal
/// parallelism ([`JobGraph`]) and the parallel executor dispatch onto.
///
/// Per design §24.1 the ECS does not own a thread pool; it reuses this shared
/// pool. Insert one into the world (`world.insert_resource(ComputeTaskPool(pool))`)
/// before running any system that takes a [`JobGraph`] param.
pub struct ComputeTaskPool(
    /// The wrapped work-stealing pool.
    pub TaskPool,
);

// SAFETY: `ComputeTaskPool` is a marker-only resource; `TaskPool` is an
// `Arc`-backed handle that is `Send + Sync + 'static`, so the resource store's
// thread-safety requirements are met.
impl Resource for ComputeTaskPool {}

impl ComputeTaskPool {
    /// Borrow the wrapped pool.
    #[inline]
    pub fn pool(&self) -> &TaskPool {
        &self.0
    }
}

/// System param granting system-internal, chunk-parallel dispatch over the
/// world's [`ComputeTaskPool`] (design §8.3).
///
/// It reads the [`ComputeTaskPool`] resource and exposes it through the
/// `par_for_each*` (row-granular) and `par_chunks*` (slice-granular) helpers,
/// which forward to the matching [`Query`] methods. Declaring it as a resource
/// read lets the conflict-graph executor keep systems that only *use* the pool
/// able to run alongside each other.
pub struct JobGraph<'w> {
    pool: &'w TaskPool,
}

impl<'w> JobGraph<'w> {
    /// The default per-batch row count used by the `*_auto` helpers when the
    /// caller does not pick one. A middling grain that amortises dispatch
    /// overhead without starving the pool on small worlds.
    pub const DEFAULT_BATCH_SIZE: usize = 256;

    /// Borrow the underlying task pool, e.g. to open a manual
    /// [`scope`](prism_tasks::TaskPool::scope) for bespoke sub-jobs.
    #[inline]
    pub fn pool(&self) -> &TaskPool {
        self.pool
    }

    /// Build and run a heterogeneous sub-job **dependency DAG** over the shared
    /// pool (design §8.3 "fiber 作业图：原子计数依赖 + 工作窃取").
    ///
    /// Unlike the `par_for_each*` / `par_chunks*` helpers, which fan **one**
    /// query's rows out homogeneously, this splits a heavy system into named,
    /// heterogeneous sub-jobs wired by dependency edges. Inside `build`, add
    /// nodes with [`JobDag::add`] / [`JobDag::add_after`]; each node is
    /// dispatched the instant its last predecessor finishes (dataflow
    /// scheduling, no false phase barriers), and the call returns once every
    /// node has run.
    ///
    /// ```ignore
    /// fn step(mut q: Query<&mut Body>, jobs: JobGraph) {
    ///     let shared = SharedState::default();
    ///     jobs.dag(|dag| {
    ///         let broad = dag.add(|| shared.broadphase());
    ///         let integ = dag.add(|| shared.integrate());
    ///         dag.add_after(&[broad, integ], || shared.resolve());
    ///     });
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the declared edges contain a cycle, or re-raises a panic from
    /// any sub-job after the graph joins. See [`JobDag`].
    #[inline]
    pub fn dag<'env, B>(&self, build: B)
    where
        B: FnOnce(&mut JobDag<'env>),
    {
        let mut dag = JobDag::new();
        build(&mut dag);
        dag.dispatch(self.pool);
    }

    /// Fan `q`'s matched rows out across the pool with shared access, in
    /// disjoint batches of at most `batch_size` rows (clamped to `>= 1`).
    ///
    /// Available only for read-only queries (`D: `[`ReadOnlyQueryData`]). The
    /// call returns once every batch completes.
    #[inline]
    pub fn par_for_each<D, F, Func>(&self, q: &Query<'_, '_, D, F>, batch_size: usize, func: Func)
    where
        D: QueryData + ReadOnlyQueryData,
        F: QueryFilter,
        Func: Fn(D::Item<'_>) + Send + Sync,
    {
        q.par_for_each(self.pool, batch_size, func);
    }

    /// Fan `q`'s matched rows out across the pool with exclusive access,
    /// permitting `&mut T` terms, in disjoint batches of at most `batch_size`
    /// rows (clamped to `>= 1`).
    ///
    /// The disjoint batches guarantee the `&mut` references different threads
    /// form never alias the same column slot. The call returns once every batch
    /// completes.
    #[inline]
    pub fn par_for_each_mut<D, F, Func>(
        &self,
        q: &mut Query<'_, '_, D, F>,
        batch_size: usize,
        func: Func,
    ) where
        D: QueryData,
        F: QueryFilter,
        Func: Fn(D::Item<'_>) + Send + Sync,
    {
        q.par_for_each_mut(self.pool, batch_size, func);
    }

    /// Fan `q`'s matched rows out across the pool as **chunk-aligned typed
    /// column slices** with shared access (design §8.3 `jobs.par_chunks`),
    /// invoking `func` once per chunk-aligned batch with a read-only slice
    /// tuple.
    ///
    /// Available only for read-only (`D: `[`ReadOnlyQueryData`]), sliceable
    /// (`D: `[`ColumnSliceData`]) queries with an archetype-uniform filter
    /// (`F: `[`ArchetypalFilter`]). `batch_size` is the soft target row count,
    /// rounded down to a whole number of 16KiB chunks (at least one). The call
    /// returns once every batch completes.
    #[inline]
    pub fn par_chunks<D, F, Func>(&self, q: &Query<'_, '_, D, F>, batch_size: usize, func: Func)
    where
        D: QueryData + ReadOnlyQueryData + ColumnSliceData,
        F: ArchetypalFilter,
        Func: Fn(<D as ColumnSliceData>::Slice<'_>) + Send + Sync,
    {
        q.par_chunks(self.pool, batch_size, func);
    }

    /// Fan `q`'s matched rows out across the pool as **chunk-aligned typed
    /// column slices** with exclusive access, permitting `&mut [T]` slice terms
    /// (design §8.3 `jobs.par_chunks`, the SIMD-skinning headline).
    ///
    /// Batches are chunk-aligned and disjoint, so the `&mut [T]` slices
    /// different threads form never overlap; a `&mut T` term stamps the change
    /// ticks of its whole handed-out range up front. The call returns once every
    /// batch completes.
    #[inline]
    pub fn par_chunks_mut<D, F, Func>(
        &self,
        q: &mut Query<'_, '_, D, F>,
        batch_size: usize,
        func: Func,
    ) where
        D: QueryData + ColumnSliceData,
        F: ArchetypalFilter,
        Func: Fn(<D as ColumnSliceData>::Slice<'_>) + Send + Sync,
    {
        q.par_chunks_mut(self.pool, batch_size, func);
    }
}

// SAFETY: `update_access` declares a resource *read* of the `ComputeTaskPool`
// id minted in `init_state`; `get_param` only ever forms `&TaskPool` from the
// resource's heap box via `get_ptr`, so no `&mut` aliasing can arise.
unsafe impl SystemParam for JobGraph<'_> {
    type State = ResourceId;
    type Item<'w, 's> = JobGraph<'w>;

    #[inline]
    fn init_state(world: &mut World) -> ResourceId {
        world.resources_mut().register::<ComputeTaskPool>()
    }

    #[inline]
    fn update_access(state: &ResourceId, access: &mut Access) {
        access.add_resource_read(*state);
    }

    #[inline]
    unsafe fn get_param<'w, 's>(
        state: &'s mut ResourceId,
        world: UnsafeWorldCell<'w>,
    ) -> JobGraph<'w> {
        // SAFETY: the caller guarantees nothing aliases this read; we form only
        // a shared `&World` from the cell.
        let world: &'w World = unsafe { world.world() };
        // SAFETY: `state` is the id minted for `ComputeTaskPool` in
        // `init_state`; the declared read guarantees no `ResMut` aliases it.
        let ptr = unsafe { world.resources().get_ptr::<ComputeTaskPool>(*state) }
            .expect("JobGraph: ComputeTaskPool resource not present in world");
        // SAFETY: `ptr` points at the live `ComputeTaskPool` inside the resource
        // store, which outlives `'w`; our declared read means no `&mut` is live,
        // so promoting the wrapped pool to `&'w TaskPool` is sound.
        let pool = unsafe { &(*ptr).0 };
        JobGraph { pool }
    }
}
