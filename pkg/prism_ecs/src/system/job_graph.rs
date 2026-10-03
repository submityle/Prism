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
//! fn skinning(mut q: Query<(&mut SkinnedMesh, &Pose)>, jobs: JobGraph) {
//!     jobs.par_for_each_mut(&mut q, 256, |(mesh, pose)| { /* SIMD 蒙皮 */ });
//! }
//! ```
//!
//! The pool itself is **not** owned by the ECS: per design §24.1 the kernel does
//! not build its own thread pool but dispatches onto `prism_tasks`. The pool is
//! stored in the world as the [`ComputeTaskPool`] resource; `JobGraph` is a
//! [`SystemParam`] that reads it. A system that uses `JobGraph` therefore
//! requires `ComputeTaskPool` to be inserted into the world first.
//!
//! # Granularity: rows now, slices later
//!
//! §8.3 sketches `jobs.par_chunks(&mut q, |chunk| ...)` handing each sub-job a
//! whole chunk/slice so the body can run SIMD kernels over it. That slice-fetch
//! form needs a column-slice `QueryData` path that does not yet exist, so it is
//! an honest follow-up. What ships here is the **row-granular** form
//! ([`par_for_each`](JobGraph::par_for_each) /
//! [`par_for_each_mut`](JobGraph::par_for_each_mut)): rows are still partitioned
//! into disjoint batches and run in parallel across the pool — the same load
//! balancing — just invoking `func` once per row rather than once per slice.

use crate::query::{Access, QueryData, QueryFilter, ReadOnlyQueryData};
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
/// `par_for_each*` helpers, which forward to [`Query::par_for_each`] /
/// [`Query::par_for_each_mut`]. Declaring it as a resource read lets the
/// conflict-graph executor keep systems that only *use* the pool able to run
/// alongside each other.
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
