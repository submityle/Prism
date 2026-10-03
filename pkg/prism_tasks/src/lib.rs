//! # `prism_tasks`
//!
//! Prism's parallel-runtime kernel: a single work-stealing thread pool that
//! every subsystem feeds fine-grained jobs into.
//!
//! ## M0 scope (this build) — the thread pool
//! - [`Counter`]: atomic fork-join dependency/wait handle.
//! - [`TaskPool`]: worker threads + per-worker deques + a global injector,
//!   with [`TaskPool::spawn`], [`TaskPool::wait`] (help-on-wait, no deadlock),
//!   and [`TaskPool::join`] for recursive fork-join.
//! - A single-threaded synchronous fallback (the `single` feature, or a pool
//!   built with zero threads) that runs jobs inline on spawn.
//!
//! ## M1 scope (this build) — structured parallelism
//! Built on the M0 pool and its help-on-wait join barrier:
//! - [`TaskPool::scope`] / [`Scope`]: structured scopes that spawn *borrowed*
//!   tasks and join them all before returning.
//! - [`TaskPool::parallel_for`] / [`TaskPool::par_for_each`] /
//!   [`TaskPool::par_for_each_mut`] / chunked variants: adaptive-grain data
//!   parallelism.
//! - [`TaskPool::reduce`]: parallel reduction with a deterministic tree-shaped
//!   combine order.
//! - [`TaskPool::prefix_sum`]: parallel inclusive scan.
//!
//! ## M3 scope (this build) — async + named threads
//! - [`TaskPool::spawn_async`] / [`TaskPool::block_on`] / [`Task`]: a minimal
//!   `Future` executor that polls futures as jobs on the same pool, with a
//!   hand-rolled [`RawWaker`](std::task::RawWaker) vtable.
//! - [`Counter::wait_async`] / [`CounterFuture`]: bridge counter completion to
//!   future wakeups (and, via [`Task::counter`], async results back to jobs).
//! - [`NamedThreads`] / [`ThreadCategory`]: Main / Render / IO / `AsyncCompute`
//!   lanes dispatched independently of the compute pool.
//!
//! Later milestones add NUMA/affinity and deterministic replay.
//!
//! The crate contains no Unreal Engine source or derived code and depends on
//! no `bevy_*` crate.

// This crate is safe apart from two audited areas, each guarded by a local
// `#[expect(unsafe_code, reason = ...)]` (the workspace denies `unsafe_code`):
// the lifetime erasure in `scope` (see `scope.rs` for its soundness argument),
// and the stackful-fiber context switching behind the off-by-default `fibers`
// feature (see the `fiber` module, especially `fiber::context`).

extern crate alloc;

mod affinity;
mod arena;
mod async_exec;
mod counter;
#[cfg(feature = "fibers")]
mod fiber;
mod job;
mod named;
mod numa;
mod parallel;
mod scheduler;
mod scope;

use alloc::sync::Arc;
use std::thread::JoinHandle;

pub use affinity::{
    affinity_supported, pin_current_thread_to_core, plan_worker_cores, CoreAssignment,
    CoreClassPolicy, WorkerCorePlan,
};
pub use arena::{FrameArena, FrameArenas, DEFAULT_ARENA_CAPACITY};
pub use async_exec::{CounterFuture, Task};
pub use counter::Counter;
pub use numa::{
    steal_order, steal_penalty, CoreClass, CoreInfo, NumaNodeId, Topology,
    CROSS_NODE_STEAL_PENALTY,
};
pub use prism_platform::AffinityError;
pub use named::{NamedThreads, NamedThreadsConfig, ThreadCategory};
pub use scope::Scope;
use scheduler::Shared;

/// Configuration for a [`TaskPool`].
#[derive(Clone, Copy, Debug)]
pub struct TaskPoolConfig {
    /// Number of worker threads. `0` selects the synchronous fallback.
    pub threads: usize,
    /// M4: pin each worker to a fixed OS core for cache/NUMA locality. Default
    /// `false` — pinning is best-effort and a no-op on platforms that cannot
    /// pin (e.g. macOS). See [`crate::affinity`].
    pub pin_workers: bool,
    /// M4: how to spread workers across hybrid (big.LITTLE) core classes when
    /// pinning. Ignored when `pin_workers` is `false`.
    pub core_class_policy: CoreClassPolicy,
}

impl Default for TaskPoolConfig {
    fn default() -> Self {
        Self {
            threads: default_thread_count(),
            pin_workers: false,
            core_class_policy: CoreClassPolicy::PerformanceFirst,
        }
    }
}

/// Default worker count: logical cores minus one (reserving the calling thread
/// which also helps during `wait`), at least one.
fn default_thread_count() -> usize {
    let cores = prism_platform::CpuInfo::detect().logical_cores;
    cores.saturating_sub(1).max(1)
}

/// A work-stealing thread pool. Cheap to clone (shares the same workers).
#[derive(Clone)]
pub struct TaskPool {
    shared: Arc<Shared>,
    handles: Arc<Vec<JoinHandle<()>>>,
    single: bool,
}

impl TaskPool {
    /// Build a pool with the default configuration.
    pub fn new() -> Self {
        Self::with_config(TaskPoolConfig::default())
    }

    /// Build a pool with an explicit worker-thread count.
    pub fn with_threads(threads: usize) -> Self {
        Self::with_config(TaskPoolConfig {
            threads,
            ..TaskPoolConfig::default()
        })
    }

    /// Build a pool from a full configuration.
    pub fn with_config(config: TaskPoolConfig) -> Self {
        let single = cfg!(feature = "single") || config.threads == 0;
        if single {
            return Self {
                shared: Arc::new(Shared::new(0)),
                handles: Arc::new(Vec::new()),
                single: true,
            };
        }

        // M4: when pinning is requested, resolve the machine topology (honest
        // single-node fallback where the platform exposes none) and build a
        // deterministic worker->core plan the workers pin themselves with.
        let affinity_plan = if config.pin_workers {
            let topology = Topology::detect();
            Some(plan_worker_cores(
                &topology,
                config.threads,
                config.core_class_policy,
            ))
        } else {
            None
        };
        let shared = Arc::new(Shared::with_affinity(config.threads, affinity_plan));
        let pool_id = shared.id();
        let mut handles = Vec::with_capacity(config.threads);
        for index in 0..config.threads {
            let shared = Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name(format!("prism-worker-{index}"))
                .spawn(move || {
                    shared.run_worker(pool_id, index);
                })
                .expect("failed to spawn worker thread");
            handles.push(handle);
        }

        Self {
            shared,
            handles: Arc::new(handles),
            single: false,
        }
    }

    /// Number of worker threads (`0` in the synchronous fallback).
    pub fn worker_count(&self) -> usize {
        if self.single {
            0
        } else {
            self.shared.num_workers()
        }
    }

    /// Whether this pool runs jobs inline (synchronous fallback).
    pub fn is_single_threaded(&self) -> bool {
        self.single
    }

    /// Spawn `f`, tracking it on `counter`. In the synchronous fallback the job
    /// runs inline before returning.
    pub fn spawn<F>(&self, counter: &Counter, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        counter.add(1);
        let counter = counter.clone();
        let job = move || {
            f();
            counter.finish_one();
        };
        if self.single {
            job();
        } else {
            self.shared.push(Box::new(job));
        }
    }

    /// Block until every job tracked by `counter` has completed. The calling
    /// thread helps run queued jobs while waiting, so nested fork-join cannot
    /// deadlock even when it is deeper than the worker count.
    pub fn wait(&self, counter: &Counter) {
        if self.single {
            debug_assert!(counter.is_complete());
            return;
        }
        // With fibers on, a job that waits suspends its fiber (yielding the OS
        // worker) instead of busy-helping. Only the running fiber takes this
        // path; external/top-level waiters fall through to help-on-wait, which
        // in the fiber build drives fibers rather than running jobs inline.
        #[cfg(feature = "fibers")]
        {
            if fiber::on_fiber() {
                fiber::suspend_current(counter);
                return;
            }
        }
        self.shared.help_until(|| counter.is_complete());
    }

    /// Run two closures that may execute in parallel, returning both results.
    /// This is the canonical fork-join primitive: `b` is spawned on the pool
    /// and `a` runs on the calling thread, which then helps until `b` finishes.
    pub fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send + 'static,
        B: FnOnce() -> RB + Send + 'static,
        RA: Send + 'static,
        RB: Send + 'static,
    {
        if self.single {
            return (a(), b());
        }
        let counter = Counter::new();
        let rb_slot: Arc<std::sync::Mutex<Option<RB>>> = Arc::new(std::sync::Mutex::new(None));
        let rb_write = Arc::clone(&rb_slot);
        self.spawn(&counter, move || {
            *rb_write.lock().unwrap() = Some(b());
        });
        let ra = a();
        self.wait(&counter);
        let rb = rb_slot.lock().unwrap().take().expect("b did not complete");
        (ra, rb)
    }

    /// The index of the current worker within this pool, if the calling thread
    /// is one of its workers. Use it to pick this worker's per-worker frame
    /// arena from a [`FrameArenas`] (M4). Returns `None` on an external thread
    /// or the single-threaded fallback.
    pub fn current_worker_index(&self) -> Option<usize> {
        if self.single {
            None
        } else {
            self.shared.current_worker_index()
        }
    }

    /// Allocate a [`FrameArenas`] sized to this pool — one arena per worker,
    /// each placed on the NUMA node its worker is pinned to (node 0 when the
    /// pool is unpinned or the platform has no NUMA map). `capacity` is the
    /// per-worker arena size in bytes; pass [`DEFAULT_ARENA_CAPACITY`] for the
    /// default. The returned arenas are reset each frame via
    /// [`FrameArenas::reset_all`] (M4, design §12).
    pub fn new_frame_arenas(&self, capacity: usize) -> FrameArenas {
        let workers = self.worker_count().max(1);
        match self.shared.affinity_plan() {
            Some(plan) => {
                FrameArenas::with_nodes(workers, capacity, |w| plan.node_of_worker(w))
            }
            None => FrameArenas::new(workers, capacity),
        }
    }

    /// Whether this build can pin workers to cores (false on macOS and other
    /// unsupported platforms). Forwarded from [`crate::affinity`].
    pub fn affinity_supported(&self) -> bool {
        affinity_supported()
    }

    /// Peak fiber-stack occupancy per size class since this pool was created
    /// (design §16 "fiber 栈占用峰值"). Use it to size the pool and to alarm on
    /// the §23-risk-6 "ran out of stacks" condition. Only present with the
    /// `fibers` feature; the synchronous fallback reports zeros since it runs
    /// no fibers.
    #[cfg(feature = "fibers")]
    pub fn fiber_stack_high_water(&self) -> FiberStackStats {
        use crate::fiber::stack::StackClass;
        if self.single {
            return FiberStackStats {
                small_high_water: 0,
                large_high_water: 0,
            };
        }
        let pool = self.shared.stack_pool();
        FiberStackStats {
            small_high_water: pool.high_water(StackClass::Small),
            large_high_water: pool.high_water(StackClass::Large),
        }
    }

    /// Push an already-boxed job directly onto the pool, waking a worker. Used
    /// by the structured-parallelism scope, which manages its own counter.
    pub(crate) fn push_job(&self, job: job::Job) {
        self.shared.push(job);
    }

    /// Shared scheduler state, used by the async executor to enqueue task polls.
    pub(crate) fn shared_state(&self) -> &Arc<Shared> {
        &self.shared
    }
}

/// Peak fiber-stack occupancy per size class, from
/// [`TaskPool::fiber_stack_high_water`] (design §16). Only built with the
/// `fibers` feature.
#[cfg(feature = "fibers")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FiberStackStats {
    /// Peak number of concurrently live small-class fiber stacks.
    pub small_high_water: usize,
    /// Peak number of concurrently live large-class fiber stacks.
    pub large_high_water: usize,
}

impl Default for TaskPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TaskPool {
    fn drop(&mut self) {
        // Only the last clone owns the join handles; when it drops, tell the
        // workers to drain and exit, then join them.
        if let Some(handles) = Arc::get_mut(&mut self.handles) {
            self.shared.begin_shutdown();
            for handle in handles.drain(..) {
                let _ = handle.join();
            }
        }
    }
}

#[cfg(test)]
mod tests;
