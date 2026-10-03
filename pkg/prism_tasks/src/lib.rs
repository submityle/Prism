//! # prism_tasks
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
//! Later milestones add structured parallelism (`scope`/`parallel_for`),
//! fibers (wait-without-blocking-a-worker), an async executor, named threads,
//! NUMA/affinity, and deterministic replay.
//!
//! The crate contains no Unreal Engine source or derived code and depends on
//! no `bevy_*` crate.

#![forbid(unsafe_code)]

mod counter;
mod job;
mod scheduler;

use std::sync::Arc;
use std::thread::JoinHandle;

pub use counter::Counter;
use scheduler::Shared;

/// Configuration for a [`TaskPool`].
#[derive(Clone, Copy, Debug)]
pub struct TaskPoolConfig {
    /// Number of worker threads. `0` selects the synchronous fallback.
    pub threads: usize,
}

impl Default for TaskPoolConfig {
    fn default() -> Self {
        Self {
            threads: default_thread_count(),
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
        Self::with_config(TaskPoolConfig { threads })
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

        let shared = Arc::new(Shared::new(config.threads));
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
