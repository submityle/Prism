//! Named thread categories (design §11).
//!
//! Some work must run somewhere other than a general compute worker: platform
//! callbacks on the main thread, GPU submission on one render thread, blocking
//! file/network IO off the compute pool, and low-priority background compute.
//! [`NamedThreads`] owns those lanes and [`NamedThreads::dispatch`] routes a job
//! to one, returning a [`Counter`] so the result bridges straight back into the
//! fork-join and async worlds ([`Counter::wait_async`](crate::Counter::wait_async)).
//!
//! | [`ThreadCategory`] | Lane | Runs on |
//! |---|---|---|
//! | [`Main`](ThreadCategory::Main) | main queue | the thread that pumps it |
//! | [`Render`](ThreadCategory::Render) | 1 dedicated thread | GPU submit |
//! | [`Io`](ThreadCategory::Io) | blocking pool | file/network waits |
//! | [`AsyncCompute`](ThreadCategory::AsyncCompute) | blocking pool | background compute |
//!
//! ## Coherence with the scheduler
//! The Render / IO / `AsyncCompute` lanes are *independent* OS threads, kept off
//! the compute worker pool so blocking work there never starves (or is starved
//! by) the job graph. `Main` has no owned thread: it is a queue the application
//! drains with [`NamedThreads::run_main_pending`], respecting the existing
//! "the main thread is also a compute worker" model — between pumps the main
//! thread can help the pool via [`TaskPool::wait`](crate::TaskPool::wait).
//!
//! ## Deadlock-freedom
//! [`NamedThreads::dispatch`] only enqueues (it never blocks), and each lane
//! runs jobs to completion independently, so dispatching from any thread — a
//! compute worker, a fiber, or another lane — cannot deadlock. Waiting for a
//! dispatched job uses a [`Counter`], which composes with help-on-wait and the
//! async bridge exactly like any other job.

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::thread::JoinHandle;

use crate::job::Job;
use crate::Counter;

/// A named execution lane with specific threading guarantees.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ThreadCategory {
    /// Window events, platform callbacks, and APIs that must run on the main
    /// thread. Jobs queue until the main thread calls
    /// [`NamedThreads::run_main_pending`].
    Main,
    /// GPU command recording / submission, which most backends require to be
    /// single-threaded; backed by exactly one dedicated thread.
    Render,
    /// Blocking file / network IO, kept off the compute pool so a blocked
    /// syscall never ties up a compute worker.
    Io,
    /// Low-priority background compute (streaming pre-process, baking).
    AsyncCompute,
}

/// Configuration for the lanes a [`NamedThreads`] owns.
#[derive(Clone, Copy, Debug)]
pub struct NamedThreadsConfig {
    /// Worker threads backing the [`Io`](ThreadCategory::Io) lane (at least 1).
    pub io_threads: usize,
    /// Worker threads backing the
    /// [`AsyncCompute`](ThreadCategory::AsyncCompute) lane (at least 1).
    pub async_compute_threads: usize,
}

impl Default for NamedThreadsConfig {
    fn default() -> Self {
        Self {
            io_threads: 2,
            async_compute_threads: 2,
        }
    }
}

/// Owns the named thread lanes. Cheap to clone (clones share the same lanes);
/// the lanes shut down and join when the last clone drops.
#[derive(Clone)]
pub struct NamedThreads {
    inner: Arc<Inner>,
}

struct Inner {
    main: Arc<Lane>,
    render: BlockingLane,
    io: BlockingLane,
    async_compute: BlockingLane,
}

impl NamedThreads {
    /// Build the lanes with the default configuration (1 render thread, plus
    /// the IO and `AsyncCompute` blocking pools from [`NamedThreadsConfig`]).
    pub fn new() -> Self {
        Self::with_config(NamedThreadsConfig::default())
    }

    /// Build the lanes from an explicit configuration.
    pub fn with_config(config: NamedThreadsConfig) -> Self {
        Self {
            inner: Arc::new(Inner {
                main: Arc::new(Lane::new()),
                render: BlockingLane::new("prism-render", 1),
                io: BlockingLane::new("prism-io", config.io_threads.max(1)),
                async_compute: BlockingLane::new(
                    "prism-async-compute",
                    config.async_compute_threads.max(1),
                ),
            }),
        }
    }

    /// Dispatch `f` to `category`, returning a [`Counter`] that drains when the
    /// job finishes. Wait on it with [`TaskPool::wait`](crate::TaskPool::wait)
    /// or `counter.wait_async().await`.
    pub fn dispatch<F>(&self, category: ThreadCategory, f: F) -> Counter
    where
        F: FnOnce() + Send + 'static,
    {
        let counter = Counter::new();
        counter.add(1);
        let tracked = counter.clone();
        let job: Job = Box::new(move || {
            f();
            tracked.finish_one();
        });
        match category {
            ThreadCategory::Main => self.inner.main.push(job),
            ThreadCategory::Render => self.inner.render.push(job),
            ThreadCategory::Io => self.inner.io.push(job),
            ThreadCategory::AsyncCompute => self.inner.async_compute.push(job),
        }
        counter
    }

    /// Run every job currently queued for [`Main`](ThreadCategory::Main) on the
    /// calling thread, then return. The application's main thread calls this
    /// (e.g. once per frame); jobs enqueued while it runs wait for the next
    /// pump. Returns the number of jobs executed.
    pub fn run_main_pending(&self) -> usize {
        self.inner.main.drain_and_run()
    }

    /// Number of jobs waiting in the [`Main`](ThreadCategory::Main) queue.
    pub fn main_pending(&self) -> usize {
        self.inner.main.len()
    }
}

impl Default for NamedThreads {
    fn default() -> Self {
        Self::new()
    }
}

/// A plain job queue with no owned thread (used for the Main lane).
struct Lane {
    queue: Mutex<VecDeque<Job>>,
}

impl Lane {
    fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
        }
    }

    fn push(&self, job: Job) {
        self.queue.lock().unwrap().push_back(job);
    }

    fn len(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Run exactly the jobs queued at entry (not any enqueued during the pump),
    /// so a self-enqueueing job cannot spin this call forever.
    fn drain_and_run(&self) -> usize {
        let batch: Vec<Job> = {
            let mut queue = self.queue.lock().unwrap();
            queue.drain(..).collect()
        };
        let count = batch.len();
        for job in batch {
            job();
        }
        count
    }
}

/// A blocking job queue served by one or more dedicated OS threads.
struct BlockingLane {
    shared: Arc<BlockingShared>,
    handles: Vec<JoinHandle<()>>,
}

struct BlockingShared {
    queue: Mutex<VecDeque<Job>>,
    cvar: Condvar,
    shutdown: AtomicBool,
}

impl BlockingLane {
    fn new(name: &str, threads: usize) -> Self {
        let shared = Arc::new(BlockingShared {
            queue: Mutex::new(VecDeque::new()),
            cvar: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });
        let mut handles = Vec::with_capacity(threads);
        for index in 0..threads {
            let shared = Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name(format!("{name}-{index}"))
                .spawn(move || shared.run())
                .expect("failed to spawn named-lane thread");
            handles.push(handle);
        }
        Self { shared, handles }
    }

    fn push(&self, job: Job) {
        self.shared.queue.lock().unwrap().push_back(job);
        self.shared.cvar.notify_one();
    }
}

impl BlockingShared {
    /// Lane worker loop: run queued jobs; park on the condvar when idle; drain
    /// everything still queued at shutdown before exiting.
    fn run(&self) {
        loop {
            let job = {
                let mut queue = self.queue.lock().unwrap();
                loop {
                    if let Some(job) = queue.pop_front() {
                        break Some(job);
                    }
                    if self.shutdown.load(Ordering::Acquire) {
                        break None;
                    }
                    queue = self.cvar.wait(queue).unwrap();
                }
            };
            match job {
                Some(job) => job(),
                None => break,
            }
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        for lane in [&mut self.render, &mut self.io, &mut self.async_compute] {
            lane.shared.shutdown.store(true, Ordering::Release);
            lane.shared.cvar.notify_all();
        }
        for lane in [&mut self.render, &mut self.io, &mut self.async_compute] {
            for handle in lane.handles.drain(..) {
                let _ = handle.join();
            }
        }
    }
}
