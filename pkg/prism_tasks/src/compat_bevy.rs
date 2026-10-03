//! A `bevy_tasks`-compatible prelude (behind the `compat-bevy` feature).
//!
//! This module provides the familiar `bevy_tasks` surface — the global
//! `ComputeTaskPool` / `AsyncComputeTaskPool` / `IoTaskPool` accessors, a
//! `TaskPoolBuilder`, and a `scope`/`spawn`-shaped pool — mapped onto this
//! crate's own [`TaskPool`](crate::TaskPool). It exists so code written against
//! `bevy_tasks` can migrate to `prism_tasks` with minimal churn; it adds no new
//! scheduling, only re-shapes the existing pool API.
//!
//! Differences from `bevy_tasks` (honest caveats):
//! - The compat [`CompatTaskPool::scope`] takes **synchronous** closures
//!   (`FnOnce() -> T`) rather than futures, and returns their results in spawn
//!   order. For futures, use [`CompatTaskPool::spawn`], which forwards to this
//!   crate's async executor and returns a [`Task`](crate::Task).
//! - The three global pools are independent [`TaskPool`](crate::TaskPool)s here
//!   rather than Bevy's shared-pool arrangement; initialize each once via its
//!   `get_or_init`.
//!
//! Everything here is re-exported from [`prelude`] for a one-line import.

use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::{Mutex, OnceLock};

use core::future::Future;
use core::ops::Deref;

use crate::{Scope, Task, TaskPool};

/// Builder for a [`CompatTaskPool`], mirroring `bevy_tasks::TaskPoolBuilder`.
#[derive(Clone, Debug, Default)]
pub struct TaskPoolBuilder {
    num_threads: Option<usize>,
}

impl TaskPoolBuilder {
    /// Start a new builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of worker threads. `0` selects the synchronous fallback.
    #[must_use]
    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Build the pool.
    #[must_use]
    pub fn build(self) -> CompatTaskPool {
        let inner = match self.num_threads {
            Some(threads) => TaskPool::with_threads(threads),
            None => TaskPool::new(),
        };
        CompatTaskPool { inner }
    }
}

/// A `bevy_tasks`-shaped wrapper around this crate's [`TaskPool`]. Derefs to the
/// inner pool, so every native pool method is available directly.
pub struct CompatTaskPool {
    inner: TaskPool,
}

impl CompatTaskPool {
    /// Build a compat pool with the default worker count.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: TaskPool::new(),
        }
    }

    /// Wrap an existing [`TaskPool`].
    #[must_use]
    pub fn from_pool(pool: TaskPool) -> Self {
        Self { inner: pool }
    }

    /// The underlying native pool.
    #[must_use]
    pub fn pool(&self) -> &TaskPool {
        &self.inner
    }

    /// Number of worker threads (mirrors `bevy_tasks::TaskPool::thread_num`).
    #[must_use]
    pub fn thread_num(&self) -> usize {
        self.inner.worker_count()
    }

    /// Spawn a future on the pool, returning a [`Task`] handle (mirrors
    /// `bevy_tasks::TaskPool::spawn`).
    pub fn spawn<F>(&self, future: F) -> Task<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.inner.spawn_async(future)
    }

    /// Open a scope that runs synchronous closures in parallel and returns
    /// their results in spawn order (mirrors the shape of
    /// `bevy_tasks::TaskPool::scope`).
    ///
    /// Like `bevy_tasks::TaskPool::scope`, the collected result type must be
    /// `Send + 'static`; the spawned closures themselves may still borrow from
    /// the enclosing `'env` environment. Results are shared through a
    /// reference-counted slot vector, which keeps the self-referential scope
    /// handle free of the native scope's `for<'scope>` lifetime quantifier.
    pub fn scope<'env, F, T>(&self, f: F) -> Vec<T>
    where
        T: Send + 'static,
        F: for<'scope> FnOnce(&CompatScope<'scope, 'env, T>),
    {
        let results: Arc<Mutex<Vec<Option<T>>>> = Arc::new(Mutex::new(Vec::new()));
        let shared = Arc::clone(&results);
        self.inner.scope(|scope| {
            let compat = CompatScope {
                scope,
                results: shared,
            };
            f(&compat);
        });
        let slots = Arc::into_inner(results)
            .expect("all compat scope tasks dropped their result handles before scope returned")
            .into_inner()
            .unwrap();
        slots
            .into_iter()
            .map(|slot| slot.expect("every spawned compat task completed before scope returned"))
            .collect()
    }
}

impl Default for CompatTaskPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for CompatTaskPool {
    type Target = TaskPool;

    fn deref(&self) -> &TaskPool {
        &self.inner
    }
}

/// The scope handle passed to [`CompatTaskPool::scope`]. Spawn synchronous
/// closures with [`CompatScope::spawn`]; their results are collected in spawn
/// order.
pub struct CompatScope<'scope, 'env, T: Send + 'static> {
    scope: &'scope Scope<'scope, 'env>,
    results: Arc<Mutex<Vec<Option<T>>>>,
}

impl<'scope, 'env, T: Send + 'static> CompatScope<'scope, 'env, T> {
    /// Spawn a synchronous closure. Its result is placed in the scope's result
    /// vector at this spawn's position.
    pub fn spawn<Func>(&self, f: Func)
    where
        Func: FnOnce() -> T + Send + 'scope,
    {
        let index = {
            let mut results = self.results.lock().unwrap();
            let index = results.len();
            results.push(None);
            index
        };
        let results = Arc::clone(&self.results);
        self.scope.spawn(move || {
            let value = f();
            results.lock().unwrap()[index] = Some(value);
        });
    }
}

/// Build the global-accessor boilerplate for a named task pool, matching the
/// `bevy_tasks` `ComputeTaskPool::get()` / `get_or_init()` shape.
macro_rules! global_pool {
    ($(#[$meta:meta])* $name:ident, $cell:ident) => {
        static $cell: OnceLock<CompatTaskPool> = OnceLock::new();

        $(#[$meta])*
        pub struct $name;

        impl $name {
            /// Get the global pool, initializing it with `init` if it has not
            /// been created yet.
            pub fn get_or_init(init: impl FnOnce() -> TaskPool) -> &'static CompatTaskPool {
                $cell.get_or_init(|| CompatTaskPool::from_pool(init()))
            }

            /// Get the global pool, panicking if it has not been initialized.
            #[must_use]
            pub fn get() -> &'static CompatTaskPool {
                $cell
                    .get()
                    .expect(concat!(stringify!($name), " not initialized; call get_or_init first"))
            }

            /// Get the global pool if it has been initialized.
            #[must_use]
            pub fn try_get() -> Option<&'static CompatTaskPool> {
                $cell.get()
            }
        }
    };
}

global_pool!(
    /// Global compute pool (mirrors `bevy_tasks::ComputeTaskPool`).
    ComputeTaskPool,
    COMPUTE_POOL
);
global_pool!(
    /// Global async-compute pool (mirrors `bevy_tasks::AsyncComputeTaskPool`).
    AsyncComputeTaskPool,
    ASYNC_COMPUTE_POOL
);
global_pool!(
    /// Global I/O pool (mirrors `bevy_tasks::IoTaskPool`).
    IoTaskPool,
    IO_POOL
);

/// One-line import of the `bevy_tasks`-compatible surface.
pub mod prelude {
    pub use super::{
        AsyncComputeTaskPool, CompatScope, CompatTaskPool, ComputeTaskPool, IoTaskPool,
        TaskPoolBuilder,
    };
    pub use crate::{Task, TaskPool};
}
