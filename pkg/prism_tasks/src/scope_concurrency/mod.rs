//! Structured concurrency with cooperative cancellation (design §24.3).
//!
//! A structured scope spawns borrowed child tasks and **joins every one before
//! it returns**, so no task (and none of its borrows) can leak past the scope —
//! the same guarantee `std::thread::scope` and Kotlin structured concurrency
//! make. On top of that join barrier this module threads a cancellation tree
//! through the scope:
//!
//! - Each scope carries a [`CancelToken`]. Spawned tasks receive that token and
//!   poll it at their own checkpoints — cancellation is **cooperative**, never
//!   a forced kill, so a task that bows out leaves consistent state.
//! - Cancelling a scope ([`StructuredScope::cancel`]) trips the token, which
//!   propagates to every descendant token: parent cancellation cascades to
//!   children (and children spawned *after* the cancel are born cancelled).
//! - A task that has not yet started when the scope is cancelled is skipped
//!   entirely rather than run and immediately bailed, and is reported in
//!   [`ScopeOutcome::skipped`].
//!
//! # Determinism
//! The *shape* of the scope tree and the exact set of nodes a cancellation
//! reaches is a pure, clock-free, thread-free function captured by the
//! [`ScopeTree`] core (see [`tree`]). It is tested directly against a serial
//! oracle. The façade here binds that model to a real [`TaskPool`] and the
//! existing [`CancelToken`] tree, inheriting help-on-wait (no deadlock) and the
//! single-threaded inline fallback from [`TaskPool::scope`].
//!
//! # Layering
//! [`StructuredScope`] is a thin wrapper over [`TaskPool::scope`]: it reuses the
//! scope's lifetime-erasure soundness and join barrier wholesale (no new
//! `unsafe` is introduced here) and only adds the cancellation token plumbing
//! and the admitted / skipped / completed accounting.

pub mod tree;

use alloc::sync::Arc;
use core::sync::atomic::{AtomicUsize, Ordering};

pub use tree::{Node, NodeId, NodeState, ScopeTree};

use crate::scope::Scope;
use crate::{CancelToken, TaskPool};

/// The result of a [`TaskPool::structured_scope`] call: the scope body's return
/// value plus deterministic accounting of what its tasks did.
///
/// The counts are over tasks spawned with [`StructuredScope::spawn`]. They are
/// a function of the submitted work and whether the scope was cancelled, not of
/// worker interleaving, so `spawned == completed + skipped` always holds once
/// the scope has joined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScopeOutcome<T> {
    /// The value returned by the scope body closure.
    pub value: T,
    /// Tasks spawned on this scope.
    pub spawned: usize,
    /// Tasks whose body ran to completion (they were not skipped by an
    /// already-tripped token before starting).
    pub completed: usize,
    /// Tasks skipped because the scope's token was already cancelled when the
    /// task was about to start.
    pub skipped: usize,
    /// Whether the scope's token was cancelled by the time the scope joined.
    pub cancelled: bool,
}

/// Shared, cloneable accounting counters for a [`StructuredScope`].
///
/// Held behind [`Arc`] so spawned task closures — which must be `Send` and
/// outlive the scope's `'scope` lifetime — can own their own handle without
/// borrowing the scope frame.
#[derive(Clone)]
struct ScopeCounters {
    spawned: Arc<AtomicUsize>,
    completed: Arc<AtomicUsize>,
    skipped: Arc<AtomicUsize>,
}

impl ScopeCounters {
    fn new() -> Self {
        Self {
            spawned: Arc::new(AtomicUsize::new(0)),
            completed: Arc::new(AtomicUsize::new(0)),
            skipped: Arc::new(AtomicUsize::new(0)),
        }
    }
}

/// A structured-concurrency scope bound to a [`TaskPool`] and a [`CancelToken`].
///
/// Obtained via [`TaskPool::structured_scope`]. Tasks spawned with
/// [`StructuredScope::spawn`] may borrow from the enclosing environment and are
/// all joined before the scope returns. Each task receives the scope's
/// [`CancelToken`] so it can bow out cooperatively at its own checkpoints.
///
/// `'scope` is the lifetime of the scope itself and `'env` is the lifetime of
/// data borrowed from outside the scope.
pub struct StructuredScope<'scope, 'env: 'scope> {
    /// The underlying structured scope that owns the join barrier.
    inner: &'scope Scope<'scope, 'env>,
    /// This scope's cancellation node; tasks poll it and `cancel` trips it.
    token: CancelToken,
    /// Shared spawned / completed / skipped accounting.
    counters: ScopeCounters,
}

impl<'scope, 'env> StructuredScope<'scope, 'env> {
    /// Spawn a child task that may borrow from the enclosing `'env`
    /// environment. The task receives the scope's [`CancelToken`] so it can
    /// poll [`CancelToken::is_cancelled`] / [`CancelToken::check`] at its
    /// checkpoints.
    ///
    /// If the scope's token is already cancelled when the task is about to
    /// start, the body is skipped entirely (counted in
    /// [`ScopeOutcome::skipped`]); otherwise it runs to completion (counted in
    /// [`ScopeOutcome::completed`]). Either way the task is joined before the
    /// scope returns.
    pub fn spawn<G>(&self, body: G)
    where
        G: FnOnce(&CancelToken) + Send + 'scope,
    {
        self.counters.spawned.fetch_add(1, Ordering::Relaxed);
        let token = self.token.clone();
        let completed = Arc::clone(&self.counters.completed);
        let skipped = Arc::clone(&self.counters.skipped);
        let inner: &'scope Scope<'scope, 'env> = self.inner;
        inner.spawn(move || {
            if token.is_cancelled() {
                skipped.fetch_add(1, Ordering::Relaxed);
                return;
            }
            body(&token);
            completed.fetch_add(1, Ordering::Relaxed);
        });
    }

    /// The scope's [`CancelToken`]. Clone it into longer-lived work, or derive
    /// an independent sub-scope token with [`CancelToken::child`].
    #[must_use]
    #[inline]
    pub fn token(&self) -> &CancelToken {
        &self.token
    }

    /// Derive a child token under this scope's token, for a nested sub-scope
    /// that the parent can still cancel. Equivalent to `self.token().child()`.
    #[must_use]
    #[inline]
    pub fn child_token(&self) -> CancelToken {
        self.token.child()
    }

    /// Cancel this scope: trip its token and, transitively, every descendant
    /// token. In-flight tasks observe the trip at their next checkpoint and
    /// not-yet-started tasks are skipped.
    #[inline]
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Whether this scope's token (or an ancestor) has been cancelled.
    #[must_use]
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }
}

impl TaskPool {
    /// Open a structured-concurrency scope under `parent` (design §24.3).
    ///
    /// The scope derives a child token from `parent`, so cancelling `parent`
    /// (or any ancestor) cancels this scope and all its tasks. Within `f`, use
    /// [`StructuredScope::spawn`] to launch borrowed tasks that each receive the
    /// scope's token; every task is joined before this call returns, even if a
    /// task or the body panics.
    ///
    /// Returns a [`ScopeOutcome`] carrying the body's value and the
    /// deterministic spawned / completed / skipped accounting.
    ///
    /// ```
    /// # use prism_tasks::{TaskPool, CancelToken};
    /// let pool = TaskPool::with_threads(4);
    /// let root = CancelToken::new();
    /// let mut data = vec![0u32; 8];
    /// let outcome = pool.structured_scope(&root, |s| {
    ///     for (i, slot) in data.iter_mut().enumerate() {
    ///         s.spawn(move |token| {
    ///             if token.is_cancelled() {
    ///                 return;
    ///             }
    ///             *slot = u32::try_from(i).unwrap() * 2;
    ///         });
    ///     }
    /// });
    /// assert_eq!(outcome.spawned, 8);
    /// assert_eq!(outcome.completed, 8);
    /// assert_eq!(data, vec![0, 2, 4, 6, 8, 10, 12, 14]);
    /// ```
    pub fn structured_scope<'env, F, T>(&self, parent: &CancelToken, f: F) -> ScopeOutcome<T>
    where
        F: for<'scope> FnOnce(&StructuredScope<'scope, 'env>) -> T,
    {
        // A child node of `parent`: cancelling an ancestor cancels this scope,
        // and if `parent` is already cancelled this scope is born cancelled.
        let token = parent.child();
        let counters = ScopeCounters::new();
        let readback = counters.clone();

        let value = self.scope(|inner| {
            let structured = StructuredScope {
                inner,
                token: token.clone(),
                counters: counters.clone(),
            };
            f(&structured)
        });

        ScopeOutcome {
            value,
            spawned: readback.spawned.load(Ordering::Relaxed),
            completed: readback.completed.load(Ordering::Relaxed),
            skipped: readback.skipped.load(Ordering::Relaxed),
            cancelled: token.is_cancelled(),
        }
    }
}
