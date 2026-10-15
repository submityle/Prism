//! Cooperative cancellation tokens (design §17 作业取消, §24.3 结构化并发与取消).
//!
//! Cancellation here is **cooperative**, never a forced kill: a long-running
//! job (pathfinding, light baking, streaming decode) carries a [`CancelToken`]
//! and voluntarily bows out at a checkpoint when the token is tripped, so state
//! stays consistent. Tokens form a **tree**: [`CancelToken::child`] derives a
//! sub-token, and cancelling a parent propagates to every descendant. A child
//! born after its parent was already cancelled starts cancelled too, closing
//! the register/cancel race.
//!
//! Clones share state: a cloned token is the *same* cancellation node, so
//! cancelling either clone cancels both. Use [`CancelToken::child`] when you
//! want an independent sub-scope that the parent can still cancel.

use alloc::boxed::Box;
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use core::any::Any;
use core::fmt;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::TaskPool;

/// Error returned by [`CancelToken::check`] when the token has been cancelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cancelled;

impl fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("operation was cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// Shared state of one node in the cancellation tree.
struct CancelState {
    /// Whether this node (and therefore its subtree) has been cancelled.
    cancelled: AtomicBool,
    /// Weak links let abandoned scopes die before their parent is cancelled.
    children: Mutex<Vec<Weak<CancelState>>>,
    /// Keep the path to live descendants intact even if intermediate tokens
    /// are dropped. Together with weak child links this cannot form a cycle.
    parent: Option<Arc<CancelState>>,
    waiters: Mutex<CancelWaiters>,
}

impl CancelState {
    fn new(parent: Option<Arc<Self>>) -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            children: Mutex::new(Vec::new()),
            parent,
            waiters: Mutex::new(CancelWaiters::default()),
        }
    }

    fn cancel_node(&self, pending: &mut Vec<Arc<Self>>, wakers: &mut Vec<Arc<Waker>>) {
        self.cancelled.store(true, Ordering::Release);
        // The flag is checked under this same lock when registering. Either
        // the waiter observes cancellation or this drain includes its waker.
        let waiters = {
            let mut guard = self.waiters.lock().unwrap();
            std::mem::take(&mut *guard)
        };
        wakers.extend(waiters.slots.into_iter().flatten());
        let children = {
            let mut guard = self.children.lock().unwrap();
            std::mem::take(&mut *guard)
        };
        // Upgrade outside the lock: even releasing a node's last Arc must not
        // run a user waker destructor under a tree mutex.
        pending.extend(children.into_iter().filter_map(|child| child.upgrade()));
    }

    fn cancel(&self) {
        let already_panicking = std::thread::panicking();
        let mut pending = Vec::new();
        let mut wakers = Vec::new();
        self.cancel_node(&mut pending, &mut wakers);
        while let Some(child) = pending.pop() {
            child.cancel_node(&mut pending, &mut wakers);
        }

        // Finish propagation before invoking user code. Catch wake and drop
        // separately so even a panicking waker destructor cannot suppress
        // another notification. Keep panic payloads until all notifications
        // finish, since their destructors can also execute user code.
        let mut failures = Vec::new();
        for waker in wakers {
            failures.extend(panic::catch_unwind(AssertUnwindSafe(|| waker.wake_by_ref())).err());
            failures.extend(panic::catch_unwind(AssertUnwindSafe(|| drop(waker))).err());
        }
        let mut failures = failures.into_iter();
        let first = failures.next();
        for failure in failures {
            discard_cancel_panic(failure);
        }
        if let Some(first) = first {
            if already_panicking {
                // Cancellation from a destructor must preserve the panic
                // already unwinding through that destructor.
                discard_cancel_panic(first);
            } else {
                panic::resume_unwind(first);
            }
        }
    }
}

/// Dispose of one suppressed payload without allowing its destructor to
/// replace the original panic. A destructor may panic with another payload
/// whose destructor also panics, so never recursively dispose of that result:
/// deliberately leak only that new payload rather than risking an abort or
/// unbounded cleanup. The original payload is still released when possible.
fn discard_cancel_panic(payload: Box<dyn Any + Send>) {
    if let Err(cleanup_panic) = panic::catch_unwind(AssertUnwindSafe(|| drop(payload))) {
        core::mem::forget(cleanup_panic);
    }
}

impl Drop for CancelState {
    fn drop(&mut self) {
        // A live leaf may be the only owner of a very deep ancestry chain.
        // Release it iteratively rather than recursively overflowing the stack.
        let mut parent = self.parent.take();
        while let Some(node) = parent.take() {
            if let Some(mut state) = Arc::into_inner(node) {
                parent = state.parent.take();
            }
        }
    }
}

/// Reusable slots keep registration/removal O(1) and storage bounded by peak
/// simultaneous waiters. Only Arc wrappers are copied while holding the lock.
#[derive(Default)]
struct CancelWaiters {
    slots: Vec<Option<Arc<Waker>>>,
    free: Vec<usize>,
}

struct CancelFuture<'a> {
    state: &'a CancelState,
    slot: Option<usize>,
}

impl CancelFuture<'_> {
    fn unregister(&mut self) {
        let Some(slot) = self.slot.take() else {
            return;
        };
        let removed = {
            let mut waiters = self.state.waiters.lock().unwrap();
            // Cancellation may already have drained the entire registry.
            let removed = waiters.slots.get_mut(slot).and_then(Option::take);
            if removed.is_some() {
                waiters.free.push(slot);
            }
            removed
        };
        drop(removed);
    }
}

impl Future for CancelFuture<'_> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.state.cancelled.load(Ordering::Acquire) {
            self.unregister();
            return Poll::Ready(());
        }

        // RawWaker clone/drop/wake callbacks are arbitrary user code. Clone
        // before locking; transfer replaced registrations out before dropping.
        let waker = Arc::new(cx.waker().clone());
        let (cancelled, replaced) = {
            let mut waiters = self.state.waiters.lock().unwrap();
            if self.state.cancelled.load(Ordering::Acquire) {
                (true, None)
            } else if let Some(slot) = self.slot {
                (false, waiters.slots[slot].replace(Arc::clone(&waker)))
            } else {
                let slot = if let Some(slot) = waiters.free.pop() {
                    waiters.slots[slot] = Some(Arc::clone(&waker));
                    slot
                } else {
                    let slot = waiters.slots.len();
                    waiters.slots.push(Some(Arc::clone(&waker)));
                    slot
                };
                self.slot = Some(slot);
                (false, None)
            }
        };
        drop(replaced);
        if cancelled {
            self.unregister();
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

impl Drop for CancelFuture<'_> {
    fn drop(&mut self) {
        self.unregister();
    }
}

/// A cooperative cancellation token forming part of a cancellation tree.
///
/// Create a root with [`CancelToken::new`], derive sub-tokens with
/// [`CancelToken::child`], trip the subtree with [`CancelToken::cancel`], and
/// have jobs poll [`CancelToken::is_cancelled`] or [`CancelToken::check`] at
/// their checkpoints. Cloning yields another handle to the *same* node.
#[derive(Clone)]
pub struct CancelToken {
    state: Arc<CancelState>,
}

impl CancelToken {
    /// Create a fresh, un-cancelled root token.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(CancelState::new(None)),
        }
    }

    /// Derive a child token registered under this one.
    ///
    /// Cancelling this token later also cancels the child (and its descendants).
    /// If this token is *already* cancelled, the returned child is born
    /// cancelled, so there is no window in which a late child escapes a parent
    /// cancellation.
    #[must_use]
    pub fn child(&self) -> CancelToken {
        let child = Arc::new(CancelState::new(Some(Arc::clone(&self.state))));
        // Hold the parent's children lock across the "is the parent cancelled?"
        // check and the registration push. `CancelState::cancel` also takes
        // this lock to drain children, so the two cannot interleave in a way
        // that both leaves the child unregistered and lets it observe the
        // parent as not-yet-cancelled.
        let mut guard = self.state.children.lock().unwrap();
        if self.state.cancelled.load(Ordering::Acquire) {
            child.cancelled.store(true, Ordering::Release);
        } else {
            // Amortize pruning so a long-lived scope does not retain an
            // unbounded list of weak links to abandoned children.
            if guard.len() == guard.capacity() {
                guard.retain(|child| child.strong_count() != 0);
            }
            guard.push(Arc::downgrade(&child));
        }
        drop(guard);
        CancelToken { state: child }
    }

    /// Cancel this token and, transitively, every descendant.
    ///
    /// # Panics
    /// Unless this thread is already unwinding, the first waiter waker panic
    /// is propagated after cancelling all descendants and notifying every
    /// registered waiter. During unwinding, waiter panics are suppressed to
    /// preserve the original panic. Suppressed payloads are dropped with panic
    /// isolation; if a payload destructor panics, its new payload is deliberately
    /// leaked to avoid recursive panicking destructors.
    pub fn cancel(&self) {
        self.state.cancel();
    }

    /// Wait until this token or an ancestor is cancelled.
    ///
    /// Every pending future registers independently and uses the waker from
    /// its most recent poll. Dropping a future unregisters it. An already
    /// cancelled token completes immediately, without cloning the waker.
    /// Clone the token into an async task to share its cancellation lifetime:
    ///
    /// ```
    /// # use prism_tasks::{CancelToken, TaskPool};
    /// let pool = TaskPool::with_threads(1);
    /// let cancel = CancelToken::new();
    /// let task_cancel = cancel.clone();
    /// let task = pool.spawn_async(async move { task_cancel.cancelled().await });
    /// cancel.cancel();
    /// pool.block_on(task);
    /// ```
    pub fn cancelled(&self) -> impl Future<Output = ()> + Send + '_ {
        CancelFuture {
            state: &self.state,
            slot: None,
        }
    }

    /// Whether this token (or an ancestor) has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    /// Return `Err(Cancelled)` if cancelled, else `Ok(())`.
    ///
    /// # Errors
    /// Returns [`Cancelled`] when [`CancelToken::is_cancelled`] holds.
    pub fn check(&self) -> Result<(), Cancelled> {
        if self.is_cancelled() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
#[path = "cancel_future_tests.rs"]
mod cancel_future_tests;

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CancelToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CancelToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

/// Outcome of a cancellable parallel loop (see
/// [`TaskPool::par_for_each_cancellable`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelOutcome {
    /// Every element was processed; the token was never tripped.
    Completed,
    /// The token was cancelled; some elements may not have been processed.
    Cancelled,
}

impl CancelOutcome {
    /// Whether the loop ran to completion without cancellation.
    #[must_use]
    pub fn is_completed(self) -> bool {
        matches!(self, Self::Completed)
    }

    /// Whether the loop stopped early because the token was cancelled.
    #[must_use]
    pub fn is_cancelled(self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

impl TaskPool {
    /// Apply `body` to every element of `data` in parallel, bailing out of any
    /// chunk that observes `token` as cancelled.
    ///
    /// Cancellation is cooperative: each chunk checks `token` before each
    /// element, so an in-flight chunk stops promptly at its next element but a
    /// chunk already past its last check still finishes that element. Chunks
    /// that have not started once the token trips are skipped. Returns
    /// [`CancelOutcome::Cancelled`] if the token was cancelled by the time the
    /// loop joined, otherwise [`CancelOutcome::Completed`].
    ///
    /// ```
    /// # use prism_tasks::{TaskPool, CancelToken, CancelOutcome};
    /// let pool = TaskPool::with_threads(4);
    /// let token = CancelToken::new();
    /// let data: Vec<u32> = (0..1000).collect();
    /// let outcome = pool.par_for_each_cancellable(&data, &token, |&x| {
    ///     let _ = x; // do work
    /// });
    /// assert_eq!(outcome, CancelOutcome::Completed);
    /// ```
    pub fn par_for_each_cancellable<T, F>(
        &self,
        data: &[T],
        token: &CancelToken,
        body: F,
    ) -> CancelOutcome
    where
        T: Sync,
        F: Fn(&T) + Sync,
    {
        if data.is_empty() {
            return if token.is_cancelled() {
                CancelOutcome::Cancelled
            } else {
                CancelOutcome::Completed
            };
        }
        let body = &body;
        self.par_chunks(data, move |chunk| {
            for item in chunk {
                if token.is_cancelled() {
                    break;
                }
                body(item);
            }
        });
        if token.is_cancelled() {
            CancelOutcome::Cancelled
        } else {
            CancelOutcome::Completed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CancelOutcome, CancelToken, Cancelled};

    #[test]
    fn new_token_is_not_cancelled() {
        let t = CancelToken::new();
        assert!(!t.is_cancelled());
        assert_eq!(t.check(), Ok(()));
    }

    #[test]
    fn cancel_sets_the_flag() {
        let t = CancelToken::new();
        t.cancel();
        assert!(t.is_cancelled());
        assert_eq!(t.check(), Err(Cancelled));
    }

    #[test]
    fn clone_shares_state() {
        let t = CancelToken::new();
        let c = t.clone();
        t.cancel();
        assert!(c.is_cancelled());
    }

    #[test]
    fn cancel_propagates_to_children_and_grandchildren() {
        let root = CancelToken::new();
        let child = root.child();
        let grandchild = child.child();
        assert!(!grandchild.is_cancelled());
        root.cancel();
        assert!(child.is_cancelled());
        assert!(grandchild.is_cancelled());
    }

    #[test]
    fn child_of_cancelled_parent_is_born_cancelled() {
        let root = CancelToken::new();
        root.cancel();
        let child = root.child();
        assert!(child.is_cancelled());
    }

    #[test]
    fn child_cancel_does_not_cancel_parent() {
        let root = CancelToken::new();
        let child = root.child();
        child.cancel();
        assert!(child.is_cancelled());
        assert!(!root.is_cancelled());
    }

    #[test]
    fn outcome_predicates() {
        assert!(CancelOutcome::Completed.is_completed());
        assert!(!CancelOutcome::Completed.is_cancelled());
        assert!(CancelOutcome::Cancelled.is_cancelled());
        assert!(!CancelOutcome::Cancelled.is_completed());
    }
}
