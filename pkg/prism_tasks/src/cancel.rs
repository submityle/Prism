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

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
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
    /// Registered child nodes, cancelled transitively by [`CancelState::cancel`].
    children: Mutex<Vec<Arc<CancelState>>>,
}

impl CancelState {
    /// Mark this node cancelled and recurse into every registered child.
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        // Snapshot under the lock, then recurse without holding it: the tree
        // has no cycles, and releasing the lock first avoids nesting child
        // locks under the parent lock.
        let children = {
            let guard = self.children.lock().unwrap();
            guard.clone()
        };
        for child in children {
            child.cancel();
        }
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
            state: Arc::new(CancelState {
                cancelled: AtomicBool::new(false),
                children: Mutex::new(Vec::new()),
            }),
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
        let child = Arc::new(CancelState {
            cancelled: AtomicBool::new(false),
            children: Mutex::new(Vec::new()),
        });
        // Hold the parent's children lock across the "is the parent cancelled?"
        // check and the registration push. `CancelState::cancel` also takes
        // this lock to snapshot children, so the two cannot interleave in a way
        // that both leaves the child unregistered and lets it observe the
        // parent as not-yet-cancelled.
        let mut guard = self.state.children.lock().unwrap();
        if self.state.cancelled.load(Ordering::Acquire) {
            child.cancelled.store(true, Ordering::Release);
        }
        guard.push(Arc::clone(&child));
        drop(guard);
        CancelToken { state: child }
    }

    /// Cancel this token and, transitively, every descendant.
    pub fn cancel(&self) {
        self.state.cancel();
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
