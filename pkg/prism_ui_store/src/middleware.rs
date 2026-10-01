//! Middleware hooks that observe committed state transitions.
//!
//! A [`Middleware`] is attached to a [`Store`](crate::Store) and is invoked
//! around every *committing* write (see the crate-level commit policy). The
//! bundled [`LoggingMiddleware`] records a `(prev, next)` snapshot for each
//! commit, which makes it a convenient observability and testing tool.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

/// A hook invoked by a [`Store`](crate::Store) around state commits.
///
/// Implementors see the state immediately before a commit via
/// [`Middleware::before_commit`] and the before/after pair via
/// [`Middleware::after_commit`]. Both methods take shared references, so a
/// middleware observes transitions but never mutates the committed state.
pub trait Middleware<S> {
    /// Called just before a commit is applied, with the current (`prev`) state.
    ///
    /// The default implementation does nothing, so middleware that only cares
    /// about completed transitions can ignore it.
    fn before_commit(&self, prev: &S) {
        let _ = prev;
    }

    /// Called immediately after a commit, with the state before (`prev`) and
    /// after (`next`) the change.
    fn after_commit(&self, prev: &S, next: &S);
}

/// A shared, cloneable buffer of `(prev, next)` snapshots recorded by a
/// [`LoggingMiddleware`].
pub type CommitLog<S> = Rc<RefCell<Vec<(S, S)>>>;

/// A [`Middleware`] that appends every observed `(prev, next)` commit to a
/// shared [`CommitLog`].
///
/// Clone the handle returned by [`LoggingMiddleware::log`] before moving the
/// middleware into a store to retain read access to the recorded history.
pub struct LoggingMiddleware<S> {
    log: CommitLog<S>,
}

impl<S> LoggingMiddleware<S> {
    /// Create a middleware backed by a fresh, empty [`CommitLog`].
    pub fn new() -> Self {
        Self {
            log: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// A cloned handle to the shared commit log, suitable for inspection.
    pub fn log(&self) -> CommitLog<S> {
        self.log.clone()
    }
}

impl<S> Default for LoggingMiddleware<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Clone> Middleware<S> for LoggingMiddleware<S> {
    fn after_commit(&self, prev: &S, next: &S) {
        self.log.borrow_mut().push((prev.clone(), next.clone()));
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under the std feature"
    )]

    use super::*;
    use crate::Store;
    use prism_ui_reactive::Runtime;
    use std::cell::RefCell as StdRefCell;
    use std::rc::Rc;

    #[derive(Clone, PartialEq, Debug)]
    struct Counter {
        value: i32,
    }

    #[test]
    fn logging_middleware_records_update_and_set_commits() {
        // Arrange.
        let rt = Runtime::new();
        let mw = LoggingMiddleware::<Counter>::new();
        let log = mw.log();
        let store = Store::new(&rt, Counter { value: 0 }).with_middleware(mw);

        // Act.
        store.update(|c| c.value += 1);
        store.set(Counter { value: 9 });

        // Assert.
        let recorded = log.borrow();
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].0.value, 0);
        assert_eq!(recorded[0].1.value, 1);
        assert_eq!(recorded[1].0.value, 1);
        assert_eq!(recorded[1].1.value, 9);
    }

    #[test]
    fn update_always_runs_middleware_even_on_no_op() {
        // Arrange: `update` cannot detect no-ops, so middleware always fires.
        let rt = Runtime::new();
        let mw = LoggingMiddleware::<Counter>::new();
        let log = mw.log();
        let store = Store::new(&rt, Counter { value: 7 });
        store.add_middleware(mw);

        // Act: a closure that leaves the value untouched still commits.
        store.update(|_c| {});

        // Assert.
        let recorded = log.borrow();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].0.value, 7);
        assert_eq!(recorded[0].1.value, 7);
    }

    #[test]
    fn set_if_changed_skips_middleware_when_equal() {
        // Arrange.
        let rt = Runtime::new();
        let mw = LoggingMiddleware::<Counter>::new();
        let log = mw.log();
        let store = Store::new(&rt, Counter { value: 3 }).with_middleware(mw);

        // Act.
        let changed_equal = store.set_if_changed(Counter { value: 3 });
        let changed_diff = store.set_if_changed(Counter { value: 4 });

        // Assert: equal value is a no-op; different value commits once.
        assert!(!changed_equal);
        assert!(changed_diff);
        let recorded = log.borrow();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].0.value, 3);
        assert_eq!(recorded[0].1.value, 4);
    }

    #[test]
    fn before_commit_default_is_noop() {
        // Arrange: a middleware that only implements `after_commit`.
        struct AfterOnly {
            hits: Rc<StdRefCell<usize>>,
        }
        impl Middleware<i32> for AfterOnly {
            fn after_commit(&self, _prev: &i32, _next: &i32) {
                *self.hits.borrow_mut() += 1;
            }
        }

        let rt = Runtime::new();
        let hits = Rc::new(StdRefCell::new(0usize));
        let store = Store::new(&rt, 0i32).with_middleware(AfterOnly { hits: hits.clone() });

        // Act.
        store.set(1);
        store.set(2);

        // Assert: default `before_commit` does nothing, `after_commit` fires.
        assert_eq!(*hits.borrow(), 2);
    }
}
