//! The reactive [`Store`] container.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use prism_ui_reactive::{Runtime, Signal};

use crate::middleware::Middleware;

/// The list of middleware attached to a [`Store`], shared across clones.
type MiddlewareList<S> = Rc<RefCell<Vec<Box<dyn Middleware<S>>>>>;

/// A predictable, reactive container for a single piece of global state `S`.
///
/// The state is backed by a [`Signal`], so reads performed inside a memo or
/// effect are tracked automatically and writes notify the affected observers.
/// Writes also run any attached [`Middleware`]. A `Store` is a cheap, cloneable
/// handle: clones share the same underlying state, runtime, and middleware.
///
/// See the [crate-level docs](crate) for the exact commit policy governing when
/// middleware runs and when observers are notified.
pub struct Store<S: 'static> {
    runtime: Runtime,
    signal: Signal<S>,
    middleware: MiddlewareList<S>,
}

impl<S: 'static> Clone for Store<S> {
    fn clone(&self) -> Self {
        Self {
            runtime: self.runtime.clone(),
            signal: self.signal.clone(),
            middleware: self.middleware.clone(),
        }
    }
}

impl<S: 'static> Store<S> {
    /// Create a store seeded with `initial`, backed by a signal on `runtime`.
    pub fn new(runtime: &Runtime, initial: S) -> Self {
        Self {
            runtime: runtime.clone(),
            signal: runtime.signal(initial),
            middleware: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// Builder-style variant of [`Store::add_middleware`] that consumes and
    /// returns the store, convenient for chaining at construction time.
    #[must_use]
    pub fn with_middleware(self, middleware: impl Middleware<S> + 'static) -> Self {
        self.middleware.borrow_mut().push(Box::new(middleware));
        self
    }

    /// Attach `middleware`, which will observe every subsequent commit.
    pub fn add_middleware(&self, middleware: impl Middleware<S> + 'static) {
        self.middleware.borrow_mut().push(Box::new(middleware));
    }

    /// The [`Runtime`] this store's state lives on.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Crate-internal access to the backing signal, used by the selector impl.
    pub(crate) fn signal(&self) -> &Signal<S> {
        &self.signal
    }

    /// Read a clone of the current state, recording a reactive dependency.
    pub fn get(&self) -> S
    where
        S: Clone,
    {
        self.signal.get()
    }

    /// Borrow the current state through `f`, recording a reactive dependency.
    pub fn with<R>(&self, f: impl FnOnce(&S) -> R) -> R {
        self.signal.with(f)
    }

    /// Mutate the state in place through `f`, then run middleware and notify.
    ///
    /// This always commits: see the [crate-level docs](crate) for why no-op
    /// detection requires [`Store::set_if_changed`].
    pub fn update(&self, f: impl FnOnce(&mut S))
    where
        S: Clone,
    {
        let prev = self.signal.get_untracked();
        self.run_before(&prev);
        self.signal.update(f);
        let next = self.signal.get_untracked();
        self.run_after(&prev, &next);
    }

    /// Replace the state with `state`, then run middleware and notify.
    ///
    /// This always commits, even when `state` equals the current value.
    pub fn set(&self, state: S)
    where
        S: Clone,
    {
        let prev = self.signal.get_untracked();
        self.run_before(&prev);
        self.signal.set(state);
        let next = self.signal.get_untracked();
        self.run_after(&prev, &next);
    }

    /// Replace the state only when `state` differs from the current value.
    ///
    /// Returns `true` when a change was committed. An equal value is dropped
    /// before any middleware runs and no observers are notified.
    pub fn set_if_changed(&self, state: S) -> bool
    where
        S: Clone + PartialEq,
    {
        if self.signal.with_untracked(|current| *current == state) {
            return false;
        }
        let prev = self.signal.get_untracked();
        self.run_before(&prev);
        self.signal.set(state);
        let next = self.signal.get_untracked();
        self.run_after(&prev, &next);
        true
    }

    fn run_before(&self, prev: &S) {
        let middleware = self.middleware.borrow();
        for entry in middleware.iter() {
            entry.before_commit(prev);
        }
    }

    fn run_after(&self, prev: &S, next: &S) {
        let middleware = self.middleware.borrow();
        for entry in middleware.iter() {
            entry.after_commit(prev, next);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under the std feature"
    )]

    use super::*;
    use std::cell::RefCell as StdRefCell;
    use std::rc::Rc as StdRc;

    #[test]
    fn get_with_and_update_mutate_state() {
        // Arrange.
        let rt = Runtime::new();
        let store = Store::new(&rt, 10i32);

        // Act.
        store.update(|n| *n += 5);

        // Assert.
        assert_eq!(store.get(), 15);
        assert_eq!(store.with(|&n| n * 2), 30);
    }

    #[test]
    fn set_replaces_state() {
        // Arrange.
        let rt = Runtime::new();
        let store = Store::new(&rt, 1i32);

        // Act.
        store.set(42);

        // Assert.
        assert_eq!(store.get(), 42);
    }

    #[test]
    fn update_notifies_subscribing_effect() {
        // Arrange: an effect counts how many times it observes the state.
        let rt = Runtime::new();
        let store = Store::new(&rt, 0i32);
        let runs = StdRc::new(StdRefCell::new(0usize));
        let _effect = rt.effect({
            let store = store.clone();
            let runs = runs.clone();
            move || {
                let _ = store.get();
                *runs.borrow_mut() += 1;
            }
        });

        // Act.
        assert_eq!(*runs.borrow(), 1); // initial run
        store.update(|n| *n += 1);
        store.set(7);

        // Assert: each commit re-runs the subscriber.
        assert_eq!(*runs.borrow(), 3);
    }

    #[test]
    fn clones_share_the_same_state() {
        // Arrange.
        let rt = Runtime::new();
        let store = Store::new(&rt, 0i32);
        let clone = store.clone();

        // Act.
        clone.set(99);

        // Assert.
        assert_eq!(store.get(), 99);
    }
}
