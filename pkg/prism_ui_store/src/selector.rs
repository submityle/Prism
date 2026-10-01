//! Fine-grained selectors over [`Store`] state.
//!
//! A selector is a derived [`Memo`] computed from a slice of the store's state.
//! Because the reactive runtime only notifies a node's observers when its value
//! actually changes, a selector recomputes on every state change but only wakes
//! its own downstream observers when the *selected* slice changes.

use prism_ui_reactive::Memo;

use crate::store::Store;

impl<S: 'static> Store<S> {
    /// Derive a fine-grained [`Memo`] over a slice of the state.
    ///
    /// `f` is re-run whenever the state changes, but the returned memo only
    /// notifies its downstream observers when the selected value `U` actually
    /// changes (compared with [`PartialEq`]). This is the fine-grained selector
    /// guarantee: an unrelated change to the state will not disturb observers of
    /// a slice they do not depend on.
    pub fn select<U>(&self, f: impl Fn(&S) -> U + 'static) -> Memo<U>
    where
        U: PartialEq + Clone + 'static,
    {
        let signal = self.signal().clone();
        self.runtime().memo(move || signal.with(|state| f(state)))
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under the std feature"
    )]

    use crate::Store;
    use prism_ui_reactive::Runtime;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Clone, PartialEq, Debug)]
    struct AppState {
        a: i32,
        b: i32,
    }

    #[test]
    fn selector_tracks_its_slice_value() {
        // Arrange.
        let rt = Runtime::new();
        let store = Store::new(&rt, AppState { a: 1, b: 2 });
        let sel_a = store.select(|s| s.a);

        // Act / Assert.
        assert_eq!(sel_a.get(), 1);
        store.update(|s| s.a = 10);
        assert_eq!(sel_a.get(), 10);
    }

    #[test]
    fn selector_is_fine_grained_about_notifications() {
        // Arrange: an effect subscribes only to slice `a` through a selector.
        let rt = Runtime::new();
        let store = Store::new(&rt, AppState { a: 0, b: 0 });
        let sel_a = store.select(|s| s.a);
        let runs = Rc::new(RefCell::new(0usize));
        let _effect = rt.effect({
            let sel_a = sel_a.clone();
            let runs = runs.clone();
            move || {
                let _ = sel_a.get();
                *runs.borrow_mut() += 1;
            }
        });
        assert_eq!(*runs.borrow(), 1); // initial run

        // Act: change the unrelated slice `b`.
        store.update(|s| s.b += 1);

        // Assert: the `a` selector did not wake its observer.
        assert_eq!(*runs.borrow(), 1);

        // Act: now change the selected slice `a`.
        store.update(|s| s.a += 1);

        // Assert: the observer re-runs exactly once.
        assert_eq!(*runs.borrow(), 2);
    }

    #[test]
    fn selector_coalesces_updates_that_do_not_change_the_slice() {
        // Arrange.
        let rt = Runtime::new();
        let store = Store::new(&rt, AppState { a: 5, b: 0 });
        let sel_a = store.select(|s| s.a);
        let runs = Rc::new(RefCell::new(0usize));
        let _effect = rt.effect({
            let sel_a = sel_a.clone();
            let runs = runs.clone();
            move || {
                let _ = sel_a.get();
                *runs.borrow_mut() += 1;
            }
        });
        assert_eq!(*runs.borrow(), 1);

        // Act: write the same value for `a` repeatedly via full `set`.
        store.set(AppState { a: 5, b: 1 });
        store.set(AppState { a: 5, b: 2 });

        // Assert: `a` never changed, so the selector stayed quiet.
        assert_eq!(*runs.borrow(), 1);
    }
}
