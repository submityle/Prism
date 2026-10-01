//! A reactive client-side router built on `prism_ui_reactive`.
//!
//! A [`Router`] owns a `Signal<Location>` for the current location plus a
//! back/forward history stack. Navigating updates the signal (so any dependent
//! [`Memo`] or effect recomputes) and truncates any forward history. The
//! [`Router::current_match`] memo re-resolves the active route whenever the
//! location changes.

use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;

use prism_ui_reactive::{Memo, Runtime, Signal};

use crate::matcher::{RouteId, RouteTable};
use crate::path::Location;
use crate::route::RouteMatch;

/// The back/forward history stack: a list of visited locations plus a cursor
/// pointing at the current entry.
#[derive(Debug)]
struct History {
    stack: Vec<Location>,
    cursor: usize,
}

/// A reactive router over a shared `Runtime`.
///
/// Create one with [`Router::new`]. Read the current location (tracked) with
/// [`Router::location`], observe the resolved route with
/// [`Router::current_match`], and move through history with
/// [`Router::navigate`], [`Router::back`], and [`Router::forward`].
///
/// `Router` is cheaply clonable; all clones share the same underlying location
/// signal and history.
#[derive(Clone)]
pub struct Router {
    location: Signal<Location>,
    history: Rc<RefCell<History>>,
    current: Memo<Option<(RouteId, RouteMatch)>>,
}

impl Router {
    /// Creates a router seeded with `initial` and backed by `table`.
    ///
    /// The initial location becomes the only history entry. The returned
    /// router shares `rt`'s reactive graph, so memos and effects created from
    /// the same [`Runtime`] observe navigation.
    #[must_use]
    pub fn new(rt: &Runtime, initial: &str, table: RouteTable) -> Self {
        let location = rt.signal(Location::new(initial));
        let history = Rc::new(RefCell::new(History {
            stack: vec![location.get_untracked()],
            cursor: 0,
        }));
        let table = Rc::new(table);
        let current = rt.memo({
            let location = location.clone();
            let table = Rc::clone(&table);
            move || location.with(|loc| table.resolve(loc))
        });
        Self {
            location,
            history,
            current,
        }
    }

    /// Returns the current [`Location`]. This read is tracked, so it registers a
    /// dependency when called inside a memo or effect.
    #[must_use]
    pub fn location(&self) -> Location {
        self.location.get()
    }

    /// Returns the memo resolving the current location to a route.
    ///
    /// The memo yields `Some((RouteId, RouteMatch))` for a matching route or
    /// `None` otherwise, recomputing whenever the location changes.
    #[must_use]
    pub fn current_match(&self) -> Memo<Option<(RouteId, RouteMatch)>> {
        self.current.clone()
    }

    /// Navigates to `path`, pushing it onto the history stack.
    ///
    /// Any forward history (entries after the current cursor) is discarded, so
    /// navigation always continues from the current position.
    pub fn navigate(&self, path: &str) {
        let next = Location::new(path);
        {
            let mut history = self.history.borrow_mut();
            let cursor = history.cursor;
            history.stack.truncate(cursor + 1);
            history.stack.push(next.clone());
            history.cursor = history.stack.len() - 1;
        }
        self.location.set(next);
    }

    /// Moves one step back in history, if possible.
    ///
    /// Returns `true` if the cursor moved, or `false` when already at the
    /// oldest entry.
    pub fn back(&self) -> bool {
        let target = {
            let mut history = self.history.borrow_mut();
            if history.cursor == 0 {
                return false;
            }
            history.cursor -= 1;
            history.stack[history.cursor].clone()
        };
        self.location.set(target);
        true
    }

    /// Moves one step forward in history, if possible.
    ///
    /// Returns `true` if the cursor moved, or `false` when already at the
    /// newest entry.
    pub fn forward(&self) -> bool {
        let target = {
            let mut history = self.history.borrow_mut();
            if history.cursor + 1 >= history.stack.len() {
                return false;
            }
            history.cursor += 1;
            history.stack[history.cursor].clone()
        };
        self.location.set(target);
        true
    }

    /// Returns `true` if there is a previous entry to go [`Router::back`] to.
    #[must_use]
    pub fn can_go_back(&self) -> bool {
        self.history.borrow().cursor > 0
    }

    /// Returns `true` if there is a later entry to go [`Router::forward`] to.
    #[must_use]
    pub fn can_go_forward(&self) -> bool {
        let history = self.history.borrow();
        history.cursor + 1 < history.stack.len()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::std_instead_of_alloc, reason = "tests run under std")]

    use std::cell::RefCell;
    use std::rc::Rc;

    use prism_ui_reactive::Runtime;

    use super::Router;
    use crate::matcher::{RouteId, RouteTable};

    fn table() -> RouteTable {
        RouteTable::new()
            .route("/", RouteId::new(0))
            .route("/users/:id", RouteId::new(1))
            .route("/files/*rest", RouteId::new(2))
    }

    #[test]
    fn navigate_updates_location_and_fires_effect() {
        let rt = Runtime::new();
        let router = Router::new(&rt, "/", table());

        let log = Rc::new(RefCell::new(Vec::new()));
        let _effect = rt.effect({
            let router = router.clone();
            let log = Rc::clone(&log);
            move || log.borrow_mut().push(router.location().path().to_string())
        });

        assert_eq!(*log.borrow(), vec!["/".to_string()]);
        router.navigate("/users/42");
        assert_eq!(router.location().path(), "/users/42");
        assert_eq!(
            *log.borrow(),
            vec!["/".to_string(), "/users/42".to_string()]
        );
    }

    #[test]
    fn current_match_memo_recomputes_on_navigate() {
        let rt = Runtime::new();
        let router = Router::new(&rt, "/", table());
        let current = router.current_match();

        let (id, _) = current.get().expect("root matches");
        assert_eq!(id, RouteId::new(0));

        router.navigate("/users/7");
        let (id, matched) = current.get().expect("user route matches");
        assert_eq!(id, RouteId::new(1));
        assert_eq!(matched.param("id"), Some("7"));

        router.navigate("/files/a/b.txt");
        let (id, matched) = current.get().expect("files route matches");
        assert_eq!(id, RouteId::new(2));
        assert_eq!(matched.wildcard(), Some("a/b.txt"));
    }

    #[test]
    fn back_and_forward_restore_locations() {
        let rt = Runtime::new();
        let router = Router::new(&rt, "/", table());
        assert!(!router.can_go_back());
        assert!(!router.can_go_forward());

        router.navigate("/users/1");
        router.navigate("/users/2");
        assert!(router.can_go_back());
        assert!(!router.can_go_forward());

        assert!(router.back());
        assert_eq!(router.location().path(), "/users/1");
        assert!(router.can_go_forward());

        assert!(router.forward());
        assert_eq!(router.location().path(), "/users/2");
        assert!(!router.forward());
    }

    #[test]
    fn navigating_truncates_forward_history() {
        let rt = Runtime::new();
        let router = Router::new(&rt, "/", table());
        router.navigate("/users/1");
        router.navigate("/users/2");
        assert!(router.back());
        assert_eq!(router.location().path(), "/users/1");

        router.navigate("/users/3");
        assert!(!router.can_go_forward());
        assert_eq!(router.location().path(), "/users/3");
    }
}
