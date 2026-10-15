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

use crate::guard::{DeepLinkState, Guard, GuardOutcome, GuardPoll, PendingNavigation};
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

struct QueuedNavigation {
    navigation: PendingNavigation,
    resume_deep_link: bool,
}

/// Errors returned when a guarded navigation cannot be queued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavigationError {
    /// A previous guarded navigation is still waiting for its guard decision.
    PendingNavigation,
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
    pending: Rc<RefCell<Option<QueuedNavigation>>>,
    deep_link: Rc<RefCell<DeepLinkState>>,
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
            pending: Rc::new(RefCell::new(None)),
            deep_link: Rc::new(RefCell::new(DeepLinkState::new())),
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
    /// navigation always continues from the current position. A direct
    /// navigation also supersedes any pending guarded navigation and clears a
    /// remembered deep link, because it is an explicit new user intent.
    pub fn navigate(&self, path: &str) {
        self.pending.borrow_mut().take();
        self.deep_link.borrow_mut().take();
        self.commit(Location::new(path));
    }

    fn commit(&self, next: Location) {
        {
            let mut history = self.history.borrow_mut();
            let cursor = history.cursor;
            history.stack.truncate(cursor + 1);
            history.stack.push(next.clone());
            history.cursor = history.stack.len() - 1;
        }
        self.location.set(next);
    }

    /// Starts a guarded navigation and drives its guard once.
    ///
    /// An `Allow` commits the target, `Reject` leaves the current location
    /// unchanged, and `Redirect` commits the redirect while remembering the
    /// original complete location for later resumption. A `Pending` result is
    /// retained inside the router and must be advanced with
    /// [`Router::poll_navigation`].
    pub fn navigate_guarded(
        &self,
        path: &str,
        guard: impl Guard + 'static,
    ) -> Result<GuardPoll, NavigationError> {
        self.start_guarded(Location::new(path), guard, false)
    }

    fn start_guarded(
        &self,
        target: Location,
        guard: impl Guard + 'static,
        resume_deep_link: bool,
    ) -> Result<GuardPoll, NavigationError> {
        let mut pending = self.pending.borrow_mut();
        if pending.is_some() {
            return Err(NavigationError::PendingNavigation);
        }
        *pending = Some(QueuedNavigation {
            navigation: PendingNavigation::new(target, guard),
            resume_deep_link,
        });
        drop(pending);
        let Some(poll) = self.poll_navigation() else {
            return Err(NavigationError::PendingNavigation);
        };
        Ok(poll)
    }

    /// Advances the currently queued guarded navigation, if one exists.
    ///
    /// The returned outcome is applied exactly once when it becomes ready.
    /// `None` means there is no guarded navigation in flight.
    pub fn poll_navigation(&self) -> Option<GuardPoll> {
        let result = {
            let mut pending = self.pending.borrow_mut();
            pending.as_mut().map(|queued| queued.navigation.poll())
        }?;
        if matches!(result, GuardPoll::Pending) {
            return Some(result);
        }

        let queued = self.pending.borrow_mut().take()?;
        let target = queued.navigation.target().clone();
        let GuardPoll::Ready(outcome) = result else {
            return Some(GuardPoll::Pending);
        };
        match &outcome {
            GuardOutcome::Allow => {
                self.commit(target.clone());
                if queued.resume_deep_link {
                    self.deep_link.borrow_mut().take();
                }
            }
            GuardOutcome::Redirect(path) => {
                self.deep_link.borrow_mut().remember(target);
                self.commit(Location::new(path));
            }
            GuardOutcome::Reject => {}
        }
        Some(GuardPoll::Ready(outcome))
    }

    /// Returns whether a guarded navigation is waiting for another poll.
    #[must_use]
    pub fn has_pending_navigation(&self) -> bool {
        self.pending.borrow().is_some()
    }

    /// Returns the target of the currently queued guarded navigation.
    #[must_use]
    pub fn pending_navigation_target(&self) -> Option<Location> {
        self.pending
            .borrow()
            .as_ref()
            .map(|queued| queued.navigation.target().clone())
    }

    /// Returns the remembered location captured before a guard redirect.
    #[must_use]
    pub fn deep_link(&self) -> Option<Location> {
        self.deep_link.borrow().peek().cloned()
    }

    /// Takes and clears the remembered deep link without navigating.
    pub fn take_deep_link(&self) -> Option<Location> {
        self.deep_link.borrow_mut().take()
    }

    /// Re-runs a guard for the remembered deep link, if any.
    ///
    /// The deep link remains remembered while the guard is pending or rejects.
    /// It is consumed only after the resumed navigation is allowed.
    pub fn resume_deep_link(
        &self,
        guard: impl Guard + 'static,
    ) -> Result<Option<GuardPoll>, NavigationError> {
        let Some(target) = self.deep_link() else {
            return Ok(None);
        };
        self.start_guarded(target, guard, true).map(Some)
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
    use crate::guard::{Guard, GuardOutcome, GuardPoll};
    use crate::matcher::{RouteId, RouteTable};
    use crate::{Location, NavigationError};

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

    #[test]
    fn allowed_guard_commits_only_after_guard_decision() {
        let rt = Runtime::new();
        let router = Router::new(&rt, "/", table());
        struct AllowOnSecondPoll {
            polls: u32,
        }
        impl Guard for AllowOnSecondPoll {
            fn poll(&mut self, _location: &Location) -> GuardPoll {
                self.polls += 1;
                if self.polls == 1 {
                    GuardPoll::Pending
                } else {
                    GuardPoll::Ready(GuardOutcome::Allow)
                }
            }
        }
        assert_eq!(
            router.navigate_guarded("/users/42", AllowOnSecondPoll { polls: 0 }),
            Ok(GuardPoll::Pending)
        );
        assert_eq!(
            router.poll_navigation(),
            Some(GuardPoll::Ready(GuardOutcome::Allow))
        );
        assert_eq!(router.location().path(), "/users/42");
    }

    #[test]
    fn redirect_remembers_full_deep_link_and_resume_consumes_it() {
        let rt = Runtime::new();
        let router = Router::new(&rt, "/", table());
        let result = router.navigate_guarded(
            "/users/42?tab=activity#comments",
            crate::guard::FnGuard::new(|_| GuardOutcome::Redirect("/login".into())),
        );
        assert_eq!(
            result,
            Ok(GuardPoll::Ready(GuardOutcome::Redirect("/login".into())))
        );
        assert_eq!(router.location().path(), "/login");
        assert_eq!(
            router
                .deep_link()
                .map(|location| location.as_str().to_string()),
            Some("/users/42?tab=activity#comments".to_string())
        );

        assert_eq!(
            router.resume_deep_link(crate::guard::FnGuard::new(|_| GuardOutcome::Allow)),
            Ok(Some(GuardPoll::Ready(GuardOutcome::Allow)))
        );
        assert_eq!(
            router.location().as_str(),
            "/users/42?tab=activity#comments"
        );
        assert_eq!(router.deep_link(), None);
    }

    #[test]
    fn rejected_guard_preserves_location_and_redirected_deep_link() {
        let rt = Runtime::new();
        let router = Router::new(&rt, "/login", table());
        let first = router.navigate_guarded(
            "/users/42",
            crate::guard::FnGuard::new(|_| GuardOutcome::Redirect("/login".into())),
        );
        assert!(first.is_ok());
        let result = router.resume_deep_link(crate::guard::FnGuard::new(|_| GuardOutcome::Reject));
        assert_eq!(result, Ok(Some(GuardPoll::Ready(GuardOutcome::Reject))));
        assert_eq!(router.location().path(), "/login");
        assert_eq!(
            router
                .deep_link()
                .map(|location| location.path().to_string()),
            Some("/users/42".to_string())
        );
    }

    #[test]
    fn second_guarded_navigation_is_rejected_while_one_is_pending() {
        let rt = Runtime::new();
        let router = Router::new(&rt, "/", table());
        let pending = crate::guard::ManualGuard::new();
        assert_eq!(
            router.navigate_guarded("/users/1", pending),
            Ok(GuardPoll::Pending)
        );
        assert_eq!(
            router.navigate_guarded(
                "/users/2",
                crate::guard::FnGuard::new(|_| GuardOutcome::Allow)
            ),
            Err(NavigationError::PendingNavigation)
        );
        assert_eq!(
            router
                .pending_navigation_target()
                .map(|location| location.path().to_string()),
            Some("/users/1".to_string())
        );
    }

    #[test]
    fn direct_navigation_supersedes_pending_guard_and_stale_deep_link() {
        let rt = Runtime::new();
        let router = Router::new(&rt, "/", table());
        assert_eq!(
            router.navigate_guarded(
                "/users/1",
                crate::guard::FnGuard::new(|_| GuardOutcome::Redirect("/login".into()))
            ),
            Ok(GuardPoll::Ready(GuardOutcome::Redirect("/login".into())))
        );
        assert!(router.deep_link().is_some());

        assert_eq!(
            router.navigate_guarded("/users/2", crate::guard::ManualGuard::new()),
            Ok(GuardPoll::Pending)
        );
        router.navigate("/");
        assert_eq!(router.location().path(), "/");
        assert!(!router.has_pending_navigation());
        assert_eq!(router.deep_link(), None);
        assert_eq!(router.poll_navigation(), None);
    }
}
