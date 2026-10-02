//! Race cancellation for competing async requests ("latest wins").
//!
//! When a view triggers overlapping fetches — e.g. the user types quickly and
//! each keystroke starts a search — results may arrive out of order. Without a
//! guard, a slow earlier request could overwrite a fast later one, showing
//! stale data. Loom models this without threads or futures: every request is
//! issued a monotonically increasing [`RaceToken`] from a shared
//! [`RaceController`], and only a result carrying the *current* token is
//! accepted. Late results from superseded requests are dropped.
//!
//! [`RaceResource`] fuses a [`RaceController`] with a [`Resource`] so the
//! accepted result flows straight into reactive state.

use alloc::rc::Rc;
use core::cell::Cell;

use prism_ui_reactive::Runtime;

use crate::resource::Resource;
use crate::state::AsyncState;

/// A shared counter that hands out monotonically increasing race tokens.
///
/// Clones share the same underlying counter, so tokens minted from any clone
/// participate in the same race. Each [`RaceController::spawn`] supersedes all
/// previously issued tokens.
#[derive(Clone, Default)]
pub struct RaceController {
    latest: Rc<Cell<u64>>,
}

impl RaceController {
    /// Creates a controller with no requests issued yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Issues a fresh [`RaceToken`], superseding every earlier token.
    pub fn spawn(&self) -> RaceToken {
        let next = self.latest.get() + 1;
        self.latest.set(next);
        RaceToken {
            epoch: next,
            latest: Rc::clone(&self.latest),
        }
    }

    /// The epoch of the most recently issued token (`0` before any
    /// [`RaceController::spawn`]).
    #[must_use]
    pub fn latest(&self) -> u64 {
        self.latest.get()
    }
}

/// A token identifying one request in a race.
///
/// A token is *current* only while no later token has been issued. Use
/// [`RaceToken::is_current`] to decide whether a just-arrived result should be
/// accepted or discarded as stale.
#[derive(Clone)]
pub struct RaceToken {
    epoch: u64,
    latest: Rc<Cell<u64>>,
}

impl RaceToken {
    /// The epoch this token was issued at.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Returns `true` while this is still the most recently issued token.
    #[must_use]
    pub fn is_current(&self) -> bool {
        self.epoch == self.latest.get()
    }

    /// Returns `true` once a later token has superseded this one.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        !self.is_current()
    }
}

/// A [`Resource`] guarded by a [`RaceController`] so only the latest request's
/// result is published.
///
/// Call [`RaceResource::begin`] to start a request and obtain its token; feed
/// the eventual result through [`RaceResource::resolve`] or
/// [`RaceResource::fail`], which apply it only if the token is still current.
///
/// # Example
///
/// ```
/// use prism_ui_async::RaceResource;
/// use prism_ui_reactive::Runtime;
///
/// let rt = Runtime::new();
/// let search: RaceResource<i32, &str> = RaceResource::pending(&rt);
///
/// // Two overlapping requests; the second supersedes the first.
/// let first = search.begin();
/// let second = search.begin();
///
/// // The slow first request arrives last and is dropped.
/// assert!(!search.resolve(&first, 1));
/// // Only the latest request's result wins.
/// assert!(search.resolve(&second, 2));
/// ```
pub struct RaceResource<T: Clone + 'static, E: Clone + 'static> {
    resource: Resource<T, E>,
    control: RaceController,
}

impl<T: Clone + 'static, E: Clone + 'static> Clone for RaceResource<T, E> {
    fn clone(&self) -> Self {
        Self {
            resource: self.resource.clone(),
            control: self.control.clone(),
        }
    }
}

impl<T: Clone + 'static, E: Clone + 'static> RaceResource<T, E> {
    /// Creates a race-guarded resource that starts in the pending state.
    #[must_use]
    pub fn pending(rt: &Runtime) -> Self {
        Self {
            resource: Resource::pending(rt),
            control: RaceController::new(),
        }
    }

    /// Borrows the underlying [`Resource`] for reactive reads.
    #[must_use]
    pub fn resource(&self) -> &Resource<T, E> {
        &self.resource
    }

    /// Borrows the [`RaceController`] backing this resource.
    #[must_use]
    pub fn control(&self) -> &RaceController {
        &self.control
    }

    /// Starts a new request: moves the resource back to pending and returns the
    /// token identifying this request.
    pub fn begin(&self) -> RaceToken {
        let token = self.control.spawn();
        self.resource.reload();
        token
    }

    /// Publishes a successful result if `token` is still current.
    ///
    /// Returns `true` when the result was accepted, or `false` when it was
    /// dropped because a newer request had superseded it.
    pub fn resolve(&self, token: &RaceToken, value: T) -> bool {
        if token.is_current() {
            self.resource.resolve(value);
            true
        } else {
            false
        }
    }

    /// Publishes a failure if `token` is still current.
    ///
    /// Returns `true` when the failure was accepted, or `false` when it was
    /// dropped as stale.
    pub fn fail(&self, token: &RaceToken, error: E) -> bool {
        if token.is_current() {
            self.resource.fail(error);
            true
        } else {
            false
        }
    }

    /// Reads a clone of the current state, recording a reactive dependency.
    #[must_use]
    pub fn state(&self) -> AsyncState<T, E> {
        self.resource.state()
    }
}

#[cfg(test)]
mod tests {
    use super::{RaceController, RaceResource};
    use crate::state::AsyncState;
    use prism_ui_reactive::Runtime;

    #[test]
    fn tokens_increase_and_latest_wins() {
        let control = RaceController::new();
        assert_eq!(control.latest(), 0);
        let first = control.spawn();
        let second = control.spawn();
        assert_eq!(first.epoch(), 1);
        assert_eq!(second.epoch(), 2);
        assert!(!first.is_current());
        assert!(first.is_cancelled());
        assert!(second.is_current());
        assert_eq!(control.latest(), 2);
    }

    #[test]
    fn stale_result_is_ignored_fresh_is_applied() {
        let rt = Runtime::new();
        let race: RaceResource<i32, &str> = RaceResource::pending(&rt);

        let first = race.begin();
        let second = race.begin();

        // The first (older) request resolves last; it must be ignored.
        assert!(!race.resolve(&first, 100));
        assert_eq!(race.state(), AsyncState::Pending);

        // The latest request's result wins.
        assert!(race.resolve(&second, 200));
        assert_eq!(race.state(), AsyncState::Ready(200));
    }

    #[test]
    fn latest_result_arriving_first_blocks_older() {
        let rt = Runtime::new();
        let race: RaceResource<i32, &str> = RaceResource::pending(&rt);
        let first = race.begin();
        let second = race.begin();

        assert!(race.resolve(&second, 2));
        assert_eq!(race.state(), AsyncState::Ready(2));
        // The older request now arrives but is dropped.
        assert!(!race.resolve(&first, 1));
        assert_eq!(race.state(), AsyncState::Ready(2));
    }

    #[test]
    fn stale_failure_is_ignored() {
        let rt = Runtime::new();
        let race: RaceResource<i32, &str> = RaceResource::pending(&rt);
        let first = race.begin();
        let _second = race.begin();
        assert!(!race.fail(&first, "boom"));
        assert_eq!(race.state(), AsyncState::Pending);
    }

    #[test]
    fn single_request_resolves_normally() {
        let rt = Runtime::new();
        let race: RaceResource<i32, &str> = RaceResource::pending(&rt);
        let token = race.begin();
        assert!(race.resolve(&token, 7));
        assert_eq!(race.state(), AsyncState::Ready(7));
        assert!(token.is_current());
    }

    #[test]
    fn begin_moves_state_back_to_pending() {
        let rt = Runtime::new();
        let race: RaceResource<i32, &str> = RaceResource::pending(&rt);
        let a = race.begin();
        assert!(race.resolve(&a, 1));
        assert_eq!(race.state(), AsyncState::Ready(1));
        let _b = race.begin();
        assert_eq!(race.state(), AsyncState::Pending);
    }

    #[test]
    fn controller_is_shared_across_clones() {
        let control = RaceController::new();
        let clone = control.clone();
        let t1 = control.spawn();
        let _t2 = clone.spawn();
        // The clone's spawn superseded t1 because they share the counter.
        assert!(t1.is_cancelled());
        assert_eq!(control.latest(), 2);
    }
}
