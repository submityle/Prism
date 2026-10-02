//! Navigation guards and recoverable deep-link state.
//!
//! A guard decides whether a navigation to a target [`Location`] is allowed. It
//! answers with a [`GuardOutcome`]: proceed, redirect elsewhere, or reject
//! outright. Because this crate is `no_std` and runs without a thread-based
//! async runtime, "asynchronous" guards (e.g. ones that await a permission
//! check) are modelled as explicitly-driven state machines: a [`Guard`] is
//! polled repeatedly and returns [`GuardPoll::Pending`] until it settles on a
//! [`GuardPoll::Ready`] outcome.
//!
//! [`PendingNavigation`] bundles a target location with its guard and tracks
//! the resolution, so a caller can drive it one poll at a time. [`DeepLinkState`]
//! captures where the user *intended* to go when a guard redirects them (the
//! classic "bounce through login, then return to the deep link" flow), so the
//! navigation can be resumed later.

use alloc::boxed::Box;
use alloc::string::String;

use crate::path::Location;

/// The decision a guard reaches for a navigation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuardOutcome {
    /// Allow the navigation to proceed to the target location.
    Allow,
    /// Reject the navigation and send the user to this path instead.
    Redirect(String),
    /// Reject the navigation outright, staying where we are.
    Reject,
}

impl GuardOutcome {
    /// Returns `true` for [`GuardOutcome::Allow`].
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        matches!(self, GuardOutcome::Allow)
    }

    /// Returns the redirect path for [`GuardOutcome::Redirect`], else `None`.
    #[must_use]
    pub fn redirect_target(&self) -> Option<&str> {
        match self {
            GuardOutcome::Redirect(path) => Some(path.as_str()),
            _ => None,
        }
    }
}

/// The result of polling a [`Guard`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuardPoll {
    /// The guard has not reached a decision yet; poll again later.
    Pending,
    /// The guard has settled on an outcome.
    Ready(GuardOutcome),
}

/// A navigation guard that is driven by explicit polling.
///
/// Synchronous guards return [`GuardPoll::Ready`] on the first poll; guards that
/// model asynchronous work return [`GuardPoll::Pending`] until their underlying
/// condition settles.
pub trait Guard {
    /// Advances the guard against `location`, returning its current state.
    fn poll(&mut self, location: &Location) -> GuardPoll;
}

/// A guard built from a closure that decides synchronously.
///
/// The closure is invoked on every poll and its [`GuardOutcome`] is returned
/// immediately as [`GuardPoll::Ready`].
pub struct FnGuard<F> {
    decide: F,
}

impl<F> FnGuard<F>
where
    F: FnMut(&Location) -> GuardOutcome,
{
    /// Wraps `decide` as a synchronous [`Guard`].
    #[must_use]
    pub fn new(decide: F) -> Self {
        Self { decide }
    }
}

impl<F> Guard for FnGuard<F>
where
    F: FnMut(&Location) -> GuardOutcome,
{
    fn poll(&mut self, location: &Location) -> GuardPoll {
        GuardPoll::Ready((self.decide)(location))
    }
}

/// A guard whose decision is supplied externally, modelling asynchronous work.
///
/// It reports [`GuardPoll::Pending`] until [`ManualGuard::resolve`] is called,
/// after which every poll yields the resolved outcome. This lets tests and
/// explicit frame loops simulate a guard that completes across several ticks
/// without any real concurrency.
#[derive(Clone, Debug, Default)]
pub struct ManualGuard {
    outcome: Option<GuardOutcome>,
}

impl ManualGuard {
    /// Creates an unresolved guard that is initially [`GuardPoll::Pending`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Supplies the outcome this guard will report from now on.
    pub fn resolve(&mut self, outcome: GuardOutcome) {
        self.outcome = Some(outcome);
    }

    /// Returns `true` once an outcome has been supplied.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.outcome.is_some()
    }
}

impl Guard for ManualGuard {
    fn poll(&mut self, _location: &Location) -> GuardPoll {
        match &self.outcome {
            Some(outcome) => GuardPoll::Ready(outcome.clone()),
            None => GuardPoll::Pending,
        }
    }
}

/// A navigation whose guard is still being driven to a decision.
///
/// Construct one with [`PendingNavigation::new`] and advance it with
/// [`PendingNavigation::poll`]. Once resolved, the outcome is cached so further
/// polls are cheap and stable.
pub struct PendingNavigation {
    target: Location,
    guard: Box<dyn Guard>,
    outcome: Option<GuardOutcome>,
}

impl PendingNavigation {
    /// Begins a guarded navigation to `target` driven by `guard`.
    #[must_use]
    pub fn new(target: Location, guard: impl Guard + 'static) -> Self {
        Self {
            target,
            guard: Box::new(guard),
            outcome: None,
        }
    }

    /// The location this navigation is trying to reach.
    #[must_use]
    pub fn target(&self) -> &Location {
        &self.target
    }

    /// Advances the guard, returning the current [`GuardPoll`].
    ///
    /// Once the guard resolves, the outcome is cached and returned on every
    /// subsequent poll.
    pub fn poll(&mut self) -> GuardPoll {
        if let Some(outcome) = &self.outcome {
            return GuardPoll::Ready(outcome.clone());
        }
        match self.guard.poll(&self.target) {
            GuardPoll::Pending => GuardPoll::Pending,
            GuardPoll::Ready(outcome) => {
                self.outcome = Some(outcome.clone());
                GuardPoll::Ready(outcome)
            }
        }
    }

    /// The resolved outcome, if the guard has settled.
    #[must_use]
    pub fn outcome(&self) -> Option<&GuardOutcome> {
        self.outcome.as_ref()
    }

    /// Returns `true` once the guard has resolved.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        self.outcome.is_some()
    }
}

/// Recoverable deep-link state: the location a user was trying to reach before
/// a guard diverted them.
///
/// When a guard returns [`GuardOutcome::Redirect`] (for example to a login
/// screen), [`DeepLinkState::remember`] stores the intended target. After the
/// blocking condition clears, [`DeepLinkState::take`] hands it back so the app
/// can resume the original navigation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeepLinkState {
    intended: Option<Location>,
}

impl DeepLinkState {
    /// Creates empty deep-link state with nothing to resume.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `location` as the target to resume later, replacing any prior
    /// pending target.
    pub fn remember(&mut self, location: Location) {
        self.intended = Some(location);
    }

    /// Borrows the remembered target, if any, without consuming it.
    #[must_use]
    pub fn peek(&self) -> Option<&Location> {
        self.intended.as_ref()
    }

    /// Removes and returns the remembered target, if any.
    pub fn take(&mut self) -> Option<Location> {
        self.intended.take()
    }

    /// Returns `true` when a target is waiting to be resumed.
    #[must_use]
    pub fn has_target(&self) -> bool {
        self.intended.is_some()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and reuse its String"
    )]

    use super::{
        DeepLinkState, FnGuard, Guard, GuardOutcome, GuardPoll, ManualGuard, PendingNavigation,
    };
    use crate::path::Location;
    use alloc::string::ToString;

    #[test]
    fn fn_guard_allows_immediately() {
        let mut guard = FnGuard::new(|_loc: &Location| GuardOutcome::Allow);
        assert_eq!(
            guard.poll(&Location::new("/dashboard")),
            GuardPoll::Ready(GuardOutcome::Allow)
        );
    }

    #[test]
    fn fn_guard_can_inspect_location() {
        let mut guard = FnGuard::new(|loc: &Location| {
            if loc.path().starts_with("/admin") {
                GuardOutcome::Redirect("/login".to_string())
            } else {
                GuardOutcome::Allow
            }
        });
        assert_eq!(
            guard.poll(&Location::new("/admin/users")),
            GuardPoll::Ready(GuardOutcome::Redirect("/login".to_string()))
        );
        assert_eq!(
            guard.poll(&Location::new("/home")),
            GuardPoll::Ready(GuardOutcome::Allow)
        );
    }

    #[test]
    fn manual_guard_pends_until_resolved() {
        let mut guard = ManualGuard::new();
        let loc = Location::new("/secret");
        assert_eq!(guard.poll(&loc), GuardPoll::Pending);
        assert!(!guard.is_resolved());
        guard.resolve(GuardOutcome::Reject);
        assert!(guard.is_resolved());
        assert_eq!(guard.poll(&loc), GuardPoll::Ready(GuardOutcome::Reject));
    }

    #[test]
    fn guard_outcome_helpers() {
        assert!(GuardOutcome::Allow.is_allowed());
        assert!(!GuardOutcome::Reject.is_allowed());
        assert_eq!(
            GuardOutcome::Redirect("/x".to_string()).redirect_target(),
            Some("/x")
        );
        assert_eq!(GuardOutcome::Allow.redirect_target(), None);
    }

    #[test]
    fn pending_navigation_drives_async_guard() {
        let mut guard = ManualGuard::new();
        // Pre-resolve a guard, then wrap it: it should settle on first poll.
        guard.resolve(GuardOutcome::Allow);
        let mut nav = PendingNavigation::new(Location::new("/profile"), guard);
        assert_eq!(nav.target().path(), "/profile");
        assert_eq!(nav.poll(), GuardPoll::Ready(GuardOutcome::Allow));
        assert!(nav.is_resolved());
        assert_eq!(nav.outcome(), Some(&GuardOutcome::Allow));
    }

    #[test]
    fn pending_navigation_stays_pending_then_resolves() {
        // A guard that stays pending for two polls, then allows.
        struct Delayed {
            ticks: u32,
        }
        impl Guard for Delayed {
            fn poll(&mut self, _loc: &Location) -> GuardPoll {
                if self.ticks == 0 {
                    GuardPoll::Ready(GuardOutcome::Allow)
                } else {
                    self.ticks -= 1;
                    GuardPoll::Pending
                }
            }
        }

        let mut nav = PendingNavigation::new(Location::new("/slow"), Delayed { ticks: 2 });
        assert_eq!(nav.poll(), GuardPoll::Pending);
        assert_eq!(nav.poll(), GuardPoll::Pending);
        assert_eq!(nav.poll(), GuardPoll::Ready(GuardOutcome::Allow));
        // Cached afterwards.
        assert_eq!(nav.poll(), GuardPoll::Ready(GuardOutcome::Allow));
    }

    #[test]
    fn deep_link_round_trip() {
        let mut deep = DeepLinkState::new();
        assert!(!deep.has_target());
        assert_eq!(deep.take(), None);

        deep.remember(Location::new("/orders/42"));
        assert!(deep.has_target());
        assert_eq!(deep.peek().map(Location::path), Some("/orders/42"));

        let resumed = deep.take().expect("intended target");
        assert_eq!(resumed.path(), "/orders/42");
        assert!(!deep.has_target());
    }

    #[test]
    fn deep_link_latest_remember_wins() {
        let mut deep = DeepLinkState::new();
        deep.remember(Location::new("/a"));
        deep.remember(Location::new("/b"));
        assert_eq!(
            deep.take().map(|l| l.path().to_string()),
            Some("/b".to_string())
        );
    }

    #[test]
    fn redirect_flow_remembers_and_resumes() {
        // Simulate: user hits a guarded page, guard redirects to login,
        // we stash the intended target, later resume it.
        let mut deep = DeepLinkState::new();
        let target = Location::new("/account/settings");
        let mut guard =
            FnGuard::new(|_loc: &Location| GuardOutcome::Redirect("/login".to_string()));

        if let GuardPoll::Ready(GuardOutcome::Redirect(_)) = guard.poll(&target) {
            deep.remember(target);
        }
        assert!(deep.has_target());
        assert_eq!(
            deep.take().map(|l| l.path().to_string()),
            Some("/account/settings".to_string())
        );
    }
}
