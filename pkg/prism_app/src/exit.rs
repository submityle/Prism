//! Application exit signalling.
//!
//! M0 models exit as a world resource: a system requests shutdown through
//! [`AppExitRequest`], and the runner stops once a request is present. The exit
//! carries an [`AppExit`] code so a process can map a clean quit vs. an error
//! to an OS exit status.
//!
//! # Honestly deferred
//!
//! Bevy-style `EventWriter<AppExit>` ergonomics (an event, not a resource) and
//! the full graceful-shutdown path — drain commands, run `Last`, run plugin
//! `cleanup` in reverse (design §12 / §21) — are M1/M4. M0 provides the
//! resource-based request and an immediate, honest stop; it does not pretend to
//! run a graceful drain it has not implemented.

use prism_ecs::resource::Resource;

/// Why an [`App`](crate::app::App) stopped running.
///
/// Returned by [`App::run`](crate::app::App::run) and by a runner so a host
/// `main` can translate it to a process exit code.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AppExit {
    /// The app exited cleanly.
    #[default]
    Success,
    /// The app exited with an error, carrying a non-zero code.
    Error(core::num::NonZeroU8),
}

impl AppExit {
    /// An error exit with code `1`.
    pub const fn error() -> Self {
        // SAFETY-free: 1 is a valid non-zero u8.
        AppExit::Error(core::num::NonZeroU8::new(1).unwrap())
    }

    /// Whether this exit represents success.
    pub const fn is_success(self) -> bool {
        matches!(self, AppExit::Success)
    }
}

/// A world resource systems use to request that the [`App`](crate::app::App)
/// stop running.
///
/// Installed into the main world by [`App::new`](crate::app::App::new). A system
/// takes `ResMut<AppExitRequest>` and calls [`send`](AppExitRequest::send) (or
/// [`send_success`](AppExitRequest::send_success)); the runner observes the
/// request between frames via [`App::should_exit`](crate::app::App::should_exit).
#[derive(Default)]
pub struct AppExitRequest {
    requested: Option<AppExit>,
}

impl Resource for AppExitRequest {}

impl AppExitRequest {
    /// Request exit with the given code. The first request wins; later requests
    /// are ignored so a specific error code is not overwritten by a later
    /// success.
    pub fn send(&mut self, exit: AppExit) {
        if self.requested.is_none() {
            self.requested = Some(exit);
        }
    }

    /// Request a clean exit ([`AppExit::Success`]).
    pub fn send_success(&mut self) {
        self.send(AppExit::Success);
    }

    /// Request an error exit ([`AppExit::error`]).
    pub fn send_error(&mut self) {
        self.send(AppExit::error());
    }

    /// The pending exit request, if any.
    pub fn get(&self) -> Option<AppExit> {
        self.requested
    }
}
