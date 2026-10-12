//! Runners: strategies that drive an assembled [`App`](crate::app::App) to an
//! [`AppExit`](crate::exit::AppExit).
//!
//! A runner is any `FnOnce(App) -> AppExit`. [`App::run`](crate::app::App::run)
//! hands the finalized app to the runner, which decides *how often* and *for
//! how long* to call [`App::update`](crate::app::App::update). Decoupling the
//! loop from the app is what lets the same engine shell drive a windowed
//! client, a headless server, or a single-frame test.
//!
//! Platform-free runners ship today:
//!
//! - [`run_once`] / [`ScheduleRunnerOnce`] — exactly one frame.
//! - [`HeadlessRunner`] — loop until exit (optionally bounded by a frame
#![cfg_attr(
    feature = "std",
    doc = "  cap, and optionally paced to a [`FrameLimit`](crate::pacing::FrameLimit)"
)]
#![cfg_attr(
    not(feature = "std"),
    doc = "  cap, and optionally paced to a `FrameLimit`"
)]
//!   under `std`, design §13).
#![cfg_attr(
    feature = "std",
    doc = "- [`DedicatedServerRunner`] — an authoritative fixed-tickrate, deterministic"
)]
#![cfg_attr(
    not(feature = "std"),
    doc = "- `DedicatedServerRunner` — an authoritative fixed-tickrate, deterministic"
)]
//!   simulation heartbeat with no rendering (design §10 / §24.4, `std`-only).
//!
//! # Honestly deferred
//!
//! The platform *suspend* / low-memory callbacks (design §12) are still absent:
//! the winit host trait surfaces only a one-shot `resumed`, and `prism_window`
//! carries no low-memory signal yet, so [`winit::WinitRunner`] forwards
//! [`Resumed`](crate::lifecycle::Resumed) but cannot yet forward `Suspended` /
//! `LowMemory`. That seam stays documented rather than stubbed.

#[cfg(feature = "std")]
mod dedicated_server;
mod headless;
mod once;
#[cfg(feature = "winit")]
mod winit;

#[cfg(feature = "std")]
pub use dedicated_server::{DedicatedServerRunner, ServerTickDiagnostics};
pub use headless::HeadlessRunner;
pub use once::{run_once, ScheduleRunnerOnce};
#[cfg(feature = "winit")]
pub use winit::{run_windowed, window_actions, WindowAction, WindowSettings, WinitRunner};
