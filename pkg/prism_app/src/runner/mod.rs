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
//! # Lifecycle forwarding
//!
//! The windowed runner forwards the platform lifecycle events (design §12): the
//! winit host trait's one-shot `resumed` and repeatable `suspended` /
//! `low_memory` callbacks become [`Resumed`](crate::lifecycle::Resumed),
//! [`Suspended`](crate::lifecycle::Suspended), and
//! [`LowMemory`](crate::lifecycle::LowMemory) app events. A platform
//! *device-loss* signal (GPU context loss / swapchain recreation) is still
//! absent from the renderer seam, so device-loss recovery remains deferred
//! rather than stubbed.

#[cfg(feature = "std")]
mod dedicated_server;
mod headless;
mod once;
#[cfg(feature = "winit")]
mod winit;
#[cfg(feature = "winit")]
mod window_bridge;
#[cfg(feature = "winit")]
pub use window_bridge::WindowRenderBridge;

#[cfg(feature = "std")]
pub use dedicated_server::{DedicatedServerRunner, ServerTickDiagnostics};
pub use headless::HeadlessRunner;
pub use once::{run_once, ScheduleRunnerOnce};
#[cfg(feature = "winit")]
pub use winit::{run_windowed, window_actions, WindowAction, WindowSettings, WinitRunner};
