//! Runners: strategies that drive an assembled [`App`](crate::app::App) to an
//! [`AppExit`](crate::exit::AppExit).
//!
//! A runner is any `FnOnce(App) -> AppExit`. [`App::run`](crate::app::App::run)
//! hands the finalized app to the runner, which decides *how often* and *for
//! how long* to call [`App::update`](crate::app::App::update). Decoupling the
//! loop from the app is what lets the same engine shell drive a windowed
//! client, a headless server, or a single-frame test.
//!
//! M0 ships two platform-free runners:
//!
//! - [`run_once`] / [`ScheduleRunnerOnce`] — exactly one frame.
//! - [`HeadlessRunner`] — loop until exit (optionally bounded by a frame cap).
//!
//! # Honestly deferred
//!
//! The windowed `WinitRunner` and `DedicatedServerRunner` (design §10 / §24.4)
//! need `prism_window` and the network stack respectively; they are M4/M5 and
//! are intentionally absent, not stubbed.

mod headless;
mod once;

pub use headless::HeadlessRunner;
pub use once::{run_once, ScheduleRunnerOnce};
