//! The single-frame runner.

use crate::app::App;
use crate::exit::AppExit;

/// A runner that drives exactly one frame and then exits.
///
/// Useful for tests and batch/CI tools that want a deterministic single tick.
/// It is also the default runner [`App::run`](crate::app::App::run) uses when
/// none was set, so an un-configured `App::new().run()` terminates instead of
/// spinning forever.
pub struct ScheduleRunnerOnce;

impl ScheduleRunnerOnce {
    /// Run one frame of `app` and return its exit code.
    pub fn run(app: App) -> AppExit {
        run_once(app)
    }
}

/// Run exactly one [`update`](App::update) frame, then return the app's exit
/// request (defaulting to [`AppExit::Success`]).
pub fn run_once(mut app: App) -> AppExit {
    app.update();
    app.should_exit().unwrap_or(AppExit::Success)
}
