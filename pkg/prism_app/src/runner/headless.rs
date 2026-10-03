//! The headless loop runner.

use crate::app::App;
use crate::exit::AppExit;

/// A runner that repeatedly drives frames with no windowing or presentation,
/// stopping when a system signals exit (via
/// [`AppExitRequest`](crate::exit::AppExitRequest)) or when an optional frame
/// cap is reached.
///
/// This is the server / CI / test workhorse: pure simulation heartbeat with no
/// platform dependencies.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeadlessRunner {
    /// Stop after this many frames even if no exit was requested. `None` loops
    /// until exit.
    max_frames: Option<u64>,
}

impl HeadlessRunner {
    /// A runner that loops until a system requests exit.
    pub fn new() -> Self {
        Self { max_frames: None }
    }

    /// A runner that stops after at most `max_frames` frames (or earlier if
    /// exit is requested).
    pub fn with_max_frames(max_frames: u64) -> Self {
        Self {
            max_frames: Some(max_frames),
        }
    }

    /// Drive `app` frame by frame until exit or the frame cap is hit.
    ///
    /// Returns the requested [`AppExit`] if a system asked to stop, otherwise
    /// [`AppExit::Success`] when the frame cap is reached.
    pub fn run(self, mut app: App) -> AppExit {
        let mut frame: u64 = 0;
        loop {
            app.update();
            if let Some(exit) = app.should_exit() {
                return exit;
            }
            frame = frame.saturating_add(1);
            if self.max_frames.is_some_and(|max| frame >= max) {
                return AppExit::Success;
            }
        }
    }
}
