//! The headless loop runner.

use crate::app::App;
use crate::exit::AppExit;
#[cfg(feature = "std")]
use crate::pacing::{FrameLimit, FramePacer};

/// A runner that repeatedly drives frames with no windowing or presentation,
/// stopping when a system signals exit (via
/// [`AppExitRequest`](crate::exit::AppExitRequest)) or when an optional frame
/// cap is reached.
///
/// This is the server / CI / test workhorse: pure simulation heartbeat with no
/// platform dependencies.
///
/// # Frame pacing
///
/// By default the loop runs as fast as the machine allows (`FrameLimit::Off`),
/// which is what tests and CI want. With the `std` feature a caller can opt
/// into a cap via [`with_frame_limit`](HeadlessRunner::with_frame_limit) — a
/// mobile-style frame limiter to save power/heat, or a dedicated-server
/// tickrate. The cap is applied by a drift-free [`FramePacer`] built inside
/// [`run`](HeadlessRunner::run); it paces to a moving cadence so the achieved
/// rate does not drift (design §13).
#[derive(Clone, Copy, Debug, Default)]
pub struct HeadlessRunner {
    /// Stop after this many frames even if no exit was requested. `None` loops
    /// until exit.
    max_frames: Option<u64>,
    /// Optional software frame cap applied once per frame (design §13). `Off`
    /// (the default) keeps the loop uncapped.
    #[cfg(feature = "std")]
    frame_limit: FrameLimit,
}

impl HeadlessRunner {
    /// A runner that loops until a system requests exit, uncapped.
    pub fn new() -> Self {
        Self {
            max_frames: None,
            #[cfg(feature = "std")]
            frame_limit: FrameLimit::Off,
        }
    }

    /// A runner that stops after at most `max_frames` frames (or earlier if
    /// exit is requested).
    pub fn with_max_frames(max_frames: u64) -> Self {
        Self {
            max_frames: Some(max_frames),
            #[cfg(feature = "std")]
            frame_limit: FrameLimit::Off,
        }
    }

    /// Set a software frame cap applied once per frame after the frame's work
    /// (design §13): a mobile frame limiter or a dedicated-server tickrate.
    /// Builder-style; the default is uncapped.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_frame_limit(mut self, frame_limit: FrameLimit) -> Self {
        self.frame_limit = frame_limit;
        self
    }

    /// Drive `app` frame by frame until exit or the frame cap is hit.
    ///
    /// Returns the requested [`AppExit`] if a system asked to stop, otherwise
    /// [`AppExit::Success`] when the frame cap is reached. When a
    /// [`FrameLimit`] is set (via
    /// [`with_frame_limit`](HeadlessRunner::with_frame_limit)) the loop is
    /// paced to it with a drift-free [`FramePacer`].
    pub fn run(self, mut app: App) -> AppExit {
        #[cfg(feature = "std")]
        let mut pacer = FramePacer::new(self.frame_limit);
        let mut frame: u64 = 0;
        let exit = loop {
            app.update();
            if let Some(exit) = app.should_exit() {
                break exit;
            }
            frame = frame.saturating_add(1);
            if self.max_frames.is_some_and(|max| frame >= max) {
                break AppExit::Success;
            }
            // Pace the loop to the configured cap (no-op when unlimited).
            #[cfg(feature = "std")]
            pacer.throttle();
        };
        // Bring any pipelined render frame home before returning so the final
        // frame's render has completed (no-op in the serial / feature-off case).
        app.sync_sub_apps();
        exit
    }
}
