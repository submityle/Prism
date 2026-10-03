//! The headless loop runner.

use crate::app::App;
use crate::exit::AppExit;
#[cfg(feature = "std")]
use crate::pacing::{AdaptiveFrameLimiter, FrameLimit, FramePacer, FrameRateLadder};
#[cfg(feature = "std")]
use prism_time::Instant;

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
#[cfg_attr(feature = "std", doc = "into a cap via [`with_frame_limit`](HeadlessRunner::with_frame_limit) — a")]
#[cfg_attr(not(feature = "std"), doc = "into a cap via `with_frame_limit` — a")]
/// mobile-style frame limiter to save power/heat, or a dedicated-server
#[cfg_attr(feature = "std", doc = "tickrate. The cap is applied by a drift-free [`FramePacer`] built inside")]
#[cfg_attr(not(feature = "std"), doc = "tickrate. The cap is applied by a drift-free `FramePacer` built inside")]
/// [`run`](HeadlessRunner::run); it paces to a moving cadence so the achieved
/// rate does not drift (design §13).
#[derive(Clone, Debug, Default)]
pub struct HeadlessRunner {
    /// Stop after this many frames even if no exit was requested. `None` loops
    /// until exit.
    max_frames: Option<u64>,
    /// Optional software frame cap applied once per frame (design §13). `Off`
    /// (the default) keeps the loop uncapped.
    #[cfg(feature = "std")]
    frame_limit: FrameLimit,
    /// Optional histogram-driven adaptive cap selector (design §13). When set,
    /// each frame's work time is fed to it and the resulting cap drives the
    /// pacer, so the loop tracks the highest cadence it can sustain within a
    /// tier ladder. `None` (the default) keeps the fixed `frame_limit`.
    #[cfg(feature = "std")]
    adaptive: Option<AdaptiveFrameLimiter>,
}

impl HeadlessRunner {
    /// A runner that loops until a system requests exit, uncapped.
    pub fn new() -> Self {
        Self {
            max_frames: None,
            #[cfg(feature = "std")]
            frame_limit: FrameLimit::Off,
            #[cfg(feature = "std")]
            adaptive: None,
        }
    }

    /// A runner that stops after at most `max_frames` frames (or earlier if
    /// exit is requested).
    pub fn with_max_frames(max_frames: u64) -> Self {
        Self {
            max_frames: Some(max_frames),
            #[cfg(feature = "std")]
            frame_limit: FrameLimit::Off,
            #[cfg(feature = "std")]
            adaptive: None,
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

    /// Drive the frame cap adaptively over `ladder` (design §13): each frame's
    /// measured work time is fed to an [`AdaptiveFrameLimiter`], which nudges
    /// the cap up or down a rung so the loop targets the highest cadence it can
    /// sustain. The initial cap is the limiter's starting rung (the most
    /// demanding by default). Builder-style; overrides any fixed
    #[cfg_attr(feature = "std", doc = "[`with_frame_limit`](HeadlessRunner::with_frame_limit).")]
    #[cfg_attr(not(feature = "std"), doc = "`with_frame_limit`.")]
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_adaptive_frame_limit(mut self, ladder: FrameRateLadder) -> Self {
        let limiter = AdaptiveFrameLimiter::new(ladder);
        self.frame_limit = limiter.current_limit();
        self.adaptive = Some(limiter);
        self
    }

    /// Drive `app` frame by frame until exit or the frame cap is hit.
    ///
    /// Returns the requested [`AppExit`] if a system asked to stop, otherwise
    /// [`AppExit::Success`] when the frame cap is reached. When a
    #[cfg_attr(feature = "std", doc = "[`FrameLimit`] is set (via [`with_frame_limit`](HeadlessRunner::with_frame_limit)) the loop is paced to it with a drift-free [`FramePacer`].")]
    #[cfg_attr(not(feature = "std"), doc = "`FrameLimit` is set (via `with_frame_limit`) the loop is paced to it with a drift-free `FramePacer`.")]
    pub fn run(self, mut app: App) -> AppExit {
        let max_frames = self.max_frames;
        #[cfg(feature = "std")]
        let mut pacer = FramePacer::new(self.frame_limit);
        #[cfg(feature = "std")]
        let mut adaptive = self.adaptive;
        let mut frame: u64 = 0;
        let exit = loop {
            // Time only the frame's work (not the pacing sleep) so the adaptive
            // limiter sees headroom, not the cadence already in force.
            #[cfg(feature = "std")]
            let work_start = adaptive.as_ref().map(|_| Instant::now());
            app.update();
            // Poll for a *confirmed* exit, running the exit-veto gate when a
            // request is pending (design §24.5); a confirmation system may cancel.
            if let Some(exit) = app.poll_exit() {
                break exit;
            }
            frame = frame.saturating_add(1);
            if max_frames.is_some_and(|max| frame >= max) {
                break AppExit::Success;
            }
            // Feed this frame's work time to the adaptive limiter; a rung change
            // retargets the pacer before it throttles (design §13).
            #[cfg(feature = "std")]
            if let (Some(limiter), Some(start)) = (adaptive.as_mut(), work_start) {
                let work = Instant::now().saturating_duration_since(start);
                if let Some(new_limit) = limiter.record(work) {
                    pacer.set_limit(new_limit);
                }
            }
            // Pace the loop to the configured cap (no-op when unlimited).
            #[cfg(feature = "std")]
            pacer.throttle();
        };
        // Bring any pipelined render frame home before returning so the final
        // frame's render has completed (no-op in the serial / feature-off case).
        app.sync_sub_apps();
        // Run the graceful-shutdown path once (design §12/§21): drain the
        // Shutdown schedule and tear plugins down in reverse order.
        app.run_shutdown();
        exit
    }
}
