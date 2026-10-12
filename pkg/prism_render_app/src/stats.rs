//! Per-frame observability for the render sub-app.

use prism_ecs::prelude::Resource;

use crate::intent::RenderViewport;

/// Counters the render sub-app publishes about the work it has done.
///
/// This is the render world's honest account of what actually happened: how
/// many frames were recorded and submitted, how many failed, how many passes
/// each frame declared, and the viewport of the most recent one. Tests and
/// diagnostics read it through [`App::get_sub_app`](prism_app::App::get_sub_app).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Resource)]
pub struct RenderFrameStats {
    /// Frames whose graph compiled, recorded, and submitted successfully.
    pub frames_submitted: u64,
    /// Frames whose graph failed to compile; no GPU work was issued for them.
    pub frames_failed: u64,
    /// Frames skipped because the extracted viewport was empty.
    pub frames_skipped: u64,
    /// The total number of passes declared across every recorded frame.
    pub passes_declared: u64,
    /// The total number of draws the queue stage produced.
    ///
    /// Separate from [`draws_recorded`](Self::draws_recorded) because they answer
    /// different questions: a frame can queue draws and fail before recording
    /// them, and a persistent gap between the two is how that shows up.
    pub draws_queued: u64,
    /// The total number of queued draws recorded across every recorded frame.
    pub draws_recorded: u64,
    /// The viewport of the most recently recorded frame, empty or not.
    pub last_viewport: RenderViewport,
}

impl RenderFrameStats {
    /// Total frames the render schedule has run (submitted + failed + skipped).
    #[must_use]
    pub const fn frames_recorded(&self) -> u64 {
        self.frames_submitted + self.frames_failed + self.frames_skipped
    }
}
