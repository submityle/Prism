//! Per-frame observability for the render sub-app.

use std::time::Duration;

use prism_ecs::prelude::{Entity, Resource};
use prism_render_architecture::diagnostics::{FrameCounters, FrameStats, FrameWindow};

use crate::intent::RenderViewport;

/// Counters the render sub-app publishes about the work it has done.
///
/// This is the render world's honest account of what actually happened: how
/// many frames were recorded and submitted, how many failed, how many passes
/// each frame declared, and the viewport of the most recent one. Tests and
/// diagnostics read it through [`App::get_sub_app`](prism_app::App::get_sub_app).
#[derive(Clone, Debug, PartialEq, Resource)]
pub struct RenderFrameStats {
    /// Frames whose graph compiled, recorded, and submitted successfully.
    pub frames_submitted: u64,
    /// Render attempts rejected during validation, preparation, graph compilation,
    /// recording or submission. Preparation may already have submitted simulation
    /// work; this is not a count of attempts with zero GPU activity.
    pub frames_failed: u64,
    /// Shared-frame admission failure from the latest render attempt.
    pub fabric_frame_error: Option<crate::fabric_frame::NativeFabricFrameAdmissionError>,
    /// Legacy aggregate diagnostic for an entirely absent visibility-view set.
    /// Camera-local admission failures and batch preparation errors are reported
    /// by [`SurfaceViewQueues`](crate::SurfaceViewQueues). `None` does not prove
    /// that every camera was accepted or that any draw/GPU pass was produced.
    pub view_queue_rejection: Option<ViewQueueAdmissionRejection>,
    /// Batch-wide camera queue, target or graph failure from the latest attempt.
    pub surface_view_error: Option<SurfaceViewExecutionError>,
    /// Cameras excluded from individual outputs because their rect does not fit.
    /// A rejected camera does not discard other cameras or explicit queues.
    pub surface_view_output_rejections: Vec<SurfaceViewOutputRejection>,
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
    /// Number of frames whose unified lighting/GI plan was compiled.
    pub lighting_plan_compiles: u64,
    /// Number of recipes admitted by the most recent plan.
    pub lighting_recipes_selected: u64,
    /// Number of recipes rejected by the most recent plan with a reason.
    pub lighting_recipes_rejected: u64,
    /// Number of recipe execution declarations admitted by the plan.
    pub lighting_executions_planned: u64,
    /// Number of cache invalidation effects observed by the latest plan.
    pub lighting_cache_invalidations: u64,
    /// Number of plan-to-DebugView snapshot synchronizations.
    pub lighting_debug_updates: u64,
    /// Number of selected recipes in the latest DebugView snapshot.
    pub lighting_debug_selected_recipes: u64,
    /// Number of structured rejection explanations in the latest DebugView snapshot.
    pub lighting_debug_rejections: u64,
    /// Frames for which the selected lighting plan recorded a real GPU pass.
    pub lighting_gpu_passes_recorded: u64,
    /// Canonical light-scene conversion failure for the latest attempted frame.
    /// A present value means that frame was rejected before GPU recording.
    pub lighting_input_error: Option<crate::lighting_scene::LightingSceneExtractError>,
    /// Stable-slot table preparation failure for the latest attempted frame.
    pub lighting_upload_error: Option<crate::lighting_gpu::LightingGpuPrepareError>,
    /// Trace compute dispatches recorded into host frame graphs.
    pub trace_gpu_dispatches_recorded: u64,
    /// Ember graph inputs successfully uploaded during preparation.
    pub ember_graphs_updated: u64,
    /// Ember updates that kept an older valid graph resident.
    pub ember_graphs_preserved: u64,
    /// Ember graph inputs that failed before any valid graph existed.
    pub ember_initial_failures: u64,
    /// Ember preparation error category from the latest attempted frame.
    pub ember_prepare_error: Option<crate::particle::EmberPrepareErrorKind>,
    /// Ember preparation outcome from the latest attempted frame.
    pub ember_prepare_outcome: crate::particle::EmberPrepareOutcome,
    /// Compute passes recorded for Ember graphs.
    pub ember_compute_passes_recorded: u64,
    /// Rolling CPU render-frame samples. GPU fields remain absent until the
    /// backend supplies timestamp and residency measurements.
    pub frame_window: FrameWindow,
    /// Most recent CPU render-frame duration in milliseconds.
    pub latest_cpu_frame_ms: Option<f32>,
    /// The viewport of the most recently recorded frame, empty or not.
    pub last_viewport: RenderViewport,
    /// The receipt for the most recently attempted frame graph execution.
    pub(crate) execution_trace: crate::systems::FrameExecutionTrace,
}

/// A failure that prevents safely consuming this frame's camera bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceViewExecutionError {
    /// Queue preflight failed; no camera bindings from this batch may execute.
    QueuePreparation(crate::SurfaceViewQueueError),
    /// Camera bindings were not published for this extraction/device generation.
    StaleQueues,
    /// Allocation or retirement could not complete safely.
    Target(crate::surface_view_targets::SurfaceViewTargetError),
    /// Target/lighting declarations failed the graph's admission contract.
    GraphAdmission,
}

/// One camera whose output-space viewport is outside a selected output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceViewOutputRejection {
    /// Full camera identity, not an index-only match.
    pub camera: Entity,
    /// Output on which this camera could not be rendered.
    pub output: crate::surface_view_targets::SurfaceViewOutput,
    /// Camera rectangle `[x, y, width, height]`.
    pub viewport: [u32; 4],
    /// Available physical output size.
    pub output_viewport: RenderViewport,
}

/// Legacy aggregate diagnostic retained for visibility producers with no views.
///
/// Multiple cameras are supported. Inspect [`SurfaceViewQueues`](crate::SurfaceViewQueues)
/// for camera-local rejections and whole-batch preparation failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewQueueAdmissionRejection {
    /// No camera produced a validated visibility snapshot, including empty work.
    MissingVisibilityView,
}

impl Default for RenderFrameStats {
    fn default() -> Self {
        Self {
            frames_submitted: 0,
            frames_failed: 0,
            fabric_frame_error: None,
            view_queue_rejection: None,
            surface_view_error: None,
            surface_view_output_rejections: Vec::new(),
            frames_skipped: 0,
            passes_declared: 0,
            draws_queued: 0,
            draws_recorded: 0,
            lighting_plan_compiles: 0,
            lighting_recipes_selected: 0,
            lighting_recipes_rejected: 0,
            lighting_executions_planned: 0,
            lighting_cache_invalidations: 0,
            lighting_debug_updates: 0,
            lighting_debug_selected_recipes: 0,
            lighting_debug_rejections: 0,
            lighting_gpu_passes_recorded: 0,
            lighting_input_error: None,
            lighting_upload_error: None,
            trace_gpu_dispatches_recorded: 0,
            ember_graphs_updated: 0,
            ember_graphs_preserved: 0,
            ember_initial_failures: 0,
            ember_prepare_error: None,
            ember_prepare_outcome: crate::particle::EmberPrepareOutcome::default(),
            ember_compute_passes_recorded: 0,
            frame_window: FrameWindow::new(240),
            latest_cpu_frame_ms: None,
            last_viewport: RenderViewport::default(),
            execution_trace: crate::systems::FrameExecutionTrace::default(),
        }
    }
}

impl RenderFrameStats {
    /// Records the render schedule's measured CPU duration.
    pub fn record_cpu_frame(&mut self, frame_index: u64, elapsed: Duration) {
        let milliseconds = (elapsed.as_secs_f64() * 1_000.0) as f32;
        if milliseconds.is_finite() {
            self.latest_cpu_frame_ms = Some(milliseconds);
            self.frame_window
                .push(FrameCounters::new(frame_index, milliseconds, None, None));
        }
    }

    /// Returns the rolling render-frame statistics.
    #[must_use]
    pub fn frame_stats(&self) -> FrameStats {
        self.frame_window.stats()
    }

    /// Total frames the render schedule has run (submitted + failed + skipped).
    #[must_use]
    pub const fn frames_recorded(&self) -> u64 {
        self.frames_submitted + self.frames_failed + self.frames_skipped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_render_sample_is_recorded_without_fabricating_gpu_values() {
        let mut stats = RenderFrameStats::default();
        stats.record_cpu_frame(7, Duration::from_millis(12));

        assert_eq!(stats.latest_cpu_frame_ms, Some(12.0));
        let frame_stats = stats.frame_stats();
        assert_eq!(frame_stats.sample_count, 1);
        assert_eq!(frame_stats.cpu_frame_ms.sample_count, 1);
        assert_eq!(frame_stats.gpu_frame_ms.sample_count, 0);
        assert_eq!(frame_stats.gpu_memory_bytes.sample_count, 0);
        assert_eq!(
            stats
                .frame_window
                .iter()
                .next()
                .map(|sample| sample.frame_index),
            Some(7)
        );
    }
}
