use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::{prelude::*, resource::Resource};
use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_driver::FrameSync;
use prism_render_driver_wgpu::WgpuQueue;

use crate::{diagnostics::GpuSceneDiagnostics, gpu::PrismGpu, scene::RenderGpuScene};

/// Tracks submitted and completed queue work without fixed frame delays.
#[derive(Resource, Clone)]
pub struct GpuCompletionTracker {
    submitted: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
}

impl Default for GpuCompletionTracker {
    fn default() -> Self {
        Self {
            submitted: Arc::new(AtomicU64::new(0)),
            completed: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl GpuCompletionTracker {
    pub fn next_submission_value(&self) -> GpuCompletionValue {
        GpuCompletionValue(self.submitted.load(Ordering::Acquire) + 1)
    }

    pub fn completed_value(&self) -> GpuCompletionValue {
        GpuCompletionValue(self.completed.load(Ordering::Acquire))
    }

    pub fn track(&self, queue: &WgpuQueue) -> GpuCompletionValue {
        let value = self.submitted.fetch_add(1, Ordering::AcqRel) + 1;
        let completed = Arc::clone(&self.completed);
        // Non-blocking completion: advance the tracker when the queue drains.
        queue.on_submitted_work_done(Box::new(move || {
            completed.fetch_max(value, Ordering::Release);
        }));
        GpuCompletionValue(value)
    }
}

pub(crate) fn track_submission(tracker: Res<GpuCompletionTracker>, gpu: Option<Res<PrismGpu>>) {
    // No Prism GPU context (GPU-less host): nothing is submitted through the
    // Prism queue this frame, so there is nothing to track.
    if let Some(gpu) = gpu {
        tracker.track(gpu.queue());
    }
}

pub(crate) fn reclaim_completed_handles(
    mut scene: ResMut<RenderGpuScene>,
    tracker: Res<GpuCompletionTracker>,
    mut diagnostics: ResMut<GpuSceneDiagnostics>,
) {
    diagnostics.reclaimed_handles = diagnostics
        .reclaimed_handles
        .saturating_add(scene.reclaim_completed(tracker.completed_value()));
}
