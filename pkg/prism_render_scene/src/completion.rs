use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::{prelude::*, resource::Resource};
use bevy_render::renderer::RenderQueue;
use prism_render_architecture::gpu_scene::GpuCompletionValue;

use crate::scene::RenderGpuScene;

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

    fn track(&self, queue: &RenderQueue) {
        let value = self.submitted.fetch_add(1, Ordering::AcqRel) + 1;
        let completed = Arc::clone(&self.completed);
        queue.on_submitted_work_done(move || {
            completed.fetch_max(value, Ordering::Release);
        });
    }
}

pub(crate) fn track_submission(tracker: Res<GpuCompletionTracker>, queue: Res<RenderQueue>) {
    tracker.track(&queue);
}

pub(crate) fn reclaim_completed_handles(
    mut scene: ResMut<RenderGpuScene>,
    tracker: Res<GpuCompletionTracker>,
) {
    scene.reclaim_completed(tracker.completed_value());
}
