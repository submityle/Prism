use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::{prelude::*, resource::Resource};
use bevy_render::renderer::RenderQueue;
use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_driver::FrameSync;
use prism_render_driver_wgpu::WgpuQueue;

use crate::{diagnostics::GpuSceneDiagnostics, scene::RenderGpuScene};

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

    fn track_with(
        &self,
        register: impl FnOnce(Box<dyn FnOnce() + Send + 'static>),
    ) -> GpuCompletionValue {
        let value = self.submitted.fetch_add(1, Ordering::AcqRel) + 1;
        let completed = Arc::clone(&self.completed);
        register(Box::new(move || {
            completed.fetch_max(value, Ordering::Release);
        }));
        GpuCompletionValue(value)
    }

    /// Tracks work submitted through Prism's standalone RHI queue.
    ///
    /// This remains available for RHI-owned tests and producers. Bevy Scene
    /// production work must use [`Self::track_host`] so completion belongs to
    /// the queue that submitted the Scene command buffers.
    pub fn track(&self, queue: &WgpuQueue) -> GpuCompletionValue {
        self.track_with(|callback| queue.on_submitted_work_done(callback))
    }

    /// Tracks work submitted through Bevy's render queue.
    ///
    /// `RenderGraphSystems::Finish` runs after Bevy's `Submit` set, so the
    /// callback observes the exact host submission containing Scene work. A
    /// callback on an unrelated `PrismGpu` queue cannot prove those resources
    /// are no longer in use.
    pub fn track_host(&self, queue: &RenderQueue) -> GpuCompletionValue {
        self.track_with(|callback| queue.on_submitted_work_done(callback))
    }
}

pub(crate) fn track_submission(
    tracker: Res<GpuCompletionTracker>,
    queue: Option<Res<RenderQueue>>,
) {
    // This system is in RenderGraphSystems::Finish, after Bevy's Submit set.
    // Track the host queue, never the optional standalone PrismGpu queue.
    if let Some(queue) = queue {
        tracker.track_host(&queue);
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

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn completion_values_advance_only_after_the_registered_callback_runs() {
        let tracker = GpuCompletionTracker::default();
        let mut callbacks: Vec<Box<dyn FnOnce() + Send + 'static>> = Vec::new();

        let first = tracker.track_with(|callback| callbacks.push(callback));
        let second = tracker.track_with(|callback| callbacks.push(callback));

        assert_eq!(first, GpuCompletionValue(1));
        assert_eq!(second, GpuCompletionValue(2));
        assert_eq!(tracker.completed_value(), GpuCompletionValue(0));
        assert_eq!(tracker.next_submission_value(), GpuCompletionValue(3));

        callbacks.remove(1)();
        assert_eq!(tracker.completed_value(), second);
        callbacks.remove(0)();
        assert_eq!(tracker.completed_value(), second);
    }

    #[test]
    fn host_tracker_registration_uses_the_same_completion_state_as_rhi_tracking() {
        let tracker = GpuCompletionTracker::default();
        let callbacks = Arc::new(Mutex::new(Vec::new()));

        let first_callbacks = Arc::clone(&callbacks);
        let first = tracker.track_with(move |callback| {
            first_callbacks.lock().unwrap().push(callback);
        });
        let second_callbacks = Arc::clone(&callbacks);
        let second = tracker.track_with(move |callback| {
            second_callbacks.lock().unwrap().push(callback);
        });

        assert_eq!(first, GpuCompletionValue(1));
        assert_eq!(second, GpuCompletionValue(2));
        let mut registered = callbacks.lock().unwrap();
        registered.pop().unwrap()();
        registered.pop().unwrap()();
        drop(registered);
        assert_eq!(tracker.completed_value(), second);
    }
}
