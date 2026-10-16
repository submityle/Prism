//! Completion-owned graph allocations, including explicitly retained history.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};
use prism_render_driver::RenderDevice;
use prism_render_graph::GraphAllocations;

/// A submitted frame's physical resources, retained through real completion.
///
/// Retain this handle while publishing graph-created persistent resources into
/// later frames, and until all those later uses complete. Imported resources
/// are not owned here. Without an external handle, the completion callback
/// releases graph-owned resources when this submission completes.
///
/// A handle belongs to its creating device generation, even after runtime
/// recovery; its resource IDs must never be imported into a replacement device.
pub struct FrameSubmission<D: RenderDevice> {
    value: u64,
    generation: u64,
    resources: Arc<SubmissionResources<D>>,
}

impl<D: RenderDevice> Clone for FrameSubmission<D> {
    fn clone(&self) -> Self {
        Self {
            value: self.value,
            generation: self.generation,
            resources: Arc::clone(&self.resources),
        }
    }
}

impl<D: RenderDevice> FrameSubmission<D> {
    pub(crate) fn new(
        value: u64,
        generation: u64,
        device: Arc<D>,
        allocations: GraphAllocations,
    ) -> Self {
        Self {
            value,
            generation,
            resources: Arc::new(SubmissionResources {
                device,
                allocations: Some(allocations),
                complete: AtomicBool::new(false),
            }),
        }
    }

    /// The queue-ordered completion value within `device_generation()`.
    #[must_use]
    pub const fn completion_value(&self) -> u64 {
        self.value
    }

    /// The creating runtime generation; resource IDs are valid only there.
    #[must_use]
    pub const fn device_generation(&self) -> u64 {
        self.generation
    }

    /// Whether the actual submitting queue has invoked its completion callback.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.resources.complete.load(Ordering::Acquire)
    }

    /// Immutable IDs of the owned resources, keyed by graph resource index.
    /// Keeping this handle alive keeps these IDs valid on the creating device.
    #[must_use]
    pub fn allocations(&self) -> &GraphAllocations {
        self.resources.allocations.as_ref().expect("live owner")
    }

    pub(crate) fn mark_complete(&self) {
        self.resources.complete.store(true, Ordering::Release);
    }
}

struct SubmissionResources<D: RenderDevice> {
    device: Arc<D>,
    allocations: Option<GraphAllocations>,
    complete: AtomicBool,
}

impl<D: RenderDevice> Drop for SubmissionResources<D> {
    fn drop(&mut self) {
        // A backend may discard a callback during device loss or teardown.
        // Dropping that closure is not evidence of completion. In that case
        // leave the native objects to the creating device's teardown instead
        // of issuing destroys against possibly in-flight work.
        if self.complete.load(Ordering::Acquire) {
            if let Some(allocations) = self.allocations.take() {
                allocations.release(&*self.device);
            }
        }
    }
}
