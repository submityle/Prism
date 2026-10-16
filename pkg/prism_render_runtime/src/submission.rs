//! Completion-owned graph allocations, including explicitly retained history.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};
use prism_render_driver::RenderDevice;
use prism_render_graph::GraphAllocations;
use std::sync::Mutex;

/// A submitted frame's physical resources, retained through real completion.
///
/// Retain this handle while publishing graph-created persistent resources into
/// later frames. Pass the creating owners to
/// [`RenderRuntime::execute_prepared_owned`](crate::RenderRuntime::execute_prepared_owned)
/// so each consumer retains them through its own completion. External native
/// imports are not owned here. Without an external handle, the completion
/// callback releases graph-owned resources when this submission completes.
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
        imports: Vec<Self>,
    ) -> Self {
        Self {
            value,
            generation,
            resources: Arc::new(SubmissionResources {
                device,
                allocations: Some(allocations),
                complete: AtomicBool::new(false),
                retirement_safe: AtomicBool::new(true),
                imports: Mutex::new(imports),
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
        // Release dependencies on completion, not on this owner's final drop:
        // retaining a completed history frame must not pin every older frame.
        let imports = core::mem::take(
            &mut *self
                .resources
                .imports
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        drop(imports);
    }

    pub(crate) fn belongs_to(&self, device: &Arc<D>, generation: u64) -> bool {
        self.generation == generation && Arc::ptr_eq(&self.resources.device, device)
    }
}

struct SubmissionResources<D: RenderDevice> {
    device: Arc<D>,
    allocations: Option<GraphAllocations>,
    complete: AtomicBool,
    // Monotonic: an abandoned, unconfirmed consumer forbids destroying its
    // imports even if their own producing submission has already completed.
    retirement_safe: AtomicBool,
    imports: Mutex<Vec<FrameSubmission<D>>>,
}

impl<D: RenderDevice> Drop for SubmissionResources<D> {
    fn drop(&mut self) {
        // A backend may discard a callback during device loss or teardown.
        // Dropping that closure is not evidence of completion. In that case
        // leave the native objects to the creating device's teardown instead
        // of issuing destroys against possibly in-flight work.
        let complete = self.complete.load(Ordering::Acquire);
        if !complete {
            for owner in self
                .imports
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .iter()
            {
                owner
                    .resources
                    .retirement_safe
                    .store(false, Ordering::Release);
            }
        }
        // Do not leak owners or device Arcs on this path. An unsafe allocation
        // remains in its creating device's table until device teardown instead.
        if complete && self.retirement_safe.load(Ordering::Acquire) {
            if let Some(allocations) = self.allocations.take() {
                allocations.release(&*self.device);
            }
        }
    }
}
