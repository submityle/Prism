//! A pass-through [`RenderQueue`] that intercepts a graph's submission.
//!
//! [`RenderGraph::execute`](prism_render_graph::RenderGraph::execute) records a
//! frame and immediately submits it, leaving the caller no handle on the
//! resulting command buffer. Bounded frames-in-flight, however, must own the
//! moment of submission so it can capture the backend's completion token and
//! block on the oldest frame when the pipeline is full.
//!
//! [`CapturingQueue`] bridges the two: it forwards uploads to the real queue but
//! *captures* command buffers handed to [`submit`](RenderQueue::submit) instead
//! of forwarding them. The runtime then submits the captured buffers itself,
//! through [`FrameSync`](crate::FrameSync).

use std::sync::{Mutex, MutexGuard};

use prism_render_driver::{BufferId, CommandBuffer, ImageDataLayout, RenderQueue, TextureWrite};

/// A [`RenderQueue`] wrapper that records submissions instead of issuing them.
///
/// Uploads ([`write_buffer`](RenderQueue::write_buffer),
/// [`write_texture`](RenderQueue::write_texture)) pass straight through to the
/// wrapped queue so recorded work observes the data. Command buffers passed to
/// [`submit`](RenderQueue::submit) are retained, in submission order, until
/// [`take`](Self::take) drains them.
pub(crate) struct CapturingQueue<'a, Q: RenderQueue> {
    /// The real queue uploads are forwarded to.
    inner: &'a Q,
    /// Command buffers captured from `submit`, oldest first.
    captured: Mutex<Vec<CommandBuffer>>,
}

impl<'a, Q: RenderQueue> CapturingQueue<'a, Q> {
    /// Wraps `inner`, capturing submissions without forwarding them.
    pub(crate) fn new(inner: &'a Q) -> Self {
        Self {
            inner,
            captured: Mutex::new(Vec::new()),
        }
    }

    /// Drains every captured command buffer, oldest first.
    pub(crate) fn take(&self) -> Vec<CommandBuffer> {
        core::mem::take(&mut *self.captured())
    }

    /// Locks the capture buffer, tolerating a poisoned mutex.
    ///
    /// The only mutation is an `extend` or `mem::take`, so a panic elsewhere
    /// cannot leave the buffer in an inconsistent state.
    fn captured(&self) -> MutexGuard<'_, Vec<CommandBuffer>> {
        self.captured
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl<Q: RenderQueue> RenderQueue for CapturingQueue<'_, Q> {
    fn write_buffer(&self, buffer: BufferId, offset: u64, data: &[u8]) {
        self.inner.write_buffer(buffer, offset, data);
    }

    fn write_texture(&self, destination: TextureWrite, data: &[u8], layout: ImageDataLayout) {
        self.inner.write_texture(destination, data, layout);
    }

    fn submit(&self, command_buffers: &[CommandBuffer]) {
        // Clone rather than forward: the runtime submits these itself so it can
        // hold the completion token.
        self.captured().extend_from_slice(command_buffers);
    }
}
