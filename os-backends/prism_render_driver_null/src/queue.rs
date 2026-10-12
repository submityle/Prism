//! The null [`RenderQueue`]: uploads and submissions are no-ops.
//!
//! The frozen RHI queue is fire-and-forget, so a headless backend has nothing
//! to *do* with an upload or a submission — there is no GPU memory to fill and
//! no command stream to execute. The methods still exist and are called, which
//! is what lets upper layers run their full submit path unchanged under CI.

use prism_render_driver as rhi;
use rhi::RenderQueue;

/// A null [`rhi::RenderQueue`] whose operations are all no-ops.
#[derive(Clone, Copy, Debug, Default)]
pub struct NullQueue;

impl NullQueue {
    /// Creates a null queue.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl RenderQueue for NullQueue {
    fn write_buffer(&self, _buffer: rhi::BufferId, _offset: u64, _data: &[u8]) {}

    fn write_texture(
        &self,
        _destination: rhi::TextureWrite,
        _data: &[u8],
        _layout: rhi::ImageDataLayout,
    ) {
    }

    fn submit(&self, _command_buffers: &[rhi::CommandBuffer]) {}
}
