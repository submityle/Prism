//! Backend completion hooks the frame pacer waits on.
//!
//! The frozen [`RenderQueue`](prism_render_driver::RenderQueue) trait is
//! fire-and-forget: [`submit`](prism_render_driver::RenderQueue::submit)
//! returns nothing, so a caller cannot tell when a frame has finished on the
//! GPU. Bounded frames-in-flight needs exactly that signal, so the runtime adds
//! this small, non-frozen trait: submit and receive an opaque
//! [`CompletionToken`], then block on that token (or on the whole queue) when
//! the [`FramePacer`](crate::FramePacer) says the pipeline is full.
//!
//! The trait is backend-agnostic by design. A token is type-erased into a
//! `Box<dyn Any + Send + Sync>` so the runtime never names a backend's native
//! submission handle; the backend downcasts its own tokens inside
//! [`FrameSync::wait_for`]. The one implementation that ships today is for the
//! `wgpu` backend's [`WgpuQueue`]; an additional backend joins by implementing
//! this trait, with the runtime unchanged.

use core::any::Any;

use prism_render_driver as rhi;
use prism_render_driver_wgpu::WgpuQueue;

/// An opaque, backend-defined completion token for a single submission.
///
/// Produced by [`FrameSync::submit_tracked`] and consumed by
/// [`FrameSync::wait_for`]. It is boxed and type-erased so the runtime and the
/// pacer stay free of any backend's native types.
pub type CompletionToken = Box<dyn Any + Send + Sync>;

/// The backend hook that turns fire-and-forget submission into a waitable
/// completion signal.
pub trait FrameSync {
    /// Submits `command_buffers` and returns a token that completes when the
    /// GPU has finished them.
    ///
    /// Submission order is preserved: the returned token is ordered after every
    /// earlier submission on the same queue.
    fn submit_tracked(&self, command_buffers: &[rhi::CommandBuffer]) -> CompletionToken;

    /// Blocks the calling thread until the submission identified by `token`
    /// has completed.
    ///
    /// Returns `true` on success. Returns `false` when the wait could not
    /// confirm completion — for example a lost device, or a token that did not
    /// originate from this backend — in which case the caller proceeds without
    /// the guarantee.
    fn wait_for(&self, token: &(dyn Any + Send + Sync)) -> bool;

    /// Blocks until every submission already handed to this queue has
    /// completed, draining the pipeline regardless of per-frame tokens.
    ///
    /// Used on shutdown and by idle points (e.g. a CPU readback) that need the
    /// GPU fully caught up. Returns `true` on success.
    fn wait_idle(&self) -> bool;
}

impl FrameSync for WgpuQueue {
    fn submit_tracked(&self, command_buffers: &[rhi::CommandBuffer]) -> CompletionToken {
        // Inherent methods take precedence over trait methods, so this resolves
        // to the backend's `WgpuQueue::submit_tracked` (returning the native
        // submission handle), not to this very method. Boxing type-erases it.
        Box::new(self.submit_tracked(command_buffers))
    }

    fn wait_for(&self, token: &(dyn Any + Send + Sync)) -> bool {
        self.wait_for_submission_erased(token)
    }

    fn wait_idle(&self) -> bool {
        self.wait_idle()
    }
}
