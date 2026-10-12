//! Frame completion: the waitable-submission hook that frame pacing needs.
//!
//! [`RenderQueue::submit`](crate::RenderQueue::submit) is deliberately
//! fire-and-forget: it returns nothing, so a caller cannot tell when the GPU
//! has finished the work. Bounded frames-in-flight needs exactly that signal,
//! so this module adds a small, **backend-implemented** trait on top of the
//! queue: submit and receive an opaque [`CompletionToken`], then block on that
//! token (or on the whole queue) when the pipeline is full.
//!
//! The trait lives in the RHI, not in any caller, because every backend must be
//! able to satisfy it — that is what keeps frame pacing backend-agnostic. A
//! token is type-erased into a `Box<dyn Any + Send + Sync>` so callers never
//! name a backend's native submission handle; the backend downcasts its own
//! tokens inside [`FrameSync::wait_for`].

use alloc::boxed::Box;
use core::any::Any;

use crate::command::CommandBuffer;

/// An opaque, backend-defined completion token for a single submission.
///
/// Produced by [`FrameSync::submit_tracked`] and consumed by
/// [`FrameSync::wait_for`]. It is boxed and type-erased so callers and the frame
/// pacer stay free of any backend's native types.
pub type CompletionToken = Box<dyn Any + Send + Sync>;

/// Turns fire-and-forget submission into a waitable completion signal.
///
/// Implemented by a backend's queue. Every method takes `&self` so the queue
/// keeps matching the frozen [`RenderQueue`](crate::RenderQueue) shape.
pub trait FrameSync {
    /// Submits `command_buffers` and returns a token that completes when the GPU
    /// has finished them.
    ///
    /// Submission order is preserved: the returned token is ordered after every
    /// earlier submission on the same queue.
    fn submit_tracked(&self, command_buffers: &[CommandBuffer]) -> CompletionToken;

    /// Blocks the calling thread until the submission identified by `token` has
    /// completed.
    ///
    /// Returns `true` on success, or `false` when completion could not be
    /// confirmed — for example a lost device, or a token that did not originate
    /// from this backend — in which case the caller proceeds without the
    /// guarantee.
    fn wait_for(&self, token: &(dyn Any + Send + Sync)) -> bool;

    /// Blocks until every submission already handed to this queue has
    /// completed, draining the pipeline regardless of per-frame tokens.
    ///
    /// Used on shutdown and by idle points (e.g. a CPU readback) that need the
    /// GPU fully caught up. Returns `true` on success.
    fn wait_idle(&self) -> bool;

    /// Registers `callback` to run once every submission handed to this queue
    /// so far has completed on the GPU.
    ///
    /// This is the non-blocking counterpart to [`wait_idle`](Self::wait_idle):
    /// a renderer that tracks GPU work completion (rather than stalling the
    /// CPU) uses it to advance its bookkeeping when the queue drains. The
    /// callback runs on a backend-chosen thread, so it must be `Send` and must
    /// not block.
    fn on_submitted_work_done(&self, callback: Box<dyn FnOnce() + Send + 'static>);
}
