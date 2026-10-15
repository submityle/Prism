//! The render world's owned GPU state.

use prism_ecs::prelude::Resource;
use prism_render_driver::{FrameSync, Instance};
use prism_render_runtime::RenderRuntime;

/// The RHI render runtime, stored as a resource on the render sub-app's world.
///
/// The runtime is the GPU ground truth — adapter, device, queue, and (once a
/// window is attached) surface. It lives in the render world and never crosses
/// into the main world, which is the whole point of the boundary: the
/// authoritative simulation never names a device.
///
/// This wrapper exists because `RenderRuntime<I>` is defined in
/// `prism_render_runtime`, so the orphan rule forbids implementing the resource
/// marker on it here; the newtype is local to this crate, so the marker impl is
/// allowed, and it gives the render world an explicit, named handle to the
/// device rather than a bare runtime type floating in the resource table.
pub struct RenderRuntimeResource<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    runtime: RenderRuntime<I>,
}

impl<I> RenderRuntimeResource<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    /// Wraps a brought-up runtime so it can live in the render world.
    #[must_use]
    pub fn new(runtime: RenderRuntime<I>) -> Self {
        Self { runtime }
    }

    /// The wrapped runtime.
    #[must_use]
    pub fn runtime(&self) -> &RenderRuntime<I> {
        &self.runtime
    }

    /// Mutable access to the wrapped runtime for device recovery.
    pub(crate) fn runtime_mut(&mut self) -> &mut RenderRuntime<I> {
        &mut self.runtime
    }
}

impl<I> Resource for RenderRuntimeResource<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
}
