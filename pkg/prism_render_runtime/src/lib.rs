//! # `prism_render_runtime`
//!
//! The RHI-backed render spine. It brings up a [`WgpuDevice`]/[`WgpuQueue`]
//! pair and drives [`prism_render_graph`] frames through the frozen
//! [`prism_render_driver`] interface — the ground that the retired Bevy render
//! stack used to provide.
//!
//! ## Scope (phase 1)
//!
//! This crate is currently **headless**: it owns the device/queue and executes
//! frame graphs against offscreen targets, with a readback hook for golden and
//! headless tests. Still to come, in order:
//!
//! 1. Window/surface acquisition (`prism_window`'s raw handle → a swapchain).
//! 2. Swapchain import into the graph + a real present path.
//! 3. A `prism_app` render sub-app that owns this runtime and extracts the main
//!    world each frame.
//!
//! Only after those land does `prism_render_scene` get re-pointed from
//! `bevy_render` to this RHI surface.
//!
//! The crate names no `wgpu` type: it depends on the backend crate only for its
//! bring-up entry points, so a future non-wgpu backend slots in behind the same
//! [`prism_render_driver`] traits.

use prism_render_driver::Extent3d;
use prism_render_driver_wgpu::{DeviceCreationError, WgpuDevice, WgpuInstance, WgpuQueue};
use prism_render_graph::{CompileError, RenderGraph};

#[cfg(test)]
mod tests;

/// Why a [`RenderRuntime`] could not be created.
#[derive(Debug)]
pub enum RenderInitError {
    /// No usable graphics adapter was found (no GPU, or none matching the
    /// request). The design treats "zero adapters" as the one fatal bring-up
    /// condition.
    NoAdapter(DeviceCreationError),
}

impl core::fmt::Display for RenderInitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoAdapter(err) => write!(f, "no usable graphics adapter: {err:?}"),
        }
    }
}

/// The RHI-backed render runtime: a device/queue pair plus the frame extent.
///
/// One runtime owns the GPU ground for an app. Headless for now; the surface
/// and present path are added without changing this surface.
pub struct RenderRuntime {
    instance: WgpuInstance,
    device: WgpuDevice,
    queue: WgpuQueue,
    extent: Extent3d,
}

impl RenderRuntime {
    /// Brings up a headless runtime on the default adapter at `extent`.
    ///
    /// # Errors
    ///
    /// [`RenderInitError::NoAdapter`] when no adapter could be requested (the
    /// usual case in a GPU-less CI sandbox).
    pub fn headless(extent: Extent3d) -> Result<Self, RenderInitError> {
        let instance = WgpuInstance::new();
        let (device, queue) = instance
            .request_default_device()
            .map_err(RenderInitError::NoAdapter)?;
        Ok(Self {
            instance,
            device,
            queue,
            extent,
        })
    }

    /// The frame extent graphs created by this runtime resolve against.
    #[must_use]
    pub fn extent(&self) -> Extent3d {
        self.extent
    }

    /// The RHI instance the runtime was created from.
    #[must_use]
    pub fn instance(&self) -> &WgpuInstance {
        &self.instance
    }

    /// The RHI render device.
    #[must_use]
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// The RHI render queue.
    #[must_use]
    pub fn queue(&self) -> &WgpuQueue {
        &self.queue
    }

    /// A fresh frame graph sized to this runtime's extent.
    #[must_use]
    pub fn new_graph(&self) -> RenderGraph {
        RenderGraph::new(self.extent)
    }

    /// Compiles, realizes transients, records commands, and submits `graph`.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] when the graph fails to compile (a dependency
    /// cycle); no GPU work is issued in that case.
    pub fn execute(&self, graph: RenderGraph) -> Result<(), CompileError> {
        graph.execute(&self.device, &self.queue)
    }

    /// Blocks until every previously submitted operation completes.
    ///
    /// Returns `false` if the device was lost while waiting.
    pub fn wait(&self) -> bool {
        self.device.poll_wait()
    }
}
