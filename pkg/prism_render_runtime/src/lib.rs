//! # `prism_render_runtime`
//!
//! The RHI-backed render spine: it brings up a `prism_render_driver` device
//! through whichever [`Instance`] backend it is given, holds it for the app's
//! lifetime, and drives [`prism_render_graph`] frames through the frozen RHI.
//! It is the seam that replaced the host role `bevy_render` used to play.
//!
//! The runtime is **generic over the backend**: it depends only on the RHI
//! traits, so switching backends means constructing it with a different
//! [`Instance`] implementation — no change to this crate, the graph, or the
//! renderer. [`RenderRuntime::<WgpuInstance>::wgpu`] is the default convenience
//! constructor for the shipped wgpu backend.
//!
//! ## Scope
//!
//! Phase 1 is currently **headless**: the runtime owns the device/queue and
//! executes frame graphs against offscreen targets. Surface acquisition and
//! present are exposed through the RHI [`Surface`](prism_render_driver::Surface)
//! trait and are wired into the app loop next.

use core::fmt;

use prism_render_driver::{
    Adapter, AdapterInfo, DeviceDescriptor, Extent3d, Instance, PowerPreference,
    RequestAdapterOptions,
};
use prism_render_graph::{CompileError, RenderGraph};

#[cfg(test)]
mod tests;

/// Why a [`RenderRuntime`] could not be brought up.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RenderInitError {
    /// No adapter matched the request (no GPU, or none usable).
    NoAdapter,
    /// The chosen adapter rejected the device request; the string carries the
    /// backend's diagnostic.
    Device(String),
}

impl fmt::Display for RenderInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => f.write_str("no usable graphics adapter"),
            Self::Device(detail) => write!(f, "device bring-up failed: {detail}"),
        }
    }
}

/// The RHI-backed render runtime: an adapter plus its device/queue, owned for
/// the app's lifetime, with the frame extent graphs resolve against.
pub struct RenderRuntime<I: Instance> {
    instance: I,
    adapter: I::Adapter,
    device: I::Device,
    queue: I::Queue,
    extent: Extent3d,
}

impl<I: Instance> RenderRuntime<I> {
    /// Brings up a headless runtime on a high-performance adapter from
    /// `instance`.
    ///
    /// # Errors
    ///
    /// [`RenderInitError::NoAdapter`] when no adapter is available, or
    /// [`RenderInitError::Device`] when the device request is rejected.
    pub fn headless(instance: I, extent: Extent3d) -> Result<Self, RenderInitError> {
        let options = RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback: false,
        };
        let adapter = instance
            .request_adapter(&options)
            .ok_or(RenderInitError::NoAdapter)?;
        let (device, queue) = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .map_err(|error| RenderInitError::Device(format!("{error}")))?;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            extent,
        })
    }

    /// The backend instance the runtime was created from.
    #[must_use]
    pub fn instance(&self) -> &I {
        &self.instance
    }

    /// The adapter the device was created on.
    #[must_use]
    pub fn adapter(&self) -> &I::Adapter {
        &self.adapter
    }

    /// The chosen adapter's identifying description.
    #[must_use]
    pub fn adapter_info(&self) -> &AdapterInfo {
        self.adapter.info()
    }

    /// The RHI render device.
    #[must_use]
    pub fn device(&self) -> &I::Device {
        &self.device
    }

    /// The RHI render queue.
    #[must_use]
    pub fn queue(&self) -> &I::Queue {
        &self.queue
    }

    /// The frame extent graphs created by this runtime resolve against.
    #[must_use]
    pub fn extent(&self) -> Extent3d {
        self.extent
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
}

impl RenderRuntime<prism_render_driver_wgpu::WgpuInstance> {
    /// Brings up a headless runtime on the default wgpu adapter.
    ///
    /// Convenience wrapper over [`RenderRuntime::headless`] for the shipped
    /// backend, so callers that do not select a backend need not name wgpu.
    ///
    /// # Errors
    ///
    /// See [`RenderRuntime::headless`].
    pub fn wgpu(extent: Extent3d) -> Result<Self, RenderInitError> {
        Self::headless(prism_render_driver_wgpu::WgpuInstance::new(), extent)
    }
}
