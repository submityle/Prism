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
//!
//! ## Frame pacing
//!
//! Submission is bounded: the runtime keeps at most
//! [`DEFAULT_FRAMES_IN_FLIGHT`] (configurable) frames outstanding and blocks on
//! the oldest before starting a new one, so the CPU cannot race arbitrarily far
//! ahead of the GPU. The policy lives in the pure [`FramePacer`]; the completion
//! signal it waits on comes from the backend through [`FrameSync`]. Both are
//! backend-agnostic — the runtime itself never names a native submission handle.

extern crate alloc;

use core::fmt;
use alloc::sync::Arc;
use std::sync::{Mutex, MutexGuard};

use prism_render_driver::{
    Adapter, AdapterInfo, DeviceDescriptor, Extent3d, Instance, PowerPreference,
    RequestAdapterOptions, Surface, SurfaceConfig, SurfaceError, TextureState, TextureUsages,
};
use prism_render_graph::{
    CompileError, ImportedTexture, RenderGraph, TextureDesc, TextureHandle,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

mod capture;
mod frame_pacing;

pub use frame_pacing::{FramePacer, DEFAULT_FRAMES_IN_FLIGHT};
// The waitable-submission contract is part of the RHI, so every backend can
// satisfy it without depending on this crate. Re-exported for convenience.
pub use prism_render_driver::{CompletionToken, FrameSync};

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

/// Why a [`RenderRuntime::render_frame`] call did not produce a presented frame.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum FrameError {
    /// No surface has been created; call
    /// [`RenderRuntime::create_surface`] first.
    NoSurface,
    /// The surface reported a failure acquiring or presenting the frame.
    Surface(SurfaceError),
    /// The frame graph failed to compile; no GPU work was submitted.
    Compile(CompileError),
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSurface => f.write_str("no surface is configured for this runtime"),
            Self::Surface(error) => write!(f, "surface error while presenting: {error}"),
            Self::Compile(error) => write!(f, "frame graph failed to compile: {error:?}"),
        }
    }
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
    /// The frames-in-flight bound and its outstanding completion tokens.
    pacer: Mutex<FramePacer<CompletionToken>>,
    /// The window surface, once created.
    surface: Option<I::Surface>,
    /// The configuration the surface was last configured with, so an
    /// `Outdated` surface can be reconfigured without the caller re-supplying it.
    surface_config: Option<SurfaceConfig>,
}

impl<I: Instance> RenderRuntime<I> {
    /// Brings up a headless runtime on a high-performance adapter from
    /// `instance`, keeping [`DEFAULT_FRAMES_IN_FLIGHT`] frames in flight.
    ///
    /// # Errors
    ///
    /// [`RenderInitError::NoAdapter`] when no adapter is available, or
    /// [`RenderInitError::Device`] when the device request is rejected.
    pub fn headless(instance: I, extent: Extent3d) -> Result<Self, RenderInitError> {
        Self::with_frames_in_flight(instance, extent, DEFAULT_FRAMES_IN_FLIGHT)
    }

    /// Brings up a headless runtime that keeps at most `max_in_flight` frames
    /// outstanding.
    ///
    /// `max_in_flight` is clamped to at least one. Higher values trade latency
    /// for throughput; the double-buffered default of
    /// [`DEFAULT_FRAMES_IN_FLIGHT`] is the usual choice.
    ///
    /// # Errors
    ///
    /// See [`RenderRuntime::headless`].
    pub fn with_frames_in_flight(
        instance: I,
        extent: Extent3d,
        max_in_flight: usize,
    ) -> Result<Self, RenderInitError> {
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
            pacer: Mutex::new(FramePacer::new(max_in_flight)),
            surface: None,
            surface_config: None,
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

    /// Creates a window surface for `window`, replacing any existing one.
    ///
    /// The window is held by the surface, so the runtime owns the presentation
    /// target for its whole lifetime.
    ///
    /// # Errors
    ///
    /// Returns the backend's [`SurfaceError`] when the platform cannot create a
    /// surface for the handle.
    pub fn create_surface<W>(&mut self, window: Arc<W>) -> Result<(), SurfaceError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let surface = self
            .instance
            .create_surface(window, &self.device, &self.queue)?;
        self.surface = Some(surface);
        self.surface_config = None;
        Ok(())
    }

    /// The current window surface, if one has been created.
    #[must_use]
    pub fn surface(&self) -> Option<&I::Surface> {
        self.surface.as_ref()
    }

    /// Mutable access to the current window surface.
    pub fn surface_mut(&mut self) -> Option<&mut I::Surface> {
        self.surface.as_mut()
    }

    /// The configuration the surface was last configured with, if any.
    ///
    /// Callers use this to build pipelines whose color targets match the
    /// swapchain format chosen by [`Surface::capabilities`].
    #[must_use]
    pub fn surface_config(&self) -> Option<&SurfaceConfig> {
        self.surface_config.as_ref()
    }

    /// (Re)configures the current surface and adopts its extent for future
    /// frame graphs.
    ///
    /// # Errors
    ///
    /// [`SurfaceError::Unsupported`] when no surface exists or the
    /// configuration cannot be satisfied.
    pub fn configure_surface(&mut self, config: SurfaceConfig) -> Result<(), SurfaceError> {
        let surface = self.surface.as_mut().ok_or(SurfaceError::Unsupported)?;
        surface.configure(&config)?;
        self.extent = Extent3d::new_2d(config.width, config.height);
        self.surface_config = Some(config);
        Ok(())
    }

    /// Configures the current surface with its first supported format and a
    /// v-synced, double-buffered swapchain of `width`×`height`.
    ///
    /// The convenience most callers want at startup and after a resize.
    ///
    /// # Errors
    ///
    /// [`SurfaceError::Unsupported`] when no surface exists or it advertises no
    /// usable swapchain format.
    pub fn configure_surface_default(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<(), SurfaceError> {
        let format = {
            let surface = self.surface.as_ref().ok_or(SurfaceError::Unsupported)?;
            *surface
                .capabilities()
                .formats
                .first()
                .ok_or(SurfaceError::Unsupported)?
        };
        self.configure_surface(SurfaceConfig::new(format, width, height))
    }

    /// Re-applies the last surface configuration after the surface reported it
    /// outdated (typically a resize).
    fn reconfigure_surface(&mut self) -> Result<(), SurfaceError> {
        let config = self.surface_config.ok_or(SurfaceError::Unsupported)?;
        self.configure_surface(config)
    }

    /// Renders and presents one windowed frame.
    ///
    /// Acquires the next swapchain image, imports it into a fresh frame graph as
    /// the presentable target, hands it to `build` to declare passes, executes
    /// the graph through the frames-in-flight bound, then presents the frame.
    ///
    /// An occluded window (e.g. minimized) is a successful no-op: the frame is
    /// skipped and `Ok(())` is returned. An outdated surface is reconfigured and
    /// reported as [`FrameError::Surface`] so the caller can retry next frame.
    ///
    /// # Errors
    ///
    /// See [`FrameError`]; no GPU work is submitted when the graph fails to
    /// compile.
    pub fn render_frame<F>(&mut self, build: F) -> Result<(), FrameError>
    where
        F: FnOnce(&mut RenderGraph, TextureHandle),
        I::Queue: FrameSync,
    {
        let acquired = {
            let surface = self.surface.as_ref().ok_or(FrameError::NoSurface)?;
            surface.acquire()
        };
        let frame = match acquired {
            Ok(frame) => frame,
            // A minimized window has nothing to present; skipping is success.
            Err(SurfaceError::Occluded) => return Ok(()),
            Err(SurfaceError::Outdated) => {
                self.reconfigure_surface().map_err(FrameError::Surface)?;
                return Err(FrameError::Surface(SurfaceError::Outdated));
            }
            Err(error) => return Err(FrameError::Surface(error)),
        };

        let image = self
            .surface
            .as_ref()
            .expect("surface checked above")
            .image(&frame);

        let mut graph = RenderGraph::new(image.size);
        let target = graph.import_texture(
            "swapchain",
            TextureDesc::color(image.format).with_usage(TextureUsages::RENDER_ATTACHMENT),
            ImportedTexture {
                texture: image.texture,
                default_view: image.view,
                entry_state: TextureState::initial(),
                exit_state: TextureState::present(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        build(&mut graph, target);

        self.execute(graph).map_err(FrameError::Compile)?;

        self.surface
            .as_ref()
            .expect("surface checked above")
            .present(frame)
            .map_err(FrameError::Surface)
    }

    /// Locks and returns the frame pacer, tolerating a poisoned mutex.
    ///
    /// A panic in a thread that held the lock cannot leave the pacer in a
    /// partially-updated state — every mutation is a single `VecDeque` push or
    /// pop — so recovering the guard is safe and keeps a single panicking frame
    /// from wedging the whole render loop.
    fn pacer(&self) -> MutexGuard<'_, FramePacer<CompletionToken>> {
        self.pacer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The number of frames currently allowed to be outstanding at once.
    #[must_use]
    pub fn max_frames_in_flight(&self) -> usize {
        self.pacer().max_in_flight()
    }

    /// The number of frames submitted but not yet confirmed complete.
    #[must_use]
    pub fn frames_in_flight(&self) -> usize {
        self.pacer().in_flight()
    }

    /// Compiles, realizes transients, records commands, and submits `graph`
    /// under the frames-in-flight bound.
    ///
    /// Before the frame is submitted, the pacer is consulted: if
    /// [`Self::max_frames_in_flight`] frames are already outstanding, this
    /// blocks on the oldest one's completion token first. As a result no more
    /// than that many frames are ever in flight.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] when the graph fails to compile (a dependency
    /// cycle); no GPU work is issued and the pacer is left untouched in that
    /// case.
    pub fn execute(&self, graph: RenderGraph) -> Result<(), CompileError>
    where
        I::Queue: FrameSync,
    {
        // Record the graph into a command buffer without submitting it, so the
        // pacer can own the actual submission and receive its completion token.
        let capture = capture::CapturingQueue::new(&self.queue);
        graph.execute(&self.device, &capture)?;
        let commands = capture.take();

        // Retire the oldest frame if the pipeline is full; this is the only
        // place the CPU blocks on the GPU.
        if let Some(oldest) = self.pacer().begin_frame() {
            let _ = self.queue.wait_for(oldest.as_ref());
        }

        let token = self.queue.submit_tracked(&commands);
        self.pacer().record(token);
        Ok(())
    }

    /// Blocks until every outstanding frame has completed, leaving the pacer
    /// empty.
    ///
    /// Call this before reading results back, resizing a surface, or tearing
    /// the runtime down. Returns `true` when every wait confirmed completion.
    pub fn wait_idle(&self) -> bool
    where
        I::Queue: FrameSync,
    {
        let outstanding: Vec<CompletionToken> = self.pacer().drain().collect();
        let all_confirmed = outstanding
            .iter()
            .all(|token| self.queue.wait_for(token.as_ref()));
        // Always reach for the queue-wide barrier too: it also covers work the
        // runtime did not submit through `execute`.
        let idle = self.queue.wait_idle();
        all_confirmed && idle
    }
}

/// The default (wgpu) backend's runtime, so callers need not name the backend
/// crate to select it.
pub type WgpuRuntime = RenderRuntime<prism_render_driver_wgpu::WgpuInstance>;

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

    /// Brings up a headless wgpu runtime that keeps at most `max_in_flight`
    /// frames outstanding.
    ///
    /// # Errors
    ///
    /// See [`RenderRuntime::headless`].
    pub fn wgpu_paced(extent: Extent3d, max_in_flight: usize) -> Result<Self, RenderInitError> {
        Self::with_frames_in_flight(
            prism_render_driver_wgpu::WgpuInstance::new(),
            extent,
            max_in_flight,
        )
    }
}
