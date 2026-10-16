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

use alloc::string::String;
use alloc::sync::Arc;
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use prism_render_driver::{
    Adapter, AdapterInfo, DeviceDescriptor, DeviceLostReason, Extent3d, Instance, PowerPreference,
    RenderDevice, RequestAdapterOptions, Surface, SurfaceConfig, SurfaceError, TextureState,
    TextureUsages,
};
use prism_render_graph::{CompileError, ImportedTexture, RenderGraph, TextureDesc, TextureHandle};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

mod frame_pacing;
mod replay;
mod submission;

pub use frame_pacing::{FramePacer, DEFAULT_FRAMES_IN_FLIGHT};
pub use replay::{ReplayError, ReplayFailure, ReplayHandle};
pub use submission::FrameSubmission;
// The waitable-submission contract is part of the RHI, so every backend can
// satisfy it without depending on this crate. Re-exported for convenience.
pub use prism_render_driver::{CompletionToken, FrameSync};

#[cfg(test)]
mod retirement_tests;
#[cfg(test)]
mod submission_gpu_tests;
#[cfg(test)]
mod tests;

/// Why a compiled frame could not be submitted safely.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SubmissionError {
    /// The graph declaration could not be compiled; no work was submitted.
    Compile(CompileError),
    /// The oldest frame is still outstanding; no new work was recorded or
    /// submitted and its completion token remains in the pacer for retry.
    CompletionUnconfirmed,
    /// This device generation has exhausted its strictly increasing values.
    /// No allocation, recording, pacing wait, or submission was performed.
    CompletionValueExhausted,
}

impl From<CompileError> for SubmissionError {
    fn from(error: CompileError) -> Self {
        Self::Compile(error)
    }
}

impl fmt::Display for SubmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(error) => write!(f, "frame graph failed to compile: {error:?}"),
            Self::CompletionUnconfirmed => {
                f.write_str("oldest frame completion could not be confirmed")
            }
            Self::CompletionValueExhausted => {
                f.write_str("device generation completion values are exhausted")
            }
        }
    }
}

impl std::error::Error for SubmissionError {}

/// Why a [`RenderRuntime`] could not be brought up.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RenderInitError {
    /// No adapter matched the request (no GPU, or none usable).
    NoAdapter,
    /// The chosen adapter rejected the device request; the string carries the
    /// backend's diagnostic.
    Device(String),
}

/// Why rebuilding a lost render device could not complete.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RecoveryError {
    /// No adapter matching the retained selection policy was available.
    NoAdapter,
    /// The replacement device/queue request failed.
    Device(String),
    /// The retained window surface could not be recreated or reconfigured.
    Surface(SurfaceError),
    /// A registered GPU resource bundle could not be replayed.
    Replay(ReplayFailure),
}

/// Evidence returned after a complete device-generation replacement.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RecoveryReport {
    /// The new runtime device generation. The initial device is generation 0.
    pub generation: u64,
    /// The backend-reported reason, when recovery was initiated by polling.
    pub reason: Option<DeviceLostReason>,
    /// Number of live typed replay entries rebuilt and committed.
    pub replayed_resources: usize,
    /// Whether a retained window surface was recreated and reconfigured.
    pub surface_recreated: bool,
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
    /// Pacing could not confirm completion; this frame was not submitted.
    CompletionUnconfirmed,
    /// This device generation has no remaining completion values.
    CompletionValueExhausted,
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSurface => f.write_str("no surface is configured for this runtime"),
            Self::Surface(error) => write!(f, "surface error while presenting: {error}"),
            Self::Compile(error) => write!(f, "frame graph failed to compile: {error:?}"),
            Self::CompletionUnconfirmed => {
                f.write_str("oldest frame completion could not be confirmed")
            }
            Self::CompletionValueExhausted => {
                f.write_str("device generation completion values are exhausted")
            }
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

impl fmt::Display for RecoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => f.write_str("no usable graphics adapter during recovery"),
            Self::Device(detail) => write!(f, "replacement device bring-up failed: {detail}"),
            Self::Surface(error) => write!(f, "surface recovery failed: {error}"),
            Self::Replay(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for RecoveryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Replay(error) => Some(error),
            Self::NoAdapter | Self::Device(_) | Self::Surface(_) => None,
        }
    }
}

trait SurfaceFactory<I: Instance>: Send + Sync {
    fn create(
        &self,
        instance: &I,
        device: &I::Device,
        queue: &I::Queue,
    ) -> Result<I::Surface, SurfaceError>;
}

struct WindowSurfaceFactory<W> {
    window: Arc<W>,
}

impl<I, W> SurfaceFactory<I> for WindowSurfaceFactory<W>
where
    I: Instance,
    W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
{
    fn create(
        &self,
        instance: &I,
        device: &I::Device,
        queue: &I::Queue,
    ) -> Result<I::Surface, SurfaceError> {
        instance.create_surface(Arc::clone(&self.window), device, queue)
    }
}

/// The RHI-backed render runtime: an adapter plus its device/queue, owned for
/// the app's lifetime, with the frame extent graphs resolve against.
pub struct RenderRuntime<I: Instance + 'static> {
    instance: I,
    adapter: I::Adapter,
    adapter_options: RequestAdapterOptions,
    device_descriptor: DeviceDescriptor,
    /// The device is kept behind an `Arc` so background pipeline compilation
    /// can hold a handle while the runtime remains the owner of the render
    /// world's lifetime.
    device: Arc<I::Device>,
    queue: I::Queue,
    extent: Extent3d,
    /// The frames-in-flight bound and its outstanding completion tokens.
    pacer: Mutex<FramePacer<CompletionToken>>,
    /// Submission values used by render resources for precise retirement.
    submitted_value: AtomicU64,
    completed_value: Arc<AtomicU64>,
    /// Installed on the first tracked submission. Retains a shutdown barrier
    /// without requiring non-executing Instance implementations to implement
    /// FrameSync. It also drains the old queue before device replacement.
    submission_barrier: OnceLock<fn(&I::Queue) -> bool>,
    /// The window surface, once created.
    surface: Option<I::Surface>,
    /// Retains the owned OS window so a replacement backend surface can be
    /// created after the old device and its surface become unusable.
    surface_factory: Option<Arc<dyn SurfaceFactory<I>>>,
    /// The configuration the surface was last configured with, so an
    /// `Outdated` surface can be reconfigured without the caller re-supplying it.
    surface_config: Option<SurfaceConfig>,
    replay: replay::ReplayRegistry<I>,
    device_generation: u64,
    pending_device_loss: Mutex<Option<DeviceLostReason>>,
}

impl<I: Instance + 'static> RenderRuntime<I> {
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
        Self::with_device_configuration(
            instance,
            extent,
            max_in_flight,
            options,
            DeviceDescriptor::default(),
        )
    }

    /// Brings up a runtime and retains the exact adapter/device request so the
    /// same requirements can be applied to a replacement device after loss.
    pub fn with_device_configuration(
        instance: I,
        extent: Extent3d,
        max_in_flight: usize,
        adapter_options: RequestAdapterOptions,
        device_descriptor: DeviceDescriptor,
    ) -> Result<Self, RenderInitError> {
        let adapter = instance
            .request_adapter(&adapter_options)
            .ok_or(RenderInitError::NoAdapter)?;
        let (device, queue) = instance
            .create_device(&adapter, &device_descriptor)
            .map_err(|error| RenderInitError::Device(format!("{error}")))?;
        Ok(Self {
            instance,
            adapter,
            adapter_options,
            device_descriptor,
            device: Arc::new(device),
            queue,
            extent,
            pacer: Mutex::new(FramePacer::new(max_in_flight)),
            submitted_value: AtomicU64::new(0),
            completed_value: Arc::new(AtomicU64::new(0)),
            submission_barrier: OnceLock::new(),
            surface: None,
            surface_factory: None,
            surface_config: None,
            replay: replay::ReplayRegistry::new(),
            device_generation: 0,
            pending_device_loss: Mutex::new(None),
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

    /// A shareable handle to the render device.
    ///
    /// Pipeline compilation is independently schedulable (design §8.4), so a
    /// background compiler needs an owned handle rather than a borrow into the
    /// render runtime. Cloning the `Arc` keeps the device alive until the
    /// compiler is dropped.
    #[must_use]
    pub fn device_handle(&self) -> Arc<I::Device> {
        Arc::clone(&self.device)
    }

    /// Returns why the GPU device was lost, retaining the record until recovery
    /// succeeds, or `None` when no loss has been reported.
    ///
    /// Delegates to the backend device's
    /// [`RenderDevice::take_device_lost`](prism_render_driver::RenderDevice::take_device_lost).
    /// A backend that cannot observe loss (a null/headless backend) returns
    /// `None`; the host layer polls this to drive device-loss recovery
    /// (design §24.3).
    #[must_use]
    pub fn device_lost(&self) -> Option<DeviceLostReason> {
        let mut pending = self
            .pending_device_loss
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if pending.is_none() {
            *pending = self.device.take_device_lost();
        }
        *pending
    }

    /// The current device generation. It increments only after device, queue,
    /// surface, and every registered replay entry have all been rebuilt.
    #[must_use]
    pub const fn device_generation(&self) -> u64 {
        self.device_generation
    }

    /// Registers and immediately creates a typed GPU resource bundle.
    ///
    /// The same recipe is executed against every replacement device. Recipes
    /// should rebuild a self-contained dependency bundle (for example a
    /// texture, its view, sampler, bind group, and pipeline) so the returned
    /// handle can be replaced atomically after all registry entries succeed.
    pub fn register_replay<T, F>(
        &mut self,
        label: impl Into<String>,
        recipe: F,
    ) -> Result<ReplayHandle<T>, ReplayFailure>
    where
        T: Send + Sync + 'static,
        F: Fn(&I::Device, &I::Queue) -> Result<T, ReplayError> + Send + Sync + 'static,
    {
        self.replay
            .register(label, &self.device, &self.queue, recipe)
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
        let factory: Arc<dyn SurfaceFactory<I>> = Arc::new(WindowSurfaceFactory { window });
        let surface = factory.create(&self.instance, &self.device, &self.queue)?;
        self.surface = Some(surface);
        self.surface_factory = Some(factory);
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

    /// Rebuilds the adapter, device, queue, retained window surface, and every
    /// registered GPU resource bundle.
    ///
    /// Recovery is transactional from the caller's point of view: replacement
    /// resources are staged first, and replay handles are updated only after
    /// adapter/device creation and surface reconfiguration have all succeeded.
    /// On failure the current generation and every replay handle remain
    /// unchanged, and the precise failure is returned.
    pub fn recover_device(&mut self) -> Result<RecoveryReport, RecoveryError> {
        self.recover_device_with_reason(None)
    }

    /// Polls the backend's edge-triggered device-loss signal and performs a
    /// complete recovery when one is present.
    ///
    /// A failed attempt retains the loss reason, so a later call retries rather
    /// than silently treating the drained backend signal as recovered.
    pub fn recover_if_lost(&mut self) -> Result<Option<RecoveryReport>, RecoveryError> {
        let reason = {
            let mut pending = self
                .pending_device_loss
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if pending.is_none() {
                *pending = self.device.take_device_lost();
            }
            *pending
        };
        let Some(reason) = reason else {
            return Ok(None);
        };
        let report = self.recover_device_with_reason(Some(reason))?;
        Ok(Some(report))
    }

    fn recover_device_with_reason(
        &mut self,
        reason: Option<DeviceLostReason>,
    ) -> Result<RecoveryReport, RecoveryError> {
        let adapter = match self.surface.as_ref() {
            Some(surface) => self
                .instance
                .request_adapter_for_surface(&self.adapter_options, surface),
            None => self.instance.request_adapter(&self.adapter_options),
        }
        .ok_or(RecoveryError::NoAdapter)?;
        let (device, queue) = self
            .instance
            .create_device(&adapter, &self.device_descriptor)
            .map_err(|error| RecoveryError::Device(format!("{error}")))?;

        let mut surface = match &self.surface_factory {
            Some(factory) => Some(
                factory
                    .create(&self.instance, &device, &queue)
                    .map_err(RecoveryError::Surface)?,
            ),
            None => None,
        };
        if let Some(config) = self.surface_config {
            surface
                .as_mut()
                .ok_or(RecoveryError::Surface(SurfaceError::Unsupported))?
                .configure(&config)
                .map_err(RecoveryError::Surface)?;
        }

        let staged = self
            .replay
            .stage(&device, &queue)
            .map_err(RecoveryError::Replay)?;
        let replayed_resources = staged.count();
        if let Some(wait_idle) = self.submission_barrier.get() {
            let _ = wait_idle(&self.queue);
        }
        staged.commit();

        let max_in_flight = self.max_frames_in_flight();
        self.adapter = adapter;
        self.device = Arc::new(device);
        self.queue = queue;
        self.surface = surface;
        self.pacer = Mutex::new(FramePacer::new(max_in_flight));
        self.submitted_value.store(0, AtomicOrdering::Release);
        self.completed_value = Arc::new(AtomicU64::new(0));
        self.device_generation = self.device_generation.saturating_add(1);
        *self
            .pending_device_loss
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;

        Ok(RecoveryReport {
            generation: self.device_generation,
            reason,
            replayed_resources,
            surface_recreated: self.surface.is_some(),
        })
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
        I::Device: Send + Sync + 'static,
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

        if let Err(error) = self.execute(graph) {
            // The frame is acquired but will not be presented. Release its
            // image instead of leaking the registered swapchain ids.
            self.surface
                .as_ref()
                .expect("surface checked above")
                .discard(frame);
            return Err(match error {
                SubmissionError::Compile(error) => FrameError::Compile(error),
                SubmissionError::CompletionUnconfirmed => FrameError::CompletionUnconfirmed,
                SubmissionError::CompletionValueExhausted => FrameError::CompletionValueExhausted,
            });
        }

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

    /// Returns the value that the next successful submission will receive.
    ///
    /// The value is a planning hint for resources prepared before the frame
    /// is submitted. It becomes valid only after [`Self::execute_tracked`]
    /// succeeds. At `u64::MAX` this hint saturates; a subsequent submission
    /// returns [`SubmissionError::CompletionValueExhausted`] instead of reusing
    /// a value. Device recovery starts a new generation with fresh values.
    #[must_use]
    pub fn next_completion_value(&self) -> u64 {
        self.submitted_value
            .load(AtomicOrdering::Acquire)
            .saturating_add(1)
    }

    /// Returns the greatest submission value whose completion callback ran.
    #[must_use]
    pub fn completed_value(&self) -> u64 {
        self.completed_value.load(AtomicOrdering::Acquire)
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
    /// Returns [`SubmissionError`] without recording or submitting work when
    /// compilation fails or the oldest frame's completion cannot be confirmed.
    pub fn execute(&self, graph: RenderGraph) -> Result<(), SubmissionError>
    where
        I::Queue: FrameSync,
        I::Device: Send + Sync + 'static,
    {
        self.execute_tracked(graph).map(|_| ())
    }

    /// Executes and submits `graph`, returning its precise completion value.
    pub fn execute_tracked(&self, graph: RenderGraph) -> Result<u64, SubmissionError>
    where
        I::Queue: FrameSync,
        I::Device: Send + Sync + 'static,
    {
        self.execute_owned(graph)
            .map(|frame| frame.completion_value())
    }

    /// Submits a graph and returns an explicit owner for its physical resources.
    ///
    /// The real queue's callback retains a second owner until GPU completion.
    /// Dropping the returned handle early therefore cannot destroy in-flight
    /// resources. Retain it beyond completion to publish persistent history;
    /// release it only after every later imported use has completed.
    pub fn execute_owned(
        &self,
        graph: RenderGraph,
    ) -> Result<FrameSubmission<I::Device>, SubmissionError>
    where
        I::Queue: FrameSync,
        I::Device: Send + Sync + 'static,
    {
        // Compile before touching the pacer, then record directly. No fake
        // queue, intercepted submission, command-buffer clone, or recompile.
        let prepared = graph.prepare()?;

        // Serialize pacing, recording, value assignment and actual submission:
        // completion values must follow queue order even for concurrent callers.
        // Wait before realization so saturated frames do not allocate early.
        let mut pacer = self.pacer();
        let value = self
            .submitted_value
            .load(AtomicOrdering::Acquire)
            .checked_add(1)
            .ok_or(SubmissionError::CompletionValueExhausted)?;
        if let Some(oldest) = pacer.begin_frame() {
            if !self.queue.wait_for(oldest.as_ref()) {
                pacer.restore_oldest(oldest);
                return Err(SubmissionError::CompletionUnconfirmed);
            }
        }
        let recorded = prepared.record(&*self.device);
        let (commands, _plan, allocations) = recorded.into_parts();

        self.submitted_value.store(value, AtomicOrdering::Release);
        self.submission_barrier
            .get_or_init(|| <I::Queue as FrameSync>::wait_idle);
        let frame = FrameSubmission::new(
            value,
            self.device_generation,
            Arc::clone(&self.device),
            allocations,
        );
        let token = self.queue.submit_tracked(core::slice::from_ref(&commands));
        pacer.record(token);
        drop(pacer);
        let completed = Arc::clone(&self.completed_value);
        let completion_owner = frame.clone();
        self.queue.on_submitted_work_done(Box::new(move || {
            completion_owner.mark_complete();
            drop(completion_owner);
            completed.fetch_max(value, AtomicOrdering::Release);
        }));
        Ok(frame)
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
        let mut pacer = self.pacer();
        let outstanding: Vec<CompletionToken> = pacer.drain().collect();
        let mut unconfirmed = Vec::new();
        for token in outstanding {
            if !self.queue.wait_for(token.as_ref()) {
                unconfirmed.push(token);
            }
        }
        // Always reach for the queue-wide barrier too: it also covers work the
        // runtime did not submit through `execute`.
        let idle = self.queue.wait_idle();
        if !idle {
            for token in unconfirmed {
                pacer.record(token);
            }
        }
        idle
    }
}

impl<I: Instance + 'static> Drop for RenderRuntime<I> {
    fn drop(&mut self) {
        // A registered callback owns allocations and its creating device. Drive
        // the queue to completion before dropping its last runtime handle, so
        // normal teardown does not leave callbacks retaining a device cycle.
        // Failed waits do not authorize early resource destruction.
        if let Some(wait_idle) = self.submission_barrier.get() {
            let _ = wait_idle(&self.queue);
        }
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
