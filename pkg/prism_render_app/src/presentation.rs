//! Per-window ownership and the single acquire → render → present driver.
//!
//! The main world publishes owned window handles and desired physical sizes.
//! Extraction copies that intent; GPU surfaces and textures stay in RenderApp.
//! Each window has its own color/depth pair and format-specific final pass.
//! Device replacement is observed through the runtime generation, without
//! participating in (or duplicating) the renderer's device recovery protocol.

use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

use prism_app::presentation::{
    FeedbackSupport, PresentationBridge, PresentationClockDomain, PresentationConfig,
    PresentationHostClock, PresentationPacing, PresentationTarget, UnsupportedFeedback,
};
use prism_app::{App, Plugin};
use prism_ecs::prelude::Resource;
use prism_ecs::world::World;
use prism_render_architecture::lighting::ocio::OcioConfig;
use prism_render_driver::{
    CompositeAlphaMode, FrameSync, Instance, PresentMode, RenderDevice, Surface, SurfaceConfig,
    SurfaceError, TextureFormat,
};
use prism_render_runtime::RenderRuntime;
pub use prism_window::WindowId;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::display_transform_gpu::DisplayTransformGpu;
use crate::lighting_gpu::LightingGpuPass;
use crate::lighting_trace_gpu::TraceGraphRecorder;
use crate::{
    DrawFunctions, RenderApp, RenderFrameStats, RenderQueues, RenderTarget, RenderViewport,
};

mod blit;
use blit::PresentPipeline;

type SurfaceFactory<I> =
    Arc<dyn Fn(&RenderRuntime<I>) -> Result<<I as Instance>::Surface, SurfaceError> + Send + Sync>;

/// Preferred swapchain settings. Unsupported preferences fall back to advertised
/// capabilities; an empty capability list is an explicit per-window failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowSurfaceOptions {
    /// Prefer this format, otherwise use the surface's first advertised format.
    pub format: Option<TextureFormat>,
    /// Preferred presentation mode (FIFO by default).
    pub present_mode: PresentMode,
    /// Preferred platform alpha composition.
    pub alpha_mode: CompositeAlphaMode,
    /// Backend frame latency hint, clamped to at least one.
    pub maximum_frame_latency: u32,
}

impl Default for WindowSurfaceOptions {
    fn default() -> Self {
        Self {
            format: None,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Auto,
            maximum_frame_latency: 2,
        }
    }
}

struct WindowRequest<I: Instance + 'static> {
    factory: SurfaceFactory<I>,
    viewport: RenderViewport,
    visible: bool,
    options: WindowSurfaceOptions,
}

impl<I: Instance + 'static> Clone for WindowRequest<I> {
    fn clone(&self) -> Self {
        Self {
            factory: Arc::clone(&self.factory),
            viewport: self.viewport,
            visible: self.visible,
            options: self.options,
        }
    }
}

/// Main-world window intent. The first attachment selects window presentation
/// in place of the legacy single offscreen frame. Removing the last window
/// produces no GPU work; call [`Self::use_offscreen`] to restore that legacy mode.
/// Handles are retained until a synchronized extraction retires their surfaces.
pub struct WindowRenderRequests<I: Instance + 'static> {
    enabled: bool,
    windows: BTreeMap<WindowId, WindowRequest<I>>,
}

impl<I: Instance + 'static> Default for WindowRenderRequests<I> {
    fn default() -> Self {
        Self {
            enabled: false,
            windows: BTreeMap::new(),
        }
    }
}

impl<I: Instance + 'static> Clone for WindowRenderRequests<I> {
    fn clone(&self) -> Self {
        Self {
            enabled: self.enabled,
            windows: self.windows.clone(),
        }
    }
}

impl<I: Instance + 'static> Resource for WindowRenderRequests<I> {}

impl<I: Instance + 'static> WindowRenderRequests<I> {
    /// Attach a real platform window. Replacing an id retires its old surface
    /// after GPU completion. The factory retains the Arc for device recovery.
    pub fn attach<W>(&mut self, id: WindowId, window: Arc<W>, viewport: RenderViewport)
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        self.attach_factory(id, viewport, move |runtime| {
            runtime.instance().create_surface(
                Arc::clone(&window),
                runtime.device(),
                runtime.queue(),
            )
        });
    }

    /// Attach a backend-owned surface recipe (also useful for embedded hosts).
    /// The recipe must create a fresh surface on the supplied runtime's device
    /// each time, including after a device-generation change.
    pub fn attach_factory<F>(&mut self, id: WindowId, viewport: RenderViewport, factory: F)
    where
        F: Fn(&RenderRuntime<I>) -> Result<I::Surface, SurfaceError> + Send + Sync + 'static,
    {
        self.enabled = true;
        self.windows.insert(
            id,
            WindowRequest {
                factory: Arc::new(factory),
                viewport,
                visible: true,
                options: WindowSurfaceOptions::default(),
            },
        );
    }

    /// Update physical size and visibility without recreating the window handle.
    /// Hidden/zero-size windows neither acquire nor submit; other windows proceed.
    pub fn update(&mut self, id: WindowId, viewport: RenderViewport, visible: bool) -> bool {
        let Some(window) = self.windows.get_mut(&id) else {
            return false;
        };
        window.viewport = viewport;
        window.visible = visible;
        true
    }

    /// Change one window's preferred swapchain settings.
    pub fn set_options(&mut self, id: WindowId, options: WindowSurfaceOptions) -> bool {
        let Some(window) = self.windows.get_mut(&id) else {
            return false;
        };
        window.options = options;
        true
    }

    /// Remove a window. The render side releases it after GPU completion.
    pub fn remove(&mut self, id: WindowId) -> bool {
        self.windows.remove(&id).is_some()
    }

    /// Whether this id already has an owned window recipe.
    pub fn contains(&self, id: WindowId) -> bool {
        self.windows.contains_key(&id)
    }

    /// Window ids in deterministic presentation order.
    pub fn ids(&self) -> impl Iterator<Item = WindowId> + '_ {
        self.windows.keys().copied()
    }

    /// Retire all windows and explicitly restore the original offscreen path.
    pub fn use_offscreen(&mut self) {
        self.enabled = false;
        self.windows.clear();
    }
}

/// Optional render-world queues per window. An absent override renders the
/// global scene queue; an explicit empty queue renders only that window's clear.
/// Overrides persist until replaced/removed, so hosts control their update rate.
#[derive(Default, Resource)]
pub struct WindowRenderQueues(pub BTreeMap<WindowId, RenderQueues>);

/// Observable failure of a single window, leaving other windows runnable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WindowRenderError {
    /// Surface creation/configuration/acquisition/presentation failed.
    Surface(SurfaceError),
    /// A graph failed before submission.
    Graph(String),
    /// GPU completion could not be confirmed; resources have been retained.
    CompletionUnconfirmed,
}

/// Latest result of driving one window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WindowFrameStatus {
    /// Zero size, hidden, occluded, or temporarily unavailable.
    Skipped,
    /// Submitted and handed to the platform compositor.
    Presented {
        /// Runtime submission value (not a display-scanout guarantee).
        submission: u64,
    },
    /// A failure scoped to this window.
    Failed(WindowRenderError),
}

struct WindowGpu<I: Instance + 'static> {
    factory: SurfaceFactory<I>,
    generation: u64,
    device_generation: u64,
    surface: I::Surface,
    config: SurfaceConfig,
    present: PresentPipeline<I::Device>,
    target: OwnedTarget<I::Device>,
    display_transform: DisplayTransformGpu,
    lifetime: Arc<WindowLifetime>,
    needs_reconfigure: bool,
    lost: bool,
}

struct WindowLifetime {
    in_flight: Arc<AtomicUsize>,
    destroy_allowed: Arc<AtomicBool>,
}

impl WindowLifetime {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            in_flight: Arc::new(AtomicUsize::new(0)),
            destroy_allowed: Arc::new(AtomicBool::new(true)),
        })
    }

    fn safe(&self) -> bool {
        self.destroy_allowed.load(Ordering::Acquire) && self.in_flight.load(Ordering::Acquire) == 0
    }

    fn completed(&self) {
        let _ = self
            .in_flight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                Some(count.saturating_sub(1))
            });
    }

    fn mark_idle(&self) {
        self.in_flight.store(0, Ordering::Release);
    }
}

struct OwnedTarget<D: RenderDevice> {
    device: Arc<D>,
    target: RenderTarget,
    lifetime: Arc<WindowLifetime>,
}

impl<D: RenderDevice> OwnedTarget<D> {
    fn new(device: Arc<D>, target: RenderTarget, lifetime: Arc<WindowLifetime>) -> Self {
        Self {
            device,
            target,
            lifetime,
        }
    }
}

impl<D: RenderDevice> std::ops::Deref for OwnedTarget<D> {
    type Target = RenderTarget;
    fn deref(&self) -> &Self::Target {
        &self.target
    }
}

impl<D: RenderDevice> Drop for OwnedTarget<D> {
    fn drop(&mut self) {
        if self.lifetime.safe() {
            self.target.destroy(&*self.device);
        }
    }
}

fn presentation_config() -> PresentationConfig {
    PresentationConfig {
        feedback: FeedbackSupport::Unsupported(UnsupportedFeedback::BackendApi),
        interval_ns: 16_666_667,
        max_frames_in_flight: 2,
        safety_margin_ns: 1_000_000,
        vrr: None,
    }
}

/// Render-world owner of all live surfaces and independent frame targets.
/// Installed by PrismRenderPlugin; no GPU object is shared with the main world.
pub struct WindowRenderTargets<I: Instance + 'static> {
    windows: BTreeMap<WindowId, WindowGpu<I>>,
    status: BTreeMap<WindowId, WindowFrameStatus>,
    next_generation: u64,
    bridge: PresentationBridge,
}

impl<I: Instance + 'static> Default for WindowRenderTargets<I> {
    fn default() -> Self {
        Self::with_bridge(PresentationBridge::new(PresentationHostClock::new(
            PresentationClockDomain(1),
        )))
    }
}

impl<I: Instance + 'static> WindowRenderTargets<I> {
    fn with_bridge(bridge: PresentationBridge) -> Self {
        Self {
            windows: BTreeMap::new(),
            status: BTreeMap::new(),
            next_generation: 0,
            bridge,
        }
    }
}

impl<I> Resource for WindowRenderTargets<I>
where
    I: Instance + Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Surface: Send + Sync + 'static,
{
}

impl<I: Instance + 'static> WindowRenderTargets<I> {
    /// The independently allocated color/depth pair, if prepared.
    pub fn target(&self, id: WindowId) -> Option<&RenderTarget> {
        self.windows.get(&id).map(|window| &*window.target)
    }

    /// The negotiated surface configuration.
    pub fn surface_config(&self, id: WindowId) -> Option<SurfaceConfig> {
        self.windows.get(&id).map(|window| window.config)
    }

    /// Last outcome for an attached window.
    pub fn status(&self, id: WindowId) -> Option<&WindowFrameStatus> {
        self.status.get(&id)
    }

    /// Number of live GPU target/surface pairs.
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Whether all GPU surfaces have been released.
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Drain GPU work before releasing surfaces (required before platform
    /// suspension). Failure retains every resource for a later safe retry.
    pub fn release(&mut self, runtime: &RenderRuntime<I>) -> bool
    where
        I::Queue: FrameSync,
    {
        if !self.windows.is_empty() && !runtime.wait_idle() {
            for (&id, gpu) in &self.windows {
                gpu.lifetime.destroy_allowed.store(false, Ordering::Release);
                self.status.insert(
                    id,
                    WindowFrameStatus::Failed(WindowRenderError::CompletionUnconfirmed),
                );
            }
            return false;
        }
        for gpu in self.windows.values() {
            gpu.lifetime.destroy_allowed.store(true, Ordering::Release);
            gpu.lifetime.mark_idle();
        }
        for (&id, gpu) in &self.windows {
            self.bridge.unregister(PresentationTarget {
                id: id.0,
                generation: gpu.generation,
            });
        }
        self.windows.clear();
        self.status.clear();
        true
    }

    pub(crate) fn drive(
        &mut self,
        runtime: &RenderRuntime<I>,
        requests: &WindowRenderRequests<I>,
        overrides: &WindowRenderQueues,
        queues: &RenderQueues,
        functions: &DrawFunctions,
        lighting: Option<LightingGpuPass>,
        trace: Option<TraceGraphRecorder>,
        stats: &mut RenderFrameStats,
    ) -> bool
    where
        I::Device: Send + Sync + 'static,
        I::Queue: FrameSync,
    {
        let obsolete: Vec<_> = self
            .windows
            .iter()
            .filter_map(|(&id, gpu)| {
                let keep = requests.enabled
                    && requests.windows.get(&id).is_some_and(|request| {
                        gpu.device_generation == runtime.device_generation()
                            && Arc::ptr_eq(&gpu.factory, &request.factory)
                    });
                (!keep).then_some(id)
            })
            .collect();
        if !obsolete.is_empty() {
            if !runtime.wait_idle() {
                for id in obsolete {
                    self.status.insert(
                        id,
                        WindowFrameStatus::Failed(WindowRenderError::CompletionUnconfirmed),
                    );
                }
                stats.frames_failed += 1;
                return true;
            }
            for id in obsolete {
                if let Some(gpu) = self.windows.remove(&id) {
                    let replaced = gpu.device_generation != runtime.device_generation();
                    if replaced {
                        gpu.lifetime.destroy_allowed.store(false, Ordering::Release);
                    } else {
                        gpu.lifetime.mark_idle();
                    }
                    self.bridge.unregister(PresentationTarget {
                        id: id.0,
                        generation: gpu.generation,
                    });
                }
            }
        }
        self.status
            .retain(|id, _| requests.windows.contains_key(id));
        if !requests.enabled {
            return false;
        }
        for (&id, request) in &requests.windows {
            stats.last_viewport = request.viewport;
            if let Some(gpu) = self.windows.get(&id) {
                self.bridge.set_active(
                    PresentationTarget {
                        id: id.0,
                        generation: gpu.generation,
                    },
                    request.visible && !request.viewport.is_empty(),
                );
            }
            let result = self.drive_window(
                id,
                request,
                runtime,
                overrides.0.get(&id).unwrap_or(queues),
                functions,
                lighting,
                trace,
                stats,
            );
            let status = match result {
                Ok(Some(submission)) => WindowFrameStatus::Presented { submission },
                Ok(None) => {
                    stats.frames_skipped += 1;
                    WindowFrameStatus::Skipped
                }
                Err(error) => {
                    stats.frames_failed += 1;
                    WindowFrameStatus::Failed(error)
                }
            };
            self.status.insert(id, status);
        }
        true
    }

    fn drive_window(
        &mut self,
        id: WindowId,
        request: &WindowRequest<I>,
        runtime: &RenderRuntime<I>,
        queues: &RenderQueues,
        functions: &DrawFunctions,
        lighting: Option<LightingGpuPass>,
        trace: Option<TraceGraphRecorder>,
        stats: &mut RenderFrameStats,
    ) -> Result<Option<u64>, WindowRenderError>
    where
        I::Device: Send + Sync + 'static,
        I::Queue: FrameSync,
    {
        if request.viewport.is_empty() || !request.visible {
            return Ok(None);
        }
        let rebuild = self.windows.get(&id).is_some_and(|gpu| gpu.lost);
        if rebuild {
            if !runtime.wait_idle() {
                return Err(WindowRenderError::CompletionUnconfirmed);
            }
            if let Some(gpu) = self.windows.remove(&id) {
                gpu.lifetime.mark_idle();
                self.bridge.unregister(PresentationTarget {
                    id: id.0,
                    generation: gpu.generation,
                });
            }
        }
        if !self.windows.contains_key(&id) {
            let mut surface = (request.factory)(runtime).map_err(WindowRenderError::Surface)?;
            let config = negotiate(&surface, request).map_err(WindowRenderError::Surface)?;
            surface
                .configure(&config)
                .map_err(WindowRenderError::Surface)?;
            self.next_generation = self.next_generation.saturating_add(1).max(1);
            let lifetime = WindowLifetime::new();
            let device = runtime.device_handle();
            let target = OwnedTarget::new(
                Arc::clone(&device),
                RenderTarget::new(&*device, request.viewport.extent()),
                Arc::clone(&lifetime),
            );
            let display_device: Arc<dyn RenderDevice + Send + Sync> = device.clone();
            let display_transform = DisplayTransformGpu::new(
                display_device,
                target.view(),
                request.viewport.extent(),
                Arc::clone(&lifetime.in_flight),
                Arc::clone(&lifetime.destroy_allowed),
            )
            .map_err(|error| WindowRenderError::Graph(format!("display transform: {error:?}")))?;
            let present = PresentPipeline::new(
                Arc::clone(&device),
                display_transform.output_view(),
                config.format,
                Arc::clone(&lifetime.in_flight),
                Arc::clone(&lifetime.destroy_allowed),
            );
            let generation = self.next_generation;
            self.windows.insert(
                id,
                WindowGpu {
                    factory: Arc::clone(&request.factory),
                    generation,
                    device_generation: runtime.device_generation(),
                    surface,
                    config,
                    target,
                    display_transform,
                    present,
                    lifetime: Arc::clone(&lifetime),
                    needs_reconfigure: false,
                    lost: false,
                },
            );
            let target = PresentationTarget {
                id: id.0,
                generation,
            };
            self.bridge.register(target, presentation_config());
            self.bridge.set_active(target, request.visible);
        }
        let gpu = self.windows.get_mut(&id).expect("window prepared");
        let config = negotiate(&gpu.surface, request).map_err(WindowRenderError::Surface)?;
        if gpu.config != config || gpu.needs_reconfigure {
            if !runtime.wait_idle() {
                return Err(WindowRenderError::CompletionUnconfirmed);
            }
            if let Err(error) = gpu.surface.configure(&config) {
                gpu.lost = error == SurfaceError::Lost;
                return Err(WindowRenderError::Surface(error));
            }
            if gpu.target.viewport() != request.viewport || gpu.config.format != config.format {
                gpu.lifetime.mark_idle();
                let lifetime = WindowLifetime::new();
                let device = runtime.device_handle();
                let target = OwnedTarget::new(
                    Arc::clone(&device),
                    RenderTarget::new(&*device, request.viewport.extent()),
                    Arc::clone(&lifetime),
                );
                let display_device: Arc<dyn RenderDevice + Send + Sync> = device.clone();
                let display_transform = DisplayTransformGpu::new(
                    display_device,
                    target.view(),
                    request.viewport.extent(),
                    Arc::clone(&lifetime.in_flight),
                    Arc::clone(&lifetime.destroy_allowed),
                )
                .map_err(|error| {
                    WindowRenderError::Graph(format!("display transform: {error:?}"))
                })?;
                let present = PresentPipeline::new(
                    Arc::clone(&device),
                    display_transform.output_view(),
                    config.format,
                    Arc::clone(&lifetime.in_flight),
                    Arc::clone(&lifetime.destroy_allowed),
                );
                gpu.present = present;
                gpu.target = target;
                gpu.display_transform = display_transform;
                gpu.lifetime = lifetime;
            }
            gpu.config = config;
            gpu.needs_reconfigure = false;
        }
        let frame = match gpu.surface.acquire() {
            Ok(frame) => frame,
            Err(SurfaceError::Occluded | SurfaceError::Timeout) => return Ok(None),
            Err(error) => {
                gpu.needs_reconfigure = error == SurfaceError::Outdated;
                gpu.lost = error == SurfaceError::Lost;
                return Err(WindowRenderError::Surface(error));
            }
        };
        let lease = SurfaceLease {
            surface: &gpu.surface,
            frame: Some(frame),
        };
        let image = lease
            .surface
            .image(lease.frame.as_ref().expect("acquired frame"));
        if image.size != request.viewport.extent() || image.format != gpu.config.format {
            gpu.needs_reconfigure = true;
            return Err(WindowRenderError::Surface(SurfaceError::Outdated));
        }
        let (mut graph, color) = crate::frame::target_graph_output_with_trace(
            &gpu.target,
            queues,
            functions,
            lighting,
            trace,
        );
        let transformed = gpu
            .display_transform
            .append(&mut graph, runtime.queue(), &OcioConfig::srgb(), color)
            .map_err(|error| WindowRenderError::Graph(format!("display transform: {error:?}")))?;
        gpu.present.append(&mut graph, transformed, image);
        let mut stages = *graph
            .blackboard()
            .get::<crate::frame::FrameGraphStages>()
            .expect("the host frame graph must publish its stage receipt");
        stages.present_source = transformed;
        stages.present_passes = 1;
        stages.assert_contract();
        graph.blackboard_mut().set(stages);
        stats.execution_trace.graph = Some(stages);
        let passes = graph.pass_count() as u64;
        gpu.lifetime.in_flight.fetch_add(1, Ordering::AcqRel);
        let submission = match runtime.execute_tracked(graph) {
            Ok(submission) => submission,
            Err(error) => {
                gpu.lifetime.in_flight.fetch_sub(1, Ordering::AcqRel);
                return Err(WindowRenderError::Graph(format!("{error:?}")));
            }
        };
        stats.frames_submitted += 1;
        stats.passes_declared += passes;
        stats.draws_recorded += queues.len() as u64;
        if let Some(lighting) = lighting {
            stats.lighting_gpu_passes_recorded += lighting.pass_count();
        }
        stats.trace_gpu_dispatches_recorded += u64::from(trace.is_some());
        let target = PresentationTarget {
            id: id.0,
            generation: gpu.generation,
        };
        self.bridge.submitted(target, submission, None);
        let lifetime = Arc::clone(&gpu.lifetime);
        let bridge = self.bridge.clone();
        runtime.queue().on_submitted_work_done(Box::new(move || {
            lifetime.completed();
            bridge.retire_unpresented(target, submission);
        }));
        if let Err(error) = lease.present() {
            gpu.needs_reconfigure = error == SurfaceError::Outdated;
            gpu.lost = error == SurfaceError::Lost;
            // Submission already succeeded; only the presentation failed.
            return Err(WindowRenderError::Surface(error));
        }
        Ok(Some(submission))
    }
}

fn negotiate<I: Instance>(
    surface: &I::Surface,
    request: &WindowRequest<I>,
) -> Result<SurfaceConfig, SurfaceError> {
    let caps = surface.capabilities();
    let format = request
        .options
        .format
        .filter(|format| caps.formats.contains(format))
        .or_else(|| caps.formats.first().copied())
        .ok_or(SurfaceError::Unsupported)?;
    let present_mode = if caps.present_modes.contains(&request.options.present_mode) {
        request.options.present_mode
    } else {
        *caps
            .present_modes
            .first()
            .ok_or(SurfaceError::Unsupported)?
    };
    let alpha_mode = if caps.alpha_modes.contains(&request.options.alpha_mode) {
        request.options.alpha_mode
    } else {
        *caps.alpha_modes.first().ok_or(SurfaceError::Unsupported)?
    };
    Ok(SurfaceConfig {
        format,
        present_mode,
        alpha_mode,
        width: request.viewport.width,
        height: request.viewport.height,
        desired_maximum_frame_latency: request.options.maximum_frame_latency.max(1),
    })
}

/// Unwinding and graph failures release the acquired image exactly once.
struct SurfaceLease<'a, S: Surface> {
    surface: &'a S,
    frame: Option<S::Frame>,
}
impl<S: Surface> SurfaceLease<'_, S> {
    fn present(mut self) -> Result<(), SurfaceError> {
        self.surface
            .present(self.frame.take().expect("owned frame"))
    }
}
impl<S: Surface> Drop for SurfaceLease<'_, S> {
    fn drop(&mut self) {
        if let Some(frame) = self.frame.take() {
            self.surface.discard(frame);
        }
    }
}

/// Narrow plugin hook: initialize the optional window path without changing the
/// existing render schedule, transform upload, or device-recovery driver.
pub(crate) fn install<I>(app: &mut App)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    let bridge = app
        .world()
        .get_resource::<PresentationBridge>()
        .cloned()
        .unwrap_or_else(|| {
            let bridge =
                PresentationBridge::new(PresentationHostClock::new(PresentationClockDomain(1)));
            app.insert_resource(bridge.clone());
            bridge
        });
    app.insert_resource(bridge.clone());
    app.init_resource::<WindowRenderRequests<I>>();
    let render = app
        .sub_app_mut(RenderApp)
        .expect("render sub-app installed");
    render.world.init_resource::<WindowRenderRequests<I>>();
    render
        .world
        .insert_resource(WindowRenderTargets::<I>::with_bridge(bridge));
    render.world.init_resource::<WindowRenderQueues>();
    app.add_extract(RenderApp, extract::<I>);
}

fn extract<I: Instance + 'static>(main: &mut World, render: &mut World) {
    render.insert_resource(main.resource::<WindowRenderRequests<I>>().clone());
}

pub(crate) fn shutdown<I>(world: &mut World)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    if let Some(mut windows) = world.remove_resource::<WindowRenderTargets<I>>() {
        windows.release(
            world
                .resource::<crate::RenderRuntimeResource<I>>()
                .runtime(),
        );
        world.insert_resource(windows);
    }
}

/// Opt-in connection to App's WinitRunner. Add after PrismRenderPlugin. Other
/// embedded hosts can directly update WindowRenderRequests instead.
#[cfg(feature = "winit")]
pub struct WinitPresentationPlugin<I>(core::marker::PhantomData<fn() -> I>);

#[cfg(feature = "winit")]
impl<I> Default for WinitPresentationPlugin<I> {
    fn default() -> Self {
        Self(core::marker::PhantomData)
    }
}

#[cfg(feature = "winit")]
impl<I> Plugin for WinitPresentationPlugin<I>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    fn build(&self, app: &mut App) {
        assert!(
            app.sub_app_mut(RenderApp).is_some(),
            "add PrismRenderPlugin first"
        );
        app.init_resource::<PresentationPacing>();
        app.insert_resource(prism_app::runner::WindowRenderBridge(publish_winit::<I>));
    }
}

#[cfg(feature = "winit")]
fn publish_winit<I>(
    app: &mut App,
    backend: &prism_window_winit::WinitBackend,
    ids: &[WindowId],
    suspended: bool,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    let requests = app.world_mut().resource_mut::<WindowRenderRequests<I>>();
    requests.enabled = true;
    requests.windows.retain(|id, _| ids.contains(id));
    for &id in ids {
        if let Some(window) = backend.window(id) {
            let size = window.inner_size();
            let viewport = RenderViewport::new(size.width, size.height);
            if !requests.contains(id) {
                requests.attach(id, Arc::clone(&window), viewport);
            }
            let visible = !suspended
                && window.is_visible().unwrap_or(true)
                && !window.is_minimized().unwrap_or(false)
                && !backend.realized(id).is_some_and(|state| state.occluded);
            requests.update(id, viewport, visible);
        }
    }
    if suspended {
        if let Some(render) = app.sub_app_mut(RenderApp) {
            shutdown::<I>(&mut render.world);
        }
    }
}

#[cfg(test)]
mod tests;
