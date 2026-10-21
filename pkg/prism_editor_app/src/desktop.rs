//! Native event-driven desktop ownership. GPU resources never own the model.

use alloc::{string::String, sync::Arc, task::Wake};
use core::{fmt, task};
use std::time::{Duration, Instant};

use prism_editor_ui::SelectionMode;
use prism_ui_layout::{Point, Rect, Size};
use prism_ui_render_backend::{DrawCommand, DrawList, GpuPresenter};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalPosition, LogicalSize},
    event::{ElementState, Ime, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    keyboard::{Key, ModifiersState, NamedKey},
    window::{Window, WindowId},
};

use crate::{
    DesktopDocumentSession, DesktopKey, DesktopUiError, EditorController, EditorDesktopUi,
};

/// Native window and bounded validation-run configuration.
#[derive(Clone, Debug)]
pub struct DesktopConfig {
    /// Window title; kept separate from project data.
    pub title: String,
    /// Initial logical client extent.
    pub logical_size: [u32; 2],
    /// Maximum expanded Loom primitives retained on the GPU.
    pub max_primitives: usize,
    /// Stop after this many successful presentations, or run interactively.
    pub frame_limit: Option<u64>,
    /// Optional wall-clock bound for automated native runs.
    pub time_limit: Option<Duration>,
    /// Maximum consecutive recoverable acquisition failures.
    pub max_surface_retries: u32,
    /// Bound device recreation across the entire native session.
    pub max_device_recoveries: u32,
}

impl Default for DesktopConfig {
    fn default() -> Self {
        Self {
            title: "Loom Studio".into(),
            logical_size: [1280, 800],
            max_primitives: 131_072,
            frame_limit: None,
            time_limit: None,
            max_surface_retries: 8,
            max_device_recoveries: 3,
        }
    }
}

/// Device handles shared by Loom presentation and an injected Prism renderer.
pub struct DesktopGpu {
    /// Adapter used to validate format capabilities and create scene targets.
    pub adapter: wgpu::Adapter,
    /// Native device.
    pub device: wgpu::Device,
    /// Single submission/presentation queue.
    pub queue: wgpu::Queue,
    /// Configured surface format.
    pub format: wgpu::TextureFormat,
}

/// Clipped viewport geometry from the actual retained Loom layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DesktopViewportBounds {
    /// Absolute logical coordinates used by native input.
    pub logical: Rect,
    /// Absolute device-pixel coordinates used by GPU viewport/scissor commands.
    pub physical: Rect,
    /// Full native client surface extent in device pixels.
    pub surface_extent: [u32; 2],
}

/// Nonblocking renderer work reported to the event loop.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DesktopWork {
    /// A scene or model change requires a new presentation.
    pub redraw: bool,
    /// GPU readback or another bounded operation needs further polling.
    pub pending: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DesktopPending {
    renderer: bool,
    document: bool,
    ui: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DesktopPendingSource {
    Renderer,
    Document,
    Ui,
}

impl DesktopPending {
    fn set(&mut self, source: DesktopPendingSource, pending: bool) {
        match source {
            DesktopPendingSource::Renderer => self.renderer = pending,
            DesktopPendingSource::Document => self.document = pending,
            DesktopPendingSource::Ui => self.ui = pending,
        }
    }

    const fn any(self) -> bool {
        self.renderer || self.document || self.ui
    }
}

/// Engine-neutral native viewport adapter, sharing the host's GPU submission.
/// Implementations retain scene resources, reject stale picks, and update the
/// controller through commands rather than keeping a second authority.
pub trait DesktopRenderer {
    /// Whether the renderer owns an unfinished transform gesture.
    fn gesture_active(&self) -> bool {
        false
    }
    /// Hit-test transform handles before starting selection.
    fn gesture_press(
        &mut self,
        _ui: &mut EditorDesktopUi,
        _position: [f32; 2],
    ) -> Result<bool, DesktopError> {
        Ok(false)
    }
    /// Update captured input, including coordinates outside the viewport.
    fn gesture_move(
        &mut self,
        _ui: &mut EditorDesktopUi,
        _position: [f32; 2],
    ) -> Result<bool, DesktopError> {
        Ok(false)
    }
    /// Commit one document command on release.
    fn gesture_release(&mut self, _ui: &mut EditorDesktopUi) -> Result<bool, DesktopError> {
        Ok(false)
    }
    /// Restore the drag-start state on Escape, focus loss or suspension.
    fn gesture_cancel(&mut self, _ui: &mut EditorDesktopUi) -> Result<bool, DesktopError> {
        Ok(false)
    }
    /// Select the translation, rotation or scale tool.
    fn transform_tool(
        &mut self,
        _mode: prism_editor_viewport::GizmoMode,
    ) -> Result<bool, DesktopError> {
        Ok(false)
    }
    /// Create device-local resources after initial launch or native resume.
    fn initialize(&mut self, gpu: &DesktopGpu) -> Result<(), DesktopError>;
    /// Draw only inside the retained viewport after Loom paints the surface.
    fn encode(
        &mut self,
        gpu: &DesktopGpu,
        ui: &mut EditorDesktopUi,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        bounds: DesktopViewportBounds,
    ) -> Result<(), DesktopError>;
    /// Enqueue a pick at viewport-local logical coordinates, without waiting.
    fn pointer(
        &mut self,
        gpu: &DesktopGpu,
        ui: &mut EditorDesktopUi,
        position: [f32; 2],
        mode: SelectionMode,
    ) -> Result<bool, DesktopError>;
    /// Enqueue a rectangular viewport selection without blocking the event
    /// loop. Renderers that do not support it leave the gesture unhandled.
    fn box_select(
        &mut self,
        _gpu: &DesktopGpu,
        _ui: &mut EditorDesktopUi,
        _start: [f32; 2],
        _end: [f32; 2],
        _mode: SelectionMode,
    ) -> Result<bool, DesktopError> {
        Ok(false)
    }
    /// Poll device-local work without blocking the native event loop.
    fn poll(
        &mut self,
        gpu: &DesktopGpu,
        ui: &mut EditorDesktopUi,
    ) -> Result<DesktopWork, DesktopError>;
    /// Builds a document-dependent renderer candidate without publishing it.
    /// The host owns the mapping from reflected document data to render
    /// descriptors; the application layer does not assume gameplay types.
    fn prepare_document(&mut self, _controller: &EditorController) -> Result<(), DesktopError> {
        Ok(())
    }
    /// Drops a document candidate after UI publication failed.
    fn discard_document(&mut self) {}
    /// Publishes the candidate prepared by `prepare_document`. Implementations
    /// must make this operation infallible after their preparation checks.
    fn commit_document(&mut self) {
        self.document_reloaded();
    }
    /// Invalidates renderer-side scene and picking state after a document load.
    /// UI-only renderers may keep the default no-op implementation.
    fn document_reloaded(&mut self) {}
    /// Drop device-local resources before a suspended host drops its surface.
    /// CPU scene data and unsaved model state must remain available for resume.
    fn suspend(&mut self);
}

/// Native lifecycle failure. The returned outcome always retains the model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DesktopError {
    /// Invalid dimensions, budgets, or run bounds.
    InvalidConfig,
    /// Native event-loop or window creation failed.
    Window(String),
    /// Device, surface, or GPU validation failed.
    Gpu(String),
    /// The injected scene renderer rejected an operation.
    Renderer(String),
    /// Retained UI reconciliation or input failed fatally.
    Ui(String),
    /// Document persistence or validation failed.
    Document(String),
    /// A configured document session blocked close to preserve editor state.
    UnsavedChanges,
    /// A validation run failed to reach its presentation bound in time.
    TimeLimit,
}

impl fmt::Display for DesktopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => f.write_str("invalid desktop configuration"),
            Self::Window(e) => write!(f, "native window: {e}"),
            Self::Gpu(e) => write!(f, "native GPU: {e}"),
            Self::Renderer(e) => write!(f, "viewport renderer: {e}"),
            Self::Ui(e) => write!(f, "retained UI: {e}"),
            Self::Document(e) => write!(f, "document: {e}"),
            Self::UnsavedChanges => {
                f.write_str("unsaved editor changes must be saved before closing")
            }
            Self::TimeLimit => f.write_str("native presentation time limit exceeded"),
        }
    }
}

impl std::error::Error for DesktopError {}

/// Evidence from this native run, separate from fixture or offscreen tests.
#[derive(Clone, Debug, Default)]
pub struct DesktopReport {
    /// Successfully submitted and presented frames.
    pub presented_frames: u64,
    /// Presented native targets that encoded at least one decoded content image.
    /// Pixel correctness is verified separately by offscreen readback tests.
    pub content_image_frames: u64,
    /// Non-synthetic native input events received (not a claim of manual QA).
    pub native_input_events: u64,
    /// Device/surface initializations, including resume.
    pub device_initializations: u64,
    /// Surface resize/reconfiguration operations.
    pub surface_configurations: u64,
    /// Recoverable surface acquisition failures.
    pub surface_retries: u64,
    /// Device losses followed by successful resource recreation.
    pub device_recoveries: u64,
    /// Actual adapter name selected for the native surface.
    pub adapter_name: String,
    /// Actual backend selected for the native surface.
    pub backend: String,
}

/// Closing or failing never silently destroys unsaved editor state.
pub struct DesktopOutcome<R> {
    /// Controller, unsaved model, focus and retained UI after native exit.
    pub ui: EditorDesktopUi,
    /// CPU scene state after native resources were released.
    pub renderer: R,
    /// Document session, including any pending or unreconciled I/O.
    pub document: Option<DesktopDocumentSession>,
    /// Measured native run counters.
    pub report: DesktopReport,
    /// Fatal failure, if any; normal close and a reached frame bound are success.
    pub error: Option<DesktopError>,
}

/// Run on the platform's main thread. On every exit path, return ownership of
/// the UI/model and renderer. Interactive idle uses `Wait`, not continuous frames.
pub fn run_desktop<R: DesktopRenderer>(
    config: DesktopConfig,
    ui: EditorDesktopUi,
    renderer: R,
) -> DesktopOutcome<R> {
    run_desktop_with_document(config, ui, renderer, None)
}

/// Run the native editor with an optional production document session.
pub fn run_desktop_with_document<R: DesktopRenderer>(
    config: DesktopConfig,
    ui: EditorDesktopUi,
    mut renderer: R,
    document: Option<DesktopDocumentSession>,
) -> DesktopOutcome<R> {
    let valid = !config.title.is_empty()
        && config.title.len() <= 4096
        && config.logical_size.iter().all(|v| *v > 0 && *v <= 32_768)
        && config.max_primitives > 0
        && config.frame_limit != Some(0)
        && config.time_limit != Some(Duration::ZERO)
        && config.max_surface_retries > 0;
    if !valid {
        return DesktopOutcome {
            ui,
            renderer,
            document,
            report: DesktopReport::default(),
            error: Some(DesktopError::InvalidConfig),
        };
    }
    if let Err(error) = prepare_initial_document(&ui, &mut renderer) {
        return DesktopOutcome {
            ui,
            renderer,
            document,
            report: DesktopReport::default(),
            error: Some(error),
        };
    }
    let event_loop = match EventLoop::<DesktopWake>::with_user_event().build() {
        Ok(event_loop) => event_loop,
        Err(error) => {
            return DesktopOutcome {
                ui,
                renderer,
                document,
                report: DesktopReport::default(),
                error: Some(DesktopError::Window(error.to_string())),
            }
        }
    };
    let proxy = event_loop.create_proxy();
    let mut host = DesktopHost {
        config,
        ui,
        renderer,
        document,
        proxy,
        native: None,
        window: None,
        report: DesktopReport::default(),
        error: None,
        cursor: None,
        drag_start: None,
        modifiers: ModifiersState::empty(),
        ime_enabled: false,
        ime_allowed: false,
        composing: false,
        ime_rect: None,
        gpu_generation: 0,
        occluded: false,
        redraw: true,
        pending: DesktopPending::default(),
        retry_at: None,
        consecutive_retries: 0,
        started: Instant::now(),
    };
    if let Err(error) = event_loop.run_app(&mut host) {
        host.error
            .get_or_insert_with(|| DesktopError::Window(error.to_string()));
    }
    if let Err(error) =
        cancel_native_interactions(&mut host.ui, &mut host.renderer, &mut host.drag_start)
    {
        host.error.get_or_insert(error);
    }
    host.renderer.suspend();
    host.native = None;
    DesktopOutcome {
        ui: host.ui,
        renderer: host.renderer,
        document: host.document,
        report: host.report,
        error: host.error,
    }
}

fn prepare_initial_document<R: DesktopRenderer>(
    ui: &EditorDesktopUi,
    renderer: &mut R,
) -> Result<(), DesktopError> {
    if let Err(error) = renderer.prepare_document(ui.controller()) {
        renderer.discard_document();
        return Err(error);
    }
    renderer.commit_document();
    Ok(())
}

#[derive(Debug)]
enum DesktopWake {
    GpuFailure { generation: u64, message: String },
    DeviceLost { generation: u64, message: String },
    Document,
}

struct Native {
    generation: u64,
    // Drop the surface before the instance and native window.
    surface: wgpu::Surface<'static>,
    instance: wgpu::Instance,
    gpu: DesktopGpu,
    config: wgpu::SurfaceConfiguration,
    presenter: GpuPresenter,
    images: crate::DesktopContentImages,
    scaled_draw: DrawList,
    prepared: bool,
    scale: f32,
}

struct DesktopHost<R> {
    config: DesktopConfig,
    ui: EditorDesktopUi,
    renderer: R,
    document: Option<DesktopDocumentSession>,
    proxy: EventLoopProxy<DesktopWake>,
    native: Option<Native>,
    window: Option<Arc<Window>>,
    report: DesktopReport,
    error: Option<DesktopError>,
    cursor: Option<[f32; 2]>,
    drag_start: Option<([f32; 2], SelectionMode)>,
    modifiers: ModifiersState,
    ime_enabled: bool,
    ime_allowed: bool,
    composing: bool,
    ime_rect: Option<Rect>,
    gpu_generation: u64,
    occluded: bool,
    redraw: bool,
    pending: DesktopPending,
    retry_at: Option<Instant>,
    consecutive_retries: u32,
    started: Instant,
}

impl<R: DesktopRenderer> DesktopHost<R> {
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: DesktopError) {
        self.error.get_or_insert(error);
        event_loop.exit();
    }

    fn initialize(&mut self, event_loop: &ActiveEventLoop) -> Result<(), DesktopError> {
        let window = if let Some(window) = &self.window {
            Arc::clone(window)
        } else {
            let attributes = Window::default_attributes()
                .with_title(&self.config.title)
                .with_inner_size(LogicalSize::new(
                    self.config.logical_size[0],
                    self.config.logical_size[1],
                ));
            let window = Arc::new(
                event_loop
                    .create_window(attributes)
                    .map_err(|e| DesktopError::Window(e.to_string()))?,
            );
            window.set_ime_allowed(false);
            self.window = Some(Arc::clone(&window));
            window
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(Arc::clone(&window))
            .map_err(|e| DesktopError::Gpu(e.to_string()))?;
        let adapter = futures_lite::future::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            },
        ))
        .map_err(|e| DesktopError::Gpu(e.to_string()))?;
        let (device, queue) =
            futures_lite::future::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("Loom Studio native device"),
                ..Default::default()
            }))
            .map_err(|e| DesktopError::Gpu(e.to_string()))?;
        self.gpu_generation = self
            .gpu_generation
            .checked_add(1)
            .ok_or_else(|| DesktopError::Gpu("device generation exhausted".into()))?;
        let generation = self.gpu_generation;
        let lost_proxy = self.proxy.clone();
        device.set_device_lost_callback(move |reason, message| {
            if reason != wgpu::DeviceLostReason::Destroyed {
                let message: String = message.chars().take(8192).collect();
                let _ = lost_proxy.send_event(DesktopWake::DeviceLost {
                    generation,
                    message: format!("device lost ({reason:?}): {message}"),
                });
            }
        });
        let error_proxy = self.proxy.clone();
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            let message: String = error.to_string().chars().take(8192).collect();
            let _ = error_proxy.send_event(DesktopWake::GpuFailure {
                generation,
                message,
            });
        }));
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| {
                DesktopError::Gpu("surface has no supported default configuration".into())
            })?;
        let formats = surface.get_capabilities(&adapter).formats;
        if let Some(format) = formats.iter().copied().find(wgpu::TextureFormat::is_srgb) {
            config.format = format;
        }
        surface.configure(&device, &config);
        let presenter =
            GpuPresenter::new(&device, &queue, config.format, self.config.max_primitives)
                .map_err(|e| DesktopError::Gpu(e.to_string()))?;
        let info = adapter.get_info();
        self.report.adapter_name = info.name;
        self.report.backend = format!("{:?}", info.backend);
        let gpu = DesktopGpu {
            adapter,
            device,
            queue,
            format: config.format,
        };
        self.renderer.initialize(&gpu)?;
        let images =
            crate::DesktopContentImages::new(&gpu).map_err(|e| DesktopError::Gpu(e.to_string()))?;
        let scale = checked_scale(window.scale_factor())?;
        self.ui
            .resize(size.width as f32 / scale, size.height as f32 / scale)
            .map_err(|e| DesktopError::Ui(e.to_string()))?;
        self.native = Some(Native {
            generation,
            surface,
            instance,
            gpu,
            config,
            presenter,
            images,
            scaled_draw: DrawList::new(),
            prepared: false,
            scale,
        });
        self.report.device_initializations += 1;
        self.report.surface_configurations += 1;
        self.redraw = true;
        Ok(())
    }

    fn resize(&mut self) -> Result<(), DesktopError> {
        let (Some(window), Some(native)) = (&self.window, &mut self.native) else {
            return Ok(());
        };
        let size = window.inner_size();
        let scale = checked_scale(window.scale_factor())?;
        native.scale = scale;
        native.prepared = false;
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        if size.width > native.gpu.device.limits().max_texture_dimension_2d
            || size.height > native.gpu.device.limits().max_texture_dimension_2d
        {
            return Err(DesktopError::Gpu(
                "native surface exceeds device texture limits".into(),
            ));
        }
        native.config.width = size.width;
        native.config.height = size.height;
        native.surface.configure(&native.gpu.device, &native.config);
        self.report.surface_configurations += 1;
        self.ui
            .resize(size.width as f32 / scale, size.height as f32 / scale)
            .map_err(|e| DesktopError::Ui(e.to_string()))?;
        self.cursor = None;
        self.redraw = true;
        self.occluded = false;
        Ok(())
    }

    fn retry_surface(&mut self, event_loop: &ActiveEventLoop) {
        self.consecutive_retries += 1;
        self.report.surface_retries += 1;
        if self.consecutive_retries > self.config.max_surface_retries {
            self.fail(
                event_loop,
                DesktopError::Gpu("surface acquisition recovery budget exceeded".into()),
            );
        } else {
            self.retry_at = Some(Instant::now() + Duration::from_millis(16));
            self.redraw = true;
        }
    }

    fn draw(&mut self, event_loop: &ActiveEventLoop) -> Result<(), DesktopError> {
        let (Some(window), Some(native)) = (&self.window, &mut self.native) else {
            return Ok(());
        };
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 || self.occluded {
            return Ok(());
        }
        if self
            .ui
            .refresh()
            .map_err(|e| DesktopError::Ui(e.to_string()))?
        {
            native.prepared = false;
        }
        if !native.prepared {
            native.scaled_draw = scale_draw_list(self.ui.draw_list(), native.scale)?;
            native
                .presenter
                .prepare(
                    &native.gpu.device,
                    &native.gpu.queue,
                    &native.scaled_draw,
                    native.config.width,
                    native.config.height,
                )
                .map_err(|e| DesktopError::Gpu(e.to_string()))?;
            #[cfg(not(feature = "runtime-preview"))]
            native
                .images
                .prepare(
                    self.ui.content_images(),
                    native.scale,
                    [native.config.width, native.config.height],
                )
                .map_err(|e| DesktopError::Gpu(e.to_string()))?;
            #[cfg(feature = "runtime-preview")]
            native
                .images
                .prepare_with_preview(
                    self.ui.content_images(),
                    self.ui.frame_graph_image(),
                    native.scale,
                    [native.config.width, native.config.height],
                )
                .map_err(|e| DesktopError::Gpu(e.to_string()))?;
            native.prepared = true;
        }
        let (texture, suboptimal) = match native.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => (texture, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => (texture, true),
            wgpu::CurrentSurfaceTexture::Timeout => {
                self.retry_surface(event_loop);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                self.occluded = true;
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                native.surface.configure(&native.gpu.device, &native.config);
                self.report.surface_configurations += 1;
                self.retry_surface(event_loop);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                native.surface = native
                    .instance
                    .create_surface(Arc::clone(window))
                    .map_err(|e| DesktopError::Gpu(e.to_string()))?;
                native.surface.configure(&native.gpu.device, &native.config);
                self.report.surface_configurations += 1;
                self.retry_surface(event_loop);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(DesktopError::Gpu(
                    "surface acquisition validation failed".into(),
                ))
            }
        };
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            native
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Loom Studio native UI and scene"),
                });
        native
            .presenter
            .encode(&mut encoder, &view, Some(wgpu::Color::BLACK))
            .map_err(|e| DesktopError::Gpu(e.to_string()))?;
        if let Some(bounds) = viewport_bounds(
            self.ui.viewport_rect(),
            native.scale,
            [native.config.width, native.config.height],
        ) {
            self.renderer
                .encode(&native.gpu, &mut self.ui, &mut encoder, &view, bounds)?;
        }
        native
            .images
            .encode(&mut encoder, &texture.texture)
            .map_err(|e| DesktopError::Gpu(e.to_string()))?;
        native.gpu.queue.submit([encoder.finish()]);
        window.pre_present_notify();
        native.gpu.queue.present(texture);
        self.report.presented_frames += 1;
        if native.images.usage().0 > 0 {
            self.report.content_image_frames += 1;
        }
        self.redraw = false;
        self.consecutive_retries = 0;
        self.retry_at = None;
        if suboptimal {
            native.surface.configure(&native.gpu.device, &native.config);
            self.report.surface_configurations += 1;
        }
        if self
            .config
            .frame_limit
            .is_some_and(|limit| self.report.presented_frames >= limit)
        {
            event_loop.exit();
        } else if self.config.frame_limit.is_some() {
            self.redraw = true;
        }
        Ok(())
    }

    fn input_result(&mut self, result: Result<bool, impl fmt::Display>) {
        match result {
            Ok(changed) => self.redraw |= changed,
            Err(error) => {
                self.ui.show_error(error.to_string());
                self.redraw = true;
            }
        }
        if self.redraw
            && let Some(native) = &mut self.native
        {
            native.prepared = false;
        }
    }

    fn poll_document(&mut self) -> Result<DesktopWork, DesktopError> {
        let Some(document) = self.document.as_mut() else {
            return Ok(DesktopWork::default());
        };
        let waker = task::Waker::from(Arc::new(DocumentWake {
            proxy: self.proxy.clone(),
        }));
        let mut context = task::Context::from_waker(&waker);
        match document.poll(&mut context, &mut self.ui, &mut self.renderer)? {
            task::Poll::Ready(work) => Ok(work),
            task::Poll::Pending => Ok(DesktopWork {
                pending: true,
                ..DesktopWork::default()
            }),
        }
    }

    fn document_command(&mut self, save: bool) {
        if let Err(error) = ensure_file_command_idle(self.renderer.gesture_active()) {
            self.input_result(Err(error));
            return;
        }
        if self
            .document
            .as_ref()
            .is_some_and(|d| d.is_busy() || d.reconciliation_required())
        {
            self.input_result(Err(DesktopUiError::Panel(
                "Resolve scene I/O before panel file commands".into(),
            )));
            return;
        }
        match self.ui.panel_document_command(save) {
            Ok(true) => {
                self.pending.set(DesktopPendingSource::Ui, true);
                return;
            }
            Err(error) => {
                self.input_result(Err(error));
                return;
            }
            Ok(false) => {}
        }
        let document = match require_scene_document(self.document.as_mut()) {
            Ok(document) => document,
            Err(error) => {
                self.input_result(Err(error));
                return;
            }
        };
        let result = if save {
            document.save(&mut self.ui)
        } else {
            document.load(&self.ui)
        };
        let (result, pending) = document_request_result(result);
        self.input_result(result);
        self.pending.set(DesktopPendingSource::Document, pending);
    }

    fn sync_text_input(&mut self) {
        let Some(window) = &self.window else {
            return;
        };
        let size = window.inner_size();
        let rect = (self.native.is_some() && size.width > 0 && size.height > 0 && !self.occluded)
            .then(|| self.ui.text_input_rect())
            .flatten();
        let allowed = rect.is_some();
        if allowed != self.ime_allowed {
            window.set_ime_allowed(allowed);
            self.ime_allowed = allowed;
            if !allowed {
                self.ime_enabled = false;
                self.composing = false;
            }
        }
        if self.ime_enabled
            && (rect != self.ime_rect)
            && let Some(rect) = rect
        {
            window.set_ime_cursor_area(
                LogicalPosition::new(f64::from(rect.left()), f64::from(rect.top())),
                LogicalSize::new(f64::from(rect.size.width), f64::from(rect.size.height)),
            );
        }
        self.ime_rect = if self.ime_enabled { rect } else { None };
    }
}

fn ensure_file_command_idle(renderer_gesture_active: bool) -> Result<(), DesktopUiError> {
    if renderer_gesture_active {
        return Err(DesktopUiError::Panel(
            "Commit or cancel the active editor interaction before file commands".into(),
        ));
    }
    Ok(())
}

fn require_scene_document(
    document: Option<&mut DesktopDocumentSession>,
) -> Result<&mut DesktopDocumentSession, DesktopUiError> {
    document.ok_or_else(|| DesktopUiError::Panel("No scene document storage is configured".into()))
}

fn document_request_result<E>(result: Result<impl Sized, E>) -> (Result<bool, E>, bool) {
    match result {
        Ok(_) => (Ok(true), true),
        Err(error) => (Err(error), false),
    }
}

fn cancel_native_interactions<R: DesktopRenderer>(
    ui: &mut EditorDesktopUi,
    renderer: &mut R,
    drag_start: &mut Option<([f32; 2], SelectionMode)>,
) -> Result<bool, DesktopError> {
    let panel = ui
        .cancel_panel_pointer()
        .map_err(|error| DesktopError::Ui(error.to_string()));
    let viewport = renderer.gesture_cancel(ui);
    *drag_start = None;
    match (panel, viewport) {
        (Ok(panel), Ok(viewport)) => Ok(panel || viewport),
        (Err(error), _) | (_, Err(error)) => Err(error),
    }
}

struct DocumentWake {
    proxy: EventLoopProxy<DesktopWake>,
}

impl Wake for DocumentWake {
    fn wake(self: Arc<Self>) {
        let _ = self.proxy.send_event(DesktopWake::Document);
    }
}

impl<R: DesktopRenderer> ApplicationHandler<DesktopWake> for DesktopHost<R> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.native.is_none()
            && let Err(error) = self.initialize(event_loop)
        {
            self.fail(event_loop, error);
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        let result =
            cancel_native_interactions(&mut self.ui, &mut self.renderer, &mut self.drag_start);
        self.input_result(result);
        self.renderer.suspend();
        self.native = None;
        self.cursor = None;
        self.pending.set(DesktopPendingSource::Renderer, false);
        self.sync_text_input();
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: DesktopWake) {
        if matches!(event, DesktopWake::Document) {
            match self.poll_document() {
                Ok(work) => {
                    self.redraw |= work.redraw;
                    self.pending
                        .set(DesktopPendingSource::Document, work.pending);
                }
                Err(error) => {
                    self.pending.set(DesktopPendingSource::Document, false);
                    self.ui.show_error(error.to_string());
                    self.redraw = true;
                }
            }
            return;
        }
        let generation = match &event {
            DesktopWake::GpuFailure { generation, .. }
            | DesktopWake::DeviceLost { generation, .. } => *generation,
            DesktopWake::Document => unreachable!(),
        };
        // Callbacks from a suspended/replaced device cannot tear down its successor.
        if self
            .native
            .as_ref()
            .is_none_or(|native| native.generation != generation)
        {
            return;
        }
        match event {
            DesktopWake::GpuFailure { message, .. } => {
                self.fail(event_loop, DesktopError::Gpu(message));
            }
            DesktopWake::DeviceLost { message, .. } => {
                if self.report.device_recoveries >= u64::from(self.config.max_device_recoveries) {
                    self.fail(event_loop, DesktopError::Gpu(message));
                    return;
                }
                let result = cancel_native_interactions(
                    &mut self.ui,
                    &mut self.renderer,
                    &mut self.drag_start,
                );
                self.input_result(result);
                self.renderer.suspend();
                self.native = None;
                self.pending.set(DesktopPendingSource::Renderer, false);
                self.retry_at = None;
                match self.initialize(event_loop) {
                    Ok(()) => {
                        self.report.device_recoveries += 1;
                    }
                    Err(error) => self.fail(event_loop, error),
                }
            }
            DesktopWake::Document => unreachable!(),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().is_none_or(|window| window.id() != id) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => {
                let blocked = close_blocked(
                    self.renderer.gesture_active(),
                    self.ui.has_uncommitted_edit(),
                    self.ui.controller().is_dirty(),
                    self.document
                        .as_ref()
                        .is_some_and(DesktopDocumentSession::is_busy),
                    self.document
                        .as_ref()
                        .is_some_and(DesktopDocumentSession::reconciliation_required),
                );
                if blocked {
                    self.ui.show_error(DesktopError::UnsavedChanges.to_string());
                    self.redraw = true;
                } else {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                let result = cancel_native_interactions(
                    &mut self.ui,
                    &mut self.renderer,
                    &mut self.drag_start,
                );
                self.input_result(result);
                if let Err(error) = self.resize() {
                    self.fail(event_loop, error);
                }
            }
            WindowEvent::Occluded(value) => {
                self.occluded = value;
                self.redraw |= !value;
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.draw(event_loop) {
                    self.fail(event_loop, error);
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::CursorMoved { position, .. } => {
                self.report.native_input_events += 1;
                if let Some(native) = &self.native {
                    self.cursor = Some([
                        position.x as f32 / native.scale,
                        position.y as f32 / native.scale,
                    ]);
                }
                if self.ui.panel_pointer_active()
                    && let Some([x, y]) = self.cursor
                {
                    let result = self.ui.panel_pointer_move(x, y);
                    self.input_result(result);
                    return;
                }
                if self.renderer.gesture_active()
                    && let Some([x, y]) = self.cursor
                {
                    let rect = self.ui.viewport_rect();
                    let result = self
                        .renderer
                        .gesture_move(&mut self.ui, [x - rect.left(), y - rect.top()]);
                    self.input_result(result);
                }
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::Focused(false) => {
                self.modifiers = ModifiersState::empty();
                self.cursor = None;
                self.composing = false;
                let result = cancel_native_interactions(
                    &mut self.ui,
                    &mut self.renderer,
                    &mut self.drag_start,
                );
                self.input_result(result);
                let result = self.ui.key(DesktopKey::Escape);
                self.input_result(result);
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Middle,
                ..
            } => {
                self.report.native_input_events += 1;
                if state == ElementState::Released && self.ui.panel_pan_active() {
                    let result = self.ui.panel_pointer_release(self.cursor);
                    self.input_result(result);
                } else if state == ElementState::Pressed
                    && !self.ui.panel_pointer_active()
                    && !self.renderer.gesture_active()
                    && self.drag_start.is_none()
                    && let Some([x, y]) = self.cursor
                {
                    if self
                        .document
                        .as_ref()
                        .is_some_and(DesktopDocumentSession::is_busy)
                    {
                        self.ui
                            .show_error("Document I/O must finish before panel input");
                        self.redraw = true;
                        return;
                    }
                    let result = self.ui.panel_pan_press(x, y);
                    self.input_result(result);
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.report.native_input_events += 1;
                if state == ElementState::Released
                    && self.ui.panel_pointer_active()
                    && !self.ui.panel_pan_active()
                {
                    let result = self.ui.panel_pointer_release(self.cursor);
                    self.input_result(result);
                    return;
                }
                if state == ElementState::Released && self.renderer.gesture_active() {
                    // Use the last known pointer even after CursorLeft; focus loss
                    // has a separate cancellation path.
                    let result = self.renderer.gesture_release(&mut self.ui);
                    self.drag_start = None;
                    self.input_result(result);
                    return;
                }
                if let Some([x, y]) = self.cursor {
                    let mode = if self.modifiers.control_key() || self.modifiers.super_key() {
                        SelectionMode::Toggle
                    } else if self.modifiers.shift_key() {
                        SelectionMode::Add
                    } else {
                        SelectionMode::Replace
                    };
                    let rect = self.ui.viewport_rect();
                    match state {
                        ElementState::Pressed if contains(rect, x, y) => {
                            if self
                                .document
                                .as_ref()
                                .is_some_and(DesktopDocumentSession::is_busy)
                            {
                                self.ui.show_error(
                                    "document I/O must finish before starting a gesture",
                                );
                                self.redraw = true;
                                return;
                            }
                            if self.ui.has_uncommitted_edit() {
                                self.ui
                                    .show_error("commit or cancel the Inspector draft first");
                                self.redraw = true;
                                return;
                            }
                            let result = self
                                .renderer
                                .gesture_press(&mut self.ui, [x - rect.left(), y - rect.top()]);
                            let captured = matches!(result, Ok(true));
                            let rejected = result.is_err();
                            self.input_result(result);
                            if captured || rejected {
                                return;
                            }
                            self.drag_start = Some(([x, y], mode));
                        }
                        ElementState::Pressed => {
                            if self
                                .document
                                .as_ref()
                                .is_some_and(DesktopDocumentSession::is_busy)
                            {
                                self.ui
                                    .show_error("Document I/O must finish before panel input");
                                self.redraw = true;
                                return;
                            }
                            let result = self.ui.panel_pointer_press(x, y, mode);
                            let captured = matches!(result, Ok(true));
                            let rejected = result.is_err();
                            self.input_result(result);
                            if captured || rejected {
                                return;
                            }
                            let result = self.ui.click(x, y, mode);
                            self.input_result(result);
                        }
                        ElementState::Released => {
                            let Some(([start_x, start_y], drag_mode)) = self.drag_start.take()
                            else {
                                return;
                            };
                            let start = [start_x - rect.location.x, start_y - rect.location.y];
                            let end = [x - rect.location.x, y - rect.location.y];
                            let moved =
                                (end[0] - start[0]).abs() > 2.0 || (end[1] - start[1]).abs() > 2.0;
                            let result = if let Some(native) = &self.native {
                                if moved {
                                    self.renderer.box_select(
                                        &native.gpu,
                                        &mut self.ui,
                                        start,
                                        end,
                                        drag_mode,
                                    )
                                } else {
                                    self.renderer
                                        .pointer(&native.gpu, &mut self.ui, end, drag_mode)
                                }
                            } else {
                                Ok(false)
                            };
                            self.pending.set(DesktopPendingSource::Renderer, true);
                            self.input_result(result);
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.report.native_input_events += 1;
                let rows = match delta {
                    MouseScrollDelta::LineDelta(_, y) => -y * 3.0,
                    MouseScrollDelta::PixelDelta(position) => -position.y as f32 / 20.0,
                };
                if rows.is_finite()
                    && let Some([x, y]) = self.cursor
                {
                    let result = self
                        .ui
                        .scroll_at(x, y, rows.round().clamp(-512.0, 512.0) as i32);
                    self.input_result(result);
                }
            }
            WindowEvent::KeyboardInput {
                event,
                is_synthetic: false,
                ..
            } if event.state == ElementState::Pressed => {
                self.report.native_input_events += 1;
                // The platform IME owns editing/navigation keys while composing.
                if self.composing {
                    return;
                }
                if self.renderer.gesture_active()
                    && event.logical_key == Key::Named(NamedKey::Escape)
                {
                    let result = self.renderer.gesture_cancel(&mut self.ui);
                    self.drag_start = None;
                    self.input_result(result);
                    return;
                }
                let primary = if cfg!(target_os = "macos") {
                    self.modifiers.super_key()
                } else {
                    self.modifiers.control_key() && !self.modifiers.alt_key()
                };
                if primary
                    && let Key::Character(character) = &event.logical_key
                    && character.eq_ignore_ascii_case("s")
                {
                    self.document_command(true);
                    return;
                }
                if primary
                    && let Key::Character(character) = &event.logical_key
                    && character.eq_ignore_ascii_case("o")
                {
                    self.document_command(false);
                    return;
                }
                if !primary
                    && !self.ui.has_uncommitted_edit()
                    && let Key::Character(character) = &event.logical_key
                {
                    let mode = match character.to_ascii_lowercase().as_str() {
                        "w" => Some(prism_editor_viewport::GizmoMode::Translate),
                        "e" => Some(prism_editor_viewport::GizmoMode::Rotate),
                        "r" => Some(prism_editor_viewport::GizmoMode::Scale),
                        _ => None,
                    };
                    if let Some(mode) = mode {
                        let result = self.renderer.transform_tool(mode);
                        self.input_result(result);
                        return;
                    }
                }
                let key = match &event.logical_key {
                    Key::Named(NamedKey::Enter) => Some(DesktopKey::Enter),
                    Key::Named(NamedKey::Escape) => Some(DesktopKey::Escape),
                    Key::Named(NamedKey::Backspace) => Some(DesktopKey::Backspace),
                    Key::Named(NamedKey::Tab) => Some(DesktopKey::Tab),
                    Key::Character(c) if primary && c.eq_ignore_ascii_case("z") => {
                        Some(if self.modifiers.shift_key() {
                            DesktopKey::Redo
                        } else {
                            DesktopKey::Undo
                        })
                    }
                    Key::Character(c) if primary && c.eq_ignore_ascii_case("y") => {
                        Some(DesktopKey::Redo)
                    }
                    _ => None,
                };
                if let Some(key) = key {
                    let result = self.ui.key(key);
                    self.input_result(result);
                } else if accepts_key_text(primary, self.modifiers, self.composing)
                    && let Some(text) = event.text
                {
                    let result = self.ui.text(&text);
                    self.input_result(result);
                }
            }
            WindowEvent::Ime(event) => {
                self.report.native_input_events += 1;
                match event {
                    Ime::Enabled => {
                        self.ime_enabled = true;
                        self.ime_rect = None;
                    }
                    Ime::Disabled => {
                        self.ime_enabled = false;
                        self.composing = false;
                        let result = self.ui.preedit("");
                        self.input_result(result);
                    }
                    Ime::Preedit(text, _) => {
                        self.composing = !text.is_empty();
                        let result = self.ui.preedit(&text);
                        self.input_result(result);
                    }
                    Ime::Commit(text) => {
                        self.composing = false;
                        let result = self.ui.text(&text);
                        self.input_result(result);
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self
            .config
            .time_limit
            .is_some_and(|limit| self.started.elapsed() >= limit)
        {
            self.fail(event_loop, DesktopError::TimeLimit);
            return;
        }
        if let Some(native) = &self.native {
            if let Err(error) = native.gpu.device.poll(wgpu::PollType::Poll) {
                self.fail(event_loop, DesktopError::Gpu(error.to_string()));
                return;
            }
            match self.renderer.poll(&native.gpu, &mut self.ui) {
                Ok(work) => {
                    self.redraw |= work.redraw;
                    self.pending
                        .set(DesktopPendingSource::Renderer, work.pending);
                }
                Err(error) => {
                    self.fail(event_loop, error);
                    return;
                }
            }
        } else {
            self.pending.set(DesktopPendingSource::Renderer, false);
        }
        match self.poll_document() {
            Ok(work) => {
                self.redraw |= work.redraw;
                self.pending
                    .set(DesktopPendingSource::Document, work.pending);
            }
            Err(error) => {
                self.pending.set(DesktopPendingSource::Document, false);
                self.ui.show_error(error.to_string());
                self.redraw = true;
            }
        }
        if let Err(error) = self.ui.poll_native_panels(Instant::now()) {
            self.ui.show_error(error.to_string());
            self.redraw = true;
        }
        match self.ui.refresh() {
            Ok(changed) => {
                self.redraw |= changed;
                if changed && let Some(native) = &mut self.native {
                    native.prepared = false;
                }
            }
            Err(error) => {
                self.fail(event_loop, DesktopError::Ui(error.to_string()));
                return;
            }
        }
        self.pending.set(
            DesktopPendingSource::Ui,
            self.ui.content_thumbnails_pending()
                || self.ui.content_git_pending()
                || self.ui.content_collections_pending(),
        );
        self.sync_text_input();
        let now = Instant::now();
        if self.redraw
            && !self.occluded
            && self.native.is_some()
            && self.retry_at.is_none_or(|retry| now >= retry)
            && let Some(window) = &self.window
        {
            let size = window.inner_size();
            if size.width > 0 && size.height > 0 {
                window.request_redraw();
            }
        }
        let poll_at = self.pending.any().then(|| now + Duration::from_millis(4));
        let deadline = self
            .config
            .time_limit
            .and_then(|duration| self.started.checked_add(duration));
        let wake = [
            poll_at,
            self.retry_at,
            deadline,
            self.ui
                .next_native_panel_wake()
                .map(|wake| wake.max(now + Duration::from_millis(1))),
        ]
        .into_iter()
        .flatten()
        .filter(|time| *time > now)
        .min();
        event_loop.set_control_flow(wake.map_or(ControlFlow::Wait, ControlFlow::WaitUntil));
    }
}

fn accepts_key_text(primary: bool, modifiers: ModifiersState, composing: bool) -> bool {
    // AltGr layouts can report Ctrl+Alt for ordinary printable text.
    !primary && (!modifiers.control_key() || modifiers.alt_key()) && !composing
}

const fn close_blocked(
    renderer_gesture_active: bool,
    ui_uncommitted: bool,
    model_dirty: bool,
    document_busy: bool,
    reconciliation_required: bool,
) -> bool {
    renderer_gesture_active
        || ui_uncommitted
        || model_dirty
        || document_busy
        || reconciliation_required
}

fn checked_scale(scale: f64) -> Result<f32, DesktopError> {
    if !scale.is_finite() || !(0.125..=16.0).contains(&scale) {
        return Err(DesktopError::InvalidConfig);
    }
    Ok(scale as f32)
}

fn contains(rect: Rect, x: f32, y: f32) -> bool {
    x.is_finite()
        && y.is_finite()
        && x >= rect.left()
        && x < rect.right()
        && y >= rect.top()
        && y < rect.bottom()
}

fn viewport_bounds(rect: Rect, scale: f32, extent: [u32; 2]) -> Option<DesktopViewportBounds> {
    if checked_scale(f64::from(scale)).is_err()
        || ![rect.left(), rect.top(), rect.right(), rect.bottom()]
            .iter()
            .all(|v| v.is_finite())
        || rect.size.width <= 0.0
        || rect.size.height <= 0.0
    {
        return None;
    }
    let x = (rect.left() * scale).ceil().clamp(0.0, extent[0] as f32);
    let y = (rect.top() * scale).ceil().clamp(0.0, extent[1] as f32);
    let right = (rect.right() * scale).floor().clamp(x, extent[0] as f32);
    let bottom = (rect.bottom() * scale).floor().clamp(y, extent[1] as f32);
    if ![x, y, right, bottom].iter().all(|v| v.is_finite()) || right <= x || bottom <= y {
        return None;
    }
    Some(DesktopViewportBounds {
        logical: rect,
        physical: Rect::new(Point::new(x, y), Size::new(right - x, bottom - y)),
        surface_extent: extent,
    })
}

fn scale_draw_list(source: &DrawList, scale: f32) -> Result<DrawList, DesktopError> {
    checked_scale(f64::from(scale))?;
    let rect = |r: Rect| {
        Rect::new(
            Point::new(r.location.x * scale, r.location.y * scale),
            Size::new(r.size.width * scale, r.size.height * scale),
        )
    };
    let mut scaled = DrawList::new();
    scaled.set_text_pool(source.text_pool().to_owned());
    for command in source.commands() {
        scaled.push(match *command {
            DrawCommand::Rect(mut c) => {
                c.rect = rect(c.rect);
                c.radius *= scale;
                c.border_width *= scale;
                DrawCommand::Rect(c)
            }
            DrawCommand::Shadow(mut c) => {
                c.rect = rect(c.rect);
                c.radius *= scale;
                c.blur *= scale;
                c.offset.x *= scale;
                c.offset.y *= scale;
                DrawCommand::Shadow(c)
            }
            DrawCommand::Glyph(mut c) => {
                c.rect = rect(c.rect);
                c.size *= scale;
                DrawCommand::Glyph(c)
            }
            DrawCommand::PushLayer(mut c) => {
                c.bounds = rect(c.bounds);
                DrawCommand::PushLayer(c)
            }
            DrawCommand::PopLayer => DrawCommand::PopLayer,
        });
    }
    Ok(scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_render_backend::{GlyphCmd, LayerCmd, RectCmd};

    #[derive(Default)]
    struct StartupRenderer {
        fail_prepare: bool,
        candidate: bool,
        prepare_count: u32,
        commit_count: u32,
        discard_count: u32,
        initialize_count: u32,
        cancel_count: u32,
        cancel_changed: bool,
        fail_cancel: bool,
    }

    impl DesktopRenderer for StartupRenderer {
        fn gesture_cancel(&mut self, _ui: &mut EditorDesktopUi) -> Result<bool, DesktopError> {
            self.cancel_count += 1;
            if self.fail_cancel {
                return Err(DesktopError::Renderer("gesture cancel rejected".into()));
            }
            Ok(self.cancel_changed)
        }

        fn initialize(&mut self, _gpu: &DesktopGpu) -> Result<(), DesktopError> {
            self.initialize_count += 1;
            Ok(())
        }

        fn encode(
            &mut self,
            _gpu: &DesktopGpu,
            _ui: &mut EditorDesktopUi,
            _encoder: &mut wgpu::CommandEncoder,
            _target: &wgpu::TextureView,
            _bounds: DesktopViewportBounds,
        ) -> Result<(), DesktopError> {
            Ok(())
        }

        fn pointer(
            &mut self,
            _gpu: &DesktopGpu,
            _ui: &mut EditorDesktopUi,
            _position: [f32; 2],
            _mode: SelectionMode,
        ) -> Result<bool, DesktopError> {
            Ok(false)
        }

        fn poll(
            &mut self,
            _gpu: &DesktopGpu,
            _ui: &mut EditorDesktopUi,
        ) -> Result<DesktopWork, DesktopError> {
            Ok(DesktopWork::default())
        }

        fn prepare_document(&mut self, _controller: &EditorController) -> Result<(), DesktopError> {
            self.prepare_count += 1;
            self.candidate = true;
            if self.fail_prepare {
                return Err(DesktopError::Renderer("initial scene rejected".into()));
            }
            Ok(())
        }

        fn discard_document(&mut self) {
            self.discard_count += 1;
            self.candidate = false;
        }

        fn commit_document(&mut self) {
            assert!(self.candidate, "commit requires a prepared candidate");
            self.commit_count += 1;
            self.candidate = false;
        }

        fn suspend(&mut self) {}
    }

    #[test]
    fn initial_document_is_committed_before_native_gpu_initialization() {
        let ui = EditorDesktopUi::new(EditorController::new()).expect("empty editor UI");
        let mut renderer = StartupRenderer::default();

        prepare_initial_document(&ui, &mut renderer).expect("prepare initial document");

        assert_eq!(renderer.prepare_count, 1);
        assert_eq!(renderer.commit_count, 1);
        assert_eq!(renderer.discard_count, 0);
        assert_eq!(renderer.initialize_count, 0);
        assert!(!renderer.candidate);
    }

    #[test]
    fn rejected_initial_document_is_discarded_without_publication() {
        let ui = EditorDesktopUi::new(EditorController::new()).expect("empty editor UI");
        let mut renderer = StartupRenderer {
            fail_prepare: true,
            ..StartupRenderer::default()
        };

        assert_eq!(
            prepare_initial_document(&ui, &mut renderer),
            Err(DesktopError::Renderer("initial scene rejected".into()))
        );
        assert_eq!(renderer.prepare_count, 1);
        assert_eq!(renderer.commit_count, 0);
        assert_eq!(renderer.discard_count, 1);
        assert_eq!(renderer.initialize_count, 0);
        assert!(!renderer.candidate);
    }

    #[test]
    fn native_interaction_cancel_calls_renderer_and_clears_box_drag() {
        let mut ui = EditorDesktopUi::new(EditorController::new()).expect("empty editor UI");
        let mut renderer = StartupRenderer {
            cancel_changed: true,
            ..StartupRenderer::default()
        };
        let mut drag_start = Some(([12.0, 24.0], SelectionMode::Add));

        assert_eq!(
            cancel_native_interactions(&mut ui, &mut renderer, &mut drag_start),
            Ok(true)
        );
        assert_eq!(renderer.cancel_count, 1);
        assert_eq!(drag_start, None);
    }

    #[test]
    fn native_interaction_cancel_clears_box_drag_when_renderer_rejects() {
        let mut ui = EditorDesktopUi::new(EditorController::new()).expect("empty editor UI");
        let mut renderer = StartupRenderer {
            fail_cancel: true,
            ..StartupRenderer::default()
        };
        let mut drag_start = Some(([12.0, 24.0], SelectionMode::Replace));

        assert_eq!(
            cancel_native_interactions(&mut ui, &mut renderer, &mut drag_start),
            Err(DesktopError::Renderer("gesture cancel rejected".into()))
        );
        assert_eq!(renderer.cancel_count, 1);
        assert_eq!(drag_start, None);
    }

    #[test]
    fn ordinary_text_is_not_blocked_by_ime_availability_and_altgr_is_printable() {
        assert!(accepts_key_text(false, ModifiersState::empty(), false));
        assert!(!accepts_key_text(false, ModifiersState::empty(), true));
        assert!(!accepts_key_text(true, ModifiersState::SUPER, false));
        assert!(!accepts_key_text(false, ModifiersState::CONTROL, false));
        assert!(accepts_key_text(
            false,
            ModifiersState::CONTROL | ModifiersState::ALT,
            false
        ));
    }

    #[test]
    fn rejected_document_request_does_not_schedule_polling() {
        let (result, pending) = document_request_result(Err::<u64, _>("rejected"));

        assert_eq!(result, Err("rejected"));
        assert!(!pending);
    }

    #[test]
    fn accepted_document_request_schedules_polling() {
        let (result, pending) = document_request_result::<&str>(Ok(7_u64));

        assert_eq!(result, Ok(true));
        assert!(pending);
    }

    #[test]
    fn document_completion_does_not_clear_other_pending_sources() {
        let mut pending = DesktopPending {
            renderer: true,
            document: true,
            ui: true,
        };

        pending.set(DesktopPendingSource::Document, false);

        assert_eq!(
            pending,
            DesktopPending {
                renderer: true,
                document: false,
                ui: true,
            }
        );
        assert!(pending.any());
    }

    #[test]
    fn document_terminal_failure_stops_document_polling() {
        let mut pending = DesktopPending {
            document: true,
            ..DesktopPending::default()
        };

        pending.set(DesktopPendingSource::Document, false);

        assert!(!pending.any());
    }

    #[test]
    fn renderer_suspend_preserves_pending_document_io() {
        let mut pending = DesktopPending {
            renderer: true,
            document: true,
            ..DesktopPending::default()
        };

        pending.set(DesktopPendingSource::Renderer, false);

        assert_eq!(
            pending,
            DesktopPending {
                renderer: false,
                document: true,
                ui: false,
            }
        );
        assert!(pending.any());
    }

    #[test]
    fn dirty_model_blocks_close_without_document_storage() {
        assert!(close_blocked(false, false, true, false, false));
    }

    #[test]
    fn clean_idle_model_can_close_without_document_storage() {
        assert!(!close_blocked(false, false, false, false, false));
    }

    #[test]
    fn active_viewport_gesture_blocks_panel_and_scene_file_commands() {
        assert!(ensure_file_command_idle(true).is_err());
        assert!(ensure_file_command_idle(false).is_ok());
    }

    #[test]
    fn missing_scene_document_storage_is_reported() {
        assert_eq!(
            require_scene_document(None).err(),
            Some(DesktopUiError::Panel(
                "No scene document storage is configured".into()
            ))
        );
    }

    #[test]
    fn dpi_scales_geometry_and_preserves_utf8_text_and_layers() {
        let mut list = DrawList::new();
        let text = list.intern("场景 🎨");
        let bounds = Rect::new(Point::new(3.0, 4.0), Size::new(30.0, 20.0));
        list.push(DrawCommand::PushLayer(LayerCmd {
            bounds,
            opacity: 0.7,
        }));
        list.push(DrawCommand::Rect(RectCmd {
            rect: bounds,
            fill: None,
            radius: 3.0,
            border_width: 1.0,
            border_color: None,
            opacity: 0.5,
        }));
        list.push(DrawCommand::Glyph(GlyphCmd {
            rect: bounds,
            size: 12.0,
            color: prism_ui::style::Color::WHITE,
            opacity: 1.0,
            text,
        }));
        list.push(DrawCommand::PopLayer);
        let scaled = scale_draw_list(&list, 2.0).expect("scale");
        assert_eq!(scaled.text(text), "场景 🎨");
        assert_eq!(list.commands().len(), scaled.commands().len());
        let DrawCommand::Glyph(glyph) = scaled.commands()[2] else {
            panic!("glyph");
        };
        assert_eq!(glyph.size, 24.0);
        assert_eq!(glyph.rect.location, Point::new(6.0, 8.0));
        assert!(scale_draw_list(&list, f32::NAN).is_err());
        assert!(scale_draw_list(&list, 0.0).is_err());
    }

    #[test]
    fn native_viewport_rounding_stays_inside_surface_and_uses_half_open_input() {
        let rect = Rect::new(Point::new(3.2, 5.1), Size::new(60.0, 40.0));
        let bounds = viewport_bounds(rect, 1.5, [80, 50]).expect("clipped bounds");
        assert_eq!(bounds.physical.location, Point::new(5.0, 8.0));
        assert_eq!(bounds.physical.right(), 80.0);
        assert_eq!(bounds.physical.bottom(), 50.0);
        assert!(contains(rect, 3.2, 5.1));
        assert!(!contains(rect, rect.right(), 6.0));
        assert!(!contains(rect, f32::NAN, 6.0));
        assert!(viewport_bounds(
            Rect::new(Point::new(100.0, 100.0), Size::new(20.0, 20.0)),
            2.0,
            [50, 50]
        )
        .is_none());
        assert!(viewport_bounds(
            Rect::new(Point::new(f32::NAN, 0.0), Size::new(20.0, 20.0)),
            1.0,
            [50, 50]
        )
        .is_none());
        assert!(viewport_bounds(rect, f32::INFINITY, [50, 50]).is_none());
        assert!(viewport_bounds(rect, 1.0, [0, 0]).is_none());
    }
}
