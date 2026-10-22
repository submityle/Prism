//! Deterministic editor assembly and engine-neutral host lifecycle.
//!
//! The application layer owns lifecycle and frame-budget policy. The editor
//! kernel and panel/content/viewport/profiler crates remain independently
//! usable, while this crate provides bounded headless execution for CI and tools
//! plus checked window/surface orchestration for host-supplied platform adapters.

#![forbid(unsafe_code)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use core::fmt;

use prism_app::prelude::{App, AppExitRequest, ResMut, Update};
use prism_ecs::resource::Resource;
use prism_editor_content::ContentCatalog;
use prism_editor_core::{DocumentId, DocumentState, Selection, ServiceRegistry};
use prism_editor_dock::DockContributionRegistry;
use prism_editor_panels::{Console, Outliner};
use prism_editor_profiler::{FrameStore, ProfilerLimits};
use prism_editor_reflect::{InspectorProjection, InspectorProjectionNode};
use prism_editor_shell::{CommandIndex, ContextKeySet, ShellContributions};
use prism_editor_ui::{
    command_palette, console_panel, dock_workspace, editor_workspace, inspector_view,
    outliner_panel, shell_chrome, viewport_host, CommandPaletteProps, ConsolePanelProps,
    DockWorkspaceProps, EditorWorkspaceProps, InspectorViewProps, OutlinerViewProps,
    ShellChromeProps, ViewportHostProps,
};
use prism_editor_viewport::ViewportRegistry;
use prism_reflect::{InspectorHints, InspectorKind, InspectorNode};
use prism_ui::Element;

mod backend;
mod controller;
mod controller_validation;
#[cfg(feature = "desktop")]
mod desktop;
#[cfg(feature = "desktop")]
mod desktop_document;
#[cfg(feature = "desktop")]
mod desktop_images;
#[cfg(feature = "desktop")]
mod desktop_scene;
mod desktop_ui;
mod document;
mod ecs_document_bridge;
mod lifecycle;
#[cfg(feature = "desktop")]
mod scene_authoring;
#[cfg(feature = "desktop")]
mod scene_mesh;
#[cfg(feature = "desktop")]
mod scene_project;
#[cfg(feature = "desktop")]
mod scene_session;
mod world_session;

pub use controller::{
    ControllerAuthoringCommand, ControllerDocumentBudget, ControllerError, ControllerFieldEdit,
    ControllerLimits, ControllerObject, ControllerOutcome, EditorController,
    PreparedControllerEdit,
};
pub use controller_validation::ControllerObjectValidator;
#[cfg(feature = "desktop")]
pub use desktop::{
    run_desktop, run_desktop_with_document, DesktopConfig, DesktopError, DesktopGpu,
    DesktopOutcome, DesktopRenderer, DesktopReport, DesktopViewportBounds, DesktopWork,
};
#[cfg(feature = "desktop")]
pub use desktop_document::{DesktopDocumentCancelOutcome, DesktopDocumentSession};
#[cfg(feature = "desktop")]
pub use desktop_images::DesktopContentImages;
#[cfg(feature = "desktop")]
pub use desktop_scene::{
    DesktopScene, DesktopSceneDocument, DesktopSceneObject, DesktopSceneStatus,
    SceneDocumentAdapter,
};
#[cfg(feature = "desktop")]
pub use desktop_scene::{GizmoFrame, SceneTransformAdapter};
pub use desktop_ui::{
    ContentGitPhase, ContentGitStatus, ContentThumbnailSource, DesktopContentImage,
    DesktopFrameGraphPanel, DesktopGraphPanel, DesktopKey, DesktopPanel, DesktopPanelCancelOutcome,
    DesktopPanelUpdate, DesktopPanelView, DesktopPrefabPanel, DesktopProfilerPanel,
    DesktopSequencePanel, DesktopSequenceStorageLimits, DesktopTimelineFrame, DesktopTimelinePanel,
    DesktopUiError, DesktopWorldPanel, DesktopWorldProjectBinding, DesktopWorldProjectError,
    EditorDesktopUi,
};
#[cfg(feature = "runtime-preview")]
pub use desktop_ui::{DesktopFrameGraphImage, DesktopRuntimePreviewSource};
pub use document::{
    DecodedDocument, DocumentCodec, DocumentError, DocumentFileAdapter, DocumentIoEvent,
    DocumentLimits, DocumentLoadReceipt, DocumentSaveReceipt, DocumentSchemaCatalog,
    PreparedDocumentLoad,
};
pub use ecs_document_bridge::{
    EditorEcsDocumentBridge, EditorEcsDocumentError, EditorEcsStructureSync,
};

pub use backend::{
    GpuSurfaceError, GpuSurfaceErrorKind, GpuSurfaceLifecycle, PixelSize, PixelSizeError,
    SurfaceDescriptor, SurfaceId, SurfaceState, WindowBackend, WindowBackendError,
    WindowBackendErrorKind, WindowDescriptor, WindowEvent, WindowEventEnvelope, WindowId,
};
pub use lifecycle::{ApplicationState, ExitReason, ExitStatus, InvalidStateError, ShutdownReport};
#[cfg(feature = "desktop")]
pub use scene_authoring::{
    install_scene_authoring_commands, SCENE_ADD_TRIANGLE_COMMAND, SCENE_DELETE_SELECTION_COMMAND,
};
#[cfg(feature = "desktop")]
pub use scene_mesh::{editor_scene_mesh_catalog, editor_scene_mesh_registry, EditorSceneMesh};
#[cfg(feature = "desktop")]
pub use scene_project::{SceneProject, SceneProjectConfig, SceneProjectError, SceneProjectOpen};
#[cfg(feature = "desktop")]
pub use scene_session::{EditorSceneSession, EditorSceneSyncWork};
pub use world_session::{
    WorldProjectSession, WorldProjectSessionFailure, WorldProjectSessionLimits,
    WorldProjectSessionWork, MAX_WORLD_SESSION_WORK_PER_POLL,
};

impl Resource for EditorApplication {}

/// Run mode supported by the editor harness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EditorRunMode {
    /// No OS window or renderer is created.
    Headless,
    /// A host window and GPU surface backend are required.
    Windowed,
}

/// Bounds and lifecycle configuration for one editor application.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EditorAppConfig {
    /// Execution mode.
    pub mode: EditorRunMode,
    /// Maximum number of console entries retained by the application.
    pub console_capacity: usize,
    /// Maximum profiler frames retained by the application.
    pub profiler_limits: ProfilerLimits,
    /// Maximum frame count accepted by one headless run.
    pub max_run_frames: u64,
    /// Maximum native window events consumed by one windowed frame.
    pub max_window_events_per_frame: usize,
    /// Maximum number of configured host windows.
    pub max_windows: usize,
}

impl Default for EditorAppConfig {
    fn default() -> Self {
        Self {
            mode: EditorRunMode::Headless,
            console_capacity: 4_096,
            profiler_limits: ProfilerLimits::default(),
            max_run_frames: 10_000,
            max_window_events_per_frame: 4_096,
            max_windows: 32,
        }
    }
}

/// Failure while assembling or running the editor harness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EditorAppError {
    /// The configured console capacity is invalid.
    InvalidConsoleCapacity,
    /// Window count and per-frame event budgets must be non-zero.
    InvalidWindowLimits,
    /// Configuring another window would exceed the host window budget.
    WindowLimitExceeded {
        /// Maximum number of configured windows.
        limit: usize,
    },
    /// A document identity is already registered.
    DuplicateDocument(DocumentId),
    /// A document identity was not registered.
    MissingDocument(DocumentId),
    /// The requested run exceeds the configured frame budget.
    RunLimitExceeded {
        /// Requested frame count.
        requested: u64,
        /// Configured maximum frame count.
        limit: u64,
    },
    /// The window backend returned more events than one frame may process.
    WindowEventLimitExceeded {
        /// Number of events returned by the backend.
        observed: usize,
        /// Configured per-frame maximum.
        limit: usize,
    },
    /// A lifecycle operation was attempted from an incompatible state.
    InvalidState(InvalidStateError),
    /// The selected mode requires a window backend.
    MissingWindowBackend,
    /// The selected mode requires a GPU surface backend.
    MissingGpuBackend,
    /// A GPU backend reused an already-live surface identity.
    DuplicateSurface(SurfaceId),
    /// A host window was not known to the application.
    MissingWindow(WindowId),
    /// A surface was not known to the application.
    MissingSurface(WindowId),
    /// A window backend failed.
    WindowBackend(WindowBackendError),
    /// A GPU lifecycle backend failed.
    GpuSurface(GpuSurfaceError),
}

impl fmt::Display for EditorAppError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConsoleCapacity => {
                formatter.write_str("console capacity must be non-zero")
            }
            Self::InvalidWindowLimits => {
                formatter.write_str("window count and event limits must be non-zero")
            }
            Self::WindowLimitExceeded { limit } => {
                write!(formatter, "configured window limit {limit} exceeded")
            }
            Self::DuplicateDocument(id) => {
                write!(formatter, "document {} is already registered", id.raw())
            }
            Self::MissingDocument(id) => {
                write!(formatter, "document {} is not registered", id.raw())
            }
            Self::RunLimitExceeded { requested, limit } => {
                write!(
                    formatter,
                    "headless run requests {requested} frames; limit is {limit}"
                )
            }
            Self::WindowEventLimitExceeded { observed, limit } => {
                write!(
                    formatter,
                    "window backend returned {observed} events; per-frame limit is {limit}"
                )
            }
            Self::InvalidState(error) => error.fmt(formatter),
            Self::MissingWindowBackend => {
                formatter.write_str("windowed mode requires a window backend")
            }
            Self::MissingGpuBackend => {
                formatter.write_str("windowed mode requires a GPU surface backend")
            }
            Self::DuplicateSurface(id) => {
                write!(formatter, "GPU surface {} is already registered", id.raw())
            }
            Self::MissingWindow(id) => write!(formatter, "window {} is not registered", id.raw()),
            Self::MissingSurface(id) => write!(formatter, "window {} has no GPU surface", id.raw()),
            Self::WindowBackend(error) => write!(formatter, "window backend error: {error}"),
            Self::GpuSurface(error) => write!(formatter, "GPU surface error: {error}"),
        }
    }
}

impl std::error::Error for EditorAppError {}

impl From<InvalidStateError> for EditorAppError {
    fn from(error: InvalidStateError) -> Self {
        Self::InvalidState(error)
    }
}

impl From<WindowBackendError> for EditorAppError {
    fn from(error: WindowBackendError) -> Self {
        Self::WindowBackend(error)
    }
}

impl From<GpuSurfaceError> for EditorAppError {
    fn from(error: GpuSurfaceError) -> Self {
        Self::GpuSurface(error)
    }
}

/// Stable snapshot used for golden tests and CI divergence checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EditorSnapshot {
    /// Current frame number.
    pub frame: u64,
    /// Number of open documents.
    pub document_count: usize,
    /// Number of selected items.
    pub selection_count: usize,
    /// Number of retained console entries.
    pub console_count: usize,
    /// Number of open viewports.
    pub viewport_count: usize,
    /// Number of catalog assets.
    pub asset_count: usize,
    /// Deterministic checksum over the public state counts.
    pub checksum: u64,
}

/// Result of one bounded headless execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeadlessRunReport {
    /// Number of frames advanced.
    pub frames: u64,
    /// Snapshot at the end of the run.
    pub snapshot: EditorSnapshot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SurfaceBinding {
    id: SurfaceId,
    state: SurfaceState,
    size: PixelSize,
}

/// Assembled editor application state.
pub struct EditorApplication {
    config: EditorAppConfig,
    frame: u64,
    documents: BTreeMap<DocumentId, DocumentState>,
    selection: Selection,
    services: ServiceRegistry,
    commands: CommandIndex,
    shell: ShellContributions,
    dock: DockContributionRegistry,
    outliner: Outliner,
    console: Console,
    content: ContentCatalog,
    viewports: ViewportRegistry,
    profiler: FrameStore,
    state: ApplicationState,
    exit_status: ExitStatus,
    exit_reason: Option<ExitReason>,
    last_error: Option<EditorAppError>,
    windows: BTreeMap<WindowId, WindowDescriptor>,
    surfaces: BTreeMap<WindowId, SurfaceBinding>,
    window_backend: Option<Box<dyn WindowBackend>>,
    gpu_backend: Option<Box<dyn GpuSurfaceLifecycle>>,
    window_start_attempted: bool,
    gpu_start_attempted: bool,
}

impl EditorApplication {
    /// Creates a bounded editor application without acquiring host resources.
    pub fn new(config: EditorAppConfig) -> Result<Self, EditorAppError> {
        if config.console_capacity == 0 {
            return Err(EditorAppError::InvalidConsoleCapacity);
        }
        if config.max_windows == 0 || config.max_window_events_per_frame == 0 {
            return Err(EditorAppError::InvalidWindowLimits);
        }
        let console = Console::new(config.console_capacity)
            .map_err(|_| EditorAppError::InvalidConsoleCapacity)?;
        let profiler = FrameStore::new(config.profiler_limits);
        Ok(Self {
            config,
            frame: 0,
            documents: BTreeMap::new(),
            selection: Selection::default(),
            services: ServiceRegistry::default(),
            commands: CommandIndex::default(),
            shell: ShellContributions::default(),
            dock: DockContributionRegistry::new(),
            outliner: Outliner::default(),
            console,
            content: ContentCatalog::new(),
            viewports: ViewportRegistry::new(),
            profiler,
            state: ApplicationState::Created,
            exit_status: ExitStatus::NotRequested,
            exit_reason: None,
            last_error: None,
            windows: BTreeMap::from([(
                WindowId::from_raw(1),
                WindowDescriptor::new(
                    WindowId::from_raw(1),
                    "Prism",
                    PixelSize {
                        width: 1280,
                        height: 720,
                    },
                ),
            )]),
            surfaces: BTreeMap::new(),
            window_backend: None,
            gpu_backend: None,
            window_start_attempted: false,
            gpu_start_attempted: false,
        })
    }

    /// Returns the immutable application configuration.
    #[must_use]
    pub const fn config(&self) -> EditorAppConfig {
        self.config
    }

    /// Returns the current application lifecycle state.
    #[must_use]
    pub const fn state(&self) -> ApplicationState {
        self.state
    }

    /// Returns the current exit status.
    #[must_use]
    pub const fn exit_status(&self) -> ExitStatus {
        self.exit_status
    }

    /// Returns the most recent lifecycle failure, if any.
    #[must_use]
    pub const fn last_error(&self) -> Option<&EditorAppError> {
        self.last_error.as_ref()
    }

    /// Attaches platform adapters before starting a windowed application.
    pub fn with_backends(
        mut self,
        window_backend: impl WindowBackend + 'static,
        gpu_backend: impl GpuSurfaceLifecycle + 'static,
    ) -> Result<Self, EditorAppError> {
        self.set_backends(window_backend, gpu_backend)?;
        Ok(self)
    }

    /// Replaces the platform adapters before the next start or recovery.
    pub fn set_backends(
        &mut self,
        window_backend: impl WindowBackend + 'static,
        gpu_backend: impl GpuSurfaceLifecycle + 'static,
    ) -> Result<(), EditorAppError> {
        if !matches!(
            self.state,
            ApplicationState::Created | ApplicationState::Stopped | ApplicationState::Failed
        ) {
            return Err(InvalidStateError::new("replace backends", self.state).into());
        }
        self.window_backend = Some(Box::new(window_backend));
        self.gpu_backend = Some(Box::new(gpu_backend));
        Ok(())
    }

    /// Registers or replaces a window before startup.
    pub fn configure_window(&mut self, descriptor: WindowDescriptor) -> Result<(), EditorAppError> {
        if !matches!(
            self.state,
            ApplicationState::Created | ApplicationState::Stopped | ApplicationState::Failed
        ) {
            return Err(InvalidStateError::new("configure a window", self.state).into());
        }
        if descriptor.title.trim().is_empty() {
            return Err(WindowBackendError::new(
                WindowBackendErrorKind::InvalidRequest,
                "window title must not be empty",
            )
            .into());
        }
        if descriptor.size.width == 0 || descriptor.size.height == 0 {
            return Err(WindowBackendError::new(
                WindowBackendErrorKind::InvalidRequest,
                "window drawable size must be non-zero",
            )
            .into());
        }
        if !self.windows.contains_key(&descriptor.id)
            && self.windows.len() >= self.config.max_windows
        {
            return Err(EditorAppError::WindowLimitExceeded {
                limit: self.config.max_windows,
            });
        }
        self.windows.insert(descriptor.id, descriptor);
        Ok(())
    }

    /// Returns configured windows in stable identity order.
    pub fn windows(&self) -> impl Iterator<Item = &WindowDescriptor> {
        self.windows.values()
    }

    /// Returns the surface bound to a window, if startup created one.
    #[must_use]
    pub fn surface_for(&self, window: WindowId) -> Option<SurfaceId> {
        self.surfaces.get(&window).map(|binding| binding.id)
    }

    /// Returns the current deterministic frame number.
    #[must_use]
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// Returns the service registry used by editor modules.
    #[must_use]
    pub const fn services(&self) -> &ServiceRegistry {
        &self.services
    }

    /// Returns the mutable service registry used by editor modules.
    #[must_use]
    pub const fn services_mut(&mut self) -> &mut ServiceRegistry {
        &mut self.services
    }

    /// Returns the command index used by the workspace command palette.
    #[must_use]
    pub const fn commands(&self) -> &CommandIndex {
        &self.commands
    }

    /// Returns the mutable command index used by the workspace command palette.
    #[must_use]
    pub const fn commands_mut(&mut self) -> &mut CommandIndex {
        &mut self.commands
    }

    /// Returns shell surface contributions used by the workspace chrome.
    #[must_use]
    pub const fn shell(&self) -> &ShellContributions {
        &self.shell
    }

    /// Returns mutable shell surface contributions used by the workspace chrome.
    #[must_use]
    pub const fn shell_mut(&mut self) -> &mut ShellContributions {
        &mut self.shell
    }

    /// Returns the dock contribution mapping used by the workspace.
    #[must_use]
    pub const fn dock(&self) -> &DockContributionRegistry {
        &self.dock
    }

    /// Returns the mutable dock contribution mapping used by the workspace.
    #[must_use]
    pub const fn dock_mut(&mut self) -> &mut DockContributionRegistry {
        &mut self.dock
    }

    /// Returns the current selection.
    #[must_use]
    pub const fn selection(&self) -> &Selection {
        &self.selection
    }

    /// Returns the outliner data model.
    #[must_use]
    pub const fn outliner(&self) -> &Outliner {
        &self.outliner
    }

    /// Returns the console data model.
    #[must_use]
    pub const fn console(&self) -> &Console {
        &self.console
    }

    /// Returns the content catalog.
    #[must_use]
    pub const fn content(&self) -> &ContentCatalog {
        &self.content
    }

    /// Returns the viewport registry.
    #[must_use]
    pub const fn viewports(&self) -> &ViewportRegistry {
        &self.viewports
    }

    /// Returns the profiler frame store.
    #[must_use]
    pub const fn profiler(&self) -> &FrameStore {
        &self.profiler
    }

    /// Registers a document by stable identity.
    pub fn insert_document(&mut self, document: DocumentState) -> Result<(), EditorAppError> {
        let id = document.id();
        match self.documents.entry(id) {
            alloc::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(document);
                Ok(())
            }
            alloc::collections::btree_map::Entry::Occupied(_) => {
                Err(EditorAppError::DuplicateDocument(id))
            }
        }
    }

    /// Removes a document by stable identity.
    pub fn remove_document(&mut self, id: DocumentId) -> Result<DocumentState, EditorAppError> {
        self.documents
            .remove(&id)
            .ok_or(EditorAppError::MissingDocument(id))
    }

    /// Starts the application and acquires host resources in dependency order.
    pub fn start(&mut self) -> Result<(), EditorAppError> {
        if !matches!(
            self.state,
            ApplicationState::Created | ApplicationState::Stopped | ApplicationState::Failed
        ) {
            return Err(InvalidStateError::new("start", self.state).into());
        }
        self.state = ApplicationState::Starting;
        self.exit_status = ExitStatus::NotRequested;
        self.exit_reason = None;
        self.last_error = None;
        self.surfaces.clear();

        match self.start_host_resources() {
            Ok(()) => {
                self.state = ApplicationState::Running;
                Ok(())
            }
            Err(error) => {
                self.cleanup_host_resources();
                self.state = ApplicationState::Failed;
                self.exit_status = ExitStatus::Failed;
                self.last_error = Some(error.clone());
                Err(error)
            }
        }
    }

    fn start_host_resources(&mut self) -> Result<(), EditorAppError> {
        if self.config.mode == EditorRunMode::Headless {
            return Ok(());
        }
        if self.window_backend.is_none() {
            return Err(EditorAppError::MissingWindowBackend);
        }
        if self.gpu_backend.is_none() {
            return Err(EditorAppError::MissingGpuBackend);
        }
        let descriptors = self.windows.values().cloned().collect::<Vec<_>>();
        self.window_start_attempted = true;
        self.window_backend
            .as_mut()
            .ok_or(EditorAppError::MissingWindowBackend)?
            .start(&descriptors)?;
        let gpu_backend = self
            .gpu_backend
            .as_mut()
            .ok_or(EditorAppError::MissingGpuBackend)?;
        self.gpu_start_attempted = true;
        gpu_backend.start()?;
        for descriptor in &descriptors {
            let id = gpu_backend.create_surface(SurfaceDescriptor {
                window: descriptor.id,
                size: descriptor.size,
            })?;
            if self.surfaces.values().any(|surface| surface.id == id) {
                return Err(EditorAppError::DuplicateSurface(id));
            }
            self.surfaces.insert(
                descriptor.id,
                SurfaceBinding {
                    id,
                    state: SurfaceState::Active,
                    size: descriptor.size,
                },
            );
        }
        Ok(())
    }

    /// Requests a graceful application stop.
    pub fn request_exit(&mut self, reason: ExitReason) -> Result<(), EditorAppError> {
        if !matches!(
            self.state,
            ApplicationState::Running | ApplicationState::Suspended
        ) {
            return Err(InvalidStateError::new("request exit", self.state).into());
        }
        self.exit_reason = Some(reason);
        self.exit_status = ExitStatus::Requested(reason);
        Ok(())
    }

    /// Stops the application and releases resources in reverse dependency order.
    pub fn stop(&mut self, reason: ExitReason) -> Result<ShutdownReport, EditorAppError> {
        if self.state == ApplicationState::Stopped {
            return Ok(ShutdownReport {
                reason: self.exit_reason.unwrap_or(reason),
                frame: self.frame,
            });
        }
        if self.state == ApplicationState::Created {
            self.state = ApplicationState::Stopped;
            self.exit_reason = Some(reason);
            self.exit_status = ExitStatus::Completed(reason);
            return Ok(ShutdownReport {
                reason,
                frame: self.frame,
            });
        }
        self.state = ApplicationState::Stopping;
        self.exit_reason = Some(reason);
        self.exit_status = ExitStatus::Requested(reason);
        let cleanup_error = self.cleanup_host_resources();
        if let Some(error) = cleanup_error {
            self.state = ApplicationState::Failed;
            self.exit_status = ExitStatus::Failed;
            self.last_error = Some(error.clone());
            return Err(error);
        }
        self.state = ApplicationState::Stopped;
        self.exit_status = ExitStatus::Completed(reason);
        Ok(ShutdownReport {
            reason,
            frame: self.frame,
        })
    }

    fn cleanup_host_resources(&mut self) -> Option<EditorAppError> {
        let mut first_error = None;
        if let Some(gpu_backend) = self
            .gpu_backend
            .as_mut()
            .filter(|_| self.gpu_start_attempted)
        {
            for binding in self.surfaces.values().rev() {
                if let Err(error) = gpu_backend.destroy_surface(binding.id)
                    && first_error.is_none()
                {
                    first_error = Some(EditorAppError::GpuSurface(error));
                }
            }
            self.surfaces.clear();
            if let Err(error) = gpu_backend.stop()
                && first_error.is_none()
            {
                first_error = Some(EditorAppError::GpuSurface(error));
            }
            self.gpu_start_attempted = false;
        }
        if let Some(window_backend) = self
            .window_backend
            .as_mut()
            .filter(|_| self.window_start_attempted)
        {
            if let Err(error) = window_backend.stop()
                && first_error.is_none()
            {
                first_error = Some(EditorAppError::WindowBackend(error));
            }
            self.window_start_attempted = false;
        }
        first_error
    }

    /// Rebuilds host resources after a lifecycle failure.
    pub fn recover(&mut self) -> Result<(), EditorAppError> {
        if self.state != ApplicationState::Failed {
            return Err(InvalidStateError::new("recover", self.state).into());
        }
        self.start()
    }

    /// Advances one deterministic editor frame.
    pub fn tick(&mut self) -> Result<EditorSnapshot, EditorAppError> {
        if !matches!(
            self.state,
            ApplicationState::Running | ApplicationState::Suspended
        ) {
            return Err(InvalidStateError::new("advance a frame", self.state).into());
        }
        match self.tick_inner() {
            Ok(snapshot) => Ok(snapshot),
            Err(error) => {
                self.cleanup_host_resources();
                self.state = ApplicationState::Failed;
                self.exit_status = ExitStatus::Failed;
                self.last_error = Some(error.clone());
                Err(error)
            }
        }
    }

    fn tick_inner(&mut self) -> Result<EditorSnapshot, EditorAppError> {
        if let Some(reason) = self.exit_reason {
            self.stop(reason)?;
            return Ok(self.snapshot());
        }
        if self.config.mode == EditorRunMode::Windowed {
            let events = self
                .window_backend
                .as_mut()
                .ok_or(EditorAppError::MissingWindowBackend)?
                .poll_events(self.config.max_window_events_per_frame)?;
            if events.len() > self.config.max_window_events_per_frame {
                return Err(EditorAppError::WindowEventLimitExceeded {
                    observed: events.len(),
                    limit: self.config.max_window_events_per_frame,
                });
            }
            for envelope in events {
                self.handle_window_event(envelope)?;
                if let Some(reason) = self.exit_reason {
                    self.stop(reason)?;
                    return Ok(self.snapshot());
                }
            }
        }
        if self.state == ApplicationState::Suspended {
            return Ok(self.snapshot());
        }
        self.frame = self.frame.saturating_add(1);
        if self.config.mode == EditorRunMode::Windowed {
            for (window, binding) in &self.surfaces {
                if binding.state != SurfaceState::Active {
                    continue;
                }
                self.gpu_backend
                    .as_mut()
                    .ok_or(EditorAppError::MissingGpuBackend)?
                    .present_surface(binding.id)?;
                self.window_backend
                    .as_mut()
                    .ok_or(EditorAppError::MissingWindowBackend)?
                    .present(*window)?;
            }
        }
        Ok(self.snapshot())
    }

    fn handle_window_event(&mut self, envelope: WindowEventEnvelope) -> Result<(), EditorAppError> {
        if !self.windows.contains_key(&envelope.window) {
            return Err(EditorAppError::MissingWindow(envelope.window));
        }
        match envelope.event {
            WindowEvent::CloseRequested => self.request_exit(ExitReason::WindowClosed),
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    return self.handle_window_event(WindowEventEnvelope {
                        window: envelope.window,
                        event: WindowEvent::Suspended,
                    });
                }
                self.windows
                    .get_mut(&envelope.window)
                    .expect("window existence checked above")
                    .size = size;
                let surface = self
                    .surfaces
                    .get(&envelope.window)
                    .ok_or(EditorAppError::MissingSurface(envelope.window))?;
                let surface_id = surface.id;
                let was_suspended = surface.state == SurfaceState::Suspended;
                self.gpu_backend
                    .as_mut()
                    .ok_or(EditorAppError::MissingGpuBackend)?
                    .resize_surface(surface_id, size)?;
                if was_suspended {
                    self.gpu_backend
                        .as_mut()
                        .ok_or(EditorAppError::MissingGpuBackend)?
                        .resume_surface(surface_id)?;
                    self.state = ApplicationState::Running;
                }
                let binding = self
                    .surfaces
                    .get_mut(&envelope.window)
                    .expect("surface existence checked above");
                binding.size = size;
                if was_suspended {
                    binding.state = SurfaceState::Active;
                }
                Ok(())
            }
            WindowEvent::Focused(_) | WindowEvent::RedrawRequested => Ok(()),
            WindowEvent::Suspended => {
                let surface_id = self
                    .surfaces
                    .get(&envelope.window)
                    .ok_or(EditorAppError::MissingSurface(envelope.window))?;
                if surface_id.state == SurfaceState::Suspended {
                    return Ok(());
                }
                let surface_id = surface_id.id;
                self.gpu_backend
                    .as_mut()
                    .ok_or(EditorAppError::MissingGpuBackend)?
                    .suspend_surface(surface_id)?;
                self.surfaces
                    .get_mut(&envelope.window)
                    .expect("surface existence checked above")
                    .state = SurfaceState::Suspended;
                if self
                    .surfaces
                    .values()
                    .all(|surface| surface.state == SurfaceState::Suspended)
                {
                    self.state = ApplicationState::Suspended;
                }
                Ok(())
            }
            WindowEvent::Resumed => {
                let surface_id = self
                    .surfaces
                    .get(&envelope.window)
                    .ok_or(EditorAppError::MissingSurface(envelope.window))?;
                if surface_id.state == SurfaceState::Active {
                    return Ok(());
                }
                let surface_id = surface_id.id;
                self.gpu_backend
                    .as_mut()
                    .ok_or(EditorAppError::MissingGpuBackend)?
                    .resume_surface(surface_id)?;
                self.surfaces
                    .get_mut(&envelope.window)
                    .expect("surface existence checked above")
                    .state = SurfaceState::Active;
                self.state = ApplicationState::Running;
                Ok(())
            }
        }
    }

    /// Runs a bounded headless loop and invokes a hook after each frame.
    pub fn run_headless<F>(
        &mut self,
        frames: u64,
        mut hook: F,
    ) -> Result<HeadlessRunReport, EditorAppError>
    where
        F: FnMut(&mut Self),
    {
        if self.config.mode != EditorRunMode::Headless {
            return Err(InvalidStateError::new("run headless", self.state).into());
        }
        if frames > self.config.max_run_frames {
            return Err(EditorAppError::RunLimitExceeded {
                requested: frames,
                limit: self.config.max_run_frames,
            });
        }
        if self.state == ApplicationState::Created {
            self.start()?;
        }
        if self.state != ApplicationState::Running {
            return Err(InvalidStateError::new("run headless", self.state).into());
        }
        let initial_frame = self.frame;
        for _ in 0..frames {
            self.tick()?;
            if self.state == ApplicationState::Stopped {
                break;
            }
            hook(self);
            if self.exit_reason.is_some() || self.state != ApplicationState::Running {
                break;
            }
        }
        if self.state != ApplicationState::Stopped {
            self.stop(self.exit_reason.unwrap_or(ExitReason::FrameLimit))?;
        }
        Ok(HeadlessRunReport {
            frames: self.frame.saturating_sub(initial_frame),
            snapshot: self.snapshot(),
        })
    }

    /// Captures a stable public-state snapshot.
    #[must_use]
    pub fn snapshot(&self) -> EditorSnapshot {
        let document_count = self.documents.len();
        let selection_count = self.selection.items().len();
        let console_count = self.console.entries().len();
        let viewport_count = self.viewports.iter().count();
        let asset_count = self.content.len();
        let checksum = checksum([
            self.frame,
            document_count as u64,
            selection_count as u64,
            console_count as u64,
            viewport_count as u64,
            asset_count as u64,
        ]);
        EditorSnapshot {
            frame: self.frame,
            document_count,
            selection_count,
            console_count,
            viewport_count,
            asset_count,
            checksum,
        }
    }

    /// Builds the complete renderer-independent Loom workspace for the current state.
    #[must_use]
    pub fn workspace(&self) -> Element {
        let context = ContextKeySet::default();
        let menus = self.shell.menu("main", &context);
        let toolbar = self.shell.toolbar("main", &context);
        let status = self.shell.status(&context);
        let chrome = shell_chrome(&ShellChromeProps {
            title: "Prism Studio",
            workspace: "Default",
            menus: &menus,
            toolbar: &toolbar,
            status: &status,
        });
        let palette = command_palette(&CommandPaletteProps {
            index: &self.commands,
            query: "",
            limit: 64,
            selected_id: None,
        });
        let outliner = outliner_panel(&OutlinerViewProps {
            outliner: &self.outliner,
            filter: "",
            start: 0,
            count: 512,
            selected: None,
        });
        let console = console_panel(&ConsolePanelProps {
            console: &self.console,
            minimum: prism_editor_panels::LogLevel::Trace,
            channel: None,
            query: "",
            start: 0,
            count: 512,
        });
        let inspector = inspector_view(&InspectorViewProps {
            projection: &empty_inspector_projection(),
            show_root: false,
        });
        let viewport = viewport_host(&ViewportHostProps {
            viewport: self.viewports.iter().next(),
            overlays: Vec::new(),
        });
        let dock = dock_workspace(&DockWorkspaceProps {
            panels: &self.dock,
            active_panel: None,
            bodies: vec![palette],
        });
        editor_workspace(&EditorWorkspaceProps {
            chrome,
            dock,
            outliner,
            viewport,
            inspector,
            console,
        })
    }
}

impl Drop for EditorApplication {
    fn drop(&mut self) {
        self.cleanup_host_resources();
    }
}

/// Builds a native Prism application with the editor harness installed as a
/// resource and one deterministic update system.
pub fn build_prism_app(config: EditorAppConfig) -> Result<App, EditorAppError> {
    let mut app = App::new();
    let mut editor = EditorApplication::new(config)?;
    editor.start()?;
    app.world_mut().insert_resource(editor);
    app.add_systems(Update, tick_editor_resource);
    Ok(app)
}

/// Builds a `prism_app` with explicit window and GPU adapters.
pub fn build_prism_app_with_backends(
    config: EditorAppConfig,
    window_backend: impl WindowBackend + 'static,
    gpu_backend: impl GpuSurfaceLifecycle + 'static,
) -> Result<App, EditorAppError> {
    let mut app = App::new();
    let mut editor = EditorApplication::new(config)?.with_backends(window_backend, gpu_backend)?;
    editor.start()?;
    app.world_mut().insert_resource(editor);
    app.add_systems(Update, tick_editor_resource);
    Ok(app)
}

fn tick_editor_resource(mut editor: ResMut<EditorApplication>, mut exit: ResMut<AppExitRequest>) {
    if matches!(
        editor.state(),
        ApplicationState::Running | ApplicationState::Suspended
    ) && editor.tick().is_err()
    {
        exit.send_error();
        return;
    }
    match editor.exit_status() {
        ExitStatus::Completed(ExitReason::Failure) | ExitStatus::Failed => exit.send_error(),
        ExitStatus::Completed(_) => exit.send_success(),
        ExitStatus::NotRequested | ExitStatus::Requested(_) => {}
    }
}

fn checksum(values: impl IntoIterator<Item = u64>) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn empty_inspector_projection() -> InspectorProjection {
    InspectorProjection {
        root: InspectorProjectionNode {
            path: None,
            node: InspectorNode {
                label: String::from("root"),
                type_name: String::from("Empty"),
                kind: InspectorKind::Value,
                variant: None,
                value: None,
                hints: InspectorHints::default(),
                children: Vec::new(),
            },
            children: Vec::new(),
        },
    }
}

#[cfg(test)]
mod tests;
