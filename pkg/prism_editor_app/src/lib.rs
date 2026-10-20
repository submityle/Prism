//! Deterministic headless assembly for the Prism editor.
//!
//! The application layer owns lifecycle and frame-budget policy. The editor
//! kernel and panel/content/viewport/profiler crates remain independently
//! usable, while this crate provides one bounded composition for CI and tools.

#![forbid(unsafe_code)]

extern crate alloc;

use alloc::collections::BTreeMap;
use core::fmt;

use prism_app::prelude::{App, ResMut, Update};
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

impl Resource for EditorApplication {}

/// Run mode supported by the editor harness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EditorRunMode {
    /// No OS window or renderer is created.
    Headless,
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
}

impl Default for EditorAppConfig {
    fn default() -> Self {
        Self {
            mode: EditorRunMode::Headless,
            console_capacity: 4_096,
            profiler_limits: ProfilerLimits::default(),
            max_run_frames: 10_000,
        }
    }
}

/// Failure while assembling or running the editor harness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EditorAppError {
    /// The configured console capacity is invalid.
    InvalidConsoleCapacity,
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
}

impl fmt::Display for EditorAppError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConsoleCapacity => {
                formatter.write_str("console capacity must be non-zero")
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
        }
    }
}

impl std::error::Error for EditorAppError {}

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
}

impl EditorApplication {
    /// Creates a bounded headless editor application.
    pub fn new(config: EditorAppConfig) -> Result<Self, EditorAppError> {
        if config.console_capacity == 0 {
            return Err(EditorAppError::InvalidConsoleCapacity);
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
        })
    }

    /// Returns the immutable application configuration.
    #[must_use]
    pub const fn config(&self) -> EditorAppConfig {
        self.config
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
        if self.documents.insert(id, document).is_some() {
            return Err(EditorAppError::DuplicateDocument(id));
        }
        Ok(())
    }

    /// Removes a document by stable identity.
    pub fn remove_document(&mut self, id: DocumentId) -> Result<DocumentState, EditorAppError> {
        self.documents
            .remove(&id)
            .ok_or(EditorAppError::MissingDocument(id))
    }

    /// Advances one deterministic editor frame.
    pub fn tick(&mut self) -> EditorSnapshot {
        self.frame = self.frame.saturating_add(1);
        self.snapshot()
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
        if frames > self.config.max_run_frames {
            return Err(EditorAppError::RunLimitExceeded {
                requested: frames,
                limit: self.config.max_run_frames,
            });
        }
        for _ in 0..frames {
            self.tick();
            hook(self);
        }
        Ok(HeadlessRunReport {
            frames,
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
        let asset_count = self.content.query(&Default::default()).len();
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

/// Builds a native Prism application with the editor harness installed as a
/// resource and one deterministic update system.
pub fn build_prism_app(config: EditorAppConfig) -> Result<App, EditorAppError> {
    let mut app = App::new();
    app.world_mut()
        .insert_resource(EditorApplication::new(config)?);
    app.add_systems(Update, tick_editor_resource);
    Ok(app)
}

fn tick_editor_resource(mut editor: ResMut<EditorApplication>) {
    editor.tick();
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
mod tests {
    use super::*;
    use prism_editor_core::{DocumentId, DocumentState};

    #[test]
    fn headless_run_is_bounded_and_deterministic() {
        let mut first = EditorApplication::new(EditorAppConfig::default()).expect("app");
        let mut second = EditorApplication::new(EditorAppConfig::default()).expect("app");
        let first_report = first.run_headless(4, |_| {}).expect("run");
        let second_report = second.run_headless(4, |_| {}).expect("run");
        assert_eq!(first_report, second_report);
        assert_eq!(first_report.snapshot.frame, 4);
        assert!(first.run_headless(10_001, |_| {}).is_err());
    }

    #[test]
    fn document_registration_changes_snapshot_without_global_state() {
        let mut app = EditorApplication::new(EditorAppConfig::default()).expect("app");
        let document = DocumentState::new(DocumentId::from_raw(7));
        app.insert_document(document).expect("insert");
        assert_eq!(app.snapshot().document_count, 1);
        assert!(app.remove_document(DocumentId::from_raw(7)).is_ok());
        assert_eq!(app.snapshot().document_count, 0);
    }

    #[test]
    fn frame_hook_can_observe_and_mutate_each_frame() {
        let mut app = EditorApplication::new(EditorAppConfig::default()).expect("app");
        let mut observed = Vec::new();
        app.run_headless(3, |app| observed.push(app.frame()))
            .expect("run");
        assert_eq!(observed, [1, 2, 3]);
    }

    #[test]
    fn workspace_assembles_the_complete_renderer_independent_surface() {
        let app = EditorApplication::new(EditorAppConfig::default()).expect("app");
        let workspace = app.workspace();
        assert_eq!(
            workspace.kind(),
            &prism_ui::ElementKind::Custom("prism.editor.workspace".into())
        );
        assert_eq!(workspace.child_elements().len(), 3);
        assert_eq!(workspace.child_elements()[1].child_elements().len(), 3);
    }
}
