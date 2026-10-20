//! Concrete Loom Element assembly for Prism Studio.
//!
//! The editor model crates own state and policy. This crate owns the final
//! renderer-independent view tree: it turns command results, panel snapshots,
//! reflection projections, dock contributions, and viewport state into stable
//! [`prism_ui::Element`] subtrees. Domain controllers use Prism contracts;
//! rendering and native device-event routing remain with the application host.

#![forbid(unsafe_code)]

mod action;
mod chrome;
mod command;
mod console;
mod content;
mod content_collection;
mod content_diagnostics;
mod content_diagnostics_view;
mod content_thumbnails;
mod content_view;
mod dock;
mod inspector;
mod outliner;
mod profiler;
mod profiler_view;
mod support;
mod timeline;
mod timeline_view;
mod viewport;
mod workspace;
mod world;
mod world_view;

pub use action::{
    ActionBinding, ActionBindings, ActionError, ActionTarget, ActionTemplate, ActionTemplateKind,
    ActionValue, BoundElement, EditorAction, SelectionMode, UiBuildError, UiEvent, UiTrigger,
    MAX_ACTION_ID_BYTES, MAX_ACTION_PATH_BYTES, MAX_ACTION_TARGET_BYTES, MAX_ACTION_TEXT_BYTES,
};
pub use chrome::shell_chrome_bound;
pub use chrome::{shell_chrome, ShellChromeProps};
pub use command::{
    command_palette, command_palette_bound, command_palette_results, command_palette_results_bound,
    CommandPaletteProps,
};
pub use console::{console_panel, ConsolePanelProps};
pub use content::{
    ContentBrowserAction, ContentBrowserController, ContentBrowserError, ContentCollection,
    ContentLayout, MAX_COLLECTION_MEMBERS, MAX_CONTENT_COLLECTIONS, MAX_CONTENT_PAGE_HISTORY,
    MAX_CONTENT_SELECTION,
};
pub use content_collection::{
    CollectionLoadReport, CollectionSaveReport, CollectionStorageError, CollectionStore,
    MissingCollectionMember, MAX_COLLECTION_FILE_BYTES,
};
pub use content_diagnostics::{
    ContentDiagnosticsAction, ContentDiagnosticsController, ContentDiagnosticsError,
    DependencyDirection, MAX_DEPENDENCY_NAVIGATION_HISTORY,
};
pub use content_diagnostics_view::BoundContentDiagnostics;
pub use content_thumbnails::{
    ContentThumbnailState, ContentThumbnails, MAX_CONTENT_THUMBNAIL_BYTES,
};
pub use content_view::{BoundContentBrowser, ContentImageVisual};
pub use dock::{dock_workspace, DockWorkspaceProps};
pub use inspector::{inspector_view, inspector_view_bound, InspectorViewProps};
pub use outliner::{outliner_panel, outliner_panel_bound, OutlinerViewProps};
pub use profiler::{
    ProfilerAction, ProfilerController, ProfilerDomain, ProfilerFrameSelection, ProfilerPanelError,
    ProfilerPanelLimits, ProfilerSources, ProfilerTimeWindow, MAX_PROFILER_INDEXED_SPANS,
    MAX_PROFILER_VISIBLE_FRAMES, MAX_PROFILER_VISIBLE_SPANS, MAX_PROFILER_VISIBLE_TRACKS,
};
pub use profiler_view::{BoundProfilerPanel, ProfilerSpanVisual, ProfilerViewStats};
pub use support::{
    MAX_ACTION_BINDINGS, MAX_INSPECTOR_DEPTH, MAX_INSPECTOR_NODES, MAX_VISIBLE_ROWS,
};
pub use timeline::{
    TimelineKeySelection, TimelinePanelAction, TimelinePanelController, TimelinePanelError,
    TimelineWindow,
};
pub use timeline_view::{BoundTimelinePanel, TimelineCurvePoint};
pub use viewport::{viewport_host, ViewportHostProps};
pub use workspace::{
    editor_workspace, editor_workspace_bound, BoundEditorWorkspaceProps, EditorWorkspaceProps,
};
pub use world::{
    WorldGridPlane, WorldPanelAction, WorldPanelController, WorldPanelError, WorldPanelIntent,
    WorldPanelIntentKind, WorldPanelLease, WorldPanelLimits, WorldPanelViewport,
    MAX_WORLD_PANEL_CELLS, MAX_WORLD_PANEL_INTENTS, MAX_WORLD_PANEL_LAYERS, MAX_WORLD_PANEL_LEASES,
};
pub use world_view::{BoundWorldPanel, WorldCellVisual, WorldPanelViewStats};

#[cfg(test)]
mod tests;
