//! Pure-data docking model and deterministic layout reducer.
//!
//! This crate deliberately contains no renderer, widget factory, or
//! window-system integration. It provides a deterministic layout model,
//! reducer, validation, and a versioned persistence core; it does not provide
//! application-level workspace integration or background autosave orchestration;
//! the `std` feature exposes a host-driven debounce controller for that host
//! contract.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

#[cfg(feature = "a11y")]
mod a11y;
#[cfg(feature = "std")]
mod autosave;
mod default_layout;
mod display;
mod drag;
mod drop_preview;
#[cfg(feature = "render")]
mod drop_preview_view;
mod frame;
mod history;
mod metrics;
mod migration;
mod model;
mod persist;
mod reducer;
mod registry;
mod render_cache;
mod render_plan;
mod tab_window;
mod validate;
#[cfg(feature = "render")]
mod view;
mod workspace;
mod workspace_default;
mod workspace_persist;
#[cfg(feature = "std")]
mod workspace_store;

#[cfg(feature = "a11y")]
pub use a11y::{DockA11yAction, DockA11ySnapshot};
#[cfg(feature = "std")]
pub use autosave::{AutosavePoll, WorkspaceAutosave};
pub use default_layout::{DefaultLayoutError, DefaultLayoutPolicy};
pub use display::{
    restore_all, restore_placement, FloatingPlacement, MonitorWorkArea, RestoredPlacement,
};
pub use drag::{
    calculate_drop_zone, map_drop_action, tab_insertion_index, transition, DragError, DragEvent,
    DragState, DropMapError, DropZone, PointerPosition,
};
pub use drop_preview::{resolve_drop_preview, DropPreview, DropPreviewError};
#[cfg(feature = "render")]
pub use drop_preview_view::drop_preview_element;
pub use frame::{DockHost, DockHostError};
pub use history::{DockHistory, HistoryError};
pub use metrics::{DockFrameSample, DockMetricSummary, DockRuntimeMetrics};
pub use migration::{
    rewrite_version, rewrite_workspace_version, MigrationError, MigrationRegistry, MigrationStep,
    WorkspaceMigrationRegistry, CURRENT_LAYOUT_VERSION, CURRENT_WORKSPACE_VERSION,
};
pub use model::{
    AutoHideBars, AutoHideEntry, Axis, DockLayout, DockNode, DocumentId, FloatingWindow,
    MaximizeRestore, PanelId, Ratio, Rect, Side, SplitChild,
};
pub use persist::{
    decode_layout, decode_layout_with_registry, encode_layout, DecodeLimits, DecodeOptions,
    PersistError,
};
pub use reducer::{reduce, Action, Placement, Target};
pub use registry::{DefaultSlot, Instancing, PanelDescriptor, PanelRegistry, PanelRegistryError};
pub use render_cache::{RenderPlanCache, RenderPlanKey};
pub use render_plan::{
    solve, solve_with, AutoHideBarLayout, AutoHideButton, AutoHideLayout, AutoHideReveal,
    DocumentLayout, FloatingLayout, FocusMetadata, FocusReason, FocusTarget, LayoutMetrics,
    MaximizedLayout, NodePath, PanelLayout, PanelVisibility, RenderPlan, RenderPlanError,
    SizeRequirement, SplitterLayout, SurfaceId, TabHeader, TabId, Viewport,
};
pub use tab_window::visible_tab_range;
pub use validate::{LayoutError, Limits, ValidationIssue};
#[cfg(feature = "render")]
pub use view::{DockView, DockViewError, DocumentViewRegistry, PanelViewRegistry, ViewFactory};
pub use workspace::{NameKind, Perspective, Workspace, WorkspaceError, WorkspaceManager};
pub use workspace_default::DefaultLayoutResetError;
pub use workspace_persist::{
    decode_workspace, decode_workspace_with_registry, encode_workspace, WorkspaceDecodeLimits,
    WorkspaceDecodeOptions, WorkspacePersistError,
};
#[cfg(feature = "std")]
pub use workspace_store::{
    LoadedWorkspace, WorkspaceFileStore, WorkspaceLoadSource, WorkspaceStoreError,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod render_plan_tests;

#[cfg(test)]
mod workspace_tests;
