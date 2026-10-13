//! Pure-data docking model and deterministic layout reducer.
//!
//! This crate deliberately contains no renderer, widget factory, or
//! window-system integration. It provides a deterministic layout model,
//! reducer, validation, and a versioned persistence core; it does not provide
//! application-level workspace integration or background autosave orchestration;
//! the `std` feature exposes a host-driven debounce controller for that boundary.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

#[cfg(feature = "std")]
mod autosave;
mod default_layout;
mod drag;
mod drop_preview;
mod model;
mod persist;
mod reducer;
mod registry;
mod render_plan;
mod validate;
#[cfg(feature = "render")]
mod view;
mod workspace;
mod workspace_default;
mod workspace_persist;
#[cfg(feature = "std")]
mod workspace_store;

#[cfg(feature = "std")]
pub use autosave::{AutosavePoll, WorkspaceAutosave};
pub use default_layout::{DefaultLayoutError, DefaultLayoutPolicy};
pub use drag::{
    calculate_drop_zone, map_drop_action, tab_insertion_index, transition, DragError, DragEvent,
    DragState, DropMapError, DropZone, PointerPosition,
};
pub use drop_preview::{resolve_drop_preview, DropPreview, DropPreviewError};
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
pub use render_plan::{
    solve, solve_with, AutoHideBarLayout, AutoHideButton, AutoHideLayout, AutoHideReveal,
    DocumentLayout, FloatingLayout, FocusMetadata, FocusReason, FocusTarget, LayoutMetrics,
    MaximizedLayout, NodePath, PanelLayout, PanelVisibility, RenderPlan, RenderPlanError,
    SizeRequirement, SplitterLayout, SurfaceId, TabHeader, TabId, Viewport,
};
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
