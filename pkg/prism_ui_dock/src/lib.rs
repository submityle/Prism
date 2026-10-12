//! Pure-data docking model and deterministic layout reducer.
//!
//! This crate deliberately contains no renderer, widget factory, or
//! window-system integration. It provides a deterministic layout model,
//! reducer, validation, and a versioned persistence core; it does not provide
//! application-level workspace integration or autosave orchestration.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod drag;
mod model;
mod persist;
mod reducer;
mod registry;
mod render_plan;
mod validate;
mod workspace;
mod workspace_persist;

pub use drag::{
    calculate_drop_zone, map_drop_action, tab_insertion_index, transition, DragError, DragEvent,
    DragState, DropMapError, DropZone, PointerPosition,
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
pub use render_plan::{
    solve, solve_with, AutoHideBarLayout, AutoHideButton, AutoHideLayout, AutoHideReveal,
    DocumentLayout, FloatingLayout, FocusMetadata, FocusReason, FocusTarget, LayoutMetrics,
    MaximizedLayout, NodePath, PanelLayout, PanelVisibility, RenderPlan, RenderPlanError,
    SizeRequirement, SplitterLayout, SurfaceId, TabHeader, TabId, Viewport,
};
pub use validate::{LayoutError, Limits, ValidationIssue};
pub use workspace::{NameKind, Perspective, Workspace, WorkspaceError, WorkspaceManager};
pub use workspace_persist::{
    decode_workspace, decode_workspace_with_registry, encode_workspace, WorkspaceDecodeLimits,
    WorkspaceDecodeOptions, WorkspacePersistError,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod render_plan_tests;

#[cfg(test)]
mod workspace_tests;
