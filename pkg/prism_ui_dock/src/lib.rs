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
mod input;
mod instance_persist;
mod metrics;
mod migration;
mod model;
mod motion;
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
#[cfg(feature = "input")]
pub use drop_preview_view::drop_preview_hit_node;
#[cfg(feature = "render")]
pub use drop_preview_view::{
    drop_preview_element, drop_preview_element_with_style, DropPreviewKind, DropPreviewVisualStyle,
    DROP_PREVIEW_BACKGROUND_TOKEN, DROP_PREVIEW_BORDER_TOKEN, DROP_PREVIEW_CLASS,
    DROP_PREVIEW_CONTRAST_CLASS, DROP_PREVIEW_CONTRAST_TOKEN,
    DROP_PREVIEW_POINTER_EVENTS_NONE_CLASS, DROP_PREVIEW_RADIUS_TOKEN,
};
pub use frame::{DockHost, DockHostError};
pub use history::{DockHistory, HistoryError};
pub use input::{DockInputError, DockInputEvent, DockInputOutput};
pub use instance_persist::{
    decode_instance_registry, encode_instance_registry, InstancePersistError,
};
pub use metrics::{DockFrameSample, DockMetricSummary, DockRuntimeMetrics};
pub use migration::{
    migrate_legacy_layout_v0, migrate_legacy_workspace_v0, rewrite_version,
    rewrite_workspace_version, MigrationError, MigrationRegistry, MigrationStep,
    WorkspaceMigrationRegistry, CURRENT_LAYOUT_VERSION, CURRENT_WORKSPACE_VERSION,
};
pub use model::{
    AutoHideBars, AutoHideEntry, Axis, DockLayout, DockNode, DocumentId, FloatingWindow,
    MaximizeRestore, PanelId, Ratio, Rect, Side, SplitChild,
};
pub use motion::{DockGeometryTransition, DockPlanTransition, MotionMode, MotionPolicy};
pub use persist::{
    decode_layout, decode_layout_with_registry, encode_layout, DecodeLimits, DecodeOptions,
    PersistError,
};
pub use reducer::{activate_tab_action, reduce, Action, Placement, Target};
pub use registry::{
    DefaultSlot, Instancing, PanelDescriptor, PanelInstance, PanelInstanceId,
    PanelInstanceRegistry, PanelInstanceRegistryError, PanelInstanceState, PanelRegistry,
    PanelRegistryError,
};
pub use render_cache::{RenderPlanCache, RenderPlanKey};
pub use render_plan::{
    hit_test, solve, solve_with, AutoHideBarLayout, AutoHideButton, AutoHideLayout, AutoHideReveal,
    DockHit, DocumentLayout, FloatingLayout, FocusMetadata, FocusReason, FocusTarget, HitTarget,
    LayoutMetrics, MaximizedLayout, NodePath, PanelLayout, PanelVisibility, RenderPlan,
    RenderPlanError, SizeRequirement, SplitterLayout, SurfaceId, TabHeader, TabId, Viewport,
};
pub use tab_window::visible_tab_range;
pub use validate::{LayoutError, Limits, ValidationIssue};
#[cfg(feature = "render")]
pub use view::{
    ContextualViewFactory, DockView, DockViewError, DockViewOptions, DocumentViewRegistry,
    FallibleViewFactory, PanelViewRegistry, ViewFactory, ViewFactoryContext, ViewFactoryError,
};
pub use workspace::{NameKind, Perspective, Workspace, WorkspaceError, WorkspaceManager};
pub use workspace_default::DefaultLayoutResetError;
pub use workspace_persist::{
    decode_workspace, decode_workspace_with_registry, encode_workspace, WorkspaceDecodeLimits,
    WorkspaceDecodeOptions, WorkspacePersistError,
};
#[cfg(feature = "std")]
pub use workspace_store::{
    LoadedInstances, LoadedWorkspace, WorkspaceFileStore, WorkspaceLoadSource, WorkspaceStoreError,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod render_plan_tests;

#[cfg(test)]
mod workspace_tests;
