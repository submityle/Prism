//! Pure-data docking model and deterministic layout reducer.
//!
//! This crate deliberately contains no renderer, widget factory, window-system
//! integration, or serialization format. Layouts can be validated and changed
//! without a GUI or GPU.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod drag;
mod model;
mod reducer;
mod validate;

pub use drag::{
    calculate_drop_zone, map_drop_action, tab_insertion_index, transition, DragError, DragEvent,
    DragState, DropMapError, DropZone, PointerPosition,
};
pub use model::{
    AutoHideBars, AutoHideEntry, Axis, DockLayout, DockNode, DocumentId, FloatingWindow,
    MaximizeRestore, PanelId, Ratio, Rect, Side, SplitChild,
};
pub use reducer::{reduce, Action, Placement, Target};
pub use validate::{LayoutError, Limits, ValidationIssue};

#[cfg(test)]
mod tests;
