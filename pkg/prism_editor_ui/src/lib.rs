//! Concrete Loom Element assembly for Prism Studio.
//!
//! The editor model crates own state and policy. This crate owns the final
//! renderer-independent view tree: it turns command results, panel snapshots,
//! reflection projections, dock contributions, and viewport state into stable
//! [`prism_ui::Element`] subtrees. No renderer or engine ECS is referenced.

#![forbid(unsafe_code)]

mod action;
mod chrome;
mod command;
mod console;
mod dock;
mod inspector;
mod outliner;
mod support;
mod viewport;
mod workspace;

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
pub use dock::{dock_workspace, DockWorkspaceProps};
pub use inspector::{inspector_view, inspector_view_bound, InspectorViewProps};
pub use outliner::{outliner_panel, outliner_panel_bound, OutlinerViewProps};
pub use support::{
    MAX_ACTION_BINDINGS, MAX_INSPECTOR_DEPTH, MAX_INSPECTOR_NODES, MAX_VISIBLE_ROWS,
};
pub use viewport::{viewport_host, ViewportHostProps};
pub use workspace::{
    editor_workspace, editor_workspace_bound, BoundEditorWorkspaceProps, EditorWorkspaceProps,
};

#[cfg(test)]
mod tests;
