//! Concrete Loom Element assembly for Prism Studio.
//!
//! The editor model crates own state and policy. This crate owns the final
//! renderer-independent view tree: it turns command results, panel snapshots,
//! reflection projections, dock contributions, and viewport state into stable
//! [`prism_ui::Element`] subtrees. No renderer or engine ECS is referenced.

#![forbid(unsafe_code)]

mod chrome;
mod command;
mod console;
mod dock;
mod inspector;
mod outliner;
mod support;
mod viewport;
mod workspace;

pub use chrome::{shell_chrome, ShellChromeProps};
pub use command::{command_palette, command_palette_results, CommandPaletteProps};
pub use console::{console_panel, ConsolePanelProps};
pub use dock::{dock_workspace, DockWorkspaceProps};
pub use inspector::{inspector_view, InspectorViewProps};
pub use outliner::{outliner_panel, OutlinerViewProps};
pub use support::MAX_VISIBLE_ROWS;
pub use viewport::{viewport_host, ViewportHostProps};
pub use workspace::{editor_workspace, EditorWorkspaceProps};

#[cfg(test)]
mod tests;
