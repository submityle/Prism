//! Renderer-independent data models for the Loom Studio editor shell.
//!
//! This crate owns command indexing, keyboard context expressions, keymap
//! conflict detection, and deterministic menu/toolbar/status-bar
//! contribution aggregation. It deliberately does not create widgets, draw
//! pixels, depend on a UI toolkit, or depend on any `bevy_*` crate.

#![forbid(unsafe_code)]

mod command;
mod contribution;
mod keymap;

pub use command::{
    CommandEntry, CommandIndex, CommandIndexError, CommandRegistration, CommandSearchResult,
};
pub use contribution::{
    ContributionError, ContributionRegistration, MenuContribution, MenuItem, ShellContributions,
    StatusContribution, StatusItem, StatusRegion, ToolbarContribution, ToolbarItem,
};
pub use keymap::{
    BindingConflict, ContextExpr, ContextKeySet, ContextParseError, KeyBinding,
    KeyBindingRegistration, KeyCodeError, KeySequence, KeyStroke, Keymap, KeymapError, Modifier,
};
