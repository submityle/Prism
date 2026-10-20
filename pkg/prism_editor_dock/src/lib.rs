//! Pure-data adapter between editor contributions and the dock model.
//!
//! `prism_editor_core` owns editor semantics, while `prism_ui_dock` owns the
//! renderer-independent dock data model. This crate connects those contracts
//! without constructing widgets, opening files, or depending on an engine.
//!
//! The persistence boundary is explicit: [`EncodedDockSnapshot`] contains two
//! deterministic byte streams, one for workspace layouts and one for retained
//! panel-instance identities. Filesystem stores, autosave policy, and UI
//! rendering remain outside this crate.

#![forbid(unsafe_code)]

mod contribution;
mod instance;
mod persistence;
mod workspace;

pub use contribution::{ContributionError, DockContributionRegistry, DockPanelContribution};
pub use instance::{PanelLifecycleAdapter, PanelLifecycleError, PanelLifecycleEvent};
pub use persistence::{
    decode_snapshot, encode_snapshot, DockPersistenceError, EncodedDockSnapshot,
};
pub use workspace::{
    DockWorkspaceController, DockWorkspaceError, DockWorkspaceSnapshot, RestoreRejection,
    RestoreReport, RestoreSource,
};
