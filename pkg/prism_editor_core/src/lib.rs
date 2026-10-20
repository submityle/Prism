//! Prism editor kernel.
//!
//! The kernel owns editor semantics, not project gameplay semantics. It is
//! deliberately independent from UI, rendering, runtime transport, and any
//! `bevy_*` crate so that the same command stream can drive a desktop editor,
//! headless automation, or a remote runtime adapter.

#![forbid(unsafe_code)]

extern crate alloc;

mod command;
mod contribution;
mod document;
pub mod prefab;
mod selection;
mod service;

pub use command::{
    AffectSet, Command, CommandError, CommandStack, CommandStatus, EditContext, MergeResult,
    Transaction,
};
pub use contribution::{
    CommandDescriptor, ContributionError, ContributionId, ContributionRegistry, EditorModule,
    PanelDescriptor, Registration, RegistrationKind,
};
pub use document::{
    DocumentError, DocumentId, DocumentLayer, DocumentMetadata, DocumentState, LayerId, LayerKind,
};
pub use selection::{Selection, SelectionItem, SelectionKind};
pub use service::{ServiceError, ServiceRegistry};
