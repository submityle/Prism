//! Versioned, deterministic persistence primitives for [`prism_settings`].
//!
//! The crate owns the file representation, not schema resolution. A document
//! contains persisted candidates and preserves keys which the current registry
//! does not know about. This makes a newer settings file safe to load and write
//! back from an older binary without silently deleting future data.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod document;
mod migration;
mod store;

pub use document::{CodecError, SettingsDocument};
pub use migration::{
    MigrationChain, MigrationError, MigrationOperation, MigrationRegistrationError,
    MigrationReport, ValueConverter,
};
pub use store::{load_with_backup, write_atomic, write_atomic_rotating, LoadOutcome, StoreError};
