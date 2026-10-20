//! Versioned, deterministic persistence primitives for [`prism_settings`].
//!
//! The crate owns the file representation, not schema resolution. A document
//! contains persisted candidates and preserves keys which the current registry
//! does not know about. This makes a newer settings file safe to load and write
//! back from an older binary without silently deleting future data.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod document;
mod journal;
mod migration;
mod store;
mod transaction;

pub use document::{CodecError, SettingsDocument};
pub use journal::{
    reconstruct_transaction, recover_journal, ApplyOp, ApplyOpState, CompensationState,
    EffectState, Journal, JournalError, JournalEvent, JournalOpState, JournalRecord,
    JournalRecovery, JournalTx, OpAction, Reversibility, TxMode, TxOutcome, TxRecoveryDecision,
};
pub use migration::{
    MigrationChain, MigrationError, MigrationOperation, MigrationRegistrationError,
    MigrationReport, ValueConverter,
};
pub use store::{load_with_backup, write_atomic, write_atomic_rotating, LoadOutcome, StoreError};
pub use transaction::{
    run_transaction, ApplyPlan, ApplyPlanError, CompensateOutcome, OpOutcome, SettingsApplier,
    TransactionError, TransactionRunOutcome,
};
