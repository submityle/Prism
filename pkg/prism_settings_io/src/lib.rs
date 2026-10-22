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
mod pipeline;
mod store;
mod transaction;

pub use document::{CodecError, SettingsDocument};
pub use journal::{
    inspect_journal, reconstruct_transaction, recover_journal, ApplyOp, ApplyOpState,
    CompensationState, EffectState, Journal, JournalError, JournalEvent, JournalInspection,
    JournalOpState, JournalRecord, JournalRecovery, JournalTx, OpAction, Reversibility, TxMode,
    TxOutcome, TxRecoveryDecision,
};
pub use migration::{
    MigrationChain, MigrationError, MigrationOperation, MigrationRegistrationError,
    MigrationReport, ValueConverter,
};
pub use pipeline::{
    recover_settings_transaction, recover_settings_transaction_with_source_rebuild,
    resolved_snapshot_digest, AuthoritativeSourceRebuilder, CandidateBuilder, CandidateError,
    CandidateSnapshot, FileSettingsPersistence, PipelineError, PipelineOutcome,
    SettingsPersistence, SettingsTransaction, SettingsTransactionSpec, SettingsWatcher,
    SnapshotController, SourceSpec, WatchPoll,
};
pub use store::{load_with_backup, write_atomic, write_atomic_rotating, LoadOutcome, StoreError};
pub use transaction::{
    recover_open_transaction, rollback_open_transaction, run_preview_transaction, run_transaction,
    ApplyPlan, ApplyPlanError, CompensateOutcome, OpOutcome, SettingsApplier, SnapshotRecovery,
    TransactionError, TransactionRunOutcome,
};
