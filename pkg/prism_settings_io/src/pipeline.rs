//! Canonical settings I/O pipeline.
//!
//! This module is the `std` boundary between a persisted document and the
//! no-`std` settings core. It intentionally keeps file watching, migration,
//! candidate resolution, persistence, and side-effect transactions in one
//! explicit path:
//!
//! `watch -> candidate -> snapshot transaction -> journaled effects`.

use crate::{
    inspect_journal, load_with_backup, rollback_open_transaction, run_preview_transaction,
    run_transaction, ApplyPlan, CodecError, Journal, JournalError, JournalEvent, MigrationChain,
    MigrationError, SettingsDocument, StoreError, TransactionError, TransactionRunOutcome, TxMode,
    TxOutcome,
};
use prism_settings::{
    Domain, LayerKind, LayerSource, Resolver, SettingId, SettingsError, SettingsRegistry, Snapshot,
    TrustLevel, Value,
};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fmt, fs,
    io::Write,
    path::{Path, PathBuf},
};

/// Stable identity and trust metadata for a watched settings source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceSpec {
    /// Domain layer populated by the source.
    pub layer: LayerKind,
    /// Trust assigned to values loaded from the source.
    pub trust: TrustLevel,
    /// Stable source identity used for replacement and tie-breaking.
    pub source_id: String,
    /// Order among sources occupying the same layer.
    pub order: i32,
}

impl SourceSpec {
    /// Creates a source specification with order zero.
    pub fn new(layer: LayerKind, trust: TrustLevel, source_id: impl Into<String>) -> Self {
        Self {
            layer,
            trust,
            source_id: source_id.into(),
            order: 0,
        }
    }
}

/// A decoded and resolved candidate produced by the canonical watcher path.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateSnapshot {
    /// Versioned document, including preserved unknown keys.
    pub document: SettingsDocument,
    /// Source generated from the document's known candidates.
    pub source: LayerSource,
    /// Resolved immutable snapshot to install atomically.
    pub snapshot: Snapshot,
    /// Digest of the canonical persisted document.
    pub document_digest: u64,
    /// Digest of the resolved candidate snapshot.
    pub snapshot_digest: u64,
    /// Monotonic candidate generation assigned by the watcher.
    pub generation: u64,
    /// Whether the primary file was unavailable and a backup supplied this
    /// candidate.
    pub recovered_from_backup: bool,
}

/// Errors while constructing a candidate from a document and source set.
#[derive(Debug)]
pub enum CandidateError {
    /// Document encoding failed while calculating its digest.
    Codec(CodecError),
    /// The resolver rejected one or more candidates.
    Resolve(Vec<SettingsError>),
    /// The generation counter overflowed.
    GenerationOverflow,
}

impl fmt::Display for CandidateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(error) => write!(formatter, "candidate document failed: {error}"),
            Self::Resolve(errors) => write!(formatter, "candidate snapshot rejected: {errors:?}"),
            Self::GenerationOverflow => formatter.write_str("candidate generation overflowed"),
        }
    }
}

impl Error for CandidateError {}

impl From<CodecError> for CandidateError {
    fn from(value: CodecError) -> Self {
        Self::Codec(value)
    }
}

/// Result of one deterministic watcher poll.
#[derive(Clone, Debug, PartialEq)]
pub enum WatchPoll {
    /// Neither the primary nor backup observation changed.
    Unchanged {
        /// Digest of the observed file pair.
        observed_digest: u64,
    },
    /// A new candidate is ready for transaction preflight.
    Changed {
        /// Candidate snapshot resolved from the changed source.
        candidate: CandidateSnapshot,
        /// Digest of the observed file pair, including invalid primary bytes.
        observed_digest: u64,
        /// Number of migration steps applied to the document.
        migrated_steps: usize,
    },
}

/// Polling watcher for one canonical settings source.
///
/// The watcher is deliberately content based. Platform-specific notification
/// adapters can call [`Self::poll`] after receiving an event, while platforms
/// without a watcher use the same method at a low frequency. A failed decode
/// never advances the observed digest, so a later repair is retried.
pub struct SettingsWatcher<'a> {
    path: PathBuf,
    backup: Option<PathBuf>,
    registry: &'a SettingsRegistry,
    migration: MigrationChain,
    source: SourceSpec,
    domain: Domain,
    /// Digest whose candidate the caller explicitly acknowledged as handled.
    last_observed_digest: Option<u64>,
    /// Most recently constructed candidate awaiting transaction disposition.
    pending_digest: Option<u64>,
    next_generation: u64,
}

impl<'a> SettingsWatcher<'a> {
    /// Creates a watcher for one primary/backup document pair.
    pub fn new(
        path: impl AsRef<Path>,
        backup: Option<impl AsRef<Path>>,
        registry: &'a SettingsRegistry,
        migration: MigrationChain,
        domain: Domain,
        source: SourceSpec,
    ) -> Self {
        Self {
            path: path.as_ref().to_owned(),
            backup: backup.map(|path| path.as_ref().to_owned()),
            registry,
            migration,
            source,
            domain,
            last_observed_digest: None,
            pending_digest: None,
            next_generation: 0,
        }
    }

    /// Returns the primary path watched by this instance.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the last successfully observed raw file-pair digest.
    pub fn last_observed_digest(&self) -> Option<u64> {
        self.last_observed_digest
    }

    /// Acknowledges a candidate after its transaction has reached a terminal
    /// disposition (committed, rejected, or explicitly deferred by policy).
    ///
    /// Returns `false` if `observed_digest` is not the current pending
    /// candidate or if the files changed since that candidate was built. A
    /// stale acknowledgment can therefore never suppress a newer edit.
    pub fn acknowledge(&mut self, observed_digest: u64) -> Result<bool, PipelineError> {
        if self.pending_digest != Some(observed_digest) {
            return Ok(false);
        }
        let current = observe_files(&self.path, self.backup.as_deref())?;
        if current != observed_digest {
            self.pending_digest = None;
            return Ok(false);
        }
        self.last_observed_digest = Some(observed_digest);
        self.pending_digest = None;
        Ok(true)
    }

    /// Polls the source and resolves a candidate against `base_sources`.
    pub fn poll(&mut self, base_sources: &[LayerSource]) -> Result<WatchPoll, PipelineError> {
        let observed_digest = observe_files(&self.path, self.backup.as_deref())?;
        if self.last_observed_digest == Some(observed_digest) {
            return Ok(WatchPoll::Unchanged { observed_digest });
        }

        let loaded = load_with_backup(&self.path, self.backup.as_deref(), Some(self.registry))?;
        let mut document = loaded.document;
        let report = self.migration.migrate(&mut document)?;
        self.next_generation = self
            .next_generation
            .checked_add(1)
            .ok_or(CandidateError::GenerationOverflow)?;
        let candidate = CandidateBuilder::new(self.registry, self.domain, self.source.clone())
            .build(
                document,
                base_sources,
                self.next_generation,
                loaded.recovered_from_backup,
            )?;
        self.pending_digest = Some(observed_digest);
        Ok(WatchPoll::Changed {
            candidate,
            observed_digest,
            migrated_steps: report.applied_steps,
        })
    }
}

/// Converts a document into a resolved candidate without touching the filesystem.
pub struct CandidateBuilder<'a> {
    registry: &'a SettingsRegistry,
    domain: Domain,
    source: SourceSpec,
}

impl<'a> CandidateBuilder<'a> {
    /// Creates a builder for one domain and source identity.
    pub fn new(registry: &'a SettingsRegistry, domain: Domain, source: SourceSpec) -> Self {
        Self {
            registry,
            domain,
            source,
        }
    }

    /// Resolves `document` after replacing the builder's source in
    /// `base_sources`. Unknown keys remain in the document and are not fed to
    /// the core resolver.
    pub fn build(
        &self,
        mut document: SettingsDocument,
        base_sources: &[LayerSource],
        generation: u64,
        recovered_from_backup: bool,
    ) -> Result<CandidateSnapshot, CandidateError> {
        document.bind_registry(self.registry);
        let mut source = LayerSource::new(
            self.source.layer,
            self.source.trust,
            self.source.source_id.clone(),
        );
        source.order = self.source.order;
        for (key, value) in document.candidates() {
            source.set(key.clone(), value.clone());
        }

        let mut sources = Vec::with_capacity(base_sources.len() + 1);
        for base in base_sources {
            if base.layer == source.layer && base.source_id == source.source_id {
                continue;
            }
            sources.push(base.clone());
        }
        sources.push(source.clone());
        let snapshot = Resolver::new(self.registry)
            .resolve(self.domain, &sources, generation)
            .map_err(CandidateError::Resolve)?;
        let document_digest = document_digest(&document)?;
        let snapshot_digest = resolved_snapshot_digest(self.registry, &snapshot);
        Ok(CandidateSnapshot {
            document,
            source,
            snapshot,
            document_digest,
            snapshot_digest,
            generation,
            recovered_from_backup,
        })
    }
}

/// Persistence boundary used by the transaction coordinator.
pub trait SettingsPersistence {
    /// Persists one candidate document and returns its manifest/digest.
    fn persist(&mut self, document: &SettingsDocument, generation: u64) -> Result<u64, String>;
    /// Restores the image observed before the most recent persistence attempt.
    /// Implementations which cannot prove rollback must return an error.
    fn rollback(&mut self) -> Result<(), String> {
        Err("persistence rollback is not supported".to_owned())
    }
    /// Restores the pre-transaction image after a process restart.
    fn recover_after_restart(&mut self) -> Result<(), String> {
        self.rollback()
    }
    /// Releases transaction-local recovery metadata after a durable terminal
    /// outcome has been written.
    fn finalize(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// Atomic file persistence implementation for one settings document.
pub struct FileSettingsPersistence {
    path: PathBuf,
    backup: Option<PathBuf>,
    max_backups: usize,
    recovery_path: PathBuf,
    previous_contents: Option<Option<Vec<u8>>>,
}

impl FileSettingsPersistence {
    /// Creates an atomic persistence target.
    pub fn new(
        path: impl AsRef<Path>,
        backup: Option<impl AsRef<Path>>,
        max_backups: usize,
    ) -> Self {
        Self {
            path: path.as_ref().to_owned(),
            backup: backup.map(|path| path.as_ref().to_owned()),
            max_backups,
            recovery_path: recovery_path(path.as_ref()),
            previous_contents: None,
        }
    }

    /// Returns the primary persistence path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl SettingsPersistence for FileSettingsPersistence {
    fn persist(&mut self, document: &SettingsDocument, _generation: u64) -> Result<u64, String> {
        let previous = match fs::read(&self.path) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.to_string()),
        };
        write_recovery_image(&self.recovery_path, previous.as_deref())
            .map_err(|error| error.to_string())?;
        self.previous_contents = Some(previous);
        crate::write_atomic_rotating(&self.path, document, self.backup.as_ref(), self.max_backups)
            .map_err(|error| error.to_string())?;
        document_digest(document).map_err(|error| error.to_string())
    }

    fn rollback(&mut self) -> Result<(), String> {
        let previous = match self.previous_contents.as_ref() {
            Some(previous) => previous.clone(),
            None => read_recovery_image(&self.recovery_path).map_err(|error| error.to_string())?,
        };
        restore_file_contents(&self.path, previous.as_deref())
            .map_err(|error| error.to_string())?;
        self.previous_contents = None;
        clear_recovery_image(&self.recovery_path).map_err(|error| error.to_string())
    }

    fn recover_after_restart(&mut self) -> Result<(), String> {
        let previous =
            read_recovery_image(&self.recovery_path).map_err(|error| error.to_string())?;
        restore_file_contents(&self.path, previous.as_deref())
            .map_err(|error| error.to_string())?;
        clear_recovery_image(&self.recovery_path).map_err(|error| error.to_string())
    }

    fn finalize(&mut self) -> Result<(), String> {
        clear_recovery_image(&self.recovery_path).map_err(|error| error.to_string())
    }
}

/// In-memory snapshot installation and durable-generation restoration boundary.
pub trait SnapshotController {
    /// Installs a complete candidate, including its source identity and
    /// provenance. Implementations that own source tables should override
    /// this method so source replacement and snapshot publication are atomic.
    fn install_candidate(&mut self, candidate: &CandidateSnapshot) -> Result<(), String> {
        self.install_snapshot(candidate.snapshot.clone())
    }
    /// Restores the complete pre-transaction runtime state.
    ///
    /// The default preserves compatibility for snapshot-only hosts. Hosts
    /// with source tables should override this and restore their matching
    /// source/configuration checkpoint as well.
    fn restore_before(&mut self, snapshot: &Snapshot) -> Result<(), String> {
        self.install_snapshot(snapshot.clone())
    }
    /// Atomically installs an immutable candidate snapshot.
    fn install_snapshot(&mut self, snapshot: Snapshot) -> Result<(), String>;
    /// Restores a durable generation during process-start recovery.
    fn restore_generation(&mut self, generation: u64, hash: u64) -> Result<(), String>;
    /// Atomically installs host-rebuilt authoritative sources and their
    /// verified resolved snapshot. The default fails closed for hosts that do
    /// not support process-start source reconstruction.
    fn install_authoritative_sources(
        &mut self,
        _sources: Vec<LayerSource>,
        _generation: u64,
        _hash: u64,
    ) -> Result<(), String> {
        Err("authoritative source installation is not supported".to_owned())
    }
}

/// Loads the authoritative source table after durable file recovery and before
/// journal effects are reconciled.
pub trait AuthoritativeSourceRebuilder {
    /// Loads all authoritative sources for the requested durable generation.
    /// The caller resolves them and verifies the requested snapshot digest
    /// before publishing them to the runtime.
    fn load_authoritative_sources(
        &mut self,
        generation: u64,
        hash: u64,
    ) -> Result<Vec<LayerSource>, String>;
}

/// Result of the complete settings transaction pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineOutcome {
    /// State, persistence, and all effects completed.
    Committed,
    /// State/persistence completed but an external effect degraded.
    Degraded,
    /// Candidate state and reversible effects were restored.
    RolledBack,
    /// An irreversible effect crossed its barrier and requires restart.
    NeedsRestart,
    /// A preview is installed and awaits explicit confirmation.
    PendingConfirmation,
    /// Recovery must be completed by a caller with stronger knowledge.
    RecoveryRequired,
}

/// Errors returned by the complete watcher/transaction pipeline.
#[derive(Debug)]
pub enum PipelineError {
    /// Store/codec I/O failed.
    Store(StoreError),
    /// Schema migration failed.
    Migration(MigrationError),
    /// Candidate resolution failed.
    Candidate(CandidateError),
    /// Journal I/O or validation failed.
    Journal(JournalError),
    /// Operation execution failed before a terminal outcome was recorded.
    Transaction(TransactionError),
    /// Snapshot installation or restoration failed.
    Snapshot(String),
    /// Candidate persistence failed.
    Persistence(String),
    /// The requested preview/recovery state is not valid.
    InvalidState(String),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => write!(formatter, "settings store failed: {error}"),
            Self::Migration(error) => write!(formatter, "settings migration failed: {error}"),
            Self::Candidate(error) => write!(formatter, "settings candidate failed: {error}"),
            Self::Journal(error) => write!(formatter, "settings journal failed: {error}"),
            Self::Transaction(error) => write!(formatter, "settings transaction failed: {error}"),
            Self::Snapshot(error) => write!(formatter, "settings snapshot failed: {error}"),
            Self::Persistence(error) => write!(formatter, "settings persistence failed: {error}"),
            Self::InvalidState(error) => {
                write!(formatter, "invalid settings pipeline state: {error}")
            }
        }
    }
}

impl Error for PipelineError {}

impl From<StoreError> for PipelineError {
    fn from(value: StoreError) -> Self {
        Self::Store(value)
    }
}
impl From<MigrationError> for PipelineError {
    fn from(value: MigrationError) -> Self {
        Self::Migration(value)
    }
}
impl From<CandidateError> for PipelineError {
    fn from(value: CandidateError) -> Self {
        Self::Candidate(value)
    }
}
impl From<JournalError> for PipelineError {
    fn from(value: JournalError) -> Self {
        Self::Journal(value)
    }
}
impl From<TransactionError> for PipelineError {
    fn from(value: TransactionError) -> Self {
        Self::Transaction(value)
    }
}

/// A fully prepared candidate transaction. The object is retained while a
/// preview waits for confirmation.
pub struct SettingsTransaction {
    tx_id: String,
    journal_path: PathBuf,
    mode: TxMode,
    before_hash: u64,
    before_snapshot: Snapshot,
    candidate: CandidateSnapshot,
    plan: ApplyPlan,
    snapshot_gen: u64,
    last_confirmed: Option<u64>,
}

/// Complete immutable input for one settings transaction.
#[derive(Debug)]
pub struct SettingsTransactionSpec {
    /// Stable transaction identity.
    pub tx_id: String,
    /// Durable journal path.
    pub journal_path: PathBuf,
    /// Direct or preview execution mode.
    pub mode: TxMode,
    /// Digest of the snapshot before the transaction.
    pub before_hash: u64,
    /// Snapshot restored when the transaction rolls back.
    pub before_snapshot: Snapshot,
    /// Resolved candidate to install and persist.
    pub candidate: CandidateSnapshot,
    /// Ordered side-effect plan.
    pub plan: ApplyPlan,
    /// Generation associated with the pre-transaction snapshot.
    pub snapshot_gen: u64,
    /// Last confirmed generation, if the caller has one.
    pub last_confirmed: Option<u64>,
}

impl SettingsTransaction {
    /// Creates a transaction from an already resolved candidate and apply plan.
    pub fn new(spec: SettingsTransactionSpec) -> Self {
        Self {
            tx_id: spec.tx_id,
            journal_path: spec.journal_path,
            mode: spec.mode,
            before_hash: spec.before_hash,
            before_snapshot: spec.before_snapshot,
            candidate: spec.candidate,
            plan: spec.plan,
            snapshot_gen: spec.snapshot_gen,
            last_confirmed: spec.last_confirmed,
        }
    }

    /// Returns the stable transaction identity used by the durable journal.
    #[must_use]
    pub fn tx_id(&self) -> &str {
        &self.tx_id
    }

    /// Returns whether this transaction is direct or confirmation-gated.
    #[must_use]
    pub const fn mode(&self) -> TxMode {
        self.mode
    }

    /// Returns the candidate held by this transaction.
    pub fn candidate(&self) -> &CandidateSnapshot {
        &self.candidate
    }

    /// Installs, persists, applies, and finalizes the candidate transaction.
    pub fn execute(
        &self,
        snapshots: &mut dyn SnapshotController,
        persistence: &mut dyn SettingsPersistence,
        appliers: &mut [&mut dyn crate::SettingsApplier],
    ) -> Result<PipelineOutcome, PipelineError> {
        let mut journal = open_new_journal(&self.journal_path, &self.tx_id)?;
        journal.append(JournalEvent::TxBegin {
            mode: self.mode,
            hash_before: self.before_hash,
            hash_after: self.candidate.snapshot_digest,
            plan_digest: plan_digest(&self.plan),
            op_count: u16::try_from(self.plan.ops.len())
                .map_err(|_| PipelineError::InvalidState("apply plan exceeds u16".to_owned()))?,
            snapshot_gen: self.snapshot_gen,
            last_confirmed: self.last_confirmed,
        })?;
        snapshots
            .install_candidate(&self.candidate)
            .map_err(PipelineError::Snapshot)?;
        if let Err(error) = journal.append(JournalEvent::SnapshotCommitted {
            snapshot_gen: self.candidate.generation,
        }) {
            let restore = snapshots.restore_before(&self.before_snapshot);
            if let Err(restore_error) = restore {
                return Err(PipelineError::InvalidState(format!(
                    "snapshot commit journal failed ({error}) and runtime rollback failed ({restore_error})"
                )));
            }
            return Err(error.into());
        }

        match self.mode {
            TxMode::Direct => {
                if let Err(error) = persist_with_journal(&mut journal, persistence, &self.candidate)
                {
                    return rollback_persistence_failure(
                        &mut journal,
                        snapshots,
                        persistence,
                        &self.before_snapshot,
                        error,
                    );
                }
                let outcome = run_transaction(&mut journal, &self.plan, appliers)?;
                finish_direct_transaction(
                    &mut journal,
                    snapshots,
                    persistence,
                    &self.before_snapshot,
                    outcome,
                )
            }
            TxMode::Preview => {
                let outcome = run_preview_transaction(&mut journal, &self.plan, appliers)?;
                if outcome == TransactionRunOutcome::RolledBack {
                    restore_snapshot(snapshots, &self.before_snapshot)?;
                }
                Ok(map_outcome(outcome))
            }
        }
    }

    /// Confirms an installed preview, persisting the candidate and closing its journal.
    pub fn confirm_preview(
        &self,
        snapshots: &mut dyn SnapshotController,
        persistence: &mut dyn SettingsPersistence,
        appliers: &mut [&mut dyn crate::SettingsApplier],
    ) -> Result<PipelineOutcome, PipelineError> {
        if self.mode != TxMode::Preview {
            return Err(PipelineError::InvalidState(
                "only preview transactions can be confirmed".to_owned(),
            ));
        }
        let mut journal = open_existing_journal(&self.journal_path, &self.tx_id)?;
        let tx =
            crate::reconstruct_transaction(&crate::recover_journal(&self.journal_path)?.records)?;
        if tx.ended != Some(TxOutcome::PendingConfirmation) {
            return Err(PipelineError::InvalidState(
                "preview is not awaiting confirmation".to_owned(),
            ));
        }
        if let Err(error) = persist_with_journal(&mut journal, persistence, &self.candidate) {
            let persistence_rollback = persistence.rollback();
            let rollback = rollback_open_transaction(
                &mut journal,
                appliers,
                &mut BeforeRestore {
                    snapshots,
                    snapshot: self.before_snapshot.clone(),
                },
            );
            if persistence_rollback.is_err() || rollback.is_err() {
                return Err(PipelineError::InvalidState(
                    "preview persistence and rollback both failed".to_owned(),
                ));
            }
            persistence.finalize().map_err(PipelineError::Persistence)?;
            return Err(error);
        }
        journal.append(JournalEvent::TxEnd {
            outcome: TxOutcome::Committed,
        })?;
        if let Err(error) = persistence.finalize() {
            let _ = journal.append(JournalEvent::TxEnd {
                outcome: TxOutcome::RecoveryRequired,
            });
            return Err(PipelineError::Persistence(error));
        }
        Ok(PipelineOutcome::Committed)
    }

    /// Rejects a preview or rolls it back after its confirmation deadline.
    pub fn rollback_preview(
        &self,
        snapshots: &mut dyn SnapshotController,
        appliers: &mut [&mut dyn crate::SettingsApplier],
    ) -> Result<PipelineOutcome, PipelineError> {
        if self.mode != TxMode::Preview {
            return Err(PipelineError::InvalidState(
                "only preview transactions can be rolled back".to_owned(),
            ));
        }
        let mut journal = open_existing_journal(&self.journal_path, &self.tx_id)?;
        let outcome = rollback_open_transaction(
            &mut journal,
            appliers,
            &mut BeforeRestore {
                snapshots,
                snapshot: self.before_snapshot.clone(),
            },
        )?;
        Ok(map_outcome(outcome))
    }
}

/// Recovers one existing journal after process restart.
pub fn recover_settings_transaction(
    journal_path: impl AsRef<Path>,
    snapshots: &mut dyn SnapshotController,
    persistence: &mut dyn SettingsPersistence,
    appliers: &mut [&mut dyn crate::SettingsApplier],
) -> Result<Option<PipelineOutcome>, PipelineError> {
    let path = journal_path.as_ref();
    let inspection = inspect_journal(path)?;
    if inspection.records.is_empty() && inspection.invalid_tail {
        return Err(PipelineError::InvalidState(
            "settings journal contains no valid records".to_owned(),
        ));
    }
    let recovery = crate::recover_journal(path)?;
    let Some(first) = recovery.records.first() else {
        return Ok(None);
    };
    let tx = crate::reconstruct_transaction(&recovery.records)?;
    if tx.persistence_in_flight
        || (tx.persisted && tx.ended.is_none() && !tx.all_operations_applied())
    {
        persistence
            .recover_after_restart()
            .map_err(PipelineError::Persistence)?;
    }
    let mut journal = Journal::open(path, first.tx_id.clone())?;
    let mut recovery = ControllerRestore { snapshots };
    let outcome = crate::recover_open_transaction(&mut journal, &mut recovery, appliers)?;
    if outcome != TransactionRunOutcome::RecoveryRequired {
        persistence.finalize().map_err(PipelineError::Persistence)?;
    }
    Ok(Some(map_outcome(outcome)))
}

/// Recovers a transaction with an explicit authoritative-source rebuild step.
///
/// This is the process-start entry point for hosts whose source table is
/// loaded from files, platform policy, or another durable authority. It first
/// restores any durable file preimage, then invokes `rebuilder`, and only then
/// allows journal effect recovery to run.
pub fn recover_settings_transaction_with_source_rebuild(
    journal_path: impl AsRef<Path>,
    snapshots: &mut dyn SnapshotController,
    rebuilder: &mut dyn AuthoritativeSourceRebuilder,
    persistence: &mut dyn SettingsPersistence,
    appliers: &mut [&mut dyn crate::SettingsApplier],
) -> Result<Option<PipelineOutcome>, PipelineError> {
    let path = journal_path.as_ref();
    let inspection = inspect_journal(path)?;
    if inspection.records.is_empty() {
        return if inspection.invalid_tail {
            Err(PipelineError::InvalidState(
                "settings journal contains no valid records".to_owned(),
            ))
        } else {
            Ok(None)
        };
    }
    let recovery = crate::recover_journal(path)?;
    let Some(first) = recovery.records.first() else {
        return Ok(None);
    };
    let tx = crate::reconstruct_transaction(&recovery.records)?;
    if tx.ended.is_none() || tx.ended == Some(crate::TxOutcome::PendingConfirmation) {
        let Some(crate::JournalEvent::TxBegin {
            snapshot_gen: before_generation,
            hash_before,
            hash_after,
            ..
        }) = tx.begin.as_ref()
        else {
            return Err(PipelineError::InvalidState(
                "open settings journal has no TxBegin".to_owned(),
            ));
        };
        let restoring_preimage =
            tx.persistence_in_flight || (tx.persisted && !tx.all_operations_applied());
        if restoring_preimage {
            persistence
                .recover_after_restart()
                .map_err(PipelineError::Persistence)?;
        }
        let (generation, hash) = if restoring_preimage {
            (*before_generation, *hash_before)
        } else {
            match tx.recovery_decision() {
                crate::TxRecoveryDecision::Commit | crate::TxRecoveryDecision::NeedsRestart => {
                    let generation = recovery
                        .records
                        .iter()
                        .find_map(|record| match &record.event {
                            crate::JournalEvent::SnapshotCommitted { snapshot_gen } => {
                                Some(*snapshot_gen)
                            }
                            _ => None,
                        })
                        .ok_or_else(|| {
                            PipelineError::InvalidState(
                                "forward recovery has no candidate snapshot generation".to_owned(),
                            )
                        })?;
                    (generation, *hash_after)
                }
                crate::TxRecoveryDecision::RecoveryRequired if tx.persisted => {
                    let generation = recovery
                        .records
                        .iter()
                        .find_map(|record| match &record.event {
                            crate::JournalEvent::SnapshotCommitted { snapshot_gen } => {
                                Some(*snapshot_gen)
                            }
                            _ => None,
                        })
                        .ok_or_else(|| {
                            PipelineError::InvalidState(
                                "forward recovery has no candidate snapshot generation".to_owned(),
                            )
                        })?;
                    (generation, *hash_after)
                }
                _ => (*before_generation, *hash_before),
            }
        };
        let sources = rebuilder
            .load_authoritative_sources(generation, hash)
            .map_err(PipelineError::Snapshot)?;
        snapshots
            .install_authoritative_sources(sources, generation, hash)
            .map_err(PipelineError::Snapshot)?;
    }
    let mut journal = Journal::open(path, first.tx_id.clone())?;
    let mut recovery = ControllerRestore { snapshots };
    let outcome = crate::recover_open_transaction(&mut journal, &mut recovery, appliers)?;
    if outcome != TransactionRunOutcome::RecoveryRequired {
        persistence.finalize().map_err(PipelineError::Persistence)?;
    }
    Ok(Some(map_outcome(outcome)))
}

struct BeforeRestore<'a> {
    snapshots: &'a mut dyn SnapshotController,
    snapshot: Snapshot,
}

impl crate::SnapshotRecovery for BeforeRestore<'_> {
    fn restore(&mut self, _generation: u64, _hash: u64) -> Result<(), String> {
        self.snapshots.restore_before(&self.snapshot)
    }
}

struct ControllerRestore<'a> {
    snapshots: &'a mut dyn SnapshotController,
}

impl crate::SnapshotRecovery for ControllerRestore<'_> {
    fn restore(&mut self, generation: u64, hash: u64) -> Result<(), String> {
        self.snapshots.restore_generation(generation, hash)
    }
}

fn persist_with_journal(
    journal: &mut Journal,
    persistence: &mut dyn SettingsPersistence,
    candidate: &CandidateSnapshot,
) -> Result<(), PipelineError> {
    journal.append(JournalEvent::PersistBegin {
        generation: candidate.generation,
        manifest_hash: candidate.document_digest,
    })?;
    let manifest_hash = persistence
        .persist(&candidate.document, candidate.generation)
        .map_err(PipelineError::Persistence)?;
    journal.append(JournalEvent::PersistEnd {
        generation: candidate.generation,
        manifest_hash,
    })?;
    Ok(())
}

fn restore_snapshot(
    snapshots: &mut dyn SnapshotController,
    before: &Snapshot,
) -> Result<(), PipelineError> {
    snapshots
        .restore_before(before)
        .map_err(PipelineError::Snapshot)
}

fn finish_direct_transaction(
    journal: &mut Journal,
    snapshots: &mut dyn SnapshotController,
    persistence: &mut dyn SettingsPersistence,
    before: &Snapshot,
    outcome: TransactionRunOutcome,
) -> Result<PipelineOutcome, PipelineError> {
    match outcome {
        TransactionRunOutcome::RolledBack => {
            let persistence_error = persistence.rollback().err();
            let snapshot_error = restore_snapshot(snapshots, before).err();
            let cleanup_error = if persistence_error.is_none() && snapshot_error.is_none() {
                persistence.finalize().err()
            } else {
                None
            };
            if persistence_error.is_some() || snapshot_error.is_some() || cleanup_error.is_some() {
                let _ = journal.append(JournalEvent::TxEnd {
                    outcome: TxOutcome::RecoveryRequired,
                });
                return Err(PipelineError::InvalidState(format!(
                    "direct rollback failed: persistence={persistence_error:?}, snapshot={snapshot_error:?}, cleanup={cleanup_error:?}"
                )));
            }
        }
        TransactionRunOutcome::Committed
        | TransactionRunOutcome::Degraded
        | TransactionRunOutcome::NeedsRestart => {
            if let Err(error) = persistence.finalize() {
                let _ = journal.append(JournalEvent::TxEnd {
                    outcome: TxOutcome::RecoveryRequired,
                });
                return Err(PipelineError::Persistence(error));
            }
        }
        TransactionRunOutcome::RecoveryRequired | TransactionRunOutcome::PendingConfirmation => {}
    }
    Ok(map_outcome(outcome))
}

fn rollback_persistence_failure(
    journal: &mut Journal,
    snapshots: &mut dyn SnapshotController,
    persistence: &mut dyn SettingsPersistence,
    before: &Snapshot,
    original_error: PipelineError,
) -> Result<PipelineOutcome, PipelineError> {
    let persistence_error = persistence.rollback().err();
    let snapshot_error = snapshots.restore_before(before).err();
    let cleanup_error = if persistence_error.is_none() && snapshot_error.is_none() {
        persistence.finalize().err()
    } else {
        None
    };
    let outcome =
        if persistence_error.is_none() && snapshot_error.is_none() && cleanup_error.is_none() {
            TxOutcome::RolledBack
        } else {
            TxOutcome::RecoveryRequired
        };
    if let Err(journal_error) = journal.append(JournalEvent::TxEnd { outcome }) {
        return Err(PipelineError::InvalidState(format!(
            "persistence rollback is undecidable; terminal journal write failed: {journal_error}"
        )));
    }
    if outcome == TxOutcome::RecoveryRequired {
        return Err(PipelineError::InvalidState(format!(
            "persistence rollback requires recovery; persistence={:?}, snapshot={:?}, cleanup={:?}",
            persistence_error, snapshot_error, cleanup_error
        )));
    }
    Err(original_error)
}

fn restore_file_contents(path: &Path, contents: Option<&[u8]>) -> Result<(), std::io::Error> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent)?;
    }
    match contents {
        Some(contents) => {
            let mut temporary = path.as_os_str().to_owned();
            temporary.push(".rollback.tmp");
            let temporary = PathBuf::from(temporary);
            let result = (|| {
                let mut file = fs::File::create(&temporary)?;
                file.write_all(contents)?;
                file.sync_all()?;
                fs::rename(&temporary, path)?;
                if let Some(parent) = parent {
                    fs::File::open(parent)?.sync_all()?;
                }
                Ok(())
            })();
            if result.is_err() {
                let _ = fs::remove_file(&temporary);
            }
            result
        }
        None => {
            match fs::remove_file(path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
            if let Some(parent) = parent {
                fs::File::open(parent)?.sync_all()?;
            }
            Ok(())
        }
    }
}

fn recovery_path(path: &Path) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(".prism-rollback");
    PathBuf::from(value)
}

fn write_recovery_image(path: &Path, contents: Option<&[u8]>) -> Result<(), std::io::Error> {
    let mut encoded = Vec::with_capacity(contents.map_or(1, |bytes| bytes.len() + 1));
    encoded.push(u8::from(contents.is_some()));
    if let Some(contents) = contents {
        encoded.extend_from_slice(contents);
    }
    write_bytes_atomic(path, &encoded)
}

fn read_recovery_image(path: &Path) -> Result<Option<Vec<u8>>, std::io::Error> {
    let bytes = fs::read(path)?;
    match bytes.split_first() {
        Some((&0, [])) => Ok(None),
        Some((&1, rest)) => Ok(Some(rest.to_vec())),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid settings rollback image",
        )),
    }
}

fn clear_recovery_image(path: &Path) -> Result<(), std::io::Error> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    let result = (|| {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::File::open(parent)?.sync_all()?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn open_new_journal(path: &Path, tx_id: &str) -> Result<Journal, PipelineError> {
    let recovery = crate::recover_journal(path)?;
    if !recovery.records.is_empty() {
        return Err(PipelineError::InvalidState(
            "journal path already contains a transaction".to_owned(),
        ));
    }
    Ok(Journal::open(path, tx_id)?)
}

fn open_existing_journal(path: &Path, tx_id: &str) -> Result<Journal, PipelineError> {
    Ok(Journal::open(path, tx_id)?)
}

fn map_outcome(outcome: TransactionRunOutcome) -> PipelineOutcome {
    match outcome {
        TransactionRunOutcome::Committed => PipelineOutcome::Committed,
        TransactionRunOutcome::Degraded => PipelineOutcome::Degraded,
        TransactionRunOutcome::RolledBack => PipelineOutcome::RolledBack,
        TransactionRunOutcome::NeedsRestart => PipelineOutcome::NeedsRestart,
        TransactionRunOutcome::RecoveryRequired => PipelineOutcome::RecoveryRequired,
        TransactionRunOutcome::PendingConfirmation => PipelineOutcome::PendingConfirmation,
    }
}

fn observe_files(primary: &Path, backup: Option<&Path>) -> Result<u64, PipelineError> {
    let primary_bytes = read_optional(primary)?;
    let backup_bytes = match backup {
        Some(path) => read_optional(path)?,
        None => None,
    };
    let mut hasher = Sha256::new();
    hash_optional_bytes(&mut hasher, primary_bytes.as_deref());
    hash_optional_bytes(&mut hasher, backup_bytes.as_deref());
    Ok(u64::from_le_bytes(
        hasher.finalize()[..8]
            .try_into()
            .expect("sha256 has at least 8 bytes"),
    ))
}

fn read_optional(path: &Path) -> Result<Option<Vec<u8>>, PipelineError> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(PipelineError::Store(StoreError::Io {
            path: path.to_owned(),
            source,
        })),
    }
}

fn hash_optional_bytes(hasher: &mut Sha256, bytes: Option<&[u8]>) {
    match bytes {
        Some(bytes) => {
            hasher.update([1]);
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        }
        None => hasher.update([0]),
    }
}

fn document_digest(document: &SettingsDocument) -> Result<u64, CodecError> {
    digest_bytes(document.encode()?.as_bytes())
}

fn digest_bytes(bytes: &[u8]) -> Result<u64, CodecError> {
    let digest = Sha256::digest(bytes);
    Ok(u64::from_le_bytes(
        digest[..8].try_into().expect("sha256 has at least 8 bytes"),
    ))
}

/// Returns the stable digest used by transaction journals to verify an
/// authoritative resolved settings snapshot. The snapshot generation is
/// intentionally excluded; values and their schema keys define the state.
pub fn resolved_snapshot_digest(registry: &SettingsRegistry, snapshot: &Snapshot) -> u64 {
    let mut hasher = Sha256::new();
    for (index, definition) in registry.defs().iter().enumerate() {
        hasher.update(definition.key.as_str().as_bytes());
        hasher.update([0]);
        if let Some(value) = snapshot.get(SettingId(index as u32)) {
            hash_value(&mut hasher, value);
        } else {
            hasher.update([0xff]);
        }
    }
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("sha256 has at least 8 bytes"))
}

fn hash_value(hasher: &mut Sha256, value: &Value) {
    match value {
        Value::Bool(value) => {
            hasher.update([1, u8::from(*value)]);
        }
        Value::Int(value) => {
            hasher.update([2]);
            hasher.update(value.to_le_bytes());
        }
        Value::Float(value) => {
            hasher.update([3]);
            hasher.update(value.to_bits().to_le_bytes());
        }
        Value::String(value) => {
            hasher.update([4]);
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        Value::Enum(value) => {
            hasher.update([5]);
            hasher.update(value.0.to_le_bytes());
        }
        Value::Array(values) => {
            hasher.update([6]);
            hasher.update((values.len() as u64).to_le_bytes());
            for value in values {
                hash_value(hasher, value);
            }
        }
        Value::Table(values) => {
            hasher.update([7]);
            hasher.update((values.len() as u64).to_le_bytes());
            for (key, value) in values {
                hasher.update((key.len() as u64).to_le_bytes());
                hasher.update(key.as_bytes());
                hash_value(hasher, value);
            }
        }
        Value::Bytes(value) => {
            hasher.update([8]);
            hasher.update(value.0.to_le_bytes());
        }
        Value::Secret(value) => {
            hasher.update([9]);
            hasher.update(value.0.to_le_bytes());
        }
        _ => {
            hasher.update([0xff]);
        }
    }
}

fn plan_digest(plan: &ApplyPlan) -> u64 {
    let mut hasher = Sha256::new();
    for op in &plan.ops {
        hasher.update(op.index.to_le_bytes());
        hasher.update(op.id.as_bytes());
        hasher.update([0]);
        hasher.update(op.group.as_bytes());
        hasher.update([0]);
        hasher.update(op.key.as_str().as_bytes());
        hasher.update([0]);
        hasher.update(op.input_digest.to_le_bytes());
        hasher.update([u8::from(op.idempotent)]);
        hasher.update([op.reversibility as u8]);
    }
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("sha256 has at least 8 bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ApplyOp, CompensateOutcome, EffectState, OpAction, OpOutcome, Reversibility,
        SettingsApplier,
    };
    use prism_settings::{
        Constraints, LayerKind, RegistryBuilder, SettingDef, SettingKey, TrustLevel, Value,
    };
    use std::{
        cell::{Cell, RefCell},
        fs::OpenOptions,
        io::Write,
    };
    use tempfile::tempdir;

    fn registry() -> SettingsRegistry {
        let mut builder = RegistryBuilder::new();
        builder.namespace("audio", "test").unwrap();
        let mut def = SettingDef::new("audio.level", "audio", Value::Int(50));
        def.constraints = Constraints {
            min: Some(0.0),
            max: Some(100.0),
            ..Constraints::default()
        };
        builder.register(def);
        builder.finish().unwrap()
    }

    fn document(value: i64) -> SettingsDocument {
        let mut document = SettingsDocument::new(1);
        document.insert("audio.level", Value::Int(value));
        document.preserve_unknown("future.key", toml::Value::String("kept".to_owned()));
        document
    }

    #[test]
    fn watcher_builds_candidate_and_preserves_unknown_keys() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let backup = directory.path().join("settings.bak");
        let first = document(60);
        crate::write_atomic(&path, &first, Some(&backup)).unwrap();
        let registry = registry();
        let migration = MigrationChain::new(1);
        let mut watcher = SettingsWatcher::new(
            &path,
            Some(&backup),
            &registry,
            migration,
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        );
        let result = watcher.poll(&[]).unwrap();
        let WatchPoll::Changed { candidate, .. } = result else {
            panic!("first poll must produce a candidate")
        };
        assert_eq!(
            candidate.document.unknown()["future.key"],
            toml::Value::String("kept".into())
        );
        assert_eq!(
            candidate.snapshot.get(prism_settings::SettingId(0)),
            Some(&Value::Int(60))
        );
        assert!(matches!(
            watcher.poll(&[]).unwrap(),
            WatchPoll::Changed { .. }
        ));
        assert_eq!(watcher.last_observed_digest(), None);
        let WatchPoll::Changed {
            observed_digest, ..
        } = watcher.poll(&[]).unwrap()
        else {
            panic!("an unacknowledged candidate must remain pending")
        };
        assert!(watcher.acknowledge(observed_digest).unwrap());
        assert_eq!(watcher.last_observed_digest(), Some(observed_digest));
        assert!(matches!(
            watcher.poll(&[]).unwrap(),
            WatchPoll::Unchanged { .. }
        ));
    }

    #[test]
    fn watcher_does_not_acknowledge_a_candidate_after_files_change() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        crate::write_atomic(&path, &document(60), None::<&Path>).unwrap();
        let registry = registry();
        let mut watcher = SettingsWatcher::new(
            &path,
            None::<&Path>,
            &registry,
            MigrationChain::new(1),
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        );
        let WatchPoll::Changed {
            observed_digest, ..
        } = watcher.poll(&[]).unwrap()
        else {
            panic!("first poll must produce a candidate")
        };
        crate::write_atomic(&path, &document(62), None::<&Path>).unwrap();
        assert!(!watcher.acknowledge(observed_digest).unwrap());
        assert_eq!(watcher.last_observed_digest(), None);
        assert!(matches!(
            watcher.poll(&[]).unwrap(),
            WatchPoll::Changed { .. }
        ));
    }

    #[test]
    fn watcher_retries_after_truncated_primary_is_repaired() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        fs::write(&path, b"broken = [").unwrap();
        let registry = registry();
        let mut watcher = SettingsWatcher::new(
            &path,
            None::<&Path>,
            &registry,
            MigrationChain::new(1),
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        );
        assert!(watcher.poll(&[]).is_err());
        crate::write_atomic(&path, &document(61), None::<&Path>).unwrap();
        assert!(matches!(
            watcher.poll(&[]).unwrap(),
            WatchPoll::Changed { .. }
        ));
    }

    struct MemorySnapshots {
        current: Snapshot,
        restores: Cell<u32>,
    }

    impl SnapshotController for MemorySnapshots {
        fn install_snapshot(&mut self, snapshot: Snapshot) -> Result<(), String> {
            self.current = snapshot;
            Ok(())
        }

        fn restore_generation(&mut self, _generation: u64, _hash: u64) -> Result<(), String> {
            self.restores.set(self.restores.get() + 1);
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecoverySnapshots {
        restored_generations: RefCell<Vec<(u64, u64)>>,
        installed_sources: RefCell<Vec<(u64, u64)>>,
    }

    impl SnapshotController for RecoverySnapshots {
        fn install_snapshot(&mut self, _snapshot: Snapshot) -> Result<(), String> {
            Ok(())
        }

        fn restore_generation(&mut self, generation: u64, hash: u64) -> Result<(), String> {
            self.restored_generations
                .borrow_mut()
                .push((generation, hash));
            Ok(())
        }

        fn install_authoritative_sources(
            &mut self,
            _sources: Vec<LayerSource>,
            generation: u64,
            hash: u64,
        ) -> Result<(), String> {
            self.installed_sources.borrow_mut().push((generation, hash));
            Ok(())
        }
    }

    struct RecordingRebuilder {
        requests: RefCell<Vec<(u64, u64)>>,
        fail: bool,
    }

    impl AuthoritativeSourceRebuilder for RecordingRebuilder {
        fn load_authoritative_sources(
            &mut self,
            generation: u64,
            hash: u64,
        ) -> Result<Vec<LayerSource>, String> {
            self.requests.borrow_mut().push((generation, hash));
            if self.fail {
                Err("injected source rebuild failure".to_owned())
            } else {
                Ok(Vec::new())
            }
        }
    }

    fn append_recovery_prefix(path: &Path, events: impl IntoIterator<Item = JournalEvent>) {
        let mut journal = Journal::open(path, "source-rebuild-recovery").unwrap();
        for event in events {
            journal.append(event).unwrap();
        }
    }

    struct MemoryApplier {
        applied: Cell<u32>,
        compensated: Cell<u32>,
    }

    impl SettingsApplier for MemoryApplier {
        fn group(&self) -> &str {
            "Audio"
        }
        fn order(&self) -> i32 {
            0
        }
        fn apply_op(&mut self, _op: &ApplyOp) -> OpOutcome {
            self.applied.set(self.applied.get() + 1);
            OpOutcome::Applied {
                effect_handle: Some("audio".into()),
                result_digest: 1,
            }
        }
        fn probe(&self, _op: &ApplyOp) -> EffectState {
            EffectState::Unknown
        }
        fn compensate(&mut self, _op: &ApplyOp) -> CompensateOutcome {
            self.compensated.set(self.compensated.get() + 1);
            CompensateOutcome::Done { verified: true }
        }
    }

    struct FailingSecondApplier {
        calls: Cell<u32>,
        compensated: Cell<u32>,
    }

    impl SettingsApplier for FailingSecondApplier {
        fn group(&self) -> &str {
            "Audio"
        }
        fn order(&self) -> i32 {
            0
        }
        fn apply_op(&mut self, _op: &ApplyOp) -> OpOutcome {
            let call = self.calls.get() + 1;
            self.calls.set(call);
            if call == 2 {
                OpOutcome::Failed {
                    error: "injected apply failure".to_owned(),
                    effect_handle: None,
                }
            } else {
                OpOutcome::Applied {
                    effect_handle: Some("audio".into()),
                    result_digest: 1,
                }
            }
        }
        fn probe(&self, _op: &ApplyOp) -> EffectState {
            EffectState::Unknown
        }
        fn compensate(&mut self, _op: &ApplyOp) -> CompensateOutcome {
            self.compensated.set(self.compensated.get() + 1);
            CompensateOutcome::Done { verified: true }
        }
    }

    struct NoopPersistence;
    impl SettingsPersistence for NoopPersistence {
        fn persist(
            &mut self,
            document: &SettingsDocument,
            _generation: u64,
        ) -> Result<u64, String> {
            document_digest(document).map_err(|error| error.to_string())
        }
    }

    struct FailingPersistence {
        rolled_back: Cell<u32>,
    }

    impl SettingsPersistence for FailingPersistence {
        fn persist(
            &mut self,
            _document: &SettingsDocument,
            _generation: u64,
        ) -> Result<u64, String> {
            Err("injected persistence failure".to_owned())
        }

        fn rollback(&mut self) -> Result<(), String> {
            self.rolled_back.set(self.rolled_back.get() + 1);
            Ok(())
        }
    }

    fn apply_op() -> ApplyOp {
        ApplyOp {
            index: 0,
            id: "audio-level".into(),
            group: "Audio".into(),
            key: SettingKey::from("audio.level"),
            action: OpAction::Set,
            reversibility: Reversibility::Reversible,
            idempotent: true,
            input_digest: 1,
            effect_handle: None,
        }
    }

    fn recovery_begin(mode: TxMode, op_count: u16, hash_after: u64) -> JournalEvent {
        JournalEvent::TxBegin {
            mode,
            hash_before: 10,
            hash_after,
            plan_digest: 20,
            op_count,
            snapshot_gen: 1,
            last_confirmed: None,
        }
    }

    #[test]
    fn source_rebuild_recovery_selects_candidate_state_for_durable_commit() {
        let directory = tempdir().unwrap();
        let journal_path = directory.path().join("settings.journal");
        let hash_after = 30;
        append_recovery_prefix(
            &journal_path,
            [
                recovery_begin(TxMode::Direct, 0, hash_after),
                JournalEvent::SnapshotCommitted { snapshot_gen: 2 },
                JournalEvent::PersistBegin {
                    generation: 2,
                    manifest_hash: 40,
                },
                JournalEvent::PersistEnd {
                    generation: 2,
                    manifest_hash: 40,
                },
            ],
        );
        let mut snapshots = RecoverySnapshots::default();
        let mut rebuilder = RecordingRebuilder {
            requests: RefCell::default(),
            fail: false,
        };
        let mut persistence = NoopPersistence;
        let mut appliers: Vec<&mut dyn SettingsApplier> = Vec::new();

        assert_eq!(
            recover_settings_transaction_with_source_rebuild(
                &journal_path,
                &mut snapshots,
                &mut rebuilder,
                &mut persistence,
                &mut appliers,
            )
            .unwrap(),
            Some(PipelineOutcome::Committed)
        );
        assert_eq!(*rebuilder.requests.borrow(), vec![(2, hash_after)]);
        assert_eq!(*snapshots.installed_sources.borrow(), vec![(2, hash_after)]);
    }

    #[test]
    fn source_rebuild_recovery_restores_file_preimage_before_loading_sources() {
        struct FileCheckingRebuilder {
            path: PathBuf,
            expected: Vec<u8>,
            requests: Vec<(u64, u64)>,
        }
        impl AuthoritativeSourceRebuilder for FileCheckingRebuilder {
            fn load_authoritative_sources(
                &mut self,
                generation: u64,
                hash: u64,
            ) -> Result<Vec<LayerSource>, String> {
                assert_eq!(fs::read(&self.path).unwrap(), self.expected);
                self.requests.push((generation, hash));
                Ok(Vec::new())
            }
        }

        let candidate = CandidateBuilder::new(
            &registry(),
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        )
        .build(document(70), &[], 2, false)
        .unwrap();
        let directory = tempdir().unwrap();
        let settings_path = directory.path().join("settings.toml");
        let backup_path = directory.path().join("settings.bak");
        let before_bytes = document(50).encode().unwrap().into_bytes();
        crate::write_atomic(&settings_path, &document(50), Some(&backup_path)).unwrap();
        let mut persistence = FileSettingsPersistence::new(&settings_path, Some(&backup_path), 1);
        persistence
            .persist(&candidate.document, candidate.generation)
            .unwrap();

        let journal_path = directory.path().join("settings.journal");
        append_recovery_prefix(
            &journal_path,
            [
                recovery_begin(TxMode::Direct, 1, candidate.snapshot_digest),
                JournalEvent::SnapshotCommitted {
                    snapshot_gen: candidate.generation,
                },
                JournalEvent::PersistBegin {
                    generation: candidate.generation,
                    manifest_hash: candidate.document_digest,
                },
            ],
        );
        let mut snapshots = RecoverySnapshots::default();
        let mut rebuilder = FileCheckingRebuilder {
            path: settings_path.clone(),
            expected: before_bytes.clone(),
            requests: Vec::new(),
        };
        let mut applier = MemoryApplier {
            applied: Cell::new(0),
            compensated: Cell::new(0),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];

        assert_eq!(
            recover_settings_transaction_with_source_rebuild(
                &journal_path,
                &mut snapshots,
                &mut rebuilder,
                &mut persistence,
                &mut appliers,
            )
            .unwrap(),
            Some(PipelineOutcome::RecoveryRequired)
        );
        assert_eq!(rebuilder.requests, vec![(1, 10)]);
        assert_eq!(fs::read(settings_path).unwrap(), before_bytes);
        assert_eq!(*snapshots.installed_sources.borrow(), vec![(1, 10)]);
        assert_eq!(applier.applied.get(), 0);
    }

    #[test]
    fn failed_source_rebuild_does_not_enter_effect_recovery() {
        let directory = tempdir().unwrap();
        let journal_path = directory.path().join("settings.journal");
        let op = apply_op();
        append_recovery_prefix(
            &journal_path,
            [
                recovery_begin(TxMode::Preview, 1, 30),
                JournalEvent::SnapshotCommitted { snapshot_gen: 2 },
                JournalEvent::OpBegin { op },
            ],
        );
        let mut snapshots = RecoverySnapshots::default();
        let mut rebuilder = RecordingRebuilder {
            requests: RefCell::default(),
            fail: true,
        };
        let mut persistence = NoopPersistence;
        let mut applier = MemoryApplier {
            applied: Cell::new(0),
            compensated: Cell::new(0),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];

        assert!(matches!(
            recover_settings_transaction_with_source_rebuild(
                &journal_path,
                &mut snapshots,
                &mut rebuilder,
                &mut persistence,
                &mut appliers,
            ),
            Err(PipelineError::Snapshot(_))
        ));
        assert_eq!(applier.applied.get(), 0);
        assert!(snapshots.installed_sources.borrow().is_empty());
        let recovered = crate::recover_journal(&journal_path).unwrap();
        assert!(crate::reconstruct_transaction(&recovered.records)
            .unwrap()
            .ended
            .is_none());
    }

    #[test]
    fn source_rebuild_recovery_rolls_back_interrupted_pending_preview() {
        let directory = tempdir().unwrap();
        let journal_path = directory.path().join("settings.journal");
        append_recovery_prefix(
            &journal_path,
            [
                recovery_begin(TxMode::Preview, 0, 30),
                JournalEvent::SnapshotCommitted { snapshot_gen: 2 },
                JournalEvent::TxEnd {
                    outcome: TxOutcome::PendingConfirmation,
                },
            ],
        );
        let mut snapshots = RecoverySnapshots::default();
        let mut rebuilder = RecordingRebuilder {
            requests: RefCell::default(),
            fail: false,
        };
        let mut persistence = NoopPersistence;
        let mut appliers: Vec<&mut dyn SettingsApplier> = Vec::new();

        assert_eq!(
            recover_settings_transaction_with_source_rebuild(
                &journal_path,
                &mut snapshots,
                &mut rebuilder,
                &mut persistence,
                &mut appliers,
            )
            .unwrap(),
            Some(PipelineOutcome::RolledBack)
        );
        assert_eq!(*rebuilder.requests.borrow(), vec![(1, 10)]);
        assert_eq!(*snapshots.installed_sources.borrow(), vec![(1, 10)]);
    }

    #[test]
    fn source_rebuild_recovery_skips_terminal_journal_and_fails_closed_on_empty_corruption() {
        let directory = tempdir().unwrap();
        let terminal_path = directory.path().join("terminal.journal");
        append_recovery_prefix(
            &terminal_path,
            [
                recovery_begin(TxMode::Preview, 0, 30),
                JournalEvent::TxEnd {
                    outcome: TxOutcome::Committed,
                },
            ],
        );
        let mut snapshots = RecoverySnapshots::default();
        let mut rebuilder = RecordingRebuilder {
            requests: RefCell::default(),
            fail: false,
        };
        let mut persistence = NoopPersistence;
        let mut appliers: Vec<&mut dyn SettingsApplier> = Vec::new();
        assert_eq!(
            recover_settings_transaction_with_source_rebuild(
                &terminal_path,
                &mut snapshots,
                &mut rebuilder,
                &mut persistence,
                &mut appliers,
            )
            .unwrap(),
            Some(PipelineOutcome::Committed)
        );
        assert!(rebuilder.requests.borrow().is_empty());

        let corrupt_path = directory.path().join("corrupt.journal");
        fs::write(&corrupt_path, b"not a journal record\n").unwrap();
        assert!(recover_settings_transaction_with_source_rebuild(
            &corrupt_path,
            &mut snapshots,
            &mut rebuilder,
            &mut persistence,
            &mut appliers,
        )
        .is_err());
        assert_eq!(fs::read(&corrupt_path).unwrap(), b"not a journal record\n");
        assert!(rebuilder.requests.borrow().is_empty());
    }

    #[test]
    fn preview_confirm_and_rollback_are_durable() {
        let registry = registry();
        let base = CandidateBuilder::new(
            &registry,
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        )
        .build(document(50), &[], 1, false)
        .unwrap();
        let candidate = CandidateBuilder::new(
            &registry,
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        )
        .build(document(70), &[], 2, false)
        .unwrap();
        let directory = tempdir().unwrap();
        let journal_path = directory.path().join("settings.journal");
        let transaction = SettingsTransaction::new(SettingsTransactionSpec {
            tx_id: "preview-1".to_owned(),
            journal_path: journal_path.clone(),
            mode: TxMode::Preview,
            before_hash: 10,
            before_snapshot: base.snapshot.clone(),
            candidate,
            plan: ApplyPlan::new(vec![apply_op()]).unwrap(),
            snapshot_gen: 1,
            last_confirmed: Some(1),
        });
        let mut snapshots = MemorySnapshots {
            current: base.snapshot,
            restores: Cell::new(0),
        };
        let mut persistence = NoopPersistence;
        let mut applier = MemoryApplier {
            applied: Cell::new(0),
            compensated: Cell::new(0),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];
        assert_eq!(
            transaction
                .execute(&mut snapshots, &mut persistence, &mut appliers)
                .unwrap(),
            PipelineOutcome::PendingConfirmation
        );
        assert_eq!(
            transaction
                .confirm_preview(&mut snapshots, &mut persistence, &mut appliers)
                .unwrap(),
            PipelineOutcome::Committed
        );
        assert!(matches!(
            crate::reconstruct_transaction(&crate::recover_journal(&journal_path).unwrap().records)
                .unwrap()
                .ended,
            Some(TxOutcome::Committed)
        ));
    }

    #[test]
    fn direct_persistence_failure_restores_snapshot_and_closes_transaction() {
        let registry = registry();
        let base = CandidateBuilder::new(
            &registry,
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        )
        .build(document(50), &[], 1, false)
        .unwrap();
        let candidate = CandidateBuilder::new(
            &registry,
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        )
        .build(document(70), &[], 2, false)
        .unwrap();
        let directory = tempdir().unwrap();
        let journal_path = directory.path().join("settings.journal");
        let transaction = SettingsTransaction::new(SettingsTransactionSpec {
            tx_id: "direct-persist-failure".to_owned(),
            journal_path: journal_path.clone(),
            mode: TxMode::Direct,
            before_hash: 10,
            before_snapshot: base.snapshot.clone(),
            candidate,
            plan: ApplyPlan::new(vec![apply_op()]).unwrap(),
            snapshot_gen: 1,
            last_confirmed: None,
        });
        let mut snapshots = MemorySnapshots {
            current: base.snapshot.clone(),
            restores: Cell::new(0),
        };
        let mut persistence = FailingPersistence {
            rolled_back: Cell::new(0),
        };
        let mut applier = MemoryApplier {
            applied: Cell::new(0),
            compensated: Cell::new(0),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];
        assert!(matches!(
            transaction.execute(&mut snapshots, &mut persistence, &mut appliers),
            Err(PipelineError::Persistence(_))
        ));
        assert_eq!(persistence.rolled_back.get(), 1);
        assert_eq!(snapshots.current, base.snapshot);
        assert_eq!(applier.applied.get(), 0);
        assert_eq!(
            crate::reconstruct_transaction(&crate::recover_journal(&journal_path).unwrap().records)
                .unwrap()
                .ended,
            Some(TxOutcome::RolledBack)
        );
    }

    #[test]
    fn direct_apply_failure_restores_file_snapshot_and_journal() {
        let registry = registry();
        let base = CandidateBuilder::new(
            &registry,
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        )
        .build(document(50), &[], 1, false)
        .unwrap();
        let candidate = CandidateBuilder::new(
            &registry,
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        )
        .build(document(70), &[], 2, false)
        .unwrap();
        let directory = tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let backup = directory.path().join("settings.toml.bak");
        let before_bytes = document(50).encode().unwrap().into_bytes();
        crate::write_atomic(&path, &document(50), Some(&backup)).unwrap();
        let journal_path = directory.path().join("settings.journal");
        let mut second_op = apply_op();
        second_op.index = 1;
        second_op.id = "audio-level-2".to_owned();
        let transaction = SettingsTransaction::new(SettingsTransactionSpec {
            tx_id: "direct-apply-failure".to_owned(),
            journal_path: journal_path.clone(),
            mode: TxMode::Direct,
            before_hash: 10,
            before_snapshot: base.snapshot.clone(),
            candidate,
            plan: ApplyPlan::new(vec![apply_op(), second_op]).unwrap(),
            snapshot_gen: 1,
            last_confirmed: None,
        });
        let mut snapshots = MemorySnapshots {
            current: base.snapshot,
            restores: Cell::new(0),
        };
        let mut persistence = FileSettingsPersistence::new(&path, Some(&backup), 1);
        let mut applier = FailingSecondApplier {
            calls: Cell::new(0),
            compensated: Cell::new(0),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];
        assert_eq!(
            transaction
                .execute(&mut snapshots, &mut persistence, &mut appliers)
                .unwrap(),
            PipelineOutcome::RolledBack
        );
        assert_eq!(fs::read(&path).unwrap(), before_bytes);
        assert_eq!(applier.compensated.get(), 1);
        assert_eq!(
            crate::reconstruct_transaction(&crate::recover_journal(&journal_path).unwrap().records)
                .unwrap()
                .ended,
            Some(TxOutcome::RolledBack)
        );
        assert!(!recovery_path(&path).exists());
    }

    #[test]
    fn restart_recovery_restores_durable_file_image_before_unknown_effect_resolution() {
        let registry = registry();
        let candidate = CandidateBuilder::new(
            &registry,
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        )
        .build(document(70), &[], 2, false)
        .unwrap();
        let directory = tempdir().unwrap();
        let path = directory.path().join("settings.toml");
        let backup = directory.path().join("settings.toml.bak");
        let before_bytes = document(50).encode().unwrap().into_bytes();
        crate::write_atomic(&path, &document(50), Some(&backup)).unwrap();
        let mut persistence = FileSettingsPersistence::new(&path, Some(&backup), 1);
        persistence
            .persist(&candidate.document, candidate.generation)
            .unwrap();
        let journal_path = directory.path().join("settings.journal");
        let plan = ApplyPlan::new(vec![apply_op()]).unwrap();
        let mut journal = Journal::open(&journal_path, "restart-recovery").unwrap();
        for event in [
            JournalEvent::TxBegin {
                mode: TxMode::Direct,
                hash_before: 10,
                hash_after: candidate.snapshot_digest,
                plan_digest: plan_digest(&plan),
                op_count: 1,
                snapshot_gen: 1,
                last_confirmed: None,
            },
            JournalEvent::SnapshotCommitted { snapshot_gen: 2 },
            JournalEvent::PersistBegin {
                generation: candidate.generation,
                manifest_hash: candidate.document_digest,
            },
            JournalEvent::PersistEnd {
                generation: candidate.generation,
                manifest_hash: candidate.document_digest,
            },
            JournalEvent::OpBegin { op: apply_op() },
        ] {
            journal.append(event).unwrap();
        }
        let mut snapshots = MemorySnapshots {
            current: candidate.snapshot,
            restores: Cell::new(0),
        };
        let mut applier = MemoryApplier {
            applied: Cell::new(0),
            compensated: Cell::new(0),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];
        assert_eq!(
            recover_settings_transaction(
                &journal_path,
                &mut snapshots,
                &mut persistence,
                &mut appliers,
            )
            .unwrap(),
            Some(PipelineOutcome::RecoveryRequired)
        );
        assert_eq!(fs::read(&path).unwrap(), before_bytes);
        assert!(!recovery_path(&path).exists());
        assert_eq!(applier.applied.get(), 0);
    }

    #[test]
    fn restart_recovery_removes_file_when_durable_preimage_was_absent() {
        let registry = registry();
        let candidate = CandidateBuilder::new(
            &registry,
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        )
        .build(document(70), &[], 2, false)
        .unwrap();
        let directory = tempdir().unwrap();
        let path = directory.path().join("new-settings.toml");
        let backup = directory.path().join("new-settings.toml.bak");
        let mut persistence = FileSettingsPersistence::new(&path, Some(&backup), 1);
        persistence
            .persist(&candidate.document, candidate.generation)
            .unwrap();
        let journal_path = directory.path().join("settings.journal");
        let plan = ApplyPlan::new(vec![apply_op()]).unwrap();
        let mut journal = Journal::open(&journal_path, "restart-delete").unwrap();
        for event in [
            JournalEvent::TxBegin {
                mode: TxMode::Direct,
                hash_before: 10,
                hash_after: candidate.snapshot_digest,
                plan_digest: plan_digest(&plan),
                op_count: 1,
                snapshot_gen: 1,
                last_confirmed: None,
            },
            JournalEvent::SnapshotCommitted { snapshot_gen: 2 },
            JournalEvent::PersistBegin {
                generation: candidate.generation,
                manifest_hash: candidate.document_digest,
            },
            JournalEvent::OpBegin { op: apply_op() },
        ] {
            journal.append(event).unwrap();
        }
        let mut snapshots = MemorySnapshots {
            current: candidate.snapshot,
            restores: Cell::new(0),
        };
        let mut applier = MemoryApplier {
            applied: Cell::new(0),
            compensated: Cell::new(0),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];
        assert_eq!(
            recover_settings_transaction(
                &journal_path,
                &mut snapshots,
                &mut persistence,
                &mut appliers,
            )
            .unwrap(),
            Some(PipelineOutcome::RecoveryRequired)
        );
        assert!(!path.exists());
        assert!(!recovery_path(&path).exists());
    }

    #[test]
    fn recovery_rolls_back_pending_preview_after_truncated_tail() {
        let registry = registry();
        let base = CandidateBuilder::new(
            &registry,
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        )
        .build(document(50), &[], 1, false)
        .unwrap();
        let candidate = CandidateBuilder::new(
            &registry,
            Domain::Runtime,
            SourceSpec::new(LayerKind::User, TrustLevel::User, "user-file"),
        )
        .build(document(70), &[], 2, false)
        .unwrap();
        let directory = tempdir().unwrap();
        let journal_path = directory.path().join("settings.journal");
        let transaction = SettingsTransaction::new(SettingsTransactionSpec {
            tx_id: "preview-2".to_owned(),
            journal_path: journal_path.clone(),
            mode: TxMode::Preview,
            before_hash: 10,
            before_snapshot: base.snapshot.clone(),
            candidate,
            plan: ApplyPlan::new(vec![apply_op()]).unwrap(),
            snapshot_gen: 1,
            last_confirmed: None,
        });
        let mut snapshots = MemorySnapshots {
            current: base.snapshot,
            restores: Cell::new(0),
        };
        let mut persistence = NoopPersistence;
        let mut applier = MemoryApplier {
            applied: Cell::new(0),
            compensated: Cell::new(0),
        };
        let mut appliers: Vec<&mut dyn SettingsApplier> = vec![&mut applier];
        assert_eq!(
            transaction
                .execute(&mut snapshots, &mut persistence, &mut appliers)
                .unwrap(),
            PipelineOutcome::PendingConfirmation
        );
        OpenOptions::new()
            .append(true)
            .open(&journal_path)
            .unwrap()
            .write_all(b"{truncated")
            .unwrap();
        let outcome = recover_settings_transaction(
            &journal_path,
            &mut snapshots,
            &mut persistence,
            &mut appliers,
        )
        .unwrap()
        .unwrap();
        assert_eq!(outcome, PipelineOutcome::RolledBack);
        assert_eq!(snapshots.restores.get(), 1);
        assert_eq!(applier.compensated.get(), 1);
    }
}
