//! Crash-safe, operation-level settings transaction journal.
//!
//! The journal deliberately stores operation metadata, digests, and opaque
//! effect handles rather than setting values. Each append is flushed and
//! synced before the caller is allowed to perform the corresponding side
//! effect. Recovery accepts only a valid prefix of the JSON-lines file; an
//! incomplete or checksum-invalid tail is truncated and never guessed.

use prism_settings::SettingKey;
use serde_json::{json, Map, Value as JsonValue};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    error::Error,
    fmt, fs,
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const CHECKSUM_KEY: &str = "checksum";

/// Transaction execution mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TxMode {
    /// Apply and persist without a user confirmation window.
    Direct,
    /// Apply a reversible preview and wait for confirmation before persisting.
    Preview,
}

impl TxMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Preview => "preview",
        }
    }

    fn parse(value: &str) -> Result<Self, JournalError> {
        match value {
            "direct" => Ok(Self::Direct),
            "preview" => Ok(Self::Preview),
            other => Err(JournalError::InvalidField {
                field: "mode".to_owned(),
                reason: format!("unknown transaction mode `{other}`"),
            }),
        }
    }
}

/// Reversibility class of an apply operation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Reversibility {
    /// The old value can be restored exactly.
    Reversible,
    /// Restoring the old value may fail and must be verified.
    Compensable,
    /// The operation cannot be undone once started.
    Irreversible,
    /// An external system owns the final effect.
    External,
}

impl Reversibility {
    fn as_str(self) -> &'static str {
        match self {
            Self::Reversible => "reversible",
            Self::Compensable => "compensable",
            Self::Irreversible => "irreversible",
            Self::External => "external",
        }
    }

    fn parse(value: &str) -> Result<Self, JournalError> {
        match value {
            "reversible" => Ok(Self::Reversible),
            "compensable" => Ok(Self::Compensable),
            "irreversible" => Ok(Self::Irreversible),
            "external" => Ok(Self::External),
            other => Err(JournalError::InvalidField {
                field: "reversibility".to_owned(),
                reason: format!("unknown reversibility `{other}`"),
            }),
        }
    }
}

/// The operation kind recorded in the journal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpAction {
    /// Set a setting to a new value.
    Set,
    /// Remove a setting override.
    Reset,
    /// Rebuild an external or in-memory resource.
    Rebuild,
    /// Notify observers without changing the setting value.
    Notify,
    /// An application-defined action name.
    Custom(String),
}

impl OpAction {
    fn as_str(&self) -> &str {
        match self {
            Self::Set => "set",
            Self::Reset => "reset",
            Self::Rebuild => "rebuild",
            Self::Notify => "notify",
            Self::Custom(value) => value,
        }
    }

    fn from_string(value: String) -> Self {
        match value.as_str() {
            "set" => Self::Set,
            "reset" => Self::Reset,
            "rebuild" => Self::Rebuild,
            "notify" => Self::Notify,
            _ => Self::Custom(value),
        }
    }
}

/// One minimal side-effect unit in an apply plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ApplyOp {
    /// Execution order, starting at zero.
    pub index: u16,
    /// Stable operation identity.
    pub id: String,
    /// Apply group such as `Display` or `Audio`.
    pub group: String,
    /// Schema key affected by the operation.
    pub key: SettingKey,
    /// Operation action.
    pub action: OpAction,
    /// Reversibility class.
    pub reversibility: Reversibility,
    /// Whether an unknown in-flight operation may be retried safely.
    pub idempotent: bool,
    /// Digest of the new value and target handle; never the value itself.
    pub input_digest: u64,
    /// Opaque handle returned by the applier.
    pub effect_handle: Option<String>,
}

/// Result of probing an external effect during recovery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EffectState {
    /// The target effect is known to be applied.
    Applied,
    /// The target effect is known not to be applied.
    NotApplied,
    /// The target effect cannot be determined safely.
    Unknown,
}

/// State of one operation reconstructed from journal events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApplyOpState {
    /// No operation invocation was started.
    Pending,
    /// An invocation began but no terminal event was durable.
    Unknown,
    /// The operation completed successfully.
    Applied,
    /// The operation completed with failure.
    Failed,
}

/// Compensation ledger state for one operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompensationState {
    /// No compensation is needed.
    NotNeeded,
    /// Compensation has not started.
    NotStarted,
    /// Compensation began but has no terminal event.
    Unknown,
    /// Compensation completed.
    Done,
    /// Compensation completed with failure.
    Failed,
}

/// Reconstructed operation ledger entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JournalOpState {
    /// Original operation metadata, if an `OpBegin` was recorded.
    pub op: ApplyOp,
    /// Reconstructed apply state.
    pub state: ApplyOpState,
    /// Compensation state.
    pub compensation: CompensationState,
    /// Number of apply attempts.
    pub attempts: u32,
    /// Number of compensation attempts.
    pub compensation_attempts: u32,
    /// Last error recorded for the operation.
    pub last_error: Option<String>,
}

/// Final transaction outcome stored by `TxEnd`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TxOutcome {
    /// State and effects completed normally.
    Committed,
    /// State committed but one or more effects were not fully applied.
    Degraded,
    /// State was restored and reversible effects were compensated.
    RolledBack,
    /// An irreversible effect crossed its barrier and a restart is required.
    NeedsRestart,
    /// Recovery could not establish a safe outcome.
    RecoveryRequired,
    /// A preview is awaiting confirmation.
    PendingConfirmation,
}

impl TxOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Committed => "committed",
            Self::Degraded => "degraded",
            Self::RolledBack => "rolled_back",
            Self::NeedsRestart => "needs_restart",
            Self::RecoveryRequired => "recovery_required",
            Self::PendingConfirmation => "pending_confirmation",
        }
    }

    fn parse(value: &str) -> Result<Self, JournalError> {
        match value {
            "committed" => Ok(Self::Committed),
            "degraded" => Ok(Self::Degraded),
            "rolled_back" => Ok(Self::RolledBack),
            "needs_restart" => Ok(Self::NeedsRestart),
            "recovery_required" => Ok(Self::RecoveryRequired),
            "pending_confirmation" => Ok(Self::PendingConfirmation),
            other => Err(JournalError::InvalidField {
                field: "outcome".to_owned(),
                reason: format!("unknown transaction outcome `{other}`"),
            }),
        }
    }
}

/// Transaction-level recovery decision derived from the durable prefix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TxRecoveryDecision {
    /// No side effect was started; the transaction can be discarded.
    Discard,
    /// Reversible operations can be compensated and the snapshot restored.
    Rollback,
    /// The transaction crossed an irreversible barrier and needs a restart.
    NeedsRestart,
    /// The journal contains an ambiguity requiring safe-mode intervention.
    RecoveryRequired,
    /// The transaction has a durable terminal event.
    Completed,
    /// Persistence and every operation completed, but the final marker was
    /// not written before the process stopped.
    Commit,
    /// A preview is durably installed and awaits confirmation or rollback.
    PendingConfirmation,
}

/// A decoded, checksum-verified journal event.
#[derive(Clone, Debug, PartialEq)]
pub enum JournalEvent {
    /// Starts a transaction and records hashes, plan identity, and snapshot metadata.
    TxBegin {
        /// Transaction mode.
        mode: TxMode,
        /// Hash of the snapshot before the transaction.
        hash_before: u64,
        /// Hash of the candidate snapshot.
        hash_after: u64,
        /// Hash of the apply plan.
        plan_digest: u64,
        /// Number of operations in the plan.
        op_count: u16,
        /// Snapshot generation before commit.
        snapshot_gen: u64,
        /// Last confirmed generation, if any.
        last_confirmed: Option<u64>,
    },
    /// Records the atomic snapshot exchange.
    SnapshotCommitted {
        /// New snapshot generation.
        snapshot_gen: u64,
    },
    /// Starts persistence of one generation.
    PersistBegin {
        /// Generation being written.
        generation: u64,
        /// Manifest digest, if already known.
        manifest_hash: u64,
    },
    /// Completes persistence of one generation.
    PersistEnd {
        /// Generation written.
        generation: u64,
        /// Manifest digest.
        manifest_hash: u64,
    },
    /// Written immediately before invoking an applier operation.
    OpBegin {
        /// Operation metadata.
        op: ApplyOp,
    },
    /// Written after an operation reports success.
    OpApplied {
        /// Operation index.
        op_index: u16,
        /// Opaque effect handle.
        effect_handle: Option<String>,
        /// Digest of the result metadata.
        result_digest: u64,
    },
    /// Written after an operation reports failure.
    OpFailed {
        /// Operation index.
        op_index: u16,
        /// Error with secrets removed by the applier.
        error: String,
        /// Opaque effect handle, if one was returned despite failure.
        effect_handle: Option<String>,
    },
    /// Marks the point after which recovery is forward-only.
    IrreversibleBarrier {
        /// Operation index at the barrier.
        op_index: u16,
    },
    /// Written before invoking compensation.
    CompensateBegin {
        /// Operation index being compensated.
        op_index: u16,
        /// Redacted recovery reason.
        reason: String,
    },
    /// Written after compensation succeeds.
    Compensated {
        /// Operation index compensated.
        op_index: u16,
        /// Whether the applier verified the old effect.
        verified: bool,
    },
    /// Written after compensation fails.
    CompensateFailed {
        /// Operation index that could not be compensated.
        op_index: u16,
        /// Error with secrets removed by the applier.
        error: String,
    },
    /// Closes the transaction and permits journal compaction.
    TxEnd {
        /// Final transaction outcome.
        outcome: TxOutcome,
    },
}

impl JournalEvent {
    fn kind(&self) -> &'static str {
        match self {
            Self::TxBegin { .. } => "TxBegin",
            Self::SnapshotCommitted { .. } => "SnapshotCommitted",
            Self::PersistBegin { .. } => "PersistBegin",
            Self::PersistEnd { .. } => "PersistEnd",
            Self::OpBegin { .. } => "OpBegin",
            Self::OpApplied { .. } => "OpApplied",
            Self::OpFailed { .. } => "OpFailed",
            Self::IrreversibleBarrier { .. } => "IrreversibleBarrier",
            Self::CompensateBegin { .. } => "CompensateBegin",
            Self::Compensated { .. } => "Compensated",
            Self::CompensateFailed { .. } => "CompensateFailed",
            Self::TxEnd { .. } => "TxEnd",
        }
    }

    fn payload(&self) -> JsonValue {
        match self {
            Self::TxBegin {
                mode,
                hash_before,
                hash_after,
                plan_digest,
                op_count,
                snapshot_gen,
                last_confirmed,
            } => json!({
                "mode": mode.as_str(),
                "hash_before": hash_before,
                "hash_after": hash_after,
                "plan_digest": plan_digest,
                "op_count": op_count,
                "snapshot_gen": snapshot_gen,
                "last_confirmed": last_confirmed,
            }),
            Self::SnapshotCommitted { snapshot_gen } => json!({
                "snapshot_gen": snapshot_gen,
            }),
            Self::PersistBegin {
                generation,
                manifest_hash,
            }
            | Self::PersistEnd {
                generation,
                manifest_hash,
            } => json!({
                "generation": generation,
                "manifest_hash": manifest_hash,
            }),
            Self::OpBegin { op } => op_to_json(op),
            Self::OpApplied {
                op_index,
                effect_handle,
                result_digest,
            } => json!({
                "op_index": op_index,
                "effect_handle": effect_handle,
                "result_digest": result_digest,
            }),
            Self::OpFailed {
                op_index,
                error,
                effect_handle,
            } => json!({
                "op_index": op_index,
                "error": error,
                "effect_handle": effect_handle,
            }),
            Self::IrreversibleBarrier { op_index } => json!({
                "op_index": op_index,
            }),
            Self::CompensateBegin { op_index, reason } => json!({
                "op_index": op_index,
                "reason": reason,
            }),
            Self::Compensated { op_index, verified } => json!({
                "op_index": op_index,
                "verified": verified,
            }),
            Self::CompensateFailed { op_index, error } => json!({
                "op_index": op_index,
                "error": error,
            }),
            Self::TxEnd { outcome } => json!({
                "outcome": outcome.as_str(),
            }),
        }
    }

    fn from_parts(kind: &str, payload: &JsonValue) -> Result<Self, JournalError> {
        match kind {
            "TxBegin" => Ok(Self::TxBegin {
                mode: TxMode::parse(required_str(payload, "mode")?)?,
                hash_before: required_u64(payload, "hash_before")?,
                hash_after: required_u64(payload, "hash_after")?,
                plan_digest: required_u64(payload, "plan_digest")?,
                op_count: required_u16(payload, "op_count")?,
                snapshot_gen: required_u64(payload, "snapshot_gen")?,
                last_confirmed: optional_u64(payload, "last_confirmed")?,
            }),
            "SnapshotCommitted" => Ok(Self::SnapshotCommitted {
                snapshot_gen: required_u64(payload, "snapshot_gen")?,
            }),
            "PersistBegin" => Ok(Self::PersistBegin {
                generation: required_u64(payload, "generation")?,
                manifest_hash: required_u64(payload, "manifest_hash")?,
            }),
            "PersistEnd" => Ok(Self::PersistEnd {
                generation: required_u64(payload, "generation")?,
                manifest_hash: required_u64(payload, "manifest_hash")?,
            }),
            "OpBegin" => Ok(Self::OpBegin {
                op: op_from_json(payload)?,
            }),
            "OpApplied" => Ok(Self::OpApplied {
                op_index: required_u16(payload, "op_index")?,
                effect_handle: optional_string(payload, "effect_handle")?,
                result_digest: required_u64(payload, "result_digest")?,
            }),
            "OpFailed" => Ok(Self::OpFailed {
                op_index: required_u16(payload, "op_index")?,
                error: required_str(payload, "error")?.to_owned(),
                effect_handle: optional_string(payload, "effect_handle")?,
            }),
            "IrreversibleBarrier" => Ok(Self::IrreversibleBarrier {
                op_index: required_u16(payload, "op_index")?,
            }),
            "CompensateBegin" => Ok(Self::CompensateBegin {
                op_index: required_u16(payload, "op_index")?,
                reason: required_str(payload, "reason")?.to_owned(),
            }),
            "Compensated" => Ok(Self::Compensated {
                op_index: required_u16(payload, "op_index")?,
                verified: required_bool(payload, "verified")?,
            }),
            "CompensateFailed" => Ok(Self::CompensateFailed {
                op_index: required_u16(payload, "op_index")?,
                error: required_str(payload, "error")?.to_owned(),
            }),
            "TxEnd" => Ok(Self::TxEnd {
                outcome: TxOutcome::parse(required_str(payload, "outcome")?)?,
            }),
            other => Err(JournalError::InvalidEvent(other.to_owned())),
        }
    }
}

/// One durable JSON-lines record.
#[derive(Clone, Debug, PartialEq)]
pub struct JournalRecord {
    /// Transaction identity.
    pub tx_id: String,
    /// Monotonically increasing event sequence within the transaction.
    pub seq: u64,
    /// Monotonic timestamp from the writer process.
    pub monotonic_ns: u128,
    /// Decoded event.
    pub event: JournalEvent,
}

/// Result of loading and repairing a journal file.
#[derive(Clone, Debug, PartialEq)]
pub struct JournalRecovery {
    /// Valid prefix of records.
    pub records: Vec<JournalRecord>,
    /// Whether invalid or incomplete tail bytes were discarded.
    pub truncated_tail: bool,
    /// Byte offset at which the valid prefix ends.
    pub valid_bytes: u64,
}

/// Read-only result of validating a journal without repairing its tail.
#[derive(Clone, Debug, PartialEq)]
pub struct JournalInspection {
    /// Checksum-verified, sequence-contiguous record prefix.
    pub records: Vec<JournalRecord>,
    /// Whether the file contains an incomplete, malformed, or invalid tail.
    pub invalid_tail: bool,
    /// Byte offset at which the valid prefix ends.
    pub valid_bytes: u64,
    /// Total file size observed during inspection.
    pub file_bytes: u64,
}

/// Errors raised by journal I/O, encoding, or recovery validation.
#[derive(Debug)]
pub enum JournalError {
    /// Filesystem error associated with a path.
    Io {
        /// Filesystem path involved in the failure.
        path: PathBuf,
        /// Underlying filesystem error.
        source: io::Error,
    },
    /// JSON encoding or decoding failure.
    Json(String),
    /// A required field was absent or had the wrong type.
    InvalidField {
        /// Field name.
        field: String,
        /// Validation explanation.
        reason: String,
    },
    /// The event kind was not recognized.
    InvalidEvent(String),
    /// The record checksum did not match its canonical body.
    ChecksumMismatch {
        /// Sequence number whose body did not match its checksum.
        seq: u64,
    },
    /// A record was structurally valid but violated journal ordering.
    InvalidSequence {
        /// Expected sequence number.
        expected: u64,
        /// Actual sequence number.
        actual: u64,
    },
}

impl fmt::Display for JournalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(
                formatter,
                "journal I/O failed for {}: {source}",
                path.display()
            ),
            Self::Json(error) => write!(formatter, "journal JSON error: {error}"),
            Self::InvalidField { field, reason } => {
                write!(formatter, "journal field `{field}` is invalid: {reason}")
            }
            Self::InvalidEvent(event) => write!(formatter, "unknown journal event `{event}`"),
            Self::ChecksumMismatch { seq } => {
                write!(formatter, "journal checksum mismatch at sequence {seq}")
            }
            Self::InvalidSequence { expected, actual } => write!(
                formatter,
                "journal sequence expected {expected}, got {actual}"
            ),
        }
    }
}

impl Error for JournalError {}

/// Append-only journal writer for one transaction.
pub struct Journal {
    path: PathBuf,
    file: fs::File,
    tx_id: String,
    next_seq: u64,
    started: Instant,
    poisoned: bool,
}

impl Journal {
    /// Opens or creates a journal and continues the sequence after its valid prefix.
    pub fn open(path: impl AsRef<Path>, tx_id: impl Into<String>) -> Result<Self, JournalError> {
        let path = path.as_ref().to_owned();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|source| io_error(&path, source))?;
        }
        let recovery = recover_journal(&path)?;
        let tx_id = tx_id.into();
        if let Some(existing) = recovery.records.first()
            && existing.tx_id != tx_id
        {
            return Err(JournalError::InvalidField {
                field: "tx_id".to_owned(),
                reason: format!(
                    "journal belongs to transaction `{}`, not `{tx_id}`",
                    existing.tx_id
                ),
            });
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| io_error(&path, source))?;
        let next_seq = recovery
            .records
            .last()
            .map_or(0, |record| record.seq.saturating_add(1));
        Ok(Self {
            path,
            file,
            tx_id,
            next_seq,
            started: Instant::now(),
            poisoned: false,
        })
    }

    /// Returns the file backing this journal.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends an event, flushes it, and fsyncs the journal before returning.
    pub fn append(&mut self, event: JournalEvent) -> Result<JournalRecord, JournalError> {
        if self.poisoned {
            return Err(JournalError::InvalidField {
                field: "journal".to_owned(),
                reason: "writer is poisoned after an uncertain append; reopen and recover"
                    .to_owned(),
            });
        }
        let current_len = self
            .file
            .metadata()
            .map_err(|source| io_error(&self.path, source))?
            .len();
        if current_len > 16 * 1024 * 1024 {
            self.poisoned = true;
            return Err(JournalError::InvalidField {
                field: "journal".to_owned(),
                reason: "journal size limit exceeded; compaction is required".to_owned(),
            });
        }
        let record = JournalRecord {
            tx_id: self.tx_id.clone(),
            seq: self.next_seq,
            monotonic_ns: self.started.elapsed().as_nanos(),
            event,
        };
        let line = encode_record(&record)?;
        if let Err(source) = self
            .file
            .write_all(line.as_bytes())
            .and_then(|()| self.file.flush())
            .and_then(|()| self.file.sync_all())
        {
            self.poisoned = true;
            return Err(io_error(&self.path, source));
        }
        self.next_seq = self.next_seq.saturating_add(1);
        Ok(record)
    }
}

/// Reads a journal, verifies checksums and truncates any invalid tail.
pub fn recover_journal(path: impl AsRef<Path>) -> Result<JournalRecovery, JournalError> {
    let path = path.as_ref();
    let mut file = match fs::OpenOptions::new().read(true).write(true).open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(JournalRecovery {
                records: Vec::new(),
                truncated_tail: false,
                valid_bytes: 0,
            });
        }
        Err(source) => return Err(io_error(path, source)),
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|source| io_error(path, source))?;
    let mut records = Vec::new();
    let mut valid_bytes = 0usize;
    let mut expected_seq = 0;
    let mut truncated_tail = false;
    for segment in bytes.split_inclusive(|byte| *byte == b'\n') {
        if !segment.ends_with(b"\n") {
            truncated_tail = true;
            break;
        }
        let line = &segment[..segment.len() - 1];
        if line.is_empty() {
            truncated_tail = true;
            break;
        }
        match decode_record(line) {
            Ok(record) if record.seq == expected_seq => {
                expected_seq = expected_seq.saturating_add(1);
                records.push(record);
                valid_bytes += segment.len();
            }
            Ok(record) => {
                truncated_tail = true;
                let _ = record;
                break;
            }
            Err(_) => {
                truncated_tail = true;
                break;
            }
        }
    }
    if valid_bytes < bytes.len() {
        file.set_len(valid_bytes as u64)
            .map_err(|source| io_error(path, source))?;
        file.seek(SeekFrom::Start(valid_bytes as u64))
            .map_err(|source| io_error(path, source))?;
        file.sync_all().map_err(|source| io_error(path, source))?;
    }
    Ok(JournalRecovery {
        records,
        truncated_tail,
        valid_bytes: valid_bytes as u64,
    })
}

/// Reads and validates a journal without changing the file.
///
/// Unlike [`recover_journal`], this function never truncates an invalid tail.
/// The same 16 MiB bound enforced by the writer is applied before allocating
/// the input buffer, so inspection cannot be used to request an unbounded read.
pub fn inspect_journal(path: impl AsRef<Path>) -> Result<JournalInspection, JournalError> {
    const MAX_JOURNAL_BYTES: u64 = 16 * 1024 * 1024;
    let path = path.as_ref();
    let mut file = match fs::OpenOptions::new().read(true).open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(JournalInspection {
                records: Vec::new(),
                invalid_tail: false,
                valid_bytes: 0,
                file_bytes: 0,
            });
        }
        Err(source) => return Err(io_error(path, source)),
    };
    let file_bytes = file
        .metadata()
        .map_err(|source| io_error(path, source))?
        .len();
    if file_bytes > MAX_JOURNAL_BYTES {
        return Err(JournalError::InvalidField {
            field: "journal".to_owned(),
            reason: format!("size {file_bytes} exceeds inspection limit {MAX_JOURNAL_BYTES}"),
        });
    }
    let mut bytes = Vec::with_capacity(file_bytes as usize);
    file.read_to_end(&mut bytes)
        .map_err(|source| io_error(path, source))?;
    let mut records = Vec::new();
    let mut valid_bytes = 0usize;
    let mut expected_seq = 0u64;
    let mut invalid_tail = false;
    for segment in bytes.split_inclusive(|byte| *byte == b'\n') {
        if !segment.ends_with(b"\n") || segment.len() == 1 {
            invalid_tail = true;
            break;
        }
        match decode_record(&segment[..segment.len() - 1]) {
            Ok(record) if record.seq == expected_seq => {
                expected_seq = expected_seq.saturating_add(1);
                records.push(record);
                valid_bytes += segment.len();
            }
            _ => {
                invalid_tail = true;
                break;
            }
        }
    }
    Ok(JournalInspection {
        records,
        invalid_tail,
        valid_bytes: valid_bytes as u64,
        file_bytes,
    })
}

/// Reconstructs transaction and operation state from a verified journal prefix.
pub fn reconstruct_transaction(records: &[JournalRecord]) -> Result<JournalTx, JournalError> {
    let first = records.first().ok_or_else(|| JournalError::InvalidField {
        field: "records".to_owned(),
        reason: "journal is empty".to_owned(),
    })?;
    let mut tx = JournalTx {
        tx_id: first.tx_id.clone(),
        begin: None,
        snapshot_committed: false,
        persisted: false,
        persistence_in_flight: false,
        ended: None,
        irreversible_barrier: None,
        operations: BTreeMap::new(),
    };
    let mut expected_seq = first.seq;
    let mut persistence_begin = None;
    for record in records {
        if record.tx_id != tx.tx_id {
            return Err(JournalError::InvalidField {
                field: "tx_id".to_owned(),
                reason: "one journal file contains multiple transaction ids".to_owned(),
            });
        }
        if record.seq != expected_seq {
            return Err(JournalError::InvalidSequence {
                expected: expected_seq,
                actual: record.seq,
            });
        }
        expected_seq = expected_seq.saturating_add(1);
        match &record.event {
            JournalEvent::TxBegin { .. } => {
                if tx.begin.is_some() {
                    return Err(JournalError::InvalidField {
                        field: "event".to_owned(),
                        reason: "duplicate TxBegin".to_owned(),
                    });
                }
                if record.seq != first.seq {
                    return Err(JournalError::InvalidField {
                        field: "event".to_owned(),
                        reason: "TxBegin must be the first journal event".to_owned(),
                    });
                }
                tx.begin = Some(record.event.clone());
            }
            JournalEvent::SnapshotCommitted { .. } => {
                if tx.begin.is_none() {
                    return Err(JournalError::InvalidField {
                        field: "event".to_owned(),
                        reason: "SnapshotCommitted precedes TxBegin".to_owned(),
                    });
                }
                if tx.snapshot_committed {
                    return Err(JournalError::InvalidField {
                        field: "event".to_owned(),
                        reason: "duplicate SnapshotCommitted".to_owned(),
                    });
                }
                tx.snapshot_committed = true;
            }
            JournalEvent::PersistBegin {
                generation,
                manifest_hash,
            } => {
                if tx.begin.is_none() {
                    return Err(JournalError::InvalidField {
                        field: "event".to_owned(),
                        reason: "PersistBegin precedes TxBegin".to_owned(),
                    });
                }
                if persistence_begin.is_some() {
                    return Err(JournalError::InvalidField {
                        field: "persistence".to_owned(),
                        reason: "overlapping PersistBegin".to_owned(),
                    });
                }
                persistence_begin = Some((*generation, *manifest_hash));
                tx.persistence_in_flight = true;
            }
            JournalEvent::PersistEnd {
                generation,
                manifest_hash,
            } => {
                let Some(expected) = persistence_begin.take() else {
                    return Err(JournalError::InvalidField {
                        field: "event".to_owned(),
                        reason: "PersistEnd has no matching PersistBegin".to_owned(),
                    });
                };
                if expected != (*generation, *manifest_hash) {
                    return Err(JournalError::InvalidField {
                        field: "persistence".to_owned(),
                        reason: "PersistEnd does not match PersistBegin".to_owned(),
                    });
                }
                tx.persisted = true;
                tx.persistence_in_flight = false;
            }
            JournalEvent::OpBegin { op } => {
                validate_operation_index(tx.begin.as_ref(), op.index)?;
                if let Some(existing) = tx.operations.get_mut(&op.index) {
                    if existing.op.id != op.id
                        || existing.op.group != op.group
                        || existing.op.key != op.key
                        || existing.op.input_digest != op.input_digest
                    {
                        return Err(JournalError::InvalidField {
                            field: "op".to_owned(),
                            reason: format!("operation {} changed between attempts", op.index),
                        });
                    }
                    existing.op.effect_handle = op.effect_handle.clone();
                    existing.state = ApplyOpState::Unknown;
                    existing.attempts = existing.attempts.saturating_add(1);
                    existing.last_error = None;
                } else {
                    tx.operations.insert(
                        op.index,
                        JournalOpState {
                            op: op.clone(),
                            state: ApplyOpState::Unknown,
                            compensation: if op.reversibility == Reversibility::Reversible
                                || op.reversibility == Reversibility::Compensable
                            {
                                CompensationState::NotStarted
                            } else {
                                CompensationState::NotNeeded
                            },
                            attempts: 1,
                            compensation_attempts: 0,
                            last_error: None,
                        },
                    );
                }
            }
            JournalEvent::OpApplied {
                op_index,
                effect_handle,
                ..
            } => {
                validate_operation_index(tx.begin.as_ref(), *op_index)?;
                let state = tx.operation_mut(*op_index)?;
                state.state = ApplyOpState::Applied;
                state.op.effect_handle = effect_handle.clone();
            }
            JournalEvent::OpFailed {
                op_index,
                error,
                effect_handle,
            } => {
                validate_operation_index(tx.begin.as_ref(), *op_index)?;
                let state = tx.operation_mut(*op_index)?;
                state.state = ApplyOpState::Failed;
                state.last_error = Some(error.clone());
                if effect_handle.is_some() {
                    state.op.effect_handle = effect_handle.clone();
                }
            }
            JournalEvent::IrreversibleBarrier { op_index } => {
                validate_operation_index(tx.begin.as_ref(), *op_index)?;
                tx.irreversible_barrier = Some(*op_index);
            }
            JournalEvent::CompensateBegin { op_index, .. } => {
                validate_operation_index(tx.begin.as_ref(), *op_index)?;
                let state = tx.operation_mut(*op_index)?;
                state.compensation = CompensationState::Unknown;
                state.compensation_attempts = state.compensation_attempts.saturating_add(1);
            }
            JournalEvent::Compensated { op_index, verified } => {
                validate_operation_index(tx.begin.as_ref(), *op_index)?;
                let state = tx.operation_mut(*op_index)?;
                state.compensation = if *verified {
                    CompensationState::Done
                } else {
                    CompensationState::Unknown
                };
            }
            JournalEvent::CompensateFailed { op_index, error } => {
                validate_operation_index(tx.begin.as_ref(), *op_index)?;
                let state = tx.operation_mut(*op_index)?;
                state.compensation = CompensationState::Failed;
                state.last_error = Some(error.clone());
            }
            JournalEvent::TxEnd { outcome } => {
                if tx.ended.is_some_and(|previous| {
                    previous != TxOutcome::PendingConfirmation
                        && !(previous == TxOutcome::RolledBack
                            && *outcome == TxOutcome::RecoveryRequired)
                }) {
                    return Err(JournalError::InvalidField {
                        field: "outcome".to_owned(),
                        reason: "transaction already has a terminal outcome".to_owned(),
                    });
                }
                tx.ended = Some(*outcome);
            }
        }
    }
    Ok(tx)
}

fn validate_operation_index(begin: Option<&JournalEvent>, index: u16) -> Result<(), JournalError> {
    let Some(JournalEvent::TxBegin { op_count, .. }) = begin else {
        return Err(JournalError::InvalidField {
            field: "event".to_owned(),
            reason: "operation event precedes TxBegin".to_owned(),
        });
    };
    if index >= *op_count {
        return Err(JournalError::InvalidField {
            field: "op_index".to_owned(),
            reason: format!("operation {index} is outside declared range 0..{op_count}"),
        });
    }
    Ok(())
}

/// Reconstructed transaction state used by recovery and doctor tooling.
#[derive(Clone, Debug, PartialEq)]
pub struct JournalTx {
    /// Transaction identity.
    pub tx_id: String,
    /// The transaction-begin event.
    pub begin: Option<JournalEvent>,
    /// Whether the candidate snapshot was committed.
    pub snapshot_committed: bool,
    /// Whether persistence completed.
    pub persisted: bool,
    /// Whether a persistence operation started without a durable completion.
    pub persistence_in_flight: bool,
    /// Terminal outcome, if present.
    pub ended: Option<TxOutcome>,
    /// Last irreversible barrier, if any.
    pub irreversible_barrier: Option<u16>,
    /// Per-operation state in execution order.
    pub operations: BTreeMap<u16, JournalOpState>,
}

impl JournalTx {
    /// Returns the recovery decision implied by the durable prefix.
    pub fn recovery_decision(&self) -> TxRecoveryDecision {
        if self.ended == Some(TxOutcome::PendingConfirmation) {
            return TxRecoveryDecision::PendingConfirmation;
        }
        if self.ended.is_some() {
            return TxRecoveryDecision::Completed;
        }
        if self.begin.is_none() || !self.snapshot_committed {
            return TxRecoveryDecision::Discard;
        }
        if self.persistence_in_flight {
            return TxRecoveryDecision::RecoveryRequired;
        }
        if let Some(barrier) = self.irreversible_barrier
            && self
                .operations
                .get(&barrier)
                .is_some_and(|operation| operation.state != ApplyOpState::Pending)
        {
            return if self
                .operations
                .get(&barrier)
                .is_some_and(|operation| operation.state == ApplyOpState::Applied)
            {
                TxRecoveryDecision::NeedsRestart
            } else {
                TxRecoveryDecision::RecoveryRequired
            };
        }
        if self.persisted && self.all_operations_applied() {
            return TxRecoveryDecision::Commit;
        }
        if self.operations.values().any(|operation| {
            operation.state == ApplyOpState::Unknown
                || operation.compensation == CompensationState::Unknown
                || operation.compensation == CompensationState::Failed
        }) {
            return TxRecoveryDecision::RecoveryRequired;
        }
        TxRecoveryDecision::Rollback
    }

    /// Returns whether the declared operation set completed successfully.
    #[must_use]
    pub fn all_operations_applied(&self) -> bool {
        let Some(JournalEvent::TxBegin { op_count, .. }) = self.begin.as_ref() else {
            return false;
        };
        self.operations.len() == usize::from(*op_count)
            && self
                .operations
                .values()
                .all(|operation| operation.state == ApplyOpState::Applied)
    }

    fn operation_mut(&mut self, index: u16) -> Result<&mut JournalOpState, JournalError> {
        self.operations
            .get_mut(&index)
            .ok_or_else(|| JournalError::InvalidField {
                field: "op_index".to_owned(),
                reason: format!("event references op {index} before OpBegin"),
            })
    }
}

fn encode_record(record: &JournalRecord) -> Result<String, JournalError> {
    let body = json!({
        "tx_id": record.tx_id,
        "seq": record.seq,
        "monotonic_ns": record.monotonic_ns.to_string(),
        "event": record.event.kind(),
        "payload": record.event.payload(),
    });
    let body_text =
        serde_json::to_string(&body).map_err(|error| JournalError::Json(error.to_string()))?;
    let checksum = checksum(&body_text);
    let line = match body {
        JsonValue::Object(mut object) => {
            object.insert(CHECKSUM_KEY.to_owned(), JsonValue::String(checksum));
            JsonValue::Object(object)
        }
        _ => unreachable!("journal body is always an object"),
    };
    let encoded =
        serde_json::to_string(&line).map_err(|error| JournalError::Json(error.to_string()))?;
    Ok(format!("{encoded}\n"))
}

fn decode_record(line: &[u8]) -> Result<JournalRecord, JournalError> {
    let value: JsonValue =
        serde_json::from_slice(line).map_err(|error| JournalError::Json(error.to_string()))?;
    let object = value
        .as_object()
        .ok_or_else(|| JournalError::Json("journal record is not an object".to_owned()))?;
    let checksum_value = object
        .get(CHECKSUM_KEY)
        .and_then(JsonValue::as_str)
        .ok_or_else(|| JournalError::InvalidField {
            field: CHECKSUM_KEY.to_owned(),
            reason: "missing checksum".to_owned(),
        })?;
    let mut body = object.clone();
    body.remove(CHECKSUM_KEY);
    let body_text = serde_json::to_string(&JsonValue::Object(body))
        .map_err(|error| JournalError::Json(error.to_string()))?;
    let expected = checksum(&body_text);
    if checksum_value != expected {
        let seq = object
            .get("seq")
            .and_then(JsonValue::as_u64)
            .unwrap_or(u64::MAX);
        return Err(JournalError::ChecksumMismatch { seq });
    }
    let tx_id = required_str_from(object, "tx_id")?.to_owned();
    let seq = required_u64_from(object, "seq")?;
    let monotonic_ns = required_str_from(object, "monotonic_ns")?
        .parse::<u128>()
        .map_err(|error| JournalError::InvalidField {
            field: "monotonic_ns".to_owned(),
            reason: error.to_string(),
        })?;
    let event = JournalEvent::from_parts(
        required_str_from(object, "event")?,
        object
            .get("payload")
            .ok_or_else(|| JournalError::InvalidField {
                field: "payload".to_owned(),
                reason: "missing payload".to_owned(),
            })?,
    )?;
    Ok(JournalRecord {
        tx_id,
        seq,
        monotonic_ns,
        event,
    })
}

fn checksum(body: &str) -> String {
    let digest = Sha256::digest(body.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn op_to_json(op: &ApplyOp) -> JsonValue {
    json!({
        "index": op.index,
        "id": op.id,
        "group": op.group,
        "key": op.key.as_str(),
        "action": op.action.as_str(),
        "reversibility": op.reversibility.as_str(),
        "idempotent": op.idempotent,
        "input_digest": op.input_digest,
        "effect_handle": op.effect_handle,
    })
}

fn op_from_json(value: &JsonValue) -> Result<ApplyOp, JournalError> {
    Ok(ApplyOp {
        index: required_u16(value, "index")?,
        id: required_str(value, "id")?.to_owned(),
        group: required_str(value, "group")?.to_owned(),
        key: SettingKey::new(required_str(value, "key")?.to_owned()),
        action: OpAction::from_string(required_str(value, "action")?.to_owned()),
        reversibility: Reversibility::parse(required_str(value, "reversibility")?)?,
        idempotent: required_bool(value, "idempotent")?,
        input_digest: required_u64(value, "input_digest")?,
        effect_handle: optional_string(value, "effect_handle")?,
    })
}

fn required_str<'a>(value: &'a JsonValue, field: &str) -> Result<&'a str, JournalError> {
    value
        .get(field)
        .and_then(JsonValue::as_str)
        .ok_or_else(|| JournalError::InvalidField {
            field: field.to_owned(),
            reason: "expected string".to_owned(),
        })
}

fn required_str_from<'a>(
    value: &'a Map<String, JsonValue>,
    field: &str,
) -> Result<&'a str, JournalError> {
    value
        .get(field)
        .and_then(JsonValue::as_str)
        .ok_or_else(|| JournalError::InvalidField {
            field: field.to_owned(),
            reason: "expected string".to_owned(),
        })
}

fn required_u64(value: &JsonValue, field: &str) -> Result<u64, JournalError> {
    value
        .get(field)
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| JournalError::InvalidField {
            field: field.to_owned(),
            reason: "expected unsigned integer".to_owned(),
        })
}

fn required_u64_from(value: &Map<String, JsonValue>, field: &str) -> Result<u64, JournalError> {
    value
        .get(field)
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| JournalError::InvalidField {
            field: field.to_owned(),
            reason: "expected unsigned integer".to_owned(),
        })
}

fn required_u16(value: &JsonValue, field: &str) -> Result<u16, JournalError> {
    u16::try_from(required_u64(value, field)?).map_err(|_| JournalError::InvalidField {
        field: field.to_owned(),
        reason: "outside u16".to_owned(),
    })
}

fn required_bool(value: &JsonValue, field: &str) -> Result<bool, JournalError> {
    value
        .get(field)
        .and_then(JsonValue::as_bool)
        .ok_or_else(|| JournalError::InvalidField {
            field: field.to_owned(),
            reason: "expected boolean".to_owned(),
        })
}

fn optional_u64(value: &JsonValue, field: &str) -> Result<Option<u64>, JournalError> {
    match value.get(field) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(value) => Ok(Some(value.as_u64().ok_or_else(|| {
            JournalError::InvalidField {
                field: field.to_owned(),
                reason: "expected unsigned integer or null".to_owned(),
            }
        })?)),
    }
}

fn optional_string(value: &JsonValue, field: &str) -> Result<Option<String>, JournalError> {
    match value.get(field) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(value) => Ok(Some(
            value
                .as_str()
                .ok_or_else(|| JournalError::InvalidField {
                    field: field.to_owned(),
                    reason: "expected string or null".to_owned(),
                })?
                .to_owned(),
        )),
    }
}

fn io_error(path: &Path, source: io::Error) -> JournalError {
    JournalError::Io {
        path: path.to_owned(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn op(index: u16, reversibility: Reversibility, idempotent: bool) -> ApplyOp {
        ApplyOp {
            index,
            id: format!("op-{index}"),
            group: "Display".to_owned(),
            key: SettingKey::new(format!("display.mode.{index}")),
            action: OpAction::Set,
            reversibility,
            idempotent,
            input_digest: 100 + u64::from(index),
            effect_handle: None,
        }
    }

    #[test]
    fn append_roundtrips_all_operation_events_with_checksum() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-1").unwrap();
        journal
            .append(JournalEvent::TxBegin {
                mode: TxMode::Direct,
                hash_before: 1,
                hash_after: 2,
                plan_digest: 3,
                op_count: 1,
                snapshot_gen: 4,
                last_confirmed: Some(3),
            })
            .unwrap();
        journal
            .append(JournalEvent::SnapshotCommitted { snapshot_gen: 5 })
            .unwrap();
        journal
            .append(JournalEvent::OpBegin {
                op: op(0, Reversibility::Compensable, true),
            })
            .unwrap();
        journal
            .append(JournalEvent::OpApplied {
                op_index: 0,
                effect_handle: Some("display:old".to_owned()),
                result_digest: 8,
            })
            .unwrap();
        journal
            .append(JournalEvent::TxEnd {
                outcome: TxOutcome::Committed,
            })
            .unwrap();
        let recovery = recover_journal(&path).unwrap();
        assert!(!recovery.truncated_tail);
        assert_eq!(recovery.records.len(), 5);
        let tx = reconstruct_transaction(&recovery.records).unwrap();
        assert_eq!(tx.ended, Some(TxOutcome::Committed));
        assert_eq!(tx.operations[&0].state, ApplyOpState::Applied);
    }

    #[test]
    fn truncated_or_tampered_tail_is_discarded_and_file_is_repaired() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-2").unwrap();
        journal
            .append(JournalEvent::TxBegin {
                mode: TxMode::Direct,
                hash_before: 0,
                hash_after: 1,
                plan_digest: 2,
                op_count: 0,
                snapshot_gen: 0,
                last_confirmed: None,
            })
            .unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{\"broken\":")
            .unwrap();
        let recovery = recover_journal(&path).unwrap();
        assert!(recovery.truncated_tail);
        assert_eq!(recovery.records.len(), 1);
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            recovery.valid_bytes
        );
    }

    #[test]
    fn checksum_corruption_discards_only_the_corrupt_tail_record() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-checksum").unwrap();
        for index in 0..2 {
            journal
                .append(JournalEvent::OpBegin {
                    op: op(index, Reversibility::Reversible, true),
                })
                .unwrap();
        }
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes
            .iter()
            .rposition(|byte| *byte == b'"')
            .expect("checksum line has a quoted field");
        bytes[last.saturating_sub(1)] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        let recovery = recover_journal(&path).unwrap();
        assert!(recovery.truncated_tail);
        assert_eq!(recovery.records.len(), 1);
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            recovery.valid_bytes
        );
    }

    #[test]
    fn inspection_reports_corrupt_tail_without_mutating_the_journal() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("inspect.journal");
        let mut journal = Journal::open(&path, "tx-inspect").unwrap();
        journal
            .append(JournalEvent::TxBegin {
                mode: TxMode::Direct,
                hash_before: 1,
                hash_after: 2,
                plan_digest: 3,
                op_count: 0,
                snapshot_gen: 4,
                last_confirmed: None,
            })
            .unwrap();
        drop(journal);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(b"{truncated");
        std::fs::write(&path, &bytes).unwrap();

        let inspection = inspect_journal(&path).unwrap();

        assert_eq!(inspection.records.len(), 1);
        assert!(inspection.invalid_tail);
        assert_eq!(inspection.file_bytes, bytes.len() as u64);
        assert!(inspection.valid_bytes < inspection.file_bytes);
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    #[test]
    fn in_flight_apply_and_compensation_are_unknown() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-3").unwrap();
        journal
            .append(JournalEvent::TxBegin {
                mode: TxMode::Direct,
                hash_before: 1,
                hash_after: 2,
                plan_digest: 3,
                op_count: 1,
                snapshot_gen: 4,
                last_confirmed: None,
            })
            .unwrap();
        journal
            .append(JournalEvent::SnapshotCommitted { snapshot_gen: 5 })
            .unwrap();
        journal
            .append(JournalEvent::OpBegin {
                op: op(0, Reversibility::Reversible, true),
            })
            .unwrap();
        let recovery = recover_journal(&path).unwrap();
        let tx = reconstruct_transaction(&recovery.records).unwrap();
        assert_eq!(tx.operations[&0].state, ApplyOpState::Unknown);
        assert_eq!(tx.recovery_decision(), TxRecoveryDecision::RecoveryRequired);

        journal
            .append(JournalEvent::OpApplied {
                op_index: 0,
                effect_handle: Some("effect".to_owned()),
                result_digest: 7,
            })
            .unwrap();
        journal
            .append(JournalEvent::CompensateBegin {
                op_index: 0,
                reason: "recovery".to_owned(),
            })
            .unwrap();
        let recovery = recover_journal(&path).unwrap();
        let tx = reconstruct_transaction(&recovery.records).unwrap();
        assert_eq!(tx.operations[&0].compensation, CompensationState::Unknown);
        assert_eq!(tx.recovery_decision(), TxRecoveryDecision::RecoveryRequired);
    }

    #[test]
    fn reconstruction_rejects_duplicate_or_mismatched_persistence_markers() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-invalid-persist").unwrap();
        for event in [
            JournalEvent::TxBegin {
                mode: TxMode::Direct,
                hash_before: 1,
                hash_after: 2,
                plan_digest: 3,
                op_count: 0,
                snapshot_gen: 4,
                last_confirmed: None,
            },
            JournalEvent::SnapshotCommitted { snapshot_gen: 5 },
            JournalEvent::PersistBegin {
                generation: 6,
                manifest_hash: 7,
            },
            JournalEvent::PersistEnd {
                generation: 6,
                manifest_hash: 8,
            },
        ] {
            journal.append(event).unwrap();
        }
        assert!(reconstruct_transaction(&recover_journal(&path).unwrap().records).is_err());

        journal
            .append(JournalEvent::PersistEnd {
                generation: 6,
                manifest_hash: 7,
            })
            .unwrap();
        assert!(reconstruct_transaction(&recover_journal(&path).unwrap().records).is_err());
    }

    #[test]
    fn reconstruction_rejects_out_of_range_operation_events() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-invalid-op").unwrap();
        for event in [
            JournalEvent::TxBegin {
                mode: TxMode::Direct,
                hash_before: 1,
                hash_after: 2,
                plan_digest: 3,
                op_count: 1,
                snapshot_gen: 4,
                last_confirmed: None,
            },
            JournalEvent::SnapshotCommitted { snapshot_gen: 5 },
            JournalEvent::OpBegin {
                op: op(1, Reversibility::Reversible, true),
            },
        ] {
            journal.append(event).unwrap();
        }
        assert!(reconstruct_transaction(&recover_journal(&path).unwrap().records).is_err());
    }

    #[test]
    fn irreversible_barrier_requires_restart_after_applied_op() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-4").unwrap();
        for event in [
            JournalEvent::TxBegin {
                mode: TxMode::Direct,
                hash_before: 1,
                hash_after: 2,
                plan_digest: 3,
                op_count: 1,
                snapshot_gen: 4,
                last_confirmed: None,
            },
            JournalEvent::SnapshotCommitted { snapshot_gen: 5 },
            JournalEvent::OpBegin {
                op: op(0, Reversibility::Irreversible, false),
            },
            JournalEvent::IrreversibleBarrier { op_index: 0 },
            JournalEvent::OpApplied {
                op_index: 0,
                effect_handle: None,
                result_digest: 7,
            },
        ] {
            journal.append(event).unwrap();
        }
        let recovery = recover_journal(&path).unwrap();
        let tx = reconstruct_transaction(&recovery.records).unwrap();
        assert_eq!(tx.recovery_decision(), TxRecoveryDecision::NeedsRestart);
    }

    #[test]
    fn incomplete_persistence_is_recovery_required() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-persist-in-flight").unwrap();
        journal
            .append(JournalEvent::TxBegin {
                mode: TxMode::Direct,
                hash_before: 1,
                hash_after: 2,
                plan_digest: 3,
                op_count: 0,
                snapshot_gen: 4,
                last_confirmed: None,
            })
            .unwrap();
        journal
            .append(JournalEvent::SnapshotCommitted { snapshot_gen: 5 })
            .unwrap();
        journal
            .append(JournalEvent::PersistBegin {
                generation: 5,
                manifest_hash: 6,
            })
            .unwrap();
        let tx = reconstruct_transaction(&recover_journal(&path).unwrap().records).unwrap();
        assert!(tx.persistence_in_flight);
        assert_eq!(tx.recovery_decision(), TxRecoveryDecision::RecoveryRequired);
    }

    #[test]
    fn reconstruction_allows_direct_failure_before_all_planned_operations_begin() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("partial.journal");
        let mut journal = Journal::open(&path, "tx-partial").unwrap();
        journal
            .append(JournalEvent::TxBegin {
                mode: TxMode::Direct,
                hash_before: 1,
                hash_after: 2,
                plan_digest: 3,
                op_count: 2,
                snapshot_gen: 4,
                last_confirmed: None,
            })
            .unwrap();
        journal
            .append(JournalEvent::SnapshotCommitted { snapshot_gen: 5 })
            .unwrap();
        journal
            .append(JournalEvent::PersistBegin {
                generation: 6,
                manifest_hash: 7,
            })
            .unwrap();
        journal
            .append(JournalEvent::PersistEnd {
                generation: 6,
                manifest_hash: 7,
            })
            .unwrap();
        journal
            .append(JournalEvent::OpBegin {
                op: op(0, Reversibility::Reversible, true),
            })
            .unwrap();
        journal
            .append(JournalEvent::OpApplied {
                op_index: 0,
                effect_handle: None,
                result_digest: 0,
            })
            .unwrap();
        journal
            .append(JournalEvent::OpBegin {
                op: op(1, Reversibility::Reversible, true),
            })
            .unwrap();
        journal
            .append(JournalEvent::OpFailed {
                op_index: 1,
                error: "expected".into(),
                effect_handle: None,
            })
            .unwrap();
        assert!(reconstruct_transaction(&recover_journal(&path).unwrap().records).is_ok());
    }

    #[test]
    fn journal_does_not_contain_setting_values() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("apply.journal");
        let mut journal = Journal::open(&path, "tx-secret").unwrap();
        journal
            .append(JournalEvent::OpBegin {
                op: ApplyOp {
                    input_digest: 0xdead_beef,
                    ..op(0, Reversibility::Reversible, true)
                },
            })
            .unwrap();
        let text = std::fs::read_to_string(path).unwrap();
        assert!(!text.contains("dead"));
        assert!(!text.contains("old_value"));
    }
}
