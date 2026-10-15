//! Durable std-backed journal and atomic snapshot files.
//!
//! The journal keeps the deterministic AppendOnlyLog as its in-memory index
//! and stores each transaction as one self-checking record. Opening a journal
//! replays the valid prefix and truncates only the first incomplete or invalid
//! tail record. A snapshot is encoded completely before it is written to a
//! sibling temporary file, flushed, synced, and renamed into place.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::hash::checksum32;
use crate::{
    decode_snapshot, decode_transaction, encode_snapshot, encode_transaction, AppendOnlyLog,
    CodecError, LogError, Snapshot, Transaction,
};

mod journal_write;
use journal_write::JournalWriter;

#[cfg(test)]
mod journal_fault_tests;
#[cfg(test)]
pub(crate) mod test_support;

const JOURNAL_MAGIC: [u8; 4] = *b"PWRJ";
const JOURNAL_VERSION: u32 = 1;
const JOURNAL_HEADER_LEN: u64 = 8;
const RECORD_HEADER_LEN: u64 = 12;
const RECORD_TRAILER_LEN: u64 = 4;
const MAX_RECORD_PAYLOAD: u32 = 16 * 1024 * 1024;
/// Maximum number of transactions returned by one replay call.
pub const MAX_REPLAY_BATCH: usize = 4096;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A durable event-log cursor pointing at the next sequence to replay.
///
/// Cursors are logical, not byte offsets. They remain stable across reopen
/// and tail recovery as long as the valid journal prefix is retained.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct EventCursor {
    next_sequence: u64,
}

impl EventCursor {
    /// Creates a cursor positioned before `next_sequence`.
    pub const fn new(next_sequence: u64) -> Self {
        Self { next_sequence }
    }

    /// Returns the next transaction sequence this cursor will replay.
    pub const fn next_sequence(self) -> u64 {
        self.next_sequence
    }
}

/// A bounded replay result from an [`EventLogBackend`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayBatch {
    events: Vec<Transaction>,
    next_cursor: EventCursor,
    end_cursor: EventCursor,
}

impl ReplayBatch {
    fn new(events: Vec<Transaction>, next_cursor: EventCursor, end_cursor: EventCursor) -> Self {
        Self {
            events,
            next_cursor,
            end_cursor,
        }
    }

    /// Returns the replayed transactions in append order.
    pub fn events(&self) -> &[Transaction] {
        &self.events
    }

    /// Consumes the batch and returns its owned transactions.
    pub fn into_events(self) -> Vec<Transaction> {
        self.events
    }

    /// Returns the cursor immediately after the returned events.
    pub const fn next_cursor(&self) -> EventCursor {
        self.next_cursor
    }

    /// Returns the current end cursor observed by this replay call.
    pub const fn end_cursor(&self) -> EventCursor {
        self.end_cursor
    }

    /// Returns whether another replay call can return more events.
    pub const fn has_more(&self) -> bool {
        self.next_cursor.next_sequence < self.end_cursor.next_sequence
    }
}

/// Provider contract for the authoritative durable event stream.
///
/// The default implementation is [`FileJournal`]. Implementations keep
/// append order, use the persist [`Transaction`] as the event unit, and make
/// replay explicitly bounded by `limit`.
pub trait EventLogBackend {
    /// Backend-specific failure type.
    type Error;

    /// Appends one transaction durably.
    fn append(&mut self, transaction: Transaction) -> Result<(), Self::Error>;

    /// Appends a validated batch durably.
    fn append_batch(&mut self, transactions: &[Transaction]) -> Result<(), Self::Error>;

    /// Returns the cursor at the current end of the log.
    fn cursor(&self) -> EventCursor;

    /// Replays at most `limit` events from `cursor`.
    fn replay(&self, cursor: EventCursor, limit: usize) -> Result<ReplayBatch, Self::Error>;

    /// Flushes userspace/OS buffered bytes without requiring storage sync.
    fn flush(&mut self) -> Result<(), Self::Error>;

    /// Requests storage durability for all bytes written so far.
    fn sync(&mut self) -> Result<(), Self::Error>;
}

/// Why a journal tail was discarded while opening it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecoveryReason {
    /// The record header, payload, or checksum was cut short.
    Truncated,
    /// The record checksum did not match its header and payload.
    ChecksumMismatch,
    /// The payload was not a valid transaction frame.
    InvalidPayload,
    /// The record sequence and transaction sequence disagreed.
    SequenceMismatch,
    /// The transaction violated append-only log ordering or uniqueness.
    LogValidation,
}

/// Details about a valid-prefix recovery performed during journal opening.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JournalRecovery {
    truncated_bytes: u64,
    reason: RecoveryReason,
}

impl JournalRecovery {
    /// Returns the number of bytes removed from the invalid tail.
    pub const fn truncated_bytes(self) -> u64 {
        self.truncated_bytes
    }

    /// Returns the reason that caused tail recovery.
    pub const fn reason(self) -> RecoveryReason {
        self.reason
    }
}

/// Errors from opening or appending a durable transaction journal.
#[derive(Debug)]
pub enum JournalError {
    /// The operating system rejected a file operation.
    Io(io::Error),
    /// Another handle holds an incompatible operating-system file lock.
    Locked,
    /// An exclusive operating-system file lock could not be acquired.
    Lock(io::Error),
    /// A previous I/O failure requires dropping and reopening this journal.
    Poisoned,
    /// An operation failed and its durable rollback also failed.
    RollbackFailed {
        /// The original append failure.
        operation: Box<JournalError>,
        /// The error encountered while restoring the previous file boundary.
        rollback: io::Error,
    },
    /// The file was shorter than its fixed version header.
    TruncatedHeader,
    /// The file magic did not identify a Prism world journal.
    InvalidHeader,
    /// The file was written by a journal version this crate cannot read.
    UnsupportedVersion(u32),
    /// A record declared a payload larger than the safety limit.
    RecordTooLarge(u32),
    /// A transaction could not be encoded or decoded.
    Codec(CodecError),
    /// The in-memory log rejected a transaction.
    Log(LogError),
    /// The record sequence did not match the transaction sequence.
    RecordSequenceMismatch {
        /// Sequence stored in the journal record header.
        record: u64,
        /// Sequence stored in the decoded transaction.
        transaction: u64,
    },
    /// A replay cursor points after the current durable log end.
    CursorAhead {
        /// Sequence requested by the cursor.
        cursor: u64,
        /// First sequence after the current log.
        end: u64,
    },
    /// A replay request exceeds the backend's fixed safety bound.
    ReplayLimitExceeded {
        /// Number of events requested by the caller.
        requested: usize,
        /// Maximum number of events returned by one replay call.
        max: usize,
    },
}

impl fmt::Display for JournalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "journal I/O error: {error}"),
            Self::Locked => formatter.write_str("journal is locked by another handle"),
            Self::Lock(error) => write!(formatter, "journal exclusive lock error: {error}"),
            Self::Poisoned => {
                formatter.write_str("journal is poisoned; drop and reopen to recover")
            }
            Self::RollbackFailed {
                operation,
                rollback,
            } => write!(
                formatter,
                "{operation}; journal rollback also failed: {rollback}"
            ),
            Self::TruncatedHeader => formatter.write_str("journal version header is truncated"),
            Self::InvalidHeader => formatter.write_str("journal version header is invalid"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported journal version {version}")
            }
            Self::RecordTooLarge(length) => {
                write!(formatter, "journal record is too large: {length}")
            }
            Self::Codec(error) => write!(formatter, "journal codec error: {error:?}"),
            Self::Log(error) => write!(formatter, "journal log error: {error:?}"),
            Self::RecordSequenceMismatch {
                record,
                transaction,
            } => write!(
                formatter,
                "journal record sequence {record} differs from transaction sequence {transaction}"
            ),
            Self::CursorAhead { cursor, end } => {
                write!(
                    formatter,
                    "journal cursor {cursor} is ahead of log end {end}"
                )
            }
            Self::ReplayLimitExceeded { requested, max } => write!(
                formatter,
                "journal replay limit {requested} exceeds maximum {max}"
            ),
        }
    }
}

impl std::error::Error for JournalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) | Self::Lock(error) => Some(error),
            Self::RollbackFailed { operation, .. } => Some(operation.as_ref()),
            _ => None,
        }
    }
}

impl From<io::Error> for JournalError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<CodecError> for JournalError {
    fn from(error: CodecError) -> Self {
        Self::Codec(error)
    }
}

impl From<LogError> for JournalError {
    fn from(error: LogError) -> Self {
        Self::Log(error)
    }
}

/// Errors from encoding, atomically replacing, or restoring a snapshot file.
#[derive(Debug)]
pub enum SnapshotError {
    /// The operating system rejected a file operation.
    Io(io::Error),
    /// The snapshot bytes failed the versioned codec validation.
    Codec(CodecError),
    /// The target path has no usable file name for its sibling temporary file.
    InvalidPath,
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "snapshot I/O error: {error}"),
            Self::Codec(error) => write!(formatter, "snapshot codec error: {error:?}"),
            Self::InvalidPath => formatter.write_str("snapshot path has no usable file name"),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl From<io::Error> for SnapshotError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<CodecError> for SnapshotError {
    fn from(error: CodecError) -> Self {
        Self::Codec(error)
    }
}

/// A durable append-only file journal backed by a validated in-memory log.
///
/// Holds an exclusive OS file lock until dropped. All journal users must honor
/// this lock; replacing the path or writing through an unlocked handle is not
/// supported. An I/O failure poisons the handle even if rollback succeeds.
pub struct FileJournal {
    file: File,
    path: PathBuf,
    log: AppendOnlyLog,
    writer: JournalWriter,
    recovery: Option<JournalRecovery>,
}

impl FileJournal {
    /// Opens or creates a journal and recovers its valid transaction prefix.
    /// Returns [`JournalError::Locked`] if another handle owns the file lock,
    /// and [`JournalError::Lock`] if the OS cannot provide the lock. Locking
    /// precedes all header reads, initialization, and tail recovery.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, JournalError> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)?;
        file.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => JournalError::Locked,
            TryLockError::Error(error) => JournalError::Lock(error),
        })?;
        let file_len = file.metadata()?.len();

        if file_len == 0 {
            file.write_all(&journal_header())?;
            file.flush()?;
            file.sync_all()?;
            return Ok(Self {
                file,
                path,
                log: AppendOnlyLog::new(),
                writer: JournalWriter::new(JOURNAL_HEADER_LEN),
                recovery: None,
            });
        }

        if file_len < JOURNAL_HEADER_LEN {
            return Err(JournalError::TruncatedHeader);
        }
        file.seek(SeekFrom::Start(0))?;
        let mut header = [0_u8; 8];
        file.read_exact(&mut header)?;
        if header[..4] != JOURNAL_MAGIC {
            return Err(JournalError::InvalidHeader);
        }
        let version = u32::from_le_bytes(header[4..8].try_into().expect("fixed header"));
        if version != JOURNAL_VERSION {
            return Err(JournalError::UnsupportedVersion(version));
        }

        let mut log = AppendOnlyLog::new();
        let mut offset = JOURNAL_HEADER_LEN;
        let mut recovery = None;
        while offset < file_len {
            let record_start = offset;
            let remaining = file_len - record_start;
            if remaining < RECORD_HEADER_LEN + RECORD_TRAILER_LEN {
                recovery = Some((record_start, RecoveryReason::Truncated));
                break;
            }

            file.seek(SeekFrom::Start(record_start))?;
            let mut record_header = [0_u8; 12];
            if !read_exact_or_truncated(&mut file, &mut record_header)? {
                recovery = Some((record_start, RecoveryReason::Truncated));
                break;
            }
            let payload_len =
                u32::from_le_bytes(record_header[..4].try_into().expect("fixed record"));
            if payload_len > MAX_RECORD_PAYLOAD {
                recovery = Some((record_start, RecoveryReason::InvalidPayload));
                break;
            }
            let total_len = RECORD_HEADER_LEN
                .checked_add(u64::from(payload_len))
                .and_then(|value| value.checked_add(RECORD_TRAILER_LEN))
                .expect("bounded record length");
            if total_len > remaining {
                recovery = Some((record_start, RecoveryReason::Truncated));
                break;
            }

            let record_sequence = u64::from_le_bytes(
                record_header[4..12]
                    .try_into()
                    .expect("fixed record sequence"),
            );
            let mut payload = vec![0_u8; payload_len as usize];
            if !read_exact_or_truncated(&mut file, &mut payload)? {
                recovery = Some((record_start, RecoveryReason::Truncated));
                break;
            }
            let mut checksum_bytes = [0_u8; 4];
            if !read_exact_or_truncated(&mut file, &mut checksum_bytes)? {
                recovery = Some((record_start, RecoveryReason::Truncated));
                break;
            }
            let expected_checksum = u32::from_le_bytes(checksum_bytes);
            let mut checksum_input = Vec::with_capacity(12 + payload.len());
            checksum_input.extend_from_slice(&record_header);
            checksum_input.extend_from_slice(&payload);
            if checksum32(&checksum_input) != expected_checksum {
                recovery = Some((record_start, RecoveryReason::ChecksumMismatch));
                break;
            }

            let transaction = match decode_transaction(&payload) {
                Ok(transaction) => transaction,
                Err(_) => {
                    recovery = Some((record_start, RecoveryReason::InvalidPayload));
                    break;
                }
            };
            if transaction.sequence() != record_sequence {
                recovery = Some((record_start, RecoveryReason::SequenceMismatch));
                break;
            }
            if log.validate_append(&transaction).is_err() {
                recovery = Some((record_start, RecoveryReason::LogValidation));
                break;
            }
            log.append(transaction).map_err(JournalError::Log)?;
            offset = record_start + total_len;
        }

        let recovery = recovery.map(|(valid_end, reason)| {
            let recovered = JournalRecovery {
                truncated_bytes: file_len - valid_end,
                reason,
            };
            (valid_end, recovered)
        });
        let (end_offset, recovery) = if let Some((valid_end, recovery)) = recovery {
            file.set_len(valid_end)?;
            file.flush()?;
            file.sync_all()?;
            (valid_end, Some(recovery))
        } else {
            (offset, None)
        };
        file.seek(SeekFrom::Start(end_offset))?;

        Ok(Self {
            file,
            path,
            log,
            writer: JournalWriter::new(end_offset),
            recovery,
        })
    }

    /// Returns the path backing this journal.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the currently loaded validated log.
    /// After an I/O failure this remains the pre-failure view. Reopening may
    /// recover additional complete records if rollback could not finish.
    pub const fn log(&self) -> &AppendOnlyLog {
        &self.log
    }

    /// Returns details of the tail recovery performed during Self::open.
    pub const fn recovery(&self) -> Option<JournalRecovery> {
        self.recovery
    }

    /// Returns the cursor at the current end of the validated log.
    pub const fn cursor(&self) -> EventCursor {
        EventCursor::new(self.log.next_sequence())
    }

    /// Replays a bounded range beginning at `cursor`.
    ///
    /// The cursor denotes the next sequence to return. A cursor equal to the
    /// current end is valid and returns an empty batch. A cursor ahead of the
    /// end is rejected so consumers cannot silently skip an unavailable
    /// prefix. `limit` may be zero, but must not exceed
    /// [`MAX_REPLAY_BATCH`].
    pub fn replay(&self, cursor: EventCursor, limit: usize) -> Result<ReplayBatch, JournalError> {
        if limit > MAX_REPLAY_BATCH {
            return Err(JournalError::ReplayLimitExceeded {
                requested: limit,
                max: MAX_REPLAY_BATCH,
            });
        }
        let end = self.cursor();
        if cursor > end {
            return Err(JournalError::CursorAhead {
                cursor: cursor.next_sequence(),
                end: end.next_sequence(),
            });
        }
        let start =
            usize::try_from(cursor.next_sequence()).map_err(|_| JournalError::CursorAhead {
                cursor: cursor.next_sequence(),
                end: end.next_sequence(),
            })?;
        let stop = start.saturating_add(limit).min(self.log.len());
        let events = self.log.as_slice()[start..stop].to_vec();
        let next_cursor = EventCursor::new(cursor.next_sequence() + events.len() as u64);
        Ok(ReplayBatch::new(events, next_cursor, end))
    }

    /// Appends one transaction, flushes it, and calls sync_all before return.
    /// Any I/O failure poisons this handle until it is dropped and reopened.
    pub fn append(&mut self, transaction: Transaction) -> Result<(), JournalError> {
        self.writer
            .append(&mut self.file, &mut self.log, transaction)
    }

    /// Appends a batch with one flush and one sync_all for better throughput.
    /// Any I/O failure poisons this handle. If rollback fails, reopening may
    /// recover a complete prefix of the batch; batches are not crash-atomic.
    pub fn append_batch(&mut self, transactions: &[Transaction]) -> Result<(), JournalError> {
        self.writer
            .append_batch(&mut self.file, &mut self.log, transactions)
    }

    /// Flushes the journal's operating-system file handle without syncing it.
    /// Rejects poisoned handles and poisons this handle on an I/O failure.
    pub fn flush(&mut self) -> Result<(), JournalError> {
        self.writer.flush(&mut self.file)
    }

    /// Requests durable persistence of all journal bytes written so far.
    /// Rejects poisoned handles and poisons this handle on an I/O failure.
    pub fn sync(&mut self) -> Result<(), JournalError> {
        self.writer.sync(&mut self.file)
    }
}

impl EventLogBackend for FileJournal {
    type Error = JournalError;

    fn append(&mut self, transaction: Transaction) -> Result<(), Self::Error> {
        FileJournal::append(self, transaction)
    }

    fn append_batch(&mut self, transactions: &[Transaction]) -> Result<(), Self::Error> {
        FileJournal::append_batch(self, transactions)
    }

    fn cursor(&self) -> EventCursor {
        FileJournal::cursor(self)
    }

    fn replay(&self, cursor: EventCursor, limit: usize) -> Result<ReplayBatch, Self::Error> {
        FileJournal::replay(self, cursor, limit)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        FileJournal::flush(self)
    }

    fn sync(&mut self) -> Result<(), Self::Error> {
        FileJournal::sync(self)
    }
}

/// The built-in durable event-log backend.
pub type DefaultEventLogBackend = FileJournal;

/// Descriptive alias for the file-backed event-log backend.
pub type FileEventLogBackend = FileJournal;

/// Stateless atomic snapshot writer and loader.
pub struct SnapshotStore;

impl SnapshotStore {
    /// Writes a complete snapshot through a flushed and synced sibling file.
    pub fn save(path: impl AsRef<Path>, snapshot: &Snapshot) -> Result<(), SnapshotError> {
        let path = path.as_ref();
        let bytes = encode_snapshot(snapshot)?;
        let temporary = create_temporary_path(path)?;
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            file.write_all(&bytes)?;
            file.flush()?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, path)?;
            sync_parent_directory(path)?;
            Ok::<(), io::Error>(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result.map_err(SnapshotError::Io)
    }

    /// Loads and validates a snapshot written by Self::save.
    pub fn load(path: impl AsRef<Path>) -> Result<Snapshot, SnapshotError> {
        let bytes = fs::read(path)?;
        decode_snapshot(&bytes).map_err(SnapshotError::Codec)
    }
}

fn journal_header() -> [u8; 8] {
    let mut header = [0_u8; 8];
    header[..4].copy_from_slice(&JOURNAL_MAGIC);
    header[4..].copy_from_slice(&JOURNAL_VERSION.to_le_bytes());
    header
}

fn encode_record(transaction: &Transaction) -> Result<Vec<u8>, JournalError> {
    let payload = encode_transaction(transaction)?;
    let payload_len =
        u32::try_from(payload.len()).map_err(|_| JournalError::RecordTooLarge(u32::MAX))?;
    if payload_len > MAX_RECORD_PAYLOAD {
        return Err(JournalError::RecordTooLarge(payload_len));
    }
    let mut record = Vec::with_capacity(12 + payload.len() + 4);
    record.extend_from_slice(&payload_len.to_le_bytes());
    record.extend_from_slice(&transaction.sequence().to_le_bytes());
    record.extend_from_slice(&payload);
    record.extend_from_slice(&checksum32(&record).to_le_bytes());
    Ok(record)
}

fn read_exact_or_truncated(file: &mut File, bytes: &mut [u8]) -> io::Result<bool> {
    match file.read_exact(bytes) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => Ok(false),
        Err(error) => Err(error),
    }
}

fn create_temporary_path(path: &Path) -> Result<PathBuf, SnapshotError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path.file_name().ok_or(SnapshotError::InvalidPath)?;
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary_name = std::format!(
        ".{}.tmp-{}-{}",
        name.to_string_lossy(),
        std::process::id(),
        counter
    );
    Ok(parent.join(temporary_name))
}

fn sync_parent_directory(path: &Path) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    #[cfg(unix)]
    {
        File::open(parent)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = parent;
        Ok(())
    }
}
