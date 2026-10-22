//! Bounded durable provider seam for logical replication events.
//!
//! This module stores opaque logical-event payloads. It deliberately does not
//! reinterpret them as `prism_world_persist::Transaction`: persist remains the
//! owner of world override-delta truth, while replication owns the transport-
//! independent replay envelope. A provider can therefore persist the encoded
//! [`crate::replay::ReplayLog`] without creating a second world-state format.
//!
//! The contract is intentionally smaller than a distributed event grid:
//! append order is local and monotonic, appends are idempotent by caller-supplied
//! key, replay is bounded and cursor-based, and `sync` is the only durability
//! barrier. There is no implicit eviction, network replication, DLQ, or
//! cross-process membership protocol here.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

use crate::replay::{ReplayError, ReplayLog, ReplayRecord, DEFAULT_MAX_PAYLOAD};

/// Maximum number of opaque events returned by one replay operation.
pub const MAX_EVENT_LOG_BATCH: usize = 4_096;

#[cfg(feature = "std")]
const FILE_MAGIC: [u8; 4] = *b"PREL";
#[cfg(feature = "std")]
const FILE_VERSION: u8 = 2;
#[cfg(feature = "std")]
const FILE_HEADER_LEN: usize = 20;
#[cfg(feature = "std")]
const FILE_RECORD_HEADER_LEN: usize = 8 + 16 + 4;
#[cfg(feature = "std")]
const FILE_RECORD_TRAILER_LEN: usize = 4;

/// Logical cursor pointing at the next event sequence to replay.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct EventLogCursor {
    next_sequence: u64,
}

impl EventLogCursor {
    /// Construct a cursor positioned before `next_sequence`.
    #[must_use]
    pub const fn new(next_sequence: u64) -> Self {
        Self { next_sequence }
    }

    /// Return the next sequence visible to this cursor.
    #[must_use]
    pub const fn next_sequence(self) -> u64 {
        self.next_sequence
    }
}

/// One opaque durable event-log entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventLogEntry {
    /// Local append sequence assigned by the backend.
    pub sequence: u64,
    /// Caller-supplied idempotency key.
    pub idempotency_key: u128,
    /// Encoded logical-event payload.
    pub payload: Vec<u8>,
}

/// Outcome of an idempotent append.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventLogAppend {
    /// Cursor immediately after the existing or newly inserted event.
    pub next_cursor: EventLogCursor,
    /// Whether a new entry was inserted.
    pub inserted: bool,
}

/// A bounded replay result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventLogBatch {
    entries: Vec<EventLogEntry>,
    next_cursor: EventLogCursor,
    end_cursor: EventLogCursor,
}

impl EventLogBatch {
    fn new(
        entries: Vec<EventLogEntry>,
        next_cursor: EventLogCursor,
        end_cursor: EventLogCursor,
    ) -> Self {
        Self {
            entries,
            next_cursor,
            end_cursor,
        }
    }

    /// Borrow entries in append order.
    #[must_use]
    pub fn entries(&self) -> &[EventLogEntry] {
        &self.entries
    }

    /// Consume the batch and return owned entries.
    #[must_use]
    pub fn into_entries(self) -> Vec<EventLogEntry> {
        self.entries
    }

    /// Cursor immediately after the returned entries.
    #[must_use]
    pub const fn next_cursor(&self) -> EventLogCursor {
        self.next_cursor
    }

    /// End cursor observed when the replay started.
    #[must_use]
    pub const fn end_cursor(&self) -> EventLogCursor {
        self.end_cursor
    }

    /// Whether another bounded replay call may return entries.
    #[must_use]
    pub const fn has_more(&self) -> bool {
        self.next_cursor.next_sequence < self.end_cursor.next_sequence
    }
}

/// Provider contract for a bounded logical event stream.
pub trait EventLogBackend {
    /// Provider-specific error type.
    type Error;

    /// Append one payload, deduplicating an already-seen idempotency key.
    /// Reuse succeeds only when the payload is byte-identical; a different
    /// payload for the same retained key is an explicit conflict.
    fn append(
        &mut self,
        idempotency_key: u128,
        payload: &[u8],
    ) -> Result<EventLogAppend, Self::Error>;

    /// Return the current end cursor.
    fn cursor(&self) -> EventLogCursor;

    /// First retained sequence. Older cursors must fail, never skip silently.
    fn start_cursor(&self) -> EventLogCursor {
        EventLogCursor::default()
    }

    /// Remaining entry slots, if the provider exposes a fixed capacity.
    fn remaining_capacity(&self) -> Option<usize> {
        None
    }

    /// Atomically replace the retained stream with a caller-supplied checkpoint
    /// at `cursor() - 1`, preserving the end cursor. The caller must encode all
    /// state needed to recover the discarded prefix. Success includes a sync
    /// barrier; failure may have installed the checkpoint, so reopen or retry the
    /// *same* checkpoint before proceeding. `None` means unsupported, not success.
    /// Idempotency keys before the checkpoint cease to be retained.
    fn checkpoint(&mut self, _entry: &EventLogEntry) -> Option<Result<(), Self::Error>> {
        None
    }

    /// Replay at most `limit` entries from `cursor`.
    fn replay(&self, cursor: EventLogCursor, limit: usize) -> Result<EventLogBatch, Self::Error>;

    /// Flush userspace/OS buffers without promising stable storage.
    fn flush(&mut self) -> Result<(), Self::Error>;

    /// Request a durability barrier for all appended bytes.
    fn sync(&mut self) -> Result<(), Self::Error>;
}

/// Errors from the provider-neutral bounded event-log implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventLogError {
    /// No more entries can be admitted.
    CapacityExceeded {
        /// Maximum number of entries retained by the backend.
        limit: usize,
    },
    /// An idempotency key was reused for a different payload.
    IdempotencyConflict {
        /// Key already assigned to another payload.
        key: u128,
    },
    /// An opaque payload exceeds the configured bound.
    PayloadTooLarge {
        /// Supplied payload length.
        len: usize,
        /// Configured maximum payload length.
        max: usize,
    },
    /// A replay request exceeds the fixed batch bound.
    ReplayLimitExceeded {
        /// Requested replay batch size.
        requested: usize,
        /// Maximum replay batch size.
        max: usize,
    },
    /// A cursor points after the current end.
    CursorAhead {
        /// Requested cursor sequence.
        cursor: u64,
        /// Current end sequence.
        end: u64,
    },
    /// The requested prefix has been replaced by a checkpoint.
    CursorExpired {
        /// Requested sequence.
        cursor: u64,
        /// First retained sequence.
        start: u64,
    },
    /// A checkpoint does not replace the last retained sequence.
    InvalidCheckpoint,
    /// The sequence space is exhausted.
    SequenceExhausted,
}

impl fmt::Display for EventLogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdempotencyConflict { key } => {
                write!(
                    formatter,
                    "event log idempotency key {key} conflicts with its payload"
                )
            }
            Self::CapacityExceeded { limit } => {
                write!(formatter, "event log capacity {limit} is full")
            }
            Self::PayloadTooLarge { len, max } => {
                write!(formatter, "event log payload {len} exceeds maximum {max}")
            }
            Self::ReplayLimitExceeded { requested, max } => write!(
                formatter,
                "event log replay limit {requested} exceeds maximum {max}"
            ),
            Self::CursorAhead { cursor, end } => {
                write!(formatter, "event log cursor {cursor} is ahead of end {end}")
            }
            Self::CursorExpired { cursor, start } => {
                write!(
                    formatter,
                    "event log cursor {cursor} precedes retained start {start}"
                )
            }
            Self::InvalidCheckpoint => formatter.write_str("invalid event log checkpoint"),
            Self::SequenceExhausted => formatter.write_str("event log sequence is exhausted"),
        }
    }
}

/// In-memory provider used by deterministic tests and local authority harnesses.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryEventLogBackend {
    base_sequence: u64,
    entries: Vec<EventLogEntry>,
    keys: BTreeMap<u128, u64>,
    max_entries: usize,
    max_payload: usize,
}

impl MemoryEventLogBackend {
    /// Create an empty bounded provider.
    #[must_use]
    pub fn new(max_entries: usize, max_payload: usize) -> Self {
        Self {
            base_sequence: 0,
            entries: Vec::new(),
            keys: BTreeMap::new(),
            max_entries,
            max_payload,
        }
    }

    /// Create a provider with the standard replay payload bound.
    #[must_use]
    pub fn with_capacity(max_entries: usize) -> Self {
        Self::new(max_entries, DEFAULT_MAX_PAYLOAD)
    }

    /// Number of entries currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no entries are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Configured entry capacity.
    #[must_use]
    pub const fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Configured payload capacity.
    #[must_use]
    pub const fn max_payload(&self) -> usize {
        self.max_payload
    }

    fn append_inner(
        &mut self,
        idempotency_key: u128,
        payload: &[u8],
    ) -> Result<EventLogAppend, EventLogError> {
        if let Some(sequence) = self.keys.get(&idempotency_key).copied() {
            let index = usize::try_from(sequence.saturating_sub(self.base_sequence))
                .map_err(|_| EventLogError::SequenceExhausted)?;
            let existing = self
                .entries
                .get(index)
                .ok_or(EventLogError::SequenceExhausted)?;
            if existing.payload != payload {
                return Err(EventLogError::IdempotencyConflict {
                    key: idempotency_key,
                });
            }
            return Ok(EventLogAppend {
                next_cursor: EventLogCursor::new(sequence.saturating_add(1)),
                inserted: false,
            });
        }
        if self.entries.len() >= self.max_entries {
            return Err(EventLogError::CapacityExceeded {
                limit: self.max_entries,
            });
        }
        if payload.len() > self.max_payload {
            return Err(EventLogError::PayloadTooLarge {
                len: payload.len(),
                max: self.max_payload,
            });
        }
        let sequence = self.cursor().next_sequence;
        let next_sequence = sequence
            .checked_add(1)
            .ok_or(EventLogError::SequenceExhausted)?;
        self.entries.push(EventLogEntry {
            sequence,
            idempotency_key,
            payload: payload.to_vec(),
        });
        self.keys.insert(idempotency_key, sequence);
        Ok(EventLogAppend {
            next_cursor: EventLogCursor::new(next_sequence),
            inserted: true,
        })
    }

    fn replay_inner(
        &self,
        cursor: EventLogCursor,
        limit: usize,
    ) -> Result<EventLogBatch, EventLogError> {
        if limit > MAX_EVENT_LOG_BATCH {
            return Err(EventLogError::ReplayLimitExceeded {
                requested: limit,
                max: MAX_EVENT_LOG_BATCH,
            });
        }
        let end = self.cursor();
        if cursor.next_sequence < self.base_sequence {
            return Err(EventLogError::CursorExpired {
                cursor: cursor.next_sequence,
                start: self.base_sequence,
            });
        }
        if cursor > end {
            return Err(EventLogError::CursorAhead {
                cursor: cursor.next_sequence,
                end: end.next_sequence,
            });
        }
        let start = usize::try_from(cursor.next_sequence - self.base_sequence).map_err(|_| {
            EventLogError::CursorAhead {
                cursor: cursor.next_sequence,
                end: end.next_sequence,
            }
        })?;
        let stop = start.saturating_add(limit).min(self.entries.len());
        let entries = self.entries[start..stop].to_vec();
        Ok(EventLogBatch::new(
            entries,
            EventLogCursor::new(self.base_sequence + stop as u64),
            end,
        ))
    }

    fn checkpoint_index(&self, entry: &EventLogEntry) -> Result<Self, EventLogError> {
        if self.is_empty() || entry.sequence.checked_add(1) != Some(self.cursor().next_sequence) {
            return Err(EventLogError::InvalidCheckpoint);
        }
        let mut replacement = Self::new(self.max_entries, self.max_payload);
        replacement.base_sequence = entry.sequence;
        replacement.append_inner(entry.idempotency_key, &entry.payload)?;
        Ok(replacement)
    }
}

impl EventLogBackend for MemoryEventLogBackend {
    type Error = EventLogError;

    fn append(
        &mut self,
        idempotency_key: u128,
        payload: &[u8],
    ) -> Result<EventLogAppend, Self::Error> {
        self.append_inner(idempotency_key, payload)
    }

    fn cursor(&self) -> EventLogCursor {
        EventLogCursor::new(self.base_sequence + self.entries.len() as u64)
    }

    fn start_cursor(&self) -> EventLogCursor {
        EventLogCursor::new(self.base_sequence)
    }

    fn remaining_capacity(&self) -> Option<usize> {
        Some(self.max_entries.saturating_sub(self.len()))
    }

    fn checkpoint(&mut self, entry: &EventLogEntry) -> Option<Result<(), Self::Error>> {
        Some(
            self.checkpoint_index(entry)
                .map(|replacement| *self = replacement),
        )
    }

    fn replay(&self, cursor: EventLogCursor, limit: usize) -> Result<EventLogBatch, Self::Error> {
        self.replay_inner(cursor, limit)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn sync(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Errors from the local file-backed provider.
#[cfg(feature = "std")]
#[derive(Debug)]
pub enum FileEventLogError {
    /// Filesystem operation failed.
    Io(std::io::Error),
    /// The file header is not this provider's format.
    InvalidHeader,
    /// The file contains an incomplete or invalid record.
    Corrupt {
        /// Byte offset where validation failed.
        offset: u64,
    },
    /// A failed append or ambiguous checkpoint poisoned the provider; reopen it.
    Poisoned,
    /// The bounded in-memory index rejected an entry or replay request.
    Contract(EventLogError),
}

#[cfg(feature = "std")]
impl fmt::Display for FileEventLogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "event log I/O error: {error}"),
            Self::InvalidHeader => formatter.write_str("invalid replication event log header"),
            Self::Corrupt { offset } => {
                write!(
                    formatter,
                    "corrupt replication event log at offset {offset}"
                )
            }
            Self::Poisoned => formatter.write_str("event log provider is poisoned; reopen it"),
            Self::Contract(error) => error.fmt(formatter),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FileEventLogError {}

/// A bounded append-only file provider for encoded logical events.
///
/// Opening validates the complete file and rebuilds the bounded idempotency
/// index. A malformed tail is rejected rather than silently discarded; a
/// caller that wants valid-prefix recovery must perform that policy outside
/// this minimal provider or use the world-persist journal for transactions.
/// Loss of a whole-record-aligned suffix is indistinguishable from a shorter
/// valid stream without an independently stored high-watermark. The checksum
/// detects malformed bytes, not adversarial changes or complete record loss.
/// Opening streams one record at a time: retained memory is bounded by
/// `max_entries * max_payload` plus one bounded record, not by file size.
/// A persistent sibling advisory lock excludes cooperating writers using the
/// same path across rename; alternate hardlink/symlink names are not fenced.
/// Checkpoint uses one bounded sibling staging file, sync, atomic rename, and
/// directory sync. Filesystems must implement these operations faithfully; this
/// is not protection from external writers, media loss, or malicious corruption.
#[cfg(feature = "std")]
pub struct FileEventLogBackend {
    file: std::fs::File,
    path: std::path::PathBuf,
    // Never unlink the lock: doing so would allow locking different inodes.
    _writer_lock: EventLogWriterLock,
    index: MemoryEventLogBackend,
    poisoned: bool,
}

#[cfg(feature = "std")]
struct EventLogWriterLock(std::fs::File);

#[cfg(feature = "std")]
impl Drop for EventLogWriterLock {
    fn drop(&mut self) {
        // Explicitly end this provider's exclusive ownership. On Unix a forked
        // or duplicated descriptor can keep the open-file-description alive
        // after File::drop, otherwise delaying release until that copy closes.
        let _ = self.0.unlock();
    }
}

#[cfg(feature = "std")]
impl FileEventLogBackend {
    /// Open or create a bounded event log.
    pub fn open(
        path: impl AsRef<std::path::Path>,
        max_entries: usize,
        max_payload: usize,
    ) -> Result<Self, FileEventLogError> {
        use std::io::{Seek, SeekFrom, Write};

        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."));
        let parent = parent.canonicalize().map_err(FileEventLogError::Io)?;
        let name = path.file_name().ok_or_else(|| {
            FileEventLogError::Io(std::io::Error::other("event log path has no filename"))
        })?;
        let path = parent.join(name);
        let mut lock_name = name.to_os_string();
        lock_name.push(".writer-lock");
        let writer_lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(parent.join(lock_name))
            .map_err(FileEventLogError::Io)?;
        writer_lock
            .try_lock()
            .map_err(|error| FileEventLogError::Io(error.into()))?;
        let writer_lock = EventLogWriterLock(writer_lock);
        let (mut file, created) = match std::fs::OpenOptions::new()
            .create_new(true)
            .read(true)
            .append(true)
            .open(&path)
        {
            Ok(file) => (file, true),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (
                std::fs::OpenOptions::new()
                    .read(true)
                    .append(true)
                    .open(&path)
                    .map_err(FileEventLogError::Io)?,
                false,
            ),
            Err(error) => return Err(FileEventLogError::Io(error)),
        };
        let mut index = MemoryEventLogBackend::new(max_entries, max_payload);
        // An existing zero-length file may be a truncated authority journal.
        // Never silently turn loss of its header into a new empty authority.
        if created {
            file.write_all(&file_header(0))
                .map_err(FileEventLogError::Io)?;
            file.flush().map_err(FileEventLogError::Io)?;
        } else {
            parse_file(&mut file, &mut index)?;
        }
        file.seek(SeekFrom::End(0)).map_err(FileEventLogError::Io)?;
        Ok(Self {
            file,
            path,
            _writer_lock: writer_lock,
            index,
            poisoned: false,
        })
    }

    /// Access the bounded in-memory index for diagnostics.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether no entries are currently retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn ensure_healthy(&self) -> Result<(), FileEventLogError> {
        if self.poisoned {
            Err(FileEventLogError::Poisoned)
        } else {
            Ok(())
        }
    }

    fn sync_parent(&self) -> Result<(), FileEventLogError> {
        std::fs::File::open(self.path.parent().expect("absolute event log path"))
            .and_then(|directory| directory.sync_all())
            .map_err(FileEventLogError::Io)
    }

    fn replace_checkpoint(&mut self, entry: &EventLogEntry) -> Result<(), FileEventLogError> {
        use std::io::Write;
        self.ensure_healthy()?;
        if entry.payload.len() > u32::MAX as usize {
            return Err(FileEventLogError::Contract(
                EventLogError::PayloadTooLarge {
                    len: entry.payload.len(),
                    max: u32::MAX as usize,
                },
            ));
        }
        let replacement = self
            .index
            .checkpoint_index(entry)
            .map_err(FileEventLogError::Contract)?;
        // One deterministic staging path bounds orphan disk usage after abrupt
        // exits. It is never adopted on open; only the active path is authority.
        let mut staging_name = self.path.as_os_str().to_os_string();
        staging_name.push(".checkpoint");
        let staging_path = std::path::PathBuf::from(staging_name);
        let mut staging = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&staging_path)
            .map_err(FileEventLogError::Io)?;
        staging
            .write_all(&file_header(entry.sequence))
            .map_err(FileEventLogError::Io)?;
        staging
            .write_all(&encode_record(entry))
            .map_err(FileEventLogError::Io)?;
        #[cfg(test)]
        checkpoint_test_point("checkpoint-written")?;
        staging.sync_all().map_err(FileEventLogError::Io)?;
        #[cfg(test)]
        checkpoint_test_point("checkpoint-synced")?;
        std::fs::rename(&staging_path, &self.path).map_err(FileEventLogError::Io)?;
        // After rename, errors are ambiguous. Do not write the old unlinked inode
        // or report successful capacity reclamation before directory durability.
        self.poisoned = true;
        self.file = staging;
        self.index = replacement;
        #[cfg(test)]
        checkpoint_test_point("checkpoint-renamed")?;
        self.sync_parent()?;
        #[cfg(test)]
        checkpoint_test_point("checkpoint-published")?;
        self.poisoned = false;
        Ok(())
    }
}

// Test-only process fault injection: no production environment-variable hooks.
#[cfg(all(test, feature = "std"))]
std::thread_local! {
    pub(crate) static CHECKPOINT_CRASH: core::cell::Cell<Option<&'static str>> = const {
        core::cell::Cell::new(None)
    };
    static CHECKPOINT_ERROR: core::cell::Cell<Option<&'static str>> = const {
        core::cell::Cell::new(None)
    };
}

#[cfg(all(test, feature = "std"))]
fn checkpoint_test_point(point: &'static str) -> Result<(), FileEventLogError> {
    if CHECKPOINT_CRASH.with(|fault| fault.get() == Some(point)) {
        process_crash_barrier();
    }
    if CHECKPOINT_ERROR.with(|fault| fault.get() == Some(point)) {
        return Err(FileEventLogError::Io(std::io::Error::other(
            "injected checkpoint failure",
        )));
    }
    Ok(())
}

/// The parent kills the child only after observing this marker. No unwinding or
/// file destructors run: the test exercises actual OS process termination.
#[cfg(all(test, feature = "std"))]
pub(crate) fn process_crash_barrier() -> ! {
    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "PRISM_DURABLE_CRASH_READY").unwrap();
    stdout.flush().unwrap();
    drop(stdout);
    loop {
        std::thread::park();
    }
}

#[cfg(feature = "std")]
impl EventLogBackend for FileEventLogBackend {
    type Error = FileEventLogError;

    fn append(
        &mut self,
        idempotency_key: u128,
        payload: &[u8],
    ) -> Result<EventLogAppend, Self::Error> {
        use std::io::Write;

        self.ensure_healthy()?;
        if payload.len() > u32::MAX as usize {
            return Err(FileEventLogError::Contract(
                EventLogError::PayloadTooLarge {
                    len: payload.len(),
                    max: u32::MAX as usize,
                },
            ));
        }
        let outcome = self
            .index
            .append_inner(idempotency_key, payload)
            .map_err(FileEventLogError::Contract)?;
        if !outcome.inserted {
            return Ok(outcome);
        }
        let entry = self
            .index
            .entries
            .last()
            .expect("new append inserted an index entry");
        if let Err(error) = self.file.write_all(&encode_record(entry)) {
            self.poisoned = true;
            return Err(FileEventLogError::Io(error));
        }
        if let Err(error) = self.file.flush() {
            self.poisoned = true;
            return Err(FileEventLogError::Io(error));
        }
        Ok(outcome)
    }

    fn cursor(&self) -> EventLogCursor {
        self.index.cursor()
    }

    fn start_cursor(&self) -> EventLogCursor {
        self.index.start_cursor()
    }

    fn remaining_capacity(&self) -> Option<usize> {
        self.index.remaining_capacity()
    }

    fn checkpoint(&mut self, entry: &EventLogEntry) -> Option<Result<(), Self::Error>> {
        Some(self.replace_checkpoint(entry))
    }

    fn replay(&self, cursor: EventLogCursor, limit: usize) -> Result<EventLogBatch, Self::Error> {
        self.ensure_healthy()?;
        self.index
            .replay_inner(cursor, limit)
            .map_err(FileEventLogError::Contract)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        use std::io::Write;

        self.ensure_healthy()?;
        self.file.flush().map_err(FileEventLogError::Io)
    }

    fn sync(&mut self) -> Result<(), Self::Error> {
        self.ensure_healthy()?;
        self.file.sync_all().map_err(FileEventLogError::Io)?;
        self.sync_parent()
    }
}

#[cfg(feature = "std")]
fn file_header(base: u64) -> [u8; FILE_HEADER_LEN] {
    let mut bytes = [0; FILE_HEADER_LEN];
    bytes[..4].copy_from_slice(&FILE_MAGIC);
    bytes[4] = FILE_VERSION;
    bytes[8..16].copy_from_slice(&base.to_le_bytes());
    let hash = checksum(&bytes[..16]);
    bytes[16..].copy_from_slice(&hash.to_le_bytes());
    bytes
}

#[cfg(feature = "std")]
fn encode_record(entry: &EventLogEntry) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(
        FILE_RECORD_HEADER_LEN
            .saturating_add(entry.payload.len())
            .saturating_add(FILE_RECORD_TRAILER_LEN),
    );
    bytes.extend_from_slice(&entry.sequence.to_le_bytes());
    bytes.extend_from_slice(&entry.idempotency_key.to_le_bytes());
    bytes.extend_from_slice(&(entry.payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&entry.payload);
    bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
    bytes
}

#[cfg(feature = "std")]
fn parse_file(
    reader: &mut impl std::io::Read,
    index: &mut MemoryEventLogBackend,
) -> Result<(), FileEventLogError> {
    use std::io::ErrorKind;

    let mut header = [0; FILE_HEADER_LEN];
    match reader.read_exact(&mut header[..8]) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => {
            return Err(FileEventLogError::InvalidHeader);
        }
        Err(error) => return Err(FileEventLogError::Io(error)),
    }
    if header[..4] != FILE_MAGIC || header[5..8] != [0; 3] {
        return Err(FileEventLogError::InvalidHeader);
    }
    let mut offset = match header[4] {
        1 => 8, // Existing PREL v1 files always start at zero.
        FILE_VERSION => {
            read_file_exact(reader, &mut header[8..], 8)?;
            if checksum(&header[..16]) != u32::from_le_bytes(header[16..].try_into().unwrap()) {
                return Err(FileEventLogError::InvalidHeader);
            }
            index.base_sequence = u64::from_le_bytes(header[8..16].try_into().unwrap());
            FILE_HEADER_LEN as u64
        }
        _ => return Err(FileEventLogError::InvalidHeader),
    };
    loop {
        let mut header = [0; FILE_RECORD_HEADER_LEN];
        // EOF is valid only between records. Interrupted/short reads must not
        // accidentally turn a truncated record into an accepted prefix.
        loop {
            match reader.read(&mut header[..1]) {
                Ok(0) if index.base_sequence != 0 && index.is_empty() => {
                    return Err(FileEventLogError::Corrupt { offset });
                }
                Ok(0) => return Ok(()),
                Ok(_) => break,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(FileEventLogError::Io(error)),
            }
        }
        read_file_exact(reader, &mut header[1..], offset)?;
        let mut position = 0;
        let sequence = read_u64(&header, &mut position).expect("fixed record header");
        let idempotency_key = read_u128(&header, &mut position).expect("fixed record header");
        let payload_len = read_u32(&header, &mut position).expect("fixed record header") as usize;
        // A duplicate physical record is corruption, not an idempotent append.
        // Validate bounds before reading/allocating the declared payload.
        if sequence != index.cursor().next_sequence || index.keys.contains_key(&idempotency_key) {
            return Err(FileEventLogError::Corrupt { offset });
        }
        if payload_len > index.max_payload {
            return Err(FileEventLogError::Contract(
                EventLogError::PayloadTooLarge {
                    len: payload_len,
                    max: index.max_payload,
                },
            ));
        }
        if index.len() >= index.max_entries {
            return Err(FileEventLogError::Contract(
                EventLogError::CapacityExceeded {
                    limit: index.max_entries,
                },
            ));
        }
        let mut payload = alloc::vec![0; payload_len];
        read_file_exact(reader, &mut payload, offset)?;
        let mut trailer = [0; FILE_RECORD_TRAILER_LEN];
        read_file_exact(reader, &mut trailer, offset)?;
        if checksum_parts(&[&header, &payload]) != u32::from_le_bytes(trailer) {
            return Err(FileEventLogError::Corrupt { offset });
        }
        let outcome = index
            .append_inner(idempotency_key, &payload)
            .map_err(FileEventLogError::Contract)?;
        if !outcome.inserted {
            return Err(FileEventLogError::Corrupt { offset });
        }
        offset = offset
            .checked_add((FILE_RECORD_HEADER_LEN + FILE_RECORD_TRAILER_LEN) as u64)
            .and_then(|offset| offset.checked_add(payload_len as u64))
            .ok_or(FileEventLogError::Corrupt { offset })?;
    }
}

#[cfg(feature = "std")]
fn read_file_exact(
    reader: &mut impl std::io::Read,
    bytes: &mut [u8],
    offset: u64,
) -> Result<(), FileEventLogError> {
    reader.read_exact(bytes).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            FileEventLogError::Corrupt { offset }
        } else {
            FileEventLogError::Io(error)
        }
    })
}

#[cfg(feature = "std")]
fn checksum(bytes: &[u8]) -> u32 {
    checksum_parts(&[bytes])
}

#[cfg(feature = "std")]
fn checksum_parts(parts: &[&[u8]]) -> u32 {
    let mut hash = 2_166_136_261_u32;
    for byte in parts.iter().flat_map(|part| part.iter()) {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

#[cfg(feature = "std")]
fn read_u64(bytes: &[u8], offset: &mut usize) -> Option<u64> {
    let end = offset.checked_add(8)?;
    let value = u64::from_le_bytes(bytes.get(*offset..end)?.try_into().ok()?);
    *offset = end;
    Some(value)
}

#[cfg(feature = "std")]
fn read_u128(bytes: &[u8], offset: &mut usize) -> Option<u128> {
    let end = offset.checked_add(16)?;
    let value = u128::from_le_bytes(bytes.get(*offset..end)?.try_into().ok()?);
    *offset = end;
    Some(value)
}

#[cfg(feature = "std")]
fn read_u32(bytes: &[u8], offset: &mut usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let value = u32::from_le_bytes(bytes.get(*offset..end)?.try_into().ok()?);
    *offset = end;
    Some(value)
}

/// A typed bridge between [`ReplayRecord`] and an opaque event-log provider.
pub struct ReplayEventLog<B> {
    backend: B,
    max_payload: usize,
}

impl<B> ReplayEventLog<B> {
    /// Wrap a provider with the payload bound used when decoding replay records.
    #[must_use]
    pub const fn new(backend: B, max_payload: usize) -> Self {
        Self {
            backend,
            max_payload,
        }
    }

    /// Borrow the underlying provider.
    #[must_use]
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Borrow the underlying provider mutably.
    #[must_use]
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B: EventLogBackend> ReplayEventLog<B> {
    /// Replace the retained prefix with one typed checkpoint. See
    /// [`EventLogBackend::checkpoint`] for durability and expired-key semantics.
    pub fn checkpoint(
        &mut self,
        idempotency_key: u128,
        record: ReplayRecord,
    ) -> Option<Result<(), ReplayEventLogError<B::Error>>> {
        let sequence = record.sequence;
        let mut log = ReplayLog::new(1, self.max_payload);
        if let Err(error) = log.append(record) {
            return Some(Err(ReplayEventLogError::Replay(error)));
        }
        let payload = match log.encode() {
            Ok(payload) => payload,
            Err(error) => return Some(Err(ReplayEventLogError::Replay(error))),
        };
        self.backend
            .checkpoint(&EventLogEntry {
                sequence,
                idempotency_key,
                payload,
            })
            .map(|result| result.map_err(ReplayEventLogError::Backend))
    }

    /// Append one replay record using an explicit idempotency key.
    pub fn append(
        &mut self,
        idempotency_key: u128,
        record: ReplayRecord,
    ) -> Result<EventLogAppend, ReplayEventLogError<B::Error>> {
        let mut log = ReplayLog::new(1, self.max_payload);
        log.append(record).map_err(ReplayEventLogError::Replay)?;
        let payload = log.encode().map_err(ReplayEventLogError::Replay)?;
        self.backend
            .append(idempotency_key, &payload)
            .map_err(ReplayEventLogError::Backend)
    }

    /// Replay and decode at most `limit` records.
    pub fn replay(
        &self,
        cursor: EventLogCursor,
        limit: usize,
    ) -> Result<ReplayEventBatch, ReplayEventLogError<B::Error>> {
        let batch = self
            .backend
            .replay(cursor, limit)
            .map_err(ReplayEventLogError::Backend)?;
        let mut records = Vec::with_capacity(batch.entries().len());
        for entry in batch.entries() {
            let log = ReplayLog::decode(&entry.payload, 1, self.max_payload)
                .map_err(ReplayEventLogError::Replay)?;
            let record = log
                .records()
                .first()
                .cloned()
                .ok_or(ReplayEventLogError::EmptyPayload)?;
            records.push(ReplayEvent {
                sequence: entry.sequence,
                idempotency_key: entry.idempotency_key,
                record,
            });
        }
        Ok(ReplayEventBatch {
            records,
            next_cursor: batch.next_cursor(),
            end_cursor: batch.end_cursor(),
        })
    }
}

/// One decoded replay event returned by [`ReplayEventLog`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayEvent {
    /// Provider-local append sequence.
    pub sequence: u64,
    /// Idempotency key used for append.
    pub idempotency_key: u128,
    /// Decoded logical replication record.
    pub record: ReplayRecord,
}

/// A bounded decoded replay batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayEventBatch {
    records: Vec<ReplayEvent>,
    next_cursor: EventLogCursor,
    end_cursor: EventLogCursor,
}

impl ReplayEventBatch {
    /// Borrow decoded events in provider order.
    #[must_use]
    pub fn records(&self) -> &[ReplayEvent] {
        &self.records
    }

    /// Cursor immediately after this batch.
    #[must_use]
    pub const fn next_cursor(&self) -> EventLogCursor {
        self.next_cursor
    }

    /// End cursor observed at replay start.
    #[must_use]
    pub const fn end_cursor(&self) -> EventLogCursor {
        self.end_cursor
    }

    /// Whether more events were available after this batch.
    #[must_use]
    pub const fn has_more(&self) -> bool {
        self.next_cursor.next_sequence < self.end_cursor.next_sequence
    }
}

/// Errors raised while encoding or decoding records through a provider.
#[derive(Debug)]
pub enum ReplayEventLogError<E> {
    /// The provider rejected the operation.
    Backend(E),
    /// The replay codec rejected the opaque payload.
    Replay(ReplayError),
    /// A provider entry decoded to an empty replay log.
    EmptyPayload,
}

impl<E: fmt::Display> fmt::Display for ReplayEventLogError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(error) => write!(formatter, "event log backend error: {error}"),
            Self::Replay(error) => write!(formatter, "replay codec error: {error:?}"),
            Self::EmptyPayload => {
                formatter.write_str("event log payload contained no replay record")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multiplex::NetChannel;
    use crate::replay::ReplayDirection;

    fn record(tick: u64, payload: &[u8]) -> ReplayRecord {
        ReplayRecord::new(
            tick,
            tick,
            NetChannel::EventsReliable,
            ReplayDirection::ServerToClient,
            Some(tick ^ 0x55),
            payload.to_vec(),
        )
    }

    #[test]
    fn checkpoint_preserves_end_expires_old_cursors_and_reclaims_capacity() {
        let mut backend = MemoryEventLogBackend::new(3, 16);
        for key in 0..3 {
            backend.append(key, b"delta").unwrap();
        }
        let end = backend.cursor();
        let checkpoint = EventLogEntry {
            sequence: 2,
            idempotency_key: 2,
            payload: b"snapshot".to_vec(),
        };
        backend.checkpoint(&checkpoint).unwrap().unwrap();
        assert_eq!(backend.cursor(), end);
        assert_eq!(backend.start_cursor(), EventLogCursor::new(2));
        assert_eq!(backend.remaining_capacity(), Some(2));
        assert!(matches!(
            backend.replay(EventLogCursor::new(1), 1),
            Err(EventLogError::CursorExpired { .. })
        ));
        assert_eq!(
            backend.replay(EventLogCursor::new(2), 1).unwrap().entries(),
            &[checkpoint]
        );
        assert_eq!(
            backend.append(3, b"next").unwrap().next_cursor,
            EventLogCursor::new(4)
        );
        let before = backend.clone();
        assert_eq!(
            backend.checkpoint(&EventLogEntry {
                sequence: 2,
                idempotency_key: 2,
                payload: Vec::new()
            }),
            Some(Err(EventLogError::InvalidCheckpoint))
        );
        assert_eq!(backend, before);
    }

    #[cfg(feature = "std")]
    struct FileFixture(std::path::PathBuf);

    #[cfg(feature = "std")]
    impl FileFixture {
        fn new() -> Self {
            let directory = std::env::temp_dir().join(std::format!(
                "prism-checkpoint-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
            ));
            std::fs::create_dir(&directory).unwrap();
            Self(directory)
        }

        fn path(&self) -> std::path::PathBuf {
            self.0.join("journal")
        }
    }

    #[cfg(feature = "std")]
    impl Drop for FileFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn file_writer_ownership_ends_even_with_a_duplicated_lock_descriptor() {
        let fixture = FileFixture::new();
        let path = fixture.path();
        let mut backend = FileEventLogBackend::open(&path, 2, 8).unwrap();
        backend.append(1, b"first").unwrap();
        backend.sync().unwrap();
        // Model a descriptor inherited by a concurrently spawned process: it
        // must not extend ownership beyond the provider's lifetime.
        let inherited = backend._writer_lock.0.try_clone().unwrap();
        assert!(matches!(
            FileEventLogBackend::open(&path, 2, 8),
            Err(FileEventLogError::Io(error))
                if error.kind() == std::io::ErrorKind::WouldBlock
        ));
        drop(backend);
        let mut reopened = FileEventLogBackend::open(&path, 2, 8).unwrap();
        drop(inherited);
        assert!(matches!(
            FileEventLogBackend::open(&path, 2, 8),
            Err(FileEventLogError::Io(error))
                if error.kind() == std::io::ErrorKind::WouldBlock
        ));
        reopened.append(2, b"next").unwrap();
        reopened.sync().unwrap();
        drop(reopened);
        let reopened = FileEventLogBackend::open(&path, 2, 8).unwrap();
        assert_eq!(reopened.cursor(), EventLogCursor::new(2));
    }

    #[cfg(feature = "std")]
    #[test]
    fn file_checkpoint_failure_atomicity_requires_reopen_after_rename() {
        for point in [
            "checkpoint-written",
            "checkpoint-synced",
            "checkpoint-renamed",
            "checkpoint-published",
        ] {
            let fixture = FileFixture::new();
            let path = fixture.path();
            let mut backend = FileEventLogBackend::open(&path, 2, 16).unwrap();
            backend.append(0, b"seed").unwrap();
            backend.append(1, b"delta").unwrap();
            backend.sync().unwrap();
            let bytes = std::fs::read(&path).unwrap();
            let before = backend.index.clone();
            let checkpoint = EventLogEntry {
                sequence: 1,
                idempotency_key: 1,
                payload: b"checkpoint".to_vec(),
            };
            CHECKPOINT_ERROR.with(|slot| slot.set(Some(point)));
            let result = backend.checkpoint(&checkpoint).unwrap();
            CHECKPOINT_ERROR.with(|slot| slot.set(None));
            assert!(matches!(result, Err(FileEventLogError::Io(_))), "{point}");
            if matches!(point, "checkpoint-written" | "checkpoint-synced") {
                assert_eq!(std::fs::read(&path).unwrap(), bytes);
                assert_eq!(backend.index, before);
                assert!(!backend.poisoned);
                // Retry overwrites the stale staging file, never adopts it.
                backend.checkpoint(&checkpoint).unwrap().unwrap();
            } else {
                assert!(backend.poisoned);
                assert_eq!(backend.cursor(), EventLogCursor::new(2));
                assert_eq!(backend.start_cursor(), EventLogCursor::new(1));
                assert!(matches!(
                    backend.append(2, b"next"),
                    Err(FileEventLogError::Poisoned)
                ));
                assert!(matches!(
                    backend.replay(EventLogCursor::new(1), 1),
                    Err(FileEventLogError::Poisoned)
                ));
                assert!(matches!(backend.flush(), Err(FileEventLogError::Poisoned)));
                assert!(matches!(backend.sync(), Err(FileEventLogError::Poisoned)));
                assert!(matches!(
                    backend.checkpoint(&checkpoint),
                    Some(Err(FileEventLogError::Poisoned))
                ));
                drop(backend);
                backend = FileEventLogBackend::open(&path, 2, 16).unwrap();
                backend.checkpoint(&checkpoint).unwrap().unwrap();
            }
            assert_eq!(backend.cursor(), EventLogCursor::new(2));
            assert_eq!(backend.remaining_capacity(), Some(1));
            assert_eq!(
                backend.replay(EventLogCursor::new(1), 1).unwrap().entries(),
                &[checkpoint]
            );
            assert!(!backend.append(1, b"checkpoint").unwrap().inserted);
            assert_eq!(
                backend.append(2, b"next").unwrap().next_cursor,
                EventLogCursor::new(3)
            );
            backend.sync().unwrap();
            drop(backend);
            let reopened = FileEventLogBackend::open(&path, 2, 16).unwrap();
            assert_eq!(reopened.cursor(), EventLogCursor::new(3));
            assert_eq!(reopened.start_cursor(), EventLogCursor::new(1));
            assert_eq!(reopened.len(), 2);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn file_bounds_and_staging_open_failure_leave_active_file_unchanged() {
        let fixture = FileFixture::new();
        let path = fixture.path();
        let mut backend = FileEventLogBackend::open(&path, 2, 8).unwrap();
        backend.append(0, b"seed").unwrap();
        backend.append(1, b"delta").unwrap();
        backend.sync().unwrap();
        let before = backend.index.clone();
        let bytes = std::fs::read(&path).unwrap();
        let mut checkpoint = EventLogEntry {
            sequence: 1,
            idempotency_key: 1,
            payload: b"snapshot".to_vec(),
        };
        assert!(matches!(
            backend.append(2, b"next"),
            Err(FileEventLogError::Contract(
                EventLogError::CapacityExceeded { .. }
            ))
        ));
        checkpoint.payload.push(0);
        assert!(matches!(
            backend.checkpoint(&checkpoint),
            Some(Err(FileEventLogError::Contract(
                EventLogError::PayloadTooLarge { .. }
            )))
        ));
        checkpoint.payload.pop();
        checkpoint.sequence = 0;
        assert!(matches!(
            backend.checkpoint(&checkpoint),
            Some(Err(FileEventLogError::Contract(
                EventLogError::InvalidCheckpoint
            )))
        ));
        checkpoint.sequence = 1;
        let staging = fixture.0.join("journal.checkpoint");
        std::fs::create_dir(&staging).unwrap();
        assert!(matches!(
            backend.checkpoint(&checkpoint),
            Some(Err(FileEventLogError::Io(_)))
        ));
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(backend.index, before);
        assert!(!backend.poisoned);
        std::fs::remove_dir(staging).unwrap();
        backend.checkpoint(&checkpoint).unwrap().unwrap();
        backend.append(2, b"next").unwrap();
        backend.sync().unwrap();
        drop(backend);
        let reopened = FileEventLogBackend::open(&path, 2, 8).unwrap();
        assert_eq!(reopened.start_cursor(), EventLogCursor::new(1));
        assert_eq!(reopened.cursor(), EventLogCursor::new(3));
        assert_eq!(reopened.len(), 2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn file_parser_rejects_duplicate_keys_truncations_bounds_and_bad_headers() {
        use std::{
            fs,
            time::{SystemTime, UNIX_EPOCH},
        };
        let directory = std::env::temp_dir().join(std::format!(
            "prism-file-matrix-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("journal");
        let first = EventLogEntry {
            sequence: 0,
            idempotency_key: 7,
            payload: b"one".to_vec(),
        };
        let second = EventLogEntry {
            sequence: 1,
            idempotency_key: 8,
            payload: b"two".to_vec(),
        };
        let mut bytes = file_header(0).to_vec();
        bytes.extend_from_slice(&encode_record(&first));
        bytes.extend_from_slice(&encode_record(&second));
        // Only exact record boundaries are valid prefixes; all other truncations fail closed.
        let boundary = FILE_HEADER_LEN + encode_record(&first).len();
        for length in 0..bytes.len() {
            if length == FILE_HEADER_LEN || length == boundary {
                continue;
            }
            fs::write(&path, &bytes[..length]).unwrap();
            assert!(
                FileEventLogBackend::open(&path, 2, 8).is_err(),
                "truncation {length}"
            );
        }
        for duplicate in [
            first.clone(),
            EventLogEntry {
                sequence: 1,
                ..first.clone()
            },
        ] {
            let mut duplicate_bytes = file_header(0).to_vec();
            duplicate_bytes.extend_from_slice(&encode_record(&first));
            duplicate_bytes.extend_from_slice(&encode_record(&duplicate));
            fs::write(&path, duplicate_bytes).unwrap();
            assert!(matches!(
                FileEventLogBackend::open(&path, 3, 8),
                Err(FileEventLogError::Corrupt { .. })
            ));
        }
        fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            FileEventLogBackend::open(&path, 1, 8),
            Err(FileEventLogError::Contract(
                EventLogError::CapacityExceeded { .. }
            ))
        ));
        assert!(matches!(
            FileEventLogBackend::open(&path, 2, 2),
            Err(FileEventLogError::Contract(
                EventLogError::PayloadTooLarge { .. }
            ))
        ));
        for offset in [
            0,
            4,
            5,
            8,
            16,
            FILE_HEADER_LEN + FILE_RECORD_HEADER_LEN,
            bytes.len() - 1,
        ] {
            let mut corrupt = bytes.clone();
            corrupt[offset] ^= 0x80;
            fs::write(&path, corrupt).unwrap();
            assert!(
                FileEventLogBackend::open(&path, 2, 8).is_err(),
                "corruption {offset}"
            );
        }
        fs::write(&path, file_header(9)).unwrap();
        assert!(
            FileEventLogBackend::open(&path, 2, 8).is_err(),
            "missing nonzero checkpoint"
        );
        // v1 remains readable, and a subsequent checkpoint upgrades it atomically.
        let mut legacy = b"PREL\x01\0\0\0".to_vec();
        legacy.extend_from_slice(&encode_record(&first));
        legacy.extend_from_slice(&encode_record(&second));
        fs::write(&path, legacy).unwrap();
        {
            let mut log = FileEventLogBackend::open(&path, 2, 8).unwrap();
            assert!(
                FileEventLogBackend::open(&path, 2, 8).is_err(),
                "exclusive writer"
            );
            log.checkpoint(&second).unwrap().unwrap();
            log.append(9, b"next").unwrap();
            log.sync().unwrap();
            assert!(
                FileEventLogBackend::open(&path, 2, 8).is_err(),
                "lock survives rename"
            );
        }
        let log = FileEventLogBackend::open(&path, 2, 8).unwrap();
        assert_eq!(log.start_cursor(), EventLogCursor::new(1));
        assert_eq!(log.cursor(), EventLogCursor::new(3));
        assert_eq!(
            log.replay(EventLogCursor::new(1), 2).unwrap().entries()[1].payload,
            b"next"
        );
        drop(log);
        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_parser_refuses_payload_and_capacity_before_reading_body() {
        use std::io::Read;
        struct GuardedReader {
            bytes: std::io::Cursor<Vec<u8>>,
            allowed: u64,
        }
        impl Read for GuardedReader {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                assert!(
                    self.bytes.position() + out.len() as u64 <= self.allowed,
                    "unbounded body read"
                );
                self.bytes.read(out)
            }
        }
        let mut bytes = file_header(0).to_vec();
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u128.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        let mut reader = GuardedReader {
            allowed: bytes.len() as u64,
            bytes: std::io::Cursor::new(bytes),
        };
        assert!(matches!(
            parse_file(&mut reader, &mut MemoryEventLogBackend::new(1, 8)),
            Err(FileEventLogError::Contract(
                EventLogError::PayloadTooLarge { .. }
            ))
        ));
        reader.bytes.set_position(0);
        assert!(matches!(
            parse_file(&mut reader, &mut MemoryEventLogBackend::new(0, usize::MAX)),
            Err(FileEventLogError::Contract(
                EventLogError::CapacityExceeded { .. }
            ))
        ));
    }

    #[test]
    fn memory_backend_is_bounded_idempotent_and_cursor_replay_is_at_least_once_friendly() {
        let mut backend = MemoryEventLogBackend::new(2, 8);
        assert_eq!(
            backend.append(7, b"one").unwrap(),
            EventLogAppend {
                next_cursor: EventLogCursor::new(1),
                inserted: true,
            }
        );
        assert_eq!(
            backend.append(7, b"one"),
            Ok(EventLogAppend {
                next_cursor: EventLogCursor::new(1),
                inserted: false,
            })
        );
        assert_eq!(
            backend.append(7, b"different duplicate"),
            Err(EventLogError::IdempotencyConflict { key: 7 })
        );
        backend.append(8, b"two").unwrap();
        let first = backend.replay(EventLogCursor::default(), 1).unwrap();
        assert_eq!(first.entries()[0].payload, b"one");
        assert!(first.has_more());
        let second = backend.replay(first.next_cursor(), 1).unwrap();
        assert_eq!(second.entries()[0].payload, b"two");
        assert!(!second.has_more());
        assert_eq!(backend.flush(), Ok(()));
        assert_eq!(backend.sync(), Ok(()));
    }

    #[cfg(feature = "std")]
    #[test]
    fn file_backend_rejects_conflicting_idempotent_append_without_mutating_log() {
        let fixture = FileFixture::new();
        let path = fixture.path();
        let mut backend = FileEventLogBackend::open(&path, 4, 64).unwrap();
        assert!(backend.append(42, b"authoritative event").unwrap().inserted);
        backend.sync().unwrap();
        let bytes_before = std::fs::read(&path).unwrap();
        assert!(matches!(
            backend.append(42, b"different event"),
            Err(FileEventLogError::Contract(
                EventLogError::IdempotencyConflict { key: 42 }
            ))
        ));
        assert_eq!(std::fs::read(&path).unwrap(), bytes_before);
        assert_eq!(backend.cursor(), EventLogCursor::new(1));
        drop(backend);

        let reopened = FileEventLogBackend::open(&path, 4, 64).unwrap();
        let replay = reopened.replay(EventLogCursor::default(), 4).unwrap();
        assert_eq!(replay.entries().len(), 1);
        assert_eq!(replay.entries()[0].payload, b"authoritative event");
    }

    #[test]
    fn replay_event_log_round_trips_typed_records_and_rejects_bounds() {
        let mut log = ReplayEventLog::new(MemoryEventLogBackend::new(1, 64), 64);
        log.append(11, record(4, b"payload")).unwrap();
        assert!(matches!(
            log.append(11, record(9, b"different")),
            Err(ReplayEventLogError::Backend(
                EventLogError::IdempotencyConflict { key: 11 }
            ))
        ));
        let batch = log.replay(EventLogCursor::default(), 1).unwrap();
        assert_eq!(batch.records().len(), 1);
        assert_eq!(batch.records()[0].idempotency_key, 11);
        assert_eq!(batch.records()[0].record, record(4, b"payload"));

        let mut bounded = MemoryEventLogBackend::new(1, 2);
        assert_eq!(
            bounded.append(1, b"too long"),
            Err(EventLogError::PayloadTooLarge { len: 8, max: 2 })
        );
        assert_eq!(
            bounded.replay(EventLogCursor::default(), MAX_EVENT_LOG_BATCH + 1),
            Err(EventLogError::ReplayLimitExceeded {
                requested: MAX_EVENT_LOG_BATCH + 1,
                max: MAX_EVENT_LOG_BATCH,
            })
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn file_backend_survives_reopen_and_rejects_corrupt_tail() {
        use std::format;
        use std::fs;
        use std::io::Write;
        use std::time::{SystemTime, UNIX_EPOCH};

        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("prism-replication-event-log-{suffix}.bin"));
        {
            let mut backend = FileEventLogBackend::open(&path, 4, 64).expect("open file backend");
            backend.append(1, b"durable").expect("append");
            backend.sync().expect("sync");
        }
        let reopened = FileEventLogBackend::open(&path, 4, 64).expect("reopen file backend");
        let batch = reopened
            .replay(EventLogCursor::default(), 4)
            .expect("replay reopened backend");
        assert_eq!(batch.entries()[0].payload, b"durable");
        drop(reopened);
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open for corruption")
            .write_all(b"tail")
            .expect("write corrupt tail");
        assert!(matches!(
            FileEventLogBackend::open(&path, 4, 64),
            Err(FileEventLogError::Corrupt { .. })
        ));
        let _ = fs::remove_file(path);
    }
}
