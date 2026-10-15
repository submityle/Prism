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
const FILE_VERSION: u8 = 1;
#[cfg(feature = "std")]
const FILE_HEADER_LEN: usize = 8;
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
    fn append(
        &mut self,
        idempotency_key: u128,
        payload: &[u8],
    ) -> Result<EventLogAppend, Self::Error>;

    /// Return the current end cursor.
    fn cursor(&self) -> EventLogCursor;

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
    /// The sequence space is exhausted.
    SequenceExhausted,
}

impl fmt::Display for EventLogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            Self::SequenceExhausted => formatter.write_str("event log sequence is exhausted"),
        }
    }
}

/// In-memory provider used by deterministic tests and local authority harnesses.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryEventLogBackend {
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
        let sequence =
            u64::try_from(self.entries.len()).map_err(|_| EventLogError::SequenceExhausted)?;
        self.entries.push(EventLogEntry {
            sequence,
            idempotency_key,
            payload: payload.to_vec(),
        });
        self.keys.insert(idempotency_key, sequence);
        Ok(EventLogAppend {
            next_cursor: EventLogCursor::new(sequence.saturating_add(1)),
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
        if cursor > end {
            return Err(EventLogError::CursorAhead {
                cursor: cursor.next_sequence,
                end: end.next_sequence,
            });
        }
        let start =
            usize::try_from(cursor.next_sequence).map_err(|_| EventLogError::CursorAhead {
                cursor: cursor.next_sequence,
                end: end.next_sequence,
            })?;
        let stop = start.saturating_add(limit).min(self.entries.len());
        let entries = self.entries[start..stop].to_vec();
        Ok(EventLogBatch::new(
            entries,
            EventLogCursor::new(stop as u64),
            end,
        ))
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
        EventLogCursor::new(self.entries.len() as u64)
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
    /// The provider was poisoned by a failed append and must be reopened.
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
#[cfg(feature = "std")]
pub struct FileEventLogBackend {
    file: std::fs::File,
    index: MemoryEventLogBackend,
    poisoned: bool,
}

#[cfg(feature = "std")]
impl FileEventLogBackend {
    /// Open or create a bounded event log.
    pub fn open(
        path: impl AsRef<std::path::Path>,
        max_entries: usize,
        max_payload: usize,
    ) -> Result<Self, FileEventLogError> {
        use std::io::{Read, Seek, SeekFrom, Write};

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(path)
            .map_err(FileEventLogError::Io)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(FileEventLogError::Io)?;
        let mut index = MemoryEventLogBackend::new(max_entries, max_payload);
        if bytes.is_empty() {
            file.write_all(&file_header())
                .map_err(FileEventLogError::Io)?;
            file.flush().map_err(FileEventLogError::Io)?;
        } else {
            parse_file(&bytes, &mut index)?;
        }
        file.seek(SeekFrom::End(0)).map_err(FileEventLogError::Io)?;
        Ok(Self {
            file,
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
        self.file.sync_all().map_err(FileEventLogError::Io)
    }
}

#[cfg(feature = "std")]
fn file_header() -> [u8; FILE_HEADER_LEN] {
    [
        FILE_MAGIC[0],
        FILE_MAGIC[1],
        FILE_MAGIC[2],
        FILE_MAGIC[3],
        FILE_VERSION,
        0,
        0,
        0,
    ]
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
fn parse_file(bytes: &[u8], index: &mut MemoryEventLogBackend) -> Result<(), FileEventLogError> {
    if bytes.len() < FILE_HEADER_LEN || bytes[..FILE_HEADER_LEN] != file_header() {
        return Err(FileEventLogError::InvalidHeader);
    }
    let mut offset = FILE_HEADER_LEN;
    while offset < bytes.len() {
        if bytes.len().saturating_sub(offset) < FILE_RECORD_HEADER_LEN + FILE_RECORD_TRAILER_LEN {
            return Err(FileEventLogError::Corrupt {
                offset: offset as u64,
            });
        }
        let sequence = read_u64(bytes, &mut offset).ok_or(FileEventLogError::Corrupt {
            offset: offset as u64,
        })?;
        let idempotency_key = read_u128(bytes, &mut offset).ok_or(FileEventLogError::Corrupt {
            offset: offset as u64,
        })?;
        let payload_len = read_u32(bytes, &mut offset).ok_or(FileEventLogError::Corrupt {
            offset: offset as u64,
        })? as usize;
        if payload_len > index.max_payload {
            return Err(FileEventLogError::Contract(
                EventLogError::PayloadTooLarge {
                    len: payload_len,
                    max: index.max_payload,
                },
            ));
        }
        let end = offset
            .checked_add(payload_len)
            .and_then(|value| value.checked_add(FILE_RECORD_TRAILER_LEN))
            .ok_or(FileEventLogError::Corrupt {
                offset: offset as u64,
            })?;
        let Some(record_bytes) = bytes.get(offset..end) else {
            return Err(FileEventLogError::Corrupt {
                offset: offset as u64,
            });
        };
        let payload = &record_bytes[..payload_len];
        let stored_checksum = u32::from_le_bytes(
            record_bytes[payload_len..]
                .try_into()
                .expect("trailer length checked"),
        );
        let mut checksum_bytes = Vec::with_capacity(FILE_RECORD_HEADER_LEN + payload_len);
        checksum_bytes.extend_from_slice(&sequence.to_le_bytes());
        checksum_bytes.extend_from_slice(&idempotency_key.to_le_bytes());
        checksum_bytes.extend_from_slice(&(payload_len as u32).to_le_bytes());
        checksum_bytes.extend_from_slice(payload);
        if checksum(&checksum_bytes) != stored_checksum {
            return Err(FileEventLogError::Corrupt {
                offset: offset as u64,
            });
        }
        if sequence != index.cursor().next_sequence {
            return Err(FileEventLogError::Corrupt {
                offset: offset as u64,
            });
        }
        index
            .append_inner(idempotency_key, payload)
            .map_err(FileEventLogError::Contract)?;
        offset = end;
    }
    Ok(())
}

#[cfg(feature = "std")]
fn checksum(bytes: &[u8]) -> u32 {
    let mut hash = 2_166_136_261_u32;
    for byte in bytes {
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
            backend.append(7, b"different duplicate").unwrap(),
            EventLogAppend {
                next_cursor: EventLogCursor::new(1),
                inserted: false,
            }
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

    #[test]
    fn replay_event_log_round_trips_typed_records_and_rejects_bounds() {
        let mut log = ReplayEventLog::new(MemoryEventLogBackend::new(1, 64), 64);
        log.append(11, record(4, b"payload")).unwrap();
        assert!(!log.append(11, record(9, b"different")).unwrap().inserted);
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
