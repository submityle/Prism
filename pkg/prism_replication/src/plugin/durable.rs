//! Single-authority write-ahead journal. One backend belongs to exactly one
//! authority and one writer; sharing it with unrelated replay traffic is invalid.

use alloc::{boxed::Box, string::String, vec::Vec};
use core::fmt::Debug;

use crate::delta::{apply_delta, FieldKind, Schema};
use crate::event_log::{EventLogBackend, EventLogCursor, ReplayEventLog};
use crate::frame::{FrameDecoder, FrameError};
use crate::multiplex::NetChannel;
use crate::replay::{ReplayDirection, ReplayRecord};
use crate::session::{ServerSession, SessionError};

use super::{encode_delta, encode_snapshot};

/// The journal operation that failed. Provider errors are retained as diagnostic
/// text so the app resource does not depend on a concrete backend's error type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DurableLogOperation {
    /// Encoding/appending an idempotent event.
    Append,
    /// Reading and decoding a journal batch.
    Replay,
    /// Establishing the provider's durability barrier.
    Sync,
    /// Replacing a full retained prefix with a complete authority checkpoint.
    Checkpoint,
}

/// Failure to open, replay, or commit an authority journal.
#[derive(Clone, Debug, PartialEq)]
pub enum DurableRecoveryError {
    /// A provider or replay codec failed; retry preserves the staged transaction.
    Log {
        /// Operation to retry (a poisoned file provider instead requires reopen).
        operation: DurableLogOperation,
        /// The original provider/codec error's debug representation.
        message: String,
    },
    /// Session encoding or replay failed.
    Session(SessionError),
    /// Wrong schema/seed, foreign records, gaps, or conflicting idempotency keys.
    InvalidJournal(&'static str),
}

impl From<SessionError> for DurableRecoveryError {
    fn from(error: SessionError) -> Self {
        Self::Session(error)
    }
}

impl core::fmt::Display for DurableRecoveryError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Log { operation, message } => {
                write!(formatter, "authority journal {operation:?}: {message}")
            }
            Self::Session(error) => write!(formatter, "authority session: {error:?}"),
            Self::InvalidJournal(message) => {
                write!(formatter, "invalid authority journal: {message}")
            }
        }
    }
}

impl std::error::Error for DurableRecoveryError {}

fn log_error(operation: DurableLogOperation, error: impl Debug) -> DurableRecoveryError {
    DurableRecoveryError::Log {
        operation,
        message: std::format!("{error:?}"),
    }
}

pub(super) trait AuthorityJournal: Send + Sync {
    fn commit(
        &mut self,
        record: &ReplayRecord,
        baseline: &ServerSession,
    ) -> Result<(), DurableRecoveryError>;
}

struct Journal<B> {
    log: ReplayEventLog<B>,
    seed: ReplayRecord,
    pending_checkpoint: Option<ReplayRecord>,
}

impl<B> AuthorityJournal for Journal<B>
where
    B: EventLogBackend + Send + Sync,
    B::Error: Debug,
{
    fn commit(
        &mut self,
        record: &ReplayRecord,
        baseline: &ServerSession,
    ) -> Result<(), DurableRecoveryError> {
        let sequence = record.sequence;
        let next = sequence
            .checked_add(1)
            .ok_or(DurableRecoveryError::InvalidJournal(
                "journal sequence exhausted",
            ))?;
        // An earlier failed append may already have inserted this exact record.
        // Never accept a provider's key-only deduplication as proof of equality.
        let end = self.log.backend().cursor().next_sequence();
        if end != sequence && end != next {
            return Err(DurableRecoveryError::InvalidJournal(
                "unexpected journal end",
            ));
        }
        // Only reclaim a prefix ending at the published baseline. In particular,
        // never compact an appended-but-unpublished candidate on an ambiguous retry.
        if self.pending_checkpoint.is_none()
            && end == sequence
            && self.log.backend().remaining_capacity() == Some(0)
            && end.saturating_sub(self.log.backend().start_cursor().next_sequence()) > 1
        {
            self.pending_checkpoint = Some(checkpoint_record(&self.seed, baseline, end - 1)?);
        }
        if let Some(checkpoint) = self.pending_checkpoint.as_ref() {
            if let Some(result) = self
                .log
                .checkpoint(u128::from(checkpoint.sequence), checkpoint.clone())
            {
                result.map_err(|error| log_error(DurableLogOperation::Checkpoint, error))?;
            }
            // Unsupported providers keep their original bounded backpressure.
            self.pending_checkpoint = None;
        }
        let outcome = self
            .log
            .append(u128::from(sequence), record.clone())
            .map_err(|error| log_error(DurableLogOperation::Append, error))?;
        let batch = self
            .log
            .replay(EventLogCursor::new(sequence), 1)
            .map_err(|error| log_error(DurableLogOperation::Replay, error))?;
        if outcome.next_cursor.next_sequence() != next
            || self.log.backend().cursor().next_sequence() != next
            || batch.next_cursor().next_sequence() != next
            || batch.end_cursor().next_sequence() != next
            || batch.records().len() != 1
            || batch.records()[0].sequence != sequence
            || batch.records()[0].idempotency_key != u128::from(sequence)
            || batch.records()[0].record != *record
        {
            return Err(DurableRecoveryError::InvalidJournal("conflicting append"));
        }
        self.log
            .backend_mut()
            .sync()
            .map_err(|error| log_error(DurableLogOperation::Sync, error))
    }
}

pub(super) struct RestoredAuthority {
    pub session: ServerSession,
    pub journal: Box<dyn AuthorityJournal>,
    pub next_sequence: u64,
}

/// A bounded in-flight transaction. The head input is not popped until this
/// candidate is both durable and retained by the reliable event queue.
pub(super) struct PreparedCommit {
    pub session: ServerSession,
    pub record: ReplayRecord,
    // A failed append or sync may already have persisted the candidate. Once
    // attempted, even a later transport refusal must not discard or replace it.
    pub journal_attempted: bool,
    pub synced: bool,
}

pub(super) fn delta_record(sequence: u64, revision: u32, wire: &[u8]) -> ReplayRecord {
    ReplayRecord::new(
        u64::from(revision),
        sequence,
        NetChannel::EventsReliable,
        ReplayDirection::ServerToClient,
        None,
        encode_delta(wire),
    )
}

// Persist exact schema parameters, not just field count: the snapshot codec
// otherwise cannot distinguish equal-width fields with different semantics.
fn seed_record(schema: &Schema, session: &ServerSession) -> Result<ReplayRecord, SessionError> {
    let mut payload = Vec::from(&b"PRA1"[..]);
    payload.extend_from_slice(&(schema.len() as u64).to_le_bytes());
    for kind in schema.kinds() {
        match *kind {
            FieldKind::Bool => payload.push(0),
            FieldKind::Int { bits } => payload.extend_from_slice(&[1, bits]),
            FieldKind::Float { min, max, bits } => {
                payload.extend_from_slice(&[2, bits]);
                payload.extend_from_slice(&min.to_bits().to_le_bytes());
                payload.extend_from_slice(&max.to_bits().to_le_bytes());
            }
        }
    }
    let mut wire = Vec::new();
    let revision = session.snapshot_frame(&mut wire)?;
    payload.extend_from_slice(&encode_snapshot(&wire));
    Ok(ReplayRecord::new(
        u64::from(revision),
        0,
        NetChannel::EventsReliable,
        ReplayDirection::ServerToClient,
        None,
        payload,
    ))
}

fn checkpoint_record(
    seed: &ReplayRecord,
    session: &ServerSession,
    sequence: u64,
) -> Result<ReplayRecord, SessionError> {
    let mut payload = Vec::from(&b"PRC1"[..]);
    payload.extend_from_slice(&(seed.payload.len() as u64).to_le_bytes());
    payload.extend_from_slice(&seed.payload);
    let mut wire = Vec::new();
    let revision = session.snapshot_frame(&mut wire)?;
    payload.extend_from_slice(&wire);
    Ok(ReplayRecord::new(
        u64::from(revision),
        sequence,
        NetChannel::EventsReliable,
        ReplayDirection::ServerToClient,
        None,
        payload,
    ))
}

fn restore_checkpoint(
    schema: &Schema,
    capacity: usize,
    seed: &ReplayRecord,
    record: &ReplayRecord,
) -> Result<ServerSession, DurableRecoveryError> {
    let prefix =
        12usize
            .checked_add(seed.payload.len())
            .ok_or(DurableRecoveryError::InvalidJournal(
                "checkpoint length overflow",
            ))?;
    if record.payload.get(..4) != Some(b"PRC1")
        || record.payload.get(4..12) != Some(&(seed.payload.len() as u64).to_le_bytes())
        || record.payload.get(12..prefix) != Some(seed.payload.as_slice())
        || record.channel != NetChannel::EventsReliable
        || record.direction != ReplayDirection::ServerToClient
        || record.state_hash.is_some()
        || seed.tick.checked_add(record.sequence) != Some(record.tick)
    {
        return Err(DurableRecoveryError::InvalidJournal(
            "invalid checkpoint schema/seed/envelope",
        ));
    }
    let wire = record
        .payload
        .get(prefix..)
        .ok_or(DurableRecoveryError::InvalidJournal("truncated checkpoint"))?;
    let mut decoder = FrameDecoder::new();
    decoder.feed(wire);
    let snapshot = decoder
        .next_frame()
        .map_err(SessionError::from)?
        .ok_or(SessionError::Frame(FrameError::Truncated))?;
    let session = ServerSession::new(schema.clone(), capacity, &snapshot)?;
    // Re-encoding checks revision, canonical snapshot bytes, and trailing frames.
    if checkpoint_record(seed, &session, record.sequence)? != *record {
        return Err(DurableRecoveryError::InvalidJournal(
            "noncanonical checkpoint",
        ));
    }
    Ok(session)
}

pub(super) fn restore<B>(
    schema: &Schema,
    capacity: usize,
    snapshot: &[u8],
    log: ReplayEventLog<B>,
) -> Result<RestoredAuthority, DurableRecoveryError>
where
    B: EventLogBackend + Send + Sync + 'static,
    B::Error: Debug,
{
    let mut session = ServerSession::new(schema.clone(), capacity, snapshot)?;
    let seed = seed_record(schema, &session)?;
    let mut journal = Journal {
        log,
        seed: seed.clone(),
        pending_checkpoint: None,
    };
    let end = journal.log.backend().cursor().next_sequence();
    if end == 0 {
        if journal.log.backend().start_cursor().next_sequence() != 0 {
            return Err(DurableRecoveryError::InvalidJournal(
                "empty nonzero journal",
            ));
        }
        journal.commit(&seed, &session)?;
        return Ok(RestoredAuthority {
            session,
            journal: Box::new(journal),
            next_sequence: 1,
        });
    }
    let mut cursor = journal.log.backend().start_cursor();
    let start = cursor.next_sequence();
    if start >= end {
        return Err(DurableRecoveryError::InvalidJournal(
            "invalid retained journal range",
        ));
    }
    while cursor.next_sequence() < end {
        let batch = journal
            .log
            .replay(cursor, crate::event_log::MAX_EVENT_LOG_BATCH)
            .map_err(|error| log_error(DurableLogOperation::Replay, error))?;
        let mut expected = cursor.next_sequence();
        for event in batch.records() {
            let record = &event.record;
            if expected >= end
                || event.sequence != expected
                || event.idempotency_key != u128::from(expected)
                || record.sequence != expected
            {
                return Err(DurableRecoveryError::InvalidJournal(
                    "journal sequence/key gap",
                ));
            }
            if expected == start && record.payload.starts_with(b"PRC1") {
                session = restore_checkpoint(schema, capacity, &seed, record)?;
            } else if expected == 0 {
                if *record != seed {
                    return Err(DurableRecoveryError::InvalidJournal(
                        "schema or initial snapshot mismatch",
                    ));
                }
            } else if expected == start {
                return Err(DurableRecoveryError::InvalidJournal(
                    "retained prefix has no checkpoint",
                ));
            } else {
                replay_delta(schema, &mut session, record)?;
            }
            expected += 1;
        }
        if expected <= cursor.next_sequence()
            || expected > end
            || batch.next_cursor().next_sequence() != expected
            || batch.end_cursor().next_sequence() != end
        {
            return Err(DurableRecoveryError::InvalidJournal(
                "invalid replay cursor",
            ));
        }
        cursor = batch.next_cursor();
    }
    // A complete record may have survived an ambiguous append/sync failure.
    // Adopt the valid prefix, then establish a fresh barrier before exposing it.
    journal
        .log
        .backend_mut()
        .sync()
        .map_err(|error| log_error(DurableLogOperation::Sync, error))?;
    Ok(RestoredAuthority {
        session,
        journal: Box::new(journal),
        next_sequence: end,
    })
}

fn replay_delta(
    schema: &Schema,
    session: &mut ServerSession,
    record: &ReplayRecord,
) -> Result<(), DurableRecoveryError> {
    let previous = session.revision().ok_or(SessionError::NoBaseline)?;
    let revision = previous
        .checked_add(1)
        .ok_or(SessionError::RevisionOverflow(previous))?;
    if record.tick != u64::from(revision)
        || record.channel != NetChannel::EventsReliable
        || record.direction != ReplayDirection::ServerToClient
        || record.state_hash.is_some()
        || record.payload.first() != Some(&super::EVENT_DELTA)
    {
        return Err(DurableRecoveryError::InvalidJournal(
            "invalid authority delta envelope",
        ));
    }
    let mut decoder = FrameDecoder::new();
    decoder.feed(&record.payload[1..]);
    let payload = decoder
        .next_frame()
        .map_err(SessionError::from)?
        .ok_or(SessionError::Frame(FrameError::Truncated))?;
    if decoder.buffered() != 0 {
        return Err(DurableRecoveryError::InvalidJournal(
            "delta contains trailing frames",
        ));
    }
    let mut next = session.record().cloned().ok_or(SessionError::NoBaseline)?;
    apply_delta(schema, &mut next, previous, &payload).map_err(SessionError::from)?;
    let mut candidate = session.clone();
    let mut wire = Vec::new();
    candidate.advance(|record| *record = next, &mut wire)?;
    // Reject noncanonical/trailing payload bytes as well as revision gaps.
    if encode_delta(&wire) != record.payload {
        return Err(DurableRecoveryError::InvalidJournal(
            "noncanonical authority delta",
        ));
    }
    *session = candidate;
    Ok(())
}
