//! Standalone durable state manager for canonical MultiPeer checkpoints.
//!
//! Each journal entry is a complete [`MultiPeerCheckpoint`], not a delta. This
//! makes a retained-prefix checkpoint independently recoverable and keeps AOI
//! and all peer outboxes in one commit unit. The manager is a lower-level
//! journal primitive; `MultiPeerAuthority` integration remains a separate step.

use core::fmt;

use crate::event_log::{
    EventLogBackend, EventLogCursor, ReplayEventLog, ReplayEventLogError, MAX_EVENT_LOG_BATCH,
};
use crate::multiplex::NetChannel;
use crate::plugin::multi_durable::{
    MultiPeerCheckpoint, MultiPeerCheckpointCodecError, MultiPeerCheckpointLimits,
};
use crate::replay::{ReplayDirection, ReplayRecord};

/// The event-log operation that failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultiPeerJournalOperation {
    /// Read and decode retained entries.
    Replay,
    /// Append a commit record.
    Append,
    /// Re-read the appended record for exact verification.
    Verify,
    /// Establish the durable barrier.
    Sync,
}

/// Recovery or commit error. A failed commit retains its exact pending record
/// inside the manager and must be retried with the same candidate.
#[derive(Debug)]
pub enum MultiPeerJournalError<E> {
    /// The backend operation failed. The pending commit, if any, is retained.
    Backend {
        /// Operation that failed.
        operation: MultiPeerJournalOperation,
        /// Provider error.
        source: E,
    },
    /// The ReplayRecord codec rejected a record.
    ReplayCodec(crate::replay::ReplayError),
    /// The checkpoint codec rejected or bounded the state.
    CheckpointCodec(MultiPeerCheckpointCodecError),
    /// Retained records do not form a valid journal.
    InvalidJournal(&'static str),
    /// A commit is pending and this retry supplied different state.
    PendingCandidateMismatch,
    /// The provider cursor changed outside this manager.
    CursorChanged {
        /// Expected next sequence.
        expected: u64,
        /// Actual next sequence.
        actual: u64,
    },
    /// The next event sequence cannot be represented.
    SequenceExhausted,
}

impl<E: fmt::Display> fmt::Display for MultiPeerJournalError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend { operation, source } => {
                write!(
                    formatter,
                    "MultiPeer journal {operation:?} failed: {source}"
                )
            }
            Self::ReplayCodec(error) => write!(formatter, "replay codec failed: {error:?}"),
            Self::CheckpointCodec(error) => {
                write!(formatter, "checkpoint codec failed: {error:?}")
            }
            Self::InvalidJournal(reason) => {
                write!(formatter, "invalid MultiPeer journal: {reason}")
            }
            Self::PendingCandidateMismatch => {
                formatter.write_str("pending MultiPeer commit must be retried byte-identically")
            }
            Self::CursorChanged { expected, actual } => write!(
                formatter,
                "MultiPeer journal cursor changed: expected {expected}, got {actual}"
            ),
            Self::SequenceExhausted => formatter.write_str("MultiPeer journal sequence exhausted"),
        }
    }
}

struct PendingCommit {
    candidate: MultiPeerCheckpoint,
    record: ReplayRecord,
    sequence: u64,
}

/// Durable journal over a caller-provided [`ReplayEventLog`].
pub struct MultiPeerDurableJournal<B> {
    log: ReplayEventLog<B>,
    limits: MultiPeerCheckpointLimits,
    state: MultiPeerCheckpoint,
    next_sequence: u64,
    pending: Option<PendingCommit>,
}

impl<B: EventLogBackend> MultiPeerDurableJournal<B> {
    /// Recover from the retained journal, or seed an empty sequence-zero log
    /// with `initial_checkpoint`. All retained records are validated before the
    /// manager is returned. The caller must not mutate the backend except via
    /// this manager for its lifetime.
    pub fn open(
        log: ReplayEventLog<B>,
        initial_checkpoint: MultiPeerCheckpoint,
        limits: MultiPeerCheckpointLimits,
    ) -> Result<Self, MultiPeerJournalError<B::Error>> {
        let initial_bytes = initial_checkpoint
            .encode(limits)
            .map_err(MultiPeerJournalError::CheckpointCodec)?;
        let start = log.backend().start_cursor().next_sequence();
        let end = log.backend().cursor().next_sequence();
        if start > end {
            return Err(MultiPeerJournalError::InvalidJournal(
                "retained start follows end cursor",
            ));
        }

        let mut journal = Self {
            log,
            limits,
            state: initial_checkpoint.clone(),
            next_sequence: end,
            pending: None,
        };

        if start == end {
            if start != 0 {
                return Err(MultiPeerJournalError::InvalidJournal(
                    "empty journal has a nonzero sequence range",
                ));
            }
            journal.commit(initial_checkpoint)?;
            return Ok(journal);
        }

        let mut cursor = EventLogCursor::new(start);
        let mut recovered_state = None;
        while cursor.next_sequence() < end {
            let batch = journal
                .log
                .replay(cursor, MAX_EVENT_LOG_BATCH)
                .map_err(|error| map_log_error(MultiPeerJournalOperation::Replay, error))?;
            if batch.end_cursor().next_sequence() != end {
                return Err(MultiPeerJournalError::InvalidJournal(
                    "event-log end cursor changed during recovery",
                ));
            }
            let mut expected = cursor.next_sequence();
            for event in batch.records() {
                let sequence = expected;
                if event.sequence != sequence
                    || event.idempotency_key != u128::from(sequence)
                    || event.record.sequence != sequence
                    || event.record.tick != sequence
                    || event.record.channel != NetChannel::EventsReliable
                    || event.record.direction != ReplayDirection::ServerToClient
                    || event.record.state_hash.is_some()
                {
                    return Err(MultiPeerJournalError::InvalidJournal(
                        "sequence, key, or replay envelope mismatch",
                    ));
                }
                let state = MultiPeerCheckpoint::decode(&event.record.payload, limits)
                    .map_err(MultiPeerJournalError::CheckpointCodec)?;
                if sequence == 0 && event.record.payload != initial_bytes {
                    return Err(MultiPeerJournalError::InvalidJournal(
                        "initial checkpoint/configuration mismatch",
                    ));
                }
                if recovered_state.replace(state).is_none() && sequence != start {
                    return Err(MultiPeerJournalError::InvalidJournal(
                        "first retained sequence mismatch",
                    ));
                }
                expected = expected
                    .checked_add(1)
                    .ok_or(MultiPeerJournalError::SequenceExhausted)?;
            }
            if expected == cursor.next_sequence()
                || batch.next_cursor().next_sequence() != expected
                || expected > end
            {
                return Err(MultiPeerJournalError::InvalidJournal(
                    "non-contiguous replay cursor",
                ));
            }
            cursor = batch.next_cursor();
        }
        journal.state = recovered_state.ok_or(MultiPeerJournalError::InvalidJournal(
            "retained journal contains no checkpoint",
        ))?;
        journal
            .log
            .backend_mut()
            .sync()
            .map_err(|source| MultiPeerJournalError::Backend {
                operation: MultiPeerJournalOperation::Sync,
                source,
            })?;
        Ok(journal)
    }

    /// Current state whose last journal record crossed the sync barrier.
    #[must_use]
    pub const fn state(&self) -> &MultiPeerCheckpoint {
        &self.state
    }

    /// Sequence assigned to the next commit.
    #[must_use]
    pub const fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Durably commit a full replacement checkpoint.
    ///
    /// Append, exact replay verification, cursor validation, and sync all
    /// complete before the published state or sequence advances. Any failure
    /// after candidate encoding keeps one immutable pending candidate/record;
    /// retries with another candidate are rejected.
    pub fn commit(
        &mut self,
        candidate: MultiPeerCheckpoint,
    ) -> Result<(), MultiPeerJournalError<B::Error>> {
        if let Some(pending) = &self.pending {
            if pending.candidate != candidate {
                return Err(MultiPeerJournalError::PendingCandidateMismatch);
            }
        } else {
            let sequence = self.next_sequence;
            let next = sequence
                .checked_add(1)
                .ok_or(MultiPeerJournalError::SequenceExhausted)?;
            let payload = candidate
                .encode(self.limits)
                .map_err(MultiPeerJournalError::CheckpointCodec)?;
            self.pending = Some(PendingCommit {
                candidate,
                record: ReplayRecord::new(
                    sequence,
                    sequence,
                    NetChannel::EventsReliable,
                    ReplayDirection::ServerToClient,
                    None,
                    payload,
                ),
                sequence,
            });
            debug_assert_eq!(next, sequence + 1);
        }

        let pending = self.pending.as_ref().expect("pending record installed");
        let sequence = pending.sequence;
        let following = sequence
            .checked_add(1)
            .ok_or(MultiPeerJournalError::SequenceExhausted)?;
        let actual_cursor = self.log.backend().cursor().next_sequence();
        if actual_cursor != sequence && actual_cursor != following {
            return Err(MultiPeerJournalError::CursorChanged {
                expected: sequence,
                actual: actual_cursor,
            });
        }
        self.log
            .append(u128::from(sequence), pending.record.clone())
            .map_err(|error| map_log_error(MultiPeerJournalOperation::Append, error))?;
        let actual_cursor = self.log.backend().cursor().next_sequence();
        if actual_cursor != following {
            return Err(MultiPeerJournalError::CursorChanged {
                expected: following,
                actual: actual_cursor,
            });
        }
        let replayed = self
            .log
            .replay(EventLogCursor::new(sequence), 1)
            .map_err(|error| map_log_error(MultiPeerJournalOperation::Verify, error))?;
        if replayed.records().len() != 1
            || replayed.next_cursor().next_sequence() != following
            || replayed.end_cursor().next_sequence() != following
            || replayed.records()[0].sequence != sequence
            || replayed.records()[0].idempotency_key != u128::from(sequence)
            || replayed.records()[0].record != pending.record
        {
            return Err(MultiPeerJournalError::InvalidJournal(
                "appended record did not replay exactly",
            ));
        }
        self.log
            .backend_mut()
            .sync()
            .map_err(|source| MultiPeerJournalError::Backend {
                operation: MultiPeerJournalOperation::Sync,
                source,
            })?;

        let committed = self.pending.take().expect("verified pending record");
        self.state = committed.candidate;
        self.next_sequence = following;
        Ok(())
    }
}

fn map_log_error<E>(
    operation: MultiPeerJournalOperation,
    error: ReplayEventLogError<E>,
) -> MultiPeerJournalError<E> {
    match error {
        ReplayEventLogError::Backend(source) => {
            MultiPeerJournalError::Backend { operation, source }
        }
        ReplayEventLogError::Replay(error) => MultiPeerJournalError::ReplayCodec(error),
        ReplayEventLogError::EmptyPayload => {
            MultiPeerJournalError::InvalidJournal("empty replay event")
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::*;
    use crate::event_log::{EventLogAppend, EventLogBatch, MemoryEventLogBackend};
    use std::sync::{Arc, Mutex};

    const LIMITS: MultiPeerCheckpointLimits = MultiPeerCheckpointLimits {
        max_total_bytes: 512,
        max_aoi_snapshot_bytes: 64,
        max_peers: 8,
        max_events_per_peer: 8,
        max_payload_bytes: 64,
    };

    fn state(aoi: u8) -> MultiPeerCheckpoint {
        MultiPeerCheckpoint {
            aoi_snapshot: vec![aoi],
            peers: Vec::new(),
        }
    }

    fn log(backend: SharedBackend) -> ReplayEventLog<SharedBackend> {
        ReplayEventLog::new(backend, LIMITS.max_total_bytes)
    }

    #[derive(Clone)]
    struct SharedBackend(Arc<Mutex<SharedState>>);

    struct SharedState {
        inner: MemoryEventLogBackend,
        fail_sync_once: bool,
        last_payload: Option<Vec<u8>>,
    }

    impl SharedBackend {
        fn new(capacity: usize) -> Self {
            Self(Arc::new(Mutex::new(SharedState {
                inner: MemoryEventLogBackend::new(capacity, LIMITS.max_total_bytes),
                fail_sync_once: false,
                last_payload: None,
            })))
        }
    }

    impl EventLogBackend for SharedBackend {
        type Error = &'static str;

        fn append(&mut self, key: u128, payload: &[u8]) -> Result<EventLogAppend, Self::Error> {
            let mut state = self.0.lock().unwrap();
            let result = state.inner.append(key, payload).map_err(|_| "append")?;
            state.last_payload = Some(payload.to_vec());
            Ok(result)
        }

        fn cursor(&self) -> EventLogCursor {
            self.0.lock().unwrap().inner.cursor()
        }

        fn start_cursor(&self) -> EventLogCursor {
            self.0.lock().unwrap().inner.start_cursor()
        }

        fn replay(
            &self,
            cursor: EventLogCursor,
            limit: usize,
        ) -> Result<EventLogBatch, Self::Error> {
            self.0
                .lock()
                .unwrap()
                .inner
                .replay(cursor, limit)
                .map_err(|_| "replay")
        }

        fn flush(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }

        fn sync(&mut self) -> Result<(), Self::Error> {
            let mut state = self.0.lock().unwrap();
            if state.fail_sync_once {
                state.fail_sync_once = false;
                return Err("sync");
            }
            state.inner.sync().map_err(|_| "sync")
        }
    }

    #[test]
    fn empty_log_seeds_and_restart_restores_latest_checkpoint() {
        let backend = SharedBackend::new(8);
        let mut journal =
            MultiPeerDurableJournal::open(log(backend.clone()), state(1), LIMITS).unwrap();
        assert_eq!(journal.state(), &state(1));
        assert_eq!(journal.next_sequence(), 1);
        journal.commit(state(2)).unwrap();
        let restored = MultiPeerDurableJournal::open(log(backend), state(1), LIMITS).unwrap();
        assert_eq!(restored.state(), &state(2));
        assert_eq!(restored.next_sequence(), 2);
    }

    #[test]
    fn sync_failure_keeps_exact_candidate_and_does_not_publish_early() {
        let backend = SharedBackend::new(8);
        let shared = backend.0.clone();
        let mut journal = MultiPeerDurableJournal::open(
            ReplayEventLog::new(backend, LIMITS.max_total_bytes),
            state(1),
            LIMITS,
        )
        .unwrap();
        let candidate = state(7);
        shared.lock().unwrap().fail_sync_once = true;
        assert!(matches!(
            journal.commit(candidate.clone()),
            Err(MultiPeerJournalError::Backend {
                operation: MultiPeerJournalOperation::Sync,
                ..
            })
        ));
        assert_eq!(journal.state(), &state(1));
        assert_eq!(journal.next_sequence(), 1);
        let first_bytes = shared.lock().unwrap().last_payload.clone().unwrap();
        assert!(matches!(
            journal.commit(state(8)),
            Err(MultiPeerJournalError::PendingCandidateMismatch)
        ));
        journal.commit(candidate.clone()).unwrap();
        assert_eq!(
            shared.lock().unwrap().last_payload.as_ref(),
            Some(&first_bytes)
        );
        assert_eq!(journal.state(), &candidate);
        assert_eq!(journal.next_sequence(), 2);
    }

    #[test]
    fn recovery_rejects_initial_configuration_mismatch_and_bounded_state() {
        let backend = SharedBackend::new(8);
        MultiPeerDurableJournal::open(log(backend.clone()), state(1), LIMITS).unwrap();
        assert!(matches!(
            MultiPeerDurableJournal::open(log(backend.clone()), state(2), LIMITS),
            Err(MultiPeerJournalError::InvalidJournal(
                "initial checkpoint/configuration mismatch"
            ))
        ));

        let mut tight = LIMITS;
        tight.max_total_bytes = 21;
        assert!(matches!(
            MultiPeerDurableJournal::open(log(SharedBackend::new(8)), state(1), tight),
            Err(MultiPeerJournalError::CheckpointCodec(_))
        ));
    }

    #[test]
    fn recovery_rejects_corrupt_and_noncontiguous_journals() {
        let backend = SharedBackend::new(8);
        let _ = MultiPeerDurableJournal::open(log(backend.clone()), state(1), LIMITS).unwrap();
        backend
            .0
            .lock()
            .unwrap()
            .inner
            .append(99, b"not a replay record")
            .unwrap();
        assert!(matches!(
            MultiPeerDurableJournal::open(log(backend), state(1), LIMITS),
            Err(MultiPeerJournalError::ReplayCodec(_))
        ));

        let backend = SharedBackend::new(8);
        let record = ReplayRecord::new(
            0,
            1,
            NetChannel::EventsReliable,
            ReplayDirection::ServerToClient,
            None,
            state(1).encode(LIMITS).unwrap(),
        );
        let mut typed = ReplayEventLog::new(backend.clone(), LIMITS.max_total_bytes);
        typed.append(1, record).unwrap();
        drop(typed);
        assert!(matches!(
            MultiPeerDurableJournal::open(log(backend), state(1), LIMITS),
            Err(MultiPeerJournalError::InvalidJournal(_))
        ));
    }
}
