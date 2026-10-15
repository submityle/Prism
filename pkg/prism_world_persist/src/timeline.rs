use alloc::vec::Vec;

use crate::{ActorId, AppendOnlyLog, DeltaError, LogicalClock, OverrideDelta, PersistentId};
use crate::{LogError, Transaction, TransactionId, WorldState};

/// A replay checkpoint containing materialized state at the next sequence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Snapshot {
    sequence: u64,
    state: WorldState,
}

impl Snapshot {
    /// Creates a snapshot. `sequence` is the first log sequence not included.
    pub fn new(sequence: u64, state: WorldState) -> Self {
        Self { sequence, state }
    }

    /// Returns the first transaction sequence not included in this snapshot.
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Returns the immutable materialized state.
    pub const fn state(&self) -> &WorldState {
        &self.state
    }
}

/// Timeline errors, including log validation and empty history operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TimelineError {
    /// The underlying append-only log rejected a transaction.
    Log(LogError),
    /// The transaction's inverse did not cover the same fields.
    InvalidDelta(DeltaError),
    /// There is no local transaction left to undo.
    NothingToUndo,
    /// There is no locally undone transaction left to redo.
    NothingToRedo,
    /// A snapshot starts after the supplied log's end.
    SnapshotAhead {
        /// Sequence requested by the snapshot.
        snapshot: u64,
        /// Sequence available at the end of the log.
        log_end: u64,
    },
}

#[derive(Clone, Debug)]
struct HistoryEntry {
    object: PersistentId,
    forward: OverrideDelta,
    inverse: OverrideDelta,
}

/// Event-sourced world state with snapshots and transaction-backed undo/redo.
#[derive(Clone, Debug, Default)]
pub struct Timeline {
    log: AppendOnlyLog,
    state: WorldState,
    undo: Vec<HistoryEntry>,
    redo: Vec<HistoryEntry>,
}

impl Timeline {
    /// Creates an empty timeline.
    pub const fn new() -> Self {
        Self {
            log: AppendOnlyLog::new(),
            state: WorldState::new(),
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    /// Commits a field delta and records the exact prior values for undo.
    pub fn commit(
        &mut self,
        id: TransactionId,
        actor: ActorId,
        clock: LogicalClock,
        object: PersistentId,
        delta: OverrideDelta,
    ) -> Result<Transaction, TimelineError> {
        let inverse = self.state.inverse_for(object, &delta);
        let transaction = Transaction::new(
            id,
            actor,
            clock,
            self.log.next_sequence(),
            object,
            delta,
            inverse,
        )
        .map_err(TimelineError::InvalidDelta)?;
        self.append_internal(transaction.clone())?;
        Ok(transaction)
    }

    /// Appends a prebuilt transaction and makes its recorded inverse undoable.
    pub fn append(&mut self, transaction: Transaction) -> Result<(), TimelineError> {
        self.append_internal(transaction)
    }

    fn append_internal(&mut self, transaction: Transaction) -> Result<(), TimelineError> {
        self.append_internal_with_redo(transaction, true)
    }

    fn append_internal_with_redo(
        &mut self,
        transaction: Transaction,
        clear_redo: bool,
    ) -> Result<(), TimelineError> {
        self.log
            .append(transaction.clone())
            .map_err(TimelineError::Log)?;
        self.state
            .apply_delta(transaction.object(), transaction.delta());
        self.undo.push(HistoryEntry {
            object: transaction.object(),
            forward: transaction.delta().clone(),
            inverse: transaction.inverse().clone(),
        });
        if clear_redo {
            self.redo.clear();
        }
        Ok(())
    }

    /// Appends a reverse transaction for the most recent local change.
    pub fn undo(
        &mut self,
        id: TransactionId,
        actor: ActorId,
        clock: LogicalClock,
    ) -> Result<Transaction, TimelineError> {
        let entry = self.undo.pop().ok_or(TimelineError::NothingToUndo)?;
        let inverse_of_inverse = self.state.inverse_for(entry.object, &entry.inverse);
        let transaction = Transaction::new(
            id,
            actor,
            clock,
            self.log.next_sequence(),
            entry.object,
            entry.inverse.clone(),
            inverse_of_inverse,
        )
        .map_err(TimelineError::InvalidDelta)?;
        if let Err(error) = self.append_internal_with_redo(transaction.clone(), false) {
            self.undo.push(entry);
            return Err(error);
        }
        self.undo.pop();
        self.redo.push(entry);
        Ok(transaction)
    }

    /// Appends a forward transaction for the most recently undone change.
    pub fn redo(
        &mut self,
        id: TransactionId,
        actor: ActorId,
        clock: LogicalClock,
    ) -> Result<Transaction, TimelineError> {
        let entry = self.redo.pop().ok_or(TimelineError::NothingToRedo)?;
        let inverse = self.state.inverse_for(entry.object, &entry.forward);
        let transaction = Transaction::new(
            id,
            actor,
            clock,
            self.log.next_sequence(),
            entry.object,
            entry.forward.clone(),
            inverse,
        )
        .map_err(TimelineError::InvalidDelta)?;
        if let Err(error) = self.append_internal_with_redo(transaction.clone(), false) {
            self.redo.push(entry);
            return Err(error);
        }
        self.undo.pop();
        self.undo.push(entry);
        Ok(transaction)
    }

    /// Captures the current state at the next log sequence.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot::new(self.log.next_sequence(), self.state.clone())
    }

    /// Returns the current materialized state.
    pub const fn state(&self) -> &WorldState {
        &self.state
    }

    /// Returns the immutable append-only log.
    pub const fn log(&self) -> &AppendOnlyLog {
        &self.log
    }

    /// Replays all log entries after the snapshot sequence.
    pub fn replay(snapshot: &Snapshot, log: &AppendOnlyLog) -> Result<WorldState, TimelineError> {
        if snapshot.sequence > log.next_sequence() {
            return Err(TimelineError::SnapshotAhead {
                snapshot: snapshot.sequence,
                log_end: log.next_sequence(),
            });
        }
        let mut state = snapshot.state.clone();
        for transaction in log.iter() {
            if transaction.sequence() >= snapshot.sequence {
                state.apply_delta(transaction.object(), transaction.delta());
            }
        }
        Ok(state)
    }
}
