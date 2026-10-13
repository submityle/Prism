//! Client-side prediction and reconciliation (design §10 "客户端预测与权威和解").
//!
//! A client cannot wait a round trip before reacting, so it applies its input
//! immediately (*predicts*) and keeps the un-acknowledged inputs. When the server's
//! authoritative change for a step arrives, the client:
//!
//! 1. applies the authoritative change to its acknowledged state,
//! 2. drops the local input the server has now confirmed,
//! 3. **rolls back** the remaining predicted state and **replays** the still
//!    un-acknowledged inputs on top of authority.
//!
//! If the prediction matched, replay reproduces it exactly (no visible change); if
//! it diverged, the state snaps to authority and the pending inputs re-apply on
//! top — standard rollback-and-resimulate reconciliation.
//!
//! The predicted state is a [`Record`] advanced by a caller-supplied `simulate`
//! closure, so one session drives any replicated field set the schema describes.
//! Everything rides on [`SnapshotStore`], so predicted revisions are addressable
//! exactly like authority's.

use alloc::vec::Vec;

use crate::delta::{DeltaError, Record, Schema};
use crate::snapshot::{SnapshotError, SnapshotStore};

/// A prediction/reconciliation failure.
#[derive(Clone, Debug, PartialEq)]
pub enum PredictionError {
    /// The authoritative message was malformed or encoded against another schema.
    Delta(DeltaError),
    /// The underlying store refused an operation.
    Snapshot(SnapshotError),
    /// The state at a revision the session needed was not retained.
    MissingState(u32),
    /// A revision counter would overflow `u32`.
    RevisionOverflow(u32),
    /// The authoritative change did not advance exactly one revision from the
    /// acknowledged step — a gap or a stale message; resync is required.
    UnexpectedRevision {
        /// The revision the client expected authority to confirm.
        expected: u32,
        /// The revision the authoritative change produced.
        found: u32,
    },
}

impl From<SnapshotError> for PredictionError {
    fn from(error: SnapshotError) -> Self {
        Self::Snapshot(error)
    }
}

impl From<DeltaError> for PredictionError {
    fn from(error: DeltaError) -> Self {
        Self::Delta(error)
    }
}

/// A client's predicted view of one replicated record.
#[derive(Clone, Debug)]
pub struct Prediction<F> {
    store: SnapshotStore,
    acked: u32,
    /// Local inputs applied since `acked`, oldest first.
    pending: Vec<F>,
}

impl<F: Clone> Prediction<F> {
    /// Start from the server's authoritative snapshot (its seed message).
    ///
    /// `capacity` is the snapshot ring size. It must exceed the largest number of
    /// un-acknowledged predictions the client will hold, or a replay evicts the
    /// acknowledged baseline and the next [`reconcile`](Self::reconcile) fails
    /// with [`SnapshotError::UnknownBaseline`] (resync from a fresh snapshot).
    ///
    /// # Errors
    /// [`PredictionError::Snapshot`] for a malformed snapshot.
    pub fn new(schema: Schema, capacity: usize, snapshot: &[u8]) -> Result<Self, PredictionError> {
        let mut store = SnapshotStore::new(schema, capacity);
        let acked = store.seed(snapshot)?;
        Ok(Self {
            store,
            acked,
            pending: Vec::new(),
        })
    }

    /// Reseed from a fresh authoritative snapshot (a resync): rebuild the store at
    /// the snapshot's revision, keeping the same schema and ring capacity, and drop
    /// every un-acknowledged prediction.
    ///
    /// # Errors
    /// [`PredictionError::Snapshot`] for a malformed snapshot.
    pub fn reseed(&mut self, snapshot: &[u8]) -> Result<u32, PredictionError> {
        let mut store = SnapshotStore::new(self.store.schema().clone(), self.store.capacity());
        let acked = store.seed(snapshot)?;
        self.store = store;
        self.acked = acked;
        self.pending.clear();
        Ok(acked)
    }

    /// The authoritative revision the server has confirmed.
    #[must_use]
    pub fn acked(&self) -> u32 {
        self.acked
    }

    /// The locally predicted revision (`acked` plus the un-acknowledged inputs).
    #[must_use]
    pub fn predicted(&self) -> u32 {
        self.acked.saturating_add(self.pending.len() as u32)
    }

    /// The currently predicted record.
    #[must_use]
    pub fn state(&self) -> Option<&Record> {
        self.store.record(self.predicted())
    }

    /// The un-acknowledged local inputs, oldest first.
    #[must_use]
    pub fn pending(&self) -> &[F] {
        &self.pending
    }

    /// Predict one local input, advancing the predicted state and returning its
    /// revision.
    ///
    /// # Errors
    /// [`PredictionError::MissingState`] when the current predicted state is not
    /// retained, or the store's errors.
    pub fn predict(
        &mut self,
        input: F,
        simulate: impl FnOnce(&mut Record, &F),
    ) -> Result<u32, PredictionError> {
        let base = self.predicted();
        let mut next = self
            .store
            .record(base)
            .cloned()
            .ok_or(PredictionError::MissingState(base))?;
        simulate(&mut next, &input);
        let revision = base
            .checked_add(1)
            .ok_or(PredictionError::RevisionOverflow(base))?;
        self.store.insert_baseline(revision, next)?;
        self.pending.push(input);
        Ok(revision)
    }

    /// Reconcile with the server's authoritative change for the next step: apply
    /// it, drop the confirmed local input, and replay the rest.
    ///
    /// Returns the resulting predicted revision. `simulate` is called once per
    /// replayed input.
    ///
    /// # Errors
    /// [`PredictionError::UnexpectedRevision`] when the authoritative change does
    /// not confirm exactly `acked + 1` (a gap or a stale message), or the store's
    /// errors.
    ///
    /// A message that does not name the acknowledged baseline is refused
    /// **before** the store is touched, so a stale or gap delta can never
    /// overwrite a retained frame on its way to the error. Not panic-safe against
    /// a `simulate` that panics mid-replay: the authoritative step is then applied
    /// and the replay is partial (the next `reconcile` resyncs from authority).
    pub fn reconcile(
        &mut self,
        authoritative: &[u8],
        simulate: impl FnMut(&mut Record, &F),
    ) -> Result<u32, PredictionError> {
        let expected = self
            .acked
            .checked_add(1)
            .ok_or(PredictionError::RevisionOverflow(self.acked))?;
        // Validate the message names the acknowledged baseline *before* touching
        // the store: `SnapshotStore::apply` writes at `baseline + 1` in place, so
        // applying a stale/gap delta first would overwrite a retained frame (which
        // `state()` may read) and only then fail the revision check.
        let baseline = crate::delta::decode_delta(self.store.schema(), authoritative)?.revision();
        if baseline != self.acked {
            return Err(PredictionError::UnexpectedRevision {
                expected,
                found: baseline.saturating_add(1),
            });
        }
        let new_acked = self.store.apply(authoritative)?;
        debug_assert_eq!(new_acked, expected);
        self.acked = new_acked;
        // The server has confirmed the input this client predicted for the step;
        // the oldest pending input (if any) is now authority's.
        if !self.pending.is_empty() {
            self.pending.remove(0);
        }
        self.replay(simulate)?;
        Ok(self.predicted())
    }

    /// Rebuild the predicted branch from the acknowledged state.
    fn replay(&mut self, mut simulate: impl FnMut(&mut Record, &F)) -> Result<(), PredictionError> {
        // Drop the stale predicted branch first; the replayed frames overwrite in
        // place, and a shorter pending list must not leave stale tail frames.
        self.store.truncate_after(self.acked);
        let mut revision = self.acked;
        for index in 0..self.pending.len() {
            let input = self.pending[index].clone();
            let mut next = self
                .store
                .record(revision)
                .cloned()
                .ok_or(PredictionError::MissingState(revision))?;
            simulate(&mut next, &input);
            let next_revision = revision
                .checked_add(1)
                .ok_or(PredictionError::RevisionOverflow(revision))?;
            self.store.insert_baseline(next_revision, next)?;
            revision = next_revision;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{encode_delta, encode_snapshot, FieldKind, FieldValue};
    use alloc::vec;
    use alloc::vec::Vec;

    fn schema() -> Schema {
        Schema::new(vec![FieldKind::Int { bits: 12 }])
    }

    fn record(value: i64) -> Record {
        Record::new(vec![FieldValue::Int(value)])
    }

    fn field(record: &Record) -> i64 {
        let FieldValue::Int(value) = record.value(0).expect("one field") else {
            panic!("field 0 is an int");
        };
        value
    }

    /// The simulate step: add the input to the single field.
    fn add(record: &mut Record, input: &i64) {
        record.set(0, FieldValue::Int(field(record) + input));
    }

    fn authoritative(base: &Record, delta: i64) -> Vec<u8> {
        let mut next = base.clone();
        add(&mut next, &delta);
        let mut bytes = Vec::new();
        encode_delta(&schema(), base, &next, 0, &mut bytes).unwrap();
        bytes
    }

    fn session() -> Prediction<i64> {
        let mut snapshot = Vec::new();
        encode_snapshot(&schema(), 0, &record(0), &mut snapshot).unwrap();
        Prediction::new(schema(), 8, &snapshot).unwrap()
    }

    #[test]
    fn a_divergent_prediction_rolls_back_to_authority() {
        let mut client = session();
        assert_eq!(client.acked(), 0);

        client.predict(5, add).unwrap();
        assert_eq!(field(client.state().unwrap()), 5);
        assert_eq!(client.predicted(), 1);

        // Authority applied +3, not the predicted +5: the state snaps to 3.
        let bytes = authoritative(&record(0), 3);
        assert_eq!(client.reconcile(&bytes, add).unwrap(), 1);
        assert_eq!(client.acked(), 1);
        assert_eq!(field(client.state().unwrap()), 3);
        assert!(client.pending().is_empty());
    }

    #[test]
    fn a_matching_prediction_survives_reconciliation() {
        let mut client = session();
        client.predict(3, add).unwrap();
        assert_eq!(field(client.state().unwrap()), 3);

        let bytes = authoritative(&record(0), 3);
        assert_eq!(client.reconcile(&bytes, add).unwrap(), 1);
        assert_eq!(field(client.state().unwrap()), 3, "the prediction held");
        assert_eq!(client.predicted(), 1);
        assert!(client.pending().is_empty());
    }

    #[test]
    fn unacknowledged_inputs_replay_after_a_correction() {
        let mut client = session();
        client.predict(5, add).unwrap(); // confirmed by authority (as +3)
        client.predict(7, add).unwrap(); // still unacknowledged
        assert_eq!(field(client.state().unwrap()), 12);
        assert_eq!(client.predicted(), 2);
        assert_eq!(client.pending(), &[5, 7]);

        let bytes = authoritative(&record(0), 3);
        assert_eq!(client.reconcile(&bytes, add).unwrap(), 2);
        // Authoritative base 3, with the one surviving local input (+7) replayed.
        assert_eq!(field(client.state().unwrap()), 10);
        assert_eq!(client.pending(), &[7]);
        assert_eq!(client.predicted(), 2);
    }

    #[test]
    fn a_gap_in_authority_is_refused() {
        let mut client = session();
        // An authoritative change whose baseline is 4, not the acked 0.
        let mut bytes = Vec::new();
        encode_delta(&schema(), &record(4), &record(6), 4, &mut bytes).unwrap();
        assert_eq!(
            client.reconcile(&bytes, add),
            Err(PredictionError::UnexpectedRevision {
                expected: 1,
                found: 5
            })
        );
        assert_eq!(client.acked(), 0, "the refused gap changed nothing");
    }

    /// A stale/gap message must be refused **without** overwriting a retained
    /// frame on its way to the error (regression: `reconcile` applied the delta —
    /// which writes at `baseline + 1` in place — before validating it, so a
    /// re-sent step corrupted the acknowledged state and the corruption then
    /// propagated through every later reconcile).
    #[test]
    fn a_stale_authoritative_message_does_not_corrupt_the_state() {
        let mut client = session();
        client.predict(5, add).unwrap();

        // Step 0 -> 1 is applied normally: authority says +3.
        let step = authoritative(&record(0), 3);
        assert_eq!(client.reconcile(&step, add).unwrap(), 1);
        assert_eq!(client.acked(), 1);
        assert_eq!(field(client.state().unwrap()), 3);

        // Re-send the already-applied step: baseline 0 < acked 1, so it would
        // land on revision 1 == acked and corrupt the state without the guard.
        assert_eq!(
            client.reconcile(&step, add),
            Err(PredictionError::UnexpectedRevision {
                expected: 2,
                found: 1
            })
        );
        assert_eq!(client.acked(), 1, "a refused message does not advance");
        assert_eq!(
            field(client.state().unwrap()),
            3,
            "the acknowledged state survived"
        );
        assert!(client.pending().is_empty());
    }
}
