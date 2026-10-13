//! End-to-end replication sessions (design §7–§10).
//!
//! The other modules are primitives; this one wires them into the two halves of
//! a real replication path:
//!
//! - [`ServerSession`] owns **authority** for one replicated record — a
//!   [`SnapshotStore`]. It seeds a joining client from a framed full snapshot,
//!   advances authority one step and emits the framed **delta**, and can
//!   re-emit a single catch-up delta from any retained baseline a lagging client
//!   still holds.
//! - [`ClientSession`] owns the **prediction** — a [`Prediction`] plus a
//!   [`FrameDecoder`]. It predicts local inputs at once and reconciles each
//!   authoritative frame the server sends, over whatever transport carried the
//!   bytes.
//!
//! Everything travels as [`frame`](crate::frame)-framed payloads, so a stream
//! transport can carry them verbatim; the two sessions never share state, only
//! bytes.

use alloc::vec::Vec;

use crate::delta::{encode_delta, DeltaError, Record, Schema};
use crate::frame::{encode_frame, FrameDecoder, FrameError};
use crate::prediction::{Prediction, PredictionError};
use crate::snapshot::{SnapshotError, SnapshotStore};

/// A session-level failure.
#[derive(Clone, Debug, PartialEq)]
pub enum SessionError {
    /// A delta could not be encoded.
    Delta(DeltaError),
    /// A frame was malformed or oversized on the wire.
    Frame(FrameError),
    /// The snapshot store refused an operation.
    Snapshot(SnapshotError),
    /// The client's prediction could not be reconciled.
    Prediction(PredictionError),
    /// The server has no authoritative baseline to serve.
    NoBaseline,
    /// A revision counter would overflow `u32`.
    RevisionOverflow(u32),
}

impl From<FrameError> for SessionError {
    fn from(error: FrameError) -> Self {
        Self::Frame(error)
    }
}

impl From<DeltaError> for SessionError {
    fn from(error: DeltaError) -> Self {
        Self::Delta(error)
    }
}

impl From<SnapshotError> for SessionError {
    fn from(error: SnapshotError) -> Self {
        Self::Snapshot(error)
    }
}

impl From<PredictionError> for SessionError {
    fn from(error: PredictionError) -> Self {
        Self::Prediction(error)
    }
}

/// The authoritative half for one replicated record.
#[derive(Clone, Debug)]
pub struct ServerSession {
    store: SnapshotStore,
}

impl ServerSession {
    /// Authority seeded from a full snapshot payload.
    ///
    /// # Errors
    /// [`SessionError::Snapshot`] for a malformed snapshot.
    pub fn new(schema: Schema, capacity: usize, snapshot: &[u8]) -> Result<Self, SessionError> {
        let mut store = SnapshotStore::new(schema, capacity);
        store.seed(snapshot)?;
        Ok(Self { store })
    }

    /// The current authoritative revision.
    #[must_use]
    pub fn revision(&self) -> Option<u32> {
        self.store.latest_revision()
    }

    /// The current authoritative record.
    #[must_use]
    pub fn record(&self) -> Option<&Record> {
        self.store
            .latest_revision()
            .and_then(|revision| self.store.record(revision))
    }

    /// Append a framed full snapshot (a joining / resyncing client's seed) and
    /// return the revision it carries.
    ///
    /// # Errors
    /// [`SessionError::NoBaseline`] when authority has no record.
    pub fn snapshot_frame(&self, out: &mut Vec<u8>) -> Result<u32, SessionError> {
        let revision = self.revision().ok_or(SessionError::NoBaseline)?;
        let mut payload = Vec::new();
        self.store.encode_snapshot(revision, &mut payload)?;
        encode_frame(&payload, out)?;
        Ok(revision)
    }

    /// Advance authority one step: `simulate` mutates the next record, and the
    /// framed delta from the previous revision is appended to `out`. Returns the
    /// new revision.
    ///
    /// # Errors
    /// [`SessionError::NoBaseline`] before any authority exists, or the store's
    /// errors.
    pub fn advance(
        &mut self,
        simulate: impl FnOnce(&mut Record),
        out: &mut Vec<u8>,
    ) -> Result<u32, SessionError> {
        let previous = self.revision().ok_or(SessionError::NoBaseline)?;
        let base = self
            .store
            .record(previous)
            .cloned()
            .ok_or(SessionError::NoBaseline)?;
        let mut next = base.clone();
        simulate(&mut next);

        let revision = previous
            .checked_add(1)
            .ok_or(SessionError::RevisionOverflow(previous))?;
        let mut payload = Vec::new();
        encode_delta(self.store.schema(), &base, &next, previous, &mut payload)?;
        let mut frame = Vec::new();
        encode_frame(&payload, &mut frame)?;
        // Commit only once both the delta and its frame are built, so a framing
        // error cannot advance authority without emitting the bytes for it.
        self.store.insert_baseline(revision, next)?;
        out.extend_from_slice(&frame);
        Ok(revision)
    }

    /// Append a framed catch-up delta covering exactly the one step above `from`.
    ///
    /// A delta header carries its **baseline** revision and the receiver infers
    /// the result as `baseline + 1` (see [`crate::delta`]), so a single frame can
    /// only advance a client by one step. This returns:
    ///
    /// - `true` and writes nothing when `from` already equals the current
    ///   revision (up to date);
    /// - `true` and writes a delta when `from + 1` is the current revision;
    /// - `false` (writing nothing) when the client is further behind — a gap the
    ///   delta format cannot bridge, so the caller must send a
    ///   [`snapshot_frame`](Self::snapshot_frame) and reseed the client.
    ///
    /// # Errors
    /// [`SessionError::NoBaseline`] / the store's errors.
    pub fn catch_up(&self, from: u32, out: &mut Vec<u8>) -> Result<bool, SessionError> {
        let latest = self.revision().ok_or(SessionError::NoBaseline)?;
        if from == latest {
            return Ok(true);
        }
        if from.checked_add(1) != Some(latest) {
            return Ok(false);
        }
        let Some(base) = self.store.record(from).cloned() else {
            return Ok(false);
        };
        let current = self
            .store
            .record(latest)
            .cloned()
            .ok_or(SessionError::NoBaseline)?;
        let mut payload = Vec::new();
        encode_delta(self.store.schema(), &base, &current, from, &mut payload)?;
        encode_frame(&payload, out)?;
        Ok(true)
    }
}

/// The client half: a prediction fed by inbound frames.
#[derive(Clone, Debug)]
pub struct ClientSession<F> {
    prediction: Prediction<F>,
    decoder: FrameDecoder,
}

impl<F: Clone> ClientSession<F> {
    /// A client seeded from a full snapshot payload.
    ///
    /// # Errors
    /// [`SessionError::Snapshot`] for a malformed snapshot.
    pub fn new(schema: Schema, capacity: usize, snapshot: &[u8]) -> Result<Self, SessionError> {
        Ok(Self {
            prediction: Prediction::new(schema, capacity, snapshot)?,
            decoder: FrameDecoder::new(),
        })
    }

    /// The authoritative revision confirmed by the server.
    #[must_use]
    pub fn acked(&self) -> u32 {
        self.prediction.acked()
    }

    /// The locally predicted revision.
    #[must_use]
    pub fn predicted(&self) -> u32 {
        self.prediction.predicted()
    }

    /// The currently predicted record.
    #[must_use]
    pub fn state(&self) -> Option<&Record> {
        self.prediction.state()
    }

    /// Predict one local input immediately.
    ///
    /// # Errors
    /// [`SessionError::Prediction`].
    pub fn predict(
        &mut self,
        input: F,
        simulate: impl FnOnce(&mut Record, &F),
    ) -> Result<u32, SessionError> {
        Ok(self.prediction.predict(input, simulate)?)
    }

    /// Feed received stream bytes and reconcile each complete delta frame,
    /// returning the predicted revision after each applied frame.
    ///
    /// A frame that fails reconciliation is **left buffered** rather than dropped,
    /// so a resumed/duplicated stream cannot silently lose an authoritative step:
    /// the caller can observe [`buffered`](Self::buffered) staying non-zero and
    /// resync from a fresh snapshot. A stream that carries a stale or gap delta out
    /// of order therefore stalls (and is visible) instead of diverging.
    ///
    /// # Errors
    /// [`SessionError::Frame`] for a malformed stream and
    /// [`SessionError::Prediction`] when a frame cannot be reconciled.
    pub fn receive(
        &mut self,
        bytes: &[u8],
        mut simulate: impl FnMut(&mut Record, &F),
    ) -> Result<Vec<u32>, SessionError> {
        self.decoder.feed(bytes);
        let mut revisions = Vec::new();
        // Peek so a frame that fails reconciliation stays buffered; only a frame
        // that applied cleanly is consumed.
        while let Some(frame) = self.decoder.peek_frame()? {
            let revision = self.prediction.reconcile(&frame, &mut simulate)?;
            self.decoder.next_frame()?;
            revisions.push(revision);
        }
        Ok(revisions)
    }

    /// Bytes still buffered in the frame decoder — non-zero after a
    /// [`receive`](Self::receive) error means the session is desynchronized and
    /// needs a fresh snapshot.
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.decoder.buffered()
    }

    /// Reseed from a fresh authoritative **snapshot** frame — the resync a client
    /// runs after [`receive`](Self::receive) reports a gap or a desync, using the
    /// bytes [`ServerReplication::seed`](crate::plugin::ServerReplication::seed)
    /// produced. Rebuilds the prediction at the snapshot's revision and drops every
    /// pending input; returns the new acknowledged revision.
    ///
    /// # Errors
    /// [`SessionError::Frame`] when the input is not exactly one complete frame,
    /// or [`SessionError::Snapshot`] for a malformed snapshot.
    pub fn reseed(&mut self, snapshot_frame: &[u8]) -> Result<u32, SessionError> {
        let mut decoder = FrameDecoder::new();
        decoder.feed(snapshot_frame);
        let snapshot = decoder
            .next_frame()?
            .ok_or(SessionError::Frame(FrameError::Truncated))?;
        let acked = self.prediction.reseed(&snapshot)?;
        self.decoder = FrameDecoder::new();
        Ok(acked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{encode_snapshot, FieldKind, FieldValue};
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

    /// The prediction's simulate step: add the input to the single field.
    fn add(record: &mut Record, input: &i64) {
        record.set(0, FieldValue::Int(field(record) + input));
    }

    /// A raw (unframed) full snapshot payload at `revision`.
    fn snapshot_at(revision: u32, value: i64) -> Vec<u8> {
        let mut bytes = Vec::new();
        encode_snapshot(&schema(), revision, &record(value), &mut bytes).unwrap();
        bytes
    }

    /// Split a byte stream into its frame payloads.
    fn frames(bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut decoder = FrameDecoder::new();
        decoder.feed(bytes);
        let mut out = Vec::new();
        while let Some(frame) = decoder.next_frame().unwrap() {
            out.push(frame);
        }
        out
    }

    #[test]
    fn a_seeded_client_reconciles_the_authoritative_step() {
        let mut server = ServerSession::new(schema(), 8, &snapshot_at(0, 0)).unwrap();

        // Seed the client from the framed snapshot.
        let mut seed = Vec::new();
        server.snapshot_frame(&mut seed).unwrap();
        let mut client = ClientSession::<i64>::new(schema(), 8, &frames(&seed)[0]).unwrap();
        assert_eq!(client.acked(), 0);

        // The client predicts +5; authority applies +3.
        client.predict(5, add).unwrap();
        assert_eq!(field(client.state().unwrap()), 5);
        let mut wire = Vec::new();
        server
            .advance(
                |record| record.set(0, FieldValue::Int(field(record) + 3)),
                &mut wire,
            )
            .unwrap();

        // The authoritative frame rolls the prediction back to 3.
        assert_eq!(client.receive(&wire, add).unwrap(), vec![1]);
        assert_eq!(client.acked(), 1);
        assert_eq!(field(client.state().unwrap()), 3);
    }

    /// A single frame can only advance a client by one step (a delta names its
    /// baseline), so a one-step gap is bridged and anything larger needs a
    /// snapshot resync.
    #[test]
    fn catch_up_covers_a_one_step_gap_only() {
        let mut server = ServerSession::new(schema(), 8, &snapshot_at(0, 0)).unwrap();
        for _ in 0..3 {
            server
                .advance(
                    |record| record.set(0, FieldValue::Int(field(record) + 10)),
                    &mut Vec::new(),
                )
                .unwrap();
        }
        assert_eq!(server.revision(), Some(3));

        // A client one step behind (at 2) gets a delta covering 2 -> 3.
        let mut wire = Vec::new();
        assert!(server.catch_up(2, &mut wire).unwrap());
        let mut client = ClientSession::<i64>::new(schema(), 8, &snapshot_at(2, 20)).unwrap();
        assert_eq!(client.receive(&wire, add).unwrap(), vec![3]);
        assert_eq!(field(client.state().unwrap()), 30);

        // Further behind (at 1) cannot be bridged by one delta: resync needed.
        let mut none = Vec::new();
        assert!(!server.catch_up(1, &mut none).unwrap());
        assert!(none.is_empty());
        // A revision the server no longer retains is refused for the same reason.
        assert!(!server.catch_up(99, &mut none).unwrap());
        // Up to date: nothing to send.
        assert!(server.catch_up(3, &mut none).unwrap());
        assert!(none.is_empty());
    }

    #[test]
    fn a_client_at_the_current_revision_gets_nothing() {
        let server = ServerSession::new(schema(), 8, &snapshot_at(4, 1)).unwrap();
        let mut wire = Vec::new();
        assert!(server.catch_up(4, &mut wire).unwrap());
        assert!(wire.is_empty());
    }

    /// A delta split across two reads is buffered until its frame completes.
    #[test]
    fn an_incomplete_delta_frame_waits_for_the_rest() {
        let mut server = ServerSession::new(schema(), 8, &snapshot_at(0, 0)).unwrap();
        let mut wire = Vec::new();
        server
            .advance(
                |record| record.set(0, FieldValue::Int(field(record) + 3)),
                &mut wire,
            )
            .unwrap();

        let mut client = ClientSession::<i64>::new(schema(), 8, &snapshot_at(0, 0)).unwrap();
        let (head, tail) = wire.split_at(wire.len() - 1);
        assert!(client.receive(head, add).unwrap().is_empty());
        assert_eq!(client.receive(tail, add).unwrap(), vec![1]);
    }

    /// A delta that names a baseline ahead of the client is a gap: it is **kept**
    /// buffered (so the desync is visible), not silently dropped, and a fresh
    /// snapshot recovers with `reseed` instead of the client diverging quietly.
    #[test]
    fn a_gap_delta_is_left_buffered_and_visible() {
        let mut server = ServerSession::new(schema(), 8, &snapshot_at(0, 0)).unwrap();
        let mut first = Vec::new();
        server
            .advance(
                |record| record.set(0, FieldValue::Int(field(record) + 1)),
                &mut first,
            )
            .unwrap();
        let mut second = Vec::new();
        server
            .advance(
                |record| record.set(0, FieldValue::Int(field(record) + 1)),
                &mut second,
            )
            .unwrap();

        let mut client = ClientSession::<i64>::new(schema(), 8, &snapshot_at(0, 0)).unwrap();
        // The 1->2 delta arrives first: it names baseline 1, but the client is at 0.
        assert!(client.receive(&second, add).is_err());
        assert!(
            client.buffered() > 0,
            "the un-applied frame is kept, so the desync is observable"
        );
        assert_eq!(client.acked(), 0, "the client did not advance onto a gap");

        // Resyncing from a fresh snapshot frame recovers cleanly.
        let mut seed = Vec::new();
        server.snapshot_frame(&mut seed).unwrap();
        assert_eq!(client.reseed(&seed).unwrap(), 2);
        assert_eq!(client.buffered(), 0, "reseed discards the stale stream");
    }
}
