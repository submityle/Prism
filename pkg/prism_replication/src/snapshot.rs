//! Baseline snapshots + delta application (design §7.2 / §10 "瞬时状态快照 delta").
//!
//! [`crate::delta`] encodes one change; this module is the store that makes the
//! change *usable*. A [`SnapshotStore`] keeps a bounded ring of full baseline
//! [`Record`]s addressed by revision, so a receiver can:
//!
//! - apply a delta to the retained baseline it names, materialising the next
//!   baseline — the forward path a client runs as authoritative deltas arrive;
//! - read the baseline at any retained revision — the seed a late-join or a
//!   resync snapshots from;
//! - roll back to a retained revision when a delta arrives out of order or a
//!   prediction is rejected (the caller re-applies the intervening deltas).
//!
//! The ring is bounded, so the oldest baseline is evicted as newer ones land: a
//! delta whose baseline has been evicted cannot be applied and is refused
//! ([`SnapshotError::UnknownBaseline`]) — the caller must resync from a snapshot
//! rather than guess. Revisions are fed in increasing order.

use alloc::collections::VecDeque;

use crate::delta::{DeltaError, Record, Schema, apply_delta, decode_delta};

/// A failure applying a delta to a [`SnapshotStore`].
#[derive(Clone, Debug, PartialEq)]
pub enum SnapshotError {
    /// The delta itself was malformed or encoded against another schema.
    Delta(DeltaError),
    /// The store does not retain the delta's baseline revision — resync needed.
    UnknownBaseline(u32),
    /// `baseline + 1` overflowed `u32` (the revision space is exhausted).
    RevisionOverflow(u32),
}

impl From<DeltaError> for SnapshotError {
    fn from(error: DeltaError) -> Self {
        Self::Delta(error)
    }
}

/// A bounded ring of baseline records, addressed by revision.
#[derive(Clone, Debug)]
pub struct SnapshotStore {
    schema: Schema,
    capacity: usize,
    /// Retained `(revision, record)` pairs, oldest at the front, ascending.
    frames: VecDeque<(u32, Record)>,
}

impl SnapshotStore {
    /// A store for `schema` retaining at most `capacity` baselines.
    ///
    /// `capacity` is clamped to at least one (a store that retains nothing could
    /// never apply a delta).
    #[must_use]
    pub fn new(schema: Schema, capacity: usize) -> Self {
        Self {
            schema,
            capacity: capacity.max(1),
            frames: VecDeque::new(),
        }
    }

    /// The schema every retained record matches.
    #[must_use]
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// The maximum number of baselines retained at once.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of retained baselines.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether no baselines are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The newest retained revision, if any.
    #[must_use]
    pub fn latest_revision(&self) -> Option<u32> {
        self.frames.back().map(|(revision, _)| *revision)
    }

    /// The oldest retained revision, if any (the next to be evicted).
    #[must_use]
    pub fn oldest_revision(&self) -> Option<u32> {
        self.frames.front().map(|(revision, _)| *revision)
    }

    /// Whether `revision` is still retained.
    #[must_use]
    pub fn contains(&self, revision: u32) -> bool {
        self.frames.iter().any(|(r, _)| *r == revision)
    }

    /// The baseline retained at `revision`, if any.
    #[must_use]
    pub fn record(&self, revision: u32) -> Option<&Record> {
        self.frames
            .iter()
            .find(|(r, _)| *r == revision)
            .map(|(_, record)| record)
    }

    /// Store a full baseline at `revision`.
    ///
    /// Re-storing an already-retained revision overwrites it in place (idempotent
    /// re-confirmation); a new revision must be newer than the current newest (the
    /// store is fed in order), and evicts the oldest past `capacity`.
    ///
    /// # Errors
    /// [`DeltaError::Shape`] / [`DeltaError::KindMismatch`] when `record` does not
    /// match the schema.
    pub fn insert_baseline(&mut self, revision: u32, record: Record) -> Result<(), DeltaError> {
        self.schema.validate(&record)?;
        if let Some(slot) = self.frames.iter_mut().find(|(r, _)| *r == revision) {
            slot.1 = record;
            return Ok(());
        }
        if self.frames.len() == self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back((revision, record));
        Ok(())
    }

    /// Apply a delta to the retained baseline it names, materialising the next
    /// baseline and returning its revision.
    ///
    /// The delta's declared revision is its **baseline**; the result is stored at
    /// `baseline + 1`.
    ///
    /// # Errors
    /// [`SnapshotError::Delta`] for a malformed delta, or
    /// [`SnapshotError::UnknownBaseline`] when the baseline has been evicted (the
    /// store is left untouched in both cases).
    pub fn apply(&mut self, bytes: &[u8]) -> Result<u32, SnapshotError> {
        let delta = decode_delta(&self.schema, bytes)?;
        let baseline = delta.revision();
        let Some(base) = self.record(baseline).cloned() else {
            return Err(SnapshotError::UnknownBaseline(baseline));
        };
        let next_revision = baseline
            .checked_add(1)
            .ok_or(SnapshotError::RevisionOverflow(baseline))?;

        // Apply onto a copy first, so a malformed delta cannot corrupt the store.
        let mut next = base;
        apply_delta(&self.schema, &mut next, baseline, bytes)?;
        self.insert_baseline(next_revision, next)?;
        Ok(next_revision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{FieldKind, FieldValue, encode_delta};
    use alloc::vec;
    use alloc::vec::Vec;

    fn schema() -> Schema {
        Schema::new(vec![
            FieldKind::Bool,
            FieldKind::Int { bits: 12 },
            FieldKind::Float {
                min: -10.0,
                max: 10.0,
                bits: 16,
            },
            FieldKind::Int { bits: 8 },
        ])
    }

    fn record(flag: bool, first: i64, angle: f32, second: i64) -> Record {
        Record::new(vec![
            FieldValue::Bool(flag),
            FieldValue::Int(first),
            FieldValue::Float(angle),
            FieldValue::Int(second),
        ])
    }

    #[test]
    fn a_baseline_plus_a_delta_chain_tracks_the_sender() {
        let schema = schema();
        let r0 = record(false, 1, 0.0, 1);
        let r1 = record(true, 1, 0.0, 2);
        let r2 = record(true, -5, 0.0, 2);

        let mut receiver = SnapshotStore::new(schema.clone(), 8);
        receiver.insert_baseline(0, r0.clone()).unwrap();

        let mut first = Vec::new();
        encode_delta(&schema, &r0, &r1, 0, &mut first).unwrap();
        assert_eq!(receiver.apply(&first).unwrap(), 1);
        assert_eq!(receiver.record(1), Some(&r1));
        assert_eq!(receiver.latest_revision(), Some(1));

        let mut second = Vec::new();
        encode_delta(&schema, &r1, &r2, 1, &mut second).unwrap();
        assert_eq!(receiver.apply(&second).unwrap(), 2);
        assert_eq!(receiver.record(2), Some(&r2));
    }

    #[test]
    fn a_delta_whose_baseline_was_evicted_is_refused() {
        let schema = schema();
        let r0 = record(false, 0, 0.0, 0);
        let r1 = record(true, 0, 0.0, 0);
        let mut receiver = SnapshotStore::new(schema.clone(), 1);
        receiver.insert_baseline(0, r0.clone()).unwrap();
        receiver.insert_baseline(1, r1.clone()).unwrap();
        assert_eq!(receiver.oldest_revision(), Some(1), "frame 0 was evicted");

        let mut stale = Vec::new();
        encode_delta(&schema, &r0, &r1, 0, &mut stale).unwrap();
        assert_eq!(
            receiver.apply(&stale),
            Err(SnapshotError::UnknownBaseline(0))
        );
    }

    #[test]
    fn the_ring_evicts_the_oldest_baseline() {
        let schema = schema();
        let mut store = SnapshotStore::new(schema, 2);
        for revision in 0..3 {
            store
                .insert_baseline(revision, record(false, revision as i64, 0.0, 0))
                .unwrap();
        }
        assert_eq!(store.len(), 2);
        assert_eq!(store.oldest_revision(), Some(1));
        assert!(!store.contains(0));
        assert!(store.contains(1) && store.contains(2));
    }

    #[test]
    fn re_storing_a_baseline_overwrites_in_place() {
        let schema = schema();
        let mut store = SnapshotStore::new(schema.clone(), 4);
        store.insert_baseline(0, record(false, 1, 0.0, 1)).unwrap();
        store.insert_baseline(1, record(false, 2, 0.0, 2)).unwrap();
        // Correct revision 0 in place: still two frames, oldest unchanged.
        store.insert_baseline(0, record(true, 9, 0.0, 9)).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.oldest_revision(), Some(0));
        assert_eq!(store.record(0), Some(&record(true, 9, 0.0, 9)));
    }

    #[test]
    fn a_mismatched_baseline_record_is_rejected() {
        let mut store = SnapshotStore::new(schema(), 4);
        let short = Record::new(vec![FieldValue::Bool(true)]);
        assert_eq!(
            store.insert_baseline(0, short),
            Err(DeltaError::Shape {
                expected: 4,
                found: 1
            })
        );
        assert!(store.is_empty());
    }
}
