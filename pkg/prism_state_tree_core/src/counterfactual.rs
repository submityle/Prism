//! Bounded counterfactual branches over deterministic [`TraceRecord`] values.
//!
//! A counterfactual branch is an inspection-time overlay. It keeps the
//! captured trace records and stores only indexed changes to an active-state
//! digest or a host-provided decision label. It does not execute a tree, call
//! guards, or mutate [`crate::replay::TraceLog`]. A debugger can therefore
//! replay a branch with its own execution adapter while this module remains a
//! small, deterministic diff contract in the `no_std + alloc` core.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

use alloc::vec::Vec;

use crate::replay::{ActiveStateDigest, TraceInput, TraceRecord};

/// Maximum number of records accepted by one counterfactual trace.
pub const MAX_COUNTERFACTUAL_RECORDS: usize = 4096;

/// Maximum number of sparse record overrides retained by one branch.
///
/// This is intentionally smaller than the trace-record bound: a useful
/// counterfactual normally changes a small number of decisions, and the
/// separate cap prevents a branch from becoming a second full trace-sized
/// patch table.
pub const MAX_COUNTERFACTUAL_OVERRIDES: usize = 1024;

/// An opaque, deterministic label for the decision observed at one trace
/// record.
///
/// The core does not assign meaning to the value. Hosts may encode a
/// transition id, a planner choice, or a compact decision-table row, while
/// keeping domain-specific interpretation outside the replay kernel.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DecisionTag(u64);

impl DecisionTag {
    /// Creates a decision tag from a host-defined stable value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the host-defined stable value.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// The requested edit to a decision tag in a counterfactual record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionTagEdit {
    /// Preserve the current tag and make no tag change.
    Keep,
    /// Replace the current tag with the supplied tag.
    Set(DecisionTag),
    /// Remove the current tag from the counterfactual record.
    Clear,
}

/// Sparse changes applied to one record in a counterfactual branch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CounterfactualOverride {
    summary: Option<ActiveStateDigest>,
    decision: DecisionTagEdit,
}

impl CounterfactualOverride {
    /// Creates an empty override.
    ///
    /// An empty override is useful when a caller builds an edit
    /// conditionally. Applying it leaves the branch unchanged.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            summary: None,
            decision: DecisionTagEdit::Keep,
        }
    }

    /// Adds an active-state summary replacement to this override.
    #[must_use]
    pub const fn with_summary(mut self, summary: ActiveStateDigest) -> Self {
        self.summary = Some(summary);
        self
    }

    /// Adds a decision-tag replacement to this override.
    #[must_use]
    pub const fn with_decision_tag(mut self, tag: DecisionTag) -> Self {
        self.decision = DecisionTagEdit::Set(tag);
        self
    }

    /// Adds a decision-tag removal to this override.
    #[must_use]
    pub const fn without_decision_tag(mut self) -> Self {
        self.decision = DecisionTagEdit::Clear;
        self
    }

    /// Returns the optional active-state summary replacement.
    #[must_use]
    pub const fn summary(self) -> Option<ActiveStateDigest> {
        self.summary
    }

    /// Returns the requested decision-tag edit.
    #[must_use]
    pub const fn decision_tag(self) -> DecisionTagEdit {
        self.decision
    }

    fn is_empty(self) -> bool {
        self.summary.is_none() && matches!(self.decision, DecisionTagEdit::Keep)
    }

    fn merge(self, incoming: Self) -> Self {
        Self {
            summary: incoming.summary.or(self.summary),
            decision: if matches!(incoming.decision, DecisionTagEdit::Keep) {
                self.decision
            } else {
                incoming.decision
            },
        }
    }
}

impl Default for CounterfactualOverride {
    fn default() -> Self {
        Self::new()
    }
}

/// A trace record plus the optional decision label associated with it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CounterfactualRecord {
    trace: TraceRecord,
    decision_tag: Option<DecisionTag>,
}

impl CounterfactualRecord {
    /// Creates a counterfactual record from a replay record and a baseline
    /// decision label.
    #[must_use]
    pub const fn new(trace: TraceRecord, decision_tag: Option<DecisionTag>) -> Self {
        Self {
            trace,
            decision_tag,
        }
    }

    /// Returns the original replay record.
    #[must_use]
    pub const fn trace_record(self) -> TraceRecord {
        self.trace
    }

    /// Returns the input operation represented by the record.
    #[must_use]
    pub const fn input(self) -> TraceInput {
        self.trace.input
    }

    /// Returns the active-state summary represented by the record.
    #[must_use]
    pub const fn summary(self) -> ActiveStateDigest {
        self.trace.digest
    }

    /// Returns the decision label represented by the record, if one exists.
    #[must_use]
    pub const fn decision_tag(self) -> Option<DecisionTag> {
        self.decision_tag
    }
}

/// A structured failure from constructing or editing a counterfactual
/// branch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CounterfactualError {
    /// The requested branch capacity exceeds the hard contract limit.
    CapacityLimit {
        /// Requested record capacity.
        requested: usize,
        /// Maximum supported record capacity.
        maximum: usize,
    },
    /// A record cannot be appended because the branch is full.
    CapacityExceeded {
        /// Configured branch record capacity.
        capacity: usize,
    },
    /// An operation referenced a record outside the branch.
    RecordIndexOutOfRange {
        /// Referenced zero-based record index.
        index: usize,
        /// Number of records currently in the branch.
        len: usize,
    },
    /// A new sparse override would exceed the hard override limit.
    OverrideCapacityExceeded {
        /// Maximum number of sparse overrides supported by the branch.
        capacity: usize,
    },
}

/// The first difference between a baseline record and its counterfactual
/// effective record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CounterfactualDiff {
    record_index: usize,
    baseline: CounterfactualRecord,
    counterfactual: CounterfactualRecord,
}

impl CounterfactualDiff {
    /// Returns the zero-based record index at which the branches first differ.
    #[must_use]
    pub const fn record_index(self) -> usize {
        self.record_index
    }

    /// Returns the unmodified baseline record.
    #[must_use]
    pub const fn baseline(self) -> CounterfactualRecord {
        self.baseline
    }

    /// Returns the effective counterfactual record.
    #[must_use]
    pub const fn counterfactual(self) -> CounterfactualRecord {
        self.counterfactual
    }

    /// Returns whether the active-state summary changed at the first
    /// divergence.
    #[must_use]
    pub fn summary_changed(self) -> bool {
        self.baseline.summary() != self.counterfactual.summary()
    }

    /// Returns whether the decision label changed at the first divergence.
    #[must_use]
    pub fn decision_tag_changed(self) -> bool {
        self.baseline.decision_tag() != self.counterfactual.decision_tag()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct IndexedOverride {
    index: usize,
    value: CounterfactualOverride,
}

/// A bounded baseline trace with sparse counterfactual record overrides.
///
/// The baseline is copied when the branch is constructed, so later changes to
/// the caller's source slice cannot alter a branch. Overrides are kept sorted
/// by record index. Lookup is logarithmic and iteration is allocation-free;
/// inserting a new index shifts at most the bounded override tail.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CounterfactualTrace {
    capacity: usize,
    records: Vec<CounterfactualRecord>,
    overrides: Vec<IndexedOverride>,
}

impl CounterfactualTrace {
    /// Creates an empty branch with a fixed maximum record count.
    pub fn new(capacity: usize) -> Result<Self, CounterfactualError> {
        if capacity > MAX_COUNTERFACTUAL_RECORDS {
            return Err(CounterfactualError::CapacityLimit {
                requested: capacity,
                maximum: MAX_COUNTERFACTUAL_RECORDS,
            });
        }
        Ok(Self {
            capacity,
            records: Vec::with_capacity(capacity),
            overrides: Vec::with_capacity(capacity.min(MAX_COUNTERFACTUAL_OVERRIDES)),
        })
    }

    /// Copies replay records into a bounded branch with no decision labels.
    pub fn from_trace_records(
        records: &[TraceRecord],
        capacity: usize,
    ) -> Result<Self, CounterfactualError> {
        let mut branch = Self::new(capacity)?;
        if records.len() > capacity {
            return Err(CounterfactualError::CapacityExceeded { capacity });
        }
        branch.records.extend(
            records
                .iter()
                .copied()
                .map(|trace| CounterfactualRecord::new(trace, None)),
        );
        Ok(branch)
    }

    /// Alias for [`Self::from_trace_records`].
    pub fn from_records(
        records: &[TraceRecord],
        capacity: usize,
    ) -> Result<Self, CounterfactualError> {
        Self::from_trace_records(records, capacity)
    }

    /// Creates an independent branch with the same baseline records and
    /// capacity, discarding all overrides from the source branch.
    #[must_use]
    pub fn fork(&self) -> Self {
        Self {
            capacity: self.capacity,
            records: self.records.clone(),
            overrides: Vec::with_capacity(self.capacity.min(MAX_COUNTERFACTUAL_OVERRIDES)),
        }
    }

    /// Returns the configured maximum number of baseline records.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of baseline records in the branch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns whether the branch contains no baseline records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Returns whether the branch has reached its baseline record capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.records.len() == self.capacity
    }

    /// Returns the number of indexed overrides currently stored.
    #[must_use]
    pub fn override_count(&self) -> usize {
        self.overrides.len()
    }

    /// Returns baseline records, including their baseline decision labels.
    #[must_use]
    pub fn records(&self) -> &[CounterfactualRecord] {
        &self.records
    }

    /// Appends a replay record without a decision label.
    pub fn push(&mut self, trace: TraceRecord) -> Result<(), CounterfactualError> {
        self.push_with_decision_tag(trace, None)
    }

    /// Appends a replay record with an optional baseline decision label.
    pub fn push_with_decision_tag(
        &mut self,
        trace: TraceRecord,
        decision_tag: Option<DecisionTag>,
    ) -> Result<(), CounterfactualError> {
        if self.records.len() == self.capacity {
            return Err(CounterfactualError::CapacityExceeded {
                capacity: self.capacity,
            });
        }
        self.records
            .push(CounterfactualRecord::new(trace, decision_tag));
        Ok(())
    }

    /// Replaces a baseline decision label without changing any branch
    /// override.
    pub fn set_baseline_decision_tag(
        &mut self,
        index: usize,
        decision_tag: Option<DecisionTag>,
    ) -> Result<(), CounterfactualError> {
        let record = self.record_mut(index)?;
        record.decision_tag = decision_tag;
        Ok(())
    }

    /// Returns a baseline record by index.
    pub fn record(&self, index: usize) -> Result<CounterfactualRecord, CounterfactualError> {
        self.records
            .get(index)
            .copied()
            .ok_or(CounterfactualError::RecordIndexOutOfRange {
                index,
                len: self.records.len(),
            })
    }

    /// Returns the effective record after applying its indexed override.
    pub fn effective_record(
        &self,
        index: usize,
    ) -> Result<CounterfactualRecord, CounterfactualError> {
        let baseline = self.record(index)?;
        Ok(self.apply_override(index, baseline))
    }

    /// Returns an allocation-free iterator over effective branch records.
    #[must_use]
    pub fn iter(&self) -> EffectiveRecordIter<'_> {
        EffectiveRecordIter {
            branch: self,
            next: 0,
        }
    }

    /// Applies or merges a sparse override at a record index.
    pub fn override_record(
        &mut self,
        index: usize,
        change: CounterfactualOverride,
    ) -> Result<(), CounterfactualError> {
        self.ensure_index(index)?;
        if change.is_empty() {
            return Ok(());
        }

        let position = self.find_override(index);
        match position {
            Ok(position) => {
                let merged = self.overrides[position].value.merge(change);
                self.overrides[position].value = merged;
            }
            Err(position) => {
                if self.overrides.len() == MAX_COUNTERFACTUAL_OVERRIDES {
                    return Err(CounterfactualError::OverrideCapacityExceeded {
                        capacity: MAX_COUNTERFACTUAL_OVERRIDES,
                    });
                }
                self.overrides.insert(
                    position,
                    IndexedOverride {
                        index,
                        value: change,
                    },
                );
            }
        }
        Ok(())
    }

    /// Replaces the active-state summary at a record index in the branch.
    pub fn override_summary(
        &mut self,
        index: usize,
        summary: ActiveStateDigest,
    ) -> Result<(), CounterfactualError> {
        self.override_record(index, CounterfactualOverride::new().with_summary(summary))
    }

    /// Replaces the decision label at a record index in the branch.
    pub fn override_decision_tag(
        &mut self,
        index: usize,
        decision_tag: DecisionTag,
    ) -> Result<(), CounterfactualError> {
        self.override_record(
            index,
            CounterfactualOverride::new().with_decision_tag(decision_tag),
        )
    }

    /// Removes the decision label at a record index in the branch.
    pub fn clear_decision_tag(&mut self, index: usize) -> Result<(), CounterfactualError> {
        self.override_record(index, CounterfactualOverride::new().without_decision_tag())
    }

    /// Removes only the summary replacement at a record index.
    pub fn clear_summary_override(&mut self, index: usize) -> Result<(), CounterfactualError> {
        self.ensure_index(index)?;
        if let Ok(position) = self.find_override(index) {
            let mut change = self.overrides[position].value;
            change.summary = None;
            if change.is_empty() {
                self.overrides.remove(position);
            } else {
                self.overrides[position].value = change;
            }
        }
        Ok(())
    }

    /// Returns the sparse override at an index, if one exists.
    pub fn override_at(
        &self,
        index: usize,
    ) -> Result<Option<CounterfactualOverride>, CounterfactualError> {
        self.ensure_index(index)?;
        Ok(self
            .find_override(index)
            .ok()
            .map(|position| self.overrides[position].value))
    }

    /// Removes every change at an index and reports whether one existed.
    pub fn clear_override(&mut self, index: usize) -> Result<bool, CounterfactualError> {
        self.ensure_index(index)?;
        match self.find_override(index) {
            Ok(position) => {
                self.overrides.remove(position);
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }

    /// Returns the first index whose effective record differs from baseline.
    #[must_use]
    pub fn first_divergence(&self) -> Option<CounterfactualDiff> {
        for index in 0..self.records.len() {
            let baseline = self.records[index];
            let counterfactual = self.apply_override(index, baseline);
            if baseline != counterfactual {
                return Some(CounterfactualDiff {
                    record_index: index,
                    baseline,
                    counterfactual,
                });
            }
        }
        None
    }

    /// Alias for [`Self::first_divergence`].
    #[must_use]
    pub fn diff(&self) -> Option<CounterfactualDiff> {
        self.first_divergence()
    }

    fn ensure_index(&self, index: usize) -> Result<(), CounterfactualError> {
        if index < self.records.len() {
            Ok(())
        } else {
            Err(CounterfactualError::RecordIndexOutOfRange {
                index,
                len: self.records.len(),
            })
        }
    }

    fn record_mut(
        &mut self,
        index: usize,
    ) -> Result<&mut CounterfactualRecord, CounterfactualError> {
        let len = self.records.len();
        self.records
            .get_mut(index)
            .ok_or(CounterfactualError::RecordIndexOutOfRange { index, len })
    }

    fn find_override(&self, index: usize) -> Result<usize, usize> {
        self.overrides
            .binary_search_by_key(&index, |entry| entry.index)
    }

    fn apply_override(
        &self,
        index: usize,
        mut baseline: CounterfactualRecord,
    ) -> CounterfactualRecord {
        let Ok(position) = self.find_override(index) else {
            return baseline;
        };
        let change = self.overrides[position].value;
        if let Some(summary) = change.summary {
            baseline.trace.digest = summary;
        }
        match change.decision {
            DecisionTagEdit::Keep => {}
            DecisionTagEdit::Set(tag) => baseline.decision_tag = Some(tag),
            DecisionTagEdit::Clear => baseline.decision_tag = None,
        }
        baseline
    }
}

/// Allocation-free iterator over effective records in record-index order.
pub struct EffectiveRecordIter<'a> {
    branch: &'a CounterfactualTrace,
    next: usize,
}

impl Iterator for EffectiveRecordIter<'_> {
    type Item = CounterfactualRecord;

    fn next(&mut self) -> Option<Self::Item> {
        let record = self.branch.records.get(self.next).copied()?;
        self.next += 1;
        Some(self.branch.apply_override(self.next - 1, record))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.branch.records.len().saturating_sub(self.next);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for EffectiveRecordIter<'_> {}

impl core::iter::FusedIterator for EffectiveRecordIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Event, EventId, StateId};
    use alloc::vec;

    fn trace_record(index: u32) -> TraceRecord {
        TraceRecord {
            input: TraceInput::Event(Event::new(EventId(index))),
            digest: ActiveStateDigest::from_states(&[StateId(index)]),
        }
    }

    fn summary(index: u32) -> ActiveStateDigest {
        ActiveStateDigest::from_states(&[StateId(index), StateId(index + 1)])
    }

    #[test]
    fn capacity_is_bounded_and_structured() {
        assert_eq!(
            CounterfactualTrace::new(MAX_COUNTERFACTUAL_RECORDS + 1),
            Err(CounterfactualError::CapacityLimit {
                requested: MAX_COUNTERFACTUAL_RECORDS + 1,
                maximum: MAX_COUNTERFACTUAL_RECORDS,
            })
        );

        let mut branch = CounterfactualTrace::new(1).unwrap();
        branch.push(trace_record(0)).unwrap();
        assert_eq!(
            branch.push(trace_record(1)),
            Err(CounterfactualError::CapacityExceeded { capacity: 1 })
        );
    }

    #[test]
    fn source_records_are_copied_and_forks_are_independent() {
        let source = vec![trace_record(0), trace_record(1)];
        let baseline = CounterfactualTrace::from_trace_records(&source, 2).unwrap();
        let mut branch = baseline.fork();
        branch.override_summary(1, summary(9)).unwrap();

        assert_eq!(source[1].digest, baseline.record(1).unwrap().summary());
        assert_eq!(baseline.first_divergence(), None);
        assert_eq!(branch.first_divergence().unwrap().record_index(), 1);
    }

    #[test]
    fn summary_and_decision_overrides_merge_at_one_index() {
        let mut branch = CounterfactualTrace::new(2).unwrap();
        branch
            .push_with_decision_tag(trace_record(0), Some(DecisionTag::new(7)))
            .unwrap();

        branch.override_summary(0, summary(4)).unwrap();
        branch
            .override_decision_tag(0, DecisionTag::new(8))
            .unwrap();

        let change = branch.override_at(0).unwrap().unwrap();
        assert_eq!(change.summary(), Some(summary(4)));
        assert_eq!(
            change.decision_tag(),
            DecisionTagEdit::Set(DecisionTag::new(8))
        );
        let effective = branch.effective_record(0).unwrap();
        assert_eq!(effective.summary(), summary(4));
        assert_eq!(effective.decision_tag(), Some(DecisionTag::new(8)));
    }

    #[test]
    fn first_divergence_reports_the_earliest_changed_field() {
        let mut branch = CounterfactualTrace::new(3).unwrap();
        branch.push(trace_record(0)).unwrap();
        branch
            .push_with_decision_tag(trace_record(1), Some(DecisionTag::new(1)))
            .unwrap();
        branch.push(trace_record(2)).unwrap();

        branch.override_summary(2, summary(12)).unwrap();
        branch
            .override_decision_tag(1, DecisionTag::new(11))
            .unwrap();

        let diff = branch.first_divergence().unwrap();
        assert_eq!(diff.record_index(), 1);
        assert!(!diff.summary_changed());
        assert!(diff.decision_tag_changed());
        assert_eq!(diff.baseline().decision_tag(), Some(DecisionTag::new(1)));
        assert_eq!(
            diff.counterfactual().decision_tag(),
            Some(DecisionTag::new(11))
        );
    }

    #[test]
    fn clearing_overrides_restores_no_diff() {
        let mut branch = CounterfactualTrace::new(1).unwrap();
        branch
            .push_with_decision_tag(trace_record(0), Some(DecisionTag::new(3)))
            .unwrap();
        branch.override_summary(0, summary(5)).unwrap();
        branch.clear_decision_tag(0).unwrap();
        assert!(branch.first_divergence().is_some());

        branch.clear_override(0).unwrap();
        assert_eq!(branch.first_divergence(), None);
        assert_eq!(branch.override_count(), 0);
    }

    #[test]
    fn invalid_indices_are_reported_without_mutation() {
        let mut branch = CounterfactualTrace::new(1).unwrap();
        let error = branch.override_summary(4, summary(4)).unwrap_err();
        assert_eq!(
            error,
            CounterfactualError::RecordIndexOutOfRange { index: 4, len: 0 }
        );
        assert_eq!(branch.override_count(), 0);
        assert_eq!(branch.clear_override(4), Err(error));
    }

    #[test]
    fn sparse_override_capacity_is_reported() {
        let record_count = MAX_COUNTERFACTUAL_OVERRIDES + 1;
        let mut branch = CounterfactualTrace::new(record_count).unwrap();
        for index in 0..record_count {
            branch.push(trace_record(index as u32)).unwrap();
        }
        for index in 0..MAX_COUNTERFACTUAL_OVERRIDES {
            branch
                .override_summary(index, summary(index as u32))
                .unwrap();
        }

        assert_eq!(
            branch.override_summary(
                MAX_COUNTERFACTUAL_OVERRIDES,
                summary(MAX_COUNTERFACTUAL_OVERRIDES as u32),
            ),
            Err(CounterfactualError::OverrideCapacityExceeded {
                capacity: MAX_COUNTERFACTUAL_OVERRIDES,
            })
        );
        assert_eq!(branch.override_count(), MAX_COUNTERFACTUAL_OVERRIDES);
    }

    #[test]
    fn effective_iterator_is_bounded_and_allocation_free() {
        let mut branch = CounterfactualTrace::new(2).unwrap();
        branch.push(trace_record(0)).unwrap();
        branch.push(trace_record(1)).unwrap();
        branch.override_summary(0, summary(9)).unwrap();

        let records: Vec<CounterfactualRecord> = branch.iter().collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].summary(), summary(9));
        assert_eq!(records[1].summary(), branch.record(1).unwrap().summary());
        assert_eq!(branch.iter().len(), 2);
    }
}
