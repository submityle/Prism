//! Deterministic aggregation of external CPU/GPU page feedback.
//!
//! This module accepts already-decoded feedback only. It has no GPU dependency
//! and does not pretend to perform feedback readback, decoding, or upload.

use super::{PageId, PageKey, PageRequestError, PageResidency};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cmp::Ordering;
use prism_tasks::Priority;

/// One decoded page demand reported by an external CPU or GPU producer.
///
/// Lower `desired_mip` means finer requested quality. `importance` is an
/// application-defined integer weight forwarded unchanged to
/// [`PageResidency::request`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeedbackDemand {
    /// Page identity, including its mounted-source revision.
    pub key: PageKey,
    /// Finest requested mip (`0` is finest).
    pub desired_mip: u8,
    /// Caller-defined importance forwarded to page residency.
    pub importance: u32,
    /// Frame that produced this feedback.
    pub frame: u64,
    /// Scheduling lane forwarded to page residency.
    pub priority: Priority,
}

impl FeedbackDemand {
    /// Creates one decoded feedback item.
    #[must_use]
    pub const fn new(
        key: PageKey,
        desired_mip: u8,
        importance: u32,
        frame: u64,
        priority: Priority,
    ) -> Self {
        Self {
            key,
            desired_mip,
            importance,
            frame,
            priority,
        }
    }

    fn merge(&mut self, incoming: Self) {
        self.desired_mip = self.desired_mip.min(incoming.desired_mip);
        self.importance = self.importance.max(incoming.importance);
        self.frame = self.frame.max(incoming.frame);
        self.priority = self.priority.max(incoming.priority);
    }
}

/// Result of inserting one feedback item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedbackPush {
    /// A new key was inserted without displacing another key.
    Inserted,
    /// An existing key absorbed the incoming feedback.
    Merged,
    /// The queue was full and the incoming key ranked below the retained set.
    DroppedOverflow,
    /// The queue was full and the incoming key displaced its worst item.
    ReplacedOverflow,
}

/// Saturating lifetime counters for queue behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FeedbackDemandStats {
    /// Total calls to [`FeedbackDemandQueue::push`].
    pub submitted: u64,
    /// Inputs merged into an already queued key.
    pub merged: u64,
    /// Unique inputs considered while the queue was full.
    pub overflowed: u64,
    /// Stale queued demands removed by expiration.
    pub expired: u64,
    /// Demands accepted by [`PageResidency::request`].
    pub admitted: u64,
    /// Demands rejected by [`PageResidency::request`].
    pub request_errors: u64,
}

/// The result of forwarding one queued demand to [`PageResidency`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeedbackAdmission {
    /// The removed queue item that was forwarded.
    pub demand: FeedbackDemand,
    /// The residency request result for that item.
    pub result: Result<PageId, PageRequestError>,
}

/// A bounded, deterministic pre-admission queue for decoded page feedback.
///
/// Duplicate [`PageKey`] values are merged monotonically. Distinct source
/// revisions remain distinct because revision is part of `PageKey`. When full,
/// the queue retains the globally best `capacity` demands, independent of
/// insertion order.
pub struct FeedbackDemandQueue {
    capacity: usize,
    max_age_frames: u64,
    demands: BTreeMap<PageKey, FeedbackDemand>,
    stats: FeedbackDemandStats,
}

impl FeedbackDemandQueue {
    /// Creates a queue. A zero capacity is valid and drops every unique item.
    #[must_use]
    pub fn new(capacity: usize, max_age_frames: u64) -> Self {
        Self {
            capacity,
            max_age_frames,
            demands: BTreeMap::new(),
            stats: FeedbackDemandStats::default(),
        }
    }

    /// Returns the maximum number of distinct queued keys.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of distinct queued keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.demands.len()
    }

    /// Returns whether no demand is queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.demands.is_empty()
    }

    /// Returns saturating lifetime counters.
    #[must_use]
    pub const fn stats(&self) -> FeedbackDemandStats {
        self.stats
    }

    /// Inserts or merges one demand while preserving the bounded top-K set.
    pub fn push(&mut self, demand: FeedbackDemand) -> FeedbackPush {
        self.stats.submitted = self.stats.submitted.saturating_add(1);
        if let Some(current) = self.demands.get_mut(&demand.key) {
            current.merge(demand);
            self.stats.merged = self.stats.merged.saturating_add(1);
            return FeedbackPush::Merged;
        }

        if self.demands.len() < self.capacity {
            self.demands.insert(demand.key.clone(), demand);
            return FeedbackPush::Inserted;
        }

        self.stats.overflowed = self.stats.overflowed.saturating_add(1);
        let worst_key = self
            .demands
            .values()
            .max_by(|left, right| compare_demands(left, right))
            .map(|worst| worst.key.clone());
        let Some(worst_key) = worst_key else {
            return FeedbackPush::DroppedOverflow;
        };
        let replace = self
            .demands
            .get(&worst_key)
            .is_some_and(|worst| compare_demands(&demand, worst) == Ordering::Less);
        if replace {
            self.demands.remove(&worst_key);
            self.demands.insert(demand.key.clone(), demand);
            FeedbackPush::ReplacedOverflow
        } else {
            FeedbackPush::DroppedOverflow
        }
    }

    /// Inserts a batch in stable key order and returns one outcome per input.
    ///
    /// Sorting first makes duplicate merging and bounded top-K retention
    /// independent of the producer's iterator order.
    pub fn push_batch(
        &mut self,
        demands: impl IntoIterator<Item = FeedbackDemand>,
    ) -> Vec<FeedbackPush> {
        let mut demands = demands.into_iter().collect::<Vec<_>>();
        demands.sort_by(|left, right| left.key.cmp(&right.key));
        demands
            .into_iter()
            .map(|demand| self.push(demand))
            .collect()
    }

    /// Removes feedback older than `max_age_frames` relative to `current_frame`.
    ///
    /// Future-dated feedback is retained without subtraction overflow.
    pub fn expire(&mut self, current_frame: u64) -> usize {
        let oldest_retained = current_frame.saturating_sub(self.max_age_frames);
        let before = self.demands.len();
        self.demands
            .retain(|_, demand| demand.frame >= oldest_retained);
        let expired = before.saturating_sub(self.demands.len());
        self.stats.expired = self
            .stats
            .expired
            .saturating_add(u64::try_from(expired).unwrap_or(u64::MAX));
        expired
    }

    /// Returns queued demands in stable dispatch order without mutating them.
    #[must_use]
    pub fn ordered(&self) -> Vec<&FeedbackDemand> {
        let mut ordered = self.demands.values().collect::<Vec<_>>();
        ordered.sort_by(|left, right| compare_demands(left, right));
        ordered
    }

    /// Expires stale feedback, removes at most `limit` best demands, and
    /// forwards them to [`PageResidency::request`].
    ///
    /// A rejected page range is reported and removed so one malformed producer
    /// cannot poison later batches. The returned vector is bounded by `limit`.
    pub fn admit_batch(
        &mut self,
        residency: &mut PageResidency,
        current_frame: u64,
        limit: usize,
    ) -> Vec<FeedbackAdmission> {
        self.expire(current_frame);
        let keys = self
            .ordered()
            .into_iter()
            .take(limit)
            .map(|demand| demand.key.clone())
            .collect::<Vec<_>>();
        let mut admitted = Vec::with_capacity(keys.len());
        for key in keys {
            let Some(demand) = self.demands.remove(&key) else {
                continue;
            };
            let result = residency.request(demand.key.clone(), demand.priority, demand.importance);
            if result.is_ok() {
                self.stats.admitted = self.stats.admitted.saturating_add(1);
            } else {
                self.stats.request_errors = self.stats.request_errors.saturating_add(1);
            }
            admitted.push(FeedbackAdmission { demand, result });
        }
        admitted
    }
}

fn compare_demands(left: &FeedbackDemand, right: &FeedbackDemand) -> Ordering {
    left.desired_mip
        .cmp(&right.desired_mip)
        .then_with(|| right.priority.cmp(&left.priority))
        .then_with(|| right.importance.cmp(&left.importance))
        .then_with(|| right.frame.cmp(&left.frame))
        .then_with(|| left.key.cmp(&right.key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PageBudget, PageSize};

    fn demand(
        path: &str,
        page: u64,
        revision: u64,
        mip: u8,
        importance: u32,
        frame: u64,
        priority: Priority,
    ) -> FeedbackDemand {
        FeedbackDemand::new(
            PageKey::new(path, page, revision),
            mip,
            importance,
            frame,
            priority,
        )
    }

    fn residency() -> PageResidency {
        PageResidency::new(PageSize::new(4).unwrap(), PageBudget::new(64, 16).unwrap()).unwrap()
    }

    #[test]
    fn stable_sort_prefers_quality_priority_importance_and_freshness() {
        let mut queue = FeedbackDemandQueue::new(8, 10);
        queue.push(demand("d.bin", 0, 0, 1, 9, 9, Priority::Normal));
        queue.push(demand("c.bin", 0, 0, 0, 1, 7, Priority::Low));
        queue.push(demand("b.bin", 0, 0, 0, 1, 8, Priority::High));
        queue.push(demand("a.bin", 0, 0, 0, 2, 8, Priority::High));
        let paths = queue
            .ordered()
            .into_iter()
            .map(|item| item.key.path().path())
            .collect::<Vec<_>>();
        assert_eq!(paths, vec!["a.bin", "b.bin", "c.bin", "d.bin"]);
    }

    #[test]
    fn duplicate_feedback_merges_monotonically() {
        let mut queue = FeedbackDemandQueue::new(2, 5);
        assert_eq!(
            queue.push(demand("a.bin", 3, 4, 5, 2, 7, Priority::Low)),
            FeedbackPush::Inserted
        );
        assert_eq!(
            queue.push(demand("a.bin", 3, 4, 2, 8, 9, Priority::High)),
            FeedbackPush::Merged
        );
        let merged = queue.ordered()[0];
        assert_eq!(merged.desired_mip, 2);
        assert_eq!(merged.importance, 8);
        assert_eq!(merged.frame, 9);
        assert_eq!(merged.priority, Priority::High);
        assert_eq!(queue.stats().merged, 1);
    }

    #[test]
    fn capacity_keeps_best_demands_independent_of_input_order() {
        let values = [
            demand("low.bin", 0, 0, 4, 0, 0, Priority::Low),
            demand("best.bin", 0, 0, 0, 0, 0, Priority::Critical),
            demand("mid.bin", 0, 0, 2, 0, 0, Priority::Normal),
        ];
        let mut forward = FeedbackDemandQueue::new(2, u64::MAX);
        forward.push_batch(values.clone());
        let mut reverse = FeedbackDemandQueue::new(2, u64::MAX);
        reverse.push_batch(values.into_iter().rev());
        let keys = |queue: &FeedbackDemandQueue| {
            queue
                .ordered()
                .into_iter()
                .map(|item| item.key.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(keys(&forward), keys(&reverse));
        assert_eq!(forward.len(), 2);
        assert_eq!(forward.stats().overflowed, 1);
    }

    #[test]
    fn batch_duplicate_merge_is_independent_of_input_order() {
        let values = [
            demand("a.bin", 0, 0, 5, 9, 7, Priority::Low),
            demand("b.bin", 0, 0, 2, 3, 8, Priority::Normal),
            demand("a.bin", 0, 0, 1, 2, 9, Priority::High),
        ];
        let mut forward = FeedbackDemandQueue::new(1, u64::MAX);
        forward.push_batch(values.clone());
        let mut reverse = FeedbackDemandQueue::new(1, u64::MAX);
        reverse.push_batch(values.into_iter().rev());
        assert_eq!(forward.ordered(), reverse.ordered());
        let retained = forward.ordered()[0];
        assert_eq!(retained.key.path().path(), "a.bin");
        assert_eq!(retained.desired_mip, 1);
        assert_eq!(retained.importance, 9);
        assert_eq!(retained.frame, 9);
        assert_eq!(retained.priority, Priority::High);
    }

    #[test]
    fn zero_capacity_and_saturated_counters_never_overflow() {
        let mut queue = FeedbackDemandQueue::new(0, 0);
        queue.stats.submitted = u64::MAX;
        queue.stats.overflowed = u64::MAX;
        assert_eq!(
            queue.push(demand("drop.bin", 0, 0, 0, 0, 0, Priority::Low)),
            FeedbackPush::DroppedOverflow
        );
        assert_eq!(queue.stats().submitted, u64::MAX);
        assert_eq!(queue.stats().overflowed, u64::MAX);
        assert!(queue.is_empty());
    }

    #[test]
    fn expiration_handles_frame_extremes_without_overflow() {
        let mut queue = FeedbackDemandQueue::new(4, 2);
        queue.push(demand("old.bin", 0, 0, 0, 0, 7, Priority::Low));
        queue.push(demand("edge.bin", 0, 0, 0, 0, 8, Priority::Low));
        queue.push(demand("future.bin", 0, 0, 0, 0, u64::MAX, Priority::Low));
        assert_eq!(queue.expire(10), 1);
        assert_eq!(queue.expire(0), 0);
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn source_revisions_are_isolated() {
        let mut queue = FeedbackDemandQueue::new(4, 5);
        queue.push(demand("a.bin", 0, 1, 0, 1, 1, Priority::Normal));
        queue.push(demand("a.bin", 0, 2, 0, 1, 1, Priority::Normal));
        assert_eq!(queue.len(), 2);
        assert_eq!(
            queue
                .ordered()
                .into_iter()
                .map(|item| item.key.source_revision())
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn batch_admission_is_bounded_sorted_and_merges_with_residency() {
        let mut queue = FeedbackDemandQueue::new(4, 10);
        queue.push(demand("late.bin", 0, 0, 3, 1, 3, Priority::Low));
        queue.push(demand("first.bin", 0, 0, 0, 5, 3, Priority::High));
        queue.push(demand("second.bin", 0, 0, 1, 2, 3, Priority::Normal));
        let mut residency = residency();
        let existing = residency
            .request(PageKey::new("first.bin", 0, 0), Priority::Low, 1)
            .unwrap();

        let admitted = queue.admit_batch(&mut residency, 3, 2);
        assert_eq!(admitted.len(), 2);
        assert_eq!(admitted[0].demand.key.path().path(), "first.bin");
        assert_eq!(admitted[0].result, Ok(existing));
        assert_eq!(admitted[1].demand.key.path().path(), "second.bin");
        assert_eq!(residency.priority(existing), Some(Priority::High));
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.stats().admitted, 2);
    }

    #[test]
    fn invalid_page_range_is_reported_without_panicking() {
        let mut queue = FeedbackDemandQueue::new(1, u64::MAX);
        queue.push(demand(
            "overflow.bin",
            u64::MAX,
            0,
            0,
            u32::MAX,
            u64::MAX,
            Priority::Critical,
        ));
        let mut residency = residency();
        let admitted = queue.admit_batch(&mut residency, u64::MAX, usize::MAX);
        assert_eq!(admitted.len(), 1);
        assert!(matches!(
            admitted[0].result,
            Err(PageRequestError::Range(_))
        ));
        assert_eq!(queue.stats().request_errors, 1);
        assert!(queue.is_empty());
    }
}
