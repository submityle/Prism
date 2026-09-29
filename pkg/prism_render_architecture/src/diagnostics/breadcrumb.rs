//! Fixed-capacity, frame-stamped breadcrumb ring buffer.
//!
//! When a subsystem misbehaves, the most useful post-mortem artifact is a short
//! ordered trail of what it was doing just before: "began page mark",
//! "history rejected", "fell back to full resolve", and so on. This module
//! provides that trail as a fixed-capacity ring buffer so it has a bounded
//! memory cost and never needs to allocate on the hot path once constructed.
//!
//! The ring keeps the most recent `capacity` [`Breadcrumb`]s in insertion
//! order. Pushing into a full ring overwrites the oldest entry. Iteration and
//! the "most recent N" query always return entries oldest-first, and the
//! behavior is fully deterministic for a given push sequence.

use alloc::string::String;
use alloc::vec::Vec;

/// Severity of a breadcrumb, ordered least to most severe.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BreadcrumbSeverity {
    /// Routine progress marker.
    Info,
    /// Something unexpected but non-fatal (a fallback engaged, a budget hit).
    Warning,
    /// A fault; typically paired with a quarantine of the owning feature.
    Error,
}

/// A single frame-stamped diagnostic event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Breadcrumb {
    /// Frame index the event was recorded on.
    pub frame_index: u64,
    /// Severity bucket.
    pub severity: BreadcrumbSeverity,
    /// Stable short label for the event (for example `"history_rejected"`).
    pub label: &'static str,
    /// Optional free-form detail.
    pub detail: Option<String>,
}

impl Breadcrumb {
    /// Creates a breadcrumb with an explicit severity and no detail.
    #[must_use]
    pub const fn new(frame_index: u64, severity: BreadcrumbSeverity, label: &'static str) -> Self {
        Self {
            frame_index,
            severity,
            label,
            detail: None,
        }
    }

    /// Creates an [`BreadcrumbSeverity::Info`] breadcrumb.
    #[must_use]
    pub const fn info(frame_index: u64, label: &'static str) -> Self {
        Self::new(frame_index, BreadcrumbSeverity::Info, label)
    }

    /// Creates a [`BreadcrumbSeverity::Warning`] breadcrumb.
    #[must_use]
    pub const fn warning(frame_index: u64, label: &'static str) -> Self {
        Self::new(frame_index, BreadcrumbSeverity::Warning, label)
    }

    /// Creates a [`BreadcrumbSeverity::Error`] breadcrumb.
    #[must_use]
    pub const fn error(frame_index: u64, label: &'static str) -> Self {
        Self::new(frame_index, BreadcrumbSeverity::Error, label)
    }

    /// Returns a copy of this breadcrumb carrying `detail`.
    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

/// A fixed-capacity ring buffer of [`Breadcrumb`]s.
///
/// The ring stores at most `capacity` entries. Once full, each push overwrites
/// the oldest entry. [`BreadcrumbTrail::iter`] and
/// [`BreadcrumbTrail::recent`] yield entries oldest-first.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BreadcrumbTrail {
    capacity: usize,
    entries: Vec<Breadcrumb>,
    /// Index of the oldest entry once the ring is full; `0` while filling.
    head: usize,
    /// Lifetime count of pushes, including those that overwrote older entries.
    recorded: u64,
}

impl BreadcrumbTrail {
    /// Creates an empty trail holding up to `capacity` entries.
    ///
    /// A `capacity` of `0` is clamped up to `1`, so the trail always retains at
    /// least the most recent breadcrumb and pushes never silently vanish.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            entries: Vec::with_capacity(capacity),
            head: 0,
            recorded: 0,
        }
    }

    /// Maximum number of entries the trail retains.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of entries currently stored (`<= capacity`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the trail holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns `true` when the trail is at capacity (further pushes overwrite).
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.entries.len() == self.capacity
    }

    /// Lifetime count of pushes, including overwrites.
    #[must_use]
    pub const fn total_recorded(&self) -> u64 {
        self.recorded
    }

    /// Records `crumb`, overwriting the oldest entry when the ring is full.
    pub fn push(&mut self, crumb: Breadcrumb) {
        if self.entries.len() < self.capacity {
            self.entries.push(crumb);
        } else {
            self.entries[self.head] = crumb;
            self.head = (self.head + 1) % self.capacity;
        }
        self.recorded = self.recorded.saturating_add(1);
    }

    /// Records an [`BreadcrumbSeverity::Info`] breadcrumb.
    pub fn info(&mut self, frame_index: u64, label: &'static str) {
        self.push(Breadcrumb::info(frame_index, label));
    }

    /// Records a [`BreadcrumbSeverity::Warning`] breadcrumb.
    pub fn warning(&mut self, frame_index: u64, label: &'static str) {
        self.push(Breadcrumb::warning(frame_index, label));
    }

    /// Records a [`BreadcrumbSeverity::Error`] breadcrumb.
    pub fn error(&mut self, frame_index: u64, label: &'static str) {
        self.push(Breadcrumb::error(frame_index, label));
    }

    /// Iterates stored entries oldest-first.
    pub fn iter(&self) -> impl Iterator<Item = &Breadcrumb> + '_ {
        let len = self.entries.len();
        let head = self.head;
        let capacity = self.capacity;
        (0..len).map(move |i| &self.entries[(head + i) % capacity])
    }

    /// Returns the most recent entry, or `None` when empty.
    #[must_use]
    pub fn last(&self) -> Option<&Breadcrumb> {
        let len = self.entries.len();
        if len == 0 {
            None
        } else {
            Some(&self.entries[(self.head + len - 1) % self.capacity])
        }
    }

    /// Returns up to the `n` most recent entries, oldest-first.
    ///
    /// When `n` exceeds the stored count, all entries are returned.
    #[must_use]
    pub fn recent(&self, n: usize) -> Vec<&Breadcrumb> {
        let len = self.entries.len();
        let skip = len.saturating_sub(n);
        self.iter().skip(skip).collect()
    }

    /// Counts stored entries at or above `min_severity`.
    #[must_use]
    pub fn count_at_least(&self, min_severity: BreadcrumbSeverity) -> usize {
        self.iter().filter(|c| c.severity >= min_severity).count()
    }

    /// Removes all stored entries.
    ///
    /// The [`BreadcrumbTrail::total_recorded`] lifetime counter is preserved so
    /// callers can still tell how many events have ever been seen.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.head = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(trail: &BreadcrumbTrail) -> Vec<&'static str> {
        trail.iter().map(|c| c.label).collect()
    }

    #[test]
    fn empty_trail_queries() {
        let trail = BreadcrumbTrail::new(4);
        assert!(trail.is_empty());
        assert_eq!(trail.len(), 0);
        assert!(trail.last().is_none());
        assert!(trail.recent(3).is_empty());
        assert_eq!(trail.total_recorded(), 0);
        assert!(!trail.is_full());
    }

    #[test]
    fn zero_capacity_is_clamped_to_one() {
        let mut trail = BreadcrumbTrail::new(0);
        assert_eq!(trail.capacity(), 1);
        trail.info(1, "a");
        trail.info(2, "b");
        assert_eq!(trail.len(), 1);
        assert_eq!(labels(&trail), ["b"]);
        assert_eq!(trail.total_recorded(), 2);
    }

    #[test]
    fn fills_in_order_without_overflow() {
        let mut trail = BreadcrumbTrail::new(3);
        trail.info(1, "a");
        trail.warning(2, "b");
        trail.error(3, "c");
        assert!(trail.is_full());
        assert_eq!(labels(&trail), ["a", "b", "c"]);
        assert_eq!(trail.last().map(|c| c.label), Some("c"));
    }

    #[test]
    fn overflow_overwrites_oldest_and_preserves_order() {
        let mut trail = BreadcrumbTrail::new(3);
        for (i, l) in ["a", "b", "c", "d", "e"].iter().enumerate() {
            trail.info(i as u64, l);
        }
        // a, b overwritten; window is c, d, e.
        assert_eq!(labels(&trail), ["c", "d", "e"]);
        assert_eq!(trail.len(), 3);
        assert_eq!(trail.total_recorded(), 5);
        assert_eq!(trail.last().map(|c| c.label), Some("e"));
    }

    #[test]
    fn recent_returns_tail_oldest_first() {
        let mut trail = BreadcrumbTrail::new(5);
        for (i, l) in ["a", "b", "c", "d"].iter().enumerate() {
            trail.info(i as u64, l);
        }
        assert_eq!(
            trail.recent(2).iter().map(|c| c.label).collect::<Vec<_>>(),
            ["c", "d"]
        );
        // n larger than stored -> everything.
        assert_eq!(
            trail.recent(99).iter().map(|c| c.label).collect::<Vec<_>>(),
            ["a", "b", "c", "d"]
        );
        // n == 0 -> nothing.
        assert!(trail.recent(0).is_empty());
    }

    #[test]
    fn recent_after_wraparound() {
        let mut trail = BreadcrumbTrail::new(3);
        for (i, l) in ["a", "b", "c", "d", "e"].iter().enumerate() {
            trail.info(i as u64, l);
        }
        assert_eq!(
            trail.recent(2).iter().map(|c| c.label).collect::<Vec<_>>(),
            ["d", "e"]
        );
    }

    #[test]
    fn frame_index_and_detail_are_retained() {
        let mut trail = BreadcrumbTrail::new(2);
        trail.push(Breadcrumb::error(42, "boom").with_detail("shader link failed"));
        let last = trail.last().unwrap();
        assert_eq!(last.frame_index, 42);
        assert_eq!(last.severity, BreadcrumbSeverity::Error);
        assert_eq!(last.detail.as_deref(), Some("shader link failed"));
    }

    #[test]
    fn count_at_least_filters_by_severity() {
        let mut trail = BreadcrumbTrail::new(8);
        trail.info(0, "i");
        trail.warning(1, "w");
        trail.error(2, "e");
        trail.warning(3, "w2");
        assert_eq!(trail.count_at_least(BreadcrumbSeverity::Info), 4);
        assert_eq!(trail.count_at_least(BreadcrumbSeverity::Warning), 3);
        assert_eq!(trail.count_at_least(BreadcrumbSeverity::Error), 1);
    }

    #[test]
    fn clear_empties_but_keeps_lifetime_count() {
        let mut trail = BreadcrumbTrail::new(3);
        trail.info(0, "a");
        trail.info(1, "b");
        trail.clear();
        assert!(trail.is_empty());
        assert_eq!(trail.total_recorded(), 2);
        // Reusable after clear.
        trail.info(2, "c");
        assert_eq!(labels(&trail), ["c"]);
    }

    #[test]
    fn push_sequence_is_deterministic() {
        let build = || {
            let mut t = BreadcrumbTrail::new(4);
            for i in 0..10u64 {
                t.info(i, "x");
            }
            t
        };
        let a = build();
        let b = build();
        assert_eq!(a, b);
        assert_eq!(a.len(), 4);
        assert_eq!(
            a.iter().map(|c| c.frame_index).collect::<Vec<_>>(),
            [6, 7, 8, 9]
        );
    }

    #[test]
    fn severity_ordering() {
        assert!(BreadcrumbSeverity::Info < BreadcrumbSeverity::Warning);
        assert!(BreadcrumbSeverity::Warning < BreadcrumbSeverity::Error);
    }
}
