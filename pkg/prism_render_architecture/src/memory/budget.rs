//! Per-frame work-budget accounting and per-heap statistics aggregation.
//!
//! Two related contracts live here. First, a ledger that accumulates the
//! transfer and build work a frame asks the `GPU` to do — bytes uploaded, bytes
//! read back, bytes relocated during defragmentation, and acceleration-structure
//! builds — and answers whether the frame stayed inside a configured budget.
//! Keeping the running total on the `CPU` lets the frame graph refuse or defer
//! work before it is ever submitted.
//!
//! Second, a small registry that collects one [`HeapStats`] per
//! [`HeapClass`] so diagnostics can report committed and used memory across the
//! whole device. Both are integer-only and deterministic; the values they hold
//! are produced by the allocators in this subsystem and, ultimately, by the
//! backend, whose device queries are *pending the GPU backend*.

use super::{FrameWorkBudget, HeapClass, HeapStats};

impl HeapClass {
    /// Every heap class, in a fixed order suitable for array indexing.
    pub const ALL: [Self; 5] = [
        Self::DeviceLocal,
        Self::Upload,
        Self::Readback,
        Self::Transient,
        Self::AccelerationStructure,
    ];

    /// Number of heap classes.
    pub const COUNT: usize = Self::ALL.len();

    /// Dense index of this class within [`ALL`](Self::ALL).
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::DeviceLocal => 0,
            Self::Upload => 1,
            Self::Readback => 2,
            Self::Transient => 3,
            Self::AccelerationStructure => 4,
        }
    }
}

/// How far a frame has exceeded its budget in each category.
///
/// A field of zero means that category is within budget. Produced by
/// [`FrameBudgetLedger::overage`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BudgetOverage {
    /// Upload bytes over the limit.
    pub upload_bytes: u64,
    /// Readback bytes over the limit.
    pub readback_bytes: u64,
    /// Relocation bytes over the limit.
    pub relocation_bytes: u64,
    /// Acceleration-structure builds over the limit.
    pub acceleration_structure_builds: u32,
}

impl BudgetOverage {
    /// Whether any category is over budget.
    #[must_use]
    pub const fn any(&self) -> bool {
        self.upload_bytes > 0
            || self.readback_bytes > 0
            || self.relocation_bytes > 0
            || self.acceleration_structure_builds > 0
    }
}

/// Accumulates a frame's `GPU` work and compares it against a fixed budget.
///
/// Every recording method saturates rather than overflows, so a pathological
/// frame reports a clamped total instead of panicking. Reuse the ledger across
/// frames via [`reset`](Self::reset).
#[derive(Clone, Copy, Debug)]
pub struct FrameBudgetLedger {
    limits: FrameWorkBudget,
    consumed: FrameWorkBudget,
}

impl FrameBudgetLedger {
    /// Creates a ledger enforcing `limits`, with nothing consumed yet.
    #[must_use]
    pub const fn new(limits: FrameWorkBudget) -> Self {
        Self {
            limits,
            consumed: FrameWorkBudget {
                upload_bytes: 0,
                readback_bytes: 0,
                relocation_bytes: 0,
                acceleration_structure_builds: 0,
            },
        }
    }

    /// The configured budget limits.
    #[must_use]
    pub const fn limits(&self) -> FrameWorkBudget {
        self.limits
    }

    /// The work recorded so far this frame.
    #[must_use]
    pub const fn consumed(&self) -> FrameWorkBudget {
        self.consumed
    }

    /// Records `bytes` of upload (host to device) traffic.
    pub const fn record_upload(&mut self, bytes: u64) {
        self.consumed.upload_bytes = self.consumed.upload_bytes.saturating_add(bytes);
    }

    /// Records `bytes` of readback (device to host) traffic.
    pub const fn record_readback(&mut self, bytes: u64) {
        self.consumed.readback_bytes = self.consumed.readback_bytes.saturating_add(bytes);
    }

    /// Records `bytes` moved by a defragmentation relocation.
    pub const fn record_relocation(&mut self, bytes: u64) {
        self.consumed.relocation_bytes = self.consumed.relocation_bytes.saturating_add(bytes);
    }

    /// Records `count` acceleration-structure builds.
    pub const fn record_acceleration_structure_builds(&mut self, count: u32) {
        self.consumed.acceleration_structure_builds = self
            .consumed
            .acceleration_structure_builds
            .saturating_add(count);
    }

    /// Whether every category is at or under its limit.
    #[must_use]
    pub const fn is_within_budget(&self) -> bool {
        self.consumed.upload_bytes <= self.limits.upload_bytes
            && self.consumed.readback_bytes <= self.limits.readback_bytes
            && self.consumed.relocation_bytes <= self.limits.relocation_bytes
            && self.consumed.acceleration_structure_builds
                <= self.limits.acceleration_structure_builds
    }

    /// Remaining headroom in each category, saturating at zero once exceeded.
    #[must_use]
    pub const fn remaining(&self) -> FrameWorkBudget {
        FrameWorkBudget {
            upload_bytes: self
                .limits
                .upload_bytes
                .saturating_sub(self.consumed.upload_bytes),
            readback_bytes: self
                .limits
                .readback_bytes
                .saturating_sub(self.consumed.readback_bytes),
            relocation_bytes: self
                .limits
                .relocation_bytes
                .saturating_sub(self.consumed.relocation_bytes),
            acceleration_structure_builds: self
                .limits
                .acceleration_structure_builds
                .saturating_sub(self.consumed.acceleration_structure_builds),
        }
    }

    /// Amount by which each category exceeds its limit (zero if within budget).
    #[must_use]
    pub const fn overage(&self) -> BudgetOverage {
        BudgetOverage {
            upload_bytes: self
                .consumed
                .upload_bytes
                .saturating_sub(self.limits.upload_bytes),
            readback_bytes: self
                .consumed
                .readback_bytes
                .saturating_sub(self.limits.readback_bytes),
            relocation_bytes: self
                .consumed
                .relocation_bytes
                .saturating_sub(self.limits.relocation_bytes),
            acceleration_structure_builds: self
                .consumed
                .acceleration_structure_builds
                .saturating_sub(self.limits.acceleration_structure_builds),
        }
    }

    /// Whether a prospective upload of `bytes` would stay within budget.
    #[must_use]
    pub const fn upload_fits(&self, bytes: u64) -> bool {
        self.consumed.upload_bytes.saturating_add(bytes) <= self.limits.upload_bytes
    }

    /// Clears consumed work for the next frame; limits are preserved.
    pub const fn reset(&mut self) {
        self.consumed = FrameWorkBudget {
            upload_bytes: 0,
            readback_bytes: 0,
            relocation_bytes: 0,
            acceleration_structure_builds: 0,
        };
    }
}

/// Per-heap-class collection of [`HeapStats`], indexed densely by class.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeapStatsRegistry {
    stats: [HeapStats; HeapClass::COUNT],
}

impl HeapStatsRegistry {
    /// Creates a registry with every class zeroed.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the latest stats for `class`, replacing any prior snapshot.
    pub const fn set(&mut self, class: HeapClass, stats: HeapStats) {
        self.stats[class.index()] = stats;
    }

    /// Current stats for `class`.
    #[must_use]
    pub const fn get(&self, class: HeapClass) -> HeapStats {
        self.stats[class.index()]
    }

    /// Committed bytes summed across every class.
    #[must_use]
    pub fn total_committed(&self) -> u64 {
        self.stats
            .iter()
            .fold(0u64, |acc, stats| acc.saturating_add(stats.committed_bytes))
    }

    /// Used bytes summed across every class.
    #[must_use]
    pub fn total_used(&self) -> u64 {
        self.stats
            .iter()
            .fold(0u64, |acc, stats| acc.saturating_add(stats.used_bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn limits() -> FrameWorkBudget {
        FrameWorkBudget {
            upload_bytes: 1000,
            readback_bytes: 500,
            relocation_bytes: 200,
            acceleration_structure_builds: 4,
        }
    }

    #[test]
    fn accumulates_and_stays_within_budget() {
        let mut ledger = FrameBudgetLedger::new(limits());
        ledger.record_upload(400);
        ledger.record_upload(300);
        ledger.record_readback(100);
        ledger.record_relocation(50);
        ledger.record_acceleration_structure_builds(2);
        assert!(ledger.is_within_budget());
        assert_eq!(ledger.consumed().upload_bytes, 700);

        let remaining = ledger.remaining();
        assert_eq!(remaining.upload_bytes, 300);
        assert_eq!(remaining.readback_bytes, 400);
        assert_eq!(remaining.acceleration_structure_builds, 2);
        assert!(!ledger.overage().any());
    }

    #[test]
    fn detects_overage_per_category() {
        let mut ledger = FrameBudgetLedger::new(limits());
        ledger.record_upload(1200);
        ledger.record_acceleration_structure_builds(5);
        assert!(!ledger.is_within_budget());

        let overage = ledger.overage();
        assert!(overage.any());
        assert_eq!(overage.upload_bytes, 200);
        assert_eq!(overage.acceleration_structure_builds, 1);
        // Untouched categories are not over budget.
        assert_eq!(overage.readback_bytes, 0);
        // Remaining saturates at zero for the exceeded category.
        assert_eq!(ledger.remaining().upload_bytes, 0);
    }

    #[test]
    fn upload_fits_predicts_the_next_record() {
        let mut ledger = FrameBudgetLedger::new(limits());
        ledger.record_upload(900);
        assert!(ledger.upload_fits(100));
        assert!(!ledger.upload_fits(101));
    }

    #[test]
    fn recording_saturates_instead_of_overflowing() {
        let mut ledger = FrameBudgetLedger::new(FrameWorkBudget {
            upload_bytes: u64::MAX,
            ..limits()
        });
        ledger.record_upload(u64::MAX);
        ledger.record_upload(u64::MAX);
        // Clamped, not wrapped.
        assert_eq!(ledger.consumed().upload_bytes, u64::MAX);
        assert!(ledger.is_within_budget());
    }

    #[test]
    fn reset_clears_consumption_but_keeps_limits() {
        let mut ledger = FrameBudgetLedger::new(limits());
        ledger.record_upload(500);
        ledger.reset();
        assert_eq!(ledger.consumed().upload_bytes, 0);
        assert_eq!(ledger.limits().upload_bytes, 1000);
    }

    #[test]
    fn registry_stores_and_totals_per_class() {
        let mut registry = HeapStatsRegistry::new();
        registry.set(
            HeapClass::DeviceLocal,
            HeapStats {
                committed_bytes: 1024,
                used_bytes: 512,
                largest_free_block: 512,
            },
        );
        registry.set(
            HeapClass::Upload,
            HeapStats {
                committed_bytes: 256,
                used_bytes: 128,
                largest_free_block: 128,
            },
        );
        assert_eq!(registry.get(HeapClass::DeviceLocal).used_bytes, 512);
        assert_eq!(registry.total_committed(), 1280);
        assert_eq!(registry.total_used(), 640);
        // Untouched classes stay zeroed.
        assert_eq!(registry.get(HeapClass::Readback).committed_bytes, 0);
    }

    #[test]
    fn every_heap_class_has_a_unique_index() {
        let mut seen = [false; HeapClass::COUNT];
        for class in HeapClass::ALL {
            let index = class.index();
            assert!(!seen[index], "duplicate index {index}");
            seen[index] = true;
        }
        assert!(seen.iter().all(|&hit| hit));
    }
}
