//! Per-category memory budget guard (§24.3).
//!
//! A shipping build dies when a pool silently grows past the memory the target
//! platform actually has. This module turns a memory budget into an explicit,
//! per-category contract: declare how many bytes `assets`/`render`/`gameplay`
//! are each allowed to hold live, feed the measured live bytes at a frame or
//! level boundary, and red-flag every category that is over — *before* the OS
//! OOM-kills the process.
//!
//! Like the time-budget layer in [`crate::budget`], this is pure `core`/`alloc`
//! arithmetic: no allocator hot-path involvement, no `unsafe`, deterministic,
//! and always compiled regardless of the `alloc-track` feature. With
//! `alloc-track` on, measured bytes can be sourced from
//! [`crate::alloc_track::tag_report`] by matching category names to tag names.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// A declared per-category memory budget: a label plus its live-byte ceiling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemBudget {
    /// Category label, e.g. `"assets"`, `"render"`, `"gameplay"`.
    pub category: String,
    /// Allowed live bytes for this category.
    pub budget_bytes: u64,
}

/// Result of comparing one category's measured live bytes against its budget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemBudgetStatus {
    /// The category this status refers to.
    pub category: String,
    /// Measured live bytes.
    pub measured_bytes: u64,
    /// Declared budget, in bytes.
    pub budget_bytes: u64,
    /// Whether the measured live bytes exceeded the budget (the red flag).
    pub over_budget: bool,
    /// Saturating `measured - budget`; `0` when within budget.
    pub overspend_bytes: u64,
}

impl MemBudgetStatus {
    /// Utilization ratio `measured / budget` in `[0, +inf)`. A zero budget
    /// yields `0.0` (an unbudgeted category is never "over" by ratio).
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.budget_bytes == 0 {
            0.0
        } else {
            self.measured_bytes as f64 / self.budget_bytes as f64
        }
    }
}

/// A whole-snapshot budget evaluation across every declared category.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemBudgetReport {
    /// Per-category statuses, in the registry's declaration order.
    pub statuses: Vec<MemBudgetStatus>,
    /// Sum of all measured live bytes across declared categories.
    pub total_measured_bytes: u64,
    /// Sum of all declared budgets.
    pub total_budget_bytes: u64,
    /// Whether any category was over budget.
    pub any_over_budget: bool,
}

impl MemBudgetReport {
    /// Iterator over just the categories that exceeded their budget.
    pub fn offenders(&self) -> impl Iterator<Item = &MemBudgetStatus> {
        self.statuses.iter().filter(|s| s.over_budget)
    }

    /// Total bytes over budget, summed across every offending category.
    #[must_use]
    pub fn total_overspend_bytes(&self) -> u64 {
        self.statuses
            .iter()
            .map(|s| s.overspend_bytes)
            .fold(0u64, u64::saturating_add)
    }
}

/// A registry of per-category memory budgets.
///
/// Categories are kept in declaration order so reports are stable for diffing
/// and HUD rendering. Re-declaring an existing category updates its budget in
/// place rather than appending a duplicate.
#[derive(Clone, Debug, Default)]
pub struct MemBudgetRegistry {
    budgets: Vec<MemBudget>,
}

impl MemBudgetRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            budgets: Vec::new(),
        }
    }

    /// Declare (or update) a category's live-byte budget. Returns `self` for
    /// chaining during setup.
    pub fn declare(&mut self, category: impl Into<String>, budget_bytes: u64) -> &mut Self {
        let category = category.into();
        if let Some(existing) = self.budgets.iter_mut().find(|b| b.category == category) {
            existing.budget_bytes = budget_bytes;
        } else {
            self.budgets.push(MemBudget {
                category,
                budget_bytes,
            });
        }
        self
    }

    /// Number of declared categories.
    #[must_use]
    pub fn len(&self) -> usize {
        self.budgets.len()
    }

    /// Whether no categories have been declared.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.budgets.is_empty()
    }

    /// Look up a single category's declared budget, if any.
    #[must_use]
    pub fn budget_of(&self, category: &str) -> Option<u64> {
        self.budgets
            .iter()
            .find(|b| b.category == category)
            .map(|b| b.budget_bytes)
    }

    /// Evaluate one category's measured live bytes against its declared budget.
    ///
    /// Returns `None` if the category was never declared.
    #[must_use]
    pub fn evaluate(&self, category: &str, measured_bytes: u64) -> Option<MemBudgetStatus> {
        let budget = self.budgets.iter().find(|b| b.category == category)?;
        Some(status(
            &budget.category,
            measured_bytes,
            budget.budget_bytes,
        ))
    }

    /// Evaluate every declared category against a lookup of measured live bytes.
    ///
    /// `measured` is called once per declared category, in declaration order; a
    /// category with no measurement is treated as `0` live bytes (within
    /// budget). Categories that are measured but never declared are ignored —
    /// declare them to enforce a ceiling.
    #[must_use]
    pub fn evaluate_all<F>(&self, mut measured: F) -> MemBudgetReport
    where
        F: FnMut(&str) -> u64,
    {
        let mut statuses = Vec::with_capacity(self.budgets.len());
        let mut total_measured_bytes = 0u64;
        let mut total_budget_bytes = 0u64;
        let mut any_over_budget = false;
        for budget in &self.budgets {
            let measured_bytes = measured(&budget.category);
            let st = status(&budget.category, measured_bytes, budget.budget_bytes);
            total_measured_bytes = total_measured_bytes.saturating_add(measured_bytes);
            total_budget_bytes = total_budget_bytes.saturating_add(budget.budget_bytes);
            any_over_budget |= st.over_budget;
            statuses.push(st);
        }
        MemBudgetReport {
            statuses,
            total_measured_bytes,
            total_budget_bytes,
            any_over_budget,
        }
    }

    /// Evaluate every declared category against the per-tag **live** bytes
    /// reported by [`crate::alloc_track::tag_report`], matching each budget
    /// category label to the allocation tag of the same name.
    ///
    /// This is the turn-key path for the per-tag live signal produced by
    /// [`LiveTrackingAllocator`]: instead of hand-rolling the category-to-tag
    /// name match at the call site, declare budgets whose labels equal the tag
    /// names (`"assets"`/`"render"`/`"gameplay"`) and read the measured live
    /// residency straight from the allocator. A category with no matching tag
    /// (or whose tag has no live bytes, e.g. under the header-free
    /// [`TrackingAllocator`]) is treated as `0` live bytes. Only available with
    /// the `alloc-track` feature.
    ///
    /// [`LiveTrackingAllocator`]: crate::alloc_track::LiveTrackingAllocator
    /// [`TrackingAllocator`]: crate::alloc_track::TrackingAllocator
    #[cfg(feature = "alloc-track")]
    #[must_use]
    pub fn evaluate_from_live_tags(&self) -> MemBudgetReport {
        let report = crate::alloc_track::tag_report();
        self.evaluate_all(|category| {
            report
                .iter()
                .find(|stat| stat.name == category)
                .map_or(0, |stat| stat.live_bytes)
        })
    }
}

/// Build a [`MemBudgetStatus`] from a measured/budget pair.
fn status(category: &str, measured_bytes: u64, budget_bytes: u64) -> MemBudgetStatus {
    let overspend_bytes = measured_bytes.saturating_sub(budget_bytes);
    MemBudgetStatus {
        category: String::from(category),
        measured_bytes,
        budget_bytes,
        over_budget: overspend_bytes > 0,
        overspend_bytes,
    }
}

#[cfg(all(test, feature = "alloc-track"))]
#[expect(
    unsafe_code,
    reason = "driving GlobalAlloc::alloc/dealloc directly to exercise per-tag \
              live-byte budgeting; every block is freed with its own layout"
)]
mod live_tag_tests {
    use super::*;
    use crate::alloc_track::{register_tag, tag_scope, LiveTrackingAllocator};
    use core::alloc::{GlobalAlloc, Layout};
    use std::alloc::System;

    // Drive a tagged allocation through the live allocator so the budget guard
    // can read real per-tag live residency, then free it and re-check.
    #[test]
    fn evaluate_from_live_tags_reads_allocator_residency() {
        let category = "prism::test::budget::live_render";
        let tag = register_tag(category).expect("tag slot");
        let alloc = LiveTrackingAllocator::new(System);

        let mut registry = MemBudgetRegistry::new();
        registry.declare(category, 4096);

        // No live bytes yet: within budget.
        let before = registry.evaluate_from_live_tags();
        let st_before = before
            .statuses
            .iter()
            .find(|s| s.category == category)
            .expect("declared category present");
        let baseline = st_before.measured_bytes;
        assert!(!st_before.over_budget, "baseline must be within budget");

        // Allocate 8192 bytes under the tag -> over the 4096 budget.
        let layout = Layout::from_size_align(8192, 16).unwrap();
        let ptr = {
            let _scope = tag_scope(tag);
            // SAFETY: non-zero layout; freed below with the same layout.
            unsafe { alloc.alloc(layout) }
        };
        assert!(!ptr.is_null());

        let over = registry.evaluate_from_live_tags();
        let st_over = over
            .statuses
            .iter()
            .find(|s| s.category == category)
            .expect("category present");
        assert_eq!(st_over.measured_bytes, baseline + 8192);
        assert!(st_over.over_budget, "8192 live must exceed the 4096 budget");
        assert_eq!(
            st_over.overspend_bytes,
            (baseline + 8192).saturating_sub(4096)
        );
        assert!(over.any_over_budget);

        // Free it: live residency returns to baseline, back within budget.
        // SAFETY: same block/layout as the allocation above.
        unsafe { alloc.dealloc(ptr, layout) };
        let after = registry.evaluate_from_live_tags();
        let st_after = after
            .statuses
            .iter()
            .find(|s| s.category == category)
            .expect("category present");
        assert_eq!(st_after.measured_bytes, baseline);
        assert!(
            !st_after.over_budget,
            "after free must be within budget again"
        );
    }

    // A declared category with no matching allocation tag reads as 0 live bytes.
    #[test]
    fn undeclared_tag_category_reads_zero() {
        let mut registry = MemBudgetRegistry::new();
        registry.declare("prism::test::budget::no_such_tag", 1024);
        let report = registry.evaluate_from_live_tags();
        let st = report
            .statuses
            .iter()
            .find(|s| s.category == "prism::test::budget::no_such_tag")
            .expect("declared category present");
        assert_eq!(st.measured_bytes, 0);
        assert!(!st.over_budget);
    }
}
