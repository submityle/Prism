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
        Some(status(&budget.category, measured_bytes, budget.budget_bytes))
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
