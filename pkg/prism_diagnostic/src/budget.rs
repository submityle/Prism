//! §24.1 real-time performance budget + automatic regression alerts.
//!
//! An AAA frame time is a *contract* (60/120Hz must hold), and it cannot be
//! policed after the fact by reading a trace. This module turns a frame into a
//! budgeted contract with three cooperating pieces, all pure `core`/`alloc`
//! arithmetic (deterministic, `no_std` + `alloc`, no clock of its own):
//!
//! 1. **Budget declaration + runtime check** ([`BudgetRegistry`]): each
//!    subsystem declares a per-frame budget (render ≤8ms, physics ≤2ms, …);
//!    [`BudgetRegistry::evaluate_frame`] compares the measured costs, red-flags
//!    overspenders ([`BudgetStatus::over_budget`]), and reports the frame-level
//!    [`FrameBudgetReport`].
//! 2. **Scheduler feedback**: [`FrameBudgetReport::remaining_background_nanos`]
//!    is the headroom left under the frame budget after the measured foreground
//!    work; it feeds `prism_tasks` (tasks §24.1 frame-budget scheduling) so
//!    background jobs defer when the frame is tight. No dependency edge: the
//!    value is a plain `u64` the scheduler reads.
//! 3. **Automatic regression detection** ([`RegressionTracker`]): CI / nightly
//!    stores a per-span [`Baseline`] (p50/p99); a new sample over the configured
//!    threshold (e.g. +5%) raises a [`RegressionAlert`] attributed to a commit,
//!    so chronic performance rot is caught at the offending change.
//! 4. **Hotspot auto-attribution** ([`hotspot_diff`]): the top of the flame
//!    graph sorted by self time, diffed against the previous baseline, points
//!    straight at the regressed function.
//!
//! Percentiles use the same nearest-rank method as [`crate::hitch`], computed
//! with integer-friendly `core` arithmetic (no `std` rounding intrinsics).

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// A declared per-subsystem frame budget: a category name plus its nanosecond
/// allowance for a single frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameBudget {
    /// Subsystem/category label, e.g. `"render"`, `"physics"`, `"gameplay"`.
    pub category: String,
    /// Allowed per-frame cost in nanoseconds.
    pub budget_nanos: u64,
}

/// Result of comparing one subsystem's measured cost against its budget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetStatus {
    /// The category this status refers to.
    pub category: String,
    /// Measured cost this frame, in nanoseconds.
    pub measured_nanos: u64,
    /// Declared budget, in nanoseconds.
    pub budget_nanos: u64,
    /// Whether the measured cost exceeded the budget (the red flag).
    pub over_budget: bool,
    /// Saturating `measured - budget`; `0` when within budget.
    pub overspend_nanos: u64,
}

impl BudgetStatus {
    /// Utilization ratio `measured / budget` in `[0, +inf)`. A zero budget
    /// yields `0.0` (an unbudgeted category is never "over" by ratio).
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.budget_nanos == 0 {
            0.0
        } else {
            self.measured_nanos as f64 / self.budget_nanos as f64
        }
    }
}

/// A whole-frame budget evaluation across every declared subsystem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameBudgetReport {
    /// Per-category statuses, in the registry's declaration order.
    pub statuses: Vec<BudgetStatus>,
    /// Sum of all measured foreground costs this frame, in nanoseconds.
    pub total_measured_nanos: u64,
    /// Overall per-frame target (e.g. `16_667_000` for 60Hz), in nanoseconds.
    pub frame_budget_nanos: u64,
    /// Whether the summed foreground cost exceeded the frame budget.
    pub over_frame: bool,
    /// Headroom left under the frame budget after measured foreground work,
    /// saturating at `0`. Feeds `prism_tasks` frame-budget scheduling so
    /// background jobs yield when the frame is tight.
    pub remaining_background_nanos: u64,
}

impl FrameBudgetReport {
    /// Iterator over the categories that exceeded their individual budgets.
    pub fn overspenders(&self) -> impl Iterator<Item = &BudgetStatus> {
        self.statuses.iter().filter(|s| s.over_budget)
    }
}

/// A registry of per-subsystem frame budgets plus an overall frame target.
///
/// Budgets are stored in declaration order and looked up by linear scan (the
/// category count is small and fixed); this keeps the type `no_std`-friendly
/// without pulling in a hash map.
#[derive(Clone, Debug, Default)]
pub struct BudgetRegistry {
    budgets: Vec<FrameBudget>,
    frame_budget_nanos: u64,
}

impl BudgetRegistry {
    /// Create a registry with an overall per-frame target in nanoseconds (e.g.
    /// `16_667_000` for 60Hz, `8_333_000` for 120Hz). `0` disables the
    /// frame-level over/under decision while per-category checks still apply.
    #[must_use]
    pub fn new(frame_budget_nanos: u64) -> Self {
        Self {
            budgets: Vec::new(),
            frame_budget_nanos,
        }
    }

    /// A 60Hz registry: ~16.67ms overall frame budget, no categories yet.
    #[must_use]
    pub fn fps_60() -> Self {
        Self::new(16_667_000)
    }

    /// A 120Hz registry: ~8.33ms overall frame budget, no categories yet.
    #[must_use]
    pub fn fps_120() -> Self {
        Self::new(8_333_000)
    }

    /// The overall per-frame target in nanoseconds.
    #[must_use]
    pub fn frame_budget_nanos(&self) -> u64 {
        self.frame_budget_nanos
    }

    /// Declare (or overwrite) a subsystem budget. Returns `&mut Self` for
    /// chaining. A repeated category replaces its previous budget in place,
    /// preserving declaration order.
    pub fn declare(&mut self, category: impl Into<String>, budget_nanos: u64) -> &mut Self {
        let category = category.into();
        if let Some(existing) = self.budgets.iter_mut().find(|b| b.category == category) {
            existing.budget_nanos = budget_nanos;
        } else {
            self.budgets.push(FrameBudget {
                category,
                budget_nanos,
            });
        }
        self
    }

    /// The declared budget for a category, if any.
    #[must_use]
    pub fn budget_of(&self, category: &str) -> Option<u64> {
        self.budgets
            .iter()
            .find(|b| b.category == category)
            .map(|b| b.budget_nanos)
    }

    /// Number of declared categories.
    #[must_use]
    pub fn len(&self) -> usize {
        self.budgets.len()
    }

    /// Whether no categories are declared.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.budgets.is_empty()
    }

    /// Evaluate a single category's measured cost against its declared budget.
    /// Returns `None` if the category was never declared.
    #[must_use]
    pub fn evaluate(&self, category: &str, measured_nanos: u64) -> Option<BudgetStatus> {
        let budget = self.budgets.iter().find(|b| b.category == category)?;
        Some(status(
            &budget.category,
            measured_nanos,
            budget.budget_nanos,
        ))
    }

    /// Evaluate a whole frame: pair each `(category, measured_nanos)` with its
    /// declared budget and roll the per-category results up into a
    /// [`FrameBudgetReport`].
    ///
    /// Categories present in `measured` but never declared are treated as
    /// having a `0` budget (always over by overspend, never over by ratio);
    /// declared categories absent from `measured` are reported as `0` measured
    /// (within budget). The summed foreground cost uses every entry in
    /// `measured`, so the scheduler headroom reflects real work regardless of
    /// declaration coverage.
    #[must_use]
    pub fn evaluate_frame(&self, measured: &[(&str, u64)]) -> FrameBudgetReport {
        let mut statuses = Vec::with_capacity(self.budgets.len());
        // Declared categories first, in declaration order.
        for budget in &self.budgets {
            let measured_nanos = measured
                .iter()
                .find(|(name, _)| *name == budget.category)
                .map_or(0, |(_, v)| *v);
            statuses.push(status(
                &budget.category,
                measured_nanos,
                budget.budget_nanos,
            ));
        }
        // Undeclared-but-measured categories afterwards, with a zero budget.
        for (name, measured_nanos) in measured {
            if !self.budgets.iter().any(|b| b.category == *name) {
                statuses.push(status(name, *measured_nanos, 0));
            }
        }

        let total_measured_nanos = measured
            .iter()
            .map(|(_, v)| *v)
            .fold(0u64, u64::saturating_add);
        let over_frame =
            self.frame_budget_nanos != 0 && total_measured_nanos > self.frame_budget_nanos;
        let remaining_background_nanos =
            self.frame_budget_nanos.saturating_sub(total_measured_nanos);

        FrameBudgetReport {
            statuses,
            total_measured_nanos,
            frame_budget_nanos: self.frame_budget_nanos,
            over_frame,
            remaining_background_nanos,
        }
    }
}

fn status(category: &str, measured_nanos: u64, budget_nanos: u64) -> BudgetStatus {
    let over_budget = budget_nanos != 0 && measured_nanos > budget_nanos;
    BudgetStatus {
        category: String::from(category),
        measured_nanos,
        budget_nanos,
        over_budget,
        overspend_nanos: measured_nanos.saturating_sub(budget_nanos),
    }
}

/// A stored performance baseline for one span key: its p50/p99 in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Baseline {
    /// Median (p50) cost, in nanoseconds.
    pub p50_nanos: u64,
    /// Tail (p99) cost, in nanoseconds.
    pub p99_nanos: u64,
}

impl Baseline {
    /// Build a baseline from a sample window by nearest-rank p50/p99.
    #[must_use]
    pub fn from_samples(samples: &[u64]) -> Self {
        Self {
            p50_nanos: percentile_nearest_rank(samples, 0.50),
            p99_nanos: percentile_nearest_rank(samples, 0.99),
        }
    }
}

/// Configuration for regression detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegressionConfig {
    /// Trigger ratio: a sample percentile greater than `threshold_ratio *
    /// baseline` is a regression. `1.05` means "+5%".
    pub threshold_ratio: f64,
    /// Absolute noise floor in nanoseconds: baselines at or below this are too
    /// small to judge by ratio (micro-spans), so they never alert. `0`
    /// disables the floor.
    pub min_baseline_nanos: u64,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            threshold_ratio: 1.05,
            min_baseline_nanos: 50_000,
        }
    }
}

/// A raised regression: a span whose p50 and/or p99 exceeded the threshold vs
/// its baseline, attributed to a commit for CI localization.
#[derive(Clone, Debug, PartialEq)]
pub struct RegressionAlert {
    /// Span key that regressed.
    pub key: String,
    /// Baseline p50 / sampled p50, in nanoseconds.
    pub baseline_p50: u64,
    /// Sampled p50 this run, in nanoseconds.
    pub sample_p50: u64,
    /// Baseline p99 / sampled p99, in nanoseconds.
    pub baseline_p99: u64,
    /// Sampled p99 this run, in nanoseconds.
    pub sample_p99: u64,
    /// Whether p50 crossed the threshold.
    pub p50_regressed: bool,
    /// Whether p99 crossed the threshold.
    pub p99_regressed: bool,
    /// `sample_p50 / baseline_p50`.
    pub p50_ratio: f64,
    /// `sample_p99 / baseline_p99`.
    pub p99_ratio: f64,
    /// Attributed commit identifier (short hash/tag), if provided.
    pub commit: Option<String>,
}

/// A baseline store that detects regressions as new samples arrive.
///
/// Baselines are keyed by span name and looked up by linear scan (keys are a
/// small, mostly fixed set of "key" spans). No clock; callers supply sampled
/// percentiles.
#[derive(Clone, Debug, Default)]
pub struct RegressionTracker {
    config: RegressionConfig,
    baselines: Vec<(String, Baseline)>,
}

impl RegressionTracker {
    /// Create a tracker with the given config.
    #[must_use]
    pub fn new(config: RegressionConfig) -> Self {
        Self {
            config,
            baselines: Vec::new(),
        }
    }

    /// Store (or overwrite) the baseline for a span key.
    pub fn set_baseline(&mut self, key: impl Into<String>, baseline: Baseline) -> &mut Self {
        let key = key.into();
        if let Some(entry) = self.baselines.iter_mut().find(|(k, _)| *k == key) {
            entry.1 = baseline;
        } else {
            self.baselines.push((key, baseline));
        }
        self
    }

    /// The stored baseline for a key, if any.
    #[must_use]
    pub fn baseline_of(&self, key: &str) -> Option<Baseline> {
        self.baselines
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, b)| *b)
    }

    /// Check a new sample against the stored baseline for `key`.
    ///
    /// Returns `Some(RegressionAlert)` only when p50 or p99 crossed the
    /// threshold and the baseline is above the noise floor. Returns `None` when
    /// there is no baseline, the baseline is below the floor, or nothing
    /// regressed. Does not mutate the stored baseline (promotion is an explicit
    /// `set_baseline` once a new baseline is accepted).
    #[must_use]
    pub fn check(
        &self,
        key: &str,
        sample_p50: u64,
        sample_p99: u64,
        commit: Option<&str>,
    ) -> Option<RegressionAlert> {
        let baseline = self.baseline_of(key)?;

        let p50_ratio = ratio(sample_p50, baseline.p50_nanos);
        let p99_ratio = ratio(sample_p99, baseline.p99_nanos);

        let p50_regressed = baseline.p50_nanos > self.config.min_baseline_nanos
            && sample_p50 as f64 > self.config.threshold_ratio * baseline.p50_nanos as f64;
        let p99_regressed = baseline.p99_nanos > self.config.min_baseline_nanos
            && sample_p99 as f64 > self.config.threshold_ratio * baseline.p99_nanos as f64;

        if !p50_regressed && !p99_regressed {
            return None;
        }

        Some(RegressionAlert {
            key: String::from(key),
            baseline_p50: baseline.p50_nanos,
            sample_p50,
            baseline_p99: baseline.p99_nanos,
            sample_p99,
            p50_regressed,
            p99_regressed,
            p50_ratio,
            p99_ratio,
            commit: commit.map(String::from),
        })
    }
}

/// One flame-graph entry: a span name plus its self (exclusive) time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hotspot {
    /// Span/function name.
    pub name: String,
    /// Self (exclusive) time this frame, in nanoseconds.
    pub self_nanos: u64,
}

impl Hotspot {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, self_nanos: u64) -> Self {
        Self {
            name: name.into(),
            self_nanos,
        }
    }
}

/// A per-hotspot diff of the current frame against a baseline.
#[derive(Clone, Debug, PartialEq)]
pub struct HotspotDelta {
    /// Span/function name.
    pub name: String,
    /// Current self time, in nanoseconds.
    pub current_nanos: u64,
    /// Baseline self time, in nanoseconds (`0` if newly appeared).
    pub baseline_nanos: u64,
    /// Signed `current - baseline`, in nanoseconds.
    pub delta_nanos: i64,
    /// `current / baseline` (`+inf`-style sentinel is avoided: a zero baseline
    /// yields `0.0`).
    pub ratio: f64,
    /// Whether this hotspot crossed the regression threshold.
    pub is_regression: bool,
}

/// Diff the current flame-graph hotspots against a baseline, sorted by current
/// self time descending (the top of the flame graph first), flagging any entry
/// over `threshold_ratio` as a regression.
///
/// Entries are matched by name; a hotspot absent from the baseline is treated
/// as having a `0` baseline (newly appeared, always flagged when non-trivial).
#[must_use]
pub fn hotspot_diff(
    current: &[Hotspot],
    baseline: &[Hotspot],
    threshold_ratio: f64,
) -> Vec<HotspotDelta> {
    let mut out: Vec<HotspotDelta> = current
        .iter()
        .map(|h| {
            let base = baseline
                .iter()
                .find(|b| b.name == h.name)
                .map_or(0, |b| b.self_nanos);
            let delta_nanos = h.self_nanos as i64 - base as i64;
            let ratio = ratio(h.self_nanos, base);
            // A newly appeared hotspot (zero baseline) with real cost regresses;
            // otherwise compare the ratio against the threshold.
            let is_regression = if base == 0 {
                h.self_nanos > 0
            } else {
                h.self_nanos as f64 > threshold_ratio * base as f64
            };
            HotspotDelta {
                name: h.name.clone(),
                current_nanos: h.self_nanos,
                baseline_nanos: base,
                delta_nanos,
                ratio,
                is_regression,
            }
        })
        .collect();
    // Stable sort by current self time descending; ties keep input order.
    out.sort_by_key(|d| core::cmp::Reverse(d.current_nanos));
    out
}

fn ratio(sample: u64, baseline: u64) -> f64 {
    if baseline == 0 {
        0.0
    } else {
        sample as f64 / baseline as f64
    }
}

/// Nearest-rank percentile of a slice, integer-friendly `core` arithmetic.
/// Returns `0` for an empty input; `p` is clamped to `[0, 1]`.
fn percentile_nearest_rank(samples: &[u64], p: f64) -> u64 {
    let n = samples.len();
    if n == 0 {
        return 0;
    }
    let mut sorted: Vec<u64> = samples.to_vec();
    sorted.sort_unstable();

    let p = p.clamp(0.0, 1.0);
    let product = p * (n as f64);
    let mut rank = product as usize;
    if (rank as f64) < product {
        rank += 1;
    }
    let rank = rank.clamp(1, n);
    sorted[rank - 1]
}
