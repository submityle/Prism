//! Bounded runtime telemetry for StateTree design feedback.
//!
//! The collector stores aggregate counters rather than individual samples.
//! Every metric vector is allocated once by [`TelemetryCollector::with_capacity`]
//! and recording an existing key, or a new key below its configured capacity,
//! does not grow any vector. Reports are sorted by their stable numeric keys so
//! their contents do not depend on event arrival order.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

use alloc::vec::Vec;

/// The metric family associated with a telemetry capacity or arithmetic error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TelemetryMetric {
    /// Per-state residency samples.
    Residency,
    /// Per-transition migration counts.
    Migration,
    /// Per-guard evaluation and false-result counts.
    Guard,
    /// Per-LOD sample and duration counts.
    Lod,
    /// Per-budget event and unit counts.
    Budget,
}

/// A structured failure returned while recording or merging telemetry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TelemetryError {
    /// A new key would exceed the selected capacity for its metric family.
    CapacityExceeded {
        /// Metric family that reached its capacity.
        metric: TelemetryMetric,
        /// Configured maximum number of distinct keys for the family.
        capacity: usize,
    },
    /// A counter would overflow its `u64` representation.
    CounterOverflow {
        /// Metric family containing the counter.
        metric: TelemetryMetric,
        /// Stable key whose counter could not be increased.
        key: u64,
    },
}

/// Per-family maximum numbers of distinct keys retained by telemetry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TelemetryCapacity {
    residency: usize,
    migration: usize,
    guard: usize,
    lod: usize,
    budget: usize,
}

impl TelemetryCapacity {
    /// Creates the same distinct-key capacity for every metric family.
    #[must_use]
    pub const fn uniform(capacity: usize) -> Self {
        Self {
            residency: capacity,
            migration: capacity,
            guard: capacity,
            lod: capacity,
            budget: capacity,
        }
    }

    /// Creates independent capacities for all metric families.
    #[must_use]
    pub const fn new(
        residency: usize,
        migration: usize,
        guard: usize,
        lod: usize,
        budget: usize,
    ) -> Self {
        Self {
            residency,
            migration,
            guard,
            lod,
            budget,
        }
    }

    /// Returns the residency-key capacity.
    #[must_use]
    pub const fn residency(self) -> usize {
        self.residency
    }

    /// Returns the migration-key capacity.
    #[must_use]
    pub const fn migration(self) -> usize {
        self.migration
    }

    /// Returns the guard-key capacity.
    #[must_use]
    pub const fn guard(self) -> usize {
        self.guard
    }

    /// Returns the LOD-key capacity.
    #[must_use]
    pub const fn lod(self) -> usize {
        self.lod
    }

    /// Returns the budget-key capacity.
    #[must_use]
    pub const fn budget(self) -> usize {
        self.budget
    }
}

/// Aggregated residency observations for one state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidencyMetric {
    /// Stable state identity supplied by the host.
    pub state: u32,
    /// Number of completed residency observations.
    pub samples: u64,
    /// Sum of all observed residency durations in host-defined ticks.
    pub total_ticks: u64,
    /// Shortest observed residency in host-defined ticks.
    pub min_ticks: u64,
    /// Longest observed residency in host-defined ticks.
    pub max_ticks: u64,
}

/// Aggregated migration observations for one transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MigrationMetric {
    /// Stable transition identity supplied by the host.
    pub transition: u32,
    /// Number of times the transition was applied.
    pub count: u64,
}

/// Aggregated guard observations for one guard.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuardMetric {
    /// Stable guard identity supplied by the host.
    pub guard: u32,
    /// Number of guard evaluations.
    pub evaluated: u64,
    /// Number of evaluations that returned false.
    pub false_count: u64,
}

/// Aggregated LOD observations for one level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodMetric {
    /// LOD level supplied by the host.
    pub lod: u8,
    /// Number of observations for this level.
    pub samples: u64,
    /// Sum of observed LOD residency durations in host-defined ticks.
    pub total_ticks: u64,
}

/// Aggregated budget observations for one budget key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BudgetMetric {
    /// Stable budget identity supplied by the host.
    pub budget: u32,
    /// Number of budget accounting events.
    pub count: u64,
    /// Sum of host-defined budget units accounted for by those events.
    pub units: u64,
}

/// A deterministic, bounded snapshot of aggregated telemetry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryReport {
    capacities: TelemetryCapacity,
    residency: Vec<ResidencyMetric>,
    migrations: Vec<MigrationMetric>,
    guards: Vec<GuardMetric>,
    lod: Vec<LodMetric>,
    budgets: Vec<BudgetMetric>,
}

impl TelemetryReport {
    /// Creates an empty report with reserved storage for every metric family.
    #[must_use]
    pub fn with_capacity(capacities: TelemetryCapacity) -> Self {
        Self {
            capacities,
            residency: Vec::with_capacity(capacities.residency),
            migrations: Vec::with_capacity(capacities.migration),
            guards: Vec::with_capacity(capacities.guard),
            lod: Vec::with_capacity(capacities.lod),
            budgets: Vec::with_capacity(capacities.budget),
        }
    }

    /// Creates an empty report with one capacity shared by every family.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self::with_capacity(TelemetryCapacity::uniform(capacity))
    }

    /// Returns the configured per-family capacities.
    #[must_use]
    pub const fn capacities(&self) -> TelemetryCapacity {
        self.capacities
    }

    /// Returns aggregated residency metrics in ascending state-key order.
    #[must_use]
    pub fn residency(&self) -> &[ResidencyMetric] {
        &self.residency
    }

    /// Returns aggregated migration metrics in ascending transition-key order.
    #[must_use]
    pub fn migrations(&self) -> &[MigrationMetric] {
        &self.migrations
    }

    /// Returns aggregated guard metrics in ascending guard-key order.
    #[must_use]
    pub fn guards(&self) -> &[GuardMetric] {
        &self.guards
    }

    /// Returns aggregated LOD metrics in ascending level order.
    #[must_use]
    pub fn lod(&self) -> &[LodMetric] {
        &self.lod
    }

    /// Returns aggregated budget metrics in ascending budget-key order.
    #[must_use]
    pub fn budgets(&self) -> &[BudgetMetric] {
        &self.budgets
    }

    /// Returns the total number of distinct metric keys in the report.
    #[must_use]
    pub fn len(&self) -> usize {
        self.residency.len()
            + self.migrations.len()
            + self.guards.len()
            + self.lod.len()
            + self.budgets.len()
    }

    /// Returns whether the report has no metric keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Removes all metrics while retaining the reserved storage.
    pub fn clear(&mut self) {
        self.residency.clear();
        self.migrations.clear();
        self.guards.clear();
        self.lod.clear();
        self.budgets.clear();
    }

    /// Merges another report without changing this report's capacities.
    pub fn merge(&mut self, other: &Self) -> Result<(), TelemetryError> {
        self.check_merge_capacity(other)?;
        self.check_merge_counters(other)?;
        self.merge_residency(other.residency())?;
        self.merge_migrations(other.migrations())?;
        self.merge_guards(other.guards())?;
        self.merge_lod(other.lod())?;
        self.merge_budgets(other.budgets())?;
        self.sort_all();
        Ok(())
    }

    fn check_merge_counters(&self, other: &Self) -> Result<(), TelemetryError> {
        for metric in other.residency() {
            if let Some(index) = find_residency(&self.residency, metric.state) {
                let current = self.residency[index];
                if current.samples.checked_add(metric.samples).is_none()
                    || current
                        .total_ticks
                        .checked_add(metric.total_ticks)
                        .is_none()
                {
                    return Err(TelemetryError::CounterOverflow {
                        metric: TelemetryMetric::Residency,
                        key: u64::from(metric.state),
                    });
                }
            }
        }
        for metric in other.migrations() {
            if let Some(index) = find_migration(&self.migrations, metric.transition)
                && self.migrations[index]
                    .count
                    .checked_add(metric.count)
                    .is_none()
            {
                return Err(TelemetryError::CounterOverflow {
                    metric: TelemetryMetric::Migration,
                    key: u64::from(metric.transition),
                });
            }
        }
        for metric in other.guards() {
            if let Some(index) = find_guard(&self.guards, metric.guard) {
                let current = self.guards[index];
                if current.evaluated.checked_add(metric.evaluated).is_none()
                    || current
                        .false_count
                        .checked_add(metric.false_count)
                        .is_none()
                {
                    return Err(TelemetryError::CounterOverflow {
                        metric: TelemetryMetric::Guard,
                        key: u64::from(metric.guard),
                    });
                }
            }
        }
        for metric in other.lod() {
            if let Some(index) = find_lod(&self.lod, metric.lod) {
                let current = self.lod[index];
                if current.samples.checked_add(metric.samples).is_none()
                    || current
                        .total_ticks
                        .checked_add(metric.total_ticks)
                        .is_none()
                {
                    return Err(TelemetryError::CounterOverflow {
                        metric: TelemetryMetric::Lod,
                        key: u64::from(metric.lod),
                    });
                }
            }
        }
        for metric in other.budgets() {
            if let Some(index) = find_budget(&self.budgets, metric.budget) {
                let current = self.budgets[index];
                if current.count.checked_add(metric.count).is_none()
                    || current.units.checked_add(metric.units).is_none()
                {
                    return Err(TelemetryError::CounterOverflow {
                        metric: TelemetryMetric::Budget,
                        key: u64::from(metric.budget),
                    });
                }
            }
        }
        Ok(())
    }

    fn check_merge_capacity(&self, other: &Self) -> Result<(), TelemetryError> {
        check_capacity(
            TelemetryMetric::Residency,
            self.residency.len(),
            self.capacities.residency,
            missing_residencies(&self.residency, &other.residency),
        )?;
        check_capacity(
            TelemetryMetric::Migration,
            self.migrations.len(),
            self.capacities.migration,
            missing_migrations(&self.migrations, &other.migrations),
        )?;
        check_capacity(
            TelemetryMetric::Guard,
            self.guards.len(),
            self.capacities.guard,
            missing_guards(&self.guards, &other.guards),
        )?;
        check_capacity(
            TelemetryMetric::Lod,
            self.lod.len(),
            self.capacities.lod,
            missing_lod(&self.lod, &other.lod),
        )?;
        check_capacity(
            TelemetryMetric::Budget,
            self.budgets.len(),
            self.capacities.budget,
            missing_budgets(&self.budgets, &other.budgets),
        )?;
        Ok(())
    }

    fn merge_residency(&mut self, metrics: &[ResidencyMetric]) -> Result<(), TelemetryError> {
        for metric in metrics {
            if let Some(index) = find_residency(&self.residency, metric.state) {
                let current = self.residency[index];
                let merged = merge_residency_metric(current, *metric)?;
                self.residency[index] = merged;
            } else {
                self.residency.push(*metric);
            }
        }
        Ok(())
    }

    fn merge_migrations(&mut self, metrics: &[MigrationMetric]) -> Result<(), TelemetryError> {
        for metric in metrics {
            if let Some(index) = find_migration(&self.migrations, metric.transition) {
                let count = self.migrations[index]
                    .count
                    .checked_add(metric.count)
                    .ok_or(TelemetryError::CounterOverflow {
                        metric: TelemetryMetric::Migration,
                        key: u64::from(metric.transition),
                    })?;
                self.migrations[index].count = count;
            } else {
                self.migrations.push(*metric);
            }
        }
        Ok(())
    }

    fn merge_guards(&mut self, metrics: &[GuardMetric]) -> Result<(), TelemetryError> {
        for metric in metrics {
            if let Some(index) = find_guard(&self.guards, metric.guard) {
                let current = self.guards[index];
                let evaluated = current.evaluated.checked_add(metric.evaluated);
                let false_count = current.false_count.checked_add(metric.false_count);
                let (Some(evaluated), Some(false_count)) = (evaluated, false_count) else {
                    return Err(TelemetryError::CounterOverflow {
                        metric: TelemetryMetric::Guard,
                        key: u64::from(metric.guard),
                    });
                };
                self.guards[index] = GuardMetric {
                    evaluated,
                    false_count,
                    ..current
                };
            } else {
                self.guards.push(*metric);
            }
        }
        Ok(())
    }

    fn merge_lod(&mut self, metrics: &[LodMetric]) -> Result<(), TelemetryError> {
        for metric in metrics {
            if let Some(index) = find_lod(&self.lod, metric.lod) {
                let current = self.lod[index];
                let samples = current.samples.checked_add(metric.samples);
                let total_ticks = current.total_ticks.checked_add(metric.total_ticks);
                let (Some(samples), Some(total_ticks)) = (samples, total_ticks) else {
                    return Err(TelemetryError::CounterOverflow {
                        metric: TelemetryMetric::Lod,
                        key: u64::from(metric.lod),
                    });
                };
                self.lod[index] = LodMetric {
                    samples,
                    total_ticks,
                    ..current
                };
            } else {
                self.lod.push(*metric);
            }
        }
        Ok(())
    }

    fn merge_budgets(&mut self, metrics: &[BudgetMetric]) -> Result<(), TelemetryError> {
        for metric in metrics {
            if let Some(index) = find_budget(&self.budgets, metric.budget) {
                let current = self.budgets[index];
                let count = current.count.checked_add(metric.count);
                let units = current.units.checked_add(metric.units);
                let (Some(count), Some(units)) = (count, units) else {
                    return Err(TelemetryError::CounterOverflow {
                        metric: TelemetryMetric::Budget,
                        key: u64::from(metric.budget),
                    });
                };
                self.budgets[index] = BudgetMetric {
                    count,
                    units,
                    ..current
                };
            } else {
                self.budgets.push(*metric);
            }
        }
        Ok(())
    }

    fn sort_all(&mut self) {
        self.residency.sort_unstable_by_key(|metric| metric.state);
        self.migrations
            .sort_unstable_by_key(|metric| metric.transition);
        self.guards.sort_unstable_by_key(|metric| metric.guard);
        self.lod.sort_unstable_by_key(|metric| metric.lod);
        self.budgets.sort_unstable_by_key(|metric| metric.budget);
    }
}

/// A bounded hot-path collector for StateTree runtime telemetry.
pub struct TelemetryCollector {
    report: TelemetryReport,
}

impl TelemetryCollector {
    /// Creates a collector with one distinct-key capacity shared by all
    /// metric families.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self::with_capacity(TelemetryCapacity::uniform(capacity))
    }

    /// Creates a collector and reserves every metric vector up front.
    #[must_use]
    pub fn with_capacity(capacities: TelemetryCapacity) -> Self {
        Self {
            report: TelemetryReport::with_capacity(capacities),
        }
    }

    /// Returns the configured per-family capacities.
    #[must_use]
    pub const fn capacities(&self) -> TelemetryCapacity {
        self.report.capacities
    }

    /// Returns the number of distinct keys currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.report.len()
    }

    /// Returns whether no metric key has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.report.is_empty()
    }

    /// Records one completed state residency in host-defined ticks.
    pub fn record_residency(&mut self, state: u32, ticks: u64) -> Result<(), TelemetryError> {
        if let Some(index) = find_residency(&self.report.residency, state) {
            let current = self.report.residency[index];
            let samples = current.samples.checked_add(1);
            let total_ticks = current.total_ticks.checked_add(ticks);
            let (Some(samples), Some(total_ticks)) = (samples, total_ticks) else {
                return Err(TelemetryError::CounterOverflow {
                    metric: TelemetryMetric::Residency,
                    key: u64::from(state),
                });
            };
            self.report.residency[index] = ResidencyMetric {
                samples,
                total_ticks,
                min_ticks: current.min_ticks.min(ticks),
                max_ticks: current.max_ticks.max(ticks),
                ..current
            };
            return Ok(());
        }
        ensure_capacity(
            TelemetryMetric::Residency,
            self.report.residency.len(),
            self.report.capacities.residency,
        )?;
        self.report.residency.push(ResidencyMetric {
            state,
            samples: 1,
            total_ticks: ticks,
            min_ticks: ticks,
            max_ticks: ticks,
        });
        Ok(())
    }

    /// Records one applied migration for a transition.
    pub fn record_migration(&mut self, transition: u32) -> Result<(), TelemetryError> {
        if let Some(index) = find_migration(&self.report.migrations, transition) {
            let count = self.report.migrations[index].count.checked_add(1).ok_or(
                TelemetryError::CounterOverflow {
                    metric: TelemetryMetric::Migration,
                    key: u64::from(transition),
                },
            )?;
            self.report.migrations[index].count = count;
            return Ok(());
        }
        ensure_capacity(
            TelemetryMetric::Migration,
            self.report.migrations.len(),
            self.report.capacities.migration,
        )?;
        self.report.migrations.push(MigrationMetric {
            transition,
            count: 1,
        });
        Ok(())
    }

    /// Records one guard evaluation that returned true.
    pub fn record_guard_evaluated(&mut self, guard: u32) -> Result<(), TelemetryError> {
        self.record_guard_result(guard, true)
    }

    /// Records one guard evaluation that returned false.
    pub fn record_guard_false(&mut self, guard: u32) -> Result<(), TelemetryError> {
        self.record_guard_result(guard, false)
    }

    /// Records one guard evaluation and its Boolean result.
    pub fn record_guard_result(&mut self, guard: u32, passed: bool) -> Result<(), TelemetryError> {
        if let Some(index) = find_guard(&self.report.guards, guard) {
            let current = self.report.guards[index];
            let evaluated = current.evaluated.checked_add(1);
            let false_count = current.false_count.checked_add(u64::from(!passed));
            let (Some(evaluated), Some(false_count)) = (evaluated, false_count) else {
                return Err(TelemetryError::CounterOverflow {
                    metric: TelemetryMetric::Guard,
                    key: u64::from(guard),
                });
            };
            self.report.guards[index] = GuardMetric {
                evaluated,
                false_count,
                ..current
            };
            return Ok(());
        }
        ensure_capacity(
            TelemetryMetric::Guard,
            self.report.guards.len(),
            self.report.capacities.guard,
        )?;
        self.report.guards.push(GuardMetric {
            guard,
            evaluated: 1,
            false_count: u64::from(!passed),
        });
        Ok(())
    }

    /// Records one observation of a LOD level.
    pub fn record_lod(&mut self, lod: u8) -> Result<(), TelemetryError> {
        self.record_lod_ticks(lod, 0)
    }

    /// Records one LOD observation with its host-defined duration.
    pub fn record_lod_ticks(&mut self, lod: u8, ticks: u64) -> Result<(), TelemetryError> {
        if let Some(index) = find_lod(&self.report.lod, lod) {
            let current = self.report.lod[index];
            let samples = current.samples.checked_add(1);
            let total_ticks = current.total_ticks.checked_add(ticks);
            let (Some(samples), Some(total_ticks)) = (samples, total_ticks) else {
                return Err(TelemetryError::CounterOverflow {
                    metric: TelemetryMetric::Lod,
                    key: u64::from(lod),
                });
            };
            self.report.lod[index] = LodMetric {
                samples,
                total_ticks,
                ..current
            };
            return Ok(());
        }
        ensure_capacity(
            TelemetryMetric::Lod,
            self.report.lod.len(),
            self.report.capacities.lod,
        )?;
        self.report.lod.push(LodMetric {
            lod,
            samples: 1,
            total_ticks: ticks,
        });
        Ok(())
    }

    /// Records one budget accounting event with one host-defined unit.
    pub fn record_budget(&mut self, budget: u32) -> Result<(), TelemetryError> {
        self.record_budget_units(budget, 1)
    }

    /// Records one budget accounting event and its host-defined unit count.
    pub fn record_budget_units(&mut self, budget: u32, units: u64) -> Result<(), TelemetryError> {
        if let Some(index) = find_budget(&self.report.budgets, budget) {
            let current = self.report.budgets[index];
            let count = current.count.checked_add(1);
            let total_units = current.units.checked_add(units);
            let (Some(count), Some(total_units)) = (count, total_units) else {
                return Err(TelemetryError::CounterOverflow {
                    metric: TelemetryMetric::Budget,
                    key: u64::from(budget),
                });
            };
            self.report.budgets[index] = BudgetMetric {
                count,
                units: total_units,
                ..current
            };
            return Ok(());
        }
        ensure_capacity(
            TelemetryMetric::Budget,
            self.report.budgets.len(),
            self.report.capacities.budget,
        )?;
        self.report.budgets.push(BudgetMetric {
            budget,
            count: 1,
            units,
        });
        Ok(())
    }

    /// Removes all metrics while retaining all reserved storage.
    pub fn clear(&mut self) {
        self.report.clear();
    }

    /// Merges an already aggregated report into this collector.
    pub fn merge(&mut self, report: &TelemetryReport) -> Result<(), TelemetryError> {
        self.report.merge(report)
    }

    /// Creates a deterministic report sorted by every metric's stable key.
    #[must_use]
    pub fn report(&self) -> TelemetryReport {
        let mut report = self.report.clone();
        report.sort_all();
        report
    }
}

fn ensure_capacity(
    metric: TelemetryMetric,
    len: usize,
    capacity: usize,
) -> Result<(), TelemetryError> {
    if len < capacity {
        Ok(())
    } else {
        Err(TelemetryError::CapacityExceeded { metric, capacity })
    }
}

fn check_capacity(
    metric: TelemetryMetric,
    len: usize,
    capacity: usize,
    missing: usize,
) -> Result<(), TelemetryError> {
    let required = len.saturating_add(missing);
    if required <= capacity {
        Ok(())
    } else {
        Err(TelemetryError::CapacityExceeded { metric, capacity })
    }
}

fn find_residency(metrics: &[ResidencyMetric], state: u32) -> Option<usize> {
    metrics.iter().position(|metric| metric.state == state)
}

fn find_migration(metrics: &[MigrationMetric], transition: u32) -> Option<usize> {
    metrics
        .iter()
        .position(|metric| metric.transition == transition)
}

fn find_guard(metrics: &[GuardMetric], guard: u32) -> Option<usize> {
    metrics.iter().position(|metric| metric.guard == guard)
}

fn find_lod(metrics: &[LodMetric], lod: u8) -> Option<usize> {
    metrics.iter().position(|metric| metric.lod == lod)
}

fn find_budget(metrics: &[BudgetMetric], budget: u32) -> Option<usize> {
    metrics.iter().position(|metric| metric.budget == budget)
}

fn missing_residencies(current: &[ResidencyMetric], other: &[ResidencyMetric]) -> usize {
    other
        .iter()
        .filter(|metric| find_residency(current, metric.state).is_none())
        .count()
}

fn missing_migrations(current: &[MigrationMetric], other: &[MigrationMetric]) -> usize {
    other
        .iter()
        .filter(|metric| find_migration(current, metric.transition).is_none())
        .count()
}

fn missing_guards(current: &[GuardMetric], other: &[GuardMetric]) -> usize {
    other
        .iter()
        .filter(|metric| find_guard(current, metric.guard).is_none())
        .count()
}

fn missing_lod(current: &[LodMetric], other: &[LodMetric]) -> usize {
    other
        .iter()
        .filter(|metric| find_lod(current, metric.lod).is_none())
        .count()
}

fn missing_budgets(current: &[BudgetMetric], other: &[BudgetMetric]) -> usize {
    other
        .iter()
        .filter(|metric| find_budget(current, metric.budget).is_none())
        .count()
}

fn merge_residency_metric(
    current: ResidencyMetric,
    other: ResidencyMetric,
) -> Result<ResidencyMetric, TelemetryError> {
    let samples = current.samples.checked_add(other.samples);
    let total_ticks = current.total_ticks.checked_add(other.total_ticks);
    let (Some(samples), Some(total_ticks)) = (samples, total_ticks) else {
        return Err(TelemetryError::CounterOverflow {
            metric: TelemetryMetric::Residency,
            key: u64::from(other.state),
        });
    };
    Ok(ResidencyMetric {
        samples,
        total_ticks,
        min_ticks: current.min_ticks.min(other.min_ticks),
        max_ticks: current.max_ticks.max(other.max_ticks),
        ..current
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_all_families_and_reports_deterministically() {
        let mut collector =
            TelemetryCollector::with_capacity(TelemetryCapacity::new(4, 4, 4, 4, 4));
        collector.record_residency(8, 5).unwrap();
        collector.record_residency(8, 7).unwrap();
        collector.record_residency(3, 2).unwrap();
        collector.record_migration(9).unwrap();
        collector.record_migration(9).unwrap();
        collector.record_guard_evaluated(4).unwrap();
        collector.record_guard_false(4).unwrap();
        collector.record_lod_ticks(2, 10).unwrap();
        collector.record_lod(2).unwrap();
        collector.record_budget_units(7, 12).unwrap();
        collector.record_budget(7).unwrap();

        let report = collector.report();
        assert_eq!(report.residency()[0].state, 3);
        assert_eq!(report.residency()[1].state, 8);
        assert_eq!(report.residency()[1].samples, 2);
        assert_eq!(report.residency()[1].total_ticks, 12);
        assert_eq!(report.residency()[1].min_ticks, 5);
        assert_eq!(report.residency()[1].max_ticks, 7);
        assert_eq!(
            report.migrations(),
            &[MigrationMetric {
                transition: 9,
                count: 2
            }]
        );
        assert_eq!(
            report.guards(),
            &[GuardMetric {
                guard: 4,
                evaluated: 2,
                false_count: 1,
            }]
        );
        assert_eq!(
            report.lod(),
            &[LodMetric {
                lod: 2,
                samples: 2,
                total_ticks: 10,
            }]
        );
        assert_eq!(
            report.budgets(),
            &[BudgetMetric {
                budget: 7,
                count: 2,
                units: 13,
            }]
        );

        let mut reversed = TelemetryCollector::new(4);
        reversed.record_residency(3, 2).unwrap();
        reversed.record_residency(8, 5).unwrap();
        reversed.record_residency(8, 7).unwrap();
        reversed.record_migration(9).unwrap();
        reversed.record_migration(9).unwrap();
        reversed.record_guard_false(4).unwrap();
        reversed.record_guard_evaluated(4).unwrap();
        reversed.record_lod(2).unwrap();
        reversed.record_lod_ticks(2, 10).unwrap();
        reversed.record_budget(7).unwrap();
        reversed.record_budget_units(7, 12).unwrap();
        assert_eq!(report, reversed.report());
    }

    #[test]
    fn capacity_errors_do_not_add_a_key() {
        let mut collector = TelemetryCollector::new(1);
        collector.record_migration(1).unwrap();
        assert_eq!(
            collector.record_migration(2),
            Err(TelemetryError::CapacityExceeded {
                metric: TelemetryMetric::Migration,
                capacity: 1,
            })
        );
        assert_eq!(collector.report().migrations().len(), 1);
    }

    #[test]
    fn clear_retains_capacity_and_merge_is_bounded() {
        let mut collector =
            TelemetryCollector::with_capacity(TelemetryCapacity::new(2, 2, 2, 2, 2));
        collector.record_residency(1, 4).unwrap();
        let before = collector.report();
        collector.clear();
        assert!(collector.is_empty());
        collector.merge(&before).unwrap();
        assert_eq!(collector.report(), before);

        let mut extra = TelemetryCollector::new(1);
        extra.record_migration(1).unwrap();
        let mut target = TelemetryReport::new(1);
        target.merge(&extra.report()).unwrap();
        let mut second = TelemetryCollector::new(1);
        second.record_migration(2).unwrap();
        assert_eq!(
            target.merge(&second.report()),
            Err(TelemetryError::CapacityExceeded {
                metric: TelemetryMetric::Migration,
                capacity: 1,
            })
        );
        assert_eq!(
            target.migrations(),
            &[MigrationMetric {
                transition: 1,
                count: 1
            }]
        );
    }

    #[test]
    fn recording_with_reserved_capacity_keeps_vector_addresses() {
        let mut collector =
            TelemetryCollector::with_capacity(TelemetryCapacity::new(2, 2, 2, 2, 2));
        let residency_ptr = collector.report.residency.as_ptr();
        let migration_ptr = collector.report.migrations.as_ptr();
        let guard_ptr = collector.report.guards.as_ptr();
        let lod_ptr = collector.report.lod.as_ptr();
        let budget_ptr = collector.report.budgets.as_ptr();

        collector.record_residency(1, 1).unwrap();
        collector.record_residency(2, 2).unwrap();
        collector.record_migration(1).unwrap();
        collector.record_migration(2).unwrap();
        collector.record_guard_evaluated(1).unwrap();
        collector.record_guard_evaluated(2).unwrap();
        collector.record_lod(1).unwrap();
        collector.record_lod(2).unwrap();
        collector.record_budget(1).unwrap();
        collector.record_budget(2).unwrap();

        assert_eq!(collector.report.residency.as_ptr(), residency_ptr);
        assert_eq!(collector.report.migrations.as_ptr(), migration_ptr);
        assert_eq!(collector.report.guards.as_ptr(), guard_ptr);
        assert_eq!(collector.report.lod.as_ptr(), lod_ptr);
        assert_eq!(collector.report.budgets.as_ptr(), budget_ptr);
    }

    #[test]
    fn counter_overflow_is_structured() {
        let mut collector = TelemetryCollector::new(1);
        collector.report.migrations.push(MigrationMetric {
            transition: 11,
            count: u64::MAX,
        });
        assert_eq!(
            collector.record_migration(11),
            Err(TelemetryError::CounterOverflow {
                metric: TelemetryMetric::Migration,
                key: 11,
            })
        );
    }

    #[test]
    fn merge_counter_overflow_leaves_target_unchanged() {
        let mut target = TelemetryReport::new(2);
        target.residency.push(ResidencyMetric {
            state: 1,
            samples: 1,
            total_ticks: 2,
            min_ticks: 2,
            max_ticks: 2,
        });
        target.migrations.push(MigrationMetric {
            transition: 3,
            count: 4,
        });
        let before = target.clone();

        let mut other = TelemetryReport::new(2);
        other.residency.push(ResidencyMetric {
            state: 1,
            samples: u64::MAX,
            total_ticks: 1,
            min_ticks: 1,
            max_ticks: 1,
        });
        other.migrations.push(MigrationMetric {
            transition: 3,
            count: 2,
        });

        assert_eq!(
            target.merge(&other),
            Err(TelemetryError::CounterOverflow {
                metric: TelemetryMetric::Residency,
                key: 1,
            })
        );
        assert_eq!(target, before);
    }
}
