//! Bounded memory budgets, samples, and deterministic threshold alerts.

use std::collections::BTreeMap;

use crate::{validate_limit, validate_text, ProfilerError, ProfilerLimits};

/// A named memory budget.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryBudget {
    category: String,
    budget_bytes: u64,
    warning_per_mille: u16,
}

impl MemoryBudget {
    /// Creates a budget with a warning threshold expressed in per mille.
    pub fn new(
        category: impl Into<String>,
        budget_bytes: u64,
        warning_per_mille: u16,
    ) -> Result<Self, ProfilerError> {
        if budget_bytes == 0 || warning_per_mille > 1_000 {
            return Err(ProfilerError::InvalidRange {
                field: "memory budget",
            });
        }
        Ok(Self {
            category: category.into(),
            budget_bytes,
            warning_per_mille,
        })
    }

    /// Returns the category name.
    #[must_use]
    pub fn category(&self) -> &str {
        &self.category
    }

    /// Returns the hard budget in bytes.
    #[must_use]
    pub const fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Returns the warning threshold in per mille of the budget.
    #[must_use]
    pub const fn warning_per_mille(&self) -> u16 {
        self.warning_per_mille
    }

    fn warning_bytes(&self) -> Result<u64, ProfilerError> {
        self.budget_bytes
            .checked_mul(u64::from(self.warning_per_mille))
            .map(|value| value / 1_000)
            .ok_or(ProfilerError::ArithmeticOverflow)
    }
}

/// One normalized memory observation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemorySample {
    id: u64,
    timestamp_ns: u64,
    category: String,
    used_bytes: u64,
}

impl MemorySample {
    /// Creates a memory sample.
    #[must_use]
    pub fn new(id: u64, timestamp_ns: u64, category: impl Into<String>, used_bytes: u64) -> Self {
        Self {
            id,
            timestamp_ns,
            category: category.into(),
            used_bytes,
        }
    }

    /// Returns the stable sample identity.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Returns the sample timestamp.
    #[must_use]
    pub const fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }

    /// Returns the budget category.
    #[must_use]
    pub fn category(&self) -> &str {
        &self.category
    }

    /// Returns used bytes.
    #[must_use]
    pub const fn used_bytes(&self) -> u64 {
        self.used_bytes
    }
}

/// Severity of a memory budget alert.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MemorySeverity {
    /// Usage crossed the warning threshold but not the hard budget.
    Warning,
    /// Usage reached or crossed the hard budget.
    Exceeded,
}

/// A threshold result associated with one sample.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryAlert {
    sample_id: u64,
    category: String,
    used_bytes: u64,
    budget_bytes: u64,
    severity: MemorySeverity,
}

impl MemoryAlert {
    /// Returns the sample identity that caused the alert.
    #[must_use]
    pub const fn sample_id(&self) -> u64 {
        self.sample_id
    }

    /// Returns the category.
    #[must_use]
    pub fn category(&self) -> &str {
        &self.category
    }

    /// Returns observed bytes.
    #[must_use]
    pub const fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Returns the configured hard budget.
    #[must_use]
    pub const fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Returns alert severity.
    #[must_use]
    pub const fn severity(&self) -> MemorySeverity {
        self.severity
    }
}

/// Bounded category budgets and samples.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryProfile {
    limits: ProfilerLimits,
    budgets: BTreeMap<String, MemoryBudget>,
    samples: Vec<MemorySample>,
}

impl MemoryProfile {
    /// Creates an empty memory profile.
    #[must_use]
    pub const fn new(limits: ProfilerLimits) -> Self {
        Self {
            limits,
            budgets: BTreeMap::new(),
            samples: Vec::new(),
        }
    }

    /// Defines or replaces one category budget.
    pub fn define_budget(&mut self, budget: MemoryBudget) -> Result<(), ProfilerError> {
        validate_text(&budget.category, "memory category", &self.limits)?;
        if !self.budgets.contains_key(&budget.category) {
            validate_limit(
                self.budgets.len() + 1,
                self.limits.max_memory_budgets,
                "memory budgets",
            )?;
        }
        budget.warning_bytes()?;
        self.budgets.insert(budget.category.clone(), budget);
        Ok(())
    }

    /// Returns budgets in category order.
    pub fn budgets(&self) -> impl Iterator<Item = &MemoryBudget> {
        self.budgets.values()
    }

    /// Records a sample and returns an alert when a threshold is crossed.
    pub fn record(&mut self, sample: MemorySample) -> Result<Option<MemoryAlert>, ProfilerError> {
        validate_text(&sample.category, "memory category", &self.limits)?;
        if self.samples.iter().any(|existing| existing.id == sample.id) {
            return Err(ProfilerError::DuplicateId {
                resource: "memory sample",
                id: sample.id,
            });
        }
        let budget = self
            .budgets
            .get(&sample.category)
            .ok_or_else(|| ProfilerError::UnknownCategory(sample.category.clone()))?;
        validate_limit(
            self.samples.len() + 1,
            self.limits.max_memory_samples,
            "memory samples",
        )?;
        let warning_bytes = budget.warning_bytes()?;
        let alert = if sample.used_bytes >= budget.budget_bytes {
            Some(MemoryAlert {
                sample_id: sample.id,
                category: sample.category.clone(),
                used_bytes: sample.used_bytes,
                budget_bytes: budget.budget_bytes,
                severity: MemorySeverity::Exceeded,
            })
        } else if sample.used_bytes >= warning_bytes {
            Some(MemoryAlert {
                sample_id: sample.id,
                category: sample.category.clone(),
                used_bytes: sample.used_bytes,
                budget_bytes: budget.budget_bytes,
                severity: MemorySeverity::Warning,
            })
        } else {
            None
        };
        let position = self
            .samples
            .binary_search_by_key(
                &(sample.timestamp_ns, sample.category.clone(), sample.id),
                |existing| {
                    (
                        existing.timestamp_ns,
                        existing.category.clone(),
                        existing.id,
                    )
                },
            )
            .unwrap_or_else(|position| position);
        self.samples.insert(position, sample);
        Ok(alert)
    }

    /// Returns samples in timestamp/category/id order.
    #[must_use]
    pub fn samples(&self) -> &[MemorySample] {
        &self.samples
    }

    /// Recomputes all current alerts in sample order.
    pub fn alerts(&self) -> Result<Vec<MemoryAlert>, ProfilerError> {
        let mut alerts = Vec::new();
        for sample in &self.samples {
            if let Some(alert) = self.alert_for(sample)? {
                alerts.push(alert);
            }
        }
        Ok(alerts)
    }

    fn alert_for(&self, sample: &MemorySample) -> Result<Option<MemoryAlert>, ProfilerError> {
        let budget = self
            .budgets
            .get(&sample.category)
            .ok_or_else(|| ProfilerError::UnknownCategory(sample.category.clone()))?;
        let warning_bytes = budget.warning_bytes()?;
        let severity = if sample.used_bytes >= budget.budget_bytes {
            Some(MemorySeverity::Exceeded)
        } else if sample.used_bytes >= warning_bytes {
            Some(MemorySeverity::Warning)
        } else {
            None
        };
        Ok(severity.map(|severity| MemoryAlert {
            sample_id: sample.id,
            category: sample.category.clone(),
            used_bytes: sample.used_bytes,
            budget_bytes: budget.budget_bytes,
            severity,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_and_alerts_are_stably_ordered() {
        let mut profile = MemoryProfile::new(ProfilerLimits {
            max_memory_samples: 2,
            ..ProfilerLimits::default()
        });
        profile
            .define_budget(MemoryBudget::new("textures", 1_000, 800).unwrap())
            .unwrap();
        assert_eq!(
            profile
                .record(MemorySample::new(2, 20, "textures", 900))
                .unwrap()
                .unwrap()
                .severity(),
            MemorySeverity::Warning
        );
        assert!(profile
            .record(MemorySample::new(1, 10, "textures", 1_100))
            .unwrap()
            .is_some());
        assert_eq!(
            profile
                .samples()
                .iter()
                .map(MemorySample::id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        let alerts = profile.alerts().unwrap();
        assert_eq!(alerts[0].severity(), MemorySeverity::Exceeded);
        assert_eq!(alerts[1].severity(), MemorySeverity::Warning);
    }

    #[test]
    fn unknown_categories_and_limits_are_rejected() {
        let mut profile = MemoryProfile::new(ProfilerLimits {
            max_memory_samples: 1,
            ..ProfilerLimits::default()
        });
        assert!(matches!(
            profile.record(MemorySample::new(1, 0, "missing", 1)),
            Err(ProfilerError::UnknownCategory(_))
        ));
        profile
            .define_budget(MemoryBudget::new("mesh", 10, 900).unwrap())
            .unwrap();
        profile.record(MemorySample::new(1, 0, "mesh", 1)).unwrap();
        assert!(matches!(
            profile.record(MemorySample::new(2, 1, "mesh", 1)),
            Err(ProfilerError::LimitExceeded {
                resource: "memory samples",
                ..
            })
        ));
    }
}
