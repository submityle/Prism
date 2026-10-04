//! Adaptive sampled-reporting rate control (§24.6).
//!
//! High-frequency per-instance metrics cannot all be shipped to the aggregator:
//! a fleet of N instances each emitting thousands of frame-time samples per
//! second would saturate the collection path. The fix is *sampled reporting* —
//! each instance forwards only `1` of every `N` samples, trading resolution for
//! bandwidth. The twist §24.6 calls for is that the rate must be *adaptive*: an
//! instance the cluster view flagged as anomalous should temporarily report at
//! full resolution so the operator can diagnose it, while healthy instances
//! stay thin.
//!
//! This module owns the deterministic policy that turns a
//! [`ClusterFrametimeReport`] into a per-instance [`SamplingDecision`]: which
//! [`SampleRate`] each instance should use next window, whether it was boosted,
//! and why. The actual network transport and the per-sample decimation on the
//! edge are upper-layer wiring; the rate math, the boost policy, and the
//! bandwidth estimate here are pure `core`/`alloc` integer arithmetic —
//! deterministic, `no_std` + `alloc`, no `unsafe`.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use super::cluster::ClusterFrametimeReport;

/// A sampled-reporting rate: forward `1` of every [`one_in`](Self::one_in)
/// captured samples.
///
/// `one_in == 1` is full resolution (every sample reported). The value is
/// always at least `1`, so a rate never silently drops every sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SampleRate {
    /// Report one sample out of every `one_in` captured (clamped to `>= 1`).
    one_in: u32,
}

impl SampleRate {
    /// Full resolution: report every sample (`one_in == 1`).
    pub const FULL: Self = Self { one_in: 1 };

    /// A rate reporting `1` of every `one_in` samples. A zero `one_in` is
    /// clamped to `1` (full resolution), since reporting zero of every zero
    /// samples is meaningless.
    #[inline]
    #[must_use]
    pub const fn one_in(one_in: u32) -> Self {
        Self {
            one_in: if one_in == 0 { 1 } else { one_in },
        }
    }

    /// The divisor: one sample is reported out of every this many captured.
    #[inline]
    #[must_use]
    pub const fn divisor(self) -> u32 {
        self.one_in
    }

    /// Whether this rate forwards every sample.
    #[inline]
    #[must_use]
    pub const fn is_full(self) -> bool {
        self.one_in == 1
    }

    /// The reporting fraction in `(0, 1]`, i.e. `1.0 / one_in`.
    #[inline]
    #[must_use]
    pub fn fraction(self) -> f64 {
        1.0 / f64::from(self.one_in)
    }

    /// Whether the capture with zero-based sequence number `seq` is reported
    /// under this rate.
    ///
    /// A capture is reported when `seq` is a multiple of
    /// [`one_in`](Self::one_in), so sequence `0` is always reported and the
    /// pattern is deterministic and stateless.
    #[inline]
    #[must_use]
    pub const fn should_report(self, seq: u64) -> bool {
        (self.one_in as u64) != 0 && seq.is_multiple_of(self.one_in as u64)
    }

    /// Exact number of reports emitted for `captured` consecutive samples
    /// (sequences `0..captured`) under this rate.
    ///
    /// This is the count of multiples of [`one_in`](Self::one_in) in
    /// `0..captured`, i.e. `0` when `captured == 0`, else
    /// `(captured - 1) / one_in + 1`.
    #[inline]
    #[must_use]
    pub const fn expected_reports(self, captured: u64) -> u64 {
        if captured == 0 {
            0
        } else {
            (captured - 1) / (self.one_in as u64) + 1
        }
    }
}

impl Default for SampleRate {
    /// Full resolution.
    #[inline]
    fn default() -> Self {
        Self::FULL
    }
}

/// Why a [`SamplingDecision`] chose its rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleReason {
    /// The instance's configured baseline rate applied (no anomaly).
    Baseline,
    /// The instance was flagged anomalous by the cluster view and boosted to
    /// the policy's boosted rate for diagnosis.
    AnomalyBoost,
}

impl SampleReason {
    /// A short, stable label.
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::AnomalyBoost => "anomaly-boost",
        }
    }
}

/// The sampled-reporting rate chosen for one instance for the next window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplingDecision {
    /// The instance this decision targets.
    pub instance: String,
    /// The rate the instance should report at next window.
    pub rate: SampleRate,
    /// Whether this rate is a boost above the instance's baseline (because it
    /// was flagged anomalous).
    pub boosted: bool,
    /// Why this rate was chosen.
    pub reason: SampleReason,
}

/// Policy thresholds controlling the adaptive sampling controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplingPolicy {
    /// The default baseline rate applied to a healthy instance with no override.
    pub base: SampleRate,
    /// The rate an anomalous instance is boosted to (typically
    /// [`SampleRate::FULL`]).
    pub boosted: SampleRate,
}

impl Default for SamplingPolicy {
    /// Report `1` of every `16` samples at baseline, full resolution on anomaly.
    #[inline]
    fn default() -> Self {
        Self {
            base: SampleRate::one_in(16),
            boosted: SampleRate::FULL,
        }
    }
}

/// Turns a [`ClusterFrametimeReport`] into per-instance [`SamplingDecision`]s.
///
/// Each instance reports at the policy baseline (or a per-instance override)
/// unless the cluster view flagged it anomalous, in which case it is boosted to
/// the policy's boosted rate so the operator gets full resolution exactly where
/// it is needed. Per-instance baseline overrides let a known-noisy shard run
/// thinner (or a VIP shard run richer) without changing the fleet default.
///
/// The controller holds no clock and no network state; [`evaluate`](Self::evaluate)
/// is a pure function of the policy, the overrides, and the cluster report.
#[derive(Clone, Debug, Default)]
pub struct SamplingController {
    /// Policy thresholds.
    policy: SamplingPolicy,
    /// Per-instance baseline overrides, sorted by label for determinism.
    overrides: Vec<(String, SampleRate)>,
}

impl SamplingController {
    /// A controller with the default [`SamplingPolicy`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_policy(SamplingPolicy::default())
    }

    /// A controller with an explicit [`SamplingPolicy`].
    #[must_use]
    pub fn with_policy(policy: SamplingPolicy) -> Self {
        Self {
            policy,
            overrides: Vec::new(),
        }
    }

    /// The active policy.
    #[inline]
    #[must_use]
    pub fn policy(&self) -> SamplingPolicy {
        self.policy
    }

    /// Set a per-instance baseline rate override, replacing any prior override
    /// for the same label. Overrides are kept sorted by label.
    pub fn set_baseline(&mut self, instance: impl Into<String>, rate: SampleRate) {
        let instance = instance.into();
        match self
            .overrides
            .binary_search_by(|(label, _)| label.as_str().cmp(instance.as_str()))
        {
            Ok(existing) => self.overrides[existing].1 = rate,
            Err(insert_at) => self.overrides.insert(insert_at, (instance, rate)),
        }
    }

    /// The baseline rate for `instance`: its override, or the policy baseline.
    #[must_use]
    pub fn baseline_for(&self, instance: &str) -> SampleRate {
        self.overrides
            .binary_search_by(|(label, _)| label.as_str().cmp(instance))
            .map_or(self.policy.base, |found| self.overrides[found].1)
    }

    /// Decide each instance's next-window rate from the cluster report.
    ///
    /// Returns one [`SamplingDecision`] per instance present in the report, in
    /// the report's (label-sorted) order. An instance flagged anomalous is
    /// boosted to the policy's boosted rate; every other instance reports at
    /// its baseline. A boost that would be *slower* than baseline (a policy
    /// misconfiguration) is clamped so a boost never reduces resolution.
    #[must_use]
    pub fn evaluate(&self, report: &ClusterFrametimeReport) -> Vec<SamplingDecision> {
        let mut decisions: Vec<SamplingDecision> = Vec::with_capacity(report.instances.len());
        for summary in &report.instances {
            let baseline = self.baseline_for(&summary.label);
            if report.is_anomalous(&summary.label) {
                // Boost, but never below baseline resolution: pick the finer
                // (smaller divisor) of the configured boost and the baseline.
                let boosted = if self.policy.boosted.divisor() <= baseline.divisor() {
                    self.policy.boosted
                } else {
                    baseline
                };
                let is_boost = boosted.divisor() < baseline.divisor();
                decisions.push(SamplingDecision {
                    instance: summary.label.clone(),
                    rate: boosted,
                    boosted: is_boost,
                    reason: if is_boost {
                        SampleReason::AnomalyBoost
                    } else {
                        SampleReason::Baseline
                    },
                });
            } else {
                decisions.push(SamplingDecision {
                    instance: summary.label.clone(),
                    rate: baseline,
                    boosted: false,
                    reason: SampleReason::Baseline,
                });
            }
        }
        decisions
    }
}

/// Estimate the total reports emitted across a fleet for one window under a set
/// of [`SamplingDecision`]s, given each instance's captured sample count.
///
/// `captured` pairs an instance label with how many samples it captured this
/// window; a decision with no matching entry contributes `0`. Returns the
/// summed [`SampleRate::expected_reports`] — the deterministic bandwidth
/// estimate the operator uses to size the collection path.
#[must_use]
pub fn estimate_fleet_reports(
    decisions: &[SamplingDecision],
    captured: &[(&str, u64)],
) -> u64 {
    let mut total = 0u64;
    for decision in decisions {
        let count = captured
            .iter()
            .find(|(label, _)| *label == decision.instance.as_str())
            .map_or(0, |(_, count)| *count);
        total = total.saturating_add(decision.rate.expected_reports(count));
    }
    total
}
