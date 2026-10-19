//! Executable, deterministic acceptance metrics for lighting/GI validation.
//!
//! This module is intentionally self-contained so it can be exercised before
//! registration in the lighting module tree. A missing measurement is a
//! failure; N/A is only emitted from an explicit declaration with a reason.

use std::fmt::Write as _;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum HardwareClass {
    H1,
    H2,
    H3,
    H4,
}

impl HardwareClass {
    pub const ALL: [Self; 4] = [Self::H1, Self::H2, Self::H3, Self::H4];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::H1 => "H1",
            Self::H2 => "H2",
            Self::H3 => "H3",
            Self::H4 => "H4",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SceneClass {
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
}

impl SceneClass {
    pub const ALL: [Self; 6] = [Self::S1, Self::S2, Self::S3, Self::S4, Self::S5, Self::S6];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::S1 => "S1",
            Self::S2 => "S2",
            Self::S3 => "S3",
            Self::S4 => "S4",
            Self::S5 => "S5",
            Self::S6 => "S6",
        }
    }
}

/// Measurements are represented in their natural units: milliseconds,
/// bytes, bytes/second, rates, linear error, or dB as appropriate.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Metric {
    CpuFrameMs,
    GpuFrameMs,
    BandwidthBytesPerSecond,
    ResidentBytes,
    IoFaultRate,
    VarianceRaw,
    VarianceAccumulated,
    VariancePostDenoise,
    BiasBoundTightness,
    TemporalInstability,
    Psnr,
    EnergyError,
    HitchFrameMs,
    DisplayableOutput,
}

impl Metric {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CpuFrameMs => "cpu_frame_ms",
            Self::GpuFrameMs => "gpu_frame_ms",
            Self::BandwidthBytesPerSecond => "bandwidth_bytes_per_second",
            Self::ResidentBytes => "resident_bytes",
            Self::IoFaultRate => "io_fault_rate",
            Self::VarianceRaw => "variance_raw",
            Self::VarianceAccumulated => "variance_accumulated",
            Self::VariancePostDenoise => "variance_post_denoise",
            Self::BiasBoundTightness => "bias_bound_tightness",
            Self::TemporalInstability => "temporal_instability",
            Self::Psnr => "psnr_db",
            Self::EnergyError => "energy_error",
            Self::HitchFrameMs => "hitch_frame_ms",
            Self::DisplayableOutput => "displayable_output",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Statistic {
    P50,
    P95,
    P99,
    P999,
    Min,
    Max,
    Mean,
}

impl Statistic {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::P50 => "P50",
            Self::P95 => "P95",
            Self::P99 => "P99",
            Self::P999 => "P99.9",
            Self::Min => "min",
            Self::Max => "max",
            Self::Mean => "mean",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Comparison {
    AtMost,
    AtLeast,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Distribution {
    pub count: usize,
    pub min: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
    pub max: f64,
    pub mean: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SampleError {
    Empty,
    NonFinite { index: usize },
}

impl Distribution {
    /// Uses nearest-rank quantiles (ceil(p*n), one-based), giving stable and
    /// conservative tail estimates without interpolation or external crates.
    pub fn from_samples(samples: &[f64]) -> Result<Self, SampleError> {
        if samples.is_empty() {
            return Err(SampleError::Empty);
        }
        for (index, value) in samples.iter().enumerate() {
            if !value.is_finite() {
                return Err(SampleError::NonFinite { index });
            }
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let quantile = |numerator: usize, denominator: usize| {
            let rank = sorted
                .len()
                .saturating_mul(numerator)
                .div_ceil(denominator)
                .max(1);
            sorted[rank.min(sorted.len()) - 1]
        };
        // Compensated summation reduces order-dependent error; sorting first
        // also makes the mean bit-stable for the same multiset of samples.
        let mut sum = 0.0;
        let mut correction = 0.0;
        for value in &sorted {
            let adjusted = *value - correction;
            let next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
        }
        Ok(Self {
            count: sorted.len(),
            min: sorted[0],
            p50: quantile(50, 100),
            p95: quantile(95, 100),
            p99: quantile(99, 100),
            p999: quantile(999, 1000),
            max: *sorted.last().expect("non-empty checked above"),
            mean: sum / sorted.len() as f64,
        })
    }

    pub fn value(&self, statistic: Statistic) -> f64 {
        match statistic {
            Statistic::P50 => self.p50,
            Statistic::P95 => self.p95,
            Statistic::P99 => self.p99,
            Statistic::P999 => self.p999,
            Statistic::Min => self.min,
            Statistic::Max => self.max,
            Statistic::Mean => self.mean,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Case {
    pub hardware: HardwareClass,
    pub scene: SceneClass,
    pub metric: Metric,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    pub case: Case,
    pub samples: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GateRule {
    pub case: Case,
    pub statistic: Statistic,
    pub comparison: Comparison,
    pub threshold: f64,
}

impl GateRule {
    pub fn validate(&self) -> Result<(), PlanError> {
        if !self.threshold.is_finite() {
            return Err(PlanError::NonFiniteThreshold(self.case));
        }
        if self.case.metric == Metric::DisplayableOutput
            && !matches!(
                self.statistic,
                Statistic::Min | Statistic::P50 | Statistic::Mean
            )
        {
            return Err(PlanError::InvalidBooleanStatistic(self.case));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NotApplicable {
    pub case: Case,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AcceptancePlan {
    pub rules: Vec<GateRule>,
    pub not_applicable: Vec<NotApplicable>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanError {
    NonFiniteThreshold(Case),
    InvalidBooleanStatistic(Case),
    EmptyNotApplicableReason(Case),
    DuplicateCase(Case),
}

impl AcceptancePlan {
    pub fn validate(&self) -> Result<(), PlanError> {
        let mut seen = Vec::new();
        for rule in &self.rules {
            rule.validate()?;
            if seen.contains(&rule.case) {
                return Err(PlanError::DuplicateCase(rule.case));
            }
            seen.push(rule.case);
        }
        for na in &self.not_applicable {
            if na.reason.trim().is_empty() {
                return Err(PlanError::EmptyNotApplicableReason(na.case));
            }
            if seen.contains(&na.case) {
                return Err(PlanError::DuplicateCase(na.case));
            }
            seen.push(na.case);
        }
        Ok(())
    }

    pub fn evaluate(&self, measurements: &[Measurement]) -> Result<Report, PlanError> {
        self.validate()?;
        let mut results = Vec::new();
        let mut declared_cases: Vec<Case> = Vec::new();
        for rule in &self.rules {
            declared_cases.push(rule.case);
            let matching: Vec<_> = measurements
                .iter()
                .filter(|m| m.case == rule.case)
                .collect();
            if matching.len() > 1 {
                return Err(PlanError::DuplicateCase(rule.case));
            }
            let Some(measurement) = matching.first() else {
                results.push(CheckResult::Missing { case: rule.case });
                continue;
            };
            let distribution = match Distribution::from_samples(&measurement.samples) {
                Ok(distribution) => distribution,
                Err(error) => {
                    results.push(CheckResult::InvalidSamples {
                        case: rule.case,
                        error,
                    });
                    continue;
                }
            };
            let observed = distribution.value(rule.statistic);
            let passed = match rule.comparison {
                Comparison::AtMost => observed <= rule.threshold,
                Comparison::AtLeast => observed >= rule.threshold,
            };
            results.push(CheckResult::Measured {
                case: rule.case,
                statistic: rule.statistic,
                comparison: rule.comparison,
                observed,
                threshold: rule.threshold,
                passed,
                distribution,
            });
        }
        for na in &self.not_applicable {
            declared_cases.push(na.case);
            results.push(CheckResult::NotApplicable {
                case: na.case,
                reason: na.reason.clone(),
            });
        }
        for measurement in measurements {
            if !declared_cases.contains(&measurement.case) {
                results.push(CheckResult::Uncovered {
                    case: measurement.case,
                });
            }
        }
        results.sort_by_key(CheckResult::case);
        Ok(Report::new(results))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CheckResult {
    Measured {
        case: Case,
        statistic: Statistic,
        comparison: Comparison,
        observed: f64,
        threshold: f64,
        passed: bool,
        distribution: Distribution,
    },
    Missing {
        case: Case,
    },
    InvalidSamples {
        case: Case,
        error: SampleError,
    },
    Uncovered {
        case: Case,
    },
    NotApplicable {
        case: Case,
        reason: String,
    },
}

impl CheckResult {
    pub fn case(&self) -> Case {
        match self {
            Self::Measured { case, .. }
            | Self::Missing { case }
            | Self::InvalidSamples { case, .. }
            | Self::Uncovered { case }
            | Self::NotApplicable { case, .. } => *case,
        }
    }

    pub const fn passed(&self) -> Option<bool> {
        match self {
            Self::Measured { passed, .. } => Some(*passed),
            Self::NotApplicable { .. } => None,
            Self::Missing { .. } | Self::InvalidSamples { .. } | Self::Uncovered { .. } => {
                Some(false)
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OverallStatus {
    Pass,
    Fail,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub status: OverallStatus,
    pub pass_count: usize,
    pub fail_count: usize,
    pub not_applicable_count: usize,
    pub results: Vec<CheckResult>,
    pub summary: String,
    pub digest: [u8; 16],
}

impl Report {
    fn new(results: Vec<CheckResult>) -> Self {
        let pass_count = results.iter().filter(|r| r.passed() == Some(true)).count();
        let fail_count = results.iter().filter(|r| r.passed() == Some(false)).count();
        let not_applicable_count = results.iter().filter(|r| r.passed().is_none()).count();
        let status = if fail_count == 0 && pass_count > 0 {
            OverallStatus::Pass
        } else {
            OverallStatus::Fail
        };
        let summary = render_summary(
            &results,
            &status,
            pass_count,
            fail_count,
            not_applicable_count,
        );
        let digest = digest128(summary.as_bytes());
        Self {
            status,
            pass_count,
            fail_count,
            not_applicable_count,
            results,
            summary,
            digest,
        }
    }
}

fn render_summary(
    results: &[CheckResult],
    status: &OverallStatus,
    passes: usize,
    failures: usize,
    nas: usize,
) -> String {
    let mut out = String::new();
    let status_text = match status {
        OverallStatus::Pass => "PASS",
        OverallStatus::Fail => "FAIL",
    };
    let _ = writeln!(
        out,
        "lighting-metrics-v1 status={status_text} pass={passes} fail={failures} na={nas}"
    );
    for result in results {
        let case = result.case();
        let prefix = format!(
            "{} {} {}",
            case.hardware.as_str(),
            case.scene.as_str(),
            case.metric.as_str()
        );
        match result {
            CheckResult::Measured {
                statistic,
                comparison,
                observed,
                threshold,
                passed,
                distribution,
                ..
            } => {
                let verdict = if *passed { "PASS" } else { "FAIL" };
                let op = match comparison {
                    Comparison::AtMost => "<=",
                    Comparison::AtLeast => ">=",
                };
                let _ = writeln!(out, "{prefix} {verdict} {}={:.9} {} {:.9} n={} p50={:.9} p95={:.9} p99={:.9} p999={:.9} min={:.9} max={:.9} mean={:.9}", statistic.as_str(), observed, op, threshold, distribution.count, distribution.p50, distribution.p95, distribution.p99, distribution.p999, distribution.min, distribution.max, distribution.mean);
            }
            CheckResult::Missing { .. } => {
                let _ = writeln!(out, "{prefix} FAIL missing_measurement");
            }
            CheckResult::InvalidSamples { error, .. } => {
                let _ = writeln!(out, "{prefix} FAIL invalid_samples={error:?}");
            }
            CheckResult::Uncovered { .. } => {
                let _ = writeln!(out, "{prefix} FAIL uncovered_case");
            }
            CheckResult::NotApplicable { reason, .. } => {
                let _ = writeln!(out, "{prefix} N/A reason={}", escape_summary(reason));
            }
        }
    }
    out
}

fn escape_summary(input: &str) -> String {
    input
        .chars()
        .flat_map(|ch| match ch {
            '\n' => "\\n".chars().collect::<Vec<_>>(),
            '\r' => "\\r".chars().collect(),
            _ => vec![ch],
        })
        .collect()
}

/// Stable, platform-independent 128-bit FNV-1a-derived report fingerprint.
/// This is an identity checksum, not a cryptographic integrity primitive.
fn digest128(bytes: &[u8]) -> [u8; 16] {
    let mut lanes = [0x6c62272e07bb0142u64, 0x62b821756295c58du64];
    for (index, byte) in bytes.iter().enumerate() {
        let lane = index & 1;
        lanes[lane] ^= u64::from(*byte);
        lanes[lane] = lanes[lane].wrapping_mul(0x00000100000001b3);
        lanes[lane] ^= lanes[1 - lane].rotate_left(17);
    }
    let mut output = [0; 16];
    output[..8].copy_from_slice(&lanes[0].to_le_bytes());
    output[8..].copy_from_slice(&lanes[1].to_le_bytes());
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(metric: Metric) -> Case {
        Case {
            hardware: HardwareClass::H3,
            scene: SceneClass::S2,
            metric,
        }
    }

    fn rule(
        metric: Metric,
        statistic: Statistic,
        comparison: Comparison,
        threshold: f64,
    ) -> GateRule {
        GateRule {
            case: case(metric),
            statistic,
            comparison,
            threshold,
        }
    }

    #[test]
    fn exposes_all_hardware_and_scene_classes_in_stable_order() {
        assert_eq!(
            HardwareClass::ALL.map(HardwareClass::as_str),
            ["H1", "H2", "H3", "H4"]
        );
        assert_eq!(
            SceneClass::ALL.map(SceneClass::as_str),
            ["S1", "S2", "S3", "S4", "S5", "S6"]
        );
    }

    #[test]
    fn nearest_rank_quantiles_and_mean_are_exact_for_integer_samples() {
        let samples: Vec<_> = (1..=100).map(f64::from).collect();
        let d = Distribution::from_samples(&samples).unwrap();
        assert_eq!((d.p50, d.p95, d.p99, d.p999), (50.0, 95.0, 99.0, 100.0));
        assert_eq!((d.min, d.max, d.mean, d.count), (1.0, 100.0, 50.5, 100));
    }

    #[test]
    fn rejects_empty_and_non_finite_samples() {
        assert_eq!(Distribution::from_samples(&[]), Err(SampleError::Empty));
        assert_eq!(
            Distribution::from_samples(&[1.0, f64::NAN]),
            Err(SampleError::NonFinite { index: 1 })
        );
    }

    #[test]
    fn evaluates_less_than_and_greater_than_gates_including_equality() {
        let plan = AcceptancePlan {
            rules: vec![
                rule(Metric::GpuFrameMs, Statistic::P95, Comparison::AtMost, 10.0),
                rule(Metric::Psnr, Statistic::P50, Comparison::AtLeast, 30.0),
            ],
            not_applicable: vec![],
        };
        let report = plan
            .evaluate(&[
                Measurement {
                    case: case(Metric::GpuFrameMs),
                    samples: vec![8.0, 10.0],
                },
                Measurement {
                    case: case(Metric::Psnr),
                    samples: vec![30.0, 31.0],
                },
            ])
            .unwrap();
        assert_eq!(report.status, OverallStatus::Pass);
        assert_eq!(report.pass_count, 2);
        assert_eq!(report.fail_count, 0);
    }

    #[test]
    fn missing_or_empty_measurement_fails_instead_of_becoming_na_or_pass() {
        let plan = AcceptancePlan {
            rules: vec![rule(
                Metric::VariancePostDenoise,
                Statistic::P95,
                Comparison::AtMost,
                1.0,
            )],
            not_applicable: vec![],
        };
        let missing = plan.evaluate(&[]).unwrap();
        assert_eq!(missing.status, OverallStatus::Fail);
        assert_eq!(missing.fail_count, 1);
        let empty = plan
            .evaluate(&[Measurement {
                case: case(Metric::VariancePostDenoise),
                samples: vec![],
            }])
            .unwrap();
        assert_eq!(empty.fail_count, 1);
        assert!(matches!(
            empty.results[0],
            CheckResult::InvalidSamples {
                error: SampleError::Empty,
                ..
            }
        ));
    }

    #[test]
    fn an_unlisted_measured_case_is_reported_as_uncovered_and_fails() {
        let plan = AcceptancePlan::default();
        let extra = Measurement {
            case: Case {
                hardware: HardwareClass::H1,
                scene: SceneClass::S4,
                metric: Metric::ResidentBytes,
            },
            samples: vec![64.0],
        };
        let report = plan.evaluate(&[extra]).unwrap();
        assert_eq!(report.status, OverallStatus::Fail);
        assert!(matches!(report.results[0], CheckResult::Uncovered { .. }));
    }

    #[test]
    fn na_requires_a_reason_and_is_never_counted_as_pass() {
        let c = case(Metric::BiasBoundTightness);
        let invalid = AcceptancePlan {
            rules: vec![],
            not_applicable: vec![NotApplicable {
                case: c,
                reason: " \n".into(),
            }],
        };
        assert_eq!(
            invalid.validate(),
            Err(PlanError::EmptyNotApplicableReason(c))
        );
        let valid = AcceptancePlan {
            rules: vec![],
            not_applicable: vec![NotApplicable {
                case: c,
                reason: "H1 has no traced contribution".into(),
            }],
        };
        let report = valid.evaluate(&[]).unwrap();
        assert_eq!(report.status, OverallStatus::Fail);
        assert_eq!(
            (
                report.pass_count,
                report.fail_count,
                report.not_applicable_count
            ),
            (0, 0, 1)
        );
    }

    #[test]
    fn duplicate_cases_and_non_finite_thresholds_are_rejected() {
        let r = rule(
            Metric::EnergyError,
            Statistic::Max,
            Comparison::AtMost,
            0.005,
        );
        let duplicate = AcceptancePlan {
            rules: vec![r.clone(), r],
            not_applicable: vec![],
        };
        assert!(matches!(
            duplicate.validate(),
            Err(PlanError::DuplicateCase(_))
        ));
        let invalid = AcceptancePlan {
            rules: vec![rule(
                Metric::EnergyError,
                Statistic::Max,
                Comparison::AtMost,
                f64::INFINITY,
            )],
            not_applicable: vec![],
        };
        assert!(matches!(
            invalid.validate(),
            Err(PlanError::NonFiniteThreshold(_))
        ));
    }

    #[test]
    fn report_summary_and_digest_are_deterministic_and_escape_newlines() {
        let plan = AcceptancePlan {
            rules: vec![rule(
                Metric::TemporalInstability,
                Statistic::P99,
                Comparison::AtMost,
                0.1,
            )],
            not_applicable: vec![NotApplicable {
                case: case(Metric::BiasBoundTightness),
                reason: "not traced\nby design".into(),
            }],
        };
        let input = [Measurement {
            case: case(Metric::TemporalInstability),
            samples: vec![0.01, 0.03, 0.02],
        }];
        let a = plan.evaluate(&input).unwrap();
        let b = plan.evaluate(&input).unwrap();
        assert_eq!(a.summary, b.summary);
        assert_eq!(a.digest, b.digest);
        assert!(a.summary.contains("reason=not traced\\nby design"));
    }

    #[test]
    fn boolean_displayability_uses_minimum_to_require_every_frame() {
        let c = case(Metric::DisplayableOutput);
        let plan = AcceptancePlan {
            rules: vec![GateRule {
                case: c,
                statistic: Statistic::Min,
                comparison: Comparison::AtLeast,
                threshold: 1.0,
            }],
            not_applicable: vec![],
        };
        let report = plan
            .evaluate(&[Measurement {
                case: c,
                samples: vec![1.0, 1.0, 0.0],
            }])
            .unwrap();
        assert_eq!(report.fail_count, 1);
    }
}
