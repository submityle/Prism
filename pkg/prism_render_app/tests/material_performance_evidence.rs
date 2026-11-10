//! Strict admission for externally collected material GPU performance evidence.
//!
//! This test does not benchmark the renderer and does not convert CPU wall time
//! into GPU time. The ignored gate validates a real-device artifact produced by
//! a timestamp-capable harness. That keeps missing RHI timestamp readback
//! visible instead of turning a source/compile check into performance evidence.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const SCHEMA: &str = "prism-material-perf-v1";
const REQUIRED_METRICS: [&str; 3] = [
    "frame_cpu_wall_ns",
    "frame_gpu_timestamp_ns",
    "material_upload_bytes",
];

#[derive(Debug, PartialEq, Eq)]
struct Evidence {
    fields: BTreeMap<String, String>,
    metrics: BTreeMap<String, Metric>,
}

#[derive(Debug, PartialEq, Eq)]
struct Metric {
    samples: u64,
    p50: u64,
    p95: u64,
    p99: u64,
}

fn parse(input: &str) -> Result<Evidence, String> {
    let mut fields = BTreeMap::new();
    let mut metrics = BTreeMap::new();

    for (line_index, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<_> = line.split(',').map(str::trim).collect();
        match parts.first().copied() {
            Some("metric") => {
                if parts.len() != 6 {
                    return Err(format!(
                        "line {}: metric requires name,samples,p50,p95,p99",
                        line_index + 1
                    ));
                }
                let name = parts[1];
                if name.is_empty() || metrics.contains_key(name) {
                    return Err(format!(
                        "line {}: metric name is empty or duplicated",
                        line_index + 1
                    ));
                }
                let values = [parts[2], parts[3], parts[4], parts[5]]
                    .into_iter()
                    .map(|value| {
                        value.parse::<u64>().map_err(|_| {
                            format!("line {}: invalid unsigned metric value", line_index + 1)
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                metrics.insert(
                    name.to_owned(),
                    Metric {
                        samples: values[0],
                        p50: values[1],
                        p95: values[2],
                        p99: values[3],
                    },
                );
            }
            Some(kind) if parts.len() == 2 && !kind.is_empty() && !parts[1].is_empty() => {
                if fields
                    .insert(kind.to_owned(), parts[1].to_owned())
                    .is_some()
                {
                    return Err(format!("line {}: field is duplicated", line_index + 1));
                }
            }
            _ => {
                return Err(format!(
                    "line {}: expected key,value or metric,name,samples,p50,p95,p99",
                    line_index + 1
                ));
            }
        }
    }

    let evidence = Evidence { fields, metrics };
    validate(&evidence)?;
    Ok(evidence)
}

fn validate(evidence: &Evidence) -> Result<(), String> {
    let field = |name: &str| {
        evidence
            .fields
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("missing required field {name}"))
    };

    if field("schema")? != SCHEMA {
        return Err(format!("schema must be {SCHEMA}"));
    }
    let commit = field("commit")?;
    if !(7..=64).contains(&commit.len()) || !commit.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("commit must be a hexadecimal revision id".into());
    }
    if field("scene")? != "default_native_material_frame" {
        return Err("scene must identify the default native material frame".into());
    }
    for name in ["backend", "adapter"] {
        if field(name)?.trim().is_empty() {
            return Err(format!("{name} must not be empty"));
        }
    }
    if field("timestamp_source")? != "gpu_timestamp_query" {
        return Err("timestamp_source must be gpu_timestamp_query".into());
    }

    let resolution = field("resolution")?;
    let (width, height) = resolution
        .split_once('x')
        .ok_or_else(|| "resolution must be WIDTHxHEIGHT".to_owned())?;
    if width
        .parse::<u32>()
        .ok()
        .filter(|value| *value > 0)
        .is_none()
        || height
            .parse::<u32>()
            .ok()
            .filter(|value| *value > 0)
            .is_none()
    {
        return Err("resolution must contain positive dimensions".into());
    }

    let warmup = field("warmup")?
        .parse::<u64>()
        .map_err(|_| "warmup must be an unsigned integer".to_owned())?;
    let sample_count = field("samples")?
        .parse::<u64>()
        .map_err(|_| "samples must be an unsigned integer".to_owned())?;
    if warmup == 0 || sample_count < 30 {
        return Err("warmup must be positive and samples must be at least 30".into());
    }

    for name in REQUIRED_METRICS {
        let metric = evidence
            .metrics
            .get(name)
            .ok_or_else(|| format!("missing required metric {name}"))?;
        if metric.samples != sample_count {
            return Err(format!(
                "metric {name} has {} samples, expected {sample_count}",
                metric.samples
            ));
        }
        if metric.p50 == 0 || metric.p50 > metric.p95 || metric.p95 > metric.p99 {
            return Err(format!("metric {name} must satisfy 0 < p50 <= p95 <= p99"));
        }
    }

    Ok(())
}

fn percentile(metric: &Metric, percentile: &str) -> u64 {
    match percentile {
        "p50" => metric.p50,
        "p95" => metric.p95,
        "p99" => metric.p99,
        _ => unreachable!("the gate only names supported percentiles"),
    }
}

fn assert_budgets(evidence: &Evidence) {
    for (metric_name, env_name, percentile_name) in [
        ("frame_cpu_wall_ns", "PRISM_MATERIAL_CPU_P50_MAX_NS", "p50"),
        ("frame_cpu_wall_ns", "PRISM_MATERIAL_CPU_P95_MAX_NS", "p95"),
        ("frame_cpu_wall_ns", "PRISM_MATERIAL_CPU_P99_MAX_NS", "p99"),
        (
            "frame_gpu_timestamp_ns",
            "PRISM_MATERIAL_GPU_P50_MAX_NS",
            "p50",
        ),
        (
            "frame_gpu_timestamp_ns",
            "PRISM_MATERIAL_GPU_P95_MAX_NS",
            "p95",
        ),
        (
            "frame_gpu_timestamp_ns",
            "PRISM_MATERIAL_GPU_P99_MAX_NS",
            "p99",
        ),
    ] {
        let budget = std::env::var(env_name)
            .unwrap_or_else(|_| panic!("set {env_name} to an agreed nanosecond budget"))
            .parse::<u64>()
            .unwrap_or_else(|_| panic!("{env_name} must be an unsigned nanosecond budget"));
        let actual = percentile(&evidence.metrics[metric_name], percentile_name);
        assert!(
            actual <= budget,
            "{metric_name} {percentile_name} is {actual} ns, above {env_name}={budget} ns"
        );
    }
}

fn complete_artifact() -> &'static str {
    "schema,prism-material-perf-v1
commit,0123456789abcdef0123456789abcdef01234567
scene,default_native_material_frame
backend,metal
adapter,physical-gpu
resolution,1280x720
warmup,30
samples,60
timestamp_source,gpu_timestamp_query
metric,frame_cpu_wall_ns,60,100000,150000,200000
metric,frame_gpu_timestamp_ns,60,80000,120000,160000
metric,material_upload_bytes,60,1024,2048,4096
"
}

#[test]
fn complete_material_performance_artifact_has_real_percentile_fields() {
    let evidence = parse(complete_artifact()).expect("complete artifact must be admissible");
    assert_eq!(evidence.fields["timestamp_source"], "gpu_timestamp_query");
    assert_eq!(evidence.metrics["frame_gpu_timestamp_ns"].p95, 120_000);
}

#[test]
fn cpu_wall_time_cannot_be_declared_gpu_timestamp_evidence() {
    let artifact = complete_artifact().replace(
        "timestamp_source,gpu_timestamp_query",
        "timestamp_source,cpu_wall_clock",
    );
    let error = parse(&artifact).expect_err("CPU wall time must not pass the GPU gate");
    assert!(
        error.contains("timestamp_source"),
        "unexpected error: {error}"
    );
}

#[test]
fn incomplete_percentiles_and_sample_counts_are_rejected() {
    let artifact = complete_artifact().replace(
        "metric,frame_gpu_timestamp_ns,60,80000,120000,160000",
        "metric,frame_gpu_timestamp_ns,29,80000,120000,160000",
    );
    let error = parse(&artifact).expect_err("incomplete GPU samples must fail the gate");
    assert!(error.contains("samples"), "unexpected error: {error}");
}

/// Run with a real-GPU artifact, for example:
///
/// `PRISM_REQUIRE_GPU=1 PRISM_MATERIAL_PERF_EVIDENCE=/abs/path/evidence.csv
/// PRISM_MATERIAL_CPU_P50_MAX_NS=... PRISM_MATERIAL_CPU_P95_MAX_NS=...
/// PRISM_MATERIAL_CPU_P99_MAX_NS=... PRISM_MATERIAL_GPU_P50_MAX_NS=...
/// PRISM_MATERIAL_GPU_P95_MAX_NS=... PRISM_MATERIAL_GPU_P99_MAX_NS=...
/// cargo test -p prism_render_app --test material_performance_evidence
/// material_gpu_performance_evidence_gate -- --ignored --nocapture`
#[test]
#[ignore = "requires a timestamped real-GPU material performance artifact"]
fn material_gpu_performance_evidence_gate() {
    assert_eq!(
        std::env::var("PRISM_REQUIRE_GPU").as_deref(),
        Ok("1"),
        "the performance gate requires PRISM_REQUIRE_GPU=1"
    );
    let path = std::env::var_os("PRISM_MATERIAL_PERF_EVIDENCE")
        .expect("set PRISM_MATERIAL_PERF_EVIDENCE to a collected artifact path");
    let path = Path::new(&path);
    let input =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let evidence = parse(&input)
        .unwrap_or_else(|error| panic!("invalid material performance evidence: {error}"));
    assert_budgets(&evidence);
    eprintln!(
        "accepted material performance evidence: adapter={}, backend={}, samples={}, frame_gpu_p95_ns={}",
        evidence.fields["adapter"],
        evidence.fields["backend"],
        evidence.fields["samples"],
        evidence.metrics["frame_gpu_timestamp_ns"].p95,
    );
}
