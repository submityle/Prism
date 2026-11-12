//! Strict admission for native material performance evidence.
//!
//! The gate accepts only raw per-sample records. Percentiles are recomputed
//! here, so a producer cannot provide a hand-written summary and call it GPU
//! evidence. The producer identity is also part of the contract: the generic
//! RHI timestamp probe is useful capability evidence, but it is not a native
//! material frame and must not pass this gate.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const SCHEMA: &str = "prism-material-perf-v2";
const REQUIRED_FIELDS: [&str; 10] = [
    "schema",
    "commit",
    "scene",
    "capture_scope",
    "producer",
    "backend",
    "adapter",
    "resolution",
    "timestamp_source",
    "timestamp_unit",
];
const PRODUCTION_SCOPE: &str = "production_native_material_frame";
const PRODUCTION_PRODUCER: &str = "prism_native_material_frame_capture_v1";

#[derive(Debug, PartialEq, Eq)]
struct Evidence {
    fields: BTreeMap<String, String>,
    samples: Vec<Sample>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sample {
    index: u64,
    cpu_wall_ns: u64,
    gpu_timestamp_ns: u64,
    material_upload_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Percentiles {
    p50: u64,
    p95: u64,
    p99: u64,
}

fn parse(input: &str) -> Result<Evidence, String> {
    let mut fields = BTreeMap::new();
    let mut samples = Vec::new();

    for (line_index, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<_> = line.split(',').map(str::trim).collect();
        match parts.first().copied() {
            Some("sample") => {
                if parts.len() != 5 {
                    return Err(format!(
                        "line {}: sample requires index,cpu_wall_ns,gpu_timestamp_ns,material_upload_bytes",
                        line_index + 1
                    ));
                }
                let values = [parts[1], parts[2], parts[3], parts[4]]
                    .into_iter()
                    .map(|value| {
                        value.parse::<u64>().map_err(|_| {
                            format!("line {}: invalid unsigned sample value", line_index + 1)
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                samples.push(Sample {
                    index: values[0],
                    cpu_wall_ns: values[1],
                    gpu_timestamp_ns: values[2],
                    material_upload_bytes: values[3],
                });
            }
            Some("metric") => {
                return Err(format!(
                    "line {}: percentile summaries are not accepted; provide raw sample lines",
                    line_index + 1
                ));
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
                    "line {}: expected key,value or sample,index,cpu_wall_ns,gpu_timestamp_ns,material_upload_bytes",
                    line_index + 1
                ));
            }
        }
    }

    let evidence = Evidence { fields, samples };
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

    for name in REQUIRED_FIELDS {
        field(name)?;
    }
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
    if field("capture_scope")? != PRODUCTION_SCOPE {
        return Err("capture_scope must identify a production native material frame".into());
    }
    if field("producer")? != PRODUCTION_PRODUCER {
        return Err("producer is not the production native material frame sampler".into());
    }
    for name in ["backend", "adapter"] {
        if field(name)?.trim().is_empty() || field(name)?.contains(',') {
            return Err(format!("{name} must be nonempty and comma-free"));
        }
    }
    if field("timestamp_source")? != "gpu_timestamp_query" {
        return Err("timestamp_source must be gpu_timestamp_query".into());
    }
    if field("timestamp_unit")? != "ns" {
        return Err("timestamp_unit must be ns".into());
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
    if evidence.samples.len() as u64 != sample_count {
        return Err(format!(
            "raw sample count is {}, expected {sample_count}",
            evidence.samples.len()
        ));
    }
    for (expected, sample) in evidence.samples.iter().enumerate() {
        if sample.index != expected as u64 {
            return Err(format!(
                "sample indices must be contiguous from zero; expected {}, got {}",
                expected, sample.index
            ));
        }
        if sample.cpu_wall_ns == 0 || sample.gpu_timestamp_ns == 0 {
            return Err(format!(
                "sample {} must contain positive CPU and GPU durations",
                sample.index
            ));
        }
    }

    Ok(())
}

fn percentiles(values: impl IntoIterator<Item = u64>) -> Percentiles {
    let mut values = values.into_iter().collect::<Vec<_>>();
    assert!(
        !values.is_empty(),
        "percentiles require at least one sample"
    );
    values.sort_unstable();
    let nearest_rank = |percent: usize| {
        let rank = (values.len() * percent).div_ceil(100);
        values[rank.saturating_sub(1)]
    };
    Percentiles {
        p50: nearest_rank(50),
        p95: nearest_rank(95),
        p99: nearest_rank(99),
    }
}

fn metric_percentiles(evidence: &Evidence, metric: &str) -> Percentiles {
    match metric {
        "frame_cpu_wall_ns" => {
            percentiles(evidence.samples.iter().map(|sample| sample.cpu_wall_ns))
        }
        "frame_gpu_timestamp_ns" => percentiles(
            evidence
                .samples
                .iter()
                .map(|sample| sample.gpu_timestamp_ns),
        ),
        "material_upload_bytes" => percentiles(
            evidence
                .samples
                .iter()
                .map(|sample| sample.material_upload_bytes),
        ),
        _ => unreachable!("the material gate only names known metrics"),
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
        let percentiles = metric_percentiles(evidence, metric_name);
        let actual = match percentile_name {
            "p50" => percentiles.p50,
            "p95" => percentiles.p95,
            "p99" => percentiles.p99,
            _ => unreachable!("the gate only names supported percentiles"),
        };
        assert!(
            actual <= budget,
            "{metric_name} {percentile_name} is {actual} ns, above {env_name}={budget} ns"
        );
    }
}

fn complete_artifact() -> String {
    let mut artifact = String::from(
        "schema,prism-material-perf-v2\n\
         commit,0123456789abcdef0123456789abcdef01234567\n\
         scene,default_native_material_frame\n\
         capture_scope,production_native_material_frame\n\
         producer,prism_native_material_frame_capture_v1\n\
         backend,metal\n\
         adapter,physical-gpu\n\
         resolution,1280x720\n\
         warmup,30\n\
         samples,60\n\
         timestamp_source,gpu_timestamp_query\n\
         timestamp_unit,ns\n",
    );
    for index in 0..60_u64 {
        let cpu = 100_000 + index * 1_000;
        let gpu = 80_000 + index * 500;
        let upload = 1_024 + index * 16;
        artifact.push_str(&format!("sample,{index},{cpu},{gpu},{upload}\n"));
    }
    artifact
}

#[test]
fn raw_material_artifact_recomputes_real_percentiles() {
    let evidence = parse(&complete_artifact()).expect("complete artifact must be admissible");
    assert_eq!(evidence.fields["timestamp_source"], "gpu_timestamp_query");
    assert_eq!(evidence.samples.len(), 60);
    assert_eq!(
        metric_percentiles(&evidence, "frame_gpu_timestamp_ns").p95,
        108_000
    );
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
fn summary_only_or_incomplete_raw_samples_are_rejected() {
    let mut summary = complete_artifact();
    let sample_start = summary.find("sample,").expect("fixture has samples");
    summary.truncate(sample_start);
    summary.push_str("metric,frame_gpu_timestamp_ns,60,80000,120000,160000\n");
    let error = parse(&summary).expect_err("summary-only evidence must fail the gate");
    assert!(error.contains("raw sample"), "unexpected error: {error}");

    let incomplete = complete_artifact().replace("sample,29,", "sample,60,");
    let error = parse(&incomplete).expect_err("non-contiguous samples must fail the gate");
    assert!(error.contains("contiguous"), "unexpected error: {error}");
}

#[test]
fn generic_rhi_timestamp_probe_cannot_pass_as_material_frame_evidence() {
    let probe = complete_artifact()
        .replace(
            "capture_scope,production_native_material_frame",
            "capture_scope,rhi_timestamp_probe",
        )
        .replace(
            "producer,prism_native_material_frame_capture_v1",
            "producer,prism_native_rhi_timestamp_probe_v1",
        );
    let error = parse(&probe).expect_err("generic RHI work must not pass material gate");
    assert!(error.contains("capture_scope"), "unexpected error: {error}");
}

/// Run with a real production-native-material artifact, for example:
///
/// `PRISM_REQUIRE_GPU=1 PRISM_MATERIAL_EXPECTED_COMMIT=$(git rev-parse HEAD)
/// PRISM_MATERIAL_PERF_EVIDENCE=/abs/path/evidence.csv
/// PRISM_MATERIAL_CPU_P50_MAX_NS=... PRISM_MATERIAL_CPU_P95_MAX_NS=...
/// PRISM_MATERIAL_CPU_P99_MAX_NS=... PRISM_MATERIAL_GPU_P50_MAX_NS=...
/// PRISM_MATERIAL_GPU_P95_MAX_NS=... PRISM_MATERIAL_GPU_P99_MAX_NS=...
/// cargo test -p prism_render_app --test material_performance_evidence
/// material_gpu_performance_evidence_gate -- --ignored --nocapture`
#[test]
#[ignore = "requires a timestamped production native-material artifact"]
fn material_gpu_performance_evidence_gate() {
    assert_eq!(
        std::env::var("PRISM_REQUIRE_GPU").as_deref(),
        Ok("1"),
        "the performance gate requires PRISM_REQUIRE_GPU=1"
    );
    let expected_commit = std::env::var("PRISM_MATERIAL_EXPECTED_COMMIT")
        .expect("set PRISM_MATERIAL_EXPECTED_COMMIT to the tested HEAD revision");
    let path = std::env::var_os("PRISM_MATERIAL_PERF_EVIDENCE")
        .expect("set PRISM_MATERIAL_PERF_EVIDENCE to a collected artifact path");
    let path = Path::new(&path);
    let input =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let evidence = parse(&input)
        .unwrap_or_else(|error| panic!("invalid material performance evidence: {error}"));
    assert_eq!(
        evidence.fields["commit"], expected_commit,
        "the artifact must be captured from the requested HEAD"
    );
    assert_budgets(&evidence);
    eprintln!(
        "accepted material performance evidence: adapter={}, backend={}, samples={}, frame_gpu_p95_ns={}",
        evidence.fields["adapter"],
        evidence.fields["backend"],
        evidence.samples.len(),
        metric_percentiles(&evidence, "frame_gpu_timestamp_ns").p95,
    );
}
