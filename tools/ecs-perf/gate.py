#!/usr/bin/env python3
"""Controlled-machine ECS regression gate (Python stdlib only).

Run from any directory; --target-dir must name the shared build directory.
An operator must reserve an idle/exclusive runner before acknowledging it:

  python3 tools/ecs-perf/gate.py measure --controlled-machine \
    --machine-id ecs-perf-01 --target-dir /private/tmp/prism-ecs-completion-target \
    --output /tmp/ecs-measurement.json
  python3 tools/ecs-perf/gate.py record-baseline \
    --measurement /tmp/ecs-measurement.json --baseline /persistent/ecs-baseline.json
  python3 tools/ecs-perf/gate.py compare --baseline /persistent/ecs-baseline.json \
    --measurement /tmp/ecs-measurement.json --max-regression-percent 10

record-baseline never silently updates a baseline; replacement requires --replace.
compare returns 1 on a regression, 2 on invalid/incompatible input, 0 on success.
Source revisions may differ; machine, toolchain/build configuration, benchmark
source, metric set and sample counts must match. No local timing is a hardware
claim, and a machine fingerprint cannot itself prove runner exclusivity.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
PREFIX = "PRISM_ECS_PERF_V1 "
PORTABLE_COMPLEXITY_PREFIX = "PRISM_ECS_COMPLEXITY_V1 "
PORTABLE_TIMING_PREFIX = "PRISM_ECS_TIMING_V1 "
SAMPLES = 9
PORTABLE_SAMPLES = 9
ENTITIES = 1_000_000
METRICS = {
    "spawn_batch": ENTITIES,
    "serial_propagate": ENTITIES,
    "parallel_propagate_1": ENTITIES,
    "parallel_propagate_4": ENTITIES,
    "query_cache_idle": 1024,
    "dirty_idle": 1,
    "dirty_clustered": 1,
    "dirty_scattered": 1,
    "changed_scan": ENTITIES,
    "simd_add": ENTITIES,
}
FEATURES = "multi_thread,simd"
TEST = "perf_gate::controlled_samples"
PORTABLE_TEST = "perf_gate::portable_complexity_samples"
MAX_SAMPLE_NS = 2**64 - 1
PORTABLE_COMPLEXITY_METRICS = {
    "query_match_cold_calls": 1,
    "query_match_warm_calls": 1,
    "query_match_new_archetype_calls": 1,
    "dirty_chunks_output": 2,
    "changed_scan_rows": 2,
    "dirty_journal_records_scanned": 1,
    "dirty_candidate_chunk_marks": 1,
    "dirty_index_summaries_tested": 1,
    "dirty_reported_chunks": 1,
}
PORTABLE_COMPLEXITY_POINTS = {
    ("query_match_cold_calls", 3),
    ("query_match_warm_calls", 3),
    ("query_match_new_archetype_calls", 4),
    ("dirty_chunks_output", 256),
    ("dirty_chunks_output", 16384),
    ("changed_scan_rows", 256),
    ("changed_scan_rows", 16384),
    ("dirty_journal_records_scanned", 131072),
    ("dirty_candidate_chunk_marks", 131072),
    ("dirty_index_summaries_tested", 131072),
    ("dirty_reported_chunks", 131072),
}
PORTABLE_TIMING_METRICS = {
    "dirty_chunks_256": 256,
    "changed_scan_256": 256,
    "dirty_chunks_16384": 16384,
    "changed_scan_16384": 16384,
}


class GateError(ValueError):
    """Invalid measurement, configuration or incompatible baseline."""


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except FileNotFoundError:
        return None


def command_output(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def machine_fingerprint(machine_id):
    if not machine_id.strip():
        raise GateError("--machine-id must identify the managed physical runner")
    info = {
        "managed_id": machine_id,
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
    if sys.platform.startswith("linux"):
        # Do not hash transient MHz, load averages, available RAM or process ids.
        fields = {"model name", "vendor_id", "cpu family", "model", "stepping", "microcode", "flags"}
        cpu = {}
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() in fields:
                cpu.setdefault(key.strip(), set()).add(value.strip())
        if not cpu.get("model name"):
            raise GateError("cannot identify CPU model on this Linux host")
        info["cpu"] = {key: sorted(values) for key, values in cpu.items()}
        info["affinity"] = sorted(os.sched_getaffinity(0))
        info["memory_total"] = next(
            line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")
        )
        info["governors"] = {
            path.parent.name: path.read_text().strip()
            for path in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor"))
        }
        info["boost"] = {
            path: read_optional(path)
            for path in [
                "/sys/devices/system/cpu/cpufreq/boost",
                "/sys/devices/system/cpu/intel_pstate/no_turbo",
            ]
        }
        # Include container limits, not only the host CPU/memory inventory.
        info["cgroup_limits"] = {
            path: read_optional(path)
            for path in [
                "/sys/fs/cgroup/cpu.max",
                "/sys/fs/cgroup/memory.max",
                "/sys/fs/cgroup/cpuset.cpus.effective",
                "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
            ]
        }
    elif sys.platform == "darwin":
        info["hardware"] = {
            key: command_output(["sysctl", "-n", key])
            for key in ["machdep.cpu.brand_string", "hw.model", "hw.memsize", "hw.physicalcpu", "hw.logicalcpu"]
        }
    else:
        raise GateError("controlled measurements currently support Linux and macOS only")
    return info


def build_fingerprint():
    # Record all inherited Cargo/Rust build overrides, excluding cache locations,
    # presentation flags and job count (which affect build time, not sample time).
    excluded = {"CARGO_HOME", "CARGO_TARGET_DIR", "CARGO_TERM_COLOR", "CARGO_TERM_PROGRESS_WHEN", "CARGO_BUILD_JOBS"}
    env = {
        key: value for key, value in sorted(os.environ.items())
        if key not in excluded and (key.startswith(("CARGO_PROFILE_", "CARGO_TARGET_", "CARGO_BUILD_"))
                                   or key in {"RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC", "RUSTUP_TOOLCHAIN", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "CARGO_INCREMENTAL", "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "MALLOC_CONF", "GLIBC_TUNABLES"})
    }
    # Cargo configuration can change code generation without an env override.
    configs = {}
    directories = list(ROOT.parents) + [ROOT]
    directories.append(Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))))
    for index, directory in enumerate(directories):
        cargo_dir = directory if index == len(directories) - 1 else directory / ".cargo"
        for name in ["config", "config.toml"]:
            path = cargo_dir / name
            if path.is_file():
                configs[f"{index}/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "rustc": command_output([os.environ.get("RUSTC", "rustc"), "-Vv"]),
        "cargo": command_output(["cargo", "-V"]),
        "features": FEATURES,
        "profile": "release",
        "environment": env,
        "cargo_configs": configs,
        "workspace_manifest_sha256": hashlib.sha256((ROOT / "Cargo.toml").read_bytes()).hexdigest(),
    }


def parse_samples(output):
    collected = {name: {} for name in METRICS}
    for line in output.splitlines():
        if PREFIX not in line:
            continue
        try:
            record = json.loads(line.split(PREFIX, 1)[1])
            metric, index, ns, units = (record[key] for key in ["metric", "sample", "ns", "units"])
        except (ValueError, KeyError, TypeError) as error:
            raise GateError(f"invalid sample: {line}") from error
        if not isinstance(metric, str) or metric not in METRICS:
            raise GateError(f"unknown metric: {metric!r}")
        if type(index) is not int or not 0 <= index < SAMPLES:
            raise GateError("invalid sample index")
        if type(ns) is not int or not 0 < ns <= MAX_SAMPLE_NS:
            raise GateError("sample duration must be a positive integer number of ns")
        if type(units) is not int or units != METRICS[metric]:
            raise GateError(f"workload units changed for {metric}")
        if index in collected[metric]:
            raise GateError(f"duplicate sample {metric}/{index}")
        collected[metric][index] = ns
    if any(len(values) != SAMPLES for values in collected.values()):
        raise GateError("missing samples (test may have been filtered out or lacked required features)")
    return {
        name: {"units": METRICS[name], "samples_ns": [values[i] for i in range(SAMPLES)]}
        for name, values in collected.items()
    }


def parse_portable_samples(output):
    complexity = {}
    timing = {name: {} for name in PORTABLE_TIMING_METRICS}
    for line in output.splitlines():
        if PORTABLE_COMPLEXITY_PREFIX in line:
            try:
                record = json.loads(line.split(PORTABLE_COMPLEXITY_PREFIX, 1)[1])
                metric, world_units, changed_units, value = (
                    record[key] for key in ["metric", "world_units", "changed_units", "value"]
                )
            except (ValueError, KeyError, TypeError) as error:
                raise GateError(f"invalid portable complexity record: {line}") from error
            if metric not in PORTABLE_COMPLEXITY_METRICS:
                raise GateError(f"unknown portable complexity metric: {metric!r}")
            if (type(world_units) is not int or world_units <= 0 or
                    type(changed_units) is not int or changed_units < 0 or
                    type(value) is not int or value < 0):
                raise GateError("portable complexity fields must be nonnegative integers")
            key = (metric, world_units)
            if key in complexity:
                raise GateError(f"duplicate portable complexity record: {key}")
            complexity[key] = {
                "metric": metric,
                "world_units": world_units,
                "changed_units": changed_units,
                "value": value,
            }
        elif PORTABLE_TIMING_PREFIX in line:
            try:
                record = json.loads(line.split(PORTABLE_TIMING_PREFIX, 1)[1])
                metric, sample, ns, units = (
                    record[key] for key in ["metric", "sample", "ns", "units"]
                )
            except (ValueError, KeyError, TypeError) as error:
                raise GateError(f"invalid portable timing record: {line}") from error
            if metric not in PORTABLE_TIMING_METRICS:
                raise GateError(f"unknown portable timing metric: {metric!r}")
            if (type(sample) is not int or not 0 <= sample < PORTABLE_SAMPLES or
                    type(ns) is not int or not 0 < ns <= MAX_SAMPLE_NS or
                    type(units) is not int or units != PORTABLE_TIMING_METRICS[metric]):
                raise GateError(f"invalid portable timing record for {metric}")
            if sample in timing[metric]:
                raise GateError(f"duplicate portable timing sample: {metric}/{sample}")
            timing[metric][sample] = ns
    if set(complexity) != PORTABLE_COMPLEXITY_POINTS:
        raise GateError("portable complexity point set differs from the gate contract")
    if any(len(values) != PORTABLE_SAMPLES for values in timing.values()):
        raise GateError("missing portable timing samples")
    return {
        "complexity": sorted(complexity.values(), key=lambda item: (item["metric"], item["world_units"])),
        "timing": {
            metric: [values[index] for index in range(PORTABLE_SAMPLES)]
            for metric, values in timing.items()
        },
    }


def portable_contract():
    benchmark = ROOT / "pkg/prism_ecs/tests/scale.rs"
    return {
        "protocol": PORTABLE_COMPLEXITY_PREFIX.strip(),
        "timing_protocol": PORTABLE_TIMING_PREFIX.strip(),
        "test": PORTABLE_TEST,
        "samples": PORTABLE_SAMPLES,
        "complexity_metrics": PORTABLE_COMPLEXITY_METRICS,
        "timing_metrics": PORTABLE_TIMING_METRICS,
        "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
    }


def validate_portable(data, kind):
    if not isinstance(data, dict) or data.get("schema") != 1 or data.get("kind") != kind:
        raise GateError("unsupported portable measurement schema")
    contract = data.get("contract")
    if contract != portable_contract():
        raise GateError("incompatible portable workload contract")
    if data.get("contract_sha256") != digest(contract):
        raise GateError("portable contract checksum mismatch")
    complexity = data.get("complexity")
    if not isinstance(complexity, list):
        raise GateError("missing portable complexity records")
    parsed_complexity = {}
    for record in complexity:
        if not isinstance(record, dict):
            raise GateError("invalid portable complexity record")
        key = (record.get("metric"), record.get("world_units"))
        if key in parsed_complexity:
            raise GateError(f"duplicate portable complexity record: {key}")
        if (key[0] not in PORTABLE_COMPLEXITY_METRICS or type(key[1]) is not int or key[1] <= 0 or
                type(record.get("changed_units")) is not int or record["changed_units"] < 0 or
                type(record.get("value")) is not int or record["value"] < 0):
            raise GateError("invalid portable complexity fields")
        if kind == "portable-baseline" and (
                type(record.get("tolerance_abs")) is not int or record["tolerance_abs"] < 0):
            raise GateError("invalid portable complexity tolerance")
        parsed_complexity[key] = record
    if set(parsed_complexity) != PORTABLE_COMPLEXITY_POINTS:
        raise GateError("portable complexity point set differs from the gate contract")
    timing = data.get("timing")
    if not isinstance(timing, dict) or set(timing) != set(PORTABLE_TIMING_METRICS):
        raise GateError("portable timing metric set differs from the gate contract")
    for metric, values in timing.items():
        if (not isinstance(values, list) or len(values) != PORTABLE_SAMPLES or
                any(type(value) is not int or not 0 < value <= MAX_SAMPLE_NS for value in values)):
            raise GateError(f"invalid portable timing samples for {metric}")
    return data


def load_portable(path):
    data = json.loads(Path(path).read_text())
    return validate_portable(data, data.get("kind"))


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def timing_stats(values):
    median = statistics.median(values)
    return {
        "samples": len(values),
        "min_ns": min(values),
        "p50_ns": percentile(values, 0.50),
        "p95_ns": percentile(values, 0.95),
        "p99_ns": percentile(values, 0.99),
        "max_ns": max(values),
        "mean_ns": statistics.fmean(values),
        "mad_ns": statistics.median(abs(value - median) for value in values),
    }


def compare_portable(baseline, measurement):
    validate_portable(baseline, "portable-baseline")
    validate_portable(measurement, "portable-measurement")
    base = {(record["metric"], record["world_units"]): record for record in baseline["complexity"]}
    candidate = {(record["metric"], record["world_units"]): record for record in measurement["complexity"]}
    if set(base) != set(candidate):
        raise GateError("portable complexity points differ from the baseline")
    rows = []
    for key in sorted(base):
        expected = base[key]
        observed = candidate[key]
        if expected["changed_units"] != observed["changed_units"]:
            raise GateError(f"changed workload differs for {key}")
        tolerance = int(expected.get("tolerance_abs", 0))
        if tolerance < 0:
            raise GateError(f"negative complexity tolerance for {key}")
        allowed = expected["value"] + tolerance
        passed = observed["value"] <= allowed
        rows.append({
            "metric": key[0], "world_units": key[1],
            "baseline_value": expected["value"], "candidate_value": observed["value"],
            "tolerance_abs": tolerance, "allowed_value": allowed, "passed": passed,
        })
    return {
        "passed": all(row["passed"] for row in rows),
        "complexity": rows,
        "timing": {
            metric: timing_stats(values)
            for metric, values in measurement["timing"].items()
        },
        "timing_policy": "diagnostic_only_on_shared_or_offline_runners",
    }


def measure_portable(args):
    target = Path(args.target_dir)
    if not target.is_absolute():
        raise GateError("--target-dir must be an explicit absolute shared target directory")
    if Path(args.output).exists():
        raise GateError("output exists; choose a new measurement path")
    contract = portable_contract()
    command = [
        "cargo", "test", "--locked", "--offline", "--release", "--target-dir", str(target),
        "--manifest-path", str(ROOT / "pkg/prism_ecs/Cargo.toml"), "-p", "prism_ecs",
        "--features", FEATURES, "--test", "scale", PORTABLE_TEST,
        "--", "--exact", "--nocapture", "--test-threads=1",
    ]
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=None, check=True)
    print(result.stdout, end="")
    parsed = parse_portable_samples(result.stdout)
    data = {
        "schema": 1, "kind": "portable-measurement",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "revision": command_output(["git", "rev-parse", "HEAD"]),
            "dirty": bool(command_output(["git", "status", "--porcelain", "--untracked-files=normal"])),
        },
        "contract": contract,
        "complexity": parsed["complexity"],
        "timing": parsed["timing"],
        "command": command,
    }
    data["contract_sha256"] = digest(contract)
    write_json(args.output, validate_portable(data, "portable-measurement"))


def record_portable_baseline(measurement, path, replace=False):
    validate_portable(measurement, "portable-measurement")
    baseline = {
        "schema": 1,
        "kind": "portable-baseline",
        "contract": measurement["contract"],
        "contract_sha256": measurement["contract_sha256"],
        "complexity": [dict(record, tolerance_abs=0) for record in measurement["complexity"]],
        "timing": measurement["timing"],
    }
    write_json(path, validate_portable(baseline, "portable-baseline"), replace=replace)


def validate(data):
    if not isinstance(data, dict) or type(data.get("schema")) is not int or data["schema"] != 1:
        raise GateError("unsupported measurement schema")
    if data.get("kind") not in ("measurement", "baseline") or data.get("controlled_machine") is not True:
        raise GateError("measurement must come from an acknowledged controlled runner")
    for key in ["machine", "build", "contract"]:
        if not isinstance(data.get(key), dict) or not data[key]:
            raise GateError(f"missing {key} fingerprint")
        if data.get(f"{key}_sha256") != digest(data[key]):
            raise GateError(f"{key} fingerprint checksum mismatch")
    contract = data["contract"]
    if (contract.get("samples") != SAMPLES or contract.get("metrics") != METRICS
            or contract.get("test") != TEST or contract.get("protocol") != PREFIX.strip()
            or not isinstance(contract.get("benchmark_sha256"), str)
            or len(contract["benchmark_sha256"]) != 64):
        raise GateError("incompatible workload contract")
    metrics = data.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
        raise GateError("metric set differs from the gate contract")
    for name, units in METRICS.items():
        entry = metrics[name]
        if not isinstance(entry, dict) or type(entry.get("units")) is not int or entry["units"] != units:
            raise GateError(f"wrong workload units for {name}")
        values = entry.get("samples_ns")
        if not isinstance(values, list) or len(values) != SAMPLES:
            raise GateError(f"wrong sample count for {name}")
        if any(type(value) is not int or not 0 < value <= MAX_SAMPLE_NS for value in values):
            raise GateError(f"nonpositive/noninteger duration for {name}")
    return data


def load(path):
    return validate(json.loads(Path(path).read_text()))


def write_json(path, data, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publication, including no-clobber baseline creation. A crash must
    # not leave a half-written baseline that looks like a valid previous run.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def record_baseline(measurement, path, replace=False):
    validate(measurement)
    if measurement["kind"] != "measurement":
        raise GateError("record-baseline requires an original measurement")
    baseline = dict(measurement, kind="baseline", recorded_at=datetime.now(timezone.utc).isoformat())
    write_json(path, baseline, replace=replace)


def compare(baseline, measurement, max_regression_percent):
    if not math.isfinite(max_regression_percent) or max_regression_percent < 0:
        raise GateError("regression threshold must be finite and nonnegative")
    validate(baseline)
    validate(measurement)
    if baseline["kind"] != "baseline" or measurement["kind"] != "measurement":
        raise GateError("compare requires a persisted baseline and a new measurement")
    for key in ["machine", "build", "contract"]:
        if baseline[key] != measurement[key]:
            raise GateError(f"{key} mismatch; use the original runner/configuration or explicitly record a new baseline")
    rows = {}
    failed = False
    for name in METRICS:
        old = statistics.median(baseline["metrics"][name]["samples_ns"])
        values = measurement["metrics"][name]["samples_ns"]
        new = statistics.median(values)
        regression = (new / old - 1) * 100
        passed = new <= old * (1 + max_regression_percent / 100)
        failed |= not passed
        rows[name] = {
            "baseline_median_ns": old, "candidate_median_ns": new,
            "candidate_mad_ns": statistics.median(abs(value - new) for value in values),
            "regression_percent": regression, "passed": passed,
        }
    return {"passed": not failed, "max_regression_percent": max_regression_percent, "metrics": rows}


def measure(args):
    if not args.controlled_machine:
        raise GateError("reserve an idle/exclusive machine and pass --controlled-machine")
    target = Path(args.target_dir)
    if not target.is_absolute():
        raise GateError("--target-dir must be an explicit absolute shared target directory")
    if Path(args.output).exists():
        raise GateError("output exists; choose a new measurement path")
    machine = machine_fingerprint(args.machine_id)
    build = build_fingerprint()
    benchmark = ROOT / "pkg/prism_ecs/tests/scale.rs"
    contract = {
        "protocol": PREFIX.strip(), "test": TEST, "samples": SAMPLES, "metrics": METRICS,
        "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
    }
    source = {
        "revision": command_output(["git", "rev-parse", "HEAD"]),
        "dirty": bool(command_output(["git", "status", "--porcelain", "--untracked-files=normal"])),
        "cargo_lock_sha256": hashlib.sha256((ROOT / "Cargo.lock").read_bytes()).hexdigest(),
    }
    command = [
        "cargo", "test", "--locked", "--release", "--target-dir", str(target),
        "--manifest-path", str(ROOT / "pkg/prism_ecs/Cargo.toml"), "-p", "prism_ecs",
        "--features", FEATURES, "--test", "scale", TEST,
        "--", "--exact", "--ignored", "--nocapture", "--test-threads=1",
    ]
    # Compile/test exit status is authoritative; partial sample output is never
    # accepted after a failed assertion or a panic late in the benchmark.
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=None, check=True)
    print(result.stdout, end="")
    metrics = parse_samples(result.stdout)
    if machine_fingerprint(args.machine_id) != machine or build_fingerprint() != build:
        raise GateError("runner/build configuration changed during measurement")
    if hashlib.sha256(benchmark.read_bytes()).hexdigest() != contract["benchmark_sha256"]:
        raise GateError("benchmark source changed during measurement")
    data = {
        "schema": 1, "kind": "measurement", "controlled_machine": True,
        "created_at": datetime.now(timezone.utc).isoformat(), "source": source,
        "machine": machine, "build": build, "contract": contract, "metrics": metrics,
        "command": command,
    }
    for key in ["machine", "build", "contract"]:
        data[f"{key}_sha256"] = digest(data[key])
    write_json(args.output, validate(data))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    collect = sub.add_parser("measure", help="build and collect real controlled-runner samples")
    collect.add_argument("--controlled-machine", action="store_true")
    collect.add_argument("--machine-id", required=True)
    collect.add_argument("--target-dir", required=True)
    collect.add_argument("--output", required=True)
    portable = sub.add_parser(
        "portable",
        help="run the offline portable complexity gate and report timing percentiles",
    )
    portable.add_argument("--baseline", required=True)
    portable.add_argument("--target-dir", required=True)
    portable.add_argument("--output", required=True)
    portable.add_argument("--report", required=True)
    portable_measure = sub.add_parser(
        "portable-measure",
        help="run the offline portable benchmark and persist its measurement",
    )
    portable_measure.add_argument("--target-dir", required=True)
    portable_measure.add_argument("--output", required=True)
    record_portable = sub.add_parser(
        "record-portable-baseline",
        help="explicitly persist a portable measurement as a complexity baseline",
    )
    record_portable.add_argument("--measurement", required=True)
    record_portable.add_argument("--baseline", required=True)
    record_portable.add_argument("--replace", action="store_true")
    record = sub.add_parser("record-baseline", help="explicitly persist an accepted measurement")
    record.add_argument("--measurement", required=True)
    record.add_argument("--baseline", required=True)
    record.add_argument("--replace", action="store_true")
    check = sub.add_parser("compare", help="reject incompatible machines/builds or timing regressions")
    check.add_argument("--measurement", required=True)
    check.add_argument("--baseline", required=True)
    check.add_argument("--max-regression-percent", type=float, required=True)
    check.add_argument("--report")
    args = parser.parse_args(argv)
    try:
        if args.action == "measure":
            measure(args)
        elif args.action == "portable":
            if Path(args.output).resolve() == Path(args.baseline).resolve() or Path(args.report).resolve() in {
                Path(args.output).resolve(), Path(args.baseline).resolve()
            }:
                raise GateError("portable output, report and baseline must be different files")
            measure_portable(args)
            report = compare_portable(load_portable(args.baseline), load_portable(args.output))
            write_json(args.report, report)
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
            return 0 if report["passed"] else 1
        elif args.action == "portable-measure":
            measure_portable(args)
        elif args.action == "record-portable-baseline":
            if Path(args.measurement).resolve() == Path(args.baseline).resolve():
                raise GateError("baseline and measurement must be different files")
            record_portable_baseline(load_portable(args.measurement), args.baseline, args.replace)
            print(f"Recorded portable baseline: {args.baseline}")
        elif args.action == "record-baseline":
            if Path(args.measurement).resolve() == Path(args.baseline).resolve():
                raise GateError("baseline and measurement must be different files")
            record_baseline(load(args.measurement), args.baseline, args.replace)
            print(f"Recorded baseline: {args.baseline}")
        else:
            protected = {Path(args.measurement).resolve(), Path(args.baseline).resolve()}
            if args.report and Path(args.report).resolve() in protected:
                raise GateError("report must not overwrite the baseline or measurement")
            report = compare(load(args.baseline), load(args.measurement), args.max_regression_percent)
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
            if args.report:
                write_json(args.report, report)
            return 0 if report["passed"] else 1
    except (GateError, OSError, ValueError, subprocess.SubprocessError, StopIteration) as error:
        print(f"ECS performance gate error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
