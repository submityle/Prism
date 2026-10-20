"""Synthetic comparator/protocol tests, not measured hardware performance."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("gate.py")
SPEC = importlib.util.spec_from_file_location("ecs_perf_gate", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def fixture():
    data = {
        "schema": 1,
        "kind": "measurement",
        "controlled_machine": True,
        "machine": {"managed_id": "SYNTHETIC-TEST-HOST", "cpu": "test-only"},
        "build": {"rustc": "SYNTHETIC-TEST-TOOLCHAIN"},
        "contract": {
            "protocol": gate.PREFIX.strip(), "test": gate.TEST,
            "samples": gate.SAMPLES, "metrics": gate.METRICS,
            "benchmark_sha256": "0" * 64,
        },
        "source": {"revision": "test-only", "dirty": False},
        "metrics": {
            name: {"units": units, "samples_ns": [100] * gate.SAMPLES}
            for name, units in gate.METRICS.items()
        },
    }
    for key in ["machine", "build", "contract"]:
        data[f"{key}_sha256"] = gate.digest(data[key])
    return data


def baseline(data=None):
    return dict(data or fixture(), kind="baseline")


def protocol():
    return "\n".join(
        gate.PREFIX + json.dumps({"metric": name, "sample": index, "ns": 100 + index, "units": units})
        for name, units in gate.METRICS.items()
        for index in range(gate.SAMPLES)
    )


def portable_fixture(kind="portable-measurement"):
    complexity = [
        {"metric": "query_match_cold_calls", "world_units": 3, "changed_units": 0, "value": 3},
        {"metric": "query_match_warm_calls", "world_units": 3, "changed_units": 0, "value": 0},
        {"metric": "query_match_new_archetype_calls", "world_units": 4, "changed_units": 1, "value": 1},
        {"metric": "dirty_chunks_output", "world_units": 256, "changed_units": 3, "value": 1},
        {"metric": "dirty_chunks_output", "world_units": 16384, "changed_units": 3, "value": 3},
        {"metric": "changed_scan_rows", "world_units": 256, "changed_units": 3, "value": 3},
        {"metric": "changed_scan_rows", "world_units": 16384, "changed_units": 3, "value": 3},
        {"metric": "dirty_journal_records_scanned", "world_units": 131072, "changed_units": 1024, "value": 1024},
        {"metric": "dirty_candidate_chunk_marks", "world_units": 131072, "changed_units": 1024, "value": 1024},
        {"metric": "dirty_index_summaries_tested", "world_units": 131072, "changed_units": 1024, "value": 0},
        {"metric": "dirty_reported_chunks", "world_units": 131072, "changed_units": 1024, "value": 1024},
    ]
    timing = {
        metric: [100 + index for index in range(gate.PORTABLE_SAMPLES)]
        for metric in gate.PORTABLE_TIMING_METRICS
    }
    data = {
        "schema": 1,
        "kind": kind,
        "contract": gate.portable_contract(),
        "complexity": complexity,
        "timing": timing,
    }
    data["contract_sha256"] = gate.digest(data["contract"])
    if kind == "portable-baseline":
        data["complexity"] = [dict(record, tolerance_abs=0) for record in complexity]
    return data


class ComparisonTests(unittest.TestCase):
    def test_equal_and_improved_measurements_pass(self):
        candidate = fixture()
        self.assertTrue(gate.compare(baseline(), candidate, 0)["passed"])
        candidate["metrics"]["spawn_batch"]["samples_ns"] = [90] * gate.SAMPLES
        candidate["source"]["revision"] = "new-source-is-allowed"
        self.assertTrue(gate.compare(baseline(), candidate, 0)["passed"])

    def test_regression_in_any_metric_fails(self):
        for name in gate.METRICS:
            with self.subTest(name=name):
                candidate = fixture()
                candidate["metrics"][name]["samples_ns"] = [111] * gate.SAMPLES
                report = gate.compare(baseline(), candidate, 10)
                self.assertFalse(report["passed"])
                self.assertFalse(report["metrics"][name]["passed"])

    def test_configurable_threshold_boundary(self):
        candidate = fixture()
        candidate["metrics"]["spawn_batch"]["samples_ns"] = [110] * gate.SAMPLES
        self.assertTrue(gate.compare(baseline(), candidate, 10)["passed"])
        self.assertFalse(gate.compare(baseline(), candidate, 9)["passed"])

    def test_one_scheduler_outlier_does_not_change_median(self):
        candidate = fixture()
        candidate["metrics"]["spawn_batch"]["samples_ns"][-1] = 100_000
        self.assertTrue(gate.compare(baseline(), candidate, 0)["passed"])

    def test_machine_build_and_contract_mismatches_fail_closed(self):
        for section, field in [("machine", "managed_id"), ("build", "rustc"), ("contract", "benchmark_sha256")]:
            with self.subTest(section=section):
                candidate = fixture()
                candidate[section][field] = "f" * 64
                candidate[f"{section}_sha256"] = gate.digest(candidate[section])
                with self.assertRaisesRegex(gate.GateError, f"{section} mismatch"):
                    gate.compare(baseline(), candidate, 10)

    def test_tampered_fingerprint_rejected(self):
        candidate = fixture()
        candidate["machine"]["managed_id"] = "changed-without-updating-checksum"
        with self.assertRaisesRegex(gate.GateError, "checksum mismatch"):
            gate.compare(baseline(), candidate, 10)

    def test_invalid_thresholds_rejected(self):
        for value in [-1, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(gate.GateError):
                gate.compare(baseline(), fixture(), value)

    def test_invalid_durations_rejected(self):
        for value in [0, -1, True, 1.5, float("nan"), float("inf"), "100", 2**65]:
            with self.subTest(value=value):
                candidate = fixture()
                candidate["metrics"]["spawn_batch"]["samples_ns"][0] = value
                with self.assertRaises(gate.GateError):
                    gate.compare(baseline(), candidate, 10)

    def test_missing_metrics_units_and_samples_rejected(self):
        for case in ["missing", "units", "samples"]:
            candidate = fixture()
            if case == "missing":
                del candidate["metrics"]["spawn_batch"]
            elif case == "units":
                candidate["metrics"]["spawn_batch"]["units"] = 1
            else:
                candidate["metrics"]["spawn_batch"]["samples_ns"].pop()
            with self.subTest(case=case), self.assertRaises(gate.GateError):
                gate.compare(baseline(), candidate, 10)

    def test_malformed_top_level_metadata_rejected(self):
        cases = [[], None, {}, dict(fixture(), schema=True), dict(fixture(), kind=[])]
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(gate.GateError):
                gate.validate(candidate)

    def test_no_exclusivity_acknowledgement_is_invalid(self):
        candidate = fixture()
        candidate["controlled_machine"] = False
        with self.assertRaises(gate.GateError):
            gate.compare(baseline(), candidate, 10)


class ProtocolTests(unittest.TestCase):
    def test_accept_complete_output_with_libtest_preamble(self):
        parsed = gate.parse_samples("running 1 test\ntest perf_gate::controlled_samples ... " + protocol())
        self.assertEqual(set(parsed), set(gate.METRICS))
        self.assertEqual(parsed["spawn_batch"]["samples_ns"], list(range(100, 109)))

    def test_empty_truncated_duplicate_and_unknown_samples_rejected(self):
        output = protocol()
        cases = [
            "running 0 tests\ntest result: ok.",
            output.rsplit("\n", 1)[0],
            output + "\n" + output.splitlines()[0],
            output.replace('"spawn_batch"', '"unknown_metric"'),
            output.replace('"units": 1000000', '"units": 1'),
            output + "\n" + gate.PREFIX + "not json",
        ]
        for index, output in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(gate.GateError):
                gate.parse_samples(output)

    def test_portable_parser_accepts_libtest_prefix_and_requires_all_records(self):
        source = portable_fixture()
        output = "running 1 test\n" + gate.PORTABLE_COMPLEXITY_PREFIX + json.dumps(source["complexity"][0])
        for record in source["complexity"][1:]:
            output += "\n" + gate.PORTABLE_COMPLEXITY_PREFIX + json.dumps(record)
        for metric, values in source["timing"].items():
            for sample, ns in enumerate(values):
                output += "\n" + gate.PORTABLE_TIMING_PREFIX + json.dumps({
                    "metric": metric,
                    "sample": sample,
                    "ns": ns,
                    "units": gate.PORTABLE_TIMING_METRICS[metric],
                })
        parsed = gate.parse_portable_samples(output)
        self.assertEqual(len(parsed["complexity"]), 11)
        self.assertEqual(parsed["timing"]["changed_scan_16384"], list(range(100, 109)))

    def test_portable_parser_rejects_missing_complexity_point(self):
        source = portable_fixture()
        output = "\n".join(
            gate.PORTABLE_COMPLEXITY_PREFIX + json.dumps(record)
            for record in source["complexity"][:-1]
        )
        for metric, values in source["timing"].items():
            for sample, ns in enumerate(values):
                output += "\n" + gate.PORTABLE_TIMING_PREFIX + json.dumps({
                    "metric": metric,
                    "sample": sample,
                    "ns": ns,
                    "units": gate.PORTABLE_TIMING_METRICS[metric],
                })
        with self.assertRaises(gate.GateError):
            gate.parse_portable_samples(output)


class PersistenceAndCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prism-ecs-perf-test-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.base = self.directory / "baseline.json"
        self.candidate = self.directory / "candidate.json"

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)], text=True,
            capture_output=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=False,
        )

    def test_persisted_baseline_no_clobber_and_explicit_replace(self):
        gate.record_baseline(fixture(), self.base)
        saved = self.base.read_bytes()
        candidate = fixture()
        candidate["metrics"]["spawn_batch"]["samples_ns"] = [200] * gate.SAMPLES
        with self.assertRaises(FileExistsError):
            gate.record_baseline(candidate, self.base)
        self.assertEqual(self.base.read_bytes(), saved)
        gate.record_baseline(candidate, self.base, replace=True)
        self.assertEqual(gate.load(self.base)["kind"], "baseline")
        self.assertEqual(gate.load(self.base)["metrics"]["spawn_batch"]["samples_ns"][0], 200)

    def test_cli_regression_returns_one_and_writes_report(self):
        candidate = fixture()
        candidate["metrics"]["spawn_batch"]["samples_ns"] = [200] * gate.SAMPLES
        gate.write_json(self.candidate, candidate)
        gate.record_baseline(fixture(), self.base)
        report = self.directory / "report.json"
        result = self.cli("compare", "--baseline", self.base, "--measurement", self.candidate,
                          "--max-regression-percent", "10", "--report", report)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(json.loads(report.read_text())["passed"])

    def test_cli_mismatched_machine_returns_two(self):
        candidate = fixture()
        candidate["machine"]["managed_id"] = "OTHER-SYNTHETIC-HOST"
        candidate["machine_sha256"] = gate.digest(candidate["machine"])
        gate.write_json(self.candidate, candidate)
        gate.record_baseline(fixture(), self.base)
        result = self.cli("compare", "--baseline", self.base, "--measurement", self.candidate,
                          "--max-regression-percent", "10")
        self.assertEqual(result.returncode, 2)
        self.assertIn("machine mismatch", result.stderr)

    def test_cli_success_and_missing_baseline(self):
        gate.write_json(self.candidate, fixture())
        result = self.cli("compare", "--baseline", self.base, "--measurement", self.candidate,
                          "--max-regression-percent", "0")
        self.assertEqual(result.returncode, 2)
        result = self.cli("record-baseline", "--baseline", self.base, "--measurement", self.candidate)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.cli("compare", "--baseline", self.base, "--measurement", self.candidate,
                          "--max-regression-percent", "0")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_report_cannot_overwrite_baseline(self):
        gate.write_json(self.candidate, fixture())
        gate.record_baseline(fixture(), self.base)
        before = self.base.read_bytes()
        result = self.cli("compare", "--baseline", self.base, "--measurement", self.candidate,
                          "--max-regression-percent", "0", "--report", self.base)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.base.read_bytes(), before)

    def test_portable_complexity_regression_is_blocking_and_timing_is_percentiled(self):
        base = portable_fixture("portable-baseline")
        candidate = portable_fixture()
        candidate["complexity"][3]["value"] = 2
        report = gate.compare_portable(base, candidate)
        self.assertFalse(report["passed"])
        dirty_256 = next(
            row for row in report["complexity"]
            if row["metric"] == "dirty_chunks_output" and row["world_units"] == 256
        )
        self.assertFalse(dirty_256["passed"])
        self.assertEqual(report["timing"]["changed_scan_16384"]["p50_ns"], 104)
        self.assertEqual(report["timing"]["changed_scan_16384"]["p95_ns"], 108)
        self.assertEqual(report["timing"]["changed_scan_16384"]["p99_ns"], 108)

        candidate = portable_fixture()
        candidate["complexity"][-2]["value"] = 131072
        report = gate.compare_portable(base, candidate)
        self.assertFalse(report["passed"])
        summary_work = next(
            row for row in report["complexity"]
            if row["metric"] == "dirty_index_summaries_tested"
        )
        self.assertFalse(summary_work["passed"])

    def test_portable_baseline_is_explicit_and_non_clobbering(self):
        measurement = portable_fixture()
        gate.record_portable_baseline(measurement, self.base)
        saved = self.base.read_bytes()
        with self.assertRaises(FileExistsError):
            gate.record_portable_baseline(measurement, self.base)
        self.assertEqual(self.base.read_bytes(), saved)

    def test_portable_baseline_rejects_invalid_tolerance(self):
        for tolerance in [-1, True, 1.5, "zero"]:
            candidate = portable_fixture("portable-baseline")
            candidate["complexity"][0]["tolerance_abs"] = tolerance
            with self.subTest(tolerance=tolerance), self.assertRaises(gate.GateError):
                gate.validate_portable(candidate, "portable-baseline")

    def test_no_acknowledgement_does_not_invoke_cargo(self):
        with patch.object(gate.subprocess, "run") as run, redirect_stderr(io.StringIO()):
            result = gate.main(["measure", "--machine-id", "test", "--target-dir", str(self.directory),
                                "--output", str(self.candidate)])
        self.assertEqual(result, 2)
        run.assert_not_called()
        self.assertFalse(self.candidate.exists())

    def test_failed_benchmark_cannot_publish_partial_samples(self):
        with patch.object(gate, "machine_fingerprint", return_value={"id": "test"}), \
                patch.object(gate, "build_fingerprint", return_value={"rustc": "test"}), \
                patch.object(gate, "command_output", return_value="test"), \
                patch.object(gate.subprocess, "run", side_effect=subprocess.CalledProcessError(101, "cargo")), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = gate.main(["measure", "--controlled-machine", "--machine-id", "test",
                                "--target-dir", "/private/tmp/prism-ecs-completion-target",
                                "--output", str(self.candidate)])
        self.assertEqual(result, 2)
        self.assertFalse(self.candidate.exists())


if __name__ == "__main__":
    unittest.main()
