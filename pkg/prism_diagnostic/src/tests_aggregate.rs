//! §24.6 distributed / multi-instance aggregation tests: per-instance
//! summarization and nearest-rank percentiles, pooled cluster distribution,
//! robust median-p99 + MAD anomaly localization, distributed-trace tree
//! assembly (critical path, service attribution, orphan handling, duplicate
//! span ids), and the adaptive sampling-rate controller. Every expectation is
//! a hand-computed oracle over fixed inputs, so a refactor cannot silently
//! change an aggregation result.

use crate::aggregate::{
    estimate_fleet_reports, AnomalyConfig, ClusterAggregator, DistributedSpan, InstanceFrameReport,
    InstanceSummary, SampleRate, SampleReason, SamplingController, SamplingPolicy, SpanId,
    SpanKind, TraceAssembler, TraceId,
};

/// One millisecond in nanoseconds; frame-time fixtures are expressed in ms.
const MS: u64 = 1_000_000;

// ---- per-instance summary + nearest-rank percentiles ------------------------

#[test]
fn summary_percentiles_match_nearest_rank_oracle() {
    // Samples 1..=10 (deliberately out of order to exercise the internal sort).
    let samples: Vec<u64> = alloc::vec![7, 3, 10, 1, 8, 4, 9, 2, 6, 5];
    let summary = InstanceSummary::from_samples("shard-a", &samples);

    assert_eq!(summary.label, "shard-a");
    assert_eq!(summary.count, 10);
    assert_eq!(summary.min_nanos, 1);
    assert_eq!(summary.max_nanos, 10);
    assert_eq!(summary.sum_nanos, 55);
    assert_eq!(summary.mean_nanos(), 5); // 55 / 10 = 5 (integer).

    // Nearest-rank, 1-indexed: rank = ceil(p * n), value = sorted[rank - 1].
    // p50: 0.50 * 10 = 5.0 -> rank 5 -> 5.
    // p90: 0.90 * 10 = 9.0 -> rank 9 -> 9.
    // p99: 0.99 * 10 = 9.9 -> rank 10 -> 10.
    // p999: 0.999 * 10 = 9.99 -> rank 10 -> 10.
    assert_eq!(summary.p50_nanos, 5);
    assert_eq!(summary.p90_nanos, 9);
    assert_eq!(summary.p99_nanos, 10);
    assert_eq!(summary.p999_nanos, 10);
    assert!(summary.is_populated());
}

#[test]
fn empty_summary_is_unpopulated_and_zeroed() {
    let summary = InstanceSummary::from_samples("empty", &[]);
    assert_eq!(summary.count, 0);
    assert_eq!(summary.min_nanos, 0);
    assert_eq!(summary.max_nanos, 0);
    assert_eq!(summary.sum_nanos, 0);
    assert_eq!(summary.mean_nanos(), 0);
    assert_eq!(summary.p50_nanos, 0);
    assert_eq!(summary.p99_nanos, 0);
    assert!(!summary.is_populated());
}

#[test]
fn frame_report_summarizes_via_record() {
    let mut report = InstanceFrameReport::new("shard-b");
    for v in [2u64, 4, 6, 8, 10] {
        report.record(v);
    }
    let summary = report.summarize();
    assert_eq!(summary.label, "shard-b");
    assert_eq!(summary.count, 5);
    assert_eq!(summary.min_nanos, 2);
    assert_eq!(summary.max_nanos, 10);
    assert_eq!(summary.sum_nanos, 30);
    // p50: 0.5 * 5 = 2.5 -> rank 3 -> sorted[2] = 6.
    assert_eq!(summary.p50_nanos, 6);
    // p99: 0.99 * 5 = 4.95 -> rank 5 -> 10.
    assert_eq!(summary.p99_nanos, 10);
}

// ---- pooled cluster distribution --------------------------------------------

#[test]
fn pooled_cluster_distribution_merges_raw_samples() {
    // Two instances whose merged, sorted samples are a known arithmetic set.
    let mut a = InstanceFrameReport::new("a");
    for v in [10u64, 20, 30, 40, 50] {
        a.record(v);
    }
    let mut b = InstanceFrameReport::new("b");
    for v in [15u64, 25, 35, 45, 55] {
        b.record(v);
    }

    let mut agg = ClusterAggregator::new();
    agg.add_report(a);
    agg.add_report(b);
    let report = agg.report();

    // Merged sorted: [10,15,20,25,30,35,40,45,50,55], n = 10.
    assert_eq!(report.total_samples, 10);
    assert_eq!(report.populated_instances, 2);
    assert_eq!(report.min_nanos, 10);
    assert_eq!(report.max_nanos, 55);
    // sum = 150 + 175 = 325, mean = 325 / 10 = 32.
    assert_eq!(report.mean_nanos, 32);
    // p50: rank 5 -> sorted[4] = 30.
    assert_eq!(report.p50_nanos, 30);
    // p99: 9.9 -> rank 10 -> 55. p999 likewise.
    assert_eq!(report.p99_nanos, 55);
    assert_eq!(report.p999_nanos, 55);

    // Per-instance p99s: a = 50, b = 55 -> sorted [50, 55];
    // median: 0.5 * 2 = 1.0 -> rank 1 -> 50.
    assert_eq!(report.median_instance_p99_nanos, 50);

    // Instances are label-sorted; median p99 (50 ns) is far below the 1 ms
    // floor, so nothing is flagged.
    assert_eq!(report.instances.len(), 2);
    assert_eq!(report.instances[0].label, "a");
    assert_eq!(report.instances[1].label, "b");
    assert!(!report.has_anomalies());
}

// ---- anomaly localization ---------------------------------------------------

/// Build an instance whose ten samples are all `value_ms` ms, so its p99 (and
/// min/max/mean) is exactly `value_ms` ms.
fn flat_instance(label: &str, value_ms: u64) -> InstanceFrameReport {
    let mut report = InstanceFrameReport::new(label);
    for _ in 0..10 {
        report.record(value_ms * MS);
    }
    report
}

/// A fleet of four healthy instances (p99 ~16 ms) plus one slow outlier
/// (p99 50 ms). Labels are chosen so `node-bad` sorts last.
fn anomaly_fleet() -> ClusterAggregator {
    let mut agg = ClusterAggregator::new();
    agg.add_report(flat_instance("node-1", 15));
    agg.add_report(flat_instance("node-2", 16));
    agg.add_report(flat_instance("node-3", 17));
    agg.add_report(flat_instance("node-4", 16));
    agg.add_report(flat_instance("node-bad", 50));
    agg
}

#[test]
fn robust_median_mad_flags_only_the_outlier() {
    let report = anomaly_fleet().report();

    // Per-instance p99s: [15,16,16,17,50] ms sorted; n = 5.
    // median: 0.5 * 5 = 2.5 -> rank 3 -> sorted[2] = 16 ms.
    assert_eq!(report.median_instance_p99_nanos, 16 * MS);

    // Factor gate: 1.5 * 16 ms = 24 ms. Only node-bad (50 ms) exceeds it.
    // MAD gate: abs devs from 16 ms = [1,0,0,1,34] ms sorted [0,0,1,1,34];
    // MAD = rank 3 -> 1 ms; node-bad excess 34 ms > 3 * 1 ms = 3 ms -> passes.
    assert!(report.has_anomalies());
    assert_eq!(report.anomalies.len(), 1);
    let anomaly = &report.anomalies[0];
    assert_eq!(anomaly.label, "node-bad");
    assert_eq!(anomaly.p99_nanos, 50 * MS);
    assert_eq!(anomaly.median_p99_nanos, 16 * MS);
    assert!((anomaly.ratio - (50.0 / 16.0)).abs() < 1e-9);

    assert!(report.is_anomalous("node-bad"));
    assert!(!report.is_anomalous("node-1"));
    assert!(report.instance("node-bad").is_some());
    assert!(report.instance("missing").is_none());
}

#[test]
fn two_outliers_both_flagged_worst_first() {
    // The median stays healthy even with two bad instances, so neither can mask
    // the other; both are flagged, worst (largest p99) first.
    let mut agg = ClusterAggregator::new();
    agg.add_report(flat_instance("n1", 15));
    agg.add_report(flat_instance("n2", 16));
    agg.add_report(flat_instance("n3", 16));
    agg.add_report(flat_instance("slow-a", 60));
    agg.add_report(flat_instance("slow-b", 40));
    let report = agg.report();

    // p99s [15,16,16,40,60] ms; median rank 3 -> 16 ms. 1.5 * 16 = 24 ms.
    assert_eq!(report.median_instance_p99_nanos, 16 * MS);
    assert_eq!(report.anomalies.len(), 2);
    // Worst first: slow-a (60 ms) before slow-b (40 ms).
    assert_eq!(report.anomalies[0].label, "slow-a");
    assert_eq!(report.anomalies[1].label, "slow-b");
}

#[test]
fn single_instance_is_never_flagged() {
    // Fewer than two populated instances: no robust baseline, no flags.
    let mut agg = ClusterAggregator::new();
    agg.add_report(flat_instance("solo", 100));
    let report = agg.report();
    assert_eq!(report.populated_instances, 1);
    assert!(!report.has_anomalies());
}

#[test]
fn tiny_jitter_fleet_below_floor_is_not_flagged() {
    // All instances fast (microsecond-scale); median p99 is below the 1 ms
    // floor, so even a relatively large multiple raises no noise.
    let mut agg = ClusterAggregator::new();
    for (i, label) in ["m1", "m2", "m3", "m4"].iter().enumerate() {
        let mut report = InstanceFrameReport::new(*label);
        // 100 us baseline, one instance a touch higher.
        let base = 100_000 + (i as u64) * 1_000;
        for _ in 0..10 {
            report.record(base);
        }
        agg.add_report(report);
    }
    let report = agg.report();
    assert!(report.median_instance_p99_nanos < 1_000_000);
    assert!(!report.has_anomalies());
}

#[test]
fn wide_fleet_mad_gate_suppresses_false_positive() {
    // A naturally wide fleet (p99s 10/20/30/40 ms): the top instance exceeds
    // 1.5x the median but the large MAD keeps it from being flagged.
    let mut agg = ClusterAggregator::new();
    agg.add_report(flat_instance("w1", 10));
    agg.add_report(flat_instance("w2", 20));
    agg.add_report(flat_instance("w3", 30));
    agg.add_report(flat_instance("w4", 40));
    let report = agg.report();

    // p99s [10,20,30,40] ms; median rank 2 -> sorted[1] = 20 ms.
    assert_eq!(report.median_instance_p99_nanos, 20 * MS);
    // Factor gate: 1.5 * 20 = 30 ms; w4 (40 ms) exceeds it.
    // MAD: abs devs [10,0,10,20] ms sorted [0,10,10,20]; MAD rank 2 -> 10 ms.
    // w4 excess = 20 ms, not > 3 * 10 ms = 30 ms -> MAD gate suppresses it.
    assert!(!report.has_anomalies());
}

#[test]
fn disabled_mad_gate_flags_on_factor_alone() {
    // With mad_factor = 0 the secondary gate is off, so the same wide fleet now
    // flags on the factor test alone.
    let config = AnomalyConfig {
        mad_factor: 0.0,
        ..AnomalyConfig::default()
    };
    let mut agg = ClusterAggregator::with_config(config);
    agg.add_report(flat_instance("w1", 10));
    agg.add_report(flat_instance("w2", 20));
    agg.add_report(flat_instance("w3", 30));
    agg.add_report(flat_instance("w4", 40));
    let report = agg.report();
    assert!(report.is_anomalous("w4"));
}

#[test]
fn edge_reduced_summary_contributes_to_detection() {
    // One instance ships raw samples, another ships only a precomputed summary;
    // both participate in anomaly detection.
    let mut agg = ClusterAggregator::new();
    agg.add_report(flat_instance("raw-1", 15));
    agg.add_report(flat_instance("raw-2", 16));
    agg.add_report(flat_instance("raw-3", 17));
    // Edge-reduced bad instance (summary only, no raw samples).
    let bad = InstanceFrameReport::new("edge-bad");
    let mut bad = bad;
    for _ in 0..10 {
        bad.record(55 * MS);
    }
    agg.add_summary(bad.summarize());
    let report = agg.report();

    // Four populated instances; summary-only instance still flagged.
    assert_eq!(report.populated_instances, 4);
    assert!(report.is_anomalous("edge-bad"));
    // Pooled percentiles only draw on raw instances (summary carries no raw
    // samples): pooled p99 reflects the three raw instances, max 17 ms.
    assert_eq!(report.max_nanos, 55 * MS); // min/max/count span both kinds.
    assert_eq!(report.p99_nanos, 17 * MS); // pooled percentiles: raw-only.
}

// ---- distributed-trace assembly ---------------------------------------------

/// The primary trace id used throughout the trace tests.
const TRACE: TraceId = TraceId(0xAAAA);

/// client (span 1) -> server (span 2) -> db (span 3), ingested out of order.
fn linear_trace() -> TraceAssembler {
    let mut asm = TraceAssembler::new();
    // Ingest leaf-first to prove ordering is derived, not ingestion order.
    asm.ingest(DistributedSpan::new(
        TRACE,
        SpanId(3),
        Some(SpanId(2)),
        "db",
        "query",
        SpanKind::Db,
        20,
        40,
    ));
    asm.ingest(DistributedSpan::new(
        TRACE,
        SpanId(1),
        None,
        "edge",
        "GET /world",
        SpanKind::Client,
        0,
        100,
    ));
    asm.ingest(DistributedSpan::new(
        TRACE,
        SpanId(2),
        Some(SpanId(1)),
        "world",
        "handle",
        SpanKind::Server,
        10,
        70,
    ));
    asm
}

#[test]
fn trace_assembles_parent_child_forest() {
    let asm = linear_trace();
    assert_eq!(asm.len(), 3);
    assert!(!asm.is_empty());
    assert_eq!(asm.trace_ids(), alloc::vec![TRACE]);

    let trace = asm.assemble(TRACE).expect("trace present");
    assert_eq!(trace.span_count(), 3);
    assert_eq!(trace.orphan_count(), 0);

    // Nodes are kept in ingestion order: [span3, span1, span2].
    assert_eq!(trace.nodes[0].span.span_id, SpanId(3));
    assert_eq!(trace.nodes[1].span.span_id, SpanId(1));
    assert_eq!(trace.nodes[2].span.span_id, SpanId(2));

    // Single true root: span 1 (node index 1).
    assert_eq!(trace.roots, alloc::vec![1]);
    // span1 -> span2 (node 2); span2 -> span3 (node 0); span3 is a leaf.
    assert_eq!(trace.nodes[1].children, alloc::vec![2]);
    assert_eq!(trace.nodes[2].children, alloc::vec![0]);
    assert!(trace.nodes[0].children.is_empty());
    assert!(trace.node(5).is_none());
}

#[test]
fn trace_critical_path_and_wall_time() {
    let trace = linear_trace().assemble(TRACE).expect("trace present");

    // Critical path root->leaf: span1(100) + span2(70) + span3(40) = 210.
    let path = trace.critical_path();
    assert_eq!(path.nodes, alloc::vec![1, 2, 0]);
    assert_eq!(path.total_nanos, 210);

    // Wall clock: min start 0 (span1), max end 100 (span1) -> 100.
    assert_eq!(trace.wall_nanos(), 100);
}

#[test]
fn trace_service_latencies_sorted_desc() {
    let trace = linear_trace().assemble(TRACE).expect("trace present");
    let services = trace.service_latencies();
    assert_eq!(services.len(), 3);
    // Desc by total duration: edge 100, world 70, db 40.
    assert_eq!(services[0].instance, "edge");
    assert_eq!(services[0].total_nanos, 100);
    assert_eq!(services[0].span_count, 1);
    assert_eq!(services[1].instance, "world");
    assert_eq!(services[1].total_nanos, 70);
    assert_eq!(services[2].instance, "db");
    assert_eq!(services[2].total_nanos, 40);
}

#[test]
fn orphan_span_becomes_flagged_root() {
    let orphan_trace = TraceId(0xBBBB);
    let mut asm = TraceAssembler::new();
    asm.ingest(DistributedSpan::new(
        orphan_trace,
        SpanId(5),
        Some(SpanId(999)), // parent never arrives
        "x",
        "work",
        SpanKind::Internal,
        5,
        10,
    ));
    let trace = asm.assemble(orphan_trace).expect("trace present");
    assert_eq!(trace.span_count(), 1);
    assert_eq!(trace.orphan_count(), 1);
    assert_eq!(trace.roots, alloc::vec![0]);
    assert!(trace.nodes[0].orphan);
}

#[test]
fn duplicate_span_id_keeps_first_ingested() {
    let dup_trace = TraceId(0xCCCC);
    let mut asm = TraceAssembler::new();
    asm.ingest(DistributedSpan::new(
        dup_trace,
        SpanId(7),
        None,
        "svc",
        "first",
        SpanKind::Internal,
        0,
        10,
    ));
    asm.ingest(DistributedSpan::new(
        dup_trace,
        SpanId(7),
        None,
        "svc",
        "second",
        SpanKind::Internal,
        5,
        20,
    ));
    let trace = asm.assemble(dup_trace).expect("trace present");
    assert_eq!(trace.span_count(), 1);
    assert_eq!(trace.nodes[0].span.operation, "first");
}

#[test]
fn assemble_all_orders_by_trace_id() {
    let mut asm = TraceAssembler::new();
    asm.ingest(DistributedSpan::new(
        TraceId(0xAAAA),
        SpanId(1),
        None,
        "a",
        "op",
        SpanKind::Internal,
        0,
        10,
    ));
    asm.ingest(DistributedSpan::new(
        TraceId(0x1111),
        SpanId(1),
        None,
        "b",
        "op",
        SpanKind::Internal,
        0,
        10,
    ));
    let all = asm.assemble_all();
    assert_eq!(all.len(), 2);
    // Ascending trace id: 0x1111 before 0xAAAA.
    assert_eq!(all[0].trace_id, TraceId(0x1111));
    assert_eq!(all[1].trace_id, TraceId(0xAAAA));
    assert!(asm.assemble(TraceId(0xDEAD)).is_none());
}

// ---- adaptive sampling rate -------------------------------------------------

#[test]
fn sample_rate_report_pattern_and_counts() {
    let full = SampleRate::FULL;
    assert!(full.is_full());
    assert_eq!(full.divisor(), 1);
    assert_eq!(full.expected_reports(100), 100);

    let r = SampleRate::one_in(16);
    assert!(!r.is_full());
    assert_eq!(r.divisor(), 16);
    assert!((r.fraction() - 1.0 / 16.0).abs() < 1e-9);

    // should_report: seq multiple of 16.
    assert!(r.should_report(0));
    assert!(!r.should_report(1));
    assert!(r.should_report(16));
    assert!(r.should_report(32));

    // expected_reports = 0 if captured == 0 else (captured - 1) / 16 + 1.
    assert_eq!(r.expected_reports(0), 0);
    assert_eq!(r.expected_reports(1), 1);
    assert_eq!(r.expected_reports(16), 1);
    assert_eq!(r.expected_reports(17), 2);
    assert_eq!(r.expected_reports(100), 7); // (99 / 16) + 1 = 6 + 1.

    // A zero divisor clamps to full resolution.
    assert!(SampleRate::one_in(0).is_full());
}

#[test]
fn controller_boosts_only_anomalous_instances() {
    let report = anomaly_fleet().report();
    let controller = SamplingController::new();
    let decisions = controller.evaluate(&report);

    // One decision per instance, in label-sorted order (node-bad last).
    let labels: Vec<&str> = decisions.iter().map(|d| d.instance.as_str()).collect();
    assert_eq!(
        labels,
        alloc::vec!["node-1", "node-2", "node-3", "node-4", "node-bad"]
    );

    for decision in &decisions[..4] {
        assert_eq!(decision.rate, SampleRate::one_in(16));
        assert!(!decision.boosted);
        assert_eq!(decision.reason, SampleReason::Baseline);
    }
    let bad = &decisions[4];
    assert_eq!(bad.instance, "node-bad");
    assert_eq!(bad.rate, SampleRate::FULL);
    assert!(bad.boosted);
    assert_eq!(bad.reason, SampleReason::AnomalyBoost);
}

#[test]
fn controller_baseline_override_applies() {
    let report = anomaly_fleet().report();
    let mut controller = SamplingController::new();
    controller.set_baseline("node-1", SampleRate::one_in(4));
    // Overwriting the same label replaces (does not duplicate).
    controller.set_baseline("node-1", SampleRate::one_in(8));
    assert_eq!(controller.baseline_for("node-1"), SampleRate::one_in(8));
    assert_eq!(controller.baseline_for("node-2"), SampleRate::one_in(16));

    let decisions = controller.evaluate(&report);
    let node1 = decisions.iter().find(|d| d.instance == "node-1").unwrap();
    assert_eq!(node1.rate, SampleRate::one_in(8));
    assert!(!node1.boosted);
}

#[test]
fn boost_never_reduces_resolution_below_baseline() {
    // A VIP shard whose baseline is already full resolution, flagged anomalous:
    // the "boost" to FULL is not a boost, so it stays baseline.
    let report = anomaly_fleet().report();
    let mut controller = SamplingController::new();
    controller.set_baseline("node-bad", SampleRate::FULL);
    let decisions = controller.evaluate(&report);
    let bad = decisions.iter().find(|d| d.instance == "node-bad").unwrap();
    assert_eq!(bad.rate, SampleRate::FULL);
    assert!(!bad.boosted);
    assert_eq!(bad.reason, SampleReason::Baseline);
}

#[test]
fn custom_policy_boost_rate_is_honored() {
    // A policy whose boost is a finer-but-not-full rate still boosts a flagged
    // instance above the baseline.
    let policy = SamplingPolicy {
        base: SampleRate::one_in(32),
        boosted: SampleRate::one_in(4),
    };
    let report = anomaly_fleet().report();
    let controller = SamplingController::with_policy(policy);
    assert_eq!(controller.policy().base, SampleRate::one_in(32));
    let decisions = controller.evaluate(&report);
    let bad = decisions.iter().find(|d| d.instance == "node-bad").unwrap();
    assert_eq!(bad.rate, SampleRate::one_in(4));
    assert!(bad.boosted);
    assert_eq!(bad.reason, SampleReason::AnomalyBoost);
}

#[test]
fn fleet_report_estimate_sums_expected_reports() {
    let report = anomaly_fleet().report();
    let controller = SamplingController::new();
    let decisions = controller.evaluate(&report);

    // node-1 at baseline one_in(16): 100 captured -> 7 reports.
    // node-bad boosted to FULL: 100 captured -> 100 reports.
    // Instances with no captured entry contribute 0.
    let captured = alloc::vec![("node-1", 100u64), ("node-bad", 100u64)];
    let total = estimate_fleet_reports(&decisions, &captured);
    assert_eq!(total, 7 + 100);
}

extern crate alloc;
