//! §24.8 backpressure / deadlock-prevention / health-monitoring tests.
//!
//! Anti-vacuous contract: each deterministic core is checked against an
//! *independent* serial oracle —
//!
//! - [`QueueBackpressure`](crate::QueueBackpressure) admission sequences vs a
//!   from-scratch watermark/hysteresis replay;
//! - [`WaitGraph`](crate::WaitGraph) cycle admission vs brute-force
//!   reachability over the edge set;
//! - [`LatencyHistogram`](crate::LatencyHistogram) percentiles vs a nearest-rank
//!   computation over the raw samples;
//! - [`StarvationDetector`](crate::StarvationDetector) /
//!   [`StealStats`](crate::StealStats) vs direct recomputation.
//!
//! The [`BackpressureQueue`](crate::BackpressureQueue) and
//! [`HealthProbe`](crate::HealthProbe) façades are then exercised on a real
//! [`TaskPool`](crate::TaskPool).

use alloc::vec;
use alloc::vec::Vec;
use alloc::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::health::wait_graph::DeadlockError;
use crate::{
    Admission, BackpressureLimits, LatencyHistogram, Priority, QueueBackpressure,
    StarvationDetector, StealStats, TaskPool, WaitGraph, WaitNodeId,
};

// ----------------------------------------------------------------------------
// Backpressure core (oracle-checked).
// ----------------------------------------------------------------------------

/// Independent admission oracle: replays the submission sequence from scratch,
/// mirroring the capacity cap, high-watermark background shedding, and
/// low-watermark hysteresis without consulting [`QueueBackpressure`].
fn backpressure_oracle(limits: BackpressureLimits, submissions: &[Priority]) -> Vec<Admission> {
    let capacity = limits.capacity();
    let high = limits.high_watermark();
    let low = limits.low_watermark();
    let mut depth = 0usize;
    let mut shedding = false;
    let mut out = Vec::new();
    for &p in submissions {
        let decision = if depth >= capacity {
            Admission::Rejected
        } else if p == Priority::Background && (shedding || depth >= high) {
            Admission::Deferred
        } else {
            Admission::Admitted
        };
        if decision.is_admitted() {
            depth += 1;
            if depth >= high {
                shedding = true;
            } else if depth <= low {
                shedding = false;
            }
        }
        out.push(decision);
    }
    out
}

fn drive_backpressure(limits: BackpressureLimits, submissions: &[Priority]) -> Vec<Admission> {
    let mut gate = QueueBackpressure::new(limits);
    submissions.iter().map(|&p| gate.offer(p)).collect()
}

#[test]
fn backpressure_limits_are_clamped() {
    let l = BackpressureLimits::new(0, 100, 50);
    assert_eq!(l.capacity(), 1);
    assert_eq!(l.high_watermark(), 1);
    assert_eq!(l.low_watermark(), 1);

    let l = BackpressureLimits::new(10, 20, 7);
    assert_eq!(l.capacity(), 10);
    assert_eq!(l.high_watermark(), 10);
    assert_eq!(l.low_watermark(), 7);

    let l = BackpressureLimits::new(10, 6, 9);
    assert_eq!(l.high_watermark(), 6);
    assert_eq!(l.low_watermark(), 6);
}

#[test]
fn background_sheds_at_high_watermark_but_foreground_flows() {
    let limits = BackpressureLimits::new(8, 6, 3);
    // Fill foreground to the high watermark, then background sheds while
    // Critical keeps being admitted up to capacity.
    let subs = [
        Priority::Normal, // depth 1
        Priority::Normal, // 2
        Priority::Normal, // 3
        Priority::Normal, // 4
        Priority::Normal, // 5
        Priority::Background, // 6? depth 5 < high 6 -> admitted -> depth 6 arms shedding
        Priority::Background, // shedding -> Deferred
        Priority::Critical,   // depth 6 -> admitted -> 7
        Priority::Critical,   // 7 -> admitted -> 8 (capacity)
        Priority::Critical,   // depth 8 == capacity -> Rejected
        Priority::Background, // Rejected (capacity)
    ];
    let got = drive_backpressure(limits, &subs);
    let want = backpressure_oracle(limits, &subs);
    assert_eq!(got, want);
    assert_eq!(got[5], Admission::Admitted);
    assert_eq!(got[6], Admission::Deferred);
    assert_eq!(got[7], Admission::Admitted);
    assert_eq!(got[9], Admission::Rejected);
    assert_eq!(got[10], Admission::Rejected);
}

#[test]
fn hysteresis_keeps_shedding_until_low_watermark() {
    let limits = BackpressureLimits::new(10, 6, 2);
    let mut gate = QueueBackpressure::new(limits);
    for _ in 0..6 {
        assert_eq!(gate.offer(Priority::Normal), Admission::Admitted);
    }
    assert!(gate.is_shedding());
    assert_eq!(gate.offer(Priority::Background), Admission::Deferred);
    // Drain to just above the low watermark: still shedding.
    gate.complete_many(3); // depth 3
    assert!(gate.is_shedding());
    assert_eq!(gate.offer(Priority::Background), Admission::Deferred);
    // Drain to the low watermark: shedding disarms.
    gate.complete_many(1); // depth 3 -> actually offer admitted nothing; depth is 3
    assert!(!gate.is_shedding());
    assert_eq!(gate.offer(Priority::Background), Admission::Admitted);
}

#[test]
fn randomised_backpressure_matches_oracle() {
    // Deterministic pseudo-random submission stream (LCG), several configs.
    let configs = [
        BackpressureLimits::new(16, 12, 4),
        BackpressureLimits::new(4, 3, 1),
        BackpressureLimits::new(32, 32, 0),
    ];
    for (ci, &limits) in configs.iter().enumerate() {
        let mut state = 0x1234_5678u32 ^ u32::try_from(ci).unwrap().wrapping_mul(0x9E37_79B1);
        let mut subs = Vec::new();
        for _ in 0..500 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Bias toward background so shedding is exercised.
            subs.push(match (state >> 16) % 5 {
                0 => Priority::Critical,
                1 => Priority::High,
                2 => Priority::Normal,
                _ => Priority::Background,
            });
        }
        assert_eq!(
            drive_backpressure(limits, &subs),
            backpressure_oracle(limits, &subs),
            "config {ci}"
        );
    }
}

// ----------------------------------------------------------------------------
// Wait-for graph / deadlock prevention (oracle-checked).
// ----------------------------------------------------------------------------

/// Brute-force reachability oracle over an explicit edge list.
fn reachable(edges: &[(usize, usize)], from: usize, to: usize, nodes: usize) -> bool {
    if from == to {
        return true;
    }
    let mut seen = vec![false; nodes];
    let mut stack = vec![from];
    seen[from] = true;
    while let Some(n) = stack.pop() {
        for &(a, b) in edges {
            if a == n && !seen[b] {
                if b == to {
                    return true;
                }
                seen[b] = true;
                stack.push(b);
            }
        }
    }
    false
}

#[test]
fn try_add_dependency_matches_reachability_oracle() {
    const N: usize = 8;
    let mut graph = WaitGraph::new();
    let ids: Vec<WaitNodeId> = (0..N).map(|_| graph.add_node()).collect();
    let mut edges: Vec<(usize, usize)> = Vec::new();

    // Deterministic stream of candidate edges.
    let mut state = 0xDEAD_BEEFu32;
    for _ in 0..200 {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let w = ((state >> 8) as usize) % N;
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let h = ((state >> 8) as usize) % N;

        // Oracle: adding w->h closes a cycle iff h already reaches w.
        let oracle_cycle = reachable(&edges, h, w, N) || w == h;
        let result = graph.try_add_dependency(ids[w], ids[h]);
        match result {
            Ok(()) => {
                assert!(!oracle_cycle, "accepted a cycle-forming edge {w}->{h}");
                if !edges.contains(&(w, h)) {
                    edges.push((w, h));
                }
            }
            Err(DeadlockError::Cycle { cycle }) => {
                assert!(oracle_cycle, "rejected an acyclic edge {w}->{h}");
                assert!(!cycle.is_empty());
            }
            Err(DeadlockError::ChainTooDeep { .. }) => {
                panic!("no chain limit configured");
            }
        }
    }
    // The accepted graph is acyclic.
    assert!(graph.find_cycle().is_none());
}

#[test]
fn find_cycle_locates_a_real_cycle() {
    let mut graph = WaitGraph::new();
    let a = graph.add_node();
    let b = graph.add_node();
    let c = graph.add_node();
    // Build a->b->c, then the closing edge c->a is rejected...
    assert!(graph.try_add_dependency(a, b).is_ok());
    assert!(graph.try_add_dependency(b, c).is_ok());
    let err = graph.try_add_dependency(c, a).unwrap_err();
    match err {
        DeadlockError::Cycle { cycle } => {
            // Reported path is a -> b -> c (the holder->...->waiter witness).
            assert_eq!(cycle, vec![a, b, c]);
        }
        DeadlockError::ChainTooDeep { .. } => panic!("unexpected"),
    }
    // Since the edge was rejected, the graph is still acyclic.
    assert!(graph.find_cycle().is_none());
    assert!(graph.creates_cycle(c, a));
    assert!(graph.creates_cycle(a, a));
}

#[test]
fn self_dependency_is_a_cycle() {
    let mut graph = WaitGraph::new();
    let a = graph.add_node();
    assert!(graph.creates_cycle(a, a));
    assert!(matches!(
        graph.try_add_dependency(a, a),
        Err(DeadlockError::Cycle { .. })
    ));
}

#[test]
fn chain_limit_rejects_overlong_wait_chains() {
    // Limit of 3 nodes per wait chain.
    let mut graph = WaitGraph::with_chain_limit(3);
    let n: Vec<WaitNodeId> = (0..5).map(|_| graph.add_node()).collect();
    assert!(graph.try_add_dependency(n[0], n[1]).is_ok()); // chain 0->1 (2 nodes)
    assert!(graph.try_add_dependency(n[1], n[2]).is_ok()); // chain 0->1->2 (3 nodes)
    assert_eq!(graph.longest_chain(), 3);
    // Extending to 4 nodes exceeds the limit.
    let err = graph.try_add_dependency(n[2], n[3]).unwrap_err();
    assert_eq!(err, DeadlockError::ChainTooDeep { depth: 4, limit: 3 });
    // A separate short chain is still fine.
    assert!(graph.try_add_dependency(n[3], n[4]).is_ok());
    assert_eq!(graph.longest_chain(), 3);
}

#[test]
fn remove_dependency_reopens_capacity() {
    let mut graph = WaitGraph::with_chain_limit(3);
    let n: Vec<WaitNodeId> = (0..4).map(|_| graph.add_node()).collect();
    assert!(graph.try_add_dependency(n[0], n[1]).is_ok());
    assert!(graph.try_add_dependency(n[1], n[2]).is_ok());
    assert!(graph.try_add_dependency(n[2], n[3]).is_err()); // too deep
    assert!(graph.remove_dependency(n[0], n[1]));
    // With the head edge gone, the chain fits again.
    assert!(graph.try_add_dependency(n[2], n[3]).is_ok());
}

// ----------------------------------------------------------------------------
// Latency histogram / percentiles (oracle-checked).
// ----------------------------------------------------------------------------

/// Nearest-rank percentile oracle computed directly over the recorded samples,
/// mapped onto the same bucket upper bounds the histogram reports.
fn percentile_oracle(bounds: &[u64], samples: &[u64], p: u8) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let p = u64::from(p.clamp(1, 100));
    let total = samples.len() as u64;
    let rank = (p * total).div_ceil(100);
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let value = sorted[(rank - 1) as usize];
    // Map the raw value to its bucket upper bound (overflow -> max sample).
    for &b in bounds {
        if value <= b {
            return b;
        }
    }
    *sorted.last().unwrap()
}

#[test]
fn histogram_percentiles_match_nearest_rank_oracle() {
    let bounds = [10u64, 50, 100, 500, 1000];
    let mut hist = LatencyHistogram::new(&bounds);
    // Deterministic sample stream.
    let mut samples = Vec::new();
    let mut state = 777u64;
    for _ in 0..1000 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (state >> 33) % 1200; // spans all buckets incl. overflow
        samples.push(v);
        hist.record(v);
    }
    assert_eq!(hist.total(), 1000);
    for p in [1u8, 25, 50, 75, 90, 99, 100] {
        assert_eq!(
            hist.percentile_nanos(p),
            percentile_oracle(&bounds, &samples, p),
            "p{p}"
        );
    }
    assert_eq!(hist.max_nanos(), *samples.iter().max().unwrap());
}

#[test]
fn empty_histogram_reports_zero() {
    let hist = LatencyHistogram::new(&[10, 20]);
    assert!(hist.is_empty());
    assert_eq!(hist.percentile_nanos(50), 0);
    assert_eq!(hist.max_nanos(), 0);
}

#[test]
fn histogram_unsorted_bounds_are_normalised() {
    let mut hist = LatencyHistogram::new(&[100, 10, 50, 10]);
    assert_eq!(hist.bounds(), &[10, 50, 100]);
    hist.record(5);
    hist.record(5);
    hist.record(60);
    // 2/3 of samples <= 10, so p50 lands in the first bucket.
    assert_eq!(hist.percentile_nanos(50), 10);
    // p99 reaches the 60 sample, which lands in the 100 bucket.
    assert_eq!(hist.percentile_nanos(99), 100);
}

// ----------------------------------------------------------------------------
// Starvation / steal stats (direct recomputation).
// ----------------------------------------------------------------------------

#[test]
fn starvation_detector_tracks_idle_streaks() {
    let mut d = StarvationDetector::new(3, 3);
    assert_eq!(d.worker_count(), 3);
    for _ in 0..3 {
        d.record_idle(0);
    }
    d.record_idle(1);
    d.record_idle(1);
    assert!(d.is_starving(0));
    assert!(!d.is_starving(1));
    assert_eq!(d.idle_streak(0), 3);
    assert_eq!(d.starving_workers(), 1);
    // Running a job resets the streak.
    d.record_busy(0);
    assert!(!d.is_starving(0));
    assert_eq!(d.starving_workers(), 0);
}

#[test]
fn steal_failure_rate_is_integer_per_mille() {
    let mut s = StealStats::new();
    for i in 0..1000 {
        s.record_attempt(i % 4 == 0); // 25% success -> 75% failure
    }
    assert_eq!(s.attempts(), 1000);
    assert_eq!(s.failures(), 750);
    assert_eq!(s.failure_per_mille(), 750);
    assert_eq!(StealStats::new().failure_per_mille(), 0);
}

// ----------------------------------------------------------------------------
// Façades over a real TaskPool.
// ----------------------------------------------------------------------------

#[test]
fn backpressure_queue_runs_admitted_jobs_and_drains_depth() {
    let pool = TaskPool::with_threads(4);
    let limits = BackpressureLimits::new(4, 3, 1);
    let mut queue = pool.backpressure_queue(limits);
    let ran = Arc::new(AtomicUsize::new(0));

    // Four foreground jobs fill to capacity; a fifth is rejected.
    for _ in 0..4 {
        let ran = Arc::clone(&ran);
        assert_eq!(
            queue.offer(Priority::Normal, move || {
                ran.fetch_add(1, Ordering::Relaxed);
            }),
            Admission::Admitted
        );
    }
    assert_eq!(queue.depth(), 4);
    let rejected = queue.offer(Priority::Normal, || {});
    assert_eq!(rejected, Admission::Rejected);
    // Background work is shed under pressure, too.
    assert_eq!(queue.offer(Priority::Background, || {}), Admission::Rejected);

    let report = queue.run();
    assert_eq!(report.ran, 4);
    assert_eq!(queue.depth(), 0);
    assert_eq!(queue.pending(), 0);
    assert_eq!(ran.load(Ordering::Relaxed), 4);
    assert!(!queue.is_shedding());

    // After draining, new work is admitted again.
    assert_eq!(queue.offer(Priority::Background, || {}), Admission::Admitted);
}

#[test]
fn health_probe_measures_real_latencies_and_reports() {
    let pool = TaskPool::with_threads(4);
    let probe = crate::HealthProbe::new(4, 4, &[1_000, 10_000, 100_000, 1_000_000]);
    probe.observe_queue_depth(7);

    pool.scope(|s| {
        for _ in 0..32 {
            let probe = probe.clone();
            s.spawn(move || {
                probe.measure(|| {
                    // A tiny, real unit of work so the clock sees nonzero time.
                    let mut acc = 0u64;
                    for i in 0..1000 {
                        acc = acc.wrapping_add(i);
                    }
                    core::hint::black_box(acc);
                });
            });
        }
    });

    let report = probe.report();
    assert_eq!(report.max_queue_depth, 7);
    // Exactly 32 real samples were recorded (deterministic, order-independent).
    assert_eq!(report.latency_sample_count, 32);
    // Nearest-rank percentiles are monotone: p50 never exceeds p99. This holds
    // regardless of OS scheduling, which only perturbs the raw sample values.
    assert!(report.p50_latency_nanos <= report.p99_latency_nanos);
}
