//! Real-device parity for the frame-time budget twin:
//! [`GpuPerfBudget`](prism_volumetric_gpu::perf_budget::GpuPerfBudget) must
//! reproduce the `CPU` golden
//! [`perf_budget`](prism_render_architecture::particle::perf_budget) across every
//! numeric term it exposes: the per-stage/total overspend evaluation
//! ([`FrameTimeBudget::evaluate`](prism_render_architecture::particle::perf_budget::FrameTimeBudget::evaluate)),
//! the per-stage target sum
//! ([`FrameTimeBudget::stage_sum_ms`](prism_render_architecture::particle::perf_budget::FrameTimeBudget::stage_sum_ms)),
//! the per-stage measured sum
//! ([`FrameTimeSample::total_ms`](prism_render_architecture::particle::perf_budget::FrameTimeSample::total_ms)),
//! the within-total predicate
//! ([`FrameTimeReport::is_within_total`](prism_render_architecture::particle::perf_budget::FrameTimeReport::is_within_total)),
//! the any-stage-over predicate
//! ([`FrameTimeReport::any_stage_over`](prism_render_architecture::particle::perf_budget::FrameTimeReport::any_stage_over)),
//! the guarded overspend ratio
//! ([`FrameTimeReport::overspend_ratio`](prism_render_architecture::particle::perf_budget::FrameTimeReport::overspend_ratio)),
//! and the microsecond-to-millisecond conversion
//! ([`micros_to_ms`](prism_render_architecture::particle::perf_budget::micros_to_ms)).
//!
//! The fixtures exercise one dedicated query per term plus a randomized mixed
//! batch compared element for element. Every scalar is an interior value held
//! well away from its guard branch: measured and target stage times are normal
//! millisecond magnitudes, the within-total and any-stage-over fixtures land
//! either clearly within the `BUDGET_EPS` guard or clearly past it, the ratio
//! fixtures use a target comfortably above the guard, and the microsecond counts
//! stay in a range an `f32` represents exactly.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each continuous term is a sum, difference, clamp or single guarded divide of
//! its inputs, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every continuous `f32` lane; the
//! boolean verdicts are compared exactly.
//!
//! # Conditioning
//!
//! Every fixture is kept clear of the guard cracks: a total overspend is either
//! exactly `0.0` or comfortably above `BUDGET_EPS`, a per-stage overspend array
//! is all-zero or has at least one element far past the guard, the overspend
//! ratio uses a target at least `1.0` so it is far from the near-zero divide
//! guard, and every stage time is a normal millisecond magnitude so no fixture
//! lands on a classification tie.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::perf_budget`；
//! 无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::perf_budget::{
    micros_to_ms, FrameTimeBudget, FrameTimeReport, FrameTimeSample,
};
use prism_volumetric_gpu::perf_budget::{GpuPerfBudget, PerfBudgetQuery, PerfBudgetResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Number of accountable frame stages, mirroring `FrameStage::COUNT`.
const STAGE_COUNT: usize = 6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// Computes the reference result for `query` through the `CPU` golden.
fn golden_result(query: &PerfBudgetQuery) -> PerfBudgetResult {
    match *query {
        PerfBudgetQuery::Evaluate {
            stage_ms,
            stage_targets_ms,
            total_target_ms,
        } => {
            let budget = FrameTimeBudget {
                stage_targets_ms,
                total_target_ms,
            };
            let report = budget.evaluate(&FrameTimeSample { stage_ms });
            PerfBudgetResult::Evaluate {
                stage_overspend_ms: report.stage_overspend_ms,
                total_measured_ms: report.total_measured_ms,
                total_overspend_ms: report.total_overspend_ms,
            }
        }
        PerfBudgetQuery::StageTargetSum { stage_targets_ms } => {
            let budget = FrameTimeBudget {
                stage_targets_ms,
                total_target_ms: 0.0,
            };
            PerfBudgetResult::StageTargetSum {
                sum_ms: budget.stage_sum_ms(),
            }
        }
        PerfBudgetQuery::SampleTotalMs { stage_ms } => PerfBudgetResult::SampleTotalMs {
            sum_ms: FrameTimeSample { stage_ms }.total_ms(),
        },
        PerfBudgetQuery::IsWithinTotal { total_overspend_ms } => {
            let report = FrameTimeReport {
                stage_overspend_ms: [0.0; STAGE_COUNT],
                total_measured_ms: 0.0,
                total_target_ms: 0.0,
                total_overspend_ms,
            };
            PerfBudgetResult::IsWithinTotal {
                within: report.is_within_total(),
            }
        }
        PerfBudgetQuery::AnyStageOver { stage_overspend_ms } => {
            let report = FrameTimeReport {
                stage_overspend_ms,
                total_measured_ms: 0.0,
                total_target_ms: 0.0,
                total_overspend_ms: 0.0,
            };
            PerfBudgetResult::AnyStageOver {
                any_over: report.any_stage_over(),
            }
        }
        PerfBudgetQuery::OverspendRatio {
            total_overspend_ms,
            total_target_ms,
        } => {
            let report = FrameTimeReport {
                stage_overspend_ms: [0.0; STAGE_COUNT],
                total_measured_ms: 0.0,
                total_target_ms,
                total_overspend_ms,
            };
            PerfBudgetResult::OverspendRatio {
                ratio: report.overspend_ratio(),
            }
        }
        PerfBudgetQuery::MicrosToMs { micros } => PerfBudgetResult::MicrosToMs {
            ms: micros_to_ms(micros),
        },
    }
}

/// Pins one `GPU` result against the reference for `query`.
fn pin(idx: usize, query: &PerfBudgetQuery, got: &PerfBudgetResult) {
    let want = golden_result(query);
    match (got, want) {
        (
            PerfBudgetResult::Evaluate {
                stage_overspend_ms: go,
                total_measured_ms: gm,
                total_overspend_ms: gt,
            },
            PerfBudgetResult::Evaluate {
                stage_overspend_ms: wo,
                total_measured_ms: wm,
                total_overspend_ms: wt,
            },
        ) => {
            for k in 0..STAGE_COUNT {
                assert!(
                    close(go[k], wo[k]),
                    "query {idx} stage overspend[{k}]: gpu {} vs cpu {}",
                    go[k],
                    wo[k]
                );
            }
            assert!(
                close(*gm, wm),
                "query {idx} total measured: gpu {gm} vs cpu {wm}"
            );
            assert!(
                close(*gt, wt),
                "query {idx} total overspend: gpu {gt} vs cpu {wt}"
            );
        }
        (
            PerfBudgetResult::StageTargetSum { sum_ms: g },
            PerfBudgetResult::StageTargetSum { sum_ms: w },
        )
        | (
            PerfBudgetResult::SampleTotalMs { sum_ms: g },
            PerfBudgetResult::SampleTotalMs { sum_ms: w },
        ) => {
            assert!(close(*g, w), "query {idx} sum: gpu {g} vs cpu {w}");
        }
        (
            PerfBudgetResult::IsWithinTotal { within: g },
            PerfBudgetResult::IsWithinTotal { within: w },
        ) => {
            assert_eq!(*g, w, "query {idx} within-total: gpu {g} vs cpu {w}");
        }
        (
            PerfBudgetResult::AnyStageOver { any_over: g },
            PerfBudgetResult::AnyStageOver { any_over: w },
        ) => {
            assert_eq!(*g, w, "query {idx} any-stage-over: gpu {g} vs cpu {w}");
        }
        (
            PerfBudgetResult::OverspendRatio { ratio: g },
            PerfBudgetResult::OverspendRatio { ratio: w },
        ) => {
            assert!(close(*g, w), "query {idx} ratio: gpu {g} vs cpu {w}");
        }
        (PerfBudgetResult::MicrosToMs { ms: g }, PerfBudgetResult::MicrosToMs { ms: w }) => {
            assert!(close(*g, w), "query {idx} micros_to_ms: gpu {g} vs cpu {w}");
        }
        (g, w) => panic!("query {idx} result variant mismatch: gpu {g:?} vs cpu {w:?}"),
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuPerfBudget, queries: &[PerfBudgetQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPerfBudget::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn evaluate_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPerfBudget::new(&ctx);
    let queries = vec![
        // A sample that fits every stage and the total: zero overspend.
        PerfBudgetQuery::Evaluate {
            stage_ms: [0.8, 1.2, 0.3, 0.1, 0.6, 1.1],
            stage_targets_ms: [1.0, 1.5, 0.4, 0.2, 0.8, 1.5],
            total_target_ms: 5.5,
        },
        // A sample that overspends several stages and the total.
        PerfBudgetQuery::Evaluate {
            stage_ms: [1.4, 2.1, 0.3, 0.5, 0.6, 2.0],
            stage_targets_ms: [1.0, 1.5, 0.4, 0.2, 0.8, 1.5],
            total_target_ms: 5.5,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn stage_target_sum_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPerfBudget::new(&ctx);
    let queries = vec![
        PerfBudgetQuery::StageTargetSum {
            stage_targets_ms: [1.0, 1.5, 0.4, 0.2, 0.8, 1.5],
        },
        PerfBudgetQuery::StageTargetSum {
            stage_targets_ms: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn sample_total_ms_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPerfBudget::new(&ctx);
    let queries = vec![
        PerfBudgetQuery::SampleTotalMs {
            stage_ms: [0.8, 1.2, 0.3, 0.1, 0.6, 1.1],
        },
        PerfBudgetQuery::SampleTotalMs {
            stage_ms: [2.0, 0.4, 0.9, 0.2, 1.3, 0.7],
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn is_within_total_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPerfBudget::new(&ctx);
    let queries = vec![
        // Clearly within budget (exactly zero overspend).
        PerfBudgetQuery::IsWithinTotal {
            total_overspend_ms: 0.0,
        },
        // Clearly over budget, far past the EPS guard.
        PerfBudgetQuery::IsWithinTotal {
            total_overspend_ms: 0.75,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn any_stage_over_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPerfBudget::new(&ctx);
    let queries = vec![
        // No stage over its target.
        PerfBudgetQuery::AnyStageOver {
            stage_overspend_ms: [0.0; STAGE_COUNT],
        },
        // One stage comfortably over the EPS guard.
        PerfBudgetQuery::AnyStageOver {
            stage_overspend_ms: [0.0, 0.0, 0.4, 0.0, 0.0, 0.0],
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn overspend_ratio_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPerfBudget::new(&ctx);
    let queries = vec![
        PerfBudgetQuery::OverspendRatio {
            total_overspend_ms: 0.55,
            total_target_ms: 5.5,
        },
        PerfBudgetQuery::OverspendRatio {
            total_overspend_ms: 0.0,
            total_target_ms: 5.5,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn micros_to_ms_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPerfBudget::new(&ctx);
    let queries = vec![
        PerfBudgetQuery::MicrosToMs { micros: 1_500 },
        PerfBudgetQuery::MicrosToMs { micros: 0 },
        PerfBudgetQuery::MicrosToMs { micros: 250_000 },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn randomized_mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPerfBudget::new(&ctx);
    let mut state: u64 = 0x5eed_1234_abcd_f00d;
    let mut queries = Vec::new();
    for _ in 0..64 {
        let pick = (lcg(&mut state) * 7.0) as u32;
        let query = match pick {
            0 => {
                let mut stage_ms = [0.0_f32; STAGE_COUNT];
                let mut stage_targets_ms = [0.0_f32; STAGE_COUNT];
                for k in 0..STAGE_COUNT {
                    stage_ms[k] = ranged(&mut state, 0.1, 3.0);
                    stage_targets_ms[k] = ranged(&mut state, 0.1, 3.0);
                }
                PerfBudgetQuery::Evaluate {
                    stage_ms,
                    stage_targets_ms,
                    total_target_ms: ranged(&mut state, 3.0, 7.0),
                }
            }
            1 => {
                let mut stage_targets_ms = [0.0_f32; STAGE_COUNT];
                for target in &mut stage_targets_ms {
                    *target = ranged(&mut state, 0.1, 3.0);
                }
                PerfBudgetQuery::StageTargetSum { stage_targets_ms }
            }
            2 => {
                let mut stage_ms = [0.0_f32; STAGE_COUNT];
                for ms in &mut stage_ms {
                    *ms = ranged(&mut state, 0.1, 3.0);
                }
                PerfBudgetQuery::SampleTotalMs { stage_ms }
            }
            3 => {
                // Either clearly within (0.0) or clearly over (>= 0.1).
                let over = if lcg(&mut state) < 0.5 {
                    0.0
                } else {
                    ranged(&mut state, 0.1, 2.0)
                };
                PerfBudgetQuery::IsWithinTotal {
                    total_overspend_ms: over,
                }
            }
            4 => {
                let mut stage_overspend_ms = [0.0_f32; STAGE_COUNT];
                // Optionally push one stage comfortably past the guard.
                if lcg(&mut state) < 0.5 {
                    let slot = (lcg(&mut state) * STAGE_COUNT as f32) as usize;
                    let slot = slot.min(STAGE_COUNT - 1);
                    stage_overspend_ms[slot] = ranged(&mut state, 0.1, 1.0);
                }
                PerfBudgetQuery::AnyStageOver { stage_overspend_ms }
            }
            5 => PerfBudgetQuery::OverspendRatio {
                total_overspend_ms: ranged(&mut state, 0.0, 2.0),
                total_target_ms: ranged(&mut state, 1.0, 7.0),
            },
            _ => PerfBudgetQuery::MicrosToMs {
                micros: (ranged(&mut state, 0.0, 5_000_000.0)) as u32,
            },
        };
        queries.push(query);
    }
    check(&ctx, &gpu, &queries);
}
