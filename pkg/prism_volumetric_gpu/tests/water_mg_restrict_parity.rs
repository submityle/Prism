//! Real-device parity for the multigrid full-weighting restriction twin:
//! [`GpuWaterMgRestrict`](prism_volumetric_gpu::water_mg_restrict::GpuWaterMgRestrict)
//! must reproduce the `CPU` golden
//! [`restrict_full_weighting`](prism_render_architecture::water::pressure_multigrid::restrict_full_weighting)
//! across the `nf in {3, 5, 9}` grid sizes plus a randomized batch compared
//! value-for-value.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`restrict_full_weighting`](prism_render_architecture::water::pressure_multigrid::restrict_full_weighting)
//! is `pub`, so each `GPU` coarse field is pinned directly against the golden
//! run on the same fine field.
//!
//! # Parity criterion
//!
//! Every coarse value is a continuous `f32` (a `27`-tap weighted sum), asserted
//! within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The coarse length `nc^3` is
//! an integer and asserted with exact `==`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pressure_multigrid`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::pressure_multigrid::restrict_full_weighting;
use prism_volumetric_gpu::water_mg_restrict::{
    GpuWaterMgRestrict, WaterMgRestrictQuery, WaterMgRestrictResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on each coarse value.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Computes the golden coarse field for one query.
fn expected(q: &WaterMgRestrictQuery) -> WaterMgRestrictResult {
    WaterMgRestrictResult {
        coarse: restrict_full_weighting(&q.fine, q.nf as usize),
    }
}

/// Pins one `GPU` result against the golden oracle: equal length and each value
/// within tolerance.
fn assert_result(idx: usize, got: &WaterMgRestrictResult, want: &WaterMgRestrictResult) {
    assert_eq!(
        got.coarse.len(),
        want.coarse.len(),
        "result {idx}: coarse length"
    );
    for (j, (g, w)) in got.coarse.iter().zip(want.coarse.iter()).enumerate() {
        assert!(
            close(*g, *w),
            "result {idx} coarse[{j}]: gpu {g} vs cpu {w}"
        );
    }
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[WaterMgRestrictQuery]) {
    let gpu = GpuWaterMgRestrict::new(ctx);
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        assert_result(idx, g, &expected(q));
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a float in `[0, 1)` from `state` using only integer work.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32
}

/// Draws a float in `[lo, hi)` from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * unit(state)
}

/// Builds a random fine field of length `nf^3` with values in `[-4, 4]`.
fn random_fine(state: &mut u64, nf: usize) -> Vec<f32> {
    let n = nf * nf * nf;
    let mut fine = Vec::with_capacity(n);
    for _ in 0..n {
        fine.push(ranged(state, -4.0, 4.0));
    }
    fine
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_mg_restrict parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterMgRestrict::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn restriction_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x1234_5678_9abc_def0_u64;
    let queries = vec![
        // nf = 3 -> nc = 2: no interior coarse node, so the field is all zero.
        WaterMgRestrictQuery {
            fine: random_fine(&mut state, 3),
            nf: 3,
        },
        // nf = 5 -> nc = 3: a single interior coarse node at (1, 1, 1).
        WaterMgRestrictQuery {
            fine: random_fine(&mut state, 5),
            nf: 5,
        },
        // nf = 9 -> nc = 5: a 3x3x3 block of interior coarse nodes.
        WaterMgRestrictQuery {
            fine: random_fine(&mut state, 9),
            nf: 9,
        },
        // A smooth ramp on nf = 5 exercises the exact separable weights.
        WaterMgRestrictQuery {
            fine: {
                let mut f = Vec::new();
                for z in 0..5 {
                    for y in 0..5 {
                        for x in 0..5 {
                            f.push((x + 2 * y + 3 * z) as f32);
                        }
                    }
                }
                f
            },
            nf: 5,
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x0fe1_dc23_ab45_6789_u64;
    let sizes = [3usize, 5, 9];

    let mut queries: Vec<WaterMgRestrictQuery> = Vec::new();
    while queries.len() < 512 {
        let nf = sizes[(lcg(&mut state) % 3) as usize];
        queries.push(WaterMgRestrictQuery {
            fine: random_fine(&mut state, nf),
            nf: nf as u32,
        });
    }
    run_and_check(&ctx, &queries);
}
