//! Real-device parity for the `APIC` velocity reconstruction twin:
//! [`GpuWaterFlipApic`](prism_volumetric_gpu::water_flip_apic::GpuWaterFlipApic)
//! must reproduce the `CPU` golden
//! [`apic_velocity`](prism_render_architecture::water::flip::apic_velocity)
//! across hand-picked fixtures plus a randomized batch compared
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
//! [`apic_velocity`](prism_render_architecture::water::flip::apic_velocity) is
//! `pub`, so each `GPU` reconstruction is pinned directly against the golden
//! run on the same base velocity, affine matrix, and offset.
//!
//! # Parity criterion
//!
//! Each output component is a continuous `f32` built from three products and an
//! add, so the parity test uses `abs <= 1e-5 || rel <= 1e-5` with a relative
//! floor of `1e-6`. A zero affine matrix must reproduce the base velocity
//! exactly, and a linear field is reproduced to tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::flip`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::flip::apic_velocity;
use prism_render_architecture::water::Vec3;
use prism_volumetric_gpu::water_flip_apic::{
    GpuWaterFlipApic, WaterFlipApicQuery, WaterFlipApicResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-comparison floor so tiny magnitudes do not inflate the relative
/// error.
const REL_FLOOR: f32 = 1.0e-6;

/// Computes the golden reconstruction for one query via the `CPU` oracle.
fn expected(q: &WaterFlipApicQuery) -> WaterFlipApicResult {
    let base = Vec3::new(q.base[0], q.base[1], q.base[2]);
    let rows = [
        Vec3::new(
            q.affine_rows[0][0],
            q.affine_rows[0][1],
            q.affine_rows[0][2],
        ),
        Vec3::new(
            q.affine_rows[1][0],
            q.affine_rows[1][1],
            q.affine_rows[1][2],
        ),
        Vec3::new(
            q.affine_rows[2][0],
            q.affine_rows[2][1],
            q.affine_rows[2][2],
        ),
    ];
    let offset = Vec3::new(q.offset[0], q.offset[1], q.offset[2]);
    let v = apic_velocity(base, rows, offset);
    WaterFlipApicResult {
        velocity: [v.x, v.y, v.z],
    }
}

/// Returns whether `got` matches `want` within the mixed absolute/relative
/// tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    if diff <= 1.0e-5 {
        return true;
    }
    let denom = want.abs().max(REL_FLOOR);
    diff / denom <= 1.0e-5
}

/// Pins one `GPU` result against the golden oracle on every component.
fn assert_result(idx: usize, got: &WaterFlipApicResult, want: &WaterFlipApicResult) {
    for (axis, (g, w)) in got.velocity.iter().zip(want.velocity.iter()).enumerate() {
        assert!(close(*g, *w), "result {idx} axis {axis}: gpu={g} cpu={w}");
    }
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[WaterFlipApicQuery]) {
    let gpu = GpuWaterFlipApic::new(ctx);
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

/// Draws a 3-vector with each component in `[lo, hi)`.
fn ranged3(state: &mut u64, lo: f32, hi: f32) -> [f32; 3] {
    [
        ranged(state, lo, hi),
        ranged(state, lo, hi),
        ranged(state, lo, hi),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_flip_apic parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterFlipApic::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn flip_apic_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries = vec![
        // Zero affine matrix collapses to plain PIC: result == base.
        WaterFlipApicQuery {
            base: [1.0, -2.0, 3.5],
            affine_rows: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            offset: [0.5, 9.0, -4.0],
        },
        // Zero offset: the affine term vanishes, result == base.
        WaterFlipApicQuery {
            base: [-5.0, 6.0, 7.0],
            affine_rows: [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 1.0]],
            offset: [0.0, 0.0, 0.0],
        },
        // Identity affine matrix: result == base + offset.
        WaterFlipApicQuery {
            base: [0.0, 0.0, 0.0],
            affine_rows: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            offset: [1.25, -3.75, 2.0],
        },
        // General linear field, reproduced exactly to tolerance.
        WaterFlipApicQuery {
            base: [0.5, -0.25, 2.0],
            affine_rows: [[1.0, -2.0, 0.5], [3.0, 0.0, -1.0], [-0.5, 4.0, 2.0]],
            offset: [2.0, -1.0, 0.75],
        },
        // Large magnitudes to exercise the relative tolerance path.
        WaterFlipApicQuery {
            base: [1.0e4, -2.0e4, 3.0e4],
            affine_rows: [
                [10.0, -20.0, 30.0],
                [-40.0, 50.0, -60.0],
                [70.0, -80.0, 90.0],
            ],
            offset: [100.0, -200.0, 300.0],
        },
        // Negative-only components.
        WaterFlipApicQuery {
            base: [-1.0, -1.0, -1.0],
            affine_rows: [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]],
            offset: [-1.5, -2.5, -3.5],
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0xa17c_5f3b_9e21_04d7_u64;

    let mut queries: Vec<WaterFlipApicQuery> = Vec::new();
    while queries.len() < 1024 {
        queries.push(WaterFlipApicQuery {
            base: ranged3(&mut state, -50.0, 50.0),
            affine_rows: [
                ranged3(&mut state, -10.0, 10.0),
                ranged3(&mut state, -10.0, 10.0),
                ranged3(&mut state, -10.0, 10.0),
            ],
            offset: ranged3(&mut state, -5.0, 5.0),
        });
    }
    run_and_check(&ctx, &queries);
}
