//! Real-device parity for the `MAC`-grid `FLIP`/`PIC` numerics twin:
//! [`GpuWaterFlipMac`](prism_volumetric_gpu::water_flip_mac::GpuWaterFlipMac)
//! must reproduce the two `CPU` golden closed forms
//! [`blend_flip_pic`](prism_render_architecture::water::flip::blend_flip_pic)
//! and
//! [`cell_divergence`](prism_render_architecture::water::flip::cell_divergence)
//! across the blend's clamped-`alpha` endpoints, the divergence's uniform,
//! net-outflow, net-inflow and degenerate-spacing cases, and a randomized
//! sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! Both golden functions are public, so the expected values come straight from
//! calling [`blend_flip_pic`] and [`cell_divergence`] on the host — no
//! reimplementation of the formula. A `GPU == golden` pass is therefore direct
//! evidence the ported kernel computes the same blend and divergence.
//!
//! # Parity criterion
//!
//! Both outputs thread through only multiply, add, subtract, `clamp` and a
//! single divide, so a `GPU` result may land a few units in the last place from
//! the scalar reference; each component is asserted within `abs_diff <= 1e-6` or
//! `rel_diff <= 1e-5`. The degenerate `dx <= EPS` divergence branch returns an
//! exact `0` on both sides.
//!
//! # Conditioning
//!
//! The randomized sweep keeps `dx` well clear of the `EPS` guard (drawing
//! `dx >= 0.25`) so the `CPU` and `GPU` stay on the same side of the
//! divergence branch, and keeps the blend factor inside `0..=1` or far outside
//! it so the clamp lands identically on both.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::flip`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::flip::{blend_flip_pic, cell_divergence, FaceVelocities};
use prism_render_architecture::water::Vec3;
use prism_volumetric_gpu::water_flip_mac::{
    GpuWaterFlipMac, WaterFlipMacQuery, WaterFlipMacResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` multiply/add/divide may land a few units in
/// the last place from the scalar reference; `1e-6` admits that legal slack
/// while still failing a wrong port.
const EPS: f32 = 1.0e-6;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-5;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Computes the expected result straight from the public golden functions, the
/// faithful oracle the `GPU` is pinned against.
fn oracle(q: &WaterFlipMacQuery) -> WaterFlipMacResult {
    let pic = Vec3::new(q.pic[0], q.pic[1], q.pic[2]);
    let flip = Vec3::new(q.flip[0], q.flip[1], q.flip[2]);
    let blended = blend_flip_pic(pic, flip, q.alpha);
    let faces = FaceVelocities {
        x_pos: q.faces[0],
        x_neg: q.faces[1],
        y_pos: q.faces[2],
        y_neg: q.faces[3],
        z_pos: q.faces[4],
        z_neg: q.faces[5],
    };
    let divergence = cell_divergence(faces, q.dx);
    WaterFlipMacResult {
        blended: [blended.x, blended.y, blended.z],
        divergence,
    }
}

/// Pins one `GPU` result against the in-host oracle: each blended component and
/// the divergence within tolerance.
fn check_one(idx: usize, got: &WaterFlipMacResult, want: &WaterFlipMacResult) {
    for axis in 0..3 {
        assert!(
            close(got.blended[axis], want.blended[axis]),
            "query {idx} blended[{axis}]: gpu {} vs cpu {}",
            got.blended[axis],
            want.blended[axis]
        );
    }
    assert!(
        close(got.divergence, want.divergence),
        "query {idx} divergence: gpu {} vs cpu {}",
        got.divergence,
        want.divergence
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterFlipMac, queries: &[WaterFlipMacQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, result, &want);
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

/// Draws a signed velocity component in `[-8.0, 8.0]` at milli resolution.
fn signed(state: &mut u64) -> f32 {
    -8.0 + (lcg(state) % 16_001) as f32 / 1000.0
}

/// Draws a blend factor in `[0.0, 1.0]` at milli resolution.
fn blend(state: &mut u64) -> f32 {
    (lcg(state) % 1001) as f32 / 1000.0
}

/// Draws a grid spacing in `[0.25, 2.25]`, well clear of the `EPS` guard.
fn spacing(state: &mut u64) -> f32 {
    0.25 + (lcg(state) % 2001) as f32 / 1000.0
}

/// Draws a random, well-conditioned query from `state`.
fn random_query(state: &mut u64) -> WaterFlipMacQuery {
    WaterFlipMacQuery::new(
        [signed(state), signed(state), signed(state)],
        [signed(state), signed(state), signed(state)],
        blend(state),
        [
            signed(state),
            signed(state),
            signed(state),
            signed(state),
            signed(state),
            signed(state),
        ],
        spacing(state),
    )
}

/// The deterministic fixture batch: blend endpoints and clamp cases, plus the
/// four divergence regimes.
fn fixture_queries() -> Vec<WaterFlipMacQuery> {
    vec![
        // alpha = 0 -> pure PIC, with a uniform (divergence-free) flow.
        WaterFlipMacQuery::new(
            [1.0, -2.0, 3.0],
            [4.0, 5.0, -6.0],
            0.0,
            [1.0, 1.0, -2.0, -2.0, 0.5, 0.5],
            0.1,
        ),
        // alpha = 0.5 -> midpoint, net outflow (positive divergence).
        WaterFlipMacQuery::new(
            [1.0, -2.0, 3.0],
            [4.0, 5.0, -6.0],
            0.5,
            [2.0, -2.0, 1.0, -1.0, 0.5, -0.5],
            1.0,
        ),
        // alpha = 1 -> pure FLIP, net inflow (negative divergence).
        WaterFlipMacQuery::new(
            [1.0, -2.0, 3.0],
            [4.0, 5.0, -6.0],
            1.0,
            [-2.0, 2.0, -1.0, 1.0, -0.5, 0.5],
            0.75,
        ),
        // alpha below 0 -> clamps to 0, degenerate spacing -> divergence 0.
        WaterFlipMacQuery::new(
            [7.0, 8.0, 9.0],
            [-7.0, -8.0, -9.0],
            -1.5,
            [3.0, -1.0, 2.0, -2.0, 1.0, 0.0],
            0.0,
        ),
        // alpha above 1 -> clamps to 1, mixed faces at unit spacing.
        WaterFlipMacQuery::new(
            [7.0, 8.0, 9.0],
            [-7.0, -8.0, -9.0],
            2.5,
            [0.3, -0.7, 1.2, 0.4, -1.1, 0.9],
            0.5,
        ),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_flip_mac parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterFlipMac::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn blend_endpoint_zero_is_pure_pic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFlipMac::new(&ctx);
    let q = WaterFlipMacQuery::new(
        [1.0, -2.0, 3.0],
        [4.0, 5.0, -6.0],
        0.0,
        [1.0, 1.0, -2.0, -2.0, 0.5, 0.5],
        0.1,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    // alpha = 0 reproduces the pic velocity exactly, and the uniform flow is
    // divergence-free.
    check_one(0, &got[0], &oracle(&q));
    assert!(close(got[0].blended[0], 1.0), "pure PIC keeps pic.x");
    assert!(
        close(got[0].divergence, 0.0),
        "uniform flow has zero divergence"
    );
}

#[test]
fn blend_endpoint_one_is_pure_flip() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFlipMac::new(&ctx);
    let q = WaterFlipMacQuery::new(
        [1.0, -2.0, 3.0],
        [4.0, 5.0, -6.0],
        1.0,
        [-2.0, 2.0, -1.0, 1.0, -0.5, 0.5],
        0.75,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(close(got[0].blended[1], 5.0), "pure FLIP keeps flip.y");
    // Net inflow is a negative divergence.
    assert!(got[0].divergence < 0.0, "net inflow is negative divergence");
}

#[test]
fn blend_clamps_out_of_range_alpha() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFlipMac::new(&ctx);
    // alpha = -1.5 clamps to 0 (pure pic); alpha = 2.5 clamps to 1 (pure flip).
    let low = WaterFlipMacQuery::new(
        [7.0, 8.0, 9.0],
        [-7.0, -8.0, -9.0],
        -1.5,
        [3.0, -1.0, 2.0, -2.0, 1.0, 0.0],
        0.0,
    );
    let high = WaterFlipMacQuery::new(
        [7.0, 8.0, 9.0],
        [-7.0, -8.0, -9.0],
        2.5,
        [0.3, -0.7, 1.2, 0.4, -1.1, 0.9],
        0.5,
    );
    let got = gpu.evaluate(&ctx, &[low, high]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&low));
    check_one(1, &got[1], &oracle(&high));
    // The low case clamps to pic and has a degenerate spacing -> zero.
    assert!(close(got[0].blended[0], 7.0), "clamp-low keeps pic.x");
    assert!(close(got[0].divergence, 0.0), "degenerate spacing is inert");
    // The high case clamps to flip.
    assert!(close(got[1].blended[2], -9.0), "clamp-high keeps flip.z");
}

#[test]
fn degenerate_spacing_is_inert() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFlipMac::new(&ctx);
    // dx exactly 0 and dx exactly the EPS guard both yield zero divergence.
    let zero_dx = WaterFlipMacQuery::new(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.5,
        [5.0, -5.0, 3.0, -3.0, 1.0, -1.0],
        0.0,
    );
    let eps_dx = WaterFlipMacQuery::new(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.5,
        [5.0, -5.0, 3.0, -3.0, 1.0, -1.0],
        1.0e-6,
    );
    let got = gpu.evaluate(&ctx, &[zero_dx, eps_dx]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&zero_dx));
    check_one(1, &got[1], &oracle(&eps_dx));
    assert!(close(got[0].divergence, 0.0), "dx = 0 is inert");
    assert!(close(got[1].divergence, 0.0), "dx = EPS is inert");
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFlipMac::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterFlipMac::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin every
    // reported blend and divergence across a wide span of inputs.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
