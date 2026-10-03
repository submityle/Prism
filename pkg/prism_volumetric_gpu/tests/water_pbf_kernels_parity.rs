//! Real-device parity for the Position-Based Fluids smoothing-kernel twin:
//! [`GpuWaterPbfKernels`](prism_volumetric_gpu::water_pbf_kernels::GpuWaterPbfKernels)
//! must reproduce the numeric core of the `CPU` golden
//! [`pbf`](prism_render_architecture::water::pbf) — the `Poly6` density weight
//! [`poly6`](prism_render_architecture::water::pbf::poly6) and the `Spiky`
//! gradient vector
//! [`spiky_gradient`](prism_render_architecture::water::pbf::spiky_gradient) —
//! across interior samples, the two support breakpoints, degenerate radii, a
//! coincident pair, a mixed batch and a randomized sweep compared
//! sample-for-sample.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected values come straight from the public golden
//! [`poly6`](prism_render_architecture::water::pbf::poly6) and
//! [`spiky_gradient`](prism_render_architecture::water::pbf::spiky_gradient), so
//! a `GPU == golden` pass is direct evidence the ported kernel computes the same
//! weights and gradients the reference does.
//!
//! # Parity criterion
//!
//! Both kernels thread through multiplies, subtracts, a guarded divide and one
//! `sqrt`, so a `GPU` `sqrt` or divide may land a few units in the last place
//! from the scalar reference; every continuous field is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (relative floor `1e-6`).
//!
//! # Conditioning
//!
//! The two support breakpoints — `r^2` crossing `h^2`, and `r^2` crossing the
//! coincident-pair floor `EPS_LEN_SQ` — are discontinuities. Fixtures and the
//! randomized sweep keep every sample well clear of both (rejection sampling the
//! sweep into `[0.05 * h^2, 0.9 * h^2]` for the `Spiky` radius and
//! `[0, 0.9 * h^2]` for the `Poly6` squared distance), so `CPU` and `GPU` cannot
//! straddle a breakpoint. The smoothing radius `h` is kept far above `EPS`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pbf`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::pbf::{poly6, spiky_gradient};
use prism_render_architecture::water::Vec3;
use prism_volumetric_gpu::water_pbf_kernels::{
    GpuWaterPbfKernels, WaterPbfKernelsQuery, WaterPbfKernelsResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous field. A `GPU` `sqrt` or divide may
/// land a few units in the last place from the scalar reference; `1e-4` admits
/// that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
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

/// Computes the golden result for one query straight from the public
/// [`poly6`](prism_render_architecture::water::pbf::poly6) and
/// [`spiky_gradient`](prism_render_architecture::water::pbf::spiky_gradient).
fn oracle(q: &WaterPbfKernelsQuery) -> WaterPbfKernelsResult {
    let weight = poly6(q.r_squared, q.h);
    let grad = spiky_gradient(Vec3::new(q.rx, q.ry, q.rz), q.h);
    WaterPbfKernelsResult {
        poly6: weight,
        spiky_x: grad.x,
        spiky_y: grad.y,
        spiky_z: grad.z,
    }
}

/// Pins one `GPU` sample against the golden: every continuous field within
/// tolerance.
fn check_sample(idx: usize, got: &WaterPbfKernelsResult, want: &WaterPbfKernelsResult) {
    assert!(
        close(got.poly6, want.poly6),
        "sample {idx} poly6: gpu {} vs cpu {}",
        got.poly6,
        want.poly6
    );
    assert!(
        close(got.spiky_x, want.spiky_x),
        "sample {idx} spiky_x: gpu {} vs cpu {}",
        got.spiky_x,
        want.spiky_x
    );
    assert!(
        close(got.spiky_y, want.spiky_y),
        "sample {idx} spiky_y: gpu {} vs cpu {}",
        got.spiky_y,
        want.spiky_y
    );
    assert!(
        close(got.spiky_z, want.spiky_z),
        "sample {idx} spiky_z: gpu {} vs cpu {}",
        got.spiky_z,
        want.spiky_z
    );
}

/// Dispatches `queries` and pins every `GPU` sample against the golden oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterPbfKernels, queries: &[WaterPbfKernelsQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query is expected");
    for (idx, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        check_sample(idx, g, &oracle(q));
    }
}

/// A small, fixed set of well-conditioned interior and edge-case fixtures.
fn fixture_queries() -> Vec<WaterPbfKernelsQuery> {
    vec![
        // Interior sample: both kernels active, radius well clear of support.
        WaterPbfKernelsQuery::new(0.2, 1.0, 0.3, 0.2, 0.1),
        // Poly6 at the self term (r_squared = 0, maximum density weight); the
        // Spiky vector is non-degenerate.
        WaterPbfKernelsQuery::new(0.0, 1.0, 0.25, 0.15, 0.05),
        // Beyond the support radius: both kernels vanish (r_squared and the
        // relative vector length both exceed h^2 with wide margin).
        WaterPbfKernelsQuery::new(2.0, 1.0, 1.4, 1.3, 1.2),
        // Coincident pair for the Spiky kernel (zero relative vector) while
        // Poly6 is still interior.
        WaterPbfKernelsQuery::new(0.1, 1.0, 0.0, 0.0, 0.0),
        // Larger smoothing radius exercises the h^6 / h^9 scaling.
        WaterPbfKernelsQuery::new(1.5, 2.5, 0.9, 0.6, 0.3),
        // Degenerate radius (h well below EPS): both kernels return zero.
        WaterPbfKernelsQuery::new(0.001, 1.0e-9, 0.0004, 0.0002, 0.0001),
    ]
}

/// A 64-bit linear-congruential generator (`SplitMix`/`PCG`-style multiplier)
/// for host-side fixtures; no transcendental and no float equality.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a value in `[0, span)` with milli-resolution from the generator.
fn draw(state: &mut u64, span: f32) -> f32 {
    (lcg(state) % 1000) as f32 / 1000.0 * span
}

/// Draws a signed value in `(-span, span)` with milli-resolution.
fn draw_signed(state: &mut u64, span: f32) -> f32 {
    draw(state, 2.0 * span) - span
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfKernels::new(&ctx);
    // An empty batch never dispatches (a storage buffer cannot be zero-sized)
    // and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn interior_sample_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfKernels::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[WaterPbfKernelsQuery::new(0.2, 1.0, 0.3, 0.2, 0.1)],
    );
}

#[test]
fn self_term_poly6_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfKernels::new(&ctx);
    // r_squared = 0 is the maximum-density self term, well inside the support.
    let q = WaterPbfKernelsQuery::new(0.0, 1.0, 0.25, 0.15, 0.05);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(want.poly6 > 0.0, "self term must have a positive weight");
    check_sample(0, &got[0], &want);
}

#[test]
fn beyond_support_returns_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfKernels::new(&ctx);
    // r_squared and the relative-vector length both exceed h^2, so both kernels
    // vanish.
    let q = WaterPbfKernelsQuery::new(2.0, 1.0, 1.4, 1.3, 1.2);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        close(want.poly6, 0.0) && close(want.spiky_x, 0.0),
        "fixture must be beyond the support radius"
    );
    check_sample(0, &got[0], &want);
}

#[test]
fn coincident_pair_spiky_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfKernels::new(&ctx);
    // A zero relative vector leaves the Spiky direction undefined, so the golden
    // returns the zero vector while Poly6 stays interior.
    let q = WaterPbfKernelsQuery::new(0.1, 1.0, 0.0, 0.0, 0.0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        close(want.spiky_x, 0.0) && close(want.spiky_y, 0.0) && close(want.spiky_z, 0.0),
        "coincident pair must give a zero gradient"
    );
    check_sample(0, &got[0], &want);
}

#[test]
fn degenerate_radius_returns_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfKernels::new(&ctx);
    // h well below EPS is the "no kernel" degenerate case: both outputs zero.
    let q = WaterPbfKernelsQuery::new(0.001, 1.0e-9, 0.0004, 0.0002, 0.0001);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        close(want.poly6, 0.0) && close(want.spiky_x, 0.0),
        "degenerate radius must zero both kernels"
    );
    check_sample(0, &got[0], &want);
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfKernels::new(&ctx);
    // Every fixture dispatched together exercises per-thread indexing and the
    // contiguous output slots.
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfKernels::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = fixture_queries();
    let mut accepted = 0u32;
    let mut tries = 0u32;
    // Many well-conditioned random samples (several workgroups' worth) pin the
    // kernels across a wide span of radii, with rejection sampling keeping every
    // sample clear of both support breakpoints.
    while accepted < 256 && tries < 20_000 {
        tries += 1;
        // Smoothing radius well above EPS: [0.5, 2.5).
        let h = 0.5 + draw(&mut state, 2.0);
        let h2 = h * h;

        // Spiky relative vector: accept only when its squared length lands in
        // [0.05 * h^2, 0.9 * h^2], far from the coincident-pair floor and the
        // support radius.
        let rx = draw_signed(&mut state, h);
        let ry = draw_signed(&mut state, h);
        let rz = draw_signed(&mut state, h);
        let vec_r2 = rx * rx + ry * ry + rz * rz;
        if vec_r2 < 0.05 * h2 || vec_r2 > 0.9 * h2 {
            continue;
        }

        // Poly6 squared distance: accept only in [0, 0.9 * h^2]; Poly6 has no
        // breakpoint at 0, only at h^2, so leave margin only on the high side.
        let r_squared = draw(&mut state, h2);
        if r_squared > 0.9 * h2 {
            continue;
        }

        queries.push(WaterPbfKernelsQuery::new(r_squared, h, rx, ry, rz));
        accepted += 1;
    }
    assert!(
        accepted >= 256,
        "expected at least 256 random samples, got {accepted}"
    );
    check(&ctx, &gpu, &queries);
}
