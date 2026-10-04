//! Real-device parity for the diameter-midpoint twin:
//! [`GpuDiameterMidpoint`](prism_volumetric_gpu::diameter_midpoint::GpuDiameterMidpoint)
//! must reproduce the `CPU` golden
//! `prism_physics_core::collider::diameter::MeshDiameter::midpoint`, which
//! forms the component-wise average `(endpoint_a + endpoint_b) * 0.5` of a
//! mesh's farthest endpoint pair.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! three component-wise `(a + b) * 0.5` evaluations — written out directly in
//! flat `f32` array math so the test never imports `prism_physics_core`,
//! `prism_render_architecture` or `glam`. It mirrors the reference operation
//! for operation.
//!
//! The fixtures cover the regimes the kernel must honor: a symmetric pair whose
//! midpoint is the origin (exercising the absolute-tolerance branch), a generic
//! pair, an all-negative pair, a large-magnitude pair that exercises the
//! relative tolerance, a multi-element mixed batch that validates the `std430`
//! array stride end to end, plus an empty batch the host short-circuits with no
//! dispatch. A sweep over random endpoint pairs follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each component is a single add then a halve, so `CPU` and `GPU` evaluate the
//! same closed form but need not be bit-exact. The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on each of the
//! three components. The closed form has no degenerate branch, so no comparison
//! sits on a branch knife edge and no validity flag is carried.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::diameter::MeshDiameter::midpoint`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::diameter_midpoint::{DiameterMidpointQuery, GpuDiameterMidpoint};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Independent host re-implementation of the golden `MeshDiameter::midpoint`,
/// returning the three midpoint components without importing the golden crate
/// or `glam`. Each component is evaluated in the same order as the kernel:
/// add the two endpoints, then halve.
fn oracle(q: &DiameterMidpointQuery) -> [f32; 3] {
    [
        (q.endpoint_a[0] + q.endpoint_b[0]) * 0.5,
        (q.endpoint_a[1] + q.endpoint_b[1]) * 0.5,
        (q.endpoint_a[2] + q.endpoint_b[2]) * 0.5,
    ]
}

/// Dispatches one pair and asserts the `GPU` midpoint matches the oracle on all
/// three components within tolerance.
fn assert_parity(ctx: &GpuContext, gpu: &GpuDiameterMidpoint, q: DiameterMidpointQuery) {
    let r = gpu.evaluate(ctx, std::slice::from_ref(&q))[0];
    let mid = oracle(&q);
    for (i, (g, c)) in r.midpoint.iter().zip(mid.iter()).enumerate() {
        assert!(
            close(*g, *c),
            "midpoint component {i} mismatch: gpu={g} cpu={c} query={q:?}"
        );
    }
}

#[test]
fn origin_symmetric_pair_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterMidpoint::new(&ctx);
    // Endpoints are exact negatives, so the midpoint is the origin and the
    // absolute-tolerance branch (not the relative one) carries the comparison.
    let q = DiameterMidpointQuery::new([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    for (i, c) in r.midpoint.iter().enumerate() {
        assert!(
            close(*c, 0.0),
            "symmetric midpoint {i} should be zero, got {c}"
        );
    }
}

#[test]
fn generic_pair_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterMidpoint::new(&ctx);
    // A generic pair with distinct non-zero components in every axis.
    let q = DiameterMidpointQuery::new([0.3, -0.6, 0.74], [1.7, 0.21, -0.44]);
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn negative_coordinates_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterMidpoint::new(&ctx);
    // Both endpoints entirely in the negative octant: the midpoint stays
    // negative, confirming no sign is lost in the add-then-halve.
    let q = DiameterMidpointQuery::new([-4.0, -8.5, -2.25], [-1.0, -3.5, -9.75]);
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn large_magnitude_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterMidpoint::new(&ctx);
    // Large coordinates: the midpoint is far from zero, so the relative
    // tolerance (not the absolute) carries the comparison.
    let q = DiameterMidpointQuery::new([1.0e6, -5.0e5, 3.3e5], [2.0e6, 4.0e5, -1.1e5]);
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterMidpoint::new(&ctx);
    // A multi-element batch with distinct pairs validates the std430 array
    // stride end to end: each thread must read its own 32-byte slot.
    let queries = [
        DiameterMidpointQuery::new([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]),
        DiameterMidpointQuery::new([0.3, -0.6, 0.74], [1.7, 0.21, -0.44]),
        DiameterMidpointQuery::new([-4.0, -8.5, -2.25], [-1.0, -3.5, -9.75]),
        DiameterMidpointQuery::new([10.0, 20.0, 30.0], [-30.0, -20.0, -10.0]),
        DiameterMidpointQuery::new([1.0e6, -5.0e5, 3.3e5], [2.0e6, 4.0e5, -1.1e5]),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len());
    for (q, r) in queries.iter().zip(results.iter()) {
        let mid = oracle(q);
        for (i, (g, c)) in r.midpoint.iter().zip(mid.iter()).enumerate() {
            assert!(
                close(*g, *c),
                "batch component {i} mismatch: gpu={g} cpu={c} query={q:?}"
            );
        }
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterMidpoint::new(&ctx);
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterMidpoint::new(&ctx);

    // Numerical Recipes LCG; the top bits drive a uniform in [0, 1).
    let mut state: u32 = 0x1234_5678;
    let mut next_unit = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    let mut next_range = |lo: f32, hi: f32, u: f32| lo + (hi - lo) * u;

    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        let a = [
            next_range(-100.0, 100.0, next_unit()),
            next_range(-100.0, 100.0, next_unit()),
            next_range(-100.0, 100.0, next_unit()),
        ];
        let b = [
            next_range(-100.0, 100.0, next_unit()),
            next_range(-100.0, 100.0, next_unit()),
            next_range(-100.0, 100.0, next_unit()),
        ];
        queries.push(DiameterMidpointQuery::new(a, b));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len());
    for (q, r) in queries.iter().zip(results.iter()) {
        let mid = oracle(q);
        for (i, (g, c)) in r.midpoint.iter().zip(mid.iter()).enumerate() {
            assert!(
                close(*g, *c),
                "sweep component {i} mismatch: gpu={g} cpu={c} query={q:?}"
            );
        }
    }
}
