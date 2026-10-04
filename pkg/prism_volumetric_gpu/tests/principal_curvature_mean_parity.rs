//! Real-device parity for the mean-curvature twin:
//! [`GpuPrincipalCurvatureMean`](prism_volumetric_gpu::principal_curvature_mean::GpuPrincipalCurvatureMean)
//! must reproduce the `CPU` golden
//! `prism_physics_core::collider::curvature_tensor::PrincipalCurvature::mean`,
//! which forms the mean curvature `H = 0.5 * (k1 + k2)` from the two principal
//! curvature magnitudes at a surface point.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! one `0.5 * (k1 + k2)` evaluation plus a finiteness gate — written out
//! directly in flat `f32` math so the test never imports `prism_physics_core`,
//! `prism_render_architecture` or `glam`. It mirrors the reference operation.
//!
//! The fixtures cover the regimes the kernel must honor: a generic pair, a
//! sign-cancelling pair whose mean is zero, non-finite inputs (`NaN` and
//! infinity) that drive `valid = 0`, large-magnitude and all-zero boundary
//! pairs, a multi-element mixed batch that validates the `std430` array stride
//! end to end, plus an empty batch the host short-circuits with no dispatch. A
//! sweep over random finite pairs follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! `mean` is a single add then a halve, so `CPU` and `GPU` evaluate the same
//! closed form but need not be bit-exact. The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on `mean`; the
//! discrete `valid` flag is compared exactly. The sweep keeps curvatures finite
//! and well away from the `3.0e38` finiteness edge so the validity decision
//! cannot be flipped by round-off.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::curvature_tensor::PrincipalCurvature::mean`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::principal_curvature_mean::{
    GpuPrincipalCurvatureMean, PrincipalCurvatureMeanQuery,
};
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

/// Independent host re-implementation of the golden `PrincipalCurvature::mean`,
/// returning the mean curvature and the `valid` flag without importing the
/// golden crate or `glam`. A non-finite `k1` or `k2` is invalid and yields a
/// zero mean, matching the kernel's finiteness gate.
fn oracle(q: &PrincipalCurvatureMeanQuery) -> (f32, u32) {
    let valid = q.k1.is_finite() && q.k2.is_finite();
    if valid {
        (0.5 * (q.k1 + q.k2), 1)
    } else {
        (0.0, 0)
    }
}

/// Dispatches one pair and asserts the `GPU` result matches the oracle on
/// `mean` (within tolerance) and `valid` (exactly).
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuPrincipalCurvatureMean,
    q: PrincipalCurvatureMeanQuery,
) {
    let r = gpu.evaluate(ctx, std::slice::from_ref(&q))[0];
    let (mean, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    if valid == 1 {
        assert!(
            close(r.mean, mean),
            "mean mismatch: gpu={} cpu={mean} query={q:?}",
            r.mean
        );
    } else {
        assert!(
            close(r.mean, 0.0),
            "invalid mean should be zero, got {} query={q:?}",
            r.mean
        );
    }
}

#[test]
fn generic_pair_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureMean::new(&ctx);
    // k1 = 2, k2 = 4 -> mean = 3.
    assert_parity(&ctx, &gpu, PrincipalCurvatureMeanQuery::new(2.0, 4.0));
}

#[test]
fn sign_cancelling_pair_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureMean::new(&ctx);
    // k1 = 1, k2 = -1 cancel to a zero mean, exercising the absolute-tolerance
    // branch (not the relative one).
    let q = PrincipalCurvatureMeanQuery::new(1.0, -1.0);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert!(
        close(r.mean, 0.0),
        "cancellation should be zero, got {}",
        r.mean
    );
}

#[test]
fn nan_curvature_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureMean::new(&ctx);
    // A NaN k1 fails the finiteness gate: valid = 0, mean = 0.
    let q = PrincipalCurvatureMeanQuery::new(f32::NAN, 4.0);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 0, "NaN input must be invalid");
    assert!(
        close(r.mean, 0.0),
        "invalid mean must be zero, got {}",
        r.mean
    );
}

#[test]
fn infinite_curvature_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureMean::new(&ctx);
    // An infinite k2 fails the finiteness gate: valid = 0, mean = 0.
    let q = PrincipalCurvatureMeanQuery::new(2.0, f32::INFINITY);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 0, "infinite input must be invalid");
    assert!(
        close(r.mean, 0.0),
        "invalid mean must be zero, got {}",
        r.mean
    );
}

#[test]
fn large_magnitude_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureMean::new(&ctx);
    // Both curvatures 1e6 -> mean = 1e6, exercising the relative tolerance.
    assert_parity(&ctx, &gpu, PrincipalCurvatureMeanQuery::new(1.0e6, 1.0e6));
}

#[test]
fn flat_vertex_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureMean::new(&ctx);
    // The flat vertex k1 = k2 = 0 maps to a zero mean and valid = 1.
    let q = PrincipalCurvatureMeanQuery::new(0.0, 0.0);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1, "flat vertex is finite and valid");
    assert!(close(r.mean, 0.0), "flat mean must be zero, got {}", r.mean);
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureMean::new(&ctx);
    // A multi-element batch (including one invalid entry) validates the std430
    // array stride end to end: each thread must read its own 16-byte slot.
    let queries = [
        PrincipalCurvatureMeanQuery::new(2.0, 4.0),
        PrincipalCurvatureMeanQuery::new(1.0, -1.0),
        PrincipalCurvatureMeanQuery::new(f32::NAN, 3.0),
        PrincipalCurvatureMeanQuery::new(0.0, 0.0),
        PrincipalCurvatureMeanQuery::new(1.0e6, 1.0e6),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len());
    for (q, r) in queries.iter().zip(results.iter()) {
        let (mean, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        if valid == 1 {
            assert!(close(r.mean, mean), "batch mean: gpu={} cpu={mean}", r.mean);
        } else {
            assert!(close(r.mean, 0.0), "batch invalid mean: got {}", r.mean);
        }
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureMean::new(&ctx);
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureMean::new(&ctx);

    // Numerical Recipes LCG; the top bits drive a uniform in [0, 1).
    let mut state: u32 = 0x1357_9bdf;
    let mut next_unit = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    let next_range = |lo: f32, hi: f32, u: f32| lo + (hi - lo) * u;

    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        // Finite curvatures well away from the 3.0e38 finiteness edge, so the
        // validity decision cannot be flipped by round-off.
        let k1 = next_range(-100.0, 100.0, next_unit());
        let k2 = next_range(-100.0, 100.0, next_unit());
        queries.push(PrincipalCurvatureMeanQuery::new(k1, k2));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len());
    for (q, r) in queries.iter().zip(results.iter()) {
        let (mean, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert!(
            close(r.mean, mean),
            "sweep mean mismatch: gpu={} cpu={mean} query={q:?}",
            r.mean
        );
    }
}
