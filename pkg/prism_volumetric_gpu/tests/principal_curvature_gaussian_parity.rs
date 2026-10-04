//! Real-device parity for the Gaussian-curvature twin:
//! [`GpuPrincipalCurvatureGaussian`](prism_volumetric_gpu::principal_curvature_gaussian::GpuPrincipalCurvatureGaussian)
//! must reproduce the `CPU` golden `PrincipalCurvature::gaussian` of
//! `prism_physics_core::collider::curvature_tensor`. The Gaussian curvature of
//! a surface vertex is `K = k1 * k2` when both principal curvatures are finite,
//! and undefined (invalid) otherwise.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the finiteness guard, then `k1 * k2` — written out directly so the test
//! never imports `prism_render_architecture` or `prism_physics_core`.
//!
//! The fixtures cover an ordinary convex pair (`k1 = 2, k2 = 3`), a saddle pair
//! (`k1 = 2, k2 = -3`, negative Gaussian), a flat vertex (`0 * 5 = 0`), a
//! large-magnitude pair (`1e3 * 1e3 = 1e6`), a non-finite curvature (`NaN` or
//! `inf`, invalid), a batch of two or more elements that mixes finite and
//! non-finite pairs to validate the `std430` stride, and an empty batch the
//! host short-circuits with no dispatch. A sweep over random finite curvatures
//! follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic is a single multiply, so `CPU` and `GPU` evaluate
//! the same closed form. The valid `gaussian` scalar is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid` flag
//! is compared exactly. The fixtures keep random curvatures finite and well
//! inside range so the validity decision agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::curvature_tensor`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::principal_curvature_gaussian::{
    GpuPrincipalCurvatureGaussian, PrincipalCurvatureGaussianQuery,
    PrincipalCurvatureGaussianResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `gaussian` as `k1 * k2`, returning the
/// Gaussian curvature and the validity flag.
fn oracle(q: &PrincipalCurvatureGaussianQuery) -> (f32, u32) {
    if !q.k1.is_finite() || !q.k2.is_finite() {
        return (0.0, 0);
    }
    (q.k1 * q.k2, 1)
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle: the discrete
/// `valid` flag exactly, and the `gaussian` scalar to tolerance when valid.
fn assert_parity(
    gpu: &PrincipalCurvatureGaussianResult,
    q: &PrincipalCurvatureGaussianQuery,
    label: &str,
) {
    let (gaussian, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        assert!(
            close(gpu.gaussian, gaussian),
            "{label}: gaussian mismatch gpu={} oracle={}",
            gpu.gaussian,
            gaussian
        );
    }
}

#[test]
fn convex_pair_has_positive_gaussian() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureGaussian::new(&ctx);
    // 2 * 3 = 6.
    let q = PrincipalCurvatureGaussianQuery::new(2.0, 3.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].gaussian, 6.0));
    assert_parity(&out[0], &q, "convex");
}

#[test]
fn saddle_pair_has_negative_gaussian() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureGaussian::new(&ctx);
    // 2 * -3 = -6.
    let q = PrincipalCurvatureGaussianQuery::new(2.0, -3.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].gaussian, -6.0));
    assert_parity(&out[0], &q, "saddle");
}

#[test]
fn flat_vertex_has_zero_gaussian() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureGaussian::new(&ctx);
    // 0 * 5 = 0.
    let q = PrincipalCurvatureGaussianQuery::new(0.0, 5.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].gaussian, 0.0));
    assert_parity(&out[0], &q, "flat");
}

#[test]
fn large_magnitude_pair_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureGaussian::new(&ctx);
    // 1e3 * 1e3 = 1e6.
    let q = PrincipalCurvatureGaussianQuery::new(1.0e3, 1.0e3);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].gaussian, 1.0e6));
    assert_parity(&out[0], &q, "large");
}

#[test]
fn non_finite_curvature_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureGaussian::new(&ctx);
    let nan = PrincipalCurvatureGaussianQuery::new(f32::NAN, 2.0);
    let inf = PrincipalCurvatureGaussianQuery::new(3.0, f32::INFINITY);
    let out = gpu.evaluate(&ctx, &[nan, inf]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 0, "NaN curvature should be invalid");
    assert_eq!(out[1].valid, 0, "inf curvature should be invalid");
    assert_eq!(out[0].gaussian, 0.0);
    assert_eq!(out[1].gaussian, 0.0);
    assert_parity(&out[0], &nan, "nan");
    assert_parity(&out[1], &inf, "inf");
}

#[test]
fn batch_mixes_finite_and_non_finite_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureGaussian::new(&ctx);
    let queries = vec![
        PrincipalCurvatureGaussianQuery::new(2.0, 2.0),
        PrincipalCurvatureGaussianQuery::new(f32::NEG_INFINITY, 5.0),
        PrincipalCurvatureGaussianQuery::new(4.0, -0.25),
        PrincipalCurvatureGaussianQuery::new(7.0, f32::NAN),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 0);
    assert_eq!(out[2].valid, 1);
    assert_eq!(out[3].valid, 0);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureGaussian::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPrincipalCurvatureGaussian::new(&ctx);
    let mut lcg = Lcg::new(0x0CA9_11A5);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Finite curvatures well inside range so validity is stable.
        let k1 = lcg.next_range(-100.0, 100.0);
        let k2 = lcg.next_range(-100.0, 100.0);
        queries.push(PrincipalCurvatureGaussianQuery::new(k1, k2));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
}

/// A small deterministic linear-congruential generator; the fixture carries no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}
