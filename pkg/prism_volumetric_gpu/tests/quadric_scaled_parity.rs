//! Real-device parity for the quadric-scaling twin:
//! [`GpuQuadricScaled`](prism_volumetric_gpu::quadric_scaled::GpuQuadricScaled)
//! must reproduce the `CPU` golden `Quadric::scaled` of
//! `prism_physics_core::collider::quadric`. Scaling a quadric by a scalar `s`
//! multiplies each of its ten upper-triangular coefficients
//! `a2, ab, ac, ad, b2, bc, bd, c2, cd, d2` by `s`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! a single per-coefficient multiply in the golden order — written out
//! directly so the test never imports `prism_render_architecture` or
//! `prism_physics_core`.
//!
//! The fixtures cover a general quadric scaled by a random factor, `scale = 0`
//! (the zero quadric), `scale = 1` (the identity), a negative scale, a large
//! scale, a batch of two or more elements validating the `std430` stride, and
//! an empty batch the host short-circuits with no dispatch. A sweep over random
//! coefficients and scalars follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each coefficient is a single multiply, so `CPU` and `GPU` evaluate the same
//! closed form but need not be bit-exact. Each scaled coefficient is compared
//! with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid`
//! flag is compared exactly. The closed form has no degenerate branch, so
//! `valid` is always `1`.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::quadric::Quadric::scaled`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::quadric_scaled::{
    GpuQuadricScaled, QuadricScaledQuery, QuadricScaledResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `Quadric::scaled` by multiplying every
/// coefficient by the scalar in the golden order, returning the ten scaled
/// coefficients and the validity flag (always `1`).
fn oracle(q: &QuadricScaledQuery) -> ([f32; 10], u32) {
    let s = q.scale;
    let mut out = [0.0f32; 10];
    for (dst, src) in out.iter_mut().zip(q.coeffs.iter()) {
        *dst = *src * s;
    }
    (out, 1)
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
/// `valid` flag exactly, and each scaled coefficient to tolerance.
fn assert_parity(gpu: &QuadricScaledResult, q: &QuadricScaledQuery, label: &str) {
    let (coeffs, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    for (i, (g, o)) in gpu.coeffs.iter().zip(coeffs.iter()).enumerate() {
        assert!(
            close(*g, *o),
            "{label}: coeff[{i}] mismatch gpu={g} oracle={o}"
        );
    }
}

/// A representative non-trivial quadric (ten distinct coefficients).
fn sample_coeffs() -> [f32; 10] {
    [1.5, -2.25, 0.75, 3.0, -0.5, 4.125, -1.0, 2.0, -3.5, 0.125]
}

#[test]
fn general_quadric_scales_each_coefficient() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricScaled::new(&ctx);
    let q = QuadricScaledQuery::new(sample_coeffs(), 2.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "general");
}

#[test]
fn scale_zero_yields_the_zero_quadric() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricScaled::new(&ctx);
    let q = QuadricScaledQuery::new(sample_coeffs(), 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    for (i, c) in out[0].coeffs.iter().enumerate() {
        assert_eq!(*c, 0.0, "coeff[{i}] should be zero");
    }
    assert_parity(&out[0], &q, "scale_zero");
}

#[test]
fn scale_one_is_the_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricScaled::new(&ctx);
    let coeffs = sample_coeffs();
    let q = QuadricScaledQuery::new(coeffs, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    for (i, (g, c)) in out[0].coeffs.iter().zip(coeffs.iter()).enumerate() {
        assert!(close(*g, *c), "coeff[{i}] should be identity");
    }
    assert_parity(&out[0], &q, "scale_one");
}

#[test]
fn negative_scale_flips_sign() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricScaled::new(&ctx);
    let q = QuadricScaledQuery::new(sample_coeffs(), -1.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "negative");
}

#[test]
fn large_scale_preserves_ratio() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricScaled::new(&ctx);
    let q = QuadricScaledQuery::new(sample_coeffs(), 1.0e4);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "large");
}

#[test]
fn batch_of_several_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricScaled::new(&ctx);
    let queries = vec![
        QuadricScaledQuery::new(sample_coeffs(), 2.0),
        QuadricScaledQuery::new([-1.0; 10], 0.0),
        QuadricScaledQuery::new([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], -3.0),
        QuadricScaledQuery::new(sample_coeffs(), 1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1);
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricScaled::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricScaled::new(&ctx);
    let mut lcg = Lcg::new(0x5CA1_ED00);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let mut coeffs = [0.0f32; 10];
        for c in coeffs.iter_mut() {
            *c = lcg.next_range(-10.0, 10.0);
        }
        let scale = lcg.next_range(-5.0, 5.0);
        queries.push(QuadricScaledQuery::new(coeffs, scale));
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
