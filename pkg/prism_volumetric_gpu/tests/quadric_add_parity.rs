//! Real-device parity for the quadric-add twin:
//! [`GpuQuadricAdd`](prism_volumetric_gpu::quadric_add::GpuQuadricAdd) must
//! reproduce the `CPU` golden `Quadric::add` of
//! `prism_physics_core::collider::quadric`, which sums two symmetric error
//! quadrics coefficient for coefficient across their ten upper-triangular
//! entries `a2, ab, ac, ad, b2, bc, bd, c2, cd, d2`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! ten element-wise additions evaluated in the golden's order — written out
//! directly in flat `f32` array math so the test never imports
//! `prism_physics_core`, `prism_render_architecture` or `glam`. It mirrors the
//! reference operation for operation.
//!
//! The fixtures cover the regimes the kernel must honor: a random pair of
//! quadrics, a pair carrying negative coefficients, adding the zero quadric
//! (the additive identity), large-magnitude coefficients that exercise the
//! relative tolerance, a multi-element mixed batch that validates the `std430`
//! array stride end to end, plus an empty batch the host short-circuits with no
//! dispatch. A sweep over random coefficient pairs follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each coefficient is a single add, so `CPU` and `GPU` evaluate the same closed
//! form but need not be bit-exact. The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on each of the
//! ten coefficients; the discrete `valid` flag is compared exactly. The closed
//! form has no degenerate branch, so `valid` is always `1` and no comparison
//! sits on a branch knife edge.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::quadric::Quadric::add`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::quadric_add::{GpuQuadricAdd, QuadricAddQuery};
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

/// Independent host re-implementation of the golden `Quadric::add`, returning
/// the ten summed coefficients and the `valid` flag without importing the
/// golden crate or `glam`. The additions are evaluated in the same order as the
/// kernel: `a2, ab, ac, ad, b2, bc, bd, c2, cd, d2`.
fn oracle(q: &QuadricAddQuery) -> ([f32; 10], u32) {
    let mut coeffs = [0.0f32; 10];
    for (i, slot) in coeffs.iter_mut().enumerate() {
        *slot = q.lhs[i] + q.rhs[i];
    }
    (coeffs, 1)
}

/// Dispatches one pair and asserts the `GPU` result matches the oracle on all
/// ten coefficients (within tolerance) and the `valid` flag (exactly).
fn assert_parity(ctx: &GpuContext, gpu: &GpuQuadricAdd, q: QuadricAddQuery) {
    let r = gpu.evaluate(ctx, std::slice::from_ref(&q))[0];
    let (coeffs, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    for (i, (g, c)) in r.coeffs.iter().zip(coeffs.iter()).enumerate() {
        assert!(
            close(*g, *c),
            "coeff {i} mismatch: gpu={g} cpu={c} query={q:?}"
        );
    }
}

#[test]
fn random_pair_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricAdd::new(&ctx);
    // Two generic quadrics with distinct non-zero coefficients, so every one of
    // the ten additions is exercised against the oracle.
    let lhs = [0.3, -0.6, 0.74, 1.7, 0.21, -0.44, 0.9, 0.05, -0.17, 2.4];
    let rhs = [-0.5, 0.5, 0.707, 0.0, -0.33, 0.12, -0.8, 0.6, 0.41, -1.1];
    assert_parity(&ctx, &gpu, QuadricAddQuery::new(lhs, rhs));
}

#[test]
fn negative_coefficients_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricAdd::new(&ctx);
    // Opposite-sign pairs: several sums cancel toward zero, so the absolute
    // tolerance branch (not the relative one) carries the comparison.
    let lhs = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0];
    let rhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let q = QuadricAddQuery::new(lhs, rhs);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    for (i, c) in r.coeffs.iter().enumerate() {
        assert!(
            close(*c, 0.0),
            "cancellation coeff {i} should be zero, got {c}"
        );
    }
}

#[test]
fn zero_quadric_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricAdd::new(&ctx);
    // Adding the zero quadric must echo the other operand coefficient for
    // coefficient — the additive identity.
    let lhs = [1.5, -2.5, 3.25, -0.75, 4.0, 0.125, -6.5, 7.0, -0.5, 9.0];
    let zero = [0.0f32; 10];
    let q = QuadricAddQuery::new(lhs, zero);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    for (i, (g, l)) in r.coeffs.iter().zip(lhs.iter()).enumerate() {
        assert!(close(*g, *l), "identity coeff {i}: gpu={g} expected={l}");
    }
}

#[test]
fn large_magnitude_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricAdd::new(&ctx);
    // Large-magnitude coefficients: the sums are far from zero, so the relative
    // tolerance (not the absolute one) carries the comparison.
    let lhs = [
        1.0e5, 2.0e5, -3.0e5, 4.0e5, -5.0e5, 6.0e5, 7.0e5, -8.0e5, 9.0e5, 1.0e6,
    ];
    let rhs = [
        9.0e4, -1.5e5, 3.3e5, -4.1e5, 5.5e5, -6.6e5, 2.2e5, 8.8e5, -9.9e5, 1.1e6,
    ];
    assert_parity(&ctx, &gpu, QuadricAddQuery::new(lhs, rhs));
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricAdd::new(&ctx);
    // A multi-element mixed batch (generic, cancelling, identity) exercises the
    // std430 array stride: every slot must decode at the right byte offset.
    let queries = [
        QuadricAddQuery::new(
            [0.3, -0.6, 0.74, 1.7, 0.21, -0.44, 0.9, 0.05, -0.17, 2.4],
            [-0.5, 0.5, 0.707, 0.0, -0.33, 0.12, -0.8, 0.6, 0.41, -1.1],
        ),
        QuadricAddQuery::new(
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
        QuadricAddQuery::new(
            [1.5, -2.5, 3.25, -0.75, 4.0, 0.125, -6.5, 7.0, -0.5, 9.0],
            [0.0; 10],
        ),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (coeffs, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        for (i, (g, c)) in r.coeffs.iter().zip(coeffs.iter()).enumerate() {
            assert!(
                close(*g, *c),
                "batch coeff {i} mismatch: gpu={g} cpu={c} query={q:?}"
            );
        }
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricAdd::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}

/// A small deterministic linear-congruential generator so the sweep needs no
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

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricAdd::new(&ctx);
    let mut rng = Lcg::new(0x0A_DD_51_7A);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let mut lhs = [0.0f32; 10];
        let mut rhs = [0.0f32; 10];
        for slot in &mut lhs {
            *slot = rng.next_range(-50.0, 50.0);
        }
        for slot in &mut rhs {
            *slot = rng.next_range(-50.0, 50.0);
        }
        queries.push(QuadricAddQuery::new(lhs, rhs));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (coeffs, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        for (i, (g, c)) in r.coeffs.iter().zip(coeffs.iter()).enumerate() {
            assert!(
                close(*g, *c),
                "sweep coeff {i} mismatch: gpu={g} cpu={c} query={q:?}"
            );
        }
    }
}
