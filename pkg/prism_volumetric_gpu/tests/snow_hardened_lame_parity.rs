//! Real-device parity for the snow hardening Lamé twin:
//! [`GpuSnowHardenedLame`](prism_volumetric_gpu::snow_hardened_lame::GpuSnowHardenedLame)
//! must reproduce the `CPU` golden `hardened_lame` of
//! `prism_physics_core::collider::tet_fem_snow_plasticity`. Compacted snow
//! stiffens: both Lamé parameters are scaled by `exp(ξ · (1 − J_p))`, where
//! `ξ ≥ 0` is the model hardening coefficient and `J_p = det Fₚ` is the plastic
//! volume ratio supplied here directly as a scalar. The exponent is clamped to
//! `[-40, 40]` so a near-singular plastic gradient cannot overflow the
//! exponential.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the finiteness gate, then `exponent = ξ · (1 − J_p)` taken in `f64` (as the
//! golden does) before `exp` and the two scaling multiplies — written out
//! directly so the test never imports `prism_render_architecture` or
//! `prism_physics_core`.
//!
//! The fixtures cover a nominal valid model, the `J_p = 1` identity where the
//! factor is one, both clamp saturation knees (exponent driven well past
//! `±40`), every degenerate rejection (non-finite `ξ`, `J_p`, `base_mu` or
//! `base_lambda`), a batch of two or more elements mixing valid and invalid
//! inputs to validate the `std430` stride, and an empty batch the host
//! short-circuits with no dispatch. A sweep over random finite inputs whose
//! exponent stays far inside the clamp window follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The golden evaluates the exponent and `exp` in `f64` before narrowing to
//! `f32`; the kernel evaluates the same closed form in `f32` with the `WGSL`
//! built-in `exp`. `CPU` and `GPU` therefore agree on the same function but
//! need not be bit-exact, so each continuous output is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid` flag
//! is compared exactly. The fixtures keep inputs finite and the exponent well
//! away from the overflow knees so the two sides agree.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::tet_fem_snow_plasticity`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::snow_hardened_lame::{
    GpuSnowHardenedLame, SnowHardenedLameQuery, SnowHardenedLameResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `hardened_lame` in the golden operator
/// order. The exponent and `exp` are evaluated in `f64` (as the reference
/// does) before narrowing the factor to `f32`; the two scaling multiplies are
/// `f32`. Returns the scaled parameters, the factor and the validity flag.
fn oracle(q: &SnowHardenedLameQuery) -> (f32, f32, f32, bool) {
    let xi = q.hardening;
    let jp = q.plastic_volume_ratio;
    let base_mu = q.base_mu;
    let base_lambda = q.base_lambda;
    if !xi.is_finite() || !jp.is_finite() || !base_mu.is_finite() || !base_lambda.is_finite() {
        return (0.0, 0.0, 0.0, false);
    }
    let one_minus = 1.0f32 - jp;
    let exponent = f64::from(xi) * f64::from(one_minus);
    let factor = exponent.clamp(-40.0, 40.0).exp() as f32;
    (base_mu * factor, base_lambda * factor, factor, true)
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
/// `valid` flag exactly, and each continuous output to tolerance when valid.
fn assert_parity(gpu: &SnowHardenedLameResult, q: &SnowHardenedLameQuery, label: &str) {
    let (mu, lambda, factor, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid {
        assert!(
            close(gpu.factor, factor),
            "{label}: factor mismatch gpu={} oracle={}",
            gpu.factor,
            factor
        );
        assert!(
            close(gpu.mu, mu),
            "{label}: mu mismatch gpu={} oracle={}",
            gpu.mu,
            mu
        );
        assert!(
            close(gpu.lambda, lambda),
            "{label}: lambda mismatch gpu={} oracle={}",
            gpu.lambda,
            lambda
        );
    }
}

#[test]
fn nominal_scales_both_parameters() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSnowHardenedLame::new(&ctx);
    // ξ=5, J_p=0.5 → exponent=2.5, factor=exp(2.5)≈12.182.
    let q = SnowHardenedLameQuery::new(5.0, 0.5, 1.0e5, 1.5e5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(close(out[0].factor, (2.5f64).exp() as f32));
    assert_parity(&out[0], &q, "nominal");
}

#[test]
fn jp_equals_one_gives_unit_factor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSnowHardenedLame::new(&ctx);
    // J_p=1 → exponent=0 → factor=1 → outputs equal the base parameters.
    let q = SnowHardenedLameQuery::new(7.0, 1.0, 2.0e4, 9.0e4);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(close(out[0].factor, 1.0));
    assert!(close(out[0].mu, 2.0e4));
    assert!(close(out[0].lambda, 9.0e4));
    assert_parity(&out[0], &q, "jp_one");
}

#[test]
fn clamp_upper_saturates() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSnowHardenedLame::new(&ctx);
    // ξ=100, J_p=-10 → one_minus=11, exponent=1100 → clamped to 40, so both
    // sides saturate at factor=exp(40) with no knee ambiguity.
    let q = SnowHardenedLameQuery::new(100.0, -10.0, 1.0e5, 1.0e5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(close(out[0].factor, (40.0f64).exp() as f32));
    assert_parity(&out[0], &q, "clamp_upper");
}

#[test]
fn clamp_lower_saturates() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSnowHardenedLame::new(&ctx);
    // ξ=100, J_p=10 → one_minus=-9, exponent=-900 → clamped to -40, so both
    // sides saturate at factor=exp(-40)≈4.25e-18 (outputs effectively zero).
    let q = SnowHardenedLameQuery::new(100.0, 10.0, 1.0e5, 1.0e5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(close(out[0].factor, (-40.0f64).exp() as f32));
    assert_parity(&out[0], &q, "clamp_lower");
}

#[test]
fn non_finite_hardening_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSnowHardenedLame::new(&ctx);
    let q = SnowHardenedLameQuery::new(f32::NAN, 0.5, 1.0e5, 1.5e5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_eq!(out[0].mu, 0.0);
    assert_eq!(out[0].lambda, 0.0);
    assert_eq!(out[0].factor, 0.0);
    assert_parity(&out[0], &q, "nan_hardening");
}

#[test]
fn infinite_plastic_ratio_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSnowHardenedLame::new(&ctx);
    let q = SnowHardenedLameQuery::new(5.0, f32::INFINITY, 1.0e5, 1.5e5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_eq!(out[0].factor, 0.0);
    assert_parity(&out[0], &q, "inf_jp");
}

#[test]
fn non_finite_base_mu_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSnowHardenedLame::new(&ctx);
    let q = SnowHardenedLameQuery::new(5.0, 0.5, f32::NAN, 1.5e5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_eq!(out[0].mu, 0.0);
    assert_parity(&out[0], &q, "nan_base_mu");
}

#[test]
fn infinite_base_lambda_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSnowHardenedLame::new(&ctx);
    let q = SnowHardenedLameQuery::new(5.0, 0.5, 1.0e5, f32::NEG_INFINITY);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_eq!(out[0].lambda, 0.0);
    assert_parity(&out[0], &q, "inf_base_lambda");
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSnowHardenedLame::new(&ctx);
    let queries = vec![
        SnowHardenedLameQuery::new(5.0, 0.5, 1.0e5, 1.5e5),
        SnowHardenedLameQuery::new(7.0, 1.0, 2.0e4, 9.0e4),
        SnowHardenedLameQuery::new(f32::NAN, 0.5, 1.0e5, 1.5e5),
        SnowHardenedLameQuery::new(100.0, -10.0, 1.0e5, 1.0e5),
        SnowHardenedLameQuery::new(5.0, 0.5, 1.0e5, f32::INFINITY),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(out[0].valid);
    assert!(out[1].valid);
    assert!(!out[2].valid);
    assert!(out[3].valid);
    assert!(!out[4].valid);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSnowHardenedLame::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSnowHardenedLame::new(&ctx);
    let mut lcg = Lcg::new(0x51A0_1BEF);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // ξ ∈ [0, 10] and J_p ∈ [0.2, 1.8] bound |1 - J_p| <= 0.8, so the
        // exponent ξ·(1 − J_p) stays within [-8, 8] — far inside the ±40 clamp
        // window with margin well over 0.05, keeping both the clamp decision
        // and the factor stable. Base parameters stay finite and positive.
        let hardening = lcg.next_range(0.0, 10.0);
        let plastic_volume_ratio = lcg.next_range(0.2, 1.8);
        let base_mu = lcg.next_range(1.0e3, 1.0e6);
        let base_lambda = lcg.next_range(1.0e3, 1.0e6);
        queries.push(SnowHardenedLameQuery::new(
            hardening,
            plastic_volume_ratio,
            base_mu,
            base_lambda,
        ));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(res.valid, "sweep[{i}] should be valid");
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
