//! Real-device parity for the Drucker–Prager friction-coefficient twin:
//! [`GpuDruckerPragerFrictionAlpha`](prism_volumetric_gpu::drucker_prager_friction_alpha::GpuDruckerPragerFrictionAlpha)
//! must reproduce the `α` branch of the `CPU` golden
//! `DruckerPragerModel::from_friction_angle` of
//! `prism_physics_core::collider::tet_fem_drucker_prager_plasticity`, the base
//! friction coefficient `α = √(2/3) · 2 sinφ / (3 − sinφ)` a cohesive-frictional
//! material develops at internal friction angle `φ` (in degrees).
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly in flat `f32`/`f64` math so the test never imports
//! `prism_physics_core`, `prism_render_architecture` or `glam`. It replicates
//! the golden exactly, including the radians conversion, the sine and the
//! `√(2/3)` factor taken in `f64` (`f64::from(angle).to_radians().sin()`,
//! `(2.0_f64 / 3.0).sqrt()`), and the golden guard: the angle must be finite
//! and strictly inside the open interval `(0, 90)` degrees.
//!
//! The fixtures cover the regimes the kernel must honor: representative valid
//! angles; angles just inside the `(0, 90)` boundary; the two exact bounds `0`
//! and `90` that must report `valid = false`; out-of-range angles; non-finite
//! inputs (`INFINITY`, `NEG_INFINITY`, `NaN`); a multi-element mixed
//! valid/invalid batch that validates the `std430` array stride end to end;
//! plus an empty batch the host short-circuits with no dispatch. A `512`-step
//! `LCG` sweep over valid interior angles follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL` plus the `sin` and `sqrt` builtins the
//! closed form requires, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The golden evaluates the radians conversion, the sine and the `√(2/3)`
//! factor in `f64` while the twin uses `f32` `sin` and `sqrt`, so `CPU` and
//! `GPU` agree only up to a small numerical gap. The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on `alpha`; the
//! discrete `valid` flag is compared exactly. The sweep keeps every angle well
//! inside `(0, 90)` so the `f64`/`f32` round-off cannot flip the validity
//! decision.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::tet_fem_drucker_prager_plasticity`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::drucker_prager_friction_alpha::{
    DruckerPragerFrictionAlphaQuery, GpuDruckerPragerFrictionAlpha,
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

/// Independent host re-implementation of the golden
/// `DruckerPragerModel::from_friction_angle` `α` branch, returning the friction
/// coefficient and the `valid` flag without importing the golden crate or
/// `glam`. The radians conversion, the sine and the `√(2/3)` factor are taken
/// in `f64` exactly as the golden does, and the validity gate mirrors the
/// golden guard: the angle finite and strictly inside the open interval
/// `(0, 90)` degrees.
fn oracle(q: &DruckerPragerFrictionAlphaQuery) -> (f32, bool) {
    let a = q.angle_degrees;
    if !(a.is_finite() && a > 0.0 && a < 90.0) {
        return (0.0, false);
    }
    let s = f64::from(a).to_radians().sin();
    let alpha = ((2.0_f64 / 3.0).sqrt() * 2.0 * s / (3.0 - s)) as f32;
    (alpha, true)
}

/// Dispatches one query and asserts the `GPU` result matches the oracle on the
/// `alpha` scalar (within tolerance) and the `valid` flag (exactly).
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuDruckerPragerFrictionAlpha,
    q: DruckerPragerFrictionAlphaQuery,
) {
    let r = gpu.evaluate(ctx, std::slice::from_ref(&q))[0];
    let (alpha, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close(r.alpha, alpha),
        "alpha mismatch: gpu={} cpu={} query={q:?}",
        r.alpha,
        alpha
    );
}

#[test]
fn normal_angles_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDruckerPragerFrictionAlpha::new(&ctx);
    // Representative internal friction angles across the valid interior.
    for angle in [10.0_f32, 30.0, 45.0, 60.0, 80.0] {
        assert_parity(&ctx, &gpu, DruckerPragerFrictionAlphaQuery::new(angle));
    }
}

#[test]
fn boundary_near_zero_and_ninety_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDruckerPragerFrictionAlpha::new(&ctx);
    // Angles just inside the open (0, 90) interval remain valid.
    for angle in [0.1_f32, 1.0, 89.0, 89.9] {
        assert_parity(&ctx, &gpu, DruckerPragerFrictionAlphaQuery::new(angle));
    }
}

#[test]
fn zero_angle_reports_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDruckerPragerFrictionAlpha::new(&ctx);
    // The exact lower bound 0 is excluded by the strict guard.
    let q = DruckerPragerFrictionAlphaQuery::new(0.0);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert!(!r.valid, "zero angle must be invalid: {q:?}");
    assert!(
        close(r.alpha, 0.0),
        "invalid alpha must be zero, got {} for {q:?}",
        r.alpha
    );
    let (alpha, valid) = oracle(&q);
    assert!(!valid, "oracle must agree zero angle is invalid: {q:?}");
    assert_eq!(alpha, 0.0, "oracle invalid alpha must be zero: {q:?}");
}

#[test]
fn ninety_degrees_reports_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDruckerPragerFrictionAlpha::new(&ctx);
    // The exact upper bound 90 is excluded by the strict guard.
    let q = DruckerPragerFrictionAlphaQuery::new(90.0);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert!(!r.valid, "ninety degrees must be invalid: {q:?}");
    assert!(
        close(r.alpha, 0.0),
        "invalid alpha must be zero, got {} for {q:?}",
        r.alpha
    );
    let (_, valid) = oracle(&q);
    assert!(!valid, "oracle must agree ninety degrees is invalid: {q:?}");
}

#[test]
fn out_of_range_reports_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDruckerPragerFrictionAlpha::new(&ctx);
    // Angles outside the open (0, 90) interval force valid = false, alpha = 0.
    for angle in [-5.0_f32, -0.001, 90.001, 120.0, 180.0] {
        let q = DruckerPragerFrictionAlphaQuery::new(angle);
        let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
        assert!(!r.valid, "out-of-range angle must be invalid: {q:?}");
        assert!(
            close(r.alpha, 0.0),
            "invalid alpha must be zero, got {} for {q:?}",
            r.alpha
        );
        let (_, valid) = oracle(&q);
        assert!(!valid, "oracle must agree angle is out of range: {q:?}");
    }
}

#[test]
fn non_finite_reports_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDruckerPragerFrictionAlpha::new(&ctx);
    // A non-finite angle forces valid = false, alpha = 0.
    for angle in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let q = DruckerPragerFrictionAlphaQuery::new(angle);
        let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
        assert!(!r.valid, "non-finite angle must be invalid: {q:?}");
        assert!(
            close(r.alpha, 0.0),
            "invalid alpha must be zero, got {} for {q:?}",
            r.alpha
        );
        let (_, valid) = oracle(&q);
        assert!(!valid, "oracle must agree angle is non-finite: {q:?}");
    }
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDruckerPragerFrictionAlpha::new(&ctx);
    // A multi-element mixed batch (valid, boundary-zero, out-of-range,
    // non-finite, valid) exercises the std430 array stride: every slot must
    // decode at the right byte offset and remain independent.
    let queries = [
        DruckerPragerFrictionAlphaQuery::new(30.0),
        DruckerPragerFrictionAlphaQuery::new(0.0),
        DruckerPragerFrictionAlphaQuery::new(120.0),
        DruckerPragerFrictionAlphaQuery::new(f32::NAN),
        DruckerPragerFrictionAlphaQuery::new(55.0),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (alpha, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close(r.alpha, alpha),
            "batch alpha mismatch: gpu={} cpu={} query={q:?}",
            r.alpha,
            alpha
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDruckerPragerFrictionAlpha::new(&ctx);
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
    let gpu = GpuDruckerPragerFrictionAlpha::new(&ctx);
    let mut rng = Lcg::new(0x3D_9B_4F_21);
    let mut queries = Vec::with_capacity(512);
    // Keep every drawn angle well inside the valid (0, 90) interior so the
    // f64/f32 round-off cannot flip the validity decision at the knees.
    while queries.len() < 512 {
        let angle = rng.next_range(1.0, 89.0);
        queries.push(DruckerPragerFrictionAlphaQuery::new(angle));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (alpha, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert!(valid, "sweep samples must be valid: query={q:?}");
        assert!(
            close(r.alpha, alpha),
            "sweep alpha mismatch: gpu={} cpu={} query={q:?}",
            r.alpha,
            alpha
        );
    }
}
