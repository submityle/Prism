//! Real-device parity for the Hertz elastic-force twin:
//! [`GpuHertzElasticForce`](prism_volumetric_gpu::hertz_elastic_force::GpuHertzElasticForce)
//! must reproduce the `CPU` golden private `hertz_elastic_force` of
//! `prism_physics_core::collider::hertz_contact`, the Hertzian normal-force
//! magnitude developed when two elastic grains press into each other by a
//! penetration `overlap`.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly in flat `f32`/`f64` math so the test never imports
//! `prism_physics_core`, `prism_render_architecture` or `glam`. It replicates
//! the golden exactly, including the square root and the `3/2` power taken in
//! `f64` (`r_eff.sqrt()`, `delta.powf(1.5)`), and the golden
//! `evaluate_hertz_contact` degeneracy gate: `effective_modulus`,
//! `effective_radius` and `overlap` all finite, `effective_radius > 0` and
//! `overlap > 0`.
//!
//! The fixtures cover the regimes the kernel must honor: a normal force at
//! sensible contact parameters; a tiny positive overlap; a very large modulus;
//! a non-positive overlap that must report `valid = 0`; a non-positive radius;
//! non-finite inputs (`INFINITY`, `NaN`) in each field; a multi-element mixed
//! valid/invalid batch that validates the `std430` array stride end to end;
//! plus an empty batch the host short-circuits with no dispatch. A `512`-step
//! `LCG` sweep over valid interior inputs follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL` plus the `sqrt` and `pow` builtins the
//! closed form requires, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The golden evaluates the square root and the `3/2` power in `f64` while the
//! twin uses `f32` `sqrt` and `pow`, so `CPU` and `GPU` agree only up to a
//! small numerical gap. The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on `force`; the
//! discrete `valid` flag is compared exactly. The sweep keeps every input well
//! inside the valid region (modulus, radius and overlap all strictly positive
//! with margin) so the `f64`/`f32` round-off cannot flip the validity decision.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::hertz_contact`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hertz_elastic_force::{GpuHertzElasticForce, HertzElasticForceQuery};
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

/// Independent host re-implementation of the golden `hertz_elastic_force`,
/// returning the force magnitude and the `valid` flag without importing the
/// golden crate or `glam`. The square root and the `3/2` power are taken in
/// `f64` exactly as the golden does, and the validity gate mirrors the golden
/// `evaluate_hertz_contact` degeneracy rule: all inputs finite,
/// `effective_radius > 0` and `overlap > 0`.
fn oracle(q: &HertzElasticForceQuery) -> (f32, u32) {
    let finite = q.effective_modulus.abs() < 3.0e38
        && q.effective_radius.abs() < 3.0e38
        && q.overlap.abs() < 3.0e38;
    if !finite || !(q.effective_radius > 0.0) || !(q.overlap > 0.0) {
        return (0.0, 0);
    }
    let e_star = f64::from(q.effective_modulus);
    let r_eff = f64::from(q.effective_radius);
    let delta = f64::from(q.overlap);
    let force = ((4.0 / 3.0) * e_star * r_eff.sqrt() * delta.powf(1.5)) as f32;
    (force, 1)
}

/// Dispatches one query and asserts the `GPU` result matches the oracle on the
/// `force` scalar (within tolerance) and the `valid` flag (exactly).
fn assert_parity(ctx: &GpuContext, gpu: &GpuHertzElasticForce, q: HertzElasticForceQuery) {
    let r = gpu.evaluate(ctx, std::slice::from_ref(&q))[0];
    let (force, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close(r.force, force),
        "force mismatch: gpu={} cpu={} query={q:?}",
        r.force,
        force
    );
}

#[test]
fn normal_force_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzElasticForce::new(&ctx);
    // Representative elastic-grain contacts: a stiff modulus, a reduced radius
    // around a half metre, and a small penetration.
    let cases = [
        HertzElasticForceQuery::new(1.0e7, 0.5, 0.01),
        HertzElasticForceQuery::new(5.0e6, 1.0, 0.02),
        HertzElasticForceQuery::new(2.5e7, 0.25, 0.005),
        HertzElasticForceQuery::new(8.0e6, 2.0, 0.1),
    ];
    for q in cases {
        assert_parity(&ctx, &gpu, q);
    }
}

#[test]
fn tiny_positive_overlap_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzElasticForce::new(&ctx);
    // Barely-touching grains: overlap just above zero must still be valid.
    for &overlap in &[1.0e-5_f32, 5.0e-5, 1.0e-4, 1.0e-3] {
        assert_parity(&ctx, &gpu, HertzElasticForceQuery::new(1.0e7, 0.5, overlap));
    }
}

#[test]
fn large_modulus_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzElasticForce::new(&ctx);
    // Very stiff material: large modulus, moderate radius and overlap.
    for &modulus in &[1.0e8_f32, 5.0e8, 1.0e9] {
        assert_parity(&ctx, &gpu, HertzElasticForceQuery::new(modulus, 1.0, 0.05));
    }
}

#[test]
fn zero_overlap_reports_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzElasticForce::new(&ctx);
    // Non-positive overlap collapses to valid = 0, force = 0.
    for &overlap in &[0.0_f32, -1.0e-4, -0.5] {
        let q = HertzElasticForceQuery::new(1.0e7, 0.5, overlap);
        let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
        assert_eq!(r.valid, 0, "non-positive overlap must be invalid: {q:?}");
        assert!(
            close(r.force, 0.0),
            "invalid force must be zero, got {} for {q:?}",
            r.force
        );
        let (force, valid) = oracle(&q);
        assert_eq!(valid, 0, "oracle must agree overlap is invalid: {q:?}");
        assert_eq!(force, 0.0, "oracle invalid force must be zero: {q:?}");
    }
}

#[test]
fn non_positive_radius_reports_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzElasticForce::new(&ctx);
    // Non-positive effective radius collapses to valid = 0, force = 0.
    for &radius in &[0.0_f32, -0.25, -2.0] {
        let q = HertzElasticForceQuery::new(1.0e7, radius, 0.01);
        let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
        assert_eq!(r.valid, 0, "non-positive radius must be invalid: {q:?}");
        assert!(
            close(r.force, 0.0),
            "invalid force must be zero, got {} for {q:?}",
            r.force
        );
        let (force, valid) = oracle(&q);
        assert_eq!(valid, 0, "oracle must agree radius is invalid: {q:?}");
        assert_eq!(force, 0.0, "oracle invalid force must be zero: {q:?}");
    }
}

#[test]
fn non_finite_reports_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzElasticForce::new(&ctx);
    // A non-finite value in any field forces valid = 0, force = 0.
    let degenerate = [
        HertzElasticForceQuery::new(f32::INFINITY, 0.5, 0.01),
        HertzElasticForceQuery::new(f32::NAN, 0.5, 0.01),
        HertzElasticForceQuery::new(1.0e7, f32::INFINITY, 0.01),
        HertzElasticForceQuery::new(1.0e7, f32::NAN, 0.01),
        HertzElasticForceQuery::new(1.0e7, 0.5, f32::INFINITY),
        HertzElasticForceQuery::new(1.0e7, 0.5, f32::NEG_INFINITY),
        HertzElasticForceQuery::new(1.0e7, 0.5, f32::NAN),
    ];
    for q in degenerate {
        let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
        assert_eq!(r.valid, 0, "non-finite input must be invalid: {q:?}");
        assert!(
            close(r.force, 0.0),
            "invalid force must be zero, got {} for {q:?}",
            r.force
        );
        let (_, valid) = oracle(&q);
        assert_eq!(valid, 0, "oracle must agree input is non-finite: {q:?}");
    }
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzElasticForce::new(&ctx);
    // A multi-element mixed batch (normal, tiny overlap, zero overlap,
    // degenerate radius, non-finite modulus) exercises the std430 array stride:
    // every slot must decode at the right byte offset and remain independent.
    let queries = [
        HertzElasticForceQuery::new(1.0e7, 0.5, 0.01),
        HertzElasticForceQuery::new(5.0e6, 1.0, 1.0e-4),
        HertzElasticForceQuery::new(1.0e7, 0.5, 0.0),
        HertzElasticForceQuery::new(1.0e7, -0.5, 0.01),
        HertzElasticForceQuery::new(f32::NAN, 0.5, 0.01),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (force, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close(r.force, force),
            "batch force mismatch: gpu={} cpu={} query={q:?}",
            r.force,
            force
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzElasticForce::new(&ctx);
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
    let gpu = GpuHertzElasticForce::new(&ctx);
    let mut rng = Lcg::new(0x5C_71_A3_0D);
    let mut queries = Vec::with_capacity(512);
    // Keep every drawn sample well inside the valid region: a physically
    // sensible modulus, reduced radius and overlap, all strictly positive with
    // a comfortable margin from the valid-flip knee so the f64/f32 round-off
    // cannot flip the validity decision.
    while queries.len() < 512 {
        let modulus = rng.next_range(1.0e4, 1.0e8);
        let radius = rng.next_range(0.05, 2.0);
        let overlap = rng.next_range(0.05, 0.5);
        queries.push(HertzElasticForceQuery::new(modulus, radius, overlap));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (force, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert_eq!(valid, 1, "sweep samples must be valid: query={q:?}");
        assert!(
            close(r.force, force),
            "sweep force mismatch: gpu={} cpu={} query={q:?}",
            r.force,
            force
        );
    }
}
