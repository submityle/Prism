//! Real-device parity for the Mohr–Coulomb shear-failure envelope twin:
//! [`GpuMohrCoulombYieldEnvelope`](prism_volumetric_gpu::mohr_coulomb_yield_envelope::GpuMohrCoulombYieldEnvelope)
//! must reproduce the `CPU` golden `MohrCoulombCriterion` of
//! `prism_physics_core::collider::mohr_coulomb_yield`. The criterion is defined
//! by an internal friction angle `phi` (radians) and a cohesion intercept `c`;
//! it exposes a friction coefficient `tan(phi)`, a shear strength
//! `c + sigma_n tan(phi)`, an unconfined compressive strength
//! `2 c cos(phi) / (1 - sin(phi))`, a cohesion apex `c / tan(phi)`, and a yield
//! evaluation on either a principal-stress triple or a single plane.
//!
//! The oracle here is an independent re-implementation of those closed forms —
//! the finiteness-and-range gate, the trigonometric readouts and the
//! principal / plane yield functions — written out directly so the test never
//! imports `prism_render_architecture` or `prism_physics_core`, and uses no
//! vector math library.
//!
//! The fixtures cover a tangent circle (`yield ≈ 0`), sub- and super-critical
//! circles, a plane loaded exactly to strength, the unconfined strength, a
//! frictionless cohesion apex (`phi = 0`, apex unbounded), every criterion
//! rejection (`phi = pi/2`, `phi < 0`, `c < 0`, `NaN`, infinity), a plane with
//! negative shear and a principal state with a non-finite component, a batch of
//! two or more elements mixing valid and invalid inputs to validate the
//! `std430` stride, and an empty batch the host short-circuits with no
//! dispatch. A sweep over random finite inputs whose friction angle stays well
//! inside `[0, 1.4]` radians (clear of `pi/2`) follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The golden evaluates its trigonometry in `f64` and casts to `f32`; this
//! oracle does the same, while the kernel uses the portable `f32`
//! `sin`/`cos`/`tan` builtins, so the two sides need not be bit-exact. Each
//! continuous output is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` and `apex_valid` flags are
//! compared exactly. The fixtures keep the friction angle away from `pi/2`
//! (where `tan` and `1 - sin(phi)` are ill-conditioned) so round-off cannot
//! flip a validity decision.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::mohr_coulomb_yield`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mohr_coulomb_yield_envelope::{
    GpuMohrCoulombYieldEnvelope, MohrCoulombMode, MohrCoulombYieldEnvelopeQuery,
    MohrCoulombYieldEnvelopeResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// `30` degrees in radians, a well-conditioned friction angle used by several
/// fixtures (`sin = 0.5`, `cos = sqrt(3)/2`).
const PHI_30: f32 = std::f32::consts::FRAC_PI_6;

/// Ordered finiteness predicate matching the kernel's `abs(x) < 3.0e38` guard
/// (which rejects both infinities and `NaN`), rather than a bare `x == x`.
fn finite(x: f32) -> bool {
    x.abs() < 3.0e38
}

/// Independent host oracle: reproduces `MohrCoulombCriterion` in the golden's
/// operator order, evaluating trigonometry in `f64` and casting to `f32`, then
/// the principal- or plane-mode state selected by the query. Returns the full
/// resolved result.
fn oracle(q: &MohrCoulombYieldEnvelopeQuery) -> MohrCoulombYieldEnvelopeResult {
    let phi = q.friction_angle;
    let coh = q.cohesion;
    let crit_valid =
        finite(phi) && finite(coh) && phi >= 0.0 && phi < std::f32::consts::FRAC_PI_2 && coh >= 0.0;

    // Trigonometry in f64, cast to f32, matching the golden.
    let ss = (phi as f64).sin() as f32;
    let cc = (phi as f64).cos() as f32;
    let mu = (phi as f64).tan() as f32;

    let friction_coefficient = if crit_valid { mu } else { 0.0 };
    let shear_strength_at_normal = if crit_valid {
        coh + q.sigma_n * mu
    } else {
        0.0
    };
    let unconfined = if crit_valid {
        2.0 * coh * cc / (1.0 - ss)
    } else {
        0.0
    };

    let apex_ok = crit_valid && phi > 0.0 && mu > 0.0;
    let cohesion_apex = if apex_ok { coh / mu } else { 0.0 };
    let apex_valid = u32::from(apex_ok);

    // Principal-mode state.
    let smax = q.principal[0].max(q.principal[1]).max(q.principal[2]);
    let smin = q.principal[0].min(q.principal[1]).min(q.principal[2]);
    let p_center = 0.5 * (smax + smin);
    let p_radius = 0.5 * (smax - smin);
    let p_strength = coh * cc + p_center * ss;
    let p_yield = p_radius - p_strength;
    let principal_valid =
        finite(q.principal[0]) && finite(q.principal[1]) && finite(q.principal[2]);

    // Plane-mode state.
    let pl_normal = q.sigma_n;
    let pl_shear = q.tau;
    let pl_strength = coh + q.sigma_n * mu;
    let pl_yield = q.tau - pl_strength;
    let plane_valid = finite(q.sigma_n) && finite(q.tau) && q.tau >= 0.0;

    let is_principal = q.mode == MohrCoulombMode::Principal;
    let state_valid = if is_principal {
        principal_valid
    } else {
        plane_valid
    };
    let overall = crit_valid && state_valid;

    let normal_stress = if is_principal { p_center } else { pl_normal };
    let shear_stress = if is_principal { p_radius } else { pl_shear };
    let state_strength = if is_principal {
        p_strength
    } else {
        pl_strength
    };
    let yield_function = if is_principal { p_yield } else { pl_yield };

    MohrCoulombYieldEnvelopeResult {
        friction_coefficient,
        shear_strength_at_normal,
        unconfined_compressive_strength: unconfined,
        cohesion_apex,
        normal_stress: if overall { normal_stress } else { 0.0 },
        shear_stress: if overall { shear_stress } else { 0.0 },
        shear_strength: if overall { state_strength } else { 0.0 },
        yield_function: if overall { yield_function } else { 0.0 },
        apex_valid,
        valid: u32::from(overall),
    }
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
/// `valid` and `apex_valid` flags exactly, the criterion readouts whenever the
/// criterion is valid, and each state output to tolerance when overall-valid.
fn assert_parity(
    gpu: &MohrCoulombYieldEnvelopeResult,
    q: &MohrCoulombYieldEnvelopeQuery,
    label: &str,
) {
    let want = oracle(q);
    assert_eq!(gpu.valid, want.valid, "{label}: valid flag mismatch");
    assert_eq!(
        gpu.apex_valid, want.apex_valid,
        "{label}: apex_valid flag mismatch"
    );
    assert!(
        close(gpu.friction_coefficient, want.friction_coefficient),
        "{label}: friction_coefficient gpu={} oracle={}",
        gpu.friction_coefficient,
        want.friction_coefficient
    );
    assert!(
        close(gpu.shear_strength_at_normal, want.shear_strength_at_normal),
        "{label}: shear_strength_at_normal gpu={} oracle={}",
        gpu.shear_strength_at_normal,
        want.shear_strength_at_normal
    );
    assert!(
        close(
            gpu.unconfined_compressive_strength,
            want.unconfined_compressive_strength
        ),
        "{label}: unconfined gpu={} oracle={}",
        gpu.unconfined_compressive_strength,
        want.unconfined_compressive_strength
    );
    assert!(
        close(gpu.cohesion_apex, want.cohesion_apex),
        "{label}: cohesion_apex gpu={} oracle={}",
        gpu.cohesion_apex,
        want.cohesion_apex
    );
    assert!(
        close(gpu.normal_stress, want.normal_stress),
        "{label}: normal_stress gpu={} oracle={}",
        gpu.normal_stress,
        want.normal_stress
    );
    assert!(
        close(gpu.shear_stress, want.shear_stress),
        "{label}: shear_stress gpu={} oracle={}",
        gpu.shear_stress,
        want.shear_stress
    );
    assert!(
        close(gpu.shear_strength, want.shear_strength),
        "{label}: shear_strength gpu={} oracle={}",
        gpu.shear_strength,
        want.shear_strength
    );
    assert!(
        close(gpu.yield_function, want.yield_function),
        "{label}: yield_function gpu={} oracle={}",
        gpu.yield_function,
        want.yield_function
    );
}

#[test]
fn principal_tangent_circle_is_neutral() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    // phi=30deg, c=0: strength = center*sin30 = 0.5*center. A circle of radius
    // 0.5*center is exactly tangent to the envelope, so yield ≈ 0.
    let center = 10.0_f32;
    let radius = 0.5 * center;
    let q = MohrCoulombYieldEnvelopeQuery::principal(
        PHI_30,
        0.0,
        0.0,
        [center + radius, center, center - radius],
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(out[0].yield_function.abs() <= 1.0e-3, "tangent yield ≈ 0");
    assert_parity(&out[0], &q, "principal_tangent");
}

#[test]
fn principal_subcritical_circle_is_safe() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    let center = 10.0_f32;
    let radius = 0.3 * center;
    let q = MohrCoulombYieldEnvelopeQuery::principal(
        PHI_30,
        0.0,
        0.0,
        [center + radius, center, center - radius],
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].yield_function < 0.0, "subcritical yield < 0");
    assert_parity(&out[0], &q, "principal_subcritical");
}

#[test]
fn principal_supercritical_circle_fails() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    let center = 10.0_f32;
    let radius = 0.8 * center;
    let q = MohrCoulombYieldEnvelopeQuery::principal(
        PHI_30,
        0.0,
        0.0,
        [center + radius, center, center - radius],
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].yield_function > 0.0, "supercritical yield > 0");
    assert_parity(&out[0], &q, "principal_supercritical");
}

#[test]
fn plane_loaded_to_strength_is_neutral() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    let coh = 3.0_f32;
    let sigma_n = 8.0_f32;
    // tau set to the available strength c + sigma_n*tan(phi), so yield ≈ 0.
    let mu = (PHI_30 as f64).tan() as f32;
    let tau = coh + sigma_n * mu;
    let q = MohrCoulombYieldEnvelopeQuery::plane(PHI_30, coh, sigma_n, tau);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(out[0].yield_function.abs() <= 1.0e-3, "plane yield ≈ 0");
    assert_parity(&out[0], &q, "plane_at_strength");
}

#[test]
fn unconfined_strength_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    // phi=30deg, c=10: sigma_c = 2*10*cos30/(1-sin30) = 20*(sqrt3/2)/0.5 = 20√3.
    let q = MohrCoulombYieldEnvelopeQuery::plane(PHI_30, 10.0, 0.0, 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    let want = 20.0 * 3.0_f32.sqrt();
    assert!(
        close(out[0].unconfined_compressive_strength, want),
        "unconfined gpu={} want={}",
        out[0].unconfined_compressive_strength,
        want
    );
    assert_parity(&out[0], &q, "unconfined");
}

#[test]
fn frictionless_cohesion_apex_is_unbounded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    // phi=0: the apex is unbounded, so apex_valid=0 while the criterion itself
    // is still valid (yield is well defined).
    let q = MohrCoulombYieldEnvelopeQuery::plane(0.0, 4.0, 5.0, 2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].apex_valid, 0);
    assert_eq!(out[0].cohesion_apex, 0.0);
    assert_eq!(out[0].friction_coefficient, 0.0);
    assert_parity(&out[0], &q, "frictionless_apex");
}

#[test]
fn friction_angle_at_half_pi_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    let q = MohrCoulombYieldEnvelopeQuery::plane(std::f32::consts::FRAC_PI_2, 3.0, 5.0, 2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[0].friction_coefficient, 0.0);
    assert_parity(&out[0], &q, "phi_half_pi");
}

#[test]
fn negative_friction_angle_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    let q = MohrCoulombYieldEnvelopeQuery::plane(-0.1, 3.0, 5.0, 2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_parity(&out[0], &q, "phi_negative");
}

#[test]
fn negative_cohesion_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    let q = MohrCoulombYieldEnvelopeQuery::plane(PHI_30, -1.0, 5.0, 2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_parity(&out[0], &q, "cohesion_negative");
}

#[test]
fn nan_friction_angle_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    let q = MohrCoulombYieldEnvelopeQuery::plane(f32::NAN, 3.0, 5.0, 2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_parity(&out[0], &q, "phi_nan");
}

#[test]
fn infinite_cohesion_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    let q = MohrCoulombYieldEnvelopeQuery::plane(PHI_30, f32::INFINITY, 5.0, 2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_parity(&out[0], &q, "cohesion_inf");
}

#[test]
fn plane_negative_shear_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    // The criterion is valid but a negative shear magnitude invalidates the
    // plane state, so the overall result is invalid.
    let q = MohrCoulombYieldEnvelopeQuery::plane(PHI_30, 3.0, 5.0, -1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    // The criterion-only readouts stay populated even with an invalid state.
    assert_eq!(out[0].apex_valid, 1);
    assert_parity(&out[0], &q, "plane_negative_shear");
}

#[test]
fn principal_non_finite_component_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    let q = MohrCoulombYieldEnvelopeQuery::principal(PHI_30, 3.0, 0.0, [10.0, f32::NAN, 2.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_parity(&out[0], &q, "principal_nan");
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    let queries = vec![
        MohrCoulombYieldEnvelopeQuery::principal(PHI_30, 0.0, 0.0, [15.0, 10.0, 5.0]),
        MohrCoulombYieldEnvelopeQuery::plane(PHI_30, 3.0, 8.0, 6.0),
        MohrCoulombYieldEnvelopeQuery::plane(std::f32::consts::FRAC_PI_2, 3.0, 5.0, 2.0),
        MohrCoulombYieldEnvelopeQuery::plane(0.0, 4.0, 5.0, 2.0),
        MohrCoulombYieldEnvelopeQuery::principal(PHI_30, 3.0, 0.0, [10.0, f32::NAN, 2.0]),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 1);
    assert_eq!(out[2].valid, 0);
    assert_eq!(out[3].valid, 1);
    assert_eq!(out[3].apex_valid, 0);
    assert_eq!(out[4].valid, 0);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombYieldEnvelope::new(&ctx);
    let mut lcg = Lcg::new(0x00D3_7A11);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // phi well inside [0, 1.4] rad (clear of pi/2 ≈ 1.5708 where tan and
        // 1 - sin(phi) are ill-conditioned); cohesion non-negative.
        let phi = lcg.next_range(0.0, 1.4);
        let coh = lcg.next_range(0.0, 50.0);
        let sigma_n = lcg.next_range(-20.0, 60.0);
        if lcg.next_u32() & 1 == 0 {
            // Principal mode: finite triple around a positive centre.
            let center = lcg.next_range(5.0, 40.0);
            let radius = lcg.next_range(0.0, 20.0);
            let mid = lcg.next_range(center - radius, center + radius);
            queries.push(MohrCoulombYieldEnvelopeQuery::principal(
                phi,
                coh,
                sigma_n,
                [center + radius, mid, center - radius],
            ));
        } else {
            // Plane mode: non-negative shear so the state is valid.
            let tau = lcg.next_range(0.0, 40.0);
            queries.push(MohrCoulombYieldEnvelopeQuery::plane(phi, coh, sigma_n, tau));
        }
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "sweep[{i}] should be valid");
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
