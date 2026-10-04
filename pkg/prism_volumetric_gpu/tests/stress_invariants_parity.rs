//! Real-device parity for the stress-invariant twin:
//! [`GpuStressInvariants`](prism_volumetric_gpu::stress_invariants::GpuStressInvariants)
//! must reproduce the `CPU` golden `StressInvariants` of
//! `prism_physics_core::collider::stress_invariants` for one principal-stress
//! triple per thread. A triple is decomposed into the standard `p`–`q`–`theta`
//! state descriptors: the mean stress `p`, the deviatoric invariants `J2` and
//! `J3`, the von Mises equivalent `q`, the octahedral shear `tau_oct`, the
//! deviatoric Frobenius norm, and the Lode parameter `mu_L` with the Lode angle
//! `theta`.
//!
//! The oracle below is an independent re-implementation of that closed form —
//! the finiteness guard, the descending sort of three, then the invariant and
//! Lode formulas in the golden operator order (with the Lode angle evaluated in
//! `f64` like the reference) — written out directly so the test never imports
//! `prism_render_architecture`, `prism_physics_core` or `glam`.
//!
//! The fixtures cover the three canonical states (triaxial compression,
//! triaxial extension, pure shear) with hand-checked analytic values, the
//! descending sort, translation covariance of the deviatoric part, the
//! `q = (3 / sqrt(2)) tau_oct` identity, a hydrostatic state (`lode_valid = 0`),
//! near-hydrostatic spans kept clear of the `1e-9` knee, non-finite inputs
//! (`valid = 0`, all outputs `0`), a mixed batch validating the `std430` stride,
//! an empty batch the host short-circuits, and a `512`-step random sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The golden evaluates the Lode angle in `f64` and casts to `f32` whereas the
//! kernel uses the portable `f32` `atan`, so the two sides are not bit-exact.
//! Each continuous output is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` and `lode_valid` flags are
//! compared exactly. The sweep keeps the span `sigma1 - sigma3` well clear of
//! the hydrostatic knee so the `lode_valid` decision cannot flip by round-off.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::stress_invariants`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::stress_invariants::{
    GpuStressInvariants, StressInvariantsQuery, StressInvariantsResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Finiteness via the same ordered `abs < 3.0e38` test the kernel uses, which
/// rejects both infinities and `NaN`.
fn finite(x: f32) -> bool {
    x.abs() < 3.0e38
}

/// Independent host oracle: reproduces `StressInvariants::from_principal_stresses`
/// plus every getter in the golden operator order, with the Lode angle computed
/// in `f64` like the reference. Produces every field of the public result.
fn oracle(q: &StressInvariantsQuery) -> StressInvariantsResult {
    let x0 = q.principal[0];
    let x1 = q.principal[1];
    let x2 = q.principal[2];
    let ok = finite(x0) && finite(x1) && finite(x2);
    if !ok {
        return StressInvariantsResult {
            mean_stress: 0.0,
            j2: 0.0,
            j3: 0.0,
            von_mises: 0.0,
            octahedral_shear: 0.0,
            deviatoric_norm: 0.0,
            lode_parameter: 0.0,
            lode_angle: 0.0,
            lode_valid: 0,
            valid: 0,
        };
    }

    // Descending sort of three via min/max compare-exchange: a >= b >= c, the
    // same ordering the kernel performs.
    let hi0 = x0.max(x1);
    let lo0 = x0.min(x1);
    let a = hi0.max(x2);
    let t = hi0.min(x2);
    let b = lo0.max(t);
    let c = lo0.min(t);

    let p = (a + b + c) / 3.0;
    let d0 = a - p;
    let d1 = b - p;
    let d2 = c - p;
    let j2 = 0.5 * (d0 * d0 + d1 * d1 + d2 * d2);
    let j3 = d0 * d1 * d2;
    let von_mises = (3.0 * j2).sqrt();
    let octahedral_shear = (2.0 * j2 / 3.0).sqrt();
    let deviatoric_norm = (2.0 * j2).sqrt();

    let span = a - c;
    let lode_ok = span > 1.0e-9;
    let (lode_parameter, lode_angle, lode_valid) = if lode_ok {
        let mu = (2.0 * b - a - c) / span;
        let theta = (f64::from(mu) / 3.0_f64.sqrt()).atan() as f32;
        (mu, theta, 1)
    } else {
        (0.0, 0.0, 0)
    };

    StressInvariantsResult {
        mean_stress: p,
        j2,
        j3,
        von_mises,
        octahedral_shear,
        deviatoric_norm,
        lode_parameter,
        lode_angle,
        lode_valid,
        valid: 1,
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
/// `valid` and `lode_valid` flags exactly, and every continuous scalar to
/// tolerance when its validity flag is set.
fn assert_parity(gpu: &StressInvariantsResult, q: &StressInvariantsQuery, label: &str) {
    let want = oracle(q);
    assert_eq!(gpu.valid, want.valid, "{label}: valid flag mismatch");
    assert_eq!(
        gpu.lode_valid, want.lode_valid,
        "{label}: lode_valid flag mismatch"
    );
    if want.valid == 1 {
        assert!(
            close(gpu.mean_stress, want.mean_stress),
            "{label}: mean_stress gpu={} oracle={}",
            gpu.mean_stress,
            want.mean_stress
        );
        assert!(
            close(gpu.j2, want.j2),
            "{label}: j2 gpu={} oracle={}",
            gpu.j2,
            want.j2
        );
        assert!(
            close(gpu.j3, want.j3),
            "{label}: j3 gpu={} oracle={}",
            gpu.j3,
            want.j3
        );
        assert!(
            close(gpu.von_mises, want.von_mises),
            "{label}: von_mises gpu={} oracle={}",
            gpu.von_mises,
            want.von_mises
        );
        assert!(
            close(gpu.octahedral_shear, want.octahedral_shear),
            "{label}: octahedral_shear gpu={} oracle={}",
            gpu.octahedral_shear,
            want.octahedral_shear
        );
        assert!(
            close(gpu.deviatoric_norm, want.deviatoric_norm),
            "{label}: deviatoric_norm gpu={} oracle={}",
            gpu.deviatoric_norm,
            want.deviatoric_norm
        );
    }
    if want.lode_valid == 1 {
        assert!(
            close(gpu.lode_parameter, want.lode_parameter),
            "{label}: lode_parameter gpu={} oracle={}",
            gpu.lode_parameter,
            want.lode_parameter
        );
        assert!(
            close(gpu.lode_angle, want.lode_angle),
            "{label}: lode_angle gpu={} oracle={}",
            gpu.lode_angle,
            want.lode_angle
        );
    }
}

/// Sixth of pi, the `+30` degree Lode-angle bound for the canonical states.
const SIXTH_PI: f32 = std::f32::consts::FRAC_PI_6;

#[test]
fn triaxial_compression_hand_values() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStressInvariants::new(&ctx);
    // sigma = [3, 1, 1] -> mu_L = -1 (triaxial compression).
    let q = StressInvariantsQuery::new([3.0, 1.0, 1.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let r = out[0];
    assert_eq!(r.valid, 1);
    assert_eq!(r.lode_valid, 1);
    assert!(close(r.mean_stress, 5.0 / 3.0));
    assert!(close(r.j2, 4.0 / 3.0));
    assert!(close(r.von_mises, 2.0));
    assert!(close(r.j3, 16.0 / 27.0));
    assert!(close(r.octahedral_shear, (8.0_f32 / 9.0).sqrt()));
    assert!(close(r.deviatoric_norm, (8.0_f32 / 3.0).sqrt()));
    assert!(close(r.lode_parameter, -1.0));
    assert!(close(r.lode_angle, -SIXTH_PI));
    assert_parity(&r, &q, "triaxial_compression");
}

#[test]
fn triaxial_extension_hand_values() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStressInvariants::new(&ctx);
    // sigma = [3, 3, 1] -> mu_L = +1 (triaxial extension).
    let q = StressInvariantsQuery::new([3.0, 3.0, 1.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let r = out[0];
    assert_eq!(r.valid, 1);
    assert_eq!(r.lode_valid, 1);
    assert!(close(r.mean_stress, 7.0 / 3.0));
    assert!(close(r.j2, 4.0 / 3.0));
    assert!(close(r.von_mises, 2.0));
    assert!(close(r.lode_parameter, 1.0));
    assert!(close(r.lode_angle, SIXTH_PI));
    assert_parity(&r, &q, "triaxial_extension");
}

#[test]
fn pure_shear_hand_values() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStressInvariants::new(&ctx);
    // sigma = [1, 0, -1] -> mu_L = 0 (pure shear).
    let q = StressInvariantsQuery::new([1.0, 0.0, -1.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let r = out[0];
    assert_eq!(r.valid, 1);
    assert_eq!(r.lode_valid, 1);
    assert!(r.mean_stress.abs() <= 1.0e-4);
    assert!(close(r.j2, 1.0));
    assert!(close(r.von_mises, 3.0_f32.sqrt()));
    assert!(r.j3.abs() <= 1.0e-4);
    assert!(r.lode_parameter.abs() <= 1.0e-4);
    assert!(r.lode_angle.abs() <= 1.0e-4);
    assert_parity(&r, &q, "pure_shear");
}

#[test]
fn sorts_principal_descending() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStressInvariants::new(&ctx);
    // Unordered input must decompose like its descending reordering [3, 1, 1].
    let q = StressInvariantsQuery::new([1.0, 3.0, 1.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let r = out[0];
    assert_eq!(r.valid, 1);
    assert!(close(r.mean_stress, 5.0 / 3.0));
    assert!(close(r.lode_parameter, -1.0));
    assert_parity(&r, &q, "sorts_descending");
}

#[test]
fn translation_covariance_of_deviatoric_part() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStressInvariants::new(&ctx);
    // Adding a hydrostatic offset shifts only p; the deviatoric part and Lode
    // parameter are invariant.
    let base = StressInvariantsQuery::new([3.0, 1.0, -2.0]);
    let shifted = StressInvariantsQuery::new([13.0, 11.0, 8.0]);
    let out = gpu.evaluate(&ctx, &[base, shifted]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 1);
    assert!(close(out[1].mean_stress - out[0].mean_stress, 10.0));
    assert!(close(out[1].j2, out[0].j2));
    assert!(close(out[1].lode_parameter, out[0].lode_parameter));
    assert_parity(&out[0], &base, "covariance_base");
    assert_parity(&out[1], &shifted, "covariance_shifted");
}

#[test]
fn von_mises_matches_octahedral_relation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStressInvariants::new(&ctx);
    // q = (3 / sqrt(2)) * tau_oct for any state.
    let q = StressInvariantsQuery::new([5.0, 2.0, -1.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let r = out[0];
    assert_eq!(r.valid, 1);
    let expected = 3.0 / 2.0_f32.sqrt() * r.octahedral_shear;
    assert!(close(r.von_mises, expected));
    assert_parity(&r, &q, "von_mises_octahedral");
}

#[test]
fn hydrostatic_has_no_lode() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStressInvariants::new(&ctx);
    let q = StressInvariantsQuery::new([2.0, 2.0, 2.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let r = out[0];
    assert_eq!(r.valid, 1);
    assert_eq!(r.lode_valid, 0);
    assert!(close(r.mean_stress, 2.0));
    assert!(r.j2.abs() <= 1.0e-4);
    assert!(r.von_mises.abs() <= 1.0e-4);
    assert!(r.deviatoric_norm.abs() <= 1.0e-4);
    assert_eq!(r.lode_parameter, 0.0);
    assert_eq!(r.lode_angle, 0.0);
    assert_parity(&r, &q, "hydrostatic");
}

#[test]
fn near_hydrostatic_boundary_is_stable() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStressInvariants::new(&ctx);
    // Spans kept clear of the 1e-9 knee: one clearly above (defined), one
    // clearly below (undefined). Both sides agree on lode_valid.
    let above = StressInvariantsQuery::new([1.0e-6, 0.0, 0.0]);
    let below = StressInvariantsQuery::new([1.0e-11, 0.0, 0.0]);
    let out = gpu.evaluate(&ctx, &[above, below]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].lode_valid, 1, "span above knee is defined");
    assert_eq!(out[1].valid, 1);
    assert_eq!(out[1].lode_valid, 0, "span below knee is undefined");
    assert_parity(&out[0], &above, "near_above");
    assert_parity(&out[1], &below, "near_below");
}

#[test]
fn non_finite_nan_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStressInvariants::new(&ctx);
    let q = StressInvariantsQuery::new([f32::NAN, 1.0, 0.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let r = out[0];
    assert_eq!(r.valid, 0);
    assert_eq!(r.lode_valid, 0);
    assert_eq!(r.mean_stress, 0.0);
    assert_eq!(r.j2, 0.0);
    assert_eq!(r.j3, 0.0);
    assert_eq!(r.von_mises, 0.0);
    assert_eq!(r.octahedral_shear, 0.0);
    assert_eq!(r.deviatoric_norm, 0.0);
    assert_eq!(r.lode_parameter, 0.0);
    assert_eq!(r.lode_angle, 0.0);
    assert_parity(&r, &q, "nan");
}

#[test]
fn non_finite_infinity_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStressInvariants::new(&ctx);
    let pos = StressInvariantsQuery::new([1.0, f32::INFINITY, 0.0]);
    let neg = StressInvariantsQuery::new([f32::NEG_INFINITY, 2.0, 3.0]);
    let out = gpu.evaluate(&ctx, &[pos, neg]);
    assert_eq!(out.len(), 2);
    for (r, label) in out.iter().zip(["pos_inf", "neg_inf"]) {
        assert_eq!(r.valid, 0, "{label}: should be invalid");
        assert_eq!(r.lode_valid, 0);
        assert_eq!(r.mean_stress, 0.0);
        assert_eq!(r.j2, 0.0);
    }
    assert_parity(&out[0], &pos, "pos_inf");
    assert_parity(&out[1], &neg, "neg_inf");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStressInvariants::new(&ctx);
    let queries = vec![
        StressInvariantsQuery::new([3.0, 1.0, 1.0]),
        StressInvariantsQuery::new([f32::NAN, 0.0, 0.0]),
        StressInvariantsQuery::new([2.0, 2.0, 2.0]),
        StressInvariantsQuery::new([5.0, 2.0, -1.0]),
        StressInvariantsQuery::new([1.0, f32::INFINITY, -3.0]),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].lode_valid, 1);
    assert_eq!(out[1].valid, 0);
    assert_eq!(out[2].valid, 1);
    assert_eq!(out[2].lode_valid, 0);
    assert_eq!(out[3].valid, 1);
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
    let gpu = GpuStressInvariants::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStressInvariants::new(&ctx);
    let mut lcg = Lcg::new(0x51A7_3E22);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Build a center and a clearly-positive radius so the span is well
        // clear of the hydrostatic knee, with a mid value inside [low, high].
        let center = lcg.next_range(-50.0, 50.0);
        let radius = lcg.next_range(0.5, 50.0);
        let high = center + radius;
        let low = center - radius;
        let mid = lcg.next_range(low, high);
        // Present the triple shuffled so the kernel's descending sort is tested.
        queries.push(StressInvariantsQuery::new([mid, high, low]));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "sweep[{i}] should be valid");
        assert_eq!(res.lode_valid, 1, "sweep[{i}] should have a defined Lode");
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
