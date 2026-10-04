//! Real-device parity for the peak-adhesion twin:
//! [`GpuCapillaryMaxForce`](prism_volumetric_gpu::capillary_max_force::GpuCapillaryMaxForce)
//! must reproduce the `CPU` golden `CapillaryBridgeModel::max_force` of
//! `prism_physics_core::collider::capillary_bridge`. The peak adhesion of a
//! pendular liquid bridge is `F₀ = 2π·R·γ·cos θ`, where the reduced radius is
//! `R = 2·r_a·r_b / (r_a + r_b)`, `γ` the liquid surface tension and `θ` the
//! solid–liquid contact angle, provided every input is finite and
//! `r_a > 0`, `r_b > 0`, `γ > 0` and `θ ∈ [0, π/2]`; the pair is invalid
//! (`force = 0`) otherwise.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the finiteness, positivity and angle-range guards, then `2·a·b / (a + b)`
//! in the golden operator order and `2π·R·γ·cos θ` with the cosine evaluated in
//! `f64` and narrowed to `f32` exactly as the golden does — written out
//! directly so the test never imports `prism_render_architecture` or
//! `prism_physics_core`.
//!
//! The fixtures cover `θ = 0` (so `cos θ = 1`), `θ = π/2` (so `cos θ ≈ 0` and
//! the force collapses to zero), equal and mismatched radii, each degenerate
//! input class (non-positive radius, non-positive tension, out-of-range angle,
//! and non-finite inputs), a batch of two or more elements that validates the
//! `std430` stride, and an empty batch the host short-circuits with no
//! dispatch. A sweep over random strictly-valid inputs follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through multiplies, a division and a
//! cosine, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a `GPU` may contract a multiply-add, and the golden's `f64`
//! cosine differs slightly from the kernel's native `f32` cosine). The valid
//! `force` scalar is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly. The
//! random fixtures keep inputs strictly inside the valid region so the
//! validity decision agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::capillary_bridge`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::capillary_max_force::{
    CapillaryMaxForceQuery, CapillaryMaxForceResult, GpuCapillaryMaxForce,
};
use prism_volumetric_gpu::GpuContext;
use std::f32::consts::PI;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Half of `π`, the inclusive upper bound of the valid contact-angle range.
const HALF_PI: f32 = PI / 2.0;

/// Independent host oracle: reproduces `max_force` in the golden operator
/// order, returning the peak force and the validity flag. The cosine is taken
/// in `f64` then narrowed to `f32`, matching the golden exactly.
fn oracle(q: &CapillaryMaxForceQuery) -> (f32, u32) {
    let ra = q.radius_a;
    let rb = q.radius_b;
    let gamma = q.surface_tension;
    let theta = q.contact_angle;
    let finite = ra.is_finite() && rb.is_finite() && gamma.is_finite() && theta.is_finite();
    let region = ra > 0.0 && rb > 0.0 && gamma > 0.0 && theta >= 0.0 && theta <= HALF_PI;
    if !finite || !region {
        return (0.0, 0);
    }
    let reduced = 2.0 * ra * rb / (ra + rb);
    let cos_theta = (theta as f64).cos() as f32;
    (2.0 * PI * reduced * gamma * cos_theta, 1)
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
/// `valid` flag exactly, and the `force` scalar to tolerance when valid.
fn assert_parity(gpu: &CapillaryMaxForceResult, q: &CapillaryMaxForceQuery, label: &str) {
    let (force, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        assert!(
            close(gpu.force, force),
            "{label}: force mismatch gpu={} oracle={}",
            gpu.force,
            force
        );
    } else {
        assert_eq!(gpu.force, 0.0, "{label}: invalid force must be zero");
    }
}

#[test]
fn zero_angle_gives_full_cosine_force() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryMaxForce::new(&ctx);
    let q = CapillaryMaxForceQuery::new(0.5, 0.5, 0.072, 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    let expected = 2.0 * PI * 0.5 * 0.072;
    assert!(close(out[0].force, expected));
    assert_parity(&out[0], &q, "zero_angle");
}

#[test]
fn right_angle_cancels_the_force() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryMaxForce::new(&ctx);
    let q = CapillaryMaxForceQuery::new(1.0, 3.0, 0.072, HALF_PI);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].force, 0.0), "force={}", out[0].force);
    assert_parity(&out[0], &q, "right_angle");
}

#[test]
fn mismatched_radii_normal_case() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryMaxForce::new(&ctx);
    let q = CapillaryMaxForceQuery::new(1.0, 3.0, 0.05, PI / 3.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    let expected = 2.0 * PI * 1.5 * 0.05 * (PI / 3.0).cos();
    assert!(close(out[0].force, expected));
    assert_parity(&out[0], &q, "mismatched");
}

#[test]
fn non_positive_radius_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryMaxForce::new(&ctx);
    let zero = CapillaryMaxForceQuery::new(0.0, 1.0, 0.072, 0.3);
    let negative = CapillaryMaxForceQuery::new(1.0, -2.0, 0.072, 0.3);
    let out = gpu.evaluate(&ctx, &[zero, negative]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[1].valid, 0);
    assert_eq!(out[0].force, 0.0);
    assert_eq!(out[1].force, 0.0);
    assert_parity(&out[0], &zero, "zero_radius");
    assert_parity(&out[1], &negative, "negative_radius");
}

#[test]
fn non_positive_tension_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryMaxForce::new(&ctx);
    let zero = CapillaryMaxForceQuery::new(1.0, 1.0, 0.0, 0.3);
    let negative = CapillaryMaxForceQuery::new(1.0, 1.0, -0.5, 0.3);
    let out = gpu.evaluate(&ctx, &[zero, negative]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[1].valid, 0);
    assert_parity(&out[0], &zero, "zero_tension");
    assert_parity(&out[1], &negative, "negative_tension");
}

#[test]
fn out_of_range_angle_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryMaxForce::new(&ctx);
    let too_large = CapillaryMaxForceQuery::new(1.0, 1.0, 0.072, HALF_PI + 0.1);
    let negative = CapillaryMaxForceQuery::new(1.0, 1.0, 0.072, -0.1);
    let out = gpu.evaluate(&ctx, &[too_large, negative]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 0, "angle above PI/2 should be invalid");
    assert_eq!(out[1].valid, 0, "negative angle should be invalid");
    assert_parity(&out[0], &too_large, "angle_too_large");
    assert_parity(&out[1], &negative, "angle_negative");
}

#[test]
fn non_finite_inputs_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryMaxForce::new(&ctx);
    let inf = CapillaryMaxForceQuery::new(f32::INFINITY, 1.0, 0.072, 0.3);
    let nan = CapillaryMaxForceQuery::new(1.0, f32::NAN, 0.072, 0.3);
    let nan_angle = CapillaryMaxForceQuery::new(1.0, 1.0, 0.072, f32::NAN);
    let out = gpu.evaluate(&ctx, &[inf, nan, nan_angle]);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].valid, 0, "infinite radius should be invalid");
    assert_eq!(out[1].valid, 0, "NaN radius should be invalid");
    assert_eq!(out[2].valid, 0, "NaN angle should be invalid");
    for res in &out {
        assert_eq!(res.force, 0.0);
    }
    assert_parity(&out[0], &inf, "infinite");
    assert_parity(&out[1], &nan, "nan_radius");
    assert_parity(&out[2], &nan_angle, "nan_angle");
}

#[test]
fn stride_regression_two_element_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryMaxForce::new(&ctx);
    let queries = [
        CapillaryMaxForceQuery::new(0.3, 0.7, 0.05, 0.2),
        CapillaryMaxForceQuery::new(2.0, 5.0, 0.1, 0.9),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 1);
    assert_ne!(out[0].force, out[1].force);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("stride[{i}]"));
    }
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryMaxForce::new(&ctx);
    let queries = vec![
        CapillaryMaxForceQuery::new(1.0, 1.0, 0.072, 0.0),
        CapillaryMaxForceQuery::new(-1.0, 5.0, 0.072, 0.3),
        CapillaryMaxForceQuery::new(4.0, 12.0, 0.05, PI / 4.0),
        CapillaryMaxForceQuery::new(1.0, 1.0, 0.0, 0.3),
        CapillaryMaxForceQuery::new(2.0, 2.0, 0.08, HALF_PI),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 0);
    assert_eq!(out[2].valid, 1);
    assert_eq!(out[3].valid, 0);
    assert_eq!(out[4].valid, 1);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("mixed[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryMaxForce::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryMaxForce::new(&ctx);
    let mut lcg = Lcg::new(0x0CA9_11A5);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let radius_a = lcg.next_range(0.1, 10.0);
        let radius_b = lcg.next_range(0.1, 10.0);
        let surface_tension = lcg.next_range(0.01, 5.0);
        let contact_angle = lcg.next_range(0.0, HALF_PI);
        queries.push(CapillaryMaxForceQuery::new(
            radius_a,
            radius_b,
            surface_tension,
            contact_angle,
        ));
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
