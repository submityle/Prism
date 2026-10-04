//! Real-device parity for the granular-transport twin:
//! [`GpuGranularTransportClosure`](prism_volumetric_gpu::granular_transport_closure::GpuGranularTransportClosure)
//! must reproduce the `CPU` golden `GranularTransport` of
//! `prism_physics_core::collider::granular_transport`. For a valid model and
//! state the closure yields
//!
//! * the bulk viscosity
//!   `xi = (4/3) * rho_s * phi^2 * d * g0 * (1 + e) * sqrt(Theta / pi)`, and
//! * the collisional dissipation rate
//!   `gamma = (12 * (1 - e^2) / (d * sqrt(pi))) * rho_s * phi^2 * g0 * Theta^(3/2)`
//!
//! with `Theta^(3/2) = Theta * sqrt(Theta)` (no `pow`). The model is rejected
//! unless `e` is finite and in `[0, 1]`; the state is rejected unless every
//! parameter is finite with `rho_s > 0`, `phi` in `[0, 1)`, `d > 0`, `g0 >= 1`
//! and `Theta >= 0`. On rejection both coefficients are `0` and `valid = 0`.
//!
//! The oracle here is an independent re-implementation of that closed form in
//! the golden operator order, written out directly so the test never imports
//! `prism_render_architecture` or `prism_physics_core` (nor `glam`).
//!
//! The fixtures cover a representative valid state (checked against a hand
//! computation), the elastic limit `e = 1` where `gamma = 0`, the quiescent
//! limit `Theta = 0` where both coefficients vanish, several invalid models and
//! states, a non-finite input, a mixed batch that validates the `std430`
//! stride, and an empty batch the host short-circuits. A sweep over random
//! strictly-interior states follows, kept well away from the `e = 1`,
//! `Theta = 0` and `phi = 1` knees so the validity decision is stable.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through multiplies, divisions and square
//! roots, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a `GPU` may contract a multiply-add). Each valid coefficient is
//! compared with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::granular_transport`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::granular_transport_closure::{
    GpuGranularTransportClosure, GranularTransportClosureQuery, GranularTransportClosureResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces the granular-transport closure in the
/// golden operator order, returning `(xi, gamma, valid)`.
fn oracle(q: &GranularTransportClosureQuery) -> (f32, f32, u32) {
    let pi = core::f32::consts::PI;
    let e = q.restitution;
    let rho_s = q.rho_s;
    let phi = q.phi;
    let d = q.d;
    let g0 = q.g0;
    let theta = q.theta;

    let finite = e.is_finite()
        && rho_s.is_finite()
        && phi.is_finite()
        && d.is_finite()
        && g0.is_finite()
        && theta.is_finite();
    let model_ok = e >= 0.0 && e <= 1.0;
    let state_ok = rho_s > 0.0 && phi >= 0.0 && phi < 1.0 && d > 0.0 && g0 >= 1.0 && theta >= 0.0;
    if !(finite && model_ok && state_ok) {
        return (0.0, 0.0, 0);
    }

    let phi_sq = phi * phi;
    let sqrt_theta = theta.sqrt();
    let xi = (4.0 / 3.0) * rho_s * phi_sq * d * g0 * (1.0 + e) * (theta / pi).sqrt();

    let one_minus_e2 = 1.0 - e * e;
    let sqrt_pi = pi.sqrt();
    let theta_three_halves = theta * sqrt_theta;
    let prefactor = 12.0 * one_minus_e2 / (d * sqrt_pi);
    let gamma = prefactor * rho_s * phi_sq * g0 * theta_three_halves;

    (xi, gamma, 1)
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
/// `valid` flag exactly, and both coefficients to tolerance when valid.
fn assert_parity(
    gpu: &GranularTransportClosureResult,
    q: &GranularTransportClosureQuery,
    label: &str,
) {
    let (xi, gamma, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        assert!(
            close(gpu.bulk_viscosity, xi),
            "{label}: bulk_viscosity mismatch gpu={} oracle={}",
            gpu.bulk_viscosity,
            xi
        );
        assert!(
            close(gpu.collisional_dissipation, gamma),
            "{label}: collisional_dissipation mismatch gpu={} oracle={}",
            gpu.collisional_dissipation,
            gamma
        );
    }
}

#[test]
fn representative_bulk_viscosity_matches_hand_value() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularTransportClosure::new(&ctx);
    let q = GranularTransportClosureQuery::new(0.9, 2000.0, 0.5, 0.01, 2.0, 0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    // Hand computation: xi ~= 10.1065.
    assert!(close(out[0].bulk_viscosity, 10.1065));
    assert_parity(&out[0], &q, "representative_xi");
}

#[test]
fn representative_dissipation_matches_hand_value() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularTransportClosure::new(&ctx);
    let q = GranularTransportClosureQuery::new(0.9, 2000.0, 0.5, 0.01, 2.0, 0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    // Hand computation: gamma ~= 45479.6.
    assert!(close(out[0].collisional_dissipation, 45479.6));
    assert_parity(&out[0], &q, "representative_gamma");
}

#[test]
fn elastic_limit_zeroes_dissipation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularTransportClosure::new(&ctx);
    // e = 1 => (1 - e^2) = 0 => gamma = 0, but xi stays positive.
    let q = GranularTransportClosureQuery::new(1.0, 2000.0, 0.5, 0.01, 2.0, 0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].collisional_dissipation, 0.0, "elastic limit gamma");
    assert!(out[0].bulk_viscosity > 0.0, "elastic limit xi positive");
    assert_parity(&out[0], &q, "elastic_limit");
}

#[test]
fn quiescent_limit_zeroes_both_coefficients() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularTransportClosure::new(&ctx);
    // Theta = 0 => sqrt(Theta) = 0 => xi = 0 and gamma = 0, still valid.
    let q = GranularTransportClosureQuery::new(0.9, 2000.0, 0.5, 0.01, 2.0, 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].bulk_viscosity, 0.0, "quiescent xi");
    assert_eq!(out[0].collisional_dissipation, 0.0, "quiescent gamma");
    assert_parity(&out[0], &q, "quiescent_limit");
}

#[test]
fn restitution_above_one_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularTransportClosure::new(&ctx);
    let q = GranularTransportClosureQuery::new(1.5, 2000.0, 0.5, 0.01, 2.0, 0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[0].bulk_viscosity, 0.0);
    assert_eq!(out[0].collisional_dissipation, 0.0);
    assert_parity(&out[0], &q, "restitution_high");
}

#[test]
fn restitution_below_zero_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularTransportClosure::new(&ctx);
    let q = GranularTransportClosureQuery::new(-0.1, 2000.0, 0.5, 0.01, 2.0, 0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[0].bulk_viscosity, 0.0);
    assert_eq!(out[0].collisional_dissipation, 0.0);
    assert_parity(&out[0], &q, "restitution_low");
}

#[test]
fn non_finite_input_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularTransportClosure::new(&ctx);
    let nan_theta = GranularTransportClosureQuery::new(0.9, 2000.0, 0.5, 0.01, 2.0, f32::NAN);
    let inf_rho = GranularTransportClosureQuery::new(0.9, f32::INFINITY, 0.5, 0.01, 2.0, 0.5);
    let out = gpu.evaluate(&ctx, &[nan_theta, inf_rho]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 0, "NaN theta should be invalid");
    assert_eq!(out[1].valid, 0, "inf rho_s should be invalid");
    assert_parity(&out[0], &nan_theta, "nan_theta");
    assert_parity(&out[1], &inf_rho, "inf_rho");
}

#[test]
fn out_of_range_state_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularTransportClosure::new(&ctx);
    // phi = 1.0 (not < 1), g0 = 0.5 (< 1), d = 0.0 (not > 0).
    let phi_one = GranularTransportClosureQuery::new(0.9, 2000.0, 1.0, 0.01, 2.0, 0.5);
    let g0_low = GranularTransportClosureQuery::new(0.9, 2000.0, 0.5, 0.01, 0.5, 0.5);
    let d_zero = GranularTransportClosureQuery::new(0.9, 2000.0, 0.5, 0.0, 2.0, 0.5);
    let out = gpu.evaluate(&ctx, &[phi_one, g0_low, d_zero]);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].valid, 0, "phi = 1 should be invalid");
    assert_eq!(out[1].valid, 0, "g0 < 1 should be invalid");
    assert_eq!(out[2].valid, 0, "d = 0 should be invalid");
    assert_parity(&out[0], &phi_one, "phi_one");
    assert_parity(&out[1], &g0_low, "g0_low");
    assert_parity(&out[2], &d_zero, "d_zero");
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularTransportClosure::new(&ctx);
    let queries = vec![
        GranularTransportClosureQuery::new(0.9, 2000.0, 0.5, 0.01, 2.0, 0.5),
        GranularTransportClosureQuery::new(1.5, 2000.0, 0.5, 0.01, 2.0, 0.5),
        GranularTransportClosureQuery::new(0.3, 1500.0, 0.4, 0.02, 1.5, 1.0),
        GranularTransportClosureQuery::new(0.9, 2000.0, 1.0, 0.01, 2.0, 0.5),
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
    let gpu = GpuGranularTransportClosure::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularTransportClosure::new(&ctx);
    let mut lcg = Lcg::new(0x51A9_3C7D);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Strictly-interior state, kept well away from the e = 1, Theta = 0 and
        // phi = 1 knees so the validity decision cannot flip under round-off.
        let restitution = lcg.next_range(0.1, 0.95);
        let rho_s = lcg.next_range(100.0, 5000.0);
        let phi = lcg.next_range(0.05, 0.9);
        let d = lcg.next_range(0.001, 0.1);
        let g0 = lcg.next_range(1.0, 5.0);
        let theta = lcg.next_range(0.05, 2.0);
        queries.push(GranularTransportClosureQuery::new(
            restitution,
            rho_s,
            phi,
            d,
            g0,
            theta,
        ));
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
