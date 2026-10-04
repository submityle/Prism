//! Real-device parity for the dense-granular-gas equation-of-state twin:
//! [`GpuGranularEquationOfState`](prism_volumetric_gpu::granular_equation_of_state::GpuGranularEquationOfState)
//! must reproduce the `CPU` golden `GranularEquationOfState` of
//! `prism_physics_core::collider::granular_eos`. For restitution `e`, solid
//! fraction `φ`, contact value `g0`, bulk density `ρ` and granular temperature
//! `T`, the compressibility factor is `Z = 1 + 2 (1 + e) φ g0`, the collisional
//! enhancement is `Z - 1`, the kinetic pressure is `p_k = ρ T`, the collisional
//! pressure is `p_c = p_k (Z - 1)`, and the total pressure is `p = p_k Z`.
//!
//! The oracle here is an independent re-implementation of those closed forms —
//! the constructor gate on `e`, the packing gate on `φ`/`g0`, the kinetic gate
//! on `ρ`/`T`, then the pure multiply/add relations in golden operator order —
//! written out directly so the test never imports `prism_physics_core` or
//! `prism_render_architecture`.
//!
//! The fixtures cover a valid interior state (with hand-computed values), the
//! dilute limit `φ = 0` (`Z = 1`), zero temperature (all pressures `0` yet
//! valid), an invalid restitution (every flag `0`), an invalid packing (only
//! the kinetic pressure survives), an invalid kinetic state (only the
//! compressibility factor and enhancement survive), non-finite inputs, a batch
//! of four heterogeneous elements validating the `std430` stride, and an empty
//! batch the host short-circuits with no dispatch. A `512`-step sweep over
//! inputs held well inside the valid region follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through multiplies and adds, so `CPU` and
//! `GPU` evaluate the same closed form but need not be bit-exact (a `GPU` may
//! contract a multiply-add). Each valid scalar is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid`
//! flags are compared exactly. The fixtures keep inputs well away from the
//! validity knees so the decisions agree on both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::granular_eos`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::granular_equation_of_state::{
    GpuGranularEquationOfState, GranularEquationOfStateQuery, GranularEquationOfStateResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces every `GranularEquationOfState` accessor
/// in the golden operator order. The restitution gate stands in for the golden
/// model's constructor, so every accessor is undefined when `e` is invalid.
fn oracle(q: &GranularEquationOfStateQuery) -> GranularEquationOfStateResult {
    let e = q.restitution;
    let phi = q.solid_fraction;
    let g0 = q.g0;
    let rho = q.bulk_density;
    let temp = q.temperature;

    let e_ok = e.is_finite() && (0.0..=1.0).contains(&e);
    let phi_ok = phi.is_finite() && (0.0..1.0).contains(&phi);
    let g0_ok = g0.is_finite() && g0 >= 1.0;
    let packing_ok = phi_ok && g0_ok;
    let rho_ok = rho.is_finite() && rho > 0.0;
    let temp_ok = temp.is_finite() && temp >= 0.0;
    let kinetic_ok = rho_ok && temp_ok;

    let z_ok = e_ok && packing_ok;
    let kin_ok = e_ok && kinetic_ok;
    let both_ok = e_ok && packing_ok && kinetic_ok;

    let enhancement = 2.0 * (1.0 + e) * phi * g0;
    let z = 1.0 + enhancement;
    let p_k = rho * temp;
    let p_c = p_k * enhancement;
    let p = p_k * z;

    GranularEquationOfStateResult {
        compressibility_factor: if z_ok { z } else { 0.0 },
        compressibility_valid: u32::from(z_ok),
        collisional_enhancement: if z_ok { enhancement } else { 0.0 },
        enhancement_valid: u32::from(z_ok),
        kinetic_pressure: if kin_ok { p_k } else { 0.0 },
        kinetic_valid: u32::from(kin_ok),
        collisional_pressure: if both_ok { p_c } else { 0.0 },
        collisional_valid: u32::from(both_ok),
        pressure: if both_ok { p } else { 0.0 },
        pressure_valid: u32::from(both_ok),
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

/// Asserts a single `GPU` result matches the independent oracle: every discrete
/// `valid` flag exactly, and each scalar to tolerance when its flag is `1`.
fn assert_parity(
    gpu: &GranularEquationOfStateResult,
    q: &GranularEquationOfStateQuery,
    label: &str,
) {
    let want = oracle(q);

    assert_eq!(
        gpu.compressibility_valid, want.compressibility_valid,
        "{label}: compressibility_valid mismatch"
    );
    assert_eq!(
        gpu.enhancement_valid, want.enhancement_valid,
        "{label}: enhancement_valid mismatch"
    );
    assert_eq!(
        gpu.kinetic_valid, want.kinetic_valid,
        "{label}: kinetic_valid mismatch"
    );
    assert_eq!(
        gpu.collisional_valid, want.collisional_valid,
        "{label}: collisional_valid mismatch"
    );
    assert_eq!(
        gpu.pressure_valid, want.pressure_valid,
        "{label}: pressure_valid mismatch"
    );

    if want.compressibility_valid == 1 {
        assert!(
            close(gpu.compressibility_factor, want.compressibility_factor),
            "{label}: Z mismatch gpu={} oracle={}",
            gpu.compressibility_factor,
            want.compressibility_factor
        );
    }
    if want.enhancement_valid == 1 {
        assert!(
            close(gpu.collisional_enhancement, want.collisional_enhancement),
            "{label}: enhancement mismatch gpu={} oracle={}",
            gpu.collisional_enhancement,
            want.collisional_enhancement
        );
    }
    if want.kinetic_valid == 1 {
        assert!(
            close(gpu.kinetic_pressure, want.kinetic_pressure),
            "{label}: p_k mismatch gpu={} oracle={}",
            gpu.kinetic_pressure,
            want.kinetic_pressure
        );
    }
    if want.collisional_valid == 1 {
        assert!(
            close(gpu.collisional_pressure, want.collisional_pressure),
            "{label}: p_c mismatch gpu={} oracle={}",
            gpu.collisional_pressure,
            want.collisional_pressure
        );
    }
    if want.pressure_valid == 1 {
        assert!(
            close(gpu.pressure, want.pressure),
            "{label}: p mismatch gpu={} oracle={}",
            gpu.pressure,
            want.pressure
        );
    }
}

#[test]
fn valid_interior_state_matches_hand_computation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularEquationOfState::new(&ctx);
    // e=0.8, phi=0.3, g0=2.5 => enhancement = 2*1.8*0.3*2.5 = 2.7, Z = 3.7.
    // rho=10, T=2 => p_k = 20, p_c = 54, p = 74.
    let q = GranularEquationOfStateQuery::new(0.8, 0.3, 2.5, 10.0, 2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].compressibility_valid, 1);
    assert_eq!(out[0].enhancement_valid, 1);
    assert_eq!(out[0].kinetic_valid, 1);
    assert_eq!(out[0].collisional_valid, 1);
    assert_eq!(out[0].pressure_valid, 1);
    assert!(close(out[0].compressibility_factor, 3.7));
    assert!(close(out[0].collisional_enhancement, 2.7));
    assert!(close(out[0].kinetic_pressure, 20.0));
    assert!(close(out[0].collisional_pressure, 54.0));
    assert!(close(out[0].pressure, 74.0));
    assert_parity(&out[0], &q, "interior");
}

#[test]
fn dilute_limit_is_ideal_gas() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularEquationOfState::new(&ctx);
    // phi = 0 => enhancement = 0, Z = 1, p = p_k.
    let q = GranularEquationOfStateQuery::new(0.9, 0.0, 1.0, 10.0, 2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].compressibility_valid, 1);
    assert!(close(out[0].compressibility_factor, 1.0));
    assert!(close(out[0].collisional_enhancement, 0.0));
    assert!(close(out[0].kinetic_pressure, 20.0));
    assert!(close(out[0].collisional_pressure, 0.0));
    assert!(close(out[0].pressure, 20.0));
    assert_parity(&out[0], &q, "dilute");
}

#[test]
fn zero_temperature_is_valid_with_zero_pressure() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularEquationOfState::new(&ctx);
    // T = 0 is admissible (T >= 0): kinetic and both pressures are 0 yet valid.
    let q = GranularEquationOfStateQuery::new(0.5, 0.3, 2.0, 10.0, 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].kinetic_valid, 1);
    assert_eq!(out[0].collisional_valid, 1);
    assert_eq!(out[0].pressure_valid, 1);
    assert!(close(out[0].kinetic_pressure, 0.0));
    assert!(close(out[0].collisional_pressure, 0.0));
    assert!(close(out[0].pressure, 0.0));
    // Z = 1 + 2*1.5*0.3*2 = 2.8 is still defined.
    assert!(close(out[0].compressibility_factor, 2.8));
    assert_parity(&out[0], &q, "zero_temp");
}

#[test]
fn invalid_restitution_clears_every_flag() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularEquationOfState::new(&ctx);
    let over = GranularEquationOfStateQuery::new(1.1, 0.3, 2.0, 10.0, 2.0);
    let nan = GranularEquationOfStateQuery::new(f32::NAN, 0.3, 2.0, 10.0, 2.0);
    let out = gpu.evaluate(&ctx, &[over, nan]);
    assert_eq!(out.len(), 2);
    for (res, label) in out.iter().zip(["over", "nan"]) {
        assert_eq!(res.compressibility_valid, 0, "{label}");
        assert_eq!(res.enhancement_valid, 0, "{label}");
        assert_eq!(res.kinetic_valid, 0, "{label}");
        assert_eq!(res.collisional_valid, 0, "{label}");
        assert_eq!(res.pressure_valid, 0, "{label}");
    }
    assert_parity(&out[0], &over, "over");
    assert_parity(&out[1], &nan, "nan");
}

#[test]
fn invalid_packing_keeps_only_the_kinetic_pressure() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularEquationOfState::new(&ctx);
    // phi = 1.0 is excluded ([0, 1)); kinetic state stays valid.
    let q = GranularEquationOfStateQuery::new(0.5, 1.0, 2.0, 10.0, 2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].compressibility_valid, 0);
    assert_eq!(out[0].enhancement_valid, 0);
    assert_eq!(out[0].kinetic_valid, 1);
    assert_eq!(out[0].collisional_valid, 0);
    assert_eq!(out[0].pressure_valid, 0);
    assert!(close(out[0].kinetic_pressure, 20.0));
    assert_parity(&out[0], &q, "bad_packing");
}

#[test]
fn invalid_kinetic_keeps_only_the_compressibility() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularEquationOfState::new(&ctx);
    // rho = 0 invalidates the kinetic state; packing stays valid.
    let q = GranularEquationOfStateQuery::new(0.5, 0.3, 2.0, 0.0, 2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].compressibility_valid, 1);
    assert_eq!(out[0].enhancement_valid, 1);
    assert_eq!(out[0].kinetic_valid, 0);
    assert_eq!(out[0].collisional_valid, 0);
    assert_eq!(out[0].pressure_valid, 0);
    // Z = 1 + 2*1.5*0.3*2 = 2.8, enhancement = 1.8.
    assert!(close(out[0].compressibility_factor, 2.8));
    assert!(close(out[0].collisional_enhancement, 1.8));
    assert_parity(&out[0], &q, "bad_kinetic");
}

#[test]
fn non_finite_inputs_are_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularEquationOfState::new(&ctx);
    // Non-finite temperature invalidates the kinetic chain; packing survives.
    let inf_temp = GranularEquationOfStateQuery::new(0.5, 0.3, 2.0, 10.0, f32::INFINITY);
    // Non-finite g0 invalidates packing; kinetic survives.
    let nan_g0 = GranularEquationOfStateQuery::new(0.5, 0.3, f32::NAN, 10.0, 2.0);
    let out = gpu.evaluate(&ctx, &[inf_temp, nan_g0]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].compressibility_valid, 1);
    assert_eq!(out[0].kinetic_valid, 0);
    assert_eq!(out[0].pressure_valid, 0);
    assert_eq!(out[1].compressibility_valid, 0);
    assert_eq!(out[1].kinetic_valid, 1);
    assert_eq!(out[1].pressure_valid, 0);
    assert_parity(&out[0], &inf_temp, "inf_temp");
    assert_parity(&out[1], &nan_g0, "nan_g0");
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularEquationOfState::new(&ctx);
    let queries = vec![
        GranularEquationOfStateQuery::new(0.8, 0.3, 2.5, 10.0, 2.0),
        GranularEquationOfStateQuery::new(1.1, 0.3, 2.0, 10.0, 2.0),
        GranularEquationOfStateQuery::new(0.5, 1.0, 2.0, 10.0, 2.0),
        GranularEquationOfStateQuery::new(0.3, 0.2, 1.5, 0.0, 2.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].pressure_valid, 1);
    assert_eq!(out[1].compressibility_valid, 0);
    assert_eq!(out[2].compressibility_valid, 0);
    assert_eq!(out[2].kinetic_valid, 1);
    assert_eq!(out[3].compressibility_valid, 1);
    assert_eq!(out[3].kinetic_valid, 0);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularEquationOfState::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularEquationOfState::new(&ctx);
    let mut lcg = Lcg::new(0x6E05_17A7);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Hold every input well inside the valid region so no flag can flip.
        let restitution = lcg.next_range(0.0, 1.0);
        let solid_fraction = lcg.next_range(0.0, 0.9);
        let g0 = lcg.next_range(1.0, 5.0);
        let bulk_density = lcg.next_range(0.5, 50.0);
        let temperature = lcg.next_range(0.0, 10.0);
        queries.push(GranularEquationOfStateQuery::new(
            restitution,
            solid_fraction,
            g0,
            bulk_density,
            temperature,
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
