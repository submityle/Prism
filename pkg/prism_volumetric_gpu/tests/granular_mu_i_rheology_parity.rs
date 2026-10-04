//! Real-device parity for the granular `μ(I)`-rheology twin:
//! [`GpuGranularMuIRheology`](prism_volumetric_gpu::granular_mu_i_rheology::GpuGranularMuIRheology)
//! must reproduce the `CPU` golden `GranularRheology` of
//! `prism_physics_core::collider::granular_rheology`, the dense-granular-flow
//! map from a shear rate and confining pressure to an inertial number, an
//! effective friction coefficient, a shear stress and an effective viscosity.
//!
//! The oracle here is an independent re-implementation of those closed forms,
//! written out directly in flat `f32` math so the test never imports
//! `prism_physics_core`, `prism_render_architecture` or `glam`. It replicates
//! the golden exactly: `I = γ̇ · d / √(P / ρ_s)` with the golden validity gate,
//! the `μ(I) = μ_s + (μ_2 − μ_s)/(I_0/I + 1)` friction law with its `I <= 0` and
//! non-finite special cases, `τ = μ(I) · P`, and `η = τ / γ̇` for a strictly
//! positive shear rate.
//!
//! The fixtures cover the regimes the kernel must honor: a valid interior flow;
//! a negative and a non-positive shear rate / pressure; non-finite inputs; a
//! zero shear rate (valid inertial number `I = 0`, defined shear stress, but no
//! viscosity); a multi-element mixed batch validating the `std430` array
//! stride; plus an empty batch the host short-circuits with no dispatch. A
//! `512`-step `LCG` sweep over valid interior inputs follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The golden is evaluated entirely in `f32`, matching the twin, so `CPU` and
//! `GPU` differ only by `GPU` operator contraction. The continuous comparison
//! is `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on each
//! continuous scalar; the discrete validity flags are compared exactly. The
//! sweep keeps every input well clear of the `I = 0` knee and the validity
//! boundaries so round-off cannot flip a discrete decision.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::granular_rheology`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::granular_mu_i_rheology::{
    GpuGranularMuIRheology, GranularMuIRheologyQuery, GranularMuIRheologyResult,
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
/// `GranularRheology::inertial_number`, returning `None` on the golden's
/// degeneracy gate without importing the golden crate.
fn inertial_number(q: &GranularMuIRheologyQuery) -> Option<f32> {
    if !q.shear_rate.is_finite() || q.shear_rate < 0.0 {
        return None;
    }
    if !q.pressure.is_finite() || q.pressure <= 0.0 {
        return None;
    }
    let micro = q.pressure / q.grain_density;
    let denom = micro.sqrt();
    if denom <= 0.0 {
        return None;
    }
    Some(q.shear_rate * q.grain_diameter / denom)
}

/// Independent host re-implementation of the golden `GranularRheology::friction`
/// `μ(I)` law, including the `I <= 0` and non-finite special cases.
fn friction(q: &GranularMuIRheologyQuery, inertial: f32) -> f32 {
    if !inertial.is_finite() {
        return if inertial.is_sign_positive() && inertial.is_infinite() {
            q.mu_dynamic
        } else {
            q.mu_static
        };
    }
    if inertial <= 0.0 {
        return q.mu_static;
    }
    let extra = (q.mu_dynamic - q.mu_static) / (q.i_ref / inertial + 1.0);
    q.mu_static + extra
}

/// Full independent oracle mirroring the twin's encoding: `None`-returning
/// quantities become value `0` with `valid = 0`, and the always-defined
/// friction is evaluated on the stored inertial number (`0` when invalid).
fn oracle(q: &GranularMuIRheologyQuery) -> GranularMuIRheologyResult {
    let inertial_opt = inertial_number(q);
    let inertial_value = inertial_opt.unwrap_or(0.0);
    let inertial_valid = u32::from(inertial_opt.is_some());

    let fric = friction(q, inertial_value);

    let shear_opt = inertial_opt.map(|i| friction(q, i) * q.pressure);
    let shear_value = shear_opt.unwrap_or(0.0);
    let shear_valid = u32::from(shear_opt.is_some());

    let eff_opt = if !q.shear_rate.is_finite() || q.shear_rate <= 0.0 {
        None
    } else {
        shear_opt.map(|tau| tau / q.shear_rate)
    };
    let eff_value = eff_opt.unwrap_or(0.0);
    let eff_valid = u32::from(eff_opt.is_some());

    GranularMuIRheologyResult {
        inertial_number: inertial_value,
        inertial_valid,
        friction: fric,
        shear_stress: shear_value,
        shear_stress_valid: shear_valid,
        effective_viscosity: eff_value,
        effective_viscosity_valid: eff_valid,
    }
}

/// Asserts a `GPU` result matches the oracle: continuous scalars to tolerance,
/// discrete flags exactly.
fn assert_matches(label: &str, q: &GranularMuIRheologyQuery, got: &GranularMuIRheologyResult) {
    let want = oracle(q);
    assert_eq!(
        got.inertial_valid, want.inertial_valid,
        "{label}: inertial_valid mismatch, query={q:?}"
    );
    assert_eq!(
        got.shear_stress_valid, want.shear_stress_valid,
        "{label}: shear_stress_valid mismatch, query={q:?}"
    );
    assert_eq!(
        got.effective_viscosity_valid, want.effective_viscosity_valid,
        "{label}: effective_viscosity_valid mismatch, query={q:?}"
    );
    assert!(
        close(got.inertial_number, want.inertial_number),
        "{label}: inertial_number gpu={} cpu={} query={q:?}",
        got.inertial_number,
        want.inertial_number
    );
    assert!(
        close(got.friction, want.friction),
        "{label}: friction gpu={} cpu={} query={q:?}",
        got.friction,
        want.friction
    );
    assert!(
        close(got.shear_stress, want.shear_stress),
        "{label}: shear_stress gpu={} cpu={} query={q:?}",
        got.shear_stress,
        want.shear_stress
    );
    assert!(
        close(got.effective_viscosity, want.effective_viscosity),
        "{label}: effective_viscosity gpu={} cpu={} query={q:?}",
        got.effective_viscosity,
        want.effective_viscosity
    );
}

#[test]
fn valid_interior_flow_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularMuIRheology::new(&ctx);
    // Dense flow with a strictly positive inertial number well above zero.
    let q = GranularMuIRheologyQuery::new(0.32, 0.62, 0.30, 1.0e-3, 2500.0, 2.0, 1.0e4);
    let results = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(results.len(), 1);
    assert_matches("valid_interior_flow", &q, &results[0]);
    // Sanity: the flow is genuinely in the continuous (I > 0) branch.
    assert_eq!(results[0].inertial_valid, 1);
    assert_eq!(results[0].effective_viscosity_valid, 1);
}

#[test]
fn negative_shear_rate_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularMuIRheology::new(&ctx);
    let q = GranularMuIRheologyQuery::new(0.32, 0.62, 0.30, 1.0e-3, 2500.0, -1.5, 1.0e4);
    let results = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(results.len(), 1);
    assert_matches("negative_shear_rate", &q, &results[0]);
    assert_eq!(results[0].inertial_valid, 0);
    assert_eq!(results[0].shear_stress_valid, 0);
    assert_eq!(results[0].effective_viscosity_valid, 0);
    // Friction is always defined; with an invalid inertial number it degrades
    // to mu_static.
    assert!(close(results[0].friction, 0.32));
}

#[test]
fn non_positive_pressure_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularMuIRheology::new(&ctx);
    let q = GranularMuIRheologyQuery::new(0.32, 0.62, 0.30, 1.0e-3, 2500.0, 2.0, 0.0);
    let results = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(results.len(), 1);
    assert_matches("non_positive_pressure", &q, &results[0]);
    assert_eq!(results[0].inertial_valid, 0);
}

#[test]
fn non_finite_inputs_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularMuIRheology::new(&ctx);
    let queries = [
        GranularMuIRheologyQuery::new(0.32, 0.62, 0.30, 1.0e-3, 2500.0, f32::NAN, 1.0e4),
        GranularMuIRheologyQuery::new(0.32, 0.62, 0.30, 1.0e-3, 2500.0, f32::INFINITY, 1.0e4),
        GranularMuIRheologyQuery::new(0.32, 0.62, 0.30, 1.0e-3, 2500.0, 2.0, f32::INFINITY),
        GranularMuIRheologyQuery::new(0.32, 0.62, 0.30, 1.0e-3, 2500.0, 2.0, f32::NAN),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len());
    for (q, r) in queries.iter().zip(results.iter()) {
        assert_matches("non_finite", q, r);
        assert_eq!(r.inertial_valid, 0);
        assert_eq!(r.effective_viscosity_valid, 0);
    }
}

#[test]
fn zero_shear_rate_defines_stress_not_viscosity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularMuIRheology::new(&ctx);
    // Shear rate exactly zero: I = 0 (valid), friction = mu_static, shear stress
    // defined, but viscosity (division by the shear rate) is undefined.
    let q = GranularMuIRheologyQuery::new(0.32, 0.62, 0.30, 1.0e-3, 2500.0, 0.0, 1.0e4);
    let results = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(results.len(), 1);
    assert_matches("zero_shear_rate", &q, &results[0]);
    assert_eq!(results[0].inertial_valid, 1);
    assert_eq!(results[0].shear_stress_valid, 1);
    assert_eq!(results[0].effective_viscosity_valid, 0);
    assert!(close(results[0].inertial_number, 0.0));
    assert!(close(results[0].friction, 0.32));
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularMuIRheology::new(&ctx);
    // A deliberately heterogeneous batch so a wrong std430 stride would scramble
    // neighbors: valid, degenerate, zero-rate, non-finite, another valid.
    let queries = [
        GranularMuIRheologyQuery::new(0.28, 0.55, 0.25, 2.0e-3, 2600.0, 1.5, 5.0e3),
        GranularMuIRheologyQuery::new(0.35, 0.70, 0.40, 1.0e-3, 2400.0, -2.0, 8.0e3),
        GranularMuIRheologyQuery::new(0.30, 0.60, 0.30, 1.5e-3, 2500.0, 0.0, 1.2e4),
        GranularMuIRheologyQuery::new(0.33, 0.64, 0.35, 1.0e-3, 2550.0, 3.0, f32::NAN),
        GranularMuIRheologyQuery::new(0.22, 0.48, 0.20, 3.0e-3, 2700.0, 4.0, 2.0e4),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len());
    for (q, r) in queries.iter().zip(results.iter()) {
        assert_matches("mixed_batch", q, r);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularMuIRheology::new(&ctx);
    let results = gpu.evaluate(&ctx, &[]);
    assert!(results.is_empty(), "empty batch returns an empty vector");
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
    let gpu = GpuGranularMuIRheology::new(&ctx);
    let mut rng = Lcg::new(0x51_7C_C1_D7);
    let mut queries = Vec::with_capacity(512);
    // Keep every drawn sample strictly inside the valid region and clear of the
    // I = 0 knee: a strictly positive shear rate and pressure with margin, a
    // positive grain diameter/density and a positive reference inertial number,
    // so both the inertial-number validity and the continuous (I > 0) friction
    // branch are exercised and no discrete flag can flip under round-off.
    while queries.len() < 512 {
        let mu_static = rng.next_range(0.2, 0.4);
        let mu_dynamic = rng.next_range(0.5, 0.8);
        let i_ref = rng.next_range(0.1, 0.6);
        let grain_diameter = rng.next_range(1.0e-4, 1.0e-2);
        let grain_density = rng.next_range(1000.0, 3000.0);
        let shear_rate = rng.next_range(0.1, 5.0);
        let pressure = rng.next_range(100.0, 1.0e5);
        let q = GranularMuIRheologyQuery::new(
            mu_static,
            mu_dynamic,
            i_ref,
            grain_diameter,
            grain_density,
            shear_rate,
            pressure,
        );
        queries.push(q);
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        assert_matches("sweep", q, r);
        // Every swept sample lands in the valid, continuous regime.
        assert_eq!(r.inertial_valid, 1);
        assert_eq!(r.shear_stress_valid, 1);
        assert_eq!(r.effective_viscosity_valid, 1);
    }
}
