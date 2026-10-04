//! Real-device parity for the Haff cooling-law twin:
//! [`GpuHaffCoolingLaw`](prism_volumetric_gpu::haff_cooling_law::GpuHaffCoolingLaw)
//! must reproduce the `CPU` golden `HaffCooling` constructor and getters of
//! `prism_physics_core::collider::haff_cooling`. An unforced inelastic granular
//! gas cools algebraically: with `zeta0 = (1 - e^2) * omega0 / 3` and
//! `tau = 2 / zeta0`, the temperature follows `T(t) = T0 / (1 + t/tau)^2`, the
//! collision frequency `omega(t) = omega0 / (1 + t/tau)`, the time to a
//! fraction `f` is `tau * (f^{-1/2} - 1)` and the half-life is
//! `tau * (sqrt(2) - 1)`.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly so the test never imports `prism_render_architecture`
//! or `prism_physics_core`. The golden is pure `f32`, so the oracle is pure
//! `f32` too and mirrors the same operator order and the same validity gates.
//!
//! The fixtures cover a mid-ratio all-valid point, cooling-rate/time sanity, a
//! temperature decay, a collision-frequency decay, the time-to-fraction getter
//! at `f = 0.5` and the identity `f = 1 -> 0`, the half-life, the degenerate
//! constructor rejections (`e = 1`, `omega0 = 0`, `T0 < 0`, `e < 0`), the
//! non-finite rejections (`T0 = NaN`, `omega0 = inf`), the per-getter sub-gates
//! (`t = inf` leaves the master valid but zeroes the two time getters; `f` out
//! of `(0, 1]` zeroes only the fraction getter), a mixed batch validating the
//! `std430` stride, and an empty batch the host short-circuits with no
//! dispatch. A sweep over random finite inputs follows; it keeps `e <= 0.95`,
//! `omega0 >= 0.1`, `T0 >= 0`, `t` and `f >= 0.01` away from the knees so every
//! query is master-valid.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Both sides evaluate the same pure-`f32` closed form, so `CPU` and `GPU` need
//! not be bit-exact under reassociation. Every continuous scalar is compared
//! with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`) and the four validity
//! words are compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::haff_cooling`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::haff_cooling_law::{
    GpuHaffCoolingLaw, HaffCoolingLawQuery, HaffCoolingLawResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// The resolved oracle outputs: six scalars plus the master flag and the three
/// per-getter sub-flags, in the same encoding as the device result.
struct Oracle {
    cooling_rate: f32,
    cooling_time: f32,
    temperature_at: f32,
    collision_frequency_at: f32,
    time_to_fraction: f32,
    half_life: f32,
    valid: bool,
    temperature_valid: bool,
    collision_frequency_valid: bool,
    time_to_fraction_valid: bool,
}

/// Independent host oracle: reproduces the golden `HaffCooling` constructor
/// gate and the five getters in the golden operator order, pure `f32`.
fn oracle(q: &HaffCoolingLawQuery) -> Oracle {
    let temp0 = q.initial_temperature;
    let freq0 = q.initial_collision_frequency;
    let rest = q.restitution;

    let temp0_ok = temp0.is_finite() && temp0 >= 0.0;
    let freq0_ok = freq0.is_finite() && freq0 >= 0.0;
    let rest_ok = rest.is_finite() && (0.0..1.0).contains(&rest);

    let inelasticity = 1.0 - rest * rest;
    let zeta0 = inelasticity * freq0 / 3.0;
    let zeta0_ok = zeta0.is_finite() && zeta0 > 0.0;

    let master_pre = temp0_ok && freq0_ok && rest_ok && zeta0_ok;
    let tau = if master_pre { 2.0 / zeta0 } else { 1.0 };
    let tau_ok = tau.is_finite() && tau > 0.0;
    let master_ok = master_pre && tau_ok;

    if !master_ok {
        return Oracle {
            cooling_rate: 0.0,
            cooling_time: 0.0,
            temperature_at: 0.0,
            collision_frequency_at: 0.0,
            time_to_fraction: 0.0,
            half_life: 0.0,
            valid: false,
            temperature_valid: false,
            collision_frequency_valid: false,
            time_to_fraction_valid: false,
        };
    }

    let cooling_rate = zeta0;
    let cooling_time = tau;
    let half_life = tau * (std::f32::consts::SQRT_2 - 1.0);

    let elapsed = q.time;
    let time_ok = elapsed.is_finite() && elapsed >= 0.0;
    let (temperature_at, collision_frequency_at) = if time_ok {
        let ratio = 1.0 + elapsed / tau;
        (temp0 / (ratio * ratio), freq0 / ratio)
    } else {
        (0.0, 0.0)
    };

    let frac = q.fraction;
    let frac_ok = frac.is_finite() && frac > 0.0 && frac <= 1.0;
    let time_to_fraction = if frac_ok {
        tau * (1.0 / frac.sqrt() - 1.0)
    } else {
        0.0
    };

    Oracle {
        cooling_rate,
        cooling_time,
        temperature_at,
        collision_frequency_at,
        time_to_fraction,
        half_life,
        valid: true,
        temperature_valid: time_ok,
        collision_frequency_valid: time_ok,
        time_to_fraction_valid: frac_ok,
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

/// Asserts a single `GPU` result matches the independent oracle: the validity
/// words match exactly, each scalar matches to tolerance when its flag is set,
/// and every scalar is zero when the master flag is cleared.
fn assert_parity(gpu: &HaffCoolingLawResult, q: &HaffCoolingLawQuery, label: &str) {
    let o = oracle(q);
    assert_eq!(gpu.valid, o.valid, "{label}: master valid flag");
    assert_eq!(
        gpu.temperature_valid, o.temperature_valid,
        "{label}: temperature valid flag"
    );
    assert_eq!(
        gpu.collision_frequency_valid, o.collision_frequency_valid,
        "{label}: collision-frequency valid flag"
    );
    assert_eq!(
        gpu.time_to_fraction_valid, o.time_to_fraction_valid,
        "{label}: time-to-fraction valid flag"
    );

    if !o.valid {
        assert_eq!(gpu.cooling_rate, 0.0, "{label}: cooling_rate zeroed");
        assert_eq!(gpu.cooling_time, 0.0, "{label}: cooling_time zeroed");
        assert_eq!(gpu.temperature_at, 0.0, "{label}: temperature_at zeroed");
        assert_eq!(
            gpu.collision_frequency_at, 0.0,
            "{label}: collision_frequency_at zeroed"
        );
        assert_eq!(
            gpu.time_to_fraction, 0.0,
            "{label}: time_to_fraction zeroed"
        );
        assert_eq!(gpu.half_life, 0.0, "{label}: half_life zeroed");
        return;
    }

    assert!(
        close(gpu.cooling_rate, o.cooling_rate),
        "{label}: cooling_rate gpu={} oracle={}",
        gpu.cooling_rate,
        o.cooling_rate
    );
    assert!(
        close(gpu.cooling_time, o.cooling_time),
        "{label}: cooling_time gpu={} oracle={}",
        gpu.cooling_time,
        o.cooling_time
    );
    assert!(
        close(gpu.half_life, o.half_life),
        "{label}: half_life gpu={} oracle={}",
        gpu.half_life,
        o.half_life
    );
    if o.temperature_valid {
        assert!(
            close(gpu.temperature_at, o.temperature_at),
            "{label}: temperature_at gpu={} oracle={}",
            gpu.temperature_at,
            o.temperature_at
        );
        assert!(
            close(gpu.collision_frequency_at, o.collision_frequency_at),
            "{label}: collision_frequency_at gpu={} oracle={}",
            gpu.collision_frequency_at,
            o.collision_frequency_at
        );
    } else {
        assert_eq!(gpu.temperature_at, 0.0, "{label}: temperature_at zeroed");
        assert_eq!(
            gpu.collision_frequency_at, 0.0,
            "{label}: collision_frequency_at zeroed"
        );
    }
    if o.time_to_fraction_valid {
        assert!(
            close(gpu.time_to_fraction, o.time_to_fraction),
            "{label}: time_to_fraction gpu={} oracle={}",
            gpu.time_to_fraction,
            o.time_to_fraction
        );
    } else {
        assert_eq!(
            gpu.time_to_fraction, 0.0,
            "{label}: time_to_fraction zeroed"
        );
    }
}

/// A representative well-conditioned query: `e = 0.9`, `omega0 = 10`,
/// `T0 = 100`, with the elapsed time and fraction supplied by the caller.
fn well_conditioned(time: f32, fraction: f32) -> HaffCoolingLawQuery {
    HaffCoolingLawQuery::new(100.0, 10.0, 0.9, time, fraction)
}

#[test]
fn midpoint_all_valid_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaffCoolingLaw::new(&ctx);
    // zeta0 = (1 - 0.81) * 10 / 3 = 0.6333..., tau = 3.1578...; sample near tau.
    let q = well_conditioned(3.0, 0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid, "constructor should accept");
    assert!(out[0].temperature_valid);
    assert!(out[0].collision_frequency_valid);
    assert!(out[0].time_to_fraction_valid);
    assert_parity(&out[0], &q, "midpoint");
}

#[test]
fn cooling_rate_and_time_are_sane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaffCoolingLaw::new(&ctx);
    let q = well_conditioned(0.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let o = oracle(&q);
    // tau = 2 / zeta0, so cooling_rate * cooling_time = 2.
    assert!(
        close(o.cooling_rate * o.cooling_time, 2.0),
        "zeta0 * tau should equal 2: {} {}",
        o.cooling_rate,
        o.cooling_time
    );
    assert_parity(&out[0], &q, "cooling_rate_time");
}

#[test]
fn temperature_decays_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaffCoolingLaw::new(&ctx);
    // t = 0 gives T(0) = T0; a later time gives strictly less temperature.
    let early = well_conditioned(0.0, 0.5);
    let late = well_conditioned(10.0, 0.5);
    let out = gpu.evaluate(&ctx, &[early, late]);
    assert_eq!(out.len(), 2);
    assert!(close(out[0].temperature_at, 100.0), "T(0) should equal T0");
    assert!(
        out[1].temperature_at < out[0].temperature_at,
        "temperature must decay"
    );
    assert_parity(&out[0], &early, "temp_early");
    assert_parity(&out[1], &late, "temp_late");
}

#[test]
fn collision_frequency_decays_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaffCoolingLaw::new(&ctx);
    let early = well_conditioned(0.0, 0.5);
    let late = well_conditioned(8.0, 0.5);
    let out = gpu.evaluate(&ctx, &[early, late]);
    assert_eq!(out.len(), 2);
    assert!(
        close(out[0].collision_frequency_at, 10.0),
        "omega(0) should equal omega0"
    );
    assert!(
        out[1].collision_frequency_at < out[0].collision_frequency_at,
        "collision frequency must decay"
    );
    assert_parity(&out[0], &early, "freq_early");
    assert_parity(&out[1], &late, "freq_late");
}

#[test]
fn time_to_fraction_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaffCoolingLaw::new(&ctx);
    // f = 1 maps to t = 0 (already at T0); f = 0.5 is the half-life.
    let unit = well_conditioned(1.0, 1.0);
    let half = well_conditioned(1.0, 0.5);
    let out = gpu.evaluate(&ctx, &[unit, half]);
    assert_eq!(out.len(), 2);
    assert!(
        close(out[0].time_to_fraction, 0.0),
        "f = 1 should give t = 0: {}",
        out[0].time_to_fraction
    );
    // time_to_fraction(0.5) must equal the reported half-life.
    assert!(
        close(out[1].time_to_fraction, out[1].half_life),
        "time_to_fraction(0.5) should equal half_life: {} {}",
        out[1].time_to_fraction,
        out[1].half_life
    );
    assert_parity(&out[0], &unit, "ttf_unit");
    assert_parity(&out[1], &half, "ttf_half");
}

#[test]
fn half_life_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaffCoolingLaw::new(&ctx);
    let q = well_conditioned(0.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let o = oracle(&q);
    assert!(o.half_life > 0.0, "half-life should be positive");
    assert_parity(&out[0], &q, "half_life");
}

#[test]
fn degenerate_constructor_rejections_zero_out() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaffCoolingLaw::new(&ctx);
    let queries = vec![
        // Elastic grain: e = 1 drives zeta0 to 0.
        HaffCoolingLawQuery::new(100.0, 10.0, 1.0, 1.0, 0.5),
        // Collisionless gas: omega0 = 0 drives zeta0 to 0.
        HaffCoolingLawQuery::new(100.0, 0.0, 0.5, 1.0, 0.5),
        // Negative initial temperature.
        HaffCoolingLawQuery::new(-1.0, 10.0, 0.5, 1.0, 0.5),
        // Restitution below range.
        HaffCoolingLawQuery::new(100.0, 10.0, -0.1, 1.0, 0.5),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(!res.valid, "degenerate[{i}] should be invalid");
        assert_eq!(res.cooling_rate, 0.0);
        assert_eq!(res.cooling_time, 0.0);
        assert_eq!(res.half_life, 0.0);
        assert_parity(res, q, &format!("degenerate[{i}]"));
    }
}

#[test]
fn non_finite_constructor_inputs_zero_out() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaffCoolingLaw::new(&ctx);
    let queries = vec![
        HaffCoolingLawQuery::new(f32::NAN, 10.0, 0.5, 1.0, 0.5),
        HaffCoolingLawQuery::new(100.0, f32::INFINITY, 0.5, 1.0, 0.5),
        HaffCoolingLawQuery::new(100.0, 10.0, f32::NAN, 1.0, 0.5),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(!res.valid, "non_finite[{i}] should be invalid");
        assert_parity(res, q, &format!("non_finite[{i}]"));
    }
}

#[test]
fn non_finite_time_zeroes_only_time_getters() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaffCoolingLaw::new(&ctx);
    // Infinite elapsed time: master stays valid, the two time getters sub-gate.
    let q = well_conditioned(f32::INFINITY, 0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid, "master should stay valid");
    assert!(!out[0].temperature_valid, "temperature sub-gate closes");
    assert!(
        !out[0].collision_frequency_valid,
        "frequency sub-gate closes"
    );
    assert!(out[0].time_to_fraction_valid, "fraction gate still open");
    assert_eq!(out[0].temperature_at, 0.0);
    assert_eq!(out[0].collision_frequency_at, 0.0);
    assert!(out[0].cooling_rate > 0.0, "cooling_rate still carried");
    assert_parity(&out[0], &q, "inf_time");
}

#[test]
fn out_of_range_fraction_zeroes_only_fraction_getter() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaffCoolingLaw::new(&ctx);
    let queries = vec![
        // f = 0 is out of (0, 1].
        well_conditioned(1.0, 0.0),
        // f > 1 is out of (0, 1].
        well_conditioned(1.0, 1.5),
        // f = NaN rejects.
        well_conditioned(1.0, f32::NAN),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(res.valid, "fraction_sub[{i}] master valid");
        assert!(
            res.temperature_valid,
            "fraction_sub[{i}] temperature still valid"
        );
        assert!(
            !res.time_to_fraction_valid,
            "fraction_sub[{i}] fraction sub-gate closes"
        );
        assert_eq!(res.time_to_fraction, 0.0);
        assert_parity(res, q, &format!("fraction_sub[{i}]"));
    }
}

#[test]
fn batch_mixes_valid_and_degenerate_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaffCoolingLaw::new(&ctx);
    let queries = vec![
        well_conditioned(3.0, 0.5),
        HaffCoolingLawQuery::new(100.0, 10.0, 1.0, 1.0, 0.5),
        well_conditioned(f32::INFINITY, 0.5),
        well_conditioned(1.0, 2.0),
        HaffCoolingLawQuery::new(-1.0, 10.0, 0.5, 1.0, 0.5),
        well_conditioned(0.0, 1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(out[0].valid && out[0].temperature_valid && out[0].time_to_fraction_valid);
    assert!(!out[1].valid);
    assert!(out[2].valid && !out[2].temperature_valid);
    assert!(out[3].valid && !out[3].time_to_fraction_valid);
    assert!(!out[4].valid);
    assert!(out[5].valid);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaffCoolingLaw::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaffCoolingLaw::new(&ctx);
    let mut lcg = Lcg::new(0x51A3_C7E9);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Keep e <= 0.95 and omega0 >= 0.1 so zeta0 stays comfortably positive;
        // t and f >= 0.01 away from the knees, so every query is master-valid.
        let temp0 = lcg.next_range(0.0, 1000.0);
        let freq0 = lcg.next_range(0.1, 50.0);
        let rest = lcg.next_range(0.0, 0.95);
        let time = lcg.next_range(0.0, 100.0);
        let fraction = lcg.next_range(0.01, 1.0);
        queries.push(HaffCoolingLawQuery::new(temp0, freq0, rest, time, fraction));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(res.valid, "sweep[{i}] should be master-valid");
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
