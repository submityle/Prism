//! Real-device parity for the Enskog granular kinetic-theory twin:
//! [`GpuKineticTheoryEnskog`](prism_volumetric_gpu::kinetic_theory_enskog::GpuKineticTheoryEnskog)
//! must reproduce the `CPU` golden `prism_physics_core::collider::kinetic_theory`'s
//! `GranularKineticState::new` plus its depth-independent getters.
//!
//! From four coarse granular-gas fields — the number density `n`, the grain
//! diameter `d`, the solid fraction `φ`, and the granular temperature `T` — the
//! reference derives the Carnahan–Starling pair correlation
//! `g0 = (2 − φ) / (2 (1 − φ)³)`, the Enskog mean free path
//! `ℓ = 1 / (√2 π n d² g0)`, the collision frequency `ω = 4 n d² g0 √(π T)`,
//! the mean collision time `1/ω`, the thermal velocity scale `√max(T, 0)`, and
//! the `RMS` fluctuation speed `√max(3T, 0)`.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly so the test never imports `prism_render_architecture`
//! or `prism_physics_core`. The reference evaluates the pair-correlation cube
//! in `f64` then narrows to `f32`; the oracle mirrors that (host `f64` is
//! allowed, unlike the device), while the device evaluates the cube in `f32`,
//! so the parity tolerance absorbs the narrowing.
//!
//! The fixtures cover the normal regime, the degenerate rejections (non-finite
//! or out-of-range input), the frozen-gas edge where construction still
//! succeeds but the collision time is invalid, a mixed batch that validates the
//! `std430` stride, and an empty batch the host short-circuits. A sweep over
//! random parameters well inside the valid band follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The device evaluates the whole chain in `f32`; the oracle mirrors the
//! reference's `f64` cube then narrows, so `CPU` and `GPU` need not be
//! bit-exact (a `GPU` may contract a multiply-add, and the `f64`/`f32` cube
//! differs). Each valid scalar is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` and `mean_collision_time_valid`
//! flags are compared exactly. The sweep keeps random parameters away from the
//! `φ → 1` and `T → 0` knees so the validity decision agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::kinetic_theory`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::kinetic_theory_enskog::{
    GpuKineticTheoryEnskog, KineticTheoryEnskogQuery, KineticTheoryEnskogResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces the Enskog kinetic-theory descriptors,
/// evaluating the pair-correlation cube in `f64` exactly as the reference does
/// then narrowing to `f32`. Returns the six scalars plus the two validity
/// flags `(pair, mfp, cf, mct, tvs, rms, mean_collision_time_valid, valid)`.
fn oracle(q: &KineticTheoryEnskogQuery) -> (f32, f32, f32, f32, f32, f32, bool, bool) {
    let n = q.number_density;
    let d = q.grain_diameter;
    let phi = q.solid_fraction;
    let t = q.granular_temperature;
    let invalid = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, false);
    if !n.is_finite() || n <= 0.0 {
        return invalid;
    }
    if !d.is_finite() || d <= 0.0 {
        return invalid;
    }
    if !phi.is_finite() || !(0.0..1.0).contains(&phi) {
        return invalid;
    }
    if !t.is_finite() || t < 0.0 {
        return invalid;
    }
    let phi64 = f64::from(phi);
    let om = 1.0 - phi64;
    let g0_64 = (2.0 - phi64) / (2.0 * om * om * om);
    if !g0_64.is_finite() || g0_64 <= 0.0 {
        return invalid;
    }
    let pair = g0_64 as f32;
    let d2 = d * d;
    let s2 = std::f32::consts::SQRT_2;
    let pi = std::f32::consts::PI;
    let denom = s2 * pi * n * d2 * pair;
    if !denom.is_finite() || denom <= 0.0 {
        return invalid;
    }
    let mfp = 1.0 / denom;
    let thermal = (pi * t).max(0.0).sqrt();
    let cf = 4.0 * n * d2 * pair * thermal;
    if !mfp.is_finite() || !cf.is_finite() {
        return invalid;
    }
    let mct_valid = cf > 0.0;
    let mct = if mct_valid { 1.0 / cf } else { 0.0 };
    let tvs = t.max(0.0).sqrt();
    let rms = (3.0 * t).max(0.0).sqrt();
    (pair, mfp, cf, mct, tvs, rms, mct_valid, true)
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
/// flags exactly, the six descriptors to tolerance when valid (and the mean
/// collision time only when its own flag is set), else all zero.
fn assert_parity(gpu: &KineticTheoryEnskogResult, q: &KineticTheoryEnskogQuery, label: &str) {
    let (pair, mfp, cf, mct, tvs, rms, mct_valid, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    assert_eq!(
        gpu.mean_collision_time_valid, mct_valid,
        "{label}: mean_collision_time_valid flag mismatch"
    );
    if valid {
        assert!(
            close(gpu.pair_correlation, pair),
            "{label}: pair_correlation mismatch gpu={} oracle={}",
            gpu.pair_correlation,
            pair
        );
        assert!(
            close(gpu.mean_free_path, mfp),
            "{label}: mean_free_path mismatch gpu={} oracle={}",
            gpu.mean_free_path,
            mfp
        );
        assert!(
            close(gpu.collision_frequency, cf),
            "{label}: collision_frequency mismatch gpu={} oracle={}",
            gpu.collision_frequency,
            cf
        );
        assert!(
            close(gpu.thermal_velocity_scale, tvs),
            "{label}: thermal_velocity_scale mismatch gpu={} oracle={}",
            gpu.thermal_velocity_scale,
            tvs
        );
        assert!(
            close(gpu.rms_fluctuation_speed, rms),
            "{label}: rms_fluctuation_speed mismatch gpu={} oracle={}",
            gpu.rms_fluctuation_speed,
            rms
        );
        if mct_valid {
            assert!(
                close(gpu.mean_collision_time, mct),
                "{label}: mean_collision_time mismatch gpu={} oracle={}",
                gpu.mean_collision_time,
                mct
            );
        } else {
            assert_eq!(
                gpu.mean_collision_time, 0.0,
                "{label}: mean_collision_time should be zero when its flag is unset"
            );
        }
    } else {
        assert_eq!(gpu.pair_correlation, 0.0, "{label}: pair not zeroed");
        assert_eq!(gpu.mean_free_path, 0.0, "{label}: mfp not zeroed");
        assert_eq!(gpu.collision_frequency, 0.0, "{label}: cf not zeroed");
        assert_eq!(gpu.mean_collision_time, 0.0, "{label}: mct not zeroed");
        assert_eq!(gpu.thermal_velocity_scale, 0.0, "{label}: tvs not zeroed");
        assert_eq!(gpu.rms_fluctuation_speed, 0.0, "{label}: rms not zeroed");
    }
}

#[test]
fn normal_regime_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKineticTheoryEnskog::new(&ctx);
    let queries = vec![
        KineticTheoryEnskogQuery::new(1.0, 1.0, 0.3, 1.0),
        KineticTheoryEnskogQuery::new(2.0, 0.5, 0.1, 4.0),
        // phi = 0.5 gives the exact g0 = (2 - 0.5) / (2 * 0.125) = 6.
        KineticTheoryEnskogQuery::new(3.0, 0.25, 0.5, 2.0),
        // A denser packing well short of the g0 blow-up.
        KineticTheoryEnskogQuery::new(5.0, 0.8, 0.6, 9.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(res.valid, "normal[{i}] should be valid");
        assert!(
            res.mean_collision_time_valid,
            "normal[{i}] should have a valid mean collision time"
        );
        assert_parity(res, q, &format!("normal[{i}]"));
    }
}

#[test]
fn degenerate_inputs_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKineticTheoryEnskog::new(&ctx);
    let queries = vec![
        // Non-positive number density.
        KineticTheoryEnskogQuery::new(0.0, 1.0, 0.3, 1.0),
        // Negative grain diameter.
        KineticTheoryEnskogQuery::new(1.0, -1.0, 0.3, 1.0),
        // phi = 1 is outside [0, 1).
        KineticTheoryEnskogQuery::new(1.0, 1.0, 1.0, 1.0),
        // Negative solid fraction.
        KineticTheoryEnskogQuery::new(1.0, 1.0, -0.1, 1.0),
        // Negative temperature.
        KineticTheoryEnskogQuery::new(1.0, 1.0, 0.3, -1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(!res.valid, "degenerate[{i}] should be invalid");
        assert!(
            !res.mean_collision_time_valid,
            "degenerate[{i}] should have no valid mean collision time"
        );
        assert_parity(res, q, &format!("degenerate[{i}]"));
    }
}

#[test]
fn non_finite_inputs_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKineticTheoryEnskog::new(&ctx);
    let queries = vec![
        KineticTheoryEnskogQuery::new(f32::NAN, 1.0, 0.3, 1.0),
        KineticTheoryEnskogQuery::new(1.0, f32::INFINITY, 0.3, 1.0),
        KineticTheoryEnskogQuery::new(1.0, 1.0, f32::NAN, 1.0),
        KineticTheoryEnskogQuery::new(1.0, 1.0, 0.3, f32::INFINITY),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(!res.valid, "non_finite[{i}] should be invalid");
        assert_parity(res, q, &format!("non_finite[{i}]"));
    }
}

#[test]
fn frozen_gas_is_valid_but_collision_time_is_not() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKineticTheoryEnskog::new(&ctx);
    // T = 0 freezes the gas: the construction still succeeds (g0, mean free
    // path are temperature-independent), but omega = 0 so the mean collision
    // time is invalid, and the velocity scales collapse to zero.
    let q = KineticTheoryEnskogQuery::new(1.0, 1.0, 0.3, 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let res = &out[0];
    assert!(res.valid, "frozen gas should still construct");
    assert!(
        !res.mean_collision_time_valid,
        "frozen gas should have no valid mean collision time"
    );
    assert_eq!(res.collision_frequency, 0.0, "omega must be zero at T = 0");
    assert_eq!(res.mean_collision_time, 0.0, "mean collision time zeroed");
    assert_eq!(res.thermal_velocity_scale, 0.0, "thermal velocity zeroed");
    assert_eq!(res.rms_fluctuation_speed, 0.0, "rms speed zeroed");
    assert_parity(res, &q, "frozen");
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKineticTheoryEnskog::new(&ctx);
    let queries = vec![
        KineticTheoryEnskogQuery::new(1.0, 1.0, 0.3, 1.0),
        KineticTheoryEnskogQuery::new(0.0, 1.0, 0.3, 1.0),
        KineticTheoryEnskogQuery::new(2.0, 0.5, 0.4, 5.0),
        KineticTheoryEnskogQuery::new(1.0, 1.0, 1.0, 1.0),
        KineticTheoryEnskogQuery::new(1.5, 0.75, 0.2, 0.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(out[0].valid);
    assert!(!out[1].valid);
    assert!(out[2].valid);
    assert!(!out[3].valid);
    assert!(out[4].valid);
    assert!(
        !out[4].mean_collision_time_valid,
        "T = 0 element has no mct"
    );
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKineticTheoryEnskog::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuKineticTheoryEnskog::new(&ctx);
    let mut lcg = Lcg::new(0x0E25_C0CA);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Draw every parameter from a strictly positive band with margin from
        // the knees: phi stays below the g0 blow-up and T above the omega -> 0
        // freeze, so the validity decision stays stable under round-off.
        let n = lcg.next_range(0.1, 10.0);
        let d = lcg.next_range(0.1, 5.0);
        let phi = lcg.next_range(0.01, 0.6);
        let t = lcg.next_range(0.01, 10.0);
        queries.push(KineticTheoryEnskogQuery::new(n, d, phi, t));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(res.valid, "sweep[{i}] should be valid");
        assert!(
            res.mean_collision_time_valid,
            "sweep[{i}] should have a valid mean collision time"
        );
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
