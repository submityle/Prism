//! Real-device parity for the granular-suspension flow-regime twin:
//! [`GpuGranularFlowRegime`](prism_volumetric_gpu::granular_flow_regime::GpuGranularFlowRegime)
//! must reproduce the `CPU` golden `GranularFlowRegime::from_state` and its
//! `bagnold_number` / `stokes_number` / `regime` getters of
//! `prism_physics_core::collider::granular_flow_regime`. When grains shear
//! through a viscous interstitial fluid the Bagnold number
//! `Ba = rho_s * d² * γ̇ / mu_f` measures the ratio of grain-inertial to
//! fluid-viscous stress; the Stokes number is `St = Ba / 18` and the classic
//! thresholds split the flow into `MacroViscous` (`Ba < 40`), `Transitional`
//! (`40 <= Ba <= 450`) and `GrainInertia` (`Ba > 450`).
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly so the test never imports `prism_render_architecture`
//! or `prism_physics_core`. The golden is pure `f32`, so the oracle is pure
//! `f32` too and mirrors the same operator order and the same validity gates.
//!
//! The fixtures cover the three regimes with hand-computed Bagnold numbers, a
//! zero-shear state (`Ba = 0`, macro-viscous), the constructor rejections
//! (`rho_s <= 0`, `d <= 0`, `shear_rate < 0`, `mu_f <= 0`, plus non-finite
//! inputs), a mixed batch validating the `std430` stride, and an empty batch
//! the host short-circuits with no dispatch. A sweep over random finite inputs
//! follows; it keeps every parameter master-valid and rejection-samples so the
//! Bagnold number stays off the `Ba = 40` / `Ba = 450` knees, covering all
//! three regimes.
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
//! with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`) and the
//! `regime`/`valid` words are compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::granular_flow_regime`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::granular_flow_regime::{
    GpuGranularFlowRegime, GranularFlowRegimeQuery, GranularFlowRegimeResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// The resolved oracle outputs, in the same encoding as the device result.
struct Oracle {
    bagnold: f32,
    stokes: f32,
    regime: u32,
    valid: bool,
}

/// Independent host oracle: reproduces the golden `from_state` gate and the
/// `bagnold_number` / `stokes_number` / `regime` getters in the golden operator
/// order, pure `f32`.
fn oracle(q: &GranularFlowRegimeQuery) -> Oracle {
    let rho_s = q.grain_density;
    let diam = q.grain_diameter;
    let shear = q.shear_rate;
    let mu = q.fluid_viscosity;

    let master_ok = rho_s.is_finite()
        && diam.is_finite()
        && shear.is_finite()
        && mu.is_finite()
        && rho_s > 0.0
        && diam > 0.0
        && shear >= 0.0
        && mu > 0.0;

    if !master_ok {
        return Oracle {
            bagnold: 0.0,
            stokes: 0.0,
            regime: 0,
            valid: false,
        };
    }

    let bagnold = rho_s * diam * diam * shear / mu;
    let stokes = bagnold / 18.0;
    let regime = if bagnold < 40.0 {
        0u32
    } else if bagnold <= 450.0 {
        1u32
    } else {
        2u32
    };

    Oracle {
        bagnold,
        stokes,
        regime,
        valid: true,
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

/// Asserts a single `GPU` result matches the independent oracle: the `regime`
/// and `valid` words match exactly, each scalar matches to tolerance when
/// valid, and every scalar is zero when the master flag is cleared.
fn assert_parity(gpu: &GranularFlowRegimeResult, q: &GranularFlowRegimeQuery, label: &str) {
    let o = oracle(q);
    let gpu_valid = gpu.valid == 1;
    assert_eq!(gpu_valid, o.valid, "{label}: master valid flag");

    if !o.valid {
        assert_eq!(gpu.bagnold, 0.0, "{label}: bagnold zeroed");
        assert_eq!(gpu.stokes, 0.0, "{label}: stokes zeroed");
        assert_eq!(gpu.regime, 0, "{label}: regime zeroed");
        return;
    }

    assert_eq!(gpu.regime, o.regime, "{label}: regime label");
    assert!(
        close(gpu.bagnold, o.bagnold),
        "{label}: bagnold gpu={} oracle={}",
        gpu.bagnold,
        o.bagnold
    );
    assert!(
        close(gpu.stokes, o.stokes),
        "{label}: stokes gpu={} oracle={}",
        gpu.stokes,
        o.stokes
    );
}

#[test]
fn transitional_hand_value() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFlowRegime::new(&ctx);
    // rho_s=2000, d=0.001, γ̇=100, mu_f=0.001 → Ba = 200, St ≈ 11.111.
    let q = GranularFlowRegimeQuery::new(2000.0, 0.001, 100.0, 0.001);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1, "well-conditioned state should be valid");
    assert_eq!(out[0].regime, 1, "Ba = 200 is transitional");
    assert_parity(&out[0], &q, "transitional_hand_value");
}

#[test]
fn macro_viscous_regime() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFlowRegime::new(&ctx);
    // rho_s=2000, d=0.0001, γ̇=1, mu_f=0.001 → Ba = 0.02 < 40.
    let q = GranularFlowRegimeQuery::new(2000.0, 0.0001, 1.0, 0.001);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].regime, 0, "Ba = 0.02 is macro-viscous");
    assert_parity(&out[0], &q, "macro_viscous_regime");
}

#[test]
fn grain_inertia_regime() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFlowRegime::new(&ctx);
    // rho_s=2000, d=0.01, γ̇=100, mu_f=0.001 → Ba = 20000 > 450.
    let q = GranularFlowRegimeQuery::new(2000.0, 0.01, 100.0, 0.001);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].regime, 2, "Ba = 20000 is grain-inertia");
    assert_parity(&out[0], &q, "grain_inertia_regime");
}

#[test]
fn zero_shear_is_macro_viscous() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFlowRegime::new(&ctx);
    // γ̇ = 0 is in-range (shear_rate >= 0), so Ba = 0 and the regime is macro.
    let q = GranularFlowRegimeQuery::new(2000.0, 0.001, 0.0, 0.001);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1, "zero shear is a valid state");
    assert_eq!(out[0].regime, 0, "Ba = 0 is macro-viscous");
    assert!(close(out[0].bagnold, 0.0), "zero shear gives Ba = 0");
    assert_parity(&out[0], &q, "zero_shear");
}

#[test]
fn invalid_states_are_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFlowRegime::new(&ctx);
    let base = GranularFlowRegimeQuery::new(2000.0, 0.001, 100.0, 0.001);
    let queries = vec![
        GranularFlowRegimeQuery {
            grain_density: 0.0,
            ..base
        },
        GranularFlowRegimeQuery {
            grain_density: -2000.0,
            ..base
        },
        GranularFlowRegimeQuery {
            grain_diameter: 0.0,
            ..base
        },
        GranularFlowRegimeQuery {
            grain_diameter: -0.001,
            ..base
        },
        GranularFlowRegimeQuery {
            shear_rate: -1.0,
            ..base
        },
        GranularFlowRegimeQuery {
            fluid_viscosity: 0.0,
            ..base
        },
        GranularFlowRegimeQuery {
            fluid_viscosity: -0.001,
            ..base
        },
        GranularFlowRegimeQuery {
            grain_density: f32::NAN,
            ..base
        },
        GranularFlowRegimeQuery {
            grain_diameter: f32::INFINITY,
            ..base
        },
        GranularFlowRegimeQuery {
            shear_rate: f32::NEG_INFINITY,
            ..base
        },
        GranularFlowRegimeQuery {
            fluid_viscosity: f32::NAN,
            ..base
        },
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0, "invalid[{i}] should be rejected");
        assert_parity(res, q, &format!("invalid[{i}]"));
    }
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFlowRegime::new(&ctx);
    let queries = vec![
        // Ba = 0.02 → macro-viscous.
        GranularFlowRegimeQuery::new(2000.0, 0.0001, 1.0, 0.001),
        // Ba = 200 → transitional.
        GranularFlowRegimeQuery::new(2000.0, 0.001, 100.0, 0.001),
        // Ba = 20000 → grain-inertia.
        GranularFlowRegimeQuery::new(2000.0, 0.01, 100.0, 0.001),
        // Invalid (mu_f <= 0).
        GranularFlowRegimeQuery::new(2000.0, 0.001, 100.0, 0.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("mixed[{i}]"));
    }
    assert_eq!(out[0].regime, 0, "slot 0 macro-viscous");
    assert_eq!(out[1].regime, 1, "slot 1 transitional");
    assert_eq!(out[2].regime, 2, "slot 2 grain-inertia");
    assert_eq!(out[3].valid, 0, "slot 3 invalid");
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFlowRegime::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularFlowRegime::new(&ctx);
    let mut lcg = Lcg::new(0x2B7E_1591);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Keep every parameter master-valid; reject-sample so the Bagnold
        // number stays off the Ba = 40 / Ba = 450 knees (relative margin), so
        // neither the validity nor the regime decision can be flipped by
        // round-off. The ranges span all three regimes.
        let rho_s = lcg.next_range(500.0, 3000.0);
        let diam = lcg.next_range(0.0002, 0.02);
        let shear = lcg.next_range(0.0, 200.0);
        let mu = lcg.next_range(0.0005, 0.01);
        let ba = rho_s * diam * diam * shear / mu;
        if (ba - 40.0).abs() < 0.4 || (ba - 450.0).abs() < 4.5 {
            continue;
        }
        queries.push(GranularFlowRegimeQuery::new(rho_s, diam, shear, mu));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    let mut macro_viscous = 0;
    let mut transitional = 0;
    let mut grain_inertia = 0;
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "sweep[{i}] should be master-valid");
        match res.regime {
            0 => macro_viscous += 1,
            1 => transitional += 1,
            2 => grain_inertia += 1,
            other => panic!("sweep[{i}] unexpected regime {other}"),
        }
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
    assert!(
        macro_viscous > 0,
        "sweep should cover the macro-viscous regime"
    );
    assert!(
        transitional > 0,
        "sweep should cover the transitional regime"
    );
    assert!(
        grain_inertia > 0,
        "sweep should cover the grain-inertia regime"
    );
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
