//! Real-device parity for the Beverloo slot-discharge twin:
//! [`GpuBeverlooSlotDischarge`](prism_volumetric_gpu::beverloo_slot_discharge::GpuBeverlooSlotDischarge)
//! must reproduce the `CPU` golden `BeverlooSlot` constructor and getters of
//! `prism_physics_core::collider::hopper_discharge`. A long slot of clear width
//! `W` and length `L` draining grains of diameter `d` has effective aperture
//! `W - k*d`; it jams when that aperture is non-positive, and otherwise the
//! mass flow rate is `Q = C * rho_b * sqrt(g) * L * (aperture * sqrt(aperture))`
//! with volumetric flow rate `Q / rho_b`.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly so the test never imports `prism_render_architecture`
//! or `prism_physics_core`. The golden is pure `f32`, so the oracle is pure
//! `f32` too and mirrors the same operator order and the same validity gates.
//!
//! The fixtures cover a normal flowing orifice, a jammed orifice (valid with a
//! zero flow rate), the constructor rejections (`C <= 0`, `k < 0`, `W <= 0`,
//! `L <= 0`), the flow-argument rejections (`rho_b <= 0`, `g <= 0`, `d <= 0`),
//! the non-finite rejections (`C = NaN`, `rho_b = inf`), a `Q / rho_b` identity
//! check, a mixed batch validating the `std430` stride, and an empty batch the
//! host short-circuits with no dispatch. A sweep over random finite inputs
//! follows; it keeps every parameter master-valid and rejection-samples so
//! `|aperture|` stays off the `aperture = 0` knee, covering both the flowing
//! and jammed sides.
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
//! with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`) and the `jams`/`valid`
//! words are compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::hopper_discharge`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::beverloo_slot_discharge::{
    BeverlooSlotDischargeQuery, BeverlooSlotDischargeResult, GpuBeverlooSlotDischarge,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// The resolved oracle outputs, in the same encoding as the device result.
struct Oracle {
    effective_aperture: f32,
    jams: bool,
    mass_flow_rate: f32,
    volumetric_flow_rate: f32,
    valid: bool,
}

/// Independent host oracle: reproduces the golden `BeverlooSlot` constructor
/// gate, the flow-argument gate and the getters in the golden operator order,
/// pure `f32`.
fn oracle(q: &BeverlooSlotDischargeQuery) -> Oracle {
    let c = q.discharge_coeff;
    let k = q.shape_factor;
    let w = q.width;
    let slot_length = q.length;
    let rho = q.bulk_density;
    let grav = q.gravity;
    let d = q.grain_diameter;

    let slot_ok = c.is_finite()
        && k.is_finite()
        && w.is_finite()
        && slot_length.is_finite()
        && c > 0.0
        && k >= 0.0
        && w > 0.0
        && slot_length > 0.0;
    let flow_ok =
        rho.is_finite() && grav.is_finite() && d.is_finite() && rho > 0.0 && grav > 0.0 && d > 0.0;
    let master_ok = slot_ok && flow_ok;

    if !master_ok {
        return Oracle {
            effective_aperture: 0.0,
            jams: false,
            mass_flow_rate: 0.0,
            volumetric_flow_rate: 0.0,
            valid: false,
        };
    }

    let aperture = w - k * d;
    let jams = aperture <= 0.0;
    let mass_flow_rate = if aperture <= 0.0 {
        0.0
    } else {
        c * rho * grav.sqrt() * slot_length * (aperture * aperture.sqrt())
    };
    let volumetric_flow_rate = mass_flow_rate / rho;

    Oracle {
        effective_aperture: aperture,
        jams,
        mass_flow_rate,
        volumetric_flow_rate,
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

/// Asserts a single `GPU` result matches the independent oracle: the `jams` and
/// `valid` words match exactly, each scalar matches to tolerance when valid, and
/// every scalar is zero when the master flag is cleared.
fn assert_parity(gpu: &BeverlooSlotDischargeResult, q: &BeverlooSlotDischargeQuery, label: &str) {
    let o = oracle(q);
    assert_eq!(gpu.valid, o.valid, "{label}: master valid flag");
    assert_eq!(gpu.jams, o.jams, "{label}: jams flag");

    if !o.valid {
        assert_eq!(
            gpu.effective_aperture, 0.0,
            "{label}: effective_aperture zeroed"
        );
        assert_eq!(gpu.mass_flow_rate, 0.0, "{label}: mass_flow_rate zeroed");
        assert_eq!(
            gpu.volumetric_flow_rate, 0.0,
            "{label}: volumetric_flow_rate zeroed"
        );
        return;
    }

    assert!(
        close(gpu.effective_aperture, o.effective_aperture),
        "{label}: effective_aperture gpu={} oracle={}",
        gpu.effective_aperture,
        o.effective_aperture
    );
    assert!(
        close(gpu.mass_flow_rate, o.mass_flow_rate),
        "{label}: mass_flow_rate gpu={} oracle={}",
        gpu.mass_flow_rate,
        o.mass_flow_rate
    );
    assert!(
        close(gpu.volumetric_flow_rate, o.volumetric_flow_rate),
        "{label}: volumetric_flow_rate gpu={} oracle={}",
        gpu.volumetric_flow_rate,
        o.volumetric_flow_rate
    );
}

/// A representative well-conditioned flowing query: `C = 0.6`, `k = 1.5`,
/// `W = 0.1`, `L = 0.5`, `rho_b = 1500`, `g = 9.81`, `d = 0.01`, so the
/// effective aperture is `0.085 > 0`.
fn flowing_query() -> BeverlooSlotDischargeQuery {
    BeverlooSlotDischargeQuery::new(0.6, 1.5, 0.1, 0.5, 1500.0, 9.81, 0.01)
}

#[test]
fn normal_flowing_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBeverlooSlotDischarge::new(&ctx);
    let q = flowing_query();
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid, "constructor and flow gate should accept");
    assert!(!out[0].jams, "aperture 0.085 should flow");
    assert!(out[0].mass_flow_rate > 0.0, "flowing orifice has Q > 0");
    assert_parity(&out[0], &q, "normal_flowing");
}

#[test]
fn jammed_is_valid_with_zero_flow() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBeverlooSlotDischarge::new(&ctx);
    // k*d = 2.5 * 0.1 = 0.25 > W = 0.05, so the aperture is negative => jam.
    let q = BeverlooSlotDischargeQuery::new(0.6, 2.5, 0.05, 0.5, 1500.0, 9.81, 0.1);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid, "a jammed slot is still a valid prediction");
    assert!(out[0].jams, "W <= k*d should jam");
    assert_eq!(out[0].mass_flow_rate, 0.0, "jammed orifice has Q = 0");
    assert_eq!(out[0].volumetric_flow_rate, 0.0, "jammed orifice has 0 vol");
    assert_parity(&out[0], &q, "jammed");
}

#[test]
fn constructor_rejections_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBeverlooSlotDischarge::new(&ctx);
    let base = flowing_query();
    let queries = vec![
        BeverlooSlotDischargeQuery {
            discharge_coeff: 0.0,
            ..base
        },
        BeverlooSlotDischargeQuery {
            discharge_coeff: -0.6,
            ..base
        },
        BeverlooSlotDischargeQuery {
            shape_factor: -1.0,
            ..base
        },
        BeverlooSlotDischargeQuery { width: 0.0, ..base },
        BeverlooSlotDischargeQuery {
            width: -0.1,
            ..base
        },
        BeverlooSlotDischargeQuery {
            length: 0.0,
            ..base
        },
        BeverlooSlotDischargeQuery {
            length: -0.5,
            ..base
        },
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(!res.valid, "constructor_rejection[{i}] should be invalid");
        assert_parity(res, q, &format!("constructor_rejection[{i}]"));
    }
}

#[test]
fn flow_argument_rejections_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBeverlooSlotDischarge::new(&ctx);
    let base = flowing_query();
    let queries = vec![
        BeverlooSlotDischargeQuery {
            bulk_density: 0.0,
            ..base
        },
        BeverlooSlotDischargeQuery {
            bulk_density: -1500.0,
            ..base
        },
        BeverlooSlotDischargeQuery {
            gravity: 0.0,
            ..base
        },
        BeverlooSlotDischargeQuery {
            gravity: -9.81,
            ..base
        },
        BeverlooSlotDischargeQuery {
            grain_diameter: 0.0,
            ..base
        },
        BeverlooSlotDischargeQuery {
            grain_diameter: -0.01,
            ..base
        },
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(!res.valid, "flow_rejection[{i}] should be invalid");
        assert_parity(res, q, &format!("flow_rejection[{i}]"));
    }
}

#[test]
fn non_finite_inputs_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBeverlooSlotDischarge::new(&ctx);
    let base = flowing_query();
    let queries = vec![
        BeverlooSlotDischargeQuery {
            discharge_coeff: f32::NAN,
            ..base
        },
        BeverlooSlotDischargeQuery {
            bulk_density: f32::INFINITY,
            ..base
        },
        BeverlooSlotDischargeQuery {
            grain_diameter: f32::NEG_INFINITY,
            ..base
        },
        BeverlooSlotDischargeQuery {
            width: f32::NAN,
            ..base
        },
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(!res.valid, "non_finite[{i}] should be invalid");
        assert_parity(res, q, &format!("non_finite[{i}]"));
    }
}

#[test]
fn volumetric_is_mass_over_density() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBeverlooSlotDischarge::new(&ctx);
    let q = flowing_query();
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let expected = out[0].mass_flow_rate / q.bulk_density;
    assert!(
        close(out[0].volumetric_flow_rate, expected),
        "volumetric should be Q / rho_b: gpu={} expected={}",
        out[0].volumetric_flow_rate,
        expected
    );
    assert_parity(&out[0], &q, "volumetric");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBeverlooSlotDischarge::new(&ctx);
    let queries = vec![
        flowing_query(),
        // jammed
        BeverlooSlotDischargeQuery::new(0.6, 2.5, 0.05, 0.5, 1500.0, 9.81, 0.1),
        // invalid (gravity <= 0)
        BeverlooSlotDischargeQuery::new(0.6, 1.5, 0.1, 0.5, 1500.0, 0.0, 0.01),
        // another flowing with a different L
        BeverlooSlotDischargeQuery::new(0.58, 1.4, 0.2, 1.0, 1200.0, 9.81, 0.02),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("mixed[{i}]"));
    }
    assert!(out[0].valid && !out[0].jams, "slot 0 flows");
    assert!(out[1].valid && out[1].jams, "slot 1 jams");
    assert!(!out[2].valid, "slot 2 invalid");
    assert!(out[3].valid && !out[3].jams, "slot 3 flows");
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBeverlooSlotDischarge::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBeverlooSlotDischarge::new(&ctx);
    let mut lcg = Lcg::new(0x2B7E_1591);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Keep every parameter master-valid; reject-sample so |aperture| stays
        // >= 0.005 off the aperture = 0 knee, covering both flowing and jammed
        // sides without the ^(3/2) relative error near zero.
        let c = lcg.next_range(0.4, 0.7);
        let k = lcg.next_range(0.0, 3.0);
        let w = lcg.next_range(0.02, 0.5);
        let slot_length = lcg.next_range(0.1, 2.0);
        let rho = lcg.next_range(500.0, 2500.0);
        let grav = lcg.next_range(1.0, 20.0);
        let d = lcg.next_range(0.001, 0.2);
        let aperture = w - k * d;
        if aperture.abs() < 0.005 {
            continue;
        }
        queries.push(BeverlooSlotDischargeQuery::new(
            c,
            k,
            w,
            slot_length,
            rho,
            grav,
            d,
        ));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    let mut flowing = 0;
    let mut jammed = 0;
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(res.valid, "sweep[{i}] should be master-valid");
        if res.jams {
            jammed += 1;
        } else {
            flowing += 1;
        }
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
    assert!(flowing > 0, "sweep should cover the flowing side");
    assert!(jammed > 0, "sweep should cover the jammed side");
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
