//! Real-device parity for the granular dilatancy-law twin:
//! [`GpuGranularDilatancyLaw`](prism_volumetric_gpu::granular_dilatancy_law::GpuGranularDilatancyLaw)
//! must reproduce the `CPU` golden `DilatancyLaw` of
//! `prism_physics_core::collider::granular_rheology`. The packing fraction of a
//! granular packing decreases from its dense static value `phi_max` as the
//! inertial number `I` grows: `phi(I) = clamp(phi_max - slope*I, 0, phi_max)`,
//! valid only when `phi_max` is finite and in `(0, 1]` and `slope` is finite and
//! `>= 0`; a non-finite or non-positive inertial number returns the dense limit
//! `phi_max`.
//!
//! The oracle here is an independent re-implementation of that closed form — the
//! finiteness and range guards, then the dense-limit short-circuit and the
//! clamped linear law — written out directly so the test never imports
//! `prism_render_architecture`, `prism_physics_core` or `glam`.
//!
//! The fixtures cover the unsaturated segment, the dense limit from a
//! non-positive or non-finite inertial number, the clamped-to-zero saturated
//! regime, out-of-range or non-finite models (invalid), a batch of two or more
//! elements that mixes valid and invalid models to validate the `std430` stride,
//! and an empty batch the host short-circuits with no dispatch. A sweep over
//! random parameters follows, split across the three regimes while staying away
//! from the clamp knee `I = phi_max / slope`.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through a multiply, a subtract and a
//! `clamp`, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a `GPU` may contract a multiply-add). The valid `volume_fraction`
//! scalar is compared with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`);
//! the discrete `valid` flag is compared exactly. The sweep stays away from the
//! clamp knee so `CPU`/`GPU` round-off cannot straddle the clamp segment
//! boundary and flip the saturated/unsaturated decision.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::granular_rheology`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::granular_dilatancy_law::{
    GpuGranularDilatancyLaw, GranularDilatancyLawQuery, GranularDilatancyLawResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `DilatancyLaw::new` validity plus
/// `DilatancyLaw::volume_fraction` in the golden operator order, returning the
/// packing fraction and the validity flag.
fn oracle(q: &GranularDilatancyLawQuery) -> (f32, u32) {
    let phi_max = q.phi_max;
    let slope = q.slope;
    if !phi_max.is_finite() || phi_max <= 0.0 || phi_max > 1.0 {
        return (0.0, 0);
    }
    if !slope.is_finite() || slope < 0.0 {
        return (0.0, 0);
    }
    let i = q.inertial_number;
    let vf = if !i.is_finite() || i <= 0.0 {
        phi_max
    } else {
        (phi_max - slope * i).clamp(0.0, phi_max)
    };
    (vf, 1)
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
/// `valid` flag exactly, and the `volume_fraction` scalar to tolerance when
/// valid.
fn assert_parity(gpu: &GranularDilatancyLawResult, q: &GranularDilatancyLawQuery, label: &str) {
    let (vf, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        assert!(
            close(gpu.volume_fraction, vf),
            "{label}: volume_fraction mismatch gpu={} oracle={}",
            gpu.volume_fraction,
            vf
        );
    }
}

#[test]
fn unsaturated_segment_follows_linear_law() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularDilatancyLaw::new(&ctx);
    // 0.62 - 0.2 * 1.0 = 0.42, inside (0, phi_max), so unclamped.
    let q = GranularDilatancyLawQuery::new(0.62, 0.2, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].volume_fraction, 0.42));
    assert_parity(&out[0], &q, "unsaturated");
}

#[test]
fn dense_limit_from_nonpositive_inertial_number() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularDilatancyLaw::new(&ctx);
    let negative = GranularDilatancyLawQuery::new(0.6, 0.5, -1.0);
    let zero = GranularDilatancyLawQuery::new(0.6, 0.5, 0.0);
    let out = gpu.evaluate(&ctx, &[negative, zero]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 1);
    // Both select the dense limit phi_max = 0.6.
    assert!(close(out[0].volume_fraction, 0.6));
    assert!(close(out[1].volume_fraction, 0.6));
    assert_parity(&out[0], &negative, "dense_negative");
    assert_parity(&out[1], &zero, "dense_zero");
}

#[test]
fn nonfinite_inertial_number_is_dense_limit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularDilatancyLaw::new(&ctx);
    let inf = GranularDilatancyLawQuery::new(0.55, 0.3, f32::INFINITY);
    let nan = GranularDilatancyLawQuery::new(0.55, 0.3, f32::NAN);
    let out = gpu.evaluate(&ctx, &[inf, nan]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 1);
    assert!(close(out[0].volume_fraction, 0.55));
    assert!(close(out[1].volume_fraction, 0.55));
    assert_parity(&out[0], &inf, "dense_inf");
    assert_parity(&out[1], &nan, "dense_nan");
}

#[test]
fn saturated_segment_clamps_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularDilatancyLaw::new(&ctx);
    // 0.5 - 2.0 * 1.0 = -1.5, clamped to 0.
    let q = GranularDilatancyLawQuery::new(0.5, 2.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].volume_fraction, 0.0));
    assert_parity(&out[0], &q, "saturated");
}

#[test]
fn out_of_range_models_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularDilatancyLaw::new(&ctx);
    let phi_zero = GranularDilatancyLawQuery::new(0.0, 0.2, 1.0);
    let phi_too_large = GranularDilatancyLawQuery::new(1.1, 0.2, 1.0);
    let slope_negative = GranularDilatancyLawQuery::new(0.6, -0.1, 1.0);
    let phi_nan = GranularDilatancyLawQuery::new(f32::NAN, 0.2, 1.0);
    let slope_inf = GranularDilatancyLawQuery::new(0.6, f32::INFINITY, 1.0);
    let queries = [phi_zero, phi_too_large, slope_negative, phi_nan, slope_inf];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0, "model[{i}] should be invalid");
        assert_eq!(res.volume_fraction, 0.0, "model[{i}] volume_fraction");
        assert_parity(res, q, &format!("invalid[{i}]"));
    }
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularDilatancyLaw::new(&ctx);
    let queries = vec![
        GranularDilatancyLawQuery::new(0.62, 0.2, 1.0),
        GranularDilatancyLawQuery::new(1.5, 0.2, 1.0),
        GranularDilatancyLawQuery::new(0.6, 0.5, -2.0),
        GranularDilatancyLawQuery::new(0.5, 2.0, 1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 0);
    assert_eq!(out[2].valid, 1);
    assert_eq!(out[3].valid, 1);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularDilatancyLaw::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGranularDilatancyLaw::new(&ctx);
    let mut lcg = Lcg::new(0x0D11_A7C9);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        match queries.len() % 3 {
            0 => {
                // Unsaturated: phi(I) sits strictly inside (0, phi_max), away
                // from both the zero clamp and the dense limit, so round-off
                // cannot cross a clamp boundary.
                let phi_max = lcg.next_range(0.3, 1.0);
                let slope = lcg.next_range(0.1, 2.0);
                // target = slope * I in [0.1*phi_max, 0.8*phi_max].
                let target = lcg.next_range(0.1 * phi_max, 0.8 * phi_max);
                let i = target / slope;
                queries.push(GranularDilatancyLawQuery::new(phi_max, slope, i));
            }
            1 => {
                // Dense limit from a strictly-negative inertial number, well
                // away from zero.
                let phi_max = lcg.next_range(0.3, 1.0);
                let slope = lcg.next_range(0.0, 2.0);
                let i = lcg.next_range(-5.0, -0.05);
                queries.push(GranularDilatancyLawQuery::new(phi_max, slope, i));
            }
            _ => {
                // Saturated: phi_max - slope*I is comfortably negative, so the
                // clamp pins it to zero on both sides.
                let phi_max = lcg.next_range(0.1, 0.9);
                let slope = lcg.next_range(0.5, 3.0);
                // target = slope * I in [phi_max + 0.2, phi_max + 5.0].
                let target = lcg.next_range(phi_max + 0.2, phi_max + 5.0);
                let i = target / slope;
                queries.push(GranularDilatancyLawQuery::new(phi_max, slope, i));
            }
        }
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
