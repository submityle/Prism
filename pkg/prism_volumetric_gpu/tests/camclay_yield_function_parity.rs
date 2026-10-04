//! Real-device parity for the Cam-Clay trial-yield twin:
//! [`GpuCamClayYieldFunction`](prism_volumetric_gpu::camclay_yield_function::GpuCamClayYieldFunction)
//! must reproduce the trial-yield branch of the `CPU` golden
//! `return_map_camclay` of
//! `prism_physics_core::collider::tet_fem_camclay_plasticity`. The modified
//! Cam-Clay yield value is `Y = q^2 / M^2 + p * (p - p_c)`; a trial state is
//! elastic when `Y <= 0` and plastic (yielded) when `Y > 0`, and the tuple is
//! valid only when all four inputs are finite with `M > 0` and `p_c > 0`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the finiteness and positivity guards, then the golden `f64` intermediate
//! steps `Y = q*q/M^2 + p*(p - p_c)` narrowed to `f32` — written out directly
//! so the test never imports `prism_render_architecture` or
//! `prism_physics_core`.
//!
//! The fixtures cover an elastic interior point (`Y < 0`), a plastic exterior
//! point (`Y > 0`), varied `M` / `p_c`, degenerate tuples (`NaN`, `+/-inf`,
//! `M <= 0`, `p_c <= 0`, all invalid), a batch of two or more elements mixing
//! valid and invalid tuples to validate the `std430` stride, and an empty batch
//! the host short-circuits with no dispatch. A sweep over random well-posed
//! tuples follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The reference computes the criterion in `f64` intermediates; the device is
//! `f32`. The host oracle replays the golden `f64` steps and narrows to `f32`,
//! and the valid `yield_value` scalar is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`). The discrete `yielded`
//! and `valid` flags are compared exactly; the sweep keeps the trial criterion
//! well away from the `Y = 0` knee so the yield decision cannot be flipped by
//! the `f32`/`f64` gap.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::tet_fem_camclay_plasticity`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::camclay_yield_function::{
    CamClayYieldFunctionQuery, CamClayYieldFunctionResult, GpuCamClayYieldFunction,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: replays the golden `f64` trial-yield steps and
/// narrows to `f32`, returning the yield value, the `yielded` flag and the
/// `valid` flag.
fn oracle(q: &CamClayYieldFunctionQuery) -> (f32, bool, bool) {
    if !q.p.is_finite()
        || !q.q.is_finite()
        || !q.slope_m.is_finite()
        || !q.pre_consolidation.is_finite()
        || q.slope_m <= 0.0
        || q.pre_consolidation <= 0.0
    {
        return (0.0, false, false);
    }
    let m = f64::from(q.slope_m);
    let m2 = m * m;
    let p = f64::from(q.p);
    let qq = f64::from(q.q);
    let pc = f64::from(q.pre_consolidation);
    let y = qq * qq / m2 + p * (p - pc);
    (y as f32, y > 0.0, true)
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
/// `valid` and `yielded` flags exactly, and the `yield_value` scalar to
/// tolerance when valid.
fn assert_parity(gpu: &CamClayYieldFunctionResult, q: &CamClayYieldFunctionQuery, label: &str) {
    let (yield_value, yielded, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid {
        assert_eq!(gpu.yielded, yielded, "{label}: yielded flag mismatch");
        assert!(
            close(gpu.yield_value, yield_value),
            "{label}: yield_value mismatch gpu={} oracle={}",
            gpu.yield_value,
            yield_value
        );
    } else {
        assert!(!gpu.yielded, "{label}: invalid should not yield");
        assert_eq!(gpu.yield_value, 0.0, "{label}: invalid value should be 0");
    }
}

#[test]
fn interior_point_is_elastic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamClayYieldFunction::new(&ctx);
    // p inside (0, p_c), small q: Y = 0 + p*(p - p_c) < 0 -> elastic.
    let q = CamClayYieldFunctionQuery::new(0.3, 0.0, 1.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(!out[0].yielded);
    assert_parity(&out[0], &q, "interior");
}

#[test]
fn exterior_point_has_yielded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamClayYieldFunction::new(&ctx);
    // Large q pushes Y above the ellipse: q=3, M=1, p=0, p_c=1 -> Y=9 > 0.
    let q = CamClayYieldFunctionQuery::new(0.0, 3.0, 1.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(out[0].yielded);
    assert_parity(&out[0], &q, "exterior");
}

#[test]
fn varied_slope_and_preconsolidation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamClayYieldFunction::new(&ctx);
    // Mixed scales: Y = q^2/M^2 + p*(p - p_c).
    let a = CamClayYieldFunctionQuery::new(2.0, 1.5, 0.75, 5.0);
    let b = CamClayYieldFunctionQuery::new(-1.0, 4.0, 2.0, 3.0);
    let out = gpu.evaluate(&ctx, &[a, b]);
    assert_eq!(out.len(), 2);
    assert_parity(&out[0], &a, "varied_a");
    assert_parity(&out[1], &b, "varied_b");
}

#[test]
fn non_finite_inputs_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamClayYieldFunction::new(&ctx);
    let nan = CamClayYieldFunctionQuery::new(f32::NAN, 1.0, 1.0, 1.0);
    let pos_inf = CamClayYieldFunctionQuery::new(1.0, f32::INFINITY, 1.0, 1.0);
    let neg_inf = CamClayYieldFunctionQuery::new(1.0, 1.0, f32::NEG_INFINITY, 1.0);
    let out = gpu.evaluate(&ctx, &[nan, pos_inf, neg_inf]);
    assert_eq!(out.len(), 3);
    for (res, q) in out.iter().zip([nan, pos_inf, neg_inf].iter()) {
        assert!(!res.valid);
        assert!(!res.yielded);
        assert_eq!(res.yield_value, 0.0);
        assert_parity(res, q, "non_finite");
    }
}

#[test]
fn non_positive_model_parameters_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamClayYieldFunction::new(&ctx);
    let bad_m = CamClayYieldFunctionQuery::new(1.0, 1.0, 0.0, 1.0);
    let neg_m = CamClayYieldFunctionQuery::new(1.0, 1.0, -2.0, 1.0);
    let bad_pc = CamClayYieldFunctionQuery::new(1.0, 1.0, 1.0, 0.0);
    let neg_pc = CamClayYieldFunctionQuery::new(1.0, 1.0, 1.0, -3.0);
    let queries = vec![bad_m, neg_m, bad_pc, neg_pc];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (res, q) in out.iter().zip(queries.iter()) {
        assert!(!res.valid);
        assert_eq!(res.yield_value, 0.0);
        assert_parity(res, q, "non_positive");
    }
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamClayYieldFunction::new(&ctx);
    let queries = vec![
        CamClayYieldFunctionQuery::new(0.3, 0.0, 1.0, 1.0), // valid elastic
        CamClayYieldFunctionQuery::new(1.0, 1.0, -1.0, 1.0), // invalid M
        CamClayYieldFunctionQuery::new(0.0, 3.0, 1.0, 1.0), // valid yielded
        CamClayYieldFunctionQuery::new(f32::NAN, 1.0, 1.0, 1.0), // invalid NaN
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(out[0].valid);
    assert!(!out[0].yielded);
    assert!(!out[1].valid);
    assert!(out[2].valid);
    assert!(out[2].yielded);
    assert!(!out[3].valid);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamClayYieldFunction::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamClayYieldFunction::new(&ctx);
    let mut lcg = Lcg::new(0x0CA9_11A5);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Well-posed tuple: M in [0.5, 2], p_c > 0, q >= 0, p scaled to p_c.
        let pc = lcg.next_range(0.5, 5.0);
        let m = lcg.next_range(0.5, 2.0);
        let p = lcg.next_range(-5.0, 5.0) * pc * 0.2;
        let q = lcg.next_range(0.0, 5.0);
        let candidate = CamClayYieldFunctionQuery::new(p, q, m, pc);
        // Reject samples whose |Y| sits within a relative margin of the Y = 0
        // knee, so the f32/f64 gap cannot flip the discrete yielded decision.
        let md = f64::from(m);
        let y =
            f64::from(q) * f64::from(q) / (md * md) + f64::from(p) * (f64::from(p) - f64::from(pc));
        let scale = (f64::from(q) * f64::from(q) / (md * md)).abs()
            + (f64::from(p) * (f64::from(p) - f64::from(pc))).abs();
        if y.abs() < 0.05 * scale.max(1.0) {
            continue;
        }
        queries.push(candidate);
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
