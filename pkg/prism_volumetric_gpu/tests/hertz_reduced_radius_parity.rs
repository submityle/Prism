//! Real-device parity for the Hertz effective-radius twin:
//! [`GpuHertzReducedRadius`](prism_volumetric_gpu::hertz_reduced_radius::GpuHertzReducedRadius)
//! must reproduce the `CPU` golden effective radius computed inside
//! `hertz_contact_between` of `prism_physics_core::collider::hertz_contact`. The
//! effective radius of two spheres in Hertz contact is
//! `R = r_a * r_b / (r_a + r_b)` when both radii are finite and strictly
//! positive, and undefined (invalid) otherwise.
//!
//! Note this is deliberately **not** the capillary-bridge reduced radius
//! `2 * r_a * r_b / (r_a + r_b)`: the Hertz form has **no factor of 2**.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the finiteness and positivity guards, then `a * b / (a + b)` in the golden
//! operator order — written out directly so the test never imports
//! `prism_render_architecture` or `prism_physics_core`.
//!
//! The fixtures cover equal radii (where `R = r / 2`), mismatched radii, one
//! degenerate radius (`<= 0`, invalid), non-finite radii, extreme magnitudes, a
//! batch of two or more elements that mixes valid and invalid pairs to validate
//! the `std430` stride, and an empty batch the host short-circuits with no
//! dispatch. A sweep over random strictly-positive radii follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through a multiply, an add and a division,
//! so `CPU` and `GPU` evaluate the same closed form but need not be bit-exact
//! (a `GPU` may contract a multiply-add). The valid `reduced` scalar is compared
//! with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid`
//! flag is compared exactly. The fixtures keep radii strictly positive and well
//! away from zero/non-finite so the validity decision agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::hertz_contact`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hertz_reduced_radius::{
    GpuHertzReducedRadius, HertzReducedRadiusQuery, HertzReducedRadiusResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces the Hertz effective radius in the golden
/// operator order, returning the effective radius and the validity flag.
fn oracle(q: &HertzReducedRadiusQuery) -> (f32, u32) {
    let ra = q.radius_a;
    let rb = q.radius_b;
    if !ra.is_finite() || !rb.is_finite() || ra <= 0.0 || rb <= 0.0 {
        return (0.0, 0);
    }
    (ra * rb / (ra + rb), 1)
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
/// `valid` flag exactly, and the `reduced` scalar to tolerance when valid.
fn assert_parity(gpu: &HertzReducedRadiusResult, q: &HertzReducedRadiusQuery, label: &str) {
    let (reduced, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        assert!(
            close(gpu.reduced, reduced),
            "{label}: reduced mismatch gpu={} oracle={}",
            gpu.reduced,
            reduced
        );
    }
}

#[test]
fn equal_radii_reduce_to_half_the_radius() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzReducedRadius::new(&ctx);
    // r*r/(r+r) = r/2.
    let q = HertzReducedRadiusQuery::new(1.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].reduced, 0.5));
    assert_parity(&out[0], &q, "equal");
}

#[test]
fn mismatched_radii_reduce_toward_the_smaller() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzReducedRadius::new(&ctx);
    // 1*3/(1+3) = 0.75.
    let q = HertzReducedRadiusQuery::new(1.0, 3.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].reduced, 0.75));
    assert_parity(&out[0], &q, "mismatched");
}

#[test]
fn disparate_radii_approach_the_smaller() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzReducedRadius::new(&ctx);
    // r_a << r_b, so R -> r_a.
    let q = HertzReducedRadiusQuery::new(1.0e-3, 1.0e3);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "disparate");
}

#[test]
fn tiny_positive_radii_are_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzReducedRadius::new(&ctx);
    let q = HertzReducedRadiusQuery::new(1.0e-3, 2.0e-3);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "tiny");
}

#[test]
fn zero_radius_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzReducedRadius::new(&ctx);
    let q = HertzReducedRadiusQuery::new(0.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[0].reduced, 0.0);
    assert_parity(&out[0], &q, "zero");
}

#[test]
fn negative_radius_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzReducedRadius::new(&ctx);
    let q = HertzReducedRadiusQuery::new(1.0, -2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[0].reduced, 0.0);
    assert_parity(&out[0], &q, "negative");
}

#[test]
fn nan_radius_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzReducedRadius::new(&ctx);
    let q = HertzReducedRadiusQuery::new(f32::NAN, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[0].reduced, 0.0);
    assert_parity(&out[0], &q, "nan");
}

#[test]
fn infinite_radius_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzReducedRadius::new(&ctx);
    let q = HertzReducedRadiusQuery::new(f32::INFINITY, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[0].reduced, 0.0);
    assert_parity(&out[0], &q, "inf");
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzReducedRadius::new(&ctx);
    let queries = vec![
        HertzReducedRadiusQuery::new(2.0, 2.0),
        HertzReducedRadiusQuery::new(-1.0, 5.0),
        HertzReducedRadiusQuery::new(4.0, 12.0),
        HertzReducedRadiusQuery::new(0.0, 7.0),
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
    let gpu = GpuHertzReducedRadius::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzReducedRadius::new(&ctx);
    let mut lcg = Lcg::new(0x0CA9_11A5);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Strictly-positive radii well away from zero so validity is stable.
        let radius_a = lcg.next_range(0.05, 10.0);
        let radius_b = lcg.next_range(0.05, 10.0);
        queries.push(HertzReducedRadiusQuery::new(radius_a, radius_b));
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
