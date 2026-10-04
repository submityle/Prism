//! Real-device parity for the reduced-radius twin:
//! [`GpuCapillaryReducedRadius`](prism_volumetric_gpu::capillary_reduced_radius::GpuCapillaryReducedRadius)
//! must reproduce the `CPU` golden `CapillaryBridgeModel::reduced_radius` of
//! `prism_physics_core::collider::capillary_bridge`. The reduced radius of two
//! spheres is `R = 2 * r_a * r_b / (r_a + r_b)` when both radii are finite and
//! strictly positive, and undefined (invalid) otherwise.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the finiteness and positivity guards, then `2 * a * b / (a + b)` in the
//! golden operator order — written out directly so the test never imports
//! `prism_render_architecture` or `prism_physics_core`.
//!
//! The fixtures cover equal radii (where `R = r`), wildly mismatched radii, one
//! degenerate radius (`<= 0`, invalid), both degenerate, a batch of two or more
//! elements that mixes valid and invalid pairs to validate the `std430` stride,
//! and an empty batch the host short-circuits with no dispatch. A sweep over
//! random strictly-positive radii follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through multiplies, an add and a division,
//! so `CPU` and `GPU` evaluate the same closed form but need not be bit-exact
//! (a `GPU` may contract a multiply-add). The valid `reduced` scalar is compared
//! with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid`
//! flag is compared exactly. The fixtures keep radii strictly positive and well
//! away from zero/non-finite so the validity decision agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::capillary_bridge`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::capillary_reduced_radius::{
    CapillaryReducedRadiusQuery, CapillaryReducedRadiusResult, GpuCapillaryReducedRadius,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `reduced_radius` in the golden operator
/// order, returning the reduced radius and the validity flag.
fn oracle(q: &CapillaryReducedRadiusQuery) -> (f32, u32) {
    let ra = q.radius_a;
    let rb = q.radius_b;
    if !ra.is_finite() || !rb.is_finite() || ra <= 0.0 || rb <= 0.0 {
        return (0.0, 0);
    }
    (2.0 * ra * rb / (ra + rb), 1)
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
fn assert_parity(gpu: &CapillaryReducedRadiusResult, q: &CapillaryReducedRadiusQuery, label: &str) {
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
fn equal_radii_reduce_to_the_radius() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryReducedRadius::new(&ctx);
    // 2*r*r/(r+r) = r.
    let q = CapillaryReducedRadiusQuery::new(0.5, 0.5);
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
    let gpu = GpuCapillaryReducedRadius::new(&ctx);
    // 2*1*3/(1+3) = 1.5.
    let q = CapillaryReducedRadiusQuery::new(1.0, 3.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].reduced, 1.5));
    assert_parity(&out[0], &q, "mismatched");
}

#[test]
fn disparate_radii_approach_twice_the_smaller() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryReducedRadius::new(&ctx);
    // r_a << r_b, so R -> 2*r_a.
    let q = CapillaryReducedRadiusQuery::new(0.01, 1000.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "disparate");
}

#[test]
fn one_degenerate_radius_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryReducedRadius::new(&ctx);
    let zero = CapillaryReducedRadiusQuery::new(0.0, 1.0);
    let negative = CapillaryReducedRadiusQuery::new(1.0, -2.0);
    let out = gpu.evaluate(&ctx, &[zero, negative]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 0, "zero radius should be invalid");
    assert_eq!(out[1].valid, 0, "negative radius should be invalid");
    assert_eq!(out[0].reduced, 0.0);
    assert_eq!(out[1].reduced, 0.0);
    assert_parity(&out[0], &zero, "zero");
    assert_parity(&out[1], &negative, "negative");
}

#[test]
fn both_degenerate_radii_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryReducedRadius::new(&ctx);
    let q = CapillaryReducedRadiusQuery::new(0.0, 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[0].reduced, 0.0);
    assert_parity(&out[0], &q, "both_zero");
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryReducedRadius::new(&ctx);
    let queries = vec![
        CapillaryReducedRadiusQuery::new(2.0, 2.0),
        CapillaryReducedRadiusQuery::new(-1.0, 5.0),
        CapillaryReducedRadiusQuery::new(4.0, 12.0),
        CapillaryReducedRadiusQuery::new(0.0, 7.0),
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
    let gpu = GpuCapillaryReducedRadius::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryReducedRadius::new(&ctx);
    let mut lcg = Lcg::new(0x0CA9_11A5);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Strictly-positive radii well away from zero so validity is stable.
        let radius_a = lcg.next_range(0.05, 50.0);
        let radius_b = lcg.next_range(0.05, 50.0);
        queries.push(CapillaryReducedRadiusQuery::new(radius_a, radius_b));
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
