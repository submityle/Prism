//! Real-device parity for the contact-material combination twin:
//! [`GpuMaterialCombine`](prism_volumetric_gpu::material_combine::GpuMaterialCombine)
//! must reproduce the `CPU` golden `PhysicsMaterial::new` +
//! `PhysicsMaterial::combine` of `prism_physics_core::collider::material`. Each
//! of two contacting surfaces carries a raw friction and restitution; `new`
//! clamps friction to non-negative and restitution to `0.0..=1.0`, then
//! `combine` takes the geometric-mean friction `sqrt(fa * fb)` and the maximum
//! restitution `max(ra, rb)`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the per-side clamps then the geometric-mean friction and max restitution in
//! the golden operator order — written out directly so the test never imports
//! `prism_render_architecture` or `prism_physics_core`.
//!
//! The fixtures cover a plain combination, friction clamped from negative,
//! restitution clamped above `1` and below `0`, a zero friction forcing a zero
//! combined friction, both restitution end points, a non-finite (`NaN` and
//! `inf`) input forcing `valid = 0`, a batch of two or more elements mixing
//! valid and invalid pairs to validate the `std430` stride, and an empty batch
//! the host short-circuits with no dispatch. A sweep over random finite inputs
//! follows; since validity flips only on non-finite inputs, the sweep stays
//! finite and needs no knee rejection.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through clamps, a multiply, a `max`-with-0
//! and a `sqrt`, so `CPU` and `GPU` evaluate the same closed form but need not
//! be bit-exact (a `GPU` may contract a multiply-add). The valid `friction` and
//! `restitution` scalars are compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::material`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::material_combine::{
    GpuMaterialCombine, MaterialCombineQuery, MaterialCombineResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `PhysicsMaterial::new` +
/// `PhysicsMaterial::combine` in the golden operator order, returning the
/// combined friction, combined restitution and the validity flag.
fn oracle(q: &MaterialCombineQuery) -> (f32, f32, u32) {
    if !q.friction_a.is_finite()
        || !q.restitution_a.is_finite()
        || !q.friction_b.is_finite()
        || !q.restitution_b.is_finite()
    {
        return (0.0, 0.0, 0);
    }
    let fa = q.friction_a.max(0.0);
    let ra = q.restitution_a.clamp(0.0, 1.0);
    let fb = q.friction_b.max(0.0);
    let rb = q.restitution_b.clamp(0.0, 1.0);
    let friction = (fa * fb).max(0.0).sqrt();
    let restitution = ra.max(rb);
    (friction, restitution, 1)
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
/// `valid` flag exactly, and the `friction`/`restitution` scalars to tolerance
/// when valid.
fn assert_parity(gpu: &MaterialCombineResult, q: &MaterialCombineQuery, label: &str) {
    let (friction, restitution, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        assert!(
            close(gpu.friction, friction),
            "{label}: friction mismatch gpu={} oracle={}",
            gpu.friction,
            friction
        );
        assert!(
            close(gpu.restitution, restitution),
            "{label}: restitution mismatch gpu={} oracle={}",
            gpu.restitution,
            restitution
        );
    }
}

#[test]
fn plain_combination_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialCombine::new(&ctx);
    // sqrt(0.4*0.9)=sqrt(0.36)=0.6; max(0.2,0.8)=0.8.
    let q = MaterialCombineQuery::new(0.4, 0.2, 0.9, 0.8);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].friction, 0.6));
    assert!(close(out[0].restitution, 0.8));
    assert_parity(&out[0], &q, "plain");
}

#[test]
fn negative_friction_is_clamped_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialCombine::new(&ctx);
    // friction_a clamps to 0, so geometric mean is sqrt(0*0.5)=0.
    let q = MaterialCombineQuery::new(-3.0, 0.3, 0.5, 0.4);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].friction, 0.0));
    assert!(close(out[0].restitution, 0.4));
    assert_parity(&out[0], &q, "negative_friction");
}

#[test]
fn restitution_above_one_is_clamped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialCombine::new(&ctx);
    // restitution_a clamps 1.7 -> 1.0, so max is 1.0.
    let q = MaterialCombineQuery::new(0.6, 1.7, 0.6, 0.2);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].restitution, 1.0));
    assert_parity(&out[0], &q, "restitution_high");
}

#[test]
fn restitution_below_zero_is_clamped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialCombine::new(&ctx);
    // Both restitutions clamp to 0, so max is 0.
    let q = MaterialCombineQuery::new(0.5, -0.5, 0.8, -2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].restitution, 0.0));
    assert_parity(&out[0], &q, "restitution_low");
}

#[test]
fn zero_friction_forces_zero_combined_friction() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialCombine::new(&ctx);
    let q = MaterialCombineQuery::new(0.0, 0.5, 10.0, 0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].friction, 0.0));
    assert_parity(&out[0], &q, "zero_friction");
}

#[test]
fn restitution_end_points_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialCombine::new(&ctx);
    // One surface fully elastic (1.0), the other fully inelastic (0.0).
    let q = MaterialCombineQuery::new(0.3, 0.0, 0.7, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].restitution, 1.0));
    assert_parity(&out[0], &q, "restitution_ends");
}

#[test]
fn nan_input_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialCombine::new(&ctx);
    let q = MaterialCombineQuery::new(f32::NAN, 0.5, 0.5, 0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[0].friction, 0.0);
    assert_eq!(out[0].restitution, 0.0);
    assert_parity(&out[0], &q, "nan");
}

#[test]
fn infinite_input_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialCombine::new(&ctx);
    let positive = MaterialCombineQuery::new(0.5, 0.5, f32::INFINITY, 0.5);
    let negative = MaterialCombineQuery::new(0.5, f32::NEG_INFINITY, 0.5, 0.5);
    let out = gpu.evaluate(&ctx, &[positive, negative]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 0, "positive infinity should be invalid");
    assert_eq!(out[1].valid, 0, "negative infinity should be invalid");
    assert_eq!(out[0].friction, 0.0);
    assert_eq!(out[0].restitution, 0.0);
    assert_eq!(out[1].friction, 0.0);
    assert_eq!(out[1].restitution, 0.0);
    assert_parity(&out[0], &positive, "inf_pos");
    assert_parity(&out[1], &negative, "inf_neg");
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialCombine::new(&ctx);
    let queries = vec![
        MaterialCombineQuery::new(0.4, 0.2, 0.9, 0.8),
        MaterialCombineQuery::new(f32::NAN, 0.1, 0.2, 0.3),
        MaterialCombineQuery::new(1.0, 0.9, 4.0, 0.1),
        MaterialCombineQuery::new(0.5, 0.5, f32::INFINITY, 0.5),
        MaterialCombineQuery::new(-2.0, 1.5, 0.25, -0.5),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 0);
    assert_eq!(out[2].valid, 1);
    assert_eq!(out[3].valid, 0);
    assert_eq!(out[4].valid, 1);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialCombine::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMaterialCombine::new(&ctx);
    let mut lcg = Lcg::new(0x0CA9_11A5);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Finite inputs: friction in [0, 5], restitution in [0, 1]. Validity
        // flips only on non-finite inputs, so finite values keep it stable.
        let friction_a = lcg.next_range(0.0, 5.0);
        let restitution_a = lcg.next_range(0.0, 1.0);
        let friction_b = lcg.next_range(0.0, 5.0);
        let restitution_b = lcg.next_range(0.0, 1.0);
        queries.push(MaterialCombineQuery::new(
            friction_a,
            restitution_a,
            friction_b,
            restitution_b,
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
