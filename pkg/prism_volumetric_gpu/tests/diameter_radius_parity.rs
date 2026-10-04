//! Real-device parity for the diameter-radius twin:
//! [`GpuDiameterRadius`](prism_volumetric_gpu::diameter_radius::GpuDiameterRadius)
//! must reproduce the `CPU` golden `MeshDiameter::radius` of
//! `prism_physics_core::collider::diameter`. Half the diameter,
//! `radius = diameter * 0.5`, is a lower bound on any enclosing-sphere radius.
//!
//! The oracle here is an independent re-implementation of that closed form — a
//! single multiply by one half — written out directly so the test never
//! imports `prism_render_architecture` or `prism_physics_core`.
//!
//! The fixtures cover a general positive diameter, zero, a negative value
//! (scaled as-is), a large value, a batch of two or more elements validating
//! the `std430` stride, and an empty batch the host short-circuits with no
//! dispatch. A sweep over random diameters follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The scalar is a single multiply, so `CPU` and `GPU` evaluate the same closed
//! form (and a half-scale is in practice exact). The `radius` is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`). The closed form has no
//! degenerate branch, so every input is valid.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::diameter::MeshDiameter::radius`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::diameter_radius::{
    DiameterRadiusQuery, DiameterRadiusResult, GpuDiameterRadius,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `MeshDiameter::radius` as a single
/// multiply by one half.
fn oracle(q: &DiameterRadiusQuery) -> f32 {
    q.diameter * 0.5
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle `radius` to
/// tolerance.
fn assert_parity(gpu: &DiameterRadiusResult, q: &DiameterRadiusQuery, label: &str) {
    let radius = oracle(q);
    assert!(
        close(gpu.radius, radius),
        "{label}: radius mismatch gpu={} oracle={}",
        gpu.radius,
        radius
    );
}

#[test]
fn positive_diameter_halves() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterRadius::new(&ctx);
    let q = DiameterRadiusQuery::new(3.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(close(out[0].radius, 1.5));
    assert_parity(&out[0], &q, "positive");
}

#[test]
fn zero_diameter_is_zero_radius() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterRadius::new(&ctx);
    let q = DiameterRadiusQuery::new(0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].radius, 0.0);
    assert_parity(&out[0], &q, "zero");
}

#[test]
fn negative_diameter_scales_as_is() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterRadius::new(&ctx);
    // The golden applies no guard: -2.0 * 0.5 = -1.0.
    let q = DiameterRadiusQuery::new(-2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(close(out[0].radius, -1.0));
    assert_parity(&out[0], &q, "negative");
}

#[test]
fn large_diameter_halves() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterRadius::new(&ctx);
    let q = DiameterRadiusQuery::new(1.0e6);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(close(out[0].radius, 5.0e5));
    assert_parity(&out[0], &q, "large");
}

#[test]
fn batch_of_several_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterRadius::new(&ctx);
    let queries = vec![
        DiameterRadiusQuery::new(1.0),
        DiameterRadiusQuery::new(7.5),
        DiameterRadiusQuery::new(-4.0),
        DiameterRadiusQuery::new(100.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterRadius::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDiameterRadius::new(&ctx);
    let mut lcg = Lcg::new(0x0D1A_3E70);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let diameter = lcg.next_range(-1000.0, 1000.0);
        queries.push(DiameterRadiusQuery::new(diameter));
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
