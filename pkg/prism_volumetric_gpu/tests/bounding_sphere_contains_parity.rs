//! Real-device parity for the sphere-containment twin:
//! [`GpuBoundingSphereContains`](prism_volumetric_gpu::bounding_sphere_contains::GpuBoundingSphereContains)
//! must reproduce the `CPU` golden `BoundingSphere::contains_point` of
//! `prism_physics_core::collider::bounding_sphere`. A point is contained when
//! its squared distance to the sphere center is within the squared radius grown
//! by the relative slack `radius * CONTAIN_EPS + CONTAIN_EPS`
//! (`CONTAIN_EPS = 1e-5`).
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the slack, the relaxed radius, the squared distance and the single ordered
//! `<=` comparison — written out directly so the test never imports
//! `prism_render_architecture` or `prism_physics_core`. It compares squared
//! quantities (no `sqrt`), matching the kernel so neither side leaks a `sqrt`
//! `ULP` into the decision.
//!
//! The fixtures cover a point at the center, a point well inside, a point far
//! outside, a point near (but deliberately off) the boundary, a large radius
//! that scales the slack, a batch of two or more elements that mixes a
//! contained and an escaped point to validate the `std430` stride, and an empty
//! batch the host short-circuits with no dispatch. A sweep over random
//! sphere/point pairs follows, rejecting samples whose squared distance lands
//! within a relative `1e-2` margin of the relaxed squared radius so a last-bit
//! difference cannot flip the discrete `contains` flag.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through multiplies, adds and a dot
//! product, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a `GPU` may contract a multiply-add). The discrete `contains` and
//! `valid` flags are compared exactly; the fixtures keep every point away from
//! the boundary knee so the comparison agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::bounding_sphere`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::bounding_sphere_contains::{
    BoundingSphereContainsQuery, BoundingSphereContainsResult, GpuBoundingSphereContains,
};
use prism_volumetric_gpu::GpuContext;

/// The containment slack base, matching the golden `CONTAIN_EPS`.
const CONTAIN_EPS: f32 = 1.0e-5;

/// Independent host oracle: reproduces `contains_point` with the same squared
/// comparison the kernel uses, returning the containment flag and the
/// always-`1` validity flag.
fn oracle(q: &BoundingSphereContainsQuery) -> (u32, u32) {
    let slack = q.radius * CONTAIN_EPS + CONTAIN_EPS;
    let r = q.radius + slack;
    let d = [
        q.point[0] - q.center[0],
        q.point[1] - q.center[1],
        q.point[2] - q.center[2],
    ];
    let dist_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let contains = u32::from(dist_sq <= r * r);
    (contains, 1)
}

/// Returns the signed relative margin of a query from its boundary knee:
/// `(relaxed_r_sq - dist_sq) / max(relaxed_r_sq, REL_FLOOR)`. A value with a
/// large absolute magnitude is safely away from the flip point.
fn boundary_margin(q: &BoundingSphereContainsQuery) -> f32 {
    let slack = q.radius * CONTAIN_EPS + CONTAIN_EPS;
    let r = q.radius + slack;
    let d = [
        q.point[0] - q.center[0],
        q.point[1] - q.center[1],
        q.point[2] - q.center[2],
    ];
    let dist_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let r_sq = r * r;
    (r_sq - dist_sq) / r_sq.max(1.0e-6)
}

/// Asserts a single `GPU` result matches the independent oracle exactly: both
/// discrete flags.
fn assert_parity(gpu: &BoundingSphereContainsResult, q: &BoundingSphereContainsQuery, label: &str) {
    let (contains, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    assert_eq!(
        gpu.contains, contains,
        "{label}: contains flag mismatch (margin={})",
        boundary_margin(q)
    );
}

#[test]
fn point_at_center_is_contained() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereContains::new(&ctx);
    let q = BoundingSphereContainsQuery::new([1.0, 2.0, 3.0], 2.0, [1.0, 2.0, 3.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].contains, 1);
    assert_parity(&out[0], &q, "center");
}

#[test]
fn point_well_inside_is_contained() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereContains::new(&ctx);
    // Distance 1.0 inside radius 5.0.
    let q = BoundingSphereContainsQuery::new([0.0, 0.0, 0.0], 5.0, [1.0, 0.0, 0.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].contains, 1);
    assert_parity(&out[0], &q, "inside");
}

#[test]
fn point_far_outside_is_excluded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereContains::new(&ctx);
    // Distance 10.0 well beyond radius 1.0.
    let q = BoundingSphereContainsQuery::new([0.0, 0.0, 0.0], 1.0, [10.0, 0.0, 0.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].contains, 0);
    assert_parity(&out[0], &q, "far_outside");
}

#[test]
fn point_near_boundary_pushed_off_the_knee() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereContains::new(&ctx);
    // Radius 3.0; place the point at 0.97 * r (safely inside) and at 1.03 * r
    // (safely outside), both at least a relative 1e-2 off the knee.
    let inside = BoundingSphereContainsQuery::new([0.0, 0.0, 0.0], 3.0, [3.0 * 0.97, 0.0, 0.0]);
    let outside = BoundingSphereContainsQuery::new([0.0, 0.0, 0.0], 3.0, [3.0 * 1.03, 0.0, 0.0]);
    let queries = vec![inside, outside];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].contains, 1, "0.97r should be inside");
    assert_eq!(out[1].contains, 0, "1.03r should be outside");
    assert_parity(&out[0], &inside, "near_inside");
    assert_parity(&out[1], &outside, "near_outside");
}

#[test]
fn large_radius_scales_slack() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereContains::new(&ctx);
    // Large radius: slack = radius*1e-5 + 1e-5. A point just inside is still
    // contained; the slack simply widens the admissible band.
    let radius = 1.0e6_f32;
    let q = BoundingSphereContainsQuery::new([0.0, 0.0, 0.0], radius, [radius * 0.5, 0.0, 0.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].contains, 1);
    assert_parity(&out[0], &q, "large_radius");
}

#[test]
fn batch_mixes_contained_and_escaped_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereContains::new(&ctx);
    let queries = vec![
        BoundingSphereContainsQuery::new([0.0, 0.0, 0.0], 2.0, [0.5, 0.5, 0.5]),
        BoundingSphereContainsQuery::new([5.0, 5.0, 5.0], 1.0, [20.0, 20.0, 20.0]),
        BoundingSphereContainsQuery::new([-3.0, 1.0, 2.0], 4.0, [-3.0, 1.0, 5.0]),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].contains, 1, "near-center inside");
    assert_eq!(out[1].contains, 0, "far point escaped");
    assert_eq!(out[2].contains, 1, "3.0 inside radius 4.0");
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereContains::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingSphereContains::new(&ctx);
    let mut lcg = Lcg::new(0xB005_0DEF);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let center = [
            lcg.next_range(-20.0, 20.0),
            lcg.next_range(-20.0, 20.0),
            lcg.next_range(-20.0, 20.0),
        ];
        let radius = lcg.next_range(0.1, 10.0);
        let point = [
            lcg.next_range(-30.0, 30.0),
            lcg.next_range(-30.0, 30.0),
            lcg.next_range(-30.0, 30.0),
        ];
        let q = BoundingSphereContainsQuery::new(center, radius, point);
        // Reject samples within a relative 1e-2 of the boundary knee so a
        // last-bit difference cannot flip the discrete contains flag.
        if boundary_margin(&q).abs() < 1.0e-2 {
            continue;
        }
        queries.push(q);
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
