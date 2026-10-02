//! Real-device parity for the circumcircle twin:
//! [`GpuTriangleCircumcircle`](prism_volumetric_gpu::triangle_circumcircle::GpuTriangleCircumcircle)
//! must reproduce the `CPU` golden
//! [`triangle_circumcircle`](prism_render_architecture::particle::triangle_circumcircle)
//! across a right triangle whose circumcenter is the hypotenuse midpoint, a
//! small acute triangle with an exact integer circumcircle, an obtuse triangle
//! whose circumcenter falls outside it, a collinear (degenerate) triangle that
//! yields no circle, and a randomized batch compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The validity flag is a discrete classification, so `CPU` and `GPU` agree on
//! it exactly and the comparison asserts `==`. The center and radius thread
//! through multiplies, adds, one guarded division and one `sqrt`, so the two
//! devices are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every continuous `f32` field.
//!
//! # Conditioning
//!
//! Every fixture is kept well away from the single degeneracy crack: the random
//! batch rejects any triangle whose doubled signed area is near the compare
//! epsilon, so `CPU` and `GPU` stay on the same side of the collapse branch
//! regardless of a few units in the last place of slack. The collinear fixture
//! uses exactly aligned corners (zero area) so both devices take the
//! invalid branch.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::triangle_circumcircle`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::triangle_circumcircle::{Triangle, Vec2};
use prism_volumetric_gpu::triangle_circumcircle::{
    golden, GpuTriangleCircumcircle, TriangleCircumcircleResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random point with each component in `[-span, span)`.
fn rand_point(state: &mut u64, span: f32) -> Vec2 {
    Vec2::new(signed(state, span), signed(state, span))
}

/// Twice the signed area of `a, b, c`, computed with integer-and-divide-free
/// `f32` arithmetic for the rejection test only.
fn doubled_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Builds a clearly-conditioned triangle by rejection sampling: three
/// well-spread corners are accepted only when the doubled signed area is
/// comfortably above the compare epsilon, so the degeneracy branch is never on a
/// tie between the two devices.
fn rand_triangle(state: &mut u64) -> Triangle {
    loop {
        let a = rand_point(state, 4.0);
        let b = rand_point(state, 4.0);
        let c = rand_point(state, 4.0);
        if doubled_area(a, b, c).abs() < 0.5 {
            continue;
        }
        return Triangle::new(a, b, c);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `triangle`: the validity
/// must match exactly, and when valid the center and radius must agree within
/// bound.
fn pin(idx: usize, triangle: &Triangle, got: &TriangleCircumcircleResult) {
    let want = golden(triangle);
    match (got.circle, want.circle) {
        (Some(g), Some(w)) => {
            assert!(
                close(g.center.x, w.center.x),
                "triangle {idx} center.x: gpu {} vs cpu {}",
                g.center.x,
                w.center.x
            );
            assert!(
                close(g.center.y, w.center.y),
                "triangle {idx} center.y: gpu {} vs cpu {}",
                g.center.y,
                w.center.y
            );
            assert!(
                close(g.radius, w.radius),
                "triangle {idx} radius: gpu {} vs cpu {}",
                g.radius,
                w.radius
            );
        }
        (None, None) => {}
        _ => panic!(
            "triangle {idx} validity mismatch: gpu {:?} vs cpu {:?}",
            got.circle, want.circle
        ),
    }
}

/// Dispatches `triangles` and pins every result against the golden.
fn check(ctx: &GpuContext, gpu: &GpuTriangleCircumcircle, triangles: &[Triangle]) {
    let got = gpu.evaluate(ctx, triangles);
    assert_eq!(
        got.len(),
        triangles.len(),
        "result count must match the input count"
    );
    for (idx, (triangle, result)) in triangles.iter().zip(got.iter()).enumerate() {
        pin(idx, triangle, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleCircumcircle::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn right_triangle_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleCircumcircle::new(&ctx);
    // A 3-4-5 right triangle: the circumcenter is the hypotenuse midpoint (2,
    // 1.5) and the radius is half the hypotenuse (2.5). Integer geometry keeps
    // both devices exact.
    let triangle = Triangle::new(
        Vec2::new(0.0, 0.0),
        Vec2::new(4.0, 0.0),
        Vec2::new(0.0, 3.0),
    );
    check(&ctx, &gpu, &[triangle]);
}

#[test]
fn acute_triangle_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleCircumcircle::new(&ctx);
    // A small acute triangle whose circumcircle is the unit circle about (1, 0),
    // an exact integer answer on both devices.
    let triangle = Triangle::new(
        Vec2::new(0.0, 0.0),
        Vec2::new(2.0, 0.0),
        Vec2::new(1.0, 1.0),
    );
    check(&ctx, &gpu, &[triangle]);
}

#[test]
fn obtuse_triangle_center_outside_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleCircumcircle::new(&ctx);
    // A flat, obtuse triangle whose circumcenter falls below the base; the
    // continuous center and radius are pinned within tolerance.
    let triangle = Triangle::new(
        Vec2::new(0.0, 0.0),
        Vec2::new(4.0, 0.0),
        Vec2::new(2.0, 0.25),
    );
    check(&ctx, &gpu, &[triangle]);
}

#[test]
fn collinear_triangle_is_invalid_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleCircumcircle::new(&ctx);
    // Three exactly collinear corners (zero doubled area), so both devices take
    // the invalid branch and report no circle.
    let triangle = Triangle::new(
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(2.0, 2.0),
    );
    let got = gpu.evaluate(&ctx, &[triangle]);
    assert_eq!(got.len(), 1, "one triangle produces one result");
    assert!(
        got[0].circle.is_none(),
        "a collinear triangle has no circumcircle"
    );
    check(&ctx, &gpu, &[triangle]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleCircumcircle::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random triangles,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut triangles = vec![
        Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(0.0, 3.0),
        ),
        Triangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(1.0, 1.0),
        ),
    ];
    for _ in 0..48 {
        triangles.push(rand_triangle(&mut state));
    }
    check(&ctx, &gpu, &triangles);
}

#[test]
fn many_triangles_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleCircumcircle::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins the circumcircle across
    // many random triangle geometries.
    let triangles: Vec<Triangle> = (0..200).map(|_| rand_triangle(&mut state)).collect();
    check(&ctx, &gpu, &triangles);
}
