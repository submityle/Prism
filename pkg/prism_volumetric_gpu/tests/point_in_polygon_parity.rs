//! Real-device parity for the 2D point-in-polygon twin:
//! [`GpuPointInPolygon`](prism_volumetric_gpu::point_in_polygon::GpuPointInPolygon)
//! must reproduce the `CPU` golden
//! [`point_in_polygon`](prism_render_architecture::particle::point_in_polygon)
//! across the even-odd crossing flag, the signed winding number, the non-zero
//! winding containment and the signed boundary distance.
//!
//! The fixtures cover the shapes the golden unit tests call out: a
//! counter-clockwise and a clockwise convex square (winding `+1` versus `-1`), a
//! triangle, a concave arrow whose reflex notch separates inside from outside, a
//! self-intersecting five-pointed star whose double-wrapped core is outside
//! under the even-odd rule yet inside under the winding rule, and the degenerate
//! single-edge and empty polygons. Query points are kept clear of edges and
//! vertices for the exact containment and winding comparison, with a few
//! near-boundary probes exercising the signed distance under tolerance. All
//! polygon vertices and probe points are written as integers or simple decimals,
//! so the fixtures stay pure and need no `bevy_math` and no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The crossing flag and the winding integer are discrete classifications, so
//! `CPU` and `GPU` must agree exactly: the comparison is an exact `==` for both,
//! and for the derived non-zero winding containment. The signed distance threads
//! through a `sqrt` and a division, so it is compared under tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`); an infinite
//! reference (an edgeless polygon) must meet an infinity of the same sign.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::point_in_polygon`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::point_in_polygon::{
    point_in_polygon_crossing, point_in_polygon_winding, signed_distance_to_polygon, winding_number,
};
use prism_volumetric_gpu::point_in_polygon::GpuPointInPolygon;
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the signed distance.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the signed distance.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Signed-distance comparison: absolute or relative tolerance for finite values,
/// and a same-signed infinity match for an edgeless polygon.
fn approx(a: f32, b: f32) -> bool {
    if a.is_infinite() || b.is_infinite() {
        return a.is_infinite()
            && b.is_infinite()
            && (a.is_sign_negative() == b.is_sign_negative());
    }
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Asserts the containment flag, winding number, non-zero winding containment
/// and signed distance all match the `CPU` golden for one `(polygon, points)`
/// fixture.
fn assert_parity(
    gpu: &GpuPointInPolygon,
    ctx: &GpuContext,
    polygon: &[[f32; 2]],
    points: &[[f32; 2]],
) {
    let got = gpu.evaluate(ctx, polygon, points);
    assert_eq!(got.len(), points.len(), "one result per query point");
    for (i, &p) in points.iter().enumerate() {
        let g = got[i];
        assert_eq!(
            g.inside_crossing,
            point_in_polygon_crossing(p, polygon),
            "crossing mismatch at {p:?} for polygon {polygon:?}"
        );
        assert_eq!(
            g.winding,
            winding_number(p, polygon),
            "winding mismatch at {p:?} for polygon {polygon:?}"
        );
        assert_eq!(
            g.inside_winding(),
            point_in_polygon_winding(p, polygon),
            "winding containment mismatch at {p:?} for polygon {polygon:?}"
        );
        let cpu_sd = signed_distance_to_polygon(p, polygon);
        assert!(
            approx(g.signed_distance, cpu_sd),
            "signed distance mismatch at {p:?}: gpu {} vs cpu {cpu_sd}",
            g.signed_distance
        );
    }
}

/// A counter-clockwise axis-aligned square spanning `[0, 2]^2`; winding `+1`.
fn ccw_square() -> [[f32; 2]; 4] {
    [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
}

/// The same square wound clockwise; winding `-1` inside.
fn cw_square() -> [[f32; 2]; 4] {
    [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]]
}

/// A right triangle with legs on the axes.
fn triangle() -> [[f32; 2]; 3] {
    [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]
}

/// A concave arrow: the reflex notch at the right carves a wedge out of the
/// otherwise convex outline, so points in the notch are outside.
fn concave_arrow() -> [[f32; 2]; 5] {
    [[0.0, 0.0], [4.0, 2.0], [0.0, 4.0], [1.0, 2.0], [0.0, 2.0]]
}

/// A five-pointed star whose self-intersecting core distinguishes the even-odd
/// rule from the non-zero winding rule, matching the golden `star` fixture.
fn star() -> [[f32; 2]; 5] {
    [
        [0.0, 3.0],
        [2.0, -3.0],
        [-3.0, 1.0],
        [3.0, 1.0],
        [-2.0, -3.0],
    ]
}

#[test]
fn ccw_square_inside_outside_and_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInPolygon::new(&ctx);
    let poly = ccw_square();
    // Interior centre, interior off-centre, and exterior probes on every side,
    // plus near-boundary points for the signed distance.
    let points = [
        [1.0, 1.0],
        [0.5, 1.5],
        [1.5, 0.5],
        [-1.0, 1.0],
        [3.0, 1.0],
        [1.0, -1.0],
        [1.0, 3.0],
        [-0.5, 1.0],
        [2.5, 1.0],
    ];
    assert_parity(&gpu, &ctx, &poly, &points);
}

#[test]
fn cw_square_winding_is_negative_inside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInPolygon::new(&ctx);
    // The clockwise ring winds -1 inside and 0 outside; the crossing rule is
    // orientation-insensitive and still reports inside.
    let poly = cw_square();
    let points = [[1.0, 1.0], [0.5, 0.5], [5.0, 5.0], [-1.0, 1.0]];
    assert_parity(&gpu, &ctx, &poly, &points);
}

#[test]
fn triangle_inside_outside_and_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInPolygon::new(&ctx);
    let poly = triangle();
    let points = [
        [1.0, 1.0],
        [0.5, 0.5],
        [3.0, 3.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [2.0, 1.5],
    ];
    assert_parity(&gpu, &ctx, &poly, &points);
}

#[test]
fn concave_arrow_notch_is_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInPolygon::new(&ctx);
    // Points inside the body versus inside the reflex notch (outside the ring),
    // the case where a concave outline matters.
    let poly = concave_arrow();
    let points = [
        [1.0, 2.0],
        [1.5, 1.0],
        [1.5, 3.0],
        [2.0, 2.0],
        [-1.0, 2.0],
        [5.0, 2.0],
    ];
    assert_parity(&gpu, &ctx, &poly, &points);
}

#[test]
fn star_core_differs_between_rules() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInPolygon::new(&ctx);
    // The central overlap is wrapped twice: winding calls it inside, the
    // even-odd rule outside. Points on the arms and well outside anchor the rest.
    let poly = star();
    let points = [
        [0.0, 0.0],
        [0.0, 2.0],
        [0.0, -2.0],
        [1.5, 0.5],
        [10.0, 10.0],
        [-10.0, 0.0],
    ];
    assert_parity(&gpu, &ctx, &poly, &points);
}

#[test]
fn degenerate_single_edge_is_infinite_or_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInPolygon::new(&ctx);
    // A two-vertex "polygon" has one edge (two reversed copies): never inside,
    // winding 0, and a finite positive distance to that segment.
    let poly = [[0.0_f32, 0.0_f32], [2.0_f32, 0.0_f32]];
    let points = [[1.0, 1.0], [-1.0, 0.0], [3.0, 0.0]];
    assert_parity(&gpu, &ctx, &poly, &points);
    // A single-vertex polygon has no edge: the signed distance is +inf.
    let point_poly = [[1.0_f32, 1.0_f32]];
    assert_parity(&gpu, &ctx, &point_poly, &points);
}

#[test]
fn empty_polygon_reports_outside_and_infinite() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInPolygon::new(&ctx);
    // Empty polygon: uploaded as a placeholder the kernel never reads. Every
    // point is outside, winding 0, infinite signed distance.
    let points = [[0.0, 0.0], [1.0, 1.0], [-5.0, 3.0]];
    assert_parity(&gpu, &ctx, &[], &points);
}

#[test]
fn empty_query_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInPolygon::new(&ctx);
    // No dispatch is issued and every entry point returns an empty result.
    let poly = ccw_square();
    assert!(gpu.evaluate(&ctx, &poly, &[]).is_empty());
    assert!(gpu.crossing(&ctx, &poly, &[]).is_empty());
    assert!(gpu.winding(&ctx, &poly, &[]).is_empty());
    assert!(gpu.winding_inside(&ctx, &poly, &[]).is_empty());
    assert!(gpu.signed_distance(&ctx, &poly, &[]).is_empty());
}
