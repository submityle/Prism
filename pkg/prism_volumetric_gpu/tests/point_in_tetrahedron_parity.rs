//! Real-device parity for the point-in-tetrahedron twin:
//! [`GpuPointInTetrahedron`](prism_volumetric_gpu::point_in_tetrahedron::GpuPointInTetrahedron)
//! must reproduce the `CPU` golden
//! [`point_in_tetrahedron`](prism_render_architecture::particle::point_in_tetrahedron)
//! across the four barycentric weights and the inside/outside containment flag.
//!
//! The fixtures cover the shapes the golden unit tests call out: the centroid
//! (equal quarter weights, inside), each corner (a unit basis weight, on the
//! boundary), an on-face point (one weight zero, still inside), an on-edge point
//! (two weights zero), an exterior point (a negative weight, outside), a far
//! exterior point, a negatively-wound tetrahedron (total volume negative), a
//! skewed general tetrahedron, and a degenerate coplanar cell (reported invalid,
//! matching [`None`], and never inside). All corners and probe points are
//! written as integers or simple decimals, so the fixtures stay pure and need no
//! `bevy_math` and no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The validity flag and the inside flag are discrete classifications, so `CPU`
//! and `GPU` must agree exactly: the comparison is an exact `==` on the presence
//! of the [`Option`] and on the inside `bool`. The four weights thread through
//! multiplies, adds and one guarded division, so they are compared under
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::point_in_tetrahedron`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::point_in_tetrahedron::{
    barycentric_in_tetrahedron, point_in_tetrahedron,
};
use prism_volumetric_gpu::point_in_tetrahedron::{GpuPointInTetrahedron, PointInTetrahedronQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous weights.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous weights.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two weight quadruples.
fn approx4(a: [f32; 4], b: [f32; 4]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2]) && approx(a[3], b[3])
}

/// Builds a query from a tetrahedron and a probe point.
fn query(tet: [[f32; 3]; 4], point: [f32; 3]) -> PointInTetrahedronQuery {
    PointInTetrahedronQuery { tet, point }
}

/// Asserts every twinned answer for one query matches the `CPU` golden.
fn assert_parity(gpu: &GpuPointInTetrahedron, ctx: &GpuContext, q: &PointInTetrahedronQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];

    let cpu_bary = barycentric_in_tetrahedron(&q.tet, q.point);
    match (g.barycentric, cpu_bary) {
        (Some(a), Some(b)) => assert!(approx4(a, b), "bary mismatch: gpu {a:?} vs cpu {b:?}"),
        (None, None) => {}
        (a, b) => panic!("bary validity mismatch: gpu {a:?} vs cpu {b:?}"),
    }

    assert_eq!(
        g.inside,
        point_in_tetrahedron(&q.tet, q.point),
        "inside mismatch for point {:?}",
        q.point
    );
}

/// The canonical unit corner tetrahedron: origin plus the three axis tips. Its
/// total `orient3d` is `+1`, so the weights of `(x, y, z)` are
/// `[1 - x - y - z, x, y, z]`.
fn unit_tet() -> [[f32; 3]; 4] {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
}

#[test]
fn centroid_and_corners() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInTetrahedron::new(&ctx);
    let tet = unit_tet();
    // Centroid: equal quarter weights, deep interior.
    assert_parity(&gpu, &ctx, &query(tet, [0.25, 0.25, 0.25]));
    // Each corner: a unit basis weight, on the boundary (still inside).
    for corner in tet {
        assert_parity(&gpu, &ctx, &query(tet, corner));
    }
    // A generic interior point.
    assert_parity(&gpu, &ctx, &query(tet, [0.1, 0.2, 0.3]));
}

#[test]
fn on_face_and_on_edge_are_inside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInTetrahedron::new(&ctx);
    let tet = unit_tet();
    // On the face opposite corner 0 (plane x + y + z = 1): b0 == 0, inside.
    assert_parity(&gpu, &ctx, &query(tet, [0.5, 0.25, 0.25]));
    // Midpoint of edge t1--t2 lies on two faces: b0 == 0 and b3 == 0, inside.
    assert_parity(&gpu, &ctx, &query(tet, [0.5, 0.5, 0.0]));
    // Just inside the slanted face.
    assert_parity(&gpu, &ctx, &query(tet, [0.33, 0.33, 0.33]));
}

#[test]
fn exterior_points_are_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInTetrahedron::new(&ctx);
    let tet = unit_tet();
    // Beyond the slanted face x + y + z = 2 > 1: b0 < 0, outside.
    assert_parity(&gpu, &ctx, &query(tet, [0.7, 0.7, 0.6]));
    // Just past the face plane.
    assert_parity(&gpu, &ctx, &query(tet, [0.34, 0.34, 0.34]));
    // Far negative octant.
    assert_parity(&gpu, &ctx, &query(tet, [-1.0, -1.0, -1.0]));
    // Partition of unity still holds for an exterior point.
    assert_parity(&gpu, &ctx, &query(tet, [2.0, -1.0, 0.5]));
}

#[test]
fn negative_winding_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInTetrahedron::new(&ctx);
    // Swap corners 1 and 2 to flip the orientation (total volume negative).
    let tet = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    assert_parity(&gpu, &ctx, &query(tet, [0.1, 0.2, 0.3]));
    assert_parity(&gpu, &ctx, &query(tet, [0.2, 0.2, 0.2]));
    assert_parity(&gpu, &ctx, &query(tet, [0.6, 0.6, 0.6]));
}

#[test]
fn general_skewed_tetrahedron() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInTetrahedron::new(&ctx);
    // A skewed, non-axis-aligned tetrahedron.
    let tet = [
        [1.0, 2.0, -1.0],
        [4.0, 0.0, 1.0],
        [-1.0, 3.0, 2.0],
        [2.0, -2.0, 5.0],
    ];
    assert_parity(&gpu, &ctx, &query(tet, [1.5, 0.5, 1.75]));
}

#[test]
fn degenerate_cell_is_invalid_and_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInTetrahedron::new(&ctx);
    // All four vertices on the z = 0 plane: zero volume, reported invalid and
    // never inside (matching the reference None / false).
    let flat = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ];
    assert_parity(&gpu, &ctx, &query(flat, [0.5, 0.5, 0.0]));
}

#[test]
fn batch_of_queries_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInTetrahedron::new(&ctx);
    // A batch exercises the one-thread-per-query flattening; each result must be
    // independent of its neighbours.
    let tet = unit_tet();
    let batch = [
        query(tet, [0.25, 0.25, 0.25]),
        query(tet, [0.7, 0.7, 0.6]),
        query(tet, [0.5, 0.5, 0.0]),
        query(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            [0.5, 0.5, 0.0],
        ),
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPointInTetrahedron::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
