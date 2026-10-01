//! Real-device parity for the isolated planar-projection barycentric solve:
//! [`GpuHairComputeBarycentric`] must reproduce the `CPU` golden
//! [`reference_compute_barycentric`](prism_hair_gpu::compute_barycentric::reference_compute_barycentric)
//! (which forwards
//! [`compute_barycentric`](prism_render_architecture::hair::follicle_bind::compute_barycentric))
//! for a batch of query points broadcast against one shared triangle, mapping
//! each point to its raw barycentric weights independently and in order. The
//! suite drives vertex recovery (a pure point on each vertex returns a unit
//! weight), an interior point (the centroid returns 1/3 each), an exterior point
//! (the raw weights go negative and are *not* clamped), a degenerate collinear
//! triangle and a degenerate collapsed triangle (both fall back to the centroid
//! for the whole batch), a non-planar triangle whose points are projected onto
//! its plane, the empty no-op, and a large multi-workgroup batch that crosses
//! the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each component is a dot-product / determinant chain plus a single reciprocal
//! and multiply-adds a `GPU` may fuse, so every value is asserted within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3` rather than bit-for-bit. Every output
//! component is also asserted finite. All inputs are explicit literals or
//! integer-derived fractions and are kept finite so the `CPU` and `GPU` walk the
//! identical branch of the degeneracy guard; no `sin`/`cos` appears anywhere.
//! The suite deliberately keeps query points near the triangle so exterior
//! weights stay modest and the raw (un-clamped) sign is easy to read.
//!
//! Provenance: Ericson "Real-Time Collision Detection" barycentric solve plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::compute_barycentric::{
    reference_compute_barycentric, GpuHairComputeBarycentric,
};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::follicle_bind::{Barycentric, TriangleFrame};

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// Asserts two scalars agree within the documented fma tolerance.
fn assert_close(got: f32, want: f32, what: &str) {
    let abs = (got - want).abs();
    let rel = abs / want.abs().max(1.0);
    assert!(
        abs < 1e-4 || rel < 1e-3,
        "{what}: got {got}, want {want} (abs {abs}, rel {rel})"
    );
}

/// Asserts a whole batch of barycentric weights matches the `CPU` golden element
/// by element, component by component, and that every component is finite.
fn assert_batch(got: &[Barycentric], points: &[[f32; 3]], tri: TriangleFrame) {
    assert_eq!(got.len(), points.len(), "one weight set per query point");
    for (i, (out, &point)) in got.iter().zip(points.iter()).enumerate() {
        let reference = reference_compute_barycentric(point, tri);
        assert_close(out.u, reference.u, &format!("point {i} u"));
        assert_close(out.v, reference.v, &format!("point {i} v"));
        assert_close(out.w, reference.w, &format!("point {i} w"));
        assert!(
            out.u.is_finite(),
            "point {i} u must be finite, got {}",
            out.u
        );
        assert!(
            out.v.is_finite(),
            "point {i} v must be finite, got {}",
            out.v
        );
        assert!(
            out.w.is_finite(),
            "point {i} w must be finite, got {}",
            out.w
        );
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, points: &[[f32; 3]], tri: TriangleFrame) -> Vec<Barycentric> {
    GpuHairComputeBarycentric::new(ctx).eval(ctx, points, tri)
}

/// A flat right-triangle in the z = 0 plane.
fn flat_tri() -> TriangleFrame {
    TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: [[0.0, 0.0, 1.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    }
}

#[test]
fn vertices_recover_unit_weights() {
    let Some(ctx) = context_or_skip("vertices_recover_unit_weights") else {
        return;
    };
    let tri = flat_tri();
    // A query sitting exactly on vertex k has a unit weight on k, zero elsewhere.
    let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let got = run(&ctx, &points, tri);
    assert_batch(&got, &points, tri);
    assert_close(got[0].u, 1.0, "vtx0 u");
    assert_close(got[0].v, 0.0, "vtx0 v");
    assert_close(got[0].w, 0.0, "vtx0 w");
    assert_close(got[1].v, 1.0, "vtx1 v");
    assert_close(got[2].w, 1.0, "vtx2 w");
}

#[test]
fn interior_centroid_is_thirds() {
    let Some(ctx) = context_or_skip("interior_centroid_is_thirds") else {
        return;
    };
    let tri = flat_tri();
    // The geometric centroid of the flat triangle has weights (1/3, 1/3, 1/3).
    let third = 1.0 / 3.0;
    let points = [[third, third, 0.0], [0.25, 0.25, 0.0]];
    let got = run(&ctx, &points, tri);
    assert_batch(&got, &points, tri);
    assert_close(got[0].u, third, "centroid u");
    assert_close(got[0].v, third, "centroid v");
    assert_close(got[0].w, third, "centroid w");
    // (0.25, 0.25) -> v = 0.25, w = 0.25, u = 0.5.
    assert_close(got[1].u, 0.5, "mid u");
    assert_close(got[1].v, 0.25, "mid v");
    assert_close(got[1].w, 0.25, "mid w");
}

#[test]
fn exterior_point_keeps_raw_negative_weights() {
    let Some(ctx) = context_or_skip("exterior_point_keeps_raw_negative_weights") else {
        return;
    };
    let tri = flat_tri();
    // A point outside the triangle produces a raw negative weight; the solve must
    // return it un-clamped (sanitisation happens later, elsewhere).
    let points = [[-0.5, -0.5, 0.0], [0.75, 0.75, 0.0]];
    let got = run(&ctx, &points, tri);
    assert_batch(&got, &points, tri);
    // (-0.5, -0.5): v = -0.5, w = -0.5, u = 2.0 — raw, out of [0, 1].
    assert!(
        got[0].v < 0.0,
        "exterior v stays negative, got {}",
        got[0].v
    );
    assert!(
        got[0].w < 0.0,
        "exterior w stays negative, got {}",
        got[0].w
    );
    assert_close(got[0].u, 2.0, "exterior u");
    // (0.75, 0.75): v = 0.75, w = 0.75, u = -0.5 — the vertex-0 weight goes
    // negative past the hypotenuse.
    assert!(
        got[1].u < 0.0,
        "beyond-hypotenuse u negative, got {}",
        got[1].u
    );
}

#[test]
fn collinear_triangle_falls_back_to_centroid() {
    let Some(ctx) = context_or_skip("collinear_triangle_falls_back_to_centroid") else {
        return;
    };
    // Three collinear vertices: zero area, degenerate determinant -> centroid.
    let tri = TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
        normals: [[0.0, 0.0, 1.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let points = [[0.5, 0.5, 0.0], [3.0, -1.0, 0.0]];
    let got = run(&ctx, &points, tri);
    assert_batch(&got, &points, tri);
    let third = 1.0 / 3.0;
    for (i, b) in got.iter().enumerate() {
        assert_close(b.u, third, &format!("collinear {i} u"));
        assert_close(b.v, third, &format!("collinear {i} v"));
        assert_close(b.w, third, &format!("collinear {i} w"));
    }
}

#[test]
fn collapsed_triangle_falls_back_to_centroid() {
    let Some(ctx) = context_or_skip("collapsed_triangle_falls_back_to_centroid") else {
        return;
    };
    // All three vertices coincident: zero area -> centroid for every query.
    let tri = TriangleFrame {
        positions: [[1.0, 2.0, 3.0]; 3],
        normals: [[0.0, 0.0, 1.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let points = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]];
    let got = run(&ctx, &points, tri);
    assert_batch(&got, &points, tri);
    let third = 1.0 / 3.0;
    assert_close(got[0].u, third, "collapsed u");
    assert_close(got[1].w, third, "collapsed w");
}

#[test]
fn non_planar_triangle_projects_onto_plane() {
    let Some(ctx) = context_or_skip("non_planar_triangle_projects_onto_plane") else {
        return;
    };
    // A triangle lifted out of any axis plane with arbitrary query points; the
    // solve projects each point onto the triangle's plane. Correctness is
    // delegated to the golden via assert_batch; points are kept near the
    // triangle so weights stay modest.
    let tri = TriangleFrame {
        positions: [[1.0, -2.0, 0.5], [-1.5, 0.0, 2.0], [0.25, 3.0, -1.0]],
        normals: [[0.2, 1.0, 0.3], [-0.4, 0.8, 0.1], [0.1, 0.9, -0.5]],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let points = [
        [0.0, 0.0, 0.0],
        [-0.5, 1.0, 1.0],
        [0.5, -0.5, 0.25],
        [-1.0, 2.0, -0.5],
    ];
    assert_batch(&run(&ctx, &points, tri), &points, tri);
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let got = run(&ctx, &[], flat_tri());
    assert!(got.is_empty(), "empty batch yields no weights");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 points span four 64-wide workgroups over a deterministic sweep of
    // integer-derived in-plane coordinates (a few land outside the triangle to
    // keep raw negative weights in the mix across the boundary).
    let tri = TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [4.0, 1.0, 0.0], [1.0, 4.0, 2.0]],
        normals: [[0.0, 0.1, 1.0], [0.1, 0.0, 1.0], [-0.1, -0.1, 1.0]],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let mut points = Vec::new();
    for k in 0u32..200 {
        let x = (k % 7) as f32 / 2.0 - 1.0;
        let y = (k % 5) as f32 / 2.0 - 1.0;
        let z = (k % 3) as f32 / 4.0;
        points.push([x, y, z]);
    }
    assert_batch(&run(&ctx, &points, tri), &points, tri);
}
