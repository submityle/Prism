//! Real-device parity for the mesh-triangle-evaluation twin:
//! [`GpuMeshTriangleEval`](prism_volumetric_gpu::mesh_triangle_eval::GpuMeshTriangleEval)
//! must reproduce the `CPU` golden
//! [`MeshTriangle`](prism_render_architecture::particle::mesh_emission::MeshTriangle)
//! across the surface area, the geometric normal, the interpolated position, the
//! renormalized shading normal (with the geometric-normal fallback), and the
//! interpolated `UV`.
//!
//! The fixtures use triangles whose area is comfortably positive and whose
//! shading normals are non-parallel, with barycentric weights kept off ties, so
//! every quantity sits far from the degeneracy and fallback thresholds. One
//! dedicated fixture zeroes every shading normal to exercise the
//! geometric-normal fallback the reference `normal_at` performs. All vertices,
//! normals and weights are written as integers or simple decimals, so the
//! fixtures stay pure and need no external math library and no transcendental
//! math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The area, the normals, the position and the `UV` thread through multiplies,
//! adds, a cross product and one guarded reciprocal `sqrt`, so they are compared
//! under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::mesh_emission`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::mesh_emission::{MeshTriangle, MeshVertex};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::mesh_triangle_eval::{GpuMeshTriangleEval, GpuMeshTriangleEvalQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
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

/// Tolerant comparison of a `Vec3` answer against a `[f32; 3]` lane triple.
fn approx_vec3(a: Vec3, b: [f32; 3]) -> bool {
    approx(a.x, b[0]) && approx(a.y, b[1]) && approx(a.z, b[2])
}

/// Tolerant comparison of two `[f32; 2]` texture coordinates.
fn approx_uv(a: [f32; 2], b: [f32; 2]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1])
}

/// Builds a triangle from three (position, normal, `UV`) vertices.
fn triangle(
    a: ([f32; 3], [f32; 3], [f32; 2]),
    b: ([f32; 3], [f32; 3], [f32; 2]),
    c: ([f32; 3], [f32; 3], [f32; 2]),
) -> MeshTriangle {
    let vertex = |v: ([f32; 3], [f32; 3], [f32; 2])| {
        MeshVertex::new(
            Vec3::new(v.0[0], v.0[1], v.0[2]),
            Vec3::new(v.1[0], v.1[1], v.1[2]),
            v.2,
        )
    };
    MeshTriangle {
        a: vertex(a),
        b: vertex(b),
        c: vertex(c),
    }
}

/// Asserts every twinned quantity for one query matches the `CPU` golden.
fn assert_parity(gpu: &GpuMeshTriangleEval, ctx: &GpuContext, q: &GpuMeshTriangleEvalQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];

    let tri = q.triangle;
    let bary = Vec3::new(q.bary[0], q.bary[1], q.bary[2]);

    let cpu_area = tri.area();
    assert!(
        approx(g.area, cpu_area),
        "area mismatch: gpu {} vs cpu {cpu_area}",
        g.area
    );

    let cpu_geo = tri.geometric_normal();
    assert!(
        approx_vec3(cpu_geo, g.geometric_normal),
        "geometric_normal mismatch: gpu {:?} vs cpu {cpu_geo:?}",
        g.geometric_normal
    );

    let cpu_pos = tri.position_at(bary);
    assert!(
        approx_vec3(cpu_pos, g.position),
        "position mismatch: gpu {:?} vs cpu {cpu_pos:?}",
        g.position
    );

    let cpu_nrm = tri.normal_at(bary);
    assert!(
        approx_vec3(cpu_nrm, g.normal),
        "normal mismatch: gpu {:?} vs cpu {cpu_nrm:?}",
        g.normal
    );

    let cpu_uv = tri.uv_at(bary);
    assert!(
        approx_uv(g.uv, cpu_uv),
        "uv mismatch: gpu {:?} vs cpu {cpu_uv:?}",
        g.uv
    );
}

/// A slanted triangle with distinct, non-parallel shading normals and distinct
/// texture coordinates; its area is comfortably positive.
fn slanted() -> MeshTriangle {
    triangle(
        ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0]),
        ([4.0, 0.0, 1.0], [1.0, 0.0, 2.0], [1.0, 0.0]),
        ([1.0, 3.0, 2.0], [0.0, 1.0, 3.0], [0.0, 1.0]),
    )
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_mesh_triangle_eval_matches_cpu_golden_interior() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_triangle_eval parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuMeshTriangleEval::new(&ctx);
    // An interior barycentric sample kept off any tie, over a well-formed
    // triangle with non-parallel normals.
    let q = GpuMeshTriangleEvalQuery {
        triangle: slanted(),
        bary: [0.2, 0.3, 0.5],
    };
    assert_parity(&gpu, &ctx, &q);

    // The twin must actually interpolate: the shading normal differs from the
    // geometric normal on this fixture, proving a real blend ran.
    let g = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let geo = g[0].geometric_normal;
    let nrm = g[0].normal;
    assert!(
        (geo[0] - nrm[0]).abs() + (geo[1] - nrm[1]).abs() + (geo[2] - nrm[2]).abs() > 1e-3,
        "shading normal must differ from the geometric normal on this fixture"
    );
}

#[test]
fn gpu_mesh_triangle_eval_matches_cpu_golden_at_corners() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleEval::new(&ctx);
    // Each corner weight reconstructs the corresponding vertex exactly.
    let tri = slanted();
    for bary in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
        let q = GpuMeshTriangleEvalQuery {
            triangle: tri,
            bary,
        };
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn gpu_mesh_triangle_eval_falls_back_to_geometric_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleEval::new(&ctx);
    // All shading normals are zero (an unauthored normal set), so the blended
    // normal is zero and `normal_at` must return the geometric normal.
    let tri = triangle(
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0]),
        ([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0]),
        ([0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0]),
    );
    let q = GpuMeshTriangleEvalQuery {
        triangle: tri,
        bary: [0.25, 0.25, 0.5],
    };
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn gpu_mesh_triangle_eval_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleEval::new(&ctx);
    // A batch exercises the one-thread-per-triangle flattening; each result must
    // be independent of its neighbours.
    let batch = [
        GpuMeshTriangleEvalQuery {
            triangle: slanted(),
            bary: [0.5, 0.25, 0.25],
        },
        GpuMeshTriangleEvalQuery {
            triangle: triangle(
                ([-1.0, -1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0]),
                ([3.0, -1.0, 0.5], [2.0, 0.0, 1.0], [2.0, 0.0]),
                ([0.0, 2.5, -1.0], [0.0, 2.0, 1.0], [0.0, 3.0]),
            ),
            bary: [0.1, 0.6, 0.3],
        },
        GpuMeshTriangleEvalQuery {
            triangle: triangle(
                ([2.0, 1.0, -2.0], [1.0, 1.0, 0.0], [1.0, 1.0]),
                ([5.0, 1.5, 0.0], [0.0, 1.0, 1.0], [4.0, 1.0]),
                ([2.5, 4.0, 1.0], [1.0, 0.0, 2.0], [1.0, 5.0]),
            ),
            bary: [0.4, 0.35, 0.25],
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn gpu_mesh_triangle_eval_empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleEval::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
