//! Real-device parity for the triangle-mesh closest-point scene query: the
//! `GPU`-driven [`GpuTrimeshClosestPoint`] must agree with the `CPU` brute
//! golden [`cpu_trimesh_closest_point`] and the `CPU` `LBVH` branch-and-bound
//! query [`cpu_trimesh_closest_point_built`] over the same static triangle mesh.
//!
//! The `CPU` `BVH` query is pinned to the brute golden in its own unit suite, so
//! matching both here transitively pins the device result to the brute golden.
//! The cases mirror the `CPU` unit scenes and add device-only coverage: a point
//! projecting straight onto a face interior; a point clamping onto an edge; a
//! point clamping onto a shared vertex; a stack of quads where only the nearest
//! may win; and a larger tiled grid queried from several points so the `CPU`
//! `LBVH` prune and the device brute sweep must still land on the identical
//! triangle and surface geometry.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing.
//!
//! Provenance: Ericson Voronoi-region closest point on triangle (Ericson,
//! *Real-Time Collision Detection*, 2005). No Unreal Engine source or derived
//! code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_trimesh_closest_point, cpu_trimesh_closest_point_built, GpuContext, GpuTrimeshClosestPoint,
    Trimesh, TrimeshClosestHit,
};

/// Tolerance on the distance, point, barycentric weights, and normal: the only
/// inexact steps are the barycentric reciprocals and the normalise.
const TOL: f32 = 1e-4;

/// A unit quad in the `z = 0` plane split into two triangles sharing the
/// `(0, 0)`-to-`(1, 1)` diagonal.
fn unit_quad() -> Trimesh {
    let vertices = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let indices = vec![[0, 1, 2], [0, 2, 3]];
    Trimesh::new(vertices, indices)
}

/// A stack of `count` axis-aligned quads, each spanning `[-1, 1]^2` at an
/// increasing `z = k` plane, every quad split into a lower-right and upper-left
/// triangle.
fn quad_fan(count: u32) -> Trimesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for k in 0..count {
        let z = k as f32;
        let base = vertices.len() as u32;
        vertices.push(Vec3::new(-1.0, -1.0, z));
        vertices.push(Vec3::new(1.0, -1.0, z));
        vertices.push(Vec3::new(1.0, 1.0, z));
        vertices.push(Vec3::new(-1.0, 1.0, z));
        indices.push([base, base + 1, base + 2]);
        indices.push([base, base + 2, base + 3]);
    }
    Trimesh::new(vertices, indices)
}

/// A flat n-by-n tiled grid of unit cells in the `z = 0` plane spanning
/// `[0, n]^2`, each cell split into two triangles. The dense fan of coplanar
/// triangles exercises the `LBVH` prune against the device brute sweep.
fn tiled_grid(n: u32) -> Trimesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let base = vertices.len() as u32;
            let (x, y) = (i as f32, j as f32);
            vertices.push(Vec3::new(x, y, 0.0));
            vertices.push(Vec3::new(x + 1.0, y, 0.0));
            vertices.push(Vec3::new(x + 1.0, y + 1.0, 0.0));
            vertices.push(Vec3::new(x, y + 1.0, 0.0));
            indices.push([base, base + 1, base + 2]);
            indices.push([base, base + 2, base + 3]);
        }
    }
    Trimesh::new(vertices, indices)
}

/// Asserts two closest-point hits agree on the owning triangle exactly and on
/// distance, point, barycentric weights, and normal within tolerance.
fn assert_hit_matches(want: TrimeshClosestHit, got: TrimeshClosestHit) {
    assert_eq!(want.triangle, got.triangle, "owning triangle index differs");
    let dd = (want.distance - got.distance).abs();
    assert!(dd <= TOL, "distance diverged by {dd}");
    let dp = (want.point - got.point).length();
    assert!(dp <= TOL, "surface point diverged by {dp}");
    let db = (want.bary - got.bary).length();
    assert!(db <= TOL, "barycentric weights diverged by {db}");
    let dn = (want.normal - got.normal).length();
    assert!(dn <= TOL, "surface normal diverged by {dn}");
}

/// Runs the `CPU` brute query, the `CPU` `LBVH` query, and the device query over
/// the same mesh and point, asserting all three agree on presence and geometry.
fn run_parity(ctx: &GpuContext, gpu: &GpuTrimeshClosestPoint, mesh: &Trimesh, point: Vec3) {
    let brute = cpu_trimesh_closest_point(mesh, point);
    let bvh = cpu_trimesh_closest_point_built(mesh, point);
    assert_eq!(brute, bvh, "CPU LBVH must equal CPU brute at {point:?}");
    let device = gpu.closest(ctx, mesh, point);
    match (brute, device) {
        (None, None) => {}
        (Some(want), Some(got)) => assert_hit_matches(want, got),
        (want, got) => panic!("hit presence differs at {point:?}: cpu {want:?} gpu {got:?}"),
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn face_projection_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh closest-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshClosestPoint::new(&ctx);
    let mesh = unit_quad();
    run_parity(&ctx, &gpu, &mesh, Vec3::new(0.6, 0.2, 3.0));
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn edge_clamp_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh closest-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshClosestPoint::new(&ctx);
    let mesh = unit_quad();
    run_parity(&ctx, &gpu, &mesh, Vec3::new(2.0, 0.5, 0.0));
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn vertex_clamp_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh closest-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshClosestPoint::new(&ctx);
    let mesh = unit_quad();
    run_parity(&ctx, &gpu, &mesh, Vec3::new(3.0, 3.0, 0.0));
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn nearest_of_many_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh closest-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshClosestPoint::new(&ctx);
    let mesh = quad_fan(8);
    // Off the shared diagonal (y < x) so the owning triangle is unambiguous.
    run_parity(&ctx, &gpu, &mesh, Vec3::new(0.3, 0.1, 10.0));
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn tiled_grid_matches_cpu_golden_from_several_points() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh closest-point parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshClosestPoint::new(&ctx);
    let mesh = tiled_grid(8);
    // Several probes above distinct cells, each off the cell diagonal so the
    // owning triangle is unambiguous, plus one outside the grid that clamps.
    for &(x, y, z) in &[
        (0.3, 0.1, 2.0),
        (3.7, 2.2, 1.5),
        (6.1, 5.8, 4.0),
        (7.4, 0.6, 2.5),
        (10.0, 10.0, 0.0),
    ] {
        run_parity(&ctx, &gpu, &mesh, Vec3::new(x, y, z));
    }
}
