//! Real-device parity for the triangle-mesh ray cast scene query: the
//! `GPU`-driven [`GpuTrimeshRayCast`] must agree with the `CPU` brute golden
//! [`cpu_trimesh_raycast`] and the `CPU` `LBVH` branch-and-bound query
//! [`cpu_trimesh_raycast_built`] over the same static triangle mesh.
//!
//! The `CPU` `BVH` query is pinned to the brute golden in its own unit suite, so
//! matching both here transitively pins the device result to the brute golden.
//! The cases mirror the `CPU` unit scenes and add device-only coverage: a single
//! quad struck on its front face; the same quad struck on its back face so the
//! reported normal flips to face the origin; a fan of parallel quads where only
//! the nearest may win; an oblique ray that still resolves the correct face and
//! barycentric weights; a miss that must report no hit on every path; and a
//! larger tiled grid queried from several directions so the `CPU` `LBVH` prune
//! and the device brute sweep must still land on the identical triangle and
//! surface geometry.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing.
//!
//! Provenance: ray-vs-triangle intersection via the Moller-Trumbore algorithm
//! (Moller and Trumbore, 1997). No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_trimesh_raycast, cpu_trimesh_raycast_built, GpuContext, GpuTrimeshRayCast, MeshRay, Trimesh,
    TrimeshRayHit,
};

/// Tolerance on the hit distance, point, barycentric weights, and normal: the
/// only inexact steps are the Moller-Trumbore reciprocal and the normalise.
const TOL: f32 = 1e-4;

/// Builds a single unit quad in the `z = 0` plane, split into two triangles that
/// share the `(0, 0)`-to-`(1, 1)` diagonal, winding counter-clockwise when seen
/// from `+z` so the geometric normal points toward `+z`.
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

/// Builds a stack of `count` axis-aligned quads, each spanning `[-1, 1]^2` at an
/// increasing `z = k` plane, every quad split into a lower-right and upper-left
/// triangle. A `+z` ray travelling `-z` must resolve the nearest quad only.
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

/// Builds a flat n-by-n tiled grid of unit cells in the `z = 0` plane spanning
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

/// Asserts two mesh ray hits agree on the struck triangle and front-face flag
/// exactly, and on distance, point, barycentric weights, and normal within
/// tolerance.
fn assert_hit_matches(want: TrimeshRayHit, got: TrimeshRayHit) {
    assert_eq!(want.triangle, got.triangle, "struck triangle index differs");
    assert_eq!(want.front_face, got.front_face, "front-face flag differs");
    let dd = (want.distance - got.distance).abs();
    assert!(dd <= TOL, "distance diverged by {dd}");
    let dp = (want.point - got.point).length();
    assert!(dp <= TOL, "hit point diverged by {dp}");
    let db = (want.bary - got.bary).length();
    assert!(db <= TOL, "barycentric weights diverged by {db}");
    let dn = (want.normal - got.normal).length();
    assert!(dn <= TOL, "surface normal diverged by {dn}");
}

/// Runs the `CPU` brute query, the `CPU` `LBVH` query, and the device query over
/// the same mesh and ray, asserting all three agree on presence and geometry.
fn run_parity(ctx: &GpuContext, gpu: &GpuTrimeshRayCast, mesh: &Trimesh, ray: &MeshRay) {
    let brute = cpu_trimesh_raycast(mesh, ray);
    let bvh = cpu_trimesh_raycast_built(mesh, ray);
    assert_eq!(brute, bvh, "CPU LBVH must equal CPU brute for {ray:?}");
    let device = gpu.raycast(ctx, mesh, ray);
    match (brute, device) {
        (None, None) => {}
        (Some(want), Some(got)) => assert_hit_matches(want, got),
        (want, got) => panic!("hit presence differs for {ray:?}: cpu {want:?} gpu {got:?}"),
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn front_face_hit_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh raycast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshRayCast::new(&ctx);
    let mesh = unit_quad();
    let ray = MeshRay::new(Vec3::new(0.25, 0.25, 5.0), Vec3::new(0.0, 0.0, -1.0), 100.0);
    run_parity(&ctx, &gpu, &mesh, &ray);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn back_face_hit_flips_normal_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh raycast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshRayCast::new(&ctx);
    let mesh = unit_quad();
    let ray = MeshRay::new(Vec3::new(0.25, 0.25, -5.0), Vec3::new(0.0, 0.0, 1.0), 100.0);
    run_parity(&ctx, &gpu, &mesh, &ray);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn nearest_of_many_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh raycast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshRayCast::new(&ctx);
    let mesh = quad_fan(8);
    // Off the shared diagonal (y < x) so the winning triangle is unambiguous.
    let ray = MeshRay::new(Vec3::new(0.3, 0.1, 20.0), Vec3::new(0.0, 0.0, -1.0), 100.0);
    run_parity(&ctx, &gpu, &mesh, &ray);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn oblique_ray_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh raycast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshRayCast::new(&ctx);
    let mesh = unit_quad();
    // A slanted descent toward the quad interior from the +z side.
    let dir = Vec3::new(0.2, 0.1, -1.0).normalize();
    let ray = MeshRay::new(Vec3::new(0.1, 0.2, 4.0), dir, 100.0);
    run_parity(&ctx, &gpu, &mesh, &ray);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn miss_reports_no_hit_on_every_path() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh raycast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshRayCast::new(&ctx);
    let mesh = unit_quad();
    // Aimed away from the quad: no path may report a hit.
    let ray = MeshRay::new(Vec3::new(0.25, 0.25, 5.0), Vec3::new(0.0, 0.0, 1.0), 100.0);
    run_parity(&ctx, &gpu, &mesh, &ray);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn tiled_grid_matches_cpu_golden_from_several_points() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh raycast parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshRayCast::new(&ctx);
    let mesh = tiled_grid(8);
    // Several straight-down probes landing inside distinct cells, each off the
    // cell diagonal so the struck triangle is unambiguous.
    for &(x, y) in &[(0.3, 0.1), (3.7, 2.2), (6.1, 5.8), (7.4, 0.6), (1.2, 7.3)] {
        let ray = MeshRay::new(Vec3::new(x, y, 9.0), Vec3::new(0.0, 0.0, -1.0), 100.0);
        run_parity(&ctx, &gpu, &mesh, &ray);
    }
}
