//! Real-device parity for the triangle-mesh swept-capsule scene query: the
//! `GPU`-driven [`GpuTrimeshCapsuleSweep`] must agree with the `CPU` brute golden
//! [`cpu_trimesh_capsule_sweep`] and the `CPU` `LBVH` branch-and-bound query
//! [`cpu_trimesh_capsule_sweep_built`] over the same static triangle mesh and
//! moving capsule.
//!
//! The `CPU` `BVH` query is pinned to the brute golden in its own unit suite, so
//! matching both here transitively pins the device result to the brute golden.
//! The cases mirror the `CPU` unit scenes and add device coverage: a capsule
//! descending onto a quad's front face; an in-plane sweep that misses the face
//! and must resolve on an edge; a capsule already overlapping the quad so the
//! contact is immediate; a sweep aimed away that must miss on every path; a fan
//! of parallel quads where only the nearest may win; and a larger tiled grid
//! probed from several points so the `CPU` `LBVH` prune and the device brute
//! sweep must still land on the identical triangle and surface geometry. A
//! horizontal capsule descending across the quad diagonal also exercises a
//! non-axis-aligned segment orientation.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing.
//!
//! Provenance: conservative advancement after Mirtich (2000) and van den Bergen
//! (2004); `GJK` distance per Gilbert-Johnson-Keerthi (1988) with Ericson's
//! Voronoi sub-distance (2005). No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_trimesh_capsule_sweep, cpu_trimesh_capsule_sweep_built, CapsuleSweep, CapsuleSweepHit,
    GpuContext, GpuTrimeshCapsuleSweep, Trimesh,
};

/// Tolerance on the time of impact, contact point, and normal: the inexact
/// steps are the square roots and reciprocals in the `GJK` and advance solve.
const TOL: f32 = 1e-3;

/// A unit quad in the `z = 0` plane spanning `[0, 1]^2`, two CCW triangles seen
/// from `+z` sharing the `(0, 0)`-to-`(1, 1)` diagonal, so the geometric normal
/// points toward `+z`.
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
/// triangle. A `+z` capsule sweeping `-z` must resolve the nearest quad only.
fn quad_fan(count: u32) -> Trimesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for k in 0..count {
        let z = k as f32;
        let base = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
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
            let base = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
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

/// Asserts two mesh sweep hits agree on the struck triangle exactly and on time
/// of impact, contact point, and push-out normal within tolerance.
fn assert_hit_matches(want: CapsuleSweepHit, got: CapsuleSweepHit) {
    assert_eq!(want.triangle, got.triangle, "struck triangle index differs");
    let dt = (want.toi - got.toi).abs();
    assert!(dt <= TOL, "time of impact diverged by {dt}");
    let dp = (want.point - got.point).length();
    assert!(dp <= TOL, "contact point diverged by {dp}");
    let dn = (want.normal - got.normal).length();
    assert!(dn <= TOL, "push-out normal diverged by {dn}");
}

/// Runs the `CPU` brute query, the `CPU` `LBVH` query, and the device query over
/// the same mesh and sweep, asserting all three agree on presence and geometry.
fn run_parity(ctx: &GpuContext, gpu: &GpuTrimeshCapsuleSweep, mesh: &Trimesh, sweep: &CapsuleSweep) {
    let brute = cpu_trimesh_capsule_sweep(mesh, sweep);
    let bvh = cpu_trimesh_capsule_sweep_built(mesh, sweep);
    assert_eq!(brute, bvh, "CPU LBVH must equal CPU brute for {sweep:?}");
    let device = gpu.sweep(ctx, mesh, sweep);
    match (brute, device) {
        (None, None) => {}
        (Some(want), Some(got)) => assert_hit_matches(want, got),
        (want, got) => panic!("hit presence differs for {sweep:?}: cpu {want:?} gpu {got:?}"),
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn front_face_sweep_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh capsule sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshCapsuleSweep::new(&ctx);
    let mesh = unit_quad();
    // A short vertical capsule descending from +z onto the quad's face; it stops
    // one radius above the plane.
    let sweep = CapsuleSweep::new(
        Vec3::new(0.3, 0.4, 5.0),
        Vec3::new(0.3, 0.4, 5.6),
        Vec3::new(0.0, 0.0, -1.0),
        100.0,
        0.5,
    );
    run_parity(&ctx, &gpu, &mesh, &sweep);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn edge_sweep_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh capsule sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshCapsuleSweep::new(&ctx);
    let mesh = unit_quad();
    // In-plane vertical capsule sweeping toward the right edge x = 1: the face is
    // parallel to the motion, so the right edge must catch the capsule.
    let sweep = CapsuleSweep::new(
        Vec3::new(1.5, 0.5, -0.2),
        Vec3::new(1.5, 0.5, 0.2),
        Vec3::new(-1.0, 0.0, 0.0),
        100.0,
        0.3,
    );
    run_parity(&ctx, &gpu, &mesh, &sweep);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn initial_overlap_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh capsule sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshCapsuleSweep::new(&ctx);
    let mesh = unit_quad();
    // Capsule already straddling the surface at the start: contact is immediate.
    let sweep = CapsuleSweep::new(
        Vec3::new(0.3, 0.4, 0.2),
        Vec3::new(0.3, 0.4, 0.8),
        Vec3::new(0.0, 0.0, -1.0),
        100.0,
        0.5,
    );
    run_parity(&ctx, &gpu, &mesh, &sweep);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn miss_reports_no_hit_on_every_path() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh capsule sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshCapsuleSweep::new(&ctx);
    let mesh = unit_quad();
    // Aimed away from the quad along +z: no path may report a contact.
    let sweep = CapsuleSweep::new(
        Vec3::new(0.3, 0.4, 5.0),
        Vec3::new(0.3, 0.4, 5.6),
        Vec3::new(0.0, 0.0, 1.0),
        100.0,
        0.5,
    );
    run_parity(&ctx, &gpu, &mesh, &sweep);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn horizontal_capsule_sweep_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh capsule sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshCapsuleSweep::new(&ctx);
    let mesh = unit_quad();
    // A capsule whose segment lies along +x (not axis-aligned with the sweep)
    // descending onto the quad from +z, exercising a non-trivial orientation.
    let sweep = CapsuleSweep::new(
        Vec3::new(0.2, 0.5, 4.0),
        Vec3::new(0.8, 0.5, 4.0),
        Vec3::new(0.0, 0.0, -1.0),
        100.0,
        0.25,
    );
    run_parity(&ctx, &gpu, &mesh, &sweep);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn nearest_of_many_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh capsule sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshCapsuleSweep::new(&ctx);
    let mesh = quad_fan(8);
    // Off the shared diagonal (y < x) so the winning triangle is unambiguous;
    // only the nearest quad (z = 7) may be reported.
    let sweep = CapsuleSweep::new(
        Vec3::new(0.3, 0.1, 20.0),
        Vec3::new(0.3, 0.1, 20.6),
        Vec3::new(0.0, 0.0, -1.0),
        100.0,
        0.25,
    );
    run_parity(&ctx, &gpu, &mesh, &sweep);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn tiled_grid_matches_cpu_golden_from_several_points() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh capsule sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshCapsuleSweep::new(&ctx);
    let mesh = tiled_grid(8);
    // Several straight-down vertical-capsule sweeps landing inside distinct
    // cells, each off the cell diagonal so the struck triangle is unambiguous.
    for &(x, y) in &[(0.3, 0.1), (3.7, 2.2), (6.1, 5.8), (7.4, 0.6), (1.2, 7.3)] {
        let sweep = CapsuleSweep::new(
            Vec3::new(x, y, 9.0),
            Vec3::new(x, y, 9.6),
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
            0.3,
        );
        run_parity(&ctx, &gpu, &mesh, &sweep);
    }
}
