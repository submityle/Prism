//! Real-device parity for the triangle-mesh swept-oriented-box (`OBB`) scene
//! query: the `GPU`-driven [`GpuTrimeshObbSweep`] must agree with the `CPU` brute
//! golden [`cpu_trimesh_obb_sweep`] and the `CPU` `LBVH` branch-and-bound query
//! [`cpu_trimesh_obb_sweep_built`] over the same static triangle mesh and moving
//! oriented box.
//!
//! The `CPU` `BVH` query is pinned to the brute golden in its own unit suite, so
//! matching both here transitively pins the device result to the brute golden.
//! The cases mirror the `CPU` unit scenes and add device coverage: a box
//! descending flat onto a quad's front face; the same box rotated 45 degrees
//! about `Y` so it lands on its lower edge; an in-plane sweep that misses the
//! face and must resolve on an edge; a box already overlapping the quad so the
//! contact is immediate; a sweep aimed away that must miss on every path; a fan
//! of parallel quads where only the nearest may win; and a larger tiled grid
//! probed from several points so the `CPU` `LBVH` prune and the device brute
//! sweep must still land on the identical triangle and surface geometry.
//!
//! A sharp (zero convex radius) box lands with a face or edge flush on the
//! surface, so the conservative-advance solver's final step is the intersecting
//! branch, whose witness is the box centre rather than a surface point. Both the
//! `CPU` paths and the `GPU` kernel run the identical arithmetic, so they report
//! the same centre witness and parity still holds within tolerance; the time of
//! impact and push-out normal are the physically meaningful invariants.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing.
//!
//! Provenance: conservative advancement after Mirtich (2000) and van den Bergen
//! (2004); `GJK` distance per Gilbert-Johnson-Keerthi (1988) with Ericson's
//! Voronoi sub-distance (2005). No Unreal Engine source or derived code.

use glam::{Quat, Vec3};
use prism_physics_gpu::{
    cpu_trimesh_obb_sweep, cpu_trimesh_obb_sweep_built, GpuContext, GpuTrimeshObbSweep, ObbSweep,
    ObbSweepHit, Trimesh,
};

/// Tolerance on the time of impact, contact point, and normal: the inexact
/// steps are the square roots and reciprocals in the `GJK` and advance solve.
const TOL: f32 = 1e-3;

/// A unit quad on the `z = 0` plane spanning `[-1, 1]` in `x` and `y`, wound so
/// its outward normal is `+z`. Two triangles: lower-right (0) and upper-left (1).
fn unit_quad() -> Trimesh {
    Trimesh::new(
        vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ],
        vec![[0, 1, 2], [0, 2, 3]],
    )
}

/// A fan of `count` axis-aligned quads, each spanning `[-1, 1]^2` at an
/// increasing `z = k` plane, every quad split into a lower-right and upper-left
/// triangle. A `+z` box sweeping `-z` must resolve the nearest quad only.
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
fn assert_hit_matches(want: ObbSweepHit, got: ObbSweepHit) {
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
fn run_parity(ctx: &GpuContext, gpu: &GpuTrimeshObbSweep, mesh: &Trimesh, sweep: &ObbSweep) {
    let brute = cpu_trimesh_obb_sweep(mesh, sweep);
    let bvh = cpu_trimesh_obb_sweep_built(mesh, sweep);
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
        eprintln!("skipping GPU trimesh OBB sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshObbSweep::new(&ctx);
    let mesh = unit_quad();
    // An axis-aligned 0.25-half-extent box descending straight down onto the
    // quad's face. It is parked at (0.4, -0.4) so its whole footprint lies in
    // the lower-right (y < x) triangle, leaving one unambiguous winner on both
    // the brute and LBVH paths; its underside reaches z = 0 one half-extent
    // before the centre, so toi = 5 - 0.25 = 4.75.
    let sweep = ObbSweep::new(
        Vec3::new(0.4, -0.4, 5.0),
        Quat::IDENTITY,
        Vec3::splat(0.25),
        Vec3::new(0.0, 0.0, -1.0),
        100.0,
    );
    run_parity(&ctx, &gpu, &mesh, &sweep);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn rotated_box_sweep_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU trimesh OBB sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshObbSweep::new(&ctx);
    let mesh = unit_quad();
    // The same box rotated 45 degrees about Y: it lands on its lower edge, whose
    // downward reach is 0.25 * sqrt(2). The non-identity orientation is the key
    // difference from the capsule and sphere sweeps. Parked at (0.4, -0.4) so
    // its footprint stays within the lower-right triangle for an unambiguous
    // winner on both paths.
    let sweep = ObbSweep::new(
        Vec3::new(0.4, -0.4, 5.0),
        Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4),
        Vec3::splat(0.25),
        Vec3::new(0.0, 0.0, -1.0),
        100.0,
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
        eprintln!("skipping GPU trimesh OBB sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshObbSweep::new(&ctx);
    let mesh = unit_quad();
    // A box to the right of the quad's right edge (x = 1), swept in -x. The face
    // plane is parallel to the sweep, so the quad's right edge catches the box.
    let sweep = ObbSweep::new(
        Vec3::new(1.5, 0.0, 0.0),
        Quat::IDENTITY,
        Vec3::splat(0.3),
        Vec3::new(-1.0, 0.0, 0.0),
        100.0,
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
        eprintln!("skipping GPU trimesh OBB sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshObbSweep::new(&ctx);
    let mesh = unit_quad();
    // A 0.25-half-extent box centred at z = 0.1 already straddles the plane, so
    // the contact is immediate (toi = 0, witness at the box centre). Parked at
    // (0.4, -0.4) so the overlapping triangle is the unambiguous lower-right one
    // on both the brute and LBVH paths.
    let sweep = ObbSweep::new(
        Vec3::new(0.4, -0.4, 0.1),
        Quat::IDENTITY,
        Vec3::splat(0.25),
        Vec3::new(0.0, 0.0, -1.0),
        100.0,
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
        eprintln!("skipping GPU trimesh OBB sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshObbSweep::new(&ctx);
    let mesh = unit_quad();
    // Aimed away from the quad along +z: no path may report a contact.
    let sweep = ObbSweep::new(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
        Vec3::splat(0.5),
        Vec3::new(0.0, 0.0, 1.0),
        100.0,
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
        eprintln!("skipping GPU trimesh OBB sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshObbSweep::new(&ctx);
    let mesh = quad_fan(8);
    // Keep the box fully inside the lower-right (y < x) half of each quad so one
    // triangle is the unambiguous winner on both paths; only the nearest quad
    // (z = 7) may be reported.
    let sweep = ObbSweep::new(
        Vec3::new(0.4, -0.4, 20.0),
        Quat::IDENTITY,
        Vec3::splat(0.25),
        Vec3::new(0.0, 0.0, -1.0),
        100.0,
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
        eprintln!("skipping GPU trimesh OBB sweep parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTrimeshObbSweep::new(&ctx);
    let mesh = tiled_grid(8);
    // Several straight-down box sweeps landing inside distinct cells, each off
    // the cell diagonal (small boxes well inside a single triangle) so the
    // struck triangle is unambiguous on both the brute and LBVH paths.
    for &(x, y) in &[(0.7, 0.2), (3.8, 2.3), (6.7, 5.2), (7.3, 0.7), (1.7, 7.2)] {
        let sweep = ObbSweep::new(
            Vec3::new(x, y, 9.0),
            Quat::IDENTITY,
            Vec3::splat(0.15),
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
        );
        run_parity(&ctx, &gpu, &mesh, &sweep);
    }
}
