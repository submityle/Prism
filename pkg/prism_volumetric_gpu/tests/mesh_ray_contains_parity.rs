//! Real-device parity for the closed-mesh point-containment twin:
//! [`GpuMeshRayContains`] must reproduce the `CPU` golden
//! [`Mesh::contains_point`](prism_render_architecture::particle::mesh_emission::Mesh::contains_point)
//! for every query point, across closed meshes (a unit cube and a corner
//! tetrahedron) and the degenerate inputs the twin guards.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The per-point output is a boolean crossing parity, not a continuous
//! quantity, so parity is asserted with an exact `==` on the recovered flag.
//! There is no floating-point tolerance on the output. Instead the fixtures
//! keep every random query point well clear of the Möller–Trumbore decision
//! thresholds (`det`, `u ∈ {0, 1}`, `v = 0`, `u + v = 1`, `t ≈ EPS`) via the
//! [`margins_safe`] rejection filter, so a legal `GPU` fused multiply-add can
//! never flip a comparison and hence never flip the boolean.
//!
//! # Mirrored private golden
//!
//! The golden ray direction `RAY_DIR` is a private constant in
//! [`mesh_emission`](prism_render_architecture::particle::mesh_emission), so
//! this test mirrors its literal as the local [`RAY_DIR`] (see its provenance
//! comment) solely to drive the margin pre-filter; the containment reference
//! itself is the public `Mesh::contains_point`, so no private function is
//! transcribed for the golden answer.
//!
//! Provenance: twins the `CPU` golden `Mesh::contains_point` in
//! `prism_render_architecture::particle::mesh_emission`; no Unreal Engine
//! source or derived code.

use prism_render_architecture::particle::mesh_emission::{Mesh, MeshVertex};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::mesh_ray_contains::{
    GpuMeshContainsQuery, GpuMeshRayContains, GpuTriangle,
};
use prism_volumetric_gpu::GpuContext;

/// MIRROR of `mesh_emission.rs::RAY_DIR` (private const, exact transcription).
///
/// The golden `Mesh::contains_point` casts this fixed, non-axis-aligned ray;
/// the literal is replicated here only so [`margins_safe`] can reject query
/// points that would land near a Möller–Trumbore decision threshold. The golden
/// answer is produced by the public `Mesh::contains_point`, which uses this same
/// direction internally.
const RAY_DIR: Vec3 = Vec3::new(1.0, 0.372_133_1, 0.211_907_3);

/// A tiny deterministic linear-congruential generator so the "random" points
/// are reproducible run to run without needing an external math library. The
/// constants are the Numerical Recipes `LCG` multiplier and increment.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    /// Advances the generator and returns the next raw word.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A reproducible `f32` in `[-range, range]`, formed with pure integer and
    /// multiply arithmetic (no transcendental method).
    fn next_signed(&mut self, range: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        (unit * 2.0 - 1.0) * range
    }

    /// A reproducible point whose components are each `center ± range`.
    fn next_point(&mut self, center: f32, range: f32) -> Vec3 {
        Vec3::new(
            center + self.next_signed(range),
            center + self.next_signed(range),
            center + self.next_signed(range),
        )
    }
}

/// Extracts the triangle positions of `mesh` into the twin's input layout so
/// the `GPU` kernel and the golden `Mesh::contains_point` test the exact same
/// geometry.
fn triangles_of(mesh: &Mesh) -> Vec<GpuTriangle> {
    (0..mesh.triangle_count())
        .map(|i| {
            let tri = mesh.triangle(i);
            GpuTriangle::new(tri.a.position, tri.b.position, tri.c.position)
        })
        .collect()
}

/// Builds the closed unit cube `[0, 1]^3` as 8 vertices and 12 triangles.
fn unit_cube() -> Mesh {
    let v = |x: f32, y: f32, z: f32| MeshVertex::at(Vec3::new(x, y, z));
    let vertices = vec![
        v(0.0, 0.0, 0.0),
        v(1.0, 0.0, 0.0),
        v(1.0, 1.0, 0.0),
        v(0.0, 1.0, 0.0),
        v(0.0, 0.0, 1.0),
        v(1.0, 0.0, 1.0),
        v(1.0, 1.0, 1.0),
        v(0.0, 1.0, 1.0),
    ];
    let indices = vec![
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 5, 1],
        [0, 4, 5],
        [3, 2, 6],
        [3, 6, 7],
        [0, 3, 7],
        [0, 7, 4],
        [1, 5, 6],
        [1, 6, 2],
    ];
    Mesh::new(vertices, indices)
}

/// Builds the closed corner tetrahedron with vertices at the origin and the
/// three unit axes as 4 triangles.
fn corner_tetrahedron() -> Mesh {
    let v = |x: f32, y: f32, z: f32| MeshVertex::at(Vec3::new(x, y, z));
    let vertices = vec![
        v(0.0, 0.0, 0.0),
        v(1.0, 0.0, 0.0),
        v(0.0, 1.0, 0.0),
        v(0.0, 0.0, 1.0),
    ];
    let indices = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    Mesh::new(vertices, indices)
}

/// Mirrors the golden Möller–Trumbore test and rejects a point whenever a
/// taken branch's intermediate lands within a comfort margin of its decision
/// threshold, so a `GPU` fused multiply-add can never flip the boolean.
///
/// The margins (`1e-3` on `u`/`v`, `1e-2` on `det`/`t`) are far larger than the
/// single-precision rounding noise (`~1e-6`) yet never reject a whole mesh: for
/// both fixtures every face determinant is `O(0.2..=1.6)` with this `RAY_DIR`,
/// clearing the `det` guard.
fn margins_safe(origin: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let e1 = b.sub(a);
    let e2 = c.sub(a);
    let h = dir.cross(e2);
    let det = e1.dot(h);
    if det.abs() < 1.0e-2 {
        return false;
    }
    let inv_det = 1.0 / det;
    let s = origin.sub(a);
    let u = inv_det * s.dot(h);
    if u.abs() < 1.0e-3 || (u - 1.0).abs() < 1.0e-3 {
        return false;
    }
    if !(0.0..=1.0).contains(&u) {
        // Clearly misses the triangle in `u`; the branch is stable, so safe.
        return true;
    }
    let q = s.cross(e1);
    let v = inv_det * dir.dot(q);
    if v.abs() < 1.0e-3 || (u + v - 1.0).abs() < 1.0e-3 {
        return false;
    }
    if v < 0.0 || u + v > 1.0 {
        // Clearly fails the barycentric test; the branch is stable, so safe.
        return true;
    }
    let t = inv_det * e2.dot(q);
    if t.abs() < 1.0e-2 {
        return false;
    }
    true
}

/// Collects `want` query points that clear [`margins_safe`] against every
/// triangle of the mesh, sampling the expanded bounding box with the `LCG`.
fn collect_safe_points(
    triangles: &[GpuTriangle],
    rng: &mut Lcg,
    center: f32,
    range: f32,
    want: usize,
) -> Vec<Vec3> {
    let mut points = Vec::with_capacity(want);
    let max_tries = want * 500;
    for _ in 0..max_tries {
        if points.len() >= want {
            break;
        }
        let p = rng.next_point(center, range);
        let safe = triangles
            .iter()
            .all(|tri| margins_safe(p, RAY_DIR, tri.a, tri.b, tri.c));
        if safe {
            points.push(p);
        }
    }
    points
}

/// Runs one `CPU`-vs-`GPU` containment scenario over the given points and
/// asserts an exact boolean parity.
fn assert_contains_parity(
    label: &str,
    engine: &GpuMeshRayContains,
    ctx: &GpuContext,
    mesh: &Mesh,
    points: &[Vec3],
) {
    let triangles = triangles_of(mesh);
    let gpu = engine.contains(
        ctx,
        &GpuMeshContainsQuery {
            triangles,
            points: points.to_vec(),
        },
    );
    assert_eq!(gpu.len(), points.len(), "{label}: flag length mismatch");
    for (i, (p, g)) in points.iter().zip(gpu.iter()).enumerate() {
        let cpu = mesh.contains_point(*p);
        assert_eq!(
            cpu, *g,
            "{label}: point {i} ({}, {}, {}) classification mismatch: cpu {cpu}, gpu {g}",
            p.x, p.y, p.z
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn cube_random_points_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_ray_contains cube parity: no wgpu adapter");
        return;
    };
    let engine = GpuMeshRayContains::new(&ctx);
    let mesh = unit_cube();
    let triangles = triangles_of(&mesh);
    let mut rng = Lcg::new(0x1234_5678);

    // Expanded bbox [-1, 2]^3 around the unit cube: naturally mixes inside and
    // outside points, all held clear of the triangle decision thresholds.
    let points = collect_safe_points(&triangles, &mut rng, 0.5, 1.5, 64);
    assert!(
        points.len() >= 48,
        "cube: expected to collect enough safe points, got {}",
        points.len()
    );

    // The scenario must exercise both classifications to be meaningful.
    let inside = points.iter().filter(|p| mesh.contains_point(**p)).count();
    assert!(
        inside > 0 && inside < points.len(),
        "cube: safe points should include both inside and outside, got {inside} inside of {}",
        points.len()
    );

    assert_contains_parity("cube random", &engine, &ctx, &mesh, &points);
}

#[test]
fn tetrahedron_random_points_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshRayContains::new(&ctx);
    let mesh = corner_tetrahedron();
    let triangles = triangles_of(&mesh);
    let mut rng = Lcg::new(0x0bad_f00d);

    // Expanded bbox [-0.5, 1.5]^3 around the corner tetrahedron.
    let points = collect_safe_points(&triangles, &mut rng, 0.5, 1.0, 32);
    assert!(
        points.len() >= 20,
        "tetra: expected to collect enough safe points, got {}",
        points.len()
    );

    let inside = points.iter().filter(|p| mesh.contains_point(**p)).count();
    assert!(
        inside > 0 && inside < points.len(),
        "tetra: safe points should include both inside and outside, got {inside} inside of {}",
        points.len()
    );

    assert_contains_parity("tetra random", &engine, &ctx, &mesh, &points);
}

#[test]
fn explicit_inside_and_outside_points_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshRayContains::new(&ctx);

    let cube = unit_cube();
    let cube_points = vec![
        // Clearly inside the unit cube.
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(0.25, 0.6, 0.4),
        // Clearly outside the unit cube.
        Vec3::new(-0.5, 0.5, 0.5),
        Vec3::new(0.5, 1.75, 0.5),
        Vec3::new(0.5, 0.5, 3.0),
    ];
    assert_contains_parity("cube explicit", &engine, &ctx, &cube, &cube_points);

    let tetra = corner_tetrahedron();
    let tetra_points = vec![
        // Clearly inside the corner tetrahedron (x + y + z well below 1).
        Vec3::new(0.1, 0.1, 0.1),
        Vec3::new(0.2, 0.15, 0.3),
        // Clearly outside.
        Vec3::new(0.9, 0.9, 0.9),
        Vec3::new(-0.4, 0.2, 0.2),
    ];
    assert_contains_parity("tetra explicit", &engine, &ctx, &tetra, &tetra_points);
}

#[test]
fn empty_mesh_classifies_every_point_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshRayContains::new(&ctx);

    let points = vec![
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-1.0, 2.0, 0.3),
        Vec3::ZERO,
    ];
    let flags = engine.contains(
        &ctx,
        &GpuMeshContainsQuery {
            triangles: Vec::new(),
            points: points.clone(),
        },
    );
    assert_eq!(flags.len(), points.len(), "empty mesh: length mismatch");
    assert!(
        flags.iter().all(|flag| !*flag),
        "empty mesh: every point should be outside"
    );
}

#[test]
fn empty_points_short_circuits_to_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshRayContains::new(&ctx);
    let mesh = unit_cube();

    let flags = engine.contains(
        &ctx,
        &GpuMeshContainsQuery {
            triangles: triangles_of(&mesh),
            points: Vec::new(),
        },
    );
    assert!(
        flags.is_empty(),
        "empty point list should yield empty flags"
    );
}
