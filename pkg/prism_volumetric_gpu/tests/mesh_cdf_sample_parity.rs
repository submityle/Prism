//! Real-device parity for the area-weighted mesh triangle selector twin:
//! [`GpuMeshCdfSample`] must reproduce the `CPU` golden
//! [`Mesh::select_triangle`](prism_render_architecture::particle::mesh_emission::Mesh::select_triangle)
//! (the prefix-sum `CDF` picker extracted from
//! [`mesh_emission`](prism_render_architecture::particle::mesh_emission)) across
//! a batch of unit draws on several meshes — a single triangle, triangles of
//! unequal area, a mesh with a trailing zero-area triangle, a fully degenerate
//! mesh, and an empty mesh.
//!
//! The golden selector is `pub`, so this test calls `mesh.select_triangle(u)`
//! directly for the reference index; no private mirror is needed. The uploaded
//! `cdf` / `triangle_area` / `total_area` are rebuilt on the host from the
//! public mesh surface (`triangle_area`, `total_area`) with the identical
//! prefix-sum arithmetic the golden private `rebuild_cdf` performs, so they
//! match the selector's inputs bit for bit.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The pick is pure integer control flow over host-supplied scalars, so `CPU`
//! and `GPU` agree bit for bit; the test asserts integer index equality, not a
//! tolerance. Draws are chosen (by rejection) to keep the scaled draw clear of
//! every `CDF` boundary, so no tie can split a legitimate port.
//!
//! Provenance: standard area-weighted inverse-`CDF` triangle sampling; no Unreal
//! Engine source or derived code.

use prism_render_architecture::particle::mesh_emission::{Mesh, MeshVertex};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::mesh_cdf_sample::{
    GpuMeshCdfSample, GpuMeshCdfSampleQuery, GpuMeshCdfSampleResult,
};
use prism_volumetric_gpu::GpuContext;

/// Area threshold below which a triangle is degenerate, matching the golden
/// `EPS`.
const EPS: f32 = 1e-6;

/// A tiny deterministic linear-congruential generator so the "random" draws are
/// reproducible run to run without pulling in any external math or random
/// crate. The constants are the Numerical Recipes `LCG` multiplier and
/// increment.
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

    /// A reproducible `f32` in `[0, 1)`, built only from integer arithmetic (no
    /// transcendental or inexact helper).
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Builds a position-only vertex at `(x, y, z)`.
fn vert(x: f32, y: f32, z: f32) -> MeshVertex {
    MeshVertex::at(Vec3::new(x, y, z))
}

/// Rebuilds the area `CDF`, per-triangle areas and total area from the mesh's
/// public surface with the identical prefix-sum arithmetic the golden private
/// `rebuild_cdf` performs, then packages a batch of draws into a query.
fn build_query(mesh: &Mesh, draws: Vec<f32>) -> GpuMeshCdfSampleQuery {
    let count = mesh.triangle_count();
    let mut cdf = Vec::with_capacity(count);
    let mut triangle_area = Vec::with_capacity(count);
    let mut running = 0.0f32;
    for i in 0..count {
        let a = mesh.triangle_area(i);
        running += a;
        triangle_area.push(a);
        cdf.push(running);
    }
    GpuMeshCdfSampleQuery {
        cdf,
        triangle_area,
        total_area: mesh.total_area(),
        draws,
    }
}

/// True when the scaled draw stays clear of every `CDF` boundary (and of zero),
/// so the `<=` comparison cannot land on a tie.
fn draw_is_clear(u: f32, cdf: &[f32], total_area: f32) -> bool {
    if total_area < EPS {
        return true;
    }
    let needle = u.clamp(0.0, 1.0) * total_area;
    let margin = (total_area * 1.0e-3).max(1.0e-4);
    if needle < margin {
        return false;
    }
    cdf.iter().all(|&c| (needle - c).abs() > margin)
}

/// Draws `want` unit values clear of the `CDF` boundaries by rejection, seeded
/// deterministically.
fn clear_draws(mesh: &Mesh, want: usize, seed: u32) -> Vec<f32> {
    let count = mesh.triangle_count();
    let mut cdf = Vec::with_capacity(count);
    let mut running = 0.0f32;
    for i in 0..count {
        running += mesh.triangle_area(i);
        cdf.push(running);
    }
    let total_area = mesh.total_area();
    let mut rng = Lcg::new(seed);
    let mut draws = Vec::with_capacity(want);
    // Bounded attempts: even a dense CDF leaves wide clear intervals, so this
    // terminates well before the cap.
    for _ in 0..(want * 64 + 1024) {
        if draws.len() >= want {
            break;
        }
        let u = rng.next_unit();
        if draw_is_clear(u, &cdf, total_area) {
            draws.push(u);
        }
    }
    assert_eq!(draws.len(), want, "could not sample enough clear draws");
    draws
}

/// Asserts the `GPU` indices equal the golden `select_triangle` for every draw.
fn assert_parity(label: &str, mesh: &Mesh, draws: &[f32], gpu: &GpuMeshCdfSampleResult) {
    assert_eq!(
        gpu.indices.len(),
        draws.len(),
        "{label}: index-count mismatch"
    );
    for (i, (&u, &g)) in draws.iter().zip(gpu.indices.iter()).enumerate() {
        let golden = mesh.select_triangle(u) as u32;
        assert_eq!(
            g, golden,
            "{label}: draw {i} (u = {u}) picked gpu {g}, golden {golden}"
        );
    }
}

/// Runs one mesh scenario end to end and asserts bit-exact parity.
fn check_scenario(
    label: &str,
    engine: &GpuMeshCdfSample,
    ctx: &GpuContext,
    mesh: &Mesh,
    draws: Vec<f32>,
) {
    let q = build_query(mesh, draws.clone());
    let gpu = engine.sample(ctx, &q);
    assert_parity(label, mesh, &draws, &gpu);
}

/// A single non-degenerate right triangle of area `0.5`.
fn single_triangle_mesh() -> Mesh {
    let vertices = vec![
        vert(0.0, 0.0, 0.0),
        vert(1.0, 0.0, 0.0),
        vert(0.0, 1.0, 0.0),
    ];
    Mesh::new(vertices, vec![[0, 1, 2]])
}

/// Two triangles of unequal area (`0.5` then `2.0`).
fn unequal_area_mesh() -> Mesh {
    let vertices = vec![
        vert(0.0, 0.0, 0.0),
        vert(1.0, 0.0, 0.0),
        vert(0.0, 1.0, 0.0),
        vert(0.0, 0.0, 0.0),
        vert(2.0, 0.0, 0.0),
        vert(0.0, 2.0, 0.0),
    ];
    Mesh::new(vertices, vec![[0, 1, 2], [3, 4, 5]])
}

/// Three triangles whose last one is a collinear zero-area degenerate; the
/// selector must walk back off it.
fn trailing_degenerate_mesh() -> Mesh {
    let vertices = vec![
        vert(0.0, 0.0, 0.0),
        vert(1.0, 0.0, 0.0),
        vert(0.0, 1.0, 0.0),
        vert(0.0, 0.0, 0.0),
        vert(2.0, 0.0, 0.0),
        vert(0.0, 2.0, 0.0),
        // Collinear trio: zero area.
        vert(0.0, 0.0, 0.0),
        vert(1.0, 0.0, 0.0),
        vert(2.0, 0.0, 0.0),
    ];
    Mesh::new(vertices, vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]])
}

/// A mesh whose every triangle is collinear, so `total_area < EPS`.
fn fully_degenerate_mesh() -> Mesh {
    let vertices = vec![
        vert(0.0, 0.0, 0.0),
        vert(1.0, 0.0, 0.0),
        vert(2.0, 0.0, 0.0),
        vert(0.0, 0.0, 0.0),
        vert(0.0, 1.0, 0.0),
        vert(0.0, 2.0, 0.0),
    ];
    Mesh::new(vertices, vec![[0, 1, 2], [3, 4, 5]])
}

/// A mesh with no triangles.
fn empty_mesh() -> Mesh {
    Mesh::new(Vec::new(), Vec::new())
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_single_triangle() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh-cdf-sample parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuMeshCdfSample::new(&ctx);

    let mesh = single_triangle_mesh();
    // A single triangle always resolves to index 0 regardless of the draw.
    let draws = clear_draws(&mesh, 32, 0x1234_5678);
    check_scenario("single triangle", &engine, &ctx, &mesh, draws);
}

#[test]
fn gpu_matches_cpu_on_unequal_areas() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshCdfSample::new(&ctx);

    // Areas 0.5 and 2.0: the larger triangle should win about four draws in
    // five. Parity is asserted per draw against the golden selector.
    let mesh = unequal_area_mesh();
    let draws = clear_draws(&mesh, 64, 0x0BAD_F00D);
    check_scenario("unequal areas", &engine, &ctx, &mesh, draws);
}

#[test]
fn gpu_matches_cpu_on_trailing_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshCdfSample::new(&ctx);

    // The trailing collinear triangle has zero area; a draw landing in its
    // (empty) interval must walk back to the previous positive-area triangle.
    let mesh = trailing_degenerate_mesh();
    let mut draws = clear_draws(&mesh, 48, 0x00C0_FFEE);
    // Add draws near the top of the range so the pick starts on the trailing
    // index before walking back. These stay clear of the boundary because the
    // last CDF step is zero width.
    draws.push(0.995);
    draws.push(0.9999);
    // u = 1.0 scales exactly to total_area, so the partition point lands on the
    // trailing zero-area triangle and the back-walk must step off it. The scale
    // is an exact f32 multiply on both host and device, so this is not a
    // rounding-sensitive tie.
    draws.push(1.0);
    check_scenario("trailing degenerate", &engine, &ctx, &mesh, draws);
}

#[test]
fn gpu_matches_cpu_on_fully_degenerate_mesh() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshCdfSample::new(&ctx);

    // total_area < EPS: every draw must resolve to index 0.
    let mesh = fully_degenerate_mesh();
    let draws = clear_draws(&mesh, 16, 0xDEAD_BEEF);
    let q = build_query(&mesh, draws.clone());
    let gpu = engine.sample(&ctx, &q);
    for (i, &g) in gpu.indices.iter().enumerate() {
        assert_eq!(g, 0, "fully degenerate draw {i} should pick index 0");
    }
    assert_parity("fully degenerate", &mesh, &draws, &gpu);
}

#[test]
fn gpu_matches_cpu_on_empty_mesh() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshCdfSample::new(&ctx);

    // No triangles: count == 0 short-circuits inside the kernel to index 0 for
    // every draw, matching the reference.
    let mesh = empty_mesh();
    let draws = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
    let q = build_query(&mesh, draws.clone());
    assert!(q.cdf.is_empty(), "empty mesh should have an empty CDF");
    let gpu = engine.sample(&ctx, &q);
    for (i, &g) in gpu.indices.iter().enumerate() {
        assert_eq!(g, 0, "empty-mesh draw {i} should pick index 0");
    }
    assert_parity("empty mesh", &mesh, &draws, &gpu);
}

#[test]
fn gpu_matches_cpu_on_empty_draw_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMeshCdfSample::new(&ctx);

    // An empty draw batch short-circuits on the host to an empty result with no
    // dispatch.
    let mesh = unequal_area_mesh();
    let q = build_query(&mesh, Vec::new());
    let gpu = engine.sample(&ctx, &q);
    assert!(
        gpu.indices.is_empty(),
        "empty draw batch should return no indices"
    );
}
