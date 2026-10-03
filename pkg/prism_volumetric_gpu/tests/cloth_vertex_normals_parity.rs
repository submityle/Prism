//! Real-device parity for the area-weighted vertex-normal twin:
//! [`GpuClothVertexNormals`](prism_volumetric_gpu::cloth_vertex_normals::GpuClothVertexNormals)
//! must reproduce the `CPU` golden
//! [`accumulate_vertex_normals`](prism_render_architecture::cloth::layers::accumulate_vertex_normals)
//! — one outward normal per vertex, area-weighted across incident faces — on a
//! single triangle, a shared-vertex mesh, a degenerate face, an out-of-range
//! face, an isolated vertex, and a jittered-grid random sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`accumulate_vertex_normals`](prism_render_architecture::cloth::layers::accumulate_vertex_normals)
//! is public and called directly as the reference: each fixture builds the
//! golden normals in-host and the `GPU` result is pinned against them, so a
//! `GPU == golden` pass is direct evidence of a faithful port.
//!
//! # Parity criterion
//!
//! Each per-vertex sum is replayed on the device in the golden's exact
//! triangle-ascending order, so the pre-normalization sum differs only by a
//! device fused multiply-add inside a single cross product; the normalized
//! components then differ by a few units in the last place. Components are
//! therefore asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. A
//! degenerate or isolated vertex is an exact zero on both sides, which the same
//! bound admits.
//!
//! # Conditioning
//!
//! The random sweep is a jittered height-field grid with consistent winding, so
//! every face normal points broadly along `+z` and no per-vertex sum cancels
//! toward the zero vector; the normalized direction is therefore well clear of
//! the degenerate band, keeping the `CPU` and `GPU` on the same side of the
//! guarded normalization.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::layers`；无第三方引擎源码或衍生代码。

use prism_render_architecture::cloth::layers::accumulate_vertex_normals;
use prism_render_architecture::cloth::Vec3;
use prism_volumetric_gpu::cloth_vertex_normals::{ClothVertexNormalsQuery, GpuClothVertexNormals};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a normal component. A `GPU` `sqrt` and a possible
/// fused multiply-add may land a few units in the last place from the scalar
/// reference; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Runs the golden on the same inputs the `GPU` sees and returns its normals as
/// plain component arrays, the faithful oracle the `GPU` is pinned against.
fn oracle(positions: &[[f32; 3]], triangles: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let verts: Vec<Vec3> = positions
        .iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect();
    let mut out: Vec<Vec3> = Vec::new();
    accumulate_vertex_normals(&verts, triangles, &mut out);
    out.iter().map(|n| [n.x, n.y, n.z]).collect()
}

/// Dispatches `query` and pins every normal component against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuClothVertexNormals, query: &ClothVertexNormalsQuery) {
    let got = gpu.evaluate(ctx, query);
    let want = oracle(&query.positions, &query.triangles);
    assert_eq!(
        got.normals.len(),
        want.len(),
        "result count must match the vertex count"
    );
    for (idx, (g, w)) in got.normals.iter().zip(want.iter()).enumerate() {
        for (axis, (a, b)) in g.iter().zip(w.iter()).enumerate() {
            assert!(
                close(*a, *b),
                "vertex {idx} axis {axis}: gpu {a} vs cpu {b}"
            );
        }
    }
}

/// Builds a jittered height-field grid of `nx` by `ny` vertices with consistent
/// winding, so every face normal points broadly along `+z`. Only integer `LCG`
/// draws feed the jitter, so no transcendental method appears.
fn grid_mesh(nx: u32, ny: u32, state: &mut u64) -> ClothVertexNormalsQuery {
    let mut positions = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            let jx = (lcg(state) % 1000) as f32 / 1000.0 * 0.2 - 0.1;
            let jy = (lcg(state) % 1000) as f32 / 1000.0 * 0.2 - 0.1;
            let jz = (lcg(state) % 1000) as f32 / 1000.0 * 0.6 - 0.3;
            positions.push([i as f32 + jx, j as f32 + jy, jz]);
        }
    }
    let mut triangles = Vec::new();
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let v00 = j * nx + i;
            let v10 = j * nx + i + 1;
            let v01 = (j + 1) * nx + i;
            let v11 = (j + 1) * nx + i + 1;
            triangles.push([v00, v10, v11]);
            triangles.push([v00, v11, v01]);
        }
    }
    ClothVertexNormalsQuery {
        positions,
        triangles,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_vertex_normals parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuClothVertexNormals::new(&ctx);
    // A zero-vertex mesh short-circuits before any dispatch (a storage buffer
    // cannot be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(
        &ctx,
        &ClothVertexNormalsQuery {
            positions: Vec::new(),
            triangles: Vec::new(),
        },
    );
    assert!(got.normals.is_empty(), "an empty mesh produces no normals");
}

#[test]
fn single_triangle_normal_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothVertexNormals::new(&ctx);
    // A single counter-clockwise face in the z = 0 plane: every vertex normal
    // is the face's outward +z direction.
    let query = ClothVertexNormalsQuery {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        triangles: vec![[0, 1, 2]],
    };
    check(&ctx, &gpu, &query);
}

#[test]
fn shared_vertex_mesh_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothVertexNormals::new(&ctx);
    // A folded quad (two faces sharing an edge) with a crease: the shared-edge
    // vertices average both face normals, exercising multi-face accumulation.
    let query = ClothVertexNormalsQuery {
        positions: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.4],
            [0.0, 1.0, 0.4],
        ],
        triangles: vec![[0, 1, 2], [0, 2, 3]],
    };
    check(&ctx, &gpu, &query);
}

#[test]
fn degenerate_face_leaves_zero_normals() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothVertexNormals::new(&ctx);
    // Three collinear points: the face cross is exactly zero, so every touched
    // vertex normal stays the zero vector on both sides.
    let query = ClothVertexNormalsQuery {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        triangles: vec![[0, 1, 2]],
    };
    let want = oracle(&query.positions, &query.triangles);
    for n in &want {
        assert_eq!(*n, [0.0, 0.0, 0.0], "golden degenerate normal must be zero");
    }
    check(&ctx, &gpu, &query);
}

#[test]
fn out_of_range_face_is_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothVertexNormals::new(&ctx);
    // The second triangle references vertex 5 of a 3-vertex mesh; the golden and
    // the twin both skip it, so only the first face contributes.
    let query = ClothVertexNormalsQuery {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        triangles: vec![[0, 1, 2], [0, 1, 5]],
    };
    check(&ctx, &gpu, &query);
}

#[test]
fn isolated_vertex_stays_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothVertexNormals::new(&ctx);
    // Vertex 3 is referenced by no face: its normal is the zero vector while the
    // others carry the single face's +z normal.
    let query = ClothVertexNormalsQuery {
        positions: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 5.0, 5.0],
        ],
        triangles: vec![[0, 1, 2]],
    };
    let want = oracle(&query.positions, &query.triangles);
    assert_eq!(
        want[3],
        [0.0, 0.0, 0.0],
        "isolated vertex normal must be zero"
    );
    check(&ctx, &gpu, &query);
}

#[test]
fn small_grid_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothVertexNormals::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    check(&ctx, &gpu, &grid_mesh(4, 3, &mut state));
}

#[test]
fn random_grid_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothVertexNormals::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // Several differently sized jittered grids pin every reported vertex across
    // many workgroups' worth of threads.
    for _ in 0..24 {
        let nx = 3 + lcg(&mut state) % 6;
        let ny = 3 + lcg(&mut state) % 6;
        check(&ctx, &gpu, &grid_mesh(nx, ny, &mut state));
    }
}
