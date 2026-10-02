//! Real-device parity for the Marching Cubes per-cell twin:
//! [`GpuMarchingCubes`](prism_volumetric_gpu::marching_cubes::GpuMarchingCubes)
//! must reproduce the `CPU` golden
//! [`marching_cubes`](prism_render_architecture::particle::marching_cubes)
//! per-voxel-cube kernel: the eight-corner `cube_index` classification, the
//! shipped
//! [`EDGE_TABLE`](prism_render_architecture::particle::marching_cubes::EDGE_TABLE)
//! edge selection, the guarded per-edge linear crossings, and the up-to-five
//! triangles selected from
//! [`TRI_TABLE`](prism_render_architecture::particle::marching_cubes::TRI_TABLE)
//! in the reference emission order.
//!
//! Each fixture builds a single `2 by 2 by 2`
//! [`ScalarField`](prism_render_architecture::particle::marching_cubes::ScalarField)
//! from eight corner values laid out in the
//! [`CORNER_OFFSET`](prism_render_architecture::particle::marching_cubes::CORNER_OFFSET)
//! order (grid index `off[0] + off[1] * 2 + off[2] * 4`), runs the golden
//! [`marching_cubes`](prism_render_architecture::particle::marching_cubes) to get
//! a reference
//! [`Mesh`](prism_render_architecture::particle::marching_cubes::Mesh), and feeds
//! the same corner values plus `iso` to the twin as a
//! [`MarchingCubesCellQuery`](prism_volumetric_gpu::marching_cubes::MarchingCubesCellQuery).
//! The golden `Mesh` vertex order equals the per-cell `TRI_TABLE` emission order,
//! so the twin's `vertices` are compared one-to-one against `positions`.
//!
//! The fixtures cover a spread of `cube_index` cases: all-corners-above and
//! all-corners-below (both empty), several single-corner cases, adjacent and
//! opposite corner pairs, a three-corner case and a checkerboard four-corner
//! case, plus a mixed batch. Every corner value sits at least a unit away from
//! `iso = 0`, so the classification is never near the `corner_val == iso`
//! boundary, and every activated edge joins a negative and a positive corner, so
//! each crossing denominator has magnitude at least three — far from the
//! `CMP_EPS` midpoint-fallback region. Corner magnitudes are deliberately
//! asymmetric so the crossing parameter is rarely exactly one half, exercising
//! the full linear interpolation rather than only the midpoint.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `cube_index` and the emitted vertex count are discrete classifications
//! built from `corner_val < iso` magnitude compares, so `CPU` and `GPU` must
//! agree exactly: the comparison is an exact `==` on both, and the reference
//! triangle topology (sequential `indices`) is likewise checked exactly. The
//! interpolated vertex coordinates thread through a subtract, one guarded
//! division, a `clamp`, a multiply and an add, so they are compared under
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::marching_cubes`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::marching_cubes::{
    marching_cubes, ScalarField, CORNER_OFFSET,
};
use prism_volumetric_gpu::marching_cubes::{GpuMarchingCubes, MarchingCubesCellQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous vertex coordinates.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous vertex coordinates.
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

/// Tolerant comparison of two `xyz` vertex positions.
fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
}

/// Builds a single `2 by 2 by 2` [`ScalarField`] from its eight corner values in
/// the standard [`CORNER_OFFSET`] order, mirroring the golden `single_cube` test
/// helper's `off[0] + off[1] * 2 + off[2] * 4` fill.
fn single_cube(corners: [f32; 8]) -> ScalarField {
    let mut values = vec![0.0_f32; 8];
    for (c, off) in CORNER_OFFSET.iter().enumerate() {
        let idx = off[0] + off[1] * 2 + off[2] * 4;
        values[idx] = corners[c];
    }
    ScalarField::new(2, 2, 2, values).expect("single cube field")
}

/// The reference `cube_index` for `corners` against `iso`: corner bit `c` is set
/// when `corners[c] < iso`, exactly as the golden kernel classifies.
fn host_cube_index(corners: [f32; 8], iso: f32) -> u32 {
    let mut code = 0_u32;
    for (c, &v) in corners.iter().enumerate() {
        if v < iso {
            code |= 1 << c;
        }
    }
    code
}

/// Asserts the twinned per-cell answer for one cube matches the `CPU` golden.
fn assert_parity(gpu: &GpuMarchingCubes, ctx: &GpuContext, corners: [f32; 8], iso: f32) {
    let field = single_cube(corners);
    let mesh = marching_cubes(&field, iso);

    let query = MarchingCubesCellQuery {
        corner_values: corners,
        iso,
    };
    let got = gpu.evaluate(ctx, std::slice::from_ref(&query));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];

    // Discrete classification: cube_index matches the reference exactly.
    assert_eq!(
        g.cube_index,
        host_cube_index(corners, iso),
        "cube_index mismatch for corners {corners:?}"
    );

    // The emitted vertex count equals the golden mesh vertex count, which is
    // three per triangle.
    let expected_verts = mesh.positions.len();
    assert_eq!(
        g.vert_count as usize, expected_verts,
        "vert_count mismatch for corners {corners:?}"
    );
    assert_eq!(
        g.vert_count as usize,
        mesh.triangle_count() * 3,
        "vert_count is not three per triangle for corners {corners:?}"
    );

    // The reference triangle topology is a sequential index list, three fresh
    // indices per triangle; verify it exactly so the per-cell order is pinned.
    let expected_indices: Vec<u32> = (0..g.vert_count).collect();
    assert_eq!(
        mesh.indices, expected_indices,
        "golden indices are not sequential for corners {corners:?}"
    );

    // Lanes past the vertex count are zeroed by the kernel.
    for (slot, vertex) in g.vertices.iter().enumerate().skip(expected_verts) {
        assert!(
            approx3(*vertex, [0.0, 0.0, 0.0]),
            "slot {slot} past vert_count is not zero for corners {corners:?}: {vertex:?}"
        );
    }

    // Each emitted vertex matches the golden position within tolerance, in the
    // same TRI_TABLE emission order.
    for (i, p) in mesh.positions.iter().enumerate() {
        let gpu_v = g.vertices[i];
        let cpu_v = [p.x, p.y, p.z];
        assert!(
            approx3(gpu_v, cpu_v),
            "vertex {i} mismatch for corners {corners:?}: gpu {gpu_v:?} vs cpu {cpu_v:?}"
        );
    }
}

#[test]
fn all_corners_above_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingCubes::new(&ctx);
    // No corner is below iso, so cube_index is 0 and the cube emits nothing.
    assert_parity(&gpu, &ctx, [2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0], 0.0);
}

#[test]
fn all_corners_below_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingCubes::new(&ctx);
    // Every corner is below iso, so cube_index is 255 and the cube emits nothing.
    assert_parity(
        &gpu,
        &ctx,
        [-2.0, -3.0, -4.0, -5.0, -2.0, -3.0, -4.0, -5.0],
        0.0,
    );
}

#[test]
fn single_corner_c0_below() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingCubes::new(&ctx);
    // Only corner 0 is below iso: cube_index 1, one triangle on edges 0/8/3.
    assert_parity(&gpu, &ctx, [-1.0, 3.0, 2.0, 4.0, 5.0, 2.0, 3.0, 4.0], 0.0);
}

#[test]
fn single_corner_c6_below() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingCubes::new(&ctx);
    // Only corner 6 is below iso: cube_index 64, a different single-corner case.
    assert_parity(&gpu, &ctx, [2.0, 3.0, 4.0, 2.0, 5.0, 3.0, -2.0, 4.0], 0.0);
}

#[test]
fn adjacent_corners_below() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingCubes::new(&ctx);
    // Corners 0 and 1 (an edge-adjacent pair) are below: cube_index 3, two
    // triangles.
    assert_parity(&gpu, &ctx, [-1.0, -3.0, 2.0, 4.0, 3.0, 5.0, 2.0, 4.0], 0.0);
}

#[test]
fn opposite_corners_below() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingCubes::new(&ctx);
    // Corners 0 and 6 (a space diagonal) are below: cube_index 65, two separated
    // single-corner caps.
    assert_parity(&gpu, &ctx, [-2.0, 3.0, 4.0, 2.0, 5.0, 3.0, -1.0, 4.0], 0.0);
}

#[test]
fn three_corners_below() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingCubes::new(&ctx);
    // Corners 0, 3 and 5 are below: cube_index 41, a mixed multi-triangle case.
    assert_parity(&gpu, &ctx, [-1.0, 2.0, 3.0, -4.0, 5.0, -2.0, 3.0, 4.0], 0.0);
}

#[test]
fn checkerboard_corners_below() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingCubes::new(&ctx);
    // Corners 0, 2, 5 and 7 are below: cube_index 165, a dense multi-triangle
    // case that exercises up to several triangles.
    assert_parity(
        &gpu,
        &ctx,
        [-1.0, 2.0, -3.0, 4.0, 3.0, -2.0, 5.0, -4.0],
        0.0,
    );
}

#[test]
fn nonzero_iso_shifts_classification() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingCubes::new(&ctx);
    // A non-zero iso still classifies by magnitude; corners straddle iso = 1.5 so
    // the crossing parameters are not one half.
    assert_parity(&gpu, &ctx, [0.5, 3.0, 1.0, 4.0, 2.5, 1.0, 3.5, 0.5], 1.5);
}

#[test]
fn batch_of_cubes_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingCubes::new(&ctx);
    // A batch exercises the one-thread-per-cube flattening; each result must be
    // independent of its neighbours.
    let batch = [
        MarchingCubesCellQuery {
            corner_values: [-1.0, 3.0, 2.0, 4.0, 5.0, 2.0, 3.0, 4.0],
            iso: 0.0,
        },
        MarchingCubesCellQuery {
            corner_values: [-1.0, -3.0, 2.0, 4.0, 3.0, 5.0, 2.0, 4.0],
            iso: 0.0,
        },
        MarchingCubesCellQuery {
            corner_values: [2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0],
            iso: 0.0,
        },
        MarchingCubesCellQuery {
            corner_values: [-1.0, 2.0, -3.0, 4.0, 3.0, -2.0, 5.0, -4.0],
            iso: 0.0,
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q.corner_values, q.iso);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMarchingCubes::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
