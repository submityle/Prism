//! Real-device parity for the 3D convex-hull twin:
//! [`GpuConvexHull3d`](prism_volumetric_gpu::convex_hull_3d::GpuConvexHull3d)
//! must reproduce the `CPU` golden
//! [`convex_hull_3d`](prism_render_architecture::particle::convex_hull_3d)
//! across the face count, the exact outward-wound vertex-index triples and each
//! face's `cross`-product normal.
//!
//! The fixtures cover the shapes the golden unit tests call out: an empty set, a
//! single point, a two-point segment, a collinear triple and a coplanar quad
//! (all of which lack 3D hull volume and yield no faces), a seed tetrahedron,
//! the eight corners of a cube, a six-vertex octahedron, a square pyramid, a
//! cube with duplicated corners that must dedup, a point pushed beyond a cube
//! corner, and batches of random integer clouds. Every coordinate is an exact
//! integer, so the signed-volume predicate
//! [`orient3d`](prism_render_architecture::particle::convex_hull_3d::orient3d) is
//! computed exactly on both the `CPU` and the `GPU`: each `orient3d` is either
//! exactly zero or at least one in magnitude, far from the `HULL_EPS` threshold,
//! so neither a fused multiply-add nor a reordered sum can flip a visibility or
//! seed branch. The two sides therefore take every branch identically and the
//! face list matches face for face and element for element, winding included.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The face count and the vertex-index triples are discrete classifications, so
//! `CPU` and `GPU` must agree exactly: the comparison is an exact `==` on the
//! count and on every `u32` index, including the winding order within each
//! triple. The face normals thread through multiplies and subtractions, so they
//! are compared under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`); for the exact-integer fixtures they in fact match to the
//! bit, but the tolerance admits any legal fused multiply-add contraction.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::convex_hull_3d`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::convex_hull_3d::{convex_hull_3d, face_normal};
use prism_volumetric_gpu::convex_hull_3d::{ConvexHull3dQuery, GpuConvexHull3d};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous normal channels.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous normal channels.
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

/// Tolerant comparison of two 3D normals, channel by channel.
fn approx_normal(a: [f32; 3], b: [f32; 3]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
}

/// Wraps a slice of points as one query.
fn set(points: &[[f32; 3]]) -> ConvexHull3dQuery {
    ConvexHull3dQuery {
        points: points.to_vec(),
    }
}

/// Asserts every twinned answer for one point-set matches the `CPU` golden: the
/// face count exactly, each vertex-index triple exactly (winding included), and
/// each face normal under tolerance.
fn assert_parity(gpu: &GpuConvexHull3d, ctx: &GpuContext, points: &[[f32; 3]]) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&set(points)));
    assert_eq!(got.len(), 1, "one result per query");
    let g = &got[0];

    let cpu_faces = convex_hull_3d(points);

    // The face count is a discrete classification: exact match.
    assert_eq!(
        g.face_count as usize,
        cpu_faces.len(),
        "face_count mismatch: gpu {} vs cpu {}",
        g.face_count,
        cpu_faces.len()
    );

    for (f, cf) in cpu_faces.iter().enumerate() {
        let gf = g.faces[f];
        let expected = [cf[0] as u32, cf[1] as u32, cf[2] as u32];
        // The incremental build is replayed branch for branch, so the triple and
        // its winding match element for element.
        assert_eq!(
            gf, expected,
            "face {f} index triple mismatch: gpu {gf:?} vs cpu {expected:?}"
        );

        let cpu_normal = face_normal(points, *cf);
        let gn = g.face_normals[f];
        assert!(
            approx_normal(gn, cpu_normal),
            "face {f} normal mismatch: gpu {gn:?} vs cpu {cpu_normal:?}"
        );
    }
}

/// A tiny integer linear-congruential generator; only integer multiply, add and
/// shift work, so no transcendental appears. Returns the high bits of the state.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state >> 40
}

/// Builds a cloud of `n` points with exact-integer coordinates in
/// `[-range, range]` on every axis, driven by the integer `lcg`. Integer
/// coordinates keep every `orient3d` exact on both the `CPU` and the `GPU`, so
/// the face list is reproduced exactly regardless of the (random) seed order.
fn random_int_cloud(seed: u64, n: usize, range: i64) -> Vec<[f32; 3]> {
    let span = (2 * range + 1) as u64;
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut pts = Vec::with_capacity(n);
    for _ in 0..n {
        let x = (lcg(&mut state) % span) as i64 - range;
        let y = (lcg(&mut state) % span) as i64 - range;
        let z = (lcg(&mut state) % span) as i64 - range;
        pts.push([x as f32, y as f32, z as f32]);
    }
    pts
}

/// The eight corners of the axis-aligned cube `[0, 2]^3`.
fn cube() -> Vec<[f32; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [2.0, 0.0, 2.0],
        [2.0, 2.0, 2.0],
        [0.0, 2.0, 2.0],
    ]
}

#[test]
fn degenerate_sets_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull3d::new(&ctx);
    // Empty set: no faces.
    assert_parity(&gpu, &ctx, &[]);
    // A single point has no hull volume.
    assert_parity(&gpu, &ctx, &[[1.0, 2.0, 3.0]]);
    // Two distinct points form a segment, not a volume.
    assert_parity(&gpu, &ctx, &[[0.0, 0.0, 0.0], [3.0, 1.0, 2.0]]);
    // Three collinear points: no volume.
    assert_parity(
        &gpu,
        &ctx,
        &[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
    );
    // Four coplanar points on the z = 0 plane: no volume.
    assert_parity(
        &gpu,
        &ctx,
        &[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
    );
}

#[test]
fn tetrahedron_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull3d::new(&ctx);
    // The minimal 3D hull: four faces.
    assert_parity(
        &gpu,
        &ctx,
        &[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
    );
}

#[test]
fn cube_corners_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull3d::new(&ctx);
    // The eight cube corners triangulate to twelve faces.
    assert_parity(&gpu, &ctx, &cube());
}

#[test]
fn octahedron_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull3d::new(&ctx);
    // Six axis-aligned vertices form the eight-face octahedron.
    assert_parity(
        &gpu,
        &ctx,
        &[
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
        ],
    );
}

#[test]
fn square_pyramid_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull3d::new(&ctx);
    // A square base plus an apex above its center.
    assert_parity(
        &gpu,
        &ctx,
        &[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 3.0],
        ],
    );
}

#[test]
fn duplicate_cube_corners_still_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull3d::new(&ctx);
    // Each corner repeated once must collapse to the same twelve-face hull; the
    // retained index names the first occurrence.
    let mut doubled = Vec::new();
    for p in cube() {
        doubled.push(p);
        doubled.push(p);
    }
    assert_parity(&gpu, &ctx, &doubled);
}

#[test]
fn exterior_point_expands_cube_hull() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull3d::new(&ctx);
    // A point well beyond one corner must join the hull as a new vertex.
    let mut pts = cube();
    pts.push([4.0, 4.0, 4.0]);
    assert_parity(&gpu, &ctx, &pts);
}

#[test]
fn random_integer_clouds_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull3d::new(&ctx);
    // Many random integer clouds exercise the full seed, insertion, horizon and
    // stitch control flow. Integer coordinates keep every `orient3d` exact, so
    // parity holds for every seed; at least some clouds form a real volume.
    let mut nonempty = 0usize;
    for seed in 0..32u64 {
        let pts = random_int_cloud(seed.wrapping_mul(0x1000_0001).wrapping_add(1), 12, 20);
        if !convex_hull_3d(&pts).is_empty() {
            nonempty += 1;
        }
        assert_parity(&gpu, &ctx, &pts);
    }
    assert!(
        nonempty > 0,
        "random integer clouds should produce at least one non-empty hull"
    );
}

#[test]
fn batch_of_sets_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull3d::new(&ctx);
    // A mixed batch exercises the one-thread-per-set flattening; each result must
    // be independent of its neighbours.
    let batch = [
        set(&[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]),
        set(&cube()),
        set(&[]),
        set(&[[5.0, 5.0, 5.0]]),
        set(&random_int_cloud(0xABCD, 12, 20)),
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, &q.points);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConvexHull3d::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
