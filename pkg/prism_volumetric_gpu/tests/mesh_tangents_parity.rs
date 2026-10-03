//! Real-device parity for the per-triangle tangent-frame twin:
//! [`GpuMeshTriangleTangent`](prism_volumetric_gpu::mesh_tangents::GpuMeshTriangleTangent)
//! must reproduce the closed-form reference answer the golden
//! `compute_tangents` applies once per triangle: solve the `2x2` `UV` edge
//! system for the surface tangent and bitangent, then per vertex normalize the
//! shading normal, Gram-Schmidt the accumulated tangent against it and record
//! the handedness `w`, with a degenerate `UV` determinant falling back to a
//! deterministic basis.
//!
//! # Independent oracle
//!
//! This suite does not depend on the reference crate. The host [`oracle`] is an
//! independent reimplementation of the same per-triangle closed form
//! documented on the twin (`det = du1*dv2 - du2*dv1`, the guarded `1/det`
//! tangent and bitangent, the per-vertex Gram-Schmidt and handedness sign, and
//! the [`fallback_tangent`] world-axis selection). A passing run is therefore
//! evidence that the `WGSL` kernel and an independent `CPU` evaluation of the
//! same frame agree, not merely that the shader compiles.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and a
//! handful of `sqrt`s, so the two evaluations compute the same closed form in
//! the same order. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar host leaves separate, perturbing the low mantissa bits by a few units
//! in the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on the tangent `xyz` components while
//! pinning the `degenerate` flag and the handedness sign exactly.
//!
//! # Conditioning
//!
//! Every random fixture is rejection-sampled well away from each branch
//! boundary: the `UV` determinant stays far from the `1e-20` guard, the tangent
//! and bitangent are comfortably non-tiny, the Gram-Schmidt residual stays far
//! above the `1e-16` collapse threshold, and the handedness dot keeps a healthy
//! margin from zero, so `CPU` and `GPU` land on the same side of every branch
//! and both the `degenerate` flag and the sign of `w` match regardless of a few
//! units in the last place. The named degenerate fixtures instead pick normals
//! with clearly separated axis magnitudes so the fallback axis choice is
//! unambiguous on both evaluators.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_tangents`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_tangents::{
    GpuMeshTriangleTangent, MeshTriangleTangentQuery, MeshTriangleTangentResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity floor on the tangent `xyz` components.
const DIST_ABS: f32 = 1.0e-4;
/// Relative parity slope on the tangent `xyz` components.
const DIST_REL: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero magnitudes stay meaningful.
const REL_FLOOR: f32 = 1.0e-6;
/// Determinant guard matching the reference `1e-20` degenerate threshold.
const DET_EPS: f32 = 1.0e-20;
/// Squared-length guard matching the reference `1e-24` normalization floor.
const LEN2_EPS: f32 = 1.0e-24;
/// Gram-Schmidt residual guard matching the reference `1e-16` collapse floor.
const ORTHO_EPS: f32 = 1.0e-16;

/// Returns whether `a` and `b` agree within the documented parity bound: an
/// absolute floor or a relative term keeping large-magnitude values meaningful.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= DIST_ABS || diff <= DIST_REL * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Component-wise difference `a - b` over a 3-vector.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Dot product of two 3-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a x b` of two 3-vectors.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalizes `v`, returning `fallback` when `v` is too short to normalize
/// (only `sqrt` is used, no transcendental), mirroring the reference guard.
fn normalize_or3(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len2 = dot3(v, v);
    if len2 > LEN2_EPS {
        let inv = 1.0 / len2.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        fallback
    }
}

/// Deterministic unit tangent orthogonal to `n`: pick the world axis least
/// aligned with `n` and project it onto the tangent plane, reimplementing the
/// reference `fallback_tangent`.
fn fallback_tangent(n: [f32; 3]) -> [f32; 3] {
    let ax = n[0].abs();
    let ay = n[1].abs();
    let az = n[2].abs();
    let axis = if ax <= ay && ax <= az {
        [1.0, 0.0, 0.0]
    } else if ay <= az {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let d = dot3(axis, n);
    let ortho = [axis[0] - n[0] * d, axis[1] - n[1] * d, axis[2] - n[2] * d];
    normalize_or3(ortho, [1.0, 0.0, 0.0])
}

/// Gram-Schmidt the accumulated tangent `t_dir` against the vertex normal, then
/// record the handedness against the accumulated bitangent `b_dir`, returning
/// `[tx, ty, tz, w]`.
fn vertex_tangent(t_dir: [f32; 3], b_dir: [f32; 3], raw_n: [f32; 3]) -> [f32; 4] {
    let n = normalize_or3(raw_n, [0.0, 0.0, 1.0]);
    let acc = t_dir;
    let ndt = dot3(n, acc);
    let ortho = [
        acc[0] - n[0] * ndt,
        acc[1] - n[1] * ndt,
        acc[2] - n[2] * ndt,
    ];
    let tangent = if dot3(ortho, ortho) > ORTHO_EPS {
        normalize_or3(ortho, fallback_tangent(n))
    } else {
        fallback_tangent(n)
    };
    let w = if dot3(cross3(n, tangent), b_dir) < 0.0 {
        -1.0
    } else {
        1.0
    };
    [tangent[0], tangent[1], tangent[2], w]
}

/// Independent `CPU` reference: reproduces the golden per-triangle tangent frame
/// for one triangle treated as the sole contributor to each of its vertices.
fn oracle(q: &MeshTriangleTangentQuery) -> MeshTriangleTangentResult {
    let e1 = sub3(q.p1, q.p0);
    let e2 = sub3(q.p2, q.p0);
    let du1 = q.uv1[0] - q.uv0[0];
    let dv1 = q.uv1[1] - q.uv0[1];
    let du2 = q.uv2[0] - q.uv0[0];
    let dv2 = q.uv2[1] - q.uv0[1];
    let det = du1 * dv2 - du2 * dv1;
    let (t_dir, b_dir, degenerate) = if det.abs() <= DET_EPS {
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1u32)
    } else {
        let r = 1.0 / det;
        let t = [
            (e1[0] * dv2 - e2[0] * dv1) * r,
            (e1[1] * dv2 - e2[1] * dv1) * r,
            (e1[2] * dv2 - e2[2] * dv1) * r,
        ];
        let b = [
            (e2[0] * du1 - e1[0] * du2) * r,
            (e2[1] * du1 - e1[1] * du2) * r,
            (e2[2] * du1 - e1[2] * du2) * r,
        ];
        (t, b, 0u32)
    };
    MeshTriangleTangentResult {
        tangent0: vertex_tangent(t_dir, b_dir, q.n0),
        tangent1: vertex_tangent(t_dir, b_dir, q.n1),
        tangent2: vertex_tangent(t_dir, b_dir, q.n2),
        degenerate,
    }
}

/// 64-bit linear-congruential step (`PCG`/`Knuth` constants), returning the
/// high word so the stream has good spread without any transcendental math.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// A deterministic pseudo-random `f32` in `[0, 1]`.
fn unit01(state: &mut u64) -> f32 {
    lcg(state) as f32 / u32::MAX as f32
}

/// A deterministic pseudo-random `f32` in `[-scale, scale]`.
fn signed(state: &mut u64, scale: f32) -> f32 {
    (unit01(state) * 2.0 - 1.0) * scale
}

/// A deterministic pseudo-random unit direction, rejection-sampled away from
/// the origin so the normalization never hits the fallback.
fn rand_unit(state: &mut u64) -> [f32; 3] {
    loop {
        let v = [signed(state, 1.0), signed(state, 1.0), signed(state, 1.0)];
        let len2 = dot3(v, v);
        if len2 > 0.1 {
            let inv = 1.0 / len2.sqrt();
            return [v[0] * inv, v[1] * inv, v[2] * inv];
        }
    }
}

/// Returns whether `q` is clear of every branch boundary that could flip the
/// `degenerate` flag or the handedness sign between the two evaluators.
fn well_conditioned(q: &MeshTriangleTangentQuery) -> bool {
    let e1 = sub3(q.p1, q.p0);
    let e2 = sub3(q.p2, q.p0);
    let du1 = q.uv1[0] - q.uv0[0];
    let dv1 = q.uv1[1] - q.uv0[1];
    let du2 = q.uv2[0] - q.uv0[0];
    let dv2 = q.uv2[1] - q.uv0[1];
    let det = du1 * dv2 - du2 * dv1;
    if det.abs() < 0.05 {
        return false;
    }
    let r = 1.0 / det;
    let t_dir = [
        (e1[0] * dv2 - e2[0] * dv1) * r,
        (e1[1] * dv2 - e2[1] * dv1) * r,
        (e1[2] * dv2 - e2[2] * dv1) * r,
    ];
    let b_dir = [
        (e2[0] * du1 - e1[0] * du2) * r,
        (e2[1] * du1 - e1[1] * du2) * r,
        (e2[2] * du1 - e1[2] * du2) * r,
    ];
    let blen2 = dot3(b_dir, b_dir);
    if blen2 < 0.04 || dot3(t_dir, t_dir) < 0.04 {
        return false;
    }
    let blen = blen2.sqrt();
    for n in [q.n0, q.n1, q.n2] {
        let nn = normalize_or3(n, [0.0, 0.0, 1.0]);
        let ndt = dot3(nn, t_dir);
        let ortho = [
            t_dir[0] - nn[0] * ndt,
            t_dir[1] - nn[1] * ndt,
            t_dir[2] - nn[2] * ndt,
        ];
        let oo = dot3(ortho, ortho);
        if oo < 0.04 {
            return false;
        }
        let inv = 1.0 / oo.sqrt();
        let tangent = [ortho[0] * inv, ortho[1] * inv, ortho[2] * inv];
        let hdot = dot3(cross3(nn, tangent), b_dir);
        if hdot.abs() < 0.15 * blen {
            return false;
        }
    }
    true
}

/// Draws one well-conditioned random triangle query via rejection sampling.
fn rand_triangle(state: &mut u64) -> MeshTriangleTangentQuery {
    loop {
        let q = MeshTriangleTangentQuery::new(
            [
                signed(state, 20.0),
                signed(state, 20.0),
                signed(state, 20.0),
            ],
            [
                signed(state, 20.0),
                signed(state, 20.0),
                signed(state, 20.0),
            ],
            [
                signed(state, 20.0),
                signed(state, 20.0),
                signed(state, 20.0),
            ],
            [signed(state, 4.0), signed(state, 4.0)],
            [signed(state, 4.0), signed(state, 4.0)],
            [signed(state, 4.0), signed(state, 4.0)],
            rand_unit(state),
            rand_unit(state),
            rand_unit(state),
        );
        if well_conditioned(&q) {
            return q;
        }
    }
}

/// Pins one `GPU` result against the independent oracle: the `degenerate` flag
/// and each vertex's handedness sign match exactly, and the tangent `xyz`
/// components match within the documented tolerance.
fn pin(idx: usize, query: &MeshTriangleTangentQuery, got: &MeshTriangleTangentResult) {
    let want = oracle(query);
    assert_eq!(
        got.degenerate, want.degenerate,
        "query {idx}: degenerate flag must match the oracle exactly"
    );
    let pairs = [
        (got.tangent0, want.tangent0),
        (got.tangent1, want.tangent1),
        (got.tangent2, want.tangent2),
    ];
    for (vtx, (g, w)) in pairs.iter().enumerate() {
        assert_eq!(
            g[3] > 0.0,
            w[3] > 0.0,
            "query {idx} vertex {vtx}: handedness sign must match the oracle exactly"
        );
        for axis in 0..3 {
            assert!(
                close(g[axis], w[axis]),
                "query {idx} vertex {vtx} axis {axis}: tangent {} vs oracle {}",
                g[axis],
                w[axis]
            );
        }
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuMeshTriangleTangent, queries: &[MeshTriangleTangentQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

/// Standard `UV` unwrap: a right triangle in the `z = 0` plane with `u` -> `+X`,
/// `v` -> `+Y` and `+Z` normals, so the tangent is `+X` and the handedness
/// `w` is `+1`.
fn standard_unwrap() -> MeshTriangleTangentQuery {
    MeshTriangleTangentQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    )
}

/// Flipped `UV`: the same geometry with the third texture coordinate mirrored
/// along `v`, so the bitangent reverses and the handedness `w` is `-1`.
fn flipped_uv() -> MeshTriangleTangentQuery {
    MeshTriangleTangentQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    )
}

/// Degenerate `UV`: three collinear texture coordinates collapse the
/// determinant, so the triangle contributes nothing and every vertex falls back
/// to the deterministic basis with `degenerate = 1`.
fn degenerate_uv() -> MeshTriangleTangentQuery {
    MeshTriangleTangentQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    )
}

/// Degenerate `UV` with three normals whose dominant axes are clearly
/// separated, so the fallback basis selects the `X`, `Y` and `Z` world axes
/// respectively on both evaluators without ambiguity.
fn fallback_axes() -> MeshTriangleTangentQuery {
    let n2 = normalize_or3([0.6, 0.7, 0.1], [0.0, 0.0, 1.0]);
    MeshTriangleTangentQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        n2,
    )
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleTangent::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn standard_unwrap_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleTangent::new(&ctx);
    check(&ctx, &gpu, &[standard_unwrap()]);
}

#[test]
fn flipped_uv_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleTangent::new(&ctx);
    check(&ctx, &gpu, &[flipped_uv()]);
}

#[test]
fn degenerate_uv_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleTangent::new(&ctx);
    let queries = [degenerate_uv(), fallback_axes()];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleTangent::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random triangles,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element for element.
    let mut queries = vec![
        standard_unwrap(),
        flipped_uv(),
        degenerate_uv(),
        fallback_axes(),
    ];
    for _ in 0..48 {
        queries.push(rand_triangle(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleTangent::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned random triangles (several
    // workgroups' worth) pins the three per-vertex tangents and handedness
    // across many random frames.
    let queries: Vec<MeshTriangleTangentQuery> =
        (0..512).map(|_| rand_triangle(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
