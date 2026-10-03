//! Real-device parity for the analytic box/triangle signed-distance twin:
//! [`GpuSdfBoxTriangle`](prism_volumetric_gpu::sdf_box_triangle::GpuSdfBoxTriangle)
//! must reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_primitives` — the exact
//! axis-aligned box distance `box_sdf` and the exact triangle distance
//! `triangle_sdf` — across interior, exterior and surface points, the three
//! triangle feature regions (face, edge, vertex) and a randomized sweep
//! compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed forms: the box
//! interior/exterior split and Inigo Quilez's exact `udTriangle`. Because the
//! reference and this oracle are both scalar `f32`, a `GPU == oracle` pass is
//! direct evidence the ported kernel computes the same distances the reference
//! does.
//!
//! # Parity criterion
//!
//! Both distances thread through products, quotients and a `sqrt`, so a `GPU`
//! result may land a few units in the last place from the scalar oracle; each
//! is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a relative
//! floor of `1e-6` so a near-zero expected value does not inflate the relative
//! error.
//!
//! # Conditioning
//!
//! The triangle distance is continuous across the face/edge classification
//! boundary — the perpendicular foot lands on an edge there, where both
//! branches agree — so a branch disagreement near that boundary cannot produce
//! a distance cliff. The face branch divides by the squared face normal, so the
//! randomized sweep rejects near-degenerate triangles (squared normal below a
//! comfortable floor) to keep that quotient well-conditioned; box queries have
//! no branch cliff at all.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene/sdf_primitives.rs`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_box_triangle::{
    GpuSdfBoxTriangle, SdfBoxTriangleQuery, SdfBoxTriangleResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on either distance. A `GPU` `sqrt`/divide may land a few
/// units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative bound on either distance, applied for larger magnitudes where a few
/// units in the last place exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound
/// (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Component-wise difference `a - b`.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Dot product of two 3-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Squared length of a 3-vector.
fn dot2(v: [f32; 3]) -> f32 {
    dot3(v, v)
}

/// Cross product `a x b`.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Rust `f32::signum` reimplemented to match the kernel's `select`-based sign:
/// `+1` for a non-negative argument, `-1` otherwise.
fn sign_of(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Independent reimplementation of
/// `ray_scene::sdf_primitives::box_sdf`: the positive exterior overshoot length
/// plus the clamped interior term.
fn box_sdf_host(point: [f32; 3], half_extent: [f32; 3]) -> f32 {
    let q = [
        point[0].abs() - half_extent[0],
        point[1].abs() - half_extent[1],
        point[2].abs() - half_extent[2],
    ];
    let ox = q[0].max(0.0);
    let oy = q[1].max(0.0);
    let oz = q[2].max(0.0);
    let outside = (ox * ox + oy * oy + oz * oz).sqrt();
    let inside = q[0].max(q[1].max(q[2])).min(0.0);
    outside + inside
}

/// Independent reimplementation of
/// `ray_scene::sdf_primitives::triangle_sdf`: Inigo Quilez's exact
/// `udTriangle`, classifying the query into the edge/vertex region or the
/// face-interior region before a single final `sqrt`.
fn triangle_sdf_host(point: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let ba = sub3(b, a);
    let pa = sub3(point, a);
    let cb = sub3(c, b);
    let pb = sub3(point, b);
    let ac = sub3(a, c);
    let pc = sub3(point, c);
    let nor = cross3(ba, ac);

    let edge_sum = sign_of(dot3(cross3(ba, nor), pa))
        + sign_of(dot3(cross3(cb, nor), pb))
        + sign_of(dot3(cross3(ac, nor), pc));

    let squared = if edge_sum < 2.0 {
        let t0 = (dot3(ba, pa) / dot2(ba)).clamp(0.0, 1.0);
        let e0 = dot2(sub3([ba[0] * t0, ba[1] * t0, ba[2] * t0], pa));
        let t1 = (dot3(cb, pb) / dot2(cb)).clamp(0.0, 1.0);
        let e1 = dot2(sub3([cb[0] * t1, cb[1] * t1, cb[2] * t1], pb));
        let t2 = (dot3(ac, pc) / dot2(ac)).clamp(0.0, 1.0);
        let e2 = dot2(sub3([ac[0] * t2, ac[1] * t2, ac[2] * t2], pc));
        e0.min(e1).min(e2)
    } else {
        let np = dot3(nor, pa);
        np * np / dot2(nor)
    };
    squared.sqrt()
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws an `f32` in `[lo, hi]` at ten-thousandth resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (lcg(state) % 10_001) as f32 / 10_000.0 * (hi - lo)
}

/// Builds one combined query.
fn query(
    point: [f32; 3],
    half_extent: [f32; 3],
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
) -> SdfBoxTriangleQuery {
    SdfBoxTriangleQuery {
        point,
        half_extent,
        a,
        b,
        c,
    }
}

/// Dispatches `queries` and asserts every distance matches the host oracles.
fn check_batch(ctx: &GpuContext, gpu: &GpuSdfBoxTriangle, queries: &[SdfBoxTriangleQuery]) {
    let got: Vec<SdfBoxTriangleResult> = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let golden_box = box_sdf_host(q.point, q.half_extent);
        let golden_tri = triangle_sdf_host(q.point, q.a, q.b, q.c);
        assert!(
            close(r.box_distance, golden_box),
            "box mismatch: gpu={} golden={} (point={:?} half_extent={:?})",
            r.box_distance,
            golden_box,
            q.point,
            q.half_extent
        );
        assert!(
            close(r.triangle_distance, golden_tri),
            "triangle mismatch: gpu={} golden={} (point={:?} a={:?} b={:?} c={:?})",
            r.triangle_distance,
            golden_tri,
            q.point,
            q.a,
            q.b,
            q.c
        );
    }
}

/// A fixed, well-conditioned reference triangle used by the box-focused
/// fixtures so the triangle output is also always exercised.
const TRI_A: [f32; 3] = [0.0, 0.0, 0.0];
const TRI_B: [f32; 3] = [2.0, 0.0, 0.0];
const TRI_C: [f32; 3] = [0.0, 2.0, 0.0];

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_box_triangle parity: no wgpu adapter available");
        return;
    };
    let gpu = GpuSdfBoxTriangle::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "empty batch yields no results");
}

#[test]
fn box_points_outside_faces() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBoxTriangle::new(&ctx);
    let he = [1.0, 1.0, 1.0];
    let queries = [
        query([2.0, 0.0, 0.0], he, TRI_A, TRI_B, TRI_C),
        query([0.0, 3.0, 0.0], he, TRI_A, TRI_B, TRI_C),
        query([0.0, 0.0, -2.5], he, TRI_A, TRI_B, TRI_C),
        query([1.7, 0.3, -0.4], he, TRI_A, TRI_B, TRI_C),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn box_points_outside_corners_and_edges() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBoxTriangle::new(&ctx);
    let he = [1.0, 0.5, 2.0];
    let queries = [
        query([2.0, 2.0, 4.0], he, TRI_A, TRI_B, TRI_C),
        query([2.0, 1.5, 0.0], he, TRI_A, TRI_B, TRI_C),
        query([-3.0, -2.0, 3.0], he, TRI_A, TRI_B, TRI_C),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn box_points_inside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBoxTriangle::new(&ctx);
    let he = [2.0, 1.0, 1.5];
    let queries = [
        query([0.0, 0.0, 0.0], he, TRI_A, TRI_B, TRI_C),
        query([1.0, 0.3, -0.5], he, TRI_A, TRI_B, TRI_C),
        query([-1.5, -0.6, 1.0], he, TRI_A, TRI_B, TRI_C),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn triangle_face_region_perpendicular() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBoxTriangle::new(&ctx);
    let he = [1.0, 1.0, 1.0];
    // Points above/below the triangle interior project perpendicularly onto the
    // plane, so the face branch governs.
    let queries = [
        query([0.5, 0.5, 1.3], he, TRI_A, TRI_B, TRI_C),
        query([0.6, 0.4, -2.0], he, TRI_A, TRI_B, TRI_C),
        query([0.3, 0.3, 0.75], he, TRI_A, TRI_B, TRI_C),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn triangle_edge_region() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBoxTriangle::new(&ctx);
    let he = [1.0, 1.0, 1.0];
    // Points beyond an edge midpoint (foot lands on the open edge) exercise the
    // edge branch.
    let queries = [
        query([1.5, -0.8, 0.4], he, TRI_A, TRI_B, TRI_C),
        query([-0.8, 1.2, -0.3], he, TRI_A, TRI_B, TRI_C),
        query([1.4, 1.4, 0.2], he, TRI_A, TRI_B, TRI_C),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn triangle_vertex_region() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBoxTriangle::new(&ctx);
    let he = [1.0, 1.0, 1.0];
    // Points beyond a vertex (edge clamps saturate) collapse to the vertex.
    let queries = [
        query([-1.0, -1.0, 0.5], he, TRI_A, TRI_B, TRI_C),
        query([3.2, -0.9, -0.4], he, TRI_A, TRI_B, TRI_C),
        query([-0.9, 3.1, 0.6], he, TRI_A, TRI_B, TRI_C),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn triangle_tilted_out_of_plane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBoxTriangle::new(&ctx);
    let he = [1.0, 1.0, 1.0];
    let a = [-1.0, 0.0, 0.0];
    let b = [1.0, 0.5, 0.3];
    let c = [0.0, 1.5, -0.4];
    let queries = [
        query([0.0, 0.5, 1.2], he, a, b, c),
        query([-2.0, -1.0, 0.0], he, a, b, c),
        query([0.2, 0.6, -0.1], he, a, b, c),
        query([2.5, 0.3, 0.3], he, a, b, c),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn mixed_single_dispatch_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBoxTriangle::new(&ctx);
    let queries = [
        query([2.0, 0.0, 0.0], [1.0, 1.0, 1.0], TRI_A, TRI_B, TRI_C),
        query([0.0, 0.0, 0.0], [2.0, 1.0, 1.5], TRI_A, TRI_B, TRI_C),
        query(
            [0.5, 0.5, 1.3],
            [1.0, 0.5, 2.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.5, 0.3],
            [0.0, 1.5, -0.4],
        ),
        query([-1.0, -1.0, 0.5], [0.4, 0.4, 0.4], TRI_A, TRI_B, TRI_C),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn randomized_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfBoxTriangle::new(&ctx);
    let mut state: u64 = 0x51f0_6a3c_9d12_7e55;
    let mut queries = Vec::with_capacity(384);
    while queries.len() < 384 {
        let point = [
            draw(&mut state, -4.0, 4.0),
            draw(&mut state, -4.0, 4.0),
            draw(&mut state, -4.0, 4.0),
        ];
        let half_extent = [
            draw(&mut state, 0.2, 3.0),
            draw(&mut state, 0.2, 3.0),
            draw(&mut state, 0.2, 3.0),
        ];
        let a = [
            draw(&mut state, -3.0, 3.0),
            draw(&mut state, -3.0, 3.0),
            draw(&mut state, -3.0, 3.0),
        ];
        let b = [
            draw(&mut state, -3.0, 3.0),
            draw(&mut state, -3.0, 3.0),
            draw(&mut state, -3.0, 3.0),
        ];
        let c = [
            draw(&mut state, -3.0, 3.0),
            draw(&mut state, -3.0, 3.0),
            draw(&mut state, -3.0, 3.0),
        ];
        // Reject near-degenerate triangles so the face branch's divide by the
        // squared normal stays well-conditioned.
        let ba = sub3(b, a);
        let ac = sub3(a, c);
        let nor = cross3(ba, ac);
        if dot2(nor) < 1.0 {
            continue;
        }
        queries.push(query(point, half_extent, a, b, c));
    }
    check_batch(&ctx, &gpu, &queries);
}
