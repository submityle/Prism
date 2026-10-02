//! Real-device parity for the triangle-versus-triangle intersection twin:
//! [`GpuTriTriIntersect`](prism_volumetric_gpu::tri_tri_intersect::GpuTriTriIntersect)
//! must reproduce the `CPU` golden
//! [`tri_tri_intersect`](prism_render_architecture::particle::tri_tri_intersect::tri_tri_intersect)
//! element for element across interior crossings, shared edges and vertices, a
//! face-piercing triangle, parallel and offset separations, a perpendicular
//! disjoint-interval miss, coplanar edge touches and coplanar separations, a
//! near-miss just beyond the plane, degenerate triangles (collapsed to a segment
//! and to a point), and a randomized batch of clearly-conditioned intersecting
//! and clearly-separated pairs compared one by one.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The output is a pure boolean, so the comparison is an exact `==` on every
//! element with no tolerance: any mismatch is a genuine port bug. The reference
//! compares `f32` signed distances, orientation determinants and projected
//! coordinates against its `EPS` slack rather than `==`, so a `GPU` fusing a
//! multiply-add perturbs a quantity by a few units in the last place but cannot
//! flip a decision as long as every fixture stays clearly on one side of each
//! comparison.
//!
//! # Conditioning
//!
//! Every fixture is deliberately far from a decision boundary: intersecting
//! fixtures cross or touch with a wide margin, separated fixtures leave a gap
//! well beyond `EPS`, and the randomized batch either rigidly transforms a known
//! crossing pair (preserving intersection) or shoves one triangle far away
//! (guaranteeing separation). This keeps `CPU` and `GPU` on the same side of
//! every `EPS` comparison regardless of a few units in the last place of slack.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::tri_tri_intersect`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::tri_tri_intersect::{tri_tri_intersect, Tri, Vec3};
use prism_volumetric_gpu::tri_tri_intersect::{GpuTriTriIntersect, TriTriQuery};
use prism_volumetric_gpu::GpuContext;

/// Converts a corner-triple into the reference [`Tri`].
fn to_tri(t: [[f32; 3]; 3]) -> Tri {
    Tri::new(
        Vec3::new(t[0][0], t[0][1], t[0][2]),
        Vec3::new(t[1][0], t[1][1], t[1][2]),
        Vec3::new(t[2][0], t[2][1], t[2][2]),
    )
}

/// The `CPU` golden boolean for one query.
fn cpu(query: &TriTriQuery) -> bool {
    tri_tri_intersect(&to_tri(query.tri_a), &to_tri(query.tri_b))
}

/// Dispatches `queries` on-device and asserts an exact boolean match against the
/// reference for every element.
fn check(ctx: &GpuContext, gpu: &GpuTriTriIntersect, queries: &[TriTriQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len());
    for (idx, query) in queries.iter().enumerate() {
        assert_eq!(
            got[idx],
            cpu(query),
            "intersection mismatch at {idx}: tri_a {:?} tri_b {:?}",
            query.tri_a,
            query.tri_b
        );
    }
}

/// Translates every corner of a triangle by `o`.
fn shift(t: [[f32; 3]; 3], o: [f32; 3]) -> [[f32; 3]; 3] {
    let mut out = t;
    for v in &mut out {
        v[0] += o[0];
        v[1] += o[1];
        v[2] += o[2];
    }
    out
}

/// Rigidly scales a triangle about the origin by `s`, then translates it by `o`.
fn transform(t: [[f32; 3]; 3], s: f32, o: [f32; 3]) -> [[f32; 3]; 3] {
    let mut out = t;
    for v in &mut out {
        v[0] = v[0] * s + o[0];
        v[1] = v[1] * s + o[1];
        v[2] = v[2] * s + o[2];
    }
    out
}

/// A unit right triangle in the `z = 0` plane.
const BASE: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

/// The first triangle of the canonical perpendicular-crossing pair.
const CROSS_A: [[f32; 3]; 3] = [[0.0, -1.0, 0.0], [2.0, -1.0, 0.0], [1.0, 1.0, 0.0]];

/// The second triangle of the canonical perpendicular-crossing pair.
const CROSS_B: [[f32; 3]; 3] = [[0.0, 0.0, -1.0], [2.0, 0.0, -1.0], [1.0, 0.0, 1.0]];

/// Curated hand fixtures: unambiguous crossings, touches, and separations lifted
/// from the reference contract, all clearly off every decision boundary.
fn hand_fixtures() -> Vec<TriTriQuery> {
    vec![
        // Identical triangles fully overlap.
        TriTriQuery::new(BASE, BASE),
        // Shared edge folded up into the y = 0 plane.
        TriTriQuery::new(BASE, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        // Shared single vertex at the origin.
        TriTriQuery::new(BASE, [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        // One triangle pierces the other's interior.
        TriTriQuery::new(
            [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ),
        // Perpendicular crossing with overlapping intervals.
        TriTriQuery::new(CROSS_A, CROSS_B),
        // Partial edge overlap on the x-axis sub-range.
        TriTriQuery::new(
            CROSS_A,
            [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        ),
        // Coplanar triangles sharing a full edge.
        TriTriQuery::new(BASE, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, -1.0, 0.0]]),
        // Coplanar triangles sharing only a vertex.
        TriTriQuery::new(BASE, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
        // Far from the origin, still crossing.
        TriTriQuery::new(
            shift(CROSS_A, [1000.0, -1000.0, 500.0]),
            shift(CROSS_B, [1000.0, -1000.0, 500.0]),
        ),
        // Tiny scaled crossing pair.
        TriTriQuery::new(
            transform(CROSS_A, 1.0e-2, [0.0; 3]),
            transform(CROSS_B, 1.0e-2, [0.0; 3]),
        ),
        // Far apart on all axes.
        TriTriQuery::new(BASE, shift(BASE, [100.0, 100.0, 100.0])),
        // Coplanar but far apart.
        TriTriQuery::new(BASE, shift(BASE, [10.0, 0.0, 0.0])),
        // Parallel planes three units apart.
        TriTriQuery::new(BASE, shift(BASE, [0.0, 0.0, 3.0])),
        // Parallel planes one unit apart, offset in xy.
        TriTriQuery::new(BASE, shift(BASE, [0.25, 0.25, 1.0])),
        // Perpendicular but disjoint intervals (t2 shifted along x).
        TriTriQuery::new(CROSS_A, shift(CROSS_B, [5.0, 0.0, 0.0])),
        // One vertex on the plane, the rest separated far away.
        TriTriQuery::new(BASE, [[5.0, 5.0, 0.0], [6.0, 5.0, 1.0], [5.0, 6.0, 1.0]]),
        // Near miss just beyond the plane (0.01 > EPS).
        TriTriQuery::new(BASE, shift(BASE, [0.0, 0.0, 0.01])),
    ]
}

/// Degenerate-geometry fixtures: a triangle collapsed to a point and one
/// collapsed to a collinear segment both lack a supporting plane and report
/// `false` on both sides.
fn degenerate_fixtures() -> Vec<TriTriQuery> {
    vec![
        // First triangle collapsed to a point.
        TriTriQuery::new([[1.0, 1.0, 1.0]; 3], BASE),
        // Second triangle collapsed to a collinear segment.
        TriTriQuery::new(BASE, [[0.2, 0.2, 0.0], [0.4, 0.4, 0.0], [0.6, 0.6, 0.0]]),
        // Both triangles collapsed to points at the same spot.
        TriTriQuery::new([[0.5, 0.5, 0.5]; 3], [[0.5, 0.5, 0.5]; 3]),
    ]
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A clearly-intersecting fixture: the canonical crossing pair rigidly scaled by
/// a positive factor and translated by a shared offset, which preserves the
/// intersection with a wide margin.
fn intersecting_query(state: &mut u64) -> TriTriQuery {
    let s = 0.5 + lcg(state) * 1.5;
    let o = [
        signed(state, 20.0),
        signed(state, 20.0),
        signed(state, 20.0),
    ];
    TriTriQuery::new(transform(CROSS_A, s, o), transform(CROSS_B, s, o))
}

/// A clearly-separated fixture: a scaled triangle near the origin and a second
/// scaled triangle shoved far away, so the two cannot meet on any axis.
fn separated_query(state: &mut u64) -> TriTriQuery {
    let s = 0.5 + lcg(state) * 1.5;
    let shift_mag = 40.0 + lcg(state) * 20.0;
    let dir = [
        signed(state, 1.0).signum() * shift_mag,
        signed(state, 1.0).signum() * shift_mag,
        signed(state, 1.0).signum() * shift_mag,
    ];
    TriTriQuery::new(transform(CROSS_A, s, [0.0; 3]), transform(CROSS_B, s, dir))
}

#[test]
fn hand_fixtures_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriTriIntersect::new(&ctx);
    check(&ctx, &gpu, &hand_fixtures());
}

#[test]
fn degenerate_fixtures_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriTriIntersect::new(&ctx);
    check(&ctx, &gpu, &degenerate_fixtures());
}

#[test]
fn randomized_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriTriIntersect::new(&ctx);
    let mut state = 0x0f0e_0d0c_0b0a_0908u64;
    let mut queries = Vec::new();
    for _ in 0..128 {
        queries.push(intersecting_query(&mut state));
        queries.push(separated_query(&mut state));
    }
    // Sanity: the batch exercises both outcomes.
    assert!(queries.iter().any(cpu));
    assert!(queries.iter().any(|q| !cpu(q)));
    check(&ctx, &gpu, &queries);
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriTriIntersect::new(&ctx);
    // No dispatch is issued and the entry point returns an empty vector.
    assert!(gpu.eval(&ctx, &[]).is_empty());
}
