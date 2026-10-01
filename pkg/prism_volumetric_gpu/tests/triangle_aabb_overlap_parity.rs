//! Real-device parity for the triangle-versus-axis-aligned-box overlap twin:
//! [`GpuTriangleAabbOverlap`](prism_volumetric_gpu::triangle_aabb_overlap::GpuTriangleAabbOverlap)
//! must reproduce the `CPU` golden
//! [`triangle_overlaps_aabb`](prism_render_architecture::particle::triangle_aabb_overlap::triangle_overlaps_aabb)
//! element for element across interior overlaps, each-face separations, a
//! face-crossing triangle, vertex/edge/corner touches, an edge-cross separation,
//! a box-enclosing triangle, degenerate triangles (collapsed to a segment and to
//! a point), a zero-volume box, a thin non-cubic slab, an offset box, and a
//! randomized batch of clearly-conditioned overlapping and clearly-separated
//! pairs compared one by one.
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
//! compares `f32` projection gaps against its `SEP_EPS` slack rather than `==`,
//! so a `GPU` fusing a multiply-add perturbs a gap by a few units in the last
//! place but cannot flip a decision as long as every fixture stays clearly on
//! one side of each axis comparison.
//!
//! # Conditioning
//!
//! Every fixture is deliberately far from the touch boundary: overlapping
//! fixtures put the triangle solidly through or inside the box, separated
//! fixtures leave a wide positive gap on at least one axis, and the randomized
//! batch either shrinks a triangle to sit well inside the box or translates it
//! far away. This keeps `CPU` and `GPU` on the same side of every `SEP_EPS`
//! comparison regardless of a few units in the last place of slack. The two
//! epsilon-exact graze cases the reference unit tests probe are intentionally
//! excluded, since those live on the boundary by design.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::triangle_aabb_overlap`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::triangle_aabb_overlap::triangle_overlaps_aabb;
use prism_volumetric_gpu::triangle_aabb_overlap::{GpuTriangleAabbOverlap, TriangleAabbQuery};
use prism_volumetric_gpu::GpuContext;

/// The canonical unit box: centered at the origin with half-extents of `1`,
/// i.e. the cube spanning `[-1, 1]` on every axis.
const UNIT_CENTER: [f32; 3] = [0.0, 0.0, 0.0];
const UNIT_HALF: [f32; 3] = [1.0, 1.0, 1.0];

/// The `CPU` golden boolean for one query.
fn cpu(query: &TriangleAabbQuery) -> bool {
    triangle_overlaps_aabb(&query.tri, query.center, query.half)
}

/// Dispatches `queries` on-device and asserts an exact boolean match against the
/// reference for every element.
fn check(ctx: &GpuContext, gpu: &GpuTriangleAabbOverlap, queries: &[TriangleAabbQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len());
    for (idx, query) in queries.iter().enumerate() {
        assert_eq!(
            got[idx],
            cpu(query),
            "overlap mismatch at {idx}: tri {:?} center {:?} half {:?}",
            query.tri,
            query.center,
            query.half
        );
    }
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

/// A pseudo-random point with each component in `[-span, span)`.
fn rand_point(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// The curated hand fixtures: unambiguous overlaps and separations lifted from
/// the reference contract, excluding the two epsilon-boundary graze cases.
fn hand_fixtures() -> Vec<TriangleAabbQuery> {
    vec![
        // Fully interior.
        TriangleAabbQuery::new(
            [[-0.5, -0.5, 0.0], [0.5, -0.25, 0.1], [0.0, 0.5, -0.2]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        // Far on +x, -y, +z: each box face axis separates.
        TriangleAabbQuery::new(
            [[3.0, 0.0, 0.0], [4.0, 1.0, 0.0], [3.5, -1.0, 1.0]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        TriangleAabbQuery::new(
            [[0.0, -3.0, 0.0], [1.0, -4.0, 0.0], [-1.0, -3.5, 0.5]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        TriangleAabbQuery::new(
            [[0.0, 0.0, 3.0], [1.0, 0.0, 4.0], [0.0, 1.0, 3.5]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        // Crossing the +x face from inside to outside.
        TriangleAabbQuery::new(
            [[0.0, 0.0, 0.0], [2.0, 0.5, 0.0], [2.0, -0.5, 0.0]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        // One vertex exactly on the +x face.
        TriangleAabbQuery::new(
            [[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, -1.0, 0.0]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        // One vertex exactly on the (1,1,1) corner.
        TriangleAabbQuery::new(
            [[1.0, 1.0, 1.0], [3.0, 2.0, 1.0], [2.0, 3.0, 1.5]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        // Large triangle in the z = 0 plane enclosing the box.
        TriangleAabbQuery::new(
            [[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [0.0, 10.0, 0.0]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        // Edge-cross axis separates (just clear of the +x,+z corner).
        TriangleAabbQuery::new(
            [[2.1, 0.0, 0.1], [0.1, 0.0, 2.1], [2.1, 0.0, 2.1]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        // Tilted triangle hovering past the (1,1,1) corner: separated by the
        // triangle-normal axis with a wide gap.
        TriangleAabbQuery::new(
            [[1.4, 1.0, 1.0], [1.0, 1.4, 1.0], [1.0, 1.0, 1.4]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        // Same tilt pulled in so the plane slices off the (1,1,1) corner.
        TriangleAabbQuery::new(
            [[1.2, 0.6, 0.6], [0.6, 1.2, 0.6], [0.6, 0.6, 1.2]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        // Offset box centered at (5,5,5), half 1: an interior and an exterior
        // triangle.
        TriangleAabbQuery::new(
            [[4.5, 5.0, 5.0], [5.5, 5.0, 5.5], [5.0, 5.5, 4.5]],
            [5.0, 5.0, 5.0],
            [1.0, 1.0, 1.0],
        ),
        TriangleAabbQuery::new(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [5.0, 5.0, 5.0],
            [1.0, 1.0, 1.0],
        ),
        // Thin slab half (2, 0.1, 2): a triangle clearing it above and one
        // threading through it.
        TriangleAabbQuery::new(
            [[-1.0, 0.5, -1.0], [1.0, 0.5, -1.0], [0.0, 0.5, 1.0]],
            [0.0, 0.0, 0.0],
            [2.0, 0.1, 2.0],
        ),
        TriangleAabbQuery::new(
            [[-1.0, -0.5, 0.0], [1.0, -0.5, 0.0], [0.0, 0.5, 0.0]],
            [0.0, 0.0, 0.0],
            [2.0, 0.1, 2.0],
        ),
        // Mis-signed half-extents are taken by magnitude: still overlaps.
        TriangleAabbQuery::new(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            UNIT_CENTER,
            [-1.0, -1.0, -1.0],
        ),
    ]
}

/// Degenerate-geometry fixtures: a triangle collapsed to a segment, a triangle
/// collapsed to a point, and a zero-volume box. All reduce to the correct
/// lower-dimensional overlap test on both sides.
fn degenerate_fixtures() -> Vec<TriangleAabbQuery> {
    vec![
        // Collinear corners (a segment) passing through the box.
        TriangleAabbQuery::new(
            [[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        // Collinear corners (a segment) clearly outside the box.
        TriangleAabbQuery::new(
            [[-2.0, 5.0, 0.0], [0.0, 5.0, 0.0], [2.0, 5.0, 0.0]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        // Coincident corners (a point) inside the box.
        TriangleAabbQuery::new(
            [[0.25, -0.25, 0.1], [0.25, -0.25, 0.1], [0.25, -0.25, 0.1]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        // Coincident corners (a point) outside the box.
        TriangleAabbQuery::new(
            [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
            UNIT_CENTER,
            UNIT_HALF,
        ),
        // Zero-volume box (a flat quad in z = 0): a triangle crossing its plane
        // within the x/y span overlaps.
        TriangleAabbQuery::new(
            [[0.0, 0.0, -1.0], [0.5, 0.0, 1.0], [0.0, 0.5, 1.0]],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ),
        // Zero-volume box with a triangle offset in z: clearly separated.
        TriangleAabbQuery::new(
            [[0.0, 0.0, 2.0], [0.5, 0.0, 3.0], [0.0, 0.5, 2.5]],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ),
    ]
}

/// A clearly-overlapping fixture: a small random triangle whose corners all sit
/// well inside the unit box, so every axis leaves the intervals overlapping by a
/// wide margin.
fn interior_query(state: &mut u64) -> TriangleAabbQuery {
    let base = rand_point(state, 0.3);
    let tri = [
        [
            base[0] + signed(state, 0.2),
            base[1] + signed(state, 0.2),
            base[2] + signed(state, 0.2),
        ],
        [
            base[0] + signed(state, 0.2),
            base[1] + signed(state, 0.2),
            base[2] + signed(state, 0.2),
        ],
        [
            base[0] + signed(state, 0.2),
            base[1] + signed(state, 0.2),
            base[2] + signed(state, 0.2),
        ],
    ];
    TriangleAabbQuery::new(tri, UNIT_CENTER, UNIT_HALF)
}

/// A clearly-separated fixture: a random small triangle shoved far away along a
/// large per-component offset, so at least one box face axis leaves a wide gap.
fn separated_query(state: &mut u64) -> TriangleAabbQuery {
    let shift = [
        signed(state, 1.0).signum() * (50.0 + lcg(state) * 20.0),
        signed(state, 1.0).signum() * (50.0 + lcg(state) * 20.0),
        signed(state, 1.0).signum() * (50.0 + lcg(state) * 20.0),
    ];
    let tri = [
        [
            shift[0] + signed(state, 1.0),
            shift[1] + signed(state, 1.0),
            shift[2] + signed(state, 1.0),
        ],
        [
            shift[0] + signed(state, 1.0),
            shift[1] + signed(state, 1.0),
            shift[2] + signed(state, 1.0),
        ],
        [
            shift[0] + signed(state, 1.0),
            shift[1] + signed(state, 1.0),
            shift[2] + signed(state, 1.0),
        ],
    ];
    TriangleAabbQuery::new(tri, UNIT_CENTER, UNIT_HALF)
}

#[test]
fn hand_fixtures_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleAabbOverlap::new(&ctx);
    check(&ctx, &gpu, &hand_fixtures());
}

#[test]
fn degenerate_fixtures_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleAabbOverlap::new(&ctx);
    check(&ctx, &gpu, &degenerate_fixtures());
}

#[test]
fn randomized_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTriangleAabbOverlap::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut queries = Vec::new();
    for _ in 0..128 {
        queries.push(interior_query(&mut state));
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
    let gpu = GpuTriangleAabbOverlap::new(&ctx);
    // No dispatch is issued and the entry point returns an empty vector.
    assert!(gpu.eval(&ctx, &[]).is_empty());
}
