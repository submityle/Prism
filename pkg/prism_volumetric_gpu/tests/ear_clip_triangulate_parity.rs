#![forbid(unsafe_code)]
//! Real-device parity for the ear-clipping predicate twin:
//! [`GpuEarClipTriangulate`](prism_volumetric_gpu::ear_clip_triangulate::GpuEarClipTriangulate)
//! must reproduce the `CPU` golden
//! [`ear_clip_triangulate`](prism_render_architecture::particle::ear_clip_triangulate)
//! predicates across convex and concave polygons, both windings (`CW` / `CCW`),
//! degenerate collinear inputs, clearly-interior and clearly-exterior
//! point-in-triangle probes, ear / non-ear vertex verdicts, and a randomized
//! sweep of star-shaped simple polygons.
//!
//! Only the parallelisable predicates are twinned: `signed_area`, `is_ccw`,
//! `point_in_triangle`, `is_convex_vertex` and `is_ear`. The sequential
//! [`triangulate`](prism_render_architecture::particle::ear_clip_triangulate::triangulate)
//! clip loop is *not* ported to the `GPU`; it remains the host's serial
//! orchestration and is exercised here only indirectly, by driving `is_ear`
//! with the same index rings a clip pass would.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every predicate is a fixed, non-reorderable sequence of multiplies, adds and
//! subtracts, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. The boolean verdicts (`is_ccw`, `point_in_triangle`,
//! `is_convex_vertex`, `is_ear`) are compared *exactly*; the continuous
//! `signed_area` allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` to admit a
//! legal fused multiply-add the scalar reference leaves separate.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from the compare-epsilon ties: the
//! triangle probes sit clearly inside or clearly outside (never on an edge or a
//! vertex), convex / reflex corners turn through a wide angle, and the random
//! star polygons use angularly-ordered vertices with radii bounded away from
//! zero. This keeps `CPU` and `GPU` on the same side of every sign test
//! regardless of a few units in the last place of slack.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ear_clip_triangulate`；
//! 无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::ear_clip_triangulate::{
    is_ccw, is_convex_vertex, is_ear, point_in_triangle, signed_area,
};
use prism_volumetric_gpu::ear_clip_triangulate::{
    EarClipAnswer, EarClipQuery, GpuEarClipTriangulate,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound for the continuous signed area. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that slack
/// while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
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

/// The `CCW` unit square, side `2`.
fn square_ccw() -> [[f32; 2]; 4] {
    [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
}

/// The same square wound clockwise.
fn square_cw() -> [[f32; 2]; 4] {
    [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]]
}

/// A convex `CCW` pentagon.
fn pentagon_ccw() -> [[f32; 2]; 5] {
    [[0.0, 0.0], [2.0, 0.0], [3.0, 1.5], [1.0, 3.0], [-1.0, 1.5]]
}

/// A concave `CCW` `L`-shape with one reflex vertex (index `3`).
fn l_shape() -> [[f32; 2]; 6] {
    [
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 1.0],
        [1.0, 1.0],
        [1.0, 3.0],
        [0.0, 3.0],
    ]
}

/// Three collinear points: a degenerate "polygon" of zero area.
fn collinear_triangle() -> [[f32; 2]; 3] {
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
}

/// Computes the reference answer for one query, tagged to match its kind.
fn golden(query: &EarClipQuery<'_>) -> EarClipAnswer {
    match *query {
        EarClipQuery::SignedArea { polygon } => EarClipAnswer::SignedArea(signed_area(polygon)),
        EarClipQuery::IsCcw { polygon } => EarClipAnswer::IsCcw(is_ccw(polygon)),
        EarClipQuery::PointInTriangle { p, a, b, c } => {
            EarClipAnswer::PointInTriangle(point_in_triangle(p, a, b, c))
        }
        EarClipQuery::IsConvexVertex {
            prev,
            cur,
            next,
            ccw,
        } => EarClipAnswer::IsConvexVertex(is_convex_vertex(prev, cur, next, ccw)),
        EarClipQuery::IsEar {
            polygon,
            ring,
            vertex,
            ccw,
        } => EarClipAnswer::IsEar(is_ear(polygon, ring, vertex, ccw)),
    }
}

/// Pins one `GPU` answer against its reference: exact for every boolean verdict,
/// tolerant for the continuous signed area.
fn pin(idx: usize, got: &EarClipAnswer, want: &EarClipAnswer) {
    match (got, want) {
        (EarClipAnswer::SignedArea(g), EarClipAnswer::SignedArea(w)) => {
            assert!(close(*g, *w), "query {idx} signed_area: gpu {g} vs cpu {w}");
        }
        (EarClipAnswer::IsCcw(g), EarClipAnswer::IsCcw(w))
        | (EarClipAnswer::PointInTriangle(g), EarClipAnswer::PointInTriangle(w))
        | (EarClipAnswer::IsConvexVertex(g), EarClipAnswer::IsConvexVertex(w))
        | (EarClipAnswer::IsEar(g), EarClipAnswer::IsEar(w)) => {
            assert_eq!(g, w, "query {idx} boolean verdict: gpu {g} vs cpu {w}");
        }
        _ => panic!("query {idx}: answer kind mismatch gpu {got:?} vs cpu {want:?}"),
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference, element for element.
fn check(ctx: &GpuContext, gpu: &GpuEarClipTriangulate, queries: &[EarClipQuery<'_>]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, result, &golden(query));
    }
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears in the random driver. Returns a value in
/// `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A trig-free unit-direction sweeping `CCW` once around the origin as `t`
/// runs over `[0, 1)`, traced along the perimeter of the unit diamond with
/// corners `(1,0) -> (0,1) -> (-1,0) -> (0,-1)`. The angle is strictly
/// monotonic in `t` and the direction never collapses to the origin (its
/// shortest reach is the edge midpoint at distance `0.5`), so it is a
/// transcendental-free stand-in for `(cos, sin)` that keeps libm determinism
/// and avoids the disallowed `f32::cos`/`f32::sin` methods.
fn diamond_dir(t: f32) -> [f32; 2] {
    const CORNERS: [[f32; 2]; 5] = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]];
    let s = (t * 4.0).clamp(0.0, 4.0);
    let q = (s as u32).min(3) as usize;
    let f = s - q as f32;
    let a = CORNERS[q];
    let b = CORNERS[q + 1];
    [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]
}

/// Builds an angularly-ordered star-shaped simple polygon of `n` vertices with
/// radii bounded well away from zero. Vertices sweep a strictly increasing
/// perimeter parameter, so the outline never self-intersects and the signed
/// area is large and positive (`CCW`), keeping both devices far from any
/// degeneracy crack.
fn star_polygon(state: &mut u64, n: usize) -> Vec<[f32; 2]> {
    let inv_n = 1.0 / n as f32;
    let mut poly = Vec::with_capacity(n);
    for k in 0..n {
        // Jitter stays strictly inside each slot (below half a step) so the
        // perimeter parameter remains strictly ordered and no two edges fold
        // onto one another.
        let jitter = (lcg(state) - 0.5) * 0.5;
        let t = (k as f32 + 0.5 + jitter) * inv_n;
        let dir = diamond_dir(t);
        let radius = 1.5 + lcg(state) * 2.5;
        poly.push([radius * dir[0], radius * dir[1]]);
    }
    poly
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEarClipTriangulate::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn signed_area_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEarClipTriangulate::new(&ctx);
    let square_ccw = square_ccw();
    let square_cw = square_cw();
    let pentagon = pentagon_ccw();
    let l = l_shape();
    let collinear = collinear_triangle();
    // A two-vertex "polygon" is degenerate and must report zero area.
    let degenerate = [[0.0, 0.0], [1.0, 0.0]];
    let queries = [
        EarClipQuery::SignedArea {
            polygon: &square_ccw,
        },
        EarClipQuery::SignedArea {
            polygon: &square_cw,
        },
        EarClipQuery::SignedArea { polygon: &pentagon },
        EarClipQuery::SignedArea { polygon: &l },
        EarClipQuery::SignedArea {
            polygon: &collinear,
        },
        EarClipQuery::SignedArea {
            polygon: &degenerate,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn is_ccw_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEarClipTriangulate::new(&ctx);
    let square_ccw = square_ccw();
    let square_cw = square_cw();
    let collinear = collinear_triangle();
    let queries = [
        EarClipQuery::IsCcw {
            polygon: &square_ccw,
        },
        EarClipQuery::IsCcw {
            polygon: &square_cw,
        },
        EarClipQuery::IsCcw {
            polygon: &collinear,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn point_in_triangle_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEarClipTriangulate::new(&ctx);
    // A fixed triangle; every probe is kept clearly interior or clearly exterior
    // (never on an edge or vertex) to stay well away from the CMP_EPS tie.
    let a = [0.0, 0.0];
    let b = [4.0, 0.0];
    let c = [0.0, 4.0];
    let queries = [
        // Clearly interior.
        EarClipQuery::PointInTriangle {
            p: [1.0, 1.0],
            a,
            b,
            c,
        },
        // Clearly outside, past the hypotenuse.
        EarClipQuery::PointInTriangle {
            p: [3.0, 3.0],
            a,
            b,
            c,
        },
        // Clearly outside, left of the vertical leg.
        EarClipQuery::PointInTriangle {
            p: [-1.0, 1.0],
            a,
            b,
            c,
        },
        // Clearly outside, below the horizontal leg.
        EarClipQuery::PointInTriangle {
            p: [1.0, -1.0],
            a,
            b,
            c,
        },
        // Interior but near (not on) the centroid region.
        EarClipQuery::PointInTriangle {
            p: [1.2, 1.3],
            a,
            b,
            c,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn is_convex_vertex_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEarClipTriangulate::new(&ctx);
    let sq = square_ccw();
    let l = l_shape();
    let collinear = collinear_triangle();
    let queries = [
        // A square corner is strictly convex for CCW winding.
        EarClipQuery::IsConvexVertex {
            prev: sq[3],
            cur: sq[0],
            next: sq[1],
            ccw: true,
        },
        // The L-shape reflex corner (vertex 3) is not convex for CCW winding.
        EarClipQuery::IsConvexVertex {
            prev: l[2],
            cur: l[3],
            next: l[4],
            ccw: true,
        },
        // A collinear triple is neither convex nor reflex: false for either
        // winding.
        EarClipQuery::IsConvexVertex {
            prev: collinear[0],
            cur: collinear[1],
            next: collinear[2],
            ccw: true,
        },
        // The same square corner read as a CW polygon is convex under ccw=false.
        EarClipQuery::IsConvexVertex {
            prev: sq[1],
            cur: sq[0],
            next: sq[3],
            ccw: false,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn is_ear_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEarClipTriangulate::new(&ctx);
    // Drive is_ear at every ring position of a convex square, a convex pentagon
    // and a concave L-shape, with the natural full ring (0..n) and the polygon's
    // own winding, mirroring the inputs an ear-clip pass would feed it.
    let square = square_ccw();
    let pentagon = pentagon_ccw();
    let l = l_shape();
    let square_ring: Vec<usize> = (0..square.len()).collect();
    let pentagon_ring: Vec<usize> = (0..pentagon.len()).collect();
    let l_ring: Vec<usize> = (0..l.len()).collect();

    let mut queries: Vec<EarClipQuery> = Vec::new();
    for v in 0..square.len() {
        queries.push(EarClipQuery::IsEar {
            polygon: &square,
            ring: &square_ring,
            vertex: v,
            ccw: is_ccw(&square),
        });
    }
    for v in 0..pentagon.len() {
        queries.push(EarClipQuery::IsEar {
            polygon: &pentagon,
            ring: &pentagon_ring,
            vertex: v,
            ccw: is_ccw(&pentagon),
        });
    }
    for v in 0..l.len() {
        queries.push(EarClipQuery::IsEar {
            polygon: &l,
            ring: &l_ring,
            vertex: v,
            ccw: is_ccw(&l),
        });
    }
    // A ring shorter than three entries can hold no ear.
    let short_ring = [0usize, 1];
    queries.push(EarClipQuery::IsEar {
        polygon: &square,
        ring: &short_ring,
        vertex: 0,
        ccw: true,
    });
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEarClipTriangulate::new(&ctx);
    // One dispatch mixing all five predicate kinds, so the shared vertex / ring
    // storage, the per-query offsets and the kind switch are all exercised
    // together, then pinned element for element.
    let square = square_ccw();
    let pentagon = pentagon_ccw();
    let l = l_shape();
    let l_ring: Vec<usize> = (0..l.len()).collect();
    let a = [0.0, 0.0];
    let b = [4.0, 0.0];
    let c = [0.0, 4.0];
    let queries = [
        EarClipQuery::SignedArea { polygon: &square },
        EarClipQuery::IsCcw { polygon: &pentagon },
        EarClipQuery::PointInTriangle {
            p: [1.0, 1.0],
            a,
            b,
            c,
        },
        EarClipQuery::IsConvexVertex {
            prev: l[2],
            cur: l[3],
            next: l[4],
            ccw: true,
        },
        EarClipQuery::IsEar {
            polygon: &l,
            ring: &l_ring,
            vertex: 0,
            ccw: is_ccw(&l),
        },
        EarClipQuery::SignedArea { polygon: &l },
        EarClipQuery::IsEar {
            polygon: &l,
            ring: &l_ring,
            vertex: 3,
            ccw: is_ccw(&l),
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_simple_polygons_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuEarClipTriangulate::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // Build a sweep of star-shaped simple polygons; keep every polygon alive in
    // `polys` so the borrowed queries stay valid for the whole dispatch.
    let mut polys: Vec<Vec<[f32; 2]>> = Vec::new();
    for i in 0..64 {
        let n = 3 + (i % 6);
        let mut poly = star_polygon(&mut state, n);
        // Flip half of them to CW so both windings are swept.
        if i % 2 == 1 {
            poly.reverse();
        }
        polys.push(poly);
    }
    let mut queries: Vec<EarClipQuery> = Vec::with_capacity(polys.len() * 2);
    for poly in &polys {
        queries.push(EarClipQuery::SignedArea { polygon: poly });
        queries.push(EarClipQuery::IsCcw { polygon: poly });
    }
    check(&ctx, &gpu, &queries);
}
