//! Real-device parity for the triangle surface-point and normal reconstruction
//! twin:
//! [`GpuRayTriangleReconstruct`](prism_volumetric_gpu::ray_triangle_reconstruct::GpuRayTriangleReconstruct)
//! must reproduce the `CPU` golden
//! [`barycentric_to_point`](prism_render_architecture::particle::ray_triangle::barycentric_to_point)
//! and
//! [`triangle_normal`](prism_render_architecture::particle::ray_triangle::triangle_normal)
//! across hand-picked fixtures plus a randomized batch compared
//! value-for-value.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! Both golden functions are `pub`, so each `GPU` reconstruction is pinned
//! directly against the golden run on the same vertices and barycentric
//! coordinates.
//!
//! # Parity criterion
//!
//! The point is a continuous `f32` triple built from multiply-adds, compared
//! with `abs <= 1e-5 || rel <= 1e-5`. The normal passes through a `sqrt` and a
//! reciprocal, so it is compared with the slightly looser
//! `abs <= 1e-5 || rel <= 1e-4`. A degenerate (zero-area or collinear)
//! triangle yields the exact zero vector on both sides, which the absolute band
//! accepts directly.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ray_triangle`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::ray_triangle::{
    barycentric_to_point, triangle_normal, Vec3,
};
use prism_volumetric_gpu::ray_triangle_reconstruct::{
    GpuRayTriangleReconstruct, RayTriangleReconstructQuery, RayTriangleReconstructResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-comparison floor so tiny magnitudes do not inflate the relative
/// error.
const REL_FLOOR: f32 = 1.0e-6;

/// Computes the golden reconstruction for one query via the `CPU` oracle.
fn expected(q: &RayTriangleReconstructQuery) -> RayTriangleReconstructResult {
    let v0 = Vec3::new(q.v0[0], q.v0[1], q.v0[2]);
    let v1 = Vec3::new(q.v1[0], q.v1[1], q.v1[2]);
    let v2 = Vec3::new(q.v2[0], q.v2[1], q.v2[2]);
    let point = barycentric_to_point(v0, v1, v2, q.u, q.v);
    let normal = triangle_normal(v0, v1, v2);
    RayTriangleReconstructResult {
        point: [point.x, point.y, point.z],
        normal: [normal.x, normal.y, normal.z],
    }
}

/// Returns whether `got` matches `want` within the mixed absolute/relative
/// tolerance `abs <= 1e-5 || rel <= rel_tol`.
fn close(got: f32, want: f32, rel_tol: f32) -> bool {
    let diff = (got - want).abs();
    if diff <= 1.0e-5 {
        return true;
    }
    let denom = want.abs().max(REL_FLOOR);
    diff / denom <= rel_tol
}

/// Pins one `GPU` result against the golden oracle: the point under the
/// multiply-add tolerance and the normal under the slightly looser `sqrt`
/// tolerance.
fn assert_result(
    idx: usize,
    got: &RayTriangleReconstructResult,
    want: &RayTriangleReconstructResult,
) {
    for (axis, (g, w)) in got.point.iter().zip(want.point.iter()).enumerate() {
        assert!(
            close(*g, *w, 1.0e-5),
            "result {idx} point axis {axis}: gpu={g} cpu={w}"
        );
    }
    for (axis, (g, w)) in got.normal.iter().zip(want.normal.iter()).enumerate() {
        assert!(
            close(*g, *w, 1.0e-4),
            "result {idx} normal axis {axis}: gpu={g} cpu={w}"
        );
    }
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[RayTriangleReconstructQuery]) {
    let gpu = GpuRayTriangleReconstruct::new(ctx);
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        assert_result(idx, g, &expected(q));
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a float in `[0, 1)` from `state` using only integer work.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32
}

/// Draws a float in `[lo, hi)` from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * unit(state)
}

/// Draws a 3-vector with each component in `[lo, hi)`.
fn ranged3(state: &mut u64, lo: f32, hi: f32) -> [f32; 3] {
    [
        ranged(state, lo, hi),
        ranged(state, lo, hi),
        ranged(state, lo, hi),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ray_triangle_reconstruct parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuRayTriangleReconstruct::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn reconstruct_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // A right triangle in the z = 0 plane with a +z unit normal.
    let unit_tri = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let queries = vec![
        // Centroid: u = v = 1/3, so point is the vertex average.
        RayTriangleReconstructQuery {
            v0: unit_tri.0,
            v1: unit_tri.1,
            v2: unit_tri.2,
            u: 1.0 / 3.0,
            v: 1.0 / 3.0,
        },
        // Vertex recovery v0: u = 0, v = 0, w = 1.
        RayTriangleReconstructQuery {
            v0: [2.0, -3.0, 5.0],
            v1: [7.0, 1.0, -2.0],
            v2: [-4.0, 6.0, 8.0],
            u: 0.0,
            v: 0.0,
        },
        // Vertex recovery v1: u = 1, v = 0, w = 0.
        RayTriangleReconstructQuery {
            v0: [2.0, -3.0, 5.0],
            v1: [7.0, 1.0, -2.0],
            v2: [-4.0, 6.0, 8.0],
            u: 1.0,
            v: 0.0,
        },
        // Vertex recovery v2: u = 0, v = 1, w = 0.
        RayTriangleReconstructQuery {
            v0: [2.0, -3.0, 5.0],
            v1: [7.0, 1.0, -2.0],
            v2: [-4.0, 6.0, 8.0],
            u: 0.0,
            v: 1.0,
        },
        // Edge midpoint between v1 and v2: u = v = 0.5, w = 0.
        RayTriangleReconstructQuery {
            v0: [2.0, -3.0, 5.0],
            v1: [7.0, 1.0, -2.0],
            v2: [-4.0, 6.0, 8.0],
            u: 0.5,
            v: 0.5,
        },
        // Out-of-range barycentric extrapolation: u + v > 1 (w negative).
        RayTriangleReconstructQuery {
            v0: unit_tri.0,
            v1: unit_tri.1,
            v2: unit_tri.2,
            u: 0.8,
            v: 0.7,
        },
        // Out-of-range barycentric extrapolation: negative u.
        RayTriangleReconstructQuery {
            v0: unit_tri.0,
            v1: unit_tri.1,
            v2: unit_tri.2,
            u: -0.2,
            v: 0.3,
        },
        // Standard non-axis-aligned triangle normal.
        RayTriangleReconstructQuery {
            v0: [1.0, 0.0, 0.0],
            v1: [0.0, 1.0, 0.0],
            v2: [0.0, 0.0, 1.0],
            u: 0.25,
            v: 0.25,
        },
        // Large-scale triangle to exercise the relative tolerance path.
        RayTriangleReconstructQuery {
            v0: [-30.0, 10.0, -20.0],
            v1: [40.0, -25.0, 15.0],
            v2: [5.0, 45.0, -35.0],
            u: 0.3,
            v: 0.4,
        },
        // Negative-only coordinates.
        RayTriangleReconstructQuery {
            v0: [-1.0, -2.0, -3.0],
            v1: [-4.0, -1.0, -6.0],
            v2: [-2.0, -5.0, -1.0],
            u: 0.2,
            v: 0.5,
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn degenerate_normals_are_exact_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries = vec![
        // Three collinear points along the x axis: zero-area, zero normal.
        RayTriangleReconstructQuery {
            v0: [0.0, 0.0, 0.0],
            v1: [1.0, 0.0, 0.0],
            v2: [3.0, 0.0, 0.0],
            u: 0.3,
            v: 0.3,
        },
        // Two coincident vertices (v0 == v1): zero-area, zero normal.
        RayTriangleReconstructQuery {
            v0: [2.0, 5.0, -1.0],
            v1: [2.0, 5.0, -1.0],
            v2: [7.0, -3.0, 4.0],
            u: 0.25,
            v: 0.5,
        },
        // All three vertices identical: zero normal.
        RayTriangleReconstructQuery {
            v0: [-4.0, 4.0, 4.0],
            v1: [-4.0, 4.0, 4.0],
            v2: [-4.0, 4.0, 4.0],
            u: 0.1,
            v: 0.1,
        },
    ];
    let gpu = GpuRayTriangleReconstruct::new(&ctx);
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = expected(q);
        // The golden degenerate normal is the exact zero vector; assert both
        // sides agree exactly on every axis.
        assert_eq!(want.normal, [0.0, 0.0, 0.0], "oracle normal is zero");
        assert_eq!(g.normal, [0.0, 0.0, 0.0], "gpu normal is zero at {idx}");
        // The point is still well defined and must match to tolerance.
        for (axis, (gp, wp)) in g.point.iter().zip(want.point.iter()).enumerate() {
            assert!(
                close(*gp, *wp, 1.0e-5),
                "degenerate {idx} point axis {axis}: gpu={gp} cpu={wp}"
            );
        }
    }
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x7f3b_a17c_04d7_9e21_u64;

    let mut queries: Vec<RayTriangleReconstructQuery> = Vec::new();
    while queries.len() < 256 {
        // Random vertices far apart make a degenerate triangle vanishingly
        // unlikely, so random normals are always well defined.
        queries.push(RayTriangleReconstructQuery {
            v0: ranged3(&mut state, -50.0, 50.0),
            v1: ranged3(&mut state, -50.0, 50.0),
            v2: ranged3(&mut state, -50.0, 50.0),
            // Barycentric coordinates include out-of-range extrapolation and
            // interior points with u + v <= 1.
            u: ranged(&mut state, -0.2, 1.2),
            v: ranged(&mut state, -0.2, 1.2),
        });
    }
    run_and_check(&ctx, &queries);
}
