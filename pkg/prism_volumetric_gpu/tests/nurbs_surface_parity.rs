//! Real-device parity for the uniform rational bicubic B-spline (`NURBS`)
//! surface twin:
//! [`GpuNurbsSurface`](prism_volumetric_gpu::nurbs_surface::GpuNurbsSurface)
//! must reproduce the surface point, unit normal and degenerate classification
//! of the host-side independent reimplementation
//! [`eval_nurbs_surface`](prism_volumetric_gpu::nurbs_surface::eval_nurbs_surface)
//! across a planar net, a domed net with non-uniform weights, a multi-span
//! net, a fully collapsed degenerate net and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected result comes from the module's own host-side independent
//! reimplementation; the twin never imports the golden crate, so both the
//! kernel and the oracle are faithful, independent ports of the same closed
//! form. A `GPU` parity pass is therefore direct evidence the ported kernel
//! runs the same `locate` span split, the same B-spline-to-Bézier conversion
//! and the same rational bicubic recursion, and classifies the same degenerate
//! normal.
//!
//! # Parity criterion
//!
//! The surface point and normal are *continuous* quantities threaded through
//! nested `lerp`s, a `cross`, a `sqrt` and guarded divisions, so every
//! continuous assertion compares with an absolute-or-relative tolerance
//! (`abs <= 1e-4 || rel <= 1e-3`, relative floor `1e-6`). The discrete
//! `degenerate` flag is pinned exactly. The genuine degeneracy — a collapsed
//! tangent frame whose cross-product magnitude crosses zero — is kept off its
//! threshold by the fixtures and the rejection-sampled sweep so the two sides
//! agree on the discrete classification.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::nurbs_surface`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::nurbs_surface::{
    eval_nurbs_surface, GpuNurbsSurface, NurbsSurfaceQuery, NurbsSurfaceResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous point/normal comparison.
const EPS: f32 = 1e-4;
/// Relative tolerance for the continuous point/normal comparison.
const REL: f32 = 1e-3;
/// Relative-tolerance floor, so near-zero magnitudes do not inflate the ratio.
const REL_FLOOR: f32 = 1e-6;

/// Smallest accepted finite-difference tangent cross-product length, keeping
/// the degenerate-normal branch off its threshold in the randomized sweep so
/// the `degenerate` flag cannot flip between host and device. The projected
/// tangents the finite difference measures differ from the kernel's
/// quotient-rule numerators only by a strictly-positive weight factor, so a
/// decisively non-zero finite-difference cross implies a decisively non-zero
/// numerator cross.
const CROSS_MARGIN: f32 = 0.05;
/// Interior margin on the global `(u, v)` parameters, keeping the sweep off the
/// `clamp` boundaries where the `eps` step-search saturates.
const UV_MARGIN: f32 = 0.05;
/// Finite-difference half-step for the conditioning tangent estimate.
const FD_STEP: f32 = 1e-2;

/// Returns whether `a` and `b` agree within the absolute-or-relative tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL
}

/// Dot product of two 3-vectors, used only by the conditioning check.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two 3-vectors, used only by the conditioning check.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A shared control grid plus matching positive weights and its row/column
/// counts, bundled so one fixture passes a complete surface to the batch.
struct Grid {
    control: Vec<[f32; 3]>,
    weights: Vec<f32>,
    rows: u32,
    cols: u32,
}

/// Computes the expected result from the host-side reimplementation.
fn oracle(grid: &Grid, q: &NurbsSurfaceQuery) -> ([f32; 3], [f32; 3], bool) {
    eval_nurbs_surface(
        &grid.control,
        &grid.weights,
        grid.rows as usize,
        grid.cols as usize,
        q.u,
        q.v,
    )
}

/// Dispatches every query against the shared grid and pins each `GPU` result
/// against the oracle: the `degenerate` flag matches exactly, and the point and
/// normal match within tolerance.
fn check(ctx: &GpuContext, gpu: &GpuNurbsSurface, grid: &Grid, queries: &[NurbsSurfaceQuery]) {
    let got: Vec<NurbsSurfaceResult> = gpu.evaluate(
        ctx,
        &grid.control,
        &grid.weights,
        grid.rows,
        grid.cols,
        queries,
    );
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let (point, normal, degenerate) = oracle(grid, q);
        assert_eq!(
            result.degenerate, degenerate,
            "query {idx}: degenerate flag mismatch"
        );
        assert!(
            close(result.point[0], point[0])
                && close(result.point[1], point[1])
                && close(result.point[2], point[2]),
            "query {idx}: point gpu {:?} vs cpu {:?}",
            result.point,
            point
        );
        assert!(
            close(result.normal[0], normal[0])
                && close(result.normal[1], normal[1])
                && close(result.normal[2], normal[2]),
            "query {idx}: normal gpu {:?} vs cpu {:?}",
            result.normal,
            normal
        );
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

/// Draws an `f32` in `[lo, hi]` at milli resolution from `state`, using only
/// integer arithmetic so no transcendental method appears.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let span = ((hi - lo) * 1000.0) as u32;
    let step = lcg(state) % (span + 1);
    lo + step as f32 / 1000.0
}

/// Central finite-difference estimate of the projected surface tangent along
/// `u`, used only by the conditioning check.
fn tangent_u(grid: &Grid, u: f32, v: f32) -> [f32; 3] {
    let (plus, _, _) = oracle(grid, &NurbsSurfaceQuery::new(u + FD_STEP, v));
    let (minus, _, _) = oracle(grid, &NurbsSurfaceQuery::new(u - FD_STEP, v));
    [
        (plus[0] - minus[0]) / (2.0 * FD_STEP),
        (plus[1] - minus[1]) / (2.0 * FD_STEP),
        (plus[2] - minus[2]) / (2.0 * FD_STEP),
    ]
}

/// Central finite-difference estimate of the projected surface tangent along
/// `v`, used only by the conditioning check.
fn tangent_v(grid: &Grid, u: f32, v: f32) -> [f32; 3] {
    let (plus, _, _) = oracle(grid, &NurbsSurfaceQuery::new(u, v + FD_STEP));
    let (minus, _, _) = oracle(grid, &NurbsSurfaceQuery::new(u, v - FD_STEP));
    [
        (plus[0] - minus[0]) / (2.0 * FD_STEP),
        (plus[1] - minus[1]) / (2.0 * FD_STEP),
        (plus[2] - minus[2]) / (2.0 * FD_STEP),
    ]
}

/// Rejects queries that sit on the genuine degeneracy of the normal solve, so
/// the discrete `degenerate` flag cannot flip between host and device: the
/// finite-difference tangent cross product must be decisively non-zero and the
/// global `(u, v)` must stay off the `clamp` boundaries. Because the projected
/// tangents differ from the kernel's quotient-rule numerators only by a
/// strictly-positive weight factor, a non-zero finite-difference cross pins the
/// exact threshold the kernel branches on.
fn well_conditioned(grid: &Grid, u: f32, v: f32) -> bool {
    if !(UV_MARGIN..=1.0 - UV_MARGIN).contains(&u) || !(UV_MARGIN..=1.0 - UV_MARGIN).contains(&v) {
        return false;
    }
    let du = tangent_u(grid, u, v);
    let dv = tangent_v(grid, u, v);
    let n = cross3(du, dv);
    dot3(n, n).sqrt() >= CROSS_MARGIN
}

/// Builds a planar `4 × 4` net in the `xy` plane with unit weights:
/// `control[r*4 + c] = [c / 3, r / 3, 0]`, whose tangents are the two axes and
/// whose normal is `±z` everywhere.
fn flat_grid() -> Grid {
    let mut control = vec![[0.0f32; 3]; 16];
    for r in 0..4 {
        for c in 0..4 {
            control[r * 4 + c] = [c as f32 / 3.0, r as f32 / 3.0, 0.0];
        }
    }
    Grid {
        control,
        weights: vec![1.0; 16],
        rows: 4,
        cols: 4,
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        #[expect(
            clippy::print_stderr,
            reason = "a skipped device test should report why on hosts without a GPU"
        )]
        {
            eprintln!("skipping nurbs_surface parity: no wgpu adapter on this host");
        }
        return;
    };
    let gpu = GpuNurbsSurface::new(&ctx);
    let grid = flat_grid();
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(
        &ctx,
        &grid.control,
        &grid.weights,
        grid.rows,
        grid.cols,
        &[],
    );
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn flat_net_has_axis_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNurbsSurface::new(&ctx);
    let grid = flat_grid();
    // A planar net lies in the z = 0 plane; its analytic normal is +-z and the
    // tangent frame never collapses, so the surface is never degenerate.
    let (u, v) = (0.5, 0.5);
    assert!(
        well_conditioned(&grid, u, v),
        "fixture must stay off the threshold"
    );
    let q = NurbsSurfaceQuery::new(u, v);
    let (point, normal, degenerate) = oracle(&grid, &q);
    assert!(!degenerate, "a planar net has a well-defined normal");
    assert!(close(point[2], 0.0), "the planar net stays in z = 0");
    assert!(
        close(normal[2].abs(), 1.0),
        "the planar net normal is the z axis"
    );
    check(&ctx, &gpu, &grid, &[q]);
}

#[test]
fn domed_net_with_weights_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNurbsSurface::new(&ctx);
    // Lift the four interior control points off the plane into a smooth dome and
    // give them non-uniform weights, so the rational quotient-rule derivative
    // path is exercised rather than the degenerate all-unit-weight case.
    let mut grid = flat_grid();
    for r in 1..=2 {
        for c in 1..=2 {
            grid.control[r * 4 + c][2] = 0.6;
        }
    }
    grid.weights[5] = 3.0;
    grid.weights[6] = 2.0;
    grid.weights[9] = 2.0;
    grid.weights[10] = 3.0;
    let queries = [
        NurbsSurfaceQuery::new(0.5, 0.5),
        NurbsSurfaceQuery::new(0.3, 0.7),
        NurbsSurfaceQuery::new(0.65, 0.4),
    ];
    for q in &queries {
        assert!(
            well_conditioned(&grid, q.u, q.v),
            "dome fixture ({}, {}) must stay off the threshold",
            q.u,
            q.v
        );
        let (_, _, degenerate) = oracle(&grid, q);
        assert!(!degenerate, "the weighted dome has a well-defined normal");
    }
    check(&ctx, &gpu, &grid, &queries);
}

#[test]
fn multi_span_net_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNurbsSurface::new(&ctx);
    // A 6 x 7 net spans (7 - 3) x (6 - 3) = 4 x 3 overlapping cubic spans, so
    // `locate` must select the right span for interior parameters. The grid is
    // a gently lifted lattice with deterministic strictly-positive weights.
    let rows = 6usize;
    let cols = 7usize;
    let mut control = vec![[0.0f32; 3]; rows * cols];
    let mut weights = vec![1.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let x = c as f32;
            let y = r as f32;
            // A smooth separable bump that keeps the surface curved everywhere.
            let lift = (c as f32 - 3.0) * (c as f32 - 3.0) * 0.05
                + (r as f32 - 2.5) * (r as f32 - 2.5) * 0.04;
            control[r * cols + c] = [x, y, lift];
            weights[r * cols + c] = 1.0 + ((r + c) % 3) as f32 * 0.5;
        }
    }
    let grid = Grid {
        control,
        weights,
        rows: rows as u32,
        cols: cols as u32,
    };
    let queries = [
        NurbsSurfaceQuery::new(0.2, 0.3),
        NurbsSurfaceQuery::new(0.5, 0.5),
        NurbsSurfaceQuery::new(0.75, 0.6),
        NurbsSurfaceQuery::new(0.4, 0.85),
    ];
    for q in &queries {
        assert!(
            well_conditioned(&grid, q.u, q.v),
            "multi-span fixture ({}, {}) must stay off the threshold",
            q.u,
            q.v
        );
    }
    check(&ctx, &gpu, &grid, &queries);
}

#[test]
fn collapsed_net_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNurbsSurface::new(&ctx);
    // Every control point coincides with unit weight, so both partials vanish at
    // every eps step: the tangent frame never recovers, the normal falls back to
    // [0, 0, 1] and the degenerate flag is set. The point collapses to the
    // shared location.
    let grid = Grid {
        control: vec![[0.7f32, -0.2, 0.5]; 16],
        weights: vec![1.0; 16],
        rows: 4,
        cols: 4,
    };
    let q = NurbsSurfaceQuery::new(0.42, 0.63);
    let (point, normal, degenerate) = oracle(&grid, &q);
    assert!(degenerate, "a collapsed net has no tangent frame");
    assert!(
        close(point[0], 0.7) && close(point[1], -0.2) && close(point[2], 0.5),
        "a collapsed net evaluates to its shared point"
    );
    assert!(
        close(normal[0], 0.0) && close(normal[1], 0.0) && close(normal[2], 1.0),
        "a degenerate normal falls back to the z axis"
    );
    check(&ctx, &gpu, &grid, &[q]);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNurbsSurface::new(&ctx);
    let mut state: u64 = 0x2468_ACE0_1357_9BDF;
    let mut queries = Vec::with_capacity(512);
    // One shared 4 x 4 grid per sweep keeps the dispatch a single surface while
    // still randomizing both the net and the query parameters.
    let mut grid = Grid {
        control: vec![[0.0f32; 3]; 16],
        weights: vec![1.0; 16],
        rows: 4,
        cols: 4,
    };
    while queries.len() < 512 {
        for point in &mut grid.control {
            point[0] = uniform(&mut state, -2.0, 2.0);
            point[1] = uniform(&mut state, -2.0, 2.0);
            point[2] = uniform(&mut state, -2.0, 2.0);
        }
        for weight in &mut grid.weights {
            // Strictly-positive weights, matching the golden `NurbsSurface::new`
            // contract.
            *weight = uniform(&mut state, 0.5, 3.0);
        }
        let u = uniform(&mut state, 0.0, 1.0);
        let v = uniform(&mut state, 0.0, 1.0);
        // Reject anything whose tangent frame is near collapse or whose
        // parameters sit on a clamp boundary, so the discrete degenerate flag
        // agrees between host and device.
        if !well_conditioned(&grid, u, v) {
            continue;
        }
        let batch = [NurbsSurfaceQuery::new(u, v)];
        check(&ctx, &gpu, &grid, &batch);
        queries.push((u, v));
    }
    assert_eq!(
        queries.len(),
        512,
        "the sweep collects 512 conditioned draws"
    );
}
