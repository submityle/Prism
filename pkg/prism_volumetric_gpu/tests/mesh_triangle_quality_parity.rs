//! Real-device parity for the per-triangle shape-quality metric twin:
//! [`GpuMeshTriangleQuality`](prism_volumetric_gpu::mesh_triangle_quality::GpuMeshTriangleQuality)
//! must reproduce the single-triangle core of
//! `prism_render_architecture::ray_scene::mesh_triangle_quality::triangle_quality`
//! — the area `A = 0.5*|e0 x (v2 - v0)|` and the normalized mean-ratio score
//! `q = 4*sqrt(3)*A / (|e0|^2 + |e1|^2 + |e2|^2)` clamped to `[0, 1]` — across
//! healthy, sliver and degenerate faces plus a randomized sweep compared
//! triangle-for-triangle.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed form: the edge vectors
//! `e0 = v1 - v0`, `e1 = v2 - v1`, `e2 = v0 - v2`, the doubled-area cross
//! product `e0 x (v2 - v0)`, the area and the squared-edge-sum denominator. It
//! also mirrors the twin's `degenerate` contract: a face whose vertices
//! coincide (`sum_sq <= 0`) or whose raw score collapses
//! (`q_raw <= `[`QUALITY_EPS`]) reports `degenerate = 1` with `area` and
//! `quality` both forced to `0`. Because the reference and this oracle are both
//! scalar `f32`, a `GPU == oracle` pass is direct evidence the ported kernel
//! computes the same scores the reference does.
//!
//! # Parity criterion
//!
//! The continuous `area` and `quality` thread through products, quotients and
//! two `sqrt`, so a `GPU` result may land a few units in the last place from the
//! scalar oracle; both are asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, with a relative floor of `1e-6` so a near-zero expected
//! value does not inflate the relative error. The discrete `degenerate` flag is
//! compared for exact equality.
//!
//! # Conditioning
//!
//! The branch-switch loci are the degenerate test (`sum_sq ~ 0` or
//! `q_raw ~ `[`QUALITY_EPS`]) and the upper clamp (`q_raw ~ 1`, the equilateral
//! limit). The named fixtures sit well inside a single branch and the randomized
//! sweep rejects any sample within a safety margin of each locus, so both sides
//! fold the identical verdict and no cliff can appear.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_triangle_quality`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_triangle_quality::{
    GpuMeshTriangleQuality, MeshTriangleQualityQuery, MeshTriangleQualityResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on any continuous output. A `GPU` `sqrt`/divide may land a few
/// units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative bound on any continuous output, applied for larger magnitudes where
/// a few units in the last place exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Degenerate-collapse floor shared with the kernel: a raw score at or below
/// this routes to the degenerate branch, mirroring the twin's contract without
/// a forbidden bare `f32` equality.
const QUALITY_EPS: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound
/// (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Subtracts `b` from `a` componentwise.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Euclidean dot product of two vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a x b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Independent reimplementation of the single-triangle core of
/// `ray_scene::mesh_triangle_quality::triangle_quality`: the area and the
/// normalized mean-ratio quality, plus the twin's degenerate zeroing. Mirrors
/// the kernel step for step so a `GPU == oracle` pass is meaningful.
fn quality_host(q: &MeshTriangleQualityQuery) -> MeshTriangleQualityResult {
    let e0 = sub(q.v1, q.v0);
    let e1 = sub(q.v2, q.v1);
    let e2 = sub(q.v0, q.v2);

    // Doubled-area cross product e0 x (v2 - v0) == e0 x (-e2).
    let crs = cross(e0, sub(q.v2, q.v0));
    let cross_mag = dot(crs, crs).sqrt();
    let area = 0.5 * cross_mag;

    let sum_sq = dot(e0, e0) + dot(e1, e1) + dot(e2, e2);
    let sqrt3 = 3.0_f32.sqrt();

    let mut out = MeshTriangleQualityResult {
        area: 0.0,
        quality: 0.0,
        degenerate: 1,
    };
    if sum_sq > 0.0 {
        let q_raw = 4.0 * sqrt3 * area / sum_sq;
        if q_raw > QUALITY_EPS {
            out.area = area;
            out.quality = q_raw.clamp(0.0, 1.0);
            out.degenerate = 0;
        }
    }
    out
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

/// Returns `true` when a sweep sample sits too close to a branch-switch locus
/// and must be rejected, so the kept comparison stays far from any cliff: a
/// near-degenerate face (`sum_sq` or raw score near the collapse floor) or a
/// near-equilateral face (raw score near the upper clamp).
fn reject_sample(q: &MeshTriangleQualityQuery) -> bool {
    let e0 = sub(q.v1, q.v0);
    let e1 = sub(q.v2, q.v1);
    let e2 = sub(q.v0, q.v2);
    let crs = cross(e0, sub(q.v2, q.v0));
    let area = 0.5 * dot(crs, crs).sqrt();
    let sum_sq = dot(e0, e0) + dot(e1, e1) + dot(e2, e2);
    // Keep the sweep well clear of the coincident floor.
    if sum_sq < 0.25 {
        return true;
    }
    let sqrt3 = 3.0_f32.sqrt();
    let q_raw = 4.0 * sqrt3 * area / sum_sq;
    // Clear separation (two orders of magnitude) from the collapse floor so the
    // degenerate flag never flips between the two sides.
    if q_raw < 1.0e-4 {
        return true;
    }
    // Stay off the upper clamp so the clamp verdict is unambiguous.
    if q_raw > 0.99 {
        return true;
    }
    false
}

/// Dispatches `queries` and asserts every output matches the host oracle: the
/// `degenerate` flag exactly, and `area`/`quality` within tolerance.
fn check_batch(
    ctx: &GpuContext,
    gpu: &GpuMeshTriangleQuality,
    queries: &[MeshTriangleQualityQuery],
) {
    let got: Vec<MeshTriangleQualityResult> = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let golden = quality_host(q);
        assert_eq!(
            r.degenerate, golden.degenerate,
            "degenerate flag mismatch: gpu={} golden={} (query={q:?})",
            r.degenerate, golden.degenerate
        );
        assert!(
            close(r.area, golden.area),
            "area mismatch: gpu={} golden={} (query={q:?})",
            r.area,
            golden.area
        );
        assert!(
            close(r.quality, golden.quality),
            "quality mismatch: gpu={} golden={} (query={q:?})",
            r.quality,
            golden.quality
        );
    }
}

/// A unit-edge equilateral triangle: the maximal-quality face (`q = 1`).
fn equilateral() -> MeshTriangleQualityQuery {
    let h = 3.0_f32.sqrt() / 2.0;
    MeshTriangleQualityQuery {
        v0: [0.0, 0.0, 0.0],
        v1: [1.0, 0.0, 0.0],
        v2: [0.5, h, 0.0],
    }
}

/// A right-isosceles triangle with unit legs: a healthy mid-quality face.
fn right_isosceles() -> MeshTriangleQualityQuery {
    MeshTriangleQualityQuery {
        v0: [0.0, 0.0, 0.0],
        v1: [1.0, 0.0, 0.0],
        v2: [0.0, 1.0, 0.0],
    }
}

/// A long thin sliver: tiny positive area, small-but-finite quality.
fn sliver() -> MeshTriangleQualityQuery {
    MeshTriangleQualityQuery {
        v0: [0.0, 0.0, 0.0],
        v1: [10.0, 0.0, 0.0],
        v2: [5.0, 0.01, 0.0],
    }
}

/// Three collinear points: zero area, degenerate.
fn collinear() -> MeshTriangleQualityQuery {
    MeshTriangleQualityQuery {
        v0: [0.0, 0.0, 0.0],
        v1: [1.0, 0.0, 0.0],
        v2: [2.0, 0.0, 0.0],
    }
}

/// Three coincident points: `sum_sq = 0`, degenerate.
fn coincident() -> MeshTriangleQualityQuery {
    MeshTriangleQualityQuery {
        v0: [1.0, -2.0, 0.5],
        v1: [1.0, -2.0, 0.5],
        v2: [1.0, -2.0, 0.5],
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mesh_triangle_quality parity: no wgpu adapter available");
        return;
    };
    let gpu = GpuMeshTriangleQuality::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "empty batch yields no results");
}

#[test]
fn equilateral_is_maximal_quality() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleQuality::new(&ctx);
    let q = equilateral();
    let golden = quality_host(&q);
    assert_eq!(golden.degenerate, 0, "equilateral is not degenerate");
    assert!(close(golden.quality, 1.0), "equilateral quality is 1");
    check_batch(&ctx, &gpu, &[q]);
}

#[test]
fn right_isosceles_mid_quality() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleQuality::new(&ctx);
    let q = right_isosceles();
    let golden = quality_host(&q);
    assert_eq!(golden.degenerate, 0, "right isosceles is not degenerate");
    // 4*sqrt(3)*0.5 / 4 = sqrt(3)/2 ~ 0.8660254.
    assert!(close(golden.quality, 3.0_f32.sqrt() / 2.0), "known quality");
    check_batch(&ctx, &gpu, &[q]);
}

#[test]
fn sliver_is_low_but_finite_quality() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleQuality::new(&ctx);
    let q = sliver();
    let golden = quality_host(&q);
    assert_eq!(golden.degenerate, 0, "thin sliver still has positive area");
    assert!(golden.quality > 0.0, "sliver quality is positive");
    assert!(golden.quality < 0.1, "sliver quality is small");
    check_batch(&ctx, &gpu, &[q]);
}

#[test]
fn collinear_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleQuality::new(&ctx);
    let q = collinear();
    let golden = quality_host(&q);
    assert_eq!(golden.degenerate, 1, "collinear points are degenerate");
    assert!(close(golden.area, 0.0), "collinear area is zero");
    assert!(close(golden.quality, 0.0), "collinear quality is zero");
    check_batch(&ctx, &gpu, &[q]);
}

#[test]
fn coincident_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleQuality::new(&ctx);
    let q = coincident();
    let golden = quality_host(&q);
    assert_eq!(golden.degenerate, 1, "coincident points are degenerate");
    check_batch(&ctx, &gpu, &[q]);
}

#[test]
fn mixed_single_dispatch_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleQuality::new(&ctx);
    let queries = vec![
        equilateral(),
        right_isosceles(),
        sliver(),
        collinear(),
        coincident(),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn randomized_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMeshTriangleQuality::new(&ctx);
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut queries: Vec<MeshTriangleQualityQuery> = Vec::new();
    let mut guard = 0u32;
    while queries.len() < 512 {
        guard += 1;
        assert!(guard < 200_000, "rejection sampling failed to converge");

        let q = MeshTriangleQualityQuery {
            v0: [
                draw(&mut state, -2.0, 2.0),
                draw(&mut state, -2.0, 2.0),
                draw(&mut state, -2.0, 2.0),
            ],
            v1: [
                draw(&mut state, -2.0, 2.0),
                draw(&mut state, -2.0, 2.0),
                draw(&mut state, -2.0, 2.0),
            ],
            v2: [
                draw(&mut state, -2.0, 2.0),
                draw(&mut state, -2.0, 2.0),
                draw(&mut state, -2.0, 2.0),
            ],
        };
        if reject_sample(&q) {
            continue;
        }
        queries.push(q);
    }
    check_batch(&ctx, &gpu, &queries);
}
