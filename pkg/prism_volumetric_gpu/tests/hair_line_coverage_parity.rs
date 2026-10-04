//! Real-device parity for the hair line-coverage twin:
//! [`GpuHairLineCoverage`](prism_volumetric_gpu::hair_line_coverage::GpuHairLineCoverage)
//! must reproduce, for one query per thread, the three closed forms of the
//! golden `prism_render_architecture::hair::line_coverage` module: the finite,
//! non-negative `sanitized_width`, the clamped point-to-segment distance
//! `point_segment_distance` (with its degenerate zero-length collapse), and the
//! monotone trapezoidal coverage ramp `pixel_coverage`.
//!
//! # Independent oracle
//!
//! This suite does not depend on the reference crate. The host [`oracle`] is an
//! independent `f32` reimplementation of the same closed form documented on the
//! twin, evaluated in the same multiply-add order as the kernel. The golden
//! `sanitized_width` is `if width.is_finite() && width > 0.0 { width } else {
//! 0.0 }`; the oracle uses the same `f32::is_finite` and ordered compare so a
//! negative, infinite or `NaN` width takes the same branch as the kernel's
//! exponent-bit guard. A passing run is therefore evidence that the `WGSL`
//! kernel and an independent `CPU` evaluation of the same maps agree, not merely
//! that the shader compiles.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of adds, multiplies,
//! absolute values, one `clamp` and one `sqrt`, so the two evaluations compute
//! the same closed form in the same order. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar host leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every
//! continuous output and an exact `==` on the discrete `valid` flag.
//!
//! # Conditioning
//!
//! The only branch is `point_segment_distance`'s degenerate test
//! `abs(len_sq) < EPS`, where `len_sq` is the squared segment length. Because
//! `len_sq` is a derived sum, a value near `EPS` could send the two evaluators
//! down different branches, so the random sweep is rejection-sampled with the
//! segment length squared comfortably above one — many orders of magnitude clear
//! of `EPS` — keeping both evaluators on the non-degenerate path. The named
//! zero-length fixture instead pins the degenerate case, where both evaluators
//! take the point-distance branch exactly. The coverage ramp's `clamp` is
//! `1`-Lipschitz, so near its knees a few-`ULP` difference in the distance moves
//! the coverage by at most the same amount, well inside the absolute tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::line_coverage`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hair_line_coverage::{
    GpuHairLineCoverage, HairLineCoverageQuery, HairLineCoverageResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity floor on every continuous output.
const ABS_TOL: f32 = 1.0e-4;
/// Relative parity slope on every continuous output.
const REL_TOL: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero magnitudes stay meaningful.
const REL_FLOOR: f32 = 1.0e-6;
/// Squared-length threshold below which a segment is degenerate, matching the
/// golden `EPS`.
const EPS_LOCAL: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the documented parity bound: an
/// absolute floor or a relative term keeping large-magnitude values meaningful.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= ABS_TOL || diff <= REL_TOL * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// `width.is_finite() && width > 0.0`, else `0.0`: an independent `f32`
/// reimplementation of the golden `sanitized_width`.
fn sanitized_width(w: f32) -> f32 {
    if w.is_finite() && w > 0.0 {
        w
    } else {
        0.0
    }
}

/// Independent `f32` reimplementation of `point_segment_distance`, in the same
/// arithmetic order as the kernel, with the degenerate zero-length collapse.
fn point_segment_distance(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let len_sq = abx * abx + aby * aby;
    let apx = p[0] - a[0];
    let apy = p[1] - a[1];
    if len_sq.abs() < EPS_LOCAL {
        return (apx * apx + apy * apy).sqrt();
    }
    let t = ((apx * abx + apy * aby) / len_sq).clamp(0.0, 1.0);
    let cx = a[0] + t * abx;
    let cy = a[1] + t * aby;
    let dx = p[0] - cx;
    let dy = p[1] - cy;
    (dx * dx + dy * dy).sqrt()
}

/// Independent `f32` reimplementation of `pixel_coverage`, returning the clamped
/// trapezoidal ramp and the underlying point-to-segment distance.
fn pixel_coverage(query: &HairLineCoverageQuery) -> (f32, f32) {
    let half_width = sanitized_width(query.width) * 0.5;
    let d = point_segment_distance(query.pixel_center, query.a, query.b);
    let cov = (half_width + 0.5 - d).clamp(0.0, 1.0);
    (cov, d)
}

/// Builds the full host-side reference result for one query.
fn oracle(query: &HairLineCoverageQuery) -> HairLineCoverageResult {
    let (coverage, distance) = pixel_coverage(query);
    HairLineCoverageResult {
        coverage,
        distance,
        valid: 1,
    }
}

/// Pins one `GPU` result against the independent host oracle: both continuous
/// outputs within the parity bound and the discrete `valid` flag exactly.
fn pin(idx: usize, query: &HairLineCoverageQuery, result: &HairLineCoverageResult) {
    let want = oracle(query);
    assert!(
        close(result.coverage, want.coverage),
        "query {idx}: coverage gpu={} oracle={}",
        result.coverage,
        want.coverage
    );
    assert!(
        close(result.distance, want.distance),
        "query {idx}: distance gpu={} oracle={}",
        result.distance,
        want.distance
    );
    assert_eq!(
        result.valid, want.valid,
        "query {idx}: valid gpu={} oracle={}",
        result.valid, want.valid
    );
}

/// Evaluates `queries` on-device and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuHairLineCoverage, queries: &[HairLineCoverageQuery]) {
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

/// 64-bit linear-congruential step (`Knuth`/`PCG` constants), returning the
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

/// A deterministic pseudo-random `f32` in `[-1, 1]`.
fn signed(state: &mut u64) -> f32 {
    unit01(state) * 2.0 - 1.0
}

/// A deterministic, well-conditioned random query: endpoints spread over a
/// modest window with the segment length kept comfortably non-degenerate, the
/// pixel centre dithered around the segment midpoint so the coverage ramp is
/// exercised across its full range, and a finite non-negative width.
fn rand_query(state: &mut u64) -> HairLineCoverageQuery {
    let ax = signed(state) * 10.0;
    let ay = signed(state) * 10.0;
    let (bx, by) = loop {
        let bx = ax + signed(state) * 8.0;
        let by = ay + signed(state) * 8.0;
        let dx = bx - ax;
        let dy = by - ay;
        // Keep the squared segment length far above EPS so both evaluators take
        // the non-degenerate branch.
        if dx * dx + dy * dy > 1.0 {
            break (bx, by);
        }
    };
    let mx = (ax + bx) * 0.5;
    let my = (ay + by) * 0.5;
    let cx = mx + signed(state) * 3.0;
    let cy = my + signed(state) * 3.0;
    let width = unit01(state) * 4.0;
    HairLineCoverageQuery::new([ax, ay], [bx, by], width, [cx, cy])
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLineCoverage::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn center_on_segment_full_coverage() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLineCoverage::new(&ctx);
    // Pixel centre on the segment: distance 0, so coverage saturates to 1.
    let query = HairLineCoverageQuery::new([0.0, 0.0], [2.0, 0.0], 1.0, [1.0, 0.0]);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].coverage, 1.0),
        "centre on segment should fully cover, got {}",
        got[0].coverage
    );
    pin(0, &query, &got[0]);
}

#[test]
fn far_pixel_zero_coverage() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLineCoverage::new(&ctx);
    // A pixel far from the segment is beyond the ramp, so coverage clamps to 0.
    let query = HairLineCoverageQuery::new([0.0, 0.0], [2.0, 0.0], 1.0, [10.0, 10.0]);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].coverage, 0.0),
        "far pixel should have zero coverage, got {}",
        got[0].coverage
    );
    pin(0, &query, &got[0]);
}

#[test]
fn degenerate_zero_length_segment() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLineCoverage::new(&ctx);
    // A zero-length segment collapses to the point distance |p - a| = 5.
    let query = HairLineCoverageQuery::new([1.0, 1.0], [1.0, 1.0], 1.0, [4.0, 5.0]);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].distance, 5.0),
        "degenerate segment distance should be |p - a| = 5, got {}",
        got[0].distance
    );
    assert!(
        close(got[0].coverage, 0.0),
        "degenerate distance 5 is beyond the ramp, coverage should be 0, got {}",
        got[0].coverage
    );
    pin(0, &query, &got[0]);
}

#[test]
fn negative_width_sanitizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLineCoverage::new(&ctx);
    // A negative width sanitizes to 0, so half-width is 0 and a centre on the
    // segment (distance 0) covers exactly the pixel-kernel radius ramp = 0.5.
    let query = HairLineCoverageQuery::new([0.0, 0.0], [2.0, 0.0], -3.0, [1.0, 0.0]);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].coverage, 0.5),
        "negative width should sanitize to 0 (coverage 0.5 at distance 0), got {}",
        got[0].coverage
    );
    pin(0, &query, &got[0]);
}

#[test]
fn nan_inf_width_sanitizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLineCoverage::new(&ctx);
    // NaN and the infinities are non-finite, so they sanitize to width 0 via the
    // kernel's exponent-bit guard; a centre on the segment then covers 0.5.
    let queries = [
        HairLineCoverageQuery::new([0.0, 0.0], [2.0, 0.0], f32::NAN, [1.0, 0.0]),
        HairLineCoverageQuery::new([0.0, 0.0], [2.0, 0.0], f32::INFINITY, [1.0, 0.0]),
        HairLineCoverageQuery::new([0.0, 0.0], [2.0, 0.0], f32::NEG_INFINITY, [1.0, 0.0]),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, (q, r)) in queries.iter().zip(got.iter()).enumerate() {
        assert!(
            close(r.coverage, 0.5),
            "non-finite width {idx} should sanitize to 0 (coverage 0.5), got {}",
            r.coverage
        );
        pin(idx, q, r);
    }
}

#[test]
fn distance_clamps_to_endpoints() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLineCoverage::new(&ctx);
    // A pixel beyond endpoint b measures to b (projection parameter clamped), so
    // the distance is 3, not the perpendicular distance to the infinite line.
    let query = HairLineCoverageQuery::new([0.0, 0.0], [2.0, 0.0], 1.0, [5.0, 0.0]);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].distance, 3.0),
        "distance should clamp to endpoint b (3), got {}",
        got[0].distance
    );
    pin(0, &query, &got[0]);
}

#[test]
fn perpendicular_offset_partial_coverage() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLineCoverage::new(&ctx);
    // A pixel offset perpendicular by 0.7 from a wide stroke lands on the ramp:
    // coverage = clamp(0.5 + 0.5 - 0.7, 0, 1) = 0.3.
    let query = HairLineCoverageQuery::new([0.0, 0.0], [4.0, 0.0], 1.0, [2.0, 0.7]);
    let got = gpu.evaluate(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one result for one query");
    assert!(
        close(got[0].distance, 0.7),
        "perpendicular distance should be 0.7, got {}",
        got[0].distance
    );
    assert!(
        close(got[0].coverage, 0.3),
        "partial coverage should be 0.3, got {}",
        got[0].coverage
    );
    pin(0, &query, &got[0]);
}

#[test]
fn stride_regression_two_element_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLineCoverage::new(&ctx);
    // Two deliberately distinct queries in one batch: if the host `std430` query
    // stride disagreed with the `WGSL` struct, the second lane would decode from
    // the wrong bytes and the pin would fail. The distinct geometry and widths
    // make such a mis-stride observable.
    let queries = [
        HairLineCoverageQuery::new([0.0, 0.0], [3.0, 0.0], 2.0, [1.5, 0.4]),
        HairLineCoverageQuery::new([-2.0, 1.0], [2.0, 5.0], 0.75, [0.0, 4.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLineCoverage::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many well-conditioned
    // random queries, dispatched together so the per-thread indexing and the
    // contiguous storage layout are both exercised.
    let mut queries = vec![
        HairLineCoverageQuery::new([0.0, 0.0], [2.0, 0.0], 1.0, [1.0, 0.0]),
        HairLineCoverageQuery::new([0.0, 0.0], [2.0, 0.0], 1.0, [10.0, 10.0]),
        HairLineCoverageQuery::new([1.0, 1.0], [1.0, 1.0], 1.0, [4.0, 5.0]),
        HairLineCoverageQuery::new([0.0, 0.0], [4.0, 0.0], 1.0, [2.0, 0.7]),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLineCoverage::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) of well-conditioned random
    // queries pins every output across many invocations.
    let queries: Vec<HairLineCoverageQuery> = (0..512).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
