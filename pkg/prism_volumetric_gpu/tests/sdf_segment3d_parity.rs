//! Real-device parity for the exact axis/segment signed-distance twin:
//! `GpuSdfSegment3d` must reproduce the `CPU` closed forms of the analytic
//! oracle `prism_render_architecture::ray_scene::sdf_primitives` — the infinite
//! `line_sdf`, the Inigo-Quilez `infinite_cylinder`, the Inigo-Quilez
//! `infinite_cone` and the Inigo-Quilez `cylinder_segment` — across interior,
//! exterior, on-surface, on-axis and reduction cases plus a randomized sweep
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
//! *independent* reimplementation of the same four closed forms in scalar
//! `f32`, operation-for-operation. Because the reference and this oracle are
//! both scalar `f32`, a `GPU == oracle` pass is direct evidence the ported
//! kernel computes the same signed distance the reference does.
//!
//! # Parity criterion
//!
//! Each distance threads through `sqrt`, products and quotients, so a `GPU`
//! result may land a few units in the last place from the scalar oracle; each
//! is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a
//! `rel_diff` floor of `1e-6` so a near-zero expected value does not inflate the
//! relative error.
//!
//! # Conditioning
//!
//! The randomized sweep draws the point in `[-4, 4]^3` and positive shape
//! extents. The `line_sdf` direction is kept non-degenerate (its length squared
//! stays above a floor so the projection divide is well conditioned); the
//! `cylinder_segment` endpoints are kept distinct (`baba` above a floor so the
//! final `1 / baba` scaling is well conditioned). It rejects samples where the
//! `infinite_cone` sign predicate `cos * qx - sin * qy` falls within a small
//! margin of zero and where the `cylinder_segment` interior/exterior split term
//! `max(x, y)` falls within a `baba`-scaled margin of zero, since there a
//! last-place wobble could flip a non-tiny distance's sign between the `CPU`
//! and `GPU`. Everywhere else every intermediate is a well-conditioned
//! non-negative square root, a `min`/`max` split (continuous across its tie) or
//! a guarded quotient, so a `GPU` evaluation lands a few units in the last
//! place from the scalar oracle and never straddles a branch cliff.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_segment3d::{GpuSdfSegment3d, SdfSegment3dQuery, SdfSegment3dResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each signed distance. A `GPU` `sqrt`/divide may land a few
/// units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const SD_ABS: f32 = 1.0e-4;

/// Relative bound on each signed distance, applied for larger magnitudes where
/// a few units in the last place exceed the absolute floor.
const SD_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Margin keeping the randomized sweep clear of the `infinite_cone` sign-flip
/// predicate, where a last-place wobble could flip a non-tiny distance's sign.
const CONE_SIGN_MARGIN: f32 = 2.0e-3;

/// Floor on the `line_sdf` direction's length squared so the projection divide
/// stays well conditioned.
const LINE_DD_MIN: f32 = 0.25;

/// Floor on the `cylinder_segment` endpoint separation squared (`baba`) so the
/// final `1 / baba` scaling stays well conditioned.
const SEG_BABA_MIN: f32 = 0.5;

/// `baba`-scaled margin keeping the randomized sweep clear of the
/// `cylinder_segment` interior/exterior split term `max(x, y)`.
const SEG_SPLIT_MARGIN: f32 = 1.0e-2;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Independent reimplementation of the reference `line_sdf`: the unsigned
/// length of the component of `point` perpendicular to the (non-zero) `line`.
fn line_sdf(point: [f32; 3], direction: [f32; 3]) -> f32 {
    let dd =
        direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
    let t = (point[0] * direction[0] + point[1] * direction[1] + point[2] * direction[2]) / dd;
    let rx = point[0] - direction[0] * t;
    let ry = point[1] - direction[1] * t;
    let rz = point[2] - direction[2] * t;
    (rx * rx + ry * ry + rz * rz).sqrt()
}

/// Independent reimplementation of the reference `infinite_cylinder`: the
/// `xz`-plane radial offset from the axis minus the radius.
fn infinite_cylinder(point: [f32; 3], axis_xz: [f32; 2], radius: f32) -> f32 {
    let dx = point[0] - axis_xz[0];
    let dz = point[2] - axis_xz[1];
    (dx * dx + dz * dz).sqrt() - radius
}

/// Independent reimplementation of the reference `infinite_cone`: the meridian
/// flank residual length, signed on the axis side of the flank.
fn infinite_cone(point: [f32; 3], sin_cos: [f32; 2]) -> f32 {
    let qx = (point[0] * point[0] + point[2] * point[2]).sqrt();
    let qy = point[1];
    let proj = (qx * sin_cos[0] + qy * sin_cos[1]).max(0.0);
    let wx = qx - sin_cos[0] * proj;
    let wy = qy - sin_cos[1] * proj;
    let d = (wx * wx + wy * wy).sqrt();
    if sin_cos[1] * qx - sin_cos[0] * qy < 0.0 {
        -d
    } else {
        d
    }
}

/// Independent reimplementation of the reference `cylinder_segment`: the exact
/// signed distance to a finite capped cylinder between distinct endpoints.
fn cylinder_segment(point: [f32; 3], a: [f32; 3], b: [f32; 3], radius: f32) -> f32 {
    let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let pa = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];
    let baba = ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2];
    let paba = pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2];
    let perp = [
        pa[0] * baba - ba[0] * paba,
        pa[1] * baba - ba[1] * paba,
        pa[2] * baba - ba[2] * paba,
    ];
    let x = (perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2]).sqrt() - radius * baba;
    let y = (paba - baba * 0.5).abs() - baba * 0.5;
    let x2 = x * x;
    let y2 = y * y * baba;
    let d = if x.max(y) < 0.0 {
        -(x2.min(y2))
    } else {
        (if x > 0.0 { x2 } else { 0.0 }) + (if y > 0.0 { y2 } else { 0.0 })
    };
    d.signum() * d.abs().sqrt() / baba
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfSegment3dQuery) -> SdfSegment3dResult {
    SdfSegment3dResult {
        line_sd: line_sdf(q.point, q.line_direction),
        infinite_cylinder_sd: infinite_cylinder(q.point, q.cyl_axis_xz, q.cyl_radius),
        infinite_cone_sd: infinite_cone(q.point, q.cone_sin_cos),
        cylinder_segment_sd: cylinder_segment(q.point, q.seg_a, q.seg_b, q.seg_radius),
    }
}

/// Pins one `GPU` result against the host oracle: each of the four signed
/// distances under the shared tolerance.
fn check_one(idx: usize, got: &SdfSegment3dResult, want: &SdfSegment3dResult) {
    assert!(
        close(got.line_sd, want.line_sd, SD_ABS, SD_REL),
        "query {idx} line_sd: gpu {} vs cpu {}",
        got.line_sd,
        want.line_sd
    );
    assert!(
        close(
            got.infinite_cylinder_sd,
            want.infinite_cylinder_sd,
            SD_ABS,
            SD_REL
        ),
        "query {idx} infinite_cylinder_sd: gpu {} vs cpu {}",
        got.infinite_cylinder_sd,
        want.infinite_cylinder_sd
    );
    assert!(
        close(got.infinite_cone_sd, want.infinite_cone_sd, SD_ABS, SD_REL),
        "query {idx} infinite_cone_sd: gpu {} vs cpu {}",
        got.infinite_cone_sd,
        want.infinite_cone_sd
    );
    assert!(
        close(
            got.cylinder_segment_sd,
            want.cylinder_segment_sd,
            SD_ABS,
            SD_REL
        ),
        "query {idx} cylinder_segment_sd: gpu {} vs cpu {}",
        got.cylinder_segment_sd,
        want.cylinder_segment_sd
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfSegment3d, queries: &[SdfSegment3dQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, result, &want);
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

/// Draws a `f32` in `[lo, hi)` from the generator, using only integer-to-float
/// division (no transcendental).
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = lcg(state) as f32 * (1.0 / 4_294_967_296.0);
    lo + (hi - lo) * u
}

/// Builds a query at `point` with a fixed, well-conditioned set of shape
/// parameters shared by the named fixtures: a `y`-axis line, a `y`-parallel
/// unit-ish cylinder centred on the origin, a `30`-degree-ish cone opening
/// along `+y`, and a `y`-axis capped segment of half-height `1`.
fn query_at(point: [f32; 3]) -> SdfSegment3dQuery {
    // Aperture `[sin, cos]` of a cone whose half-angle is near 30 degrees; the
    // pair is a unit vector, derived with a host-side sqrt (not transcendental).
    let sin = 0.5_f32;
    let cos = (1.0_f32 - sin * sin).sqrt();
    SdfSegment3dQuery::new(
        point,
        [0.0, 1.0, 0.0],  // line_direction: y axis
        [0.0, 0.0],       // cyl_axis_xz
        0.8,              // cyl_radius
        [sin, cos],       // cone_sin_cos
        [0.0, -1.0, 0.0], // seg_a
        [0.0, 1.0, 0.0],  // seg_b
        0.5,              // seg_radius
    )
}

/// A fixed battery of named points spanning interior, exterior, on-surface,
/// on-axis and off-axis cases, dispatched together.
fn fixture_queries() -> Vec<SdfSegment3dQuery> {
    vec![
        // Deep interior near the origin.
        query_at([0.0, 0.0, 0.0]),
        // On the central axis, above and below.
        query_at([0.0, 2.0, 0.0]),
        query_at([0.0, -2.0, 0.0]),
        // Radially outside the lateral walls.
        query_at([2.0, 0.0, 0.0]),
        query_at([0.0, 0.0, 2.5]),
        // Diagonally outside a top corner.
        query_at([1.5, 1.5, 0.0]),
        // Diagonally outside a bottom corner.
        query_at([1.3, -1.4, 0.6]),
        // Just inside the lateral wall.
        query_at([0.6, 0.2, 0.1]),
        // Mid-height off the axis.
        query_at([0.3, 0.0, 0.2]),
        // Interior of the cone, above the apex.
        query_at([0.0, 1.0, 0.0]),
        // Far away in every direction.
        query_at([3.0, 3.0, 3.0]),
        query_at([-3.0, -2.0, 1.0]),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_segment3d parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfSegment3d::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn line_distance_is_nonnegative_and_zero_on_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSegment3d::new(&ctx);
    // On the y axis (the line direction) the perpendicular residual is zero.
    let on_axis = query_at([0.0, 3.0, 0.0]);
    // Off the axis the distance is the xz radial offset.
    let off_axis = query_at([2.0, 1.0, 0.0]);
    let got = gpu.evaluate(&ctx, &[on_axis, off_axis]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&on_axis));
    check_one(1, &got[1], &oracle(&off_axis));
    assert!(
        got[0].line_sd <= SD_ABS,
        "on-axis line distance should be ~0: {:?}",
        got[0]
    );
    assert!(
        got[1].line_sd > 0.0,
        "off-axis line distance should be positive: {:?}",
        got[1]
    );
}

#[test]
fn infinite_cylinder_interior_is_negative_exterior_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSegment3d::new(&ctx);
    let inside = query_at([0.0, 5.0, 0.0]);
    let outside = query_at([2.0, 0.0, 0.0]);
    let got = gpu.evaluate(&ctx, &[inside, outside]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&inside));
    check_one(1, &got[1], &oracle(&outside));
    assert!(
        got[0].infinite_cylinder_sd < 0.0,
        "interior infinite-cylinder distance should be negative: {:?}",
        got[0]
    );
    assert!(
        got[1].infinite_cylinder_sd > 0.0,
        "exterior infinite-cylinder distance should be positive: {:?}",
        got[1]
    );
}

#[test]
fn infinite_cone_interior_is_negative_exterior_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSegment3d::new(&ctx);
    // On the axis above the apex, well inside the solid cone.
    let inside = query_at([0.0, 1.0, 0.0]);
    // Far off the flank, outside the cone.
    let outside = query_at([3.0, 0.1, 0.0]);
    let got = gpu.evaluate(&ctx, &[inside, outside]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&inside));
    check_one(1, &got[1], &oracle(&outside));
    assert!(
        got[0].infinite_cone_sd < 0.0,
        "interior infinite-cone distance should be negative: {:?}",
        got[0]
    );
    assert!(
        got[1].infinite_cone_sd > 0.0,
        "exterior infinite-cone distance should be positive: {:?}",
        got[1]
    );
}

#[test]
fn cylinder_segment_interior_is_negative_exterior_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSegment3d::new(&ctx);
    let inside = query_at([0.0, 0.0, 0.0]);
    let outside = query_at([2.0, 0.0, 0.0]);
    let got = gpu.evaluate(&ctx, &[inside, outside]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&inside));
    check_one(1, &got[1], &oracle(&outside));
    assert!(
        got[0].cylinder_segment_sd < 0.0,
        "interior cylinder-segment distance should be negative: {:?}",
        got[0]
    );
    assert!(
        got[1].cylinder_segment_sd > 0.0,
        "exterior cylinder-segment distance should be positive: {:?}",
        got[1]
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSegment3d::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

/// Builds one well-conditioned random query: the point drawn in `[-4, 4]^3`,
/// positive extents, a non-degenerate line direction and distinct segment
/// endpoints. Samples near the `infinite_cone` sign predicate or the
/// `cylinder_segment` interior/exterior split are rejected (see
/// `# Conditioning`).
fn random_query(state: &mut u64) -> SdfSegment3dQuery {
    loop {
        let point = [
            uniform(state, -4.0, 4.0),
            uniform(state, -4.0, 4.0),
            uniform(state, -4.0, 4.0),
        ];
        let line_direction = [
            uniform(state, -2.0, 2.0),
            uniform(state, -2.0, 2.0),
            uniform(state, -2.0, 2.0),
        ];
        let cyl_axis_xz = [uniform(state, -1.0, 1.0), uniform(state, -1.0, 1.0)];
        let cyl_radius = uniform(state, 0.2, 3.0);
        // Aperture as a unit vector; sqrt is not transcendental.
        let sin = uniform(state, 0.2, 0.9);
        let cos = (1.0_f32 - sin * sin).sqrt();
        let cone_sin_cos = [sin, cos];
        let seg_a = [
            uniform(state, -2.0, 2.0),
            uniform(state, -2.0, 2.0),
            uniform(state, -2.0, 2.0),
        ];
        let seg_b = [
            uniform(state, -2.0, 2.0),
            uniform(state, -2.0, 2.0),
            uniform(state, -2.0, 2.0),
        ];
        let seg_radius = uniform(state, 0.2, 2.0);

        // Reject a near-degenerate line direction (projection divide).
        let dd = line_direction[0] * line_direction[0]
            + line_direction[1] * line_direction[1]
            + line_direction[2] * line_direction[2];
        if dd < LINE_DD_MIN {
            continue;
        }

        // Reject near-coincident segment endpoints (final 1/baba scaling).
        let ba = [
            seg_b[0] - seg_a[0],
            seg_b[1] - seg_a[1],
            seg_b[2] - seg_a[2],
        ];
        let baba = ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2];
        if baba < SEG_BABA_MIN {
            continue;
        }

        // Reject samples near the cone sign predicate, where a last-place
        // wobble could flip a non-tiny distance's sign between CPU and GPU.
        let qx = (point[0] * point[0] + point[2] * point[2]).sqrt();
        let qy = point[1];
        if (cos * qx - sin * qy).abs() < CONE_SIGN_MARGIN {
            continue;
        }

        // Reject samples near the cylinder-segment interior/exterior split.
        let pa = [
            point[0] - seg_a[0],
            point[1] - seg_a[1],
            point[2] - seg_a[2],
        ];
        let paba = pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2];
        let perp = [
            pa[0] * baba - ba[0] * paba,
            pa[1] * baba - ba[1] * paba,
            pa[2] * baba - ba[2] * paba,
        ];
        let x =
            (perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2]).sqrt() - seg_radius * baba;
        let y = (paba - baba * 0.5).abs() - baba * 0.5;
        if x.max(y).abs() < SEG_SPLIT_MARGIN * baba {
            continue;
        }

        return SdfSegment3dQuery::new(
            point,
            line_direction,
            cyl_axis_xz,
            cyl_radius,
            cone_sin_cos,
            seg_a,
            seg_b,
            seg_radius,
        );
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSegment3d::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin every
    // reported distance across a wide span of points and shape extents.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
