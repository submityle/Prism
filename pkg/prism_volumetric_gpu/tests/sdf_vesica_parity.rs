//! Real-device parity for the exact analytic signed-distance twin:
//! [`GpuSdfVesica`](prism_volumetric_gpu::sdf_vesica::GpuSdfVesica) must
//! reproduce the `CPU` closed form of three primitives from the oracle
//! `prism_render_architecture::ray_scene::sdf_primitives` — the 2D vesica
//! `vesica_2d`, the arbitrary-segment 3D vesica `vesica_segment` and the 2D
//! uneven capsule `uneven_capsule_2d` — across the inside, surface and outside
//! of each shape, every selection branch, and a randomized sweep compared
//! query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same three closed forms (Inigo
//! Quilez's `sdVesica`, segment `sdVesicaSegment` and `sdUnevenCapsule`),
//! transcribed from the reference signatures. Because the reference and this
//! oracle are both scalar `f32`, a `GPU == oracle` pass is direct evidence the
//! ported kernel computes the same signed distances the reference does.
//!
//! # Parity criterion
//!
//! Every distance threads through `sqrt`, products and quotients, so a `GPU`
//! result may land a few units in the last place from the scalar oracle; each
//! of the three distances is asserted within `abs_diff <= 1e-5` or
//! `rel_diff <= 1e-4`, with a relative floor of `1e-6` so a near-zero expected
//! value does not inflate the relative error.
//!
//! # Conditioning
//!
//! Each primitive selects its nearest feature with one linear predicate, and a
//! query sitting exactly on a predicate's switch-over has an ambiguous branch
//! whose two sides can round differently. The named fixtures stay well clear of
//! those cliffs, and the randomized sweep rejection-samples any candidate whose
//! branch gap — `(|py|-b)*offset - |px|*b` for the vesica, `r*qx - d*(qy-r)`
//! for the segment, and both `k` and `k - a*h` for the uneven capsule — falls
//! inside a guard band. The sweep also constrains the inputs so every
//! intermediate square root is well defined: `radius > offset > 0` (so
//! `b = sqrt(radius^2 - offset^2)` is real), `0.5*l > width > 0` (so the
//! segment `d` is a positive, well-conditioned quotient) and
//! `|r_bottom - r_top| < h` with all three positive (so the uneven slope
//! `a = sqrt(1 - b^2)` is real).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_vesica::{GpuSdfVesica, SdfVesicaQuery, SdfVesicaResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each signed distance. A `GPU` `sqrt`/divide may land a few
/// units in the last place from the scalar oracle; `1e-5` admits that legal
/// slack while still failing a genuinely wrong port.
const DIST_ABS: f32 = 1.0e-5;

/// Relative bound on each signed distance, applied for larger magnitudes where
/// a few units in the last place exceed the absolute floor.
const DIST_REL: f32 = 1.0e-4;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Guard band keeping the sweep clear of every branch switch-over, in the units
/// of each branch gap.
const BRANCH_GUARD: f32 = 0.08;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Scalar 2-vector Euclidean length, matching the reference `length2`.
fn len2(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

/// Scalar 3-vector Euclidean length, matching the reference `length`.
fn len3(x: f32, y: f32, z: f32) -> f32 {
    (x * x + y * y + z * z).sqrt()
}

/// Independent reimplementation of the reference `vesica_2d`: fold the query to
/// the first quadrant, then choose the nearest feature — a cusp or one of the
/// two arcs — with a single linear test.
fn vesica_2d(point: [f32; 2], radius: f32, offset: f32) -> f32 {
    let px = point[0].abs();
    let py = point[1].abs();
    let b = (radius * radius - offset * offset).sqrt();
    if (py - b) * offset > px * b {
        len2(px, py - b) * offset.signum()
    } else {
        len2(px + offset, py) - radius
    }
}

/// Independent reimplementation of the reference `vesica_segment`: project onto
/// the segment axis, reduce to an axial/radial pair, then measure against the
/// shared tip or the generating arc.
fn vesica_segment(point: [f32; 3], a: [f32; 3], b: [f32; 3], width: f32) -> f32 {
    let c = [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ];
    let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let l = len3(ba[0], ba[1], ba[2]);
    let v = [ba[0] / l, ba[1] / l, ba[2] / l];
    let pc = [point[0] - c[0], point[1] - c[1], point[2] - c[2]];
    let y = pc[0] * v[0] + pc[1] * v[1] + pc[2] * v[2];
    let perp = [pc[0] - y * v[0], pc[1] - y * v[1], pc[2] - y * v[2]];
    let qx = len3(perp[0], perp[1], perp[2]);
    let qy = y.abs();
    let r = 0.5 * l;
    let d = 0.5 * (r * r - width * width) / width;
    let (hx, hy, hz) = if r * qx < d * (qy - r) {
        (0.0f32, r, 0.0f32)
    } else {
        (-d, 0.0f32, d + width)
    };
    len2(qx - hx, qy - hy) - hz
}

/// Independent reimplementation of the reference `uneven_capsule_2d`: fold `x`
/// to its magnitude, then one slope test selects the bottom cap, the top cap or
/// the tangent flank.
fn uneven_capsule_2d(point: [f32; 2], r_bottom: f32, r_top: f32, h: f32) -> f32 {
    let p = [point[0].abs(), point[1]];
    let b = (r_bottom - r_top) / h;
    let a = (1.0 - b * b).max(0.0).sqrt();
    let k = -b * p[0] + a * p[1];
    if k < 0.0 {
        len2(p[0], p[1]) - r_bottom
    } else if k > a * h {
        len2(p[0], p[1] - h) - r_top
    } else {
        a * p[0] + b * p[1] - r_bottom
    }
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfVesicaQuery) -> SdfVesicaResult {
    SdfVesicaResult {
        vesica_2d_value: vesica_2d(q.vesica_point, q.vesica_radius, q.vesica_offset),
        vesica_segment_value: vesica_segment(
            q.segment_point,
            q.segment_a,
            q.segment_b,
            q.segment_width,
        ),
        uneven_capsule_2d_value: uneven_capsule_2d(
            q.uneven_point,
            q.uneven_r_bottom,
            q.uneven_r_top,
            q.uneven_h,
        ),
    }
}

/// Pins one `GPU` result against the host oracle: each of the three signed
/// distances under the shared bound.
fn check_one(idx: usize, got: &SdfVesicaResult, want: &SdfVesicaResult) {
    assert!(
        close(
            got.vesica_2d_value,
            want.vesica_2d_value,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} vesica_2d: gpu {} vs cpu {}",
        got.vesica_2d_value,
        want.vesica_2d_value
    );
    assert!(
        close(
            got.vesica_segment_value,
            want.vesica_segment_value,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} vesica_segment: gpu {} vs cpu {}",
        got.vesica_segment_value,
        want.vesica_segment_value
    );
    assert!(
        close(
            got.uneven_capsule_2d_value,
            want.uneven_capsule_2d_value,
            DIST_ABS,
            DIST_REL
        ),
        "query {idx} uneven_capsule_2d: gpu {} vs cpu {}",
        got.uneven_capsule_2d_value,
        want.uneven_capsule_2d_value
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfVesica, queries: &[SdfVesicaQuery]) {
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

/// Signed branch gap of `vesica_2d`: the vesica picks the cusp when this is
/// positive and an arc when negative, so the sweep rejects candidates whose gap
/// sits inside the guard band.
fn vesica_gap(point: [f32; 2], radius: f32, offset: f32) -> f32 {
    let px = point[0].abs();
    let py = point[1].abs();
    let b = (radius * radius - offset * offset).sqrt();
    (py - b) * offset - px * b
}

/// Signed branch gap of `vesica_segment`: the segment picks the shared tip when
/// this is negative and the generating arc when positive.
fn segment_gap(point: [f32; 3], a: [f32; 3], b: [f32; 3], width: f32) -> f32 {
    let c = [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ];
    let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let l = len3(ba[0], ba[1], ba[2]);
    let v = [ba[0] / l, ba[1] / l, ba[2] / l];
    let pc = [point[0] - c[0], point[1] - c[1], point[2] - c[2]];
    let y = pc[0] * v[0] + pc[1] * v[1] + pc[2] * v[2];
    let perp = [pc[0] - y * v[0], pc[1] - y * v[1], pc[2] - y * v[2]];
    let qx = len3(perp[0], perp[1], perp[2]);
    let qy = y.abs();
    let r = 0.5 * l;
    let d = 0.5 * (r * r - width * width) / width;
    r * qx - d * (qy - r)
}

/// The two branch gaps of `uneven_capsule_2d`: `k` separates the bottom cap
/// from the flank, and `k - a*h` the flank from the top cap.
fn uneven_gaps(point: [f32; 2], r_bottom: f32, r_top: f32, h: f32) -> (f32, f32) {
    let px = point[0].abs();
    let py = point[1];
    let b = (r_bottom - r_top) / h;
    let a = (1.0 - b * b).max(0.0).sqrt();
    let k = -b * px + a * py;
    (k, k - a * h)
}

/// Draws one well-conditioned random query, rejection-sampling until every
/// primitive's inputs are valid and its branch gaps clear the guard band.
fn random_query(state: &mut u64) -> SdfVesicaQuery {
    loop {
        // vesica: radius > offset > 0 keeps `b` real; the point roams a box
        // comfortably larger than the lens.
        let vesica_radius = uniform(state, 1.5, 3.0);
        let vesica_offset = uniform(state, 0.3, vesica_radius - 0.4);
        let vesica_point = [uniform(state, -4.0, 4.0), uniform(state, -4.0, 4.0)];

        // segment: endpoints in a box, half-thickness small enough that
        // `0.5*l > width` keeps `d` positive and well conditioned.
        let segment_a = [
            uniform(state, -2.5, 2.5),
            uniform(state, -2.5, 2.5),
            uniform(state, -2.5, 2.5),
        ];
        let segment_b = [
            uniform(state, -2.5, 2.5),
            uniform(state, -2.5, 2.5),
            uniform(state, -2.5, 2.5),
        ];
        let segment_width = uniform(state, 0.3, 0.9);
        let segment_point = [
            uniform(state, -3.5, 3.5),
            uniform(state, -3.5, 3.5),
            uniform(state, -3.5, 3.5),
        ];

        // uneven capsule: all positive and `|r_bottom - r_top| < h` keeps `a`
        // real.
        let uneven_r_bottom = uniform(state, 0.3, 1.4);
        let uneven_r_top = uniform(state, 0.3, 1.4);
        let uneven_h = uniform(state, (uneven_r_bottom - uneven_r_top).abs() + 0.6, 3.0);
        let uneven_point = [uniform(state, -3.0, 3.0), uniform(state, -1.5, 4.5)];

        let seg_len = len3(
            segment_b[0] - segment_a[0],
            segment_b[1] - segment_a[1],
            segment_b[2] - segment_a[2],
        );
        if seg_len <= 2.0 * segment_width + 0.6 {
            continue;
        }

        if vesica_gap(vesica_point, vesica_radius, vesica_offset).abs() < BRANCH_GUARD {
            continue;
        }
        if segment_gap(segment_point, segment_a, segment_b, segment_width).abs() < BRANCH_GUARD {
            continue;
        }
        let (k, k_minus) = uneven_gaps(uneven_point, uneven_r_bottom, uneven_r_top, uneven_h);
        if k.abs() < BRANCH_GUARD || k_minus.abs() < BRANCH_GUARD {
            continue;
        }

        return SdfVesicaQuery::new(
            vesica_point,
            vesica_radius,
            vesica_offset,
            segment_point,
            segment_a,
            segment_b,
            segment_width,
            uneven_point,
            uneven_r_bottom,
            uneven_r_top,
            uneven_h,
        );
    }
}

/// A fixed battery of named cases. Each query carries all three primitives, and
/// across the battery every selection branch and the inside/surface/outside of
/// each shape is covered; the comment on each case names the branch it targets
/// per primitive.
fn fixture_queries() -> Vec<SdfVesicaQuery> {
    vec![
        // vesica cusp branch, outside, high on the y-axis; segment tip branch,
        // outside past the taper; uneven bottom-cap branch, outside below.
        SdfVesicaQuery::new(
            [0.1, 4.0],
            2.0,
            1.0,
            [2.5, 0.05, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.5,
            [0.3, -1.5],
            1.0,
            0.5,
            2.0,
        ),
        // vesica arc branch, outside to the right; segment arc branch, outside;
        // uneven top-cap branch, just outside the top disc.
        SdfVesicaQuery::new(
            [1.5, 0.5],
            2.0,
            1.0,
            [1.0, 1.4, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.5,
            [0.2, 2.5],
            1.0,
            0.5,
            2.0,
        ),
        // vesica arc branch, inside the lens; segment arc branch, inside the
        // tube; uneven flank branch, just outside the tangent.
        SdfVesicaQuery::new(
            [0.3, 0.0],
            2.0,
            1.0,
            [1.0, 0.2, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.5,
            [1.0, 1.0],
            1.0,
            0.5,
            2.0,
        ),
        // vesica cusp branch, mirrored into negative x to exercise the fold;
        // segment tip branch far past the end; uneven bottom cap, deep inside.
        SdfVesicaQuery::new(
            [-0.08, 3.5],
            2.5,
            1.2,
            [-1.5, 0.04, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.5,
            [0.1, -0.3],
            1.0,
            0.5,
            2.0,
        ),
        // vesica arc branch with negative x and y (full fold); segment arc on a
        // tilted axis; uneven flank with negative x fold.
        SdfVesicaQuery::new(
            [-1.6, -0.4],
            2.2,
            0.9,
            [0.5, 1.1, 0.3],
            [-1.0, -0.5, 0.0],
            [1.5, 0.5, 1.0],
            0.4,
            [-1.0, 1.1],
            0.9,
            0.5,
            2.2,
        ),
        // vesica deep inside near origin; segment inside near centre; uneven
        // deep inside the bottom disc.
        SdfVesicaQuery::new(
            [0.05, 0.1],
            2.0,
            1.0,
            [1.0, 0.1, 0.05],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.5,
            [0.3, -0.4],
            1.2,
            0.6,
            2.0,
        ),
        // vesica arc outside far right; segment outside far off-axis; uneven top
        // cap outside, mirrored x.
        SdfVesicaQuery::new(
            [3.0, 0.2],
            1.8,
            0.7,
            [1.0, 2.5, 1.5],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.6,
            [-0.3, 3.2],
            1.1,
            0.4,
            2.5,
        ),
        // vesica cusp outside, offset lens; segment tip at the other end;
        // uneven flank with a near-cylindrical taper.
        SdfVesicaQuery::new(
            [0.12, 2.8],
            1.6,
            0.6,
            [-0.4, -0.03, 0.0],
            [0.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            0.45,
            [0.9, 1.3],
            0.8,
            0.7,
            2.4,
        ),
        // Equal radii uneven capsule (b = 0, a = 1, a plain stadium); vesica arc
        // outside; segment arc inside.
        SdfVesicaQuery::new(
            [1.4, 0.6],
            2.0,
            1.0,
            [0.8, 0.15, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.5,
            [0.6, 0.9],
            0.7,
            0.7,
            2.0,
        ),
        // Near-circular vesica (small offset); tilted segment arc; tall uneven
        // capsule flank.
        SdfVesicaQuery::new(
            [0.9, 1.2],
            2.3,
            0.35,
            [1.2, -0.8, 0.6],
            [-0.5, -1.0, -0.5],
            [1.5, 1.0, 0.5],
            0.35,
            [0.7, 2.0],
            0.6,
            0.5,
            3.0,
        ),
        // vesica surface-adjacent arc (small positive distance); segment arc
        // just outside; uneven bottom cap just below the origin.
        SdfVesicaQuery::new(
            [1.0, 0.3],
            2.0,
            1.0,
            [1.0, 1.1, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.5,
            [0.2, -0.1],
            1.0,
            0.5,
            2.0,
        ),
        // A mixed stress case: wide vesica, long tilted segment, asymmetric
        // uneven capsule, all comfortably off every branch cliff.
        SdfVesicaQuery::new(
            [2.1, 1.4],
            2.8,
            1.5,
            [2.0, 1.5, -1.0],
            [-1.5, -1.0, -1.0],
            [2.0, 1.5, 1.0],
            0.5,
            [1.3, 0.9],
            1.3,
            0.5,
            2.6,
        ),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_vesica parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfVesica::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn vesica_inside_is_negative() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfVesica::new(&ctx);
    // A point near the lens centre sits inside all three shapes, so every
    // reported distance is negative.
    let q = SdfVesicaQuery::new(
        [0.05, 0.1],
        2.0,
        1.0,
        [1.0, 0.1, 0.05],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        0.5,
        [0.3, -0.4],
        1.2,
        0.6,
        2.0,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].vesica_2d_value < 0.0
            && got[0].vesica_segment_value < 0.0
            && got[0].uneven_capsule_2d_value < 0.0,
        "an interior point reports negative distances: {:?}",
        got[0]
    );
}

#[test]
fn vesica_outside_is_positive() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfVesica::new(&ctx);
    // A point far outside every shape reports positive distances.
    let q = SdfVesicaQuery::new(
        [3.0, 4.0],
        2.0,
        1.0,
        [5.0, 3.0, 2.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        0.5,
        [3.0, 5.0],
        1.0,
        0.5,
        2.0,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].vesica_2d_value > 0.0
            && got[0].vesica_segment_value > 0.0
            && got[0].uneven_capsule_2d_value > 0.0,
        "an exterior point reports positive distances: {:?}",
        got[0]
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfVesica::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfVesica::new(&ctx);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin every
    // reported distance across a wide span of shapes and query points.
    for _ in 0..384 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
