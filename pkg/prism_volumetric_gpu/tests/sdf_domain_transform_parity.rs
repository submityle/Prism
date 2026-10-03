//! Real-device parity for the signed-distance *domain transform* twin:
//! `GpuSdfDomainTransform` must reproduce the `CPU` closed forms of the oracle
//! `prism_render_architecture::ray_scene::sdf_domain` — the planar `rotate_2d`,
//! the Rodrigues `rotate_axis`, the origin-plane `fold_plane` and the
//! offset-plane `fold_plane_offset` — across identity, rotated, positive-side
//! and reflected cases plus a randomized sweep compared query-for-query.
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
//! kernel computes the same domain transforms the reference does.
//!
//! # Parity criterion
//!
//! Each transform threads through products, sums and (for the folds) a `min`
//! branch, so a `GPU` result may land a few units in the last place from the
//! scalar oracle; each component is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, with a `rel_diff` floor of `1e-6` so a near-zero
//! expected component does not inflate the relative error.
//!
//! # Conditioning
//!
//! Fixtures and the randomized sweep stay clear of the only branch cliff in the
//! batch — the fold's signed distance `min(dot(p, n) - offset, 0)`, which
//! switches side at `dot == offset` (and at `dot == 0` for `fold_plane`). The
//! sweep rejects points whose signed distance to either fold plane lands within
//! a small margin of zero, so a last-place wobble never flips which side of the
//! fold is taken. `axis` and `normal` are normalized to unit length (rejecting
//! near-zero draws before the divide), and each `(sin, cos)` pair is placed on
//! the unit circle via `cos = sqrt(1 - sin^2)`, so every rotation stays an
//! isometry and the port matches operation-for-operation.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_domain`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_domain_transform::{
    GpuSdfDomainTransform, SdfDomainTransformQuery, SdfDomainTransformResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each transformed component. A `GPU` multiply-add may land a
/// few units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const SD_ABS: f32 = 1.0e-4;

/// Relative bound on each transformed component, applied for larger magnitudes
/// where a few units in the last place exceed the absolute floor.
const SD_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected component
/// does not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Margin keeping each fold's signed distance clear of zero, where the
/// `min(..., 0)` reflection branch switches side.
const FOLD_MARGIN: f32 = 1.0e-2;

/// Floor on the raw length of `axis`/`normal` before normalizing, so the
/// normalize divide stays well conditioned.
const UNIT_MIN: f32 = 0.3;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Independent reimplementation of the reference `rotate_2d`: the planar
/// rotation matrix `[[cos, -sin], [sin, cos]]` applied to the point.
fn rotate_2d(point: [f32; 2], sin: f32, cos: f32) -> [f32; 2] {
    [
        cos * point[0] - sin * point[1],
        sin * point[0] + cos * point[1],
    ]
}

/// Independent reimplementation of the reference `rotate_axis`: Rodrigues'
/// rotation `v*cos + (axis x v)*sin + axis*(axis . v)*(1 - cos)`.
fn rotate_axis(point: [f32; 3], axis: [f32; 3], sin: f32, cos: f32) -> [f32; 3] {
    let cross = [
        axis[1] * point[2] - axis[2] * point[1],
        axis[2] * point[0] - axis[0] * point[2],
        axis[0] * point[1] - axis[1] * point[0],
    ];
    let axis_dot = axis[0] * point[0] + axis[1] * point[1] + axis[2] * point[2];
    let w = axis_dot * (1.0 - cos);
    [
        point[0] * cos + cross[0] * sin + axis[0] * w,
        point[1] * cos + cross[1] * sin + axis[1] * w,
        point[2] * cos + cross[2] * sin + axis[2] * w,
    ]
}

/// Independent reimplementation of the reference `fold_plane`: reflect the
/// plane's negative side through the origin plane with unit `normal`.
fn fold_plane(point: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    let signed = (point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2]).min(0.0);
    let k = 2.0 * signed;
    [
        point[0] - k * normal[0],
        point[1] - k * normal[1],
        point[2] - k * normal[2],
    ]
}

/// Independent reimplementation of the reference `fold_plane_offset`: reflect
/// across the parallel plane `dot(p, n) = offset`.
fn fold_plane_offset(point: [f32; 3], normal: [f32; 3], offset: f32) -> [f32; 3] {
    let signed =
        (point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2] - offset).min(0.0);
    let k = 2.0 * signed;
    [
        point[0] - k * normal[0],
        point[1] - k * normal[1],
        point[2] - k * normal[2],
    ]
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfDomainTransformQuery) -> SdfDomainTransformResult {
    SdfDomainTransformResult {
        rotated_2d: rotate_2d(q.point_2d, q.rotate_2d_sin_cos[0], q.rotate_2d_sin_cos[1]),
        rotated_axis: rotate_axis(
            q.point_3d,
            q.axis,
            q.rotate_axis_sin_cos[0],
            q.rotate_axis_sin_cos[1],
        ),
        folded_plane: fold_plane(q.point_3d, q.normal),
        folded_plane_offset: fold_plane_offset(q.point_3d, q.normal, q.offset),
    }
}

/// Pins one `GPU` 2-vector against the oracle, component by component.
fn check_vec2(idx: usize, name: &str, got: [f32; 2], want: [f32; 2]) {
    for (axis, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(*g, *w, SD_ABS, SD_REL),
            "query {idx} {name}[{axis}]: gpu {g} vs cpu {w}"
        );
    }
}

/// Pins one `GPU` 3-vector against the oracle, component by component.
fn check_vec3(idx: usize, name: &str, got: [f32; 3], want: [f32; 3]) {
    for (axis, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(*g, *w, SD_ABS, SD_REL),
            "query {idx} {name}[{axis}]: gpu {g} vs cpu {w}"
        );
    }
}

/// Pins one `GPU` result against the host oracle: all four transformed points
/// under the shared tolerance.
fn check_one(idx: usize, got: &SdfDomainTransformResult, want: &SdfDomainTransformResult) {
    check_vec2(idx, "rotated_2d", got.rotated_2d, want.rotated_2d);
    check_vec3(idx, "rotated_axis", got.rotated_axis, want.rotated_axis);
    check_vec3(idx, "folded_plane", got.folded_plane, want.folded_plane);
    check_vec3(
        idx,
        "folded_plane_offset",
        got.folded_plane_offset,
        want.folded_plane_offset,
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfDomainTransform, queries: &[SdfDomainTransformQuery]) {
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

/// Length of a 3-vector via a host-side `sqrt` (not transcendental).
fn length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Returns a unit copy of `v`; callers reject near-zero draws first.
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = length(v);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Builds a `[sin, cos]` pair on the unit circle from a sine in `[-1, 1]`, so
/// the rotation is a true isometry (`sin^2 + cos^2 = 1`).
fn unit_sin_cos(sin: f32) -> [f32; 2] {
    let s = sin.clamp(-1.0, 1.0);
    let c = (1.0 - s * s).max(0.0).sqrt();
    [s, c]
}

/// Signed distance of `point` to the plane `dot(p, n) = offset` (unit `n`).
fn signed_distance(point: [f32; 3], normal: [f32; 3], offset: f32) -> f32 {
    point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2] - offset
}

/// Builds a well-conditioned fixture query from explicit parts, placing both
/// `(sin, cos)` pairs on the unit circle and normalizing `axis`/`normal`.
fn make_query(
    point_2d: [f32; 2],
    sin_2d: f32,
    point_3d: [f32; 3],
    axis: [f32; 3],
    sin_axis: f32,
    normal: [f32; 3],
    offset: f32,
) -> SdfDomainTransformQuery {
    SdfDomainTransformQuery::new(
        point_2d,
        unit_sin_cos(sin_2d),
        point_3d,
        normalize(axis),
        unit_sin_cos(sin_axis),
        normalize(normal),
        offset,
    )
}

/// A fixed battery of named queries spanning identity rotations, quarter and
/// partial turns, positive-side (pass-through) folds and reflected folds. Each
/// point is kept clear of both fold planes by at least the fold margin.
fn fixture_queries() -> Vec<SdfDomainTransformQuery> {
    vec![
        // Identity rotations (sin = 0, cos = 1); point on the positive side of
        // both fold planes so the folds pass through unchanged.
        make_query(
            [1.0, 0.0],
            0.0,
            [1.2, 0.8, 0.6],
            [0.0, 1.0, 0.0],
            0.0,
            [1.0, 0.0, 0.0],
            -0.5,
        ),
        // Quarter turn (sin = 1, cos = 0) in 2D; axis rotation about +y.
        make_query(
            [2.0, 0.0],
            1.0,
            [0.5, 1.3, 0.4],
            [0.0, 1.0, 0.0],
            1.0,
            [0.0, 1.0, 0.0],
            -0.4,
        ),
        // Negative-side point: fold_plane reflects it across the origin plane.
        make_query(
            [-1.5, 0.7],
            0.6,
            [-2.0, 0.3, 0.5],
            [1.0, 0.0, 0.0],
            0.5,
            [1.0, 0.0, 0.0],
            0.5,
        ),
        // Diagonal axis rotation; point below an offset plane with diagonal
        // normal is reflected by both folds.
        make_query(
            [0.4, -1.1],
            -0.7,
            [-1.0, -1.2, -0.8],
            [1.0, 1.0, 1.0],
            0.8,
            [1.0, 1.0, 0.0],
            0.6,
        ),
        // Point clearly on the positive side of a tilted offset plane.
        make_query(
            [3.0, 2.0],
            0.3,
            [2.5, 2.2, 1.9],
            [0.2, 0.9, 0.3],
            -0.4,
            [0.3, 0.6, 0.7],
            -0.8,
        ),
        // Axis along +z, moderate turn; point reflected by the origin fold.
        make_query(
            [-0.6, -0.9],
            -0.4,
            [0.7, -0.5, -2.1],
            [0.0, 0.0, 1.0],
            0.9,
            [0.0, 0.0, 1.0],
            0.3,
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
        eprintln!("skipping sdf_domain_transform parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfDomainTransform::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn rotate_2d_preserves_length_for_a_unit_pair() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainTransform::new(&ctx);
    let q = make_query(
        [1.3, -0.7],
        0.5,
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        0.0,
        [1.0, 0.0, 0.0],
        -1.0,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    // A unit (sin, cos) pair makes rotate_2d an isometry: length is preserved.
    let r = got[0].rotated_2d;
    let in_len = (q.point_2d[0] * q.point_2d[0] + q.point_2d[1] * q.point_2d[1]).sqrt();
    let out_len = (r[0] * r[0] + r[1] * r[1]).sqrt();
    assert!(
        close(in_len, out_len, SD_ABS, SD_REL),
        "rotate_2d should preserve length: {in_len} vs {out_len}"
    );
}

#[test]
fn rotate_axis_preserves_length_about_a_unit_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainTransform::new(&ctx);
    let q = make_query(
        [0.0, 0.0],
        0.0,
        [1.4, -0.6, 2.0],
        [0.3, 0.9, 0.2],
        0.7,
        [1.0, 0.0, 0.0],
        -2.0,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    // Rodrigues' rotation about a unit axis with a unit (sin, cos) pair is an
    // isometry: it preserves the point's length.
    let in_len = length(q.point_3d);
    let out_len = length(got[0].rotated_axis);
    assert!(
        close(in_len, out_len, SD_ABS, SD_REL),
        "rotate_axis should preserve length: {in_len} vs {out_len}"
    );
}

#[test]
fn fold_plane_passes_positive_side_and_reflects_negative_side() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainTransform::new(&ctx);
    let normal = normalize([1.0, 0.0, 0.0]);
    // Positive-side point passes through unchanged; negative-side is mirrored.
    let positive = make_query(
        [0.0, 0.0],
        0.0,
        [1.5, 0.4, 0.3],
        [0.0, 1.0, 0.0],
        0.0,
        normal,
        0.0,
    );
    let negative = make_query(
        [0.0, 0.0],
        0.0,
        [-1.5, 0.4, 0.3],
        [0.0, 1.0, 0.0],
        0.0,
        normal,
        0.0,
    );
    let got = gpu.evaluate(&ctx, &[positive, negative]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&positive));
    check_one(1, &got[1], &oracle(&negative));
    // The positive-side point is unchanged.
    check_vec3(0, "folded_plane", got[0].folded_plane, positive.point_3d);
    // The negative-side point is reflected to +x.
    assert!(
        got[1].folded_plane[0] > 0.0,
        "fold_plane should mirror the negative side to the positive side: {:?}",
        got[1].folded_plane
    );
}

#[test]
fn fold_plane_offset_folds_about_the_offset_plane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainTransform::new(&ctx);
    let normal = normalize([0.0, 1.0, 0.0]);
    let offset = 1.0;
    // Below the plane y = 1: reflected upward. Above it: unchanged.
    let below = make_query(
        [0.0, 0.0],
        0.0,
        [0.5, 0.2, 0.3],
        [1.0, 0.0, 0.0],
        0.0,
        normal,
        offset,
    );
    let above = make_query(
        [0.0, 0.0],
        0.0,
        [0.5, 1.8, 0.3],
        [1.0, 0.0, 0.0],
        0.0,
        normal,
        offset,
    );
    let got = gpu.evaluate(&ctx, &[below, above]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&below));
    check_one(1, &got[1], &oracle(&above));
    // The below-plane point is reflected across y = 1 to y = 1.8.
    assert!(
        got[0].folded_plane_offset[1] > offset,
        "fold_plane_offset should mirror a below-plane point above it: {:?}",
        got[0].folded_plane_offset
    );
    // The above-plane point passes through unchanged.
    check_vec3(
        1,
        "folded_plane_offset",
        got[1].folded_plane_offset,
        above.point_3d,
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainTransform::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

/// Builds one well-conditioned random query: the 2D and 3D points in a bounded
/// box, unit `(sin, cos)` pairs and unit `axis`/`normal`. Points whose signed
/// distance to either fold plane lands within the fold margin of zero are
/// rejected (see `# Conditioning`).
fn random_query(state: &mut u64) -> SdfDomainTransformQuery {
    loop {
        let point_2d = [uniform(state, -4.0, 4.0), uniform(state, -4.0, 4.0)];
        let point_3d = [
            uniform(state, -4.0, 4.0),
            uniform(state, -4.0, 4.0),
            uniform(state, -4.0, 4.0),
        ];
        let axis = [
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
        ];
        let normal = [
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
        ];
        let sin_2d = uniform(state, -1.0, 1.0);
        let sin_axis = uniform(state, -1.0, 1.0);
        let offset = uniform(state, -2.0, 2.0);

        // Reject near-degenerate axis/normal before the normalize divide.
        if length(axis) < UNIT_MIN || length(normal) < UNIT_MIN {
            continue;
        }
        let unit_normal = normalize(normal);

        // Reject points whose signed distance to either fold plane sits on the
        // reflection cliff (dot == 0 for fold_plane, dot == offset for the
        // offset fold), so a last-place wobble cannot flip the branch.
        if signed_distance(point_3d, unit_normal, 0.0).abs() < FOLD_MARGIN {
            continue;
        }
        if signed_distance(point_3d, unit_normal, offset).abs() < FOLD_MARGIN {
            continue;
        }

        return SdfDomainTransformQuery::new(
            point_2d,
            unit_sin_cos(sin_2d),
            point_3d,
            normalize(axis),
            unit_sin_cos(sin_axis),
            unit_normal,
            offset,
        );
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfDomainTransform::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin every
    // reported transform across a wide span of points and orientations.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
