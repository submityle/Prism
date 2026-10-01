//! Real-device parity for the capsule-vs-capsule closest twin:
//! [`GpuCapsuleCapsuleClosest`](prism_volumetric_gpu::capsule_capsule_closest::GpuCapsuleCapsuleClosest)
//! must reproduce the `CPU` golden
//! [`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest)
//! across parallel separated capsules (analytic perpendicular gap), crossing
//! capsules whose surfaces overlap (positive penetration, flipped sign), a
//! collinear end-to-end pair whose surfaces just touch (gap exactly zero), a
//! collinear overlapping-axis pair that drives `axis_dist` to zero and forces
//! the stable default normal, an endpoint-cap pair (both parameters pinned to a
//! bound), a capsule-vs-sphere pair (capsule B collapsed to a point), and a
//! randomized batch of clearly-conditioned skew pairs compared
//! element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each pair is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every `f32` field and an exact match on the discrete
//! `intersecting` flag.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from a branch tie and from the
//! degeneracy cracks: skew pairs have a determinant far above the compare
//! epsilon with both parameters clearly interior and an axis distance
//! comfortably above `EPS` so the unit normal is well defined, the parallel and
//! collinear fixtures use integer coordinates whose geometry is exact on both
//! devices, and the collapsed capsule has coincident endpoints (squared length
//! exactly zero). This keeps `CPU` and `GPU` on the same side of every branch
//! regardless of a few units in the last place of slack.
//!
//! Provenance: twinned from this repository's
//! [`capsule_capsule_closest`](prism_render_architecture::particle::capsule_capsule_closest);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::capsule_capsule_closest::{
    capsule_capsule_closest, segment_segment_closest, v_length, v_sub, CapsuleHit,
};
use prism_volumetric_gpu::capsule_capsule_closest::{
    CapsuleClosestQuery, CapsuleClosestResult, GpuCapsuleCapsuleClosest,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
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

/// Asserts two vectors agree channel-for-channel within the parity bound.
fn close_vec(label: &str, idx: usize, got: [f32; 3], want: [f32; 3]) {
    assert!(
        close(got[0], want[0]) && close(got[1], want[1]) && close(got[2], want[2]),
        "query {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got[0],
        got[1],
        got[2],
        want[0],
        want[1],
        want[2]
    );
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

/// A pseudo-random vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// Builds a clearly-conditioned skew capsule pair by rejection sampling: two
/// core axes are drawn from well-spread endpoints and accepted only when the
/// `CPU` segment-segment core places both parameters comfortably interior
/// (`s` and `t` in `[0.08, 0.92]`) and the two axis closest points stay a safe
/// distance apart (so the unit normal is far from the coincident-axis
/// degeneracy). Each capsule gets a small positive radius.
fn skew_query(state: &mut u64) -> CapsuleClosestQuery {
    loop {
        let a0 = rand_vec(state, 4.0);
        let a1 = [
            a0[0] + signed(state, 3.0),
            a0[1] + signed(state, 3.0),
            a0[2] + 3.0 + lcg(state),
        ];
        let b0 = rand_vec(state, 4.0);
        let b1 = [
            b0[0] + 3.0 + lcg(state),
            b0[1] + signed(state, 3.0),
            b0[2] + signed(state, 3.0),
        ];

        let (s, t, pa, pb) = segment_segment_closest(a0, a1, b0, b1);
        if !(0.08..=0.92).contains(&s) || !(0.08..=0.92).contains(&t) {
            continue;
        }
        let axis_dist = v_length(v_sub(pb, pa));
        if axis_dist < 0.5 {
            continue;
        }

        let ra = lcg(state) * 0.3 + 0.05;
        let rb = lcg(state) * 0.3 + 0.05;
        return CapsuleClosestQuery::new(a0, a1, ra, b0, b1, rb);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the signed gap,
/// the penetration depth, the discrete intersecting flag, the unit normal and
/// the two surface closest points must all agree within bound.
fn pin(idx: usize, query: &CapsuleClosestQuery, got: &CapsuleClosestResult) {
    let want: CapsuleHit =
        capsule_capsule_closest(query.a0, query.a1, query.ra, query.b0, query.b1, query.rb);

    assert!(
        close(got.distance, want.distance),
        "query {idx} distance: gpu {} vs cpu {}",
        got.distance,
        want.distance
    );
    assert!(
        close(got.penetration, want.penetration),
        "query {idx} penetration: gpu {} vs cpu {}",
        got.penetration,
        want.penetration
    );
    assert_eq!(
        got.intersecting, want.intersecting,
        "query {idx} intersecting: gpu {} vs cpu {}",
        got.intersecting, want.intersecting
    );
    close_vec("normal", idx, got.normal, want.normal);
    close_vec("point_a", idx, got.point_a, want.point_a);
    close_vec("point_b", idx, got.point_b, want.point_b);
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuCapsuleCapsuleClosest, queries: &[CapsuleClosestQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleCapsuleClosest::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn parallel_separated_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleCapsuleClosest::new(&ctx);
    // Two x-aligned capsules 3 apart in y, each radius 0.5: the analytic surface
    // gap is 2 and the normal is +y. Integer geometry is exact on both devices.
    let query = CapsuleClosestQuery::new(
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        0.5,
        [0.0, 3.0, 0.0],
        [2.0, 3.0, 0.0],
        0.5,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn crossing_overlap_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleCapsuleClosest::new(&ctx);
    // A along x, B along y offset by 0.5 in z; axes pass within 0.5 so the
    // surfaces overlap (negative gap, positive penetration), normal is +z.
    let query = CapsuleClosestQuery::new(
        [-2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        0.5,
        [0.0, -2.0, 0.5],
        [0.0, 2.0, 0.5],
        0.5,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn collinear_just_touch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleCapsuleClosest::new(&ctx);
    // Collinear x capsules whose axis gap of 1 exactly equals ra + rb: the
    // surfaces touch, gap is zero, penetration is zero, intersecting is true.
    let query = CapsuleClosestQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.5,
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        0.5,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn coincident_axes_use_default_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleCapsuleClosest::new(&ctx);
    // Overlapping x ranges drive axis_dist to 0, so the stable default normal
    // [0, 1, 0] is returned and the penetration is the full radius sum.
    let query = CapsuleClosestQuery::new(
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        1.0,
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        1.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn endpoint_caps_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleCapsuleClosest::new(&ctx);
    // Collinear x capsules with a clear gap: the closest axis points fall on the
    // inner endpoints (s = 1, t = 0) and the normal is +x.
    let query = CapsuleClosestQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.5,
        [5.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        0.5,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn capsule_vs_sphere_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleCapsuleClosest::new(&ctx);
    // Capsule B collapsed to a point (coincident endpoints) is a sphere; the
    // second-degenerate branch projects it onto capsule A. Integer 3-4-5 offset
    // keeps the axis distance exact on both devices.
    let query = CapsuleClosestQuery::new(
        [-2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        0.5,
        [0.0, 3.0, 4.0],
        [0.0, 3.0, 4.0],
        1.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleCapsuleClosest::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random skew pairs,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        CapsuleClosestQuery::new(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.5,
            [0.0, 3.0, 0.0],
            [2.0, 3.0, 0.0],
            0.5,
        ),
        CapsuleClosestQuery::new(
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.5,
            [0.0, -2.0, 0.5],
            [0.0, 2.0, 0.5],
            0.5,
        ),
    ];
    for _ in 0..48 {
        queries.push(skew_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_skew_pairs_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleCapsuleClosest::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned skew pairs (several workgroups'
    // worth) pins every reported field across many random capsule geometries.
    let queries: Vec<CapsuleClosestQuery> = (0..200).map(|_| skew_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
