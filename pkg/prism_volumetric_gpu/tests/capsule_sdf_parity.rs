//! Real-device parity for the capsule-`SDF` twin:
//! [`GpuCapsuleSdf`](prism_volumetric_gpu::capsule_sdf::GpuCapsuleSdf) must
//! reproduce the `CPU` golden
//! [`capsule_sdf`](prism_render_architecture::particle::capsule_sdf) across a
//! point outside the capsule flank (positive signed distance), a point inside
//! the swept volume (negative signed distance), a point beyond an endpoint cap
//! (projection pinned to the nearer end), a point on the surface (signed
//! distance about zero), a collapsed-segment capsule that falls back to a sphere
//! distance about `a`, and a randomized batch compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every `f32` field.
//!
//! # Conditioning
//!
//! Every fixture is kept well away from the single degeneracy crack: the random
//! batch rejects any capsule whose core-axis squared length is near the compare
//! epsilon, so `CPU` and `GPU` stay on the same side of the collapse branch
//! regardless of a few units in the last place of slack. The collapsed-segment
//! fixture uses coincident endpoints (squared length exactly zero) so both
//! devices take the sphere fallback.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::capsule_sdf`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::capsule_sdf::Vec3;
use prism_volumetric_gpu::capsule_sdf::{golden, CapsuleSdfQuery, CapsuleSdfResult, GpuCapsuleSdf};
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
fn rand_vec(state: &mut u64, span: f32) -> Vec3 {
    Vec3::new(
        signed(state, span),
        signed(state, span),
        signed(state, span),
    )
}

/// Builds a clearly-conditioned capsule query by rejection sampling: the core
/// axis is drawn from well-spread endpoints and accepted only when its squared
/// length is comfortably above the compare epsilon, so the collapse branch is
/// never on a tie. The sweep radius is a small positive value.
fn rand_query(state: &mut u64) -> CapsuleSdfQuery {
    loop {
        let a = rand_vec(state, 4.0);
        let b = rand_vec(state, 4.0);
        let d = b.minus(a);
        if d.length_squared() < 0.5 {
            continue;
        }
        let point = rand_vec(state, 5.0);
        let r = lcg(state) * 0.9 + 0.1;
        return CapsuleSdfQuery::new(point, a, b, r);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: both the unsigned
/// segment distance and the signed capsule distance must agree within bound.
fn pin(idx: usize, query: &CapsuleSdfQuery, got: &CapsuleSdfResult) {
    let want = golden(query);
    assert!(
        close(got.segment_distance, want.segment_distance),
        "query {idx} segment_distance: gpu {} vs cpu {}",
        got.segment_distance,
        want.segment_distance
    );
    assert!(
        close(got.signed_distance, want.signed_distance),
        "query {idx} signed_distance: gpu {} vs cpu {}",
        got.signed_distance,
        want.signed_distance
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuCapsuleSdf, queries: &[CapsuleSdfQuery]) {
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
    let gpu = GpuCapsuleSdf::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn outside_flank_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleSdf::new(&ctx);
    // An x-aligned capsule of radius 0.5; the point is 2 above the mid-axis, so
    // the perpendicular projection lands interior and the signed distance is
    // 2 - 0.5 = 1.5. Integer geometry is exact on both devices.
    let query = CapsuleSdfQuery::new(
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        0.5,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn inside_volume_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleSdf::new(&ctx);
    // The point sits 0.25 off the axis of a radius-1 capsule, so it is inside the
    // swept volume and the signed distance is negative (0.25 - 1 = -0.75).
    let query = CapsuleSdfQuery::new(
        Vec3::new(0.0, 0.25, 0.0),
        Vec3::new(-2.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        1.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn endpoint_cap_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleSdf::new(&ctx);
    // The point is beyond the b endpoint, so the clamped projection pins to b and
    // the distance is measured from the cap. A 3-4-5 offset keeps the length
    // exact on both devices: segment distance 5, signed 5 - 0.5 = 4.5.
    let query = CapsuleSdfQuery::new(
        Vec3::new(4.0, 3.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        0.5,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn on_surface_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleSdf::new(&ctx);
    // The point is exactly one radius off the axis, so the signed distance is
    // about zero while the unsigned segment distance equals the radius.
    let query = CapsuleSdfQuery::new(
        Vec3::new(0.5, 0.75, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        0.75,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn collapsed_segment_is_sphere_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleSdf::new(&ctx);
    // Coincident endpoints collapse the core axis (squared length exactly zero),
    // so both devices take the sphere fallback: the distance is measured from the
    // shared endpoint. A 3-4-0 offset keeps the length exact: segment 5, signed
    // 5 - 1 = 4.
    let query = CapsuleSdfQuery::new(
        Vec3::new(3.0, 4.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        1.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleSdf::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random queries,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        CapsuleSdfQuery::new(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            0.5,
        ),
        CapsuleSdfQuery::new(
            Vec3::new(0.0, 0.25, 0.0),
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            1.0,
        ),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapsuleSdf::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins both distances across many
    // random capsule geometries.
    let queries: Vec<CapsuleSdfQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
