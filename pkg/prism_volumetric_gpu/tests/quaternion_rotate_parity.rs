//! Real-device parity for the quaternion-rotation twin:
//! [`GpuQuaternionRotate`](prism_volumetric_gpu::quaternion_rotate::GpuQuaternionRotate)
//! must reproduce the `CPU` golden
//! [`quaternion_rotate`](prism_render_architecture::particle::quaternion_rotate)
//! across the identity rotation (vector unchanged), a `90`-degree turn about
//! `z` (the `x` axis maps to `y`), a `180`-degree turn about `z` (the `x` axis
//! flips), an unnormalized `quaternion` that renormalizes to a unit rotation, a
//! near-degenerate `quaternion` that must still produce a finite result, and a
//! randomized batch compared element-for-element.
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
//! `rel_diff <= 1e-3` on every `f32` field.
//!
//! # Conditioning
//!
//! Every fixture is kept well away from the single degeneracy crack: the random
//! batch rejects any `quaternion` whose squared length is near the compare
//! epsilon, so `CPU` and `GPU` stay on the same side of the identity fallback
//! regardless of a few units in the last place of slack. The near-degenerate
//! fixture uses an exact zero `quaternion` so both devices take the identity
//! fallback.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::quaternion_rotate`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::quaternion_rotate::Quat;
use prism_volumetric_gpu::quaternion_rotate::{
    golden, GpuQuaternionRotate, QuatRotateQuery, QuatRotateResult,
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

/// Half-angle cosine/sine for a `90`-degree rotation (`45`-degree half-angle),
/// i.e. `1/sqrt(2)`, taken from the core constant to avoid a transcendental
/// call.
const HALF_90: f32 = core::f32::consts::FRAC_1_SQRT_2;

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

/// Builds a clearly-conditioned query by rejection sampling: the `quaternion`
/// components are drawn from a well-spread range and accepted only when the
/// squared length is comfortably above the compare epsilon, so the identity
/// fallback is never on a tie. The vector is unconstrained.
fn rand_query(state: &mut u64) -> QuatRotateQuery {
    loop {
        let quat = Quat::new(
            signed(state, 2.0),
            signed(state, 2.0),
            signed(state, 2.0),
            signed(state, 2.0),
        );
        let len_sq = quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w;
        if len_sq < 0.25 {
            continue;
        }
        let vector = [signed(state, 5.0), signed(state, 5.0), signed(state, 5.0)];
        return QuatRotateQuery::new(quat, vector);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: both the
/// renormalized `quaternion` and the rotated vector must agree within bound.
fn pin(idx: usize, query: &QuatRotateQuery, got: &QuatRotateResult) {
    let want = golden(query);
    assert!(
        close(got.normalized.x, want.normalized.x)
            && close(got.normalized.y, want.normalized.y)
            && close(got.normalized.z, want.normalized.z)
            && close(got.normalized.w, want.normalized.w),
        "query {idx} normalized: gpu {:?} vs cpu {:?}",
        got.normalized,
        want.normalized
    );
    assert!(
        close(got.rotated[0], want.rotated[0])
            && close(got.rotated[1], want.rotated[1])
            && close(got.rotated[2], want.rotated[2]),
        "query {idx} rotated: gpu {:?} vs cpu {:?}",
        got.rotated,
        want.rotated
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuQuaternionRotate, queries: &[QuatRotateQuery]) {
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
    let gpu = GpuQuaternionRotate::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn identity_leaves_vector_unchanged() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuaternionRotate::new(&ctx);
    // The identity rotation leaves any vector unchanged; integer geometry is
    // exact on both devices.
    let query = QuatRotateQuery::new(Quat::identity(), [1.0, 2.0, 3.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn ninety_about_z_maps_x_to_y() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuaternionRotate::new(&ctx);
    // A 90-degree turn about z: the unit quaternion is (0, 0, sin45, cos45), and
    // the x axis maps to y.
    let query = QuatRotateQuery::new(Quat::new(0.0, 0.0, HALF_90, HALF_90), [1.0, 0.0, 0.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn one_eighty_about_z_flips_x() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuaternionRotate::new(&ctx);
    // A 180-degree turn about z: the unit quaternion is (0, 0, 1, 0), and the x
    // axis flips to -x. Integer components keep this exact on both devices.
    let query = QuatRotateQuery::new(Quat::new(0.0, 0.0, 1.0, 0.0), [1.0, 0.0, 0.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn unnormalized_quaternion_renormalizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuaternionRotate::new(&ctx);
    // A unit 90-degree-about-z quaternion scaled by 3; renormalization recovers
    // the unit rotation and the rotated vector matches the normalized rotation.
    let query = QuatRotateQuery::new(
        Quat::new(0.0, 0.0, 3.0 * HALF_90, 3.0 * HALF_90),
        [1.0, 0.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn zero_quaternion_falls_back_to_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuaternionRotate::new(&ctx);
    // An exact zero quaternion has an undefined direction; both devices take the
    // identity fallback, so the vector is returned unchanged.
    let query = QuatRotateQuery::new(Quat::new(0.0, 0.0, 0.0, 0.0), [1.0, 2.0, 3.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuaternionRotate::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random queries,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        QuatRotateQuery::new(Quat::identity(), [1.0, 2.0, 3.0]),
        QuatRotateQuery::new(Quat::new(0.0, 0.0, HALF_90, HALF_90), [1.0, 0.0, 0.0]),
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
    let gpu = GpuQuaternionRotate::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins both answers across many
    // random orientations and vectors.
    let queries: Vec<QuatRotateQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
