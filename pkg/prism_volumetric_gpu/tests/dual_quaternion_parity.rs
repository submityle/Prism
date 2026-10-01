//! Real-device parity for the dual-quaternion rigid-transform twin:
//! [`GpuDualQuaternion`](prism_volumetric_gpu::dual_quaternion::GpuDualQuaternion)
//! must reproduce the `CPU` golden
//! [`dual_quaternion`](prism_render_architecture::particle::dual_quaternion)
//! across a pure translation (identity rotation), a 90° rotation about `+z`, a
//! non-unit rotation quaternion that is renormalized to the same rotation, a
//! degenerate zero rotation that falls back to the identity (point only
//! translated), and randomized batches compared element-for-element.
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
//! `rel_diff <= 1e-3` on every `f32` lane of the transformed point.
//!
//! # Conditioning
//!
//! Every fixture is kept well away from the single degeneracy crack: the random
//! batch rejects any rotation quaternion whose squared norm is near the compare
//! epsilon, so `CPU` and `GPU` stay on the same side of the fallback branch
//! regardless of a few units in the last place of slack. The degenerate fixture
//! uses an exactly zero rotation quaternion so both devices take the identity
//! fallback. No `f32` transcendental method appears: rotation quaternions are
//! built from integer components or the allowed `sqrt`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::dual_quaternion`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::dual_quaternion::Quat;
use prism_volumetric_gpu::dual_quaternion::{golden, DualQuatTransformQuery, GpuDualQuaternion};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
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

/// `sqrt(1/2)`, the half-angle sine/cosine of a 90° rotation, derived with the
/// allowed `sqrt` rather than a transcendental call.
fn sqrt_half() -> f32 {
    0.5_f32.sqrt()
}

/// Unit rotation of 90° about the `+z` axis.
fn rot_z90() -> Quat {
    Quat::new(0.0, 0.0, sqrt_half(), sqrt_half())
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

/// A pseudo-random point with each component in `[-span, span)`.
fn rand_point(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// Builds a clearly-conditioned transform query by rejection sampling: the
/// rotation quaternion is drawn from four signed components and accepted only
/// when its squared norm is comfortably above the compare epsilon, so the
/// identity fallback is never on a tie. The transform renormalizes it, so a
/// non-unit draw is fine. Translation and point are well-spread.
fn rand_query(state: &mut u64) -> DualQuatTransformQuery {
    loop {
        let rot = Quat::new(
            signed(state, 1.0),
            signed(state, 1.0),
            signed(state, 1.0),
            signed(state, 1.0),
        );
        let norm_sq = rot.x * rot.x + rot.y * rot.y + rot.z * rot.z + rot.w * rot.w;
        if norm_sq < 0.5 {
            continue;
        }
        let translation = rand_point(state, 4.0);
        let point = rand_point(state, 5.0);
        return DualQuatTransformQuery::new(rot, translation, point);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: every lane of the
/// transformed point must agree within bound.
fn pin(idx: usize, query: &DualQuatTransformQuery, got: &[f32; 3]) {
    let want = golden(query);
    for (lane, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(close(*g, *w), "query {idx} lane {lane}: gpu {g} vs cpu {w}");
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuDualQuaternion, queries: &[DualQuatTransformQuery]) {
    let got = gpu.transform(ctx, queries);
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
    let gpu = GpuDualQuaternion::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.transform(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn identity_rotation_is_pure_translation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuaternion::new(&ctx);
    // The identity rotation leaves the point untouched, so the transform is a
    // pure translation. Integer geometry is exact on both devices.
    let query = DualQuatTransformQuery::new(Quat::identity(), [3.0, -4.0, 5.0], [1.0, 2.0, 3.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn z90_rotation_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuaternion::new(&ctx);
    // A 90° rotation about +z sends (1, 0, 0) to (0, 1, 0); with a translation
    // the result is (0, 1, 0) + t.
    let query = DualQuatTransformQuery::new(rot_z90(), [2.0, -1.0, 0.5], [1.0, 0.0, 0.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn non_unit_rotation_normalizes_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuaternion::new(&ctx);
    // A non-unit quaternion that renormalizes to the same +z 90° rotation; the
    // transform's guarded normalize must recover the unit rotation.
    let query = DualQuatTransformQuery::new(
        Quat::new(0.0, 0.0, 2.0, 2.0),
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn degenerate_zero_rotation_falls_back_to_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuaternion::new(&ctx);
    // A zero rotation quaternion has no stable direction, so the guarded
    // normalize falls back to the identity rotation and the point is only
    // translated. Both devices take the fallback for an exactly zero norm.
    let query = DualQuatTransformQuery::new(
        Quat::new(0.0, 0.0, 0.0, 0.0),
        [1.0, 2.0, 3.0],
        [4.0, -5.0, 6.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDualQuaternion::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random queries,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        DualQuatTransformQuery::new(Quat::identity(), [3.0, -4.0, 5.0], [1.0, 2.0, 3.0]),
        DualQuatTransformQuery::new(rot_z90(), [2.0, -1.0, 0.5], [1.0, 0.0, 0.0]),
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
    let gpu = GpuDualQuaternion::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins the transformed point
    // across many random rotations, translations and points.
    let queries: Vec<DualQuatTransformQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
