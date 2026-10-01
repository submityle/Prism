//! Real-device parity for the affine `AABB`-transform twin:
//! [`GpuAabbTransform`](prism_volumetric_gpu::aabb_transform::GpuAabbTransform)
//! must reproduce the `CPU` golden
//! [`Aabb::transform_by_mat4`](prism_render_architecture::particle::aabb_transform::Aabb::transform_by_mat4)
//! across the identity map, a pure translation, a per-axis scale with a negative
//! (flipping) component, an exact integer `90`-degree rotation about `z`, a
//! Pythagorean rotation composed with scale and translation, a flat
//! (zero-thickness) box, a degenerate (empty) box that must stay empty, and a
//! randomized batch of well-conditioned matrix/box pairs compared
//! element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each box is a fixed, non-reorderable sequence of multiplies and adds, so
//! `CPU` and `GPU` evaluate the same closed form. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The
//! comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on both
//! transformed corners.
//!
//! # Conditioning
//!
//! The rotation fixtures use exact integer or Pythagorean (`0.6` / `0.8`)
//! matrix entries so no transcendental call ever appears, and every random
//! fixture draws moderate rational entries so the products stay far from
//! overflow and the two devices share the same empty-vs-non-empty branch. The
//! empty-box fixture seeds `min > max`, which both devices map to the identical
//! `f32::MAX` / `f32::MIN` sentinel.
//!
//! Provenance: twinned from this repository's
//! [`aabb_transform`](prism_render_architecture::particle::aabb_transform);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::aabb_transform::Aabb;
use prism_volumetric_gpu::aabb_transform::{
    AabbTransformQuery, AabbTransformResult, GpuAabbTransform,
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

/// Asserts two 3-component corners agree channel-for-channel within the bound.
fn close3(label: &str, idx: usize, got: [f32; 3], want: [f32; 3]) {
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

/// Builds a well-conditioned random query: a moderate-magnitude affine matrix
/// (all sixteen entries rational and in `[-2, 2)` save a `1` in the final lane)
/// paired with a non-empty box drawn from a random center and a strictly
/// positive extent. Both devices run Arvo's method on the same inputs, so no
/// orthonormality is required: the only requirement is a shared non-empty
/// branch, which a positive extent guarantees.
fn rand_query(state: &mut u64) -> AabbTransformQuery {
    let matrix = [
        [
            signed(state, 2.0),
            signed(state, 2.0),
            signed(state, 2.0),
            0.0,
        ],
        [
            signed(state, 2.0),
            signed(state, 2.0),
            signed(state, 2.0),
            0.0,
        ],
        [
            signed(state, 2.0),
            signed(state, 2.0),
            signed(state, 2.0),
            0.0,
        ],
        [
            signed(state, 4.0),
            signed(state, 4.0),
            signed(state, 4.0),
            1.0,
        ],
    ];
    let center = [signed(state, 5.0), signed(state, 5.0), signed(state, 5.0)];
    // Strictly positive extents (0.5 .. 4.5) keep the box non-empty.
    let extent = [
        lcg(state) * 4.0 + 0.5,
        lcg(state) * 4.0 + 0.5,
        lcg(state) * 4.0 + 0.5,
    ];
    AabbTransformQuery::new(matrix, Aabb::from_center_extent(center, extent))
}

/// Pins one `GPU` result against the `CPU` golden for `query`: both transformed
/// corners must agree within bound.
fn pin(idx: usize, query: &AabbTransformQuery, got: &AabbTransformResult) {
    let want = query.aabb.transform_by_mat4(&query.matrix);
    close3("min", idx, got.transformed.min, want.min);
    close3("max", idx, got.transformed.max, want.max);
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuAabbTransform, queries: &[AabbTransformQuery]) {
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

/// The column-major identity `mat4`.
const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbTransform::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn identity_is_noop() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbTransform::new(&ctx);
    // The identity map leaves the box unchanged.
    let query = AabbTransformQuery::new(IDENTITY, Aabb::new([-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]));
    check(&ctx, &gpu, &[query]);
}

#[test]
fn translation_shifts_box() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbTransform::new(&ctx);
    // A pure translation in the fourth column moves both corners by the same
    // offset, leaving the extent untouched.
    let mut m = IDENTITY;
    m[3] = [10.0, -5.0, 2.0, 1.0];
    let query = AabbTransformQuery::new(m, Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    check(&ctx, &gpu, &[query]);
}

#[test]
fn negative_scale_flips_without_inverting_box() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbTransform::new(&ctx);
    // A per-axis scale whose x component is negative mirrors the box; Arvo's
    // |m[j][i]| keeps the half-extent positive so the result stays a valid box.
    let mut m = IDENTITY;
    m[0][0] = -3.0;
    m[1][1] = 4.0;
    m[2][2] = 0.5;
    let query = AabbTransformQuery::new(
        m,
        Aabb::from_center_extent([1.0, -2.0, 3.0], [2.0, 2.0, 2.0]),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn rotation_90_about_z_swaps_axes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbTransform::new(&ctx);
    // Column-major 90-degree rotation about z with exact integer entries, so the
    // matrix is identical on both devices.
    let m = [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let query = AabbTransformQuery::new(m, Aabb::new([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]));
    check(&ctx, &gpu, &[query]);
}

#[test]
fn pythagorean_rotation_scale_translation_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbTransform::new(&ctx);
    // A 3-4-5 in-plane rotation about z (orthonormal, no transcendentals: the
    // local names cos/sin are literal rational constants), composed with a scale
    // on z and a translation.
    let (cos, sin) = (0.6f32, 0.8f32);
    let m = [
        [cos, sin, 0.0, 0.0],
        [-sin, cos, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [7.0, -3.0, 1.0, 1.0],
    ];
    let query = AabbTransformQuery::new(m, Aabb::new([-1.0, -2.0, -0.5], [3.0, 1.0, 2.5]));
    check(&ctx, &gpu, &[query]);
}

#[test]
fn flat_box_stays_flat() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbTransform::new(&ctx);
    // A zero-thickness box on z (min.z == max.z): a valid non-empty degenerate
    // box whose transformed half-extent on the mapped axis stays zero.
    let (cos, sin) = (0.6f32, 0.8f32);
    let m = [
        [cos, sin, 0.0, 0.0],
        [-sin, cos, 0.0, 0.0],
        [0.0, 0.0, 1.5, 0.0],
        [2.0, 2.0, 2.0, 1.0],
    ];
    let query = AabbTransformQuery::new(m, Aabb::new([-2.0, -1.0, 4.0], [2.0, 3.0, 4.0]));
    check(&ctx, &gpu, &[query]);
}

#[test]
fn empty_box_stays_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbTransform::new(&ctx);
    // An empty input box (min > max on every axis) maps to the empty-box
    // sentinel on both devices, independent of the matrix.
    let mut m = IDENTITY;
    m[3] = [5.0, 6.0, 7.0, 1.0];
    let query = AabbTransformQuery::new(m, Aabb::empty());
    let got = gpu.eval(&ctx, &[query]);
    assert_eq!(got.len(), 1, "one input yields one result");
    assert!(
        got[0].transformed.is_empty(),
        "an empty box must transform to an empty box, got {:?}",
        got[0].transformed
    );
    // And it must match the reference sentinel field-for-field.
    pin(0, &query, &got[0]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbTransform::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random pairs, dispatched
    // together so the per-thread indexing and contiguous storage layout are both
    // exercised, then pinned element-for-element.
    let mut m_scale = IDENTITY;
    m_scale[0][0] = -3.0;
    m_scale[1][1] = 4.0;
    m_scale[2][2] = 0.5;
    let mut queries = vec![
        AabbTransformQuery::new(IDENTITY, Aabb::new([-1.0, -2.0, -3.0], [4.0, 5.0, 6.0])),
        AabbTransformQuery::new(
            m_scale,
            Aabb::from_center_extent([1.0, -2.0, 3.0], [2.0, 2.0, 2.0]),
        ),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_random_pairs_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAabbTransform::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins both transformed corners
    // across many random matrix/box pairs.
    let queries: Vec<AabbTransformQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
