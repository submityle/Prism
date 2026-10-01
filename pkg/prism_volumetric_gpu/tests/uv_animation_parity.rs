//! Real-device parity for the `UV`-animation twin:
//! [`GpuUvAnimation`](prism_volumetric_gpu::uv_animation::GpuUvAnimation) must
//! reproduce the `CPU` golden
//! [`uv_animation`](prism_render_architecture::particle::uv_animation) across the
//! atomic scroll / tile / rotate steps, the time-driven scroll offset, the
//! half-open wrap, the flowmap dual panner and the composed `apply`.
//!
//! The fixtures mirror the shapes the golden unit tests call out: a componentwise
//! scroll, a tile that scales about and fixes its pivot, a `+90` degree rotation
//! and an identity rotation, a linear scroll offset, the full
//! pivot-scale-rotate-scroll `apply` order with a pivot that maps to
//! `pivot + scroll`, positive and negative wrap folds, and a dual panner whose
//! blend weight is clamped above and below. A randomized batch then pins every
//! field element-for-element. All inputs are integers or simple decimals, so the
//! fixtures stay pure and need no `bevy_math` and no transcendental math; the
//! rotation reads a caller-supplied `(cos, sin)` pair exactly as the reference
//! does.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each element is a fixed, non-reorderable sequence of adds, multiplies and one
//! `floor`, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every coordinate; the clamped
//! blend weight matches under the same bound.
//!
//! # Conditioning
//!
//! The wrap inputs are kept clear of the half-open snap threshold (a folded
//! value within `EPS` of `1.0`), so `CPU` and `GPU` take the same branch. In
//! practice the fold `x - floor(x)` is a single subtraction after an exact
//! `floor`, so it is identical on both devices regardless, but the fixtures stay
//! in the interior to make that obvious.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::uv_animation`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::uv_animation::{golden, GpuUvAnimation, UvAnimationQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous coordinates.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous coordinates.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two `UV` pairs.
fn approx2(a: [f32; 2], b: [f32; 2]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1])
}

/// A base query with an identity-ish transform; per-fixture helpers override the
/// fields they exercise so a single query still drives every twinned function.
fn base() -> UvAnimationQuery {
    UvAnimationQuery {
        uv: [0.2, 0.3],
        offset: [0.5, -0.1],
        tiling: [1.0, 1.0],
        pivot: [0.0, 0.0],
        cos_rot: 1.0,
        sin_rot: 0.0,
        velocity: [0.25, -0.5],
        time: 1.0,
        scroll: [0.1, 0.2],
        offset_a: [0.5, 0.0],
        offset_b: [0.0, 0.5],
        blend: 0.3,
        wrap_input: [1.25, -0.25],
    }
}

/// Asserts every twinned answer for one query matches the `CPU` golden.
fn pin(
    idx: usize,
    q: &UvAnimationQuery,
    got: &prism_volumetric_gpu::uv_animation::UvAnimationResult,
) {
    let want = golden(q);
    assert!(
        approx2(got.scrolled, want.scrolled),
        "query {idx} scrolled: gpu {:?} vs cpu {:?}",
        got.scrolled,
        want.scrolled
    );
    assert!(
        approx2(got.tiled, want.tiled),
        "query {idx} tiled: gpu {:?} vs cpu {:?}",
        got.tiled,
        want.tiled
    );
    assert!(
        approx2(got.rotated, want.rotated),
        "query {idx} rotated: gpu {:?} vs cpu {:?}",
        got.rotated,
        want.rotated
    );
    assert!(
        approx2(got.scroll_offset, want.scroll_offset),
        "query {idx} scroll_offset: gpu {:?} vs cpu {:?}",
        got.scroll_offset,
        want.scroll_offset
    );
    assert!(
        approx2(got.wrapped, want.wrapped),
        "query {idx} wrapped: gpu {:?} vs cpu {:?}",
        got.wrapped,
        want.wrapped
    );
    assert!(
        approx2(got.dual_a, want.dual_a),
        "query {idx} dual_a: gpu {:?} vs cpu {:?}",
        got.dual_a,
        want.dual_a
    );
    assert!(
        approx2(got.dual_b, want.dual_b),
        "query {idx} dual_b: gpu {:?} vs cpu {:?}",
        got.dual_b,
        want.dual_b
    );
    assert!(
        approx(got.dual_blend, want.dual_blend),
        "query {idx} dual_blend: gpu {} vs cpu {}",
        got.dual_blend,
        want.dual_blend
    );
    assert!(
        approx2(got.applied, want.applied),
        "query {idx} applied: gpu {:?} vs cpu {:?}",
        got.applied,
        want.applied
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the golden.
fn check(ctx: &GpuContext, gpu: &GpuUvAnimation, queries: &[UvAnimationQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, q, result);
    }
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

/// A pseudo-random `UV` pair with each lane in `[-span, span)`.
fn rand_pair(state: &mut u64, span: f32) -> [f32; 2] {
    [signed(state, span), signed(state, span)]
}

/// Builds a randomized but well-conditioned query. Scale factors stay positive
/// and bounded, and the wrap input is a moderate magnitude so the fold stays
/// clear of the half-open snap threshold on both devices.
fn rand_query(state: &mut u64) -> UvAnimationQuery {
    UvAnimationQuery {
        uv: rand_pair(state, 2.0),
        offset: rand_pair(state, 1.0),
        tiling: [lcg(state) * 3.5 + 0.25, lcg(state) * 3.5 + 0.25],
        pivot: rand_pair(state, 1.0),
        cos_rot: signed(state, 1.0),
        sin_rot: signed(state, 1.0),
        velocity: rand_pair(state, 1.5),
        time: lcg(state) * 4.0,
        scroll: rand_pair(state, 1.0),
        offset_a: rand_pair(state, 1.0),
        offset_b: rand_pair(state, 1.0),
        blend: signed(state, 1.5),
        wrap_input: rand_pair(state, 3.0),
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUvAnimation::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn scroll_translates_componentwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUvAnimation::new(&ctx);
    // uv_scroll adds the offset lane for lane: (0.2, 0.3) + (0.5, -0.1).
    let q = UvAnimationQuery {
        uv: [0.2, 0.3],
        offset: [0.5, -0.1],
        ..base()
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn tile_scales_about_and_fixes_pivot() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUvAnimation::new(&ctx);
    // 2x tiling about (0.5, 0.5) doubles a unit offset; the pivot is a fixed
    // point of the tile stage.
    let q = UvAnimationQuery {
        uv: [1.0, 0.5],
        tiling: [2.0, 2.0],
        pivot: [0.5, 0.5],
        ..base()
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn rotate_ninety_degrees_maps_x_to_y() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUvAnimation::new(&ctx);
    // cos = 0, sin = 1 is a +90 degree rotation about the origin: (1, 0) -> (0, 1).
    let q = UvAnimationQuery {
        uv: [1.0, 0.0],
        pivot: [0.0, 0.0],
        cos_rot: 0.0,
        sin_rot: 1.0,
        ..base()
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn identity_rotation_is_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUvAnimation::new(&ctx);
    // cos = 1, sin = 0 leaves the coordinate untouched regardless of pivot.
    let q = UvAnimationQuery {
        uv: [0.42, -0.17],
        pivot: [0.5, 0.5],
        cos_rot: 1.0,
        sin_rot: 0.0,
        ..base()
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn scroll_offset_is_linear_in_time() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUvAnimation::new(&ctx);
    // scroll_offset is velocity * time, linear in time.
    let q = UvAnimationQuery {
        velocity: [0.25, -0.5],
        time: 2.0,
        ..base()
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn apply_follows_pivot_scale_rotate_scroll_order() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUvAnimation::new(&ctx);
    // pivot at origin, 3x tiling, +90 degree rotation, then scroll:
    // (1, 0) -> scale (3, 0) -> rotate (0, 3) -> +scroll (0.1, 3.2).
    let q = UvAnimationQuery {
        uv: [1.0, 0.0],
        tiling: [3.0, 3.0],
        pivot: [0.0, 0.0],
        cos_rot: 0.0,
        sin_rot: 1.0,
        scroll: [0.1, 0.2],
        ..base()
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn apply_fixes_pivot_up_to_scroll() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUvAnimation::new(&ctx);
    // A uv equal to the pivot returns pivot + scroll under any tiling / rotation.
    let q = UvAnimationQuery {
        uv: [0.5, 0.5],
        tiling: [7.0, 0.5],
        pivot: [0.5, 0.5],
        cos_rot: 0.0,
        sin_rot: 1.0,
        scroll: [0.05, -0.05],
        ..base()
    };
    check(&ctx, &gpu, &[q]);
}

#[test]
fn wrap_folds_positive_and_negative_values() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUvAnimation::new(&ctx);
    // (1.25, -0.25) folds to (0.25, 0.75); exact integers fold to 0.
    let interior = UvAnimationQuery {
        wrap_input: [1.25, -0.25],
        ..base()
    };
    let integral = UvAnimationQuery {
        wrap_input: [1.0, 2.0],
        ..base()
    };
    let drifted = UvAnimationQuery {
        wrap_input: [3.5, -1.25],
        ..base()
    };
    check(&ctx, &gpu, &[interior, integral, drifted]);
}

#[test]
fn dual_panner_clamps_blend_weight() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUvAnimation::new(&ctx);
    // Interior, over-range and under-range blend weights: clamp pins to [0, 1].
    let interior = UvAnimationQuery {
        offset_a: [0.5, 0.0],
        offset_b: [0.0, 0.5],
        blend: 0.3,
        ..base()
    };
    let over = UvAnimationQuery {
        blend: 1.5,
        ..base()
    };
    let under = UvAnimationQuery {
        blend: -0.5,
        ..base()
    };
    check(&ctx, &gpu, &[interior, over, under]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUvAnimation::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random queries,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        UvAnimationQuery {
            uv: [1.0, 0.0],
            tiling: [3.0, 3.0],
            pivot: [0.0, 0.0],
            cos_rot: 0.0,
            sin_rot: 1.0,
            scroll: [0.1, 0.2],
            ..base()
        },
        base(),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_elements_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUvAnimation::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins every field across many
    // random coordinate warps.
    let queries: Vec<UvAnimationQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
