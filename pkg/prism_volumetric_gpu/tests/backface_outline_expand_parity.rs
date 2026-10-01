//! Real-device parity for the backface outline-expansion twin:
//! [`GpuBackfaceOutlineExpand`](prism_volumetric_gpu::backface_outline_expand::GpuBackfaceOutlineExpand)
//! must reproduce the `CPU` golden
//! [`backface_outline_expand`](prism_render_architecture::particle::backface_outline_expand)
//! across world-space strokes, screen-space strokes resolved at a view depth,
//! clamped near-camera strokes, the two degeneracy guards (a zero-length normal
//! that falls through to the geometric fallback, and both normals degenerate so
//! the vertex is left untouched), a degenerate `focal_scale` that collapses the
//! screen distance to zero, and a randomized batch of well-conditioned vertices
//! compared value-for-value.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each vertex is a fixed, non-reorderable sequence of multiplies, adds and
//! divides (plus one `sqrt` inside the guarded normalize), so `CPU` and `GPU`
//! evaluate the same closed form in the same order. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The
//! comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on every
//! `f32` field.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from the degeneracy cracks: the
//! fallback and both-degenerate cases use exactly zero normals (squared length
//! exactly zero on both devices), the normal-present cases use normals whose
//! squared length is far above the normalize threshold, and the clamp fixtures
//! land clearly on one side of the `max_distance` bound. This keeps `CPU` and
//! `GPU` on the same side of every branch regardless of a few units in the last
//! place of slack.
//!
//! Provenance: twinned from this repository's
//! [`backface_outline_expand`](prism_render_architecture::particle::backface_outline_expand);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::backface_outline_expand::{
    displace_with_fallback, outward_offset, OutlineExpandParams,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::backface_outline_expand::{
    GpuBackfaceOutlineExpand, OutlineExpandResult, OutlineExpandVertex,
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
fn close_vec(label: &str, idx: usize, got: Vec3, want: Vec3) {
    assert!(
        close(got.x, want.x) && close(got.y, want.y) && close(got.z, want.z),
        "vertex {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got.x,
        got.y,
        got.z,
        want.x,
        want.y,
        want.z
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
fn rand_vec(state: &mut u64, span: f32) -> Vec3 {
    Vec3::new(
        signed(state, span),
        signed(state, span),
        signed(state, span),
    )
}

/// Pins one `GPU` result against the `CPU` golden for `vertex` under `params`
/// at `view_depth`: the resolved distance, the normal displacement, the
/// fallback displacement and the signed outward offset must all agree within
/// bound.
fn pin(
    idx: usize,
    params: &OutlineExpandParams,
    view_depth: f32,
    vertex: &OutlineExpandVertex,
    got: &OutlineExpandResult,
) {
    let want_distance = params.expand_distance(view_depth);
    let want_displaced = params.displace_vertex(vertex.position, vertex.normal, view_depth);
    let want_fallback = displace_with_fallback(
        vertex.position,
        vertex.normal,
        vertex.fallback,
        want_distance,
    );
    let want_offset = outward_offset(vertex.position, want_displaced, vertex.normal);

    assert!(
        close(got.distance, want_distance),
        "vertex {idx} distance: gpu {} vs cpu {}",
        got.distance,
        want_distance
    );
    close_vec("displaced", idx, got.displaced, want_displaced);
    close_vec(
        "fallback_displaced",
        idx,
        got.fallback_displaced,
        want_fallback,
    );
    assert!(
        close(got.outward_offset, want_offset),
        "vertex {idx} outward_offset: gpu {} vs cpu {}",
        got.outward_offset,
        want_offset
    );
}

/// Dispatches `vertices` on the `GPU` and pins every result against the
/// reference.
fn check(
    ctx: &GpuContext,
    gpu: &GpuBackfaceOutlineExpand,
    params: &OutlineExpandParams,
    view_depth: f32,
    vertices: &[OutlineExpandVertex],
) {
    let got = gpu.eval(ctx, params, view_depth, vertices);
    assert_eq!(
        got.len(),
        vertices.len(),
        "result count must match the input count"
    );
    for (idx, (vertex, result)) in vertices.iter().zip(got.iter()).enumerate() {
        pin(idx, params, view_depth, vertex, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBackfaceOutlineExpand::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &OutlineExpandParams::world(0.2), 1.0, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn world_space_stroke_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBackfaceOutlineExpand::new(&ctx);
    // A fixed world-space thickness is depth-independent; view_depth is ignored.
    let params = OutlineExpandParams::world(0.25);
    let vertices = [
        OutlineExpandVertex::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ),
        // Non-unit normal must be normalized before displacement.
        OutlineExpandVertex::new(
            Vec3::new(0.0, 2.0, -1.0),
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ),
    ];
    check(&ctx, &gpu, &params, 7.5, &vertices);
}

#[test]
fn screen_space_stroke_scales_with_depth() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBackfaceOutlineExpand::new(&ctx);
    // d = pixel_width * view_depth / focal_scale = 3 * 12 / 600 = 0.06.
    let params = OutlineExpandParams::screen(3.0, 600.0);
    let vertices = [OutlineExpandVertex::new(
        Vec3::new(0.5, -0.5, 2.0),
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(1.0, 0.0, 0.0),
    )];
    check(&ctx, &gpu, &params, 12.0, &vertices);
}

#[test]
fn screen_space_stroke_clamps_to_max_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBackfaceOutlineExpand::new(&ctx);
    // Raw = 4 * 50 / 100 = 2.0 at a near depth, clamped down to 0.3.
    let params = OutlineExpandParams::screen(4.0, 100.0).with_max_distance(0.3);
    let vertices = [OutlineExpandVertex::new(
        Vec3::new(-1.0, 1.0, 0.5),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    )];
    check(&ctx, &gpu, &params, 50.0, &vertices);
}

#[test]
fn degenerate_focal_collapses_distance_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBackfaceOutlineExpand::new(&ctx);
    // A focal length at the guard magnitude collapses the resolved distance to
    // zero rather than dividing by (near) zero, leaving the vertex unmoved.
    let params = OutlineExpandParams::screen(5.0, 0.0);
    let vertices = [OutlineExpandVertex::new(
        Vec3::new(2.0, 3.0, 4.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
    )];
    check(&ctx, &gpu, &params, 9.0, &vertices);
}

#[test]
fn degenerate_normal_uses_fallback_branch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBackfaceOutlineExpand::new(&ctx);
    // A zero shading normal leaves displace_along_normal untouched, while
    // displace_with_fallback moves along the geometric fallback normal.
    let params = OutlineExpandParams::world(0.4);
    let vertices = [OutlineExpandVertex::new(
        Vec3::new(1.0, -2.0, 3.0),
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 2.0),
    )];
    check(&ctx, &gpu, &params, 1.0, &vertices);
}

#[test]
fn both_normals_degenerate_leaves_vertex_unchanged() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBackfaceOutlineExpand::new(&ctx);
    // Both normals zero: neither displacement moves the vertex and the offset is
    // zero, with no NaN emitted.
    let params = OutlineExpandParams::world(0.5);
    let vertices = [OutlineExpandVertex::new(
        Vec3::new(-4.0, 5.0, -6.0),
        Vec3::ZERO,
        Vec3::ZERO,
    )];
    let got = gpu.eval(&ctx, &params, 1.0, &vertices);
    assert_eq!(got.len(), 1);
    close_vec("displaced", 0, got[0].displaced, vertices[0].position);
    close_vec(
        "fallback_displaced",
        0,
        got[0].fallback_displaced,
        vertices[0].position,
    );
    assert!(close(got[0].outward_offset, 0.0));
    assert!(
        got[0].displaced.x.is_finite()
            && got[0].displaced.y.is_finite()
            && got[0].displaced.z.is_finite(),
        "no NaN is emitted for a fully degenerate vertex"
    );
    check(&ctx, &gpu, &params, 1.0, &vertices);
}

#[test]
fn randomized_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBackfaceOutlineExpand::new(&ctx);
    let params = OutlineExpandParams::screen(2.5, 480.0).with_max_distance(1.0);
    let view_depth = 15.0_f32;

    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut vertices = Vec::new();
    for _ in 0..256 {
        // Normals are drawn with a floor on magnitude so every random fixture
        // sits far above the normalize threshold, clear of the degeneracy crack.
        let mut normal = rand_vec(&mut state, 3.0);
        if normal.length_squared() < 1.0 {
            normal = Vec3::new(normal.x + 2.0, normal.y + 2.0, normal.z + 2.0);
        }
        let mut fallback = rand_vec(&mut state, 3.0);
        if fallback.length_squared() < 1.0 {
            fallback = Vec3::new(fallback.x - 2.0, fallback.y - 2.0, fallback.z - 2.0);
        }
        vertices.push(OutlineExpandVertex::new(
            rand_vec(&mut state, 10.0),
            normal,
            fallback,
        ));
    }
    check(&ctx, &gpu, &params, view_depth, &vertices);
}
