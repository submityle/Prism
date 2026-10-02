//! Real-device parity for the parallax-occlusion `UV`-offset twin:
//! [`GpuParallaxOffset`](prism_volumetric_gpu::parallax_offset::GpuParallaxOffset)
//! must reproduce the `CPU` golden
//! [`parallax_offset`](prism_render_architecture::particle::parallax_offset)
//! across the unit-interval clamp, the multiply-only `smoothstep01`, the plain
//! `lerp`, the tangent-space `normalize3`, the full-depth offset vector
//! `full_offset`, the view-dependent `layer_count`, the one-step `secant_refine`,
//! one steep-parallax march step, the steep-hit bracket resolution, one
//! self-shadow march step and the final self-shadow resolve, plus a randomized
//! batch compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every routine threads through multiplies, adds and one guarded divide /
//! `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. Continuous channels are
//! compared with `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The normalization
//! validity flag and the steep-step penetration flag are integer / `bool`
//! codes, so they are compared for exact equality.
//!
//! # Conditioning
//!
//! Fixtures are drawn clear of every branch knee: `normalize3` inputs keep the
//! squared length well above `CMP_EPS` squared; `full_offset` view vectors keep
//! the `z` component well above the division floor; `layer_count` cosines stay
//! inside `(0, 1)` so the `clamp01` knees never bite; secant brackets keep
//! `before >= 0`, `after <= 0` with a wide `before - after`; the steep-step gap
//! between ray and surface depth is held above `0.1` so a fused multiply-add
//! cannot flip the penetration `bool`; and the self-shadow resolve ratio stays
//! in the `smoothstep01` interior. All randomness comes from a host-side
//! integer generator, so no transcendental appears in a fixture.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::parallax_offset`；no
//! third-party engine source or derived code.

use prism_render_architecture::particle::parallax_offset::{
    secant_refine, ParallaxConfig, SteepHit, CMP_EPS,
};
use prism_volumetric_gpu::parallax_offset::{
    GpuParallaxOffset, ParallaxOffsetQuery, ParallaxOffsetResult,
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

/// Clamps a scalar to the closed unit interval, mirroring the reference private
/// `clamp01`.
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// The multiply-only `smoothstep01` shaper `t^2 (3 - 2 t)` on a clamped `t`,
/// mirroring the reference private `smoothstep01`.
fn smoothstep01(t: f32) -> f32 {
    let c = clamp01(t);
    c * c * (3.0 - 2.0 * c)
}

/// Plain linear interpolation `a + (b - a) * t`, mirroring the reference `lerp`.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Normalizes a tangent-space vector, returning `None` for a degenerate
/// (near-zero) input, mirroring the reference private `normalize3` (squared
/// length guarded against `CMP_EPS` squared, one `sqrt`).
fn normalize3(v: [f32; 3]) -> Option<[f32; 3]> {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq < CMP_EPS * CMP_EPS {
        return None;
    }
    let inv = 1.0 / len_sq.sqrt();
    Some([v[0] * inv, v[1] * inv, v[2] * inv])
}

/// The full-depth tangent-space offset vector `height_scale / max(z, eps)`,
/// mirroring the reference private `full_offset`.
fn full_offset(view_unit: [f32; 3], height_scale: f32) -> [f32; 2] {
    let denom = view_unit[2].max(CMP_EPS);
    let k = height_scale / denom;
    [view_unit[0] * k, view_unit[1] * k]
}

/// One steep-parallax march step, mirroring the body of the reference
/// `steep_parallax` loop: slide the `UV` by `delta_uv`, advance the ray depth by
/// `layer_step`, re-evaluate the surface depth `1 - clamp01(height)` and report
/// whether the ray reached or passed below the surface.
fn steep_step_want(
    cur_uv: [f32; 2],
    delta_uv: [f32; 2],
    cur_layer_depth: f32,
    layer_step: f32,
    height_sample: f32,
) -> ([f32; 2], f32, f32, bool) {
    let next_uv = [cur_uv[0] - delta_uv[0], cur_uv[1] - delta_uv[1]];
    let next_layer_depth = cur_layer_depth + layer_step;
    let next_surface_depth = 1.0 - clamp01(height_sample);
    let penetrated = next_layer_depth >= next_surface_depth;
    (next_uv, next_layer_depth, next_surface_depth, penetrated)
}

/// One self-shadow march step, mirroring the body of the reference
/// `self_shadow` loop: slide the shadow ray by `delta_uv`, drop its depth by
/// `depth_step`, re-evaluate the surface depth and update the running maximum
/// overlap.
fn self_shadow_step_want(
    ray_uv: [f32; 2],
    delta_uv: [f32; 2],
    ray_depth: f32,
    depth_step: f32,
    height_sample: f32,
    max_overlap: f32,
) -> ([f32; 2], f32, f32, f32) {
    let new_ray_uv = [ray_uv[0] + delta_uv[0], ray_uv[1] + delta_uv[1]];
    let new_ray_depth = ray_depth - depth_step;
    let surface_depth = 1.0 - clamp01(height_sample);
    let overlap = new_ray_depth - surface_depth;
    let new_max_overlap = max_overlap.max(overlap);
    (new_ray_uv, new_ray_depth, surface_depth, new_max_overlap)
}

/// The final self-shadow resolve, mirroring the reference `self_shadow` tail:
/// a non-positive `max_overlap` is fully lit, otherwise the softness-scaled
/// `smoothstep01` drives the shadow and the lit fraction is `clamp01(1 -
/// shadow)`.
fn self_shadow_resolve_want(max_overlap: f32, softness: f32) -> f32 {
    if max_overlap <= 0.0 {
        return 1.0;
    }
    let shadow = if softness < CMP_EPS {
        1.0
    } else {
        smoothstep01(max_overlap / softness)
    };
    clamp01(1.0 - shadow)
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

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// A pseudo-random `UV` with each lane in `[-1, 1)`.
fn rand_uv(state: &mut u64) -> [f32; 2] {
    [range(state, -1.0, 1.0), range(state, -1.0, 1.0)]
}

/// A pseudo-random per-layer `UV` slide with each lane in `[-0.1, 0.1)`.
fn rand_delta(state: &mut u64) -> [f32; 2] {
    [range(state, -0.1, 0.1), range(state, -0.1, 0.1)]
}

/// A non-degenerate vector for `normalize3`, rejection-sampled so the squared
/// length stays well above `CMP_EPS` squared.
fn rand_vec3(state: &mut u64) -> [f32; 3] {
    loop {
        let v = [
            range(state, -1.0, 1.0),
            range(state, -1.0, 1.0),
            range(state, -1.0, 1.0),
        ];
        let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if len_sq >= 0.25 {
            return v;
        }
    }
}

/// A normalized view direction whose `z` lane stays well above the division
/// floor, so `full_offset`'s `max(z, eps)` never reaches the guard.
fn rand_view_unit(state: &mut u64) -> [f32; 3] {
    loop {
        let x = range(state, -1.0, 1.0);
        let y = range(state, -1.0, 1.0);
        let z = range(state, 0.3, 1.0);
        let len_sq = x * x + y * y + z * z;
        let inv = 1.0 / len_sq.sqrt();
        let v = [x * inv, y * inv, z * inv];
        if v[2] >= 0.2 {
            return v;
        }
    }
}

/// A `Clamp01` query with the scalar spanning both clamp knees (the clamp is
/// identical on both devices, so the knees are parity-safe).
fn make_clamp01(state: &mut u64) -> ParallaxOffsetQuery {
    ParallaxOffsetQuery::Clamp01 {
        x: range(state, -0.5, 1.5),
    }
}

/// A `Smoothstep01` query spanning both clamp knees.
fn make_smoothstep01(state: &mut u64) -> ParallaxOffsetQuery {
    ParallaxOffsetQuery::Smoothstep01 {
        t: range(state, -0.2, 1.2),
    }
}

/// A `Lerp` query with endpoints in `[-1, 1)` and parameter in `[0, 1)`.
fn make_lerp(state: &mut u64) -> ParallaxOffsetQuery {
    ParallaxOffsetQuery::Lerp {
        a: range(state, -1.0, 1.0),
        b: range(state, -1.0, 1.0),
        t: range(state, 0.0, 1.0),
    }
}

/// A `Normalize3` query over a non-degenerate vector.
fn make_normalize3(state: &mut u64) -> ParallaxOffsetQuery {
    ParallaxOffsetQuery::Normalize3 {
        v: rand_vec3(state),
    }
}

/// A `FullOffset` query over a seam-safe view direction and a positive depth.
fn make_full_offset(state: &mut u64) -> ParallaxOffsetQuery {
    ParallaxOffsetQuery::FullOffset {
        view_unit: rand_view_unit(state),
        height_scale: range(state, 0.05, 0.5),
    }
}

/// A `LayerCount` query with an interior cosine and ordered layer bounds.
fn make_layer_count(state: &mut u64) -> ParallaxOffsetQuery {
    let min_layers = 4 + (lcg(state) * 12.0) as u32;
    let max_layers = min_layers + 8 + (lcg(state) * 24.0) as u32;
    ParallaxOffsetQuery::LayerCount {
        cos_view: range(state, 0.05, 0.95),
        min_layers,
        max_layers,
    }
}

/// A `SecantRefine` query whose bracket keeps `before >= 0`, `after <= 0` and a
/// wide `before - after`, away from the degenerate-bracket guard.
fn make_secant_refine(state: &mut u64) -> ParallaxOffsetQuery {
    ParallaxOffsetQuery::SecantRefine {
        prev_uv: rand_uv(state),
        hit_uv: rand_uv(state),
        before: range(state, 0.05, 1.0),
        after: range(state, -1.0, -0.05),
    }
}

/// A `SteepStep` query whose ray-minus-surface gap stays above `0.1`, so a
/// fused multiply-add cannot flip the penetration `bool`.
fn make_steep_step(state: &mut u64) -> ParallaxOffsetQuery {
    loop {
        let cur_layer_depth = range(state, 0.0, 0.8);
        let layer_step = range(state, 0.02, 0.2);
        let height_sample = range(state, 0.1, 0.9);
        let next_layer_depth = cur_layer_depth + layer_step;
        let next_surface_depth = 1.0 - clamp01(height_sample);
        if (next_layer_depth - next_surface_depth).abs() >= 0.1 {
            return ParallaxOffsetQuery::SteepStep {
                cur_uv: rand_uv(state),
                delta_uv: rand_delta(state),
                cur_layer_depth,
                layer_step,
                height_sample,
            };
        }
    }
}

/// A `SteepResolve` query built from an outside sample (`before >= 0.05`) and an
/// inside sample (`after <= -0.05`) so the bracket resolves well clear of the
/// degenerate guard.
fn make_steep_resolve(state: &mut u64) -> ParallaxOffsetQuery {
    let prev_layer_depth = range(state, 0.1, 0.4);
    let before = range(state, 0.05, 0.5);
    let prev_surface_depth = prev_layer_depth + before;
    let hit_layer_depth = range(state, 0.5, 0.9);
    let after = range(state, -0.5, -0.05);
    let hit_surface_depth = hit_layer_depth + after;
    ParallaxOffsetQuery::SteepResolve {
        prev_uv: rand_uv(state),
        hit_uv: rand_uv(state),
        prev_layer_depth,
        prev_surface_depth,
        hit_layer_depth,
        hit_surface_depth,
    }
}

/// A `SelfShadowStep` query with no branch knees in the step arithmetic.
fn make_self_shadow_step(state: &mut u64) -> ParallaxOffsetQuery {
    ParallaxOffsetQuery::SelfShadowStep {
        ray_uv: rand_uv(state),
        delta_uv: rand_delta(state),
        ray_depth: range(state, 0.2, 1.0),
        depth_step: range(state, 0.02, 0.2),
        height_sample: range(state, 0.1, 0.9),
        max_overlap: range(state, 0.0, 0.5),
    }
}

/// A `SelfShadowResolve` query drawn either in the fully-lit branch
/// (`max_overlap <= 0`) or in the `smoothstep01` interior, both clear of the
/// resolve knees.
fn make_self_shadow_resolve(state: &mut u64) -> ParallaxOffsetQuery {
    let softness = range(state, 0.1, 1.0);
    if lcg(state) < 0.5 {
        ParallaxOffsetQuery::SelfShadowResolve {
            max_overlap: range(state, -1.0, -0.1),
            softness,
        }
    } else {
        let ratio = range(state, 0.1, 0.9);
        ParallaxOffsetQuery::SelfShadowResolve {
            max_overlap: ratio * softness,
            softness,
        }
    }
}

/// Pins a vector output against the reference channel-for-channel.
fn close_lanes(idx: usize, label: &str, got: &[f32], want: &[f32]) {
    for (lane, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(*g, *w),
            "query {idx} {label} lane {lane}: gpu {g} vs cpu {w}"
        );
    }
}

/// Pins a scalar output against the reference.
fn close_scalar(idx: usize, label: &str, got: f32, want: f32) {
    assert!(
        close(got, want),
        "query {idx} {label}: gpu {got} vs cpu {want}"
    );
}

/// Pins one `GPU` result against the `CPU` golden for `query`, matching the
/// result variant to the query variant and comparing channel-for-channel. The
/// march-step variants carry the host-sampled height, so their golden is the
/// reference loop body replayed on that same single step.
#[expect(
    clippy::too_many_lines,
    reason = "one match arm per parallax routine keeps the full dispatch table in one readable place"
)]
fn pin(idx: usize, query: &ParallaxOffsetQuery, got: &ParallaxOffsetResult) {
    match (query, got) {
        (ParallaxOffsetQuery::Clamp01 { x }, ParallaxOffsetResult::Clamp01(g)) => {
            close_scalar(idx, "clamp01", *g, clamp01(*x));
        }
        (ParallaxOffsetQuery::Smoothstep01 { t }, ParallaxOffsetResult::Smoothstep01(g)) => {
            close_scalar(idx, "smoothstep01", *g, smoothstep01(*t));
        }
        (ParallaxOffsetQuery::Lerp { a, b, t }, ParallaxOffsetResult::Lerp(g)) => {
            close_scalar(idx, "lerp", *g, lerp(*a, *b, *t));
        }
        (ParallaxOffsetQuery::Normalize3 { v }, ParallaxOffsetResult::Normalize3(g)) => {
            match (g, normalize3(*v)) {
                (None, None) => {}
                (Some(gv), Some(cv)) => close_lanes(idx, "normalize3", gv, &cv),
                _ => panic!("query {idx}: normalize3 presence differs (gpu {g:?})"),
            }
        }
        (
            ParallaxOffsetQuery::FullOffset {
                view_unit,
                height_scale,
            },
            ParallaxOffsetResult::FullOffset(g),
        ) => {
            close_lanes(
                idx,
                "full_offset",
                g,
                &full_offset(*view_unit, *height_scale),
            );
        }
        (
            ParallaxOffsetQuery::LayerCount {
                cos_view,
                min_layers,
                max_layers,
            },
            ParallaxOffsetResult::LayerCount(g),
        ) => {
            let cfg = ParallaxConfig::new(1.0, 1.0, *min_layers as u16, *max_layers as u16);
            close_scalar(idx, "layer_count", *g, cfg.layer_count(*cos_view));
        }
        (
            ParallaxOffsetQuery::SecantRefine {
                prev_uv,
                hit_uv,
                before,
                after,
            },
            ParallaxOffsetResult::SecantRefine(g),
        ) => {
            close_lanes(
                idx,
                "secant_refine",
                g,
                &secant_refine(*prev_uv, *hit_uv, *before, *after),
            );
        }
        (
            ParallaxOffsetQuery::SteepStep {
                cur_uv,
                delta_uv,
                cur_layer_depth,
                layer_step,
                height_sample,
            },
            ParallaxOffsetResult::SteepStep {
                next_uv,
                next_layer_depth,
                next_surface_depth,
                penetrated,
            },
        ) => {
            let (w_uv, w_layer, w_surface, w_pen) = steep_step_want(
                *cur_uv,
                *delta_uv,
                *cur_layer_depth,
                *layer_step,
                *height_sample,
            );
            close_lanes(idx, "steep_step uv", next_uv, &w_uv);
            close_scalar(idx, "steep_step layer_depth", *next_layer_depth, w_layer);
            close_scalar(
                idx,
                "steep_step surface_depth",
                *next_surface_depth,
                w_surface,
            );
            assert_eq!(*penetrated, w_pen, "query {idx} steep_step penetrated");
        }
        (
            ParallaxOffsetQuery::SteepResolve {
                prev_uv,
                hit_uv,
                prev_layer_depth,
                prev_surface_depth,
                hit_layer_depth,
                hit_surface_depth,
            },
            ParallaxOffsetResult::SteepResolve {
                before,
                after,
                refined_uv,
            },
        ) => {
            let hit = SteepHit {
                prev_uv: *prev_uv,
                hit_uv: *hit_uv,
                prev_layer_depth: *prev_layer_depth,
                prev_surface_depth: *prev_surface_depth,
                hit_layer_depth: *hit_layer_depth,
                hit_surface_depth: *hit_surface_depth,
                penetrated: true,
            };
            close_scalar(idx, "steep_resolve before", *before, hit.before());
            close_scalar(idx, "steep_resolve after", *after, hit.after());
            close_lanes(idx, "steep_resolve uv", refined_uv, &hit.refined_uv());
        }
        (
            ParallaxOffsetQuery::SelfShadowStep {
                ray_uv,
                delta_uv,
                ray_depth,
                depth_step,
                height_sample,
                max_overlap,
            },
            ParallaxOffsetResult::SelfShadowStep {
                ray_uv: g_uv,
                ray_depth: g_depth,
                surface_depth: g_surface,
                max_overlap: g_overlap,
            },
        ) => {
            let (w_uv, w_depth, w_surface, w_overlap) = self_shadow_step_want(
                *ray_uv,
                *delta_uv,
                *ray_depth,
                *depth_step,
                *height_sample,
                *max_overlap,
            );
            close_lanes(idx, "self_shadow_step uv", g_uv, &w_uv);
            close_scalar(idx, "self_shadow_step ray_depth", *g_depth, w_depth);
            close_scalar(idx, "self_shadow_step surface_depth", *g_surface, w_surface);
            close_scalar(idx, "self_shadow_step max_overlap", *g_overlap, w_overlap);
        }
        (
            ParallaxOffsetQuery::SelfShadowResolve {
                max_overlap,
                softness,
            },
            ParallaxOffsetResult::SelfShadowResolve(g),
        ) => {
            close_scalar(
                idx,
                "self_shadow_resolve",
                *g,
                self_shadow_resolve_want(*max_overlap, *softness),
            );
        }
        _ => panic!("query {idx}: result variant does not match the query variant"),
    }
}

/// Draws a random query of a random routine with seam-safe fixtures.
fn rand_query(state: &mut u64) -> ParallaxOffsetQuery {
    match (lcg(state) * 11.0) as u32 {
        0 => make_clamp01(state),
        1 => make_smoothstep01(state),
        2 => make_lerp(state),
        3 => make_normalize3(state),
        4 => make_full_offset(state),
        5 => make_layer_count(state),
        6 => make_secant_refine(state),
        7 => make_steep_step(state),
        8 => make_steep_resolve(state),
        9 => make_self_shadow_step(state),
        _ => make_self_shadow_resolve(state),
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuParallaxOffset, queries: &[ParallaxOffsetQuery]) {
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
    let gpu = GpuParallaxOffset::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn clamp01_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParallaxOffset::new(&ctx);
    let mut state = 0x1111_2222_3333_4444_u64;
    let queries: Vec<ParallaxOffsetQuery> = (0..64).map(|_| make_clamp01(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn smoothstep01_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParallaxOffset::new(&ctx);
    let mut state = 0x5555_6666_7777_8888_u64;
    let queries: Vec<ParallaxOffsetQuery> =
        (0..64).map(|_| make_smoothstep01(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn lerp_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParallaxOffset::new(&ctx);
    let mut state = 0x9999_aaaa_bbbb_cccc_u64;
    let queries: Vec<ParallaxOffsetQuery> = (0..64).map(|_| make_lerp(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn normalize3_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParallaxOffset::new(&ctx);
    let mut state = 0xdddd_eeee_ffff_0000_u64;
    let mut queries: Vec<ParallaxOffsetQuery> =
        (0..64).map(|_| make_normalize3(&mut state)).collect();
    // A deterministic exact-zero vector exercises the `None` (invalid) branch on
    // both devices.
    queries.push(ParallaxOffsetQuery::Normalize3 { v: [0.0, 0.0, 0.0] });
    check(&ctx, &gpu, &queries);
}

#[test]
fn full_offset_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParallaxOffset::new(&ctx);
    let mut state = 0x0102_0304_0506_0708_u64;
    let queries: Vec<ParallaxOffsetQuery> = (0..64).map(|_| make_full_offset(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn layer_count_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParallaxOffset::new(&ctx);
    let mut state = 0x1a2b_3c4d_5e6f_7a8b_u64;
    let queries: Vec<ParallaxOffsetQuery> = (0..64).map(|_| make_layer_count(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn secant_refine_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParallaxOffset::new(&ctx);
    let mut state = 0x2718_2818_2845_9045_u64;
    let queries: Vec<ParallaxOffsetQuery> =
        (0..64).map(|_| make_secant_refine(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn steep_step_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParallaxOffset::new(&ctx);
    let mut state = 0x3141_5926_5358_9793_u64;
    let queries: Vec<ParallaxOffsetQuery> = (0..64).map(|_| make_steep_step(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn steep_resolve_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParallaxOffset::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    let queries: Vec<ParallaxOffsetQuery> =
        (0..64).map(|_| make_steep_resolve(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn self_shadow_step_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParallaxOffset::new(&ctx);
    let mut state = 0x5157_1a2b_3c4d_5e6f_u64;
    let queries: Vec<ParallaxOffsetQuery> =
        (0..64).map(|_| make_self_shadow_step(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn self_shadow_resolve_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParallaxOffset::new(&ctx);
    let mut state = 0x5a5a_a5a5_0f0f_f0f0_u64;
    let queries: Vec<ParallaxOffsetQuery> = (0..64)
        .map(|_| make_self_shadow_resolve(&mut state))
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParallaxOffset::new(&ctx);
    let mut state = 0x2b2b_1a1a_3c3c_4d4d_u64;
    // One batch mixing deterministic fixtures with many random queries of every
    // routine, dispatched together so the per-thread indexing and the contiguous
    // storage layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        ParallaxOffsetQuery::Clamp01 { x: 1.3 },
        ParallaxOffsetQuery::Smoothstep01 { t: 0.4 },
        ParallaxOffsetQuery::Lerp {
            a: -0.5,
            b: 0.5,
            t: 0.25,
        },
        ParallaxOffsetQuery::Normalize3 { v: [0.3, 0.4, 0.5] },
        ParallaxOffsetQuery::FullOffset {
            view_unit: rand_view_unit(&mut state),
            height_scale: 0.2,
        },
        ParallaxOffsetQuery::LayerCount {
            cos_view: 0.5,
            min_layers: 8,
            max_layers: 32,
        },
        ParallaxOffsetQuery::SecantRefine {
            prev_uv: [0.1, 0.2],
            hit_uv: [0.3, 0.4],
            before: 0.3,
            after: -0.2,
        },
    ];
    for _ in 0..57 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParallaxOffset::new(&ctx);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    // A larger sweep (several workgroups' worth) pins every routine across many
    // random fixtures.
    let queries: Vec<ParallaxOffsetQuery> = (0..256).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
