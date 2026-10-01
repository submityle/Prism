//! Real-device parity for the sprite motion-stretch twin:
//! [`GpuSpriteStretch`](prism_volumetric_gpu::sprite_stretch::GpuSpriteStretch) must
//! reproduce the `CPU` golden
//! [`sprite_stretch`](prism_render_architecture::particle::sprite_stretch),
//! field for field, across an empty batch, the zero-velocity isotropic
//! fallback, speed-driven stretch that saturates at the clamp window, the
//! lower-clamp and width-scale paths, the degenerate-axis fallbacks (zero
//! velocity, zero forward, collapsed binormal), the trail quad whose length is
//! the travel distance, and a large random batch kept well clear of every
//! decision boundary.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full dispatch-and-
//! readback on any real device such as an Apple `M`-series `GPU`. The kernel is
//! portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every field (the stretched half-extents, the aspect, the trail length and
//! all three four-corner blocks) is a fixed, non-reorderable sequence of dots,
//! `sqrt`s, clamps, scales and adds, so `CPU` and `GPU` evaluate the same
//! closed form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (relative floor `1e-6`) --
//! loose enough to admit a legal fused multiply-add contraction yet tight
//! enough to fail a genuinely wrong port (a swapped clamp, a dropped width
//! scale, a flipped axis fallback, a wrong corner layout). Every fixture is
//! kept well clear of the `EPS` zero-velocity / zero-axis branch splits, the
//! clamp-window edges and the `perpendicular_to` axis-selection ties, so a
//! `GPU`'s fused multiply-add cannot flip a branch.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sprite_stretch`；
//! 无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::sprite_stretch::{
    stretched_corners, stretched_from_prev, velocity_stretched_corners, StretchParams,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::sprite_stretch::{
    GpuSpriteStretch, SpriteStretchQuery, SpriteStretchResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the continuous fields.
const EPS_ABS: f32 = 1.0e-4;

/// Relative parity bound on the continuous fields.
const EPS_REL: f32 = 1.0e-3;

/// Floor keeping the relative-error denominator away from zero.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS_ABS || rel <= EPS_REL
}

/// Asserts two 3-vectors agree component-wise within [`close`].
fn close_vec(a: Vec3, b: Vec3, what: &str, idx: usize) {
    assert!(
        close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z),
        "{what} mismatch at query {idx}: gpu ({}, {}, {}), cpu ({}, {}, {})",
        a.x,
        a.y,
        a.z,
        b.x,
        b.y,
        b.z,
    );
}

/// Asserts the four corners of a quad agree with the reference within [`close`].
fn close_corners(got: [Vec3; 4], want: [Vec3; 4], what: &str, idx: usize) {
    for (corner, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        close_vec(g, w, &format!("{what} corner {corner}"), idx);
    }
}

/// The velocity-stretch parameters shared by most fixtures: a wide clamp window
/// so the half-length stays strictly interior (clear of a clamp-edge tie) and a
/// unit width scale.
fn params() -> StretchParams {
    StretchParams {
        stretch_scale: 1.0,
        min_length: 0.0,
        max_length: 1_000.0,
        width_scale: 1.0,
    }
}

/// Builds a full sprite-stretch query from explicit vectors and scalars.
#[expect(
    clippy::too_many_arguments,
    reason = "a single query drives every portable function, so it carries all inputs"
)]
fn q(
    center: Vec3,
    velocity: Vec3,
    binormal: Vec3,
    prev_position: Vec3,
    forward_axis: Vec3,
    base_half: f32,
    half_length: f32,
    half_width: f32,
    trail_half_width: f32,
    params: StretchParams,
) -> SpriteStretchQuery {
    SpriteStretchQuery {
        center,
        velocity,
        binormal,
        prev_position,
        forward_axis,
        base_half,
        half_length,
        half_width,
        trail_half_width,
        params,
    }
}

/// Asserts one `GPU` [`SpriteStretchResult`] matches the `CPU` golden for its
/// originating query, field for field.
fn assert_parity(got: &SpriteStretchResult, qq: &SpriteStretchQuery, idx: usize) {
    let size = qq.params.stretched_size(qq.velocity, qq.base_half);
    assert!(
        close(got.size.half_length, size.half_length),
        "size.half_length mismatch at query {idx}: gpu {}, cpu {}",
        got.size.half_length,
        size.half_length,
    );
    assert!(
        close(got.size.half_width, size.half_width),
        "size.half_width mismatch at query {idx}: gpu {}, cpu {}",
        got.size.half_width,
        size.half_width,
    );
    assert!(
        close(got.aspect, size.aspect()),
        "aspect mismatch at query {idx}: gpu {}, cpu {}",
        got.aspect,
        size.aspect(),
    );

    let want_vc = velocity_stretched_corners(
        qq.center,
        qq.velocity,
        qq.binormal,
        qq.base_half,
        qq.params,
    );
    close_corners(got.velocity_corners, want_vc, "velocity", idx);

    let want_trail = stretched_from_prev(
        qq.center,
        qq.prev_position,
        qq.binormal,
        qq.trail_half_width,
    );
    assert!(
        close(got.trail.length, want_trail.length),
        "trail.length mismatch at query {idx}: gpu {}, cpu {}",
        got.trail.length,
        want_trail.length,
    );
    close_corners(got.trail.corners, want_trail.corners, "trail", idx);

    let want_direct = stretched_corners(
        qq.center,
        qq.forward_axis,
        qq.binormal,
        qq.half_length,
        qq.half_width,
    );
    close_corners(got.corners, want_direct, "direct", idx);
}

/// Runs the `GPU` dispatch and asserts per-lane parity against the `CPU`
/// golden, returning the `GPU` results for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuSpriteStretch,
    queries: &[SpriteStretchQuery],
) -> Vec<SpriteStretchResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (g, qq)) in got.iter().zip(queries.iter()).enumerate() {
        assert_parity(g, qq, idx);
    }
    got
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

/// A random vector with each component in `[-span, span)`.
fn rand_vec(span: f32, state: &mut u64) -> Vec3 {
    Vec3::new(
        lcg(state) * 2.0 * span - span,
        lcg(state) * 2.0 * span - span,
        lcg(state) * 2.0 * span - span,
    )
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpriteStretch::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "empty batch should produce no results");
}

#[test]
fn zero_velocity_is_isotropic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpriteStretch::new(&ctx);
    // Exactly zero velocity takes the isotropic branch on both paths: both
    // half-extents equal base_half and no stretch or clamp is applied.
    let queries = [q(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(0.0, 1.0, 0.0),
        2.0,
        4.0,
        0.5,
        0.5,
        params(),
    )];
    let got = check(&ctx, &gpu, &queries);
    assert!(
        close(got[0].size.half_length, 2.0) && close(got[0].size.half_width, 2.0),
        "zero velocity leaves the quad isotropic at base_half"
    );
}

#[test]
fn speed_stretches_then_saturates_at_max() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpriteStretch::new(&ctx);
    // A tight window so a fast particle saturates at max_length; both the
    // slow (interior) and fast (saturated) fixtures are kept clear of the edge.
    let tight = StretchParams {
        stretch_scale: 1.0,
        min_length: 0.0,
        max_length: 5.0,
        width_scale: 1.0,
    };
    let bn = Vec3::new(0.0, 0.0, 1.0);
    let queries = [
        // speed 1 => raw 1.5, interior.
        q(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            bn,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            0.5,
            1.0,
            0.5,
            0.5,
            tight,
        ),
        // speed 100 => raw 100.5, saturates to 5.0 with a clear margin.
        q(
            Vec3::ZERO,
            Vec3::new(100.0, 0.0, 0.0),
            bn,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            0.5,
            1.0,
            0.5,
            0.5,
            tight,
        ),
    ];
    let got = check(&ctx, &gpu, &queries);
    assert!(
        got[1].size.half_length > got[0].size.half_length,
        "a faster particle stretches at least as far before clamping"
    );
    assert!(
        close(got[1].size.half_length, 5.0),
        "a fast particle saturates at max_length"
    );
}

#[test]
fn lower_clamp_and_width_scale_apply() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpriteStretch::new(&ctx);
    // raw = 0.05 + 1.0 * 1.0 = 1.05 is below min_length 2.0, so the half-length
    // clamps up to 2.0 (clear of the edge), and width_scale 0.25 scales the
    // half-width to base_half * 0.25.
    let p = StretchParams {
        stretch_scale: 1.0,
        min_length: 2.0,
        max_length: 100.0,
        width_scale: 0.25,
    };
    let queries = [q(
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        0.05,
        1.0,
        0.5,
        0.5,
        p,
    )];
    let got = check(&ctx, &gpu, &queries);
    assert!(
        close(got[0].size.half_length, 2.0),
        "a short raw length clamps up to min_length"
    );
    assert!(
        close(got[0].size.half_width, 0.05 * 0.25),
        "the half-width is base_half * width_scale"
    );
}

#[test]
fn trail_length_equals_travel_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpriteStretch::new(&ctx);
    // A 3-4-5 travel: the trail length must equal the distance travelled, and
    // the near/far edge midpoints sit on center / prev_position.
    let center = Vec3::new(3.0, 4.0, 0.0);
    let prev = Vec3::ZERO;
    let queries = [q(
        center,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        prev,
        Vec3::new(0.0, 1.0, 0.0),
        1.0,
        1.0,
        0.5,
        0.5,
        params(),
    )];
    let got = check(&ctx, &gpu, &queries);
    assert!(
        close(got[0].trail.length, 5.0),
        "the trail length equals the travel distance"
    );
    let near_mid = got[0].trail.corners[0]
        .add(got[0].trail.corners[1])
        .scale(0.5);
    let far_mid = got[0].trail.corners[2]
        .add(got[0].trail.corners[3])
        .scale(0.5);
    close_vec(near_mid, center, "trail near-edge midpoint", 0);
    close_vec(far_mid, prev, "trail far-edge midpoint", 0);
}

#[test]
fn degenerate_axes_stay_finite_and_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpriteStretch::new(&ctx);
    // Collapsed binormal with a clearly non-tied velocity/forward so the
    // perpendicular_to axis selection is unambiguous on both paths, plus a
    // fully zero-axis fixture exercising the +Y / +X fallbacks. `check`
    // asserts GPU/CPU parity; here we additionally require finiteness.
    let queries = [
        q(
            Vec3::new(0.5, -0.5, 0.25),
            Vec3::new(0.3, 0.5, 0.9),
            Vec3::ZERO,
            Vec3::new(-0.2, 0.1, 0.4),
            Vec3::new(0.2, 0.7, 0.1),
            1.5,
            2.0,
            0.3,
            0.4,
            params(),
        ),
        q(
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            2.0,
            1.0,
            0.5,
            0.5,
            params(),
        ),
    ];
    let got = check(&ctx, &gpu, &queries);
    for r in &got {
        for c in r
            .velocity_corners
            .iter()
            .chain(r.trail.corners.iter())
            .chain(r.corners.iter())
        {
            assert!(
                c.x.is_finite() && c.y.is_finite() && c.z.is_finite(),
                "degenerate axes must still yield finite corners"
            );
        }
    }
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpriteStretch::new(&ctx);
    let mut state = 0x_9e37_79b9_7f4a_7c15_u64;

    for _round in 0u32..8 {
        let mut queries: Vec<SpriteStretchQuery> = Vec::with_capacity(128);
        while queries.len() < 128 {
            // Velocity is either exactly zero (isotropic branch) or clearly
            // non-zero (magnitude >= ~0.5), so no fixture lands near the
            // sqrt(EPS) ~ 1e-3 branch split.
            let velocity = if lcg(&mut state) < 0.2 {
                Vec3::ZERO
            } else {
                let dir = rand_vec(1.0, &mut state);
                // Push the magnitude clearly above the branch split.
                dir.add(Vec3::new(0.5, 0.5, 0.5)).scale(2.0)
            };
            // A binormal and forward kept clearly non-zero so the primary
            // (non-fallback) axis path is taken, away from perpendicular ties.
            let binormal = rand_vec(1.0, &mut state).add(Vec3::new(1.0, 0.5, 2.0));
            let forward = rand_vec(1.0, &mut state).add(Vec3::new(2.0, 1.0, 0.5));
            let center = rand_vec(5.0, &mut state);
            // A clearly non-zero travel so the trail length is unambiguous.
            let prev = center.sub(forward.add(Vec3::new(1.0, 1.0, 1.0)));
            let base_half = lcg(&mut state) * 2.0 + 0.5;
            let half_length = lcg(&mut state) * 3.0 + 0.5;
            let half_width = lcg(&mut state) * 2.0 + 0.5;
            let trail_half_width = lcg(&mut state) * 2.0 + 0.25;
            // A wide clamp window keeps the half-length strictly interior.
            let p = StretchParams {
                stretch_scale: lcg(&mut state) * 2.0 + 0.5,
                min_length: 0.0,
                max_length: 10_000.0,
                width_scale: lcg(&mut state) * 1.5 + 0.25,
            };
            queries.push(q(
                center,
                velocity,
                binormal,
                prev,
                forward,
                base_half,
                half_length,
                half_width,
                trail_half_width,
                p,
            ));
        }
        // check asserts per-lane parity against the CPU golden.
        let got = check(&ctx, &gpu, &queries);
        // Sanity: a mixed batch should produce both isotropic and stretched
        // lanes, so the test is not trivially passing.
        let any_iso = got
            .iter()
            .any(|r| close(r.size.half_length, r.size.half_width));
        let any_stretched = got
            .iter()
            .any(|r| !close(r.size.half_length, r.size.half_width));
        assert!(
            any_iso && any_stretched,
            "expected a mix of isotropic and stretched lanes"
        );
    }
}
