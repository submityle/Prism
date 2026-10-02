//! Real-device parity for the ray-traced particle-collision and lit-sampling
//! twin:
//! [`GpuRaytrace`](prism_volumetric_gpu::raytrace::GpuRaytrace) must reproduce
//! the `CPU` golden
//! [`raytrace`](prism_render_architecture::particle::raytrace) across the mirror
//! velocity reflection, the restitution-plus-friction bounce response (with the
//! back-face normal flip and the separating-velocity no-op), the
//! hardware/quality collision-method choice and the ray-traced lit sample
//! (clamped visibility, isotropic-aware irradiance and the gated `GI` bounce),
//! plus a randomized batch of clearly-conditioned queries compared
//! element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt` per normalize, so `CPU` and `GPU` evaluate the same closed
//! form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on every continuous `f32`
//! field (the `reflected`, `bounced`, `irradiance` and `bounce` vectors and the
//! `visibility` scalar), while the discrete `method_code` and the
//! `uses_ray_tracing` flag are compared bit-exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately clear of the twin's branch ties. The bounce
//! turns on the sign of `v·n`, so each query keeps `|v·n| > 0.2` and `CPU` and
//! `GPU` share the separating-versus-approaching verdict regardless of a few
//! units in the last place. Hit and light normals are unit length so the
//! `EPS_LEN_SQ` normalize guard never trips (except in the deliberate
//! zero-normal fixtures), and the lit sample keeps `|N·L| > 0.05` off the
//! `max(N·L, 0)` hinge. The collision-method choice is driven only by integer
//! flags and the quality code, so it is exact by construction.
//!
//! Provenance: twinned from this repository's
//! [`raytrace`](prism_render_architecture::particle::raytrace); no third-party
//! engine source or derived code.

use prism_volumetric_gpu::raytrace::{
    cpu_reference, GpuRaytrace, GpuRaytraceQuery, GpuRaytraceResult, METHOD_DEPTH_BUFFER,
    METHOD_RAYTRACE, METHOD_SDF, QUALITY_HIGH, QUALITY_LOW, QUALITY_MEDIUM,
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

/// Asserts two 3-vectors agree channel-for-channel within the parity bound.
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

/// A pseudo-random 3-vector with each component in `[-span, span)`.
fn rand_vec3(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// A pseudo-random 3-vector with each component in `[0, span)`.
fn rand_pos_vec3(state: &mut u64, span: f32) -> [f32; 3] {
    [lcg(state) * span, lcg(state) * span, lcg(state) * span]
}

/// A pseudo-random boolean, true for roughly half the stream.
fn rand_bool(state: &mut u64) -> bool {
    lcg(state) < 0.5
}

/// The squared length of a 3-vector.
fn len_sq(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// The dot product of two 3-vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The unit vector in the direction of `v`, via `f32::sqrt` (no transcendental).
fn unit(v: [f32; 3]) -> [f32; 3] {
    let inv = 1.0 / len_sq(v).sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Rejection-samples a well-spread unit 3-vector, retrying until the raw draw is
/// clear of the near-zero region so the normalise is well defined.
fn rand_unit(state: &mut u64) -> [f32; 3] {
    loop {
        let raw = rand_vec3(state, 1.0);
        if len_sq(raw) >= 0.09 {
            return unit(raw);
        }
    }
}

/// A benign, clearly-conditioned base query that each fixture overrides field by
/// field via struct-update syntax. The velocity drives straight into the
/// up-facing normal (`v·n = -1`, well inside the approaching branch), the light
/// is head-on against an up-facing surface, and the coefficients sit mid-range.
fn base_query() -> GpuRaytraceQuery {
    GpuRaytraceQuery {
        velocity: [0.0, -1.0, 0.0],
        hit_normal: [0.0, 1.0, 0.0],
        light_normal: [0.0, 1.0, 0.0],
        to_light: [0.0, 1.0, 0.0],
        light_radiance: [1.0, 0.8, 0.6],
        raw_bounce: [0.2, 0.3, 0.4],
        restitution: 0.5,
        friction: 0.25,
        raw_visibility: 0.75,
        back_face: false,
        ray_tracing: false,
        sdf_volume: false,
        sample_gi: false,
        quality: QUALITY_LOW,
    }
}

/// Builds a clearly-conditioned random query by rejection sampling: the hit and
/// light normals are well-spread unit vectors, the velocity drives the bounce a
/// safe margin off the `v·n = 0` separating tie (`|v·n| > 0.2`), and the light
/// direction stays off the `max(N·L, 0)` hinge (`|N·L| > 0.05`). Restitution,
/// friction and visibility span their clamped ranges, and the capability flags
/// and quality code are drawn freely so the batch covers every method branch.
fn well_conditioned(state: &mut u64) -> GpuRaytraceQuery {
    loop {
        let hit_normal = rand_unit(state);
        let velocity = rand_vec3(state, 3.0);
        // The bounce branches on the sign of v·n; keep it clear of the tie.
        if dot(velocity, hit_normal).abs() <= 0.2 {
            continue;
        }
        let light_normal = rand_unit(state);
        let to_light = rand_unit(state);
        // The lit sample clamps N·L at zero; keep it clear of the hinge.
        if dot(light_normal, to_light).abs() <= 0.05 {
            continue;
        }
        let quality = (lcg(state) * 3.0) as u32;
        return GpuRaytraceQuery {
            velocity,
            hit_normal,
            light_normal,
            to_light,
            light_radiance: rand_pos_vec3(state, 2.0),
            raw_bounce: rand_vec3(state, 1.0),
            restitution: lcg(state),
            friction: lcg(state),
            raw_visibility: lcg(state),
            back_face: rand_bool(state),
            ray_tracing: rand_bool(state),
            sdf_volume: rand_bool(state),
            sample_gi: rand_bool(state),
            quality,
        };
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the mirror
/// reflection, the bounce response, the chosen collision method (code and
/// ray-tracing flag), the clamped visibility, the irradiance and the gated `GI`
/// bounce must all agree within bound (the discrete method code and flag
/// bit-exactly).
fn pin(idx: usize, query: &GpuRaytraceQuery, got: &GpuRaytraceResult) {
    let want = cpu_reference(query);

    close_vec("reflected", idx, got.reflected, want.reflected);
    close_vec("bounced", idx, got.bounced, want.bounced);

    assert_eq!(
        got.method_code, want.method_code,
        "query {idx} method_code: gpu {} vs cpu {}",
        got.method_code, want.method_code
    );
    assert_eq!(
        got.uses_ray_tracing, want.uses_ray_tracing,
        "query {idx} uses_ray_tracing: gpu {} vs cpu {}",
        got.uses_ray_tracing, want.uses_ray_tracing
    );

    assert!(
        close(got.visibility, want.visibility),
        "query {idx} visibility: gpu {} vs cpu {}",
        got.visibility,
        want.visibility
    );
    close_vec("irradiance", idx, got.irradiance, want.irradiance);
    close_vec("bounce", idx, got.bounce, want.bounce);
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuRaytrace, queries: &[GpuRaytraceQuery]) {
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
    let gpu = GpuRaytrace::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn head_on_bounce_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaytrace::new(&ctx);
    // Velocity straight into an up-facing surface: v·n = -1 is well inside the
    // approaching branch, so the bounce reverses and scales the normal part.
    let query = GpuRaytraceQuery {
        velocity: [0.4, -1.0, 0.0],
        restitution: 0.6,
        friction: 0.2,
        ..base_query()
    };
    check(&ctx, &gpu, &[query]);
}

#[test]
fn back_face_hit_flips_the_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaytrace::new(&ctx);
    // A back-face hit flips the geometric normal before the bounce. With the
    // normal pointing away from the velocity, the raw normal would be a
    // separating no-op; the flip makes it an approaching bounce instead.
    let query = GpuRaytraceQuery {
        velocity: [0.0, -1.0, 0.3],
        hit_normal: [0.0, -1.0, 0.0],
        back_face: true,
        restitution: 0.7,
        friction: 0.1,
        ..base_query()
    };
    check(&ctx, &gpu, &[query]);
}

#[test]
fn separating_velocity_is_a_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaytrace::new(&ctx);
    // Velocity moving away from the surface (v·n = +1, well clear of the tie):
    // the bounce leaves it unchanged while the mirror reflection still fires.
    let query = GpuRaytraceQuery {
        velocity: [0.2, 1.0, 0.0],
        hit_normal: [0.0, 1.0, 0.0],
        ..base_query()
    };
    check(&ctx, &gpu, &[query]);
}

#[test]
fn zero_normal_bounce_is_a_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaytrace::new(&ctx);
    // A (numerically) zero hit normal trips the normalize guard, so both the
    // reflection and the bounce return the velocity untouched.
    let query = GpuRaytraceQuery {
        velocity: [0.5, -1.2, 0.3],
        hit_normal: [0.0, 0.0, 0.0],
        ..base_query()
    };
    check(&ctx, &gpu, &[query]);
}

#[test]
fn restitution_and_friction_boundaries_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaytrace::new(&ctx);
    // The restitution and friction coefficients clamp to 0..=1. Both the exact
    // boundary values and deliberately out-of-range inputs fold identically on
    // CPU and GPU, so the bounce agrees for every case.
    let queries = [
        // Perfectly elastic, frictionless: full normal reversal, tangent kept.
        GpuRaytraceQuery {
            velocity: [0.6, -1.0, 0.0],
            restitution: 1.0,
            friction: 0.0,
            ..base_query()
        },
        // Perfectly inelastic, full friction: normal killed, tangent killed.
        GpuRaytraceQuery {
            velocity: [0.6, -1.0, 0.0],
            restitution: 0.0,
            friction: 1.0,
            ..base_query()
        },
        // Out-of-range coefficients clamp back into 0..=1 on both sides.
        GpuRaytraceQuery {
            velocity: [0.6, -1.0, 0.4],
            restitution: 1.8,
            friction: -0.5,
            ..base_query()
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn collision_method_matches_every_caps_and_quality_combo() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaytrace::new(&ctx);
    // The quality ladder degrades to a runnable path. These fixtures walk every
    // reachable outcome; the asserted expectations document the ladder.
    let queries = [
        // High + ray tracing -> hardware ray trace (uses_ray_tracing).
        GpuRaytraceQuery {
            quality: QUALITY_HIGH,
            ray_tracing: true,
            sdf_volume: true,
            ..base_query()
        },
        // High, no RT, SDF present -> SDF query.
        GpuRaytraceQuery {
            quality: QUALITY_HIGH,
            ray_tracing: false,
            sdf_volume: true,
            ..base_query()
        },
        // High, no RT, no SDF -> depth-buffer fallback.
        GpuRaytraceQuery {
            quality: QUALITY_HIGH,
            ray_tracing: false,
            sdf_volume: false,
            ..base_query()
        },
        // Medium ignores ray tracing: SDF present -> SDF.
        GpuRaytraceQuery {
            quality: QUALITY_MEDIUM,
            ray_tracing: true,
            sdf_volume: true,
            ..base_query()
        },
        // Medium, no SDF -> depth-buffer fallback.
        GpuRaytraceQuery {
            quality: QUALITY_MEDIUM,
            ray_tracing: true,
            sdf_volume: false,
            ..base_query()
        },
        // Low always falls to the depth-buffer path regardless of hardware.
        GpuRaytraceQuery {
            quality: QUALITY_LOW,
            ray_tracing: true,
            sdf_volume: true,
            ..base_query()
        },
        // An unknown quality code folds to the cheapest tier (depth buffer).
        GpuRaytraceQuery {
            quality: 7,
            ray_tracing: true,
            sdf_volume: true,
            ..base_query()
        },
    ];
    check(&ctx, &gpu, &queries);

    // Pin the expected ladder outcomes directly so a wrong port is caught even
    // if the golden ever regressed in lockstep with the twin.
    let got = gpu.eval(&ctx, &queries);
    let expected = [
        (METHOD_RAYTRACE, true),
        (METHOD_SDF, false),
        (METHOD_DEPTH_BUFFER, false),
        (METHOD_SDF, false),
        (METHOD_DEPTH_BUFFER, false),
        (METHOD_DEPTH_BUFFER, false),
        (METHOD_DEPTH_BUFFER, false),
    ];
    for (idx, (code, uses)) in expected.iter().enumerate() {
        assert_eq!(got[idx].method_code, *code, "method_code at {idx}");
        assert_eq!(
            got[idx].uses_ray_tracing, *uses,
            "uses_ray_tracing at {idx}"
        );
    }
}

#[test]
fn isotropic_lit_sample_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaytrace::new(&ctx);
    // A zero light normal marks an isotropic sample: N·L folds to 1 so the
    // irradiance is visibility * radiance, independent of the light direction.
    let query = GpuRaytraceQuery {
        light_normal: [0.0, 0.0, 0.0],
        to_light: [0.3, 0.7, 0.1],
        light_radiance: [1.5, 1.2, 0.9],
        raw_visibility: 0.6,
        ..base_query()
    };
    check(&ctx, &gpu, &[query]);
}

#[test]
fn directional_lit_samples_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaytrace::new(&ctx);
    // Head-on, oblique and back-lit directional samples, each clear of the
    // N·L = 0 hinge, plus visibility clamping and the GI gate.
    let queries = [
        // Head-on light, GI gathered: bounce passes through.
        GpuRaytraceQuery {
            light_normal: [0.0, 1.0, 0.0],
            to_light: [0.0, 1.0, 0.0],
            raw_bounce: [0.1, 0.2, 0.5],
            sample_gi: true,
            ..base_query()
        },
        // Oblique light, GI off: bounce stays zero.
        GpuRaytraceQuery {
            light_normal: [0.0, 1.0, 0.0],
            to_light: unit([0.6, 0.8, 0.0]),
            sample_gi: false,
            ..base_query()
        },
        // Back-lit (N·L < 0 before the clamp): irradiance folds to zero.
        GpuRaytraceQuery {
            light_normal: [0.0, 1.0, 0.0],
            to_light: unit([0.3, -0.9, 0.0]),
            raw_visibility: 0.9,
            ..base_query()
        },
        // Over-range visibility clamps to one on both sides.
        GpuRaytraceQuery {
            light_normal: [0.0, 1.0, 0.0],
            to_light: [0.0, 1.0, 0.0],
            raw_visibility: 1.7,
            ..base_query()
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaytrace::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing a few deterministic fixtures with many clearly-conditioned
    // random queries, dispatched together so the per-thread indexing and the
    // contiguous storage layout are both exercised, then pinned element-wise.
    let mut queries = vec![
        GpuRaytraceQuery {
            velocity: [0.4, -1.0, 0.0],
            quality: QUALITY_HIGH,
            ray_tracing: true,
            ..base_query()
        },
        GpuRaytraceQuery {
            velocity: [0.0, -1.0, 0.3],
            hit_normal: [0.0, -1.0, 0.0],
            back_face: true,
            sample_gi: true,
            raw_bounce: [0.3, 0.1, 0.2],
            ..base_query()
        },
    ];
    for _ in 0..64 {
        queries.push(well_conditioned(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_random_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaytrace::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned queries (several workgroups' worth)
    // pins every reported field across many random collision and lighting
    // geometries.
    let queries: Vec<GpuRaytraceQuery> = (0..256).map(|_| well_conditioned(&mut state)).collect();
    check(&ctx, &gpu, &queries);

    // The random stream must span more than one collision-method branch, proving
    // the discrete classifier is genuinely exercised rather than stuck.
    let got = gpu.eval(&ctx, &queries);
    let mut seen_raytrace = false;
    let mut seen_sdf = false;
    let mut seen_depth = false;
    for result in &got {
        match result.method_code {
            METHOD_RAYTRACE => seen_raytrace = true,
            METHOD_SDF => seen_sdf = true,
            _ => seen_depth = true,
        }
    }
    assert!(
        seen_raytrace && seen_sdf && seen_depth,
        "random sweep should cover all three collision methods (rt {seen_raytrace}, sdf {seen_sdf}, depth {seen_depth})"
    );
}
