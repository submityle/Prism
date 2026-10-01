//! Real-device parity for the Variance-Shadow-Map resolve twin:
//! [`GpuVarianceShadow`](prism_volumetric_gpu::variance_shadow::GpuVarianceShadow)
//! must reproduce the `CPU` golden
//! [`resolve_visibility`](prism_render_architecture::particle::variance_shadow::resolve_visibility)
//! across empty batches, fully-lit and fully-shadowed receivers, the penumbra
//! probability-bound band, the light-bleed `linstep` remap, the `min_variance`
//! floor and a large batch of random moments and random receiver depths.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each lane is a fixed, non-reorderable sequence of multiplies, adds and one
//! divide (the variance recovery, the `Chebyshev` ratio and the light-bleed
//! `linstep`), so `CPU` and `GPU` evaluate the same closed form in the same
//! order. The comparison allows `abs_diff <= 1e-5` or `rel_diff <= 1e-5` —
//! loose enough to admit a legal fused multiply-add contraction, yet tight
//! enough to fail a wrong port (a dropped variance floor, a missing in-front
//! early-out, a flipped light-bleed span).
//!
//! Provenance: standard two-moment Variance Shadow Maps; no Unreal Engine
//! source or derived code.

use prism_render_architecture::particle::variance_shadow::{
    chebyshev_upper_bound, filter_moments, light_bleed_reduction, resolve_visibility, Moments,
};
use prism_volumetric_gpu::variance_shadow::{GpuVarianceShadow, VarianceShadowQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute/relative parity bound. A `GPU` may fuse a multiply-add the scalar
/// reference leaves separate, perturbing the low mantissa bits by a few units
/// in the last place; `1e-5` admits that legal slack while still failing a
/// genuinely wrong port.
const EPS: f32 = 1.0e-5;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= EPS
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    // 24 usable mantissa bits mapped onto [0, 1).
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Runs the `GPU` dispatch and compares every receiver against the `CPU` golden
/// [`resolve_visibility`], returning the `GPU` results for any extra per-test
/// assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuVarianceShadow,
    min_variance: f32,
    bleed: f32,
    queries: &[VarianceShadowQuery],
) -> Vec<f32> {
    let got = gpu.eval(ctx, min_variance, bleed, queries);
    assert_eq!(got.len(), queries.len(), "one visibility per receiver");

    let map: Vec<Moments> = queries.iter().map(|q| q.moments).collect();
    let depths: Vec<f32> = queries.iter().map(|q| q.receiver_depth).collect();
    let want = resolve_visibility(&map, &depths, min_variance, bleed);

    for (lane, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(g, w),
            "lane {lane}: gpu {g} vs cpu {w} (min_variance {min_variance}, bleed {bleed})"
        );
    }
    got
}

/// Convenience constructor for a receiver testing `receiver_depth` against a
/// single-occluder moment pair at `occluder_depth`.
fn point_occluder(occluder_depth: f32, receiver_depth: f32) -> VarianceShadowQuery {
    VarianceShadowQuery {
        moments: Moments::from_depth(occluder_depth),
        receiver_depth,
    }
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVarianceShadow::new(&ctx);
    let got = gpu.eval(&ctx, 1.0e-4, 0.0, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn receiver_in_front_is_fully_lit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVarianceShadow::new(&ctx);
    // Every receiver sits in front of (or exactly at) the mean occluder depth,
    // so the Chebyshev bound is full visibility and no light-bleed remap can
    // darken it (bleed 0 is the identity on 1.0).
    let queries = [
        point_occluder(5.0, 4.999),
        point_occluder(5.0, 5.0),
        point_occluder(10.0, 2.0),
        point_occluder(3.0, -1.0),
    ];
    let got = check(&ctx, &gpu, 1.0e-4, 0.0, &queries);
    for (lane, &visibility) in got.iter().enumerate() {
        assert!(
            close(visibility, 1.0),
            "lane {lane}: receiver in front should be fully lit, got {visibility}"
        );
    }
}

#[test]
fn receiver_behind_point_occluder_is_fully_shadowed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVarianceShadow::new(&ctx);
    // A point distribution has zero variance; behind the occluder with no
    // variance floor the Chebyshev bound collapses to a hard zero.
    let queries = [
        point_occluder(5.0, 5.001),
        point_occluder(5.0, 6.0),
        point_occluder(2.0, 50.0),
    ];
    let got = check(&ctx, &gpu, 0.0, 0.0, &queries);
    for (lane, &visibility) in got.iter().enumerate() {
        assert!(
            close(visibility, 0.0),
            "lane {lane}: receiver behind a point occluder should be shadowed, got {visibility}"
        );
    }
}

#[test]
fn penumbra_band_matches_chebyshev_bound() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVarianceShadow::new(&ctx);
    // A filtered moment pair with real variance gives a partial (penumbra)
    // visibility in the open interval (0, 1); the dispatch must track the
    // Chebyshev bound across a swept depth and stay monotonically decreasing.
    let moments = filter_moments(
        &[Moments::from_depth(3.0), Moments::from_depth(7.0)],
        &[1.0, 1.0],
    );
    let min_variance = 1.0e-4;
    let mut queries = Vec::with_capacity(31);
    for step in 0u8..=30 {
        let receiver_depth = 5.0 + f32::from(step) * 0.5;
        queries.push(VarianceShadowQuery {
            moments,
            receiver_depth,
        });
    }
    let got = check(&ctx, &gpu, min_variance, 0.0, &queries);

    // At least one lane lands strictly inside the penumbra band.
    let has_partial = got.iter().any(|&v| v > EPS && v < 1.0 - EPS);
    assert!(has_partial, "expected a partial-visibility penumbra lane");

    // The bound is monotonically non-increasing with receiver depth.
    let mut prev = f32::INFINITY;
    for (lane, &visibility) in got.iter().enumerate() {
        assert!(
            visibility <= prev + EPS,
            "lane {lane}: visibility should not increase with depth ({visibility} > {prev})"
        );
        prev = visibility;
    }
}

#[test]
fn light_bleed_reduction_takes_effect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVarianceShadow::new(&ctx);
    // A wide occluder distribution produces a bright partial bound that the
    // light-bleed remap must darken; the parity check already compares against
    // the CPU remap, and we additionally confirm the remap moved the result.
    let moments = filter_moments(
        &[Moments::from_depth(0.0), Moments::from_depth(20.0)],
        &[1.0, 1.0],
    );
    let receiver_depth = 12.0;
    let min_variance = 1.0e-4;
    let query = VarianceShadowQuery {
        moments,
        receiver_depth,
    };

    let raw = chebyshev_upper_bound(&moments, receiver_depth, min_variance);
    // Pick a bleed amount strictly below the raw bound so the remap is active
    // (neither clamps to zero nor is the identity).
    assert!(
        raw > 0.3 && raw < 1.0,
        "test fixture should sit in the bleed band, raw bound was {raw}"
    );
    let bleed = 0.25;
    let expected = light_bleed_reduction(raw, bleed);
    assert!(
        expected < raw - EPS,
        "the chosen bleed should darken the raw bound ({expected} vs {raw})"
    );

    let got = check(&ctx, &gpu, min_variance, bleed, &[query]);
    assert!(
        close(got[0], expected),
        "gpu {} should match the CPU light-bleed remap {expected}",
        got[0]
    );
}

#[test]
fn min_variance_floor_clamps_bound() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVarianceShadow::new(&ctx);
    // A point occluder has zero variance, so behind it the raw bound is zero.
    // Raising `min_variance` floors the variance and lifts the bound; the
    // dispatch must reproduce that floor exactly for both settings.
    let queries = [point_occluder(5.0, 6.0)];

    let low = check(&ctx, &gpu, 1.0e-4, 0.0, &queries);
    let high = check(&ctx, &gpu, 1.0, 0.0, &queries);
    assert!(
        high[0] >= low[0] - EPS,
        "a larger min_variance should not lower the bound ({} vs {})",
        high[0],
        low[0]
    );
    assert!(
        high[0] > low[0] + EPS,
        "a larger min_variance should visibly lift the floored bound ({} vs {})",
        high[0],
        low[0]
    );
}

#[test]
fn random_moments_and_receivers_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVarianceShadow::new(&ctx);
    let mut state = 0x5eed_1234_abcd_0001_u64;

    for round in 0u32..8 {
        // Sweep a spread of variance floors and bleed amounts across rounds so
        // every branch (in-front early-out, floored variance, degenerate span)
        // is exercised against random inputs.
        let min_variance = lcg(&mut state) * 0.5;
        let bleed = lcg(&mut state);
        let mut queries = Vec::with_capacity(96);
        for _ in 0..96 {
            // Build each receiver from a convex blend of two random occluder
            // depths so the moment pair is physically consistent
            // (`m2 >= m1^2`), then test it against a random receiver depth that
            // spans in-front, penumbra and fully-shadowed regions.
            let depth_a = lcg(&mut state) * 20.0;
            let depth_b = lcg(&mut state) * 20.0;
            let weight_a = lcg(&mut state) + 1.0e-3;
            let weight_b = lcg(&mut state) + 1.0e-3;
            let moments = filter_moments(
                &[Moments::from_depth(depth_a), Moments::from_depth(depth_b)],
                &[weight_a, weight_b],
            );
            let receiver_depth = lcg(&mut state) * 24.0 - 2.0;
            queries.push(VarianceShadowQuery {
                moments,
                receiver_depth,
            });
        }
        // `check` asserts lane-for-lane parity against the CPU golden.
        let _ = check(&ctx, &gpu, min_variance, bleed, &queries);
        let _ = round;
    }
}
