//! Real-device parity for the clustered light-vs-cluster overlap twin:
//! [`GpuLightClusterCull`](prism_volumetric_gpu::lighting_cluster_cull::GpuLightClusterCull)
//! must reproduce the `CPU` golden
//! [`light_overlaps_cluster`](prism_render_architecture::lighting::culling::light_overlaps_cluster)
//! — the per-axis clamped squared distance from a light's sphere center to a
//! cluster box compared against the squared radius — across a center inside the
//! box, a distant center, the grazing face-tangent, a zero-radius point light,
//! and a randomized `LCG` sweep compared pairing-for-pairing.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`light_overlaps_cluster`](prism_render_architecture::lighting::culling::light_overlaps_cluster)
//! is public, so each test calls it directly as the oracle and asserts
//! `GPU == golden` on the boolean decision. The kernel takes no `sqrt`: it
//! accumulates the clamped squared distance with `+ - *` and compares with
//! `<=`, so for pairings clear of the exact tangent tie the two sides land on
//! the same side of the comparison and the boolean agrees exactly.
//!
//! # Conditioning
//!
//! The randomized sweep rejects any pairing whose squared distance lands within
//! a margin of the squared radius, keeping every sampled decision clear of the
//! `distance_sq == radius^2` grazing boundary where a legal last-place
//! difference in the squared-distance sum could otherwise flip the result. The
//! one deliberate face-tangent fixture uses only small integers, which are
//! exactly representable in `f32`, so its squared distance and squared radius
//! are computed without rounding and the decision stays unambiguous.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::culling`；无第三方引擎源码或衍生代码。

use prism_render_architecture::lighting::culling::{
    light_overlaps_cluster, ClusterBounds, LightVolume,
};
use prism_render_architecture::lighting::LightHandle;
use prism_volumetric_gpu::lighting_cluster_cull::{
    GpuLightClusterCull, LightClusterCullQuery, LightClusterCullResult,
};
use prism_volumetric_gpu::GpuContext;

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental method and no external math dependency are involved. The
/// multiplier and increment are the well-known 64-bit `PCG` constants.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a float in `[0, 1)` from the generator using the top 24 mantissa bits,
/// built purely from integer arithmetic.
fn lcg_f32(state: &mut u64) -> f32 {
    (lcg(state) & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Host mirror of the reference clamped squared distance, used only to steer
/// rejection sampling away from the tangent boundary. Uses only `+ - *` and
/// magnitude comparisons, matching the house rules for fixtures.
fn distance_sq(center: [f32; 3], min: [f32; 3], max: [f32; 3]) -> f32 {
    let mut total = 0.0_f32;
    let mut axis = 0;
    while axis < 3 {
        let p = center[axis];
        if p < min[axis] {
            let d = min[axis] - p;
            total += d * d;
        } else if p > max[axis] {
            let d = p - max[axis];
            total += d * d;
        }
        axis += 1;
    }
    total
}

/// Builds the golden [`LightVolume`] / [`ClusterBounds`] pair from a query and
/// returns the oracle boolean.
fn golden(query: &LightClusterCullQuery) -> bool {
    let volume = LightVolume {
        light: LightHandle(0),
        center: query.center,
        radius: query.radius,
    };
    let cluster = ClusterBounds {
        min: query.cluster_min,
        max: query.cluster_max,
    };
    light_overlaps_cluster(volume, cluster)
}

/// Dispatches every pairing and pins each boolean against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuLightClusterCull, queries: &[LightClusterCullQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = LightClusterCullResult {
            overlaps: golden(q),
        };
        assert_eq!(
            *result, want,
            "pairing {idx}: gpu {:?} vs cpu {:?}",
            result, want
        );
    }
}

/// A unit cube cluster anchored at `origin`.
fn unit_cluster(origin: [f32; 3]) -> [f32; 3] {
    origin
}

#[test]
fn empty_input_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClusterCull::new(&ctx);
    // A zero-pairing batch is short-circuited on the host and never dispatched.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "empty batch must return an empty vector");
}

#[test]
fn center_inside_box_overlaps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClusterCull::new(&ctx);
    // Center well inside a unit cube with a small radius: squared distance 0.
    let origin = unit_cluster([0.0, 0.0, 0.0]);
    let q = LightClusterCullQuery::new(
        [0.5, 0.5, 0.5],
        0.1,
        origin,
        [origin[0] + 1.0, origin[1] + 1.0, origin[2] + 1.0],
    );
    assert!(golden(&q), "fixture must be an overlap on the oracle");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn distant_center_does_not_overlap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClusterCull::new(&ctx);
    // Center 10 units away on X, radius 1: comfortably short of the box.
    let origin = unit_cluster([0.0, 0.0, 0.0]);
    let q = LightClusterCullQuery::new(
        [11.0, 0.5, 0.5],
        1.0,
        origin,
        [origin[0] + 1.0, origin[1] + 1.0, origin[2] + 1.0],
    );
    assert!(!golden(&q), "fixture must be a miss on the oracle");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn grazing_face_is_handled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClusterCull::new(&ctx);
    // Center at x=3, box max x=1 -> gap 2. These small integers are exactly
    // representable in f32, so the squared distance (4) and squared radius are
    // computed without rounding: radius 2 just reaches, radius 1.9 falls short.
    let origin = unit_cluster([0.0, 0.0, 0.0]);
    let top = [origin[0] + 1.0, origin[1] + 1.0, origin[2] + 1.0];
    let reaches = LightClusterCullQuery::new([3.0, 0.5, 0.5], 2.0, origin, top);
    let short = LightClusterCullQuery::new([3.0, 0.5, 0.5], 1.9, origin, top);
    assert!(
        golden(&reaches),
        "radius 2 must reach the face on the oracle"
    );
    assert!(!golden(&short), "radius 1.9 must fall short on the oracle");
    check(&ctx, &gpu, &[reaches, short]);
}

#[test]
fn zero_radius_point_light() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClusterCull::new(&ctx);
    let origin = unit_cluster([0.0, 0.0, 0.0]);
    let top = [origin[0] + 1.0, origin[1] + 1.0, origin[2] + 1.0];
    // A zero-radius point strictly inside the slab overlaps (squared distance 0
    // <= 0); the same point pulled outside the box does not.
    let inside = LightClusterCullQuery::new([0.5, 0.5, 0.5], 0.0, origin, top);
    let outside = LightClusterCullQuery::new([2.0, 0.5, 0.5], 0.0, origin, top);
    assert!(golden(&inside), "interior zero-radius point must overlap");
    assert!(!golden(&outside), "exterior zero-radius point must miss");
    check(&ctx, &gpu, &[inside, outside]);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClusterCull::new(&ctx);

    let mut state = 0x0bad_c0de_dead_beefu64;
    let mut queries: Vec<LightClusterCullQuery> = Vec::new();
    // Margin that keeps every sampled squared distance clear of the squared
    // radius, so no decision sits on the grazing boundary.
    let margin = 0.25_f32;

    while queries.len() < 256 {
        // Box with a random anchor in [-4, 4] and random positive extents.
        let min = [
            lcg_f32(&mut state) * 8.0 - 4.0,
            lcg_f32(&mut state) * 8.0 - 4.0,
            lcg_f32(&mut state) * 8.0 - 4.0,
        ];
        let max = [
            min[0] + 0.5 + lcg_f32(&mut state) * 2.0,
            min[1] + 0.5 + lcg_f32(&mut state) * 2.0,
            min[2] + 0.5 + lcg_f32(&mut state) * 2.0,
        ];
        // Center in a wider box so both hits and misses are sampled.
        let center = [
            lcg_f32(&mut state) * 12.0 - 6.0,
            lcg_f32(&mut state) * 12.0 - 6.0,
            lcg_f32(&mut state) * 12.0 - 6.0,
        ];
        let radius = lcg_f32(&mut state) * 4.0;

        let dsq = distance_sq(center, min, max);
        let rsq = radius * radius;
        // Reject pairings grazing the tangent tie.
        if (dsq - rsq).abs() < margin {
            continue;
        }
        queries.push(LightClusterCullQuery::new(center, radius, min, max));
    }

    // Sanity: the sweep must exercise both outcomes.
    let hits = queries.iter().filter(|q| golden(q)).count();
    assert!(hits > 0, "sweep produced no overlaps");
    assert!(hits < queries.len(), "sweep produced no misses");

    check(&ctx, &gpu, &queries);
}
