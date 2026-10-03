//! Real-device parity for the per-pair cluster/light importance twin:
//! [`GpuLightClusterImportance`](prism_volumetric_gpu::lighting_cluster_importance::GpuLightClusterImportance)
//! must reproduce the `CPU` golden
//! [`cluster_light_importance`](prism_render_architecture::lighting::stochastic::cluster_light_importance) —
//! the inverse-square falloff weight and its zero / non-zero overlap
//! classification — across a light-inside-the-box pair, a near-tangent (but
//! clearly inside) pair, an out-of-range pair, a zero-power pair, a tiny-`d²`
//! pair that exercises the `MIN_DISTANCE_SQ` clamp, and a randomized batch
//! compared pair-for-pair.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`cluster_light_importance`](prism_render_architecture::lighting::stochastic::cluster_light_importance)
//! is `pub`, so each `GPU` importance is pinned directly against the reference
//! run on the same `(cluster, light)` pair. The pair is built from the golden
//! [`ClusterBounds`](prism_render_architecture::lighting::culling::ClusterBounds)
//! and [`MegaLight`](prism_render_architecture::lighting::stochastic::MegaLight)
//! so the `GPU` and the oracle share identical inputs.
//!
//! # Parity criterion
//!
//! The non-zero importance threads through subtracts, multiplies, and a single
//! divide only, so it is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`; the zero / non-zero overlap classification is asserted
//! exactly for pairs clear of the tangency boundary.
//!
//! # Conditioning
//!
//! Every fixture keeps the squared distance `d²` far from the squared radius
//! (the discrete zero-branch boundary), so a `GPU` divide and a `CPU` divide
//! never disagree about whether the light reaches the cluster. The randomized
//! sweep rejection-samples any pair whose `|d² − radius²|` margin is small
//! relative to `radius²`, keeping both machines on the same side of the overlap
//! test.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::stochastic`；无第三方引擎源码或衍生代码。

use prism_render_architecture::lighting::culling::{ClusterBounds, LightVolume};
use prism_render_architecture::lighting::stochastic::{cluster_light_importance, MegaLight};
use prism_render_architecture::lighting::LightHandle;
use prism_volumetric_gpu::lighting_cluster_importance::{
    GpuLightClusterImportance, LightClusterImportanceQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on an importance.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes.
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

/// Builds a golden `ClusterBounds` from axis-aligned corners.
fn cluster(min: [f32; 3], max: [f32; 3]) -> ClusterBounds {
    ClusterBounds { min, max }
}

/// Builds a golden `MegaLight` from a bounding sphere and scalar power.
fn light(center: [f32; 3], radius: f32, power: f32) -> MegaLight {
    MegaLight {
        volume: LightVolume {
            light: LightHandle(0),
            center,
            radius,
        },
        power,
    }
}

/// Mirrors the golden `(cluster, light)` pair into the twin's query struct.
fn query(c: ClusterBounds, l: MegaLight) -> LightClusterImportanceQuery {
    LightClusterImportanceQuery::new(c.min, c.max, l.volume.center, l.volume.radius, l.power)
}

/// Pins one `GPU` importance against the golden oracle.
fn check_one(idx: usize, c: ClusterBounds, l: MegaLight, got: f32) {
    let want = cluster_light_importance(c, l);
    assert!(
        close(got, want),
        "pair {idx} importance: gpu {got} vs cpu {want}"
    );
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a float in `[0, 1)` from `state` using only integer work.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32
}

/// Draws a float in `[lo, hi)` from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * unit(state)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping lighting_cluster_importance parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuLightClusterImportance::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn light_inside_box_is_high_importance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClusterImportance::new(&ctx);
    // Light center inside the box: d² == 0, clamped to MIN_DISTANCE_SQ, so the
    // importance is power / 1e-4 = large but finite.
    let c = cluster([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let l = light([0.0, 0.0, 0.0], 5.0, 2.0);
    let got = gpu.evaluate(&ctx, &[query(c, l)]);
    assert_eq!(got.len(), 1);
    assert!(
        got[0].importance > 1.0,
        "a light inside the box earns a large importance, got {}",
        got[0].importance
    );
    check_one(0, c, l, got[0].importance);
}

#[test]
fn near_tangent_but_inside_is_nonzero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClusterImportance::new(&ctx);
    // Nearest box face is at x = 1; light center at x = 4 gives d² = 9. Radius
    // 4 gives radius² = 16, so d² is clearly below radius² (margin 7), well off
    // the tangency boundary.
    let c = cluster([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let l = light([4.0, 0.0, 0.0], 4.0, 3.0);
    let got = gpu.evaluate(&ctx, &[query(c, l)]);
    assert_eq!(got.len(), 1);
    assert!(
        got[0].importance > 0.0,
        "a light inside its radius earns a non-zero importance, got {}",
        got[0].importance
    );
    check_one(0, c, l, got[0].importance);
}

#[test]
fn out_of_range_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClusterImportance::new(&ctx);
    // Nearest box face at x = 1; light center at x = 10 gives d² = 81, far above
    // radius² = 4, so the light does not reach the cluster.
    let c = cluster([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let l = light([10.0, 0.0, 0.0], 2.0, 5.0);
    let got = gpu.evaluate(&ctx, &[query(c, l)]);
    assert_eq!(got.len(), 1);
    assert!(
        got[0].importance <= 0.0,
        "an out-of-range light earns exactly zero importance, got {}",
        got[0].importance
    );
    check_one(0, c, l, got[0].importance);
}

#[test]
fn zero_power_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClusterImportance::new(&ctx);
    // Light reaches the cluster (d² = 4 <= radius² = 25), but non-positive power
    // zeroes the importance.
    let c = cluster([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let l = light([3.0, 0.0, 0.0], 5.0, -2.0);
    let got = gpu.evaluate(&ctx, &[query(c, l)]);
    assert_eq!(got.len(), 1);
    assert!(
        got[0].importance <= 0.0,
        "non-positive power earns exactly zero importance, got {}",
        got[0].importance
    );
    check_one(0, c, l, got[0].importance);
}

#[test]
fn tiny_distance_hits_min_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClusterImportance::new(&ctx);
    // Nearest box face at x = 1; light center at x = 1.001 gives d² = 1e-6,
    // below MIN_DISTANCE_SQ = 1e-4, so the denominator clamps to 1e-4 and the
    // importance is power / 1e-4.
    let c = cluster([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let l = light([1.001, 0.0, 0.0], 5.0, 4.0);
    let got = gpu.evaluate(&ctx, &[query(c, l)]);
    assert_eq!(got.len(), 1);
    check_one(0, c, l, got[0].importance);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClusterImportance::new(&ctx);
    let mut state = 0x2fd1_8c47_6b3e_90a5_u64;

    let mut clusters: Vec<ClusterBounds> = Vec::new();
    let mut lights: Vec<MegaLight> = Vec::new();
    let mut queries: Vec<LightClusterImportanceQuery> = Vec::new();

    // Collect a few hundred well-conditioned pairs via rejection sampling.
    let mut guard = 0u32;
    while queries.len() < 256 && guard < 100_000 {
        guard += 1;

        // A box with positive extent around a random center.
        let cx = ranged(&mut state, -5.0, 5.0);
        let cy = ranged(&mut state, -5.0, 5.0);
        let cz = ranged(&mut state, -5.0, 5.0);
        let hx = ranged(&mut state, 0.3, 2.0);
        let hy = ranged(&mut state, 0.3, 2.0);
        let hz = ranged(&mut state, 0.3, 2.0);
        let c = cluster([cx - hx, cy - hy, cz - hz], [cx + hx, cy + hy, cz + hz]);

        // A random light center and radius; power sometimes non-positive.
        let lx = ranged(&mut state, -8.0, 8.0);
        let ly = ranged(&mut state, -8.0, 8.0);
        let lz = ranged(&mut state, -8.0, 8.0);
        let radius = ranged(&mut state, 1.0, 6.0);
        let power = if lcg(&mut state) & 7 == 0 {
            ranged(&mut state, -2.0, 0.0)
        } else {
            ranged(&mut state, 0.5, 20.0)
        };
        let l = light([lx, ly, lz], radius, power);

        // Reject pairs whose squared distance is near the squared radius (the
        // zero-branch boundary), so CPU and GPU never disagree about overlap.
        let d2 = dist_sq_point_aabb([lx, ly, lz], c.min, c.max);
        let r2 = radius * radius;
        let margin = (d2 - r2).abs();
        if margin < 0.1 * r2 {
            continue;
        }

        clusters.push(c);
        lights.push(l);
        queries.push(query(c, l));
    }

    assert!(
        queries.len() >= 256,
        "expected a full random batch, got {}",
        queries.len()
    );

    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match input count"
    );
    for (idx, result) in got.iter().enumerate() {
        check_one(idx, clusters[idx], lights[idx], result.importance);
    }
}

/// Host mirror of the golden per-axis clamped squared distance, used only to
/// reject fixtures near the overlap boundary (no transcendental, no `sqrt`).
fn dist_sq_point_aabb(p: [f32; 3], min: [f32; 3], max: [f32; 3]) -> f32 {
    let mut total = 0.0_f32;
    let mut axis = 0;
    while axis < 3 {
        if p[axis] < min[axis] {
            let d = min[axis] - p[axis];
            total += d * d;
        } else if p[axis] > max[axis] {
            let d = p[axis] - max[axis];
            total += d * d;
        }
        axis += 1;
    }
    total
}
