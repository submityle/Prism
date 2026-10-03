//! Real-device parity for the `ReSTIR` DI spatial-admissibility twin:
//! [`GpuRestirSpatialAdmissible`](prism_volumetric_gpu::restir_spatial_admissible::GpuRestirSpatialAdmissible)
//! must reproduce the exact accept/reject `bool` of the `CPU` golden
//! [`spatial_admissible`](prism_render_architecture::lighting::restir_spatial::spatial_admissible)
//! across coplanar near-depth acceptances, depth and normal rejections, invalid
//! (non-positive, infinite, `NaN`) surfaces, boundary-adjacent cases kept clear
//! of a tie, and a randomized sweep compared sample-for-sample.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! Each `GPU` decision is pinned directly against the public golden
//! [`spatial_admissible`](prism_render_architecture::lighting::restir_spatial::spatial_admissible),
//! fed the same two [`SurfaceGeometry`](prism_render_architecture::lighting::restir_temporal::SurfaceGeometry)
//! surfaces and [`SpatialParams`](prism_render_architecture::lighting::restir_spatial::SpatialParams)
//! tolerances, so a `GPU == golden` pass is direct evidence the ported kernel
//! computes the same admissibility the reference does.
//!
//! # Parity criterion
//!
//! The decision is a single `bool`, so it is asserted with an exact `==`.
//!
//! # Conditioning
//!
//! A `GPU` multiply-add can land a few units in the last place from the scalar
//! reference, so a fixture whose `depth_diff` sits right on
//! `depth_rel_tolerance * center.view_depth`, or whose normal `dot` sits right
//! on `normal_cos_tolerance`, could flip. Every random fixture is held a `0.02`
//! margin clear of both thresholds (rejection sampling), far beyond the `f32`
//! slack, so `CPU` and `GPU` stay on the same side of every comparison. Random
//! normals are built from integer-derived components and normalized with
//! `sqrt`; no `f32` transcendental method and no `f32` `==` appears.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_spatial`；无第三方引擎源码或衍生代码。

use prism_render_architecture::lighting::restir_spatial::{spatial_admissible, SpatialParams};
use prism_render_architecture::lighting::restir_temporal::SurfaceGeometry;
use prism_volumetric_gpu::restir_spatial_admissible::{
    GpuRestirSpatialAdmissible, RestirSpatialAdmissibleQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Default relative view-depth tolerance, mirroring the reference default.
const DEPTH_REL_TOL: f32 = 0.1;

/// Default minimum normal-agreement cosine, mirroring the reference default.
const NORMAL_COS_TOL: f32 = 0.906;

/// The authoritative decision for one query: feeds the exact same two surfaces
/// and tolerances to the public golden
/// [`spatial_admissible`](prism_render_architecture::lighting::restir_spatial::spatial_admissible).
fn golden(q: &RestirSpatialAdmissibleQuery) -> bool {
    let center = SurfaceGeometry::new(q.center_depth, q.center_normal);
    let neighbor = SurfaceGeometry::new(q.neighbor_depth, q.neighbor_normal);
    let params = SpatialParams {
        depth_rel_tolerance: q.depth_rel_tolerance,
        normal_cos_tolerance: q.normal_cos_tolerance,
    };
    spatial_admissible(center, neighbor, params)
}

/// Dispatches every query and pins each `GPU` decision against the golden with
/// an exact `bool` equality.
fn check(
    ctx: &GpuContext,
    gpu: &GpuRestirSpatialAdmissible,
    queries: &[RestirSpatialAdmissibleQuery],
) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = golden(q);
        assert_eq!(
            result.admissible, want,
            "sample {idx}: gpu {} vs cpu {}",
            result.admissible, want
        );
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a signed component in `[-1.0, 1.0]` at milli resolution from `state`.
fn signed_component(state: &mut u64) -> f32 {
    (lcg(state) % 2001) as f32 / 1000.0 - 1.0
}

/// Draws a raw 3-vector with each component in `[-1.0, 1.0]`.
fn rand_vec(state: &mut u64) -> [f32; 3] {
    [
        signed_component(state),
        signed_component(state),
        signed_component(state),
    ]
}

/// Euclidean length (uses only `sqrt`, no transcendental method).
fn length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Normalizes a vector whose length the caller has already guarded above zero.
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = length(v);
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Dot product of two 3-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Builds a randomized, well-conditioned batch: random unit normals (the
/// neighbor a bounded perturbation of the center so acceptances actually occur)
/// and random positive depths with a relative offset, every sample held a
/// `0.02` margin clear of both the depth and the normal thresholds so `CPU` and
/// `GPU` land on the same side of each comparison.
fn conditioned_sweep() -> Vec<RestirSpatialAdmissibleQuery> {
    let mut state = 0x0bad_c0de_1234_5678_u64;
    let mut out: Vec<RestirSpatialAdmissibleQuery> = Vec::new();
    let mut tries = 0u32;
    while out.len() < 256 && tries < 200_000 {
        tries += 1;

        let cn_raw = rand_vec(&mut state);
        if length(cn_raw) < 0.3 {
            continue;
        }
        let center_normal = normalize(cn_raw);

        let pert_raw = rand_vec(&mut state);
        if length(pert_raw) < 0.3 {
            continue;
        }
        let pert = normalize(pert_raw);
        // Perturbation scale in `[0.0, 1.5)` sweeps the dot across the cosine
        // threshold, giving a mix of agreeing and disagreeing normals.
        let scale = (lcg(&mut state) % 1500) as f32 / 1000.0;
        let nn_raw = [
            center_normal[0] + scale * pert[0],
            center_normal[1] + scale * pert[1],
            center_normal[2] + scale * pert[2],
        ];
        if length(nn_raw) < 0.3 {
            continue;
        }
        let neighbor_normal = normalize(nn_raw);
        let dot = dot3(center_normal, neighbor_normal);
        if (dot - NORMAL_COS_TOL).abs() < 0.02 {
            continue;
        }

        // Center depth in `[0.5, 50.5)`.
        let center_depth = 0.5 + (lcg(&mut state) % 50_000) as f32 / 1000.0;
        // Relative depth offset in `[-0.25, 0.251)` sweeps the depth test.
        let offset = (lcg(&mut state) % 501) as f32 / 1000.0 - 0.25;
        let neighbor_depth = center_depth * (1.0 + offset);
        let depth_diff = (neighbor_depth - center_depth).abs();
        let depth_bound = DEPTH_REL_TOL * center_depth;
        if (depth_diff - depth_bound).abs() < 0.02 {
            continue;
        }

        out.push(RestirSpatialAdmissibleQuery::new(
            center_depth,
            center_normal,
            neighbor_depth,
            neighbor_normal,
            DEPTH_REL_TOL,
            NORMAL_COS_TOL,
        ));
    }
    out
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirSpatialAdmissible::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector with no dispatch.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn coplanar_near_depth_is_admissible() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirSpatialAdmissible::new(&ctx);
    // Same normal, depths within the 10% band: must be admissible.
    let q = RestirSpatialAdmissibleQuery::new(
        10.0,
        [0.0, 0.0, 1.0],
        10.3,
        [0.0, 0.0, 1.0],
        DEPTH_REL_TOL,
        NORMAL_COS_TOL,
    );
    assert!(golden(&q), "fixture must be admissible for the golden");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn depth_difference_over_limit_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirSpatialAdmissible::new(&ctx);
    // Depth differs by 2.5 against a 1.0 bound (0.1 * 10): rejected despite the
    // identical normals.
    let q = RestirSpatialAdmissibleQuery::new(
        10.0,
        [0.0, 0.0, 1.0],
        12.5,
        [0.0, 0.0, 1.0],
        DEPTH_REL_TOL,
        NORMAL_COS_TOL,
    );
    assert!(!golden(&q), "fixture must be rejected for the golden");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn normal_deviation_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirSpatialAdmissible::new(&ctx);
    // Identical depths, but the normals meet at a dot of 0.6 < 0.906: rejected.
    let q = RestirSpatialAdmissibleQuery::new(
        10.0,
        [0.0, 0.0, 1.0],
        10.0,
        [0.0, 0.8, 0.6],
        DEPTH_REL_TOL,
        NORMAL_COS_TOL,
    );
    assert!(!golden(&q), "fixture must be rejected for the golden");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn invalid_center_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirSpatialAdmissible::new(&ctx);
    // Every non-positive / non-finite center depth is invalid, regardless of a
    // perfectly matching neighbor. The infinity and NaN cases exercise the
    // ordered-compare `is_finite` replica on the device.
    let bad_depths = [0.0_f32, -2.0, f32::INFINITY, f32::NAN];
    let queries: Vec<RestirSpatialAdmissibleQuery> = bad_depths
        .iter()
        .map(|&d| {
            RestirSpatialAdmissibleQuery::new(
                d,
                [0.0, 0.0, 1.0],
                10.0,
                [0.0, 0.0, 1.0],
                DEPTH_REL_TOL,
                NORMAL_COS_TOL,
            )
        })
        .collect();
    for q in &queries {
        assert!(!golden(q), "invalid center must be rejected by the golden");
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn invalid_neighbor_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirSpatialAdmissible::new(&ctx);
    let bad_depths = [0.0_f32, -4.5, f32::INFINITY, f32::NAN];
    let queries: Vec<RestirSpatialAdmissibleQuery> = bad_depths
        .iter()
        .map(|&d| {
            RestirSpatialAdmissibleQuery::new(
                10.0,
                [0.0, 0.0, 1.0],
                d,
                [0.0, 0.0, 1.0],
                DEPTH_REL_TOL,
                NORMAL_COS_TOL,
            )
        })
        .collect();
    for q in &queries {
        assert!(
            !golden(q),
            "invalid neighbor must be rejected by the golden"
        );
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn boundary_adjacent_cases_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirSpatialAdmissible::new(&ctx);
    // Four cases, each a clear 0.02+ margin from its threshold: depth just
    // inside, depth just outside, normal just above the cosine, normal just
    // below it.
    let just_above_cos = normalize([0.0, 0.3, 1.0]);
    let just_below_cos = normalize([0.0, 0.55, 1.0]);
    let queries = vec![
        // depth_diff 0.5 vs bound 1.0 (admissible), normals equal.
        RestirSpatialAdmissibleQuery::new(
            10.0,
            [0.0, 0.0, 1.0],
            10.5,
            [0.0, 0.0, 1.0],
            DEPTH_REL_TOL,
            NORMAL_COS_TOL,
        ),
        // depth_diff 1.5 vs bound 1.0 (rejected), normals equal.
        RestirSpatialAdmissibleQuery::new(
            10.0,
            [0.0, 0.0, 1.0],
            11.5,
            [0.0, 0.0, 1.0],
            DEPTH_REL_TOL,
            NORMAL_COS_TOL,
        ),
        // dot ~0.958 >= 0.906 (admissible), depths equal.
        RestirSpatialAdmissibleQuery::new(
            10.0,
            [0.0, 0.0, 1.0],
            10.0,
            just_above_cos,
            DEPTH_REL_TOL,
            NORMAL_COS_TOL,
        ),
        // dot ~0.876 < 0.906 (rejected), depths equal.
        RestirSpatialAdmissibleQuery::new(
            10.0,
            [0.0, 0.0, 1.0],
            10.0,
            just_below_cos,
            DEPTH_REL_TOL,
            NORMAL_COS_TOL,
        ),
    ];
    let expected = [true, false, true, false];
    for (q, &want) in queries.iter().zip(expected.iter()) {
        assert_eq!(golden(q), want, "boundary fixture must match the golden");
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirSpatialAdmissible::new(&ctx);
    let queries = conditioned_sweep();
    assert!(
        queries.len() >= 128,
        "the conditioned sweep must yield a sizable batch, got {}",
        queries.len()
    );
    // Both branches must be exercised so the sweep is meaningful.
    let admissible = queries.iter().filter(|q| golden(q)).count();
    assert!(admissible > 0, "sweep must contain admissible samples");
    assert!(
        admissible < queries.len(),
        "sweep must contain rejected samples"
    );
    check(&ctx, &gpu, &queries);
}
