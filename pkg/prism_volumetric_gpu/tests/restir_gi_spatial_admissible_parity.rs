//! Real-device parity for the `ReSTIR` GI spatial-admissibility twin:
//! [`GpuRestirGiSpatialAdmissible`](prism_volumetric_gpu::restir_gi_spatial_admissible::GpuRestirGiSpatialAdmissible)
//! must reproduce the `CPU` golden
//! [`gi_spatial_admissible`](prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible) —
//! the depth / normal / validity screen a spatial-reuse neighbor must pass — on
//! clear accepts, clear rejects (depth mismatch, normal mismatch, invalid
//! surfaces) and a conditioned randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! [`gi_spatial_admissible`](prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible)
//! is public, so each `GPU` flag is pinned directly against a host call:
//! [`GiSurface`](prism_render_architecture::lighting::restir_gi_resolve::GiSurface)
//! pair and a
//! [`GiSpatialParams`](prism_render_architecture::lighting::restir_gi_resolve::GiSpatialParams)
//! are rebuilt from the query and evaluated on the host. The shading-point
//! positions are irrelevant to the predicate, so they carry a placeholder.
//!
//! # Parity criterion
//!
//! The predicate is discrete, so the admissibility flag is asserted with an
//! exact `==`.
//!
//! # Conditioning
//!
//! The randomized sweep keeps every pair well clear of both comparison
//! boundaries — the depth agreement margin `|z_n − z_c| vs tol · z_c` and the
//! cosine threshold `n_c · n_n vs cos_tol` — so a last-place difference in the
//! shared `+ - * /` sequence can never flip the decision and desync the two
//! sides. All depths are finite and positive by construction.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_gi_resolve::gi_spatial_admissible`；无第三方引擎源码或衍生代码。

use prism_render_architecture::lighting::restir_gi::ShadingPoint;
use prism_render_architecture::lighting::restir_gi_resolve::{
    gi_spatial_admissible, GiSpatialParams, GiSurface,
};
use prism_volumetric_gpu::restir_gi_spatial_admissible::{
    GpuRestirGiSpatialAdmissible, RestirGiSpatialAdmissibleQuery, RestirGiSpatialAdmissibleResult,
};
use prism_volumetric_gpu::GpuContext;

/// Computes the expected admissibility in-host via the golden
/// `gi_spatial_admissible`: the faithful oracle the `GPU` is pinned against. The
/// shading-point positions are unused by the predicate, so they carry a
/// placeholder.
fn oracle(q: &RestirGiSpatialAdmissibleQuery) -> bool {
    let center = GiSurface::new(
        q.center_view_depth,
        ShadingPoint::new([0.0, 0.0, 0.0], q.center_normal),
    );
    let neighbor = GiSurface::new(
        q.neighbor_view_depth,
        ShadingPoint::new([0.0, 0.0, 0.0], q.neighbor_normal),
    );
    let params = GiSpatialParams {
        depth_rel_tolerance: q.depth_rel_tolerance,
        normal_cos_tolerance: q.normal_cos_tolerance,
    };
    gi_spatial_admissible(center, neighbor, params)
}

/// Pins one `GPU` flag against the in-host oracle with an exact match.
fn check_sample(idx: usize, got: &RestirGiSpatialAdmissibleResult, want: bool) {
    assert_eq!(
        got.admissible, want,
        "sample {idx} admissible: gpu {} vs cpu {}",
        got.admissible, want
    );
}

/// Dispatches every sample and pins each result against the oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuRestirGiSpatialAdmissible,
    queries: &[RestirGiSpatialAdmissibleQuery],
) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "one result per query must come back"
    );
    for (idx, (res, q)) in got.iter().zip(queries.iter()).enumerate() {
        check_sample(idx, res, oracle(q));
    }
}

/// A 64-bit linear-congruential generator (`PCG`-style multiplier); only
/// integer work, so no transcendental appears. Returns the raw high bits.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws an `f32` in `[0.0, 1.0]` at milli resolution from `state`.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) % 1001) as f32 / 1000.0
}

/// Draws a normalized direction from three centered components, rejecting a
/// near-zero vector so the normalization is well defined. Uses only `+ - * /`
/// and `sqrt` (not a transcendental), so no forbidden host method appears.
fn normal(state: &mut u64) -> [f32; 3] {
    loop {
        let v = [
            unit(state) * 2.0 - 1.0,
            unit(state) * 2.0 - 1.0,
            unit(state) * 2.0 - 1.0,
        ];
        let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if len2 < 0.25 {
            continue;
        }
        let inv = 1.0 / len2.sqrt();
        return [v[0] * inv, v[1] * inv, v[2] * inv];
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping restir_gi_spatial_admissible parity: no wgpu adapter");
        return;
    };
    let gpu = GpuRestirGiSpatialAdmissible::new(&ctx);
    // An empty batch must short-circuit on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn identical_surface_is_admissible() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirGiSpatialAdmissible::new(&ctx);
    // Same depth and normal: zero depth diff, cosine 1 >= tol → admissible.
    let q =
        RestirGiSpatialAdmissibleQuery::new(10.0, 10.0, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.1, 0.9);
    let got = gpu.evaluate(&ctx, &[q]);
    assert!(got[0].admissible, "identical surface must be admissible");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn depth_mismatch_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirGiSpatialAdmissible::new(&ctx);
    // Depth diff 5 far exceeds tol*z = 0.1*10 = 1 → rejected despite aligned
    // normals.
    let q =
        RestirGiSpatialAdmissibleQuery::new(10.0, 15.0, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.1, 0.9);
    let got = gpu.evaluate(&ctx, &[q]);
    assert!(!got[0].admissible, "a large depth gap must be rejected");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn normal_mismatch_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirGiSpatialAdmissible::new(&ctx);
    // Opposing normals: cosine -1 < 0.9 → rejected despite matching depth.
    let q = RestirGiSpatialAdmissibleQuery::new(
        10.0,
        10.0,
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        0.1,
        0.9,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert!(!got[0].admissible, "opposing normals must be rejected");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn invalid_surfaces_are_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirGiSpatialAdmissible::new(&ctx);
    // A non-positive depth, a +inf depth and a NaN depth each mark an invalid
    // surface and must be rejected, matching the golden `is_valid`.
    let n = [0.0, 0.0, 1.0];
    let queries = [
        RestirGiSpatialAdmissibleQuery::new(0.0, 10.0, n, n, 0.1, 0.9),
        RestirGiSpatialAdmissibleQuery::new(-3.0, 10.0, n, n, 0.1, 0.9),
        RestirGiSpatialAdmissibleQuery::new(10.0, f32::INFINITY, n, n, 0.1, 0.9),
        RestirGiSpatialAdmissibleQuery::new(f32::NAN, 10.0, n, n, 0.1, 0.9),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    for (idx, res) in got.iter().enumerate() {
        assert!(!res.admissible, "invalid surface {idx} must be rejected");
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirGiSpatialAdmissible::new(&ctx);
    // A mix of accepts and rejects dispatched together so per-thread indexing
    // and contiguous output slots are both exercised.
    let queries = [
        RestirGiSpatialAdmissibleQuery::new(4.0, 4.2, [0.0, 0.0, 1.0], [0.1, 0.0, 0.995], 0.1, 0.9),
        RestirGiSpatialAdmissibleQuery::new(8.0, 9.5, [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 0.1, 0.9),
        RestirGiSpatialAdmissibleQuery::new(2.0, 2.05, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.1, 0.5),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRestirGiSpatialAdmissible::new(&ctx);
    let mut state = 0x2f6a_1c93_84db_e715_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of random pairs, each conditioned away from both
    // comparison boundaries so a last-place difference cannot flip the discrete
    // decision.
    while queries.len() < 256 {
        let center_depth = 0.5 + unit(&mut state) * 9.5;
        let neighbor_depth = 0.5 + unit(&mut state) * 9.5;
        let cn = normal(&mut state);
        let nn = normal(&mut state);
        let depth_rel_tolerance = 0.05 + unit(&mut state) * 0.25;
        let normal_cos_tolerance = 0.3 + unit(&mut state) * 0.6;

        // Keep the depth comparison clear of its boundary by a relative margin.
        let depth_diff = (neighbor_depth - center_depth).abs();
        let depth_bound = depth_rel_tolerance * center_depth;
        if (depth_diff - depth_bound).abs() < 0.02 * center_depth {
            continue;
        }
        // Keep the cosine comparison clear of its boundary.
        let cos = cn[0] * nn[0] + cn[1] * nn[1] + cn[2] * nn[2];
        if (cos - normal_cos_tolerance).abs() < 0.02 {
            continue;
        }

        queries.push(RestirGiSpatialAdmissibleQuery::new(
            center_depth,
            neighbor_depth,
            cn,
            nn,
            depth_rel_tolerance,
            normal_cos_tolerance,
        ));
    }
    check(&ctx, &gpu, &queries);
}
