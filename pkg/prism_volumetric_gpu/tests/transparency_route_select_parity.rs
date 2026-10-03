//! Real-device parity for the transparency-route selector twin:
//! [`GpuTransparencyRouteSelect`](prism_volumetric_gpu::transparency_route_select::GpuTransparencyRouteSelect)
//! must reproduce the `CPU` golden
//! [`select_transparency_path`](prism_render_architecture::transparency::routing::select_transparency_path)
//! and
//! [`outputs_for`](prism_render_architecture::transparency::routing::outputs_for)
//! across every content class, flag combination, and glass layer split, plus a
//! randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected outcome is produced by calling the golden
//! `select_transparency_path` and `outputs_for` directly, so the test pins
//! `GPU == golden`, not merely that the shader compiles. The path discriminant
//! is read from the golden enum's `as u32`, so both sides share one encoding.
//!
//! # Parity criterion
//!
//! Every field is a discrete discriminant or boolean, so parity is asserted
//! with exact equality.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::transparency::routing`；无第三方引擎源码或衍生代码。

use prism_render_architecture::transparency::routing::{
    outputs_for, select_transparency_path, TransparencyCapability, TransparentKind,
    TransparentSurface,
};
use prism_render_architecture::transparency::TransparencyPath;
use prism_volumetric_gpu::transparency_route_select::{
    GpuTransparencyRouteSelect, TransparencyRouteSelectQuery, TransparencyRouteSelectResult,
};
use prism_volumetric_gpu::GpuContext;

/// Maps a content-class discriminant back to the golden `TransparentKind`,
/// matching the twin's `TransparentKind as u32` encoding.
fn kind_from_code(code: u32) -> TransparentKind {
    match code {
        1 => TransparentKind::Water,
        2 => TransparentKind::Hair,
        3 => TransparentKind::Glass,
        4 => TransparentKind::Volume,
        _ => TransparentKind::General,
    }
}

/// Reconstructs the golden outcome in-host by calling the reference routines
/// directly. The path discriminant is read from `TransparencyPath as u32`.
fn oracle(q: &TransparencyRouteSelectQuery) -> TransparencyRouteSelectResult {
    let surface = TransparentSurface {
        kind: kind_from_code(q.kind),
        order_independent: q.order_independent,
        high_fidelity: q.high_fidelity,
        layer_count: q.layer_count,
    };
    let capability = TransparencyCapability {
        moment_oit: q.moment_oit,
    };
    let path: TransparencyPath = select_transparency_path(surface, capability);
    let outputs = outputs_for(path);
    TransparencyRouteSelectResult {
        path_code: path as u32,
        writes_reactive_mask: outputs.writes_reactive_mask,
        writes_motion: outputs.writes_motion,
        contributes_to_ray_scene: outputs.contributes_to_ray_scene,
    }
}

/// Pins one `GPU` result against the in-host oracle: exact path discriminant and
/// exact output booleans.
fn check_query(
    idx: usize,
    got: &TransparencyRouteSelectResult,
    want: &TransparencyRouteSelectResult,
) {
    assert_eq!(
        got.path_code, want.path_code,
        "query {idx} path_code: gpu {} vs cpu {}",
        got.path_code, want.path_code
    );
    assert_eq!(
        got.writes_reactive_mask, want.writes_reactive_mask,
        "query {idx} writes_reactive_mask: gpu {} vs cpu {}",
        got.writes_reactive_mask, want.writes_reactive_mask
    );
    assert_eq!(
        got.writes_motion, want.writes_motion,
        "query {idx} writes_motion: gpu {} vs cpu {}",
        got.writes_motion, want.writes_motion
    );
    assert_eq!(
        got.contributes_to_ray_scene, want.contributes_to_ray_scene,
        "query {idx} contributes_to_ray_scene: gpu {} vs cpu {}",
        got.contributes_to_ray_scene, want.contributes_to_ray_scene
    );
}

/// Dispatches `queries` and checks every result against the oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuTransparencyRouteSelect,
    queries: &[TransparencyRouteSelectQuery],
) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_query(idx, g, &want);
    }
}

/// A small `LCG` for the randomized sweep (host-only; the kernel is portable).
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

#[test]
fn empty_batch_produces_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTransparencyRouteSelect::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn full_enum_cross_product_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTransparencyRouteSelect::new(&ctx);
    // Every content class crossed with both values of each flag and a
    // single-layer vs multi-layer glass split: 5 * 2 * 2 * 2 * 2 = 80 queries,
    // which exhausts every branch of the selector and every output arm.
    let mut queries: Vec<TransparencyRouteSelectQuery> = Vec::new();
    for kind in 0u32..5 {
        for order_independent in [false, true] {
            for high_fidelity in [false, true] {
                for moment_oit in [false, true] {
                    for layer_count in [1u32, 3] {
                        queries.push(TransparencyRouteSelectQuery::new(
                            kind,
                            order_independent,
                            high_fidelity,
                            moment_oit,
                            layer_count,
                        ));
                    }
                }
            }
        }
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn moment_oit_needs_both_request_and_capability() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTransparencyRouteSelect::new(&ctx);
    // A high-fidelity order-independent general surface: moment OIT only when
    // the backend supports it, weighted OIT otherwise.
    let requested_unsupported = TransparencyRouteSelectQuery::new(0, true, true, false, 1);
    let requested_supported = TransparencyRouteSelectQuery::new(0, true, true, true, 1);
    let queries = [requested_unsupported, requested_supported];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), 2);
    // WeightedOit = 1, MomentOit = 2 in the shared encoding.
    assert_eq!(
        got[0].path_code, 1,
        "unsupported request falls back to weighted"
    );
    assert_eq!(got[1].path_code, 2, "supported request takes moment OIT");
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTransparencyRouteSelect::new(&ctx);
    let mut state = 0x5151_f00d_c0de_1234u64;
    let mut queries: Vec<TransparencyRouteSelectQuery> = Vec::new();
    // Several workgroups' worth of random surfaces: random content class,
    // random flags, and a layer count spanning the single/multi split.
    while queries.len() < 512 {
        let kind = lcg(&mut state) % 5;
        let order_independent = lcg(&mut state) & 1 == 1;
        let high_fidelity = lcg(&mut state) & 1 == 1;
        let moment_oit = lcg(&mut state) & 1 == 1;
        let layer_count = lcg(&mut state) % 5;
        queries.push(TransparencyRouteSelectQuery::new(
            kind,
            order_independent,
            high_fidelity,
            moment_oit,
            layer_count,
        ));
    }
    check(&ctx, &gpu, &queries);
}
