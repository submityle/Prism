//! Real-device parity for the per-surface transparency routing twin:
//! [`GpuTransparencyPathSelect`](prism_volumetric_gpu::transparency_path_select::GpuTransparencyPathSelect)
//! must reproduce the `CPU` golden
//! [`select_transparency_path`](prism_render_architecture::transparency::routing::select_transparency_path)
//! and
//! [`outputs_for`](prism_render_architecture::transparency::routing::outputs_for)
//! across every content class and routing branch, each path's shared-target
//! writes, and a randomized combination sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`select_transparency_path`](prism_render_architecture::transparency::routing::select_transparency_path)
//! and
//! [`outputs_for`](prism_render_architecture::transparency::routing::outputs_for)
//! are called directly in-host to produce the expected
//! [`TransparencyPath`](prism_render_architecture::transparency::TransparencyPath)
//! and
//! [`TransparencyOutputs`](prism_render_architecture::transparency::TransparencyOutputs),
//! which are then mapped to the `path_code` and `outputs_bits` encoding the
//! kernel emits. A `GPU == oracle` pass therefore establishes `GPU == golden`.
//!
//! # Parity criterion
//!
//! Every output is a discriminant or a bit mask built from integer comparisons
//! and boolean logic — there is no floating-point arithmetic anywhere — so the
//! `CPU` and `GPU` agree bit-for-bit and every output is asserted with an exact
//! `==`. There is no tolerance and no degenerate numeric region.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::transparency::routing`；无第三方引擎源码或衍生代码。

use prism_render_architecture::transparency::routing::{
    outputs_for, select_transparency_path, TransparencyCapability, TransparentKind,
    TransparentSurface,
};
use prism_render_architecture::transparency::{TransparencyOutputs, TransparencyPath};
use prism_volumetric_gpu::transparency_path_select::{
    GpuTransparencyPathSelect, TransparencyPathSelectQuery, TransparencyPathSelectResult,
};
use prism_volumetric_gpu::GpuContext;

/// Maps a golden [`TransparencyPath`] to the kernel's `path_code`, mirroring the
/// golden `as u32`.
fn path_code(path: TransparencyPath) -> u32 {
    match path {
        TransparencyPath::Sorted => TransparencyPathSelectResult::PATH_SORTED,
        TransparencyPath::WeightedOit => TransparencyPathSelectResult::PATH_WEIGHTED_OIT,
        TransparencyPath::MomentOit => TransparencyPathSelectResult::PATH_MOMENT_OIT,
        TransparencyPath::LayeredGlass => TransparencyPathSelectResult::PATH_LAYERED_GLASS,
        TransparencyPath::SingleLayerWater => TransparencyPathSelectResult::PATH_SINGLE_LAYER_WATER,
        TransparencyPath::HairVisibility => TransparencyPathSelectResult::PATH_HAIR_VISIBILITY,
        TransparencyPath::Volumetric => TransparencyPathSelectResult::PATH_VOLUMETRIC,
    }
}

/// Packs a golden [`TransparencyOutputs`] into the kernel's `outputs_bits`
/// encoding (bit0 reactive, bit1 motion, bit2 ray scene).
fn outputs_bits(outputs: TransparencyOutputs) -> u32 {
    let mut bits = 0u32;
    if outputs.writes_reactive_mask {
        bits |= TransparencyPathSelectResult::OUT_REACTIVE_MASK;
    }
    if outputs.writes_motion {
        bits |= TransparencyPathSelectResult::OUT_MOTION;
    }
    if outputs.contributes_to_ray_scene {
        bits |= TransparencyPathSelectResult::OUT_RAY_SCENE;
    }
    bits
}

/// Builds a golden surface from the twin query fields.
fn golden_surface(q: &TransparencyPathSelectQuery, kind: TransparentKind) -> TransparentSurface {
    TransparentSurface {
        kind,
        order_independent: q.order_independent,
        high_fidelity: q.high_fidelity,
        layer_count: q.layer_count,
    }
}

/// Returns the golden [`TransparentKind`] for a `kind_code`.
fn kind_from_code(code: u32) -> TransparentKind {
    match code {
        TransparencyPathSelectQuery::KIND_WATER => TransparentKind::Water,
        TransparencyPathSelectQuery::KIND_HAIR => TransparentKind::Hair,
        TransparencyPathSelectQuery::KIND_GLASS => TransparentKind::Glass,
        TransparencyPathSelectQuery::KIND_VOLUME => TransparentKind::Volume,
        _ => TransparentKind::General,
    }
}

/// Computes the expected [`TransparencyPathSelectResult`] from the golden.
fn oracle(q: &TransparencyPathSelectQuery) -> TransparencyPathSelectResult {
    let kind = kind_from_code(q.kind_code);
    let surface = golden_surface(q, kind);
    let capability = TransparencyCapability {
        moment_oit: q.moment_oit,
    };
    let path = select_transparency_path(surface, capability);
    TransparencyPathSelectResult {
        path_code: path_code(path),
        outputs_bits: outputs_bits(outputs_for(path)),
    }
}

/// Pins one `GPU` surface result against the in-host oracle: both the path code
/// and the packed output mask exactly.
fn check_surface(
    idx: usize,
    got: &TransparencyPathSelectResult,
    want: &TransparencyPathSelectResult,
) {
    assert_eq!(
        got.path_code, want.path_code,
        "surface {idx} path_code: gpu {} vs cpu {}",
        got.path_code, want.path_code
    );
    assert_eq!(
        got.outputs_bits, want.outputs_bits,
        "surface {idx} outputs_bits: gpu {} vs cpu {}",
        got.outputs_bits, want.outputs_bits
    );
}

/// Dispatches `queries` and checks every result against the oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuTransparencyPathSelect,
    queries: &[TransparencyPathSelectQuery],
) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_surface(idx, g, &want);
    }
}

/// Enumerates every content class and routing branch at least once.
fn enumerated_fixtures() -> Vec<TransparencyPathSelectQuery> {
    vec![
        // Water, hair and volume always take their dedicated paths.
        TransparencyPathSelectQuery::new(
            TransparencyPathSelectQuery::KIND_WATER,
            1,
            false,
            false,
            false,
        ),
        TransparencyPathSelectQuery::new(
            TransparencyPathSelectQuery::KIND_HAIR,
            1,
            false,
            false,
            false,
        ),
        TransparencyPathSelectQuery::new(
            TransparencyPathSelectQuery::KIND_VOLUME,
            1,
            false,
            false,
            false,
        ),
        // Glass: single layer sorts, multi-layer takes the layered resolve.
        TransparencyPathSelectQuery::new(
            TransparencyPathSelectQuery::KIND_GLASS,
            1,
            false,
            false,
            false,
        ),
        TransparencyPathSelectQuery::new(
            TransparencyPathSelectQuery::KIND_GLASS,
            4,
            false,
            false,
            false,
        ),
        // General, order-dependent: cheap sorted path.
        TransparencyPathSelectQuery::new(
            TransparencyPathSelectQuery::KIND_GENERAL,
            1,
            false,
            false,
            false,
        ),
        // General, order-independent, no fidelity: weighted OIT.
        TransparencyPathSelectQuery::new(
            TransparencyPathSelectQuery::KIND_GENERAL,
            1,
            true,
            false,
            true,
        ),
        // General, order-independent, fidelity + capability: moment OIT.
        TransparencyPathSelectQuery::new(
            TransparencyPathSelectQuery::KIND_GENERAL,
            1,
            true,
            true,
            true,
        ),
        // General, order-independent, fidelity but no capability: weighted OIT.
        TransparencyPathSelectQuery::new(
            TransparencyPathSelectQuery::KIND_GENERAL,
            1,
            true,
            true,
            false,
        ),
    ]
}

/// A small `LCG` for the randomized sweep (host-only; the kernel is integer).
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
    let gpu = GpuTransparencyPathSelect::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn all_branches_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTransparencyPathSelect::new(&ctx);
    check(&ctx, &gpu, &enumerated_fixtures());
}

#[test]
fn each_path_outputs_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTransparencyPathSelect::new(&ctx);
    // Every path appears at least once in the enumerated fixtures, so this
    // exercises the full `outputs_for` table against the oracle.
    let fixtures = enumerated_fixtures();
    let got = gpu.evaluate(&ctx, &fixtures);
    for (idx, (q, g)) in fixtures.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert_eq!(
            g.outputs_bits, want.outputs_bits,
            "surface {idx} outputs_bits mismatch"
        );
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTransparencyPathSelect::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = enumerated_fixtures();
    // Many random combinations across every content class, layer count, and
    // flag triple, several workgroups' worth, pin every routing branch.
    for _ in 0..512 {
        let kind_code = lcg(&mut state) % 5;
        let layer_count = lcg(&mut state) % 6;
        let order_independent = lcg(&mut state) & 1 == 1;
        let high_fidelity = lcg(&mut state) & 1 == 1;
        let moment_oit = lcg(&mut state) & 1 == 1;
        queries.push(TransparencyPathSelectQuery::new(
            kind_code,
            layer_count,
            order_independent,
            high_fidelity,
            moment_oit,
        ));
    }
    check(&ctx, &gpu, &queries);
}
