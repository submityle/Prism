//! Real-device parity for the transparent-draw binner twin:
//! [`GpuTransparencyBinDraws`](prism_volumetric_gpu::transparency_bin_draws::GpuTransparencyBinDraws)
//! must reproduce the `CPU` golden
//! [`bin_transparent_draws`](prism_render_architecture::transparency::routing::bin_transparent_draws)
//! across stale-draw skips, every content class, flag combination, glass layer
//! split, intra-bucket order preservation, and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected bins are produced by calling the golden `bin_transparent_draws`
//! directly, so the test pins `GPU == golden`, not merely that the shader
//! compiles. The per-bucket order and lengths are read straight from the golden
//! [`TransparencyBins`](prism_render_architecture::transparency::routing::TransparencyBins),
//! and the twin's bucket indices follow `TransparencyPath as u32`, so both
//! sides share one encoding.
//!
//! # Parity criterion
//!
//! Every value is a discrete count or draw index, so parity is asserted with
//! exact equality, including the first-seen order inside every bucket.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::transparency::routing`；无第三方引擎源码或衍生代码。

use prism_render_architecture::transparency::routing::{
    bin_transparent_draws, TransparencyCapability, TransparentKind, TransparentSurface,
};
use prism_render_architecture::transparency::TransparencyPath;
use prism_volumetric_gpu::transparency_bin_draws::{
    GpuTransparencyBinDraws, TransparencyBinDrawsQuery, TransparencyBinDrawsResult,
};
use prism_volumetric_gpu::GpuContext;

/// Maps a content-class discriminant to the golden `TransparentKind`, matching
/// the twin's `TransparentKind as u32` encoding.
fn kind_from_code(code: u32) -> TransparentKind {
    match code {
        1 => TransparentKind::Water,
        2 => TransparentKind::Hair,
        3 => TransparentKind::Glass,
        4 => TransparentKind::Volume,
        _ => TransparentKind::General,
    }
}

/// A host-side description of one binning scenario: the draw list and the
/// parallel per-slot surface fields plus the backend capability.
#[derive(Clone)]
struct Scenario {
    draws: Vec<u32>,
    kinds: Vec<u32>,
    order_independent: Vec<bool>,
    high_fidelity: Vec<bool>,
    layer_count: Vec<u32>,
    moment_oit: bool,
}

impl Scenario {
    /// Builds the twin query from the parallel per-slot fields.
    fn query(&self) -> TransparencyBinDrawsQuery {
        TransparencyBinDrawsQuery::new(
            &self.draws,
            &self.kinds,
            &self.order_independent,
            &self.high_fidelity,
            &self.layer_count,
            self.moment_oit,
        )
    }

    /// Rebuilds the golden surfaces (one per slot described) for the oracle.
    fn surfaces(&self) -> Vec<TransparentSurface> {
        (0..self.kinds.len())
            .map(|i| TransparentSurface {
                kind: kind_from_code(self.kinds[i]),
                order_independent: self.order_independent[i],
                high_fidelity: self.high_fidelity[i],
                layer_count: self.layer_count[i],
            })
            .collect()
    }
}

/// Pins one `GPU` result against the in-host oracle: for each of the seven
/// paths, the exact bucket length and the exact first-seen-order draw indices.
fn check_result(idx: usize, got: &TransparencyBinDrawsResult, scenario: &Scenario) {
    let surfaces = scenario.surfaces();
    let capability = TransparencyCapability {
        moment_oit: scenario.moment_oit,
    };
    let golden = bin_transparent_draws(&scenario.draws, &surfaces, capability);
    let paths = [
        TransparencyPath::Sorted,
        TransparencyPath::WeightedOit,
        TransparencyPath::MomentOit,
        TransparencyPath::LayeredGlass,
        TransparencyPath::SingleLayerWater,
        TransparencyPath::HairVisibility,
        TransparencyPath::Volumetric,
    ];
    for path in paths {
        let p = path as u32 as usize;
        let want = golden.bucket(path);
        assert_eq!(
            got.count(p) as usize,
            want.len(),
            "scenario {idx} path {p} count: gpu {} vs cpu {}",
            got.count(p),
            want.len()
        );
        assert_eq!(got.bucket(p), want, "scenario {idx} path {p} bucket order");
    }
}

/// Dispatches every scenario's query and checks each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuTransparencyBinDraws, scenarios: &[Scenario]) {
    let queries: Vec<TransparencyBinDrawsQuery> = scenarios.iter().map(Scenario::query).collect();
    let got = gpu.evaluate(ctx, &queries);
    assert_eq!(got.len(), scenarios.len(), "one result per query");
    for (idx, (scenario, g)) in scenarios.iter().zip(got.iter()).enumerate() {
        check_result(idx, g, scenario);
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
    let gpu = GpuTransparencyBinDraws::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn stale_draws_past_surfaces_are_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTransparencyBinDraws::new(&ctx);
    // Six draws but only three surfaces: slots 3..6 have no surface and are
    // dropped, mirroring the golden surfaces.get(slot) returning None.
    let scenario = Scenario {
        draws: vec![10, 11, 12, 13, 14, 15],
        kinds: vec![1, 2, 4], // Water, Hair, Volume
        order_independent: vec![false, false, false],
        high_fidelity: vec![false, false, false],
        layer_count: vec![1, 1, 1],
        moment_oit: false,
    };
    check(&ctx, &gpu, &[scenario]);
}

#[test]
fn all_kinds_route_into_their_buckets() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTransparencyBinDraws::new(&ctx);
    // One draw per content class plus the two glass splits and the OIT splits,
    // with moment_oit toggled across two scenarios.
    let mut scenarios: Vec<Scenario> = Vec::new();
    for moment_oit in [false, true] {
        // Slots: Water, Hair, Volume, Glass(1 layer), Glass(3 layers),
        // General(sortable), General(OIT, low fidelity),
        // General(OIT, high fidelity).
        scenarios.push(Scenario {
            draws: vec![100, 101, 102, 103, 104, 105, 106, 107],
            kinds: vec![1, 2, 4, 3, 3, 0, 0, 0],
            order_independent: vec![false, false, false, false, false, false, true, true],
            high_fidelity: vec![false, false, false, false, false, false, false, true],
            layer_count: vec![1, 1, 1, 1, 3, 1, 1, 1],
            moment_oit,
        });
    }
    check(&ctx, &gpu, &scenarios);
}

#[test]
fn intra_bucket_order_is_preserved() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTransparencyBinDraws::new(&ctx);
    // Interleave surfaces that route to the same bucket (all Water) so the
    // bucket must keep first-seen order 20, 21, 22, 23, 24.
    let scenario = Scenario {
        draws: vec![20, 21, 22, 23, 24],
        kinds: vec![1, 1, 1, 1, 1],
        order_independent: vec![false; 5],
        high_fidelity: vec![false; 5],
        layer_count: vec![1; 5],
        moment_oit: true,
    };
    // Also interleave two paths to confirm independent ordering per bucket.
    let mixed = Scenario {
        draws: vec![30, 31, 32, 33, 34, 35],
        // Water, Hair, Water, Hair, Water, Hair.
        kinds: vec![1, 2, 1, 2, 1, 2],
        order_independent: vec![false; 6],
        high_fidelity: vec![false; 6],
        layer_count: vec![1; 6],
        moment_oit: false,
    };
    check(&ctx, &gpu, &[scenario, mixed]);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTransparencyBinDraws::new(&ctx);
    let mut state = 0x00bd_a4ac_0de7_7788u64;
    let mut scenarios: Vec<Scenario> = Vec::new();
    // A batch of random draw lists: random draw count, a surface count that may
    // be shorter than the draw list (exercising the stale-draw skip), random
    // content class, flags, layer count, and backend capability. Draw ids are
    // sequential within a scenario so order mismatches surface immediately.
    while scenarios.len() < 128 {
        let draw_count = (lcg(&mut state) % 24) as usize + 1;
        // Surface count sometimes shorter than the draw list.
        let surface_count = (lcg(&mut state) as usize % (draw_count + 1)).min(draw_count);
        let draws: Vec<u32> = (0..draw_count).map(|i| 1000 + i as u32).collect();
        let kinds: Vec<u32> = (0..surface_count).map(|_| lcg(&mut state) % 5).collect();
        let order_independent: Vec<bool> = (0..surface_count)
            .map(|_| lcg(&mut state) & 1 == 1)
            .collect();
        let high_fidelity: Vec<bool> = (0..surface_count)
            .map(|_| lcg(&mut state) & 1 == 1)
            .collect();
        let layer_count: Vec<u32> = (0..surface_count)
            .map(|_| 1 + lcg(&mut state) % 4)
            .collect();
        let moment_oit = lcg(&mut state) & 1 == 1;
        scenarios.push(Scenario {
            draws,
            kinds,
            order_independent,
            high_fidelity,
            layer_count,
            moment_oit,
        });
    }
    check(&ctx, &gpu, &scenarios);
}
