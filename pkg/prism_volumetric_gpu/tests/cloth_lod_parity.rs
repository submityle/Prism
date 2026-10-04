//! Real-device parity for the cloth level-of-detail twin:
//! [`GpuClothLod`](prism_volumetric_gpu::cloth_lod::GpuClothLod) must reproduce
//! the `CPU` golden `resolve_cloth_lod` (its coverage-selected tier clamped to
//! `native_form`, plus the sim-vertex / constraint budget) and
//! `select_cloth_lod_tier_hysteretic` of `prism_render_architecture::cloth::lod`
//! in a single query.
//!
//! The oracle here is an independent re-implementation of those closed-form
//! classifiers — the ordered coverage test, the `max(count / 4, 1)` integer
//! decimation, the `coarser_of` clamp (a plain `max` of coarseness indices) and
//! the symmetric hysteresis dead-band — written out directly so the test never
//! imports `prism_render_architecture`. It mirrors the reference branch for
//! branch, including the out-of-range rejection that reports `valid = 0`.
//!
//! The fixtures cover every tier, the hysteresis dead-band hold (a `current`
//! tier held between the two edges), the `hysteresis == 0` collapse onto the
//! stateless classifier, the `native_form` clamp that keeps a skin-only garment
//! skinned at full coverage, a hard multi-tier camera cut, and the invalid
//! `current` / `native_form` rejection. A mixed-tier batch validates the
//! `std430` stride, and a `512`-query `LCG` sweep follows, plus an empty batch
//! the host short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every output is a discrete tier or an integer budget: there is no continuous
//! channel that could admit a fused-multiply-add drift. Every field is compared
//! exactly with `assert_eq!`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::lod`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cloth_lod::{ClothLodQuery, ClothLodResult, GpuClothLod};
use prism_volumetric_gpu::GpuContext;

/// Tier encoding mirrored from the golden coarseness index.
const FULL_SIM: u32 = 0;
/// Tier encoding for reduced-resolution simulation.
const REDUCED_SIM: u32 = 1;
/// Tier encoding for the static skinned proxy.
const SKINNED_PROXY: u32 = 2;

/// Independent host re-implementation of `select_cloth_lod_tier`: ordered
/// coverage classifier, coarsest-last.
fn select_tier(coverage: f32, reduced_sim_below: f32, skinned_below: f32) -> u32 {
    if coverage >= reduced_sim_below {
        FULL_SIM
    } else if coverage >= skinned_below {
        REDUCED_SIM
    } else {
        SKINNED_PROXY
    }
}

/// Independent host re-implementation of the `cloth_lod_budget` decimation
/// rule: full sim keeps the authored counts, reduced sim keeps `max(_ / 4, 1)`,
/// the skinned proxy keeps none.
fn budget(tier: u32, sim_vertex_count: u32, constraint_count: u32) -> (u32, u32) {
    match tier {
        FULL_SIM => (sim_vertex_count, constraint_count),
        REDUCED_SIM => ((sim_vertex_count / 4).max(1), (constraint_count / 4).max(1)),
        _ => (0, 0),
    }
}

/// Independent host re-implementation of `select_cloth_lod_tier_hysteretic`:
/// a symmetric dead-band holds `current` between the two edges.
fn hysteretic(
    coverage: f32,
    reduced_sim_below: f32,
    skinned_below: f32,
    hysteresis: f32,
    current: u32,
) -> u32 {
    let band = hysteresis.max(0.0);
    let reduced_down = reduced_sim_below - band;
    let reduced_up = reduced_sim_below + band;
    let skinned_down = skinned_below - band;
    let skinned_up = skinned_below + band;
    match current {
        FULL_SIM => {
            if coverage < skinned_down {
                SKINNED_PROXY
            } else if coverage < reduced_down {
                REDUCED_SIM
            } else {
                FULL_SIM
            }
        }
        REDUCED_SIM => {
            if coverage >= reduced_up {
                FULL_SIM
            } else if coverage < skinned_down {
                SKINNED_PROXY
            } else {
                REDUCED_SIM
            }
        }
        _ => {
            if coverage >= reduced_up {
                FULL_SIM
            } else if coverage >= skinned_up {
                REDUCED_SIM
            } else {
                SKINNED_PROXY
            }
        }
    }
}

/// Full independent oracle for one query: the stateless resolve (tier clamped
/// to `native_form`, plus its budget) and the hysteretic tier, with the
/// out-of-range rejection mirrored.
fn oracle(q: &ClothLodQuery) -> ClothLodResult {
    if q.current > 2 || q.native_form > 2 {
        return ClothLodResult {
            tier_stateless: 0,
            sim_vertices: 0,
            constraints: 0,
            tier_hysteretic: 0,
            valid: 0,
        };
    }
    let selected = select_tier(q.coverage, q.reduced_sim_below, q.skinned_below);
    let tier = selected.max(q.native_form);
    let (sim_vertices, constraints) = budget(tier, q.sim_vertex_count, q.constraint_count);
    let tier_hysteretic = hysteretic(
        q.coverage,
        q.reduced_sim_below,
        q.skinned_below,
        q.hysteresis,
        q.current,
    );
    ClothLodResult {
        tier_stateless: tier,
        sim_vertices,
        constraints,
        tier_hysteretic,
        valid: 1,
    }
}

/// Asserts the single-query GPU result matches the oracle exactly.
fn assert_parity(ctx: &GpuContext, gpu: &GpuClothLod, q: ClothLodQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    assert_eq!(got[0], oracle(&q), "result mismatch: query={q:?}");
}

#[test]
fn full_sim_at_high_coverage() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // Coverage above reduced_sim_below -> FullSim, authored counts kept intact.
    assert_parity(
        &ctx,
        &gpu,
        ClothLodQuery::new(0.9, 0.5, 0.2, 0.0, FULL_SIM, FULL_SIM, 4000, 12000),
    );
}

#[test]
fn reduced_sim_in_mid_band() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // Coverage between the two thresholds -> ReducedSim, counts quartered.
    assert_parity(
        &ctx,
        &gpu,
        ClothLodQuery::new(0.35, 0.5, 0.2, 0.0, FULL_SIM, FULL_SIM, 4000, 12000),
    );
}

#[test]
fn skinned_proxy_at_low_coverage() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // Coverage below skinned_below -> SkinnedProxy, zero sim budget.
    assert_parity(
        &ctx,
        &gpu,
        ClothLodQuery::new(0.05, 0.5, 0.2, 0.0, FULL_SIM, FULL_SIM, 4000, 12000),
    );
}

#[test]
fn reduced_sim_floor_keeps_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // Tiny authored counts decimate to the max(_, 1) floor, not zero.
    assert_parity(
        &ctx,
        &gpu,
        ClothLodQuery::new(0.3, 0.5, 0.2, 0.0, FULL_SIM, FULL_SIM, 2, 3),
    );
}

#[test]
fn hysteresis_holds_current_in_dead_band() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // Coverage 0.46 sits inside the FullSim dead-band (reduced_down = 0.45),
    // so a FullSim garment is held even though the stateless tier is ReducedSim.
    let q = ClothLodQuery::new(0.46, 0.5, 0.2, 0.05, FULL_SIM, FULL_SIM, 4000, 12000);
    assert_eq!(oracle(&q).tier_hysteretic, FULL_SIM, "fixture sanity");
    assert_eq!(oracle(&q).tier_stateless, REDUCED_SIM, "fixture sanity");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn hysteresis_holds_reduced_between_edges() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // From ReducedSim: 0.3 is below reduced_up (0.55) and at/above skinned_down
    // (0.15), so the tier is held at ReducedSim.
    let q = ClothLodQuery::new(0.3, 0.5, 0.2, 0.05, REDUCED_SIM, FULL_SIM, 4000, 12000);
    assert_eq!(oracle(&q).tier_hysteretic, REDUCED_SIM, "fixture sanity");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn hysteresis_holds_skinned_between_edges() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // From SkinnedProxy: 0.22 is below reduced_up (0.55) and below skinned_up
    // (0.25), so the tier is held at SkinnedProxy.
    let q = ClothLodQuery::new(0.22, 0.5, 0.2, 0.05, SKINNED_PROXY, FULL_SIM, 4000, 12000);
    assert_eq!(oracle(&q).tier_hysteretic, SKINNED_PROXY, "fixture sanity");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn zero_band_reduces_to_stateless() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // With hysteresis == 0 the hysteretic tier equals the stateless classifier
    // for every current tier.
    for current in [FULL_SIM, REDUCED_SIM, SKINNED_PROXY] {
        let q = ClothLodQuery::new(0.35, 0.5, 0.2, 0.0, current, FULL_SIM, 4000, 12000);
        let stateless = select_tier(q.coverage, q.reduced_sim_below, q.skinned_below);
        assert_eq!(oracle(&q).tier_hysteretic, stateless, "fixture sanity");
        assert_parity(&ctx, &gpu, q);
    }
}

#[test]
fn native_form_clamps_skin_only_garment() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // Full coverage would select FullSim, but a skin-only piece
    // (native_form = SkinnedProxy) stays skinned with zero sim budget.
    let q = ClothLodQuery::new(0.99, 0.5, 0.2, 0.0, FULL_SIM, SKINNED_PROXY, 4000, 12000);
    assert_eq!(oracle(&q).tier_stateless, SKINNED_PROXY, "fixture sanity");
    assert_eq!(oracle(&q).sim_vertices, 0, "fixture sanity");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn native_form_clamps_to_reduced() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // Full coverage selects FullSim, but native_form = ReducedSim coarsens it.
    let q = ClothLodQuery::new(0.99, 0.5, 0.2, 0.0, FULL_SIM, REDUCED_SIM, 4000, 12000);
    assert_eq!(oracle(&q).tier_stateless, REDUCED_SIM, "fixture sanity");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn hard_cut_full_to_skinned() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // A hard camera cut: from FullSim straight to a very low coverage resolves
    // directly to SkinnedProxy in one frame, not one tier per frame.
    let q = ClothLodQuery::new(0.001, 0.5, 0.2, 0.05, FULL_SIM, FULL_SIM, 4000, 12000);
    assert_eq!(oracle(&q).tier_hysteretic, SKINNED_PROXY, "fixture sanity");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn hard_cut_skinned_to_full() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // The symmetric cut back: from SkinnedProxy to near-full coverage climbs
    // straight to FullSim.
    let q = ClothLodQuery::new(0.99, 0.5, 0.2, 0.05, SKINNED_PROXY, FULL_SIM, 4000, 12000);
    assert_eq!(oracle(&q).tier_hysteretic, FULL_SIM, "fixture sanity");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn out_of_range_current_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // current = 3 is out of range -> valid = 0 with cleared outputs.
    let q = ClothLodQuery::new(0.9, 0.5, 0.2, 0.0, 3, FULL_SIM, 4000, 12000);
    assert_eq!(oracle(&q).valid, 0, "fixture sanity");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn out_of_range_native_form_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // native_form = 7 is out of range -> valid = 0 with cleared outputs.
    let q = ClothLodQuery::new(0.9, 0.5, 0.2, 0.0, FULL_SIM, 7, 4000, 12000);
    assert_eq!(oracle(&q).valid, 0, "fixture sanity");
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    // A >=2-element batch mixing every tier, a held dead-band, a native-form
    // clamp and an invalid query validates the std430 stride end to end.
    let queries = vec![
        ClothLodQuery::new(0.9, 0.5, 0.2, 0.0, FULL_SIM, FULL_SIM, 4000, 12000),
        ClothLodQuery::new(0.35, 0.5, 0.2, 0.0, FULL_SIM, FULL_SIM, 4000, 12000),
        ClothLodQuery::new(0.05, 0.5, 0.2, 0.0, FULL_SIM, FULL_SIM, 4000, 12000),
        ClothLodQuery::new(0.46, 0.5, 0.2, 0.05, FULL_SIM, FULL_SIM, 4000, 12000),
        ClothLodQuery::new(0.99, 0.5, 0.2, 0.0, FULL_SIM, SKINNED_PROXY, 4000, 12000),
        ClothLodQuery::new(0.9, 0.5, 0.2, 0.0, 3, FULL_SIM, 4000, 12000),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        assert_eq!(*r, oracle(q), "mixed batch mismatch: query={q:?}");
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}

/// A small deterministic linear-congruential generator so the sweep needs no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }

    /// An integer in `[0, n)`.
    fn next_below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothLod::new(&ctx);
    let mut rng = Lcg::new(0x51_0D_3A_77);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        // Ordered thresholds in [0, 1], reduced_sim_below >= skinned_below.
        let skinned_below = rng.next_range(0.05, 0.45);
        let reduced_sim_below = rng.next_range(skinned_below + 0.05, 0.95);
        // Coverage spans the whole range so every tier and both dead-band edges
        // are exercised; the outputs are discrete so no knife-edge guard is
        // needed, but a mix of exact and random hysteresis keeps the band live.
        let coverage = rng.next_range(0.0, 1.0);
        let hysteresis = if rng.next_below(4) == 0 {
            0.0
        } else {
            rng.next_range(0.0, 0.1)
        };
        // current and native_form occasionally stray out of range to drive the
        // valid = 0 rejection path.
        let current = rng.next_below(4);
        let native_form = rng.next_below(4);
        let sim_vertex_count = rng.next_below(20_000);
        let constraint_count = rng.next_below(60_000);
        queries.push(ClothLodQuery::new(
            coverage,
            reduced_sim_below,
            skinned_below,
            hysteresis,
            current,
            native_form,
            sim_vertex_count,
            constraint_count,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        assert_eq!(*r, oracle(q), "sweep mismatch: query={q:?}");
    }
}
