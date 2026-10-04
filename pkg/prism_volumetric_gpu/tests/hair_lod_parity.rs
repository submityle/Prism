//! Real-device parity for the screen-coverage hair LOD twin:
//! [`GpuHairLod`](prism_volumetric_gpu::hair_lod::GpuHairLod) must reproduce the
//! `CPU` golden `select_hair_lod_tier` and `resolve_hair_lod` of
//! `prism_render_architecture::hair::lod`, which map a hair group's projected
//! screen coverage to a discrete LOD tier and resolve the render-strand and
//! control-point budget that tier keeps.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the ordered coverage ladder (`Strands` / `ReducedStrands` / `Cards` /
//! `Mesh`), the coarser-of clamp against the group's authored `native_form`,
//! and the per-tier integer strand/segment decimation (never below one) — so
//! the test never imports `prism_render_architecture`.
//!
//! The fixtures cover every tier, the `native_form` clamp (a card-authored
//! groom is never promoted to strands at any coverage), the exact decimation of
//! the reduced-strand tier, the `.max(1)` floor on tiny budgets, the
//! out-of-range `native_form` rejection, and a batch of several distinct
//! queries that catches any `std430` stride aliasing. A sweep over random
//! coverage, threshold triples, `native_form` and budgets follows, plus an
//! empty batch the host short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Every output is a discrete integer (the tier index, the strand and segment
//! budgets, the validity flag), so every field is compared with an exact `==`:
//! there is no floating-point result to tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::lod`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hair_lod::{GpuHairLod, HairLodQuery};
use prism_volumetric_gpu::GpuContext;

/// Independent oracle for the coverage ladder, returning the tier index
/// (`0` = `Strands` .. `3` = `Mesh`), mirroring the golden if / else-if chain.
fn select_tier(coverage: f32, reduced_below: f32, cards_below: f32, mesh_below: f32) -> u32 {
    if coverage >= reduced_below {
        0
    } else if coverage >= cards_below {
        1
    } else if coverage >= mesh_below {
        2
    } else {
        3
    }
}

/// Independent oracle for one query: `(tier, render_strands, segments, valid)`.
///
/// A `native_form` greater than `3` is not a valid tier index and is rejected
/// with `valid = 0` and all outputs cleared, mirroring the kernel guard.
fn oracle(q: &HairLodQuery) -> (u32, u32, u32, u32) {
    if q.native_form > 3 {
        return (0, 0, 0, 0);
    }
    let coverage_tier = select_tier(
        q.coverage,
        q.reduced_strands_below,
        q.cards_below,
        q.mesh_below,
    );
    // coarser_of: the tier with the larger coarseness index wins.
    let tier = coverage_tier.max(q.native_form);
    let (render_strands, segments) = match tier {
        0 => (q.max_render_strands, q.segments_per_strand),
        1 => (
            (q.max_render_strands / 4).max(1),
            (q.segments_per_strand / 2).max(1),
        ),
        _ => (0, 0),
    };
    (tier, render_strands, segments, 1)
}

/// Dispatches one query and asserts every field against the oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuHairLod, q: HairLodQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let (tier, render_strands, segments, valid) = oracle(&q);
    let r = got[0];
    assert_eq!(r.valid, valid, "valid flag mismatch: query={q:?}");
    assert_eq!(r.tier, tier, "tier mismatch: query={q:?}");
    assert_eq!(
        r.render_strands, render_strands,
        "render_strands mismatch: query={q:?}"
    );
    assert_eq!(
        r.segments_per_strand, segments,
        "segments_per_strand mismatch: query={q:?}"
    );
}

#[test]
fn strands_tier_keeps_authored_budget() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLod::new(&ctx);
    // Coverage above the reduced-strands threshold with a strand-authored groom:
    // the full strand tier keeps the authored counts.
    assert_parity(
        &ctx,
        &gpu,
        HairLodQuery::new(0.95, 0.5, 0.2, 0.05, 0, 40_000, 8),
    );
}

#[test]
fn reduced_strands_tier_decimates_exactly() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLod::new(&ctx);
    // Coverage between cards and reduced thresholds: the reduced-strand tier
    // decimates strands to a quarter (40000 -> 10000) and segments to a half
    // (8 -> 4).
    assert_parity(
        &ctx,
        &gpu,
        HairLodQuery::new(0.35, 0.5, 0.2, 0.05, 0, 40_000, 8),
    );
}

#[test]
fn cards_tier_keeps_no_geometry() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLod::new(&ctx);
    // Coverage between mesh and cards thresholds: the cards tier drops all
    // per-strand geometry.
    assert_parity(
        &ctx,
        &gpu,
        HairLodQuery::new(0.1, 0.5, 0.2, 0.05, 0, 40_000, 8),
    );
}

#[test]
fn mesh_tier_keeps_no_geometry() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLod::new(&ctx);
    // Coverage below the mesh threshold: the groom collapses to a static mesh
    // shell with no per-strand geometry.
    assert_parity(
        &ctx,
        &gpu,
        HairLodQuery::new(0.01, 0.5, 0.2, 0.05, 0, 40_000, 8),
    );
}

#[test]
fn native_form_clamps_card_authored_groom() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLod::new(&ctx);
    // A card-authored groom (native_form = Cards = 2) at near-full coverage: the
    // coverage would pick Strands, but the coarser-of clamp keeps it at Cards,
    // never promoting it to strands it does not own.
    assert_parity(
        &ctx,
        &gpu,
        HairLodQuery::new(0.99, 0.5, 0.2, 0.05, 2, 40_000, 8),
    );
}

#[test]
fn native_form_mesh_stays_mesh() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLod::new(&ctx);
    // A mesh-only groom (native_form = Mesh = 3) stays a mesh shell even at full
    // coverage.
    assert_parity(
        &ctx,
        &gpu,
        HairLodQuery::new(1.0, 0.5, 0.2, 0.05, 3, 40_000, 8),
    );
}

#[test]
fn reduced_strands_budget_floors_at_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLod::new(&ctx);
    // Tiny authored budgets at the reduced-strand tier: 2 / 4 = 0 and 1 / 2 = 0
    // both floor to 1, so the groom keeps at least one strand and segment.
    assert_parity(&ctx, &gpu, HairLodQuery::new(0.35, 0.5, 0.2, 0.05, 0, 2, 1));
}

#[test]
fn native_form_clamps_coverage_tier() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLod::new(&ctx);
    // Coverage picks ReducedStrands (1) but native_form is Cards (2): the
    // coarser tier wins and the groom renders as cards.
    assert_parity(
        &ctx,
        &gpu,
        HairLodQuery::new(0.35, 0.5, 0.2, 0.05, 2, 40_000, 8),
    );
}

#[test]
fn out_of_range_native_form_rejects() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLod::new(&ctx);
    // A native_form outside 0..=3 is not a valid tier index: the query is
    // rejected with valid = 0 and all outputs cleared.
    assert_parity(
        &ctx,
        &gpu,
        HairLodQuery::new(0.5, 0.5, 0.2, 0.05, 4, 40_000, 8),
    );
}

#[test]
fn coverage_exactly_on_threshold_picks_finer() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLod::new(&ctx);
    // Coverage exactly equal to the reduced-strands threshold: the `>=` ladder
    // keeps the finer Strands tier.
    assert_parity(
        &ctx,
        &gpu,
        HairLodQuery::new(0.5, 0.5, 0.2, 0.05, 0, 40_000, 8),
    );
}

#[test]
fn batch_stride_reads_non_aliased_slots() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLod::new(&ctx);
    // A batch of several distinct queries exercises the std430 query/result
    // stride: every slot must read and write its own non-aliased data.
    let queries = [
        HairLodQuery::new(0.95, 0.5, 0.2, 0.05, 0, 40_000, 8),
        HairLodQuery::new(0.35, 0.5, 0.2, 0.05, 0, 32_000, 6),
        HairLodQuery::new(0.1, 0.5, 0.2, 0.05, 0, 20_000, 4),
        HairLodQuery::new(0.01, 0.5, 0.2, 0.05, 0, 10_000, 2),
        HairLodQuery::new(0.99, 0.5, 0.2, 0.05, 2, 40_000, 8),
        HairLodQuery::new(0.5, 0.5, 0.2, 0.05, 4, 40_000, 8),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (tier, render_strands, segments, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert_eq!(r.tier, tier, "batch tier mismatch: query={q:?}");
        assert_eq!(
            r.render_strands, render_strands,
            "batch render_strands mismatch: query={q:?}"
        );
        assert_eq!(
            r.segments_per_strand, segments,
            "batch segments_per_strand mismatch: query={q:?}"
        );
    }
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

    /// A `[lo, hi]` integer.
    fn next_u32_in(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u32() % (hi - lo + 1))
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLod::new(&ctx);
    let mut rng = Lcg::new(0x1D_7A_C3_51);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        let coverage = rng.next_range(0.0, 1.0);
        // Ordered threshold triple (coarsest-last), spread so the coverage lands
        // in every ladder band across the sweep.
        let reduced_below = rng.next_range(0.5, 0.9);
        let cards_below = rng.next_range(0.2, 0.5);
        let mesh_below = rng.next_range(0.02, 0.2);
        // Mostly valid tier indices 0..=3, occasionally out of range to hit the
        // rejection path.
        let roll = rng.next_u32_in(0, 9);
        let native_form = if roll == 9 {
            rng.next_u32_in(4, 7)
        } else {
            rng.next_u32_in(0, 3)
        };
        let max_render_strands = rng.next_u32_in(1, 60_000);
        let segments_per_strand = rng.next_u32_in(1, 16);
        queries.push(HairLodQuery::new(
            coverage,
            reduced_below,
            cards_below,
            mesh_below,
            native_form,
            max_render_strands,
            segments_per_strand,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (tier, render_strands, segments, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert_eq!(r.tier, tier, "sweep tier mismatch: query={q:?}");
        assert_eq!(
            r.render_strands, render_strands,
            "sweep render_strands mismatch: query={q:?}"
        );
        assert_eq!(
            r.segments_per_strand, segments,
            "sweep segments_per_strand mismatch: query={q:?}"
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLod::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}
