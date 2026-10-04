//! Real-device parity for the hair LOD cross-fade twin:
//! [`GpuHairLodTransition`](prism_volumetric_gpu::hair_lod_transition::GpuHairLodTransition)
//! must reproduce the `CPU` golden
//! `prism_render_architecture::hair::transition::resolve_hair_lod_transition`,
//! which spreads each discrete LOD tier switch across a small coverage band so a
//! groom cross-fades between a finer and a coarser tier instead of popping in a
//! single frame.
//!
//! The oracle here is an independent re-implementation of that closed form — the
//! hard tier pick `select_hair_lod_tier`, the `native_form` clamp via
//! `coarser_of` (integer `max` of coarseness ranks) and the fixed
//! three-boundary nearest-threshold scan with its single clamped `blend`
//! divide — written out directly so the test never imports
//! `prism_render_architecture`. It mirrors the reference branch for branch,
//! including the `band <= 0` / non-finite degrade to a hard pick and the
//! `native_form` collapse that settles a cross-fade.
//!
//! The per-strand screen-door `strand_survives_dither` is deliberately not
//! twinned: it hashes with a `splitmix64`-style `u64` mixer the portable
//! core-`WGSL` subset cannot express, so there is nothing to compare for it.
//!
//! The fixtures cover the branches the kernel must honor: a disabled band that
//! settles, a coverage comfortably inside a band that cross-fades, a
//! `native_form` clamp that collapses a would-be fade to a settle, two
//! equidistant bands that must resolve first-seen, an out-of-range
//! `native_form` that reports `valid = 0`, and a mixed multi-element batch that
//! validates the `std430` stride. A `512`-step sweep over random coverage,
//! band and authored tier follows, plus an empty batch the host
//! short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The only continuous output is `blend`, a single clamped divide, so the
//! comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`);
//! the discrete `from_tier`, `to_tier`, `is_cross_fading` and `valid` fields are
//! compared exactly. The sweep keeps the coverage a safe margin away from every
//! `threshold ± band` knee (both the branch boundary `distance == band` and the
//! `blend` clamp edges land there) and away from any equidistant-threshold tie,
//! so parity never sits on a branch knife edge.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::transition`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hair_lod_transition::{GpuHairLodTransition, HairLodTransitionQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous `blend` comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous `blend` comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Largest magnitude accepted as finite, matching the kernel and golden.
const F32_MAX_FINITE: f32 = 3.4e38;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Portable finiteness test, replicated identically to the kernel: `x == x` is
/// the `NaN` self-compare (a `NaN` is the only value unequal to itself) and the
/// magnitude guard rejects the two infinities.
fn is_finite(x: f32) -> bool {
    (x == x) && (x.abs() < F32_MAX_FINITE)
}

/// Hard coverage-to-tier classification, mirroring `select_hair_lod_tier`.
fn select_tier(cov: f32, rsb: f32, cb: f32, mb: f32) -> u32 {
    if cov >= rsb {
        0
    } else if cov >= cb {
        1
    } else if cov >= mb {
        2
    } else {
        3
    }
}

/// Integer `coarser_of`: the coarser (higher-rank) of two tiers.
fn coarser_of(a: u32, b: u32) -> u32 {
    a.max(b)
}

/// Independent oracle for one query, returning
/// `(from_tier, to_tier, blend, is_cross_fading, valid)`.
fn oracle(q: &HairLodTransitionQuery) -> (u32, u32, f32, u32, u32) {
    if q.native_form > 3 {
        return (0, 0, 0.0, 0, 0);
    }
    let native = q.native_form;
    let cov = q.coverage;
    let band = q.band;

    let hard_tier = coarser_of(
        select_tier(cov, q.reduced_strands_below, q.cards_below, q.mesh_below),
        native,
    );

    let settled = |tier: u32| (tier, tier, 0.0_f32, 0_u32, 1_u32);

    if band <= 0.0 || !is_finite(band) || !is_finite(cov) {
        return settled(hard_tier);
    }

    let boundaries = [
        (q.reduced_strands_below, 0u32, 1u32),
        (q.cards_below, 1u32, 2u32),
        (q.mesh_below, 2u32, 3u32),
    ];

    let mut best: Option<(f32, u32, u32, f32)> = None;
    for (threshold, finer, coarser) in boundaries {
        if !is_finite(threshold) {
            continue;
        }
        let distance = (cov - threshold).abs();
        if distance >= band {
            continue;
        }
        let blend = ((threshold + band - cov) / (2.0 * band)).clamp(0.0, 1.0);
        let is_closer = match best {
            Some((best_distance, ..)) => distance < best_distance,
            None => true,
        };
        if is_closer {
            best = Some((distance, finer, coarser, blend));
        }
    }

    match best {
        Some((_, finer, coarser, blend)) => {
            let from = coarser_of(finer, native);
            let to = coarser_of(coarser, native);
            if from == to {
                settled(from)
            } else {
                (from, to, blend, 1, 1)
            }
        }
        None => settled(hard_tier),
    }
}

/// Asserts GPU-vs-oracle parity for a single query.
fn assert_parity(ctx: &GpuContext, gpu: &GpuHairLodTransition, q: HairLodTransitionQuery) {
    let results = gpu.evaluate(ctx, &[q]);
    assert_eq!(results.len(), 1, "one result per query");
    let r = results[0];
    let (from, to, blend, xfade, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert_eq!(r.from_tier, from, "from_tier mismatch: query={q:?}");
    assert_eq!(r.to_tier, to, "to_tier mismatch: query={q:?}");
    assert_eq!(
        r.is_cross_fading, xfade,
        "is_cross_fading mismatch: query={q:?}"
    );
    if valid == 1 {
        assert!(
            close(r.blend, blend),
            "blend mismatch: gpu={} cpu={blend} query={q:?}",
            r.blend
        );
    }
}

/// Descending thresholds reused across the named fixtures.
const RSB: f32 = 0.6;
const CB: f32 = 0.3;
const MB: f32 = 0.1;

#[test]
fn disabled_band_settles() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLodTransition::new(&ctx);
    // band == 0 disables fading: a hard pick at Strands for high coverage.
    assert_parity(
        &ctx,
        &gpu,
        HairLodTransitionQuery::new(0.8, RSB, CB, MB, 0.0, 0),
    );
}

#[test]
fn inside_band_cross_fades() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLodTransition::new(&ctx);
    // Coverage 0.62 is 0.02 inside the 0.05 band around the 0.6 threshold:
    // Strands fades to ReducedStrands with blend 0.3.
    assert_parity(
        &ctx,
        &gpu,
        HairLodTransitionQuery::new(0.62, RSB, CB, MB, 0.05, 0),
    );
}

#[test]
fn native_form_collapse_settles() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLodTransition::new(&ctx);
    // A Cards-authored groom (native = 2) in the Strands/ReducedStrands band:
    // both tiers clamp up to Cards, so the fade collapses to a settle.
    assert_parity(
        &ctx,
        &gpu,
        HairLodTransitionQuery::new(0.62, RSB, CB, MB, 0.05, 2),
    );
}

#[test]
fn equidistant_bands_pick_first_seen() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLodTransition::new(&ctx);
    // Coverage 0.45 is 0.15 from both the 0.6 and 0.3 thresholds; with a wide
    // 0.2 band both bands contain it at equal distance, so the first-seen
    // (reduced-strands) boundary must win.
    assert_parity(
        &ctx,
        &gpu,
        HairLodTransitionQuery::new(0.45, RSB, CB, MB, 0.2, 0),
    );
}

#[test]
fn out_of_range_native_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLodTransition::new(&ctx);
    // native_form == 4 has no reference tier: valid = 0, outputs cleared.
    assert_parity(
        &ctx,
        &gpu,
        HairLodTransitionQuery::new(0.5, RSB, CB, MB, 0.05, 4),
    );
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLodTransition::new(&ctx);
    // A multi-element batch exercises the std430 stride: adjacent slots must
    // decode independently and in order.
    let queries = [
        HairLodTransitionQuery::new(0.8, RSB, CB, MB, 0.0, 0),
        HairLodTransitionQuery::new(0.62, RSB, CB, MB, 0.05, 0),
        HairLodTransitionQuery::new(0.62, RSB, CB, MB, 0.05, 2),
        HairLodTransitionQuery::new(0.45, RSB, CB, MB, 0.2, 0),
        HairLodTransitionQuery::new(0.5, RSB, CB, MB, 0.05, 4),
        HairLodTransitionQuery::new(0.2, RSB, CB, MB, 0.03, 1),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (from, to, blend, xfade, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert_eq!(r.from_tier, from, "batch from_tier mismatch: query={q:?}");
        assert_eq!(r.to_tier, to, "batch to_tier mismatch: query={q:?}");
        assert_eq!(
            r.is_cross_fading, xfade,
            "batch is_cross_fading mismatch: query={q:?}"
        );
        if valid == 1 {
            assert!(
                close(r.blend, blend),
                "batch blend mismatch: gpu={} cpu={blend} query={q:?}",
                r.blend
            );
        }
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLodTransition::new(&ctx);
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
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairLodTransition::new(&ctx);
    let mut rng = Lcg::new(0x5E_3A_C1_07);
    // Margin kept well above the 1e-4 tolerance so f32 noise never flips a
    // branch at a threshold +- band knee or an equidistant-threshold tie.
    const KNEE_MARGIN: f32 = 0.01;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let cov = rng.next_range(0.0, 1.0);
        let band = rng.next_range(0.02, 0.08);
        let thresholds = [RSB, CB, MB];
        // Reject coverage sitting on a branch/clamp knee of any threshold.
        let on_knee = thresholds
            .iter()
            .any(|&t| ((cov - t).abs() - band).abs() < KNEE_MARGIN);
        if on_knee {
            continue;
        }
        // Reject near-equidistant pairs where both bands could contain the
        // coverage, so the first-seen tie-break is unambiguous.
        let d0 = (cov - RSB).abs();
        let d1 = (cov - CB).abs();
        let d2 = (cov - MB).abs();
        let tie = ((d0 - d1).abs() < KNEE_MARGIN && d0 < band && d1 < band)
            || ((d1 - d2).abs() < KNEE_MARGIN && d1 < band && d2 < band)
            || ((d0 - d2).abs() < KNEE_MARGIN && d0 < band && d2 < band);
        if tie {
            continue;
        }
        let native = rng.next_u32() % 4;
        queries.push(HairLodTransitionQuery::new(cov, RSB, CB, MB, band, native));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (from, to, blend, xfade, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert_eq!(r.from_tier, from, "sweep from_tier mismatch: query={q:?}");
        assert_eq!(r.to_tier, to, "sweep to_tier mismatch: query={q:?}");
        assert_eq!(
            r.is_cross_fading, xfade,
            "sweep is_cross_fading mismatch: query={q:?}"
        );
        if valid == 1 {
            assert!(
                close(r.blend, blend),
                "sweep blend mismatch: gpu={} cpu={blend} query={q:?}",
                r.blend
            );
        }
    }
}
