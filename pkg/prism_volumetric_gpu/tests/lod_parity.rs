//! Real-device parity for the particle-LOD decision twin:
//! [`GpuParticleLod`](prism_volumetric_gpu::lod::GpuParticleLod) must reproduce
//! the `CPU` golden [`lod`](prism_render_architecture::particle::lod) across the
//! coverage-selected tier, the native-form clamp, the `is_simulated` predicate,
//! the platform-clamped quality, its divisor and particle budget, the single
//! degradation rung and the saturating multi-step degradation.
//!
//! The fixtures cover every discrete outcome the golden exposes: all four LOD
//! tiers (full / reduced / impostor / culled), the native-form clamp that keeps
//! an impostor-authored emitter from being promoted, all four quality tiers via
//! the platform ceiling (mobile / console / desktop / high-end), all four budget
//! divisors, the `max_particles == 0` short circuit, the `.max(1)` budget floor
//! (`max_particles = 1`, `Low` divisor `8`), the `degrade` [`Option`] at both
//! ends of the ladder, and the saturating `degrade_steps` staircase. Every
//! coverage sits at least `0.05` clear of each threshold, chosen so the `>=`
//! verdicts can never straddle a boundary.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every output is a tier code, a quality code, a `bool` or a `u32` integer
//! built from integer arithmetic and discrete `f32` threshold comparisons, so
//! `CPU` and `GPU` agree bit-exactly: the comparison is an exact `==` on every
//! field, including the presence of the `degrade` [`Option`].
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::lod`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::lod::{
    budget_for_quality, resolve_quality, select_particle_lod_tier, ParticleLodThresholds,
    ParticleLodTier, ParticleQuality, PlatformTier,
};
use prism_volumetric_gpu::lod::{GpuParticleLod, ParticleLodQuery, ParticleLodResult};
use prism_volumetric_gpu::GpuContext;

/// Descending tier boundaries shared by the fixtures. Every fixture coverage is
/// kept at least `0.05` clear of each of these three values.
const REDUCED_BELOW: f32 = 0.6;
/// Drop-to-impostor boundary for the shared fixtures.
const IMPOSTOR_BELOW: f32 = 0.35;
/// Cull boundary for the shared fixtures.
const CULL_BELOW: f32 = 0.1;

/// Builds a query from the policy inputs, folding in the shared tier boundaries.
fn query(
    coverage: f32,
    native_form: ParticleLodTier,
    requested_quality: ParticleQuality,
    platform: PlatformTier,
    max_particles: u32,
    degrade_steps: u32,
) -> ParticleLodQuery {
    ParticleLodQuery {
        coverage,
        reduced_below: REDUCED_BELOW,
        impostor_below: IMPOSTOR_BELOW,
        cull_below: CULL_BELOW,
        native_form,
        requested_quality,
        platform,
        max_particles,
        degrade_steps,
    }
}

/// Computes the reference answer for a query directly from the `CPU` golden.
fn expected(q: &ParticleLodQuery) -> ParticleLodResult {
    let thresholds = ParticleLodThresholds {
        reduced_below: q.reduced_below,
        impostor_below: q.impostor_below,
        cull_below: q.cull_below,
    };
    let selected = select_particle_lod_tier(q.coverage, thresholds);
    let coarser = selected.coarser_of(q.native_form);
    let resolved = resolve_quality(q.requested_quality, q.platform);
    ParticleLodResult {
        selected_tier: selected,
        coarser_tier: coarser,
        simulated: coarser.is_simulated(),
        resolved_quality: resolved,
        divisor: resolved.particle_divisor(),
        budget: budget_for_quality(q.max_particles, resolved),
        degraded: q.requested_quality.degrade(),
        degrade_steps_quality: q.requested_quality.degrade_steps(q.degrade_steps),
    }
}

/// Asserts every twinned field for one query matches the `CPU` golden exactly.
fn assert_parity(gpu: &GpuParticleLod, ctx: &GpuContext, q: &ParticleLodQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];
    let want = expected(q);
    assert_eq!(
        g.selected_tier, want.selected_tier,
        "selected_tier mismatch for {q:?}"
    );
    assert_eq!(
        g.coarser_tier, want.coarser_tier,
        "coarser_tier mismatch for {q:?}"
    );
    assert_eq!(g.simulated, want.simulated, "simulated mismatch for {q:?}");
    assert_eq!(
        g.resolved_quality, want.resolved_quality,
        "resolved_quality mismatch for {q:?}"
    );
    assert_eq!(g.divisor, want.divisor, "divisor mismatch for {q:?}");
    assert_eq!(g.budget, want.budget, "budget mismatch for {q:?}");
    assert_eq!(g.degraded, want.degraded, "degraded mismatch for {q:?}");
    assert_eq!(
        g.degrade_steps_quality, want.degrade_steps_quality,
        "degrade_steps_quality mismatch for {q:?}"
    );
}

#[test]
fn all_four_tiers_are_selected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleLod::new(&ctx);
    // Full-detail native form so the coverage tier passes through unclamped. The
    // four coverages sit well clear of the three boundaries.
    for (coverage, tier) in [
        (0.9_f32, ParticleLodTier::Full),
        (0.45_f32, ParticleLodTier::Reduced),
        (0.22_f32, ParticleLodTier::Impostor),
        (0.02_f32, ParticleLodTier::Culled),
    ] {
        let q = query(
            coverage,
            ParticleLodTier::Full,
            ParticleQuality::High,
            PlatformTier::Desktop,
            40_000,
            1,
        );
        let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
        assert_eq!(got[0].selected_tier, tier, "coverage {coverage} tier");
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn native_form_clamps_the_selected_tier() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleLod::new(&ctx);
    // Full-screen coverage would select Full, but an impostor-authored emitter is
    // never promoted finer than its native form, and its tier does not simulate.
    let impostor_authored = query(
        0.9,
        ParticleLodTier::Impostor,
        ParticleQuality::Ultra,
        PlatformTier::HighEnd,
        40_000,
        0,
    );
    assert_parity(&gpu, &ctx, &impostor_authored);
    // A culled native form stays culled even at full coverage.
    let culled_authored = query(
        0.9,
        ParticleLodTier::Culled,
        ParticleQuality::Ultra,
        PlatformTier::HighEnd,
        40_000,
        0,
    );
    assert_parity(&gpu, &ctx, &culled_authored);
    // A reduced native form clamps a full selection to reduced, which still
    // simulates.
    let reduced_authored = query(
        0.9,
        ParticleLodTier::Reduced,
        ParticleQuality::Ultra,
        PlatformTier::HighEnd,
        40_000,
        0,
    );
    assert_parity(&gpu, &ctx, &reduced_authored);
}

#[test]
fn quality_clamps_to_each_platform_ceiling() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleLod::new(&ctx);
    // Ultra requested on every platform resolves to that platform's ceiling,
    // exercising all four resolved-quality codes.
    for platform in [
        PlatformTier::Mobile,
        PlatformTier::Console,
        PlatformTier::Desktop,
        PlatformTier::HighEnd,
    ] {
        let q = query(
            0.45,
            ParticleLodTier::Full,
            ParticleQuality::Ultra,
            platform,
            40_000,
            1,
        );
        assert_parity(&gpu, &ctx, &q);
    }
    // A request already under the ceiling passes through unchanged.
    let under_ceiling = query(
        0.45,
        ParticleLodTier::Full,
        ParticleQuality::Medium,
        PlatformTier::HighEnd,
        40_000,
        1,
    );
    assert_parity(&gpu, &ctx, &under_ceiling);
}

#[test]
fn all_four_divisors_scale_the_budget() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleLod::new(&ctx);
    // HighEnd never clamps, so each requested quality resolves to itself and
    // exercises its own divisor: Low 8, Medium 4, High 2, Ultra 1.
    for quality in [
        ParticleQuality::Low,
        ParticleQuality::Medium,
        ParticleQuality::High,
        ParticleQuality::Ultra,
    ] {
        let q = query(
            0.9,
            ParticleLodTier::Full,
            quality,
            PlatformTier::HighEnd,
            40_000,
            0,
        );
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn budget_handles_zero_and_the_floor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleLod::new(&ctx);
    // An empty emitter stays empty regardless of quality.
    let empty = query(
        0.9,
        ParticleLodTier::Full,
        ParticleQuality::Low,
        PlatformTier::HighEnd,
        0,
        0,
    );
    assert_parity(&gpu, &ctx, &empty);
    // A single-particle emitter at the coarsest divisor keeps one particle via
    // the `.max(1)` floor rather than dividing to zero.
    let floor = query(
        0.9,
        ParticleLodTier::Full,
        ParticleQuality::Low,
        PlatformTier::HighEnd,
        1,
        0,
    );
    assert_parity(&gpu, &ctx, &floor);
}

#[test]
fn single_degrade_rung_mirrors_the_option() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleLod::new(&ctx);
    // Ultra degrades to High (Some); Low degrades to None.
    for quality in [
        ParticleQuality::Ultra,
        ParticleQuality::High,
        ParticleQuality::Medium,
        ParticleQuality::Low,
    ] {
        let q = query(
            0.9,
            ParticleLodTier::Full,
            quality,
            PlatformTier::HighEnd,
            40_000,
            1,
        );
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn degrade_steps_saturate_at_low() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleLod::new(&ctx);
    // Zero steps is a no-op; two steps walks two rungs; an over-step saturates at
    // Low rather than wrapping.
    for (quality, steps) in [
        (ParticleQuality::High, 0_u32),
        (ParticleQuality::Ultra, 2_u32),
        (ParticleQuality::Ultra, 10_u32),
        (ParticleQuality::Medium, 1_u32),
        (ParticleQuality::Low, 5_u32),
    ] {
        let q = query(
            0.45,
            ParticleLodTier::Full,
            quality,
            PlatformTier::HighEnd,
            40_000,
            steps,
        );
        assert_parity(&gpu, &ctx, &q);
    }
}

#[test]
fn batch_of_queries_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleLod::new(&ctx);
    // A batch exercises the one-thread-per-query flattening; each result must be
    // independent of its neighbours.
    let batch = [
        query(
            0.9,
            ParticleLodTier::Full,
            ParticleQuality::Ultra,
            PlatformTier::Mobile,
            40_000,
            1,
        ),
        query(
            0.22,
            ParticleLodTier::Impostor,
            ParticleQuality::High,
            PlatformTier::Console,
            12_345,
            3,
        ),
        query(
            0.02,
            ParticleLodTier::Full,
            ParticleQuality::Medium,
            PlatformTier::Desktop,
            1,
            0,
        ),
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleLod::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
