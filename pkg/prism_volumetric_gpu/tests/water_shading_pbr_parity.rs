//! Real-device parity for the water-`PBR` shading planner twin:
//! [`GpuWaterShadingPbr`](prism_volumetric_gpu::water_shading_pbr::GpuWaterShadingPbr)
//! must reproduce the per-sample response of the `CPU` golden
//! [`plan_pbr`](prism_render_architecture::water::shading::pbr::plan_pbr) — the
//! base reflectance `F0`, the `Schlick` `Fresnel` term, the `SSR` -> `RT` ->
//! probe reflection tier, the `Jacobian` foam mask, the foam-roughened `GGX`
//! roughness and the grazing subsurface transmission — across fixed fixtures
//! and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! Each expected value is produced by calling the public golden
//! [`plan_pbr`](prism_render_architecture::water::shading::pbr::plan_pbr) with
//! the same inputs, reconstructed into its
//! [`PbrShadingParams`](prism_render_architecture::water::shading::PbrShadingParams)
//! and
//! [`SurfaceShadingInputs`](prism_render_architecture::water::shading::SurfaceShadingInputs)
//! argument structs.
//!
//! # Parity criterion
//!
//! The five continuous fields agree within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (relative floor `1e-6`); the reflection-tier code is a
//! chain of ordered comparisons and is asserted with `==`. The randomized sweep
//! keeps each sample away from the tier-selection and branch thresholds so a
//! last-place rounding difference cannot flip a discrete decision.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::shading::pbr::plan_pbr`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::shading::pbr::plan_pbr;
use prism_render_architecture::water::shading::{
    PbrShadingParams, ReflectionTier, SurfaceShadingInputs,
};
use prism_render_architecture::water::underwater::RgbColor;
use prism_volumetric_gpu::water_shading_pbr::{
    GpuWaterShadingPbr, WaterShadingPbrQuery, WaterShadingPbrResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute agreement bound for a continuous field.
const EPS: f32 = 1.0e-4;

/// Relative agreement bound for a continuous field.
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

/// Maps a golden [`ReflectionTier`] to its stable code, matching the
/// discriminant order the twin documents.
fn tier_code(tier: ReflectionTier) -> u32 {
    match tier {
        ReflectionTier::ScreenSpace => 0,
        ReflectionTier::RayTraced => 1,
        ReflectionTier::Probe => 2,
    }
}

/// Rebuilds the golden argument structs from a flat query and calls the public
/// oracle, returning the expected per-field response.
fn oracle(q: &WaterShadingPbrQuery) -> WaterShadingPbrResult {
    let params = PbrShadingParams {
        f0_override: q.f0_override,
        foam_fold_threshold: q.foam_fold_threshold,
        base_roughness: q.base_roughness,
        grazing_transmission: q.grazing_transmission,
        ssr_min_confidence: q.ssr_min_confidence,
        rt_min_budget: q.rt_min_budget,
    };
    let inputs = SurfaceShadingInputs {
        cos_view: q.cos_view,
        ior: q.ior,
        jacobian: q.jacobian,
        water_color: RgbColor {
            r: 0.1,
            g: 0.3,
            b: 0.5,
        },
        specular_intensity: 0.8,
        caustic_intensity: 0.6,
        flow_speed: 1.2,
        depth: 1.0,
        dist_to_shore: 1.0,
        ssr_confidence: q.ssr_confidence,
        ray_budget: q.ray_budget,
    };
    let r = plan_pbr(params, inputs);
    WaterShadingPbrResult {
        f0: r.f0,
        fresnel: r.fresnel,
        foam_mask: r.foam_mask,
        specular_roughness: r.specular_roughness,
        subsurface_transmission: r.subsurface_transmission,
        reflection_tier: tier_code(r.reflection_tier),
    }
}

/// Pins one `GPU` sample result against the in-host oracle: the tier code
/// exactly, every continuous field within tolerance.
fn check_sample(idx: usize, got: &WaterShadingPbrResult, want: &WaterShadingPbrResult) {
    assert_eq!(
        got.reflection_tier, want.reflection_tier,
        "sample {idx} reflection_tier: gpu {} vs cpu {}",
        got.reflection_tier, want.reflection_tier
    );
    assert!(
        close(got.f0, want.f0),
        "sample {idx} f0: gpu {} vs cpu {}",
        got.f0,
        want.f0
    );
    assert!(
        close(got.fresnel, want.fresnel),
        "sample {idx} fresnel: gpu {} vs cpu {}",
        got.fresnel,
        want.fresnel
    );
    assert!(
        close(got.foam_mask, want.foam_mask),
        "sample {idx} foam_mask: gpu {} vs cpu {}",
        got.foam_mask,
        want.foam_mask
    );
    assert!(
        close(got.specular_roughness, want.specular_roughness),
        "sample {idx} specular_roughness: gpu {} vs cpu {}",
        got.specular_roughness,
        want.specular_roughness
    );
    assert!(
        close(got.subsurface_transmission, want.subsurface_transmission),
        "sample {idx} subsurface_transmission: gpu {} vs cpu {}",
        got.subsurface_transmission,
        want.subsurface_transmission
    );
}

/// Dispatches every sample and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterShadingPbr, queries: &[WaterShadingPbrQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_sample(idx, result, &want);
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

/// Draws a `f32` in `[lo, hi]` at milli resolution from `state`.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let t = (lcg(state) % 1_000_001) as f32 / 1_000_000.0;
    lo + (hi - lo) * t
}

/// The clear open-water tuning used as the fixed fixture base.
fn base_query() -> WaterShadingPbrQuery {
    WaterShadingPbrQuery {
        f0_override: 0.0,
        ior: 1.33,
        cos_view: 0.5,
        jacobian: 0.4,
        foam_fold_threshold: 1.0,
        base_roughness: 0.08,
        grazing_transmission: 0.6,
        ssr_confidence: 0.7,
        ssr_min_confidence: 0.5,
        ray_budget: 0.5,
        rt_min_budget: 0.25,
    }
}

#[test]
fn fixed_fixtures_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterShadingPbr::new(&ctx);

    let mut queries = Vec::new();

    // Canonical clear water, SSR tier.
    queries.push(base_query());

    // Head-on view (cos = 1): Fresnel collapses to F0, no subsurface.
    let mut head_on = base_query();
    head_on.cos_view = 1.0;
    queries.push(head_on);

    // Grazing view (cos = 0): Fresnel rises to 1, maximal subsurface.
    let mut grazing = base_query();
    grazing.cos_view = 0.0;
    queries.push(grazing);

    // Explicit F0 override well above the epsilon bypasses the IOR relation.
    let mut overridden = base_query();
    overridden.f0_override = 0.25;
    queries.push(overridden);

    // SSR misses but the RT budget is available: RayTraced tier.
    let mut rt = base_query();
    rt.ssr_confidence = 0.1;
    rt.ray_budget = 0.9;
    queries.push(rt);

    // SSR and RT both miss: Probe tier.
    let mut probe = base_query();
    probe.ssr_confidence = 0.1;
    probe.ray_budget = 0.1;
    queries.push(probe);

    // Fully folded Jacobian: foam mask saturates and roughens the surface.
    let mut folded = base_query();
    folded.jacobian = 0.0;
    queries.push(folded);

    // Flat surface (Jacobian at the threshold): foam mask is zero.
    let mut flat = base_query();
    flat.jacobian = 1.0;
    queries.push(flat);

    check(&ctx, &gpu, &queries);
}

#[test]
fn randomized_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterShadingPbr::new(&ctx);

    let mut state = 0x5eed_1234_abcd_f00du64;
    let mut queries = Vec::new();
    while queries.len() < 4096 {
        // Thresholds first, then the deciding quantities placed a clear margin
        // off each threshold so a last-place rounding difference cannot flip the
        // reflection tier.
        let ssr_min_confidence = uniform(&mut state, 0.2, 0.8);
        let rt_min_budget = uniform(&mut state, 0.2, 0.8);

        // ssr_confidence lands either clearly above or clearly below its gate.
        let ssr_above = lcg(&mut state) & 1 == 0;
        let ssr_confidence = if ssr_above {
            (ssr_min_confidence + uniform(&mut state, 0.05, 0.2)).min(1.0)
        } else {
            (ssr_min_confidence - uniform(&mut state, 0.05, 0.2)).max(0.0)
        };
        // ray_budget likewise sits a clear margin off its gate.
        let rt_above = lcg(&mut state) & 1 == 0;
        let ray_budget = if rt_above {
            (rt_min_budget + uniform(&mut state, 0.05, 0.2)).min(1.0)
        } else {
            (rt_min_budget - uniform(&mut state, 0.05, 0.2)).max(0.0)
        };

        // f0_override is either an exact zero (derive from IOR) or a value a
        // clear margin above the epsilon branch, never straddling it.
        let f0_override = if lcg(&mut state) & 1 == 0 {
            0.0
        } else {
            uniform(&mut state, 0.02, 0.4)
        };

        let q = WaterShadingPbrQuery {
            f0_override,
            ior: uniform(&mut state, 1.05, 2.5),
            cos_view: uniform(&mut state, 0.0, 1.0),
            jacobian: uniform(&mut state, 0.0, 2.0),
            // foam_fold_threshold stays a clear margin above the epsilon branch.
            foam_fold_threshold: uniform(&mut state, 0.5, 2.0),
            base_roughness: uniform(&mut state, 0.0, 1.0),
            grazing_transmission: uniform(&mut state, 0.0, 1.0),
            ssr_confidence,
            ssr_min_confidence,
            ray_budget,
            rt_min_budget,
        };
        queries.push(q);
    }

    check(&ctx, &gpu, &queries);
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterShadingPbr::new(&ctx);
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
