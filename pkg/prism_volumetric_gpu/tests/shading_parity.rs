//! Real-device parity for the particle shading-router twin:
//! [`GpuShading`](prism_volumetric_gpu::shading::GpuShading) must reproduce the
//! `CPU` golden
//! [`render_phase_for`](prism_render_architecture::particle::shading::render_phase_for),
//! [`attribute_footprint`](prism_render_architecture::particle::shading::attribute_footprint),
//! [`lighting_services`](prism_render_architecture::particle::shading::lighting_services),
//! [`resolve_oit_route`](prism_render_architecture::particle::shading::resolve_oit_route),
//! [`motion_vector_request`](prism_render_architecture::particle::shading::motion_vector_request),
//! [`six_way_response`](prism_render_architecture::particle::shading::six_way_response),
//! [`quantize_cel_bands`](prism_render_architecture::particle::shading::quantize_cel_bands),
//! [`resolve_deep_shadow`](prism_render_architecture::particle::shading::resolve_deep_shadow)
//! and
//! [`resolve_shading_program`](prism_render_architecture::particle::shading::resolve_shading_program)
//! across an empty batch, each blend mode's render phase, each shading model's
//! attribute footprint and lighting subscription, every lighting-capability
//! combination, all four `OIT` routes, the motion-vector decision for culled /
//! opaque / transparent / fast-flipbook renderers, the six-way directional
//! response for single-axis and blended light directions, the cel-band
//! quantization ladder, the deep-shadow quality ladder, the composite program,
//! and a large pseudo-random batch compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every classification, bitfield and boolean lane is a deterministic branch on
//! integer inputs, so `CPU` and `GPU` agree *exactly* and those lanes are
//! compared with `==`. The two continuous lanes — the six-way response and the
//! cel quantization — plus the reactive-mask weight and the isotropic phase
//! parameters are closed-form multiply/add/divide with no transcendental call,
//! so they are compared within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The named
//! fixtures and the random batch are both placed clear of the cel floor's
//! integer boundaries and the motion-vector instability threshold, so no
//! discrete lane sits on a tie that a legal `ULP` perturbation could flip.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::shading`；pure
//! classification/bit-union plus directional algebra and a `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use prism_render_architecture::particle::lod::ParticleQuality;
use prism_render_architecture::particle::shading::{
    DeepShadowMode, LightingServiceCaps, MotionVectorInput, OitRoute, ParticleRenderPhase,
    ShadingAttributeFootprint, SixWayLuminance,
};
use prism_render_architecture::particle::sort_cull::{BlendMode, SortDecision};
use prism_render_architecture::particle::{EmberShadingModel, ShadingBasis, SortStrategy, Vec3};
use prism_volumetric_gpu::shading::{cpu_reference, GpuShading, GpuShadingQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the continuous lanes. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
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

/// All lighting capabilities enabled.
fn all_caps() -> LightingServiceCaps {
    LightingServiceCaps {
        shadow_maps: true,
        global_illumination: true,
        ray_tracing: true,
    }
}

/// A neutral baseline query every fixture tweaks: an unlit opaque renderer with
/// no lighting services, an order-independent sort, a visible opaque motion
/// decision, a zero six-way rig, an axis-aligned light, and a mid-band cel
/// request placed clear of the floor's integer boundaries.
fn base_query() -> GpuShadingQuery {
    GpuShadingQuery {
        model: EmberShadingModel::Unlit,
        blend: BlendMode::Opaque,
        caps: LightingServiceCaps::default(),
        volumetric: false,
        quality: ParticleQuality::Low,
        footprint_a: ShadingAttributeFootprint::default(),
        footprint_b: ShadingAttributeFootprint::default(),
        oit: SortDecision {
            blend: BlendMode::Opaque,
            particle_count: 0,
            radix_min_count: 1_024,
            prefer_shared_oit: false,
        },
        motion: MotionVectorInput {
            visible: true,
            blend: BlendMode::Opaque,
            flipbook_rate: 0.0,
            flipbook_unstable_rate: 30.0,
        },
        lum: SixWayLuminance {
            right: 0.0,
            left: 0.0,
            up: 0.0,
            down: 0.0,
            front: 0.0,
            back: 0.0,
        },
        light_dir: Vec3::new(1.0, 0.0, 0.0),
        cel_response: 0.0,
        cel_bands: 1,
    }
}

/// Asserts strict lane-for-lane parity of one `GPU` result against the `CPU`
/// golden: every discrete classification, bitfield and boolean lane matches
/// exactly, and the continuous lanes (six-way response, cel quantization,
/// reactive mask, isotropic phase parameters) match within tolerance.
fn check(ctx: &GpuContext, gpu: &GpuShading, queries: &[GpuShadingQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = cpu_reference(q);

        assert_eq!(g.phase, want.phase, "lane {lane}: phase");
        assert_eq!(
            g.footprint_model, want.footprint_model,
            "lane {lane}: footprint_model"
        );
        assert_eq!(
            g.footprint_union, want.footprint_union,
            "lane {lane}: footprint_union"
        );
        assert_eq!(g.lighting, want.lighting, "lane {lane}: lighting");
        assert_eq!(
            g.needs_lighting, want.needs_lighting,
            "lane {lane}: needs_lighting"
        );
        assert_eq!(g.oit_route, want.oit_route, "lane {lane}: oit_route");
        assert_eq!(
            g.motion.write_motion_vectors, want.motion.write_motion_vectors,
            "lane {lane}: write_motion_vectors"
        );
        assert_eq!(
            g.motion.temporally_unstable, want.motion.temporally_unstable,
            "lane {lane}: temporally_unstable"
        );
        assert!(
            close(g.motion.reactive_mask, want.motion.reactive_mask),
            "lane {lane}: reactive_mask gpu {} vs cpu {}",
            g.motion.reactive_mask,
            want.motion.reactive_mask
        );
        assert!(
            close(g.six_way_response, want.six_way_response),
            "lane {lane}: six_way_response gpu {} vs cpu {}",
            g.six_way_response,
            want.six_way_response
        );
        assert!(
            close(g.phase_params.g, want.phase_params.g),
            "lane {lane}: phase_params.g"
        );
        assert!(
            close(
                g.phase_params.back_lobe_weight,
                want.phase_params.back_lobe_weight
            ),
            "lane {lane}: phase_params.back_lobe_weight"
        );
        assert!(
            close(g.phase_params.back_g, want.phase_params.back_g),
            "lane {lane}: phase_params.back_g"
        );
        assert!(
            close(g.cel_quantized, want.cel_quantized),
            "lane {lane}: cel_quantized gpu {} vs cpu {}",
            g.cel_quantized,
            want.cel_quantized
        );
        assert_eq!(g.deep_shadow, want.deep_shadow, "lane {lane}: deep_shadow");
        assert_eq!(g.program, want.program, "lane {lane}: program");
    }
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Draws one of four blend modes from the generator.
fn pick_blend(state: &mut u64) -> BlendMode {
    match (lcg(state) * 4.0) as u32 {
        0 => BlendMode::Opaque,
        1 => BlendMode::Additive,
        2 => BlendMode::Premultiplied,
        _ => BlendMode::AlphaBlend,
    }
}

/// Draws one basis for a hybrid lobe (custom carries an arbitrary stable id).
fn pick_basis(state: &mut u64) -> ShadingBasis {
    match (lcg(state) * 4.0) as u32 {
        0 => ShadingBasis::Unlit,
        1 => ShadingBasis::Pbr,
        2 => ShadingBasis::Npr,
        _ => ShadingBasis::Custom(7),
    }
}

/// Draws one of five shading models from the generator.
fn pick_model(state: &mut u64) -> EmberShadingModel {
    match (lcg(state) * 5.0) as u32 {
        0 => EmberShadingModel::Unlit,
        1 => EmberShadingModel::Pbr,
        2 => EmberShadingModel::Npr,
        3 => EmberShadingModel::Custom(3),
        _ => EmberShadingModel::Hybrid {
            base: pick_basis(state),
            overlay: pick_basis(state),
            weight: lcg(state),
        },
    }
}

/// Draws one of four quality tiers from the generator.
fn pick_quality(state: &mut u64) -> ParticleQuality {
    match (lcg(state) * 4.0) as u32 {
        0 => ParticleQuality::Low,
        1 => ParticleQuality::Medium,
        2 => ParticleQuality::High,
        _ => ParticleQuality::Ultra,
    }
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuShading::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn render_phase_maps_every_blend() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuShading::new(&ctx);
    let blends = [
        BlendMode::Opaque,
        BlendMode::Additive,
        BlendMode::Premultiplied,
        BlendMode::AlphaBlend,
    ];
    let queries: Vec<GpuShadingQuery> = blends
        .iter()
        .map(|&blend| GpuShadingQuery {
            blend,
            motion: MotionVectorInput {
                blend,
                ..base_query().motion
            },
            oit: SortDecision {
                blend,
                ..base_query().oit
            },
            ..base_query()
        })
        .collect();
    check(&ctx, &gpu, &queries);
    // Spot-check the two phase classes directly.
    let got = gpu.eval(&ctx, &queries);
    assert_eq!(got[0].phase, ParticleRenderPhase::Opaque);
    assert_eq!(got[3].phase, ParticleRenderPhase::Transparent);
}

#[test]
fn footprint_and_lighting_per_model() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuShading::new(&ctx);
    let models = [
        EmberShadingModel::Unlit,
        EmberShadingModel::Pbr,
        EmberShadingModel::Npr,
        EmberShadingModel::Custom(42),
        EmberShadingModel::Hybrid {
            base: ShadingBasis::Pbr,
            overlay: ShadingBasis::Npr,
            weight: 0.5,
        },
        EmberShadingModel::Hybrid {
            base: ShadingBasis::Unlit,
            overlay: ShadingBasis::Unlit,
            weight: 0.25,
        },
        EmberShadingModel::Hybrid {
            base: ShadingBasis::Unlit,
            overlay: ShadingBasis::Custom(1),
            weight: 0.75,
        },
    ];
    let queries: Vec<GpuShadingQuery> = models
        .iter()
        .map(|&model| GpuShadingQuery {
            model,
            caps: all_caps(),
            ..base_query()
        })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn footprint_union_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuShading::new(&ctx);
    let pbr = ShadingAttributeFootprint {
        normal: true,
        tangent: true,
        material_params: true,
        ..ShadingAttributeFootprint::default()
    };
    let npr = ShadingAttributeFootprint {
        normal: true,
        ramp_lut: true,
        ..ShadingAttributeFootprint::default()
    };
    let custom = ShadingAttributeFootprint {
        custom_params: true,
        ..ShadingAttributeFootprint::default()
    };
    let queries = vec![
        GpuShadingQuery {
            footprint_a: pbr,
            footprint_b: npr,
            ..base_query()
        },
        GpuShadingQuery {
            footprint_a: ShadingAttributeFootprint::default(),
            footprint_b: custom,
            ..base_query()
        },
        GpuShadingQuery {
            footprint_a: pbr,
            footprint_b: ShadingAttributeFootprint::default(),
            ..base_query()
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn lighting_services_over_capability_combinations() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuShading::new(&ctx);
    let mut queries = Vec::new();
    // All eight capability combinations, once lit (Pbr) and once unlit.
    for mask in 0_u32..8 {
        let caps = LightingServiceCaps {
            shadow_maps: mask & 1 != 0,
            global_illumination: mask & 2 != 0,
            ray_tracing: mask & 4 != 0,
        };
        queries.push(GpuShadingQuery {
            model: EmberShadingModel::Pbr,
            caps,
            ..base_query()
        });
        queries.push(GpuShadingQuery {
            model: EmberShadingModel::Unlit,
            caps,
            ..base_query()
        });
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn oit_route_covers_every_branch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuShading::new(&ctx);
    let radix_min = 256_u32;
    let decisions = [
        // Order-independent: opaque, additive, premultiplied — no sort.
        SortDecision {
            blend: BlendMode::Opaque,
            particle_count: 10_000,
            radix_min_count: radix_min,
            prefer_shared_oit: false,
        },
        SortDecision {
            blend: BlendMode::Additive,
            particle_count: 10_000,
            radix_min_count: radix_min,
            prefer_shared_oit: false,
        },
        // Alpha blend with a single particle: still order-independent (count<=1).
        SortDecision {
            blend: BlendMode::AlphaBlend,
            particle_count: 1,
            radix_min_count: radix_min,
            prefer_shared_oit: false,
        },
        // Shared OIT preferred.
        SortDecision {
            blend: BlendMode::AlphaBlend,
            particle_count: 5_000,
            radix_min_count: radix_min,
            prefer_shared_oit: true,
        },
        // Standalone radix: large count, no shared OIT.
        SortDecision {
            blend: BlendMode::AlphaBlend,
            particle_count: 5_000,
            radix_min_count: radix_min,
            prefer_shared_oit: false,
        },
        // Standalone bitonic: small count (but > 1), no shared OIT.
        SortDecision {
            blend: BlendMode::AlphaBlend,
            particle_count: 32,
            radix_min_count: radix_min,
            prefer_shared_oit: false,
        },
    ];
    let queries: Vec<GpuShadingQuery> = decisions
        .iter()
        .map(|&oit| GpuShadingQuery {
            blend: oit.blend,
            oit,
            ..base_query()
        })
        .collect();
    check(&ctx, &gpu, &queries);
    // Spot-check each distinct route reached.
    let got = gpu.eval(&ctx, &queries);
    assert_eq!(got[0].oit_route, OitRoute::OrderIndependent);
    assert_eq!(got[3].oit_route, OitRoute::SharedOit);
    assert_eq!(
        got[4].oit_route,
        OitRoute::StandaloneSort(SortStrategy::ViewDepthRadix)
    );
    assert_eq!(
        got[5].oit_route,
        OitRoute::StandaloneSort(SortStrategy::ViewDepthBitonic)
    );
}

#[test]
fn motion_vector_decision_cases() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuShading::new(&ctx);
    let motions = [
        // Culled: writes nothing.
        MotionVectorInput {
            visible: false,
            blend: BlendMode::AlphaBlend,
            flipbook_rate: 60.0,
            flipbook_unstable_rate: 30.0,
        },
        // Visible opaque, slow: write, stable, zero reactive bias.
        MotionVectorInput {
            visible: true,
            blend: BlendMode::Opaque,
            flipbook_rate: 5.0,
            flipbook_unstable_rate: 30.0,
        },
        // Visible transparent, slow: write, stable, half reactive bias.
        MotionVectorInput {
            visible: true,
            blend: BlendMode::AlphaBlend,
            flipbook_rate: 5.0,
            flipbook_unstable_rate: 30.0,
        },
        // Visible transparent, fast flipbook: write, unstable, saturated mask.
        MotionVectorInput {
            visible: true,
            blend: BlendMode::Premultiplied,
            flipbook_rate: 60.0,
            flipbook_unstable_rate: 30.0,
        },
    ];
    let queries: Vec<GpuShadingQuery> = motions
        .iter()
        .map(|&motion| GpuShadingQuery {
            blend: motion.blend,
            motion,
            ..base_query()
        })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn six_way_single_axis_and_blend() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuShading::new(&ctx);
    let lum = SixWayLuminance {
        right: 1.0,
        left: 0.1,
        up: 0.2,
        down: 0.3,
        front: 0.4,
        back: 0.5,
    };
    let blend_lum = SixWayLuminance {
        right: 1.0,
        left: 0.0,
        up: 1.0,
        down: 0.0,
        front: 0.0,
        back: 0.0,
    };
    let queries = vec![
        // +X selects `right`.
        GpuShadingQuery {
            lum,
            light_dir: Vec3::new(2.0, 0.0, 0.0),
            ..base_query()
        },
        // -X selects `left`.
        GpuShadingQuery {
            lum,
            light_dir: Vec3::new(-1.0, 0.0, 0.0),
            ..base_query()
        },
        // -Y selects `down`.
        GpuShadingQuery {
            lum,
            light_dir: Vec3::new(0.0, -3.0, 0.0),
            ..base_query()
        },
        // 45 degrees between +X and +Y.
        GpuShadingQuery {
            lum: blend_lum,
            light_dir: Vec3::new(1.0, 1.0, 0.0),
            ..base_query()
        },
        // Zero direction yields zero response.
        GpuShadingQuery {
            lum,
            light_dir: Vec3::ZERO,
            ..base_query()
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn cel_band_quantization_ladder() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuShading::new(&ctx);
    // Responses placed clear of the floor's integer boundaries (fractional part
    // of `response * bands` near 0.5), plus the passthrough and clamp cases.
    let cases = [
        (0.0_f32, 1_u32),
        (0.37, 1),
        (0.5 / 3.0, 3),
        (1.5 / 3.0, 3),
        (2.5 / 3.0, 3),
        (3.5 / 8.0, 8),
        (2.0, 4),
        (-1.0, 4),
    ];
    let queries: Vec<GpuShadingQuery> = cases
        .iter()
        .map(|&(cel_response, cel_bands)| GpuShadingQuery {
            cel_response,
            cel_bands,
            ..base_query()
        })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn deep_shadow_quality_ladder() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuShading::new(&ctx);
    let queries = vec![
        // Non-volumetric: no self-shadow.
        GpuShadingQuery {
            model: EmberShadingModel::Pbr,
            volumetric: false,
            quality: ParticleQuality::Ultra,
            ..base_query()
        },
        // Volumetric but unlit: no self-shadow.
        GpuShadingQuery {
            model: EmberShadingModel::Unlit,
            volumetric: true,
            quality: ParticleQuality::Ultra,
            ..base_query()
        },
        // Volumetric lit, climbing the ladder.
        GpuShadingQuery {
            model: EmberShadingModel::Pbr,
            volumetric: true,
            quality: ParticleQuality::Low,
            ..base_query()
        },
        GpuShadingQuery {
            model: EmberShadingModel::Npr,
            volumetric: true,
            quality: ParticleQuality::Medium,
            ..base_query()
        },
        GpuShadingQuery {
            model: EmberShadingModel::Pbr,
            volumetric: true,
            quality: ParticleQuality::High,
            ..base_query()
        },
        GpuShadingQuery {
            model: EmberShadingModel::Pbr,
            volumetric: true,
            quality: ParticleQuality::Ultra,
            ..base_query()
        },
    ];
    check(&ctx, &gpu, &queries);
    // Spot-check the ladder endpoints.
    let got = gpu.eval(&ctx, &queries);
    assert_eq!(got[0].deep_shadow, DeepShadowMode::None);
    assert_eq!(got[1].deep_shadow, DeepShadowMode::None);
    assert_eq!(got[2].deep_shadow, DeepShadowMode::SixWay);
    assert_eq!(
        got[3].deep_shadow,
        DeepShadowMode::DeepOpacity { layers: 4 }
    );
    assert_eq!(
        got[4].deep_shadow,
        DeepShadowMode::DeepOpacity { layers: 8 }
    );
    assert_eq!(
        got[5].deep_shadow,
        DeepShadowMode::DeepOpacity { layers: 16 }
    );
}

#[test]
fn composite_program_cases() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuShading::new(&ctx);
    let queries = vec![
        // Full PBR volumetric program.
        GpuShadingQuery {
            model: EmberShadingModel::Pbr,
            blend: BlendMode::AlphaBlend,
            caps: all_caps(),
            volumetric: true,
            quality: ParticleQuality::High,
            ..base_query()
        },
        // Minimal unlit additive program.
        GpuShadingQuery {
            model: EmberShadingModel::Unlit,
            blend: BlendMode::Additive,
            caps: all_caps(),
            volumetric: true,
            quality: ParticleQuality::Ultra,
            ..base_query()
        },
        // Hybrid NPR overlay, no caps.
        GpuShadingQuery {
            model: EmberShadingModel::Hybrid {
                base: ShadingBasis::Pbr,
                overlay: ShadingBasis::Npr,
                weight: 0.5,
            },
            blend: BlendMode::Premultiplied,
            caps: LightingServiceCaps::default(),
            volumetric: true,
            quality: ParticleQuality::Medium,
            ..base_query()
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_batch_lane_for_lane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuShading::new(&ctx);

    let mut state = 0x5151_a1a1_c3c3_f0f0_u64;
    let mut queries = Vec::with_capacity(2_048);
    for _ in 0..2_048 {
        let model = pick_model(&mut state);
        let blend = pick_blend(&mut state);
        let caps = LightingServiceCaps {
            shadow_maps: lcg(&mut state) > 0.5,
            global_illumination: lcg(&mut state) > 0.5,
            ray_tracing: lcg(&mut state) > 0.5,
        };
        let volumetric = lcg(&mut state) > 0.5;
        let quality = pick_quality(&mut state);

        let footprint_a = ShadingAttributeFootprint {
            normal: lcg(&mut state) > 0.5,
            tangent: lcg(&mut state) > 0.5,
            material_params: lcg(&mut state) > 0.5,
            ramp_lut: lcg(&mut state) > 0.5,
            custom_params: lcg(&mut state) > 0.5,
        };
        let footprint_b = ShadingAttributeFootprint {
            normal: lcg(&mut state) > 0.5,
            tangent: lcg(&mut state) > 0.5,
            material_params: lcg(&mut state) > 0.5,
            ramp_lut: lcg(&mut state) > 0.5,
            custom_params: lcg(&mut state) > 0.5,
        };

        // OIT inputs: keep particle_count clear of both 1 and radix_min_count so
        // no discrete branch sits on a tie a ULP could flip.
        let radix_min_count = 512_u32;
        let count_pick = (lcg(&mut state) * 3.0) as u32;
        let particle_count = match count_pick {
            0 => 0,
            1 => 64,
            _ => 4_096,
        };
        let oit_blend = pick_blend(&mut state);
        let oit = SortDecision {
            blend: oit_blend,
            particle_count,
            radix_min_count,
            prefer_shared_oit: lcg(&mut state) > 0.5,
        };

        // Motion inputs: flipbook_rate kept well away from the instability
        // threshold so `>=` never sits on a tie.
        let mv_blend = pick_blend(&mut state);
        let flipbook_unstable_rate = 30.0_f32;
        let flipbook_rate = if lcg(&mut state) > 0.5 { 60.0 } else { 5.0 };
        let motion = MotionVectorInput {
            visible: lcg(&mut state) > 0.2,
            blend: mv_blend,
            flipbook_rate,
            flipbook_unstable_rate,
        };

        // Six-way: positive luminances; a normalized, non-degenerate light
        // direction built without any transcendental.
        let lum = SixWayLuminance {
            right: lcg(&mut state) * 2.0,
            left: lcg(&mut state) * 2.0,
            up: lcg(&mut state) * 2.0,
            down: lcg(&mut state) * 2.0,
            front: lcg(&mut state) * 2.0,
            back: lcg(&mut state) * 2.0,
        };
        let mut dx = lcg(&mut state) * 2.0 - 1.0;
        let dy = lcg(&mut state) * 2.0 - 1.0;
        let dz = lcg(&mut state) * 2.0 - 1.0;
        let mut len_sq = dx * dx + dy * dy + dz * dz;
        if len_sq < 0.01 {
            dx += 1.0;
            len_sq = dx * dx + dy * dy + dz * dz;
        }
        let inv_len = 1.0 / len_sq.sqrt();
        let light_dir = Vec3::new(dx * inv_len, dy * inv_len, dz * inv_len);

        // Cel: bands >= 2 and a response whose `response * bands` lands at an
        // integer plus one half, so the floor is stable far from any boundary.
        let cel_bands = 2 + (lcg(&mut state) * 7.0) as u32;
        let mut idx = (lcg(&mut state) * cel_bands as f32) as u32;
        if idx >= cel_bands {
            idx = cel_bands - 1;
        }
        let cel_response = (idx as f32 + 0.5) / cel_bands as f32;

        queries.push(GpuShadingQuery {
            model,
            blend,
            caps,
            volumetric,
            quality,
            footprint_a,
            footprint_b,
            oit,
            motion,
            lum,
            light_dir,
            cel_response,
            cel_bands,
        });
    }

    check(&ctx, &gpu, &queries);

    // The spread must exercise both phase classes, both lighting verdicts and
    // more than one OIT route, so the batch is not trivially uniform.
    let got = gpu.eval(&ctx, &queries);
    let mut saw_opaque = false;
    let mut saw_transparent = false;
    let mut saw_lit = false;
    let mut saw_unlit = false;
    let mut saw_order_independent = false;
    let mut saw_sorted = false;
    for g in &got {
        match g.phase {
            ParticleRenderPhase::Opaque => saw_opaque = true,
            ParticleRenderPhase::Transparent => saw_transparent = true,
            ParticleRenderPhase::AlphaMask => {}
        }
        saw_lit |= g.needs_lighting;
        saw_unlit |= !g.needs_lighting;
        match g.oit_route {
            OitRoute::OrderIndependent => saw_order_independent = true,
            OitRoute::SharedOit | OitRoute::StandaloneSort(_) => saw_sorted = true,
        }
    }
    assert!(
        saw_opaque && saw_transparent,
        "random batch should produce both render phases"
    );
    assert!(
        saw_lit && saw_unlit,
        "random batch should produce both lighting verdicts"
    );
    assert!(
        saw_order_independent && saw_sorted,
        "random batch should produce more than one OIT route"
    );
}
