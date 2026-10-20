//! Executable Hybrid-water behavior goldens.
//!
//! CPU equation goldens compare a synthetic composition transcription with
//! the public `MaskBlend` contract; they are NOT historical image captures.
//! Separate provenance/link gates pin the exact 8400fe972^ shader graph used
//! by the independent GPU oracle in `water_hybrid_gpu`.
//! Source checks pin the WESL consumer and the four raster entries,
//! while the optional module-validation test classifies adapter/device absence
//! as an explicit skip unless `PRISM_REQUIRE_GPU=1` turns it into a failure.
//! Module validation is not output evidence. Actual production raster,
//! render-FX, resident composition and HDR readbacks live in the sibling
//! `water_hybrid_gpu` integration target.
//! Geometry-to-region conversion here is a CPU contract: the current raster
//! consumes resolved foam and an authored shallow scalar, not shore-distance
//! or bathymetric-depth textures. Reactive reaches its overlay control rather
//! than an independent temporal-history output.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use core::sync::atomic::{AtomicUsize, Ordering};
use prism_render_architecture::water::shading::hybrid::{
    compose_hybrid_scalar, compose_mask_blend, hybrid_motion_vector, hybrid_reactive_mask,
    plan_hybrid, resolve_hybrid_geometry_weights, resolve_hybrid_weights, HybridCompositionInput,
    HybridCompositionOutput, HybridResponseArm, HybridWeightInputs, HybridWeights,
};
use prism_render_architecture::water::shading::npr::plan_npr;
use prism_render_architecture::water::shading::pbr::plan_pbr;
use prism_render_architecture::water::shading::HybridBlendParams;
use prism_render_architecture::water::shading::{
    NprShadingParams, PbrShadingParams, ReflectionTier, SurfaceShadingInputs,
};
use prism_render_architecture::water::underwater::RgbColor;
use wgpu::{
    Backends, Device, DeviceDescriptor, Instance, InstanceDescriptor, Queue,
    ShaderModuleDescriptor, ShaderSource,
};

#[path = "fixtures/water_legacy_41a31cee/mod.rs"]
mod legacy;

const RASTER_UUID: u128 = 0x5052_4953_4d5f_5741_5445_525f_4859_4201;
// The raster pipeline uses groups 0..=6. Keep this gate tied to the actual
// descriptor count so Apple Metal's eight-group limit remains enforced.
const REQUIRED_BIND_GROUPS: u32 = 7;

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("water Hybrid uses WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

type RasterShaderCache = ShaderCache<String, ()>;

fn raster_cache(
    loader: fn(&(), ShaderCacheSource, &ValidateShader) -> Result<String, ShaderCacheError>,
) -> (RasterShaderCache, AssetId<Shader>) {
    let mut cache = ShaderCache::new((), loader);
    let lighting = shader_id(RASTER_UUID.wrapping_add(1));
    let brdf = shader_id(RASTER_UUID.wrapping_add(2));
    let water = shader_id(RASTER_UUID.wrapping_add(3));
    let raster = shader_id(RASTER_UUID);
    for (index, (name, source)) in [
        ("tangent", include_str!("../src/shaders/tangent.wesl")),
        (
            "specular_aa",
            include_str!("../src/shaders/specular_aa.wesl"),
        ),
        ("thin_film", include_str!("../src/shaders/thin_film.wesl")),
        ("glint", include_str!("../src/shaders/glint.wesl")),
    ]
    .into_iter()
    .enumerate()
    {
        cache.set_shader(
            shader_id(RASTER_UUID.wrapping_add(4 + index as u128)),
            Shader::from_wesl(
                source,
                format!("embedded://prism_render_scene/shaders/{name}.wesl"),
            ),
        );
    }
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../src/shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../src/shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );
    cache.set_shader(
        water,
        Shader::from_wesl(
            include_str!("../src/shaders/water.wesl"),
            "embedded://prism_render_scene/shaders/water.wesl",
        ),
    );
    cache.set_shader(
        raster,
        Shader::from_wesl(
            include_str!("../src/shaders/water_surface_raster.wesl"),
            "embedded://prism_render_scene/shaders/water_surface_raster.wesl",
        ),
    );
    (cache, raster)
}

fn compile_raster_wgsl() -> String {
    let (mut cache, raster) = raster_cache(load_source);
    (*cache
        .get(0, raster, &[])
        .unwrap_or_else(|error| panic!("water_surface_raster.wesl failed to compile: {error}")))
    .clone()
}

fn source() -> &'static str {
    include_str!("../src/shaders/water_surface_raster.wesl")
}

fn composition_source() -> &'static str {
    include_str!("../src/shaders/response_composition.wgsl")
}

fn hdr_export_source() -> &'static str {
    include_str!("../src/shaders/response_to_hdr.wgsl")
}

fn pbr_params() -> PbrShadingParams {
    PbrShadingParams {
        f0_override: 0.0,
        foam_fold_threshold: 1.0,
        base_roughness: 0.08,
        grazing_transmission: 0.6,
        ssr_min_confidence: 0.5,
        rt_min_budget: 0.25,
    }
}

fn npr_params() -> NprShadingParams {
    NprShadingParams {
        color_bands: 4,
        specular_threshold: 0.7,
        foam_edge_width: 1.5,
        halftone_scale: 8.0,
        flow_line_gain: 0.5,
    }
}

fn blend_params() -> HybridBlendParams {
    HybridBlendParams {
        shore_blend_dist: 3.0,
        deep_blend_depth: 2.0,
        overlay_opacity: 0.35,
    }
}

fn inputs(depth: f32, dist_to_shore: f32) -> SurfaceShadingInputs {
    SurfaceShadingInputs {
        cos_view: 0.5,
        ior: 1.33,
        jacobian: 0.4,
        water_color: RgbColor {
            r: 0.1,
            g: 0.3,
            b: 0.5,
        },
        specular_intensity: 0.8,
        caustic_intensity: 0.6,
        flow_speed: 1.2,
        depth,
        dist_to_shore,
        ssr_confidence: 0.7,
        ray_budget: 0.5,
    }
}

fn composition_input(
    shore_signal: f32,
    shallow_signal: f32,
    foam: f32,
    motion: [f32; 2],
) -> HybridCompositionInput {
    HybridCompositionInput {
        pbr: HybridResponseArm {
            color: [0.12, 0.28, 0.46, 0.42],
            reflection: 0.82,
            caustics: 0.18,
        },
        npr: HybridResponseArm {
            color: [0.76, 0.91, 1.0, 0.88],
            reflection: 0.31,
            caustics: 0.74,
        },
        weights: HybridWeightInputs {
            shore_signal,
            shallow_signal,
            overlay_opacity: 0.35,
        },
        foam,
        motion,
    }
}

fn legacy_mix_rgba(left: [f32; 4], right: [f32; 4], factor: f32) -> [f32; 4] {
    [
        left[0] * (1.0 - factor) + right[0] * factor,
        left[1] * (1.0 - factor) + right[1] * factor,
        left[2] * (1.0 - factor) + right[2] * factor,
        left[3] * (1.0 - factor) + right[3] * factor,
    ]
}

fn legacy_mix_scalar(left: f32, right: f32, factor: f32) -> f32 {
    left * (1.0 - factor) + right * factor
}

/// Synthetic CPU composition equation check, not a historical shading oracle.
/// It does not call the composition helper, but cannot detect a shared PBR/NPR
/// shading regression; the frozen-source GPU target supplies that evidence.
fn legacy_hybrid_reference(input: HybridCompositionInput) -> HybridCompositionOutput {
    let shore = input.weights.shore_signal.clamp(0.0, 1.0);
    let shallow = input.weights.shallow_signal.clamp(0.0, 1.0);
    let npr_weight = shore.max(shallow);
    let pbr_weight = 1.0 - npr_weight;
    let overlay_opacity = input.weights.overlay_opacity.clamp(0.0, 1.0);

    let blended_color = legacy_mix_rgba(input.pbr.color, input.npr.color, npr_weight);
    let blended_reflection =
        legacy_mix_scalar(input.pbr.reflection, input.npr.reflection, npr_weight);
    let blended_caustics = legacy_mix_scalar(input.pbr.caustics, input.npr.caustics, npr_weight);
    let motion_strength =
        (input.motion[0] * input.motion[0] + input.motion[1] * input.motion[1]).sqrt();
    let reactive_mask = input
        .foam
        .clamp(0.0, 1.0)
        .max((motion_strength * 32.0).clamp(0.0, 1.0))
        .max((1.0 - blended_color[3]).clamp(0.0, 1.0));
    let overlay_factor = (overlay_opacity * reactive_mask.max(0.25)).clamp(0.0, 1.0);
    let color = legacy_mix_rgba(blended_color, input.npr.color, overlay_factor);

    HybridCompositionOutput {
        color,
        reflection: legacy_mix_scalar(blended_reflection, input.npr.reflection, overlay_factor),
        caustics: legacy_mix_scalar(blended_caustics, input.npr.caustics, overlay_factor),
        transparency: color[3],
        motion: input.motion,
        npr_weight,
        pbr_weight,
        reactive_mask,
        overlay_factor,
    }
}

fn assert_close(actual: f32, expected: f32, label: &str) {
    assert!(
        (actual - expected).abs() <= 1.0e-6,
        "{label}: actual={actual:?}, expected={expected:?}"
    );
}

fn assert_composition_equivalent(input: HybridCompositionInput) -> HybridCompositionOutput {
    let legacy = legacy_hybrid_reference(input);
    let replacement = compose_mask_blend(input);
    for (index, (actual, expected)) in replacement.color.into_iter().zip(legacy.color).enumerate() {
        assert_close(actual, expected, &format!("color[{index}]"));
    }
    assert_close(replacement.reflection, legacy.reflection, "reflection");
    assert_close(replacement.caustics, legacy.caustics, "caustics");
    assert_close(
        replacement.transparency,
        legacy.transparency,
        "transparency",
    );
    assert_eq!(replacement.motion, legacy.motion, "motion");
    assert_close(replacement.npr_weight, legacy.npr_weight, "npr weight");
    assert_close(replacement.pbr_weight, legacy.pbr_weight, "pbr weight");
    assert_close(
        replacement.reactive_mask,
        legacy.reactive_mask,
        "reactive mask",
    );
    assert_close(
        replacement.overlay_factor,
        legacy.overlay_factor,
        "overlay factor",
    );
    replacement
}

#[test]
fn golden_depth_mixing_contract() {
    let replacement = assert_composition_equivalent(composition_input(0.0, 0.75, 0.0, [0.0, 0.0]));
    assert_close(replacement.npr_weight, 0.75, "depth npr weight");
    assert_close(replacement.pbr_weight, 0.25, "depth pbr weight");
    let shallow = resolve_hybrid_geometry_weights(blend_params(), inputs(0.0, 10.0));
    let deep = resolve_hybrid_geometry_weights(blend_params(), inputs(10.0, 10.0));
    assert_eq!(shallow.npr_weight, 1.0);
    assert_eq!(shallow.pbr_weight, 0.0);
    assert_eq!(deep.npr_weight, 0.0);
    assert_eq!(deep.pbr_weight, 1.0);
    assert!(source().contains("water_hybrid_weight_contract"));
}

#[test]
fn golden_shore_mixing_contract() {
    let replacement = assert_composition_equivalent(composition_input(0.65, 0.0, 0.0, [0.0, 0.0]));
    assert_close(replacement.npr_weight, 0.65, "shore npr weight");
    assert_close(replacement.pbr_weight, 0.35, "shore pbr weight");
    let shore = plan_hybrid(
        blend_params(),
        pbr_params(),
        npr_params(),
        inputs(10.0, 0.0),
    );
    let offshore = plan_hybrid(
        blend_params(),
        pbr_params(),
        npr_params(),
        inputs(10.0, 10.0),
    );
    assert_eq!(shore.npr_weight, 1.0);
    assert_eq!(offshore.npr_weight, 0.0);
    assert!((shore.pbr_weight + shore.npr_weight - 1.0).abs() < 1.0e-6);
    assert!(source().contains("max(shore, shallow)"));
}

#[test]
fn golden_foam_contract() {
    let calm = assert_composition_equivalent(composition_input(0.25, 0.0, 0.0, [0.0, 0.0]));
    let foamy = assert_composition_equivalent(composition_input(0.25, 0.0, 0.8, [0.0, 0.0]));
    assert!(foamy.reactive_mask > calm.reactive_mask);
    assert!(foamy.overlay_factor > calm.overlay_factor);
    let mut flat = inputs(1.0, 1.0);
    flat.jacobian = 1.0;
    let mut folded = flat;
    folded.jacobian = 0.0;
    let flat_hybrid = plan_hybrid(blend_params(), pbr_params(), npr_params(), flat);
    let folded_hybrid = plan_hybrid(blend_params(), pbr_params(), npr_params(), folded);
    assert!(folded_hybrid.pbr.foam_mask > flat_hybrid.pbr.foam_mask);
    assert!(source().contains("surface_from(in.foam)"));
    assert!(source().contains("water_hybrid_reactive_mask"));
}

#[test]
fn golden_caustics_contract() {
    let mut sample = composition_input(0.4, 0.2, 0.0, [0.0, 0.0]);
    sample.pbr.caustics = 0.11;
    sample.npr.caustics = 0.93;
    let replacement = assert_composition_equivalent(sample);
    assert!(replacement.caustics > sample.pbr.caustics);
    assert!(replacement.caustics < sample.npr.caustics);
    let mut low = inputs(1.0, 1.0);
    low.caustic_intensity = 0.1;
    let mut high = low;
    high.caustic_intensity = 0.9;
    let low_npr = plan_npr(npr_params(), low);
    let high_npr = plan_npr(npr_params(), high);
    assert!(high_npr.halftone_coverage > low_npr.halftone_coverage);
    assert!(source().contains("fn water_caustic_gain("));
    assert!(source().contains("water_caustic_gain(in.uv)"));
}

#[test]
fn golden_reflection_contract() {
    let mut sample = composition_input(0.4, 0.2, 0.0, [0.0, 0.0]);
    sample.pbr.reflection = 0.95;
    sample.npr.reflection = 0.15;
    let replacement = assert_composition_equivalent(sample);
    assert!(replacement.reflection > sample.npr.reflection);
    assert!(replacement.reflection < sample.pbr.reflection);
    let mut ssr = inputs(1.0, 1.0);
    ssr.ssr_confidence = 0.9;
    assert_eq!(
        plan_pbr(pbr_params(), ssr).reflection_tier,
        ReflectionTier::ScreenSpace
    );
    ssr.ssr_confidence = 0.1;
    ssr.ray_budget = 0.5;
    assert_eq!(
        plan_pbr(pbr_params(), ssr).reflection_tier,
        ReflectionTier::RayTraced
    );
    ssr.ray_budget = 0.1;
    assert_eq!(
        plan_pbr(pbr_params(), ssr).reflection_tier,
        ReflectionTier::Probe
    );
    assert!(source().contains("water_ssr_reflection"));
    assert!(source().contains("water_ibl"));
}

#[test]
fn golden_transparency_and_overlay_contract() {
    let mut sample = composition_input(0.25, 0.0, 0.0, [0.0, 0.0]);
    sample.pbr.color[3] = 0.2;
    sample.npr.color[3] = 0.9;
    let replacement = assert_composition_equivalent(sample);
    assert_close(
        replacement.transparency,
        replacement.color[3],
        "alpha output",
    );
    assert!((0.2..=0.9).contains(&replacement.transparency));
    let weights = HybridWeights {
        npr_weight: 0.25,
        pbr_weight: 0.75,
        overlay_opacity: 0.5,
    };
    let without_overlay = compose_hybrid_scalar(
        0.0,
        1.0,
        HybridWeights {
            overlay_opacity: 0.0,
            ..weights
        },
    );
    let with_overlay = compose_hybrid_scalar(0.0, 1.0, weights);
    assert!(with_overlay > without_overlay);
    assert!(source().contains("water_hybrid_apply_overlay"));
    assert!(source().contains("weights.overlay_opacity"));
    assert!(source().contains("view.viewport.z"));
    assert!(source().contains("output_alpha"));
    assert!(source().contains("return FragmentOutput(color, motion,"));
}

#[test]
fn golden_motion_vector_contract() {
    let replacement =
        assert_composition_equivalent(composition_input(0.25, 0.0, 0.0, [0.25, -0.75]));
    assert_eq!(replacement.motion, [0.25, -0.75]);
    assert_eq!(
        hybrid_motion_vector([0.75, 0.25], [0.5, 0.5]),
        [0.25, -0.25]
    );
    assert!(source().contains("previous_displacement"));
    assert!(source().contains("return cur.xy - prev.xy"));
    assert!(source().contains("let motion = water_motion_vector(in)"));
}

#[test]
fn golden_reactive_mask_contract() {
    let replacement = assert_composition_equivalent(composition_input(0.0, 0.0, 0.8, [0.0, 0.0]));
    assert_close(replacement.reactive_mask, 0.8, "foam reactive mask");
    let moving_replacement =
        assert_composition_equivalent(composition_input(0.0, 0.0, 0.0, [0.1, 0.0]));
    assert_close(
        moving_replacement.reactive_mask,
        1.0,
        "motion reactive mask",
    );
    let calm = hybrid_reactive_mask(0.0, [0.0, 0.0], 1.0);
    let moving = hybrid_reactive_mask(0.0, [0.1, 0.0], 1.0);
    let foamy = hybrid_reactive_mask(0.8, [0.0, 0.0], 1.0);
    assert_eq!(calm, 0.0);
    assert!(moving > calm);
    assert!(foamy > calm);
    assert!(source().contains("fn water_hybrid_reactive_mask("));
    assert!(source().contains("max(reactive, 0.25)"));
}

#[test]
fn hybrid_composition_golden_grid_matches_legacy_reference() {
    let region_signals = [-1.0, 0.0, 0.2, 0.75, 1.0, 2.0];
    let foam_values = [-0.5, 0.0, 0.4, 1.0, 1.5];
    let motions = [[0.0, 0.0], [0.01, -0.02], [0.1, 0.0], [-0.4, 0.3]];

    for shore_signal in region_signals {
        for shallow_signal in region_signals {
            for foam in foam_values {
                for motion in motions {
                    assert_composition_equivalent(composition_input(
                        shore_signal,
                        shallow_signal,
                        foam,
                        motion,
                    ));
                }
            }
        }
    }
}

#[test]
fn hybrid_weight_contract_exhaustively_clamps_and_partitions() {
    let signals = [-2.0, 0.0, 0.25, 1.0, 2.0];
    for shore_signal in signals {
        for shallow_signal in signals {
            let weights = resolve_hybrid_weights(HybridWeightInputs {
                shore_signal,
                shallow_signal,
                overlay_opacity: shore_signal - shallow_signal,
            });
            let expected_npr = shore_signal
                .clamp(0.0, 1.0)
                .max(shallow_signal.clamp(0.0, 1.0));
            assert_eq!(weights.npr_weight, expected_npr);
            assert_eq!(weights.pbr_weight, 1.0 - expected_npr);
            assert!((0.0..=1.0).contains(&weights.overlay_opacity));
            assert!((weights.pbr_weight + weights.npr_weight - 1.0).abs() <= f32::EPSILON);
        }
    }
}

#[test]
fn hybrid_weight_contract_matches_wesl_reference_formula() {
    fn wesl_reference(
        shore_signal: f32,
        shallow_signal: f32,
        overlay_opacity: f32,
    ) -> HybridWeights {
        let shore = shore_signal.clamp(0.0, 1.0);
        let shallow = shallow_signal.clamp(0.0, 1.0);
        let npr_weight = shore.max(shallow);
        HybridWeights {
            npr_weight,
            pbr_weight: 1.0 - npr_weight,
            overlay_opacity: overlay_opacity.clamp(0.0, 1.0),
        }
    }

    for (shore_signal, shallow_signal, overlay_opacity) in
        [(-1.0, 0.0, -0.5), (0.25, 0.75, 0.5), (1.5, 0.25, 2.0)]
    {
        let inputs = HybridWeightInputs {
            shore_signal,
            shallow_signal,
            overlay_opacity,
        };
        assert_eq!(
            resolve_hybrid_weights(inputs),
            wesl_reference(shore_signal, shallow_signal, overlay_opacity)
        );
    }
}

#[test]
fn hybrid_cpu_smoke_workload_is_deterministic_and_finite() {
    fn run() -> u64 {
        let mut digest = 0_u64;
        for index in 0..4096_u32 {
            let depth = (index % 37) as f32 * 0.125;
            let distance = (index % 29) as f32 * 0.25;
            let response = plan_hybrid(
                blend_params(),
                pbr_params(),
                npr_params(),
                inputs(depth, distance),
            );
            assert!(response.pbr_weight.is_finite());
            assert!(response.npr_weight.is_finite());
            digest = digest.rotate_left(5)
                ^ u64::from(response.pbr_weight.to_bits())
                ^ (u64::from(response.npr_weight.to_bits()) << 17);
        }
        digest
    }

    let first = run();
    let second = run();
    assert_eq!(
        first, second,
        "fixed Hybrid smoke workload must be repeatable"
    );
    assert_ne!(
        first, 0,
        "smoke workload must exercise non-zero response weights"
    );
}

#[test]
fn split_response_intermediate_memory_accounting_is_explicit() {
    const RESPONSE_OUTPUT_FLOATS: u64 = 16;
    const BYTES_PER_FLOAT: u64 = 4;
    const BYTES_PER_PIXEL: u64 = RESPONSE_OUTPUT_FLOATS * BYTES_PER_FLOAT;
    let bytes = |width: u64, height: u64| width * height * BYTES_PER_PIXEL;

    assert_eq!(BYTES_PER_PIXEL, 64);
    assert_eq!(bytes(1920, 1080), 132_710_400);
    assert_eq!(bytes(3840, 2160), 530_841_600);
    assert!(bytes(3840, 2160) > bytes(1920, 1080));
}

static SHADER_LOADS: AtomicUsize = AtomicUsize::new(0);

fn counting_load_source(
    device: &(),
    source: ShaderCacheSource,
    validate: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    SHADER_LOADS.fetch_add(1, Ordering::SeqCst);
    load_source(device, source, validate)
}

#[test]
fn water_shader_cache_has_observable_cold_then_warm_behavior() {
    SHADER_LOADS.store(0, Ordering::SeqCst);
    let (mut cache, raster) = raster_cache(counting_load_source);

    let cold = cache
        .get(0, raster, &[])
        .unwrap_or_else(|error| panic!("cold water shader compile failed: {error}"))
        .clone();
    let loads_after_cold = SHADER_LOADS.load(Ordering::SeqCst);
    assert!(
        loads_after_cold > 0,
        "cold lookup must invoke the shader loader"
    );

    let warm = cache
        .get(0, raster, &[])
        .unwrap_or_else(|error| panic!("warm water shader lookup failed: {error}"))
        .clone();
    assert_eq!(cold, warm);
    assert_eq!(
        SHADER_LOADS.load(Ordering::SeqCst),
        loads_after_cold,
        "warm lookup must reuse the composed shader"
    );
}

#[test]
fn hybrid_wesl_compiles_and_preserves_all_four_entries() {
    let wgsl = compile_raster_wgsl();
    assert!(wgsl.contains("water_surface_fs_pbr"));
    assert!(wgsl.contains("water_surface_fs_npr"));
    assert!(wgsl.contains("water_surface_fs_custom"));
    assert!(wgsl.contains("water_surface_fs_mask_blend"));
    assert!(
        !wgsl.contains("water_surface_fs_hybrid"),
        "the removed legacy Hybrid entry must not return as a live shader API"
    );
}

#[test]
fn historical_water_fixture_has_pinned_provenance_and_closed_imports() {
    legacy::verify();
    assert_eq!(legacy::COMMIT, "41a31ceef24e59ab7f8dda076b2b4c7f5e11f6ec");
    assert_eq!(legacy::SOURCES.len(), 11);
    for frozen in legacy::SOURCES {
        assert_eq!(frozen.blob.len(), 40, "Git blob identity: {}", frozen.name);
        // These historical modules have no external imports. Catch any later
        // edit adding a live dependency even before running a GPU gate.
        for line in frozen
            .source
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("import "))
        {
            let path = line
                .strip_prefix("import prism_render_scene::shaders::")
                .unwrap_or_else(|| panic!("unexpected external historical import: {line}"));
            let name = path.split("::").next().expect("imported module name");
            assert!(
                legacy::SOURCES
                    .iter()
                    .any(|s| s.name == format!("{name}.wesl")),
                "{} imports unfrozen module {name}",
                frozen.name
            );
        }
    }
    let raster = legacy::source("water_surface_raster.wesl");
    assert!(raster.contains("fn water_surface_fs_hybrid("));
    assert!(!raster.contains("fn water_surface_fs_mask_blend("));
    assert!(raster.contains(
        "water_hybrid_weight_contract(in.foam, view.style_params.w, view.style_params.w)"
    ));
    assert!(raster.contains("CompositionControl(mask, 1u, left.a, 0u)"));
    assert!(!legacy::source("brdf.wesl").contains("clearcoat_interface_normal"));
}

#[test]
fn historical_water_wesl_graph_links_without_live_shader_dependencies() {
    let raster = legacy::compile("water_surface_raster.wesl");
    for entry in [
        "water_surface_vs",
        "water_surface_fs_pbr",
        "water_surface_fs_npr",
        "water_surface_fs_hybrid",
    ] {
        assert!(raster.contains(entry), "historical entry missing: {entry}");
    }
    assert!(!raster.contains("water_surface_fs_mask_blend"));
    let fx = legacy::compile("water_render_fx.wesl");
    assert!(fx.contains("water_caustics_project"));
    assert!(fx.contains("water_underwater_volume"));
}

#[test]
fn hybrid_runtime_is_wired_through_resident_composition_and_hdr_export() {
    let raster = source();
    let composition = composition_source();
    let hdr = hdr_export_source();

    assert!(raster.contains("response_left[pixel] = response_value(left)"));
    assert!(raster.contains("response_right[pixel] = response_value(right)"));
    assert!(raster.contains("CompositionControl(mask, 31u, output_alpha, 0u)"));
    assert!(!raster.contains("CompositionControl(mask, 1u, output_alpha, 0u)"));
    assert!(raster.contains("water_coverage[pixel] = 1u"));

    assert!(composition.contains("water_coverage"));
    assert!(
        composition.contains("out.words[word] = linear(out.words[word], b.words[word], c.mask)")
    );
    assert!(composition.contains("out.words[15] = c.shared_alpha"));
    assert!(hdr.contains("textureStore(hdr, coord, vec4<f32>(rgb, words[15]))"));
}

fn try_device() -> Option<(Device, Queue)> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: wgpu::BackendOptions::default(),
    });
    let adapter = match block_on(instance.request_adapter(&Default::default())) {
        Ok(adapter) => adapter,
        Err(error) => {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "PRISM_REQUIRE_GPU set: water Hybrid GPU adapter request failed: {error}"
            );
            eprintln!("water Hybrid GPU SKIP: adapter unavailable: {error}");
            return None;
        }
    };
    let info = adapter.get_info();
    if !matches!(
        info.device_type,
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu
    ) {
        assert!(
            std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
            "PRISM_REQUIRE_GPU set: water Hybrid requires physical GPU hardware: {info:?}"
        );
        eprintln!("water Hybrid module-validation SKIP: non-hardware adapter: {info:?}");
        return None;
    }
    eprintln!(
        "water Hybrid module-validation adapter: {} / {:?} / {:?}",
        info.name, info.backend, info.device_type
    );
    if adapter.limits().max_bind_groups < REQUIRED_BIND_GROUPS {
        let message = format!(
            "adapter exposes max_bind_groups={}, but water raster requires {}",
            adapter.limits().max_bind_groups,
            REQUIRED_BIND_GROUPS
        );
        assert!(
            std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
            "PRISM_REQUIRE_GPU set: water Hybrid GPU capability gate failed: {message}"
        );
        eprintln!("water Hybrid GPU SKIP: {message}");
        return None;
    }
    let required_limits = wgpu::Limits {
        max_bind_groups: REQUIRED_BIND_GROUPS,
        ..wgpu::Limits::default()
    };
    match block_on(adapter.request_device(&DeviceDescriptor {
        required_limits,
        ..Default::default()
    })) {
        Ok(device_queue) => Some(device_queue),
        Err(error) => {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "PRISM_REQUIRE_GPU set: water Hybrid GPU device request failed: {error}"
            );
            eprintln!("water Hybrid GPU SKIP: device unavailable: {error}");
            None
        }
    }
}

#[test]
fn hybrid_wesl_gpu_shader_module_validation_only() {
    let Some((device, _queue)) = try_device() else {
        eprintln!("hybrid_wesl_gpu_shader_module_validation_only: SKIP");
        return;
    };
    let wgsl = compile_raster_wgsl();
    let _module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water Hybrid raster golden"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let _composition = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water Hybrid resident composition golden"),
        source: ShaderSource::Wgsl(composition_source().into()),
    });
    let _hdr = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water Hybrid HDR export golden"),
        source: ShaderSource::Wgsl(hdr_export_source().into()),
    });
    eprintln!(
        "water Hybrid shader modules validated; no draw, dispatch, or output readback in this test"
    );
}
