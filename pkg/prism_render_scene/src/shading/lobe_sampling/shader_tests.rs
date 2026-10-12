//! WESL link coverage for deterministic stochastic single-lobe sampling.
//!
//! This test compiles a tiny compute entry point that references every public
//! helper and constant in `lobe_sampling.wesl`. It proves import visibility,
//! signatures, and type-checking. Numeric parity with the CPU golden
//! (`pkg/prism_render_shading/src/lobe_sampling.rs`) is achieved by mirroring
//! hash arithmetic, R2 constants, clamps, cast order, and branch structure;
//! once the rendering pass is wired, the same fixture inputs can be read back
//! from a GPU buffer without changing the kernel.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("lobe-sampling shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

const LINK_TEST_SOURCE: &str = r#"
import prism_render_scene::shaders::lobe_sampling::{
    LS_SEED_SALT, LS_GOLDEN_STRIDE, LS_TEMPORAL_SALT, LS_SEQUENCE_PERIOD,
    LS_SEQUENCE_MASK, LS_R2_A1, LS_R2_A2, LS_MIN_WEIGHT,
    LS_ONE_MINUS_EPSILON, LS_NO_LOBE,
    ls_is_finite, ls_hash_u32, ls_hash_combine, ls_uniform_from_hash,
    ls_uniform_at, ls_sanitize_weight, ls_clamp_unit, ls_rotate_left_16,
    ls_r2_sample_2d, ls_pixel_seed, ls_temporal_stride, ls_blue_noise_2d,
    ls_blue_noise_1d, ls_max_full_lobes, ls_should_use_stochastic,
    ls_should_use_stochastic_for_quality, ls_full_evaluations,
    ls_weight_probability, ls_estimator_weight, ls_sample_is_valid,
    ls_make_sample, ls_estimate_scalar, ls_cdf_target, ls_select_bucket,
    ls_uniform_fallback_index, LobeSample,
};

@group(0) @binding(0)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn lobe_sampling_link_test(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) {
        return;
    }

    let h0 = ls_hash_u32(0x12345678u);
    let h1 = ls_hash_combine(h0, 678u);
    let u0 = ls_uniform_from_hash(h1);
    let u1 = ls_uniform_at(h1, 3u);
    let w = ls_sanitize_weight(2.0);
    let clamped_u = ls_clamp_unit(1.25);
    let rotated = ls_rotate_left_16(h0);

    let pixel_seed = ls_pixel_seed(137u, 901u, 0x12345678u);
    let stride = ls_temporal_stride(pixel_seed);
    let r2 = ls_r2_sample_2d(17u, u0, u1);
    let blue2 = ls_blue_noise_2d(137u, 901u, 42u, 0u, 0x12345678u);
    let blue1 = ls_blue_noise_1d(137u, 901u, 42u, 0u, 0x12345678u);

    let low_max = ls_max_full_lobes(0u);
    let medium_max = ls_max_full_lobes(1u);
    let high_max = ls_max_full_lobes(2u);
    let stochastic = ls_should_use_stochastic(low_max, 4u);
    let stochastic_quality = ls_should_use_stochastic_for_quality(1u, 3u);
    let evaluations = ls_full_evaluations(medium_max, 3u);

    let probability = ls_weight_probability(w, 5.0);
    let estimator_weight = ls_estimator_weight(probability);
    let sample: LobeSample = ls_make_sample(2u, probability);
    let valid = ls_sample_is_valid(sample);
    let estimate = ls_estimate_scalar(0.25, sample);
    let cdf_target = ls_cdf_target(blue1, 5.0);
    let selected = ls_select_bucket(cdf_target, 3.0);
    let fallback = ls_uniform_fallback_index(blue1, 4u);

    output[0] = f32(h0);
    output[1] = f32(h1);
    output[2] = u0;
    output[3] = u1;
    output[4] = w;
    output[5] = clamped_u;
    output[6] = f32(rotated);
    output[7] = f32(pixel_seed);
    output[8] = f32(stride);
    output[9] = r2.x;
    output[10] = r2.y;
    output[11] = blue2.x;
    output[12] = blue2.y;
    output[13] = blue1;
    output[14] = f32(low_max);
    output[15] = f32(medium_max);
    output[16] = f32(high_max);
    output[17] = u32(select(0u, 1u, stochastic));
    output[18] = u32(select(0u, 1u, stochastic_quality));
    output[19] = f32(evaluations);
    output[20] = probability;
    output[21] = estimator_weight;
    output[22] = f32(sample.index);
    output[23] = sample.probability;
    output[24] = sample.estimator_weight;
    output[25] = u32(select(0u, 1u, valid));
    output[26] = estimate;
    output[27] = cdf_target;
    output[28] = u32(select(0u, 1u, selected));
    output[29] = f32(fallback);
    output[30] = LS_R2_A1;
    output[31] = LS_R2_A2;
    output[32] = LS_MIN_WEIGHT;
    output[33] = LS_ONE_MINUS_EPSILON;
    output[34] = f32(LS_NO_LOBE);
    output[35] = f32(LS_SEED_SALT);
    output[36] = f32(LS_GOLDEN_STRIDE);
    output[37] = f32(LS_TEMPORAL_SALT);
    output[38] = f32(LS_SEQUENCE_PERIOD);
    output[39] = f32(LS_SEQUENCE_MASK);
    output[40] = select(0.0, 1.0, ls_is_finite(estimate));
}
"#;

#[test]
fn lobe_sampling_wesl_compiles_and_links_every_helper() {
    let mut cache = ShaderCache::new((), load_source);

    let module = shader_id(0x5052_4953_4d5f_4c53_414d_504c_455f_0001);
    cache.set_shader(
        module,
        Shader::from_wesl(
            include_str!("../../shaders/lobe_sampling.wesl"),
            "embedded://prism_render_scene/shaders/lobe_sampling.wesl",
        ),
    );

    let link_test = shader_id(0x5052_4953_4d5f_4c53_414d_504c_455f_0002);
    cache.set_shader(
        link_test,
        Shader::from_wesl(
            LINK_TEST_SOURCE,
            "embedded://prism_render_scene/shading/lobe_sampling/link_test.wesl",
        ),
    );

    cache.get(0, link_test, &[]).unwrap_or_else(|error| {
        panic!("lobe_sampling.wesl failed to compile/link every helper: {error}")
    });
}
