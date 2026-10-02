//! WESL link coverage for the shared specular-AA helper module.
//!
//! This deliberately links a tiny compute entry point against the module rather
//! than merely parsing its source. Every helper is referenced by the live entry
//! point, so import visibility, signatures, and type-checking are covered.
//! Numeric fidelity is guaranteed by exact mirroring of the CPU golden's
//! operation order, clamps, floors, and finite-sanitization branches.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("specular-AA shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

const LINK_TEST_SOURCE: &str = r#"
import prism_render_scene::shaders::specular_aa::{
    SPEC_AA_MIN_ALPHA, SPEC_AA_SIGMA2, SPEC_AA_KAPPA_MAX,
    TOKSVIG_MIN_LEN, TOKSVIG_MAX_SHININESS,
    spec_aa_is_finite,
    toksvig_sanitize_len,
    toksvig_sanitize_shininess,
    spec_aa_roughness_to_alpha,
    spec_aa_safe_sq_len,
    spec_aa_screen_space_variance,
    spec_aa_kernel_roughness_sq,
    spec_aa_filter_alpha_sq,
    spec_aa_delta_alpha_sq_from_derivatives,
    spec_aa_geometric_specular_aa_roughness,
    toksvig_factor,
    toksvig_effective_shininess,
    toksvig_shininess_from_alpha,
    toksvig_alpha_from_shininess,
    toksvig_delta_alpha_sq,
    toksvig_effective_alpha_sq_from_len,
    toksvig_roughness,
    toksvig_combine_roughness,
};

@group(0) @binding(0)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn specular_aa_link_test(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) {
        return;
    }
    let ddx = vec3<f32>(0.2, 0.05, 0.01);
    let ddy = vec3<f32>(0.03, 0.22, 0.0);
    let alpha = spec_aa_roughness_to_alpha(0.2);
    let sq_len = spec_aa_safe_sq_len(ddx);
    let variance = spec_aa_screen_space_variance(ddx, ddy, SPEC_AA_SIGMA2);
    let kernel = spec_aa_kernel_roughness_sq(variance, SPEC_AA_KAPPA_MAX);
    let filtered_alpha_sq = spec_aa_filter_alpha_sq(alpha * alpha, variance, SPEC_AA_KAPPA_MAX);
    let geometric_delta = spec_aa_delta_alpha_sq_from_derivatives(ddx, ddy, SPEC_AA_SIGMA2, SPEC_AA_KAPPA_MAX);
    let geometric_roughness = spec_aa_geometric_specular_aa_roughness(0.2, ddx, ddy, SPEC_AA_SIGMA2, SPEC_AA_KAPPA_MAX);
    let factor = toksvig_factor(0.7, 100.0);
    let effective_shininess = toksvig_effective_shininess(100.0, 0.7);
    let shininess = toksvig_shininess_from_alpha(alpha);
    let effective_alpha = toksvig_alpha_from_shininess(shininess);
    let toksvig_delta = toksvig_delta_alpha_sq(0.7);
    let toksvig_alpha_sq = toksvig_effective_alpha_sq_from_len(alpha, 0.7);
    let toksvig_rough = toksvig_roughness(0.2, 0.7);
    let combined = toksvig_combine_roughness(geometric_roughness, toksvig_rough);
    output[0] = sq_len;
    output[1] = variance;
    output[2] = kernel;
    output[3] = filtered_alpha_sq;
    output[4] = geometric_delta;
    output[5] = geometric_roughness;
    output[6] = factor;
    output[7] = effective_shininess;
    output[8] = shininess;
    output[9] = effective_alpha;
    output[10] = toksvig_delta;
    output[11] = toksvig_alpha_sq;
    output[12] = toksvig_rough;
    output[13] = combined;
    output[14] = select(0.0, 1.0, spec_aa_is_finite(alpha));
    output[15] = toksvig_sanitize_len(TOKSVIG_MIN_LEN);
    output[16] = toksvig_sanitize_shininess(TOKSVIG_MAX_SHININESS);
    output[17] = SPEC_AA_MIN_ALPHA;
}
"#;

/// Links the shared module through a real compute entry point, exercising every
/// exported helper so none can disappear behind dead-code elimination.
#[test]
fn specular_aa_wesl_compiles_and_links_every_helper() {
    let mut cache = ShaderCache::new((), load_source);

    let module = shader_id(0x5052_4953_4d5f_5350_4543_4141_0001);
    cache.set_shader(
        module,
        Shader::from_wesl(
            include_str!("../../shaders/specular_aa.wesl"),
            "embedded://prism_render_scene/shaders/specular_aa.wesl",
        ),
    );

    let link_test = shader_id(0x5052_4953_4d5f_5350_4543_4141_0002);
    cache.set_shader(
        link_test,
        Shader::from_wesl(
            LINK_TEST_SOURCE,
            "embedded://prism_render_scene/shading/specular_aa/link_test.wesl",
        ),
    );

    cache.get(0, link_test, &[]).unwrap_or_else(|error| {
        panic!("specular_aa.wesl failed to compile/link every helper: {error}")
    });
}
