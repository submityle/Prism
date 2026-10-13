//! WESL link coverage for the shared procedural glint / sparkle helper module.
//!
//! Like the thin-film and specular-AA link tests, this builds a tiny compute
//! entry point that references every exported helper, so import visibility,
//! signatures, and type-checking are all exercised rather than merely parsing
//! the source. Numeric fidelity against the CPU golden (`glint.rs`) is
//! guaranteed by exact mirroring of the arithmetic order, clamps, floors, the
//! `triple32` avalanche hash, and the Knuth/Gaussian Poisson branches; this
//! test only proves the GPU twin compiles and links its full public surface.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("glint shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

const LINK_TEST_SOURCE: &str = r#"
import prism_render_scene::shaders::glint::{
    GLINT_PI, GLINT_MIN_ALPHA, GLINT_MIN_COS, GLINT_MIN_POSITIVE,
    GLINT_KNUTH_MAX_LAMBDA, GLINT_POISSON_MAX_ITERS, GLINT_SEED_SALT,
    glint_is_finite,
    glint_hash_u32,
    glint_hash_combine,
    glint_uniform_from_hash,
    glint_uniform_at,
    glint_quantize_axis,
    glint_half_slope,
    glint_seed,
    glint_bin_solid_angle,
    glint_ggx_ndf,
    glint_smith_g1,
    glint_smith_g2_height_correlated,
    glint_fresnel_schlick,
    glint_microfacet_count,
    glint_poisson_knuth,
    glint_poisson_gaussian,
    glint_poisson_sample,
    glint_ndf,
    glint_ndf_blended,
    glint_specular_brdf,
};

@group(0) @binding(0)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn glint_link_test(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) {
        return;
    }

    let alpha = 0.3;
    let n_dot_h = 0.95;
    let n_dot_v = 0.8;
    let n_dot_l = 0.6;

    let h0 = glint_hash_u32(12345u);
    let h1 = glint_hash_combine(h0, 678u);
    let u0 = glint_uniform_from_hash(h1);
    let u1 = glint_uniform_at(h1, 3u);

    let cell = glint_quantize_axis(1.37, 64.0);
    let slope = glint_half_slope(0.1, n_dot_h);
    let seed = glint_seed(0.25, -0.75, 64.0, slope, 0.2, 256.0, 2u);

    let dw = glint_bin_solid_angle(0.02);
    let n = glint_microfacet_count(5.0e3, 1.0e-2);

    let d_smooth = glint_ggx_ndf(n_dot_h, alpha);
    let g1 = glint_smith_g1(n_dot_v, alpha);
    let g2 = glint_smith_g2_height_correlated(n_dot_v, n_dot_l, alpha);
    let fr = glint_fresnel_schlick(0.04, n_dot_v);

    let lambda = max(d_smooth * dw * n, 0.0);
    let k_knuth = glint_poisson_knuth(min(lambda, GLINT_KNUTH_MAX_LAMBDA), seed);
    let k_gauss = glint_poisson_gaussian(lambda, seed);
    let k = glint_poisson_sample(lambda, seed);

    let d_glint = glint_ndf(n_dot_h, alpha, n, dw, seed);
    let d_blend = glint_ndf_blended(n_dot_h, alpha, n, dw, 0.5, seed);
    let f_r = glint_specular_brdf(
        n_dot_v, n_dot_l, n_dot_h, alpha, fr, n, dw, 0.5, seed,
    );

    output[0] = f32(h0);
    output[1] = f32(h1);
    output[2] = u0;
    output[3] = u1;
    output[4] = f32(cell);
    output[5] = slope;
    output[6] = f32(seed);
    output[7] = dw;
    output[8] = n;
    output[9] = d_smooth;
    output[10] = g1;
    output[11] = g2;
    output[12] = fr;
    output[13] = k_knuth;
    output[14] = k_gauss;
    output[15] = k;
    output[16] = d_glint;
    output[17] = d_blend;
    output[18] = f_r;
    output[19] = select(0.0, 1.0, glint_is_finite(f_r));
    output[20] = GLINT_PI;
    output[21] = GLINT_MIN_ALPHA;
    output[22] = GLINT_MIN_COS;
    output[23] = GLINT_MIN_POSITIVE;
    output[24] = GLINT_KNUTH_MAX_LAMBDA;
    output[25] = f32(GLINT_POISSON_MAX_ITERS);
    output[26] = f32(GLINT_SEED_SALT);
}
"#;

/// Links the shared module through a real compute entry point, exercising every
/// exported helper so none can disappear behind dead-code elimination.
#[test]
fn glint_wesl_compiles_and_links_every_helper() {
    let mut cache = ShaderCache::new((), load_source);

    let module = shader_id(0x5052_4953_4d5f_474c_494e_545f_0001);
    cache.set_shader(
        module,
        Shader::from_wesl(
            include_str!("../../shaders/glint.wesl"),
            "embedded://prism_render_scene/shaders/glint.wesl",
        ),
    );

    let link_test = shader_id(0x5052_4953_4d5f_474c_494e_545f_0002);
    cache.set_shader(
        link_test,
        Shader::from_wesl(
            LINK_TEST_SOURCE,
            "embedded://prism_render_scene/shading/glint/link_test.wesl",
        ),
    );

    cache
        .get(0, link_test, &[])
        .unwrap_or_else(|error| panic!("glint.wesl failed to compile/link every helper: {error}"));
}
