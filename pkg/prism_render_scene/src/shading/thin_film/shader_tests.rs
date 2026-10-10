//! WESL link coverage for the shared thin-film iridescence helper module.
//!
//! Like the specular-AA link test, this builds a tiny compute entry point that
//! references every exported helper, so import visibility, signatures, and
//! type-checking are all exercised rather than merely parsing the source.
//! Numeric fidelity is guaranteed by exact mirroring of the CPU golden's
//! operation order, clamps, floors, and total-internal-reflection branches
//! (`gi/material/thin_film.rs`), where the Rust `Option<cos_t>` result is
//! encoded as a negative sentinel.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("thin-film shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

const LINK_TEST_SOURCE: &str = r#"
import prism_render_scene::shaders::thin_film::{
    TF_WAVELENGTH_R_NM, TF_WAVELENGTH_G_NM, TF_WAVELENGTH_B_NM,
    TF_MIN_COS, TF_MIN_IOR, TF_MIN_ALPHA, TF_MIN_POSITIVE, TF_PI,
    tf_is_finite,
    tf_clamp_ior,
    tf_clamp_cos,
    tf_transmitted_cos,
    tf_fresnel_amplitudes,
    tf_fresnel_dielectric_unpolarized,
    tf_optical_phase,
    tf_airy_one,
    tf_airy_reflectance,
    tf_iridescent_reflectance_rgb,
    tf_normalize_or,
    tf_anisotropic_alphas,
    tf_ggx_aniso_ndf,
    tf_smith_g1_aniso,
};

@group(0) @binding(0)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn thin_film_link_test(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) {
        return;
    }
    let n = vec3<f32>(0.0, 0.0, 1.0);
    let t = vec3<f32>(1.0, 0.0, 0.0);
    let b = vec3<f32>(0.0, 1.0, 0.0);
    let h = tf_normalize_or(vec3<f32>(0.1, 0.0, 1.0), n);

    let ior = tf_clamp_ior(1.5);
    let c = tf_clamp_cos(0.8);
    let cos_t = tf_transmitted_cos(c, TF_MIN_IOR, ior);
    let amps = tf_fresnel_amplitudes(c, max(cos_t, 0.0), TF_MIN_IOR, ior);
    let fr = tf_fresnel_dielectric_unpolarized(c, TF_MIN_IOR, ior);
    let phi = tf_optical_phase(ior, c, 300.0, TF_WAVELENGTH_G_NM);
    let airy = tf_airy_one(amps.x, amps.y, cos(phi));
    let r = tf_airy_reflectance(TF_MIN_IOR, ior, 1.7, c, 300.0, TF_WAVELENGTH_R_NM);
    let rgb = tf_iridescent_reflectance_rgb(TF_MIN_IOR, ior, 1.7, c, 300.0);
    let alphas = tf_anisotropic_alphas(0.4, 0.3);
    let ndf = tf_ggx_aniso_ndf(h, t, b, n, alphas.x, alphas.y);
    let g1 = tf_smith_g1_aniso(h, t, b, n, alphas.x, alphas.y);

    output[0] = cos_t;
    output[1] = amps.x;
    output[2] = amps.y;
    output[3] = fr;
    output[4] = phi;
    output[5] = airy;
    output[6] = r;
    output[7] = rgb.x;
    output[8] = rgb.y;
    output[9] = rgb.z;
    output[10] = alphas.x;
    output[11] = alphas.y;
    output[12] = ndf;
    output[13] = g1;
    output[14] = select(0.0, 1.0, tf_is_finite(fr));
    output[15] = TF_MIN_COS;
    output[16] = TF_MIN_ALPHA;
    output[17] = TF_MIN_POSITIVE;
    output[18] = TF_PI;
    output[19] = TF_WAVELENGTH_B_NM;
}
"#;

/// Links the shared module through a real compute entry point, exercising every
/// exported helper so none can disappear behind dead-code elimination.
#[test]
fn thin_film_wesl_compiles_and_links_every_helper() {
    let mut cache = ShaderCache::new((), load_source);

    let module = shader_id(0x5052_4953_4d5f_5446_494c_4d5f_0001);
    cache.set_shader(
        module,
        Shader::from_wesl(
            include_str!("../../shaders/thin_film.wesl"),
            "embedded://prism_render_scene/shaders/thin_film.wesl",
        ),
    );

    let link_test = shader_id(0x5052_4953_4d5f_5446_494c_4d5f_0002);
    cache.set_shader(
        link_test,
        Shader::from_wesl(
            LINK_TEST_SOURCE,
            "embedded://prism_render_scene/shading/thin_film/link_test.wesl",
        ),
    );

    cache.get(0, link_test, &[]).unwrap_or_else(|error| {
        panic!("thin_film.wesl failed to compile/link every helper: {error}")
    });
}
