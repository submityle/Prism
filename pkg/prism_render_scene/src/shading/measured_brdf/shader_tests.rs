//! WESL link coverage for the shared measured / tabulated BRDF helper module.
//!
//! Like the thin-film link test, this builds a tiny compute entry point that
//! references every exported helper, so import visibility, signatures, and
//! type-checking are all exercised rather than merely parsing the source.
//! Numeric fidelity is guaranteed by exact mirroring of the CPU golden's
//! operation order, clamps, warps, and degenerate-input fallbacks
//! (`prism_render_shading/src/measured_brdf.rs`): the Rusinkiewicz coordinate
//! transform, the square-root half-elevation warp, the continuous grid mapping,
//! trilinear interpolation, and the analytic synthesiser.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("measured-brdf shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

const LINK_TEST_SOURCE: &str = r#"
import prism_render_scene::shaders::measured_brdf::{
    MB_PI, MB_FRAC_PI_2, MB_TAU, MB_MIN_COS, MB_MIN_LEN_SQ, MB_MIN_ROUGHNESS, MB_MIN_POSITIVE,
    mb_is_finite,
    mb_clamp01,
    mb_pow5,
    mb_lerp3,
    mb_normalize_or,
    mb_normalize_or_zero,
    mb_half_vector,
    mb_rotate_about_z,
    mb_rotate_about_y,
    mb_wrap_tau,
    mb_fold_phi_d,
    mb_theta_h_to_unit,
    mb_unit_to_theta_h,
    mb_directions_to_rusinkiewicz,
    mb_rusinkiewicz_wi,
    mb_rusinkiewicz_wo,
    mb_unit_to_grid,
    mb_grid_lower,
    mb_grid_frac,
    mb_grid_coord_theta_h,
    mb_grid_coord_theta_d,
    mb_grid_coord_phi_d,
    mb_trilinear,
    mb_scalar_brdf,
    mb_analytic_brdf,
};

@group(0) @binding(0)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn measured_brdf_link_test(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) {
        return;
    }
    let n = vec3<f32>(0.0, 0.0, 1.0);
    let wi = mb_normalize_or(vec3<f32>(0.3, 0.1, 0.9), n);
    let wo = mb_normalize_or_zero(vec3<f32>(-0.2, 0.2, 0.8));
    let h = mb_half_vector(wi, wo);

    let coords = mb_directions_to_rusinkiewicz(wi, wo);
    let wi_rt = mb_rusinkiewicz_wi(coords);
    let wo_rt = mb_rusinkiewicz_wo(coords);

    let nh = 8.0;
    let nd = 8.0;
    let np = 16.0;
    let g_h = mb_grid_coord_theta_h(coords.x, nh);
    let g_d = mb_grid_coord_theta_d(coords.y, nd);
    let g_p = mb_grid_coord_phi_d(coords.z, np);
    let fi = mb_grid_frac(g_h, nh);
    let fj = mb_grid_frac(g_d, nd);
    let fk = mb_grid_frac(g_p, np);
    let lo = mb_grid_lower(g_h, nh);

    let base = vec3<f32>(0.82, 0.67, 0.16);
    let c000 = mb_analytic_brdf(wi, wo, base, 0.35, 1.0, 0.5);
    let c = mb_trilinear(c000, c000, c000, c000, c000, c000, c000, c000, fi, fj, fk);

    let u = mb_theta_h_to_unit(coords.x);
    let th = mb_unit_to_theta_h(u);
    let gg = mb_unit_to_grid(u, nh);
    let wrapped = mb_wrap_tau(coords.z + MB_TAU);
    let folded = mb_fold_phi_d(coords.z);
    let scalar = mb_scalar_brdf(c);

    output[0] = coords.x;
    output[1] = coords.y;
    output[2] = coords.z;
    output[3] = h.z;
    output[4] = wi_rt.z;
    output[5] = wo_rt.z;
    output[6] = g_h;
    output[7] = g_d;
    output[8] = g_p;
    output[9] = fi;
    output[10] = fj;
    output[11] = fk;
    output[12] = lo;
    output[13] = c.x;
    output[14] = c.y;
    output[15] = c.z;
    output[16] = u;
    output[17] = th;
    output[18] = gg;
    output[19] = wrapped;
    output[20] = folded;
    output[21] = scalar;
    output[22] = mb_pow5(0.5) + mb_clamp01(2.0);
    output[23] = mb_lerp3(base, base, 0.5).x;
    output[24] = select(0.0, 1.0, mb_is_finite(scalar));
    output[25] = MB_PI + MB_FRAC_PI_2 + MB_TAU + MB_MIN_COS + MB_MIN_LEN_SQ + MB_MIN_ROUGHNESS + MB_MIN_POSITIVE;
    output[26] = mb_rotate_about_z(wi, 0.5).x + mb_rotate_about_y(wo, 0.5).z;
}
"#;

/// Links the shared module through a real compute entry point, exercising every
/// exported helper so none can disappear behind dead-code elimination.
#[test]
fn measured_brdf_wesl_compiles_and_links_every_helper() {
    let mut cache = ShaderCache::new((), load_source);

    let module = shader_id(0x5052_4953_4d5f_4d42_5244_465f_0001);
    cache.set_shader(
        module,
        Shader::from_wesl(
            include_str!("../../shaders/measured_brdf.wesl"),
            "embedded://prism_render_scene/shaders/measured_brdf.wesl",
        ),
    );

    let link_test = shader_id(0x5052_4953_4d5f_4d42_5244_465f_0002);
    cache.set_shader(
        link_test,
        Shader::from_wesl(
            LINK_TEST_SOURCE,
            "embedded://prism_render_scene/shading/measured_brdf/link_test.wesl",
        ),
    );

    cache.get(0, link_test, &[]).unwrap_or_else(|error| {
        panic!("measured_brdf.wesl failed to compile/link every helper: {error}")
    });
}
