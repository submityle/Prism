//! WESL link coverage for the ray-cone texture-LOD GPU twin.
//!
//! This mirrors the CPU golden in `prism_render_material::texture_lod`. Rather
//! than merely parsing the module, it links a real compute entry point that
//! references every exported helper and constant, so import visibility,
//! signatures, and type-checking are all exercised. Numeric fidelity is
//! anchored by exact mirroring of the CPU golden's operation order, clamps,
//! floors, and finite-sanitization branches (see `texture_lod.wesl`).

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("texture-LOD shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

const LINK_TEST_SOURCE: &str = r#"
import prism_render_scene::shaders::texture_lod::{
    TEX_LOD_MIN_AREA,
    TEX_LOD_DELTA_CLAMP,
    TEX_LOD_MIN_COS_INCIDENCE,
    TEX_LOD_F32_MIN_POSITIVE,
    TEX_LOD_MAX_SPREAD_ANGLE,
    RayCone,
    tex_lod_dot3,
    tex_lod_sub3,
    tex_lod_cross3,
    tex_lod_length3,
    tex_lod_uv_double_area,
    tex_lod_triangle_delta,
    tex_lod_mip_from_isotropic_footprint,
    tex_lod_cone_mip_level,
    tex_lod_ray_cone_new,
    tex_lod_ray_cone_from_pinhole_pixel,
    tex_lod_ray_cone_advanced,
    tex_lod_ray_cone_scattered,
    tex_lod_ray_cone_reflected,
    tex_lod_is_finite,
    RayDifferential,
    AnisotropicMip,
    tex_lod_ray_diff_axis_lengths,
    tex_lod_ray_diff_isotropic_mip,
    tex_lod_ray_diff_major_axis_uv,
    tex_lod_ray_diff_anisotropic_mip,
    TEX_LOD_MAX_ANISO_TAPS,
    tex_lod_aniso_tap_count,
    tex_lod_aniso_tap_weight,
    tex_lod_aniso_tap_uv,
};

@group(0) @binding(0)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn texture_lod_link_test(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) {
        return;
    }

    let w0 = vec3<f32>(0.0, 0.0, 0.0);
    let w1 = vec3<f32>(1.0, 0.0, 0.0);
    let w2 = vec3<f32>(0.0, 1.0, 0.0);
    let uv0 = vec2<f32>(0.0, 0.0);
    let uv1 = vec2<f32>(1.0, 0.0);
    let uv2 = vec2<f32>(0.0, 1.0);

    let d = tex_lod_dot3(w1, w2);
    let s = tex_lod_sub3(w1, w0);
    let c = tex_lod_cross3(s, tex_lod_sub3(w2, w0));
    let len = tex_lod_length3(c);
    let uv_area = tex_lod_uv_double_area(uv0, uv1, uv2);
    let delta = tex_lod_triangle_delta(w0, w1, w2, uv0, uv1, uv2, 256.0, 256.0);
    let iso_mip = tex_lod_mip_from_isotropic_footprint(4.0, 12.0);
    let cone_mip = tex_lod_cone_mip_level(delta, 2.0, 0.5, 12.0);

    var cone: RayCone = tex_lod_ray_cone_from_pinhole_pixel(TEX_LOD_MAX_SPREAD_ANGLE, 1080.0);
    cone = tex_lod_ray_cone_advanced(cone, 10.0);
    cone = tex_lod_ray_cone_scattered(cone, 0.01, 0.02);
    cone = tex_lod_ray_cone_reflected(cone, 0.02, 0.0);
    let explicit_cone = tex_lod_ray_cone_new(0.1, 0.01);

    output[0] = d;
    output[1] = s.x;
    output[2] = len;
    output[3] = uv_area;
    output[4] = delta;
    output[5] = iso_mip;
    output[6] = cone_mip;
    output[7] = cone.width;
    output[8] = cone.spread_angle;
    output[9] = explicit_cone.width;
    output[10] = explicit_cone.spread_angle;
    output[11] = TEX_LOD_MIN_AREA;
    output[12] = TEX_LOD_DELTA_CLAMP;
    output[13] = TEX_LOD_MIN_COS_INCIDENCE;
    output[14] = TEX_LOD_F32_MIN_POSITIVE;

    let rd = RayDifferential(vec2<f32>(2.0 / 256.0, 0.0), vec2<f32>(0.0, 1.0 / 256.0));
    let lengths = tex_lod_ray_diff_axis_lengths(rd.d_dx, rd.d_dy, 256.0, 256.0);
    let iso = tex_lod_ray_diff_isotropic_mip(rd.d_dx, rd.d_dy, 256.0, 256.0, 8.0);
    let major = tex_lod_ray_diff_major_axis_uv(rd.d_dx, rd.d_dy, 256.0, 256.0);
    let aniso: AnisotropicMip = tex_lod_ray_diff_anisotropic_mip(rd.d_dx, rd.d_dy, 256.0, 256.0, 8.0, 16.0);
    let n = tex_lod_aniso_tap_count(aniso.anisotropy);
    let w = tex_lod_aniso_tap_weight(n);
    let tap = tex_lod_aniso_tap_uv(vec2<f32>(0.5, 0.5), major, 0u, n);

    output[15] = lengths.x;
    output[16] = iso;
    output[17] = major.x;
    output[18] = aniso.lod;
    output[19] = aniso.anisotropy;
    output[20] = w;
    output[21] = tap.x;
    output[22] = f32(n);
    output[23] = f32(TEX_LOD_MAX_ANISO_TAPS);
    output[24] = select(0.0, 1.0, tex_lod_is_finite(iso));
}
"#;

/// Links the twin module through a real compute entry point, forcing every
/// exported helper and constant to resolve, type-check, and survive dead-code
/// elimination.
#[test]
fn texture_lod_wesl_compiles_and_links_every_helper() {
    let mut cache = ShaderCache::new((), load_source);

    let module = shader_id(0x5052_4953_4d5f_5445_584c_4f44_0001);
    cache.set_shader(
        module,
        Shader::from_wesl(
            include_str!("../src/shaders/texture_lod.wesl"),
            "embedded://prism_render_scene/shaders/texture_lod.wesl",
        ),
    );

    let link_test = shader_id(0x5052_4953_4d5f_5445_584c_4f44_0002);
    cache.set_shader(
        link_test,
        Shader::from_wesl(
            LINK_TEST_SOURCE,
            "embedded://prism_render_scene/tests/texture_lod_link_test.wesl",
        ),
    );

    cache.get(0, link_test, &[]).unwrap_or_else(|error| {
        panic!("texture_lod.wesl failed to compile/link every helper: {error}")
    });
}
