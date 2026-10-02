//! WESL compile coverage. The kernel is a scalar-for-scalar GPU twin of the
//! CPU golden `scattering::sky_view_radiance` (single scatter plus LUT-sampled
//! multiple scatter per march step), so this test guards the shader-side
//! implementation (constants, boundary roots, phases, LUT sampling and march
//! order) at compile time.
use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
fn load(_: &(), s: ShaderCacheSource, _: &ValidateShader) -> Result<String, ShaderCacheError> {
    match s {
        ShaderCacheSource::Wgsl(x) => Ok(x),
        ShaderCacheSource::SpirV(_) => unreachable!(),
    }
}
#[test]
fn sky_view_lut_wesl_compiles_and_type_checks() {
    let mut c = ShaderCache::new((), load);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x505249534d5f534b595f53564c555401),
    };
    c.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../../shaders/sky_view_lut.wesl"),
            "embedded://prism_render_scene/shaders/sky_view_lut.wesl",
        ),
    );
    c.get(0, id, &[])
        .unwrap_or_else(|e| panic!("sky-view LUT failed: {e}"));
}
