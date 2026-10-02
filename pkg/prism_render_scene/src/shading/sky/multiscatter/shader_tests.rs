//! WESL compile coverage. The kernel is a scalar-for-scalar GPU twin of the
//! CPU golden `multiscatter_estimate`, so this test guards numeric fidelity's
//! shader-side implementation (constants, sample order and march) at compile time.
use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
fn load(_: &(), s: ShaderCacheSource, _: &ValidateShader) -> Result<String, ShaderCacheError> {
    match s {
        ShaderCacheSource::Wgsl(x) => Ok(x),
        ShaderCacheSource::SpirV(_) => unreachable!(),
    }
}
#[test]
fn sky_multiscatter_lut_wesl_compiles_and_type_checks() {
    let mut c = ShaderCache::new((), load);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x505249534d5f534b595f4d534c555401),
    };
    c.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../../shaders/sky_multiscatter_lut.wesl"),
            "embedded://prism_render_scene/shaders/sky_multiscatter_lut.wesl",
        ),
    );
    c.get(0, id, &[])
        .unwrap_or_else(|e| panic!("sky LUT failed: {e}"));
}
