//! WESL compile coverage. The kernel is a scalar-for-scalar GPU twin of the
//! CPU golden `transmittance::transmittance_to_boundary`, so this test guards
//! numeric fidelity's shader-side implementation (Earth constants, altitude
//! parameterization, midpoint optical-depth march and exp) at compile time.
use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
fn load(_: &(), s: ShaderCacheSource, _: &ValidateShader) -> Result<String, ShaderCacheError> {
    match s {
        ShaderCacheSource::Wgsl(x) => Ok(x),
        ShaderCacheSource::SpirV(_) => unreachable!(),
    }
}
#[test]
fn sky_transmittance_lut_wesl_compiles_and_type_checks() {
    let mut c = ShaderCache::new((), load);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x505249534d5f534b595f54524c555401),
    };
    c.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../../shaders/sky_transmittance_lut.wesl"),
            "embedded://prism_render_scene/shaders/sky_transmittance_lut.wesl",
        ),
    );
    c.get(0, id, &[])
        .unwrap_or_else(|e| panic!("sky transmittance LUT failed: {e}"));
}
