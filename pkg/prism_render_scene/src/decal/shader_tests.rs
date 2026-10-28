//! Standalone production WESL compilation coverage for the Decal kernel.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("Decal shader is WESL"),
    }
}

#[test]
fn decal_resolve_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_4445_4341_4c5f_0001),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/decal_resolve.wesl"),
            "embedded://prism_render_scene/shaders/decal_resolve.wesl",
        ),
    );
    cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("decal_resolve.wesl failed to compile: {error}"));
}
