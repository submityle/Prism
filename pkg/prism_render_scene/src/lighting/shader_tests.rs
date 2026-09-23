//! Standalone WESL compilation coverage for the lighting shader library.
//!
//! `lighting.wesl` is intentionally import-free so it can be validated in
//! isolation here; the same source is imported by the shading resolve pass.
//! Compiling it through [`ShaderCache`] guarantees the analytic light math and
//! the ABI `struct` layouts stay syntactically valid as the resolve pass and
//! CPU golden reference evolve together.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheSource};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &bevy_shader::ValidateShader,
) -> Result<String, bevy_shader::ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("lighting shader is WESL"),
    }
}

#[test]
fn lighting_wesl_compiles_standalone() {
    let shader_id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d4c_4947_4854_494e_4700_0001),
    };
    let mut cache = ShaderCache::new((), load_source);
    cache.set_shader(
        shader_id,
        Shader::from_wesl(
            include_str!("../shaders/lighting.wesl"),
            "shaders/prism_lighting.wesl",
        ),
    );
    cache
        .get(0, shader_id, &[])
        .unwrap_or_else(|error| panic!("lighting shader failed to compile: {error}"));
}
