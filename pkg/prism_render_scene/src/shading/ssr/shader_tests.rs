//! WESL compilation coverage for the SSR trace kernel.
//!
//! The sandbox has no GPU, so this test compiles `ssr.wesl` through the same
//! `ShaderCache` / `wesl` pipeline the render world uses, validating that the
//! hierarchical march, the reflection-ray setup and every confidence fade
//! parse and type-check exactly as they will once the render-world pipeline
//! binds them.  The kernel is self-contained (no `import`s), matching the
//! `gtao.wesl` / `brdf_lut.wesl` precedent.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("SSR shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `ssr.wesl`, proving the screen-space reflection trace kernel parses
/// and type-checks exactly as it will in the render world (HZB pyramid, scene
/// depth, normal/roughness and previous-frame colour in; reflected radiance and
/// confidence out).
#[test]
fn ssr_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let ssr = shader_id(0x5052_4953_4d5f_5353_525f_5452_4143_0001);
    cache.set_shader(
        ssr,
        Shader::from_wesl(
            include_str!("../../shaders/ssr.wesl"),
            "embedded://prism_render_scene/shaders/ssr.wesl",
        ),
    );

    cache
        .get(0, ssr, &[])
        .unwrap_or_else(|error| panic!("ssr.wesl failed to compile: {error}"));
}
