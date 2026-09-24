//! WESL compilation coverage for the composite shader.
//!
//! The sandbox has no GPU, so this test does **not** exercise the pass at
//! runtime.  It compiles `composite.wesl` through the same `ShaderCache` /
//! `wesl` pipeline the render world uses, which validates that the fullscreen
//! `vertex`/`fragment` entry points, the group-0 texture bindings and the
//! background-sentinel discard all parse and type-check as WESL.
//!
//! `composite.wesl` has no `import` statements, so a green result also proves it
//! stands alone and needs none of the shading-resolve module graph registered.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("composite shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `composite.wesl` standalone.  It has no imports, so a green result
/// proves the fullscreen triangle generator, the `textureLoad`-only group-0
/// bindings and the covered-pixel discard all parse and type-check on their own.
#[test]
fn composite_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let composite = shader_id(0x5052_4953_4d5f_434f_4d50_4f53_4954_0001);
    cache.set_shader(
        composite,
        Shader::from_wesl(
            include_str!("../../shaders/composite.wesl"),
            "embedded://prism_render_scene/shaders/composite.wesl",
        ),
    );

    cache
        .get(0, composite, &[])
        .unwrap_or_else(|error| panic!("composite.wesl failed to compile: {error}"));
}
