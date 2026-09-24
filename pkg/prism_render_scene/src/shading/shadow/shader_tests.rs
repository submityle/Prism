//! WESL compilation coverage for the shadow-visibility shader.
//!
//! The sandbox has no GPU, so this test does **not** exercise any shadow pass
//! at runtime.  It compiles `shadow.wesl` through the same `ShaderCache` /
//! `wesl` pipeline the render world uses, which validates that the column-major
//! projection, the PSSM cascade split/select/blend, the normal-offset and
//! slope-scaled bias, the manual PCF/PCSS filters and the directional/point
//! orchestrators all parse and type-check as WESL — and that the atlas array
//! texture bindings and the whole `shadow_selftest` call graph are valid.
//!
//! `shadow.wesl` has no `import` statements, so a green result also proves it
//! stands alone and needs none of the shading-resolve module graph registered.
//!
//! Numerical parity against `prism_render_shading::shadow` is guaranteed by
//! construction (each function mirrors its CPU twin line for line) but must be
//! confirmed on real hardware once the atlas-filling passes and resolve wiring
//! land, since the sandbox cannot run the shader.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("shadow shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `shadow.wesl` standalone.  It has no imports, so a green result
/// proves the cascade math, bias helpers, PCF/PCSS filters, directional and
/// point orchestrators, the array-texture atlas bindings and the
/// `shadow_selftest` compute entry all parse and type-check on their own.
#[test]
fn shadow_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let shadow = shader_id(0x5052_4953_4d5f_5348_4144_4f57_0000_0001);
    cache.set_shader(
        shadow,
        Shader::from_wesl(
            include_str!("../../shaders/shadow.wesl"),
            "embedded://prism_render_scene/shaders/shadow.wesl",
        ),
    );

    cache
        .get(0, shadow, &[])
        .unwrap_or_else(|error| panic!("shadow.wesl failed to compile: {error}"));
}
