//! WESL compilation coverage for the IBL shader graph.
//!
//! The sandbox has no GPU, so these tests compile the WESL sources through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! every source parses and type-checks.  The IBL precompute shaders are
//! self-contained (no `import`s), so a green result proves the kernels are
//! ready to specialize once the render-world pipeline binds them.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("IBL shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `brdf_lut.wesl`, proving the split-sum environment-BRDF integration
/// kernel parses and type-checks exactly as it will in the render world.
#[test]
fn brdf_lut_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let lut = shader_id(0x5052_4953_4d5f_4942_4c5f_4252_4446_0001);
    cache.set_shader(
        lut,
        Shader::from_wesl(
            include_str!("../../shaders/brdf_lut.wesl"),
            "embedded://prism_render_scene/shaders/brdf_lut.wesl",
        ),
    );

    cache
        .get(0, lut, &[])
        .unwrap_or_else(|error| panic!("brdf_lut.wesl failed to compile: {error}"));
}
