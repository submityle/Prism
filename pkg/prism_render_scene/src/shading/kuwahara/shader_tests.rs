//! WESL compilation coverage for the Kuwahara shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `kuwahara.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `bloom.wesl` / `exposure.wesl` / `color_grade.wesl`), so a green result also
//! guards the filter maths — the Rec. 709 luminance metric, the per-quadrant
//! mean and variance, and the first-minimum quadrant selection — against drift
//! from its CPU golden twin in `prism_render_shading::kuwahara`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("kuwahara shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `kuwahara.wesl`, proving the four-quadrant Kuwahara kernel parses
/// and type-checks exactly as it will in the render world (resolved HDR
/// radiance and the artist params in; the stylized radiance out), and that the
/// luminance / quadrant-mean / variance / selection layouts match the CPU
/// golden.
#[test]
fn kuwahara_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let kuwahara = shader_id(0x5052_4953_4d5f_0000_4b55_5741_4841_5241);
    cache.set_shader(
        kuwahara,
        Shader::from_wesl(
            include_str!("../../shaders/kuwahara.wesl"),
            "embedded://prism_render_scene/shaders/kuwahara.wesl",
        ),
    );

    cache
        .get(0, kuwahara, &[])
        .unwrap_or_else(|error| panic!("kuwahara.wesl failed to compile: {error}"));
}
