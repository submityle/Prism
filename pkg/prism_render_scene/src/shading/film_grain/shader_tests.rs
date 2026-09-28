//! WESL compilation coverage for the film-grain shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `film_grain.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `color_grade.wesl`
//! / `bloom.wesl` / `exposure.wesl`), so a green result also guards the grain
//! maths — the *Hash without Sine* noise, the shadow-biased luminance response
//! and the additive intensity mix — against drift from its CPU golden twin in
//! `prism_render_shading::film_grain`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("film grain shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `film_grain.wesl`, proving the linear-HDR film-grain kernel parses
/// and type-checks exactly as it will in the render world (pre-exposed HDR
/// radiance and the artist params in; the grained radiance out), and that the
/// hash-noise / luminance-response / apply layouts match the CPU golden.
#[test]
fn film_grain_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let film_grain = shader_id(0x5052_4953_4d5f_0000_4649_4c4d_4752_414e);
    cache.set_shader(
        film_grain,
        Shader::from_wesl(
            include_str!("../../shaders/film_grain.wesl"),
            "embedded://prism_render_scene/shaders/film_grain.wesl",
        ),
    );

    cache
        .get(0, film_grain, &[])
        .unwrap_or_else(|error| panic!("film_grain.wesl failed to compile: {error}"));
}
