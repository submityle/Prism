//! WESL compilation coverage for the gamut-map shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `gamut_map.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `bloom.wesl` / `exposure.wesl` / `color_grade.wesl`), so a green result also
//! guards the gamut-mapping maths — the achromatic anchor, the relative
//! distance, the rational distance-compression knee and the reconstruction —
//! against drift from its CPU golden twin in
//! `prism_render_shading::gamut_map`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("gamut map shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `gamut_map.wesl`, proving the linear-HDR gamut-map kernel parses and
/// type-checks exactly as it will in the render world (pre-exposed HDR radiance
/// and the artist params in; the gamut-compressed radiance out), and that the
/// anchor / distance / compression / reconstruction layout matches the CPU
/// golden.
#[test]
fn gamut_map_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let gamut_map = shader_id(0x5052_4953_4d5f_0000_4741_4d55_5447_4d50);
    cache.set_shader(
        gamut_map,
        Shader::from_wesl(
            include_str!("../../shaders/gamut_map.wesl"),
            "embedded://prism_render_scene/shaders/gamut_map.wesl",
        ),
    );

    cache
        .get(0, gamut_map, &[])
        .unwrap_or_else(|error| panic!("gamut_map.wesl failed to compile: {error}"));
}
