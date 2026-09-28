//! WESL compilation coverage for the vignette shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `vignette.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `bloom.wesl` / `color_grade.wesl`), so a green result also guards the
//! vignette maths — the physical `cos^4` optical fall-off and the artist
//! `smoothstep` vignette — against drift from its CPU golden twin in
//! `prism_render_shading::vignette`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("vignette shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `vignette.wesl`, proving the vignette kernel parses and type-checks
/// exactly as it will in the render world (pre-exposed HDR radiance and the
/// artist params in; the darkened radiance out), and that the natural / artistic
/// fall-off layouts match the CPU golden.
#[test]
fn vignette_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let vignette = shader_id(0x5052_4953_4d5f_0000_5649_474e_4554_5445);
    cache.set_shader(
        vignette,
        Shader::from_wesl(
            include_str!("../../shaders/vignette.wesl"),
            "embedded://prism_render_scene/shaders/vignette.wesl",
        ),
    );

    cache
        .get(0, vignette, &[])
        .unwrap_or_else(|error| panic!("vignette.wesl failed to compile: {error}"));
}
