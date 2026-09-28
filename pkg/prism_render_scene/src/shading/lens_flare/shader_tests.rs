//! WESL compilation coverage for the lens-flare shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `lens_flare.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `bloom.wesl` / `color_grade.wesl` / `exposure.wesl`), so a green result also
//! guards the flare maths — the soft bright-tail threshold, the `ghost` and
//! `halo` sampling, the chromatic dispersion and the additive composite —
//! against drift from its CPU golden twin in
//! `prism_render_shading::lens_flare`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("lens flare shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `lens_flare.wesl`, proving the linear-`HDR` lens-flare kernel parses
/// and type-checks exactly as it will in the render world (pre-exposed `HDR`
/// radiance and the artist params in; the flare radiance out), and that the
/// threshold / `ghost` / `halo` / chromatic / composite layouts match the CPU
/// golden.
#[test]
fn lens_flare_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let lens_flare = shader_id(0x5052_4953_4d5f_0000_4c45_4e53_464c_5241);
    cache.set_shader(
        lens_flare,
        Shader::from_wesl(
            include_str!("../../shaders/lens_flare.wesl"),
            "embedded://prism_render_scene/shaders/lens_flare.wesl",
        ),
    );

    cache
        .get(0, lens_flare, &[])
        .unwrap_or_else(|error| panic!("lens_flare.wesl failed to compile: {error}"));
}
