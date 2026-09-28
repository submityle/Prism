//! WESL compilation coverage for the tone-map shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `tonemap.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `exposure.wesl`
//! / `ssgi.wesl`), so a green result also guards the tone-map maths — Reinhard
//! and its extended white-point variant, the Narkowicz and Hill ACES fits and
//! the minimal AgX pipeline — against drift from its CPU golden twin in
//! `prism_render_shading::tonemap`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("tonemap shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `tonemap.wesl`, proving the display tone-mapping kernel parses and
/// type-checks exactly as it will in the render world (pre-exposed linear HDR
/// in; display-referred `[0, 1]` colour out), and that its operators match the
/// CPU golden.
#[test]
fn tonemap_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let tonemap = shader_id(0x5052_4953_4d5f_544f_4e45_4d41_5000_0001);
    cache.set_shader(
        tonemap,
        Shader::from_wesl(
            include_str!("../../shaders/tonemap.wesl"),
            "embedded://prism_render_scene/shaders/tonemap.wesl",
        ),
    );

    cache
        .get(0, tonemap, &[])
        .unwrap_or_else(|error| panic!("tonemap.wesl failed to compile: {error}"));
}
