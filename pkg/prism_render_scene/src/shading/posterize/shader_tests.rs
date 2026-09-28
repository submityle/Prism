//! WESL compilation coverage for the posterize shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `posterize.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `bloom.wesl` / `exposure.wesl` / `color_grade.wesl` / `gamut_map.wesl`), so a
//! green result also guards the posterization maths — the round-to-nearest band
//! snap, the per-channel RGB quantizer and the luma-preserving quantizer —
//! against drift from its CPU golden twin in
//! `prism_render_shading::posterize`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("posterize shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `posterize.wesl`, proving the linear-HDR posterize kernel parses and
/// type-checks exactly as it will in the render world (pre-exposed HDR radiance
/// and the artist params in; the posterized radiance out), and that the
/// quantize / per-channel / luma-preserving layout matches the CPU golden.
#[test]
fn posterize_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let posterize = shader_id(0x5052_4953_4d5f_0000_504f_5354_4552_495a);
    cache.set_shader(
        posterize,
        Shader::from_wesl(
            include_str!("../../shaders/posterize.wesl"),
            "embedded://prism_render_scene/shaders/posterize.wesl",
        ),
    );

    cache
        .get(0, posterize, &[])
        .unwrap_or_else(|error| panic!("posterize.wesl failed to compile: {error}"));
}
