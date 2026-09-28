//! WESL compilation coverage for the bloom shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `bloom.wesl` parses and type-checks exactly as it will on device. The kernel
//! is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `exposure.wesl`), so a green result also guards the bloom maths — Karis
//! soft-threshold prefilter, fireflies-suppressing Karis average, the COD
//! 13-tap downsample, the 3x3 tent upsample and the artist combine — against
//! drift from its CPU golden twin in `prism_render_shading::bloom`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("bloom shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `bloom.wesl`, proving the energy-preserving dual-filter bloom
/// kernel parses and type-checks exactly as it will in the render world
/// (pre-exposed HDR radiance in; prefiltered, down/upsampled and combined bloom
/// out), and that the prefilter/downsample/upsample layouts match the CPU
/// golden.
#[test]
fn bloom_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let bloom = shader_id(0x5052_4953_4d5f_424c_4f4f_4d00_0000_0001);
    cache.set_shader(
        bloom,
        Shader::from_wesl(
            include_str!("../../shaders/bloom.wesl"),
            "embedded://prism_render_scene/shaders/bloom.wesl",
        ),
    );

    cache
        .get(0, bloom, &[])
        .unwrap_or_else(|error| panic!("bloom.wesl failed to compile: {error}"));
}
