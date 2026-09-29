//! WESL compilation coverage for the ordered-dither shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `ordered_dither.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `bloom.wesl` / `color_grade.wesl` / `exposure.wesl`), so a green result also
//! guards the dither maths — the 4x4 `Bayer` threshold, the palette quantiser and
//! the strength blend — against drift from its CPU golden twin in
//! `prism_render_shading::ordered_dither`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("ordered dither shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `ordered_dither.wesl`, proving the ordered-dither kernel parses and
/// type-checks exactly as it will in the render world (pre-exposed `HDR` radiance
/// and the artist params in; the dithered radiance out), and that the `Bayer`
/// threshold, quantiser and strength-blend layouts match the CPU golden.
#[test]
fn ordered_dither_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let ordered_dither = shader_id(0x5052_4953_4d5f_0000_4f52_4452_4449_5448);
    cache.set_shader(
        ordered_dither,
        Shader::from_wesl(
            include_str!("../../shaders/ordered_dither.wesl"),
            "embedded://prism_render_scene/shaders/ordered_dither.wesl",
        ),
    );

    cache
        .get(0, ordered_dither, &[])
        .unwrap_or_else(|error| panic!("ordered_dither.wesl failed to compile: {error}"));
}

/// Guards the Rust immediate-block ABI against drift from the WESL struct: the
/// single `GpuOrderedDitherParams` block is the 16-byte extent-led block
/// matching `ordered_dither.wesl`'s one `var<immediate>` global, and the
/// workgroup constant matches `@workgroup_size(8, 8, 1)`.
#[test]
fn ordered_dither_abi_matches_the_shader_layout() {
    use super::abi::{GpuOrderedDitherParams, ORDERED_DITHER_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuOrderedDitherParams>(), 16);
    assert_eq!(align_of::<GpuOrderedDitherParams>(), 4);
    assert_eq!(ORDERED_DITHER_WORKGROUP_SIZE, 8);
}
