//! WESL compilation coverage for the gamut-map shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `gamut_map.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `bloom.wesl` / `color_grade.wesl`), so a green result also guards the
//! gamut-compress maths — the achromatic anchor, the per-channel relative
//! distance and the rational-knee distance compression — against drift from its
//! CPU golden twin in `prism_render_shading::gamut_map`.

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

/// Compiles `gamut_map.wesl`, proving the linear-HDR gamut-compress kernel
/// parses and type-checks exactly as it will in the render world (pre-exposed
/// HDR radiance and the artist params in; the compressed radiance out), and that
/// the anchor / distance / compression layouts match the CPU golden.
#[test]
fn gamut_map_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let gamut_map = shader_id(0x5052_4953_4d5f_0000_4741_4d55_544d_4150);
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

/// Guards the Rust immediate-block ABI against drift from the WESL struct: the
/// single `GpuGamutMapParams` block is the 48-byte two-`vec4`-led block matching
/// `gamut_map.wesl`'s one `var<immediate>` global, and the workgroup constant
/// matches `@workgroup_size(8, 8, 1)`.
#[test]
fn gamut_map_abi_matches_the_shader_layout() {
    use super::abi::{GpuGamutMapParams, GAMUT_MAP_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuGamutMapParams>(), 48);
    assert_eq!(align_of::<GpuGamutMapParams>(), 4);
    assert_eq!(GAMUT_MAP_WORKGROUP_SIZE, 8);
}
