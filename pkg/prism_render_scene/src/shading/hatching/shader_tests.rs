//! WESL compilation coverage for the cross-hatching shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `hatching.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `bloom.wesl` / `exposure.wesl` / `color_grade.wesl` / `kuwahara.wesl`), so a
//! green result also guards the hatching maths — the Rec. 709 luminance ramp,
//! the rotated periodic line coverage, the hand-written smoothstep and the
//! ink-over-paper composite — against drift from its CPU golden twin in
//! `prism_render_shading::hatching`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("hatching shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `hatching.wesl`, proving the layered cross-hatching kernel parses
/// and type-checks exactly as it will in the render world (resolved HDR
/// radiance and the artist params in; the stylized radiance out), and that the
/// luminance / rotate / line-coverage / composite layouts match the CPU golden.
#[test]
fn hatching_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hatching = shader_id(0x5052_4953_4d5f_0000_4841_5443_4849_4e47);
    cache.set_shader(
        hatching,
        Shader::from_wesl(
            include_str!("../../shaders/hatching.wesl"),
            "embedded://prism_render_scene/shaders/hatching.wesl",
        ),
    );

    cache
        .get(0, hatching, &[])
        .unwrap_or_else(|error| panic!("hatching.wesl failed to compile: {error}"));
}

/// Guards the Rust immediate-block ABI against drift from the WESL struct: the
/// single `GpuHatchingParams` block is the 48-byte extent-led block matching
/// `hatching.wesl`'s one `var<immediate>` global, and the workgroup constant
/// matches `@workgroup_size(8, 8, 1)`.
#[test]
fn hatching_abi_matches_the_shader_layout() {
    use super::abi::{GpuHatchingParams, HATCHING_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuHatchingParams>(), 48);
    assert_eq!(align_of::<GpuHatchingParams>(), 4);
    assert_eq!(HATCHING_WORKGROUP_SIZE, 8);
}
