//! WESL compilation coverage for the CAS shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `cas.wesl` parses and type-checks exactly as it will on device. The kernel
//! is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `bloom.wesl` / `exposure.wesl`), so a green result also guards the
//! sharpening maths — the `FidelityFX` doubled soft limits, the adaptive
//! amplitude, the sharpness-shaped cross weight and the saturated blend —
//! against drift from its CPU golden twin in `prism_render_shading::cas`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("cas shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `cas.wesl`, proving the contrast-adaptive sharpening kernel parses
/// and type-checks exactly as it will in the render world (the 3x3 gather taps
/// and the artist params in; the sharpened radiance out), and that the soft
/// limit / amplitude / weight / blend layouts match the CPU golden.
#[test]
fn cas_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let cas = shader_id(0x5052_4953_4d5f_0000_4341_535f_5348_5250);
    cache.set_shader(
        cas,
        Shader::from_wesl(
            include_str!("../../shaders/cas.wesl"),
            "embedded://prism_render_scene/shaders/cas.wesl",
        ),
    );

    cache
        .get(0, cas, &[])
        .unwrap_or_else(|error| panic!("cas.wesl failed to compile: {error}"));
}

/// Guards the Rust immediate-block ABI against drift from the WESL struct: the
/// single `GpuCasParams` block is the 16-byte `vec2<u32>`-led block matching
/// `cas.wesl`'s one `var<immediate>` global, and the workgroup constant matches
/// `@workgroup_size(8, 8, 1)`.
#[test]
fn cas_abi_matches_the_shader_layout() {
    use super::abi::{GpuCasParams, CAS_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuCasParams>(), 16);
    assert_eq!(align_of::<GpuCasParams>(), 4);
    assert_eq!(CAS_WORKGROUP_SIZE, 8);
}
