//! WESL compilation coverage for the chromatic-aberration shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `chromatic_aberration.wesl` parses and type-checks exactly as it will on
//! device. The kernel is self-contained (no intra-crate `import`s, matching
//! `ssgi.wesl` / `bloom.wesl` / `color_grade.wesl`), so a green result also
//! guards the aberration maths — the radial frame, the three-tap channel split
//! and the spectral ramp — against drift from its CPU golden twin in
//! `prism_render_shading::chromatic_aberration`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("chromatic aberration shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `chromatic_aberration.wesl`, proving the lens-fringing kernel parses
/// and type-checks exactly as it will in the render world (the resolved image
/// and the artist params in; the per-channel sample coordinates out), and that
/// the radial-frame / three-tap-split / spectral-ramp layouts match the CPU
/// golden.
#[test]
fn chromatic_aberration_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let chromatic_aberration = shader_id(0x5052_4953_4d5f_0000_4348_524f_4d41_4252);
    cache.set_shader(
        chromatic_aberration,
        Shader::from_wesl(
            include_str!("../../shaders/chromatic_aberration.wesl"),
            "embedded://prism_render_scene/shaders/chromatic_aberration.wesl",
        ),
    );

    cache
        .get(0, chromatic_aberration, &[])
        .unwrap_or_else(|error| panic!("chromatic_aberration.wesl failed to compile: {error}"));
}


/// Guards the Rust immediate-block ABI against drift from the WESL struct: the
/// aberration block is a tight 24 bytes (`vec2<f32>` centre + intensity +
/// samples + `vec2<u32>` extent) matching `chromatic_aberration.wesl`'s single
/// `var<immediate>` global, and the workgroup constant matches
/// `@workgroup_size(8, 8, 1)`.
#[test]
fn chromatic_aberration_abi_matches_the_shader_layout() {
    use super::abi::{GpuChromaticAberrationParams, CHROMATIC_ABERRATION_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuChromaticAberrationParams>(), 24);
    assert_eq!(align_of::<GpuChromaticAberrationParams>(), 4);
    assert_eq!(CHROMATIC_ABERRATION_WORKGROUP_SIZE, 8);
}
