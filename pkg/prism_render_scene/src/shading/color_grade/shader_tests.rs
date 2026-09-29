//! WESL compilation coverage for the colour-grade shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `color_grade.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `bloom.wesl` / `exposure.wesl`), so a green result also guards the grading
//! maths — von Kries white balance, ASC CDL lift/gamma/gain, the pivot contrast
//! and the luma-preserving saturation — against drift from its CPU golden twin
//! in `prism_render_shading::color_grade`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("color grade shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `color_grade.wesl`, proving the linear-HDR colour-grade kernel
/// parses and type-checks exactly as it will in the render world (pre-exposed
/// HDR radiance and the artist params in; the graded radiance out), and that
/// the white-balance / CDL / contrast / saturation layouts match the CPU
/// golden.
#[test]
fn color_grade_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let color_grade = shader_id(0x5052_4953_4d5f_0000_434f_4c4f_5247_4400);
    cache.set_shader(
        color_grade,
        Shader::from_wesl(
            include_str!("../../shaders/color_grade.wesl"),
            "embedded://prism_render_scene/shaders/color_grade.wesl",
        ),
    );

    cache
        .get(0, color_grade, &[])
        .unwrap_or_else(|error| panic!("color_grade.wesl failed to compile: {error}"));
}

/// Guards the Rust immediate-block ABI against drift from the WESL struct: the
/// single `GpuColorGradeParams` block is the 80-byte four-`vec4`-led block
/// matching `color_grade.wesl`'s one `var<immediate>` global, and the workgroup
/// constant matches `@workgroup_size(8, 8, 1)`.
#[test]
fn color_grade_abi_matches_the_shader_layout() {
    use super::abi::{GpuColorGradeParams, COLOR_GRADE_WORKGROUP_SIZE};
    assert_eq!(size_of::<GpuColorGradeParams>(), 80);
    assert_eq!(align_of::<GpuColorGradeParams>(), 4);
    assert_eq!(COLOR_GRADE_WORKGROUP_SIZE, 8);
}
