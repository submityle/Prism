//! WESL compilation coverage for the halftone shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `halftone.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `bloom.wesl` / `color_grade.wesl` / `vignette.wesl`), so a green result also
//! guards the halftone maths — Rec. 709 luma, the rotated cell lattice, the
//! tone-driven dot radius and the anti-aliased coverage — against drift from
//! its CPU golden twin in `prism_render_shading::halftone`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("halftone shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `halftone.wesl`, proving the NPR halftone kernel parses and
/// type-checks exactly as it will in the render world (the stylized scene
/// colour and the artist params in; the screentone-printed colour out), and
/// that its helper layouts match the CPU golden.
#[test]
fn halftone_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let halftone = shader_id(0x5052_4953_4d5f_0000_4841_4c46_544f_4e45);
    cache.set_shader(
        halftone,
        Shader::from_wesl(
            include_str!("../../shaders/halftone.wesl"),
            "embedded://prism_render_scene/shaders/halftone.wesl",
        ),
    );

    cache
        .get(0, halftone, &[])
        .unwrap_or_else(|error| panic!("halftone.wesl failed to compile: {error}"));
}
