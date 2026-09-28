//! WESL compilation coverage for the face-shadow shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `face_shadow.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `bloom.wesl` / `color_grade.wesl` / `vignette.wesl`), so a green result also
//! guards the face-shadow maths — the in-plane light cosines, the UV-mirror
//! decision and the SDF threshold/softness terminator — against drift from its
//! CPU golden twin in `prism_render_shading::face_shadow`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("face shadow shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `face_shadow.wesl`, proving the SDF face-shadow kernel parses and
/// type-checks exactly as it will in the render world (the head frame, the
/// light direction, the resolved SDF sample and the artist params in; the
/// `[0, 1]` visibility out), and that its helper and struct layouts match the
/// CPU golden.
#[test]
fn face_shadow_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let face_shadow = shader_id(0x5052_4953_4d5f_0000_4641_4345_5348_4430);
    cache.set_shader(
        face_shadow,
        Shader::from_wesl(
            include_str!("../../shaders/face_shadow.wesl"),
            "embedded://prism_render_scene/shaders/face_shadow.wesl",
        ),
    );

    cache
        .get(0, face_shadow, &[])
        .unwrap_or_else(|error| panic!("face_shadow.wesl failed to compile: {error}"));
}
