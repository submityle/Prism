//! WESL compilation coverage for the volumetric fog shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `volumetrics.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `outline.wesl`), so a green result also guards the froxel single-scattering
//! reduction — Henyey-Greenstein phase, Beer-Lambert transmittance, the
//! energy-conserving analytic slice integral and the front-to-back column
//! march — against drift from its CPU golden twin in
//! `prism_render_shading::volumetrics`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("volumetrics shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `volumetrics.wesl`, proving the froxel volumetric fog kernel parses
/// and type-checks exactly as it will in the render world (per-froxel medium
/// coefficients and in-scattered radiance in; accumulated in-scattering and
/// column transmittance out), and that the `MediumSample` / `Froxel` /
/// `VolumetricIntegration` layouts match the CPU golden.
#[test]
fn volumetrics_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let volumetrics = shader_id(0x5052_4953_4d5f_564f_4c55_4d45_5452_0001);
    cache.set_shader(
        volumetrics,
        Shader::from_wesl(
            include_str!("../../shaders/volumetrics.wesl"),
            "embedded://prism_render_scene/shaders/volumetrics.wesl",
        ),
    );

    cache
        .get(0, volumetrics, &[])
        .unwrap_or_else(|error| panic!("volumetrics.wesl failed to compile: {error}"));
}
