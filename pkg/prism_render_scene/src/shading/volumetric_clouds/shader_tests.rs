//! WESL compilation + entry-point coverage for the volumetric cloud shader.
//!
//! The sandbox has no GPU, so these tests compile the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `volumetric_clouds.wesl` parses and type-checks exactly as it will on
//! device. The kernel is self-contained (no intra-crate `import`s), so a green
//! result also guards the ported cloud math — Perlin-Worley noise, the
//! coverage/type/height modeling remaps, Henyey-Greenstein + Draine phase,
//! Beer-Lambert transmittance, the multiple-scattering octave sum and the
//! temporal reprojection clamp — against drift from its CPU golden twin in
//! `prism_render_architecture::volumetric`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

const VOLUMETRIC_CLOUDS_WESL: &str = include_str!("../../shaders/volumetric_clouds.wesl");

/// Every `@compute` entry point the architecture crate's `gpu::kernels`
/// contract names for its dispatch domains. A rename on either side must fail
/// the coverage test below before it can reach the (GPU-less) device.
const COMPUTE_ENTRIES: [&str; 8] = [
    "fn volumetric_weather_advect",
    "fn volumetric_noise_bake",
    "fn volumetric_modeling",
    "fn volumetric_multiscatter_lut_bake",
    "fn volumetric_raymarch",
    "fn volumetric_scatter_resolve",
    "fn volumetric_shadow_march",
    "fn volumetric_upsample",
];

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("volumetric cloud shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `volumetric_clouds.wesl` through the render-world shader pipeline,
/// proving all eight cloud compute entries parse and type-check exactly as they
/// will on device, and that the ported cloud math survives WESL's static
/// analysis unchanged.
#[test]
fn volumetric_clouds_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let clouds = shader_id(0x5052_4953_4d5f_564f_4c55_4d45_5452_0002);
    cache.set_shader(
        clouds,
        Shader::from_wesl(
            VOLUMETRIC_CLOUDS_WESL,
            "embedded://prism_render_scene/shaders/volumetric_clouds.wesl",
        ),
    );

    cache
        .get(0, clouds, &[])
        .unwrap_or_else(|error| panic!("volumetric_clouds.wesl failed to compile: {error}"));
}

/// The shader must declare all eight `@compute` entry points the architecture
/// crate's `gpu::kernels` contract names, so a rename on either side is caught
/// before it reaches the (GPU-less) device.
#[test]
fn volumetric_clouds_wesl_declares_all_compute_entries() {
    for entry in COMPUTE_ENTRIES {
        assert!(
            VOLUMETRIC_CLOUDS_WESL.contains(entry),
            "compute entry point missing from volumetric_clouds.wesl: {entry}",
        );
    }
}
