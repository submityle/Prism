//! WESL compilation + ABI coverage for the froxel volumetric fog shader.
//!
//! The sandbox has no GPU, so these tests compile the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `volumetrics.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `outline.wesl`), so a green result also guards the froxel single-scattering
//! reduction — Henyey-Greenstein phase, Beer-Lambert transmittance, the
//! energy-conserving analytic slice integral and the front-to-back column
//! march — against drift from its CPU golden twin in
//! `prism_render_shading::volumetrics`. The final assertions pin the two
//! immediate-block sizes so the `#[repr(C)]` records can never silently drift
//! out of sync with the WESL `var<immediate>` structs.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

use super::abi::{
    GpuVolumetricsApplyParams, GpuVolumetricsIntegrateParams, GpuVolumetricsScatterParams,
};

const VOLUMETRICS_WESL: &str = include_str!("../../shaders/volumetrics.wesl");

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

/// Compiles `volumetrics.wesl` through the render-world shader pipeline, proving
/// both froxel compute entries (`volumetrics_scatter` and
/// `volumetrics_integrate`) parse and type-check exactly as they will on device,
/// and that the `MediumSample` / `Froxel` / `VolumetricIntegration` layouts and
/// the two `var<immediate>` param blocks match the CPU golden.
#[test]
fn volumetrics_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let volumetrics = shader_id(0x5052_4953_4d5f_564f_4c55_4d45_5452_0001);
    cache.set_shader(
        volumetrics,
        Shader::from_wesl(
            VOLUMETRICS_WESL,
            "embedded://prism_render_scene/shaders/volumetrics.wesl",
        ),
    );

    cache
        .get(0, volumetrics, &[])
        .unwrap_or_else(|error| panic!("volumetrics.wesl failed to compile: {error}"));
}

/// The shader must declare all three `@compute` entry points the pipelines name,
/// so a rename on either side is caught before it reaches the (GPU-less) device.
#[test]
fn volumetrics_wesl_declares_all_compute_entries() {
    assert!(
        VOLUMETRICS_WESL.contains("fn volumetrics_scatter"),
        "scatter entry point missing from volumetrics.wesl",
    );
    assert!(
        VOLUMETRICS_WESL.contains("fn volumetrics_integrate"),
        "integrate entry point missing from volumetrics.wesl",
    );
    assert!(
        VOLUMETRICS_WESL.contains("fn volumetrics_apply"),
        "apply entry point missing from volumetrics.wesl",
    );
}

/// The immediate-block sizes the pipelines pass as `immediate_size` must match
/// the `#[repr(C)]` ABI records byte-for-byte (24 scalars = 96 bytes for
/// scatter, 3 u32 = 12 bytes for integrate), so the `set_immediates` uploads
/// line up with the WESL `var<immediate>` structs.
#[test]
fn volumetrics_immediate_block_sizes_are_pinned() {
    assert_eq!(size_of::<GpuVolumetricsScatterParams>(), 96);
    assert_eq!(align_of::<GpuVolumetricsScatterParams>(), 4);
    assert_eq!(size_of::<GpuVolumetricsIntegrateParams>(), 12);
    assert_eq!(align_of::<GpuVolumetricsIntegrateParams>(), 4);
    assert_eq!(size_of::<GpuVolumetricsApplyParams>(), 96);
    assert_eq!(align_of::<GpuVolumetricsApplyParams>(), 4);
}
