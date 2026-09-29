//! WESL compilation coverage for the temporal-upscale shaders.
//!
//! The sandbox has no GPU, so these tests compile each WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `upscale_reconstruct.wesl` and `upscale_rcas.wesl` parse and type-check
//! exactly as they will when their pipelines specialize them. Both kernels are
//! self-contained (no intra-crate `import`s, matching `taa_resolve.wesl` /
//! `cas.wesl`), so a green result also guards their maths — the Lanczos-2
//! reconstruction, motion reproject / disocclusion, `YCoCg` clip and temporal
//! accumulation on the one side, and the RCAS five-tap cross on the other —
//! against drift from the CPU golden twin in `prism_render_shading::upscale`,
//! and pins each immediate block layout against the ABI contract in
//! [`super::abi`].

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("upscale shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `upscale_reconstruct.wesl` standalone. The pass reconstructs the
/// low-resolution current frame onto the display grid with a separable
/// Lanczos-2 kernel, reprojects the display-resolution history through the
/// motion vectors, clips it to the current 3x3 neighbourhood's `YCoCg`
/// variance box, locks thin features and blends by the growing temporal
/// accumulation. A green result guards its `UpscaleReconstructParams` layout
/// against drift from the 56-byte `GpuUpscaleReconstructParams` contract.
#[test]
fn upscale_reconstruct_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let reconstruct = shader_id(0x5052_4953_4d5f_5550_5343_4c5f_5243_4e53);
    cache.set_shader(
        reconstruct,
        Shader::from_wesl(
            include_str!("../../shaders/upscale_reconstruct.wesl"),
            "embedded://prism_render_scene/shaders/upscale_reconstruct.wesl",
        ),
    );

    cache
        .get(0, reconstruct, &[])
        .unwrap_or_else(|error| panic!("upscale_reconstruct.wesl failed to compile: {error}"));
}

/// Compiles `upscale_rcas.wesl` standalone. The pass runs FSR's Robust
/// Contrast-Adaptive Sharpening over the reconstructed image's five-tap cross,
/// deriving one lobe from the per-channel contrast limits and clamping it to
/// `RCAS_LIMIT`. A green result guards its `UpscaleRcasParams` layout against
/// drift from the 16-byte `GpuUpscaleRcasParams` contract and the RCAS maths
/// against its golden twin in `prism_render_shading::upscale::sharpen`.
#[test]
fn upscale_rcas_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let rcas = shader_id(0x5052_4953_4d5f_5550_5343_4c5f_5243_4153);
    cache.set_shader(
        rcas,
        Shader::from_wesl(
            include_str!("../../shaders/upscale_rcas.wesl"),
            "embedded://prism_render_scene/shaders/upscale_rcas.wesl",
        ),
    );

    cache
        .get(0, rcas, &[])
        .unwrap_or_else(|error| panic!("upscale_rcas.wesl failed to compile: {error}"));
}
