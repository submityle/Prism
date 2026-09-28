//! WESL compilation coverage for the TAA shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `taa_resolve.wesl` parses and type-checks exactly as it will when the
//! resolve pipeline specializes it.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("TAA shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `taa_resolve.wesl` standalone. The resolve reprojects the previous
/// frame's accumulated colour through the motion-vector G-buffer, clips it to
/// the current 3x3 neighbourhood's `YCoCg` variance box and luminance-feedback
/// blends it with the current composited `scene_color`. The kernel is
/// self-contained (no intra-crate imports), so a green result also guards its
/// immediate `TaaParams` layout against drift from the 20-byte
/// `GpuTaaResolveParams` contract (extent + golden tunables + validity flag).
#[test]
fn taa_resolve_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let resolve = shader_id(0x5052_4953_4d5f_5441_415f_5253_4c56_0001);
    cache.set_shader(
        resolve,
        Shader::from_wesl(
            include_str!("../../shaders/taa_resolve.wesl"),
            "embedded://prism_render_scene/shaders/taa_resolve.wesl",
        ),
    );

    cache
        .get(0, resolve, &[])
        .unwrap_or_else(|error| panic!("taa_resolve.wesl failed to compile: {error}"));
}
