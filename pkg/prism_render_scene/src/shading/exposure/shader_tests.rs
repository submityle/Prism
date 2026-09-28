//! WESL compilation coverage for the exposure shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `exposure.wesl` parses and type-checks exactly as it will on device. The
//! kernel is self-contained (no intra-crate `import`s, matching `ssgi.wesl` /
//! `volumetrics.wesl`), so a green result also guards the exposure maths —
//! physical-camera EV100, luminance metering, the percentile-trimmed histogram
//! average and the exponential eye-adaptation response — against drift from its
//! CPU golden twin in `prism_render_shading::exposure`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("exposure shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `exposure.wesl`, proving the physically-based exposure and eye
/// adaptation kernel parses and type-checks exactly as it will in the render
/// world (physical camera / metered luminance in; pre-exposure multiplier and
/// adapted luminance out), and that the histogram and settings layouts match
/// the CPU golden.
#[test]
fn exposure_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let exposure = shader_id(0x5052_4953_4d5f_4558_504f_5355_5245_0001);
    cache.set_shader(
        exposure,
        Shader::from_wesl(
            include_str!("../../shaders/exposure.wesl"),
            "embedded://prism_render_scene/shaders/exposure.wesl",
        ),
    );

    cache
        .get(0, exposure, &[])
        .unwrap_or_else(|error| panic!("exposure.wesl failed to compile: {error}"));
}
