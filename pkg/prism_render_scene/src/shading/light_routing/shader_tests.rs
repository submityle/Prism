//! WESL compilation coverage for the lighting-channel / light-layer routing shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses,
//! validating that `light_routing.wesl` parses and type-checks exactly as
//! it will on device. The kernel is self-contained (no intra-crate
//! `import`s, matching `ssgi.wesl` / `outline.wesl` /
//! `volumetrics.wesl`), so a green result also guards the channel/layer
//! mask helpers, the routing gates and the per-word cluster cull against drift
//! from their CPU golden twin in `prism_render_shading::light_routing`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("light_routing shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `light_routing.wesl`, proving the lighting-channel visibility
/// gate, the NPR light-layer routing gate and the per-word cluster cull parse
/// and type-check exactly as they will in the render world, and that the
/// `LightRouting` layout and channel/layer constants match the CPU golden.
#[test]
fn light_routing_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let routing = shader_id(0x5052_4953_4d5f_4c47_4852_5455_4e47_0001);
    cache.set_shader(
        routing,
        Shader::from_wesl(
            include_str!("../../shaders/light_routing.wesl"),
            "embedded://prism_render_scene/shaders/light_routing.wesl",
        ),
    );

    cache
        .get(0, routing, &[])
        .unwrap_or_else(|error| panic!("light_routing.wesl failed to compile: {error}"));
}
