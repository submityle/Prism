//! WESL compilation coverage for the SSGI shader.
//!
//! The sandbox has no GPU, so this test compiles the WESL source through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! `ssgi.wesl` parses and type-checks exactly as it will on device. The kernel
//! is self-contained (no intra-crate `import`s, matching `ssr.wesl` /
//! `gtao.wesl`), so a green result also guards the shared immediate
//! `SsgiConfig` layout against drift.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("SSGI shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `ssgi.wesl`, proving the screen-space global-illumination trace
/// kernel parses and type-checks exactly as it will in the render world (HZB
/// pyramid, scene depth, packed normal/roughness, current-frame colour pyramid
/// and IBL/SH ambient in; pre-albedo indirect radiance plus a blend confidence
/// out).
#[test]
fn ssgi_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let ssgi = shader_id(0x5052_4953_4d5f_5353_4749_5f54_5243_0001);
    cache.set_shader(
        ssgi,
        Shader::from_wesl(
            include_str!("../../shaders/ssgi.wesl"),
            "embedded://prism_render_scene/shaders/ssgi.wesl",
        ),
    );

    cache
        .get(0, ssgi, &[])
        .unwrap_or_else(|error| panic!("ssgi.wesl failed to compile: {error}"));
}

/// Compiles `ssgi_composite.wesl`, proving the two composite entry points parse
/// and type-check exactly as they will in the render world: `ssgi_copy_base`
/// lifts `scene_color` into the scratch base, and `ssgi_composite` folds the
/// gather over that base under the resolve's albedo/ambient exports. A green
/// result also guards the shared `CompositeParams` immediate layout against
/// drift.
#[test]
fn ssgi_composite_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let composite = shader_id(0x5052_4953_4d5f_5353_4749_5f43_4d50_0001);
    cache.set_shader(
        composite,
        Shader::from_wesl(
            include_str!("../../shaders/ssgi_composite.wesl"),
            "embedded://prism_render_scene/shaders/ssgi_composite.wesl",
        ),
    );

    cache
        .get(0, composite, &[])
        .unwrap_or_else(|error| panic!("ssgi_composite.wesl failed to compile: {error}"));
}
