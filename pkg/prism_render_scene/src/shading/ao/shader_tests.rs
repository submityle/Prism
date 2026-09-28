//! WESL compilation coverage for the GTAO shader graph.
//!
//! The sandbox has no GPU, so these tests compile the WESL sources through the
//! same `ShaderCache` / `wesl` pipeline the render world uses, validating that
//! every source parses and type-checks and that the intra-crate
//! `import prism_render_scene::shaders::...` statements resolve against the
//! module paths the crate's embedded-asset registration produces.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("GTAO shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Registers `tangent.wesl`, `surface.wesl` and `gpu_scene.wesl` under their
/// canonical module paths and compiles `gtao_prepass.wesl`, forcing every
/// `import prism_render_scene::shaders::*` to resolve exactly as it will in the
/// render world.  A green result proves the prepass decodes the visibility
/// buffer through the identical `surface.wesl` helpers the resolve stage uses.
#[test]
fn gtao_prepass_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let deps: [(u128, &str, &str); 3] = [
        (
            0x5052_4953_4d5f_4741_4f5f_5441_4e47_0001,
            include_str!("../../shaders/tangent.wesl"),
            "embedded://prism_render_scene/shaders/tangent.wesl",
        ),
        (
            0x5052_4953_4d5f_4741_4f5f_5355_5246_0001,
            include_str!("../../shaders/surface.wesl"),
            "embedded://prism_render_scene/shaders/surface.wesl",
        ),
        (
            0x5052_4953_4d5f_4741_4f5f_5343_4e45_0001,
            include_str!("../../shaders/gpu_scene.wesl"),
            "embedded://prism_render_scene/shaders/gpu_scene.wesl",
        ),
    ];
    for (tag, source, path) in deps {
        cache.set_shader(shader_id(tag), Shader::from_wesl(source, path));
    }

    let prepass = shader_id(0x5052_4953_4d5f_4741_4f5f_5052_4550_0001);
    cache.set_shader(
        prepass,
        Shader::from_wesl(
            include_str!("../../shaders/gtao_prepass.wesl"),
            "embedded://prism_render_scene/shaders/gtao_prepass.wesl",
        ),
    );

    cache.get(0, prepass, &[]).unwrap_or_else(|error| {
        panic!("gtao_prepass.wesl failed to compile/resolve imports: {error}")
    });
}
