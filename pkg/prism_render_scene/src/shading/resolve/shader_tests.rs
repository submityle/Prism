//! WESL compilation coverage for the shading-resolve shader graph.
//!
//! The sandbox has no GPU, so these tests do **not** exercise the pass at
//! runtime.  They compile the WESL sources through the same `ShaderCache` /
//! `wesl` pipeline the render world uses, which validates:
//!
//! * every source parses and type-checks as WESL, and
//! * the intra-crate `import prism_render_scene::shaders::...` statements
//!   resolve against the module paths produced by the crate's embedded-asset
//!   registration convention (`embedded://<crate>/shaders/<name>.wesl`).
//!
//! The module paths registered here are byte-identical to the ones
//! `load_shader_library!(app, "shaders/<name>.wesl")` produces from
//! `prism_render_scene/src/lib.rs`, so a green test here proves the runtime
//! import wiring resolves too.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("shading resolve shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Registers `lighting.wesl` and `brdf.wesl` under their canonical module
/// paths and compiles `brdf.wesl`, forcing the importer to resolve
/// `prism_render_scene::shaders::lighting::{...}`.
#[test]
fn brdf_wesl_compiles_and_resolves_lighting_import() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_0001);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_0001);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    cache
        .get(0, brdf, &[])
        .unwrap_or_else(|error| panic!("brdf.wesl failed to compile/resolve imports: {error}"));
}
