//! WESL compilation coverage for the SSR shader graph.
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
        ShaderCacheSource::SpirV(_) => unreachable!("SSR shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles `ssr.wesl`, proving the screen-space reflection trace kernel parses
/// and type-checks exactly as it will in the render world (HZB pyramid, scene
/// depth, normal/roughness and previous-frame colour in; reflected radiance and
/// confidence out).
#[test]
fn ssr_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let ssr = shader_id(0x5052_4953_4d5f_5353_525f_5452_4143_0001);
    cache.set_shader(
        ssr,
        Shader::from_wesl(
            include_str!("../../shaders/ssr.wesl"),
            "embedded://prism_render_scene/shaders/ssr.wesl",
        ),
    );

    cache
        .get(0, ssr, &[])
        .unwrap_or_else(|error| panic!("ssr.wesl failed to compile: {error}"));
}

/// Registers `tangent.wesl`, `surface.wesl` and `gpu_scene.wesl` under their
/// canonical module paths and compiles `ssr_prepass.wesl`, forcing every
/// `import prism_render_scene::shaders::*` to resolve exactly as it will in the
/// render world. A green result proves the SSR prepass decodes the visibility
/// buffer through the identical `surface.wesl` helpers the GTAO prepass and the
/// resolve stage use, and that its reverse-Z device-depth output type-checks.
#[test]
fn ssr_prepass_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let deps: [(u128, &str, &str); 3] = [
        (
            0x5052_4953_4d5f_5353_525f_5441_4e47_0001,
            include_str!("../../shaders/tangent.wesl"),
            "embedded://prism_render_scene/shaders/tangent.wesl",
        ),
        (
            0x5052_4953_4d5f_5353_525f_5355_5246_0001,
            include_str!("../../shaders/surface.wesl"),
            "embedded://prism_render_scene/shaders/surface.wesl",
        ),
        (
            0x5052_4953_4d5f_5353_525f_5343_4e45_0001,
            include_str!("../../shaders/gpu_scene.wesl"),
            "embedded://prism_render_scene/shaders/gpu_scene.wesl",
        ),
    ];
    for (tag, source, path) in deps {
        cache.set_shader(shader_id(tag), Shader::from_wesl(source, path));
    }

    let prepass = shader_id(0x5052_4953_4d5f_5353_525f_5052_4550_0001);
    cache.set_shader(
        prepass,
        Shader::from_wesl(
            include_str!("../../shaders/ssr_prepass.wesl"),
            "embedded://prism_render_scene/shaders/ssr_prepass.wesl",
        ),
    );

    cache.get(0, prepass, &[]).unwrap_or_else(|error| {
        panic!("ssr_prepass.wesl failed to compile/resolve imports: {error}")
    });
}

/// Compiles `ssr_hzb.wesl`, proving both pyramid-build entry points parse and
/// type-check as they will in the render world: `ssr_hzb_copy` lifts the
/// full-resolution device depth into pyramid level 0, and `ssr_hzb_reduce`
/// writes each coarser level as the reverse-Z 2x2 max-reduction of the finer
/// one. The kernel is self-contained (no intra-crate imports), so a green
/// result also guards the shared immediate `HzbParams` layout against drift.
#[test]
fn ssr_hzb_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hzb = shader_id(0x5052_4953_4d5f_5353_525f_484a_5a42_0001);
    cache.set_shader(
        hzb,
        Shader::from_wesl(
            include_str!("../../shaders/ssr_hzb.wesl"),
            "embedded://prism_render_scene/shaders/ssr_hzb.wesl",
        ),
    );

    cache
        .get(0, hzb, &[])
        .unwrap_or_else(|error| panic!("ssr_hzb.wesl failed to compile: {error}"));
}

/// Registers `material.wesl` under its canonical module path and compiles
/// `ssr_repack.wesl`, forcing its `import prism_render_scene::shaders::material`
/// to resolve exactly as it will in the render world. A green result proves the
/// repack reads the covered pixel's material through the same
/// `PrismMaterialHeader` / `PrismSurfaceParameters` tables the resolve stage
/// uses, and that packing the biased normal plus roughness into the trace's
/// `normal_roughness` output type-checks. `material.wesl` is self-contained (no
/// intra-crate imports), so registering it alone satisfies the graph.
#[test]
fn ssr_repack_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let material = shader_id(0x5052_4953_4d5f_5353_525f_4d41_5450_0001);
    cache.set_shader(
        material,
        Shader::from_wesl(
            include_str!("../../shaders/material.wesl"),
            "embedded://prism_render_scene/shaders/material.wesl",
        ),
    );

    let repack = shader_id(0x5052_4953_4d5f_5353_525f_5250_434b_0001);
    cache.set_shader(
        repack,
        Shader::from_wesl(
            include_str!("../../shaders/ssr_repack.wesl"),
            "embedded://prism_render_scene/shaders/ssr_repack.wesl",
        ),
    );

    cache.get(0, repack, &[]).unwrap_or_else(|error| {
        panic!("ssr_repack.wesl failed to compile/resolve imports: {error}")
    });
}
