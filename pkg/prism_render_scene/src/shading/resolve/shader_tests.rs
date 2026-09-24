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


/// Registers `lighting.wesl`, `brdf.wesl` and `cloth.wesl` under their canonical
/// module paths and compiles `cloth.wesl`, forcing the importer to resolve the
/// `prism_render_scene::shaders::{lighting, brdf}::{...}` imports the cloth lobe
/// depends on.
#[test]
fn cloth_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_0003);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_0003);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let cloth = shader_id(0x5052_4953_4d5f_434c_4f54_4800_0000_0001);
    cache.set_shader(
        cloth,
        Shader::from_wesl(
            include_str!("../../shaders/cloth.wesl"),
            "embedded://prism_render_scene/shaders/cloth.wesl",
        ),
    );

    cache
        .get(0, cloth, &[])
        .unwrap_or_else(|error| panic!("cloth.wesl failed to compile/resolve imports: {error}"));
}


/// Registers `lighting.wesl`, `brdf.wesl` and `subsurface.wesl` under their
/// canonical module paths and compiles `subsurface.wesl`, forcing the importer
/// to resolve the `prism_render_scene::shaders::{lighting, brdf}::{...}` imports
/// the subsurface lobe depends on (including the reused GGX helpers).
#[test]
fn subsurface_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_0004);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_0004);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let subsurface = shader_id(0x5052_4953_4d5f_5355_4253_0000_0000_0001);
    cache.set_shader(
        subsurface,
        Shader::from_wesl(
            include_str!("../../shaders/subsurface.wesl"),
            "embedded://prism_render_scene/shaders/subsurface.wesl",
        ),
    );

    cache.get(0, subsurface, &[]).unwrap_or_else(|error| {
        panic!("subsurface.wesl failed to compile/resolve imports: {error}")
    });
}

/// Compiles `tangent.wesl` standalone.  It has no imports, so a green result
/// proves the tangent-basis reconstruction math (Duff orthonormal basis,
/// Lengyel analytic tangent, authored re-orthonormalization) parses and
/// type-checks as WESL on its own, in lock-step with the CPU golden reference.
#[test]
fn tangent_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let tangent = shader_id(0x5052_4953_4d5f_5441_4e47_454e_5400_0001);
    cache.set_shader(
        tangent,
        Shader::from_wesl(
            include_str!("../../shaders/tangent.wesl"),
            "embedded://prism_render_scene/shaders/tangent.wesl",
        ),
    );

    cache
        .get(0, tangent, &[])
        .unwrap_or_else(|error| panic!("tangent.wesl failed to compile: {error}"));
}

/// Registers `tangent.wesl` under its canonical module path and compiles
/// `surface.wesl`, forcing the importer to resolve the
/// `prism_render_scene::shaders::tangent::{...}` import the geometry-table ABI,
/// barycentric decode and vertex/tangent-frame interpolation depend on.
#[test]
fn surface_wesl_compiles_and_resolves_tangent_import() {
    let mut cache = ShaderCache::new((), load_source);

    let tangent = shader_id(0x5052_4953_4d5f_5441_4e47_454e_5400_0003);
    cache.set_shader(
        tangent,
        Shader::from_wesl(
            include_str!("../../shaders/tangent.wesl"),
            "embedded://prism_render_scene/shaders/tangent.wesl",
        ),
    );

    let surface = shader_id(0x5052_4953_4d5f_5355_5246_4143_4500_0001);
    cache.set_shader(
        surface,
        Shader::from_wesl(
            include_str!("../../shaders/surface.wesl"),
            "embedded://prism_render_scene/shaders/surface.wesl",
        ),
    );

    cache.get(0, surface, &[]).unwrap_or_else(|error| {
        panic!("surface.wesl failed to compile/resolve tangent import: {error}")
    });
}

/// Registers the full dependency graph (`surface`, `brdf`, `lighting`,
/// `material`, `gpu_scene`) under their canonical module paths and compiles
/// `shading_resolve.wesl`, forcing every `import prism_render_scene::shaders::*`
/// to resolve exactly as it will in the render world.
#[test]
fn shading_resolve_wesl_compiles_and_resolves_all_imports() {
    let mut cache = ShaderCache::new((), load_source);

    // Register each dependency under the byte-identical embedded module path
    // that `load_shader_library!` produces at runtime.
    let deps: [(u128, &str, &str); 8] = [
        (
            0x5052_4953_4d5f_5441_4e47_454e_5400_0002,
            include_str!("../../shaders/tangent.wesl"),
            "embedded://prism_render_scene/shaders/tangent.wesl",
        ),
        (
            0x5052_4953_4d5f_5355_5246_4143_4500_0002,
            include_str!("../../shaders/surface.wesl"),
            "embedded://prism_render_scene/shaders/surface.wesl",
        ),
        (
            0x5052_4953_4d5f_4c49_4748_5449_4e47_0002,
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
        (
            0x5052_4953_4d5f_4252_4446_0000_0000_0002,
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
        (
            0x5052_4953_4d5f_4d41_5445_5249_414c_0002,
            include_str!("../../shaders/material.wesl"),
            "embedded://prism_render_scene/shaders/material.wesl",
        ),
        (
            0x5052_4953_4d5f_5343_454e_4500_0000_0002,
            include_str!("../../shaders/gpu_scene.wesl"),
            "embedded://prism_render_scene/shaders/gpu_scene.wesl",
        ),
        (
            0x5052_4953_4d5f_434c_4f54_4800_0000_0002,
            include_str!("../../shaders/cloth.wesl"),
            "embedded://prism_render_scene/shaders/cloth.wesl",
        ),
        (
            0x5052_4953_4d5f_5355_4253_0000_0000_0002,
            include_str!("../../shaders/subsurface.wesl"),
            "embedded://prism_render_scene/shaders/subsurface.wesl",
        ),
    ];
    for (tag, source, path) in deps {
        cache.set_shader(shader_id(tag), Shader::from_wesl(source, path));
    }

    let resolve = shader_id(0x5052_4953_4d5f_5245_534f_4c56_4500_0001);
    cache.set_shader(
        resolve,
        Shader::from_wesl(
            include_str!("../../shaders/shading_resolve.wesl"),
            "embedded://prism_render_scene/shaders/shading_resolve.wesl",
        ),
    );

    cache.get(0, resolve, &[]).unwrap_or_else(|error| {
        panic!("shading_resolve.wesl failed to compile/resolve imports: {error}")
    });
}
/// Registers `material.wesl` under its canonical module path and compiles
/// `material_sample.wesl`, forcing the importer to resolve the
/// `prism_render_scene::shaders::material::{PrismMaterialHeader,
/// PrismMaterialTexture, PrismSurfaceParameters}` import the bindless sampler
/// depends on.  A green result proves the `enable wgpu_binding_array;`
/// directive, the `binding_array<texture_2d<f32>>` / `binding_array<sampler>`
/// declarations, the sRGB/normal decode helpers and the semantic `switch` all
/// parse and type-check as WESL in lock-step with the CPU golden
/// `prism_render_shading::texture_sample`.
#[test]
fn material_sample_wesl_compiles_and_resolves_material_import() {
    let mut cache = ShaderCache::new((), load_source);

    let material = shader_id(0x5052_4953_4d5f_4d41_5445_5249_414c_0009);
    cache.set_shader(
        material,
        Shader::from_wesl(
            include_str!("../../shaders/material.wesl"),
            "embedded://prism_render_scene/shaders/material.wesl",
        ),
    );

    let material_sample = shader_id(0x5052_4953_4d5f_4d54_5853_414d_5000_0001);
    cache.set_shader(
        material_sample,
        Shader::from_wesl(
            include_str!("../../shaders/material_sample.wesl"),
            "embedded://prism_render_scene/shaders/material_sample.wesl",
        ),
    );

    cache.get(0, material_sample, &[]).unwrap_or_else(|error| {
        panic!("material_sample.wesl failed to compile/resolve material import: {error}")
    });
}
