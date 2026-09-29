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

/// Registers `lighting.wesl`, `brdf.wesl` and `hair.wesl` under their canonical
/// module paths and compiles `hair.wesl`, forcing the importer to resolve the
/// `prism_render_scene::shaders::{lighting, brdf}::{...}` imports the hair
/// strand lobe depends on (`brdf_normalize_or` and `INV_PI`).
#[test]
fn hair_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_0006);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_0006);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let hair = shader_id(0x5052_4953_4d5f_4841_4952_0000_0000_0001);
    cache.set_shader(
        hair,
        Shader::from_wesl(
            include_str!("../../shaders/hair.wesl"),
            "embedded://prism_render_scene/shaders/hair.wesl",
        ),
    );

    cache
        .get(0, hair, &[])
        .unwrap_or_else(|error| panic!("hair.wesl failed to compile/resolve imports: {error}"));
}

/// Registers `lighting.wesl`, `brdf.wesl` and `water.wesl` under their canonical
/// module paths and compiles `water.wesl`, forcing the importer to resolve the
/// `prism_render_scene::shaders::{lighting, brdf}::{...}` imports the single-
/// layer water lobe depends on (`brdf_normalize_or`, `fresnel_schlick`,
/// `distribution_ggx`, `vis_smith` and `INV_PI`).
#[test]
fn water_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_0007);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_0007);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let water = shader_id(0x5052_4953_4d5f_5741_5445_5200_0000_0001);
    cache.set_shader(
        water,
        Shader::from_wesl(
            include_str!("../../shaders/water.wesl"),
            "embedded://prism_render_scene/shaders/water.wesl",
        ),
    );

    cache
        .get(0, water, &[])
        .unwrap_or_else(|error| panic!("water.wesl failed to compile/resolve imports: {error}"));
}

/// Registers `lighting.wesl`, `brdf.wesl` and `clearcoat.wesl` under their
/// canonical module paths and compiles `clearcoat.wesl`, forcing the importer
/// to resolve the `prism_render_scene::shaders::{lighting, brdf}::{...}` imports
/// the two-layer clear-coat lobe depends on (`brdf_normalize_or`,
/// `fresnel_schlick`, `distribution_ggx`, `vis_smith` and `INV_PI`).
#[test]
fn clearcoat_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_0008);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_0008);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let clearcoat = shader_id(0x5052_4953_4d5f_434c_4541_5243_4f41_0001);
    cache.set_shader(
        clearcoat,
        Shader::from_wesl(
            include_str!("../../shaders/clearcoat.wesl"),
            "embedded://prism_render_scene/shaders/clearcoat.wesl",
        ),
    );

    cache.get(0, clearcoat, &[]).unwrap_or_else(|error| {
        panic!("clearcoat.wesl failed to compile/resolve imports: {error}")
    });
}

/// Registers the full dependency graph (`surface`, `brdf`, `lighting`,
/// `material`, `gpu_scene`) under their canonical module paths and compiles
/// `shading_resolve.wesl`, forcing every `import prism_render_scene::shaders::*`
/// to resolve exactly as it will in the render world.
/// Byte-faithful copy of Bevy's `affine3_to_square` (transpose of the three
/// affine rows plus a `[0,0,0,1]` bottom row), used only to satisfy the
/// `bevy_render::maths` import while compiling `shading_resolve.wesl` off-GPU.
const MATHS_STUB: &str = "fn affine3_to_square(affine: mat3x4<f32>) -> mat4x4<f32> {\n    return transpose(mat4x4<f32>(\n        affine[0],\n        affine[1],\n        affine[2],\n        vec4<f32>(0.0, 0.0, 0.0, 1.0),\n    ));\n}\n";

#[test]
fn shading_resolve_wesl_compiles_and_resolves_all_imports() {
    let mut cache = ShaderCache::new((), load_source);

    // Register each dependency under the byte-identical embedded module path
    // that `load_shader_library!` produces at runtime.
    let deps: [(u128, &str, &str); 15] = [
        (
            0x5052_4953_4d5f_5441_4e47_454e_5400_0002,
            include_str!("../../shaders/tangent.wesl"),
            "embedded://prism_render_scene/shaders/tangent.wesl",
        ),
        (
            0x5052_4953_4d5f_4d54_5341_4d50_4c45_0002,
            include_str!("../../shaders/material_sample.wesl"),
            "embedded://prism_render_scene/shaders/material_sample.wesl",
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
            0x5052_4953_4d5f_5245_534f_554e_5041_0002,
            include_str!("../../shaders/material_unpack.wesl"),
            "embedded://prism_render_scene/shaders/material_unpack.wesl",
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
        (
            0x5052_4953_4d5f_4841_4952_0000_0000_0002,
            include_str!("../../shaders/hair.wesl"),
            "embedded://prism_render_scene/shaders/hair.wesl",
        ),
        (
            0x5052_4953_4d5f_5741_5445_5200_0000_0002,
            include_str!("../../shaders/water.wesl"),
            "embedded://prism_render_scene/shaders/water.wesl",
        ),
        (
            0x5052_4953_4d5f_434c_4541_5243_4f41_0002,
            include_str!("../../shaders/clearcoat.wesl"),
            "embedded://prism_render_scene/shaders/clearcoat.wesl",
        ),
        (
            0x5052_4953_4d5f_5348_4144_4f57_0000_0002,
            include_str!("../../shaders/shadow.wesl"),
            "embedded://prism_render_scene/shaders/shadow.wesl",
        ),
        // Minimal stand-in for Bevy's `bevy_render::maths`: the resolve shader
        // imports `affine3_to_square` from it exactly like `opaque.wesl` and
        // `visibility_raster.wesl` do at runtime. Bevy registers the real module
        // in the render world; the sandbox test has no GPU/render app, so we
        // register a byte-faithful copy of `affine3_to_square` under the same
        // module path (`bevy_render::maths`) so the importer resolves it.
        (
            0x4245_5659_5f52_4e44_5f4d_4154_4853_0001,
            MATHS_STUB,
            "embedded://bevy_render/maths.wesl",
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

/// Compiles `oit.wesl` standalone. It has no imports, so a green result
/// proves the weighted-blended OIT math (`McGuire` & Bavoil 2013 eq. 10 depth
/// weight, MRT accumulation and the fullscreen composite resolve) parses and
/// type-checks as WESL on its own, in lock-step with the CPU golden in
/// `prism_render_shading::oit`.
#[test]
fn oit_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let oit = shader_id(0x5052_4953_4d5f_4f49_5400_0000_0000_0001);
    cache.set_shader(
        oit,
        Shader::from_wesl(
            include_str!("../../shaders/oit.wesl"),
            "embedded://prism_render_scene/shaders/oit.wesl",
        ),
    );

    cache
        .get(0, oit, &[])
        .unwrap_or_else(|error| panic!("oit.wesl failed to compile: {error}"));
}

/// Compiles `gtao.wesl` standalone. It has no imports, so a green result
/// proves the GTAO horizon search, the Jimenez 2016 closed-form slice integral,
/// the view-space reconstruction/`uv_radius` helpers and the compute entry
/// point (group-0 depth/normal textures + storage AO output, `var<immediate>`
/// config) all parse and type-check as WESL on their own, in lock-step with the
/// CPU golden in `prism_render_shading::ao`.
#[test]
fn gtao_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let gtao = shader_id(0x5052_4953_4d5f_4741_4f00_0000_0000_0001);
    cache.set_shader(
        gtao,
        Shader::from_wesl(
            include_str!("../../shaders/gtao.wesl"),
            "embedded://prism_render_scene/shaders/gtao.wesl",
        ),
    );

    cache
        .get(0, gtao, &[])
        .unwrap_or_else(|error| panic!("gtao.wesl failed to compile: {error}"));
}
