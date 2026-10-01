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

/// Compiles `hair_sim.wesl` standalone. It has no imports, so a green result
/// proves the GPU guide-strand XPBD sim compute entry point
/// (`@compute @workgroup_size(64)` with group-0 storage particle/prev/goal/
/// rest-length/strand buffers and a `var<immediate>` `HairXpbdParams`), the
/// four constraint families (edge-length / local bending / global goal /
/// long-range attachment) and the substep×iteration Gauss-Seidel schedule all
/// parse and type-check as WESL on their own, in lock-step with the CPU golden
/// in `prism_render_architecture::hair::dynamics::simulate_guides`.
#[test]
fn hair_sim_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_sim = shader_id(0x5052_4953_4d5f_4841_4952_5f53_494d_0001);
    cache.set_shader(
        hair_sim,
        Shader::from_wesl(
            include_str!("../../shaders/hair_sim.wesl"),
            "embedded://prism_render_scene/shaders/hair_sim.wesl",
        ),
    );

    cache
        .get(0, hair_sim, &[])
        .unwrap_or_else(|error| panic!("hair_sim.wesl failed to compile: {error}"));
}

/// Compiles `hair_raster.wesl` standalone. It has no imports, so a green result
/// proves the GPU compute software hair rasterizer — the two vis-buffer entry
/// points (`hair_raster_depth` atomicMin nearest-depth scatter and
/// `hair_raster_resolve` id/coverage publish), the analytic capsule sub-pixel
/// coverage, the point-to-segment distance helper and the group-0 storage
/// segment/vis-buffer bindings plus the `var<immediate>` `HairRasterParamsGpu`
/// (mirroring the CPU `HairSoftRasterAbi`) all parse and type-check as WESL on
/// their own, in lock-step with the CPU classifier/binner in
/// `prism_render_architecture::hair::raster`.
#[test]
fn hair_raster_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_raster = shader_id(0x5052_4953_4d5f_4841_4952_5f52_5354_0001);
    cache.set_shader(
        hair_raster,
        Shader::from_wesl(
            include_str!("../../shaders/hair_raster.wesl"),
            "embedded://prism_render_scene/shaders/hair_raster.wesl",
        ),
    );

    cache
        .get(0, hair_raster, &[])
        .unwrap_or_else(|error| panic!("hair_raster.wesl failed to compile: {error}"));
}

/// Compiles `hair_transmittance.wesl` standalone. It has no imports, so a green
/// result proves the GPU strand self-shadow transmittance kernel — the voxel
/// density scatter, the running `product(1 - sigma)` alpha-composite over a
/// fixed per-invocation voxel slab, the per-texel `ranges` slicing into the
/// flat `samples` buffer and the `var<immediate>` `HairTransmittanceParams` —
/// parses and type-checks as WESL on its own, in lock-step with the voxel path
/// of the CPU golden in `prism_render_architecture::hair::deep_transmittance`.
#[test]
fn hair_transmittance_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_transmittance = shader_id(0x5052_4953_4d5f_4841_4952_5f54_5241_0001);
    cache.set_shader(
        hair_transmittance,
        Shader::from_wesl(
            include_str!("../../shaders/hair_transmittance.wesl"),
            "embedded://prism_render_scene/shaders/hair_transmittance.wesl",
        ),
    );

    cache
        .get(0, hair_transmittance, &[])
        .unwrap_or_else(|error| panic!("hair_transmittance.wesl failed to compile: {error}"));
}

/// Compiles `hair_interp.wesl` standalone. It has no imports, so a green result
/// proves the GPU guide-to-render interpolation kernel — the `splitmix64` hash
/// emulated over `vec2<u32>` 64-bit words, the range-reduced `sin_turns` curl,
/// the weighted blend, length jitter, clump pull and per-point position jitter,
/// the per-guide `guide_ranges` slicing into the flat `guide_points` buffer and
/// the `var<immediate>` `HairInterpParams` — parses and type-checks as WESL on
/// its own, in lock-step with the CPU golden in
/// `prism_render_architecture::hair::interpolation`.
#[test]
fn hair_interp_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_interp = shader_id(0x5052_4953_4d5f_4841_4952_5f49_4e54_0001);
    cache.set_shader(
        hair_interp,
        Shader::from_wesl(
            include_str!("../../shaders/hair_interp.wesl"),
            "embedded://prism_render_scene/shaders/hair_interp.wesl",
        ),
    );

    cache
        .get(0, hair_interp, &[])
        .unwrap_or_else(|error| panic!("hair_interp.wesl failed to compile: {error}"));
}

/// Compiles `hair_ribbon.wesl` standalone. It has no imports, so a green result
/// proves the GPU strand-to-ribbon card-meshing kernel — the two-pass arc
/// length accumulation driving `v`, the `+/-bitangent*radius` edge expansion,
/// the even-spacing zero-length fallback and the `vertex_offset`-biased
/// two-triangle-per-segment index emission — parses and type-checks as WESL on
/// its own, in lock-step with the CPU golden in
/// `prism_render_architecture::hair::ribbon::build_ribbon`.
#[test]
fn hair_ribbon_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_ribbon = shader_id(0x5052_4953_4d5f_4841_4952_5f52_424e_0001);
    cache.set_shader(
        hair_ribbon,
        Shader::from_wesl(
            include_str!("../../shaders/hair_ribbon.wesl"),
            "embedded://prism_render_scene/shaders/hair_ribbon.wesl",
        ),
    );

    cache
        .get(0, hair_ribbon, &[])
        .unwrap_or_else(|error| panic!("hair_ribbon.wesl failed to compile: {error}"));
}

/// Compiles `hair_mesh_shell.wesl` standalone. It has no imports, so a green
/// result proves the GPU strand-to-shell mesh-LOD kernel — the two-pass arc
/// length accumulation driving `v`, the four-corner rectangular section swept
/// by `+/-bitangent*half_width` and `+/-normal*half_thickness`, the diagonal
/// `normalize_or` outward normals, the even-spacing zero-length fallback and
/// the `vertex_offset`-biased eight-side-triangle-per-segment plus root/tip cap
/// index emission — parses and type-checks as WESL on its own, in lock-step
/// with the CPU golden in
/// `prism_render_architecture::hair::mesh_shell::build_shell`.
#[test]
fn hair_mesh_shell_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_mesh_shell = shader_id(0x5052_4953_4d5f_4841_4952_5f4d_5348_0001);
    cache.set_shader(
        hair_mesh_shell,
        Shader::from_wesl(
            include_str!("../../shaders/hair_mesh_shell.wesl"),
            "embedded://prism_render_scene/shaders/hair_mesh_shell.wesl",
        ),
    );

    cache
        .get(0, hair_mesh_shell, &[])
        .unwrap_or_else(|error| panic!("hair_mesh_shell.wesl failed to compile: {error}"));
}

/// Registers and compiles `hair_frames.wesl` standalone. A green result proves
/// the GPU coherent per-vertex strand-frame kernel — the double-reflection
/// rotation-minimizing transport (Wang et al. 2008) walked root-to-tip per
/// strand, the `normalize_or` degeneracy guard, the cardinal-axis
/// `orthonormal_reference` seed, the two `reflect_plane` transport steps with
/// their `c > EPSILON` coincident-point guards, the forward-difference
/// `strand_tangent`, the re-orthonormalization against the next tangent and the
/// right-handed `tangent x normal` bitangent — parses and type-checks as WESL
/// on its own, in lock-step with the CPU golden in
/// `prism_render_architecture::hair::frames::build_strand_frames`.
#[test]
fn hair_frames_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_frames = shader_id(0x5052_4953_4d5f_4841_4952_5f46_524d_0001);
    cache.set_shader(
        hair_frames,
        Shader::from_wesl(
            include_str!("../../shaders/hair_frames.wesl"),
            "embedded://prism_render_scene/shaders/hair_frames.wesl",
        ),
    );

    cache
        .get(0, hair_frames, &[])
        .unwrap_or_else(|error| panic!("hair_frames.wesl failed to compile: {error}"));
}

/// Registers and compiles `hair_wind.wesl` standalone. A green result proves
/// the GPU wind-field pre-pass kernel — the `normalize_or_zero` direction
/// guard, the `round_away` half-away-from-zero range reduction, the `fma`
/// Horner `sin_turns` mirroring the CPU `mul_add` chain, the gust phase mixing
/// position and time through `GUST_SWIRL`, the phase-shifted per-axis flutter,
/// the pinned-particle skip and the semi-implicit `acceleration * dt^2`
/// displacement with its non-positive / non-finite `dt` guards — parses and
/// type-checks as WESL on its own, in lock-step with the CPU golden in
/// `prism_render_architecture::hair::wind::apply_wind`.
#[test]
fn hair_wind_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_wind = shader_id(0x5052_4953_4d5f_4841_4952_5f57_4e44_0001);
    cache.set_shader(
        hair_wind,
        Shader::from_wesl(
            include_str!("../../shaders/hair_wind.wesl"),
            "embedded://prism_render_scene/shaders/hair_wind.wesl",
        ),
    );

    cache
        .get(0, hair_wind, &[])
        .unwrap_or_else(|error| panic!("hair_wind.wesl failed to compile: {error}"));
}

/// Registers and compiles `hair_sdf_collision.wesl` standalone. A green result
/// proves the GPU SDF body-collision post-pass — the packed
/// sphere/capsule/half-space/box primitive distances, the union min-reduction,
/// the central-difference field gradient with its `normalize_or_zero` guard,
/// and the per-iteration push-out relaxation (with the flat-field `+Y` escape,
/// the NaN self-inequality guard and the pinned-particle skip) — parses and
/// type-checks as WESL on its own, in lock-step with the CPU golden in
/// `prism_render_architecture::hair::sdf_collision::resolve_sdf_collisions`.
#[test]
fn hair_sdf_collision_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_sdf = shader_id(0x5052_4953_4d5f_4841_4952_5f53_4446_0001);
    cache.set_shader(
        hair_sdf,
        Shader::from_wesl(
            include_str!("../../shaders/hair_sdf_collision.wesl"),
            "embedded://prism_render_scene/shaders/hair_sdf_collision.wesl",
        ),
    );

    cache
        .get(0, hair_sdf, &[])
        .unwrap_or_else(|error| panic!("hair_sdf_collision.wesl failed to compile: {error}"));
}

/// Registers and compiles `hair_root_skinning.wesl` standalone. A green result
/// proves the GPU groom-root skinning resolve — the packed barycentric+height
/// mesh binding, the flat `u32` triangle index reads with their out-of-range
/// degrade-to-identity guards, the zero-area face rejection, the
/// `f32::EPSILON`-guarded `normalize_or` mirror and the right-handed
/// tangent/normal/bitangent re-orthonormalization — parses and type-checks as
/// WESL on its own, in lock-step with the CPU golden in
/// `prism_render_architecture::hair::binding::resolve_root_frames`.
#[test]
fn hair_root_skinning_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_root = shader_id(0x5052_4953_4d5f_4841_4952_5f52_534b_0001);
    cache.set_shader(
        hair_root,
        Shader::from_wesl(
            include_str!("../../shaders/hair_root_skinning.wesl"),
            "embedded://prism_render_scene/shaders/hair_root_skinning.wesl",
        ),
    );

    cache
        .get(0, hair_root, &[])
        .unwrap_or_else(|error| panic!("hair_root_skinning.wesl failed to compile: {error}"));
}

/// Registers and compiles `hair_root_bind.wesl` standalone. A green result
/// proves the GPU import-time groom-root binding projection — the brute-force
/// closest-triangle scan with its first-strict-minimum tie-break, the Ericson
/// Voronoi-region `closest_point_on_triangle` clamp (all vertex/edge/face
/// branches with their `EPS_LEN_SQ`-guarded denominators), the signed-height
/// projection onto the `f32::EPSILON`-guarded face normal, the out-of-range
/// vertex-index skip and the unbound-sentinel fallback — parses and type-checks
/// as WESL on its own, in lock-step with the CPU golden in
/// `prism_render_architecture::hair::binding::bind_roots`.
#[test]
fn hair_root_bind_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_bind = shader_id(0x5052_4953_4d5f_4841_4952_5f42_4e44_0001);
    cache.set_shader(
        hair_bind,
        Shader::from_wesl(
            include_str!("../../shaders/hair_root_bind.wesl"),
            "embedded://prism_render_scene/shaders/hair_root_bind.wesl",
        ),
    );

    cache
        .get(0, hair_bind, &[])
        .unwrap_or_else(|error| panic!("hair_root_bind.wesl failed to compile: {error}"));
}

/// Registers and compiles `hair_lod_dither.wesl` standalone. A green result
/// proves the GPU per-strand LOD cross-fade dither mask — the `vec2<u32>`
/// `splitmix64` 64-bit finalizer shared with the interpolation kernel, the
/// `hash_to_unit` top-24-bit mantissa, and the `hash >= clamp(blend, 0, 1)`
/// keep test — parses and type-checks as WESL on its own, in lock-step with the
/// CPU golden in
/// `prism_render_architecture::hair::transition::strand_survives_dither`.
#[test]
fn hair_lod_dither_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_dither = shader_id(0x5052_4953_4d5f_4841_4952_5f44_5448_0001);
    cache.set_shader(
        hair_dither,
        Shader::from_wesl(
            include_str!("../../shaders/hair_lod_dither.wesl"),
            "embedded://prism_render_scene/shaders/hair_lod_dither.wesl",
        ),
    );

    cache
        .get(0, hair_dither, &[])
        .unwrap_or_else(|error| panic!("hair_lod_dither.wesl failed to compile: {error}"));
}

/// Compiles the hair arc-length resampling compute twin standalone. A green
/// result proves the per-strand groom-import resampler — the two-pass total
/// arc length plus incremental segment-cursor reparameterization, with the
/// `len == 1` and zero-length degenerate branches — parses and type-checks as
/// WESL through the render-world `ShaderCache` / `wesl` pipeline, in lock-step
/// with the CPU golden
/// `prism_render_architecture::hair::groom_import::resample_strand`.
#[test]
fn hair_resample_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_resample = shader_id(0x5052_4953_4d5f_4841_4952_5f52_534d_0001);
    cache.set_shader(
        hair_resample,
        Shader::from_wesl(
            include_str!("../../shaders/hair_resample.wesl"),
            "embedded://prism_render_scene/shaders/hair_resample.wesl",
        ),
    );

    cache
        .get(0, hair_resample, &[])
        .unwrap_or_else(|error| panic!("hair_resample.wesl failed to compile: {error}"));
}

/// Compiles the hair deep opacity map packing compute twin standalone. A green
/// result proves the per-texel self-shadow slab packer — the fixed equal-width
/// depth-layer slicing plus the `alpha`-composite running product over each
/// texel's host-pre-sorted sample slice, with the empty-texel fully
/// transmissive branch — parses and type-checks as WESL through the
/// render-world `ShaderCache` / `wesl` pipeline, in lock-step with the CPU
/// golden
/// `prism_render_architecture::hair::deep_opacity_layout::build_deep_opacity_map`.
#[test]
fn hair_deep_opacity_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_deep_opacity = shader_id(0x5052_4953_4d5f_4841_4952_5f44_4f50_0001);
    cache.set_shader(
        hair_deep_opacity,
        Shader::from_wesl(
            include_str!("../../shaders/hair_deep_opacity.wesl"),
            "embedded://prism_render_scene/shaders/hair_deep_opacity.wesl",
        ),
    );

    cache
        .get(0, hair_deep_opacity, &[])
        .unwrap_or_else(|error| panic!("hair_deep_opacity.wesl failed to compile: {error}"));
}

/// Compiles the hair forward-scatter crossing-count packing compute twin
/// standalone. A green result proves the additive sibling of the deep-opacity
/// packer — the fixed equal-width depth-layer slicing plus the `alpha`-sum
/// running crossing count over each texel's host-pre-sorted sample slice, with
/// the empty-texel zero-crossing branch — parses and type-checks as WESL
/// through the render-world `ShaderCache` / `wesl` pipeline, in lock-step with
/// the CPU golden
/// `prism_render_architecture::hair::forward_scatter_layout::build_forward_scatter_map`.
#[test]
fn hair_forward_scatter_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_forward_scatter = shader_id(0x5052_4953_4d5f_4841_4952_5f46_5343_0001);
    cache.set_shader(
        hair_forward_scatter,
        Shader::from_wesl(
            include_str!("../../shaders/hair_forward_scatter.wesl"),
            "embedded://prism_render_scene/shaders/hair_forward_scatter.wesl",
        ),
    );

    cache
        .get(0, hair_forward_scatter, &[])
        .unwrap_or_else(|error| panic!("hair_forward_scatter.wesl failed to compile: {error}"));
}

/// Compiles the hair per-guide density-LOD metric compute twin standalone. A
/// green result proves the per-guide measurement kernel — the left-to-right
/// arc-length sum, the interior-vertex `1 - dot(t_in, t_out)` curvature sum
/// with the `normalize_or(ZERO)` degenerate-segment guard, and the clamped
/// authored root-radius readout — parses and type-checks as WESL through the
/// render-world `ShaderCache` / `wesl` pipeline, in lock-step with the CPU
/// golden `prism_render_architecture::hair::density_lod::guide_metrics`
/// (`decimation::strand_arc_length` / `strand_curvature`).
#[test]
fn hair_guide_metrics_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_guide_metrics = shader_id(0x5052_4953_4d5f_4841_4952_5f47_4d54_0001);
    cache.set_shader(
        hair_guide_metrics,
        Shader::from_wesl(
            include_str!("../../shaders/hair_guide_metrics.wesl"),
            "embedded://prism_render_scene/shaders/hair_guide_metrics.wesl",
        ),
    );

    cache
        .get(0, hair_guide_metrics, &[])
        .unwrap_or_else(|error| panic!("hair_guide_metrics.wesl failed to compile: {error}"));
}

/// Compiles the hair per-binding density-LOD metric blend compute twin
/// standalone. A green result proves the widest fan-out in the density-LOD
/// path — each render strand inheriting the weight-blended `(length,
/// curvature, authored)` triple of its up-to-four skinning guides — parses
/// and type-checks as WESL through the render-world `ShaderCache` / `wesl`
/// pipeline, in lock-step with the per-binding blend loop of the CPU golden
/// `prism_render_architecture::hair::density_lod::binding_importances`
/// (the `weight > 0 && idx < guide_count` guard and index-order `weight *
/// metric` accumulation before the groom-global `compute_importance` fold).
#[test]
fn hair_binding_metrics_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_binding_metrics = shader_id(0x5052_4953_4d5f_4841_4952_5f42_4d54_0001);
    cache.set_shader(
        hair_binding_metrics,
        Shader::from_wesl(
            include_str!("../../shaders/hair_binding_metrics.wesl"),
            "embedded://prism_render_scene/shaders/hair_binding_metrics.wesl",
        ),
    );

    cache
        .get(0, hair_binding_metrics, &[])
        .unwrap_or_else(|error| panic!("hair_binding_metrics.wesl failed to compile: {error}"));
}

/// Compiles the hair groom-global density-LOD importance fold compute twin
/// standalone. A green result proves the final density-LOD step — folding each
/// render strand's blended `(length, curvature, authored)` triple into a single
/// normalized `[0, 1]` importance via max-normalization, weighted blend and
/// weight-sum renormalization — parses and type-checks as WESL through the
/// render-world `ShaderCache` / `wesl` pipeline, in lock-step with the
/// per-element body of the CPU golden
/// `prism_render_architecture::hair::decimation::compute_importance` (the
/// `weight_sum <= 0` short-circuit, the `max > 0` normalization guards, the
/// `clamp(authored, 0, 1)` readout and the `clamp(blended / weight_sum, 0, 1)`
/// result). Together with `hair_guide_metrics` and `hair_binding_metrics` this
/// closes the density-LOD GPU twin path up to the CPU-only ranking sort.
#[test]
fn hair_importance_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_importance = shader_id(0x5052_4953_4d5f_4841_4952_5f49_4d50_0001);
    cache.set_shader(
        hair_importance,
        Shader::from_wesl(
            include_str!("../../shaders/hair_importance.wesl"),
            "embedded://prism_render_scene/shaders/hair_importance.wesl",
        ),
    );

    cache
        .get(0, hair_importance, &[])
        .unwrap_or_else(|error| panic!("hair_importance.wesl failed to compile: {error}"));
}

/// Compiles the hair per-particle motion-energy map compute twin standalone. A
/// green result proves the sleep-gate proxy — mapping each particle's implicit
/// velocity `position - prev_position` to its squared length `dot(v, v)`, the
/// per-element body summed host-side into a groom's motion energy — parses and
/// type-checks as WESL through the render-world `ShaderCache` / `wesl` pipeline,
/// in lock-step with the per-particle body of the CPU golden
/// `prism_render_architecture::hair::sleep::groom_motion_energy` (its
/// `particle_speed_squared` map, with pinned particles contributing `0`). The
/// particle layout matches the shared `hair_sim` state so no repacking is needed
/// between simulation and this proxy.
#[test]
fn hair_motion_energy_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_motion_energy = shader_id(0x5052_4953_4d5f_4841_4952_5f4d_4f5f_0001);
    cache.set_shader(
        hair_motion_energy,
        Shader::from_wesl(
            include_str!("../../shaders/hair_motion_energy.wesl"),
            "embedded://prism_render_scene/shaders/hair_motion_energy.wesl",
        ),
    );

    cache
        .get(0, hair_motion_energy, &[])
        .unwrap_or_else(|error| panic!("hair_motion_energy.wesl failed to compile: {error}"));
}

/// Registers `lighting.wesl`, `brdf.wesl` and `stylized_hair.wesl` under their
/// canonical module paths and compiles `stylized_hair.wesl`, forcing the
/// importer to resolve the `prism_render_scene::shaders::brdf::{...}` imports
/// the NPR front end depends on (`SurfaceSample`, `ShadingFrame`,
/// `DirectLightSample` and `brdf_normalize_or`; `brdf.wesl` in turn imports
/// `lighting.wesl`). A green result proves the stylized (angel-ring) hair
/// shading kernel — the wrapped cel diffuse ramp, the two shifted thresholded
/// `Kajiya-Kay` highlight bands and the unshadowed Fresnel rim — parses,
/// resolves its imports and type-checks as WESL, in lock-step with the CPU
/// golden `prism_render_shading::stylized_hair::evaluate_stylized_hair_direct`.
#[test]
fn stylized_hair_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_0009);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_0009);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let stylized_hair = shader_id(0x5052_4953_4d5f_4841_4952_5f4e_5052_0001);
    cache.set_shader(
        stylized_hair,
        Shader::from_wesl(
            include_str!("../../shaders/stylized_hair.wesl"),
            "embedded://prism_render_scene/shaders/stylized_hair.wesl",
        ),
    );

    cache.get(0, stylized_hair, &[]).unwrap_or_else(|error| {
        panic!("stylized_hair.wesl failed to compile/resolve imports: {error}")
    });
}

/// Registers `lighting.wesl`, `brdf.wesl` and `hair_angel_ring.wesl` under their
/// canonical module paths and compiles `hair_angel_ring.wesl`, forcing the
/// importer to resolve the `prism_render_scene::shaders::brdf::{...}` imports
/// the NPR multi-ring front end depends on (`SurfaceSample`, `ShadingFrame`,
/// `DirectLightSample` and `brdf_normalize_or`; `brdf.wesl` in turn imports
/// `lighting.wesl`). A green result proves the N-ring "angel ring" highlight
/// stack — the fixed-capacity `AngelRing` array, the tangent-decoupled
/// thresholded `Kajiya-Kay` band and the additive per-light accumulation —
/// parses, resolves its imports and type-checks as WESL, in lock-step with the
/// CPU golden `prism_render_shading::hair_angel_ring::accumulate_angel_rings`.
#[test]
fn hair_angel_ring_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_0009);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_0009);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let angel_ring = shader_id(0x5052_4953_4d5f_4841_4952_5f41_4e47_0001);
    cache.set_shader(
        angel_ring,
        Shader::from_wesl(
            include_str!("../../shaders/hair_angel_ring.wesl"),
            "embedded://prism_render_scene/shaders/hair_angel_ring.wesl",
        ),
    );

    cache.get(0, angel_ring, &[]).unwrap_or_else(|error| {
        panic!("hair_angel_ring.wesl failed to compile/resolve imports: {error}")
    });
}

/// Registers `lighting.wesl`, `brdf.wesl` and `hair_matcap.wesl` under their
/// canonical module paths and compiles `hair_matcap.wesl`, forcing the importer
/// to resolve the `prism_render_scene::shaders::brdf::{...}` imports the NPR
/// `MatCap` body shade depends on (`SurfaceSample`, `ShadingFrame` and
/// `brdf_normalize_or`; `brdf.wesl` in turn imports `lighting.wesl`). A green
/// result proves the material-capture front end — the view-basis `MatCap`
/// coordinate reconstruction (`hair_matcap_uv`) and the base/tint/strength
/// modulation with single emissive add (`hair_matcap_shade` /
/// `evaluate_hair_matcap`) — parses, resolves its imports and type-checks as
/// WESL, in lock-step with the CPU golden
/// `prism_render_shading::hair_matcap::evaluate_hair_matcap`.
#[test]
fn hair_matcap_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_0009);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_0009);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let matcap = shader_id(0x5052_4953_4d5f_4d41_5443_4150_0000_0001);
    cache.set_shader(
        matcap,
        Shader::from_wesl(
            include_str!("../../shaders/hair_matcap.wesl"),
            "embedded://prism_render_scene/shaders/hair_matcap.wesl",
        ),
    );

    cache.get(0, matcap, &[]).unwrap_or_else(|error| {
        panic!("hair_matcap.wesl failed to compile/resolve imports: {error}")
    });
}

/// Registers `lighting.wesl`, `brdf.wesl` and `hair_chiang.wesl` under their
/// canonical module paths and compiles `hair_chiang.wesl`, forcing the importer
/// to resolve the `prism_render_scene::shaders::brdf::{...}` imports the Chiang
/// near-field front end depends on (`SurfaceSample`, `ShadingFrame`,
/// `DirectLightSample` and `brdf_normalize_or`; `brdf.wesl` in turn imports
/// `lighting.wesl` for `INV_PI`). A green result proves the energy-conserving
/// close-up hair kernel — the color-to-`sigma_a` absorption fit, the
/// Beer-Lambert cortex transmittance, the R/TT/TRT cuticle lobes and the
/// hollow-core medulla forward-scatter fill — parses, resolves its imports and
/// type-checks as WESL, in lock-step with the CPU golden
/// `prism_render_shading::hair_chiang::evaluate_hair_chiang_direct`.
#[test]
fn hair_chiang_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_0009);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_0009);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let hair_chiang = shader_id(0x5052_4953_4d5f_4841_4952_5f43_4849_0001);
    cache.set_shader(
        hair_chiang,
        Shader::from_wesl(
            include_str!("../../shaders/hair_chiang.wesl"),
            "embedded://prism_render_scene/shaders/hair_chiang.wesl",
        ),
    );

    cache.get(0, hair_chiang, &[]).unwrap_or_else(|error| {
        panic!("hair_chiang.wesl failed to compile/resolve imports: {error}")
    });
}

/// Ensures the fibre-level fur/hair BSDF twin `hair_fiber.wesl` — the Yan et al.
/// dual-cylinder model with three cuticle lobes (R/TT/TRT) plus two medulla
/// scattered lobes (`TTs`/`TRTs`) shaped by a Henyey-Greenstein phase function —
/// parses, resolves its imports and type-checks as WESL, in lock-step with the
/// CPU golden `prism_render_shading::hair_fiber::evaluate_hair_fiber_direct`.
#[test]
fn hair_fiber_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_0009);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_0009);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let hair_fiber = shader_id(0x5052_4953_4d5f_4841_4952_5f46_4942_0001);
    cache.set_shader(
        hair_fiber,
        Shader::from_wesl(
            include_str!("../../shaders/hair_fiber.wesl"),
            "embedded://prism_render_scene/shaders/hair_fiber.wesl",
        ),
    );

    cache.get(0, hair_fiber, &[]).unwrap_or_else(|error| {
        panic!("hair_fiber.wesl failed to compile/resolve imports: {error}")
    });
}

/// Ensures the Kajiya-Kay strand-highlight twin `hair_kajiya.wesl` — the cheap
/// real-time fallback front end with one anisotropic `sin(T, L)` diffuse term
/// and one shifted anisotropic specular term about the strand tangent — parses,
/// resolves its imports and type-checks as WESL, in lock-step with the CPU
/// golden `prism_render_shading::hair_kajiya::evaluate_hair_kajiya_direct`.
#[test]
fn hair_kajiya_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_4c49_4748_5449_4e47_000a);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_4252_4446_0000_0000_000a);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let hair_kajiya = shader_id(0x5052_4953_4d5f_4841_4952_5f4b_4159_0001);
    cache.set_shader(
        hair_kajiya,
        Shader::from_wesl(
            include_str!("../../shaders/hair_kajiya.wesl"),
            "embedded://prism_render_scene/shaders/hair_kajiya.wesl",
        ),
    );

    cache.get(0, hair_kajiya, &[]).unwrap_or_else(|error| {
        panic!("hair_kajiya.wesl failed to compile/resolve imports: {error}")
    });
}

/// Ensures the advanced cloth twin `cloth_advanced.wesl` — energy-conserving
/// `Charlie` sheen, `Ashikhmin`-`Shirley` woven warp/weft anisotropy, thin
/// double-sided transmission, multiple-scattering compensation, thin-film
/// interference and a tension-driven wrinkle blend — parses, resolves its
/// `prism_render_scene::shaders::{lighting, brdf}::{...}` imports and
/// type-checks as WESL, in lock-step with the CPU golden
/// `prism_render_shading::cloth_advanced::evaluate_cloth_advanced_direct`.
#[test]
fn cloth_advanced_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let lighting = shader_id(0x5052_4953_4d5f_434c_4f54_485f_4144_1001);
    cache.set_shader(
        lighting,
        Shader::from_wesl(
            include_str!("../../shaders/lighting.wesl"),
            "embedded://prism_render_scene/shaders/lighting.wesl",
        ),
    );

    let brdf = shader_id(0x5052_4953_4d5f_434c_4f54_485f_4144_2001);
    cache.set_shader(
        brdf,
        Shader::from_wesl(
            include_str!("../../shaders/brdf.wesl"),
            "embedded://prism_render_scene/shaders/brdf.wesl",
        ),
    );

    let cloth_advanced = shader_id(0x5052_4953_4d5f_434c_4f54_485f_4144_0001);
    cache.set_shader(
        cloth_advanced,
        Shader::from_wesl(
            include_str!("../../shaders/cloth_advanced.wesl"),
            "embedded://prism_render_scene/shaders/cloth_advanced.wesl",
        ),
    );

    cache.get(0, cloth_advanced, &[]).unwrap_or_else(|error| {
        panic!("cloth_advanced.wesl failed to compile/resolve imports: {error}")
    });
}

/// Compiles `cloth_sim.wesl` on its own, asserting the GPU-driven XPBD cloth
/// solver kernels — `cloth_predict`, `cloth_project_distance`,
/// `cloth_project_long_range`, `cloth_strain_limit` and
/// `cloth_velocity_update` — parse and type-check as WESL, in lock-step with
/// the CPU golden
/// `prism_render_architecture::cloth::dynamics::solve_cloth_with_collision`.
#[test]
fn cloth_sim_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let cloth_sim = shader_id(0x5052_4953_4d5f_434c_4f54_485f_5349_4d01);
    cache.set_shader(
        cloth_sim,
        Shader::from_wesl(
            include_str!("../../shaders/cloth_sim.wesl"),
            "embedded://prism_render_scene/shaders/cloth_sim.wesl",
        ),
    );

    cache
        .get(0, cloth_sim, &[])
        .unwrap_or_else(|error| panic!("cloth_sim.wesl failed to compile: {error}"));
}

/// Compiles the GPU cloth body/self-collision kernel module standalone, so a
/// regression in `cloth_collision.wesl` (a bad binding, entry name, or `WGSL`
/// construct) fails the build even though the sandbox never dispatches it.
#[test]
fn cloth_collision_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let cloth_collision = shader_id(0x5052_4953_4d5f_434c_4f54_485f_434f_4c01);
    cache.set_shader(
        cloth_collision,
        Shader::from_wesl(
            include_str!("../../shaders/cloth_collision.wesl"),
            "embedded://prism_render_scene/shaders/cloth_collision.wesl",
        ),
    );

    cache
        .get(0, cloth_collision, &[])
        .unwrap_or_else(|error| panic!("cloth_collision.wesl failed to compile: {error}"));
}

/// Compiles the GPU render-mesh embedding kernel module standalone, guarding
/// `cloth_embed.wesl` against binding / entry-name / `WGSL` regressions.
#[test]
fn cloth_embed_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let cloth_embed = shader_id(0x5052_4953_4d5f_434c_4f54_485f_454d_4201);
    cache.set_shader(
        cloth_embed,
        Shader::from_wesl(
            include_str!("../../shaders/cloth_embed.wesl"),
            "embedded://prism_render_scene/shaders/cloth_embed.wesl",
        ),
    );

    cache
        .get(0, cloth_embed, &[])
        .unwrap_or_else(|error| panic!("cloth_embed.wesl failed to compile: {error}"));
}

/// Compiles `water_ocean.wesl` standalone. It has no imports, so a green result
/// proves the two GPU ocean-field compute entry points — `water_spectrum_ifft`
/// (Hermitian spectrum phase advance `h(k,t)=h0·e^{iωt}+conj(h0(−k))·e^{−iωt}`
/// with dispersion `ω=√(gk)`, direct inverse transform, choppy displacement,
/// slope normal and fold `Jacobian`) and `water_gerstner_displace` (multi-wave
/// `Gerstner` trochoidal displacement with the closed-form `GPU` Gems normal) —
/// parse and type-check as WESL, in lock-step with the CPU golden in
/// `prism_render_architecture::water::spectrum`.
#[test]
fn water_ocean_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let water_ocean = shader_id(0x5052_4953_4d5f_5741_5445_525f_4f43_4e01);
    cache.set_shader(
        water_ocean,
        Shader::from_wesl(
            include_str!("../../shaders/water_ocean.wesl"),
            "embedded://prism_render_scene/shaders/water_ocean.wesl",
        ),
    );

    cache
        .get(0, water_ocean, &[])
        .unwrap_or_else(|error| panic!("water_ocean.wesl failed to compile: {error}"));
}

/// Compiles `water_surface.wesl` standalone. It has no imports, so a green
/// result proves the three GPU shallow-water/surface compute entry points —
/// `water_swe_step` (conservative flux divergence, `CFL`-clamped timestep,
/// upwind self-advection, source injection), `water_foam_advect`
/// (semi-Lagrangian foam backtrace with bilinear resample and flow-aware
/// exponential decay) and `water_waterline_mask` (signed submersion depth,
/// soft transition band, shallow shoreline band) — parse and type-check as
/// WESL, in lock-step with the CPU goldens in
/// `prism_render_architecture::water::{swe, foam, waterline}`.
#[test]
fn water_surface_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let water_surface = shader_id(0x5052_4953_4d5f_5741_5445_525f_5346_4301);
    cache.set_shader(
        water_surface,
        Shader::from_wesl(
            include_str!("../../shaders/water_surface.wesl"),
            "embedded://prism_render_scene/shaders/water_surface.wesl",
        ),
    );

    cache
        .get(0, water_surface, &[])
        .unwrap_or_else(|error| panic!("water_surface.wesl failed to compile: {error}"));
}

/// Compiles `water_pbf.wesl` standalone. It has no imports, so a green result
/// proves the two GPU position-based-fluids compute entry points —
/// `water_pbf_density_solve` (spatial-hash 27-cell neighborhood, `Poly6`
/// density, `XPBD` lambda, `Spiky`-gradient position correction with the
/// `Macklin` artificial-pressure term) and `water_spray_emit` (breaking-source
/// classification and jet spawn with a summing `atomic` counter) — parse and
/// type-check as WESL, in lock-step with the CPU goldens in
/// `prism_render_architecture::water::{pbf, surface_fx}`.
#[test]
fn water_pbf_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let water_pbf = shader_id(0x5052_4953_4d5f_5741_5445_525f_5042_4601);
    cache.set_shader(
        water_pbf,
        Shader::from_wesl(
            include_str!("../../shaders/water_pbf.wesl"),
            "embedded://prism_render_scene/shaders/water_pbf.wesl",
        ),
    );

    cache
        .get(0, water_pbf, &[])
        .unwrap_or_else(|error| panic!("water_pbf.wesl failed to compile: {error}"));
}

/// Compiles `water_flip.wesl` standalone. It has no imports, so a green result
/// proves the four GPU `FLIP`/`APIC` compute entry points — `water_flip_p2g`
/// (two's-complement fixed-point `atomicAdd` momentum/mass scatter),
/// `water_flip_pressure_solve` (damped `Jacobi` pressure iteration toward a
/// divergence-free field), `water_flip_g2p` (grid-to-particle gather with the
/// projected velocity correction) and `water_surface_reconstruct` (screen-space
/// / anisotropic surface field) — parse and type-check as WESL, in lock-step
/// with the CPU goldens in `prism_render_architecture::water::{flip,
/// reconstruct}`.
#[test]
fn water_flip_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let water_flip = shader_id(0x5052_4953_4d5f_5741_5445_525f_464c_5001);
    cache.set_shader(
        water_flip,
        Shader::from_wesl(
            include_str!("../../shaders/water_flip.wesl"),
            "embedded://prism_render_scene/shaders/water_flip.wesl",
        ),
    );

    cache
        .get(0, water_flip, &[])
        .unwrap_or_else(|error| panic!("water_flip.wesl failed to compile: {error}"));
}

/// Compiles `water_render_fx.wesl` standalone. It has no imports, so a green
/// result proves the five GPU water render-effect compute entry points —
/// `water_caustics_project` (refraction `Jacobian` focus gain plus photon
/// splat), `water_dispersion_refract` (`Cauchy` `IOR` per-`RGB`-wavelength
/// refraction), `water_underwater_volume` (per-froxel `Beer-Lambert` extinction
/// with `Henyey-Greenstein` phase and god-ray in-scatter), `water_wetness_step`
/// (exponential wet/dry envelope, puddle integration, capillary band) and
/// `water_coupling_readback` (bounded buoyancy/drag/added-mass readback) —
/// parse and type-check as WESL, in lock-step with the CPU goldens in
/// `prism_render_architecture::water::{caustics, dispersion, underwater,
/// wetness, coupling}`.
#[test]
fn water_render_fx_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let water_render_fx = shader_id(0x5052_4953_4d5f_5741_5445_525f_4658_0001);
    cache.set_shader(
        water_render_fx,
        Shader::from_wesl(
            include_str!("../../shaders/water_render_fx.wesl"),
            "embedded://prism_render_scene/shaders/water_render_fx.wesl",
        ),
    );

    cache
        .get(0, water_render_fx, &[])
        .unwrap_or_else(|error| panic!("water_render_fx.wesl failed to compile: {error}"));
}

/// Compiles the hair self-collision compute twin standalone. It has no imports,
/// so a green result proves both entry points — `accumulate_self_collision`
/// (per-particle Jacobi correction gathered from the CSR uniform grid built by
/// `hair::self_collision_grid`, over the 27 surrounding cells via binary search)
/// and `apply_self_collision` (in-place correction add) — parse and type-check
/// as WESL through the render-world `ShaderCache` / `wesl` pipeline, in
/// lock-step with the CPU golden
/// `prism_render_architecture::hair::self_collision_jacobi`
/// (`accumulate_jacobi_corrections` / `apply_corrections`). The particle layout
/// (`xyz` position, `w` inverse mass) matches the shared `hair_sim` state so no
/// repacking is needed between simulation and this post-pass.
#[test]
fn hair_self_collision_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_self_collision = shader_id(0x5052_4953_4d5f_4841_4952_5f53_435f_0001);
    cache.set_shader(
        hair_self_collision,
        Shader::from_wesl(
            include_str!("../../shaders/hair_self_collision.wesl"),
            "embedded://prism_render_scene/shaders/hair_self_collision.wesl",
        ),
    );

    cache
        .get(0, hair_self_collision, &[])
        .unwrap_or_else(|error| panic!("hair_self_collision.wesl failed to compile: {error}"));
}

/// Compiles the hair VBD strand-solver compute twin standalone. It has no
/// imports, so a green result proves the `simulate_strand_vbd` entry point —
/// one invocation per guide strand running semi-implicit substeps whose
/// Gauss-Seidel vertex sweeps take one exact per-vertex Newton step against an
/// inertia + stretch + bending 3x3 Hessian (cofactor-inverted), then projecting
/// out of the analytic body colliders — parses and type-checks as WESL through
/// the render-world `ShaderCache` / `wesl` pipeline, in lock-step with the CPU
/// golden `prism_render_architecture::hair::solver::simulate_strand_vbd`. The
/// particle layout (`xyz` position, `w` inverse mass) matches the shared
/// `hair_sim` state so a groom can switch between the XPBD and VBD solver slots
/// without repacking.
#[test]
fn hair_vbd_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let hair_vbd = shader_id(0x5052_4953_4d5f_4841_4952_5f56_425f_0001);
    cache.set_shader(
        hair_vbd,
        Shader::from_wesl(
            include_str!("../../shaders/hair_vbd.wesl"),
            "embedded://prism_render_scene/shaders/hair_vbd.wesl",
        ),
    );

    cache
        .get(0, hair_vbd, &[])
        .unwrap_or_else(|error| panic!("hair_vbd.wesl failed to compile: {error}"));
}

// ───────────────────────── hair 真机 GPU parity ─────────────────────────
//
// 以上均为纯编译验证；`hair_wind` 以下是 hair 子系统**第一个真机设备 parity 测试**：
// 把 `hair_wind.wesl` 外力预 pass 内核在原生 compute 设备上 dispatch，与架构层 CPU
// 黄金 `apply_wind` 逐粒子对拍（design §9）。沙盒无 `GPU` 时跳过保绿；有真实设备
// （如 `Apple` `M` 系列 `GPU`）时跑满。

/// 真机 `GPU`-对-`CPU` 逐分量绝对容差。
///
/// `hair_wind` 内核与 CPU 黄金 `apply_wind` 跑同一份 `f32` 算术，唯一自由度是归一化里
/// CPU 的 `1.0 / sqrt` 与 `WESL` `inverseSqrt`（多为原生 `rsqrt`）之间几个 `ULP` 之差；
/// 位移量级 `O(dt^2)`，`1e-4` 远紧于任何真实内核 bug 的 `O(0.1)` 级发散。
const HAIR_WIND_PARITY_EPS: f32 = 1.0e-4;

/// `hair_wind.wesl` 的 `var<immediate> params: HairWindParams` 推常量块的 host 镜像。
///
/// 字段偏移逐一镜像 `WESL` 结构体：`vec3<f32>` 对齐 16，`speed` 填入其尾部 `12..16`
/// 的填充位；尾部 `_pad` 把尺寸补到 `vec3` 对齐的 48 字节，与 `naga` 为该块算出的推
/// 常量尺寸一致。无隐式填充，可安全派生 `Pod`。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HairWindImmediate {
    direction: [f32; 3],
    speed: f32,
    gust_amplitude: f32,
    gust_frequency: f32,
    turbulence: f32,
    time: f32,
    dt: f32,
    particle_count: u32,
    _pad: [u32; 2],
}

/// 把嵌入式 `hair_wind.wesl` 经 render-world [`ShaderCache`] 编译回 `Wgsl`（复用本
/// 模块的 `load_source` 闭包，不建设备）。
fn compile_hair_wind_wgsl() -> String {
    let mut cache = ShaderCache::new((), load_source);
    let id = shader_id(0x5052_4953_4d5f_4841_4952_5f57_4e44_0001);
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../shaders/hair_wind.wesl"),
            "embedded://prism_render_scene/shaders/hair_wind.wesl",
        ),
    );
    (*cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("hair_wind.wesl failed to compile: {error}")))
    .clone()
}

/// 在编译后的 `Wgsl` 里按子串定位 compute 入口的真实符号名（`WESL` 可能给模块内名字
/// 加前缀，故按子串而非固定符号查找）。
fn hair_wind_entry_point(wgsl: &str) -> String {
    for line in wgsl.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("fn ")
            && let Some(paren) = rest.find('(')
        {
            let name = &rest[..paren];
            if name.contains("hair_wind") {
                return name.to_string();
            }
        }
    }
    panic!("no compute entry point containing `hair_wind` in compiled Wgsl");
}

/// `hair_wind.wesl` 外力预 pass 内核在真机上 dispatch 一帧后，必须与架构层黄金
/// [`apply_wind`](prism_render_architecture::hair::wind::apply_wind) 逐粒子落在 `f32`
/// 舍入容差内：自由粒子按 `accel * dt^2` 位移、pin 根（逆质量 `0`）不动、`w`（逆质量）
/// 严格保持。
///
/// 无 `wgpu` adapter、或设备不支持 `immediate`（push-constant）的无头机上打印跳过提示
/// 而非失败，让套件在任何机器上保持绿；有真实设备时跑满 dispatch 并逐值对拍。
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
#[expect(
    clippy::too_many_lines,
    reason = "一条线性的取设备-建管线-建缓冲-dispatch-读回-对拍让 parity 路径整体可审计"
)]
fn hair_wind_gpu_matches_cpu_golden() {
    use bevy_platform::future::block_on;
    use prism_render_architecture::hair::dynamics::{StrandParticle, Vec3};
    use prism_render_architecture::hair::wind::{apply_wind, WindField};
    use wgpu::util::{BufferInitDescriptor, DeviceExt};
    use wgpu::{
        BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
        CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
        DeviceDescriptor, Features, Instance, InstanceDescriptor, InstanceFlags, MapMode,
        PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, RequestAdapterOptions,
        ShaderModuleDescriptor, ShaderSource, ShaderStages,
    };

    let immediate_size = size_of::<HairWindImmediate>() as u32;

    // --- 尽力取一个支持 immediate 的 compute 设备（无则跳过保绿）---
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let Some(adapter) = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()
    else {
        eprintln!("hair_wind_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity");
        return;
    };
    if !adapter.features().contains(Features::IMMEDIATES)
        || adapter.limits().max_immediate_size < immediate_size
    {
        eprintln!(
            "hair_wind_gpu_matches_cpu_golden: adapter lacks IMMEDIATES / immediate size, \
             skipping on-device parity"
        );
        return;
    }
    let limits = adapter.limits();
    let Some((device, queue)) = block_on(adapter.request_device(&DeviceDescriptor {
        required_features: Features::IMMEDIATES,
        required_limits: limits,
        ..Default::default()
    }))
    .ok() else {
        eprintln!(
            "hair_wind_gpu_matches_cpu_golden: request_device failed, skipping on-device parity"
        );
        return;
    };

    // --- 确定性工况：3 根 strand，每根 1 pin 根 + 3 自由粒子，非零风场带 gust/flutter ---
    let field = WindField {
        direction: Vec3::new(1.0, 0.2, -0.3),
        speed: 2.5,
        gust_amplitude: 1.2,
        gust_frequency: 0.8,
        turbulence: 0.4,
    };
    let time = 1.37_f32;
    let dt = 1.0_f32 / 120.0;

    let mut particles = Vec::new();
    for s in 0..3u32 {
        for v in 0..4u32 {
            let base = Vec3::new(
                s as f32 * 0.05 - 0.05,
                0.4 - v as f32 * 0.1,
                s as f32 * 0.02,
            );
            if v == 0 {
                particles.push(StrandParticle::pinned(base));
            } else {
                particles.push(StrandParticle::free(base));
            }
        }
    }
    let count = particles.len();

    // --- CPU 黄金：副本上原地推进一帧风场 ---
    let mut golden = particles.clone();
    apply_wind(&mut golden, field, time, dt);

    // --- host 上传口径：positions.w = 逆质量（与 `hair_sim` 同布局）---
    let positions: Vec<[f32; 4]> = particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect();

    // --- GPU 重放：编译 → 建管线 → dispatch → 读回 ---
    let wgsl = compile_hair_wind_wgsl();
    let entry = hair_wind_entry_point(&wgsl);

    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("hair_wind_parity_group0"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("hair_wind_parity_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("hair_wind_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("hair_wind_parity_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(&entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let bytes = size_of_val(positions.as_slice()) as u64;
    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("hair_wind_positions"),
        contents: bytemuck::cast_slice(&positions),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("hair_wind_positions_stage"),
        size: bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("hair_wind_parity_bind"),
        layout: &layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: positions_buf.as_entire_binding(),
        }],
    });

    let immediate = HairWindImmediate {
        direction: [field.direction.x, field.direction.y, field.direction.z],
        speed: field.speed,
        gust_amplitude: field.gust_amplitude,
        gust_frequency: field.gust_frequency,
        turbulence: field.turbulence,
        time,
        dt,
        particle_count: count as u32,
        _pad: [0; 2],
    };
    let groups = (count as u32).div_ceil(64).max(1);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("hair_wind_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("hair_wind_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&immediate));
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&positions_buf, 0, &stage, 0, bytes);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped positions readback range should be available after poll");
    let out: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&view).to_vec();
    drop(view);
    stage.unmap();

    // --- 逐粒子对拍：xyz 落容差内，w（逆质量）保持 ---
    assert_eq!(out.len(), golden.len());
    for (i, (gpu, cpu)) in out.iter().zip(golden.iter()).enumerate() {
        assert!(
            (gpu[0] - cpu.position.x).abs() < HAIR_WIND_PARITY_EPS
                && (gpu[1] - cpu.position.y).abs() < HAIR_WIND_PARITY_EPS
                && (gpu[2] - cpu.position.z).abs() < HAIR_WIND_PARITY_EPS,
            "particle {i}: GPU {gpu:?} vs CPU golden ({}, {}, {})",
            cpu.position.x,
            cpu.position.y,
            cpu.position.z,
        );
        assert!(
            (gpu[3] - cpu.inverse_mass).abs() < HAIR_WIND_PARITY_EPS,
            "particle {i}: inverse mass must be preserved (GPU {} vs {})",
            gpu[3],
            cpu.inverse_mass,
        );
    }
}
