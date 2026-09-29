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
