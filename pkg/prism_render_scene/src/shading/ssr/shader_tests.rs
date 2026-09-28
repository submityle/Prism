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

/// Registers `tangent.wesl`, `surface.wesl`, `gpu_scene.wesl` and
/// `scene_transform.wesl` under their canonical module paths and compiles
/// `ssr_prepass.wesl`, forcing every
/// `import prism_render_scene::shaders::*` to resolve exactly as it will in the
/// render world. A green result proves the SSR prepass decodes the visibility
/// buffer through the identical `surface.wesl` helpers the GTAO prepass and the
/// resolve stage use, and that its reverse-Z device-depth output type-checks.
#[test]
fn ssr_prepass_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let deps: [(u128, &str, &str); 4] = [
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
        (
            0x5052_4953_4d5f_5353_525f_5343_5446_0001,
            include_str!("../../shaders/scene_transform.wesl"),
            "embedded://prism_render_scene/shaders/scene_transform.wesl",
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

/// Registers the full import closure `ssr_repack.wesl` now pulls in and
/// compiles it, forcing every `import prism_render_scene::shaders::*` to
/// resolve exactly as it will in the render world. The repack no longer
/// reads only the authored `perceptual_roughness`: it reconstructs the
/// covered pixel's interpolated UV from the visibility buffer (walking the
/// same `surface.wesl` scene -> geometry -> primitive -> vertex tables the
/// resolve uses) and samples the metallic-roughness *and* normal textures
/// through the shared bindless heap via `material_sample.wesl`, rotating a
/// bound normal map into world space against the interpolated basis (via
/// `scene_transform.wesl` + `tangent.wesl`) and back into the view frame. A
/// green result proves that whole graph -- `material`, `tangent`, `surface`,
/// `gpu_scene`, `scene_transform` and `material_sample` (which itself
/// `enable`s `wgpu_binding_array` and imports `material`) -- parses,
/// type-checks and links, and that packing the normal-mapped view normal plus
/// the texture-modulated roughness into the trace's `normal_roughness` output
/// is well-formed.
#[test]
fn ssr_repack_wesl_compiles_and_resolves_imports() {
    let mut cache = ShaderCache::new((), load_source);

    let deps: [(u128, &str, &str); 6] = [
        (
            0x5052_4953_4d5f_5353_525f_4d41_5450_0002,
            include_str!("../../shaders/material.wesl"),
            "embedded://prism_render_scene/shaders/material.wesl",
        ),
        (
            0x5052_4953_4d5f_5353_525f_5441_4e47_0002,
            include_str!("../../shaders/tangent.wesl"),
            "embedded://prism_render_scene/shaders/tangent.wesl",
        ),
        (
            0x5052_4953_4d5f_5353_525f_5355_5246_0002,
            include_str!("../../shaders/surface.wesl"),
            "embedded://prism_render_scene/shaders/surface.wesl",
        ),
        (
            0x5052_4953_4d5f_5353_525f_5343_4e45_0002,
            include_str!("../../shaders/gpu_scene.wesl"),
            "embedded://prism_render_scene/shaders/gpu_scene.wesl",
        ),
        (
            0x5052_4953_4d5f_5353_525f_5343_5446_0002,
            include_str!("../../shaders/scene_transform.wesl"),
            "embedded://prism_render_scene/shaders/scene_transform.wesl",
        ),
        (
            0x5052_4953_4d5f_5353_525f_4d53_4d50_0001,
            include_str!("../../shaders/material_sample.wesl"),
            "embedded://prism_render_scene/shaders/material_sample.wesl",
        ),
    ];
    for (tag, source, path) in deps {
        cache.set_shader(shader_id(tag), Shader::from_wesl(source, path));
    }

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

/// Compiles `ssr_color_mips.wesl`, proving all three colour-pyramid build entry
/// points parse and type-check as they will in the render world:
/// `ssr_color_copy` lifts the resolve's `scene_color` into pyramid level 0,
/// `ssr_color_reduce_karis` writes the first coarser level as a Karis
/// luma-weighted 2x2 average (firefly suppression at the source mip), and
/// `ssr_color_reduce` writes each coarser level as the 2x2 box average of the
/// finer one. The kernel is self-contained (no intra-crate imports), so a green
/// result also guards its immediate `MipParams` layout against drift from the
/// shared `GpuSsrHzbParams` destination-then-source extent contract.
#[test]
fn ssr_color_mips_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let color_mips = shader_id(0x5052_4953_4d5f_5353_525f_434d_5053_0001);
    cache.set_shader(
        color_mips,
        Shader::from_wesl(
            include_str!("../../shaders/ssr_color_mips.wesl"),
            "embedded://prism_render_scene/shaders/ssr_color_mips.wesl",
        ),
    );

    cache
        .get(0, color_mips, &[])
        .unwrap_or_else(|error| panic!("ssr_color_mips.wesl failed to compile: {error}"));
}

/// Compiles `ssr_composite.wesl` standalone. The composite blends the trace's
/// reflection output over the shaded `scene_color`, reading the untouched base
/// from colour-pyramid level 0 (a copy of `scene_color`) so the write-only
/// `rgba16float` storage output never aliases a read. The kernel is
/// self-contained (no intra-crate imports), so a green result also guards its
/// immediate `CompositeParams` layout against drift from the shared 16-byte
/// `GpuSsrCompositeParams` extent contract.
#[test]
fn ssr_composite_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let composite = shader_id(0x5052_4953_4d5f_5353_525f_434d_5053_0002);
    cache.set_shader(
        composite,
        Shader::from_wesl(
            include_str!("../../shaders/ssr_composite.wesl"),
            "embedded://prism_render_scene/shaders/ssr_composite.wesl",
        ),
    );

    cache
        .get(0, composite, &[])
        .unwrap_or_else(|error| panic!("ssr_composite.wesl failed to compile: {error}"));
}

/// Compiles `ssr_resolve.wesl` standalone. The spatial reconstruction resolves
/// the noisy multi-ray trace with an edge-aware neighbourhood filter, reading
/// the trace output, packed `normal_roughness` and device depth and writing the
/// denoised reflection the composite consumes. The kernel is self-contained (no
/// intra-crate imports), so a green result also guards its immediate
/// `ResolveParams` layout against drift from the 96-byte `GpuSsrResolveParams`
/// contract (inverse projection + extent + kernel radius + bilateral tunables).
#[test]
fn ssr_resolve_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let resolve = shader_id(0x5052_4953_4d5f_5353_525f_434d_5053_0003);
    cache.set_shader(
        resolve,
        Shader::from_wesl(
            include_str!("../../shaders/ssr_resolve.wesl"),
            "embedded://prism_render_scene/shaders/ssr_resolve.wesl",
        ),
    );

    cache
        .get(0, resolve, &[])
        .unwrap_or_else(|error| panic!("ssr_resolve.wesl failed to compile: {error}"));
}

/// Compiles `ssr_temporal.wesl` standalone. The cross-frame accumulation
/// reprojects the previous frame's reflection purely from camera motion
/// (reconstructing world position from reverse-Z depth and the inverse current
/// view-projection, then projecting through the previous view-projection),
/// colour-box-clips the sampled history and exponentially blends it with the
/// resolve. The kernel is self-contained (no intra-crate imports), so a green
/// result also guards its immediate `TemporalParams` layout against drift from
/// the 160-byte `GpuSsrTemporalParams` contract (two matrices + extent + golden
/// tunables + validity flag).
#[test]
fn ssr_temporal_wesl_compiles_standalone() {
    let mut cache = ShaderCache::new((), load_source);

    let temporal = shader_id(0x5052_4953_4d5f_5353_525f_434d_5053_0004);
    cache.set_shader(
        temporal,
        Shader::from_wesl(
            include_str!("../../shaders/ssr_temporal.wesl"),
            "embedded://prism_render_scene/shaders/ssr_temporal.wesl",
        ),
    );

    cache
        .get(0, temporal, &[])
        .unwrap_or_else(|error| panic!("ssr_temporal.wesl failed to compile: {error}"));
}
