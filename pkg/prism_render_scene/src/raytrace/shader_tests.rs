//! `WESL` compilation coverage for the ray-traversal kernel.
//!
//! This compiles `shaders/ray_traverse.wesl` through the same [`ShaderCache`] /
//! `wesl` pipeline the render world uses, so a green result proves the source
//! parses and type-checks exactly as it will on device. The kernel is
//! self-contained (one nine-word `@group(0)` resource set, no cross-shader
//! `import`s), so compiling it also guards the packed-buffer traversal maths -
//! the reciprocal-slab rejection, the near/far child ordering by split-axis
//! sign, the running `t_max` shrink and the double-sided Möller–Trumbore test -
//! against drift from its `CPU` golden twin,
//! `prism_render_architecture::ray_scene::gpu_layout::GpuBvhBuffers`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("the ray-traversal shader is WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

/// Compiles a single self-contained compute shader, panicking with the shader
/// name on any parse / type-check failure.
fn compile_standalone(source: &'static str, path: &'static str, tag: u128) {
    let mut cache = ShaderCache::new((), load_source);
    let id = shader_id(tag);
    cache.set_shader(id, Shader::from_wesl(source, path));
    cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("{path} failed to compile: {error}"));
}

/// The bottom-level `BVH` closest-hit / any-hit walk, the real-device twin of
/// `GpuBvhBuffers::closest_hit` / `GpuBvhBuffers::any_hit`.
#[test]
fn ray_traverse_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/ray_traverse.wesl"),
        "embedded://prism_render_scene/shaders/ray_traverse.wesl",
        0x5052_4953_4d5f_5241_5954_5241_5645_0001,
    );
}

/// The top-level `TLAS` closest-hit / any-hit walk, the real-device twin of
/// `GpuTlasBuffers::closest_hit` / `GpuTlasBuffers::any_hit` over the shared
/// `GpuBlasPool`.
#[test]
fn tlas_traverse_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/tlas_traverse.wesl"),
        "embedded://prism_render_scene/shaders/tlas_traverse.wesl",
        0x5052_4953_4d5f_5449_4c41_5354_5256_0001,
    );
}

/// The ray-cone footprint / texture-`LOD` kernel, the real-device twin of the
/// golden `RayFootprint` mip math (`projected_width` / `texel_span` /
/// `mip_level` / `mip_floor`) and `log2_linear`.
#[test]
fn ray_footprint_wesl_compiles_standalone() {
    compile_standalone(
        include_str!("../shaders/ray_footprint.wesl"),
        "embedded://prism_render_scene/shaders/ray_footprint.wesl",
        0x5052_4953_4d5f_4655_5450_5249_4e54_0001,
    );
}
