//! Compute-pipeline construction for the production ray-traversal service.
//!
//! The three sibling `WESL` kernels (`shaders/ray_traverse.wesl`,
//! `shaders/tlas_traverse.wesl`, `shaders/ray_footprint.wesl`) are composed
//! through the render world's [`ShaderCache`] - the exact path the
//! `#[cfg(test)]` parity harness uses - so the production service and the
//! on-device parity tests compile byte-identical `WGSL`. Each compiled module
//! is handed to a `layout: None` (auto-layout) compute pipeline, matching the
//! proven test dispatch and letting `wgpu` derive the bind-group layout from the
//! shader's `@group(0)` bindings.
//!
//! Provenance: classical `BVH` / `TLAS` traversal and ray-cone footprint math
//! owned by `prism_render_architecture::ray_scene`; no Unreal Engine source or
//! derived code, and no AI/ML.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_render::render_resource::{
    ComputePipeline, PipelineCompilationOptions, RawComputePipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource,
};
use bevy_render::renderer::RenderDevice;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

/// The three compiled compute pipelines backing the ray-traversal service.
///
/// Each pipeline is built once from the auto-derived bind-group layout of its
/// kernel and reused across every dispatch; the service owns one of these for
/// the lifetime of the device.
pub(crate) struct RayTraversalPipelines {
    /// Single-`BLAS` `ray_traverse` nearest-hit / any-hit pipeline.
    pub(crate) bvh: ComputePipeline,
    /// Two-level `tlas_traverse` nearest-hit / any-hit pipeline.
    pub(crate) tlas: ComputePipeline,
    /// Ray-cone `ray_footprint` texture-`LOD` pipeline.
    pub(crate) footprint: ComputePipeline,
}

impl RayTraversalPipelines {
    /// Compiles the three kernels and builds their compute pipelines on
    /// `device`.
    pub(crate) fn new(device: &RenderDevice) -> Self {
        let bvh_wgsl = compile_wgsl(
            include_str!("../shaders/ray_traverse.wesl"),
            "embedded://prism_render_scene/shaders/ray_traverse.wesl",
            0x5052_4953_4d5f_5241_5954_5241_5645_0003,
        );
        let tlas_wgsl = compile_wgsl(
            include_str!("../shaders/tlas_traverse.wesl"),
            "embedded://prism_render_scene/shaders/tlas_traverse.wesl",
            0x5052_4953_4d5f_5449_4c41_5354_5256_0003,
        );
        let footprint_wgsl = compile_wgsl(
            include_str!("../shaders/ray_footprint.wesl"),
            "embedded://prism_render_scene/shaders/ray_footprint.wesl",
            0x5052_4953_4d5f_464f_4f54_5052_4e54_0003,
        );

        let bvh = build_pipeline(device, &bvh_wgsl, "ray_traverse", "prism_rt_bvh");
        let tlas = build_pipeline(device, &tlas_wgsl, "tlas_traverse", "prism_rt_tlas");
        let footprint = build_pipeline(
            device,
            &footprint_wgsl,
            "ray_footprint",
            "prism_rt_footprint",
        );

        Self {
            bvh,
            tlas,
            footprint,
        }
    }
}

/// Streams the `WGSL` translation back out of the shader cache without a device.
///
/// Mirrors the closure the `#[cfg(test)]` parity harness uses so the production
/// service composes its `WESL` through the identical render-world pipeline;
/// here the compiled `WGSL` string is kept (rather than a device module) so it
/// can be handed to the service's own device.
fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("the ray-traversal shaders are WESL"),
    }
}

/// Compiles one `WESL` `source` (identified by `path` and a stable `uuid`) to
/// its `WGSL` translation through the render-world [`ShaderCache`].
fn compile_wgsl(source: &'static str, path: &'static str, uuid: u128) -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(uuid),
    };
    cache.set_shader(id, Shader::from_wesl(source, path));
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("{path} failed to compile: {error}"));
    (*module).clone()
}

/// Finds the compute entry point whose name contains `needle` in `wgsl`.
///
/// The `WESL` compiler may prefix module-local names, so the entry point is
/// located by substring rather than assuming a fixed symbol - identical to the
/// parity harness.
fn find_entry_point(wgsl: &str, needle: &str) -> String {
    for line in wgsl.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("fn ")
            && let Some(paren) = rest.find('(')
        {
            let name = &rest[..paren];
            if name.contains(needle) {
                return name.to_string();
            }
        }
    }
    panic!("no compute entry point containing `{needle}` in compiled WGSL");
}

/// Builds one auto-layout compute pipeline for the `needle` entry point of
/// `wgsl`.
fn build_pipeline(device: &RenderDevice, wgsl: &str, needle: &str, label: &str) -> ComputePipeline {
    let entry = find_entry_point(wgsl, needle);
    let module = device
        .wgpu_device()
        .create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(wgsl.into()),
        });
    device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some(&entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    })
}
