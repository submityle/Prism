//! The production ray-traversal service: a reusable raw-`wgpu` facade that
//! turns the golden packed acceleration structures into real on-device compute
//! dispatches.
//!
//! [`GpuRayTraversal`] compiles the three kernels once (via
//! [`RayTraversalPipelines`]) and exposes synchronous, consumer-callable walks
//! that each upload the inputs, record a single compute pass, copy the hit
//! buffer to a mappable staging buffer, submit, block on the device poll, and
//! decode the read-back words:
//!
//! * [`GpuRayTraversal::closest_hits`] / [`GpuRayTraversal::any_hits`] - the
//!   single-`BLAS` `ray_traverse` nearest-hit and occlusion walks, the device
//!   twins of `GpuBvhBuffers::closest_hit` / `any_hit`;
//! * [`GpuRayTraversal::tlas_closest_hits`] / [`GpuRayTraversal::tlas_any_hits`]
//!   - the two-level `tlas_traverse` twins of `GpuTlasBuffers::closest_hit` /
//!   `any_hit`;
//! * [`GpuRayTraversal::footprint_mips`] - the ray-cone `ray_footprint`
//!   texture-`LOD` twin of `RayFootprint`.
//!
//! Unlike the render-graph compute nodes elsewhere in this crate, the service
//! is deliberately a plain `RenderDevice` / `RenderQueue` object: there is no
//! single render-graph consumer yet (screen-space / world-space reflections and
//! ray-traced shadows each want a different schedule), so promoting the proven
//! parity harness into a reusable service is the honest production shape. A
//! consumer that later needs a persistent graph node can wrap this service
//! without re-porting the kernels.

use bevy_render::render_resource::{
    Buffer, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, MapMode, PollType,
};
use bevy_render::renderer::{RenderDevice, RenderQueue};

use prism_render_architecture::ray_scene::{GpuBlasPool, GpuBvhBuffers, GpuTlasBuffers, Ray};

use super::abi::{
    FOOTPRINT_WORKGROUP, MISS_PRIMITIVE, RAYTRACE_MODE_ANY, RAYTRACE_MODE_CLOSEST,
    RAYTRACE_WORKGROUP,
};
use super::bind_groups::{bvh_bind_group, footprint_bind_group, tlas_bind_group};
use super::pipeline::RayTraversalPipelines;
use super::resources::{
    decode_bvh_hits, decode_footprint_results, decode_tlas_hits, pack_footprints, pack_rays,
    BvhTraversalResources, FootprintRequest, FootprintResources, GpuFootprintResult, GpuRayHit,
    GpuTlasHit, TlasTraversalResources,
};

/// A reusable ray-traversal compute service bound to one render device.
///
/// Build it once with [`GpuRayTraversal::new`] and call the walk methods with
/// the render world's [`RenderDevice`] / [`RenderQueue`]; the compiled pipelines
/// are shared across every dispatch.
pub(crate) struct GpuRayTraversal {
    /// The three compiled kernels (`ray_traverse`, `tlas_traverse`,
    /// `ray_footprint`).
    pipelines: RayTraversalPipelines,
}

impl GpuRayTraversal {
    /// Compiles the three ray-traversal kernels on `device`.
    pub(crate) fn new(device: &RenderDevice) -> Self {
        Self {
            pipelines: RayTraversalPipelines::new(device),
        }
    }

    /// Returns the nearest single-`BLAS` hit for every ray, the device twin of
    /// `GpuBvhBuffers::closest_hit`.
    pub(crate) fn closest_hits(
        &self,
        device: &RenderDevice,
        queue: &RenderQueue,
        buffers: &GpuBvhBuffers,
        rays: &[Ray],
    ) -> Vec<GpuRayHit> {
        self.bvh_walk(device, queue, buffers, rays, RAYTRACE_MODE_CLOSEST)
    }

    /// Returns `true` for every ray occluded within its interval, the device
    /// twin of `GpuBvhBuffers::any_hit`.
    pub(crate) fn any_hits(
        &self,
        device: &RenderDevice,
        queue: &RenderQueue,
        buffers: &GpuBvhBuffers,
        rays: &[Ray],
    ) -> Vec<bool> {
        self.bvh_walk(device, queue, buffers, rays, RAYTRACE_MODE_ANY)
            .into_iter()
            .map(|hit| hit.primitive != MISS_PRIMITIVE)
            .collect()
    }

    /// Returns the nearest two-level hit for every ray, the device twin of
    /// `GpuTlasBuffers::closest_hit`.
    pub(crate) fn tlas_closest_hits(
        &self,
        device: &RenderDevice,
        queue: &RenderQueue,
        tlas: &GpuTlasBuffers,
        pool: &GpuBlasPool,
        rays: &[Ray],
    ) -> Vec<GpuTlasHit> {
        self.tlas_walk(device, queue, tlas, pool, rays, RAYTRACE_MODE_CLOSEST)
    }

    /// Returns `true` for every ray occluded within its interval across the
    /// `TLAS`, the device twin of `GpuTlasBuffers::any_hit`.
    pub(crate) fn tlas_any_hits(
        &self,
        device: &RenderDevice,
        queue: &RenderQueue,
        tlas: &GpuTlasBuffers,
        pool: &GpuBlasPool,
        rays: &[Ray],
    ) -> Vec<bool> {
        self.tlas_walk(device, queue, tlas, pool, rays, RAYTRACE_MODE_ANY)
            .into_iter()
            .map(|hit| hit.primitive != MISS_PRIMITIVE)
            .collect()
    }

    /// Returns the ray-cone footprint / texture-`LOD` result for every request,
    /// the device twin of `RayFootprint`, with the continuous mip clamped to
    /// `max_mip`.
    pub(crate) fn footprint_mips(
        &self,
        device: &RenderDevice,
        queue: &RenderQueue,
        requests: &[FootprintRequest],
        max_mip: u32,
    ) -> Vec<GpuFootprintResult> {
        let count = requests.len() as u32;
        if count == 0 {
            return Vec::new();
        }
        let words = pack_footprints(requests);
        let resources = FootprintResources::new(device, &words, count, max_mip);
        let bind_group = footprint_bind_group(device, &self.pipelines.footprint, &resources);
        let raw = run_pass(
            device,
            queue,
            &self.pipelines.footprint,
            &bind_group,
            count.div_ceil(FOOTPRINT_WORKGROUP),
            &resources.results,
            &resources.stage,
            resources.result_bytes,
            "prism_rt_footprint",
        );
        decode_footprint_results(&raw, count as usize)
    }

    /// Shared single-`BLAS` dispatch for a `mode` walk.
    fn bvh_walk(
        &self,
        device: &RenderDevice,
        queue: &RenderQueue,
        buffers: &GpuBvhBuffers,
        rays: &[Ray],
        mode: u32,
    ) -> Vec<GpuRayHit> {
        let ray_count = rays.len() as u32;
        if ray_count == 0 {
            return Vec::new();
        }
        let ray_words = pack_rays(rays);
        let resources = BvhTraversalResources::new(device, buffers, &ray_words, ray_count, mode);
        let bind_group = bvh_bind_group(device, &self.pipelines.bvh, &resources);
        let raw = run_pass(
            device,
            queue,
            &self.pipelines.bvh,
            &bind_group,
            ray_count.div_ceil(RAYTRACE_WORKGROUP),
            &resources.hits,
            &resources.stage,
            resources.hit_bytes,
            "prism_rt_bvh",
        );
        decode_bvh_hits(&raw, ray_count as usize)
    }

    /// Shared two-level dispatch for a `mode` walk.
    fn tlas_walk(
        &self,
        device: &RenderDevice,
        queue: &RenderQueue,
        tlas: &GpuTlasBuffers,
        pool: &GpuBlasPool,
        rays: &[Ray],
        mode: u32,
    ) -> Vec<GpuTlasHit> {
        let ray_count = rays.len() as u32;
        if ray_count == 0 {
            return Vec::new();
        }
        let ray_words = pack_rays(rays);
        let resources =
            TlasTraversalResources::new(device, tlas, pool, &ray_words, ray_count, mode);
        let bind_group = tlas_bind_group(device, &self.pipelines.tlas, &resources);
        let raw = run_pass(
            device,
            queue,
            &self.pipelines.tlas,
            &bind_group,
            ray_count.div_ceil(RAYTRACE_WORKGROUP),
            &resources.hits,
            &resources.stage,
            resources.hit_bytes,
            "prism_rt_tlas",
        );
        decode_tlas_hits(&raw, ray_count as usize)
    }
}

/// Records one compute pass, copies the output to its staging buffer, submits,
/// blocks on the device poll and returns the read-back `u32` words.
#[expect(
    clippy::too_many_arguments,
    reason = "the three walks share one dispatch body; grouping the pipeline, bind group, output and staging buffers into a struct would only move the arity without clarifying it"
)]
fn run_pass(
    device: &RenderDevice,
    queue: &RenderQueue,
    pipeline: &ComputePipeline,
    bind_group: &bevy_render::render_resource::BindGroup,
    workgroups: u32,
    output: &Buffer,
    stage: &Buffer,
    bytes: u64,
    label: &str,
) -> Vec<u32> {
    let mut encoder =
        device.create_command_encoder(&CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &**bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(output, 0, stage, 0, bytes);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted ray-traversal work");

    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let raw: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
    drop(view);
    stage.unmap();
    raw
}
