//! Real-device `wgpu` compute twin of the triangle-mesh ray cast: the device
//! form of [`cpu_trimesh_raycast`](super::cpu_trimesh_raycast).
//!
//! [`GpuTrimeshRayCast`] compiles `shaders/collider_trimesh_raycast.wgsl` once
//! and exposes [`GpuTrimeshRayCast::raycast`], which casts one ray at a static
//! [`Trimesh`] and returns the nearest triangle hit. One device lane handles one
//! triangle: it runs the shared Moller-Trumbore solve and writes that triangle's
//! `(t, u, v)` and a hit flag. The host then reduces the per-triangle rows to the
//! nearest hit and finalises point, barycentric weights, and the origin-facing
//! normal through the shared
//! [`finalize_hit`](super::trimesh_raycast::finalize_hit) rule, so the result is
//! the identical [`TrimeshRayHit`] the `CPU` brute and `BVH` queries return.
//!
//! # Honest brute reduction
//!
//! A nearest-hit query has no device-side priority queue, so this kernel
//! evaluates every triangle (one lane each) and the host keeps the least-`t` hit.
//! That is the `GPU` analogue of the `CPU` brute golden, not the `BVH` branch and
//! bound; the `BVH` prune lives on the `CPU` query and the device result is pinned
//! to it by the parity suite.
//!
//! # Buffer layout
//!
//! The mesh is flattened to three vertices per triangle so no shared index pool
//! is needed: `vertices` holds `tri0.a, tri0.b, tri0.c, tri1.a, ...` as
//! `vec4<f32>` rows (`xyz` used, `w` padding) and `indices` row `i` is
//! `(3i, 3i + 1, 3i + 2, 0)` as a `vec4<u32>`. [`Params`] carries the ray and the
//! triangle count. Results read back as one `vec4<f32>` per triangle:
//! `(t, u, v, hit_flag)` with `hit_flag` `1.0` on a hit.
//!
//! Provenance: Moller-Trumbore ray/triangle intersection (1997). No Unreal
//! Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::bvh::layout::{buffer_entry, entry};

use super::trimesh_raycast::{closer_hit, finalize_hit, MeshRay, TrimeshRayHit};
use super::Trimesh;

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// `hit_flag` value the kernel writes on a hit.
const HIT_FLAG: f32 = 1.0;

/// Uniform parameters shared with `Params` in
/// `shaders/collider_trimesh_raycast.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// `xyz`: ray origin; `w`: max travel distance.
    origin_maxdist: [f32; 4],
    /// `xyz`: ray direction; `w`: unused.
    dir: [f32; 4],
    /// `x`: triangle count; `y`, `z`, `w`: padding.
    counts: [u32; 4],
}

/// One per-triangle result row, mirroring the `WGSL` output `vec4<f32>`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RayOut {
    /// `(t, u, v, hit_flag)`.
    data: [f32; 4],
}

/// A reusable triangle-mesh ray-cast query that runs the per-triangle
/// intersection on the `GPU`.
///
/// Holds the compiled pipeline and bind-group layout so repeated casts reuse
/// them; build it once per [`GpuContext`] and call
/// [`raycast`](GpuTrimeshRayCast::raycast) per query.
pub struct GpuTrimeshRayCast {
    /// Retained shader module (kept alive for the pipeline).
    _module: ShaderModule,
    /// Bind-group layout for the four bindings.
    layout: BindGroupLayout,
    /// Compiled compute pipeline.
    pipeline: ComputePipeline,
}

impl GpuTrimeshRayCast {
    /// Builds the pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTrimeshRayCast {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_collider_trimesh_raycast"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/collider_trimesh_raycast.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_collider_trimesh_raycast_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_collider_trimesh_raycast_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_collider_trimesh_raycast_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("collider_trimesh_raycast"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTrimeshRayCast {
            _module: module,
            layout,
            pipeline,
        }
    }

    /// Casts `ray` at `mesh` on the `GPU`, returning the nearest triangle hit.
    ///
    /// Returns [`None`] when the ray misses every triangle or the mesh is empty.
    /// The result matches [`cpu_trimesh_raycast`](super::cpu_trimesh_raycast)
    /// including the lowest-index rule on a distance tie.
    #[must_use]
    pub fn raycast(&self, ctx: &GpuContext, mesh: &Trimesh, ray: &MeshRay) -> Option<TrimeshRayHit> {
        let n = mesh.triangle_count();
        if n == 0 {
            return None;
        }
        let device = ctx.device();

        // Flatten to three vertices per triangle; indices address that run.
        let mut vertices: Vec<[f32; 4]> = Vec::with_capacity(n * 3);
        let mut indices: Vec<[u32; 4]> = Vec::with_capacity(n);
        for i in 0..n {
            let tri = mesh.triangle(i);
            let base = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
            vertices.push([tri.a.x, tri.a.y, tri.a.z, 0.0]);
            vertices.push([tri.b.x, tri.b.y, tri.b.z, 0.0]);
            vertices.push([tri.c.x, tri.c.y, tri.c.z, 0.0]);
            indices.push([base, base + 1, base + 2, 0]);
        }

        let params = Params {
            origin_maxdist: [ray.origin.x, ray.origin.y, ray.origin.z, ray.max_distance],
            dir: [ray.direction.x, ray.direction.y, ray.direction.z, 0.0],
            counts: [u32::try_from(n).unwrap_or(u32::MAX), 0, 0, 0],
        };
        let params_buf = buffer::uniform(device, "prism_collider_trimesh_raycast_params", &params);
        let vertices_buf =
            buffer::storage_read(device, "prism_collider_trimesh_raycast_vertices", &vertices);
        let indices_buf =
            buffer::storage_read(device, "prism_collider_trimesh_raycast_indices", &indices);

        let out_bytes = (size_of::<RayOut>() * n) as u64;
        let out_buf =
            buffer::storage_rw_zeroed(device, "prism_collider_trimesh_raycast_out", out_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_collider_trimesh_raycast_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &vertices_buf),
                entry(2, &indices_buf),
                entry(3, &out_buf),
            ],
        });

        let out_stage =
            buffer::staging(device, "prism_collider_trimesh_raycast_stage", out_bytes);
        let groups = u32::try_from(n.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_collider_trimesh_raycast_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_collider_trimesh_raycast_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        buffer::copy(&mut encoder, &out_buf, &out_stage, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        let raw = buffer::read_back::<RayOut>(ctx, &out_stage);

        // Reduce to the nearest hit, finalising through the shared host rule so
        // the device hit is byte-compatible with the CPU golden.
        let mut best: Option<TrimeshRayHit> = None;
        for (i, slot) in raw.iter().enumerate().take(n) {
            if slot.data[3] != HIT_FLAG {
                continue;
            }
            let tri = mesh.triangle(i);
            let index = u32::try_from(i).unwrap_or(u32::MAX);
            let hit = finalize_hit(
                index,
                ray.direction,
                ray.origin,
                tri.a,
                tri.b,
                tri.c,
                slot.data[0],
                slot.data[1],
                slot.data[2],
            );
            if closer_hit(&hit, &best) {
                best = Some(hit);
            }
        }
        best
    }
}
