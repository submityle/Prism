//! `wgpu` compute twin of the virtual-geometry cluster culler
//! ([`cluster_cull`](prism_render_architecture::virtual_geometry::cluster_cull)).
//!
//! A GPU-driven virtual-geometry pipeline culls every candidate cluster before
//! it selects a LOD or resolves a resident page: a cluster that leaves the view
//! frustum, or that a nearer occluder fully hides, needs neither. The CPU
//! golden
//! [`cluster_cull`](prism_render_architecture::virtual_geometry::cluster_cull)
//! owns that decision; [`GpuClusterCuller`] is the on-device twin that runs one
//! thread per cluster and returns the same
//! [`CullVerdict`](prism_render_architecture::virtual_geometry::CullVerdict)
//! index the reference does.
//!
//! # Portability
//!
//! The kernel is a fixed six-plane loop of multiply-adds plus two compares, in
//! the portable core-WGSL subset, so unlike the 64-bit payload twin it needs no
//! optional device feature and runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The emitted verdict is a discrete decision — `Visible`, `FrustumCulled` or
//! `OcclusionCulled` — derived from sign comparisons, not a continuous value.
//! The kernel mirrors the reference's term order (`nx*px + ny*py + nz*pz + d`
//! for the signed distance and `|nx|*hx + |ny|*hy + |nz|*hz` for the projected
//! extent), so away from the razor-thin plane boundary the verdict is identical
//! to the reference regardless of fused-multiply-add contraction. The parity
//! test picks clusters that clear each boundary by a wide margin, so the
//! integer verdict is stable under any legal float reassociation and can be
//! asserted index-for-index rather than with a tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard frustum projected-radius / Hi-Z occlusion culling and
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_geometry::{Frustum, OcclusionProbe};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one cull dispatch. Layout matches `Params` in
/// `shaders/cluster_cull.wesl`: six `vec4` planes then the cluster count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    planes: [[f32; 4]; 6],
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One cluster's bounds plus optional occlusion probe. `48`-byte stride,
/// matching `Cluster` in the shader (`center`/`half_extents` are `vec3<f32>`,
/// 16-byte aligned, with a trailing scalar packed into each vec3's padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCluster {
    center: [f32; 3],
    has_occlusion: u32,
    half_extents: [f32; 3],
    closest_depth: f32,
    occluder_depth: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// A compiled, reusable cluster-cull pipeline.
pub struct GpuClusterCuller {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClusterCuller {
    /// Compiles the cull kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-WGSL subset, so unlike
    /// [`GpuPayloadRaster::new`](crate::GpuPayloadRaster::new) this never
    /// returns [`None`]: it compiles on any backend `ctx` acquired.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClusterCuller {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cluster_cull"),
            source: ShaderSource::Wgsl(include_str!("../shaders/cluster_cull.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cluster_cull_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cluster_cull_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_cluster_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cull"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClusterCuller {
            module,
            layout,
            pipeline,
        }
    }

    /// Culls each cluster in `clusters` against `frustum`, returning one verdict
    /// index per cluster in input order.
    ///
    /// Each input pairs a cluster's [`SceneBounds`] with an optional
    /// [`OcclusionProbe`] (`None` skips the occlusion phase for that cluster,
    /// e.g. the first depth-prepass wave with no Hi-Z yet). The returned `u32`
    /// equals
    /// [`cluster_cull`](prism_render_architecture::virtual_geometry::cluster_cull)`(..) as u32`
    /// for the same inputs: `0` = `Visible`, `1` = `FrustumCulled`,
    /// `2` = `OcclusionCulled`.
    #[must_use]
    pub fn cull(
        &self,
        ctx: &GpuContext,
        frustum: &Frustum,
        clusters: &[(SceneBounds, Option<OcclusionProbe>)],
    ) -> Vec<u32> {
        if clusters.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let mut planes = [[0.0f32; 4]; 6];
        for (dst, plane) in planes.iter_mut().zip(frustum.planes.iter()) {
            *dst = [
                plane.normal[0],
                plane.normal[1],
                plane.normal[2],
                plane.distance,
            ];
        }
        let params = Params {
            planes,
            count: clusters.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let gpu_clusters: Vec<GpuCluster> = clusters
            .iter()
            .map(|(bounds, probe)| {
                let (has_occlusion, closest_depth, occluder_depth) = match probe {
                    Some(p) => (1u32, p.closest_depth, p.occluder_depth),
                    None => (0u32, 0.0, 0.0),
                };
                GpuCluster {
                    center: bounds.center,
                    has_occlusion,
                    half_extents: bounds.half_extents,
                    closest_depth,
                    occluder_depth,
                    pad0: 0.0,
                    pad1: 0.0,
                    pad2: 0.0,
                }
            })
            .collect();

        let out_bytes = (clusters.len() as u64) * 4;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_cluster_cull_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let clusters_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_cluster_cull_clusters"),
            contents: bytemuck::cast_slice(&gpu_clusters),
            usage: BufferUsages::STORAGE,
        });
        let verdicts_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_cluster_cull_verdicts"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let verdicts_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_cluster_cull_verdicts_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_cluster_cull_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: clusters_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: verdicts_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cluster_cull_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cluster_cull_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (clusters.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&verdicts_buf, 0, &verdicts_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        verdicts_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = verdicts_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let verdicts = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        verdicts_stage.unmap();
        debug_assert_eq!(verdicts.len(), clusters.len());
        verdicts
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
