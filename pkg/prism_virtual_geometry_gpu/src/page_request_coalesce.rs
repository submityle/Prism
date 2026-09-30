//! `wgpu` compute twin of the virtual-geometry per-frame page-request coalescer.
//!
//! When a GPU-driven virtual-geometry frame selects its cut, every drawn
//! cluster names the page that backs it and reports a screen-coverage priority.
//! Dozens of clusters routinely share one page, so the CPU golden
//! [`plan_frame`](prism_render_architecture::virtual_geometry::plan_frame)
//! coalesces those references into one request per unique page at the highest
//! priority any referencing cluster reported (via
//! [`PageRequestBatch::record`](prism_render_architecture::virtual_geometry::PageRequestBatch),
//! keyed on the page's [`coverage_priority`]). [`GpuPageRequestCoalescer`] is the
//! on-device twin of that reduction: one thread per drawn cluster evaluates the
//! coverage priority and folds it into its page's dense slot with a single
//! `atomicMax`, so the per-slot maxima it returns match the reference
//! per-page priorities.
//!
//! # Host / device split
//!
//! The mutable residency bookkeeping stays on the host, exactly as the CPU
//! golden keeps `table.request` outside the pure per-frame decision: the host
//! assigns each unique [`GeometryPageKey`] a dense slot, dispatches the max
//! reduction, and reconstructs a [`PageRequestBatch`] from the per-slot maxima.
//! The device kernel is a pure max-reduction over non-negative priorities.
//!
//! # Correctness model
//!
//! [`coverage_priority`] is non-negative (`extent^2 >= 0`, `distance_sq >=
//! f32::EPSILON > 0`), and for non-negative `f32` the IEEE-754 bit pattern is
//! monotonic in value, so `atomicMax` on `bitcast<u32>(priority)` selects the
//! true maximum. The kernel mirrors the reference term order
//! (`max(radius, 0) * focal`, squared, over `max(distance_sq, EPSILON)`), so
//! away from a tie within a single ULP the coalesced priority is identical to
//! the reference regardless of fused-multiply-add contraction.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset (a storage buffer of
//! `atomic<u32>` plus `atomicMax`), so it needs no optional device feature and
//! runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard screen-coverage page-priority coalescing plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::virtual_geometry::{
    GeometryPageKey, LodProjection, PageRequestBatch,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one coalesce dispatch. Layout matches `Params` in
/// `shaders/page_request_coalesce.wesl`: the view origin (three `f32`) and focal
/// length, then the cluster and slot counts and two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    focal_length_pixels: f32,
    cluster_count: u32,
    slot_count: u32,
    pad0: u32,
    pad1: u32,
}

/// One drawn cluster's page-request input. `32`-byte stride, matching `Cluster`
/// in the shader: bounds center (three `f32`) and radius, then the dense page
/// slot this cluster references and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCluster {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    radius: f32,
    slot: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One drawn cluster's reference to the page that backs it.
///
/// * `page` — the page key this cluster streams from.
/// * `center` — world-space bounds center, used for the coverage distance.
/// * `radius` — world-space bounds radius, used for the projected extent.
#[derive(Clone, Copy, Debug)]
pub struct PageReference {
    /// The page key this cluster references.
    pub page: GeometryPageKey,
    /// World-space bounds center of the cluster.
    pub center: [f32; 3],
    /// World-space bounds radius of the cluster.
    pub radius: f32,
}

/// A compiled, reusable page-request coalescing pipeline.
pub struct GpuPageRequestCoalescer {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPageRequestCoalescer {
    /// Compiles the coalesce kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so this never
    /// returns [`None`]: it compiles on any backend `ctx` acquired.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPageRequestCoalescer {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_page_request_coalesce"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/page_request_coalesce.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_page_request_coalesce_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_page_request_coalesce_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_page_request_coalesce_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("coalesce"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPageRequestCoalescer {
            module,
            layout,
            pipeline,
        }
    }

    /// Coalesces `references` into one request per unique page, keeping the
    /// highest coverage priority any referencing cluster reported.
    ///
    /// The returned [`PageRequestBatch`] equals the `page_requests` field the
    /// CPU golden
    /// [`plan_frame`](prism_render_architecture::virtual_geometry::plan_frame)
    /// records for the same drawn clusters, view origin and projection. An empty
    /// `references` slice yields an empty batch (storage buffers cannot be
    /// zero-sized, so it is handled by an early return).
    #[must_use]
    pub fn coalesce(
        &self,
        ctx: &GpuContext,
        view_origin: [f32; 3],
        projection: LodProjection,
        references: &[PageReference],
    ) -> PageRequestBatch {
        let mut batch = PageRequestBatch::new();
        if references.is_empty() {
            return batch;
        }
        let device = ctx.device();

        // Assign each unique page a dense slot. The slot ordering is irrelevant
        // to the per-page maxima; only the mapping must be injective.
        let mut slot_of: HashMap<GeometryPageKey, u32> = HashMap::new();
        let mut slot_key: Vec<GeometryPageKey> = Vec::new();
        for reference in references {
            if !slot_of.contains_key(&reference.page) {
                let slot = u32::try_from(slot_key.len())
                    .expect("page-request slot count exceeds u32 range");
                slot_of.insert(reference.page, slot);
                slot_key.push(reference.page);
            }
        }
        let slot_count = slot_key.len();

        let params = Params {
            origin_x: view_origin[0],
            origin_y: view_origin[1],
            origin_z: view_origin[2],
            focal_length_pixels: projection.focal_length_pixels,
            cluster_count: u32::try_from(references.len())
                .expect("drawn cluster count exceeds u32 range"),
            slot_count: u32::try_from(slot_count).expect("slot count exceeds u32 range"),
            pad0: 0,
            pad1: 0,
        };

        let gpu_clusters: Vec<GpuCluster> = references
            .iter()
            .map(|reference| GpuCluster {
                center_x: reference.center[0],
                center_y: reference.center[1],
                center_z: reference.center[2],
                radius: reference.radius,
                slot: slot_of[&reference.page],
                pad0: 0,
                pad1: 0,
                pad2: 0,
            })
            .collect();

        // Max-reduction identity for non-negative priorities is +0.0 == 0 bits.
        let zeroed = vec![0u32; slot_count];
        let out_bytes = (slot_count as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_page_request_coalesce_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let clusters_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_page_request_coalesce_clusters"),
            contents: bytemuck::cast_slice(&gpu_clusters),
            usage: BufferUsages::STORAGE,
        });
        let priorities_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_page_request_coalesce_priorities"),
            contents: bytemuck::cast_slice(&zeroed),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let priorities_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_page_request_coalesce_priorities_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_page_request_coalesce_bind_group"),
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
                    resource: priorities_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_page_request_coalesce_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_page_request_coalesce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = u32::try_from(references.len())
                .expect("drawn cluster count exceeds u32 range")
                .div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&priorities_buf, 0, &priorities_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        priorities_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = priorities_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_priorities = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        priorities_stage.unmap();
        debug_assert_eq!(gpu_priorities.len(), slot_count);

        for (slot, &bits) in gpu_priorities.iter().enumerate() {
            batch.record(slot_key[slot], f32::from_bits(bits));
        }
        batch
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
