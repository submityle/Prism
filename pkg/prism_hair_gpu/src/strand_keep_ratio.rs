//! `wgpu` compute twin of Prism's continuous strand keep ratio
//! ([`strand_keep_ratio`](prism_render_architecture::hair::cluster::strand_keep_ratio)).
//!
//! As a hair cluster recedes its projected pixel footprint shrinks, and the
//! continuous decimation LOD thins the strand set to match: full strands at a
//! large footprint, a `min_ratio` floor at a tiny one, a linear ramp between so
//! the groom never pops (see [`crate::cluster_cull`] for the cull stage that
//! precedes it). This twin evaluates that ramp batch-wide on the device, one
//! thread per cluster footprint — the array-in/array-out form the LOD stage
//! consumes per cluster.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairStrandKeepRatio::eval`] takes a slice of cluster footprints (in
//! pixels) and a single [`DecimationThresholds`], and returns one keep ratio in
//! `[min_ratio, 1]` per footprint in order. The footprint index is the
//! invocation id (`@compute @workgroup_size(64)`, a one-dimensional dispatch
//! over `global_invocation_id.x`); invocations past the element count
//! early-return.
//!
//! # Correctness model
//!
//! The thresholds are a per-dispatch constant, so the host sanitises them once
//! with the golden's [`DecimationThresholds::sanitized`] (`full_px >= cull_px >=
//! 0`, finite, `min_ratio` in `[0, 1]`) and uploads the finished bounds as
//! uniforms; the per-element work sanitises the footprint (non-finite or
//! negative collapses to `0`) and evaluates the ramp exactly as the reference.
//! The hard saturation returns are exact; only the in-band `(px - cull) / span`
//! ramp divides, which a `GPU` may round a few `ULP` differently, so the parity
//! test matches the ratio within the fma tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`).
//!
//! # Portability
//!
//! The kernel uses only compares, `max`, a divide and multiply/add in the
//! portable core-`WGSL` subset — no `exp`, `pow` or optional device feature — so
//! the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: continuous screen-footprint strand decimation ramp plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::cluster::{strand_keep_ratio, DecimationThresholds};

use crate::context::GpuContext;

/// Uniform parameters for one keep-ratio dispatch. Layout matches `Params` in
/// `shaders/strand_keep_ratio.wesl`: the element count plus the sanitised
/// decimation bounds, one `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    elem_count: u32,
    full_px: f32,
    cull_px: f32,
    min_ratio: f32,
}

/// A compiled, reusable per-footprint strand keep-ratio pipeline.
pub struct GpuHairStrandKeepRatio {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairStrandKeepRatio {
    /// Compiles the per-footprint strand keep-ratio kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairStrandKeepRatio {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_strand_keep_ratio"),
            source: ShaderSource::Wgsl(include_str!("../shaders/strand_keep_ratio.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_strand_keep_ratio_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_strand_keep_ratio_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_strand_keep_ratio_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairStrandKeepRatio {
            module,
            layout,
            pipeline,
        }
    }

    /// Maps each cluster footprint to its continuous strand keep ratio under
    /// `thresholds`, returning one `f32` per input in order.
    ///
    /// The ratio for footprint `i` matches the `CPU` golden
    /// [`strand_keep_ratio`](prism_render_architecture::hair::cluster::strand_keep_ratio)
    /// within the fma tolerance, with the thresholds sanitised host-side exactly
    /// as the reference does internally. An empty batch yields an empty vector
    /// without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        footprints: &[f32],
        thresholds: DecimationThresholds,
    ) -> Vec<f32> {
        let elem_count = footprints.len();
        if elem_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        // Sanitise the thresholds once, exactly as the golden does internally, so
        // the uniform the shader reads matches the reference's working bounds.
        let t = thresholds.sanitized();
        let uniforms = Params {
            elem_count: elem_count as u32,
            full_px: t.full_px,
            cull_px: t.cull_px,
            min_ratio: t.min_ratio,
        };

        let out_bytes = (elem_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_strand_keep_ratio_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let footprints_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_strand_keep_ratio_footprints"),
            contents: bytemuck::cast_slice(footprints),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_strand_keep_ratio_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_strand_keep_ratio_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_strand_keep_ratio_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: footprints_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_strand_keep_ratio_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_strand_keep_ratio_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (elem_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        out
    }
}

/// The `CPU` golden strand keep ratio for one footprint, re-exported so the
/// parity test can assert the device twin against the identical reference it
/// mirrors.
#[must_use]
pub fn reference_strand_keep_ratio(cluster_px: f32, thresholds: DecimationThresholds) -> f32 {
    strand_keep_ratio(cluster_px, thresholds)
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
