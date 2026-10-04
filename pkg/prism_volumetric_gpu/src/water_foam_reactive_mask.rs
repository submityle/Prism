//! `wgpu` compute twin of the dependency-free `CPU` golden foam reactive mask
//! ([`reactive_mask_into`](prism_render_architecture::water::foam::reactive_mask_into)).
//!
//! The foam reactive mask is a per-cell visibility predicate: `true` wherever
//! foam coverage meets a shading threshold. The renderer uses it to gate the
//! foam shading pass to the cells that actually carry foam, so the expensive
//! pass skips the (usually large) calm majority of the surface. This twin
//! evaluates that predicate for a whole grid in one dispatch, keeping the mask
//! on device next to the foam field the companion
//! [`water_foam_step`](crate::water_foam_step) twin already evolves there.
//!
//! # Correctness model
//!
//! The predicate is embarrassingly parallel: one invocation owns one cell and
//! reads only that cell's own coverage, so there is no write hazard, no barrier
//! and no workgroup memory. Each cell performs the identical single
//! floating-point comparison `coverage >= threshold` the golden performs — no
//! arithmetic at all — so the readback is exactly bit-identical to the golden,
//! with no last-place slack to tolerate. The device stores each boolean as a
//! `u32` (`1` set, `0` clear); the host unpacks it back to `bool`. A degenerate
//! request (zero cells) returns an empty mask, the twin's device-buffer
//! contract.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset (a comparison, `select`
//! and index math) — no transcendental intrinsic, no optional device feature —
//! so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::foam`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Workgroup width of the mask pass; must match `@workgroup_size(256)` in
/// `shaders/water_foam_reactive_mask.wesl`.
const WORKGROUP_SIZE: u32 = 256;

/// One evaluated foam reactive mask.
///
/// Mirrors the boolean buffer the `CPU`
/// [`reactive_mask_into`](prism_render_architecture::water::foam::reactive_mask_into)
/// fills, one flag per cell in row-major order: `true` where foam coverage
/// meets the threshold. A degenerate request yields an empty mask.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaterFoamReactiveMask {
    /// Per-cell visibility flag: `true` where coverage `>= threshold`.
    pub mask: Vec<bool>,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/water_foam_reactive_mask.wesl` (`16` bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    threshold: f32,
    _pad0: u32,
    _pad1: u32,
}

/// A compiled, reusable foam-reactive-mask pipeline.
pub struct GpuWaterFoamReactiveMask {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterFoamReactiveMask {
    /// Compiles the mask pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_foam_reactive_mask"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/water_foam_reactive_mask.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_foam_reactive_mask_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_foam_reactive_mask_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_foam_reactive_mask_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("mask_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterFoamReactiveMask {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the reactive mask for a grid of foam coverage on device.
    ///
    /// `field` is the row-major foam coverage; the mask is `true` wherever a
    /// cell's coverage is at least `threshold`. Returns an empty mask for a
    /// degenerate request (zero cells), the twin's device-buffer contract.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        field: &[f32],
        threshold: f32,
    ) -> WaterFoamReactiveMask {
        let count = field.len();
        if count == 0 {
            return WaterFoamReactiveMask { mask: Vec::new() };
        }
        let device = ctx.device();

        let params = Params {
            count: count as u32,
            threshold,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_foam_reactive_mask_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let field_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_foam_reactive_mask_field"),
            contents: bytemuck::cast_slice(field),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<u32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_foam_reactive_mask_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_foam_reactive_mask_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: field_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_foam_reactive_mask_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_foam_reactive_mask_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_foam_reactive_mask_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per cell; round the last workgroup up (the shader
            // guards `idx >= count`).
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let raw = read_u32(&stage);
        let mask = raw.into_iter().map(|v| v != 0).collect();
        WaterFoamReactiveMask { mask }
    }
}

/// Reads back a mapped `u32` staging buffer into an owned vector, unmapping it.
fn read_u32(stage: &wgpu::Buffer) -> Vec<u32> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let out = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
    drop(view);
    stage.unmap();
    out
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
