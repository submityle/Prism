//! `wgpu` compute twin of the dependency-free `CPU` golden still-flood-depth
//! map ([`flood_depth`](prism_render_architecture::water::coastline::flood_depth)
//! applied per cell).
//!
//! `UE5` Water floods every part of a terrain that sits below the still sea
//! level. The golden's per-cell flood depth is `max(0, sea_level - height)`:
//! positive where the terrain dips under the sea level, and exactly zero on dry
//! land so a coast never reports negative water. This twin evaluates that map on
//! device so a terrain-streaming pass can derive the flood field in the same
//! place the rest of the water subsystem already lives on the `GPU`, feeding the
//! shoreline / wetness passes without a round trip to the host.
//!
//! # Correctness model
//!
//! The map is embarrassingly parallel: one invocation owns one cell, with no
//! barrier and no workgroup memory. Each cell is a single subtraction followed
//! by a clamp to zero — the identical floating-point operation
//! [`flood_depth`](prism_render_architecture::water::coastline::flood_depth)
//! performs — so the readback is bit-identical to the golden up to the
//! last-place slack of a `GPU` fused multiply-add. A degenerate request (an
//! empty terrain) returns an empty field, matching an empty golden map.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset (a subtract plus
//! `max`) — no transcendental, no optional device feature — so it runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::coastline`；无第三方引擎源码或衍生代码。
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

/// Workgroup width of the flood-depth map; must match `@workgroup_size(256)` in
/// `shaders/water_coastline_flood_depth.wesl`.
const WORKGROUP_SIZE: u32 = 256;

/// A per-cell still-flood-depth map in meters, one entry per terrain sample.
///
/// Mirrors the vector a host loop over the `CPU`
/// [`flood_depth`](prism_render_architecture::water::coastline::flood_depth)
/// produces. A degenerate request (empty terrain) yields an empty buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterCoastlineFloodDepth {
    /// Per-cell still flood depth in meters, in the terrain's input order;
    /// `0` on dry land.
    pub depth: Vec<f32>,
}

impl WaterCoastlineFloodDepth {
    /// The degenerate empty field a rejected request returns.
    #[must_use]
    fn empty() -> WaterCoastlineFloodDepth {
        WaterCoastlineFloodDepth { depth: Vec::new() }
    }
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/water_coastline_flood_depth.wesl` (`16` bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    sea_level: f32,
    pad3: f32,
    pad4: f32,
    pad5: f32,
}

/// A compiled, reusable coastline flood-depth pipeline.
pub struct GpuWaterCoastlineFloodDepth {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterCoastlineFloodDepth {
    /// Compiles the flood-depth kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterCoastlineFloodDepth {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_coastline_flood_depth"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/water_coastline_flood_depth.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_coastline_flood_depth_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_coastline_flood_depth_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_coastline_flood_depth_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("flood_depth_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterCoastlineFloodDepth {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the still flood depth of each terrain sample against
    /// `sea_level` on device.
    ///
    /// Returns [`WaterCoastlineFloodDepth::empty`] for a degenerate request
    /// (an empty terrain), matching an empty golden map.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        heights: &[f32],
        sea_level: f32,
    ) -> WaterCoastlineFloodDepth {
        if heights.is_empty() {
            return WaterCoastlineFloodDepth::empty();
        }
        let count = heights.len();
        let device = ctx.device();

        let params = Params {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            sea_level,
            pad3: 0.0,
            pad4: 0.0,
            pad5: 0.0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_coastline_flood_depth_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let heights_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_coastline_flood_depth_heights"),
            contents: bytemuck::cast_slice(heights),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = size_of_val(heights) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_coastline_flood_depth_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_coastline_flood_depth_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: heights_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let stage_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_coastline_flood_depth_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_coastline_flood_depth_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_coastline_flood_depth_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per cell; round the last workgroup up (the shader
            // guards `idx >= count`).
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage_buf, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage_buf.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let depth = read_f32(&stage_buf);

        WaterCoastlineFloodDepth { depth }
    }
}

/// Reads back a mapped `f32` staging buffer into an owned vector, unmapping it.
fn read_f32(stage: &wgpu::Buffer) -> Vec<f32> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
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
