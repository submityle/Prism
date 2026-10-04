//! `wgpu` compute twin of the dependency-free `CPU` golden coastline
//! heightfield downsample
//! ([`downsample_heightfield`](prism_render_architecture::water::coastline::downsample_heightfield)).
//!
//! `UE5` Water decimates a large Landscape heightfield to the water solver grid
//! it can afford before flooding the coast. The golden coarsens a row-major
//! `nx x nz` terrain tile by an integer `stride`: each output cell is the
//! unbiased mean of the up-to `stride x stride` source samples it covers, with
//! partial edge blocks averaging only the samples that exist. This twin runs
//! that reduction on device so a terrain-streaming pass can coarsen tiles in
//! the same place the rest of the water subsystem already lives on the `GPU`.
//!
//! # Correctness model
//!
//! The reduction is embarrassingly parallel: one invocation owns one output
//! cell, with no barrier and no workgroup memory. Each thread walks its source
//! block in the *same* row-major order the golden uses (`for row { for col }`),
//! so the floating-point summation order — and therefore the rounding — matches
//! cell for cell. `WGSL` `u32` arithmetic wraps exactly like the golden's index
//! math, so the only residual is the last-place slack of a `GPU` fused
//! multiply-add in the running sum. The parity test asserts each coarsened
//! sample within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! A `stride` of one coarsens nothing: every output cell is a one-sample block,
//! so the kernel reproduces the input exactly (the golden short-circuits this,
//! but the device path yields the identical field). A degenerate request (zero
//! `stride`/`nx`/`nz` or a `heights` length that does not match `nx * nz`)
//! returns an empty field, matching the golden's [`None`].
//!
//! # Portability
//!
//! The kernel is integer index math plus add/divide in the portable
//! core-`WGSL` subset — no transcendental, no optional device feature — so it
//! runs unmodified on Metal, Vulkan and DX12.
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

/// Workgroup width of the downsample; must match `@workgroup_size(256)` in
/// `shaders/water_coastline_downsample.wesl`.
const WORKGROUP_SIZE: u32 = 256;

/// One coarsened terrain heightfield, row-major over the `out_nx x out_nz`
/// grid.
///
/// Mirrors the tuple the `CPU`
/// [`downsample_heightfield`](prism_render_architecture::water::coastline::downsample_heightfield)
/// returns. A degenerate request yields `out_nx == 0`, `out_nz == 0` and an
/// empty buffer — the twin's stand-in for the golden's [`None`].
#[derive(Clone, Debug, PartialEq)]
pub struct WaterCoastlineDownsample {
    /// Output grid width, `ceil(nx / stride)` (`0` for a degenerate request).
    pub out_nx: u32,
    /// Output grid height, `ceil(nz / stride)` (`0` for a degenerate request).
    pub out_nz: u32,
    /// Row-major `out_nx * out_nz` coarsened heights (meters).
    pub heights: Vec<f32>,
}

impl WaterCoastlineDownsample {
    /// The degenerate empty field a rejected request returns.
    #[must_use]
    fn empty() -> WaterCoastlineDownsample {
        WaterCoastlineDownsample {
            out_nx: 0,
            out_nz: 0,
            heights: Vec::new(),
        }
    }
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/water_coastline_downsample.wesl` (`32` bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    nx: u32,
    nz: u32,
    stride: u32,
    out_w: u32,
    out_h: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable coastline-downsample pipeline.
pub struct GpuWaterCoastlineDownsample {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterCoastlineDownsample {
    /// Compiles the downsample kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterCoastlineDownsample {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_coastline_downsample"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/water_coastline_downsample.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_coastline_downsample_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_coastline_downsample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_coastline_downsample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("downsample_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterCoastlineDownsample {
            module,
            layout,
            pipeline,
        }
    }

    /// Coarsens a row-major `nx x nz` terrain heightfield by `stride` on device.
    ///
    /// Returns [`WaterCoastlineDownsample::empty`] for a degenerate request
    /// (zero `stride`/`nx`/`nz` or a `heights` length that does not match
    /// `nx * nz`), matching the golden's [`None`].
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        heights: &[f32],
        nx: u32,
        nz: u32,
        stride: u32,
    ) -> WaterCoastlineDownsample {
        if stride == 0 || nx == 0 || nz == 0 {
            return WaterCoastlineDownsample::empty();
        }
        let Some(cells) = (nx as usize).checked_mul(nz as usize) else {
            return WaterCoastlineDownsample::empty();
        };
        if heights.len() != cells {
            return WaterCoastlineDownsample::empty();
        }

        let out_w = nx.div_ceil(stride);
        let out_h = nz.div_ceil(stride);
        let out_cells = (out_w as usize) * (out_h as usize);
        let device = ctx.device();

        let params = Params {
            nx,
            nz,
            stride,
            out_w,
            out_h,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_coastline_downsample_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let heights_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_coastline_downsample_heights"),
            contents: bytemuck::cast_slice(heights),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (out_cells * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_coastline_downsample_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_coastline_downsample_bind_group"),
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
            label: Some("prism_volumetric_water_coastline_downsample_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_coastline_downsample_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_coastline_downsample_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per output cell; round the last workgroup up (the
            // shader guards `idx >= out_w*out_h`).
            let groups = (out_cells as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage_buf, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage_buf.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let heights_out = read_f32(&stage_buf);

        WaterCoastlineDownsample {
            out_nx: out_w,
            out_nz: out_h,
            heights: heights_out,
        }
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
