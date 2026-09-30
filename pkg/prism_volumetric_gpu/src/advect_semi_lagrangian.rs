//! `wgpu` compute twin of the semi-Lagrangian weather-map advection
//! ([`advect_semi_lagrangian`](prism_render_architecture::volumetric::weather::advect_semi_lagrangian)).
//!
//! The weather map is a row-major grid of 4-channel samples (`coverage`,
//! `cloud_type`, `precipitation`, `wind_disturbance`) that drives global cloud
//! distribution. Advancing it one step along a uniform wind is done by a
//! semi-Lagrangian backtrace: each output cell centre `(x, y)` is traced
//! backward to `(x - wind.x * dt, y - wind.y * dt)` in cell units and the
//! previous field is bilinearly sampled there with clamp-to-edge. The scheme is
//! unconditionally stable and convex, so every output cell is independent and
//! the whole grid advects in parallel; [`GpuAdvectSemiLagrangian`] uploads the
//! field once and traces one invocation per cell.
//!
//! # Correctness model
//!
//! The kernel mirrors the `CPU` step exactly: it reproduces
//! [`WeatherField::sample_bilinear`](prism_render_architecture::volumetric::weather::WeatherField::sample_bilinear)
//! including the coordinate clamp, the truncating floor, the edge-clamped
//! neighbour indices, and the per-channel convex `lerp`. Each channel is one
//! multiply-add per tap, so `CPU` and `GPU` agree to a few `ULP`. The parity
//! test advects real fields under several winds and steps (including winds that
//! backtrace off the grid so the clamp-to-edge path fires) and compares every
//! cell against the golden, so a wrong neighbour or a dropped clamp could not
//! pass.
//!
//! # Portability
//!
//! The kernel is `clamp`, truncation and multiply-add in the portable
//! core-`WGSL` subset, so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard semi-Lagrangian backtrace with bilinear resampling plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

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

/// One weather-map cell: the four `RGBA` channels that drive cloud
/// distribution. Mirrors `WeatherSample` in the `CPU` golden.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeatherAdvectSample {
    /// Coverage `0..=1` (`R` channel).
    pub coverage: f32,
    /// Cloud type `0..=1` (`G` channel): stratus (0) to cumulus (1).
    pub cloud_type: f32,
    /// Precipitation intensity `0..=1` (`B` channel).
    pub precipitation: f32,
    /// Wind disturbance `0..=1` (`A` channel).
    pub wind_disturbance: f32,
}

/// One cell as uploaded. `16`-byte `repr(C)` matching `Sample` in
/// `shaders/advect_semi_lagrangian.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSample {
    coverage: f32,
    cloud_type: f32,
    precipitation: f32,
    wind_disturbance: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/advect_semi_lagrangian.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    wind_x: f32,
    wind_y: f32,
    dt: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable semi-Lagrangian advection pipeline.
pub struct GpuAdvectSemiLagrangian {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAdvectSemiLagrangian {
    /// Compiles the advection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAdvectSemiLagrangian {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_advect_semi_lagrangian"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/advect_semi_lagrangian.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_advect_semi_lagrangian_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_advect_semi_lagrangian_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_advect_semi_lagrangian_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("advect_semi_lagrangian_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAdvectSemiLagrangian {
            module,
            layout,
            pipeline,
        }
    }

    /// Advects the `width` x `height` row-major weather field `cells` one step
    /// along the uniform `wind` (in cells per unit time) over `dt`, returning
    /// the advected field in the same row-major order.
    ///
    /// Each output cell equals the `CPU`
    /// [`advect_semi_lagrangian`](prism_render_architecture::volumetric::weather::advect_semi_lagrangian)
    /// result for the same field to within the tolerance documented on this
    /// module. `cells` must have exactly `width * height` entries in row-major
    /// order. An empty field (`cells` empty, or `width`/`height` zero) yields an
    /// empty result, matching the golden's empty-field fast path (storage
    /// buffers cannot be zero-sized).
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        cells: &[WeatherAdvectSample],
        width: u32,
        height: u32,
        wind: (f32, f32),
        dt: f32,
    ) -> Vec<WeatherAdvectSample> {
        let total = (width as usize) * (height as usize);
        if total == 0 || cells.is_empty() {
            return Vec::new();
        }
        assert_eq!(cells.len(), total, "cells length must equal width * height");
        let device = ctx.device();

        let gpu_cells: Vec<GpuSample> = cells
            .iter()
            .map(|c| GpuSample {
                coverage: c.coverage,
                cloud_type: c.cloud_type,
                precipitation: c.precipitation,
                wind_disturbance: c.wind_disturbance,
            })
            .collect();

        let gpu_params = Params {
            width,
            height,
            wind_x: wind.0,
            wind_y: wind.1,
            dt,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (total as u64) * (size_of::<GpuSample>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_advect_semi_lagrangian_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_advect_semi_lagrangian_input"),
            contents: bytemuck::cast_slice(&gpu_cells),
            usage: BufferUsages::STORAGE,
        });
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_advect_semi_lagrangian_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_advect_semi_lagrangian_output_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_advect_semi_lagrangian_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_advect_semi_lagrangian_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_advect_semi_lagrangian_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (total as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &output_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        output_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = output_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuSample>(&view).to_vec();
        drop(view);
        output_stage.unmap();
        debug_assert_eq!(raw.len(), total);
        raw.into_iter()
            .map(|s| WeatherAdvectSample {
                coverage: s.coverage,
                cloud_type: s.cloud_type,
                precipitation: s.precipitation,
                wind_disturbance: s.wind_disturbance,
            })
            .collect()
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
