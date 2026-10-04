//! `wgpu` compute twin of the dependency-free `CPU` golden per-surface moisture
//! step
//! ([`step_moisture`](prism_render_architecture::water::wetness::step_moisture)).
//!
//! The companion
//! [`water_wetness_response`](crate::water_wetness_response) twin ports the
//! stateless wetness *response* kernels (albedo darkening, capillary height,
//! puddle depth, puddle predicate) but deliberately leaves the time-stepping
//! envelopes on the host. This twin ports exactly those envelopes: it advances
//! a whole grid of surface-moisture cells one weather / contact step on device,
//! so a wetness field can evolve in the same place the rest of the water
//! subsystem already lives, with no round trip to the host.
//!
//! Each cell couples two fields to the same rain drive:
//!
//! * wetness saturation in `0..=1`
//!   ([`update_wetness`](prism_render_architecture::water::wetness::update_wetness)):
//!   direct water contact soaks at the full `absorb_rate`; otherwise rain wets
//!   at `absorb_rate * clamp(rain, 0, 1)` and a rain-free surface dries at
//!   `dry_rate`. Both soak and dry are the exponential envelopes
//!   ([`absorb`](prism_render_architecture::water::wetness::absorb),
//!   [`dry`](prism_render_architecture::water::wetness::dry)) built on the
//!   shared non-negative `exp_approx`.
//! * standing puddle depth `>= 0`
//!   ([`puddle_depth`](prism_render_architecture::water::wetness::puddle_depth)):
//!   `max(depth + (rain - drain) * dt, 0)`.
//!
//! # Correctness model
//!
//! The step is embarrassingly parallel: one invocation owns one cell and reads
//! only that cell's own state and drivers, so there is no write hazard, no
//! barrier and no workgroup memory. Each cell performs the identical
//! floating-point operations the golden performs, in the same order. The
//! `exp_approx` envelope is the shared water transcendental
//! ([`exp_approx`](prism_render_architecture::water::exp_approx)); the shader
//! replicates it bit for bit — `base = 1 + x/4096`, clamp-to-zero, then twelve
//! squarings — using only multiply/add, so the readback is bit-identical to the
//! golden up to the last-place slack of a `GPU` fused multiply-add. A degenerate
//! request (zero cells or an input shorter than the cell count) returns the
//! wetness and puddle inputs unchanged, the twin's device-buffer contract.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset (multiply/add, `clamp`,
//! `min`/`max`, and index math) — no transcendental intrinsic, no optional
//! device feature — so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::wetness`；无第三方引擎源码或衍生代码。
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

/// Workgroup width of the moisture step; must match `@workgroup_size(256)` in
/// `shaders/water_wetness_step.wesl`.
const WORKGROUP_SIZE: u32 = 256;

/// One advanced surface-moisture grid.
///
/// Mirrors the per-cell [`SurfaceMoisture`](prism_render_architecture::water::wetness::SurfaceMoisture)
/// the `CPU` [`step_moisture`](prism_render_architecture::water::wetness::step_moisture)
/// returns, split into two parallel row-major fields. A degenerate request
/// yields the inputs unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterWetnessStep {
    /// Wetness saturation per cell, each in `0..=1`.
    pub wetness: Vec<f32>,
    /// Standing puddle depth per cell, each `>= 0`.
    pub puddle_depth: Vec<f32>,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/water_wetness_step.wesl` (`32` bytes, 16-byte aligned).
///
/// Only the fields [`step_moisture`](prism_render_architecture::water::wetness::step_moisture)
/// actually reads are carried; the response-only `WetnessParams` fields
/// (`max_capillary_height`, `darkening_strength`, `puddle_threshold`) do not
/// influence a step and are intentionally omitted.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    absorb_rate: f32,
    dry_rate: f32,
    drain_rate: f32,
    dt: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// A compiled, reusable moisture-step pipeline.
pub struct GpuWaterWetnessStep {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterWetnessStep {
    /// Compiles the moisture-step kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterWetnessStep {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_wetness_step"),
            source: ShaderSource::Wgsl(include_str!("../shaders/water_wetness_step.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_wetness_step_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_wetness_step_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_wetness_step_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("step_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterWetnessStep {
            module,
            layout,
            pipeline,
        }
    }

    /// Advances a grid of surface-moisture cells one step on device.
    ///
    /// `wetness`, `puddle` and `rain` are parallel row-major fields; `contact`
    /// carries one `0`/non-zero flag per cell (direct water contact). The three
    /// rates and `dt` are shared. Returns the inputs unchanged for a degenerate
    /// request (zero cells or any input shorter than `count`), the twin's
    /// device-buffer contract.
    #[expect(clippy::too_many_arguments, reason = "mirrors the golden drivers")]
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        wetness: &[f32],
        puddle: &[f32],
        contact: &[u32],
        rain: &[f32],
        absorb_rate: f32,
        dry_rate: f32,
        drain_rate: f32,
        dt: f32,
    ) -> WaterWetnessStep {
        let count = wetness.len();
        if count == 0 || puddle.len() < count || contact.len() < count || rain.len() < count {
            return WaterWetnessStep {
                wetness: wetness.to_vec(),
                puddle_depth: puddle.to_vec(),
            };
        }
        let device = ctx.device();

        let params = Params {
            count: count as u32,
            absorb_rate,
            dry_rate,
            drain_rate,
            dt,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_wetness_step_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let wetness_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_wetness_step_wetness"),
            contents: bytemuck::cast_slice(&wetness[..count]),
            usage: BufferUsages::STORAGE,
        });
        let puddle_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_wetness_step_puddle"),
            contents: bytemuck::cast_slice(&puddle[..count]),
            usage: BufferUsages::STORAGE,
        });
        let contact_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_wetness_step_contact"),
            contents: bytemuck::cast_slice(&contact[..count]),
            usage: BufferUsages::STORAGE,
        });
        let rain_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_wetness_step_rain"),
            contents: bytemuck::cast_slice(&rain[..count]),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = size_of_val(wetness) as u64;
        let wetness_out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_wetness_step_wetness_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let puddle_out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_wetness_step_puddle_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_wetness_step_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wetness_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: puddle_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: contact_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: rain_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: wetness_out_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: puddle_out_buf.as_entire_binding(),
                },
            ],
        });

        let wetness_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_wetness_step_wetness_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let puddle_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_wetness_step_puddle_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_wetness_step_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_wetness_step_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per cell; round the last workgroup up (the shader
            // guards `idx >= count`).
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&wetness_out_buf, 0, &wetness_stage, 0, out_bytes);
        encoder.copy_buffer_to_buffer(&puddle_out_buf, 0, &puddle_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        wetness_stage.slice(..).map_async(MapMode::Read, |_| {});
        puddle_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let wetness_out = read_f32(&wetness_stage);
        let puddle_out = read_f32(&puddle_stage);

        WaterWetnessStep {
            wetness: wetness_out,
            puddle_depth: puddle_out,
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
