//! `wgpu` compute twin of the horizon sunset-reddening falloff
//! ([`sunset_reddening`](prism_render_architecture::volumetric::spectral::sunset_reddening)).
//!
//! The spectral sky model (design section 8) reddens the light near the horizon
//! as the sun sinks: short wavelengths scatter away along the long grazing path
//! so the surviving light shifts red. The amount is
//! `saturate(1 - smoothstep(0, REDDEN_MAX_ALTITUDE, sun_altitude))`, where
//! `sun_altitude` is the sun's angular altitude in radians (zero at the horizon,
//! positive above). It is one at and below the horizon (full reddening),
//! decreases monotonically as the sun climbs, and is zero once the sun rises
//! above `REDDEN_MAX_ALTITUDE`, so it stays in `0..=1` and is monotone
//! non-increasing in altitude. The `CPU` golden
//! [`sunset_reddening`](prism_render_architecture::volumetric::spectral::sunset_reddening)
//! owns that math; [`GpuSunsetReddening`] is the on-device twin that runs one
//! thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The `smoothstep` is expanded to the *same* closed form the CPU
//! `math::smoothstep` uses (a degenerate-span guard, a saturated ramp, then the
//! cubic `t*t*(3-2t)`), the reddening constant is the same `REDDEN_MAX_ALTITUDE`,
//! and the kernel contains no transcendental call — clamp/saturate and a
//! multiply-add — so `CPU` and `GPU` evaluate the same closed-form algebra. The
//! only slack is a legal multiply-add contraction of a few `ULP`, so the parity
//! test asserts a tight tolerance (`abs_diff < 1e-6` or `rel_diff < 1e-5`). The
//! scenes also assert the value stays in `0..=1` and is monotone non-increasing
//! in altitude (a lower sun always reddens at least as much), so a degenerate
//! kernel could not pass.
//!
//! # Portability
//!
//! The kernel is clamp/saturate and multiply/add in the portable core-`WGSL`
//! subset — no `exp`, `pow` or optional device feature — so it runs unmodified
//! on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard horizon-reddening falloff plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.
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

/// One sunset-reddening query: the sun's angular altitude in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SunsetReddeningQuery {
    /// Sun's angular altitude in radians; zero at the horizon, positive above.
    pub sun_altitude: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/sunset_reddening.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    sun_altitude: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/sunset_reddening.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable sunset-reddening pipeline.
pub struct GpuSunsetReddening {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSunsetReddening {
    /// Compiles the sunset-reddening kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSunsetReddening {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sunset_reddening"),
            source: ShaderSource::Wgsl(include_str!("../shaders/sunset_reddening.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sunset_reddening_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sunset_reddening_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sunset_reddening_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sunset_reddening_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSunsetReddening {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the sunset-reddening amount for every query in `queries`,
    /// returning one value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`sunset_reddening`](prism_render_architecture::volumetric::spectral::sunset_reddening)`(q.sun_altitude)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[SunsetReddeningQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                sun_altitude: q.sun_altitude,
                pad0: 0.0,
                pad1: 0.0,
                pad2: 0.0,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sunset_reddening_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sunset_reddening_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sunset_reddening_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sunset_reddening_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sunset_reddening_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sunset_reddening_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sunset_reddening_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let values = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(values.len(), queries.len());
        values
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
