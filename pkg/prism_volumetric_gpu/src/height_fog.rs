//! `wgpu` compute twin of the exponential height-fog density
//! ([`height_fog_density`](prism_render_architecture::volumetric::fog::height_fog_density)).
//!
//! The ground fog layer (design section 9f) is densest at sea level and thins
//! with altitude: `density_at_sea_level * e^{-falloff * altitude}`, culled to
//! exactly zero at or above `max_height` so the fog volume has a hard,
//! artist-controlled ceiling. The sea-level density, falloff and ceiling are
//! each floored at zero, and negative altitudes are treated as sea level so the
//! value never exceeds the sea-level density. This is an extinction
//! coefficient, not a probability, so it is only floored at zero, not clamped
//! to one. The `CPU` golden
//! [`height_fog_density`](prism_render_architecture::volumetric::fog::height_fog_density)
//! owns that math; [`GpuHeightFog`] is the on-device twin that runs one thread
//! per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The exponential is evaluated with the *same* hand-rolled `exp_approx` the
//! reference uses — base-two range reduction with a fractional seven-term
//! polynomial times an integer power assembled from the `f32` exponent field —
//! not the device-native `exp`. Mirroring the polynomial keeps the twin
//! bit-close to the reference, so the parity test asserts a tight tolerance
//! (`abs_diff < 1e-6` or `rel_diff < 1e-5`). The scenes also assert the density
//! is non-negative, never exceeds the sea-level density, decays monotonically
//! with altitude, is culled to exactly zero at the ceiling and treats negative
//! altitudes as sea level, so a degenerate kernel could not pass.
//!
//! # Portability
//!
//! The kernel is `floor`, `bitcast`, integer ops and multiply/add in the
//! portable core-`WGSL` subset — no `exp`, `pow` or optional device feature —
//! so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard exponential height-fog extinction plus `wgpu` compute
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

/// One height-fog query: the sample altitude plus the exponential layer
/// parameters (mirroring
/// [`HeightFogParams`](prism_render_architecture::volumetric::fog::HeightFogParams)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightFogQuery {
    /// Altitude at which to sample the layer, in world units.
    pub altitude: f32,
    /// Extinction coefficient at sea level (altitude `0`); floored at `0`.
    pub density_at_sea_level: f32,
    /// Exponential falloff rate per world unit of altitude; floored at `0`.
    pub falloff: f32,
    /// Altitude ceiling above which the fog density is culled to zero.
    pub max_height: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/height_fog.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    altitude: f32,
    density_at_sea_level: f32,
    falloff: f32,
    max_height: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/height_fog.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable height-fog density pipeline.
pub struct GpuHeightFog {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHeightFog {
    /// Compiles the height-fog density kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHeightFog {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_height_fog"),
            source: ShaderSource::Wgsl(include_str!("../shaders/height_fog.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_height_fog_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_height_fog_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_height_fog_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("height_fog_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHeightFog {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the height-fog density for every query in `queries`, returning
    /// one value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`height_fog_density`](prism_render_architecture::volumetric::fog::height_fog_density)`(q.altitude, params)`
    /// (with `params` built from the query's layer fields) to within the
    /// tolerance documented on this module. An empty `queries` slice yields an
    /// empty result — storage buffers cannot be zero-sized, so it is handled by
    /// an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[HeightFogQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                altitude: q.altitude,
                density_at_sea_level: q.density_at_sea_level,
                falloff: q.falloff,
                max_height: q.max_height,
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
            label: Some("prism_volumetric_height_fog_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_height_fog_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_height_fog_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_height_fog_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_height_fog_bind_group"),
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
            label: Some("prism_volumetric_height_fog_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_height_fog_pass"),
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
