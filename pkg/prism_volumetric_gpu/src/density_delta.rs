//! `wgpu` compute twin of the carve-brush density delta
//! ([`density_delta`](prism_render_architecture::volumetric::coupling::density_delta)).
//!
//! Two-way scene coupling (design section 9d) lets aircraft and projectiles
//! punch cavities and wakes into the cloud density field. The brush is a purely
//! negative source: the delta is `-strength` at the brush centre and eases
//! smoothly to `0` at (and beyond) the brush radius via a [`smoothstep`] falloff,
//! so the result lies in `-1..=0`. Applied through the saturating `apply_carve`
//! choke point it can only ever remove density, never manufacturing a negative
//! final density or `transmittance`. Out-of-range radii (`<= 0`) are floored to
//! `EPS` and strengths are saturated, so the kernel never divides by zero. The
//! `CPU` golden
//! [`density_delta`](prism_render_architecture::volumetric::coupling::density_delta)
//! owns that math; [`GpuDensityDelta`] is the on-device twin that runs one
//! thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The distance uses `sqrt` (mirroring
//! [`Vec3::distance`](prism_render_architecture::volumetric::Vec3) `==`
//! `length_squared().sqrt()`) and the falloff mirrors the reference's
//! hand-rolled Hermite [`smoothstep`](prism_render_architecture::volumetric::math)
//! bit-for-bit — the collapsed-edge `EPS` guard and the `t*t*(3-2t)` polynomial
//! — not the device-native `smoothstep`. Because the kernel contains no
//! transcendental call, `CPU` and `GPU` evaluate the same closed-form algebra;
//! the only slack is a legal multiply-add contraction of a few `ULP`, so the
//! parity test asserts a tight tolerance (`abs_diff < 1e-6` or `rel_diff < 1e-5`).
//! The scenes also assert the documented `-1..=0` range, the deepest carve at
//! the centre, the vanishing delta beyond the radius, and that degenerate
//! (`radius <= 0`, out-of-range strength) brushes stay bounded, so a degenerate
//! kernel could not pass.
//!
//! # Portability
//!
//! The kernel is `sqrt`, `clamp` and multiply/add in the portable core-`WGSL`
//! subset — no `exp`, `pow` or optional device feature — so it runs unmodified
//! on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard spherical-falloff density carving plus `wgpu` compute
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

use prism_render_architecture::volumetric::Vec3;

use crate::context::GpuContext;

/// A spherical negative-density carve brush; mirrors
/// [`CarveBrush`](prism_render_architecture::volumetric::coupling::CarveBrush).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarveBrush {
    /// World-space centre of the spherical brush.
    pub center: Vec3,
    /// Brush radius in world units; values below `EPS` are floored so the
    /// falloff never divides by (near) zero.
    pub radius: f32,
    /// Carve strength in `0..=1`; the peak density removed at the centre.
    pub strength: f32,
}

/// One carve query: the sampled world position and the brush that acts on it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityDeltaQuery {
    /// World-space position being sampled.
    pub pos: Vec3,
    /// The spherical negative-density brush.
    pub brush: CarveBrush,
}

/// One query as uploaded. `32`-byte `repr(C)` matching `Query` in
/// `shaders/density_delta.wesl`: two 16-byte rows keep it 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    radius: f32,
    center_x: f32,
    center_y: f32,
    center_z: f32,
    strength: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/density_delta.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable carve-brush density-delta pipeline.
pub struct GpuDensityDelta {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDensityDelta {
    /// Compiles the carve-brush density-delta kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDensityDelta {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_density_delta"),
            source: ShaderSource::Wgsl(include_str!("../shaders/density_delta.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_density_delta_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_density_delta_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_density_delta_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("density_delta_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDensityDelta {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the carve-brush density delta for every query in `queries`,
    /// returning one value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`density_delta`](prism_render_architecture::volumetric::coupling::density_delta)`(q.pos, q.brush)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[DensityDeltaQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                pos_x: q.pos.x,
                pos_y: q.pos.y,
                pos_z: q.pos.z,
                radius: q.brush.radius,
                center_x: q.brush.center.x,
                center_y: q.brush.center.y,
                center_z: q.brush.center.z,
                strength: q.brush.strength,
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
            label: Some("prism_volumetric_density_delta_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_density_delta_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_density_delta_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_density_delta_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_density_delta_bind_group"),
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
            label: Some("prism_volumetric_density_delta_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_density_delta_pass"),
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
