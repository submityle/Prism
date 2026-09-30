//! `wgpu` compute twin of the Beer-Lambert transmittance kernel
//! ([`analytic_transmittance`](prism_render_architecture::volumetric::reference::analytic_transmittance)).
//!
//! This is the closed-form transmittance of a homogeneous medium — the analytic
//! value the delta / ratio tracking estimators must converge to (design
//! section 9c reference). It returns `exp(-sigma_t * distance)` with negative
//! `sigma_t` and `distance` clamped to zero and the result saturated into
//! `[0, 1]`, so a zero distance yields exactly `1` and the value is never
//! outside the physical range.
//!
//! The `CPU` golden
//! [`analytic_transmittance`](prism_render_architecture::volumetric::reference::analytic_transmittance)
//! owns that logic; [`GpuAnalyticTransmittance`] is the on-device twin that runs
//! one thread per query.
//!
//! # Correctness model
//!
//! The `CPU` golden uses a hand-rolled `exp_approx` (a polynomial, no float
//! intrinsic, for cross-target determinism); this kernel mirrors that same
//! polynomial `exp` so the on-device result matches bit-close, rather than
//! calling the `WGSL` `exp` builtin. The parity test asserts the transmittance
//! within a tight tolerance and checks that it stays in `[0, 1]` and decreases
//! monotonically with optical depth.
//!
//! # Portability
//!
//! The kernel is arithmetic plus the polynomial `exp` in the portable
//! core-`WGSL` subset — no `exp` builtin or optional device feature — so it runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard Beer-Lambert transmittance plus `wgpu` compute
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

/// One transmittance query: the extinction coefficient and the path distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalyticTransmittanceQuery {
    /// The medium extinction coefficient (clamped to `>= 0`).
    pub sigma_t: f32,
    /// The path distance through the medium (clamped to `>= 0`).
    pub distance: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/analytic_transmittance.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    sigma_t: f32,
    distance: f32,
    pad0: u32,
    pad1: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/analytic_transmittance.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable Beer-Lambert transmittance pipeline.
pub struct GpuAnalyticTransmittance {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAnalyticTransmittance {
    /// Compiles the Beer-Lambert transmittance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAnalyticTransmittance {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_analytic_transmittance_shader"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/analytic_transmittance.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_analytic_transmittance_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_analytic_transmittance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_analytic_transmittance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("analytic_transmittance_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAnalyticTransmittance {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning one transmittance per query
    /// in input order.
    ///
    /// The returned value for query `q` equals
    /// [`analytic_transmittance`](prism_render_architecture::volumetric::reference::analytic_transmittance)`(q.sigma_t, q.distance)`
    /// within a tight floating-point tolerance. An empty `queries` slice yields
    /// an empty result — storage buffers cannot be zero-sized, so it is handled
    /// by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[AnalyticTransmittanceQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                sigma_t: q.sigma_t,
                distance: q.distance,
                pad0: 0,
                pad1: 0,
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
            label: Some("prism_volumetric_analytic_transmittance_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_analytic_transmittance_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_analytic_transmittance_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_analytic_transmittance_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_analytic_transmittance_bind_group"),
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
            label: Some("prism_volumetric_analytic_transmittance_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_analytic_transmittance_pass"),
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
        let raw = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());
        raw
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
