//! `wgpu` compute twin of the weather coverage-relaxation kernel
//! ([`relax_coverage`](prism_render_architecture::volumetric::weather::relax_coverage)).
//!
//! The sky-state machine (`Clear` -> `Fair` -> `Overcast` -> `Storm`) drives a
//! target cloud coverage per cell; each cell approaches its target with a
//! smooth exponential blend rather than snapping, coupling generation and decay
//! to the weather field (design section 9). Over a step `dt` at rate `rate`
//! (per unit time) the blend weight is
//! `k = saturate(1 - exp(-max(rate, 0) * max(dt, 0)))` and the new coverage is
//! `current + (target - current) * k`.
//!
//! The `CPU` golden
//! [`relax_coverage`](prism_render_architecture::volumetric::weather::relax_coverage)
//! owns that logic; [`GpuRelaxCoverage`] is the on-device twin that runs one
//! thread per query.
//!
//! # Correctness model
//!
//! Clamping `rate` and `dt` to non-negative and saturating `k` keeps the weight
//! in `0..=1`, so the result is monotone toward `target`, never overshoots, and
//! stays in `0..=1` whenever both `current` and `target` do. The `CPU` golden
//! uses a hand-rolled `exp_approx` (no float intrinsic, for determinism across
//! targets); this kernel mirrors that same polynomial exp, so the parity test
//! asserts agreement within a tight floating-point tolerance rather than bit
//! equality.
//!
//! # Portability
//!
//! The kernel is the portable core-`WGSL` subset — the mirrored polynomial exp,
//! `clamp` and arithmetic, no optional device feature — so it runs unmodified
//! on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard exponential relaxation / weather coverage coupling plus
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

/// One relaxation query: the current coverage, its target, the relaxation rate
/// and the timestep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelaxCoverageQuery {
    /// The cell's current coverage.
    pub current: f32,
    /// The coverage the cell is relaxing toward.
    pub target: f32,
    /// The relaxation rate per unit time (clamped non-negative).
    pub rate: f32,
    /// The timestep (clamped non-negative).
    pub dt: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/relax_coverage.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    current: f32,
    target: f32,
    rate: f32,
    dt: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/relax_coverage.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable coverage-relaxation pipeline.
pub struct GpuRelaxCoverage {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRelaxCoverage {
    /// Compiles the coverage-relaxation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRelaxCoverage {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_relax_coverage"),
            source: ShaderSource::Wgsl(include_str!("../shaders/relax_coverage.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_relax_coverage_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_relax_coverage_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_relax_coverage_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("relax_coverage_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRelaxCoverage {
            module,
            layout,
            pipeline,
        }
    }

    /// Relaxes every query in `queries`, returning one new coverage per query in
    /// input order.
    ///
    /// The returned value for query `q` equals
    /// [`relax_coverage`](prism_render_architecture::volumetric::weather::relax_coverage)`(q.current, q.target, q.rate, q.dt)`
    /// within a tight floating-point tolerance (the mirrored polynomial exp). An
    /// empty `queries` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[RelaxCoverageQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                current: q.current,
                target: q.target,
                rate: q.rate,
                dt: q.dt,
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
            label: Some("prism_volumetric_relax_coverage_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_relax_coverage_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_relax_coverage_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_relax_coverage_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_relax_coverage_bind_group"),
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
            label: Some("prism_volumetric_relax_coverage_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_relax_coverage_pass"),
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
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(out.len(), queries.len());
        out
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
