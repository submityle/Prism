//! `wgpu` compute twin of the segment-integration kernel
//! ([`integrate_segment`](prism_render_architecture::volumetric::raymarch::integrate_segment)).
//!
//! Each marched raymarch segment folds its in-scattered radiance into the
//! accumulator and attenuates the running `transmittance` (design section 6).
//! Negative coefficients are clamped to zero and `light_transmittance` /
//! `powder_factor` are saturated, so the update never amplifies energy or
//! produces `NaN`. The in-scatter uses the analytic segment integral
//! `sigma_s * (1 - exp(-sigma_t*step)) / sigma_t`, degrading to `sigma_s * step`
//! as `sigma_t -> 0` (the thin-medium limit).
//!
//! The `CPU` golden
//! [`integrate_segment`](prism_render_architecture::volumetric::raymarch::integrate_segment)
//! owns that logic; [`GpuIntegrateSegment`] is the on-device twin that runs one
//! thread per query.
//!
//! # Correctness model
//!
//! The `CPU` golden uses a hand-rolled `exp_approx` (a polynomial, no float
//! intrinsic, for cross-target determinism); this kernel mirrors that same
//! polynomial `exp` so the on-device result matches bit-close, rather than
//! calling the `WGSL` `exp` builtin. The parity test asserts every state field
//! within a tight tolerance (`steps_taken` exactly) and checks the physical
//! invariants: `transmittance` stays in `[0, 1]` and never increases,
//! `scattered` never decreases, and a zero `sigma_t` leaves `transmittance`
//! unchanged.
//!
//! # Portability
//!
//! The kernel is arithmetic plus the polynomial `exp` in the portable
//! core-`WGSL` subset — no `exp`/`pow` builtin or optional device feature — so
//! it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard analytic volumetric segment integration plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::volumetric::raymarch::RaymarchState;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One segment-integration query: the incoming accumulator state plus the
/// homogeneous segment coefficients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegrateSegmentQuery {
    /// The accumulator state before this segment is folded in.
    pub state: RaymarchState,
    /// Segment extinction coefficient (clamped to `>= 0`).
    pub sigma_t: f32,
    /// Segment scattering coefficient (clamped to `>= 0`).
    pub sigma_s: f32,
    /// Phase-function value for the view/light geometry (clamped to `>= 0`).
    pub phase: f32,
    /// Segment length along the ray (clamped to `>= 0`).
    pub step: f32,
    /// Light visibility at the segment (saturated to `0..=1`).
    pub light_transmittance: f32,
    /// Powder/edge-darkening factor (saturated to `0..=1`).
    pub powder_factor: f32,
}

/// One query as uploaded. `48`-byte `repr(C)` matching `Query` in
/// `shaders/integrate_segment.wesl`: the four state fields followed by the six
/// segment coefficients and two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    transmittance: f32,
    optical_depth: f32,
    scattered: f32,
    steps_taken: u32,
    sigma_t: f32,
    sigma_s: f32,
    phase: f32,
    step: f32,
    light_transmittance: f32,
    powder_factor: f32,
    pad0: u32,
    pad1: u32,
}

/// One result as read back. `16`-byte `repr(C)` matching `Res` in
/// `shaders/integrate_segment.wesl`: the updated accumulator state.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    transmittance: f32,
    optical_depth: f32,
    scattered: f32,
    steps_taken: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/integrate_segment.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable segment-integration pipeline.
pub struct GpuIntegrateSegment {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuIntegrateSegment {
    /// Compiles the segment-integration kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuIntegrateSegment {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_integrate_segment_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/integrate_segment.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_integrate_segment_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_integrate_segment_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_integrate_segment_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("integrate_segment_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuIntegrateSegment {
            module,
            layout,
            pipeline,
        }
    }

    /// Integrates every query in `queries`, returning one updated
    /// [`RaymarchState`] per query in input order.
    ///
    /// The returned value for query `q` equals the `CPU` golden
    /// [`integrate_segment`](prism_render_architecture::volumetric::raymarch::integrate_segment)
    /// applied to a copy of `q.state` with the same coefficients: every field
    /// matches within a tight floating-point tolerance (`steps_taken` exactly).
    /// An empty `queries` slice yields an empty result — storage buffers cannot
    /// be zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[IntegrateSegmentQuery]) -> Vec<RaymarchState> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                transmittance: q.state.transmittance,
                optical_depth: q.state.optical_depth,
                scattered: q.state.scattered,
                steps_taken: q.state.steps_taken,
                sigma_t: q.sigma_t,
                sigma_s: q.sigma_s,
                phase: q.phase,
                step: q.step,
                light_transmittance: q.light_transmittance,
                powder_factor: q.powder_factor,
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

        let out_bytes = (queries.len() as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_integrate_segment_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_integrate_segment_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_integrate_segment_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_integrate_segment_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_integrate_segment_bind_group"),
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
            label: Some("prism_volumetric_integrate_segment_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_integrate_segment_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());
        raw.into_iter()
            .map(|r| RaymarchState {
                transmittance: r.transmittance,
                optical_depth: r.optical_depth,
                scattered: r.scattered,
                steps_taken: r.steps_taken,
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
