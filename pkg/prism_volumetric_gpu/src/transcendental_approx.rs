//! `wgpu` compute twin of the hand-rolled transcendentals
//! ([`exp_approx`](prism_render_architecture::volumetric::math::exp_approx),
//! [`ln_approx`](prism_render_architecture::volumetric::math::ln_approx) and
//! [`pow_approx`](prism_render_architecture::volumetric::math::pow_approx)).
//!
//! The determinism policy forbids the hardware transcendentals, so the CPU
//! builds `exp(x)` as `2^(x*log2e)` (an integer power assembled from the `f32`
//! exponent field plus a fractional polynomial) and `ln(x)` from exponent
//! extraction plus an `atanh` series on the centred mantissa; `pow(b, e)` is
//! `exp(e * ln(b))` with non-positive bases mapped to `0`. Several volume
//! kernels embed `exp_approx` / `ln_approx` inline, but only over the narrow
//! ranges those kernels produce. [`GpuTranscendental`] isolates all three so a
//! wide-domain sweep exercises the saturation branches — `+inf` for large
//! positive `exp`, the `0` flush for large negative, the non-positive `ln`
//! clamp, the `SQRT_2` mantissa-centring branch and the negative-base `pow`
//! guard — in one place.
//!
//! # Correctness model
//!
//! The `exp_approx` and `ln_approx` blocks are copied verbatim from the
//! already-verified `shaders/analytic_transmittance.wesl` and
//! `shaders/single_scatter_reference.wesl`, so `CPU` and `GPU` evaluate the
//! identical closed-form algebra and differ at most by a legal multiply-add
//! contraction of a few `ULP`. The parity test sweeps a wide domain and
//! compares with a combined absolute/relative tolerance (and exact equality for
//! infinities), so a wrong branch or a dropped polynomial term could not pass.
//!
//! # Portability
//!
//! The kernel is `floor`/`bitcast` plus multiply/add in the portable
//! core-`WGSL` subset, so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard base-two range-reduced exp/ln plus `wgpu` compute
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

/// One transcendental query: the `exp` argument, the `ln` argument and the
/// `pow` base/exponent pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TranscendentalQuery {
    /// Argument for `exp_approx`.
    pub exp_x: f32,
    /// Argument for `ln_approx`.
    pub ln_x: f32,
    /// Base for `pow_approx`.
    pub pow_base: f32,
    /// Exponent for `pow_approx`.
    pub pow_exp: f32,
}

/// The three transcendentals evaluated for one query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TranscendentalResult {
    /// `exp_approx(exp_x)`.
    pub exp: f32,
    /// `ln_approx(ln_x)`.
    pub ln: f32,
    /// `pow_approx(pow_base, pow_exp)`.
    pub pow: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/transcendental_approx.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    exp_x: f32,
    ln_x: f32,
    pow_base: f32,
    pow_exp: f32,
}

/// One result as read back. `16`-byte `repr(C)` matching `Res` in
/// `shaders/transcendental_approx.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    exp: f32,
    ln: f32,
    pow: f32,
    pad: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/transcendental_approx.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable transcendental pipeline.
pub struct GpuTranscendental {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTranscendental {
    /// Compiles the transcendental kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTranscendental {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_transcendental_approx"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/transcendental_approx.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_transcendental_approx_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_transcendental_approx_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_transcendental_approx_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("transcendental_approx_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTranscendental {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates `exp_approx`, `ln_approx` and `pow_approx` for every query,
    /// returning one [`TranscendentalResult`] per query in input order.
    ///
    /// Each field equals the corresponding `CPU` golden to within the tolerance
    /// documented on this module. An empty `queries` slice yields an empty
    /// result — storage buffers cannot be zero-sized, so it is handled by an
    /// early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[TranscendentalQuery],
    ) -> Vec<TranscendentalResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                exp_x: q.exp_x,
                ln_x: q.ln_x,
                pow_base: q.pow_base,
                pow_exp: q.pow_exp,
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
            label: Some("prism_volumetric_transcendental_approx_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_transcendental_approx_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_transcendental_approx_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_transcendental_approx_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_transcendental_approx_bind_group"),
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
            label: Some("prism_volumetric_transcendental_approx_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_transcendental_approx_pass"),
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
            .map(|r| TranscendentalResult {
                exp: r.exp,
                ln: r.ln,
                pow: r.pow,
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
