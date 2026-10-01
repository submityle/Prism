//! `wgpu` compute twin of Prism's accumulated global forward-scatter
//! transmittance
//! ([`forward_scatter_power`](prism_render_architecture::hair::dual_scatter_sh::forward_scatter_power)),
//! the `Zinke` dual-scattering global multiplier `Ψ = a_f^n`.
//!
//! A dual-scattering groom reconstructs the multiple-scattering lobe from two
//! pieces: the local forward-scatter factor `a_f` and the number of fibres `n`
//! a path crosses on the way to the light. The global forward multiplier is
//! `a_f` raised to that crossing count — the fraction of light that survives
//! `n` successive forward-scatter events. Like `UE5` Groom and AMD `TressFX`,
//! the groom evaluates this per receiver sample once the crossing count is
//! known, so a whole tile of `(a_f, n)` pairs is raised in one dispatch.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairForwardScatterPower::eval`] takes a batch of
//! [`PowerQuery`] pairs — a forward-scatter factor `base` and a crossed-strand
//! count `exponent` — and returns one `base^exponent` per query, preserving
//! input order. The query index is the invocation id (`@compute
//! @workgroup_size(64)`, one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the query count early-return.
//!
//! # A data-dependent multiply loop, not `pow`
//!
//! The power is an explicit integer multiply loop (`accum = 1`, multiplied by
//! `base` exactly `exponent` times) exactly like the golden, **not** `powi` /
//! `powf` / `exp(n·ln a_f)`: the exponent is a per-thread `u32` read from a
//! storage buffer, so the loop trip count is data-dependent and diverges
//! between invocations in the same workgroup. `exponent = 0` yields exactly
//! `1.0` without entering the loop; `exponent = 1` yields the sanitised base.
//!
//! # Portability
//!
//! The kernel uses only integer arithmetic and `f32` multiply — no `exp`,
//! `pow`, `sqrt`, `sin` or optional device feature — so the twin runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Unlike the tolerance-checked analytic twins this kernel is checked
//! **bit-exact**. The accumulator is a *dependent* chain of IEEE-754
//! single-precision multiplies: there is no add for the device to fuse into an
//! fma, and because the trip count is read from a buffer the compiler cannot
//! fold the loop into a closed-form `pow`, so the device must reproduce the
//! scalar reference's rounding to the bit. The parity test therefore compares
//! raw bit patterns ([`f32::to_bits`]) rather than an approximate difference.
//! The sanitiser mirrors the golden exactly (`is_finite` test rejecting
//! `NaN`/±inf, collapsing a non-finite base to `0`), so a non-finite base
//! produces the same bit-exact result (`0^0 = 1`, `0^{n>0} = 0`).
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Zinke` dual-scattering global multiplier plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::dual_scatter_sh::forward_scatter_power;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One forward-scatter power query: raise the forward-scatter factor `base` to
/// the crossed-strand count `exponent`.
///
/// `exponent = 0` yields `1.0` (no crossings survive trivially); larger counts
/// multiply `base` into the accumulator `exponent` times. A non-finite `base`
/// is sanitised to `0` before the power, matching the golden.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowerQuery {
    /// The local forward-scatter factor `a_f`.
    pub base: f32,
    /// The crossed-strand count `n`.
    pub exponent: u32,
}

/// Uniform parameters for one power dispatch. Layout matches `Params` in
/// `shaders/forward_scatter_power.wesl`: the query count in a single `16`-byte
/// uniform slot (one `u32` plus padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    element_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One query uploaded to the kernel. `8`-byte `repr(C)` matching `PowerQuery`
/// in `shaders/forward_scatter_power.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPowerQuery {
    base: f32,
    exponent: u32,
}

/// A compiled, reusable per-query forward-scatter power pipeline.
pub struct GpuHairForwardScatterPower {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairForwardScatterPower {
    /// Compiles the per-query forward-scatter power kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset (integer arithmetic
    /// plus `f32` multiply), so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairForwardScatterPower {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_forward_scatter_power"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/forward_scatter_power.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_forward_scatter_power_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_forward_scatter_power_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_forward_scatter_power_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairForwardScatterPower {
            module,
            layout,
            pipeline,
        }
    }

    /// Raises each query's `base` to its `exponent`, returning one value per
    /// query in input order.
    ///
    /// The value for query `i` equals the `CPU` golden
    /// [`forward_scatter_power`](prism_render_architecture::hair::dual_scatter_sh::forward_scatter_power)
    /// of `(base, exponent)` **bit-for-bit** (a dependent multiply chain with no
    /// fma to fuse), with a non-finite base sanitised to the same result. An
    /// empty batch yields an empty vector without a dispatch — storage buffers
    /// cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[PowerQuery]) -> Vec<f32> {
        let element_count = queries.len();
        if element_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            element_count: element_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let gpu_queries: Vec<GpuPowerQuery> = queries
            .iter()
            .map(|q| GpuPowerQuery {
                base: q.base,
                exponent: q.exponent,
            })
            .collect();

        // Output is one f32 (4 bytes) per query.
        let out_bytes = (element_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_forward_scatter_power_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_forward_scatter_power_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_forward_scatter_power_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_forward_scatter_power_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_forward_scatter_power_bind_group"),
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
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_forward_scatter_power_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_forward_scatter_power_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (element_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        out
    }
}

/// The `CPU` golden forward-scatter power for one query, re-exported so the
/// parity test can assert the device twin against the identical reference it
/// mirrors.
#[must_use]
pub fn reference_forward_scatter_power(query: PowerQuery) -> f32 {
    forward_scatter_power(query.base, query.exponent)
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
