//! `wgpu` compute twin of the two multiple-importance-sampling weights inside
//! the reference path tracer
//! ([`mis`](prism_render_architecture::reference_pt::mis)).
//!
//! When the same lighting integral can be estimated by more than one sampling
//! strategy — a point on the light versus the surface `BSDF` lobe — `MIS`
//! combines both into one unbiased estimator by weighting each strategy's
//! single sample by a function of the two solid-angle densities. This module is
//! the on-device twin of the two classical weights:
//!
//! - [`balance_heuristic`](prism_render_architecture::reference_pt::mis::balance_heuristic):
//!   `pdf_a / (pdf_a + pdf_b)`, the minimum-variance linear blend, returning `0`
//!   when both densities vanish.
//! - [`power_heuristic`](prism_render_architecture::reference_pt::mis::power_heuristic):
//!   `pdf_a^2 / (pdf_a^2 + pdf_b^2)` (Veach's beta `= 2`), which sharpens the
//!   balance weight so a locally far better strategy dominates, again returning
//!   `0` when both densities vanish.
//!
//! [`GpuMisHeuristics`] evaluates both for one query per thread, reproducing the
//! reference's exact multiply-and-divide closed form, so a passing real-device
//! parity test is direct evidence the ported kernel computes the same weights
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`MisHeuristicsQuery`] — the two solid-angle densities
//! `pdf_a` and `pdf_b` — and writes one [`MisHeuristicsResult`] holding the
//! balance and power weights. Both are a sum, a `<= 0` guard and a single divide
//! (the power weight squaring the densities first); there is no loop and no
//! transcendental.
//!
//! # Precision note
//!
//! The `CPU` golden promotes the densities to `f64` before dividing so a large
//! light-sampling density cannot overflow the squaring. `WGSL` has no `f64`, so
//! the kernel computes in `f32` directly. The host therefore restricts the
//! parity fixtures to densities in `[0, 1e3]`, where the squared sum stays far
//! from `f32` overflow and the `f32` and `f64` divides agree to within the
//! documented tolerance.
//!
//! # What stays on the host
//!
//! The strategy selection, the per-bounce density evaluation and the estimator
//! accumulation stay on the host; the device sees only the two fixed-width
//! weights, one query at a time, so a storage buffer is never zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /` and unsigned
//! index arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`, `sqrt`,
//! `round` or `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::mis`；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` `MIS` balance-and-power kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`balance_heuristic`](prism_render_architecture::reference_pt::mis::balance_heuristic)
/// and
/// [`power_heuristic`](prism_render_architecture::reference_pt::mis::power_heuristic)
/// closed forms; see the module documentation for the algorithm.
const MIS_HEURISTICS_WGSL: &str = r#"
// MIS heuristics twin: one thread computes one query's balance and power
// weights, mirroring the CPU golden
// `reference_pt::mis::{balance_heuristic, power_heuristic}` with only + - * /
// and a <= 0 guard. It owns no strategy selection and no estimator
// accumulation; those stay on the host. The golden promotes to f64 before
// dividing; WGSL has no f64, so this computes in f32 and the host keeps the
// fixtures in [0, 1e3] where the two agree within tolerance.
//
// Provenance: 孪生自本仓 prism_render_architecture::reference_pt::mis；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // The first strategy's solid-angle density.
    pdf_a: f32,
    // The second strategy's solid-angle density.
    pdf_b: f32,
    pad0: f32,
    pad1: f32,
}

struct Result {
    // The balance-heuristic weight pdf_a / (pdf_a + pdf_b).
    balance: f32,
    // The power-heuristic weight pdf_a^2 / (pdf_a^2 + pdf_b^2).
    power: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let a = q.pdf_a;
    let b = q.pdf_b;

    var out: Result;

    // Balance: pdf_a / (pdf_a + pdf_b), zero when the sum is non-positive.
    let denom = a + b;
    var balance: f32 = 0.0;
    if (denom <= 0.0) {
        balance = 0.0;
    } else {
        balance = a / denom;
    }
    out.balance = balance;

    // Power (beta = 2): square both densities, then the same guarded divide.
    let a2 = a * a;
    let b2 = b * b;
    let denom2 = a2 + b2;
    var power: f32 = 0.0;
    if (denom2 <= 0.0) {
        power = 0.0;
    } else {
        power = a2 / denom2;
    }
    out.power = power;

    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MIS_HEURISTICS_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the two densities plus two pad words to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// The first strategy's solid-angle density `pdf_a`.
    pdf_a: f32,
    /// The second strategy's solid-angle density `pdf_b`.
    pdf_b: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the balance and power weights plus two pad words to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// The balance-heuristic weight.
    balance: f32,
    /// The power-heuristic weight.
    power: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One query for the `MIS` heuristics twin: the two solid-angle densities of
/// the strategies being combined.
///
/// `pdf_a` is the density of the strategy whose weight is returned; `pdf_b` is
/// the competing strategy's density. The host owns strategy selection and
/// enqueues one [`MisHeuristicsQuery`] per sample it needs weighted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MisHeuristicsQuery {
    /// The first strategy's solid-angle density `pdf_a`.
    pub pdf_a: f32,
    /// The second strategy's solid-angle density `pdf_b`.
    pub pdf_b: f32,
}

impl MisHeuristicsQuery {
    /// Builds a query from the two densities `pdf_a` and `pdf_b`.
    #[must_use]
    pub const fn new(pdf_a: f32, pdf_b: f32) -> MisHeuristicsQuery {
        MisHeuristicsQuery { pdf_a, pdf_b }
    }
}

/// One resolved query of the `MIS` heuristics twin: the balance and power
/// weights applied to the first strategy.
///
/// `balance` is
/// [`balance_heuristic`](prism_render_architecture::reference_pt::mis::balance_heuristic),
/// `power` is
/// [`power_heuristic`](prism_render_architecture::reference_pt::mis::power_heuristic);
/// both are `0` when both densities vanish.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MisHeuristicsResult {
    /// The balance-heuristic weight `pdf_a / (pdf_a + pdf_b)`.
    pub balance: f32,
    /// The power-heuristic weight `pdf_a^2 / (pdf_a^2 + pdf_b^2)`.
    pub power: f32,
}

/// Encodes one [`MisHeuristicsQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &MisHeuristicsQuery) -> GpuQuery {
    GpuQuery {
        pdf_a: q.pdf_a,
        pdf_b: q.pdf_b,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`MisHeuristicsResult`].
fn decode_result(raw: &GpuResult) -> MisHeuristicsResult {
    MisHeuristicsResult {
        balance: raw.balance,
        power: raw.power,
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

/// A compiled, reusable `MIS` balance-and-power compute pipeline, twinning the
/// `CPU` golden
/// [`balance_heuristic`](prism_render_architecture::reference_pt::mis::balance_heuristic)
/// and
/// [`power_heuristic`](prism_render_architecture::reference_pt::mis::power_heuristic).
pub struct GpuMisHeuristics {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMisHeuristics {
    /// Compiles the `MIS` heuristics kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMisHeuristics {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mis_heuristics"),
            source: ShaderSource::Wgsl(MIS_HEURISTICS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mis_heuristics_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mis_heuristics_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mis_heuristics_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMisHeuristics {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`MisHeuristicsResult`]
    /// per input, in order.
    ///
    /// The balance and power weights match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MisHeuristicsQuery],
    ) -> Vec<MisHeuristicsResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mis_heuristics_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mis_heuristics_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mis_heuristics_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mis_heuristics_bind_group"),
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
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mis_heuristics_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mis_heuristics_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mis_heuristics_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
