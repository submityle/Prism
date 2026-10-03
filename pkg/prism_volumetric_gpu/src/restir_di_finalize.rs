//! `wgpu` compute twin of the `ReSTIR` DI unbiased contribution-weight
//! finalize
//! ([`DiReservoir::finalize`](prism_render_architecture::lighting::restir_di::DiReservoir::finalize)).
//!
//! The `CPU` golden
//! [`DiReservoir::finalize`](prism_render_architecture::lighting::restir_di::DiReservoir::finalize)
//! turns a finished reservoir into its unbiased Monte-Carlo contribution weight
//! `W = w_sum / (M * p̂(y))`, where `w_sum` is the running RIS weight sum, `M`
//! the folded candidate count, and `p̂(y)` the held sample's target density. It
//! guards a vanishing density and the empty reservoir, forcing `W = 0` when
//! `p̂(y) <= 1e-6` or `M == 0`, so the estimator never divides by a near-zero
//! denominator. The computation is a pure, fixed-width per-reservoir transform:
//! no container, no sort, no random state, so the whole function is a clean
//! device twin.
//!
//! [`GpuRestirDiFinalize`] is the on-device twin: one thread finalizes one
//! reservoir, reproducing the reference's guarded ratio exactly, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same unbiased weight the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For one reservoir the twin reads `w_sum`, the count `M` and the target
//! density `target_pdf`, applies the guard (`target_pdf <= 1e-6` or `M == 0`
//! yields `W = 0`), and otherwise returns `W = w_sum / (M * target_pdf)`,
//! matching the reference's delegation to
//! [`Reservoir::finalize_w`](prism_render_architecture::particle::reservoir_sample::Reservoir::finalize_w).
//!
//! # What stays on the host
//!
//! Nothing of the finalize itself: it is per-element and fixed-width. The host
//! owns the surrounding `ReSTIR` pipeline — the RIS streaming, the temporal and
//! spatial reuse, and the variable-length neighbor gathers — plus the
//! empty-batch short-circuit (a storage buffer cannot be zero-sized).
//!
//! # Correctness model
//!
//! The guard is a magnitude compare and an integer compare, so the discrete
//! decision (clamp to zero or not) agrees exactly between host and device for
//! fixtures clear of the `1e-6` density knot. The surviving weight threads
//! through one divide, so `CPU` and `GPU` are not bit-exact: a `GPU` divide may
//! land a few units in the last place from the scalar reference. The parity
//! test therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`)
//! on `W`, tight enough to catch a genuinely wrong port (a dropped guard, a
//! swapped numerator, a missing `M` factor) yet loose enough to admit a legal
//! last-place divide difference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — a magnitude compare,
//! an integer compare, one multiply and one divide — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `sqrt`, no inverse trigonometry, no `round` and no
//! `ceil`, and no bare f32 equality. No optional device feature is required, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each
//! thread performs a fixed, bounded sequence, so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_di`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `ReSTIR` DI finalize kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`DiReservoir::finalize`](prism_render_architecture::lighting::restir_di::DiReservoir::finalize)
/// guarded contribution-weight ratio; see the module documentation for the
/// algorithm.
const RESTIR_DI_FINALIZE_WGSL: &str = r#"
// ReSTIR DI finalize twin: one thread computes one reservoir's unbiased
// contribution weight W = w_sum / (M * target_pdf), forcing W = 0 when the
// target density vanishes (<= 1e-6) or the reservoir is empty (M == 0),
// mirroring the CPU golden `lighting::restir_di::DiReservoir::finalize` with
// only a compare, a multiply and a divide. It owns no RIS streaming, no reuse
// and no neighbor gather; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::lighting::restir_di；无第三方
// 引擎源码或衍生代码。

// Below this target density the weight is forced to zero, matching the
// reference FINALIZE_PDF_EPS / TARGET_PDF_EPS guard.
const TARGET_PDF_EPS: f32 = 1.0e-6;

struct Params {
    // Number of reservoirs in the storage arrays; threads past this
    // short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Running RIS weight sum w_sum of the finished reservoir.
    w_sum: f32,
    // Target density p̂(y) of the held sample at the owning pixel.
    target_pdf: f32,
    // Folded candidate count M.
    m: u32,
    pad0: u32,
}

struct Result {
    // Unbiased contribution weight W.
    w: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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

    // Guard a vanishing target density or an empty reservoir, forcing W = 0;
    // otherwise W = w_sum / (M * target_pdf). The compare is <= (never a bare
    // f32 equality) and the count test is an integer compare.
    var w: f32 = 0.0;
    if (q.target_pdf > TARGET_PDF_EPS && q.m != 0u) {
        let denom = f32(q.m) * q.target_pdf;
        w = q.w_sum / denom;
    }

    var out: Result;
    out.w = w;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the reservoir count plus three pad words
/// to fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RESTIR_DI_FINALIZE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid reservoirs in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one reservoir query: the weight sum, the target
/// density and the count plus one pad word to a `16`-byte stride, matching the
/// `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Running RIS weight sum `w_sum`.
    w_sum: f32,
    /// Target density `p̂(y)` of the held sample.
    target_pdf: f32,
    /// Folded candidate count `M`.
    m: u32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one finalize result, matching the `WGSL`
/// `Result` struct: the contribution weight plus three pad words to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Unbiased contribution weight `W`.
    w: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One finalize query: a finished reservoir's weight sum `w_sum`, folded count
/// `m`, and held-sample target density `target_pdf`.
///
/// Mirrors the state the `CPU` golden
/// [`DiReservoir::finalize`](prism_render_architecture::lighting::restir_di::DiReservoir::finalize)
/// reads to compute the unbiased contribution weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestirDiFinalizeQuery {
    /// Running RIS weight sum `w_sum` of the finished reservoir.
    pub w_sum: f32,
    /// Target density `p̂(y)` of the held sample at the owning pixel.
    pub target_pdf: f32,
    /// Folded candidate count `M`.
    pub m: u32,
}

impl RestirDiFinalizeQuery {
    /// Builds a finalize query from a reservoir's `w_sum`, held-sample
    /// `target_pdf`, and folded count `m`.
    #[must_use]
    pub const fn new(w_sum: f32, target_pdf: f32, m: u32) -> RestirDiFinalizeQuery {
        RestirDiFinalizeQuery {
            w_sum,
            target_pdf,
            m,
        }
    }
}

/// One finalized reservoir weight, mirroring the `W` the `CPU` golden
/// [`DiReservoir::finalize`](prism_render_architecture::lighting::restir_di::DiReservoir::finalize)
/// writes.
///
/// `W` is `0` when the target density vanishes (`<= 1e-6`) or the reservoir is
/// empty (`M == 0`), and `w_sum / (M * target_pdf)` otherwise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestirDiFinalizeResult {
    /// Unbiased contribution weight `W`.
    pub w: f32,
}

/// Encodes one [`RestirDiFinalizeQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &RestirDiFinalizeQuery) -> GpuQuery {
    GpuQuery {
        w_sum: q.w_sum,
        target_pdf: q.target_pdf,
        m: q.m,
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`RestirDiFinalizeResult`].
fn decode_result(raw: &GpuResult) -> RestirDiFinalizeResult {
    RestirDiFinalizeResult { w: raw.w }
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

/// A compiled, reusable `ReSTIR` DI finalize compute pipeline, twinning the
/// `CPU` golden
/// [`DiReservoir::finalize`](prism_render_architecture::lighting::restir_di::DiReservoir::finalize).
pub struct GpuRestirDiFinalize {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRestirDiFinalize {
    /// Compiles the `ReSTIR` DI finalize kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRestirDiFinalize {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_restir_di_finalize"),
            source: ShaderSource::Wgsl(RESTIR_DI_FINALIZE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_restir_di_finalize_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_restir_di_finalize_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_restir_di_finalize_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRestirDiFinalize {
            module,
            layout,
            pipeline,
        }
    }

    /// Finalizes every reservoir in `queries` and returns one
    /// [`RestirDiFinalizeResult`] per input, in order.
    ///
    /// Each `W` equals the reference
    /// [`DiReservoir::finalize`](prism_render_architecture::lighting::restir_di::DiReservoir::finalize)
    /// weight for the same inputs, within the tolerance documented on this
    /// module. An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RestirDiFinalizeQuery],
    ) -> Vec<RestirDiFinalizeResult> {
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
            label: Some("prism_volumetric_restir_di_finalize_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_restir_di_finalize_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_restir_di_finalize_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_restir_di_finalize_bind_group"),
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
            label: Some("prism_volumetric_restir_di_finalize_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_restir_di_finalize_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_restir_di_finalize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per reservoir, flattened to a 1-D dispatch.
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
