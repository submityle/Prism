//! `wgpu` compute twin of the `ReSTIR` DI reuse-weight re-weighting step
//! (`reuse_weight` in
//! [`prism_render_architecture::lighting::restir_di`]).
//!
//! The `CPU` golden `reuse_weight` re-weights a source reservoir's held sample
//! for a destination pixel, forming the combined weight contribution
//! `w_i = p̂_dst(y) * W * M`: the destination-pixel target density `p̂_dst(y)` of
//! the held light `y`, times the source's finalized contribution weight `W`,
//! times its folded candidate count `M`. This is the per-source resampling
//! weight that spatial / temporal reuse folds into a destination reservoir (see
//! `fold_sources` / [`combine_biased`](prism_render_architecture::lighting::restir_di::combine_biased)).
//! It is a pure, fixed-width per-source transform: no container, no sort, no
//! random state — just a two-step product — so the whole function is a clean
//! device twin.
//!
//! [`GpuRestirDiReuseWeight`] is the on-device twin: one thread re-weights one
//! source reservoir, reproducing the reference product exactly, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same reuse weight the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one source the twin reads the destination target density
//! `target_pdf_at_dst`, the finalized weight `w` and the count `m`, and returns
//! `weight = target_pdf_at_dst * w * f32(m)`, matching the reference `reuse_weight`
//! evaluation order exactly.
//!
//! # What stays on the host
//!
//! Nothing of the re-weight itself: it is per-element and fixed-width. The host
//! owns the surrounding `ReSTIR` pipeline — the RIS streaming, the reservoir
//! `update` acceptance (which consumes a random draw), the variable-length
//! neighbor gather, and the normalization — plus the empty-batch short-circuit
//! (a storage buffer cannot be zero-sized).
//!
//! # Correctness model
//!
//! The weight threads through two multiplies, so `CPU` and `GPU` are not
//! guaranteed bit-exact: a `GPU` multiply may land a few units in the last place
//! from the scalar reference. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on `weight`, tight enough to catch
//! a genuinely wrong port (a dropped factor, a swapped operand) yet loose enough
//! to admit a legal last-place multiply difference. The count factor is an exact
//! integer-to-`f32` widening (counts stay well within the exact-integer range),
//! so it introduces no additional error.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — one integer-to-`f32`
//! widening and two multiplies — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `sqrt`, no inverse trigonometry, no `round`, and no bare f32 equality. No
//! optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. There is no loop: each thread performs a fixed, bounded
//! sequence, so the kernel provably terminates.
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

/// The portable core-`WGSL` `ReSTIR` DI reuse-weight kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `reuse_weight` product; see the module
/// documentation for the algorithm.
const RESTIR_DI_REUSE_WEIGHT_WGSL: &str = r#"
// ReSTIR DI reuse-weight twin: one thread computes one source reservoir's
// combined weight contribution weight = target_pdf_at_dst * w * f32(m),
// mirroring the CPU golden `lighting::restir_di::reuse_weight` with a single
// integer-to-f32 widening and two multiplies. It owns no RIS streaming, no
// reservoir update and no neighbor gather; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::lighting::restir_di；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of sources in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Destination-pixel target density p̂_dst(y) of the held light.
    target_pdf_at_dst: f32,
    // Finalized contribution weight W of the source reservoir.
    w: f32,
    // Folded candidate count M of the source reservoir.
    m: u32,
    pad0: u32,
}

struct Result {
    // Combined reuse weight w_i = p̂_dst(y) * W * M.
    weight: f32,
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

    // Reference evaluation order: (target_pdf_at_dst * w) * f32(m).
    let weight = q.target_pdf_at_dst * q.w * f32(q.m);

    var out: Result;
    out.weight = weight;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the source count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RESTIR_DI_REUSE_WEIGHT_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid sources in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one source query: the destination target
/// density, the finalized weight and the count plus one pad word to a `16`-byte
/// stride, matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Destination-pixel target density `p̂_dst(y)`.
    target_pdf_at_dst: f32,
    /// Finalized contribution weight `W`.
    w: f32,
    /// Folded candidate count `M`.
    m: u32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one reuse-weight result, matching the `WGSL`
/// `Result` struct: the combined weight plus three pad words to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Combined reuse weight `w_i`.
    weight: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One reuse-weight query: a source reservoir's destination-pixel target density
/// `target_pdf_at_dst`, finalized weight `w`, and folded count `m`.
///
/// Mirrors the state the `CPU` golden `reuse_weight` reads to compute the
/// combined reuse weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestirDiReuseWeightQuery {
    /// Destination-pixel target density `p̂_dst(y)` of the held light.
    pub target_pdf_at_dst: f32,
    /// Finalized contribution weight `W` of the source reservoir.
    pub w: f32,
    /// Folded candidate count `M` of the source reservoir.
    pub m: u32,
}

impl RestirDiReuseWeightQuery {
    /// Builds a reuse-weight query from the destination target density
    /// `target_pdf_at_dst`, the source finalized weight `w`, and its count `m`.
    #[must_use]
    pub const fn new(target_pdf_at_dst: f32, w: f32, m: u32) -> RestirDiReuseWeightQuery {
        RestirDiReuseWeightQuery {
            target_pdf_at_dst,
            w,
            m,
        }
    }
}

/// One reuse weight, mirroring the value the `CPU` golden `reuse_weight`
/// returns.
///
/// `weight` is `target_pdf_at_dst * w * m` — the combined weight contribution
/// of one source reservoir re-weighted for the destination pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestirDiReuseWeightResult {
    /// Combined reuse weight `w_i`.
    pub weight: f32,
}

/// Encodes one [`RestirDiReuseWeightQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &RestirDiReuseWeightQuery) -> GpuQuery {
    GpuQuery {
        target_pdf_at_dst: q.target_pdf_at_dst,
        w: q.w,
        m: q.m,
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`RestirDiReuseWeightResult`].
fn decode_result(raw: &GpuResult) -> RestirDiReuseWeightResult {
    RestirDiReuseWeightResult { weight: raw.weight }
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

/// A compiled, reusable `ReSTIR` DI reuse-weight compute pipeline, twinning the
/// `CPU` golden `reuse_weight` in
/// [`prism_render_architecture::lighting::restir_di`].
pub struct GpuRestirDiReuseWeight {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRestirDiReuseWeight {
    /// Compiles the reuse-weight kernel and builds its reusable pipeline on
    /// `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRestirDiReuseWeight {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_restir_di_reuse_weight_module"),
            source: ShaderSource::Wgsl(RESTIR_DI_REUSE_WEIGHT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_restir_di_reuse_weight_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_restir_di_reuse_weight_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_restir_di_reuse_weight_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRestirDiReuseWeight {
            module,
            layout,
            pipeline,
        }
    }

    /// Re-weights every source in `queries` and returns one
    /// [`RestirDiReuseWeightResult`] per input, in order.
    ///
    /// Each `weight` equals the reference `reuse_weight` value for the same
    /// inputs, within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RestirDiReuseWeightQuery],
    ) -> Vec<RestirDiReuseWeightResult> {
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
            label: Some("prism_volumetric_restir_di_reuse_weight_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_restir_di_reuse_weight_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_restir_di_reuse_weight_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_restir_di_reuse_weight_bind_group"),
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
            label: Some("prism_volumetric_restir_di_reuse_weight_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_restir_di_reuse_weight_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_restir_di_reuse_weight_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per source reservoir, flattened to a 1-D dispatch.
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
