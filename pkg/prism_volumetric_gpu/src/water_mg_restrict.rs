//! `wgpu` compute twin of the multigrid full-weighting restriction operator
//! ([`restrict_full_weighting`](prism_render_architecture::water::pressure_multigrid::restrict_full_weighting)).
//!
//! The pressure Poisson solve for the water surface runs a geometric multigrid
//! `V`-cycle. One of its two inter-grid transfer operators is the full-weighting
//! restriction that maps a fine residual field onto the next coarser grid: each
//! interior coarse node gathers the `27` fine nodes in its `3x3x3`
//! neighbourhood with the separable tensor weight `[1/4, 1/2, 1/4]` per axis.
//! That gather is a fixed-width, closed-form stencil — pure multiply/add over
//! integer-addressed samples — so it ports cleanly to the device, and a passing
//! real-device parity run is direct evidence the ported kernel folds the same
//! weighted sum the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread computes one coarse-grid cell. Given a fine size `nf` and the
//! coarse size `nc = (nf - 1) / 2 + 1`, a cell at coarse coordinate
//! `(cx, cy, cz)` that is interior (`1 <= c < nc - 1` on every axis) centres on
//! the fine node `(2*cx, 2*cy, 2*cz)` and accumulates
//! `acc = sum over dx,dy,dz in 0..3 of w[dx]*w[dy]*w[dz] *
//! fine[idx(nf, fx+dx-1, fy+dy-1, fz+dz-1)]` with `w = [0.25, 0.5, 0.25]` and
//! the reference's row-major `idx(n, x, y, z) = (z*n + y)*n + x`. Boundary
//! coarse cells write `0`, matching the reference's `Dirichlet` boundary layer.
//!
//! # What stays on the host
//!
//! The variable-length `V`-cycle schedule, the grid-size bookkeeping, the
//! residual and smoother passes, the trilinear prolongation, and the
//! `fine.len() != nf^3` guard (which the host enforces by only enqueueing
//! well-formed, zero-padded queries) are host responsibilities. The device sees
//! only the fixed-capacity fine field and the single scalar `nf`.
//!
//! # Correctness model
//!
//! Every output is a short, fixed sequence of multiplies and adds over
//! integer-addressed fine samples — no transcendental, no `sqrt`, no divide
//! other than the integer coarse-size arithmetic — so the `CPU` and `GPU` agree
//! to within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The coarse size,
//! boundary classification, and fine indices are integer quantities computed
//! identically on both sides.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+`, `-`, `*`,
//! unsigned index arithmetic, and bounded `3x3x3` loops — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`, no
//! `round`, and no `sqrt`. Each thread performs a bounded sequence of
//! arithmetic, so the kernel provably terminates. No optional device feature is
//! required, so it runs unmodified across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pressure_multigrid`；无第三方引擎源码或衍生代码。
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

/// Fixed upper bound on the fine-grid edge size `nf` a query may carry. The
/// largest supported fine field is therefore `MAX_NF^3`.
pub const MAX_NF: usize = 9;

/// Fixed upper bound on the coarse-grid edge size, `(MAX_NF - 1) / 2 + 1`. The
/// largest coarse field is `MAX_NC^3`.
pub const MAX_NC: usize = 5;

/// Fixed capacity of a query's fine-field array (`MAX_NF^3 = 729`).
const MAX_FINE: usize = MAX_NF * MAX_NF * MAX_NF;

/// Fixed capacity of a result's coarse-field array (`MAX_NC^3 = 125`).
const MAX_COARSE: usize = MAX_NC * MAX_NC * MAX_NC;

/// The inlined `WGSL` twin of
/// [`restrict_full_weighting`](prism_render_architecture::water::pressure_multigrid::restrict_full_weighting):
/// one thread per coarse cell, folding the separable `[1/4, 1/2, 1/4]`
/// full-weighting stencil over the `3x3x3` fine neighbourhood with only `+`,
/// `-`, `*`, and unsigned index arithmetic.
const WATER_MG_RESTRICT_WGSL: &str = r#"
// Twin of water::pressure_multigrid::restrict_full_weighting. The separable
// full-weighting stencil [0.25, 0.5, 0.25] is folded over the 3x3x3 fine
// neighbourhood of each interior coarse cell; no transcendental, no sqrt, only
// multiply/add over integer-addressed samples.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::pressure_multigrid；无第三方引擎源码或衍生代码。

// Fixed capacities matching the Rust side (MAX_NF^3 and MAX_NC^3).
const MAX_FINE: u32 = 729u;
const MAX_COARSE: u32 = 125u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Fixed-capacity fine field, row-major, zero-padded beyond nf^3.
    fine: array<f32, 729>,
    // Fine-grid edge size nf (<= 9).
    nf: u32,
}

struct Result {
    // Fixed-capacity coarse field, row-major; cells beyond nc^3 stay zero.
    coarse: array<f32, 125>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Separable full-weighting weight for an offset d in {0, 1, 2}: 0.5 at the
// centre, 0.25 on either side.
fn weight(d: u32) -> f32 {
    if (d == 1u) {
        return 0.5;
    }
    return 0.25;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    // Thread tid serves coarse cell (tid % MAX_COARSE) of query (tid / MAX_COARSE).
    let q = tid / MAX_COARSE;
    if (q >= params.count) {
        return;
    }
    let c = tid % MAX_COARSE;

    let nf = queries[q].nf;
    let nc = (nf - 1u) / 2u + 1u;
    let total = nc * nc * nc;
    if (c >= total) {
        // Slot outside this query's coarse grid; keep it zero.
        results[q].coarse[c] = 0.0;
        return;
    }

    // Decode the row-major coarse linear index c into (cx, cy, cz).
    let cx = c % nc;
    let cy = (c / nc) % nc;
    let cz = c / (nc * nc);

    var acc: f32 = 0.0;
    // Interior cells (1 <= coord < nc - 1 on every axis) gather; boundary = 0.
    if (cx >= 1u && cx + 1u < nc && cy >= 1u && cy + 1u < nc && cz >= 1u && cz + 1u < nc) {
        let fx = 2u * cx;
        let fy = 2u * cy;
        let fz = 2u * cz;
        for (var dz: u32 = 0u; dz < 3u; dz = dz + 1u) {
            for (var dy: u32 = 0u; dy < 3u; dy = dy + 1u) {
                for (var dx: u32 = 0u; dx < 3u; dx = dx + 1u) {
                    let sx = fx + dx - 1u;
                    let sy = fy + dy - 1u;
                    let sz = fz + dz - 1u;
                    let fi = (sz * nf + sy) * nf + sx;
                    acc = acc + weight(dx) * weight(dy) * weight(dz) * queries[q].fine[fi];
                }
            }
        }
    }

    results[q].coarse[c] = acc;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_MG_RESTRICT_WGSL`].
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

/// `repr(C)` `std430` layout of one restriction query, matching the `WGSL`
/// `Query` struct: the fixed-capacity fine field followed by the fine edge
/// size (`MAX_FINE` `f32` plus one `u32`, alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Fixed-capacity fine field, row-major, zero-padded beyond `nf^3`.
    fine: [f32; MAX_FINE],
    /// Fine-grid edge size `nf`.
    nf: u32,
}

/// `repr(C)` `std430` layout of one restriction result, matching the `WGSL`
/// `Result` struct: the fixed-capacity coarse field (`MAX_COARSE` `f32`,
/// alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Fixed-capacity coarse field, row-major; cells beyond `nc^3` stay zero.
    coarse: [f32; MAX_COARSE],
}

/// One full-weighting restriction query to run on the device, mirroring the
/// inputs of
/// [`restrict_full_weighting`](prism_render_architecture::water::pressure_multigrid::restrict_full_weighting).
///
/// The host supplies the fine field (truncated and zero-padded to `MAX_NF^3` on
/// encode) and the fine edge size `nf` (clamped to [`MAX_NF`]).
#[derive(Clone, Debug, PartialEq)]
pub struct WaterMgRestrictQuery {
    /// Fine field, row-major, length `nf^3` (shorter inputs are zero-padded on
    /// encode, longer ones truncated to [`MAX_NF`]`^3`).
    pub fine: Vec<f32>,
    /// Fine-grid edge size `nf`.
    pub nf: u32,
}

/// One restriction result, mirroring the golden coarse field. The `coarse`
/// vector has length `nc^3` where `nc = (nf - 1) / 2 + 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterMgRestrictResult {
    /// Coarse field, row-major, length `nc^3`.
    pub coarse: Vec<f32>,
}

/// Returns the coarse edge size for a fine edge size `nf`, matching the golden
/// [`coarse_size`](prism_render_architecture::water::pressure_multigrid::coarse_size).
fn coarse_size(nf: u32) -> u32 {
    (nf - 1) / 2 + 1
}

/// Encodes one [`WaterMgRestrictQuery`] into its `std430` [`GpuQuery`] slot,
/// zero-filling the fine array and copying the input up to [`MAX_NF`]`^3`.
fn encode_query(q: &WaterMgRestrictQuery) -> GpuQuery {
    let mut fine = [0.0f32; MAX_FINE];
    let count = q.fine.len().min(MAX_FINE);
    fine[..count].copy_from_slice(&q.fine[..count]);
    let nf = (q.nf as usize).min(MAX_NF) as u32;
    GpuQuery { fine, nf }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterMgRestrictResult`],
/// keeping only the first `nc^3` entries for the query's fine size `nf`.
fn decode_result(nf: u32, raw: &GpuResult) -> WaterMgRestrictResult {
    let nc = coarse_size((nf as usize).min(MAX_NF) as u32) as usize;
    let total = (nc * nc * nc).min(MAX_COARSE);
    WaterMgRestrictResult {
        coarse: raw.coarse[..total].to_vec(),
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

/// A compiled, reusable full-weighting restriction compute pipeline, twinning
/// the `CPU` golden
/// [`restrict_full_weighting`](prism_render_architecture::water::pressure_multigrid::restrict_full_weighting).
pub struct GpuWaterMgRestrict {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterMgRestrict {
    /// Compiles the restriction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterMgRestrict {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_mg_restrict"),
            source: ShaderSource::Wgsl(WATER_MG_RESTRICT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_mg_restrict_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_mg_restrict_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_mg_restrict_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterMgRestrict {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every query in `queries` and returns one [`WaterMgRestrictResult`]
    /// per input, in order.
    ///
    /// The outputs match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterMgRestrictQuery],
    ) -> Vec<WaterMgRestrictResult> {
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
            label: Some("prism_volumetric_water_mg_restrict_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_mg_restrict_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_mg_restrict_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_mg_restrict_bind_group"),
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
            label: Some("prism_volumetric_water_mg_restrict_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_mg_restrict_encoder"),
        });
        {
            // One thread per coarse cell: count * MAX_COARSE threads total.
            let threads = (count * MAX_COARSE) as u32;
            let groups = threads.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_mg_restrict_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(q, r)| decode_result(q.nf, r))
            .collect()
    }
}
