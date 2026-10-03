//! `wgpu` compute twin of the trilinear multigrid prolongation primitive
//! ([`prolong_trilinear`](prism_render_architecture::water::pressure_multigrid::prolong_trilinear),
//! the coarse-to-fine transfer operator of the pressure-`Poisson` multigrid
//! `V`-cycle).
//!
//! The `CPU` golden
//! [`prolong_trilinear`](prism_render_architecture::water::pressure_multigrid::prolong_trilinear)
//! interpolates a coarse cubic grid of edge `nc` onto the fine grid of edge
//! `nf = (nc - 1) * 2 + 1`. For every fine lattice index it gathers the per-axis
//! coarse contributors — an *even* fine index copies the single coincident
//! coarse node with weight `1`, while an *odd* fine index averages the two
//! straddling coarse nodes with weight `1/2` each — and accumulates the tensor
//! product `wx * wy * wz * coarse[idx(nc, ix, iy, iz)]` over the (at most eight)
//! surviving corner terms. The result is the dense fine field laid out with the
//! same row-major `idx(n, x, y, z) = (z * n + y) * n + x` ordering the reference
//! uses.
//!
//! [`GpuWaterMgProlong`] is the on-device twin of that closed-form gather: one
//! thread resolves one coarse grid (one query), looping over its `nf^3` fine
//! cells and reproducing the reference's exact per-axis contributor rule with
//! only integer index arithmetic, `+ - *` and the parity test `i & 1u`. A
//! passing real-device parity run is therefore direct evidence the ported
//! kernel computes the same interpolation the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! The numeric interpolation gather: for each fine `(x, y, z)` the per-axis
//! contributors `(coarse_index, weight)` and the weighted corner sum that
//! writes `out[idx(nf, x, y, z)]`. The device reproduces the reference's parity
//! split exactly — even index single-node copy, odd index two-node average —
//! because it is pure integer indexing plus multiply-add with no transcendental
//! term.
//!
//! # What stays on the host
//!
//! The variable-length container work of the reference stays host-side: the
//! `Vec` allocation sized to `nf^3`, the degenerate `coarse.len() != nc^3`
//! all-zero guard, and the surrounding `V`-cycle recursion
//! (`restrict`/`smooth`/`residual`). The host enqueues one query per coarse
//! grid with its data already zero-padded into the fixed
//! [`MAX_NC`]`^3`-element slot, so a storage buffer is never zero-sized and the
//! length guard never fires on the device.
//!
//! # Correctness model
//!
//! Every quantity on the device is built from integer index arithmetic and
//! exact binary-representable weights (`1.0`, `0.5`), so the only arithmetic
//! that can diverge from the scalar reference is the floating accumulation
//! order of the corner sum. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on each fine value, tight enough
//! to catch a genuinely wrong port (a dropped weight, a swapped parity branch,
//! a wrong index) yet loose enough to admit a legal last-place reassociation
//! difference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned index
//! arithmetic, `+ - *`, and the bitwise parity test `i & 1u` — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `sqrt`, no inverse trigonometry, no `round` and
//! no `floor`. No optional device feature is required, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. Each thread performs a fixed, bounded triple
//! loop over at most `MAX_NF^3` cells, so the kernel provably terminates.
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

/// Maximum coarse grid edge length the device twin accepts. A coarse grid holds
/// `MAX_NC * MAX_NC * MAX_NC` samples; the host zero-pads shorter grids into the
/// fixed-width slot.
pub const MAX_NC: usize = 5;

/// Maximum fine grid edge length, `(MAX_NC - 1) * 2 + 1`. A fine grid holds
/// `MAX_NF * MAX_NF * MAX_NF` samples.
pub const MAX_NF: usize = 9;

/// Fixed coarse slot length, `MAX_NC^3`.
const COARSE_LEN: usize = MAX_NC * MAX_NC * MAX_NC;

/// Fixed fine slot length, `MAX_NF^3`.
const FINE_LEN: usize = MAX_NF * MAX_NF * MAX_NF;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` trilinear prolongation kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`prolong_trilinear`](prism_render_architecture::water::pressure_multigrid::prolong_trilinear)
/// gather; see the module documentation for the algorithm.
const WATER_MG_PROLONG_WGSL: &str = r#"
// Trilinear multigrid prolongation twin: one thread resolves one coarse grid,
// looping over its nf^3 fine cells and reproducing the CPU golden
// `water::pressure_multigrid::prolong_trilinear` per-axis contributor rule with
// only integer index arithmetic, + - * and the parity test `i & 1u`. It owns no
// Vec allocation, no length guard and no V-cycle recursion; those stay on the
// host.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::pressure_multigrid；无第三方
// 引擎源码或衍生代码。

const MAX_NF: u32 = 9u;
const COARSE_LEN: u32 = 125u;
const FINE_LEN: u32 = 729u;

struct Params {
    // Number of coarse grids (queries) in the storage arrays; threads past this
    // short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // The coarse field, zero-padded to COARSE_LEN; only the first nc^3 are used.
    coarse: array<f32, 125>,
    // The coarse grid edge length nc.
    nc: u32,
}

struct Result {
    // The fine field, zero-padded to FINE_LEN; only the first nf^3 are written.
    fine: array<f32, 729>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Row-major coarse index idx(n, x, y, z) = (z*n + y)*n + x, mirroring the
// reference's private `idx`.
fn coarse_idx(n: u32, x: u32, y: u32, z: u32) -> u32 {
    return (z * n + y) * n + x;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= params.count) {
        return;
    }
    let nc = queries[qi].nc;
    // nf = (nc - 1) * 2 + 1 for the fine grid edge length.
    let nf = (nc - 1u) * 2u + 1u;

    // Clear the whole fixed slot so the padding beyond nf^3 is a deterministic
    // zero, matching the host-side zero-pad of the oracle.
    for (var i: u32 = 0u; i < FINE_LEN; i = i + 1u) {
        results[qi].fine[i] = 0.0;
    }

    for (var z: u32 = 0u; z < nf; z = z + 1u) {
        // Per-axis contributors for z: even -> single node weight 1; odd -> two
        // straddling nodes weight 1/2 each. The weight-0 slot of the even case
        // contributes nothing to the sum, exactly as the reference's skip does.
        var iz: array<u32, 2>;
        var wz: array<f32, 2>;
        if ((z & 1u) == 0u) {
            iz[0] = z / 2u;
            wz[0] = 1.0;
            iz[1] = z / 2u;
            wz[1] = 0.0;
        } else {
            iz[0] = (z - 1u) / 2u;
            wz[0] = 0.5;
            iz[1] = (z + 1u) / 2u;
            wz[1] = 0.5;
        }
        for (var y: u32 = 0u; y < nf; y = y + 1u) {
            var iy: array<u32, 2>;
            var wy: array<f32, 2>;
            if ((y & 1u) == 0u) {
                iy[0] = y / 2u;
                wy[0] = 1.0;
                iy[1] = y / 2u;
                wy[1] = 0.0;
            } else {
                iy[0] = (y - 1u) / 2u;
                wy[0] = 0.5;
                iy[1] = (y + 1u) / 2u;
                wy[1] = 0.5;
            }
            for (var x: u32 = 0u; x < nf; x = x + 1u) {
                var ix: array<u32, 2>;
                var wx: array<f32, 2>;
                if ((x & 1u) == 0u) {
                    ix[0] = x / 2u;
                    wx[0] = 1.0;
                    ix[1] = x / 2u;
                    wx[1] = 0.0;
                } else {
                    ix[0] = (x - 1u) / 2u;
                    wx[0] = 0.5;
                    ix[1] = (x + 1u) / 2u;
                    wx[1] = 0.5;
                }

                var acc: f32 = 0.0;
                for (var a: u32 = 0u; a < 2u; a = a + 1u) {
                    for (var b: u32 = 0u; b < 2u; b = b + 1u) {
                        for (var c: u32 = 0u; c < 2u; c = c + 1u) {
                            let w = wz[a] * wy[b] * wx[c];
                            let ci = coarse_idx(nc, ix[c], iy[b], iz[a]);
                            acc = acc + w * queries[qi].coarse[ci];
                        }
                    }
                }
                let fine_i = (z * nf + y) * nf + x;
                results[qi].fine[fine_i] = acc;
            }
        }
    }
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_MG_PROLONG_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid coarse grids in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one coarse-grid query: the zero-padded coarse
/// field followed by its edge length `nc`, matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// The coarse field, zero-padded to [`COARSE_LEN`] entries.
    coarse: [f32; COARSE_LEN],
    /// The coarse grid edge length `nc`.
    nc: u32,
}

/// `repr(C)` `std430` layout of one fine-grid result, matching the `WGSL`
/// `Result` struct: the fine field zero-padded to [`FINE_LEN`] entries.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// The fine field, zero-padded to [`FINE_LEN`] entries.
    fine: [f32; FINE_LEN],
}

/// One coarse-grid prolongation query: the coarse field and its edge length.
///
/// The host owns the surrounding multigrid aggregate — the `Vec` allocation,
/// the degenerate `coarse.len() != nc^3` all-zero guard, and the `V`-cycle
/// recursion — and enqueues one [`WaterMgProlongQuery`] per coarse grid,
/// mirroring the reference
/// [`prolong_trilinear`](prism_render_architecture::water::pressure_multigrid::prolong_trilinear).
#[derive(Clone, Debug, PartialEq)]
pub struct WaterMgProlongQuery {
    /// The coarse field in row-major order; its length must be `nc * nc * nc`
    /// and at most [`COARSE_LEN`].
    pub coarse: Vec<f32>,
    /// The coarse grid edge length `nc`; at most [`MAX_NC`].
    pub nc: usize,
}

impl WaterMgProlongQuery {
    /// Builds a query from a coarse `field` of edge length `nc`.
    #[must_use]
    pub fn new(coarse: Vec<f32>, nc: usize) -> WaterMgProlongQuery {
        WaterMgProlongQuery { coarse, nc }
    }
}

/// One resolved fine grid, mirroring the dense output of the reference
/// [`prolong_trilinear`](prism_render_architecture::water::pressure_multigrid::prolong_trilinear).
///
/// The `fine` field is laid out row-major with the reference's
/// `idx(n, x, y, z) = (z * n + y) * n + x` ordering and has length
/// `nf * nf * nf`, where `nf = (nc - 1) * 2 + 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterMgProlongResult {
    /// The fine field in row-major order, length `nf^3`.
    pub fine: Vec<f32>,
    /// The fine grid edge length `nf`.
    pub nf: usize,
}

/// Encodes one [`WaterMgProlongQuery`] into its `std430` [`GpuQuery`] slot,
/// zero-padding the coarse field to [`COARSE_LEN`].
fn encode_query(q: &WaterMgProlongQuery) -> GpuQuery {
    let mut coarse = [0.0f32; COARSE_LEN];
    for (slot, &v) in coarse.iter_mut().zip(q.coarse.iter()) {
        *slot = v;
    }
    GpuQuery {
        coarse,
        nc: q.nc as u32,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterMgProlongResult`],
/// trimming the fixed slot to the `nf^3` meaningful entries of the query's grid.
fn decode_result(raw: &GpuResult, nc: usize) -> WaterMgProlongResult {
    let nf = if nc == 0 { 0 } else { (nc - 1) * 2 + 1 };
    let len = nf * nf * nf;
    WaterMgProlongResult {
        fine: raw.fine[..len].to_vec(),
        nf,
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

/// A compiled, reusable trilinear multigrid prolongation compute pipeline,
/// twinning the numeric core of the `CPU` golden
/// [`prolong_trilinear`](prism_render_architecture::water::pressure_multigrid::prolong_trilinear).
pub struct GpuWaterMgProlong {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterMgProlong {
    /// Compiles the prolongation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterMgProlong {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_mg_prolong"),
            source: ShaderSource::Wgsl(WATER_MG_PROLONG_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_mg_prolong_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_mg_prolong_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_mg_prolong_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterMgProlong {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every coarse grid in `queries` and returns one
    /// [`WaterMgProlongResult`] per input, in order.
    ///
    /// Each fine value matches the reference to within the tolerance documented
    /// on this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterMgProlongQuery],
    ) -> Vec<WaterMgProlongResult> {
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
            label: Some("prism_volumetric_water_mg_prolong_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_mg_prolong_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_mg_prolong_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_mg_prolong_bind_group"),
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
            label: Some("prism_volumetric_water_mg_prolong_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_mg_prolong_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_mg_prolong_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per coarse grid, flattened to a 1-D dispatch.
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

        raw.iter()
            .zip(queries.iter())
            .map(|(r, q)| decode_result(r, q.nc))
            .collect()
    }
}
