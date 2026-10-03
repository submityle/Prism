//! `wgpu` compute twin of the damped-`Jacobi` smoother from the water
//! geometric-`multigrid` pressure solver
//! ([`pressure_multigrid`](prism_render_architecture::water::pressure_multigrid)).
//!
//! Removing divergence from a liquid velocity field means solving a Poisson
//! system `A p = b` on a vertex-centred grid, where `A` is the `7`-point
//! negative Laplacian `(6 p_c - sum of 6 neighbours) / h^2` on interior nodes
//! and the boundary layer holds a `Dirichlet` `p = 0`. The golden
//! [`smooth`](prism_render_architecture::water::pressure_multigrid::smooth) runs
//! `iters` damped-`Jacobi` sweeps in place: each sweep forms the residual
//! `r = b - A p` on interior nodes and then updates `p[c] += factor * r[c]` with
//! `factor = omega * h * h / 6`, leaving the boundary layer untouched.
//!
//! `Jacobi` is embarrassingly parallel because a sweep reads only the previous
//! iterate, so [`GpuWaterMgSmooth`] ports it directly: one thread owns one cell,
//! each sweep is one dispatch that reads the old field and writes the new field,
//! and the host ping-pongs two storage buffers for `iters` sweeps back to back.
//! A passing real-device parity test is direct evidence the ported kernel
//! reproduces the reference arithmetic, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one query carrying a per-axis node count `n`, the grid spacing `h`, the
//! damping `omega`, the sweep count `iters`, an initial field `p0` and the
//! right-hand side `b`, the kernel reproduces the full `smooth` loop:
//! - interior nodes update by `p_new[c] = p[c] + factor * (b[c] - A p[c])` with
//!   the `6`-neighbour stencil accumulated in the reference order and
//!   `factor = omega * h * h / 6`; and
//! - boundary nodes, padding cells beyond `n^3`, and sweeps past a query's
//!   `iters` copy the previous value through unchanged.
//!
//! The six face neighbours are summed in the identical order the scalar
//! reference uses, and the residual then `p`-update are evaluated as the golden
//! does, so the two run the same arithmetic.
//!
//! # What stays on the host
//!
//! The sweep loop lives on the host: it issues one dispatch per sweep and swaps
//! the two field buffers, mirroring the golden host loop, so the device never
//! needs cross-cell synchronisation within a sweep. Each grid is bounded to
//! [`MAX_N`] nodes per axis (`MAX_N^3` cells, [`MAX_N_CUBED`]) so the storage
//! layout is fixed; the host pads shorter grids. An empty batch short-circuits
//! on the host, since a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! No sweep term contains a transcendental call, so `CPU` and `GPU` evaluate the
//! same `f32` arithmetic in the same order and differ only by the legal rounding
//! of a reciprocal. Because the error of `iters` sweeps accumulates, the parity
//! test keeps `iters` small (`<= 4`) and widens the continuous tolerance to
//! `abs_diff <= 2e-4` or `rel_diff <= 2e-3`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer index
//! arithmetic plus `+`, `-`, `*`, `/` on `f32` — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, and no `64`-bit integers or
//! floats. No optional device feature is required, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
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

/// Maximum per-axis node count carried by one query. The golden grid is
/// vertex-centred; the twin bounds it so the `std430` field layout is fixed.
pub const MAX_N: usize = 9;

/// Maximum number of cells per grid (`MAX_N^3`). Each query's `p0` and `b`
/// buffers are host-padded to this length.
pub const MAX_N_CUBED: usize = MAX_N * MAX_N * MAX_N;

/// The portable core-`WGSL` damped-`Jacobi` smoother kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `sweep`
/// performs one `Jacobi` sweep; the host loops it `iters` times with a
/// ping-pong buffer pair, mirroring the `CPU` golden `smooth`.
const WATER_MG_SMOOTH_WGSL: &str = r#"
// Damped-Jacobi smoother twin: one thread owns one cell, one dispatch is one
// sweep reading the old field and writing the new field. Interior nodes update
// by p_new = p + factor * (b - A p) with the 7-point negative Laplacian; the
// boundary layer, padding cells, and sweeps past a query's iters copy through.
// Mirrors the CPU golden `water::pressure_multigrid::smooth`.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::pressure_multigrid；无第三方
// 引擎源码或衍生代码。

const CELLS: u32 = 729u;

// Row-major flat index matching the golden idx(n, x, y, z) = (z*n + y)*n + x.
fn flat(n: u32, x: u32, y: u32, z: u32) -> u32 {
    return (z * n + y) * n + x;
}

struct Params {
    // Number of queries in the batch; threads past this short-circuit.
    count: u32,
    // Index of the sweep being issued (0-based); queries whose iters are at or
    // below this copy their field through unchanged.
    sweep: u32,
    pad0: u32,
    pad1: u32,
}

struct Query {
    // Per-axis node count (<= 9).
    n: u32,
    // Number of damped-Jacobi sweeps requested.
    iters: u32,
    // Grid spacing.
    h: f32,
    // Damping factor.
    omega: f32,
    // Right-hand side, host-padded to CELLS entries.
    b: array<f32, 729>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read> p_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> p_out: array<f32>;

@compute @workgroup_size(64)
fn sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.count * CELLS;
    if (gid.x >= total) {
        return;
    }
    let q_index = gid.x / CELLS;
    let cell = gid.x - q_index * CELLS;
    let base = q_index * CELLS;
    let slot = base + cell;

    let query = queries[q_index];
    let n = query.n;
    let ncells = n * n * n;

    // Invalid grid (n < 3), padding cells beyond n^3, or a sweep past this
    // query's iters: copy the previous value through unchanged.
    if (n < 3u || cell >= ncells || params.sweep >= query.iters) {
        p_out[slot] = p_in[slot];
        return;
    }

    let plane = n * n;
    let z = cell / plane;
    let rem = cell - z * plane;
    let y = rem / n;
    let x = rem - y * n;

    // Boundary layer is a fixed Dirichlet value: copy through.
    if (x == 0u || y == 0u || z == 0u || x == n - 1u || y == n - 1u || z == n - 1u) {
        p_out[slot] = p_in[slot];
        return;
    }

    // Interior node: residual r = b - A p, then p_new = p + factor * r. The six
    // face neighbours are summed in the reference order (+-x, +-y, +-z).
    let pc = p_in[slot];
    let s = 6.0 * pc
        - p_in[base + flat(n, x - 1u, y, z)]
        - p_in[base + flat(n, x + 1u, y, z)]
        - p_in[base + flat(n, x, y - 1u, z)]
        - p_in[base + flat(n, x, y + 1u, z)]
        - p_in[base + flat(n, x, y, z - 1u)]
        - p_in[base + flat(n, x, y, z + 1u)];
    let inv_h2 = 1.0 / (query.h * query.h);
    let ap = s * inv_h2;
    let r = query.b[cell] - ap;
    let factor = query.omega * query.h * query.h / 6.0;
    p_out[slot] = pc + factor * r;
}
"#;

/// Uniform parameters for one sweep dispatch: the query count, the sweep index,
/// and two pad words, filling a `16`-byte uniform struct matching `Params` in
/// [`WATER_MG_SMOOTH_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the batch.
    count: u32,
    /// Index of the sweep being issued.
    sweep: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one query's static data: the node count, the
/// sweep count, the spacing and damping, and the host-padded right-hand side,
/// matching the `WGSL` `Query` struct. The field iterate itself lives in the
/// separate ping-pong buffers.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Per-axis node count.
    n: u32,
    /// Number of damped-`Jacobi` sweeps.
    iters: u32,
    /// Grid spacing.
    h: f32,
    /// Damping factor.
    omega: f32,
    /// Right-hand side, host-padded to `MAX_N_CUBED` entries.
    b: [f32; MAX_N_CUBED],
}

/// One damped-`Jacobi` smoother query for the twin: the grid size and tuning,
/// the initial field `p0`, and the right-hand side `b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterMgSmoothQuery {
    /// Per-axis node count (must be `<= MAX_N`).
    pub n: u32,
    /// Grid spacing.
    pub h: f32,
    /// Damping factor.
    pub omega: f32,
    /// Number of damped-`Jacobi` sweeps.
    pub iters: u32,
    /// Initial field, host-padded to `MAX_N_CUBED` entries.
    pub p0: [f32; MAX_N_CUBED],
    /// Right-hand side, host-padded to `MAX_N_CUBED` entries.
    pub b: [f32; MAX_N_CUBED],
}

impl WaterMgSmoothQuery {
    /// Builds a query from the grid size, tuning, and the `p0` and `b` slices.
    ///
    /// Each slice is copied into a fixed `MAX_N_CUBED`-entry buffer, with the
    /// remaining entries left at `0`. Slices longer than `MAX_N_CUBED` are
    /// truncated to the capacity; the caller is expected to pass exactly `n^3`
    /// entries for a grid of per-axis size `n`.
    #[must_use]
    pub fn new(
        n: u32,
        h: f32,
        omega: f32,
        iters: u32,
        p0: &[f32],
        b: &[f32],
    ) -> WaterMgSmoothQuery {
        let mut p0_buf = [0.0f32; MAX_N_CUBED];
        let mut b_buf = [0.0f32; MAX_N_CUBED];
        let p0_len = p0.len().min(MAX_N_CUBED);
        let b_len = b.len().min(MAX_N_CUBED);
        p0_buf[..p0_len].copy_from_slice(&p0[..p0_len]);
        b_buf[..b_len].copy_from_slice(&b[..b_len]);
        WaterMgSmoothQuery {
            n,
            h,
            omega,
            iters,
            p0: p0_buf,
            b: b_buf,
        }
    }
}

/// One resolved damped-`Jacobi` smoother query: the smoothed field after
/// `iters` sweeps, mirroring the reference
/// [`smooth`](prism_render_architecture::water::pressure_multigrid::smooth).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterMgSmoothResult {
    /// Smoothed field, row-major, host-padded to `MAX_N_CUBED` entries. Only the
    /// first `n^3` entries are meaningful for a grid of per-axis size `n`.
    pub p: [f32; MAX_N_CUBED],
}

/// Encodes one [`WaterMgSmoothQuery`] into its `std430` [`GpuQuery`] slot. The
/// initial field `p0` is uploaded separately into the ping-pong buffers.
fn encode_query(q: &WaterMgSmoothQuery) -> GpuQuery {
    GpuQuery {
        n: q.n,
        iters: q.iters,
        h: q.h,
        omega: q.omega,
        b: q.b,
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

/// A compiled, reusable damped-`Jacobi` smoother compute pipeline, twinning the
/// `CPU` golden `smooth` from
/// [`pressure_multigrid`](prism_render_architecture::water::pressure_multigrid).
pub struct GpuWaterMgSmooth {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterMgSmooth {
    /// Compiles the smoother kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterMgSmooth {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_mg_smooth"),
            source: ShaderSource::Wgsl(WATER_MG_SMOOTH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_mg_smooth_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_mg_smooth_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_mg_smooth_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sweep"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterMgSmooth {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one [`WaterMgSmoothResult`]
    /// per input, in order.
    ///
    /// The host issues one dispatch per sweep up to the batch maximum `iters`,
    /// ping-ponging two field buffers; each query freezes once its own `iters`
    /// sweeps are done. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterMgSmoothQuery],
    ) -> Vec<WaterMgSmoothResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_mg_smooth_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        // Concatenated initial fields, one MAX_N_CUBED block per query.
        let mut p_init: Vec<f32> = Vec::with_capacity(count * MAX_N_CUBED);
        for q in queries {
            p_init.extend_from_slice(&q.p0);
        }
        let field_bytes = (count * MAX_N_CUBED * size_of::<f32>()) as u64;
        let mut p_in = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_mg_smooth_field_a"),
            contents: bytemuck::cast_slice(&p_init),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let mut p_out = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_mg_smooth_field_b"),
            contents: bytemuck::cast_slice(&p_init),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let max_iters = queries.iter().map(|q| q.iters).max().unwrap_or(0);
        let threads = (count * MAX_N_CUBED) as u32;
        let groups = threads.div_ceil(WORKGROUP_SIZE);

        for sweep in 0..max_iters {
            let params = GpuParams {
                count: count as u32,
                sweep,
                pad0: 0,
                pad1: 0,
            };
            let params_buf = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("prism_volumetric_water_mg_smooth_params"),
                contents: bytemuck::bytes_of(&params),
                usage: BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("prism_volumetric_water_mg_smooth_bind_group"),
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
                        resource: p_in.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: p_out.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("prism_volumetric_water_mg_smooth_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("prism_volumetric_water_mg_smooth_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                // One thread per cell across all queries, flattened to a 1-D
                // dispatch.
                pass.dispatch_workgroups(groups, 1, 1);
            }
            ctx.queue().submit([encoder.finish()]);
            // The new iterate now lives in p_out; swap so the next sweep reads
            // it, mirroring the golden in-place update.
            core::mem::swap(&mut p_in, &mut p_out);
        }

        // After the final swap, the current iterate is in p_in.
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_mg_smooth_stage"),
            size: field_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_mg_smooth_readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(&p_in, 0, &stage, 0, field_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        flat.chunks_exact(MAX_N_CUBED)
            .map(|chunk| {
                let mut p = [0.0f32; MAX_N_CUBED];
                p.copy_from_slice(chunk);
                WaterMgSmoothResult { p }
            })
            .collect()
    }
}
