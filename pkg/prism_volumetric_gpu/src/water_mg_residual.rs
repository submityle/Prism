//! `wgpu` compute twin of the geometric-`multigrid` level operators
//! [`apply_operator`](prism_render_architecture::water::pressure_multigrid::apply_operator)
//! and
//! [`residual`](prism_render_architecture::water::pressure_multigrid::residual)
//! from the `FLIP`/`APIC` pressure-projection solver.
//!
//! Removing divergence from a liquid velocity field means solving the Poisson
//! system `A p = b`, where `A` is the discrete negative `Laplacian` on a
//! vertex-centred grid with a `Dirichlet` `p = 0` boundary layer. The two
//! stateless primitives at the heart of every V-cycle are the operator apply
//! `A p` and the residual `r = b - A p`: both run the `7`-point stencil over the
//! interior nodes and leave the boundary layer at `0`. They are pure integer
//! index arithmetic plus `+ - * /` on `f32`, so they port to the device with no
//! floating-point transcendental.
//!
//! [`GpuWaterMgResidual`] is the on-device twin of
//! [`residual`](prism_render_architecture::water::pressure_multigrid::residual)
//! (which internally applies the same operator as
//! [`apply_operator`](prism_render_architecture::water::pressure_multigrid::apply_operator)).
//! One thread owns one grid cell of one query, reproducing the interior
//! `7`-point stencil and the zeroed boundary layer, so a passing real-device
//! parity test is direct evidence the ported kernel computes the same residual
//! field the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the residual field `r = b - A p`:
//!
//! * Interior nodes (`x`, `y`, `z` each in `1 .. n - 1`) evaluate
//!   `r[c] = b[c] - (6 p[c] - p[x-1] - p[x+1] - p[y-1] - p[y+1] - p[z-1] - p[z+1]) / (h * h)`,
//!   matching
//!   [`apply_operator`](prism_render_architecture::water::pressure_multigrid::apply_operator)
//!   followed by `b - A p`.
//! * Boundary-layer nodes and nodes past the active `n^3` grid stay `0`,
//!   matching the reference `Dirichlet` layer.
//! * A degenerate `n < 3` yields an all-zero field, matching the reference
//!   guard.
//!
//! # What stays on the host
//!
//! The full V-cycle schedule — the damped-`Jacobi` smoother, full-weighting
//! restriction, trilinear prolongation, the coarse solve, the mean removal, the
//! residual-norm stop test and the multi-level recursion — stays on the host.
//! This twin models only the single-level residual field, with the host owning
//! the variable-length grid hierarchy. Each query carries one grid, zero-padded
//! to the fixed cap `MAX_N`, and the host dispatches by the active `n`.
//!
//! # Correctness model
//!
//! The host and the device evaluate the identical integer index arithmetic and
//! the identical stencil, so the residual field matches to within
//! floating-point tolerance. The one branch decision — interior versus boundary
//! — is a pure integer comparison on the decoded `(x, y, z)`, so there is no
//! floating-point crossing to flip.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned integer
//! index math and `+ - * /` on `f32` — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, no inverse trigonometry, no `sqrt` and no `u64`. No optional device
//! feature is required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
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

/// Fixed per-axis node-count cap. The host zero-pads each query's grid to this
/// size and dispatches by the active `n`; the reference level sizes used by the
/// V-cycle (`2^L + 1` for small `L`) all fit within it.
pub const MAX_N: usize = 9;

/// Fixed per-query cell cap `MAX_N^3`, the length of the padded `p`, `b` and `r`
/// arrays.
const MAX_CELLS: usize = MAX_N * MAX_N * MAX_N;

/// The portable core-`WGSL` single-level residual kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`residual`](prism_render_architecture::water::pressure_multigrid::residual)
/// over the operator
/// [`apply_operator`](prism_render_architecture::water::pressure_multigrid::apply_operator);
/// see the module documentation for the algorithm.
const WATER_MG_RESIDUAL_WGSL: &str = r#"
// Single-level multigrid residual twin: one thread owns one grid cell of one
// query. Interior nodes (x,y,z in 1..n-1) write r = b - A*p with the 7-point
// negative-Laplacian stencil A*p = (6*p_c - 6 neighbors)/h^2; boundary-layer
// nodes, nodes past the active n^3 grid, and a degenerate n<3 all stay 0,
// mirroring the CPU golden `water::pressure_multigrid::residual` over
// `apply_operator`. It owns no smoother, restriction, prolongation, coarse
// solve or V-cycle recursion; those stay on the host. Only unsigned integer
// index math and + - * / on f32 are used — no transcendental, no u64.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::pressure_multigrid；
// 无第三方引擎源码或衍生代码。

// MAX_N^3 = 9^3; the fixed padded cell count of each query's p/b/r arrays.
const CELLS: u32 = 729u;

struct Params {
    // Number of queries in the storage arrays; threads past count*CELLS exit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Zero-padded pressure field, flat (z*n+y)*n+x within the active n^3.
    p: array<f32, 729>,
    // Zero-padded right-hand side, same layout as p.
    b: array<f32, 729>,
    // Active per-axis node count (host guarantees 3 <= n <= MAX_N on a solve).
    n: u32,
    // Grid spacing; the operator scale is 1/(h*h).
    h: f32,
}

struct Result {
    // Residual field r = b - A*p, boundary layer and padding zeroed.
    r: array<f32, 729>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total = params.count * CELLS;
    if (tid >= total) {
        return;
    }
    let qi = tid / CELLS;
    let cell = tid % CELLS;
    let n = queries[qi].n;
    let h = queries[qi].h;

    // Default every cell to 0: the Dirichlet boundary layer, the padding past
    // the active grid, and the whole field for a degenerate n all stay 0.
    results[qi].r[cell] = 0.0;
    if (n < 3u) {
        return;
    }
    let ncells = n * n * n;
    if (cell >= ncells) {
        return;
    }

    // Decode the flat cell index back into (x, y, z); the inverse of
    // (z*n+y)*n+x.
    let x = cell % n;
    let plane = cell / n;
    let y = plane % n;
    let z = plane / n;

    // Boundary-layer nodes are fixed Dirichlet values, not unknowns.
    if (x == 0u || y == 0u || z == 0u) {
        return;
    }
    if (x + 1u >= n || y + 1u >= n || z + 1u >= n) {
        return;
    }

    let inv_h2 = 1.0 / (h * h);
    let s = 6.0 * queries[qi].p[cell]
        - queries[qi].p[(z * n + y) * n + (x - 1u)]
        - queries[qi].p[(z * n + y) * n + (x + 1u)]
        - queries[qi].p[(z * n + (y - 1u)) * n + x]
        - queries[qi].p[(z * n + (y + 1u)) * n + x]
        - queries[qi].p[((z - 1u) * n + y) * n + x]
        - queries[qi].p[((z + 1u) * n + y) * n + x];
    results[qi].r[cell] = queries[qi].b[cell] - s * inv_h2;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_MG_RESIDUAL_WGSL`].
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
/// the padded pressure and right-hand-side fields, the active node count and the
/// grid spacing.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Zero-padded pressure field (the golden `p`).
    p: [f32; MAX_CELLS],
    /// Zero-padded right-hand side (the golden `b`).
    b: [f32; MAX_CELLS],
    /// Active per-axis node count (the golden `n`).
    n: u32,
    /// Grid spacing (the golden `h`).
    h: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// the padded residual field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Zero-padded residual field `r = b - A p` (the golden return value).
    r: [f32; MAX_CELLS],
}

/// One residual query: the padded pressure and right-hand-side fields, the
/// active node count and the grid spacing, mirroring the arguments the golden
/// [`residual`](prism_render_architecture::water::pressure_multigrid::residual)
/// reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterMgResidualQuery {
    /// Zero-padded pressure field, flat `(z * n + y) * n + x` within the active
    /// `n^3` (the golden `p`).
    pub p: [f32; MAX_CELLS],
    /// Zero-padded right-hand side, same layout as `p` (the golden `b`).
    pub b: [f32; MAX_CELLS],
    /// Active per-axis node count; the host guarantees `3 <= n <= MAX_N` on a
    /// solve (the golden `n`).
    pub n: u32,
    /// Grid spacing; the operator scale is `1 / (h * h)` (the golden `h`).
    pub h: f32,
}

impl WaterMgResidualQuery {
    /// Builds a query from the padded pressure and right-hand-side fields, the
    /// active node count and the grid spacing.
    #[must_use]
    pub const fn new(
        p: [f32; MAX_CELLS],
        b: [f32; MAX_CELLS],
        n: u32,
        h: f32,
    ) -> WaterMgResidualQuery {
        WaterMgResidualQuery { p, b, n, h }
    }
}

/// One resolved residual response: the padded residual field, with the
/// boundary layer and the padding past the active `n^3` grid zeroed, mirroring
/// the golden
/// [`residual`](prism_render_architecture::water::pressure_multigrid::residual)
/// return value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterMgResidualResult {
    /// Residual field `r = b - A p`, flat `(z * n + y) * n + x`, boundary layer
    /// and padding zeroed (the golden return value).
    pub r: [f32; MAX_CELLS],
}

/// Encodes one [`WaterMgResidualQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterMgResidualQuery) -> GpuQuery {
    GpuQuery {
        p: q.p,
        b: q.b,
        n: q.n,
        h: q.h,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterMgResidualResult`].
fn decode_result(raw: &GpuResult) -> WaterMgResidualResult {
    WaterMgResidualResult { r: raw.r }
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

/// A compiled, reusable single-level residual compute pipeline, twinning the
/// stateless primitive of the `CPU` golden
/// [`residual`](prism_render_architecture::water::pressure_multigrid::residual).
pub struct GpuWaterMgResidual {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterMgResidual {
    /// Compiles the single-level residual kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterMgResidual {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_mg_residual"),
            source: ShaderSource::Wgsl(WATER_MG_RESIDUAL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_mg_residual_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_mg_residual_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_mg_residual_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterMgResidual {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves the residual field of every query in `queries` and returns one
    /// [`WaterMgResidualResult`] per input, in order.
    ///
    /// The residual field matches the reference to within floating-point
    /// tolerance. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterMgResidualQuery],
    ) -> Vec<WaterMgResidualResult> {
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
            label: Some("prism_volumetric_water_mg_residual_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_mg_residual_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_mg_residual_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_mg_residual_bind_group"),
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
            label: Some("prism_volumetric_water_mg_residual_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_mg_residual_encoder"),
        });
        {
            // One thread per grid cell of each query, flattened to a 1-D
            // dispatch of count * MAX_N^3 threads.
            let total = (count as u32) * (MAX_CELLS as u32);
            let groups = total.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_mg_residual_pass"),
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

        raw.iter().map(decode_result).collect()
    }
}
