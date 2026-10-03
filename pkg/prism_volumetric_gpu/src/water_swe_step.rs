//! `wgpu` compute twin of the shallow-water explicit time step
//! [`step`](prism_render_architecture::water::swe::step) from the regular-grid
//! shallow-water surface solver.
//!
//! One `step` advances the whole depth-velocity field by `dt` on a row-major
//! grid (`idx = z * nx + x`). Continuity is integrated in conservative flux
//! form: each interior face carries the averaged momentum `0.5 * (hu + hu_n)`
//! of its two cells and reflective (no-flux) walls zero the boundary faces, so
//! the summed depth change telescopes to zero and total volume is preserved to
//! rounding. Momentum uses a central pressure gradient, first-order upwind
//! self-advection, and linear damping `(1 - damping * dt).max(0)`. The whole
//! update is multiply/add on `f32` with ordered-comparison branch selects, so
//! it ports to the device with no floating-point transcendental.
//!
//! [`GpuWaterSweStep`] is the on-device twin of
//! [`step`](prism_render_architecture::water::swe::step). One thread owns one
//! grid cell, reading the shared depth and velocity buffers plus its reflective
//! neighbours and recomputing the per-face momentum `hu = h * u`, `hv = h * v`
//! in place (the flux stencil only needs the cell and its direct neighbours, so
//! no separate momentum pass is required and the result is bit-identical to the
//! reference's precomputed arrays). A passing real-device parity test is direct
//! evidence the ported kernel computes the same next state the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the next `SweState`:
//!
//! * The degenerate guard: an empty grid (`nx * nz == 0`) or depth/velocity
//!   lengths shorter than the cell count return the input unchanged.
//! * Conservative continuity with zeroed reflective boundary faces:
//!   `new_h = h - dt * inv_dx * ((fx_right - fx_left) + (fz_up - fz_down))`.
//! * Central depth gradients `dhdx`, `dhdz` from reflective neighbours.
//! * First-order upwind self-advection `adv_u`, `adv_v` chosen by the sign of
//!   the local velocity, matching the reference `upwind` helper exactly.
//! * Damped momentum `un = (u - dt * (adv_u + g * dhdx)) * damp` and the `v`
//!   analogue.
//!
//! # What stays on the host
//!
//! The variable-length depth, `x`-velocity and `z`-velocity vectors, the solver
//! schedule across sub-steps, the volume-changing source injection and the
//! `CFL` time-step budget stay on the host. This twin models only the single
//! stateless field update, with each query carrying one grid zero-padded to the
//! fixed cap `MAX_CELLS` and the host dispatching by the active cell count.
//!
//! # Correctness model
//!
//! The host and the device evaluate the identical row-major index arithmetic,
//! the identical boundary selects and the identical upwind sign choice, so the
//! next state matches to within floating-point tolerance over the longer
//! multiply/add chain. The branch decisions — interior versus boundary face and
//! the upwind direction — are ordered comparisons on `f32` that the random
//! sweep keeps away from their crossings, so no branch flips between host and
//! device.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned integer
//! index math, `+ - * /`, `min`, `max` and `select` on `f32` — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, no inverse trigonometry, no `sqrt` and no `u64`.
//! No optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::swe`；无第三方引擎源码或衍生代码。
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

/// Fixed per-query cell cap. The host zero-pads each query's grid to this length
/// and dispatches by the active cell count; a cap of `4096` covers a `64 x 64`
/// shallow-water grid.
pub const MAX_CELLS: usize = 4096;

/// The portable core-`WGSL` shallow-water step kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden [`step`](prism_render_architecture::water::swe::step); see
/// the module documentation for the algorithm.
const WATER_SWE_STEP_WGSL: &str = r#"
// Shallow-water explicit-step twin: one thread owns one grid cell of one query.
// It reproduces the degenerate guard, conservative continuity with zeroed
// reflective boundary faces, central depth gradients, first-order upwind
// self-advection and linear damping, mirroring the CPU golden
// `water::swe::step`. Per-face momentum hu=h*u, hv=h*v is recomputed in place
// from the cell and its direct neighbours (bit-identical to the reference's
// precomputed arrays). It owns no sub-step schedule, source injection or CFL
// budget; those stay on the host. Only unsigned integer index math and
// + - * /, min, max, select on f32 are used — no transcendental, no u64.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::swe；
// 无第三方引擎源码或衍生代码。

// MAX_CELLS: the fixed padded cell count of each query's h/u/v arrays.
const CELLS: u32 = 4096u;

struct Params {
    // Number of queries in the storage arrays; threads past count*CELLS exit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Zero-padded depth, x-velocity and z-velocity, row-major z*nx+x.
    h: array<f32, 4096>,
    u: array<f32, 4096>,
    v: array<f32, 4096>,
    // Grid shape and logical array lengths for the degenerate guard.
    nx: u32,
    nz: u32,
    h_len: u32,
    u_len: u32,
    v_len: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Cell size, gravity, linear damping and the time step.
    dx: f32,
    gravity: f32,
    damping: f32,
    dt: f32,
}

struct Result {
    // Next-state depth, x-velocity and z-velocity.
    h: array<f32, 4096>,
    u: array<f32, 4096>,
    v: array<f32, 4096>,
    // Active cell count nx*nz (informational).
    valid_n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// First-order upwind advective derivative u*dF/dx + v*dF/dz; the gradient is
// taken from the upwind neighbour (the direction the flow comes from).
fn upwind(
    flow_u: f32,
    flow_v: f32,
    center: f32,
    xm: f32,
    xp: f32,
    zm: f32,
    zp: f32,
    inv_dx: f32,
) -> f32 {
    let ddx = select((xp - center) * inv_dx, (center - xm) * inv_dx, flow_u > 0.0);
    let ddz = select((zp - center) * inv_dx, (center - zm) * inv_dx, flow_v > 0.0);
    return flow_u * ddx + flow_v * ddz;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total = params.count * CELLS;
    if (tid >= total) {
        return;
    }
    let qi = tid / CELLS;
    let cell = tid % CELLS;

    let nx = queries[qi].nx;
    let nz = queries[qi].nz;
    let n = nx * nz;

    // Degenerate guard: an empty grid or arrays shorter than the cell count
    // return the input state unchanged.
    let degenerate = (n == 0u)
        || (queries[qi].h_len < n)
        || (queries[qi].u_len < n)
        || (queries[qi].v_len < n);
    results[qi].valid_n = n;
    if (degenerate || cell >= n) {
        results[qi].h[cell] = queries[qi].h[cell];
        results[qi].u[cell] = queries[qi].u[cell];
        results[qi].v[cell] = queries[qi].v[cell];
        return;
    }

    let dx = queries[qi].dx;
    let gravity = queries[qi].gravity;
    let dt = queries[qi].dt;
    let inv_dx = 1.0 / dx;
    let damp = max(1.0 - queries[qi].damping * dt, 0.0);

    let idx = cell;
    let x = cell % nx;
    let z = cell / nx;

    // Reflective neighbour indices: clamp to self at the boundary so an
    // out-of-support read never leaves the active grid.
    let has_xp = x + 1u < nx;
    let has_xm = x >= 1u;
    let has_zp = z + 1u < nz;
    let has_zm = z >= 1u;
    let ixp = select(idx, idx + 1u, has_xp);
    let ixm = select(idx, idx - 1u, has_xm);
    let izp = select(idx, idx + nx, has_zp);
    let izm = select(idx, idx - nx, has_zm);

    // Per-face momentum, recomputed in place from h*u, h*v.
    let hu_c = queries[qi].h[idx] * queries[qi].u[idx];
    let hu_xp = queries[qi].h[ixp] * queries[qi].u[ixp];
    let hu_xm = queries[qi].h[ixm] * queries[qi].u[ixm];
    let hv_c = queries[qi].h[idx] * queries[qi].v[idx];
    let hv_zp = queries[qi].h[izp] * queries[qi].v[izp];
    let hv_zm = queries[qi].h[izm] * queries[qi].v[izm];

    // Conservative continuity: boundary faces carry no flux.
    let fx_right = select(0.0, 0.5 * (hu_c + hu_xp), has_xp);
    let fx_left = select(0.0, 0.5 * (hu_xm + hu_c), has_xm);
    let fz_up = select(0.0, 0.5 * (hv_c + hv_zp), has_zp);
    let fz_down = select(0.0, 0.5 * (hv_zm + hv_c), has_zm);
    let new_h = queries[qi].h[idx]
        - dt * inv_dx * ((fx_right - fx_left) + (fz_up - fz_down));

    // Reflective neighbour samples for the central pressure gradient.
    let h_xp = queries[qi].h[ixp];
    let h_xm = queries[qi].h[ixm];
    let h_zp = queries[qi].h[izp];
    let h_zm = queries[qi].h[izm];
    let dhdx = (h_xp - h_xm) * 0.5 * inv_dx;
    let dhdz = (h_zp - h_zm) * 0.5 * inv_dx;

    // First-order upwind self-advection of the velocity field.
    let u_xp = queries[qi].u[ixp];
    let u_xm = queries[qi].u[ixm];
    let u_zp = queries[qi].u[izp];
    let u_zm = queries[qi].u[izm];
    let v_xp = queries[qi].v[ixp];
    let v_xm = queries[qi].v[ixm];
    let v_zp = queries[qi].v[izp];
    let v_zm = queries[qi].v[izm];

    let cu = queries[qi].u[idx];
    let cv = queries[qi].v[idx];
    let adv_u = upwind(cu, cv, cu, u_xm, u_xp, u_zm, u_zp, inv_dx);
    let adv_v = upwind(cu, cv, cv, v_xm, v_xp, v_zm, v_zp, inv_dx);

    let un = (cu - dt * (adv_u + gravity * dhdx)) * damp;
    let vn = (cv - dt * (adv_v + gravity * dhdz)) * damp;

    results[qi].h[cell] = new_h;
    results[qi].u[cell] = un;
    results[qi].v[cell] = vn;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_SWE_STEP_WGSL`].
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
/// the padded field arrays, the grid shape and lengths, and the step scalars.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Zero-padded depth field (the golden `state.h`).
    h: [f32; MAX_CELLS],
    /// Zero-padded `x`-velocity field (the golden `state.u`).
    u: [f32; MAX_CELLS],
    /// Zero-padded `z`-velocity field (the golden `state.v`).
    v: [f32; MAX_CELLS],
    /// Cell count along `x` (the golden `cfg.nx`).
    nx: u32,
    /// Cell count along `z` (the golden `cfg.nz`).
    nz: u32,
    /// Logical length of the depth array for the degenerate guard.
    h_len: u32,
    /// Logical length of the `x`-velocity array for the degenerate guard.
    u_len: u32,
    /// Logical length of the `z`-velocity array for the degenerate guard.
    v_len: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Cell size (the golden `cfg.dx`).
    dx: f32,
    /// Gravitational acceleration (the golden `cfg.gravity`).
    gravity: f32,
    /// Linear velocity damping (the golden `cfg.damping`).
    damping: f32,
    /// Time step (the golden `dt`).
    dt: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// the next-state field arrays and the active cell count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Next-state depth field (the golden return `h`).
    h: [f32; MAX_CELLS],
    /// Next-state `x`-velocity field (the golden return `u`).
    u: [f32; MAX_CELLS],
    /// Next-state `z`-velocity field (the golden return `v`).
    v: [f32; MAX_CELLS],
    /// Active cell count `nx * nz`.
    valid_n: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One step query: the padded depth and velocity fields, the grid shape and
/// lengths, and the step scalars, mirroring the arguments the golden
/// [`step`](prism_render_architecture::water::swe::step) reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSweStepQuery {
    /// Zero-padded depth field, row-major `z * nx + x` (the golden `state.h`).
    pub h: [f32; MAX_CELLS],
    /// Zero-padded `x`-velocity field, same layout (the golden `state.u`).
    pub u: [f32; MAX_CELLS],
    /// Zero-padded `z`-velocity field, same layout (the golden `state.v`).
    pub v: [f32; MAX_CELLS],
    /// Cell count along `x` (the golden `cfg.nx`).
    pub nx: u32,
    /// Cell count along `z` (the golden `cfg.nz`).
    pub nz: u32,
    /// Logical length of the depth array; the degenerate guard fires when it is
    /// below `nx * nz` (the length of the golden `state.h`).
    pub h_len: u32,
    /// Logical length of the `x`-velocity array (the length of the golden
    /// `state.u`).
    pub u_len: u32,
    /// Logical length of the `z`-velocity array (the length of the golden
    /// `state.v`).
    pub v_len: u32,
    /// Cell size (the golden `cfg.dx`).
    pub dx: f32,
    /// Gravitational acceleration (the golden `cfg.gravity`).
    pub gravity: f32,
    /// Linear velocity damping (the golden `cfg.damping`).
    pub damping: f32,
    /// Time step (the golden `dt`).
    pub dt: f32,
}

impl WaterSweStepQuery {
    /// Builds a query from the padded field arrays, the grid shape and lengths,
    /// and the step scalars.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the golden step's flat state-plus-config argument list"
    )]
    pub const fn new(
        h: [f32; MAX_CELLS],
        u: [f32; MAX_CELLS],
        v: [f32; MAX_CELLS],
        nx: u32,
        nz: u32,
        h_len: u32,
        u_len: u32,
        v_len: u32,
        dx: f32,
        gravity: f32,
        damping: f32,
        dt: f32,
    ) -> WaterSweStepQuery {
        WaterSweStepQuery {
            h,
            u,
            v,
            nx,
            nz,
            h_len,
            u_len,
            v_len,
            dx,
            gravity,
            damping,
            dt,
        }
    }
}

/// One resolved step response: the next-state field arrays and the active cell
/// count, mirroring the golden
/// [`step`](prism_render_architecture::water::swe::step) return value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSweStepResult {
    /// Next-state depth field (the golden return `h`).
    pub h: [f32; MAX_CELLS],
    /// Next-state `x`-velocity field (the golden return `u`).
    pub u: [f32; MAX_CELLS],
    /// Next-state `z`-velocity field (the golden return `v`).
    pub v: [f32; MAX_CELLS],
    /// Active cell count `nx * nz`.
    pub valid_n: u32,
}

/// Encodes one [`WaterSweStepQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterSweStepQuery) -> GpuQuery {
    GpuQuery {
        h: q.h,
        u: q.u,
        v: q.v,
        nx: q.nx,
        nz: q.nz,
        h_len: q.h_len,
        u_len: q.u_len,
        v_len: q.v_len,
        pad0: 0,
        pad1: 0,
        pad2: 0,
        dx: q.dx,
        gravity: q.gravity,
        damping: q.damping,
        dt: q.dt,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterSweStepResult`].
fn decode_result(raw: &GpuResult) -> WaterSweStepResult {
    WaterSweStepResult {
        h: raw.h,
        u: raw.u,
        v: raw.v,
        valid_n: raw.valid_n,
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

/// A compiled, reusable shallow-water step compute pipeline, twinning the
/// stateless field update of the `CPU` golden
/// [`step`](prism_render_architecture::water::swe::step).
pub struct GpuWaterSweStep {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterSweStep {
    /// Compiles the shallow-water step kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterSweStep {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_swe_step"),
            source: ShaderSource::Wgsl(WATER_SWE_STEP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_swe_step_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_swe_step_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_swe_step_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterSweStep {
            module,
            layout,
            pipeline,
        }
    }

    /// Advances every query in `queries` by one step and returns one
    /// [`WaterSweStepResult`] per input, in order.
    ///
    /// The next state matches the reference to within floating-point tolerance;
    /// a degenerate query returns the input unchanged. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterSweStepQuery],
    ) -> Vec<WaterSweStepResult> {
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
            label: Some("prism_volumetric_water_swe_step_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_swe_step_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_swe_step_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_swe_step_bind_group"),
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
            label: Some("prism_volumetric_water_swe_step_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_swe_step_encoder"),
        });
        {
            // One thread per grid cell of each query, flattened to a 1-D
            // dispatch of count * MAX_CELLS threads.
            let total = (count as u32) * (MAX_CELLS as u32);
            let groups = total.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_swe_step_pass"),
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
