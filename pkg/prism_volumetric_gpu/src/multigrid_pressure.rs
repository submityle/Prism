//! `wgpu` compute twin of the geometric-multigrid pressure-projection
//! primitives
//! ([`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure),
//! design section 10 and open question 3), exposed one pure-numeric per-cell
//! routine at a time.
//!
//! The golden module is a full `V`-/`W`-cycle solver: a recursive schedule over
//! a pyramid of [`GridResolution`](prism_render_architecture::particle::fluid)
//! levels that pre-smooths, restricts a residual, recurses, prolongs a
//! correction and post-smooths. That orchestration is inherently host work —
//! `WGSL` has no recursion, no dynamic neighbour indexing and no variable-length
//! arrays — so this twin takes the complementary *unit-test* view: a single
//! compute kernel with an **operation-code dispatch** evaluates exactly one
//! golden per-cell routine per lane, over a fixed-length neighbourhood the host
//! pre-samples. A parity test can then pin each building block in isolation on a
//! real device.
//!
//! # Why op-code dispatch
//!
//! Each twinned routine is a small, independently testable numeric kernel.
//! Rather than compile one pipeline per routine, every
//! [`MultigridPressureQuery`] carries an operation code, and the single `solve`
//! kernel branches on it (an `if` / `else if` ladder on an unsigned code, an
//! exact integer compare). One thread handles one query; the batch may freely
//! mix operations. This keeps a single shader module and bind-group layout while
//! still surfacing every primitive to the parity suite.
//!
//! # What is twinned
//!
//! The pure-numeric per-cell routines, each fed a host-presampled fixed-length
//! neighbourhood because `WGSL` cannot gather a variable neighbour set:
//!
//! * `divergence_forward` for one cell — the forward-difference velocity
//!   divergence `(vx − here.x) + (vy − here.y) + (vz − here.z)`, with an
//!   out-of-grid forward face contributing zero.
//! * `gradient_backward` for one cell — the wall-aware backward-difference
//!   pressure gradient, dropped on the low face of each axis.
//! * `jacobi_smooth` for one cell, one sweep — the weighted-`Jacobi` relaxation
//!   `(1 − ω)·center + ω·(Σ neighbours − rhs / inv_h2) / diagonal`.
//! * `level_residual` for one cell — `rhs − inv_h2·(Σ neighbours − diagonal·center)`.
//! * `diagonal` — the stencil diagonal under a wall model and a live-neighbour
//!   count.
//! * `axis_contributors` — the cell-centred `(3/4, 1/4)` prolongation stencil
//!   for one fine index, mirrored at a grid edge.
//! * `prolong` for one fine cell — the tensor-product trilinear fold of eight
//!   host-gathered coarse contributors.
//! * the restriction scale `1 / 2^k` for `k` coarsened axes (the golden
//!   `pow2_f32` reciprocal).
//!
//! # Left on the host (not twinned)
//!
//! The `V`-cycle schedule and recursion
//! (`v_cycle`, `multigrid_pressure_solve`), the
//! [`GridResolution`](prism_render_architecture::particle::fluid) linear
//! indexing and variable-length `Vec` traversal, the full-field `L2` residual
//! norm reduction (`level_residual_l2`), the boundary-condition scan,
//! `remove_mean`, `solve_coarsest` and the whole
//! `project_velocity_field` write-back all stay on the host: they are reductions
//! and dynamic-length gathers with no fixed-size on-device counterpart. This
//! twin consumes the fixed six-neighbour (plus centre) or eight-coarse-contributor
//! windows the host pre-samples.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `+ - * /`, bit operations and integer/`f32` value conversions, with one
//! guarded reciprocal — and no `exp`, `pow`, `sin` / `cos` or any optional
//! device feature, so it runs unmodified on Metal, Vulkan and DX12. The golden
//! `pow2_f32` is reproduced as the same integer-doubling loop; no transcendental
//! appears.
//!
//! # Correctness model
//!
//! Every twinned routine is fixed closed-form algebra, so `CPU` and `GPU`
//! evaluate the same expression. They are not bit-exact in general: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts an absolute-or-relative tolerance on continuous values and an *exact*
//! equality on the discrete diagonal-count, stencil-index and stencil-length
//! codes.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::multigrid_pressure`；
//! standard geometric-multigrid pressure-projection primitives (forward
//! divergence, wall-aware backward gradient, weighted-`Jacobi` relaxation,
//! 7-point `Laplacian` residual, full-weighting restriction and trilinear
//! prolongation stencils) plus `wgpu` compute dispatch; no third-party engine
//! source or derived code.
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::multigrid_pressure::PressureBoundary;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// The number of axis-aligned face neighbours a voxel has in a 3-D grid; the
/// full `Dirichlet` diagonal of the 7-point `Laplacian`, matching the golden
/// `FACE_NEIGHBOR_COUNT`.
const FACE_NEIGHBOR_COUNT: f32 = 6.0;

// Operation codes shared by the host encoder and the `solve` kernel. Each tags
// one golden per-cell routine; the kernel branches on the code with an exact
// integer compare.
const OP_DIVERGENCE: u32 = 0;
const OP_GRADIENT: u32 = 1;
const OP_JACOBI: u32 = 2;
const OP_RESIDUAL: u32 = 3;
const OP_DIAGONAL: u32 = 4;
const OP_AXIS: u32 = 5;
const OP_PROLONG: u32 = 6;
const OP_RESTRICT_SCALE: u32 = 7;

/// Wall-model code for [`PressureBoundary::Dirichlet`] (full six-face diagonal).
const BOUNDARY_DIRICHLET: u32 = 0;
/// Wall-model code for [`PressureBoundary::Neumann`] (live-neighbour diagonal).
const BOUNDARY_NEUMANN: u32 = 1;

/// The portable core-`WGSL` multigrid pressure per-cell primitive kernel,
/// embedded inline so the twin ships as a single source file. One thread
/// evaluates one query, branching on its operation code; see the module
/// documentation for the op-dispatch rationale.
const MULTIGRID_PRESSURE_WGSL: &str = r#"
// Multigrid pressure per-cell primitive twin: one thread per query evaluates a
// single golden routine selected by an operation code, over a fixed-length
// neighbourhood the host pre-samples. It mirrors the CPU golden
// `particle::multigrid_pressure`, uses only the portable core-WGSL subset
// (min/max/abs and + - * / plus one guarded reciprocal and integer/f32 value
// conversions) and takes no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: standard geometric-multigrid pressure-projection primitives; no
// third-party engine source or derived code.

struct Params {
    // Number of valid lanes in the batch, one thread each.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query's packed inputs. 160-byte std430 stride matching the host GpuQuery.
struct Query {
    // Operation code selecting the golden routine.
    op: u32,
    // Wall-model code (0 Dirichlet, 1 Neumann).
    boundary: u32,
    // Packed in-grid flags: bits 0..5 are the six neighbour faces for the
    // Jacobi/residual stencil, or bits 0..2 the forward/backward faces for the
    // divergence/gradient difference.
    flag_pack: u32,
    // Fine index for the axis-contributor stencil.
    fine_index: u32,
    // Coarse axis length for the axis-contributor stencil.
    coarse_n: u32,
    // Live-neighbour count for the standalone diagonal query.
    live: u32,
    // Number of coarsened axes for the restriction scale (k in 1/2^k).
    coarsened_axes: u32,
    // Prolongation stencil lengths per axis (0..=2).
    len_x: u32,
    len_y: u32,
    len_z: u32,
    ipad0: u32,
    ipad1: u32,
    // center, rhs, inv_h2, omega.
    scalars: vec4<f32>,
    // Divergence centre velocity (x,y,z); gradient centre pressure in x.
    here: vec4<f32>,
    // Divergence forward-face velocities / gradient backward-face pressures.
    forward: vec4<f32>,
    // Six-neighbour pressures 0..3 (+x,-x,+y,-y) or coarse contributors 0..3.
    nb_lo: vec4<f32>,
    // Six-neighbour pressures 4..5 (+z,-z) or coarse contributors 4..7.
    nb_hi: vec4<f32>,
    // Prolongation per-axis weights wx0,wx1,wy0,wy1.
    wx_wy: vec4<f32>,
    // Prolongation per-axis weights wz0,wz1 (two pad lanes).
    wz: vec4<f32>,
}

// One query's packed outputs. 48-byte std430 stride matching the host GpuResult.
struct Res {
    // Primary scalar in x, or the three-vector gradient in xyz.
    v0: vec4<f32>,
    // Axis-contributor indices idx0,idx1, stencil length in z (one pad lane).
    idx: vec4<u32>,
    // Axis-contributor weights w0,w1 (two pad lanes).
    w: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

const OP_DIVERGENCE: u32 = 0u;
const OP_GRADIENT: u32 = 1u;
const OP_JACOBI: u32 = 2u;
const OP_RESIDUAL: u32 = 3u;
const OP_DIAGONAL: u32 = 4u;
const OP_AXIS: u32 = 5u;
const OP_PROLONG: u32 = 6u;
const OP_RESTRICT_SCALE: u32 = 7u;

const BOUNDARY_DIRICHLET: u32 = 0u;

// Full six-face Dirichlet diagonal, matching the golden FACE_NEIGHBOR_COUNT.
const FACE_NEIGHBOR_COUNT: f32 = 6.0;

// Stencil diagonal under a wall model and a live-neighbour count, matching the
// golden `diagonal`: the full six faces for a Dirichlet wall, else the live
// neighbour count (a lone cell keeps a unit diagonal so the sweep is a safe
// no-op rather than a divide by zero).
fn diagonal_value(boundary: u32, live: u32) -> f32 {
    if (boundary == BOUNDARY_DIRICHLET) {
        return FACE_NEIGHBOR_COUNT;
    }
    if (live == 0u) {
        return 1.0;
    }
    return f32(live);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lane = gid.x;
    if (lane >= params.count) {
        return;
    }
    let q = queries[lane];
    var out: Res;
    out.v0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.idx = vec4<u32>(0u, 0u, 0u, 0u);
    out.w = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let op = q.op;

    if (op == OP_DIVERGENCE) {
        // Out-of-grid forward faces contribute zero (masked by the in-grid bit).
        let mx = f32(q.flag_pack & 1u);
        let my = f32((q.flag_pack >> 1u) & 1u);
        let mz = f32((q.flag_pack >> 2u) & 1u);
        let fx = q.forward.x * mx;
        let fy = q.forward.y * my;
        let fz = q.forward.z * mz;
        let d = (fx - q.here.x) + (fy - q.here.y) + (fz - q.here.z);
        out.v0.x = d;
    } else if (op == OP_GRADIENT) {
        // Wall-aware backward difference: dropped on the low face of each axis.
        let mx = f32(q.flag_pack & 1u);
        let my = f32((q.flag_pack >> 1u) & 1u);
        let mz = f32((q.flag_pack >> 2u) & 1u);
        let here = q.scalars.x;
        let gx = (here - q.forward.x) * mx;
        let gy = (here - q.forward.y) * my;
        let gz = (here - q.forward.z) * mz;
        out.v0 = vec4<f32>(gx, gy, gz, 0.0);
    } else if (op == OP_JACOBI) {
        let m0 = q.flag_pack & 1u;
        let m1 = (q.flag_pack >> 1u) & 1u;
        let m2 = (q.flag_pack >> 2u) & 1u;
        let m3 = (q.flag_pack >> 3u) & 1u;
        let m4 = (q.flag_pack >> 4u) & 1u;
        let m5 = (q.flag_pack >> 5u) & 1u;
        let nsum = q.nb_lo.x * f32(m0) + q.nb_lo.y * f32(m1) + q.nb_lo.z * f32(m2)
            + q.nb_lo.w * f32(m3) + q.nb_hi.x * f32(m4) + q.nb_hi.y * f32(m5);
        let live = m0 + m1 + m2 + m3 + m4 + m5;
        let diag = diagonal_value(q.boundary, live);
        let inv_scale = 1.0 / q.scalars.z;
        let relaxed = (nsum - q.scalars.y * inv_scale) / diag;
        let next = (1.0 - q.scalars.w) * q.scalars.x + q.scalars.w * relaxed;
        out.v0.x = next;
    } else if (op == OP_RESIDUAL) {
        let m0 = q.flag_pack & 1u;
        let m1 = (q.flag_pack >> 1u) & 1u;
        let m2 = (q.flag_pack >> 2u) & 1u;
        let m3 = (q.flag_pack >> 3u) & 1u;
        let m4 = (q.flag_pack >> 4u) & 1u;
        let m5 = (q.flag_pack >> 5u) & 1u;
        let nsum = q.nb_lo.x * f32(m0) + q.nb_lo.y * f32(m1) + q.nb_lo.z * f32(m2)
            + q.nb_lo.w * f32(m3) + q.nb_hi.x * f32(m4) + q.nb_hi.y * f32(m5);
        let live = m0 + m1 + m2 + m3 + m4 + m5;
        let diag = diagonal_value(q.boundary, live);
        let laplacian = q.scalars.z * (nsum - diag * q.scalars.x);
        out.v0.x = q.scalars.y - laplacian;
    } else if (op == OP_DIAGONAL) {
        out.v0.x = diagonal_value(q.boundary, q.live);
    } else if (op == OP_AXIS) {
        let fine = q.fine_index;
        let cn = q.coarse_n;
        if (cn == 0u) {
            out.idx = vec4<u32>(0u, 0u, 0u, 0u);
            out.w = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        } else {
            var parent = fine / 2u;
            if (parent >= cn) {
                parent = cn - 1u;
            }
            let far_is_lower = (fine & 1u) == 0u;
            var far_in_range = false;
            if (far_is_lower) {
                far_in_range = parent > 0u;
            } else {
                far_in_range = (parent + 1u) < cn;
            }
            if (far_in_range) {
                var far = parent + 1u;
                if (far_is_lower) {
                    far = parent - 1u;
                }
                out.idx = vec4<u32>(parent, far, 2u, 0u);
                out.w = vec4<f32>(0.75, 0.25, 0.0, 0.0);
            } else {
                // Mirror: the far quarter folds back onto the parent.
                out.idx = vec4<u32>(parent, 0u, 1u, 0u);
                out.w = vec4<f32>(1.0, 0.0, 0.0, 0.0);
            }
        }
    } else if (op == OP_PROLONG) {
        var cv: array<f32, 8>;
        cv[0] = q.nb_lo.x;
        cv[1] = q.nb_lo.y;
        cv[2] = q.nb_lo.z;
        cv[3] = q.nb_lo.w;
        cv[4] = q.nb_hi.x;
        cv[5] = q.nb_hi.y;
        cv[6] = q.nb_hi.z;
        cv[7] = q.nb_hi.w;
        var wx: array<f32, 2>;
        wx[0] = q.wx_wy.x;
        wx[1] = q.wx_wy.y;
        var wy: array<f32, 2>;
        wy[0] = q.wx_wy.z;
        wy[1] = q.wx_wy.w;
        var wz: array<f32, 2>;
        wz[0] = q.wz.x;
        wz[1] = q.wz.y;
        var acc = 0.0;
        for (var iz = 0u; iz < q.len_z; iz = iz + 1u) {
            for (var iy = 0u; iy < q.len_y; iy = iy + 1u) {
                for (var ix = 0u; ix < q.len_x; ix = ix + 1u) {
                    let cidx = iz * 4u + iy * 2u + ix;
                    acc = acc + wx[ix] * wy[iy] * wz[iz] * cv[cidx];
                }
            }
        }
        out.v0.x = acc;
    } else if (op == OP_RESTRICT_SCALE) {
        // 2^k by an integer-doubling loop, matching the golden `pow2_f32`.
        var value = 1.0;
        var remaining = q.coarsened_axes;
        loop {
            if (remaining == 0u) {
                break;
            }
            value = value * 2.0;
            remaining = remaining - 1u;
        }
        out.v0.x = 1.0 / value;
    }

    results[lane] = out;
}
"#;

/// One multigrid pressure per-cell primitive query: a tagged request to evaluate
/// a single golden routine on the device over a host-presampled fixed-length
/// neighbourhood.
///
/// Each variant names one per-cell routine of the `CPU` golden
/// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
/// and carries just that routine's fixed-size operands; the variable-length
/// grid gathers stay on the host. Holds `f32` operands, so it derives only
/// [`Clone`], [`Copy`], [`Debug`] and [`PartialEq`] (no [`Eq`] / [`Hash`]).
/// Provenance: query tagging for the `multigrid_pressure` twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MultigridPressureQuery {
    /// Forward-difference velocity divergence for one cell, mirroring the golden
    /// `divergence_forward` cell body. `Provenance:` `divergence_forward`.
    DivergenceForward {
        /// The centre-cell velocity components `(vx, vy, vz)`.
        here: [f32; 3],
        /// The forward-face velocities `(v[x+1].x, v[y+1].y, v[z+1].z)`.
        forward: [f32; 3],
        /// Whether each forward face lies inside the grid (else it reads zero).
        forward_in_grid: [bool; 3],
    },
    /// Wall-aware backward-difference pressure gradient for one cell, mirroring
    /// the golden `gradient_backward`. `Provenance:` `gradient_backward`.
    GradientBackward {
        /// The centre-cell pressure.
        here: f32,
        /// The backward-face pressures `(p[x-1], p[y-1], p[z-1])`.
        backward: [f32; 3],
        /// Whether each backward face lies inside the grid (else the difference
        /// is dropped to zero).
        backward_in_grid: [bool; 3],
    },
    /// One weighted-`Jacobi` relaxation step for one cell, mirroring the golden
    /// `jacobi_smooth` cell body. `Provenance:` `jacobi_smooth`.
    JacobiCell {
        /// The current centre-cell pressure.
        center: f32,
        /// The six face-neighbour pressures in order `+x, -x, +y, -y, +z, -z`.
        neighbors: [f32; 6],
        /// Whether each face neighbour lies inside the grid.
        neighbor_in_grid: [bool; 6],
        /// The right-hand side at the centre cell.
        rhs: f32,
        /// The level operator scale `1 / h²`.
        inv_h2: f32,
        /// The weighted-`Jacobi` damping factor `ω`.
        omega: f32,
        /// The wall model applied outside the grid.
        boundary: PressureBoundary,
    },
    /// The level residual for one cell, mirroring the golden `level_residual_l2`
    /// per-cell term before the sum. `Provenance:` `level_residual_l2`.
    ResidualCell {
        /// The current centre-cell pressure.
        center: f32,
        /// The six face-neighbour pressures in order `+x, -x, +y, -y, +z, -z`.
        neighbors: [f32; 6],
        /// Whether each face neighbour lies inside the grid.
        neighbor_in_grid: [bool; 6],
        /// The right-hand side at the centre cell.
        rhs: f32,
        /// The level operator scale `1 / h²`.
        inv_h2: f32,
        /// The wall model applied outside the grid.
        boundary: PressureBoundary,
    },
    /// The stencil diagonal under a wall model and a live-neighbour count,
    /// mirroring the golden `diagonal`. `Provenance:` `diagonal`.
    Diagonal {
        /// The wall model applied outside the grid.
        boundary: PressureBoundary,
        /// The count of live (in-grid) face neighbours (`0..=6`).
        live: u32,
    },
    /// The cell-centred prolongation stencil for one fine index on one axis,
    /// mirroring the golden `axis_contributors`. `Provenance:` `axis_contributors`.
    AxisContributors {
        /// The fine-grid index on the axis.
        fine_index: u32,
        /// The coarse-grid axis length.
        coarse_n: u32,
    },
    /// Trilinear prolongation for one fine cell, mirroring the golden `prolong`
    /// inner tensor fold over eight host-gathered coarse contributors.
    /// `Provenance:` `prolong`.
    ProlongCell {
        /// The eight coarse contributors indexed `iz*4 + iy*2 + ix`.
        coarse_values: [f32; 8],
        /// The two `x`-axis interpolation weights.
        weight_x: [f32; 2],
        /// The two `y`-axis interpolation weights.
        weight_y: [f32; 2],
        /// The two `z`-axis interpolation weights.
        weight_z: [f32; 2],
        /// The number of live `x`-axis contributors (`1` or `2`).
        len_x: u32,
        /// The number of live `y`-axis contributors (`1` or `2`).
        len_y: u32,
        /// The number of live `z`-axis contributors (`1` or `2`).
        len_z: u32,
    },
    /// The full-weighting restriction scale `1 / 2^k`, mirroring the golden
    /// `pow2_f32` reciprocal in `restrict`. `Provenance:` `restrict`.
    RestrictScale {
        /// The number of genuinely coarsened axes `k` (`0..=3`).
        coarsened_axes: u32,
    },
}

/// The result of one multigrid pressure per-cell query, mirroring the golden
/// return of the routine the query names.
///
/// Provenance: result tagging for the `multigrid_pressure` twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MultigridPressureResult {
    /// A scalar result (divergence, relaxed pressure, residual, diagonal,
    /// restriction scale).
    Scalar {
        /// The scalar value.
        value: f32,
    },
    /// A three-vector result (the backward-difference gradient).
    Vector {
        /// The `[x, y, z]` components.
        v: [f32; 3],
    },
    /// A one-axis prolongation stencil: up to two coarse indices and weights.
    Axis {
        /// The coarse indices (only the first `len` are meaningful).
        idx: [u32; 2],
        /// The interpolation weights (only the first `len` are meaningful).
        weight: [f32; 2],
        /// The number of live contributors (`0`, `1` or `2`).
        len: u32,
    },
}

/// The `CPU` golden verdict for one query, reproducing the matching per-cell
/// routine of
/// [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
/// in its exact closed form so callers (and the parity test) can pin the twin
/// lane for lane.
///
/// The twinned routines are private helpers of the golden module, so each arm
/// reimplements the published formula rather than importing the symbol; the
/// provenance of each formula is noted on the corresponding
/// [`MultigridPressureQuery`] variant.
///
/// Provenance: `CPU` reference for the `multigrid_pressure` twin, 孪生自本仓
/// `prism_render_architecture::particle::multigrid_pressure`.
#[must_use]
pub fn cpu_reference(query: &MultigridPressureQuery) -> MultigridPressureResult {
    match query {
        MultigridPressureQuery::DivergenceForward {
            here,
            forward,
            forward_in_grid,
        } => {
            let fx = if forward_in_grid[0] { forward[0] } else { 0.0 };
            let fy = if forward_in_grid[1] { forward[1] } else { 0.0 };
            let fz = if forward_in_grid[2] { forward[2] } else { 0.0 };
            let d = (fx - here[0]) + (fy - here[1]) + (fz - here[2]);
            MultigridPressureResult::Scalar { value: d }
        }
        MultigridPressureQuery::GradientBackward {
            here,
            backward,
            backward_in_grid,
        } => {
            let gx = if backward_in_grid[0] {
                here - backward[0]
            } else {
                0.0
            };
            let gy = if backward_in_grid[1] {
                here - backward[1]
            } else {
                0.0
            };
            let gz = if backward_in_grid[2] {
                here - backward[2]
            } else {
                0.0
            };
            MultigridPressureResult::Vector { v: [gx, gy, gz] }
        }
        MultigridPressureQuery::JacobiCell {
            center,
            neighbors,
            neighbor_in_grid,
            rhs,
            inv_h2,
            omega,
            boundary,
        } => {
            let (nsum, live) = neighbor_sum_count(neighbors, neighbor_in_grid);
            let diag = diagonal(*boundary, live);
            let inv_scale = 1.0 / inv_h2;
            let relaxed = (nsum - rhs * inv_scale) / diag;
            let next = (1.0 - omega) * center + omega * relaxed;
            MultigridPressureResult::Scalar { value: next }
        }
        MultigridPressureQuery::ResidualCell {
            center,
            neighbors,
            neighbor_in_grid,
            rhs,
            inv_h2,
            boundary,
        } => {
            let (nsum, live) = neighbor_sum_count(neighbors, neighbor_in_grid);
            let diag = diagonal(*boundary, live);
            let laplacian = inv_h2 * (nsum - diag * center);
            MultigridPressureResult::Scalar {
                value: rhs - laplacian,
            }
        }
        MultigridPressureQuery::Diagonal { boundary, live } => MultigridPressureResult::Scalar {
            value: diagonal(*boundary, *live),
        },
        MultigridPressureQuery::AxisContributors {
            fine_index,
            coarse_n,
        } => {
            let (idx, weight, len) = axis_contributors(*fine_index, *coarse_n);
            MultigridPressureResult::Axis { idx, weight, len }
        }
        MultigridPressureQuery::ProlongCell {
            coarse_values,
            weight_x,
            weight_y,
            weight_z,
            len_x,
            len_y,
            len_z,
        } => {
            let mut acc = 0.0f32;
            let mut iz = 0u32;
            while iz < *len_z {
                let mut iy = 0u32;
                while iy < *len_y {
                    let mut ix = 0u32;
                    while ix < *len_x {
                        let cidx = (iz * 4 + iy * 2 + ix) as usize;
                        acc += weight_x[ix as usize]
                            * weight_y[iy as usize]
                            * weight_z[iz as usize]
                            * coarse_values[cidx];
                        ix += 1;
                    }
                    iy += 1;
                }
                iz += 1;
            }
            MultigridPressureResult::Scalar { value: acc }
        }
        MultigridPressureQuery::RestrictScale { coarsened_axes } => {
            MultigridPressureResult::Scalar {
                value: 1.0 / pow2_f32(*coarsened_axes),
            }
        }
    }
}

/// Sum of the in-grid face-neighbour pressures together with the live-neighbour
/// count, mirroring the golden `neighbor_sum_count` over a fixed six-neighbour
/// window in order `+x, -x, +y, -y, +z, -z`.
fn neighbor_sum_count(neighbors: &[f32; 6], in_grid: &[bool; 6]) -> (f32, u32) {
    let mut sum = 0.0f32;
    let mut count = 0u32;
    let mut k = 0usize;
    while k < 6 {
        if in_grid[k] {
            sum += neighbors[k];
            count += 1;
        }
        k += 1;
    }
    (sum, count)
}

/// The stencil diagonal under a wall model and a live-neighbour count, mirroring
/// the golden `diagonal`: the full six faces for a `Dirichlet` wall, else the
/// live-neighbour count clamped to one.
fn diagonal(boundary: PressureBoundary, live: u32) -> f32 {
    match boundary {
        PressureBoundary::Dirichlet => FACE_NEIGHBOR_COUNT,
        PressureBoundary::Neumann => {
            if live == 0 {
                1.0
            } else {
                live as f32
            }
        }
    }
}

/// The cell-centred `(3/4, 1/4)` prolongation stencil for one fine index on one
/// axis, mirroring the golden `axis_contributors`, returned as
/// `(idx, weight, len)`.
fn axis_contributors(fine: u32, coarse_n: u32) -> ([u32; 2], [f32; 2], u32) {
    if coarse_n == 0 {
        return ([0, 0], [0.0, 0.0], 0);
    }
    let mut parent = fine / 2;
    if parent >= coarse_n {
        parent = coarse_n - 1;
    }
    let own_weight = 0.75;
    let far_weight = 0.25;
    let far_is_lower = fine.is_multiple_of(2);
    let far_in_range = if far_is_lower {
        parent > 0
    } else {
        parent + 1 < coarse_n
    };
    if far_in_range {
        let far = if far_is_lower { parent - 1 } else { parent + 1 };
        ([parent, far], [own_weight, far_weight], 2)
    } else {
        ([parent, 0], [own_weight + far_weight, 0.0], 1)
    }
}

/// `2^k` as an `f32`, computed with an integer-doubling loop so no `pow` is
/// used, mirroring the golden `pow2_f32`. `k` is at most three here.
fn pow2_f32(k: u32) -> f32 {
    let mut value = 1.0f32;
    let mut remaining = k;
    while remaining > 0 {
        value *= 2.0;
        remaining -= 1;
    }
    value
}

/// Uniform dispatch parameters. `repr(C)` `std430` layout matching `Params` in
/// [`MULTIGRID_PRESSURE_WGSL`]: the lane count and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One query as uploaded. `160`-byte `std430` stride matching `Query` in the
/// shader: twelve leading `u32` codes/counts and seven operand `vec4`s.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    op: u32,
    boundary: u32,
    flag_pack: u32,
    fine_index: u32,
    coarse_n: u32,
    live: u32,
    coarsened_axes: u32,
    len_x: u32,
    len_y: u32,
    len_z: u32,
    ipad0: u32,
    ipad1: u32,
    scalars: [f32; 4],
    here: [f32; 4],
    forward: [f32; 4],
    nb_lo: [f32; 4],
    nb_hi: [f32; 4],
    wx_wy: [f32; 4],
    wz: [f32; 4],
}

/// One result as read back. `48`-byte `std430` stride matching `Res` in the
/// shader: the primary vector, the axis indices and the axis weights.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    v0: [f32; 4],
    idx: [u32; 4],
    w: [f32; 4],
}

/// Maps a wall model to its shader code.
fn boundary_code(boundary: PressureBoundary) -> u32 {
    match boundary {
        PressureBoundary::Dirichlet => BOUNDARY_DIRICHLET,
        PressureBoundary::Neumann => BOUNDARY_NEUMANN,
    }
}

/// Packs a fixed boolean flag slice into the low bits of a `u32` mask.
fn pack_flags(flags: &[bool]) -> u32 {
    let mut bits = 0u32;
    for (k, &f) in flags.iter().enumerate() {
        if f {
            bits |= 1u32 << (k as u32);
        }
    }
    bits
}

/// Encodes one query into its packed `GpuQuery`, placing each operand in the
/// slot the kernel reads for that operation code.
fn encode(query: &MultigridPressureQuery) -> GpuQuery {
    let mut g = GpuQuery::zeroed();
    match query {
        MultigridPressureQuery::DivergenceForward {
            here,
            forward,
            forward_in_grid,
        } => {
            g.op = OP_DIVERGENCE;
            g.flag_pack = pack_flags(forward_in_grid);
            g.here = [here[0], here[1], here[2], 0.0];
            g.forward = [forward[0], forward[1], forward[2], 0.0];
        }
        MultigridPressureQuery::GradientBackward {
            here,
            backward,
            backward_in_grid,
        } => {
            g.op = OP_GRADIENT;
            g.flag_pack = pack_flags(backward_in_grid);
            g.scalars = [*here, 0.0, 0.0, 0.0];
            g.forward = [backward[0], backward[1], backward[2], 0.0];
        }
        MultigridPressureQuery::JacobiCell {
            center,
            neighbors,
            neighbor_in_grid,
            rhs,
            inv_h2,
            omega,
            boundary,
        } => {
            g.op = OP_JACOBI;
            g.boundary = boundary_code(*boundary);
            g.flag_pack = pack_flags(neighbor_in_grid);
            g.scalars = [*center, *rhs, *inv_h2, *omega];
            g.nb_lo = [neighbors[0], neighbors[1], neighbors[2], neighbors[3]];
            g.nb_hi = [neighbors[4], neighbors[5], 0.0, 0.0];
        }
        MultigridPressureQuery::ResidualCell {
            center,
            neighbors,
            neighbor_in_grid,
            rhs,
            inv_h2,
            boundary,
        } => {
            g.op = OP_RESIDUAL;
            g.boundary = boundary_code(*boundary);
            g.flag_pack = pack_flags(neighbor_in_grid);
            g.scalars = [*center, *rhs, *inv_h2, 0.0];
            g.nb_lo = [neighbors[0], neighbors[1], neighbors[2], neighbors[3]];
            g.nb_hi = [neighbors[4], neighbors[5], 0.0, 0.0];
        }
        MultigridPressureQuery::Diagonal { boundary, live } => {
            g.op = OP_DIAGONAL;
            g.boundary = boundary_code(*boundary);
            g.live = *live;
        }
        MultigridPressureQuery::AxisContributors {
            fine_index,
            coarse_n,
        } => {
            g.op = OP_AXIS;
            g.fine_index = *fine_index;
            g.coarse_n = *coarse_n;
        }
        MultigridPressureQuery::ProlongCell {
            coarse_values,
            weight_x,
            weight_y,
            weight_z,
            len_x,
            len_y,
            len_z,
        } => {
            g.op = OP_PROLONG;
            g.len_x = *len_x;
            g.len_y = *len_y;
            g.len_z = *len_z;
            g.nb_lo = [
                coarse_values[0],
                coarse_values[1],
                coarse_values[2],
                coarse_values[3],
            ];
            g.nb_hi = [
                coarse_values[4],
                coarse_values[5],
                coarse_values[6],
                coarse_values[7],
            ];
            g.wx_wy = [weight_x[0], weight_x[1], weight_y[0], weight_y[1]];
            g.wz = [weight_z[0], weight_z[1], 0.0, 0.0];
        }
        MultigridPressureQuery::RestrictScale { coarsened_axes } => {
            g.op = OP_RESTRICT_SCALE;
            g.coarsened_axes = *coarsened_axes;
        }
    }
    g
}

/// Decodes one raw `GpuResult` into the typed result the query shape implies.
fn decode(query: &MultigridPressureQuery, r: &GpuResult) -> MultigridPressureResult {
    match query {
        MultigridPressureQuery::GradientBackward { .. } => MultigridPressureResult::Vector {
            v: [r.v0[0], r.v0[1], r.v0[2]],
        },
        MultigridPressureQuery::AxisContributors { .. } => MultigridPressureResult::Axis {
            idx: [r.idx[0], r.idx[1]],
            weight: [r.w[0], r.w[1]],
            len: r.idx[2],
        },
        MultigridPressureQuery::DivergenceForward { .. }
        | MultigridPressureQuery::JacobiCell { .. }
        | MultigridPressureQuery::ResidualCell { .. }
        | MultigridPressureQuery::Diagonal { .. }
        | MultigridPressureQuery::ProlongCell { .. }
        | MultigridPressureQuery::RestrictScale { .. } => {
            MultigridPressureResult::Scalar { value: r.v0[0] }
        }
    }
}

/// A compiled, reusable multigrid pressure per-cell primitive-evaluation
/// pipeline.
///
/// Provenance: `wgpu` compute twin of
/// `prism_render_architecture::particle::multigrid_pressure`.
pub struct GpuMultigridPressure {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMultigridPressure {
    /// Compiles the multigrid pressure per-cell kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required. Provenance: pipeline construction for the
    /// `multigrid_pressure` twin.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMultigridPressure {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_multigrid_pressure"),
            source: ShaderSource::Wgsl(MULTIGRID_PRESSURE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_multigrid_pressure_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_multigrid_pressure_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_multigrid_pressure_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMultigridPressure {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning one
    /// [`MultigridPressureResult`] per query in input order.
    ///
    /// Each lane reproduces the golden per-cell routine its query names, to
    /// within the tolerance documented on this module (the stencil index and
    /// length codes are exact). An empty `queries` slice yields an empty result
    /// — storage buffers cannot be zero-sized, so it is handled by an early
    /// return. Provenance: primitive evaluation for the `multigrid_pressure`
    /// twin.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[MultigridPressureQuery],
    ) -> Vec<MultigridPressureResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries.iter().map(encode).collect();
        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_multigrid_pressure_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_multigrid_pressure_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_multigrid_pressure_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_multigrid_pressure_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_multigrid_pressure_bind_group"),
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
            label: Some("prism_volumetric_multigrid_pressure_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_multigrid_pressure_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, in workgroups of 64 (the kernel's size).
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        queries
            .iter()
            .zip(gpu_results.iter())
            .map(|(query, raw)| decode(query, raw))
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
