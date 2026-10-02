//! `wgpu` compute twin of the 3D Marching Cubes *per-cell* iso-surface
//! primitive
//! ([`marching_cubes`](prism_render_architecture::particle::marching_cubes),
//! particle design §8.2 geometry contract).
//!
//! The `CPU` golden
//! [`marching_cubes`](prism_render_architecture::particle::marching_cubes) walks
//! a row-major `nx by ny by nz` scalar voxel grid and emits a growable triangle
//! [`Mesh`](prism_render_architecture::particle::marching_cubes::Mesh). That
//! whole-grid assembly produces a variable-length `Vec`, which does not fit a
//! fixed one-thread-per-element parity contract. Instead [`GpuMarchingCubes`]
//! twins the fixed-size, per-cell geometric kernel the whole-grid loop is built
//! from: the exact classification and interpolation each voxel cube performs
//! independently. One thread solves one cube, so a passing real-device parity
//! test is direct evidence the ported kernel classifies the same `cube_index`,
//! selects the same edges from the shipped
//! [`EDGE_TABLE`](prism_render_architecture::particle::marching_cubes::EDGE_TABLE)
//! and
//! [`TRI_TABLE`](prism_render_architecture::particle::marching_cubes::TRI_TABLE),
//! and places the same interpolated vertices in the same emission order the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For a batch of independent cube queries the kernel reproduces, per cube: the
//! `cube_index` classification code (`0..=255`, corner bit `c` set when
//! `corner_val[c] < iso`), the emitted vertex count (a multiple of three, at
//! most `15`), and the up-to-five triangles' interpolated vertex positions in
//! cube-local grid coordinates, in the exact
//! [`TRI_TABLE`](prism_render_architecture::particle::marching_cubes::TRI_TABLE)
//! emission order. Each corner sits at its
//! [`CORNER_OFFSET`](prism_render_architecture::particle::marching_cubes::CORNER_OFFSET)
//! integer position, and each crossed edge joins the two
//! [`EDGE_CORNERS`](prism_render_architecture::particle::marching_cubes::EDGE_CORNERS)
//! endpoints the reference uses. The whole-grid cube iteration and the growable
//! [`Mesh`](prism_render_architecture::particle::marching_cubes::Mesh)
//! concatenation are deliberately excluded, since their `Vec` output has no
//! fixed per-thread shape; a host loop over cells recovers the full mesh from
//! the per-cell twin.
//!
//! # Correctness model
//!
//! The `cube_index` and the vertex count are discrete classifications built from
//! `f32` magnitude comparisons (`corner_val < iso`), so for inputs clear of the
//! `corner_val == iso` classification boundary the `CPU` and `GPU` agree exactly
//! and the parity test asserts an exact `==` on both the code and the count. The
//! interpolated vertex coordinates thread through a subtract, one guarded
//! division, a `clamp`, a multiply and an add, so `CPU` and `GPU` are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits. The parity test therefore asserts
//! a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous coordinate.
//!
//! # Degenerate inputs
//!
//! When the two corner values along a crossed edge differ by less than
//! [`CMP_EPS`] the linear crossing parameter is ill-conditioned (a near-`0/0`),
//! so the kernel places the crossing at the edge midpoint `(p1 + p2) * 0.5`
//! instead of dividing by a vanishing denominator, matching the reference
//! `interp_edge`. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `clamp`,
//! `+ - * /`, bitwise `and` / `or` / shift and unsigned index arithmetic — with
//! no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `sqrt` and no optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. Every loop is bounded by the fixed eight corners, twelve
//! edges or sixteen triangle-table slots, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::marching_cubes`；无第三方引擎源码或衍生代码。
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

/// Maximum vertices one voxel cube can emit: five triangles of three vertices,
/// matching the sixteen-slot reference
/// [`TRI_TABLE`](prism_render_architecture::particle::marching_cubes::TRI_TABLE)
/// row (fifteen edge indices plus the `-1` terminator).
pub const MAX_CELL_VERTS: usize = 15;

/// Magnitude below which a crossed edge's corner-value difference is treated as
/// zero, mirroring the reference
/// [`EPS`](prism_render_architecture::particle::marching_cubes::EPS). When the
/// two corner values differ by less than this the crossing is placed at the edge
/// midpoint instead of dividing by a vanishing denominator; the compare rule
/// used instead of an `f32` `==`.
pub const CMP_EPS: f32 = 1.0e-6;

/// The portable core-`WGSL` Marching Cubes per-cell kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`marching_cubes`](prism_render_architecture::particle::marching_cubes)
/// per-cube path branch for branch; see the module documentation for the
/// algorithm.
const MARCHING_CUBES_WGSL: &str = r#"
// Marching Cubes per-cell twin: one thread per voxel cube reproduces the 8-bit
// corner classification, the shipped edge table, the guarded per-edge linear
// crossings and the up-to-five triangles a single cube contributes in cube-local
// grid coordinates. It mirrors the CPU golden `particle::marching_cubes` per-cube
// path branch for branch, uses only the portable core-WGSL subset (abs/clamp and
// + - * / plus bitwise and/or/shift and unsigned index math), needs no sqrt and
// no transcendental call and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12. Every loop is bounded by the fixed eight corners,
// twelve edges or sixteen triangle slots, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::marching_cubes；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a crossed edge's corner-value difference is treated as
// zero. Matches the reference `EPS`; the compare rule used instead of an f32
// `==`.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of cube queries in the storage arrays; threads past this
    // short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Corner scalar values 0..4 in the standard corner order (c0, c1, c2, c3).
    corners_lo: vec4<f32>,
    // Corner scalar values 4..8 in the standard corner order (c4, c5, c6, c7).
    corners_hi: vec4<f32>,
    // Iso threshold.
    iso: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // cube_index classification code in 0..=255.
    cube_index: u32,
    // Number of emitted vertices (a multiple of three, at most 15).
    vert_count: u32,
    pad0: u32,
    pad1: u32,
    // Up to 15 emitted vertex positions in cube-local grid coordinates (xyz; the
    // w lane pads each to 16 bytes). Lanes past vert_count are zero.
    verts: array<vec4<f32>, 15>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Cube-local grid position of corner `c`, i.e. the reference `CORNER_OFFSET[c]`
// cast to f32. Held in a function `var` so the runtime index lands in
// addressable memory.
fn corner_pos(c: u32) -> vec3<f32> {
    var t = array<vec3<f32>, 8>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(1.0, 0.0, 1.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(0.0, 1.0, 1.0),
    );
    return t[c];
}

// The `k`-th corner (k in 0..2) joined by edge `e`, i.e. the reference
// `EDGE_CORNERS[e][k]`. Held in a function `var` so the runtime index lands in
// addressable memory.
fn edge_corner(e: u32, k: u32) -> u32 {
    var t = array<u32, 24>(
        0u, 1u, 1u, 2u, 2u, 3u, 3u, 0u, 4u, 5u, 5u, 6u, 6u, 7u, 7u, 4u, 0u, 4u,
        1u, 5u, 2u, 6u, 3u, 7u,
    );
    return t[e * 2u + k];
}

// Linearly interpolates the surface vertex on the edge from p1 to p2, whose
// corner values are v1 and v2, at the iso crossing; mirrors the reference
// `interp_edge`. When the two corner values are within CMP_EPS of each other the
// denominator is ill-conditioned, so the edge midpoint is returned instead.
fn interp_edge(iso: f32, p1: vec3<f32>, p2: vec3<f32>, v1: f32, v2: f32) -> vec3<f32> {
    let denom = v2 - v1;
    if (abs(denom) <= CMP_EPS) {
        return (p1 + p2) * 0.5;
    }
    let mu = clamp((iso - v1) / denom, 0.0, 1.0);
    return p1 + (p2 - p1) * mu;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let iso = q.iso;

    // Eight corner scalar values in the standard corner order.
    var cv = array<f32, 8>(
        q.corners_lo.x, q.corners_lo.y, q.corners_lo.z, q.corners_lo.w,
        q.corners_hi.x, q.corners_hi.y, q.corners_hi.z, q.corners_hi.w,
    );

    // Corner bit c is set when its value is strictly below iso; mirrors the
    // reference classification exactly.
    var cube_index: u32 = 0u;
    for (var c: u32 = 0u; c < 8u; c = c + 1u) {
        if (cv[c] < iso) {
            cube_index = cube_index | (1u << c);
        }
    }

    // Edge table: the 12-bit mask of edges that straddle iso for this case. Held
    // in a function `var` so the runtime cube_index lands in addressable memory.
    var edge_tbl = array<u32, 256>(
        0u, 265u, 515u, 778u, 1030u, 1295u, 1541u, 1804u, 2060u, 2309u, 2575u, 2822u, 3082u,
        3331u, 3593u, 3840u, 400u, 153u, 915u, 666u, 1430u, 1183u, 1941u, 1692u, 2460u, 2197u,
        2975u, 2710u, 3482u, 3219u, 3993u, 3728u, 560u, 825u, 51u, 314u, 1590u, 1855u, 1077u,
        1340u, 2620u, 2869u, 2111u, 2358u, 3642u, 3891u, 3129u, 3376u, 928u, 681u, 419u, 170u,
        1958u, 1711u, 1445u, 1196u, 2988u, 2725u, 2479u, 2214u, 4010u, 3747u, 3497u, 3232u,
        1120u, 1385u, 1635u, 1898u, 102u, 367u, 613u, 876u, 3180u, 3429u, 3695u, 3942u, 2154u,
        2403u, 2665u, 2912u, 1520u, 1273u, 2035u, 1786u, 502u, 255u, 1013u, 764u, 3580u, 3317u,
        4095u, 3830u, 2554u, 2291u, 3065u, 2800u, 1616u, 1881u, 1107u, 1370u, 598u, 863u, 85u,
        348u, 3676u, 3925u, 3167u, 3414u, 2650u, 2899u, 2137u, 2384u, 1984u, 1737u, 1475u,
        1226u, 966u, 719u, 453u, 204u, 4044u, 3781u, 3535u, 3270u, 3018u, 2755u, 2505u, 2240u,
        2240u, 2505u, 2755u, 3018u, 3270u, 3535u, 3781u, 4044u, 204u, 453u, 719u, 966u, 1226u,
        1475u, 1737u, 1984u, 2384u, 2137u, 2899u, 2650u, 3414u, 3167u, 3925u, 3676u, 348u, 85u,
        863u, 598u, 1370u, 1107u, 1881u, 1616u, 2800u, 3065u, 2291u, 2554u, 3830u, 4095u,
        3317u, 3580u, 764u, 1013u, 255u, 502u, 1786u, 2035u, 1273u, 1520u, 2912u, 2665u, 2403u,
        2154u, 3942u, 3695u, 3429u, 3180u, 876u, 613u, 367u, 102u, 1898u, 1635u, 1385u, 1120u,
        3232u, 3497u, 3747u, 4010u, 2214u, 2479u, 2725u, 2988u, 1196u, 1445u, 1711u, 1958u,
        170u, 419u, 681u, 928u, 3376u, 3129u, 3891u, 3642u, 2358u, 2111u, 2869u, 2620u, 1340u,
        1077u, 1855u, 1590u, 314u, 51u, 825u, 560u, 3728u, 3993u, 3219u, 3482u, 2710u, 2975u,
        2197u, 2460u, 1692u, 1941u, 1183u, 1430u, 666u, 915u, 153u, 400u, 3840u, 3593u, 3331u,
        3082u, 2822u, 2575u, 2309u, 2060u, 1804u, 1541u, 1295u, 1030u, 778u, 515u, 265u, 0u,
    );
    let edges = edge_tbl[cube_index];

    // Interpolate a vertex on every crossed edge, exactly as the reference does
    // before consulting the triangle table.
    var vert = array<vec3<f32>, 12>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );
    for (var e: u32 = 0u; e < 12u; e = e + 1u) {
        if ((edges & (1u << e)) != 0u) {
            let a = edge_corner(e, 0u);
            let b = edge_corner(e, 1u);
            vert[e] = interp_edge(iso, corner_pos(a), corner_pos(b), cv[a], cv[b]);
        }
    }

    var out: Result;
    out.cube_index = cube_index;
    out.pad0 = 0u;
    out.pad1 = 0u;
    for (var s: u32 = 0u; s < 15u; s = s + 1u) {
        out.verts[s] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Triangle table: for each case, up to five triangles given as edge indices
    // in groups of three, terminated by -1. Held in a function `var` so the
    // runtime cube_index lands in addressable memory.
    var tri = array<i32, 4096>(
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 8, 3, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 10, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 8, 3,
        2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
        9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 11, 2, 1, 9, 11, 9, 8, 11,
        -1, -1, -1, -1, -1, -1, -1, 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1, 3, 9, 0, 3, 11, 9, 11,
        10, 9, -1, -1, -1, -1, -1, -1, -1, 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 0, 7, 3, 4,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 10, 8, 4, 7,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1,
        -1, -1, 9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, 2, 10, 9, 2, 9, 7, 2,
        7, 3, 7, 9, 4, -1, -1, -1, -1, 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1, 9, 0, 1, 8, 4, 7, 2, 3,
        11, -1, -1, -1, -1, -1, -1, -1, 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1,
        3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1, 1, 11, 10, 1, 4, 11, 1, 0, 4,
        7, 11, 4, -1, -1, -1, -1, 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1, 4, 7,
        11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1, 9, 5, 4, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 5, 4, 8, 3, 5, 3, 1, 5,
        -1, -1, -1, -1, -1, -1, -1, 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1, 5, 2, 10, 5, 4, 2, 4, 0, 2, -1,
        -1, -1, -1, -1, -1, -1, 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1, 9, 5, 4,
        2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1,
        -1, -1, -1, -1, -1, 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, 2, 1, 5, 2,
        5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1, 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1,
        -1, -1, -1, 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1, 5, 4, 0, 5, 0, 11,
        5, 11, 10, 11, 0, 3, -1, -1, -1, -1, 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1,
        -1, -1, 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 3, 0, 9, 5, 3, 5,
        7, 3, -1, -1, -1, -1, -1, -1, -1, 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1,
        -1, 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 7, 8, 9, 5, 7, 10, 1,
        2, -1, -1, -1, -1, -1, -1, -1, 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1, 8,
        0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1, 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1,
        -1, -1, -1, -1, -1, 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, 9, 5, 7, 9,
        7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1, 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1,
        -1, -1, 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1, 9, 5, 8, 8, 5, 7, 10,
        1, 3, 10, 3, 11, -1, -1, -1, -1, 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1,
        11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1, 11, 10, 5, 7, 11, 5, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 0, 1, 5, 10, 6, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1,
        -1, 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 6, 5, 1, 2, 6, 3, 0,
        8, -1, -1, -1, -1, -1, -1, -1, 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1,
        5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1, 2, 3, 11, 10, 6, 5, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, 0, 1,
        9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8,
        11, -1, -1, -1, -1, 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1, 0, 8, 11,
        0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1, 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1,
        -1, -1, -1, 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1, 5, 10, 6, 4, 7, 8,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1,
        -1, -1, 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, 10, 6, 5, 1, 9, 7, 1,
        7, 3, 7, 9, 4, -1, -1, -1, -1, 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1,
        1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1, 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6,
        -1, -1, -1, -1, 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1, 3, 11, 2, 7, 8, 4, 10,
        6, 5, -1, -1, -1, -1, -1, -1, -1, 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1,
        0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, 9, 2, 1, 9, 11, 2, 9, 4, 11, 7,
        11, 4, 5, 10, 6, -1, 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1, 5, 1, 11, 5,
        11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1, 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7,
        -1, 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1, 10, 4, 9, 6, 4, 10, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1,
        -1, 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1, 8, 3, 1, 8, 1, 6, 8, 6, 4,
        6, 1, 10, -1, -1, -1, -1, 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1, 3, 0,
        8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1, 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, 10, 4, 9, 10, 6,
        4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1, 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1,
        -1, -1, 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1, 6, 4, 1, 6, 1, 10, 4, 8,
        1, 2, 1, 11, 8, 11, 1, -1, 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1, 8, 11,
        1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1, 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1,
        -1, -1, -1, 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, 10, 6, 7, 8,
        10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1, 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1,
        -1, -1, -1, 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1, 10, 6, 7, 10, 7, 1,
        1, 7, 3, -1, -1, -1, -1, -1, -1, -1, 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1,
        -1, 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1, 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1,
        -1, -1, -1, -1, -1, 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 11,
        10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1, 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9,
        10, 7, -1, 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1, 11, 2, 1, 11, 1, 7, 10,
        6, 1, 6, 7, 1, -1, -1, -1, -1, 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1, 0, 9,
        1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6,
        0, -1, -1, -1, -1, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, 6,
        11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 0, 8, 11, 7, 6, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, 10, 1, 2, 6, 11, 7, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, 2,
        9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, 6, 11, 7, 2, 10, 3, 10, 8, 3, 10,
        9, 8, -1, -1, -1, -1, 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, 0,
        8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1, 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1,
        -1, -1, -1, -1, 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1, 10, 7, 6, 10, 1, 7,
        1, 3, 7, -1, -1, -1, -1, -1, -1, -1, 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1,
        -1, 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1, 7, 6, 10, 7, 10, 8, 8, 10,
        9, -1, -1, -1, -1, -1, -1, -1, 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1, 8, 6, 11, 8, 4, 6, 9, 0, 1,
        -1, -1, -1, -1, -1, -1, -1, 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1, 6, 8,
        4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4,
        6, -1, -1, -1, -1, 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1, 10, 9, 3, 10,
        3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1, 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1,
        -1, -1, 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 9, 0, 2, 3, 4, 2,
        4, 6, 4, 3, 8, -1, -1, -1, -1, 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1,
        8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1, 10, 1, 0, 10, 0, 6, 6, 0, 4, -1,
        -1, -1, -1, -1, -1, -1, 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1, 10, 9, 4, 6,
        10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, 5, 0, 1, 5,
        4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1,
        -1, -1, -1, 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, 6, 11, 7, 1, 2,
        10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1, 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1,
        -1, -1, 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1, 7, 2, 3, 7, 6, 2, 5, 4, 9,
        -1, -1, -1, -1, -1, -1, -1, 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1, 3, 6,
        2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1, 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5,
        8, -1, 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1, 1, 6, 10, 1, 7, 6, 1, 0, 7,
        8, 7, 0, 9, 5, 4, -1, 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1, 7, 6, 10,
        7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1, 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1,
        -1, -1, -1, -1, 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1, 0, 11, 8, 0, 5,
        11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1, 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1,
        -1, -1, 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1, 0, 11, 3, 0, 6, 11, 0,
        9, 6, 5, 6, 9, 1, 2, 10, -1, 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1, 6, 11,
        3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1, 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1,
        -1, -1, -1, 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1, 1, 5, 8, 1, 8, 0, 5,
        6, 8, 3, 8, 2, 6, 2, 8, -1, 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1, 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0,
        -1, -1, -1, -1, 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, 5, 6,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 5, 10, 7, 5, 11, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, 11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1,
        5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1, 10, 7, 5, 10, 11, 7, 9, 8, 1,
        8, 3, 1, -1, -1, -1, -1, 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1, 0, 8,
        3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1, 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1,
        -1, -1, -1, 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1, 2, 5, 10, 2, 3, 5, 3, 7,
        5, -1, -1, -1, -1, -1, -1, -1, 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1, 9,
        0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1, 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5,
        7, 5, 2, -1, 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 8, 7, 0, 7,
        1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1, 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1,
        -1, -1, 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 8, 4, 5, 10, 8,
        10, 11, 8, -1, -1, -1, -1, -1, -1, -1, 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1,
        -1, -1, 0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1, 10, 11, 4, 10, 4, 5,
        11, 3, 4, 9, 4, 1, 3, 1, 4, -1, 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1, 0,
        4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1, 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8,
        11, 8, 5, -1, 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 10, 3,
        5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1, 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1,
        -1, -1, 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1, 5, 10, 2, 5, 2, 4, 1, 9, 2,
        9, 4, 2, -1, -1, -1, -1, 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1, 0, 4,
        5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5,
        -1, -1, -1, -1, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 11, 7,
        4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11,
        -1, -1, -1, -1, 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1, 3, 1, 4, 3, 4,
        8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1, 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1,
        -1, -1, 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1, 11, 7, 4, 11, 4, 2, 2, 4,
        0, -1, -1, -1, -1, -1, -1, -1, 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1, 2,
        9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1, 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0,
        2, 0, 7, -1, 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1, 1, 10, 2, 8, 7, 4,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1,
        -1, -1, 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1, 4, 0, 3, 7, 4, 3, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 0, 9, 3, 9, 11,
        11, 9, 10, -1, -1, -1, -1, -1, -1, -1, 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1,
        -1, -1, -1, 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 11, 1,
        11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1, 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1,
        -1, -1, -1, 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 2, 11, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1,
        -1, -1, -1, -1, -1, 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8,
        2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1, 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 9, 1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 8, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1,
    );

    // Emit triangles in TRI_TABLE order: step three edge indices at a time and
    // stop at the terminator, mirroring the reference `while i + 2 < 16 &&
    // row[i] >= 0`.
    var count: u32 = 0u;
    for (var i: u32 = 0u; i + 2u < 16u; i = i + 3u) {
        let base = cube_index * 16u + i;
        if (tri[base] < 0) {
            break;
        }
        let e0 = u32(tri[base]);
        let e1 = u32(tri[base + 1u]);
        let e2 = u32(tri[base + 2u]);
        out.verts[count] = vec4<f32>(vert[e0], 0.0);
        out.verts[count + 1u] = vec4<f32>(vert[e1], 0.0);
        out.verts[count + 2u] = vec4<f32>(vert[e2], 0.0);
        count = count + 3u;
    }
    out.vert_count = count;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MARCHING_CUBES_WGSL`].
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

/// `repr(C)` `std430` layout of one cube query, matching the `WGSL` `Query`
/// struct. The eight corner values are split into two `vec4` lanes so each stays
/// `16`-byte aligned on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Corner values `0..4` `(c0, c1, c2, c3)`.
    corners_lo: [f32; 4],
    /// Corner values `4..8` `(c4, c5, c6, c7)`.
    corners_hi: [f32; 4],
    /// Iso threshold.
    iso: f32,
    /// Pad lane.
    pad0: f32,
    /// Pad lane.
    pad1: f32,
    /// Pad lane.
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
/// Each emitted vertex occupies a `vec4` lane (`xyz` position, `w` pad).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `cube_index` classification code in `0..=255`.
    cube_index: u32,
    /// Number of emitted vertices (a multiple of three, at most `15`).
    vert_count: u32,
    /// Pad word.
    pad0: u32,
    /// Pad word.
    pad1: u32,
    /// Up to `15` emitted vertex positions (`xyz` plus a `w` pad lane); lanes
    /// past `vert_count` are zero.
    verts: [[f32; 4]; MAX_CELL_VERTS],
}

/// One per-cube query for the Marching Cubes twin: the eight corner scalar
/// values in the standard corner order and the iso threshold.
///
/// Corner values follow the
/// [`CORNER_OFFSET`](prism_render_architecture::particle::marching_cubes::CORNER_OFFSET)
/// order, so `corner_values[c]` is the sample at that corner of the unit cube.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarchingCubesCellQuery {
    /// Eight corner scalar values in the standard corner order.
    pub corner_values: [f32; 8],
    /// Iso threshold.
    pub iso: f32,
}

/// One resolved answer for a single voxel cube, mirroring the reference per-cube
/// classification and emitted triangle vertices.
///
/// Only the leading `vert_count` entries of `vertices` are meaningful; the
/// remainder are zero. Vertices appear in the
/// [`TRI_TABLE`](prism_render_architecture::particle::marching_cubes::TRI_TABLE)
/// emission order, three per triangle, each an `[x, y, z]` position in
/// cube-local grid coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarchingCubesCellResult {
    /// `cube_index` classification code in `0..=255`, with corner bit `c` set
    /// when `corner_values[c] < iso`.
    pub cube_index: u32,
    /// Number of emitted vertices (a multiple of three, at most `15`).
    pub vert_count: u32,
    /// Up to `15` emitted vertex positions; only the leading `vert_count` are
    /// meaningful.
    pub vertices: [[f32; 3]; MAX_CELL_VERTS],
}

/// Encodes one [`MarchingCubesCellQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &MarchingCubesCellQuery) -> GpuQuery {
    GpuQuery {
        corners_lo: [
            q.corner_values[0],
            q.corner_values[1],
            q.corner_values[2],
            q.corner_values[3],
        ],
        corners_hi: [
            q.corner_values[4],
            q.corner_values[5],
            q.corner_values[6],
            q.corner_values[7],
        ],
        iso: q.iso,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`MarchingCubesCellResult`],
/// dropping each vertex's `w` pad lane.
fn decode_result(raw: &GpuResult) -> MarchingCubesCellResult {
    let mut vertices = [[0.0_f32; 3]; MAX_CELL_VERTS];
    for (slot, v) in vertices.iter_mut().zip(raw.verts.iter()) {
        *slot = [v[0], v[1], v[2]];
    }
    MarchingCubesCellResult {
        cube_index: raw.cube_index,
        vert_count: raw.vert_count,
        vertices,
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

/// A compiled, reusable Marching Cubes per-cell compute pipeline, twinning the
/// `CPU` golden
/// [`marching_cubes`](prism_render_architecture::particle::marching_cubes).
pub struct GpuMarchingCubes {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMarchingCubes {
    /// Compiles the Marching Cubes per-cell kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMarchingCubes {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_marching_cubes"),
            source: ShaderSource::Wgsl(MARCHING_CUBES_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_marching_cubes_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_marching_cubes_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_marching_cubes_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMarchingCubes {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every cube in `queries` and returns one [`MarchingCubesCellResult`]
    /// per input, in order.
    ///
    /// The `cube_index` and vertex count equal the reference exactly for inputs
    /// clear of the `corner_values == iso` classification boundary; the vertex
    /// coordinates match to within the tolerance documented on this module. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MarchingCubesCellQuery],
    ) -> Vec<MarchingCubesCellResult> {
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
            label: Some("prism_volumetric_marching_cubes_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_marching_cubes_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_marching_cubes_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_marching_cubes_bind_group"),
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
            label: Some("prism_volumetric_marching_cubes_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_marching_cubes_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_marching_cubes_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per cube, flattened to a 1-D dispatch.
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
