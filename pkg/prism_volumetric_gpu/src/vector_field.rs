//! `wgpu` compute twin of the discrete 3D vector-field sampling, indexing,
//! boundary-wrap and world⇄grid transform contract
//! ([`vector_field`](prism_render_architecture::particle::vector_field),
//! particle design §8, §10).
//!
//! The `CPU` golden
//! [`vector_field`](prism_render_architecture::particle::vector_field) owns the
//! stored-data flow field a particle reads through a `Texture3d` binding: a
//! row-major (`X`-fastest) grid of `Vec3` texels
//! ([`VectorField`](prism_render_architecture::particle::vector_field::VectorField)),
//! a boundary policy
//! ([`WrapMode`](prism_render_architecture::particle::vector_field::WrapMode))
//! deciding how an out-of-range index resolves, trilinear reconstruction
//! ([`VectorField::sample_grid`](prism_render_architecture::particle::vector_field::VectorField::sample_grid)),
//! and the axis-aligned-box map between world space and continuous grid space
//! ([`VectorFieldTransform`](prism_render_architecture::particle::vector_field::VectorFieldTransform)).
//! [`GpuVectorField`] is the on-device twin: a single read-only grid is uploaded
//! once and every thread samples it, so one thread resolves one query and a
//! passing real-device parity test is direct evidence the ported kernel
//! reproduces the same texels, indices and interpolated vectors the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query reproduces, against the one shared grid, every per-sample answer
//! the reference computes: the row-major linear index
//! ([`VectorField::linear_index`](prism_render_architecture::particle::vector_field::VectorField::linear_index)),
//! the clamped integer texel read
//! ([`VectorField::sample_texel`](prism_render_architecture::particle::vector_field::VectorField::sample_texel)),
//! the trilinear sample at a continuous grid coordinate under a wrap code
//! ([`VectorField::sample_grid`](prism_render_architecture::particle::vector_field::VectorField::sample_grid)),
//! the world-to-grid and grid-to-world maps
//! ([`VectorFieldTransform::world_to_grid`](prism_render_architecture::particle::vector_field::VectorFieldTransform::world_to_grid)
//! and
//! [`VectorFieldTransform::grid_to_world`](prism_render_architecture::particle::vector_field::VectorFieldTransform::grid_to_world)),
//! and the central-difference
//! [`VectorField::divergence`](prism_render_architecture::particle::vector_field::VectorField::divergence)
//! and
//! [`VectorField::curl`](prism_render_architecture::particle::vector_field::VectorField::curl)
//! diagnostics. The private `resolve_index` boundary arithmetic is twinned too
//! and is exercised indirectly through the eight-corner `fetch` of the
//! trilinear sampler at out-of-range coordinates under each wrap code.
//!
//! # Boundary codes
//!
//! The reference
//! [`WrapMode`](prism_render_architecture::particle::vector_field::WrapMode) has
//! two variants, carried into the kernel as a `u32` code: `0` is
//! [`WRAP_CLAMP`] (clamp a signed index to `[0, dim)`) and `1` is [`WRAP_TILE`]
//! (periodic tiling by the positive-normalised integer modulo
//! `(((i % d) + d) % d)`). The index arithmetic is integer and exact, so the
//! resolved texel address matches the reference bit for bit.
//!
//! # Correctness model
//!
//! The linear index and the resolved texel address are integer, so `CPU` and
//! `GPU` agree exactly and the parity test asserts `==` on the index. The
//! sampled vectors, the transform maps and the derivatives thread through
//! multiplies, adds and one guarded division, so they are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits. The parity test therefore asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous quantity, tight enough to catch a dropped term or a swapped
//! axis yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A zero-extent transform axis (`hi - lo` within [`CMP_EPS`] of zero) maps to
//! `0.0` on that axis instead of dividing by zero, mirroring the reference
//! guard. A `dim = 1` axis clamps every texel read to index `0`. An empty query
//! batch short-circuits on the host with no dispatch, since a storage buffer
//! cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `floor`, `abs`, `+ - * /` and signed/unsigned index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry,
//! no `sqrt` and no optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. There is no loop: each thread performs a fixed, bounded
//! sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::vector_field`；无第三方引擎源码或衍生代码。
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

/// Boundary code for the clamp policy: a signed index is clamped into
/// `[0, dim)`, matching
/// [`WrapMode::Clamp`](prism_render_architecture::particle::vector_field::WrapMode::Clamp).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::vector_field`。
pub const WRAP_CLAMP: u32 = 0;

/// Boundary code for the tile policy: a signed index wraps periodically by the
/// positive-normalised integer modulo, matching
/// [`WrapMode::Tile`](prism_render_architecture::particle::vector_field::WrapMode::Tile).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::vector_field`。
pub const WRAP_TILE: u32 = 1;

/// The portable core-`WGSL` vector-field kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`vector_field`](prism_render_architecture::particle::vector_field) branch
/// for branch; see the module documentation for the algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::vector_field`。
const VECTOR_FIELD_WGSL: &str = r#"
// Vector-field twin: one shared read-only grid, one thread per query. Each
// thread reproduces the linear index, the clamped texel read, the trilinear
// sample under a wrap code, the world<->grid transform maps and the
// central-difference divergence and curl. It mirrors the CPU golden
// `particle::vector_field` branch for branch, uses only the portable core-WGSL
// subset (min/max/clamp/floor/abs and + - * / plus index arithmetic), needs no
// sqrt and no transcendental call and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12. There is no loop, so the kernel
// provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::vector_field；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a transform axis extent is treated as zero. Matches the
// reference `EPS`; the compare rule used instead of an f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

// Boundary codes mirroring the reference `WrapMode`.
const WRAP_CLAMP: u32 = 0u;
const WRAP_TILE: u32 = 1u;

struct Params {
    // Texel counts along (X, Y, Z); every axis is at least 1.
    dims: vec3<u32>,
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    // World-space minimum corner of the field's bounding box.
    bounds_min: vec3<f32>,
    pad0: f32,
    // World-space maximum corner of the field's bounding box.
    bounds_max: vec3<f32>,
    pad1: f32,
}

struct Query {
    // Continuous grid coordinate fed to `sample_grid`.
    grid: vec3<f32>,
    // Boundary code (WRAP_CLAMP or WRAP_TILE) for the trilinear fetches.
    wrap: u32,
    // World position fed to `world_to_grid` (then round-tripped back).
    world: vec3<f32>,
    pad_w: f32,
    // Integer texel coordinate fed to `sample_texel`, `linear_index`,
    // `divergence` and `curl`.
    texel: vec3<u32>,
    pad_t: u32,
}

struct Result {
    // sample_grid vector, with the row-major linear index packed into w's lane.
    sampled: vec3<f32>,
    lindex: u32,
    // world_to_grid of `world`, with the divergence packed into w's lane.
    grid_from_world: vec3<f32>,
    divergence: f32,
    // grid_to_world of `grid_from_world` (the transform round-trip).
    world_roundtrip: vec3<f32>,
    pad_r: f32,
    // sample_texel at the integer `texel` coordinate.
    texel_value: vec3<f32>,
    pad_v: f32,
    // curl at the integer `texel` coordinate.
    curl: vec3<f32>,
    pad_c: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;

// Row-major (X-fastest) linear index of texel (i, j, k); mirrors the reference
// `linear_index`.
fn linear_index(i: u32, j: u32, k: u32) -> u32 {
    let nx = params.dims.x;
    let ny = params.dims.y;
    return ((k * ny) + j) * nx + i;
}

// Reads the stored vector at an in-range linear index.
fn texel_at(idx: u32) -> vec3<f32> {
    return field[idx].xyz;
}

// Stored vector at integer texel (i, j, k), clamped into range first; mirrors
// the reference `sample_texel`.
fn sample_texel(i: u32, j: u32, k: u32) -> vec3<f32> {
    let nx = params.dims.x;
    let ny = params.dims.y;
    let nz = params.dims.z;
    let ci = min(i, nx - 1u);
    let cj = min(j, ny - 1u);
    let ck = min(k, nz - 1u);
    return texel_at(linear_index(ci, cj, ck));
}

// Resolves a signed grid index into a valid [0, dim) index under a boundary
// code; mirrors the reference `resolve_index`. `dim` is non-zero.
fn resolve_index(i: i32, dim: u32, wrap: u32) -> u32 {
    let d = i32(dim);
    if (wrap == WRAP_CLAMP) {
        return u32(clamp(i, 0, d - 1));
    }
    // WRAP_TILE: positive-normalised integer modulo.
    return u32(((i % d) + d) % d);
}

// Fetches a texel by signed coordinates under a boundary code; mirrors the
// reference `fetch` used by the trilinear sampler for the eight corners.
fn fetch(i: i32, j: i32, k: i32, wrap: u32) -> vec3<f32> {
    let ri = resolve_index(i, params.dims.x, wrap);
    let rj = resolve_index(j, params.dims.y, wrap);
    let rk = resolve_index(k, params.dims.z, wrap);
    return texel_at(linear_index(ri, rj, rk));
}

// Trilinearly samples the field at a continuous grid coordinate; mirrors the
// reference `sample_grid` (floor to the eight corners, frac to the weights,
// multiply-add blend).
fn sample_grid(grid: vec3<f32>, wrap: u32) -> vec3<f32> {
    let i0f = floor(grid.x);
    let j0f = floor(grid.y);
    let k0f = floor(grid.z);
    let fx = grid.x - i0f;
    let fy = grid.y - j0f;
    let fz = grid.z - k0f;
    let i0 = i32(i0f);
    let j0 = i32(j0f);
    let k0 = i32(k0f);
    let i1 = i0 + 1;
    let j1 = j0 + 1;
    let k1 = k0 + 1;

    let c000 = fetch(i0, j0, k0, wrap);
    let c100 = fetch(i1, j0, k0, wrap);
    let c010 = fetch(i0, j1, k0, wrap);
    let c110 = fetch(i1, j1, k0, wrap);
    let c001 = fetch(i0, j0, k1, wrap);
    let c101 = fetch(i1, j0, k1, wrap);
    let c011 = fetch(i0, j1, k1, wrap);
    let c111 = fetch(i1, j1, k1, wrap);

    let gx = 1.0 - fx;
    let gy = 1.0 - fy;
    let gz = 1.0 - fz;

    let w000 = gx * gy * gz;
    let w100 = fx * gy * gz;
    let w010 = gx * fy * gz;
    let w110 = fx * fy * gz;
    let w001 = gx * gy * fz;
    let w101 = fx * gy * fz;
    let w011 = gx * fy * fz;
    let w111 = fx * fy * fz;

    return c000 * w000
        + c100 * w100
        + c010 * w010
        + c110 * w110
        + c001 * w001
        + c101 * w101
        + c011 * w011
        + c111 * w111;
}

// One-axis world-to-grid map with a degenerate-extent guard; mirrors the
// reference `axis_world_to_grid`.
fn axis_world_to_grid(world: f32, lo: f32, hi: f32, dim: u32) -> f32 {
    let size = hi - lo;
    if (abs(size) < CMP_EPS) {
        return 0.0;
    }
    let norm = (world - lo) / size;
    return norm * f32(dim) - 0.5;
}

// One-axis grid-to-world map (inverse of axis_world_to_grid); mirrors the
// reference `axis_grid_to_world`.
fn axis_grid_to_world(grid: f32, lo: f32, hi: f32, dim: u32) -> f32 {
    let size = hi - lo;
    return lo + ((grid + 0.5) / f32(dim)) * size;
}

// world_to_grid: per-axis `axis_world_to_grid`.
fn world_to_grid(world: vec3<f32>) -> vec3<f32> {
    let lo = params.bounds_min;
    let hi = params.bounds_max;
    return vec3<f32>(
        axis_world_to_grid(world.x, lo.x, hi.x, params.dims.x),
        axis_world_to_grid(world.y, lo.y, hi.y, params.dims.y),
        axis_world_to_grid(world.z, lo.z, hi.z, params.dims.z),
    );
}

// grid_to_world: per-axis `axis_grid_to_world`.
fn grid_to_world(grid: vec3<f32>) -> vec3<f32> {
    let lo = params.bounds_min;
    let hi = params.bounds_max;
    return vec3<f32>(
        axis_grid_to_world(grid.x, lo.x, hi.x, params.dims.x),
        axis_grid_to_world(grid.y, lo.y, hi.y, params.dims.y),
        axis_grid_to_world(grid.z, lo.z, hi.z, params.dims.z),
    );
}

// Neighbour difference v(i+1) - v(i-1) along X (Clamp-wrapped); mirrors the
// reference `central_x`.
fn central_x(i: u32, j: u32, k: u32) -> vec3<f32> {
    let ii = i32(i);
    let plus = fetch(ii + 1, i32(j), i32(k), WRAP_CLAMP);
    let minus = fetch(ii - 1, i32(j), i32(k), WRAP_CLAMP);
    return plus - minus;
}

// Neighbour difference v(j+1) - v(j-1) along Y (Clamp-wrapped); mirrors the
// reference `central_y`.
fn central_y(i: u32, j: u32, k: u32) -> vec3<f32> {
    let jj = i32(j);
    let plus = fetch(i32(i), jj + 1, i32(k), WRAP_CLAMP);
    let minus = fetch(i32(i), jj - 1, i32(k), WRAP_CLAMP);
    return plus - minus;
}

// Neighbour difference v(k+1) - v(k-1) along Z (Clamp-wrapped); mirrors the
// reference `central_z`.
fn central_z(i: u32, j: u32, k: u32) -> vec3<f32> {
    let kk = i32(k);
    let plus = fetch(i32(i), i32(j), kk + 1, WRAP_CLAMP);
    let minus = fetch(i32(i), i32(j), kk - 1, WRAP_CLAMP);
    return plus - minus;
}

// Central-difference divergence; mirrors the reference `divergence`.
fn divergence(i: u32, j: u32, k: u32) -> f32 {
    let dx = central_x(i, j, k);
    let dy = central_y(i, j, k);
    let dz = central_z(i, j, k);
    return (dx.x + dy.y + dz.z) * 0.5;
}

// Central-difference curl; mirrors the reference `curl`.
fn curl(i: u32, j: u32, k: u32) -> vec3<f32> {
    let dx = central_x(i, j, k);
    let dy = central_y(i, j, k);
    let dz = central_z(i, j, k);
    return vec3<f32>(
        (dy.z - dz.y) * 0.5,
        (dz.x - dx.z) * 0.5,
        (dx.y - dy.x) * 0.5,
    );
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let sampled = sample_grid(q.grid, q.wrap);
    let grid_from_world = world_to_grid(q.world);
    let world_roundtrip = grid_to_world(grid_from_world);
    let texel_value = sample_texel(q.texel.x, q.texel.y, q.texel.z);
    let lindex = linear_index(q.texel.x, q.texel.y, q.texel.z);
    let div = divergence(q.texel.x, q.texel.y, q.texel.z);
    let cu = curl(q.texel.x, q.texel.y, q.texel.z);

    var out: Result;
    out.sampled = sampled;
    out.lindex = lindex;
    out.grid_from_world = grid_from_world;
    out.divergence = div;
    out.world_roundtrip = world_roundtrip;
    out.pad_r = 0.0;
    out.texel_value = texel_value;
    out.pad_v = 0.0;
    out.curl = cu;
    out.pad_c = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the field dimensions, the query count
/// and the world-space bounding box, padded to the `std140` `16`-byte alignment
/// matching `Params` in [`VECTOR_FIELD_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Texel counts along `(X, Y, Z)`.
    dims: [u32; 3],
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// World-space minimum corner of the field box.
    bounds_min: [f32; 3],
    /// Padding word.
    pad0: f32,
    /// World-space maximum corner of the field box.
    bounds_max: [f32; 3],
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// Each `vec3` lane carries a trailing pad word so it stays `16`-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Continuous grid coordinate for `sample_grid`.
    grid: [f32; 3],
    /// Boundary code for the trilinear fetches.
    wrap: u32,
    /// World position for `world_to_grid`.
    world: [f32; 3],
    /// Pad lane after `world`.
    pad_w: f32,
    /// Integer texel coordinate for `sample_texel`, `divergence` and `curl`.
    texel: [u32; 3],
    /// Pad lane after `texel`.
    pad_t: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `sample_grid` vector.
    sampled: [f32; 3],
    /// Row-major linear index of the integer texel.
    lindex: u32,
    /// `world_to_grid` of the query world position.
    grid_from_world: [f32; 3],
    /// Central-difference divergence at the integer texel.
    divergence: f32,
    /// `grid_to_world` of `grid_from_world` (the transform round-trip).
    world_roundtrip: [f32; 3],
    /// Padding lane.
    pad_r: f32,
    /// `sample_texel` at the integer texel.
    texel_value: [f32; 3],
    /// Padding lane.
    pad_v: f32,
    /// Central-difference curl at the integer texel.
    curl: [f32; 3],
    /// Padding lane.
    pad_c: f32,
}

/// One query for the vector-field twin against the shared uploaded grid: a
/// continuous grid coordinate and a wrap code for the trilinear sample, a world
/// position for the transform round-trip, and an integer texel coordinate for
/// the index, texel read and derivatives.
///
/// The sampling, transform and derivative paths are independent, so a single
/// query exercises every twinned function at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorFieldQuery {
    /// Continuous grid coordinate fed to the trilinear sampler, matching
    /// [`VectorField::sample_grid`](prism_render_architecture::particle::vector_field::VectorField::sample_grid).
    pub grid: [f32; 3],
    /// Boundary code ([`WRAP_CLAMP`] or [`WRAP_TILE`]) for the sampler fetches,
    /// matching the reference
    /// [`WrapMode`](prism_render_architecture::particle::vector_field::WrapMode).
    pub wrap: u32,
    /// World position fed to
    /// [`VectorFieldTransform::world_to_grid`](prism_render_architecture::particle::vector_field::VectorFieldTransform::world_to_grid).
    pub world: [f32; 3],
    /// Integer texel coordinate fed to
    /// [`VectorField::sample_texel`](prism_render_architecture::particle::vector_field::VectorField::sample_texel),
    /// [`VectorField::linear_index`](prism_render_architecture::particle::vector_field::VectorField::linear_index),
    /// [`VectorField::divergence`](prism_render_architecture::particle::vector_field::VectorField::divergence)
    /// and
    /// [`VectorField::curl`](prism_render_architecture::particle::vector_field::VectorField::curl).
    pub texel: [u32; 3],
}

/// One resolved answer for a single query, mirroring every value the reference
/// reports across its twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorFieldResult {
    /// Trilinear sample, matching
    /// [`VectorField::sample_grid`](prism_render_architecture::particle::vector_field::VectorField::sample_grid).
    pub sampled: [f32; 3],
    /// World-to-grid map of the query world position, matching
    /// [`VectorFieldTransform::world_to_grid`](prism_render_architecture::particle::vector_field::VectorFieldTransform::world_to_grid).
    pub grid_from_world: [f32; 3],
    /// Grid-to-world round-trip of `grid_from_world`, matching
    /// [`VectorFieldTransform::grid_to_world`](prism_render_architecture::particle::vector_field::VectorFieldTransform::grid_to_world).
    pub world_roundtrip: [f32; 3],
    /// Clamped integer texel read, matching
    /// [`VectorField::sample_texel`](prism_render_architecture::particle::vector_field::VectorField::sample_texel).
    pub texel_value: [f32; 3],
    /// Row-major linear index, matching
    /// [`VectorField::linear_index`](prism_render_architecture::particle::vector_field::VectorField::linear_index).
    pub linear_index: u32,
    /// Central-difference divergence, matching
    /// [`VectorField::divergence`](prism_render_architecture::particle::vector_field::VectorField::divergence).
    pub divergence: f32,
    /// Central-difference curl, matching
    /// [`VectorField::curl`](prism_render_architecture::particle::vector_field::VectorField::curl).
    pub curl: [f32; 3],
}

/// Encodes one [`VectorFieldQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &VectorFieldQuery) -> GpuQuery {
    GpuQuery {
        grid: q.grid,
        wrap: q.wrap,
        world: q.world,
        pad_w: 0.0,
        texel: q.texel,
        pad_t: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`VectorFieldResult`].
fn decode_result(raw: &GpuResult) -> VectorFieldResult {
    VectorFieldResult {
        sampled: raw.sampled,
        grid_from_world: raw.grid_from_world,
        world_roundtrip: raw.world_roundtrip,
        texel_value: raw.texel_value,
        linear_index: raw.lindex,
        divergence: raw.divergence,
        curl: raw.curl,
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

/// A compiled, reusable vector-field compute pipeline, twinning the `CPU`
/// golden
/// [`vector_field`](prism_render_architecture::particle::vector_field).
pub struct GpuVectorField {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVectorField {
    /// Compiles the vector-field kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVectorField {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_vector_field"),
            source: ShaderSource::Wgsl(VECTOR_FIELD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_vector_field_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_vector_field_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_vector_field_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVectorField {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` against one shared field and returns one
    /// [`VectorFieldResult`] per input, in order.
    ///
    /// `dims` is the `(X, Y, Z)` texel count, `bounds_min` / `bounds_max` are
    /// the world-space box corners the transform uses, and `texels` is the
    /// row-major (`X`-fastest) grid of `[x, y, z]` vectors, `dims.0 * dims.1 *
    /// dims.2` long. The linear index equals the reference exactly; the sampled
    /// vectors, transform maps and derivatives match to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        dims: (u32, u32, u32),
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        texels: &[[f32; 3]],
        queries: &[VectorFieldQuery],
    ) -> Vec<VectorFieldResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            dims: [dims.0, dims.1, dims.2],
            count: count as u32,
            bounds_min,
            pad0: 0.0,
            bounds_max,
            pad1: 0.0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vector_field_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        // Each texel is padded to a vec4 lane so the storage array stays
        // 16-byte aligned on device.
        let field: Vec<[f32; 4]> = texels.iter().map(|t| [t[0], t[1], t[2], 0.0]).collect();
        let field_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vector_field_grid"),
            contents: bytemuck::cast_slice(&field),
            usage: BufferUsages::STORAGE,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vector_field_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_vector_field_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_vector_field_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: field_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_vector_field_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_vector_field_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_vector_field_pass"),
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
