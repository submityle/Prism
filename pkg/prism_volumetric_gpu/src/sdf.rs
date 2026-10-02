//! `wgpu` compute twin of the discrete 3D scalar signed-distance-field sampling,
//! indexing and boundary-wrap contract
//! ([`sdf`](prism_render_architecture::particle::sdf), particle design §8.3,
//! §10).
//!
//! The `CPU` golden [`sdf`](prism_render_architecture::particle::sdf) owns the
//! pre-baked scalar field a particle reads: a row-major (`X`-fastest) grid of
//! one `f32` signed distance per texel
//! ([`SdfField`](prism_render_architecture::particle::sdf::SdfField)), a boundary
//! policy ([`WrapMode`](prism_render_architecture::particle::sdf::WrapMode))
//! deciding how an out-of-range index resolves, trilinear distance
//! reconstruction
//! ([`SdfField::sample_distance`](prism_render_architecture::particle::sdf::SdfField::sample_distance)),
//! and a gradient-derived grid-space surface normal
//! ([`SdfField::gradient`](prism_render_architecture::particle::sdf::SdfField::gradient)).
//! [`GpuSignedDistanceField`] is the on-device twin: a single read-only grid is
//! uploaded once and every thread samples it, so one thread resolves one query
//! and a passing real-device parity test is direct evidence the ported kernel
//! reproduces the same texels, indices, interpolated distances and normals the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query reproduces, against the one shared grid, every per-sample answer
//! the reference computes: the row-major linear index
//! ([`SdfField::linear_index`](prism_render_architecture::particle::sdf::SdfField::linear_index)),
//! the clamped integer texel read
//! ([`SdfField::sample_texel`](prism_render_architecture::particle::sdf::SdfField::sample_texel)),
//! the trilinear signed distance at a continuous grid coordinate under a wrap
//! code
//! ([`SdfField::sample_distance`](prism_render_architecture::particle::sdf::SdfField::sample_distance)),
//! and the normalised central-difference grid-space surface normal
//! ([`SdfField::gradient`](prism_render_architecture::particle::sdf::SdfField::gradient)).
//! The private `resolve_index` boundary arithmetic and the eight-corner `fetch`
//! are twinned too and are exercised indirectly through the trilinear sampler at
//! out-of-range coordinates under each wrap code.
//!
//! # Boundary codes
//!
//! The reference [`WrapMode`](prism_render_architecture::particle::sdf::WrapMode)
//! has two variants, carried into the kernel as a `u32` code: `0` is
//! [`WRAP_CLAMP`] (clamp a signed index to `[0, dim)`) and `1` is [`WRAP_TILE`]
//! (periodic tiling by the positive-normalised integer modulo
//! `(((i % d) + d) % d)`). The index arithmetic is integer and exact, so the
//! resolved texel address matches the reference bit for bit.
//!
//! # Correctness model
//!
//! The linear index and the resolved texel address are integer, so `CPU` and
//! `GPU` agree exactly and the parity test asserts `==` on the index. The
//! sampled distance and the gradient thread through multiplies, adds, one
//! guarded division and (for the normal) a single `sqrt`, so they are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`) on every continuous quantity, tight enough to catch a
//! dropped term or a swapped axis yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A `dim = 1` axis clamps every texel read to index `0`. A degenerate gradient
//! (a flat region, where the central-difference magnitude is within
//! [`EPS_LEN_SQ`] of zero) normalises to the zero vector rather than dividing by
//! zero, mirroring the reference
//! [`Vec3::normalize_or_zero`](prism_render_architecture::particle::Vec3::normalize_or_zero)
//! guard. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `floor`, `abs`, `sqrt`, `+ - * /` and signed/unsigned index
//! arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse
//! trigonometry and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. The lone `sqrt` sits on the normal path only;
//! the distance path has none. There is no loop: each thread performs a fixed,
//! bounded sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sdf`；无第三方引擎源码或衍生代码。
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
/// [`WrapMode::Clamp`](prism_render_architecture::particle::sdf::WrapMode::Clamp).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sdf`。
pub const WRAP_CLAMP: u32 = 0;

/// Boundary code for the tile policy: a signed index wraps periodically by the
/// positive-normalised integer modulo, matching
/// [`WrapMode::Tile`](prism_render_architecture::particle::sdf::WrapMode::Tile).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sdf`。
pub const WRAP_TILE: u32 = 1;

/// The portable core-`WGSL` signed-distance-field kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden [`sdf`](prism_render_architecture::particle::sdf) branch for
/// branch; see the module documentation for the algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sdf`。
const SDF_WGSL: &str = r#"
// SDF twin: one shared read-only scalar grid, one thread per query. Each thread
// reproduces the linear index, the clamped texel read, the trilinear signed
// distance under a wrap code and the normalised central-difference grid-space
// surface normal. It mirrors the CPU golden `particle::sdf` branch for branch,
// uses only the portable core-WGSL subset (min/max/clamp/floor/abs/sqrt and
// + - * / plus index arithmetic) and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12. The lone sqrt is on the normal path
// only; the distance path has none. There is no loop, so the kernel provably
// terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::sdf；无第三方
// 引擎源码或衍生代码。

// Half-texel finite-difference step for the central-difference gradient.
// Matches the reference `GRAD_STEP`.
const GRAD_STEP: f32 = 0.5;

// Squared-length floor below which a gradient is treated as degenerate and
// normalises to zero. Matches the reference `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

// Boundary codes mirroring the reference `WrapMode`.
const WRAP_CLAMP: u32 = 0u;
const WRAP_TILE: u32 = 1u;

struct Params {
    // Texel counts along (X, Y, Z); every axis is at least 1.
    dims: vec3<u32>,
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
}

struct Query {
    // Continuous grid coordinate fed to `sample_distance` and `gradient`.
    grid: vec3<f32>,
    // Boundary code (WRAP_CLAMP or WRAP_TILE) for the trilinear fetches.
    wrap: u32,
    // Integer texel coordinate fed to `sample_texel` and `linear_index`.
    texel: vec3<u32>,
    pad_t: u32,
}

struct Result {
    // Normalised grid-space surface normal (the gradient).
    gradient: vec3<f32>,
    // sample_distance of the continuous grid coordinate.
    distance: f32,
    // sample_texel at the integer `texel` coordinate.
    texel_value: f32,
    // Row-major linear index of the integer `texel` coordinate.
    lindex: u32,
    pad_a: f32,
    pad_b: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;

// Row-major (X-fastest) linear index of texel (i, j, k); mirrors the reference
// `linear_index`.
fn linear_index(i: u32, j: u32, k: u32) -> u32 {
    let nx = params.dims.x;
    let ny = params.dims.y;
    return ((k * ny) + j) * nx + i;
}

// Stored signed distance at integer texel (i, j, k), clamped into range first;
// mirrors the reference `sample_texel`.
fn sample_texel(i: u32, j: u32, k: u32) -> f32 {
    let nx = params.dims.x;
    let ny = params.dims.y;
    let nz = params.dims.z;
    let ci = min(i, nx - 1u);
    let cj = min(j, ny - 1u);
    let ck = min(k, nz - 1u);
    return field[linear_index(ci, cj, ck)];
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

// Fetches a texel distance by signed coordinates under a boundary code; mirrors
// the reference `fetch` used by the trilinear sampler for the eight corners.
fn fetch(i: i32, j: i32, k: i32, wrap: u32) -> f32 {
    let ri = resolve_index(i, params.dims.x, wrap);
    let rj = resolve_index(j, params.dims.y, wrap);
    let rk = resolve_index(k, params.dims.z, wrap);
    return field[linear_index(ri, rj, rk)];
}

// Trilinearly samples the signed distance at a continuous grid coordinate;
// mirrors the reference `sample_distance` (floor to the eight corners, frac to
// the weights, nested lerp along X then Y then Z in that exact multiply-add
// order).
fn sample_distance(grid: vec3<f32>, wrap: u32) -> f32 {
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

    let c00 = c000 * gx + c100 * fx;
    let c10 = c010 * gx + c110 * fx;
    let c01 = c001 * gx + c101 * fx;
    let c11 = c011 * gx + c111 * fx;

    let c0 = c00 * gy + c10 * fy;
    let c1 = c01 * gy + c11 * fy;

    return c0 * gz + c1 * fz;
}

// Raw (unnormalised) central-difference gradient of the trilinear field in grid
// space; mirrors the reference `grad_grid` (half-texel steps on either side of
// each axis, scaled by 1 / (2 * GRAD_STEP)).
fn grad_grid(grid: vec3<f32>, wrap: u32) -> vec3<f32> {
    let dx = sample_distance(vec3<f32>(grid.x + GRAD_STEP, grid.y, grid.z), wrap)
        - sample_distance(vec3<f32>(grid.x - GRAD_STEP, grid.y, grid.z), wrap);
    let dy = sample_distance(vec3<f32>(grid.x, grid.y + GRAD_STEP, grid.z), wrap)
        - sample_distance(vec3<f32>(grid.x, grid.y - GRAD_STEP, grid.z), wrap);
    let dz = sample_distance(vec3<f32>(grid.x, grid.y, grid.z + GRAD_STEP), wrap)
        - sample_distance(vec3<f32>(grid.x, grid.y, grid.z - GRAD_STEP), wrap);
    let inv = 1.0 / (2.0 * GRAD_STEP);
    return vec3<f32>(dx * inv, dy * inv, dz * inv);
}

// Normalises a vector, returning the zero vector when its squared length is
// within EPS_LEN_SQ of zero; mirrors the reference `Vec3::normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len_sq > EPS_LEN_SQ) {
        let inv = 1.0 / sqrt(len_sq);
        return vec3<f32>(v.x * inv, v.y * inv, v.z * inv);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Unit grid-space surface normal: the normalised central-difference gradient;
// mirrors the reference `gradient`.
fn gradient(grid: vec3<f32>, wrap: u32) -> vec3<f32> {
    return normalize_or_zero(grad_grid(grid, wrap));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let distance = sample_distance(q.grid, q.wrap);
    let grad = gradient(q.grid, q.wrap);
    let texel_value = sample_texel(q.texel.x, q.texel.y, q.texel.z);
    let lindex = linear_index(q.texel.x, q.texel.y, q.texel.z);

    var out: Result;
    out.gradient = grad;
    out.distance = distance;
    out.texel_value = texel_value;
    out.lindex = lindex;
    out.pad_a = 0.0;
    out.pad_b = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the field dimensions and the query
/// count, padded to the `std140` `16`-byte alignment matching `Params` in
/// [`SDF_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Texel counts along `(X, Y, Z)`.
    dims: [u32; 3],
    /// Number of valid queries in the input and output buffers.
    count: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The `vec3` lane carries a trailing pad word so it stays `16`-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Continuous grid coordinate for `sample_distance` and `gradient`.
    grid: [f32; 3],
    /// Boundary code for the trilinear fetches.
    wrap: u32,
    /// Integer texel coordinate for `sample_texel` and `linear_index`.
    texel: [u32; 3],
    /// Pad lane after `texel`.
    pad_t: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Normalised grid-space surface normal (the gradient).
    gradient: [f32; 3],
    /// Trilinear signed distance.
    distance: f32,
    /// Clamped integer texel read.
    texel_value: f32,
    /// Row-major linear index of the integer texel.
    lindex: u32,
    /// Padding lane.
    pad_a: f32,
    /// Padding lane.
    pad_b: f32,
}

/// One query for the signed-distance-field twin against the shared uploaded
/// grid: a continuous grid coordinate and a wrap code for the trilinear sample
/// and gradient, and an integer texel coordinate for the index and texel read.
///
/// The sampling and texel paths are independent, so a single query exercises
/// every twinned function at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfQuery {
    /// Continuous grid coordinate fed to the trilinear sampler and gradient,
    /// matching
    /// [`SdfField::sample_distance`](prism_render_architecture::particle::sdf::SdfField::sample_distance)
    /// and
    /// [`SdfField::gradient`](prism_render_architecture::particle::sdf::SdfField::gradient).
    pub grid: [f32; 3],
    /// Boundary code ([`WRAP_CLAMP`] or [`WRAP_TILE`]) for the sampler fetches,
    /// matching the reference
    /// [`WrapMode`](prism_render_architecture::particle::sdf::WrapMode).
    pub wrap: u32,
    /// Integer texel coordinate fed to
    /// [`SdfField::sample_texel`](prism_render_architecture::particle::sdf::SdfField::sample_texel)
    /// and
    /// [`SdfField::linear_index`](prism_render_architecture::particle::sdf::SdfField::linear_index).
    pub texel: [u32; 3],
}

/// One resolved answer for a single query, mirroring every value the reference
/// reports across its twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfResult {
    /// Trilinear signed distance, matching
    /// [`SdfField::sample_distance`](prism_render_architecture::particle::sdf::SdfField::sample_distance).
    pub distance: f32,
    /// Normalised grid-space surface normal, matching
    /// [`SdfField::gradient`](prism_render_architecture::particle::sdf::SdfField::gradient).
    pub gradient: [f32; 3],
    /// Clamped integer texel read, matching
    /// [`SdfField::sample_texel`](prism_render_architecture::particle::sdf::SdfField::sample_texel).
    pub texel_value: f32,
    /// Row-major linear index, matching
    /// [`SdfField::linear_index`](prism_render_architecture::particle::sdf::SdfField::linear_index).
    pub linear_index: u32,
}

/// Encodes one [`SdfQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfQuery) -> GpuQuery {
    GpuQuery {
        grid: q.grid,
        wrap: q.wrap,
        texel: q.texel,
        pad_t: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfResult`].
fn decode_result(raw: &GpuResult) -> SdfResult {
    SdfResult {
        distance: raw.distance,
        gradient: raw.gradient,
        texel_value: raw.texel_value,
        linear_index: raw.lindex,
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

/// A compiled, reusable signed-distance-field compute pipeline, twinning the
/// `CPU` golden [`sdf`](prism_render_architecture::particle::sdf).
pub struct GpuSignedDistanceField {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSignedDistanceField {
    /// Compiles the signed-distance-field kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSignedDistanceField {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf"),
            source: ShaderSource::Wgsl(SDF_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSignedDistanceField {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` against one shared field and returns one
    /// [`SdfResult`] per input, in order.
    ///
    /// `dims` is the `(X, Y, Z)` texel count and `texels` is the row-major
    /// (`X`-fastest) grid of `f32` signed distances, `dims.0 * dims.1 * dims.2`
    /// long. The linear index equals the reference exactly; the sampled
    /// distance, texel read and gradient match to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        dims: (u32, u32, u32),
        texels: &[f32],
        queries: &[SdfQuery],
    ) -> Vec<SdfResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            dims: [dims.0, dims.1, dims.2],
            count: count as u32,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let field_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_grid"),
            contents: bytemuck::cast_slice(texels),
            usage: BufferUsages::STORAGE,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_bind_group"),
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
            label: Some("prism_volumetric_sdf_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_pass"),
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
