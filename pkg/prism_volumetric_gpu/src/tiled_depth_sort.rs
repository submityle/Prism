//! `wgpu` compute twin of the per-particle `tile` assignment half of the
//! blocked transparency depth sort
//! ([`tiled_depth_sort`](prism_render_architecture::particle::tiled_depth_sort),
//! design §12).
//!
//! The blocked depth sort buckets transparent particles into screen-space
//! `tile`s (optionally sub-divided into depth `slice`s) and orders each bucket
//! locally. The bucketing decision — which grid cell a single particle maps to
//! — is the one piece that is embarrassingly parallel (one thread per
//! particle), so it is the piece this crate twins. The `CPU` golden
//! [`tile_of`](prism_render_architecture::particle::tiled_depth_sort::tile_of)
//! (and its private `axis_bucket` helper) owns that math; [`GpuTileOf`] is the
//! on-device twin that runs one thread per query and reproduces every lane. A
//! passing real-device parity test is therefore direct evidence the ported
//! kernel folds the same per-axis bucket guards the reference does, not merely
//! that its shader compiles.
//!
//! # What is twinned
//!
//! Only the per-particle `tile_of` is twinned. The variable-length
//! `tiled_depth_sort` itself (a stable counting sort plus per-`tile` local
//! ordering) stays host-only: it is a reduction over the whole batch, not a
//! per-element map, so it does not fit the one-thread-one-element dispatch
//! shape and is out of scope here.
//!
//! The kernel reproduces `axis_bucket` guard-for-guard on each of the three
//! axes: a count of one collapses the axis to bucket `0`; a `NaN` or
//! non-positive span collapses to bucket `0`; a `NaN` or non-positive
//! normalized coordinate `t` lands in bucket `0`; a `t` at or past the top edge
//! lands in the last bucket; otherwise the truncating cast of `t * count` picks
//! the bucket, re-clamped against rounding at the top. The three buckets are
//! then flattened column-fastest (`(z * tiles_y + y) * tiles_x + x`) exactly as
//! the reference does.
//!
//! `WGSL` has no `isNan`, so the kernel reproduces the reference `f32::is_nan`
//! guard with a `bitcast`: a value is `NaN` when its exponent field is all ones
//! and its mantissa field is non-zero. That test is pure `u32` bit algebra, so
//! it uses integer `==`/`!=` (never an `f32` equality).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max`, integer and
//! `f32` arithmetic, comparisons, bit masking and one `bitcast` for the `NaN`
//! probe — with no `sin`, `cos`, `exp`, `log`, `pow`, `sqrt` or optional device
//! feature, so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Every output is a discrete bucket index: there are no continuous values to
//! compare. The per-axis map is a fixed sequence of guards and one truncating
//! cast, so `CPU` and `GPU` agree exactly whenever the query is placed clear of
//! a bucket boundary (where a legal `ULP`-scale perturbation of `t * count`
//! could flip the truncation across an integer). The parity test therefore
//! asserts an *exact* `==` on all four output words and keeps its fixtures away
//! from those integer boundaries by rejection sampling.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓
//! `prism_render_architecture::particle::tiled_depth_sort`；standard
//! screen-space `tile` bucketing plus `wgpu` compute dispatch; no third-party
//! engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::tiled_depth_sort::{TileGridParams, TiledParticle};
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

/// The portable core-`WGSL` per-particle `tile`-assignment kernel, embedded
/// inline so the twin ships as a single source file. Mirrors the `CPU` golden
/// [`tile_of`](prism_render_architecture::particle::tiled_depth_sort::tile_of)
/// and its private `axis_bucket` helper guard-for-guard; see the module
/// documentation for the algorithm.
const TILE_OF_WGSL: &str = r#"
// Per-particle tile assignment twin: one thread per (grid params, particle)
// query maps the particle's screen x/y and depth into integer buckets on the
// three grid axes, then flattens them column-fastest into the tile index. It
// mirrors the CPU golden `particle::tiled_depth_sort::axis_bucket`/`tile_of`
// guard for guard, uses only the portable core-WGSL subset (max, integer and
// f32 arithmetic, comparisons, bit masking and one bitcast for the NaN probe),
// and takes no optional feature, so it runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: standard screen-space tile bucketing; no third-party engine
// source or derived code.

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 64-byte std430 stride matching the host `GpuQuery`: the screen
// rect, the three axis counts, the near/far depth range and the particle
// position, each packed into a vec4 slot so the storage array needs no manual
// alignment arithmetic.
struct Query {
    // (min_x, min_y, max_x, max_y) of the screen rect.
    rect: vec4<f32>,
    // (tiles_x, tiles_y, depth_slices, pad); clamped to >= 1 in the kernel.
    counts: vec4<u32>,
    // (near, far, pad, pad) depth range.
    near_far: vec4<f32>,
    // (screen_x, screen_y, depth, pad) particle position.
    pos: vec4<f32>,
}

// One result. 16-byte std430 stride matching the host `GpuResult`: the three
// bucket indices and the flattened tile index.
struct TileResult {
    x: u32,
    y: u32,
    z: u32,
    index: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<TileResult>;

// Reproduces the reference `f32::is_nan`: an IEEE-754 f32 is NaN when its
// exponent field is all ones and its mantissa field is non-zero. Pure u32 bit
// algebra, so the comparisons are integer `==`/`!=`, never an f32 equality.
fn is_nan_f32(value: f32) -> bool {
    let bits = bitcast<u32>(value);
    return ((bits & 0x7f800000u) == 0x7f800000u) && ((bits & 0x007fffffu) != 0u);
}

// Maps a coordinate on one axis to a bucket in `0..count`, mirroring the
// private golden `axis_bucket`: a NaN coordinate or anything at/below the
// minimum lands in bucket 0; a coordinate at/above the maximum lands in the
// last bucket; a degenerate span collapses to bucket 0.
fn axis_bucket(coord: f32, lo: f32, hi: f32, count: u32) -> u32 {
    if (count <= 1u) {
        return 0u;
    }
    let span = hi - lo;
    if (is_nan_f32(span) || span <= 0.0) {
        return 0u;
    }
    let t = (coord - lo) / span;
    if (is_nan_f32(t) || t <= 0.0) {
        return 0u;
    }
    if (t >= 1.0) {
        return count - 1u;
    }
    // `t` is in the open interval (0, 1), so the truncating cast yields a
    // bucket in `0..=count-1`; the guard keeps it in range against rounding at
    // the top.
    let bucket = u32(t * f32(count));
    if (bucket >= count) {
        return count - 1u;
    }
    return bucket;
}

@compute @workgroup_size(64)
fn assign(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    let nx = max(q.counts.x, 1u);
    let ny = max(q.counts.y, 1u);
    let nz = max(q.counts.z, 1u);

    let x = axis_bucket(q.pos.x, q.rect.x, q.rect.z, nx);
    let y = axis_bucket(q.pos.y, q.rect.y, q.rect.w, ny);
    let z = axis_bucket(q.pos.z, q.near_far.x, q.near_far.y, nz);

    var out: TileResult;
    out.x = x;
    out.y = y;
    out.z = z;
    out.index = (z * ny + y) * nx + x;
    results[idx] = out;
}
"#;

/// One `tile`-assignment query: the grid parameters and the particle to place.
///
/// Mirrors a single reference
/// [`tile_of`](prism_render_architecture::particle::tiled_depth_sort::tile_of)
/// call. Carrying the grid per query lets one dispatch mix particles against
/// many distinct grids. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it
/// holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileOfQuery {
    /// The grid the particle is bucketed into.
    pub params: TileGridParams,
    /// The particle whose screen position and depth pick the grid cell.
    pub particle: TiledParticle,
}

/// The resolved grid cell for one query, the host-side mirror of the kernel's
/// `TileResult` lane and of the golden
/// [`TileCoord`](prism_render_architecture::particle::tiled_depth_sort::TileCoord).
///
/// Every field is a discrete bucket index, so this derives [`Eq`] and [`Hash`]
/// and the parity test compares it with an exact `==`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TileOfResult {
    /// Column (`0..tiles_x`).
    pub x: u32,
    /// Row (`0..tiles_y`).
    pub y: u32,
    /// Depth `slice` (`0..depth_slices`).
    pub z: u32,
    /// Flattened `tile` id, column-fastest.
    pub index: u32,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`TILE_OF_WGSL`]: the query count and three pad words — `16`
/// bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `64`-byte `std430` stride matching `Query` in the
/// shader: the screen rect, the three axis counts, the near/far range and the
/// particle position, each padded to a `vec4` slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `(min_x, min_y, max_x, max_y)` of the screen rect.
    rect: [f32; 4],
    /// `(tiles_x, tiles_y, depth_slices, pad)`; clamped to at least one in the
    /// kernel so a zero count degenerates the axis to a single bucket.
    counts: [u32; 4],
    /// `(near, far, pad, pad)` depth range.
    near_far: [f32; 4],
    /// `(screen_x, screen_y, depth, pad)` particle position.
    pos: [f32; 4],
}

impl GpuQuery {
    /// Packs a [`TileOfQuery`] into the `std430` upload layout.
    ///
    /// The axis counts are forwarded raw (not pre-clamped): the kernel applies
    /// the `.max(1)` that the golden `effective_*` accessors apply, so the
    /// clamp happens in exactly one place on both paths.
    fn from_query(query: &TileOfQuery) -> GpuQuery {
        let p = query.params;
        let rect = p.rect;
        let part = query.particle;
        GpuQuery {
            rect: [rect.min_x, rect.min_y, rect.max_x, rect.max_y],
            counts: [p.tiles_x, p.tiles_y, p.depth_slices, 0],
            near_far: [p.near, p.far, 0.0, 0.0],
            pos: [part.screen_x, part.screen_y, part.depth, 0.0],
        }
    }
}

/// One result as read back. `16`-byte `std430` stride matching `TileResult` in
/// the shader: the three bucket indices and the flattened `tile` id.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Column bucket.
    x: u32,
    /// Row bucket.
    y: u32,
    /// Depth `slice` bucket.
    z: u32,
    /// Flattened `tile` id.
    index: u32,
}

/// Maps one kernel `TileResult` lane back to the host [`TileOfResult`].
fn decode_result(raw: &GpuResult) -> TileOfResult {
    TileOfResult {
        x: raw.x,
        y: raw.y,
        z: raw.z,
        index: raw.index,
    }
}

/// A compiled, reusable per-particle `tile`-assignment pipeline.
pub struct GpuTileOf {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTileOf {
    /// Compiles the `tile`-assignment kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTileOf {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_tiled_depth_sort"),
            source: ShaderSource::Wgsl(TILE_OF_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_tiled_depth_sort_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_tiled_depth_sort_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_tiled_depth_sort_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("assign"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTileOf {
            module,
            layout,
            pipeline,
        }
    }

    /// Assigns every query in `queries` to its grid cell, returning one
    /// [`TileOfResult`] per query in input order.
    ///
    /// The returned result for query `q` mirrors
    /// [`tile_of`](prism_render_architecture::particle::tiled_depth_sort::tile_of)
    /// evaluated on `q.params` and `q.particle`. An empty `queries` slice yields
    /// an empty result — storage buffers cannot be zero-sized, so it is handled
    /// by an early return before any dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[TileOfQuery]) -> Vec<TileOfResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_tiled_depth_sort_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_tiled_depth_sort_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_tiled_depth_sort_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_tiled_depth_sort_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_tiled_depth_sort_bind_group"),
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
            label: Some("prism_volumetric_tiled_depth_sort_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_tiled_depth_sort_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
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
