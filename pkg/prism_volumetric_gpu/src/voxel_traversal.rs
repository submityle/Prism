#![forbid(unsafe_code)]
//! `wgpu` compute twin of the `Amanatides-Woo` integer voxel-grid ray
//! traversal golden
//! ([`voxel_traversal`](prism_render_architecture::particle::voxel_traversal),
//! particle design §7, §10, §29).
//!
//! The `CPU` golden
//! [`traverse`](prism_render_architecture::particle::voxel_traversal::traverse)
//! owns the small, verifiable contract several particle stages share: it turns
//! a ray plus a uniform [`VoxelGrid`](prism_render_architecture::particle::voxel_traversal::VoxelGrid)
//! into the ordered list of `(voxel, t_enter)` pairs the ray crosses, front to
//! back, bounded by a world-space distance, a voxel count, or both.
//! [`GpuVoxelTraversal`] is the on-device twin: one thread walks one ray, so a
//! passing real-device parity test is direct evidence the ported kernel folds
//! the same direction normalization, the same `floor`-to-`i32` voxel snap, the
//! same per-axis `seed`, the same minimum-`t_max` step selection and the same
//! distance/step termination the reference does, not merely that its shader
//! compiles.
//!
//! # What is twinned
//!
//! For each ray the kernel reproduces the whole reference walk: it normalizes
//! the direction with [`f32::sqrt`] (a degenerate ray shorter than
//! [`EPS`](prism_render_architecture::particle::voxel_traversal::EPS) yields an
//! empty walk), converts the origin and direction to grid space (dividing by a
//! per-axis cell size floored at `EPS`), seeds each axis' `step`, `t_max` and
//! `t_delta` (a direction component below `EPS` marks the axis *parallel*, with
//! both `t_max` and `t_delta` set to the `IEEE`-754 `+inf` sentinel so the
//! minimum selection never picks it), emits the origin voxel at `t = 0` and
//! then advances whichever axis has the nearest upcoming boundary, pushing that
//! axis' `t_max` forward by its `t_delta`. Ties resolve `x`, then `y`, then `z`
//! exactly as the reference. The walk stops when the next entry parameter would
//! exceed `max_distance`, when `max_steps` voxels have been emitted, when all
//! axes are parallel (`+inf`), or immediately for a degenerate ray, a zero step
//! cap, a negative distance bound or a fully-unbounded limit.
//!
//! # Layout
//!
//! Each query uploads the ray origin and direction, the grid origin and cell
//! size, and the limit (an optional world `max_distance` and an optional
//! `max_steps`, each with a presence flag). Each result carries the emitted
//! voxel count plus fixed-capacity [`MAX_HITS`] slots of integer voxel
//! coordinates and world-distance entry parameters; unused slots are zeroed.
//! The host passes `max_steps` no greater than [`MAX_HITS`], and the kernel
//! bounds its loop by [`MAX_HITS`] regardless, so the walk provably terminates
//! and never overflows the output.
//!
//! # Correctness model
//!
//! Each ray is a fixed, non-reorderable sequence of guarded divisions and
//! incremental additions, so `CPU` and `GPU` evaluate the same closed form in
//! the same associativity. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts an *exact* match on the emitted count and on every integer voxel
//! coordinate (the discrete `floor`-to-`i32` snap is routed through the same
//! `EPS` grid-space arithmetic the reference uses) yet a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the world-distance `t_enter`
//! values, with fixtures kept clear of voxel-face boundaries so the integer
//! path never flips under a legal perturbation.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `floor`, `sqrt`, `select`, `+ - * /` and two `bitcast`s (one for the
//! `IEEE`-754 `+inf` sentinel that mirrors the reference's `f32::INFINITY`
//! bounds, one for the `i32::MIN` saturation constant) — with no `sin`, `cos`,
//! `exp`, `log`, `pow` or optional device feature, so it runs unmodified on
//! Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::voxel_traversal`；
//! classic `Amanatides-Woo` fast voxel-traversal grid `DDA` plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::voxel_traversal::{Limit, Ray, VoxelGrid, VoxelHit};
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

/// Maximum number of voxels one ray may emit, the fixed output capacity per
/// lane. The host passes a `max_steps` no greater than this and the kernel
/// bounds its loop by it, so the walk provably terminates and never overruns
/// the result slots.
///
/// Provenance: fixed-capacity upper bound for the twin of
/// `prism_render_architecture::particle::voxel_traversal`.
pub const MAX_HITS: usize = 64;

/// The portable core-`WGSL` `Amanatides-Woo` voxel-traversal kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`traverse`](prism_render_architecture::particle::voxel_traversal::traverse)
/// step for step; see the module documentation for the algorithm.
const VOXEL_TRAVERSAL_WGSL: &str = r#"
// Amanatides-Woo integer voxel-grid ray traversal twin: one thread walks one
// ray through a uniform grid and writes the ordered (voxel, t_enter) pairs it
// crosses. It normalizes the direction with sqrt, snaps the origin to a voxel
// with floor, seeds each axis' step/t_max/t_delta (a parallel axis holds the
// +inf sentinel), emits the origin voxel at t = 0 and then advances the axis
// with the nearest boundary, pushing that axis' t_max forward by t_delta, until
// a distance or step bound is reached. It mirrors the CPU golden
// `particle::voxel_traversal` step for step, uses only the portable core-WGSL
// subset (abs/min/max/floor/sqrt/select, + - * / and two bitcasts for the +inf
// and i32::MIN constants), needs no transcendental call and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12. Each thread emits at
// most MAX_HITS voxels, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::voxel_traversal；无第三方
// 引擎源码或衍生代码。

const MAX_HITS: u32 = 64u;

// Magnitude floor matching the reference `EPS`: a direction component (world or
// grid space) whose magnitude is below it is parallel (no boundary crossing),
// and each cell size is floored at it so a division can never be by zero. The
// compare rule used instead of an f32 `==`.
const EPS: f32 = 1.0e-6;

struct Params {
    // Number of valid queries; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Ray origin in xyz; the w lane is unused padding.
    origin: vec4<f32>,
    // Ray direction in xyz (any non-zero length; normalized on device).
    dir: vec4<f32>,
    // Grid (0,0,0)-corner world position in xyz.
    grid_origin: vec4<f32>,
    // Per-axis voxel edge length in xyz (floored at EPS on device).
    cell: vec4<f32>,
    // World-distance bound, meaningful only when has_distance is 1.
    max_distance: f32,
    // Voxel-count bound, meaningful only when has_steps is 1.
    max_steps: u32,
    // 1 when a distance bound is present, 0 otherwise.
    has_distance: u32,
    // 1 when a step bound is present, 0 otherwise.
    has_steps: u32,
}

struct Result {
    // Number of emitted voxels (0 on an empty walk).
    hit_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Visited voxel coordinates in order (xyz used, w padding); zeroed past
    // hit_count.
    voxels: array<vec4<i32>, 64>,
    // World-distance entry parameter per voxel; zeroed past hit_count.
    t_enters: array<f32, 64>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Floors `v` to the largest i32 not greater than it, saturating on out-of-range
// input exactly as the reference `floor_i32`. i32::MIN is materialized by
// bitcast because its token overflows an i32 literal.
fn floor_i32(v: f32) -> i32 {
    let f = floor(v);
    if (f >= 2147483648.0) {
        return 2147483647;
    }
    if (f <= -2147483648.0) {
        return bitcast<i32>(0x80000000u);
    }
    return i32(f);
}

// Per-axis DDA state: the current voxel index, the +1/-1/0 step direction, the
// ray parameter of the next boundary crossing and the per-boundary delta.
struct Axis {
    coord: i32,
    step: i32,
    t_max: f32,
    t_delta: f32,
}

// Seeds one axis from the grid-space origin coordinate and grid-space direction
// component, mirroring the reference `AxisState::seed`. A component magnitude
// below EPS marks the axis parallel: step 0 and both t_max/t_delta set to the
// +inf sentinel the minimum selection never picks.
fn seed(pos: f32, dir: f32) -> Axis {
    var a: Axis;
    a.coord = floor_i32(pos);
    let pos_inf = bitcast<f32>(0x7f800000u);
    if (abs(dir) < EPS) {
        a.step = 0;
        a.t_max = pos_inf;
        a.t_delta = pos_inf;
        return a;
    }
    let frac = pos - f32(a.coord);
    if (dir > 0.0) {
        let dist_to_boundary = 1.0 - frac;
        a.step = 1;
        a.t_max = dist_to_boundary / dir;
        a.t_delta = 1.0 / dir;
        return a;
    }
    let speed = -dir;
    a.step = -1;
    a.t_max = frac / speed;
    a.t_delta = 1.0 / speed;
    return a;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    var res: Result;
    res.hit_count = 0u;
    res.pad0 = 0u;
    res.pad1 = 0u;
    res.pad2 = 0u;
    for (var i = 0u; i < MAX_HITS; i = i + 1u) {
        res.voxels[i] = vec4<i32>(0, 0, 0, 0);
        res.t_enters[i] = 0.0;
    }

    let has_distance = queries[idx].has_distance != 0u;
    let has_steps = queries[idx].has_steps != 0u;

    // A fully-unbounded limit cannot be safely enumerated: refuse by contract.
    if (!has_distance && !has_steps) {
        results[idx] = res;
        return;
    }
    // A zero step cap can never emit a voxel.
    if (has_steps && queries[idx].max_steps == 0u) {
        results[idx] = res;
        return;
    }
    // A negative distance bound admits no voxel entry.
    if (has_distance && queries[idx].max_distance < 0.0) {
        results[idx] = res;
        return;
    }

    // Normalize the direction so t reads as world distance; a length below EPS
    // is the degenerate ray the reference maps to the zero vector, an empty walk.
    let d = queries[idx].dir.xyz;
    let len = sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    if (len < EPS) {
        results[idx] = res;
        return;
    }
    let dir = d * (1.0 / len);

    // Work in grid space (cell units); each cell size is floored at EPS so the
    // per-axis divisions are safe.
    let cell = vec3<f32>(
        max(queries[idx].cell.x, EPS),
        max(queries[idx].cell.y, EPS),
        max(queries[idx].cell.z, EPS),
    );
    let rel = queries[idx].origin.xyz - queries[idx].grid_origin.xyz;
    let origin_g = vec3<f32>(rel.x / cell.x, rel.y / cell.y, rel.z / cell.z);
    let dir_g = vec3<f32>(dir.x / cell.x, dir.y / cell.y, dir.z / cell.z);

    var ax = seed(origin_g.x, dir_g.x);
    var ay = seed(origin_g.y, dir_g.y);
    var az = seed(origin_g.z, dir_g.z);

    let pos_inf = bitcast<f32>(0x7f800000u);
    let max_distance = select(pos_inf, queries[idx].max_distance, has_distance);
    var cap = MAX_HITS;
    if (has_steps) {
        cap = min(queries[idx].max_steps, MAX_HITS);
    }

    // Emit the origin voxel first (entered at t = 0), respecting the caps.
    res.voxels[0] = vec4<i32>(ax.coord, ay.coord, az.coord, 0);
    res.t_enters[0] = 0.0;
    var n = 1u;

    loop {
        if (n >= cap) {
            break;
        }
        let next_t = min(ax.t_max, min(ay.t_max, az.t_max));
        // No finite boundary remains (all axes parallel): the ray never enters
        // another voxel.
        if (next_t >= pos_inf) {
            break;
        }
        // Stepping past the distance bound ends the walk.
        if (next_t > max_distance) {
            break;
        }

        // Select and step the axis with the nearest boundary; ties resolve
        // x, then y, then z, exactly as the reference.
        if (ax.t_max <= ay.t_max && ax.t_max <= az.t_max) {
            ax.coord = ax.coord + ax.step;
            ax.t_max = ax.t_max + ax.t_delta;
        } else if (ay.t_max <= az.t_max) {
            ay.coord = ay.coord + ay.step;
            ay.t_max = ay.t_max + ay.t_delta;
        } else {
            az.coord = az.coord + az.step;
            az.t_max = az.t_max + az.t_delta;
        }

        res.voxels[n] = vec4<i32>(ax.coord, ay.coord, az.coord, 0);
        res.t_enters[n] = next_t;
        n = n + 1u;
    }

    res.hit_count = n;
    results[idx] = res;
}
"#;

/// One traversal query: a ray, the uniform grid it walks and the limit that
/// bounds the walk.
///
/// Reuses the golden
/// [`Ray`](prism_render_architecture::particle::voxel_traversal::Ray),
/// [`VoxelGrid`](prism_render_architecture::particle::voxel_traversal::VoxelGrid)
/// and [`Limit`](prism_render_architecture::particle::voxel_traversal::Limit)
/// so one dispatch can mix distinct rays, grids and bounds. The host passes a
/// `Limit::max_steps` no greater than [`MAX_HITS`]; a larger cap is clamped on
/// upload. Holds `f32` geometry, so it derives only [`PartialEq`] (no
/// `Eq`/`Hash`).
///
/// Provenance: twin query of
/// `prism_render_architecture::particle::voxel_traversal`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelTraversalQuery {
    /// The ray whose grid walk is traced (direction normalized on device).
    pub ray: Ray,
    /// The uniform voxel grid the ray is walked through.
    pub grid: VoxelGrid,
    /// How the walk is bounded (world distance, voxel count, or both).
    pub limit: Limit,
}

/// The resolved walk for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `hit_count` is the number of valid entries in `voxels` and `t_enters`
/// (`0` on an empty walk); only the first `hit_count` slots are meaningful, the
/// rest are zeroed. `voxels[i]` is the integer voxel coordinate entered at
/// world distance `t_enters[i]`, in strictly front-to-back order. Holds `f32`
/// parameters, so it derives only [`PartialEq`] (no `Eq`/`Hash`).
///
/// Provenance: twin result of
/// `prism_render_architecture::particle::voxel_traversal`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelTraversalResult {
    /// Number of emitted voxels; only the first `hit_count` slots are valid.
    pub hit_count: u32,
    /// Visited voxel coordinates in order; unused slots are zeroed.
    pub voxels: [[i32; 3]; MAX_HITS],
    /// World-distance entry parameter per voxel; unused slots are zeroed.
    pub t_enters: [f32; MAX_HITS],
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`VOXEL_TRAVERSAL_WGSL`]: the query count and three pad words,
/// `16` bytes.
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

/// One query as uploaded. `repr(C)` `std430` layout matching `Query` in the
/// shader: the ray origin and direction and the grid origin and cell size each
/// padded to a `vec4` lane, then the limit (distance, steps and two presence
/// flags).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin in `xyz`; the `w` lane is unused padding.
    origin: [f32; 4],
    /// Ray direction in `xyz`; the `w` lane is unused padding.
    dir: [f32; 4],
    /// Grid `(0,0,0)`-corner world position in `xyz`; `w` is padding.
    grid_origin: [f32; 4],
    /// Per-axis voxel edge length in `xyz`; `w` is padding.
    cell: [f32; 4],
    /// World-distance bound, meaningful only when `has_distance` is `1`.
    max_distance: f32,
    /// Voxel-count bound, meaningful only when `has_steps` is `1`.
    max_steps: u32,
    /// `1` when a distance bound is present, `0` otherwise.
    has_distance: u32,
    /// `1` when a step bound is present, `0` otherwise.
    has_steps: u32,
}

/// One result as read back. `repr(C)` `std430` layout matching `Result` in the
/// shader: the emitted voxel count, three pad words, the fixed `vec4<i32>` voxel
/// slots and the fixed `f32` entry-parameter slots.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Number of emitted voxels.
    hit_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Visited voxel coordinates (`xyz` used, `w` padding); zeroed past
    /// `hit_count`.
    voxels: [[i32; 4]; MAX_HITS],
    /// World-distance entry parameter per voxel; zeroed past `hit_count`.
    t_enters: [f32; MAX_HITS],
}

/// Encodes one [`VoxelTraversalQuery`] into its `std430` [`GpuQuery`] slot,
/// splitting the [`Limit`](prism_render_architecture::particle::voxel_traversal::Limit)
/// into presence flags plus values and clamping the step cap to [`MAX_HITS`].
fn encode_query(q: &VoxelTraversalQuery) -> GpuQuery {
    let o = q.ray.origin;
    let d = q.ray.dir;
    let go = q.grid.origin;
    let c = q.grid.cell;
    let (has_distance, max_distance) = match q.limit.max_distance {
        Some(dist) => (1u32, dist),
        None => (0u32, 0.0),
    };
    let (has_steps, max_steps) = match q.limit.max_steps {
        Some(steps) => (1u32, steps.min(MAX_HITS) as u32),
        None => (0u32, 0u32),
    };
    GpuQuery {
        origin: [o.x, o.y, o.z, 0.0],
        dir: [d.x, d.y, d.z, 0.0],
        grid_origin: [go.x, go.y, go.z, 0.0],
        cell: [c.x, c.y, c.z, 0.0],
        max_distance,
        max_steps,
        has_distance,
        has_steps,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`VoxelTraversalResult`],
/// dropping the `w` padding lane from each voxel and zero-filling slots past
/// `hit_count`.
fn decode_result(raw: &GpuResult) -> VoxelTraversalResult {
    let n = (raw.hit_count as usize).min(MAX_HITS);
    let mut voxels = [[0i32; 3]; MAX_HITS];
    let mut t_enters = [0.0f32; MAX_HITS];
    for i in 0..n {
        voxels[i] = [raw.voxels[i][0], raw.voxels[i][1], raw.voxels[i][2]];
        t_enters[i] = raw.t_enters[i];
    }
    VoxelTraversalResult {
        hit_count: n as u32,
        voxels,
        t_enters,
    }
}

/// The `CPU` golden walk for one query, dispatching to the reference
/// [`traverse`](prism_render_architecture::particle::voxel_traversal::traverse)
/// so callers (and the parity test) can pin the twin voxel for voxel.
#[must_use]
pub fn cpu_reference(query: &VoxelTraversalQuery) -> Vec<VoxelHit> {
    prism_render_architecture::particle::voxel_traversal::traverse(
        query.ray,
        query.grid,
        query.limit,
    )
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

/// A compiled, reusable `Amanatides-Woo` voxel-traversal pipeline.
pub struct GpuVoxelTraversal {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVoxelTraversal {
    /// Compiles the voxel-traversal kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVoxelTraversal {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_voxel_traversal"),
            source: ShaderSource::Wgsl(VOXEL_TRAVERSAL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_voxel_traversal_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_voxel_traversal_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_voxel_traversal_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVoxelTraversal {
            module,
            layout,
            pipeline,
        }
    }

    /// Walks every query in `queries`, returning one [`VoxelTraversalResult`]
    /// per query in input order.
    ///
    /// The returned result for query `q` mirrors
    /// [`traverse`](prism_render_architecture::particle::voxel_traversal::traverse)
    /// evaluated on `q.ray`, `q.grid` and `q.limit`. An empty `queries` slice
    /// yields an empty result — storage buffers cannot be zero-sized, so it is
    /// handled by an early return before any dispatch.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[VoxelTraversalQuery],
    ) -> Vec<VoxelTraversalResult> {
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
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(encode_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_voxel_traversal_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_voxel_traversal_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_voxel_traversal_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_voxel_traversal_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_voxel_traversal_bind_group"),
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
            label: Some("prism_volumetric_voxel_traversal_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_voxel_traversal_pass"),
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
