//! `wgpu` compute twin of the sphere-traced signed-distance-field ray marcher
//! of the `CPU` golden path — `sphere_trace` in
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch`, which advances a
//! ray through a discrete field's world-space bounding box, sampling the
//! continuous trilinear signed distance at every step and converging on the
//! first surface crossing.
//!
//! Sphere tracing is the canonical way an `AAA` renderer consumes a signed
//! distance field: because the sampled value at any point is a conservative
//! lower bound on the distance to the nearest surface, a ray can safely step by
//! exactly that value and never overshoot the first hit. The same traversal
//! drives distance-field soft shadows, ambient occlusion and global-illumination
//! cone stepping. [`GpuSdfSphereTrace`] is the on-device twin: each thread reads
//! one ray and writes back the hit flag, the ray parameter, the hit position,
//! the sampled distance there and the step count.
//!
//! # What is twinned
//!
//! The kernel reproduces the reference step for step. It first normalizes the
//! direction (a zero-length direction misses). It then clips the ray against
//! the grid's axis-aligned bounding box with the same slab method
//! (`t_enter >= 0`, exit clamped to `max_distance`), misses when the clipped
//! interval is empty, and otherwise marches from `t_enter`: at each step it
//! samples the trilinear signed distance at `origin + t * direction`, reports a
//! hit the instant the sample drops to or below `hit_epsilon` (with
//! `steps = step + 1`), advances `t` by `max(distance, voxel_size * 0.125)` so a
//! near-zero sample cannot freeze the march, and misses once `t` leaves the
//! exit parameter or the step budget is spent.
//!
//! The field *construction* (the Euclidean transform, the solid classification
//! and the `signed_squared` integer storage of
//! `prism_render_architecture::ray_scene::mesh_signed_distance_field`) stays on
//! the host: the twin consumes a flat `f32` array of per-cell signed distances
//! uploaded as a read-only storage buffer, which is the only data the reference
//! sampler ever reads.
//!
//! # Correctness model
//!
//! Sampling is linear interpolation (`+`, `-`, `*`) plus integer `clamp`
//! addressing; the slab clip is a handful of divides and `min`/`max`; direction
//! normalization uses a single `sqrt`. `CPU` and `GPU` evaluate the same closed
//! form but need not be bit-exact — a `GPU` may contract a multiply-add — so the
//! parity test asserts `abs_diff <= 1e-4 || rel_diff <= 1e-3` on the ray
//! parameter, the position and the distance. The `hit` flag and the `steps`
//! count are integer decisions and match with no tolerance; the parity sweep
//! rejects rays whose march grazes the hit epsilon on any step or whose exit
//! parameter sits on a step boundary, so neither discrete decision flips between
//! the two sides.
//!
//! # Degenerate inputs
//!
//! A direction too short to normalize misses (`hit = 0`, zeroed result),
//! matching the golden `None`. A ray that never overlaps the grid box misses on
//! the slab clip. A spent step budget or an exit parameter crossed before any
//! hit also misses. An empty query batch short-circuits on the host with no
//! dispatch (a storage buffer cannot be zero-sized); the field itself always has
//! at least one cell.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `min`,
//! `max`, `clamp`, `sqrt`, `abs`, `bitcast`, `+ - * /` and unsigned index
//! arithmetic — with no `sin`, `cos`, `exp`, `pow`, `round`, optional device
//! feature or `u64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! march loop has a per-query upper bound (`max_steps`) and always terminates on
//! the step counter.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_raymarch`；无第三方引擎源码或衍生代码。
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
/// that divides evenly on every target backend.
const WORKGROUP_SIZE: u32 = 64;

/// Inlined `WGSL` sphere-tracing kernel: one thread marches one ray through the
/// field's bounding box, mirroring the golden `sphere_trace`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_raymarch`；无第三方引擎源码或衍生代码。
const MESH_SDF_RAYMARCH_WGSL: &str = r#"
struct Params {
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    query_count: u32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    voxel_size: f32,
};

struct Query {
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    max_distance: f32,
    hit_epsilon: f32,
    max_steps: u32,
};

struct Hit {
    hit: u32,
    t: f32,
    px: f32,
    py: f32,
    pz: f32,
    distance: f32,
    steps: u32,
};

// Slab-clip result: ok flag plus the clamped entry/exit parameters.
struct Slab {
    ok: u32,
    t_enter: f32,
    t_exit: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> dst: array<Hit>;

// Linear interpolation between a and b by s (no clamping of s), matching the
// golden `lerp`.
fn lerp(a: f32, b: f32, s: f32) -> f32 {
    return a + (b - a) * s;
}

// Continuous cell-center split for one axis: returns the lower-corner cell as
// an f32 in slot x and the interpolation fraction in slot y. A single-layer
// (degenerate) axis contributes no interpolation weight.
fn axis_split(c: f32, o: f32, vs: f32, dim: u32) -> vec2<f32> {
    if (dim <= 1u) {
        return vec2<f32>(0.0, 0.0);
    }
    let last = f32(dim - 1u);
    let continuous = (c - o) / vs - 0.5;
    let clamped = clamp(continuous, 0.0, last);
    let lower = clamp(floor(clamped), 0.0, last - 1.0);
    return vec2<f32>(lower, clamped - lower);
}

// Fetches one cell-center value with integer clamp-to-border addressing. The
// field always holds at least one cell, so dim - 1 never underflows.
fn fetch(x: u32, y: u32, z: u32) -> f32 {
    let cx = min(x, params.dim_x - 1u);
    let cy = min(y, params.dim_y - 1u);
    let cz = min(z, params.dim_z - 1u);
    let idx = (cz * params.dim_y + cy) * params.dim_x + cx;
    return field[idx];
}

// Trilinearly samples the continuous signed distance at a world-space point,
// matching the golden `sample_signed_distance`.
fn signed_distance_at(p: vec3<f32>) -> f32 {
    let sx = axis_split(p.x, params.origin_x, params.voxel_size, params.dim_x);
    let sy = axis_split(p.y, params.origin_y, params.voxel_size, params.dim_y);
    let sz = axis_split(p.z, params.origin_z, params.voxel_size, params.dim_z);
    let bx = u32(sx.x);
    let by = u32(sy.x);
    let bz = u32(sz.x);
    let fx = sx.y;
    let fy = sy.y;
    let fz = sz.y;

    let d000 = fetch(bx, by, bz);
    let d100 = fetch(bx + 1u, by, bz);
    let d010 = fetch(bx, by + 1u, bz);
    let d110 = fetch(bx + 1u, by + 1u, bz);
    let d001 = fetch(bx, by, bz + 1u);
    let d101 = fetch(bx + 1u, by, bz + 1u);
    let d011 = fetch(bx, by + 1u, bz + 1u);
    let d111 = fetch(bx + 1u, by + 1u, bz + 1u);

    let c00 = lerp(d000, d100, fx);
    let c01 = lerp(d001, d101, fx);
    let c10 = lerp(d010, d110, fx);
    let c11 = lerp(d011, d111, fx);
    let c0 = lerp(c00, c10, fy);
    let c1 = lerp(c01, c11, fy);
    return lerp(c0, c1, fz);
}

// Clips a ray against the grid's axis-aligned bounding box with the slab
// method, matching the golden `ray_aabb`: t_enter starts at 0, t_exit at +inf,
// and an axis parallel to a slab misses unless the origin is between the planes.
fn ray_aabb(o: vec3<f32>, dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> Slab {
    // Smallest positive normal f32 (2^-126), the golden `f32::MIN_POSITIVE`.
    let min_positive = bitcast<f32>(8388608u);
    // Positive infinity via its IEEE-754 bit pattern.
    let inf = bitcast<f32>(2139095040u);
    var t0 = 0.0;
    var t1 = inf;
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        let d = dir[axis];
        let oa = o[axis];
        let lo = bmin[axis];
        let hi = bmax[axis];
        if (abs(d) <= min_positive) {
            if (oa < lo || oa > hi) {
                return Slab(0u, 0.0, 0.0);
            }
            continue;
        }
        let inv = 1.0 / d;
        var ta = (lo - oa) * inv;
        var tb = (hi - oa) * inv;
        if (ta > tb) {
            let tmp = ta;
            ta = tb;
            tb = tmp;
        }
        t0 = max(t0, ta);
        t1 = min(t1, tb);
        if (t0 > t1) {
            return Slab(0u, 0.0, 0.0);
        }
    }
    return Slab(1u, t0, t1);
}

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    let q = queries[idx];
    let origin = vec3<f32>(q.ox, q.oy, q.oz);

    // Default miss: zeroed result with the hit flag cleared (golden `None`).
    var out_hit = 0u;
    var out_t = 0.0;
    var out_px = 0.0;
    var out_py = 0.0;
    var out_pz = 0.0;
    var out_dist = 0.0;
    var out_steps = 0u;

    // Normalize the direction; a direction too short to normalize misses.
    let raw = vec3<f32>(q.dx, q.dy, q.dz);
    let len2 = raw.x * raw.x + raw.y * raw.y + raw.z * raw.z;
    let min_positive = bitcast<f32>(8388608u);
    if (len2 <= min_positive) {
        dst[idx] = Hit(out_hit, out_t, out_px, out_py, out_pz, out_dist, out_steps);
        return;
    }
    let len = sqrt(len2);
    let dir = vec3<f32>(raw.x / len, raw.y / len, raw.z / len);

    let dims = vec3<f32>(f32(params.dim_x), f32(params.dim_y), f32(params.dim_z));
    let grid_origin = vec3<f32>(params.origin_x, params.origin_y, params.origin_z);
    let grid_max = vec3<f32>(
        grid_origin.x + dims.x * params.voxel_size,
        grid_origin.y + dims.y * params.voxel_size,
        grid_origin.z + dims.z * params.voxel_size,
    );

    let slab = ray_aabb(origin, dir, grid_origin, grid_max);
    if (slab.ok == 0u) {
        dst[idx] = Hit(out_hit, out_t, out_px, out_py, out_pz, out_dist, out_steps);
        return;
    }
    let t_enter = slab.t_enter;
    let t_exit = min(slab.t_exit, q.max_distance);
    if (t_enter > t_exit) {
        dst[idx] = Hit(out_hit, out_t, out_px, out_py, out_pz, out_dist, out_steps);
        return;
    }

    // Floor on each step so a near-zero sample cannot freeze the march.
    let min_step = params.voxel_size * 0.125;
    var t = t_enter;
    var step = 0u;
    loop {
        if (step >= q.max_steps) {
            break;
        }
        let pos = vec3<f32>(
            origin.x + t * dir.x,
            origin.y + t * dir.y,
            origin.z + t * dir.z,
        );
        let distance = signed_distance_at(pos);
        if (distance <= q.hit_epsilon) {
            out_hit = 1u;
            out_t = t;
            out_px = pos.x;
            out_py = pos.y;
            out_pz = pos.z;
            out_dist = distance;
            out_steps = step + 1u;
            break;
        }
        t = t + max(distance, min_step);
        if (t > t_exit) {
            break;
        }
        step = step + 1u;
    }

    dst[idx] = Hit(out_hit, out_t, out_px, out_py, out_pz, out_dist, out_steps);
}
"#;

/// Uniform parameters for one dispatch: the field dimensions, the query count,
/// the field origin and the voxel edge length, laid out to match `Params` in
/// [`MESH_SDF_RAYMARCH_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Field cells along `x`.
    dim_x: u32,
    /// Field cells along `y`.
    dim_y: u32,
    /// Field cells along `z`.
    dim_z: u32,
    /// Number of valid query rays in the input and output buffers.
    query_count: u32,
    /// World-space minimum-corner `x`.
    origin_x: f32,
    /// World-space minimum-corner `y`.
    origin_y: f32,
    /// World-space minimum-corner `z`.
    origin_z: f32,
    /// Edge length of every (cubic) voxel.
    voxel_size: f32,
}

/// One query ray as the device sees it, matching `Query` in
/// [`MESH_SDF_RAYMARCH_WGSL`]: the ray origin and (unnormalized) direction, the
/// maximum travel distance, the hit epsilon and the step budget. A tight
/// `36`-byte stride with no padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin `x`.
    ox: f32,
    /// Ray origin `y`.
    oy: f32,
    /// Ray origin `z`.
    oz: f32,
    /// Ray direction `x` (normalized on device).
    dx: f32,
    /// Ray direction `y` (normalized on device).
    dy: f32,
    /// Ray direction `z` (normalized on device).
    dz: f32,
    /// Maximum travel distance along the normalized direction.
    max_distance: f32,
    /// Signed-distance hit threshold.
    hit_epsilon: f32,
    /// Maximum number of march steps.
    max_steps: u32,
}

/// One result as the device writes it, matching `Hit` in
/// [`MESH_SDF_RAYMARCH_WGSL`]: the hit flag, the ray parameter, the hit
/// position, the sampled distance and the step count. A tight `28`-byte stride
/// with no padding, read back by `bytemuck` so the integer fields stay exact.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` on a hit, `0` on a miss.
    hit: u32,
    /// Ray parameter of the hit, equal to the world-space distance travelled.
    t: f32,
    /// Hit position `x`.
    px: f32,
    /// Hit position `y`.
    py: f32,
    /// Hit position `z`.
    pz: f32,
    /// Sampled signed distance at the hit position.
    distance: f32,
    /// Number of march steps taken to reach the hit.
    steps: u32,
}

/// A uniform signed-distance field as seen by [`GpuSdfSphereTrace`]: the
/// per-cell signed distances plus the grid metadata the trilinear sampler
/// needs.
///
/// Cells are row-major with `x` varying fastest, matching the golden
/// `linear_index` of
/// `prism_render_architecture::ray_scene::mesh_signed_distance_field`. The host
/// builds this from any source (the golden `signed_distance_field`, an analytic
/// field, or a test fixture); the twin only ever reads the flat `distances`
/// array.
#[derive(Clone, Debug)]
pub struct SdfSphereTraceField {
    dims: [u32; 3],
    origin: [f32; 3],
    voxel_size: f32,
    distances: Vec<f32>,
}

impl SdfSphereTraceField {
    /// Builds a field from its `dims`, world-space `origin`, voxel edge length
    /// `voxel_size` and row-major per-cell signed `distances` (`x` fastest).
    ///
    /// # Panics
    ///
    /// Panics when any axis is empty or when `distances.len()` does not equal
    /// `dims[0] * dims[1] * dims[2]`, so a malformed field is rejected before a
    /// dispatch rather than reading out of bounds on the device.
    #[must_use]
    pub fn new(dims: [u32; 3], origin: [f32; 3], voxel_size: f32, distances: Vec<f32>) -> Self {
        assert!(
            dims[0] >= 1 && dims[1] >= 1 && dims[2] >= 1,
            "field must have at least one cell on every axis",
        );
        let expected = dims[0] as usize * dims[1] as usize * dims[2] as usize;
        assert_eq!(
            distances.len(),
            expected,
            "distances length must equal the cell count",
        );
        SdfSphereTraceField {
            dims,
            origin,
            voxel_size,
            distances,
        }
    }

    /// Cells along each axis.
    #[must_use]
    pub fn dims(&self) -> [u32; 3] {
        self.dims
    }

    /// World-space minimum corner.
    #[must_use]
    pub fn origin(&self) -> [f32; 3] {
        self.origin
    }

    /// Voxel edge length.
    #[must_use]
    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    /// Row-major per-cell signed distances.
    #[must_use]
    pub fn distances(&self) -> &[f32] {
        &self.distances
    }
}

/// One ray to sphere-trace against the field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSphereTraceQuery {
    /// World-space ray origin.
    pub origin: [f32; 3],
    /// Ray direction (normalized on device; need not be unit length).
    pub direction: [f32; 3],
    /// Maximum travel distance along the normalized direction.
    pub max_distance: f32,
    /// Signed-distance hit threshold.
    pub hit_epsilon: f32,
    /// Maximum number of march steps.
    pub max_steps: u32,
}

impl SdfSphereTraceQuery {
    /// Builds a ray query from an `origin`, a `direction`, a `max_distance`, a
    /// `hit_epsilon` and a `max_steps` budget.
    #[must_use]
    pub fn new(
        origin: [f32; 3],
        direction: [f32; 3],
        max_distance: f32,
        hit_epsilon: f32,
        max_steps: u32,
    ) -> Self {
        SdfSphereTraceQuery {
            origin,
            direction,
            max_distance,
            hit_epsilon,
            max_steps,
        }
    }
}

/// The sphere-tracing result for one ray.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSphereTraceResult {
    /// `1` on a hit, `0` on a miss (zeroed fields otherwise).
    pub hit: u32,
    /// Ray parameter of the hit, equal to the world-space distance travelled.
    pub t: f32,
    /// World-space hit position (`origin + t * normalized_direction`).
    pub position: [f32; 3],
    /// Sampled signed distance at the hit position.
    pub distance: f32,
    /// Number of march steps taken to reach the hit.
    pub steps: u32,
}

/// A compiled, reusable sphere-tracing kernel, twinning the `CPU` golden
/// `sphere_trace` of `prism_render_architecture::ray_scene::mesh_sdf_raymarch`.
pub struct GpuSdfSphereTrace {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfSphereTrace {
    /// Compiles the sphere-tracing kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is requested and compilation succeeds on any `Metal`,
    /// `Vulkan` or `DX12` backend.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_sdf_raymarch_module"),
            source: ShaderSource::Wgsl(MESH_SDF_RAYMARCH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_raymarch_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_raymarch_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_sdf_raymarch_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("trace"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfSphereTrace {
            module,
            layout,
            pipeline,
        }
    }

    /// Sphere-traces every ray in `queries` against the field, mirroring the
    /// golden `sphere_trace`.
    ///
    /// Returns one [`SdfSphereTraceResult`] per query, in order. An empty query
    /// batch returns an empty vector with no dispatch issued (a storage buffer
    /// cannot be zero-sized).
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        field: &SdfSphereTraceField,
        queries: &[SdfSphereTraceQuery],
    ) -> Vec<SdfSphereTraceResult> {
        let query_count = queries.len();
        if query_count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            dim_x: field.dims[0],
            dim_y: field.dims[1],
            dim_z: field.dims[2],
            query_count: query_count as u32,
            origin_x: field.origin[0],
            origin_y: field.origin[1],
            origin_z: field.origin[2],
            voxel_size: field.voxel_size,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_raymarch_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let field_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_raymarch_field"),
            contents: bytemuck::cast_slice(&field.distances),
            usage: BufferUsages::STORAGE,
        });

        let mut gpu_queries = Vec::with_capacity(query_count);
        for q in queries {
            gpu_queries.push(GpuQuery {
                ox: q.origin[0],
                oy: q.origin[1],
                oz: q.origin[2],
                dx: q.direction[0],
                dy: q.direction[1],
                dz: q.direction[2],
                max_distance: q.max_distance,
                hit_epsilon: q.hit_epsilon,
                max_steps: q.max_steps,
            });
        }
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_raymarch_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (query_count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_sdf_raymarch_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_sdf_raymarch_bind_group"),
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
            label: Some("prism_volumetric_mesh_sdf_raymarch_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_sdf_raymarch_encoder"),
        });
        {
            let groups = (query_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_sdf_raymarch_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per ray, flattened to a 1-D dispatch.
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
        let hits = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        let mut out = Vec::with_capacity(query_count);
        for hit in &hits {
            out.push(SdfSphereTraceResult {
                hit: hit.hit,
                t: hit.t,
                position: [hit.px, hit.py, hit.pz],
                distance: hit.distance,
                steps: hit.steps,
            });
        }
        out
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
