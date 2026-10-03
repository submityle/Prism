//! `wgpu` compute twin of the enhanced (over-relaxed) sphere-tracing reference
//! `enhanced_sphere_trace` in
//! `prism_render_architecture::ray_scene::mesh_sdf_enhanced_trace`, which
//! marches a ray through a signed distance field using the Keinert et al.
//! (2014) over-relaxation scheme on top of the trilinear sampler
//! `sample_signed_distance`, the direction `normalize` and the slab clip
//! `ray_aabb` of `prism_render_architecture::ray_scene::mesh_sdf_raymarch`.
//!
//! Naive sphere tracing advances a ray by exactly the sampled unbounding
//! distance each step, which stalls near grazing surfaces. Over-relaxation
//! multiplies each step by a factor `omega` in `[1, 2)`, speculatively
//! overshooting; whenever two successive unbounding spheres fail to overlap the
//! step is undone and the march falls back to a conservative `omega = 1` for the
//! remainder, so the converged hit matches naive tracing exactly while usually
//! taking far fewer steps. [`GpuSdfEnhancedTrace`] is the on-device twin: each
//! thread marches one ray and reports the first hit (or a miss).
//!
//! # What is twinned
//!
//! The kernel reproduces the reference operation for operation: the same
//! `normalize` guard (`length_squared <= f32::MIN_POSITIVE` is a miss), the same
//! slab `ray_aabb` with its parallel-axis and `t_enter > t_exit` rejections, the
//! same `t_exit = min(t_exit, max_distance)` clip, the same `omega` clamp into
//! `[1, 1.999999]`, the same entry-sign fold, and then the same per-step
//! `sor_failed = omega_cur > 1 && radius + previous_radius < step_length`
//! branch, the same `step_length` update, the same multiplicative hit test
//! `radius < pixel_radius * t` that never fires on a rolled-back step, and the
//! same `t += step_length` / `t > t_exit` termination.
//!
//! The field *construction* stays on the host: the twin consumes a flat `f32`
//! array of per-cell signed distances (row-major, `x` fastest) uploaded as a
//! read-only storage buffer, which is the only data the reference sampler ever
//! reads.
//!
//! # Correctness model
//!
//! The march is iterative and its `sor_failed` branch is `f32`-sensitive, so the
//! host oracle steps `f32`-for-`f32` in the same operation order as the kernel;
//! the discrete outputs `hit`, `steps` and `relaxation_resets` then match
//! exactly. The continuous outputs `t`, `position` and `distance` are a chain of
//! linear-interpolation adds plus one `sqrt` in normalization, so `CPU` and
//! `GPU` evaluate the same closed form but need not be bit-exact (a `GPU` may
//! contract a multiply-add); the parity test asserts
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3`. The parity sweep rejects fixtures
//! that sit near a branch boundary so the discrete counts never flip between the
//! two sides.
//!
//! # Degenerate inputs
//!
//! A direction too short to normalize is a miss (`hit = 0`). A ray that never
//! overlaps the field box, exits before converging, exceeds `max_distance` or
//! exhausts `max_steps` is likewise a miss. An empty query batch short-circuits
//! on the host with no dispatch (a storage buffer cannot be zero-sized); the
//! field itself always has at least one cell.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `min`,
//! `max`, `clamp`, `abs`, `sqrt`, `bitcast`, `+ - * /` and unsigned index
//! arithmetic — with no `sin`, `cos`, `exp`, `pow`, `round`, `%`, optional
//! device feature or `u64`, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. The march loop is bounded by `max_steps`, so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_enhanced_trace`；无第三方引擎源码或衍生代码。
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

/// Flat `f32` slots consumed per query: `origin xyz`, `direction xyz`,
/// `max_distance`, `pixel_radius`, `max_steps` and `over_relaxation`
/// (a `10`-wide stride).
const QUERY_STRIDE: usize = 10;

/// Flat `f32` slots written per query: `hit`, `t`, `position xyz`, `distance`,
/// `steps` and `relaxation_resets` (an `8`-wide, `32`-byte stride). The three
/// discrete counts are stored as exact integer-valued `f32` and decoded on the
/// host.
const RESULT_STRIDE: usize = 8;

/// The enhanced over-relaxed sphere-tracing kernel, mirroring the `CPU` golden
/// `enhanced_sphere_trace` of
/// `prism_render_architecture::ray_scene::mesh_sdf_enhanced_trace`. One thread
/// marches one ray through the shared field and reports the first hit. `field`
/// holds the row-major per-cell signed distances (`x` varying fastest),
/// `queries` holds ten `f32` per ray and `dst` holds eight `f32` per ray. Pure
/// linear interpolation, integer addressing and one `sqrt` in normalization: no
/// transcendental, no intrinsic, no `u64`, portable on `Metal`, `Vulkan` and
/// `DX12`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_enhanced_trace`；无第三方引擎源码或衍生代码。
const MESH_SDF_ENHANCED_TRACE_WGSL: &str = r#"
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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read> queries: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

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

// Updates the slab entry/exit parameters for one axis. Returns the new
// (t0, t1) in xy and an ok flag in z (0.0 = the ray misses this slab), matching
// the golden `ray_aabb`.
fn slab_axis(ro: f32, rd: f32, bmin: f32, bmax: f32, t0: f32, t1: f32) -> vec3<f32> {
    let min_pos = bitcast<f32>(8388608u);
    if (abs(rd) <= min_pos) {
        // Ray is parallel to this slab: miss unless the origin lies between the
        // planes.
        if (ro < bmin || ro > bmax) {
            return vec3<f32>(t0, t1, 0.0);
        }
        return vec3<f32>(t0, t1, 1.0);
    }
    let inv = 1.0 / rd;
    var ta = (bmin - ro) * inv;
    var tb = (bmax - ro) * inv;
    if (ta > tb) {
        let tmp = ta;
        ta = tb;
        tb = tmp;
    }
    let nt0 = max(t0, ta);
    let nt1 = min(t1, tb);
    if (nt0 > nt1) {
        return vec3<f32>(nt0, nt1, 0.0);
    }
    return vec3<f32>(nt0, nt1, 1.0);
}

@compute @workgroup_size(64)
fn enhanced_sphere_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    let in_base = idx * 10u;
    let origin = vec3<f32>(
        queries[in_base],
        queries[in_base + 1u],
        queries[in_base + 2u],
    );
    let direction = vec3<f32>(
        queries[in_base + 3u],
        queries[in_base + 4u],
        queries[in_base + 5u],
    );
    let max_distance = queries[in_base + 6u];
    let pixel_radius = queries[in_base + 7u];
    let max_steps = u32(queries[in_base + 8u]);
    let over_relaxation = queries[in_base + 9u];

    let out_base = idx * 8u;

    // Default miss output; overwritten only on a converged hit.
    var out_hit = 0.0;
    var out_t = 0.0;
    var out_px = 0.0;
    var out_py = 0.0;
    var out_pz = 0.0;
    var out_distance = 0.0;
    var out_steps = 0.0;
    var out_resets = 0.0;

    let min_pos = bitcast<f32>(8388608u);
    let inf = bitcast<f32>(2139095040u);

    // Normalize the direction; a direction too short to normalize is a miss.
    let length_squared =
        direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
    var hit_resolved = false;
    if (length_squared <= min_pos) {
        hit_resolved = true;
    }

    if (!hit_resolved) {
        let length = sqrt(length_squared);
        let dir = vec3<f32>(direction.x / length, direction.y / length, direction.z / length);

        let grid_origin = vec3<f32>(params.origin_x, params.origin_y, params.origin_z);
        let grid_max = vec3<f32>(
            params.origin_x + f32(params.dim_x) * params.voxel_size,
            params.origin_y + f32(params.dim_y) * params.voxel_size,
            params.origin_z + f32(params.dim_z) * params.voxel_size,
        );

        // Slab clip against the field bounding box.
        var t0 = 0.0;
        var t1 = inf;
        var ok = 1.0;
        let rx = slab_axis(origin.x, dir.x, grid_origin.x, grid_max.x, t0, t1);
        t0 = rx.x;
        t1 = rx.y;
        if (rx.z < 0.5) {
            ok = 0.0;
        }
        let ry = slab_axis(origin.y, dir.y, grid_origin.y, grid_max.y, t0, t1);
        t0 = ry.x;
        t1 = ry.y;
        if (ry.z < 0.5) {
            ok = 0.0;
        }
        let rz = slab_axis(origin.z, dir.z, grid_origin.z, grid_max.z, t0, t1);
        t0 = rz.x;
        t1 = rz.y;
        if (rz.z < 0.5) {
            ok = 0.0;
        }

        if (ok < 0.5) {
            hit_resolved = true;
        }

        if (!hit_resolved) {
            let t_enter = t0;
            let t_exit = min(t1, max_distance);
            if (t_enter > t_exit) {
                hit_resolved = true;
            }

            if (!hit_resolved) {
                // Keep over-relaxation strictly below 2 so the overlap test
                // stays valid.
                let omega = clamp(over_relaxation, 1.0, 1.999999);

                let entry_pos = origin + t_enter * dir;
                let entry_sample = signed_distance_at(entry_pos);
                var sign0 = 1.0;
                if (entry_sample < 0.0) {
                    sign0 = -1.0;
                }

                var t = t_enter;
                var previous_radius = 0.0;
                var step_length = 0.0;
                var omega_cur = omega;
                var relaxation_resets = 0u;

                for (var step: u32 = 0u; step < max_steps; step = step + 1u) {
                    let position = origin + t * dir;
                    let signed = sign0 * signed_distance_at(position);
                    let radius = abs(signed);

                    // The over-relaxed step overshot when the current and
                    // previous unbounding spheres no longer overlap.
                    let sor_failed = (omega_cur > 1.0) && ((radius + previous_radius) < step_length);
                    if (sor_failed) {
                        // Undo the previous over-relaxed advance and drop to
                        // conservative stepping for the rest of the march.
                        step_length = step_length - omega_cur * step_length;
                        omega_cur = 1.0;
                        relaxation_resets = relaxation_resets + 1u;
                    } else {
                        step_length = signed * omega_cur;
                    }

                    previous_radius = radius;

                    // Multiplicative hit test; never fires on a rolled-back step.
                    if ((!sor_failed) && (radius < pixel_radius * t)) {
                        out_hit = 1.0;
                        out_t = t;
                        out_px = position.x;
                        out_py = position.y;
                        out_pz = position.z;
                        out_distance = sign0 * signed;
                        out_steps = f32(step + 1u);
                        out_resets = f32(relaxation_resets);
                        hit_resolved = true;
                        break;
                    }

                    t = t + step_length;
                    if (t > t_exit) {
                        break;
                    }
                }
            }
        }
    }

    dst[out_base] = out_hit;
    dst[out_base + 1u] = out_t;
    dst[out_base + 2u] = out_px;
    dst[out_base + 3u] = out_py;
    dst[out_base + 4u] = out_pz;
    dst[out_base + 5u] = out_distance;
    dst[out_base + 6u] = out_steps;
    dst[out_base + 7u] = out_resets;
}
"#;

/// Uniform parameters for one dispatch: the field dimensions, the query count,
/// the field origin and the voxel edge length, laid out to match `Params` in
/// [`MESH_SDF_ENHANCED_TRACE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Field cells along `x`.
    dim_x: u32,
    /// Field cells along `y`.
    dim_y: u32,
    /// Field cells along `z`.
    dim_z: u32,
    /// Number of valid rays in the input and output buffers.
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

/// A uniform signed-distance field as seen by [`GpuSdfEnhancedTrace`]: the
/// per-cell signed distances plus the grid metadata the trilinear sampler
/// needs.
///
/// Cells are row-major with `x` varying fastest, matching the golden
/// `linear_index` of
/// `prism_render_architecture::ray_scene::mesh_signed_distance_field`. The
/// host builds this from any source (the golden `signed_distance_field`, an
/// analytic field, or a test fixture); the twin only ever reads the flat
/// `distances` array.
#[derive(Clone, Debug)]
pub struct SdfEnhancedTraceField {
    dims: [u32; 3],
    origin: [f32; 3],
    voxel_size: f32,
    distances: Vec<f32>,
}

impl SdfEnhancedTraceField {
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
        SdfEnhancedTraceField {
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

/// One ray the twin over-relaxed sphere traces against the shared field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfEnhancedTraceQuery {
    /// World-space ray origin.
    pub origin: [f32; 3],
    /// Ray direction; need not be unit length (normalized on the device).
    pub direction: [f32; 3],
    /// Maximum world-space distance marched before giving up.
    pub max_distance: f32,
    /// Pixel cone half-footprint per unit ray distance; the march stops once
    /// `radius < pixel_radius * t`.
    pub pixel_radius: f32,
    /// Maximum marching iterations before giving up.
    pub max_steps: u32,
    /// Over-relaxation factor, clamped into `[1, 1.999999]` on the device.
    pub over_relaxation: f32,
}

impl SdfEnhancedTraceQuery {
    /// Builds a ray query from its origin, direction and march parameters.
    #[must_use]
    pub fn new(
        origin: [f32; 3],
        direction: [f32; 3],
        max_distance: f32,
        pixel_radius: f32,
        max_steps: u32,
        over_relaxation: f32,
    ) -> Self {
        SdfEnhancedTraceQuery {
            origin,
            direction,
            max_distance,
            pixel_radius,
            max_steps,
            over_relaxation,
        }
    }
}

/// The result of over-relaxed sphere tracing one ray.
///
/// On a miss `hit` is `0` and every other field is a zero placeholder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfEnhancedTraceResult {
    /// `1` when the ray converged on a surface, `0` on a miss.
    pub hit: u32,
    /// Ray parameter of the hit, equal to the world-space distance travelled.
    pub t: f32,
    /// World-space hit position (`origin + t * normalized_direction`).
    pub position: [f32; 3],
    /// Raw signed field value sampled at the hit (negative inside).
    pub distance: f32,
    /// Number of marching iterations taken to reach the hit.
    pub steps: u32,
    /// Number of over-relaxation back-offs performed during the march.
    pub relaxation_resets: u32,
}

/// A compiled, reusable enhanced sphere-tracing kernel, twinning the `CPU`
/// golden `enhanced_sphere_trace` of
/// `prism_render_architecture::ray_scene::mesh_sdf_enhanced_trace`.
pub struct GpuSdfEnhancedTrace {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfEnhancedTrace {
    /// Compiles the enhanced sphere-tracing kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is requested and compilation succeeds on any `Metal`,
    /// `Vulkan` or `DX12` backend.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_sdf_enhanced_trace_module"),
            source: ShaderSource::Wgsl(MESH_SDF_ENHANCED_TRACE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_enhanced_trace_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_enhanced_trace_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_sdf_enhanced_trace_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("enhanced_sphere_trace"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfEnhancedTrace {
            module,
            layout,
            pipeline,
        }
    }

    /// Over-relaxed sphere traces every ray in `queries` against `field`,
    /// mirroring the golden `enhanced_sphere_trace`.
    ///
    /// Returns one [`SdfEnhancedTraceResult`] per ray, in order. An empty query
    /// batch returns an empty vector with no dispatch issued (a storage buffer
    /// cannot be zero-sized).
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        field: &SdfEnhancedTraceField,
        queries: &[SdfEnhancedTraceQuery],
    ) -> Vec<SdfEnhancedTraceResult> {
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
            label: Some("prism_volumetric_mesh_sdf_enhanced_trace_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let field_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_enhanced_trace_field"),
            contents: bytemuck::cast_slice(&field.distances),
            usage: BufferUsages::STORAGE,
        });

        let mut flat_queries = Vec::with_capacity(query_count * QUERY_STRIDE);
        for q in queries {
            flat_queries.push(q.origin[0]);
            flat_queries.push(q.origin[1]);
            flat_queries.push(q.origin[2]);
            flat_queries.push(q.direction[0]);
            flat_queries.push(q.direction[1]);
            flat_queries.push(q.direction[2]);
            flat_queries.push(q.max_distance);
            flat_queries.push(q.pixel_radius);
            flat_queries.push(q.max_steps as f32);
            flat_queries.push(q.over_relaxation);
        }
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_enhanced_trace_queries"),
            contents: bytemuck::cast_slice(&flat_queries),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (query_count * RESULT_STRIDE * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_sdf_enhanced_trace_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_sdf_enhanced_trace_bind_group"),
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
            label: Some("prism_volumetric_mesh_sdf_enhanced_trace_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_sdf_enhanced_trace_encoder"),
        });
        {
            let groups = (query_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_sdf_enhanced_trace_pass"),
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
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        let mut out = Vec::with_capacity(query_count);
        for chunk in flat.chunks_exact(RESULT_STRIDE) {
            // `hit` is written as exactly 0.0 or 1.0; decode without an f32
            // equality by testing the midpoint. The three counts are exact
            // integer-valued f32, decoded by rounding back to the nearest u32.
            let hit = u32::from(chunk[0] > 0.5);
            out.push(SdfEnhancedTraceResult {
                hit,
                t: chunk[1],
                position: [chunk[2], chunk[3], chunk[4]],
                distance: chunk[5],
                steps: chunk[6].round() as u32,
                relaxation_resets: chunk[7].round() as u32,
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
