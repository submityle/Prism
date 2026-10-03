//! `wgpu` compute twin of the signed-distance-field surface *projection*
//! reference of the `CPU` golden path — `project_to_surface` in
//! `prism_render_architecture::ray_scene::mesh_sdf_surface_projection`, which
//! composes the trilinear sampler `sample_signed_distance` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch` with the gradient
//! normal `sdf_normal` of
//! `prism_render_architecture::ray_scene::mesh_sdf_normal`.
//!
//! A signed distance field answers two questions at every world-space point:
//! how far the nearest surface is (the magnitude) and which way it lies (the
//! gradient). Combining them lets us *snap* a point onto the zero level set by
//! repeatedly stepping along the oriented normal by the signed distance — a
//! Newton-style root find on the field. `AAA` engines use this to resolve
//! penetrations in distance-field collisions, stick decals and particles to
//! geometry, and seed contact points for soft-body and cloth solvers.
//! [`GpuSdfSurfaceProjection`] is the on-device twin: each thread reads one
//! world-space start point, iterates the Newton step and writes back the
//! converged (or best-effort) location, its residual, the iteration count and a
//! `valid` flag.
//!
//! # What is twinned
//!
//! The kernel reproduces the reference step for step. Each iteration first
//! trilinearly samples the signed distance at the current point; if its unsigned
//! value is at or below the tolerance it returns immediately with that iteration
//! index (so convergence wins even when the gradient would be degenerate). It
//! otherwise forms the same symmetric central-difference gradient the normal
//! twin uses (`step = voxel_size`, `inv = 1 / (2 * step)`, full-step forward and
//! backward taps per axis), guards it against the same
//! `length_squared <= f32::MIN_POSITIVE` zero-gradient threshold, and — when the
//! gradient vanishes mid-loop — stops with `valid = 0` and a zeroed result,
//! matching the golden `None`. When the loop instead runs out of iterations it
//! samples one final residual and returns `valid = 1` with the iteration count
//! pinned to `max_iterations`.
//!
//! The field *construction* (the exact Euclidean transform, the solid
//! classification and the `signed_squared` integer storage) stays on the host:
//! the twin consumes a flat `f32` array of per-cell signed distances uploaded as
//! a read-only storage buffer, which is the only data the reference sampler ever
//! reads.
//!
//! # Correctness model
//!
//! Sampling is linear interpolation (`+`, `-`, `*`) plus integer `clamp`
//! addressing; the gradient is three differences scaled by a constant; the
//! normal divides by a single `sqrt`; the Newton step is one multiply-subtract
//! per axis. `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact — a `GPU` may contract a multiply-add — so the parity test asserts
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` on the point and residual. The
//! `iterations` count and the `valid` flag are integer decisions and match with
//! no tolerance; the parity sweep rejects start points whose path grazes either
//! the tolerance band or the zero-gradient guard so neither discrete decision
//! flips between the two sides.
//!
//! # Degenerate inputs
//!
//! A constant field (every tap equal) yields a vanishing central difference, so
//! the first non-converged iteration stops with `valid = 0` and a zeroed
//! result. `max_iterations == 0` runs an empty loop and returns the start point
//! unchanged with `iterations = 0` and `valid = 1`, matching the golden
//! best-effort return. An empty query batch short-circuits on the host with no
//! dispatch (a storage buffer cannot be zero-sized); the field itself always has
//! at least one cell.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `min`,
//! `max`, `clamp`, `sqrt`, `abs`, `bitcast`, `+ - * /` and unsigned index
//! arithmetic — with no `sin`, `cos`, `exp`, `pow`, `round`, optional device
//! feature or `u64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! Newton loop has a per-query upper bound (`max_iterations`) and always
//! terminates on the iteration counter.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_surface_projection`；无第三方引擎源码或衍生代码。
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

/// The signed-distance-field surface-projection kernel, mirroring the `CPU`
/// golden `project_to_surface` of
/// `prism_render_architecture::ray_scene::mesh_sdf_surface_projection`. One
/// thread handles one world-space start point: it iterates the Newton step
/// `point -= normal * signed_distance`, trilinearly sampling the shared field
/// and forming a symmetric central-difference gradient each iteration until the
/// unsigned signed distance drops to or below the tolerance or the iteration
/// budget is spent. `field` holds the row-major per-cell signed distances (`x`
/// varying fastest), `queries` holds one `Query` per start point and `dst`
/// holds one `Hit` per start point. Pure linear interpolation, integer
/// addressing and one `sqrt` per iteration: no transcendental, no intrinsic, no
/// `u64`, portable on `Metal`, `Vulkan` and `DX12`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_surface_projection`；无第三方引擎源码或衍生代码。
const MESH_SDF_SURFACE_PROJECTION_WGSL: &str = r#"
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
    px: f32,
    py: f32,
    pz: f32,
    max_iterations: u32,
    tolerance: f32,
};

struct Hit {
    px: f32,
    py: f32,
    pz: f32,
    residual: f32,
    iterations: u32,
    valid: u32,
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

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    let q = queries[idx];
    var cur = vec3<f32>(q.px, q.py, q.pz);
    let tol = q.tolerance;
    let max_it = q.max_iterations;

    // Smallest positive normal f32 (2^-126), the golden `f32::MIN_POSITIVE`.
    let min_positive = bitcast<f32>(8388608u);
    let step = params.voxel_size;
    let inv = 1.0 / (2.0 * step);

    // Defaults for the degenerate (gradient-vanishes) outcome: a zeroed result
    // with the valid flag cleared, matching the golden `None`.
    var out_px = 0.0;
    var out_py = 0.0;
    var out_pz = 0.0;
    var out_res = 0.0;
    var out_it = 0u;
    var out_valid = 0u;
    var done = false;

    var i = 0u;
    loop {
        if (i >= max_it) {
            break;
        }
        let signed = signed_distance_at(cur);
        if (abs(signed) <= tol) {
            out_px = cur.x;
            out_py = cur.y;
            out_pz = cur.z;
            out_res = abs(signed);
            out_it = i;
            out_valid = 1u;
            done = true;
            break;
        }

        let fwd_x = vec3<f32>(cur.x + step, cur.y, cur.z);
        let bwd_x = vec3<f32>(cur.x - step, cur.y, cur.z);
        let fwd_y = vec3<f32>(cur.x, cur.y + step, cur.z);
        let bwd_y = vec3<f32>(cur.x, cur.y - step, cur.z);
        let fwd_z = vec3<f32>(cur.x, cur.y, cur.z + step);
        let bwd_z = vec3<f32>(cur.x, cur.y, cur.z - step);

        let gx = (signed_distance_at(fwd_x) - signed_distance_at(bwd_x)) * inv;
        let gy = (signed_distance_at(fwd_y) - signed_distance_at(bwd_y)) * inv;
        let gz = (signed_distance_at(fwd_z) - signed_distance_at(bwd_z)) * inv;

        let len2 = gx * gx + gy * gy + gz * gz;
        if (len2 <= min_positive) {
            // Gradient vanished before convergence: no direction to step, so
            // stop with the zeroed, invalid default (golden `None`).
            done = true;
            break;
        }
        let len = sqrt(len2);
        cur = vec3<f32>(
            cur.x - (gx / len) * signed,
            cur.y - (gy / len) * signed,
            cur.z - (gz / len) * signed,
        );
        i = i + 1u;
    }

    if (!done) {
        // Ran out of iterations: best-effort return with one final residual.
        let residual = abs(signed_distance_at(cur));
        out_px = cur.x;
        out_py = cur.y;
        out_pz = cur.z;
        out_res = residual;
        out_it = max_it;
        out_valid = 1u;
    }

    dst[idx] = Hit(out_px, out_py, out_pz, out_res, out_it, out_valid);
}
"#;

/// Uniform parameters for one dispatch: the field dimensions, the query count,
/// the field origin and the voxel edge length, laid out to match `Params` in
/// [`MESH_SDF_SURFACE_PROJECTION_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Field cells along `x`.
    dim_x: u32,
    /// Field cells along `y`.
    dim_y: u32,
    /// Field cells along `z`.
    dim_z: u32,
    /// Number of valid query points in the input and output buffers.
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

/// One query as the device sees it, matching `Query` in
/// [`MESH_SDF_SURFACE_PROJECTION_WGSL`]: the start point, the iteration budget
/// and the convergence tolerance. A tight `20`-byte stride with no padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// World-space start point `x`.
    px: f32,
    /// World-space start point `y`.
    py: f32,
    /// World-space start point `z`.
    pz: f32,
    /// Maximum number of Newton steps.
    max_iterations: u32,
    /// Unsigned-distance convergence tolerance.
    tolerance: f32,
}

/// One result as the device writes it, matching `Hit` in
/// [`MESH_SDF_SURFACE_PROJECTION_WGSL`]: the projected point, its residual, the
/// iteration count and the `valid` flag. A tight `24`-byte stride with no
/// padding, read back by `bytemuck` so the integer fields stay exact.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Projected point `x`.
    px: f32,
    /// Projected point `y`.
    py: f32,
    /// Projected point `z`.
    pz: f32,
    /// Remaining unsigned distance to the surface.
    residual: f32,
    /// Number of Newton steps taken.
    iterations: u32,
    /// `1` on a projection (converged or best-effort), `0` when the gradient
    /// vanished before convergence.
    valid: u32,
}

/// A uniform signed-distance field as seen by [`GpuSdfSurfaceProjection`]: the
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
pub struct SdfSurfaceProjectionField {
    dims: [u32; 3],
    origin: [f32; 3],
    voxel_size: f32,
    distances: Vec<f32>,
}

impl SdfSurfaceProjectionField {
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
        SdfSurfaceProjectionField {
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

/// One world-space start point to project onto the field's zero level set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSurfaceProjectionQuery {
    /// World-space start point of the Newton iteration.
    pub point: [f32; 3],
    /// Maximum number of Newton steps before the best-effort return.
    pub max_iterations: u32,
    /// Unsigned-distance convergence tolerance.
    pub tolerance: f32,
}

impl SdfSurfaceProjectionQuery {
    /// Builds a query from a start `point`, an iteration budget and a
    /// convergence `tolerance`.
    #[must_use]
    pub fn new(point: [f32; 3], max_iterations: u32, tolerance: f32) -> Self {
        SdfSurfaceProjectionQuery {
            point,
            max_iterations,
            tolerance,
        }
    }
}

/// The projection of one start point onto the field surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSurfaceProjectionResult {
    /// World-space position after projection (zeroed when `valid` is `0`).
    pub point: [f32; 3],
    /// Remaining unsigned distance to the surface (zeroed when `valid` is `0`).
    pub residual: f32,
    /// Number of Newton steps taken (zeroed when `valid` is `0`).
    pub iterations: u32,
    /// `1` on a projection (converged or best-effort), `0` when the gradient
    /// vanished before convergence (the golden `None`).
    pub valid: u32,
}

/// A compiled, reusable signed-distance-field surface-projection kernel,
/// twinning the `CPU` golden `project_to_surface` of
/// `prism_render_architecture::ray_scene::mesh_sdf_surface_projection`.
pub struct GpuSdfSurfaceProjection {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfSurfaceProjection {
    /// Compiles the signed-distance-field surface-projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is requested and compilation succeeds on any `Metal`,
    /// `Vulkan` or `DX12` backend.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_sdf_surface_projection_module"),
            source: ShaderSource::Wgsl(MESH_SDF_SURFACE_PROJECTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_surface_projection_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_surface_projection_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_sdf_surface_projection_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("project"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfSurfaceProjection {
            module,
            layout,
            pipeline,
        }
    }

    /// Projects every start point in `queries` onto the field surface,
    /// mirroring the golden `project_to_surface`.
    ///
    /// Returns one [`SdfSurfaceProjectionResult`] per query, in order. An empty
    /// query batch returns an empty vector with no dispatch issued (a storage
    /// buffer cannot be zero-sized).
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        field: &SdfSurfaceProjectionField,
        queries: &[SdfSurfaceProjectionQuery],
    ) -> Vec<SdfSurfaceProjectionResult> {
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
            label: Some("prism_volumetric_mesh_sdf_surface_projection_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let field_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_surface_projection_field"),
            contents: bytemuck::cast_slice(&field.distances),
            usage: BufferUsages::STORAGE,
        });

        let mut gpu_queries = Vec::with_capacity(query_count);
        for q in queries {
            gpu_queries.push(GpuQuery {
                px: q.point[0],
                py: q.point[1],
                pz: q.point[2],
                max_iterations: q.max_iterations,
                tolerance: q.tolerance,
            });
        }
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_surface_projection_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (query_count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_sdf_surface_projection_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_sdf_surface_projection_bind_group"),
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
            label: Some("prism_volumetric_mesh_sdf_surface_projection_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_sdf_surface_projection_encoder"),
        });
        {
            let groups = (query_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_sdf_surface_projection_pass"),
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
        let hits = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        let mut out = Vec::with_capacity(query_count);
        for hit in &hits {
            out.push(SdfSurfaceProjectionResult {
                point: [hit.px, hit.py, hit.pz],
                residual: hit.residual,
                iterations: hit.iterations,
                valid: hit.valid,
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
