//! `wgpu` compute twin of the signed-distance-field cone-traced occlusion
//! reference of the `CPU` golden path — `sdf_cone_occlusion` in
//! `prism_render_architecture::ray_scene::mesh_sdf_cone_occlusion`, which sweeps
//! a baked seven-cone hemisphere fan through the trilinear sampler
//! `sample_signed_distance` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch`.
//!
//! A single occlusion tap along the surface normal is noisy; modern `AAA`
//! renderers instead cone-trace a small fan across the hemisphere and keep, per
//! cone, the fraction of the cone cross-section the field leaves unblocked.
//! Averaging the cone visibilities yields a far less noisy occlusion term, and
//! the visibility-weighted mean of the cone directions yields the **bent
//! normal** — the mean unoccluded direction — which image-based lighting samples
//! instead of the geometric normal so shading leans away from occluders.
//! [`GpuSdfConeOcclusion`] is the on-device twin: each thread cone-traces one
//! shaded point through the shared field.
//!
//! # What is twinned
//!
//! The kernel reproduces the reference exactly: the same branchless Duff et al.
//! (2017) orthonormal basis, the same seven baked tangent-space cone directions
//! (`CONE_DIRS`: a center cone along the normal plus a six-way `45°` ring), the
//! same `min_step = voxel_size * 0.5` floor, the same per-cone running minimum
//! of `sampled_distance / (tan_half_angle * t)`, the same early exit when a cone
//! is fully occluded, the same cosine (tangent-space `z`) weighting, the same
//! final `visibility_sum / weight_sum` clamp and the same visibility-weighted
//! bent-normal renormalization that falls back to the geometric normal when the
//! accumulated direction is a zero-length sum.
//!
//! The golden `f32::signum` is replicated as `select(1.0, -1.0, n_z < 0.0)` so
//! the basis sign folds in exactly as the reference does and the Duff
//! denominator `sign + n_z` always has magnitude at least one (no pole). The
//! golden `f32::MIN_POSITIVE` guard is replicated with the `2^-126` bit pattern
//! so no non-finite value ever enters the arithmetic (`WGSL` has no portable
//! infinity literal and no `isfinite`).
//!
//! The golden returns [`None`] only when `normal` cannot be normalized; the twin
//! maps that to a `valid` flag cleared to `0` with all other outputs zeroed, and
//! a successful trace sets `valid = 1`.
//!
//! The field *construction* (the exact Euclidean transform, the solid
//! classification and the integer storage) stays on the host: the twin consumes
//! a flat `f32` array of per-cell signed distances uploaded as a read-only
//! storage buffer, which is the only data the reference sampler ever reads.
//!
//! # Correctness model
//!
//! Each cone march is a bounded loop of multiplies, a few `clamp`/`min`/`max`
//! guards and a trilinear tap (linear interpolation plus integer `clamp`
//! addressing), closed by one `sqrt` per normalization. `CPU` and `GPU` evaluate
//! the same closed form but need not be bit-exact — a `GPU` may contract a
//! multiply-add — so the parity test asserts `abs_diff <= 1e-4 || rel_diff <=
//! 1e-3` on `visibility` and each `bent_normal` component. The `valid` output is
//! a boolean decision and matches with no tolerance; the parity fixtures use an
//! all-positive field and a far `max_distance` so each cone runs its full step
//! budget with no early break, keeping the march fully deterministic.
//!
//! # Degenerate inputs
//!
//! A zero-length `normal` cannot be normalized, so both sides report `valid = 0`
//! with a zero visibility and zero bent normal. An empty query batch
//! short-circuits on the host with no dispatch (a storage buffer cannot be
//! zero-sized); the field itself always has at least one cell.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `min`,
//! `max`, `clamp`, `sqrt`, `select`, `bitcast`, `+ - * /` and unsigned index
//! arithmetic — with no `sin`, `cos`, `exp`, `pow`, `round`, optional device
//! feature or `u64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Both
//! the cone fan (seven) and each march (`step_count`) are bounded, so the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_cone_occlusion`；无第三方引擎源码或衍生代码。
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

/// Flat `f32` slots read per query: `position` (`3`), `normal` (`3`),
/// `tan_half_angle`, `max_distance`, the `step_count` and one pad to a `10`-wide
/// stride.
const QUERY_STRIDE: usize = 10;

/// Flat `f32` slots written per query: `visibility`, the three `bent_normal`
/// components and the `valid` flag (a `5`-wide stride).
const RESULT_STRIDE: usize = 5;

/// Inlined `WGSL` for the signed-distance-field cone-occlusion kernel. One
/// thread cone-traces one shaded point through the shared field: seven baked
/// hemisphere cones rotated into the Duff orthonormal frame, each marched with
/// the running minimum of `sampled / radius`. Pure linear interpolation,
/// integer addressing, bounded loops and one `sqrt` per normalization: no
/// transcendental, no intrinsic, no `u64`, portable on `Metal`, `Vulkan` and
/// `DX12`.
const MESH_SDF_CONE_OCCLUSION_WGSL: &str = r#"
// Provenance: twin of prism_render_architecture::ray_scene::mesh_sdf_cone_occlusion.
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

// Linear interpolation between a and b by s, matching the golden `lerp`.
fn lerp(a: f32, b: f32, s: f32) -> f32 {
    return a + (b - a) * s;
}

// Continuous cell-center split for one axis: returns the lower-corner cell in
// slot x and the interpolation fraction in slot y. A single-layer axis
// contributes no interpolation weight.
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

// One of the seven baked tangent-space cone directions (z is the surface
// normal): the center cone plus a six-way ring at 45 degrees elevation.
fn cone_dir(i: u32) -> vec3<f32> {
    let cos45 = 0.70710677;
    let ring_half = 0.35355338;
    let ring_tall = 0.61237244;
    var d = vec3<f32>(0.0, 0.0, 1.0);
    if (i == 1u) {
        d = vec3<f32>(cos45, 0.0, cos45);
    } else if (i == 2u) {
        d = vec3<f32>(ring_half, ring_tall, cos45);
    } else if (i == 3u) {
        d = vec3<f32>(-ring_half, ring_tall, cos45);
    } else if (i == 4u) {
        d = vec3<f32>(-cos45, 0.0, cos45);
    } else if (i == 5u) {
        d = vec3<f32>(-ring_half, -ring_tall, cos45);
    } else if (i == 6u) {
        d = vec3<f32>(ring_half, -ring_tall, cos45);
    }
    return d;
}

@compute @workgroup_size(64)
fn sdf_cone_occlusion(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    let base = idx * 10u;
    let px = queries[base];
    let py = queries[base + 1u];
    let pz = queries[base + 2u];
    let nx = queries[base + 3u];
    let ny = queries[base + 4u];
    let nz = queries[base + 5u];
    let tan_half_angle = queries[base + 6u];
    let max_distance = queries[base + 7u];
    let step_count_f = queries[base + 8u];

    let out_base = idx * 5u;

    // Smallest positive normal f32 (2^-126), the golden `f32::MIN_POSITIVE`.
    let min_positive = bitcast<f32>(8388608u);
    let nlen2 = nx * nx + ny * ny + nz * nz;
    if (nlen2 <= min_positive) {
        // Degenerate normal: the golden returns None.
        dst[out_base] = 0.0;
        dst[out_base + 1u] = 0.0;
        dst[out_base + 2u] = 0.0;
        dst[out_base + 3u] = 0.0;
        dst[out_base + 4u] = 0.0;
        return;
    }

    let inv_nlen = 1.0 / sqrt(nlen2);
    let n = vec3<f32>(nx * inv_nlen, ny * inv_nlen, nz * inv_nlen);

    // Branchless Duff et al. (2017) orthonormal basis. `signum` is replicated as
    // a select so the sign of a zero component folds to +1 as the golden does.
    let z_sign = select(1.0, -1.0, n.z < 0.0);
    let a = -1.0 / (z_sign + n.z);
    let b = n.x * n.y * a;
    let tangent = vec3<f32>(1.0 + z_sign * n.x * n.x * a, z_sign * b, -z_sign * n.x);
    let bitangent = vec3<f32>(b, z_sign + n.y * n.y * a, -n.y);

    let min_step = params.voxel_size * 0.5;
    let step_count_u = u32(step_count_f);

    var visibility_sum = 0.0;
    var weight_sum = 0.0;
    var bent = vec3<f32>(0.0, 0.0, 0.0);

    for (var ci = 0u; ci < 7u; ci = ci + 1u) {
        let cone = cone_dir(ci);
        // Rotate the tangent-space cone direction into world space.
        let dir = vec3<f32>(
            tangent.x * cone.x + bitangent.x * cone.y + n.x * cone.z,
            tangent.y * cone.x + bitangent.y * cone.y + n.y * cone.z,
            tangent.z * cone.x + bitangent.z * cone.y + n.z * cone.z,
        );
        let weight = cone.z;

        var cone_visibility = 1.0;
        var t = min_step;
        for (var s = 0u; s < step_count_u; s = s + 1u) {
            if (t >= max_distance) {
                break;
            }
            let sample_point = vec3<f32>(px + t * dir.x, py + t * dir.y, pz + t * dir.z);
            let sampled = signed_distance_at(sample_point);
            let radius = tan_half_angle * t;
            // Fraction of the cone cross-section left open at this step.
            var open_frac = 1.0;
            if (radius > min_positive) {
                open_frac = clamp(sampled / radius, 0.0, 1.0);
            }
            cone_visibility = min(cone_visibility, open_frac);
            if (cone_visibility <= 0.0) {
                break;
            }
            t = t + max(sampled, min_step);
        }

        visibility_sum = visibility_sum + cone_visibility * weight;
        weight_sum = weight_sum + weight;
        bent = bent + dir * (cone_visibility * weight);
    }

    var visibility = 1.0;
    if (weight_sum > min_positive) {
        visibility = clamp(visibility_sum / weight_sum, 0.0, 1.0);
    }

    // Renormalize the accumulated direction; fall back to the geometric normal
    // when every cone is fully occluded (zero-length sum).
    let bent_len2 = bent.x * bent.x + bent.y * bent.y + bent.z * bent.z;
    var bent_normal = n;
    if (bent_len2 > min_positive) {
        let inv_bent = 1.0 / sqrt(bent_len2);
        bent_normal = vec3<f32>(bent.x * inv_bent, bent.y * inv_bent, bent.z * inv_bent);
    }

    dst[out_base] = visibility;
    dst[out_base + 1u] = bent_normal.x;
    dst[out_base + 2u] = bent_normal.y;
    dst[out_base + 3u] = bent_normal.z;
    dst[out_base + 4u] = 1.0;
}
"#;

/// Uniform parameters for one dispatch: the field dimensions, the query count,
/// the field origin and the voxel edge length, laid out to match `Params` in
/// [`MESH_SDF_CONE_OCCLUSION_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Field cells along `x`.
    dim_x: u32,
    /// Field cells along `y`.
    dim_y: u32,
    /// Field cells along `z`.
    dim_z: u32,
    /// Number of valid shaded points in the input and output buffers.
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

/// A uniform signed-distance field as seen by [`GpuSdfConeOcclusion`]: the
/// per-cell signed distances plus the grid metadata the trilinear sampler
/// needs.
///
/// Cells are row-major with `x` varying fastest, matching the golden
/// `linear_index` of
/// `prism_render_architecture::ray_scene::mesh_signed_distance_field`. The host
/// builds this from any source (the golden `signed_distance_field`, an analytic
/// field, or a test fixture); the twin only ever reads the flat `distances`
/// array. The same field is shared by every shaded point in one dispatch.
#[derive(Clone, Debug)]
pub struct SdfConeOcclusionField {
    dims: [u32; 3],
    origin: [f32; 3],
    voxel_size: f32,
    distances: Vec<f32>,
}

impl SdfConeOcclusionField {
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
        SdfConeOcclusionField {
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

/// One cone-occlusion query the twin cone-traces through the shared field.
///
/// `position` is the shaded surface point and `normal` points away from the
/// surface (normalized on the device). `tan_half_angle` is the tangent of each
/// cone's half-angle (wider cones catch more occluders) and `max_distance` and
/// `step_count` bound each cone's march.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfConeOcclusionQuery {
    /// World-space shaded point the cones fan out from.
    pub position: [f32; 3],
    /// World-space surface normal (normalized on the device).
    pub normal: [f32; 3],
    /// Tangent of each cone's half-angle.
    pub tan_half_angle: f32,
    /// World-space march end bound for every cone.
    pub max_distance: f32,
    /// Maximum number of march iterations per cone.
    pub step_count: u32,
}

impl SdfConeOcclusionQuery {
    /// Builds a cone-occlusion query from its point, normal and march controls.
    #[must_use]
    pub fn new(
        position: [f32; 3],
        normal: [f32; 3],
        tan_half_angle: f32,
        max_distance: f32,
        step_count: u32,
    ) -> Self {
        SdfConeOcclusionQuery {
            position,
            normal,
            tan_half_angle,
            max_distance,
            step_count,
        }
    }
}

/// The cone-occlusion result for one shaded point: the hemisphere visibility
/// plus the bent normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfConeOcclusionResult {
    /// Hemisphere visibility factor in `0..=1` (one fully open).
    pub visibility: f32,
    /// Unit bent normal: the visibility-weighted mean cone direction, or the
    /// geometric normal when every cone is fully occluded.
    pub bent_normal: [f32; 3],
    /// `1` when the point was traced, `0` for a degenerate normal (the golden
    /// `None`).
    pub valid: u32,
}

/// A compiled, reusable signed-distance-field cone-occlusion kernel, twinning
/// the `CPU` golden `sdf_cone_occlusion` of
/// `prism_render_architecture::ray_scene::mesh_sdf_cone_occlusion`.
pub struct GpuSdfConeOcclusion {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfConeOcclusion {
    /// Compiles the signed-distance-field cone-occlusion kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is requested and compilation succeeds on any `Metal`,
    /// `Vulkan` or `DX12` backend.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_sdf_cone_occlusion_module"),
            source: ShaderSource::Wgsl(MESH_SDF_CONE_OCCLUSION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_cone_occlusion_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_cone_occlusion_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_sdf_cone_occlusion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sdf_cone_occlusion"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfConeOcclusion {
            module,
            layout,
            pipeline,
        }
    }

    /// Cone-traces every shaded point in `queries` through the shared `field`,
    /// mirroring the golden `sdf_cone_occlusion`.
    ///
    /// Returns one [`SdfConeOcclusionResult`] per query, in order. An empty query
    /// batch returns an empty vector with no dispatch issued (a storage buffer
    /// cannot be zero-sized).
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        field: &SdfConeOcclusionField,
        queries: &[SdfConeOcclusionQuery],
    ) -> Vec<SdfConeOcclusionResult> {
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
            label: Some("prism_volumetric_mesh_sdf_cone_occlusion_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let field_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_cone_occlusion_field"),
            contents: bytemuck::cast_slice(&field.distances),
            usage: BufferUsages::STORAGE,
        });

        let mut flat_queries = Vec::with_capacity(query_count * QUERY_STRIDE);
        for q in queries {
            flat_queries.push(q.position[0]);
            flat_queries.push(q.position[1]);
            flat_queries.push(q.position[2]);
            flat_queries.push(q.normal[0]);
            flat_queries.push(q.normal[1]);
            flat_queries.push(q.normal[2]);
            flat_queries.push(q.tan_half_angle);
            flat_queries.push(q.max_distance);
            flat_queries.push(q.step_count as f32);
            flat_queries.push(0.0);
        }
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_cone_occlusion_queries"),
            contents: bytemuck::cast_slice(&flat_queries),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (query_count * RESULT_STRIDE * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_sdf_cone_occlusion_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_sdf_cone_occlusion_bind_group"),
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
            label: Some("prism_volumetric_mesh_sdf_cone_occlusion_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_sdf_cone_occlusion_encoder"),
        });
        {
            let groups = (query_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_sdf_cone_occlusion_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per shaded point, flattened to a 1-D dispatch.
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
            // `valid` is written as exactly 0.0 or 1.0; it is decoded without an
            // f32 equality by testing the midpoint.
            out.push(SdfConeOcclusionResult {
                visibility: chunk[0],
                bent_normal: [chunk[1], chunk[2], chunk[3]],
                valid: u32::from(chunk[4] > 0.5),
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
