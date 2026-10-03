//! `wgpu` compute twin of the signed-distance-field surface-curvature reference
//! of the `CPU` golden path — `sdf_curvature` and `curvature_from_derivatives`
//! in `prism_render_architecture::ray_scene::mesh_sdf_curvature`, which
//! differentiate the trilinear sampler `sample_signed_distance` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch` twice.
//!
//! Where [`crate::mesh_sdf_normal`] differentiates the field once to recover
//! the shading normal, this module differentiates it twice to recover how the
//! surface bends. Curvature drives cavity and edge-wear masks, curvature
//! adaptive tessellation and anisotropic highlight shaping. [`GpuSdfCurvature`]
//! is the on-device twin: each thread reads one world-space query point,
//! trilinearly samples the shared field on a one-voxel central-difference
//! stencil to build the gradient and the symmetric Hessian, then evaluates
//! Ron Goldman's implicit-surface mean and Gaussian curvatures and splits them
//! into the two principal curvatures.
//!
//! # What is twinned
//!
//! The kernel reproduces the reference tap for tap: the same continuous
//! cell-center split `(point - origin) / voxel_size - 0.5`, the same `floor`
//! lower-corner clamp into `0..=dims-2`, the same degenerate single-layer axis
//! rule, the same eight-corner `clamp-to-border` trilinear blend, the same
//! `h = voxel_size` central-difference gradient, the same diagonal
//! `f(+) - 2 f0 + f(-)` and four-corner mixed second derivatives, the same
//! `g^T` quadratic forms with the Hessian and its adjugate, and the same
//! `g2 <= f32::MIN_POSITIVE` zero-gradient guard that clears the `valid` flag
//! and zeroes the four curvature outputs. The discriminant is clamped to zero
//! exactly as the reference does so discretization noise never yields a `NaN`.
//!
//! The field *construction* (the exact Euclidean transform, the solid
//! classification and the `signed_squared` integer storage) stays on the host:
//! the twin consumes a flat `f32` array of per-cell signed distances uploaded
//! as a read-only storage buffer, which is the only data the reference sampler
//! ever reads.
//!
//! # Correctness model
//!
//! Sampling is linear interpolation (`+`, `-`, `*`) plus integer `clamp`
//! addressing; the gradient and Hessian are fixed sequences of differences
//! scaled by constants; the curvatures use two `sqrt`s (the gradient length and
//! the principal-curvature discriminant) and divisions. `CPU` and `GPU`
//! evaluate the same closed form but need not be bit-exact — a `GPU` may
//! contract a multiply-add — so the parity test asserts
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3`. The `valid` flag is an integer
//! decision and matches with no tolerance.
//!
//! # Degenerate inputs
//!
//! A constant field (every tap equal) yields a vanishing gradient, so both
//! sides return zero curvatures and `valid = 0`. A single layer along an axis
//! contributes no interpolation weight on that axis. An empty query batch
//! short-circuits on the host with no dispatch (a storage buffer cannot be
//! zero-sized); the field itself always has at least one cell.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `min`,
//! `max`, `clamp`, `sqrt`, `bitcast`, `+ - * /` and unsigned index arithmetic —
//! with no `sin`, `cos`, `exp`, `pow`, `round`, optional device feature or
//! `u64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The sample
//! taps and three axes are fixed, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_curvature`；无第三方引擎源码或衍生代码。
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

/// Flat `f32` slots written per query: `mean`, `gaussian`, `principal_max`,
/// `principal_min`, the `valid` flag and one padding word (a `6`-wide,
/// `24`-byte stride).
const RESULT_STRIDE: usize = 6;

/// The signed-distance-field curvature kernel, mirroring the `CPU` golden
/// `sdf_curvature` of `prism_render_architecture::ray_scene::mesh_sdf_curvature`.
///
/// `const MESH_SDF_CURVATURE_WGSL` is the inline, self-contained shader source;
/// it is the only program the twin ever compiles (there is no external
/// `wesl`/`wgsl` include).
const MESH_SDF_CURVATURE_WGSL: &str = r#"
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
@group(0) @binding(2) var<storage, read> points: array<f32>;
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

// Samples the field at point + h * (dx, dy, dz), matching the golden
// `offset_sample`.
fn offset_sample(p: vec3<f32>, dx: f32, dy: f32, dz: f32, h: f32) -> f32 {
    return signed_distance_at(vec3<f32>(p.x + h * dx, p.y + h * dy, p.z + h * dz));
}

@compute @workgroup_size(64)
fn sdf_curvature(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    let in_base = idx * 3u;
    let p = vec3<f32>(points[in_base], points[in_base + 1u], points[in_base + 2u]);

    let h = params.voxel_size;

    // First-order gradient from a symmetric one-voxel central difference.
    let inv = 1.0 / (2.0 * h);
    let gx = (offset_sample(p, 1.0, 0.0, 0.0, h) - offset_sample(p, -1.0, 0.0, 0.0, h)) * inv;
    let gy = (offset_sample(p, 0.0, 1.0, 0.0, h) - offset_sample(p, 0.0, -1.0, 0.0, h)) * inv;
    let gz = (offset_sample(p, 0.0, 0.0, 1.0, h) - offset_sample(p, 0.0, 0.0, -1.0, h)) * inv;

    // Symmetric Hessian: diagonal f(+) - 2 f0 + f(-), mixed four-corner.
    let centre = offset_sample(p, 0.0, 0.0, 0.0, h);
    let inv_sq = 1.0 / (h * h);
    let inv_quad = 1.0 / (4.0 * h * h);

    let m00 = (offset_sample(p, 1.0, 0.0, 0.0, h) - 2.0 * centre
        + offset_sample(p, -1.0, 0.0, 0.0, h)) * inv_sq;
    let m11 = (offset_sample(p, 0.0, 1.0, 0.0, h) - 2.0 * centre
        + offset_sample(p, 0.0, -1.0, 0.0, h)) * inv_sq;
    let m22 = (offset_sample(p, 0.0, 0.0, 1.0, h) - 2.0 * centre
        + offset_sample(p, 0.0, 0.0, -1.0, h)) * inv_sq;

    let m01 = (offset_sample(p, 1.0, 1.0, 0.0, h)
        - offset_sample(p, 1.0, -1.0, 0.0, h)
        - offset_sample(p, -1.0, 1.0, 0.0, h)
        + offset_sample(p, -1.0, -1.0, 0.0, h)) * inv_quad;
    let m02 = (offset_sample(p, 1.0, 0.0, 1.0, h)
        - offset_sample(p, 1.0, 0.0, -1.0, h)
        - offset_sample(p, -1.0, 0.0, 1.0, h)
        + offset_sample(p, -1.0, 0.0, -1.0, h)) * inv_quad;
    let m12 = (offset_sample(p, 0.0, 1.0, 1.0, h)
        - offset_sample(p, 0.0, 1.0, -1.0, h)
        - offset_sample(p, 0.0, -1.0, 1.0, h)
        + offset_sample(p, 0.0, -1.0, -1.0, h)) * inv_quad;

    // Goldman's implicit-surface curvatures from the gradient and Hessian.
    let g2 = gx * gx + gy * gy + gz * gz;
    // Smallest positive normal f32 (2^-126), the golden `f32::MIN_POSITIVE`.
    let min_positive = bitcast<f32>(8388608u);

    var mean = 0.0;
    var gaussian = 0.0;
    var principal_max = 0.0;
    var principal_min = 0.0;
    var valid = 0.0;
    if (g2 > min_positive) {
        let g_len = sqrt(g2);
        let g_len3 = g2 * g_len;
        let g4 = g2 * g2;

        // Symmetric Hessian entries: [[a, b, c], [b, d, e], [c, e, f]]
        //   a = m00, b = m01, c = m02, d = m11, e = m12, f = m22.
        let trace = m00 + m11 + m22;
        let ghg = gx * gx * m00 + gy * gy * m11 + gz * gz * m22
            + 2.0 * (gx * gy * m01 + gx * gz * m02 + gy * gz * m12);
        mean = (trace * g2 - ghg) / (2.0 * g_len3);

        // Adjugate of the symmetric Hessian (itself symmetric).
        let adj00 = m11 * m22 - m12 * m12;
        let adj11 = m00 * m22 - m02 * m02;
        let adj22 = m00 * m11 - m01 * m01;
        let adj01 = m02 * m12 - m01 * m22;
        let adj02 = m01 * m12 - m02 * m11;
        let adj12 = m01 * m02 - m00 * m12;
        let g_adj_g = gx * gx * adj00 + gy * gy * adj11 + gz * gz * adj22
            + 2.0 * (gx * gy * adj01 + gx * gz * adj02 + gy * gz * adj12);
        gaussian = g_adj_g / g4;

        let discriminant = max(mean * mean - gaussian, 0.0);
        let root = sqrt(discriminant);
        principal_max = mean + root;
        principal_min = mean - root;
        valid = 1.0;
    }

    let out_base = idx * 6u;
    dst[out_base] = mean;
    dst[out_base + 1u] = gaussian;
    dst[out_base + 2u] = principal_max;
    dst[out_base + 3u] = principal_min;
    dst[out_base + 4u] = valid;
    dst[out_base + 5u] = 0.0;
}
"#;

/// Uniform parameters for one dispatch: the field dimensions, the query count,
/// the field origin and the voxel edge length, laid out to match `Params` in
/// [`MESH_SDF_CURVATURE_WGSL`].
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

/// A uniform signed-distance field as seen by [`GpuSdfCurvature`]: the per-cell
/// signed distances plus the grid metadata the trilinear sampler needs.
///
/// Cells are row-major with `x` varying fastest, matching the golden
/// `linear_index` of
/// `prism_render_architecture::ray_scene::mesh_signed_distance_field`. The
/// host builds this from any source (the golden `signed_distance_field`, an
/// analytic field, or a test fixture); the twin only ever reads the flat
/// `distances` array.
#[derive(Clone, Debug)]
pub struct SdfCurvatureField {
    dims: [u32; 3],
    origin: [f32; 3],
    voxel_size: f32,
    distances: Vec<f32>,
}

impl SdfCurvatureField {
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
        SdfCurvatureField {
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

/// One world-space query point whose surface curvature the twin evaluates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfCurvatureQuery {
    /// World-space position sampled by the central-difference stencil.
    pub point: [f32; 3],
}

impl SdfCurvatureQuery {
    /// Builds a query at world-space `point`.
    #[must_use]
    pub fn new(point: [f32; 3]) -> Self {
        SdfCurvatureQuery { point }
    }
}

/// The mean, Gaussian and principal curvatures of the field at one query point.
///
/// All quantities share the convex-positive sign convention of the golden
/// module (the field's outward gradient is the surface normal). When the
/// gradient is too short to orient reliably the four curvatures are zero and
/// `valid` is cleared.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfCurvatureResult {
    /// Mean curvature `(k1 + k2) / 2` (convex positive, concave negative).
    pub mean: f32,
    /// Gaussian curvature `k1 * k2` (elliptic positive, hyperbolic negative).
    pub gaussian: f32,
    /// Larger (more convex) principal curvature `k1 = mean + sqrt(mean^2 - K)`.
    pub principal_max: f32,
    /// Smaller (more concave) principal curvature `k2 = mean - sqrt(mean^2 - K)`.
    pub principal_min: f32,
    /// `1` when the curvature is defined, `0` for a degenerate zero gradient.
    pub valid: u32,
}

/// A compiled, reusable signed-distance-field curvature kernel, twinning the
/// `CPU` golden `sdf_curvature` of
/// `prism_render_architecture::ray_scene::mesh_sdf_curvature`.
pub struct GpuSdfCurvature {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfCurvature {
    /// Compiles the signed-distance-field curvature kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is requested and compilation succeeds on any `Metal`,
    /// `Vulkan` or `DX12` backend.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_module"),
            source: ShaderSource::Wgsl(MESH_SDF_CURVATURE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sdf_curvature"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfCurvature {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the field curvature at every query point in `queries`,
    /// mirroring the golden `sdf_curvature`.
    ///
    /// Returns one [`SdfCurvatureResult`] per query, in order. An empty query
    /// batch returns an empty vector with no dispatch issued (a storage buffer
    /// cannot be zero-sized).
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        field: &SdfCurvatureField,
        queries: &[SdfCurvatureQuery],
    ) -> Vec<SdfCurvatureResult> {
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
            label: Some("prism_volumetric_mesh_sdf_curvature_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let field_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_field"),
            contents: bytemuck::cast_slice(&field.distances),
            usage: BufferUsages::STORAGE,
        });

        let mut flat_points = Vec::with_capacity(query_count * 3);
        for q in queries {
            flat_points.extend_from_slice(&q.point);
        }
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_points"),
            contents: bytemuck::cast_slice(&flat_points),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (query_count * RESULT_STRIDE * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_bind_group"),
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
                    resource: points_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_encoder"),
        });
        {
            let groups = (query_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_sdf_curvature_pass"),
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
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        let mut out = Vec::with_capacity(query_count);
        for chunk in flat.chunks_exact(RESULT_STRIDE) {
            // `valid` is written as exactly 0.0 or 1.0; decode without an f32
            // equality by testing the midpoint.
            let valid = u32::from(chunk[4] > 0.5);
            out.push(SdfCurvatureResult {
                mean: chunk[0],
                gaussian: chunk[1],
                principal_max: chunk[2],
                principal_min: chunk[3],
                valid,
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
