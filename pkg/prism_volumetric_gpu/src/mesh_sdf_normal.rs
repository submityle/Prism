//! `wgpu` compute twin of the signed-distance-field surface-normal reference
//! of the `CPU` golden path — `sdf_gradient` and `sdf_normal` in
//! `prism_render_architecture::ray_scene::mesh_sdf_normal`, differentiating the
//! trilinear sampler `sample_signed_distance` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch`.
//!
//! A signed distance field stores one signed length per voxel cell center; the
//! gradient of that field points from the solid interior toward the exterior,
//! so once normalized it is the outward surface normal used to shade
//! sphere-traced hits and to bend ambient-occlusion and global-illumination
//! cones around geometry. [`GpuSdfNormal`] is the on-device twin: each thread
//! reads one world-space query point, trilinearly samples the shared field at
//! six half-step neighbours, forms a symmetric central-difference gradient and
//! then guards the normalization.
//!
//! # What is twinned
//!
//! The kernel reproduces the reference tap for tap: the same continuous
//! cell-center split `(point - origin) / voxel_size - 0.5`, the same `floor`
//! lower-corner clamp into `0..=dims-2`, the same degenerate single-layer axis
//! rule, the same eight-corner `clamp-to-border` trilinear blend, the same
//! `step = voxel_size`, `inv = 1 / (2 * step)` central difference, and the same
//! `length_squared <= f32::MIN_POSITIVE` zero-gradient guard that falls back to
//! a zero normal with the `valid` flag cleared.
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
//! addressing; the gradient is three differences scaled by a constant; the
//! normal divides by a single `sqrt`. `CPU` and `GPU` evaluate the same closed
//! form but need not be bit-exact — a `GPU` may contract a multiply-add — so
//! the parity test asserts `abs_diff <= 1e-4 || rel_diff <= 1e-3`. The `valid`
//! flag and the degenerate zero-normal fallback are integer/exact decisions and
//! match with no tolerance.
//!
//! # Degenerate inputs
//!
//! A constant field (every tap equal) yields a vanishing central difference, so
//! both sides return a zero gradient, a zero normal and `valid = 0`. A single
//! layer along an axis contributes no interpolation weight on that axis. An
//! empty query batch short-circuits on the host with no dispatch (a storage
//! buffer cannot be zero-sized); the field itself always has at least one cell.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `min`,
//! `max`, `clamp`, `sqrt`, `bitcast`, `+ - * /` and unsigned index arithmetic —
//! with no `sin`, `cos`, `exp`, `pow`, `round`, optional device feature or
//! `u64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The six sample
//! calls and three axes are fixed, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_normal`；无第三方引擎源码或衍生代码。
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

/// Flat `f32` slots written per query: gradient `xyz`, normal `xyz`, the
/// `valid` flag and one padding word (an `8`-wide, `32`-byte stride).
const RESULT_STRIDE: usize = 8;

/// The signed-distance-field normal kernel, mirroring the `CPU` golden
/// `sdf_gradient` and `sdf_normal` of
/// `prism_render_architecture::ray_scene::mesh_sdf_normal`. One thread handles
/// one world-space query point: it trilinearly samples the shared field at six
/// half-step neighbours, forms a symmetric central-difference gradient and
/// guards the normalization. `field` holds the row-major per-cell signed
/// distances (`x` varying fastest), `points` holds three `f32` per query and
/// `dst` holds eight `f32` per query. Pure linear interpolation, integer
/// addressing and one `sqrt`: no transcendental, no intrinsic, no `u64`,
/// portable on `Metal`, `Vulkan` and `DX12`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_normal`；无第三方引擎源码或衍生代码。
const MESH_SDF_NORMAL_WGSL: &str = r#"
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

@compute @workgroup_size(64)
fn sdf_normal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    let in_base = idx * 3u;
    let p = vec3<f32>(points[in_base], points[in_base + 1u], points[in_base + 2u]);

    let step = params.voxel_size;
    let inv = 1.0 / (2.0 * step);

    let fwd_x = vec3<f32>(p.x + step, p.y, p.z);
    let bwd_x = vec3<f32>(p.x - step, p.y, p.z);
    let fwd_y = vec3<f32>(p.x, p.y + step, p.z);
    let bwd_y = vec3<f32>(p.x, p.y - step, p.z);
    let fwd_z = vec3<f32>(p.x, p.y, p.z + step);
    let bwd_z = vec3<f32>(p.x, p.y, p.z - step);

    let gx = (signed_distance_at(fwd_x) - signed_distance_at(bwd_x)) * inv;
    let gy = (signed_distance_at(fwd_y) - signed_distance_at(bwd_y)) * inv;
    let gz = (signed_distance_at(fwd_z) - signed_distance_at(bwd_z)) * inv;

    let len2 = gx * gx + gy * gy + gz * gz;
    // Smallest positive normal f32 (2^-126), the golden `f32::MIN_POSITIVE`.
    let min_positive = bitcast<f32>(8388608u);

    var nx = 0.0;
    var ny = 0.0;
    var nz = 0.0;
    var valid = 0.0;
    if (len2 > min_positive) {
        let len = sqrt(len2);
        nx = gx / len;
        ny = gy / len;
        nz = gz / len;
        valid = 1.0;
    }

    let out_base = idx * 8u;
    dst[out_base] = gx;
    dst[out_base + 1u] = gy;
    dst[out_base + 2u] = gz;
    dst[out_base + 3u] = nx;
    dst[out_base + 4u] = ny;
    dst[out_base + 5u] = nz;
    dst[out_base + 6u] = valid;
    dst[out_base + 7u] = 0.0;
}
"#;

/// Uniform parameters for one dispatch: the field dimensions, the query count,
/// the field origin and the voxel edge length, laid out to match `Params` in
/// [`MESH_SDF_NORMAL_WGSL`].
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

/// A uniform signed-distance field as seen by [`GpuSdfNormal`]: the per-cell
/// signed distances plus the grid metadata the trilinear sampler needs.
///
/// Cells are row-major with `x` varying fastest, matching the golden
/// `linear_index` of
/// `prism_render_architecture::ray_scene::mesh_signed_distance_field`. The
/// host builds this from any source (the golden `signed_distance_field`, an
/// analytic field, or a test fixture); the twin only ever reads the flat
/// `distances` array.
#[derive(Clone, Debug)]
pub struct SdfNormalField {
    dims: [u32; 3],
    origin: [f32; 3],
    voxel_size: f32,
    distances: Vec<f32>,
}

impl SdfNormalField {
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
        SdfNormalField {
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

/// One world-space query point whose surface normal the twin evaluates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfNormalQuery {
    /// World-space position sampled by the central difference.
    pub point: [f32; 3],
}

impl SdfNormalQuery {
    /// Builds a query at world-space `point`.
    #[must_use]
    pub fn new(point: [f32; 3]) -> Self {
        SdfNormalQuery { point }
    }
}

/// The gradient and oriented normal of the field at one query point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfNormalResult {
    /// Central-difference gradient (not normalized).
    pub gradient: [f32; 3],
    /// Outward unit normal, or `[0, 0, 0]` when the gradient is degenerate.
    pub normal: [f32; 3],
    /// `1` when the normal is oriented, `0` for a degenerate zero gradient.
    pub valid: u32,
}

/// A compiled, reusable signed-distance-field normal kernel, twinning the `CPU`
/// golden `sdf_gradient` and `sdf_normal` of
/// `prism_render_architecture::ray_scene::mesh_sdf_normal`.
pub struct GpuSdfNormal {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfNormal {
    /// Compiles the signed-distance-field normal kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is requested and compilation succeeds on any `Metal`,
    /// `Vulkan` or `DX12` backend.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_sdf_normal_module"),
            source: ShaderSource::Wgsl(MESH_SDF_NORMAL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_normal_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_normal_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_sdf_normal_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sdf_normal"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfNormal {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the field normal at every query point in `queries`, mirroring
    /// the golden `sdf_normal` (with its unnormalized `sdf_gradient` reported
    /// alongside).
    ///
    /// Returns one [`SdfNormalResult`] per query, in order. An empty query batch
    /// returns an empty vector with no dispatch issued (a storage buffer cannot
    /// be zero-sized).
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        field: &SdfNormalField,
        queries: &[SdfNormalQuery],
    ) -> Vec<SdfNormalResult> {
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
            label: Some("prism_volumetric_mesh_sdf_normal_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let field_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_normal_field"),
            contents: bytemuck::cast_slice(&field.distances),
            usage: BufferUsages::STORAGE,
        });

        let mut flat_points = Vec::with_capacity(query_count * 3);
        for q in queries {
            flat_points.extend_from_slice(&q.point);
        }
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_normal_points"),
            contents: bytemuck::cast_slice(&flat_points),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (query_count * RESULT_STRIDE * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_sdf_normal_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_sdf_normal_bind_group"),
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
            label: Some("prism_volumetric_mesh_sdf_normal_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_sdf_normal_encoder"),
        });
        {
            let groups = (query_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_sdf_normal_pass"),
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
            let valid = u32::from(chunk[6] > 0.5);
            out.push(SdfNormalResult {
                gradient: [chunk[0], chunk[1], chunk[2]],
                normal: [chunk[3], chunk[4], chunk[5]],
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
