//! `wgpu` compute twin of the signed-distance-field soft-shadow reference of
//! the `CPU` golden path — `sdf_soft_shadow` in
//! `prism_render_architecture::ray_scene::mesh_sdf_soft_shadow`, which marches
//! a penumbra ray through the trilinear sampler `sample_signed_distance` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch`.
//!
//! A hard shadow ray answers only "is the light blocked?"; a *soft* shadow also
//! estimates how much of the light disc is occluded, producing the smooth
//! penumbra that `AAA` renderers rely on for grounded contact shadows. Inigo
//! Quilez's technique computes this for free during a sphere trace: at every
//! march step the ratio of the sampled distance to the travelled distance
//! bounds the cone half-angle that reached the surface unobstructed, and the
//! running minimum of that ratio approximates the visible fraction of the
//! light. [`GpuSdfSoftShadow`] is the on-device twin: each thread marches one
//! shadow ray through the shared field.
//!
//! # What is twinned
//!
//! The kernel reproduces the reference step for step: the same
//! `min_step = voxel_size * 0.125` floor, the same `t = min_distance.max(0)`
//! start, the same improved penumbra estimator with the overshoot correction
//! `y = sampled^2 / (2 * prev)` (zero on the first step), the same
//! `chord = sqrt(max(sampled^2 - y^2, 0))`, the same `reach = max(t - y,
//! min_step)`, the same running `min(visibility, sharpness * chord / reach)`,
//! the same early full-occlusion exit when `sampled <= hit_epsilon`, the same
//! `max_distance` and `max_steps` bounds and the same final
//! `visibility.clamp(0, 1)`.
//!
//! The golden sentinel `prev_distance = f32::INFINITY` is replicated with a
//! `has_prev` boolean so the first-step correction is exactly zero without ever
//! introducing a non-finite value into the arithmetic (`WGSL` has no portable
//! infinity literal and no `isfinite`).
//!
//! The golden returns [`None`] only when `direction` cannot be normalized; the
//! twin maps that to a `valid` flag cleared to `0` with all other outputs
//! zeroed, and a successful trace sets `valid = 1`.
//!
//! The field *construction* (the exact Euclidean transform, the solid
//! classification and the `signed_squared` integer storage) stays on the host:
//! the twin consumes a flat `f32` array of per-cell signed distances uploaded
//! as a read-only storage buffer, which is the only data the reference sampler
//! ever reads.
//!
//! # Correctness model
//!
//! The march is a bounded loop of multiplies, a `max`-guarded `sqrt`, a step
//! floor and a trilinear tap (linear interpolation plus integer `clamp`
//! addressing). `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact — a `GPU` may contract a multiply-add — so the parity test asserts
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` on `visibility`. The `steps`,
//! `occluded` and `valid` outputs are integer/boolean decisions and match with
//! no tolerance; the parity fixtures are shaped so the march terminates
//! unambiguously (hits avoided or forced, `max_distance` never crossed) so
//! those counts never straddle a floating-point boundary.
//!
//! # Degenerate inputs
//!
//! A zero-length `direction` cannot be normalized, so both sides report
//! `valid = 0` with a zero visibility, zero steps and no occlusion. A
//! `max_steps` of zero runs no iterations and leaves the light fully lit. An
//! empty query batch short-circuits on the host with no dispatch (a storage
//! buffer cannot be zero-sized); the field itself always has at least one cell.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `min`,
//! `max`, `clamp`, `sqrt`, `select`, `bitcast`, `+ - * /` and unsigned index
//! arithmetic — with no `sin`, `cos`, `exp`, `pow`, `round`, optional device
//! feature or `u64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! march is bounded by `max_steps`, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_soft_shadow`；无第三方引擎源码或衍生代码。
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

/// Flat `f32` slots read per query: origin `xyz`, direction `xyz`,
/// `min_distance`, `max_distance`, `sharpness`, `hit_epsilon`, `max_steps`
/// (encoded as `f32`) and one padding word (a `12`-wide stride).
const QUERY_STRIDE: usize = 12;

/// Flat `f32` slots written per query: `visibility`, the `steps` count, the
/// `occluded` flag and the `valid` flag (a `4`-wide, `16`-byte stride).
const RESULT_STRIDE: usize = 4;

/// Inlined `WGSL` for the signed-distance-field soft-shadow kernel. One thread
/// marches one shadow ray through the shared field with the improved Quilez
/// penumbra estimator. Pure linear interpolation, integer addressing, bounded
/// loop and one `sqrt` per step: no transcendental, no intrinsic, no `u64`,
/// portable on `Metal`, `Vulkan` and `DX12`.
const MESH_SDF_SOFT_SHADOW_WGSL: &str = r#"
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

@compute @workgroup_size(64)
fn sdf_soft_shadow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    let base = idx * 12u;
    let ox = queries[base];
    let oy = queries[base + 1u];
    let oz = queries[base + 2u];
    let dx = queries[base + 3u];
    let dy = queries[base + 4u];
    let dz = queries[base + 5u];
    let min_distance = queries[base + 6u];
    let max_distance = queries[base + 7u];
    let sharpness = queries[base + 8u];
    let hit_epsilon = queries[base + 9u];
    let max_steps_f = queries[base + 10u];

    let out_base = idx * 4u;

    // Smallest positive normal f32 (2^-126), the golden `f32::MIN_POSITIVE`.
    let min_positive = bitcast<f32>(8388608u);
    let len2 = dx * dx + dy * dy + dz * dz;
    if (len2 <= min_positive) {
        // Degenerate direction: the golden returns None.
        dst[out_base] = 0.0;
        dst[out_base + 1u] = 0.0;
        dst[out_base + 2u] = 0.0;
        dst[out_base + 3u] = 0.0;
        return;
    }

    let inv_len = 1.0 / sqrt(len2);
    let dir = vec3<f32>(dx * inv_len, dy * inv_len, dz * inv_len);
    let min_step = params.voxel_size * 0.125;

    var visibility = 1.0;
    var t = max(min_distance, 0.0);
    var has_prev = false;
    var prev_distance = 0.0;
    var steps = 0u;
    var occluded = false;
    let max_steps_u = u32(max_steps_f);

    for (var i = 0u; i < max_steps_u; i = i + 1u) {
        if (t > max_distance) {
            break;
        }
        steps = i + 1u;
        let position = vec3<f32>(ox + t * dir.x, oy + t * dir.y, oz + t * dir.z);
        let sampled = signed_distance_at(position);
        if (sampled <= hit_epsilon) {
            visibility = 0.0;
            occluded = true;
            break;
        }
        var y = 0.0;
        if (has_prev) {
            y = sampled * sampled / (2.0 * prev_distance);
        }
        let chord = sqrt(max(sampled * sampled - y * y, 0.0));
        let reach = max(t - y, min_step);
        visibility = min(visibility, sharpness * chord / reach);
        prev_distance = sampled;
        has_prev = true;
        t = t + max(sampled, min_step);
    }

    dst[out_base] = clamp(visibility, 0.0, 1.0);
    dst[out_base + 1u] = f32(steps);
    dst[out_base + 2u] = select(0.0, 1.0, occluded);
    dst[out_base + 3u] = 1.0;
}
"#;

/// Uniform parameters for one dispatch: the field dimensions, the query count,
/// the field origin and the voxel edge length, laid out to match `Params` in
/// [`MESH_SDF_SOFT_SHADOW_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Field cells along `x`.
    dim_x: u32,
    /// Field cells along `y`.
    dim_y: u32,
    /// Field cells along `z`.
    dim_z: u32,
    /// Number of valid shadow rays in the input and output buffers.
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

/// A uniform signed-distance field as seen by [`GpuSdfSoftShadow`]: the
/// per-cell signed distances plus the grid metadata the trilinear sampler
/// needs.
///
/// Cells are row-major with `x` varying fastest, matching the golden
/// `linear_index` of
/// `prism_render_architecture::ray_scene::mesh_signed_distance_field`. The host
/// builds this from any source (the golden `signed_distance_field`, an analytic
/// field, or a test fixture); the twin only ever reads the flat `distances`
/// array. The same field is shared by every shadow ray in one dispatch.
#[derive(Clone, Debug)]
pub struct SdfSoftShadowField {
    dims: [u32; 3],
    origin: [f32; 3],
    voxel_size: f32,
    distances: Vec<f32>,
}

impl SdfSoftShadowField {
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
        SdfSoftShadowField {
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

/// One soft-shadow ray the twin marches through the shared field.
///
/// `origin` is the shaded surface point and `direction` points toward the
/// light (normalized on the device, so `min_distance` and `max_distance` are
/// world-space march bounds). `sharpness` is the penumbra factor (larger
/// narrows the penumbra toward a hard shadow), `hit_epsilon` is the sampled
/// distance at which the march reports a full hit and `max_steps` bounds the
/// loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSoftShadowQuery {
    /// World-space surface point the ray starts from.
    pub origin: [f32; 3],
    /// World-space direction toward the light (normalized on the device).
    pub direction: [f32; 3],
    /// World-space march start offset (clamped to zero below).
    pub min_distance: f32,
    /// World-space march end bound.
    pub max_distance: f32,
    /// Penumbra sharpness factor.
    pub sharpness: f32,
    /// Sampled distance at or below which the ray reports a full hit.
    pub hit_epsilon: f32,
    /// Maximum number of march iterations.
    pub max_steps: u32,
}

impl SdfSoftShadowQuery {
    /// Builds a soft-shadow ray from its origin, direction and march controls.
    #[must_use]
    pub fn new(
        origin: [f32; 3],
        direction: [f32; 3],
        min_distance: f32,
        max_distance: f32,
        sharpness: f32,
        hit_epsilon: f32,
        max_steps: u32,
    ) -> Self {
        SdfSoftShadowQuery {
            origin,
            direction,
            min_distance,
            max_distance,
            sharpness,
            hit_epsilon,
            max_steps,
        }
    }
}

/// The soft-shadow result for one ray: the visible fraction plus the march
/// bookkeeping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSoftShadowResult {
    /// Estimated visible fraction of the light in `0..=1` (one fully lit).
    pub visibility: f32,
    /// Number of march iterations performed.
    pub steps: u32,
    /// `1` when the ray struck the surface (fully shadowed), `0` otherwise.
    pub occluded: u32,
    /// `1` when the ray was traced, `0` for a degenerate direction (the golden
    /// `None`).
    pub valid: u32,
}

/// A compiled, reusable signed-distance-field soft-shadow kernel, twinning the
/// `CPU` golden `sdf_soft_shadow` of
/// `prism_render_architecture::ray_scene::mesh_sdf_soft_shadow`.
pub struct GpuSdfSoftShadow {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfSoftShadow {
    /// Compiles the signed-distance-field soft-shadow kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is requested and compilation succeeds on any `Metal`,
    /// `Vulkan` or `DX12` backend.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_sdf_soft_shadow_module"),
            source: ShaderSource::Wgsl(MESH_SDF_SOFT_SHADOW_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_soft_shadow_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_soft_shadow_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_sdf_soft_shadow_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sdf_soft_shadow"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfSoftShadow {
            module,
            layout,
            pipeline,
        }
    }

    /// Marches every soft-shadow ray in `queries` through the shared `field`,
    /// mirroring the golden `sdf_soft_shadow`.
    ///
    /// Returns one [`SdfSoftShadowResult`] per query, in order. An empty query
    /// batch returns an empty vector with no dispatch issued (a storage buffer
    /// cannot be zero-sized).
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        field: &SdfSoftShadowField,
        queries: &[SdfSoftShadowQuery],
    ) -> Vec<SdfSoftShadowResult> {
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
            label: Some("prism_volumetric_mesh_sdf_soft_shadow_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let field_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_soft_shadow_field"),
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
            flat_queries.push(q.min_distance);
            flat_queries.push(q.max_distance);
            flat_queries.push(q.sharpness);
            flat_queries.push(q.hit_epsilon);
            flat_queries.push(q.max_steps as f32);
            flat_queries.push(0.0);
        }
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_soft_shadow_queries"),
            contents: bytemuck::cast_slice(&flat_queries),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (query_count * RESULT_STRIDE * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_sdf_soft_shadow_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_sdf_soft_shadow_bind_group"),
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
            label: Some("prism_volumetric_mesh_sdf_soft_shadow_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_sdf_soft_shadow_encoder"),
        });
        {
            let groups = (query_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_sdf_soft_shadow_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per shadow ray, flattened to a 1-D dispatch.
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
            // `steps` is an exact integer encoded as f32; `occluded` and `valid`
            // are written as exactly 0.0 or 1.0, decoded without an f32 equality
            // by testing the midpoint.
            out.push(SdfSoftShadowResult {
                visibility: chunk[0],
                steps: chunk[1] as u32,
                occluded: u32::from(chunk[2] > 0.5),
                valid: u32::from(chunk[3] > 0.5),
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
