//! `wgpu` compute twin of the signed-distance-field thickness probe of the
//! `CPU` golden path — `sdf_thickness` in
//! `prism_render_architecture::ray_scene::mesh_sdf_thickness`, which marches
//! inward along a surface normal across the trilinear sampler
//! `sample_signed_distance` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch`.
//!
//! Translucency and subsurface shading need to know how much solid lies behind
//! a shaded point: a thin leaf lets light bleed through, a thick torso does
//! not. The golden path answers this by shooting a short ray *into* the
//! surface (along the negated outward normal) and watching the signed distance
//! field flip sign — negative inside the mesh — so the travelled distance at
//! the first re-emergence is the local thickness. [`GpuSdfThickness`] is the
//! on-device twin: each thread reads one surface sample, normalizes its
//! outward normal, and marches the inward ray in fixed `step` increments.
//!
//! # What is twinned
//!
//! The kernel reproduces the reference branch for branch: the same
//! `length_squared <= f32::MIN_POSITIVE` normalize guard (a degenerate normal
//! clears the `valid` flag), the same inward march bounded by both `max_steps`
//! iterations and the `max_distance` world budget, the same `distance < 0`
//! solid test setting the entered flag, the same `else if entered` re-emerge
//! early-out returning `t.min(max_distance)`, and the same loop-exhaustion
//! fallback of the full budget when still inside or zero when the ray never
//! entered. The trilinear sampler is twinned tap for tap: the same continuous
//! cell-center split `(point - origin) / voxel_size - 0.5`, the same `floor`
//! lower-corner clamp into `0..=dims-2`, the same degenerate single-layer axis
//! rule and the same eight-corner `clamp-to-border` blend.
//!
//! The field *construction* (the exact Euclidean transform, the solid
//! classification and the `signed_squared` integer storage with its `i64`/`f64`
//! `sqrt`) stays on the host: the twin consumes a flat `f32` array of per-cell
//! signed distances, which is the only data the reference sampler ever reads.
//!
//! # Correctness model
//!
//! Sampling is linear interpolation (`+`, `-`, `*`) plus integer `clamp`
//! addressing; the march is adds and ordered comparisons; the only `sqrt` is
//! the normal normalization. `CPU` and `GPU` evaluate the same closed form but
//! need not be bit-exact — a `GPU` may contract a multiply-add — so the parity
//! test asserts `abs_diff <= 1e-4 || rel_diff <= 1e-3` on the thickness. The
//! `valid`, `entered_solid` and `reemerged` flags are integer decisions and
//! match with no tolerance.
//!
//! # Degenerate inputs
//!
//! A zero-length normal cannot be normalized, so both sides report `valid = 0`
//! with every other field cleared (the golden `None`). An inward ray that
//! never meets a negative sample reports zero thickness; a solid thicker than
//! the probe budget reports the full budget. An empty query batch
//! short-circuits on the host with no dispatch (a storage buffer cannot be
//! zero-sized); the field itself always has at least one cell.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `min`,
//! `max`, `clamp`, `sqrt`, `select`, `bitcast`, `+ - * /` and unsigned index
//! arithmetic — with no `sin`, `cos`, `exp`, `pow`, `round`, optional device
//! feature or `u64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! The march loop is bounded by `max_steps`, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_thickness`；无第三方引擎源码或衍生代码。
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

/// The signed-distance-field thickness kernel, mirroring the `CPU` golden
/// `sdf_thickness` of
/// `prism_render_architecture::ray_scene::mesh_sdf_thickness`. One thread
/// handles one surface sample: it normalizes the outward normal, then marches
/// the inward ray in fixed `step` increments over the trilinearly sampled
/// field, tracking solid entry and the first re-emergence. `field` holds the
/// row-major per-cell signed distances (`x` varying fastest), `queries` holds
/// one `Query` per sample and `results` holds one `Res` per sample. Pure
/// linear interpolation, integer addressing, ordered comparisons and one
/// `sqrt`: no transcendental, no intrinsic, no `u64`, portable on `Metal`,
/// `Vulkan` and `DX12`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_thickness`；无第三方引擎源码或衍生代码。
const MESH_SDF_THICKNESS_WGSL: &str = r#"
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
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    nrm_x: f32,
    nrm_y: f32,
    nrm_z: f32,
    max_distance: f32,
    step: f32,
    max_steps: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Res {
    thickness: f32,
    entered_solid: u32,
    reemerged: u32,
    valid: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<Res>;

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
fn thickness_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    let q = queries[idx];

    // Normalize the outward normal exactly like the golden `normalize` guard.
    let len2 = q.nrm_x * q.nrm_x + q.nrm_y * q.nrm_y + q.nrm_z * q.nrm_z;
    // Smallest positive normal f32 (2^-126), the golden `f32::MIN_POSITIVE`.
    let min_positive = bitcast<f32>(8388608u);

    var thickness = 0.0;
    var entered = 0u;
    var reemerged = 0u;
    var valid = 0u;

    if (len2 > min_positive) {
        valid = 1u;
        let len = sqrt(len2);
        // March along the inward normal (negated outward normal).
        let inward = vec3<f32>(-q.nrm_x / len, -q.nrm_y / len, -q.nrm_z / len);
        let pos = vec3<f32>(q.pos_x, q.pos_y, q.pos_z);

        var t = 0.0;
        for (var i = 0u; i < q.max_steps; i = i + 1u) {
            if (t > q.max_distance) {
                break;
            }
            let sample = vec3<f32>(
                pos.x + t * inward.x,
                pos.y + t * inward.y,
                pos.z + t * inward.z,
            );
            let distance = signed_distance_at(sample);
            if (distance < 0.0) {
                entered = 1u;
            } else {
                if (entered == 1u) {
                    // Re-emerged into free space: t is the traversed thickness.
                    thickness = min(t, q.max_distance);
                    reemerged = 1u;
                    break;
                }
            }
            t = t + q.step;
        }

        if (reemerged == 0u) {
            // The march never exited: full budget when inside, else no solid.
            thickness = select(0.0, q.max_distance, entered == 1u);
        }
    }

    results[idx].thickness = thickness;
    results[idx].entered_solid = entered;
    results[idx].reemerged = reemerged;
    results[idx].valid = valid;
}
"#;

/// Uniform parameters for one dispatch: the field dimensions, the query count,
/// the field origin and the voxel edge length, laid out to match `Params` in
/// [`MESH_SDF_THICKNESS_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Field cells along `x`.
    dim_x: u32,
    /// Field cells along `y`.
    dim_y: u32,
    /// Field cells along `z`.
    dim_z: u32,
    /// Number of valid queries in the input and output buffers.
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

/// One query packed for the device, matching `Query` in
/// [`MESH_SDF_THICKNESS_WGSL`]. All fields are `4`-byte scalars, so the std430
/// stride is a clean `48` bytes with no interior vector-alignment padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    nrm_x: f32,
    nrm_y: f32,
    nrm_z: f32,
    max_distance: f32,
    step: f32,
    max_steps: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One result read back from the device, matching `Res` in
/// [`MESH_SDF_THICKNESS_WGSL`] (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    thickness: f32,
    entered_solid: u32,
    reemerged: u32,
    valid: u32,
}

/// A uniform signed-distance field as seen by [`GpuSdfThickness`]: the per-cell
/// signed distances plus the grid metadata the trilinear sampler needs.
///
/// Cells are row-major with `x` varying fastest, matching the golden
/// `linear_index` of
/// `prism_render_architecture::ray_scene::mesh_signed_distance_field`. The host
/// builds this from any source (the golden `signed_distance_field`, an analytic
/// field, or a test fixture); the twin only ever reads the flat `distances`
/// array.
#[derive(Clone, Debug)]
pub struct SdfThicknessField {
    dims: [u32; 3],
    origin: [f32; 3],
    voxel_size: f32,
    distances: Vec<f32>,
}

impl SdfThicknessField {
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
        SdfThicknessField {
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

/// One surface sample whose inward solid thickness the twin probes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfThicknessQuery {
    /// World-space surface position the inward march starts from.
    pub position: [f32; 3],
    /// Outward surface normal (normalized internally); the march follows its
    /// negation into the mesh.
    pub normal: [f32; 3],
    /// Probe budget in world units: the march stops once `t` exceeds this.
    pub max_distance: f32,
    /// Fixed world-space step between samples along the inward ray.
    pub step: f32,
    /// Maximum number of marching iterations before giving up.
    pub max_steps: u32,
}

impl SdfThicknessQuery {
    /// Builds a thickness query from its surface `position`, outward `normal`,
    /// probe `max_distance` budget, `step` size and `max_steps` iteration cap.
    #[must_use]
    pub fn new(
        position: [f32; 3],
        normal: [f32; 3],
        max_distance: f32,
        step: f32,
        max_steps: u32,
    ) -> Self {
        SdfThicknessQuery {
            position,
            normal,
            max_distance,
            step,
            max_steps,
        }
    }
}

/// The thickness and classification flags for one probed surface sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfThicknessResult {
    /// Travelled distance in `0..=max_distance`; meaningful only when `valid`.
    pub thickness: f32,
    /// `1` when the inward march ever sampled a negative (interior) distance.
    pub entered_solid: u32,
    /// `1` when the march re-emerged from the solid mid-flight (the early
    /// return), `0` when the loop exhausted its budget or step count.
    pub reemerged: u32,
    /// `1` when the outward normal was normalizable (the golden `Some`), `0`
    /// for a degenerate normal (the golden `None`), in which case every other
    /// field is zero.
    pub valid: u32,
}

/// A compiled, reusable signed-distance-field thickness kernel, twinning the
/// `CPU` golden `sdf_thickness` of
/// `prism_render_architecture::ray_scene::mesh_sdf_thickness`.
pub struct GpuSdfThickness {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfThickness {
    /// Compiles the signed-distance-field thickness kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is requested and compilation succeeds on any `Metal`,
    /// `Vulkan` or `DX12` backend.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_sdf_thickness_module"),
            source: ShaderSource::Wgsl(MESH_SDF_THICKNESS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_thickness_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_thickness_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_sdf_thickness_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("thickness_probe"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfThickness {
            module,
            layout,
            pipeline,
        }
    }

    /// Probes the inward solid thickness for every query in `queries`,
    /// mirroring the golden `sdf_thickness`.
    ///
    /// Returns one [`SdfThicknessResult`] per query, in order. An empty query
    /// batch returns an empty vector with no dispatch issued (a storage buffer
    /// cannot be zero-sized).
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        field: &SdfThicknessField,
        queries: &[SdfThicknessQuery],
    ) -> Vec<SdfThicknessResult> {
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
            label: Some("prism_volumetric_mesh_sdf_thickness_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let field_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_thickness_field"),
            contents: bytemuck::cast_slice(&field.distances),
            usage: BufferUsages::STORAGE,
        });

        let mut packed = Vec::with_capacity(query_count);
        for q in queries {
            packed.push(GpuQuery {
                pos_x: q.position[0],
                pos_y: q.position[1],
                pos_z: q.position[2],
                nrm_x: q.normal[0],
                nrm_y: q.normal[1],
                nrm_z: q.normal[2],
                max_distance: q.max_distance,
                step: q.step,
                max_steps: q.max_steps,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            });
        }
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_thickness_queries"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (query_count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_sdf_thickness_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_sdf_thickness_bind_group"),
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
            label: Some("prism_volumetric_mesh_sdf_thickness_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_sdf_thickness_encoder"),
        });
        {
            let groups = (query_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_sdf_thickness_pass"),
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

        let mut out = Vec::with_capacity(query_count);
        for r in raw {
            out.push(SdfThicknessResult {
                thickness: r.thickness,
                entered_solid: r.entered_solid,
                reemerged: r.reemerged,
                valid: r.valid,
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
