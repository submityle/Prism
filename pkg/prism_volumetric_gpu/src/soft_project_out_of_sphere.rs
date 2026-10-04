//! `wgpu` compute twin of the single-point sphere projection from the `CPU`
//! golden `prism_physics_core::soft::collision::body::project_out_of_sphere`.
//!
//! A soft-body collision pass pushes a particle that has sunk inside a sphere
//! collider back out onto the sphere surface along the radial direction. This
//! is the stateless single-point kernel of that pass: given a position, a
//! sphere center and a radius it returns the projected position and a flag
//! telling whether a correction was applied.
//!
//! [`GpuSoftProjectOutOfSphere`] is the on-device twin: one thread solves one
//! query, reproducing the golden's branch structure
//!
//! ```text
//! radius <= 0            -> echo pos            (no collider)
//! dist_sq >= radius^2    -> echo pos            (already on/outside)
//! dist_sq <= EPS_LEN_SQ  -> center + (0,r,0)    (coincident +Y fallback)
//! otherwise              -> center + dir*radius (radial push to surface)
//! ```
//!
//! where `delta = pos - center`, `dist_sq = dot(delta, delta)` and
//! `dir = delta / sqrt(dist_sq)`.
//!
//! # What is twinned
//!
//! The single stateless, no-`RNG` entry point `project_out_of_sphere`. The
//! `valid` flag is `1` when the point was projected onto the surface (including
//! the deterministic `+Y` fallback for a center-coincident point) and `0` when
//! the point is echoed unchanged (a non-positive radius, or a point already on
//! or outside the sphere).
//!
//! # Correctness model
//!
//! Each quantity threads through multiplies, adds, guarded compares and one
//! `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous quantity; the discrete `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! A non-positive radius and a point already on or outside the sphere echo the
//! position unchanged with `valid = 0`. A point coincident with the center
//! (`dist_sq <= EPS_LEN_SQ`, `EPS_LEN_SQ = 1e-12`) has no defined radial
//! direction, so it is nudged out along `+Y` to a deterministic, non-`NaN`
//! result and reported `valid = 1`. All degenerate guards are ordered compares,
//! never a self-equality test, so a fast-math device cannot fold them away.
//! Fixtures and the sweep keep the squared distance a safe margin clear of both
//! the `radius^2` knee and the `EPS_LEN_SQ` floor. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `dot`,
//! ordered compares, `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, no
//! `round`, no `f32` remainder, no `u64`/`i64`, no `f64` and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::body::project_out_of_sphere`；无第三方引擎源码或衍生代码。
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
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` sphere-projection kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `project`
/// mirrors the `CPU` golden `project_out_of_sphere` branch for branch; see the
/// module documentation for the algorithm.
const SOFT_PROJECT_OUT_OF_SPHERE_WGSL: &str = r#"
// Single-point sphere projection twin: one thread per query pushes a particle
// that has sunk inside a sphere collider back out onto the surface along the
// radial direction, echoing the position when there is nothing to correct. It
// uses only the portable core-WGSL subset (sqrt, dot, ordered compares and
// + - * /) with no u64/i64/f64 and no transcendental, so it runs unmodified on
// Metal, Vulkan and DX12. Every degenerate guard is an ordered compare, never a
// self-equality test, so a fast-math device cannot fold it away.
//
// Provenance: 孪生自本仓 prism_physics_core::soft::collision::body::project_out_of_sphere；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the particle position, the sphere center and the sphere radius.
// Every vec3 is flattened to scalar lanes so the std430 layout never trips a
// 16-byte vector-alignment rule; the kernel rebuilds each vec3<f32>.
struct Query {
    posx: f32,
    posy: f32,
    posz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
}

// One result: the projected position and a valid flag.
struct Res {
    new_posx: f32,
    new_posy: f32,
    new_posz: f32,
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// Squared-length floor below which the point is treated as coincident with the
// center; mirrors EPS_LEN_SQ = 1e-12 in the golden crate.
const EPS_LEN_SQ: f32 = 1e-12;

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let pos = vec3<f32>(q.posx, q.posy, q.posz);
    let center = vec3<f32>(q.cx, q.cy, q.cz);

    // Default: echo the position unchanged, invalid.
    var out: Res;
    out.new_posx = q.posx;
    out.new_posy = q.posy;
    out.new_posz = q.posz;
    out.valid = 0u;

    // A non-positive radius has no collider; echo the point.
    if (q.radius <= 0.0) {
        results[idx] = out;
        return;
    }

    let delta = pos - center;
    let dist_sq = dot(delta, delta);

    // Already on or outside the sphere; echo the point.
    if (dist_sq >= q.radius * q.radius) {
        results[idx] = out;
        return;
    }

    // Coincident with the center: no defined radial direction, so nudge out
    // along +Y for a deterministic, non-NaN result.
    if (dist_sq <= EPS_LEN_SQ) {
        out.new_posx = q.cx;
        out.new_posy = q.cy + q.radius;
        out.new_posz = q.cz;
        out.valid = 1u;
        results[idx] = out;
        return;
    }

    // dist_sq > EPS_LEN_SQ guards this division; push the point onto the
    // surface along the unit radial direction.
    let dir = delta / sqrt(dist_sq);
    let projected = center + dir * q.radius;
    out.new_posx = projected.x;
    out.new_posy = projected.y;
    out.new_posz = projected.z;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SOFT_PROJECT_OUT_OF_SPHERE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The `vec3` inputs are flattened to scalar lanes so the layout never trips a
/// `16`-byte vector-alignment rule; the kernel rebuilds each `vec3<f32>`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    posx: f32,
    posy: f32,
    posz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Res` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    new_posx: f32,
    new_posy: f32,
    new_posz: f32,
    valid: u32,
}

/// One query for the sphere-projection twin: the particle `position`, the
/// sphere `center` and the sphere `radius`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftProjectOutOfSphereQuery {
    /// Particle world position.
    pub position: [f32; 3],
    /// Sphere center.
    pub center: [f32; 3],
    /// Sphere radius; a non-positive radius echoes the point unchanged.
    pub radius: f32,
}

impl SoftProjectOutOfSphereQuery {
    /// Builds a query from the particle position, sphere center and radius.
    #[must_use]
    pub fn new(position: [f32; 3], center: [f32; 3], radius: f32) -> SoftProjectOutOfSphereQuery {
        SoftProjectOutOfSphereQuery {
            position,
            center,
            radius,
        }
    }
}

/// One resolved answer for a single query, mirroring the projected position
/// returned by the reference `project_out_of_sphere`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftProjectOutOfSphereResult {
    /// The projected particle position; unchanged for an echoed query.
    pub position: [f32; 3],
    /// `1` when the point was projected onto the surface (including the `+Y`
    /// fallback for a center-coincident point), `0` when the point was echoed
    /// unchanged (non-positive radius, or already on or outside the sphere).
    pub valid: u32,
}

/// Encodes one [`SoftProjectOutOfSphereQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &SoftProjectOutOfSphereQuery) -> GpuQuery {
    GpuQuery {
        posx: q.position[0],
        posy: q.position[1],
        posz: q.position[2],
        cx: q.center[0],
        cy: q.center[1],
        cz: q.center[2],
        radius: q.radius,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`SoftProjectOutOfSphereResult`].
fn decode_result(raw: &GpuResult) -> SoftProjectOutOfSphereResult {
    SoftProjectOutOfSphereResult {
        position: [raw.new_posx, raw.new_posy, raw.new_posz],
        valid: raw.valid,
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

/// A compiled, reusable sphere-projection compute pipeline, twinning the `CPU`
/// golden `project_out_of_sphere`.
pub struct GpuSoftProjectOutOfSphere {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSoftProjectOutOfSphere {
    /// Compiles the sphere-projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftProjectOutOfSphere {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_sphere"),
            source: ShaderSource::Wgsl(SOFT_PROJECT_OUT_OF_SPHERE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_sphere_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_sphere_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_sphere_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("project"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftProjectOutOfSphere {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SoftProjectOutOfSphereResult`] per input, in order.
    ///
    /// Each continuous channel matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SoftProjectOutOfSphereQuery],
    ) -> Vec<SoftProjectOutOfSphereResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_sphere_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_sphere_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_sphere_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_sphere_bind_group"),
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
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_sphere_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_sphere_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_soft_project_out_of_sphere_pass"),
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

        raw.iter().map(decode_result).collect()
    }
}
