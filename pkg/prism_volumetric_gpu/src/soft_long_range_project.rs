//! `wgpu` compute twin of the one-sided long-range-attachment (LRA) leash
//! projection from the `CPU` golden
//! `prism_physics_core::soft::constraint::long_range::project_long_range`.
//!
//! A long-range attachment is a *one-sided* distance constraint tying one cloth
//! particle to a fixed anchor: the particle moves freely while it stays within
//! `max_distance` of the anchor, and is pulled back onto the leash sphere only
//! once it drifts farther. Seeding every particle with a geodesic leash to a
//! nearby kinematic attachment removes the long-wavelength stretch that pure
//! local distance constraints need many iterations to resolve, so a garment
//! stops sagging like rubber under fast motion without a global solve.
//!
//! [`GpuSoftLongRangeProject`] is the on-device twin: one thread solves one
//! constraint, reproducing the golden's `XPBD` update
//!
//! ```text
//! C          = |p - anchor| - max_distance   (projected only while C > 0)
//! n          = (p - anchor) / |p - anchor|
//! alpha      = compliance / (dt * dt)
//! d_lambda   = (-C - alpha * lambda) / (w + alpha)
//! p'         = p + n * (d_lambda * w)
//! lambda'    = lambda + d_lambda
//! ```
//!
//! The anchor is a fixed (infinite-mass) point, so the particle takes the whole
//! correction; a `compliance` of `0` makes the leash perfectly rigid.
//!
//! # What is twinned
//!
//! The single stateless, no-`RNG` entry point `project_long_range` over a raw
//! particle index. For each query the kernel reproduces the branch structure of
//! the golden: a pinned particle (`w <= 0`) and a particle coincident with its
//! anchor (`|p - anchor| < EPSILON`) are inert and reported invalid; a particle
//! still inside the leash sphere (`C <= 0`) is a valid no-op; otherwise the
//! particle is pulled toward the anchor by the compliant correction.
//!
//! # Correctness model
//!
//! Every quantity threads through multiplies, adds, guarded divisions and one
//! `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous quantity; the discrete `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! A pinned particle (`w <= 0`) and one coincident with its anchor
//! (`|p - anchor| < EPSILON`, `EPSILON = f32::EPSILON`) are both inert and
//! reported `valid = 0` with the position and multiplier passed through
//! unchanged. A particle inside the leash sphere is `valid = 1` but still a
//! no-op. Fixtures and the sweep keep the stretch a safe margin clear of the
//! `C = 0` knee so a few units in the last place cannot flip the branch, and a
//! strictly positive `dt` keeps `alpha` finite. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `length`, `select`,
//! ordered compares, `+ - * /` and one `sqrt` via `length` — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, no `round`, no `f32` remainder, no `u64`/`i64`,
//! no `f64` and no optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::long_range::project_long_range`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` long-range-attachment projection kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `project` mirrors the `CPU` golden `project_long_range` branch for branch;
/// see the module documentation for the algorithm.
const SOFT_LONG_RANGE_PROJECT_WGSL: &str = r#"
// One-sided long-range-attachment twin: one thread per constraint pulls an
// overstretched particle back onto its leash sphere using the XPBD update
// d_lambda = (-C - alpha*lambda)/(w + alpha), p' = p + n*(d_lambda*w). It uses
// only the portable core-WGSL subset (length, select, ordered compares and
// + - * /) with no u64/i64/f64 and no transcendental, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_physics_core::soft::constraint::long_range::project_long_range；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the particle position, its inverse mass, the fixed anchor, the
// leash radius, the compliance, the accumulated multiplier and the timestep.
// Every vec3 is flattened to scalar lanes so the std430 layout never trips a
// 16-byte vector-alignment rule; the kernel rebuilds each vec3<f32>.
struct Query {
    px: f32,
    py: f32,
    pz: f32,
    w: f32,
    ax: f32,
    ay: f32,
    az: f32,
    max_distance: f32,
    compliance: f32,
    lambda: f32,
    dt: f32,
}

// One result: the projected position, the updated multiplier and a valid flag.
struct Res {
    new_px: f32,
    new_py: f32,
    new_pz: f32,
    new_lambda: f32,
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// Coincidence epsilon; mirrors EPSILON = f32::EPSILON in the golden crate.
const EPSILON: f32 = 1.1920929e-7;

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Default: pass the position and multiplier through unchanged, invalid.
    var out: Res;
    out.new_px = q.px;
    out.new_py = q.py;
    out.new_pz = q.pz;
    out.new_lambda = q.lambda;
    out.valid = 0u;

    // A pinned (infinite-mass) particle is inert and reported invalid.
    if (q.w <= 0.0) {
        results[idx] = out;
        return;
    }

    let delta = vec3<f32>(q.px - q.ax, q.py - q.ay, q.pz - q.az);
    let dist = length(delta);

    // Coincident with the anchor: no defined gradient, inert and invalid.
    if (dist < EPSILON) {
        results[idx] = out;
        return;
    }

    // One-sided: slack inside the leash sphere is a valid no-op.
    let c = dist - q.max_distance;
    if (c <= 0.0) {
        out.valid = 1u;
        results[idx] = out;
        return;
    }

    // dist >= EPSILON guards this division; the normal is the unit gradient.
    let normal = delta / dist;
    let alpha_tilde = q.compliance / (q.dt * q.dt);
    // w > 0 and alpha_tilde >= 0, so the denominator is strictly positive; the
    // ordered guard keeps the division out of an undefined divide-by-zero.
    let denom = q.w + alpha_tilde;
    let delta_lambda = select(0.0, (-c - alpha_tilde * q.lambda) / denom, denom > 0.0);
    let corr = normal * (delta_lambda * q.w);

    out.new_px = q.px + corr.x;
    out.new_py = q.py + corr.y;
    out.new_pz = q.pz + corr.z;
    out.new_lambda = q.lambda + delta_lambda;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SOFT_LONG_RANGE_PROJECT_WGSL`].
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
    px: f32,
    py: f32,
    pz: f32,
    w: f32,
    ax: f32,
    ay: f32,
    az: f32,
    max_distance: f32,
    compliance: f32,
    lambda: f32,
    dt: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Res` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    new_px: f32,
    new_py: f32,
    new_pz: f32,
    new_lambda: f32,
    valid: u32,
}

/// One query for the long-range-attachment twin: the particle position, its
/// inverse mass `w`, the fixed `anchor`, the leash `max_distance`, the
/// `compliance`, the accumulated Lagrange `lambda` and the timestep `dt`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftLongRangeProjectQuery {
    /// Particle world position.
    pub position: [f32; 3],
    /// Particle inverse mass; `0` (or negative) marks a pinned particle.
    pub w: f32,
    /// Fixed world-space anchor the leash is measured from.
    pub anchor: [f32; 3],
    /// Maximum allowed distance from the anchor; the leash acts only beyond it.
    pub max_distance: f32,
    /// Compliance (inverse stiffness); `0` is a perfectly rigid leash.
    pub compliance: f32,
    /// Accumulated Lagrange multiplier for the current substep.
    pub lambda: f32,
    /// Substep timestep; expected strictly positive.
    pub dt: f32,
}

impl SoftLongRangeProjectQuery {
    /// Builds a query from the particle position, inverse mass, anchor, leash
    /// radius, compliance, accumulated multiplier and timestep.
    #[must_use]
    pub fn new(
        position: [f32; 3],
        w: f32,
        anchor: [f32; 3],
        max_distance: f32,
        compliance: f32,
        lambda: f32,
        dt: f32,
    ) -> SoftLongRangeProjectQuery {
        SoftLongRangeProjectQuery {
            position,
            w,
            anchor,
            max_distance,
            compliance,
            lambda,
            dt,
        }
    }
}

/// One resolved answer for a single query, mirroring the in-place position and
/// returned multiplier of the reference `project_long_range`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftLongRangeProjectResult {
    /// The projected particle position; unchanged for an inert query.
    pub position: [f32; 3],
    /// The updated accumulated Lagrange multiplier; unchanged for an inert
    /// query.
    pub lambda: f32,
    /// `1` when the projection produced a defined decision (either a correcting
    /// pull or a valid slack no-op), `0` for a pinned or anchor-coincident
    /// particle.
    pub valid: u32,
}

/// Encodes one [`SoftLongRangeProjectQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &SoftLongRangeProjectQuery) -> GpuQuery {
    GpuQuery {
        px: q.position[0],
        py: q.position[1],
        pz: q.position[2],
        w: q.w,
        ax: q.anchor[0],
        ay: q.anchor[1],
        az: q.anchor[2],
        max_distance: q.max_distance,
        compliance: q.compliance,
        lambda: q.lambda,
        dt: q.dt,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`SoftLongRangeProjectResult`].
fn decode_result(raw: &GpuResult) -> SoftLongRangeProjectResult {
    SoftLongRangeProjectResult {
        position: [raw.new_px, raw.new_py, raw.new_pz],
        lambda: raw.new_lambda,
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

/// A compiled, reusable long-range-attachment compute pipeline, twinning the
/// `CPU` golden `project_long_range`.
pub struct GpuSoftLongRangeProject {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSoftLongRangeProject {
    /// Compiles the long-range-attachment kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftLongRangeProject {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_long_range_project"),
            source: ShaderSource::Wgsl(SOFT_LONG_RANGE_PROJECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_soft_long_range_project_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_soft_long_range_project_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_long_range_project_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("project"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftLongRangeProject {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SoftLongRangeProjectResult`] per input, in order.
    ///
    /// Each continuous channel matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SoftLongRangeProjectQuery],
    ) -> Vec<SoftLongRangeProjectResult> {
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
            label: Some("prism_volumetric_soft_long_range_project_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_long_range_project_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_long_range_project_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_soft_long_range_project_bind_group"),
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
            label: Some("prism_volumetric_soft_long_range_project_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_soft_long_range_project_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_soft_long_range_project_pass"),
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
