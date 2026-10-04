//! `wgpu` compute twin of the XPBD distance (stretch) constraint projection
//! from the `CPU` golden
//! `prism_physics_core::soft::constraint::distance::project_distance_constraint`.
//!
//! A distance constraint keeps two particles at a rest length; it is the
//! workhorse edge of the unified soft-body solver (cloth structural/shear
//! edges, rope links, lattice edges). The compliant XPBD projection drives the
//! constraint function `C = |p_a - p_b| - rest_length` to zero with a
//! warm-started Lagrange multiplier, splitting the correction between the two
//! endpoints by inverse mass so a pinned particle never moves. This module
//! ports that stateless, no-`RNG` projection onto the device: one compute
//! thread resolves one edge, so a passing real-device parity test is direct
//! evidence the kernel takes the same degeneracy guards and computes the same
//! compliant correction and multiplier update, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Each query is one edge, modelled explicitly by its two endpoints, two
//! inverse masses, rest length, compliance, incoming Lagrange multiplier and
//! timestep so the twin never touches the golden's `a == b` / index bounds
//! guards (those are host-side concerns the golden applies before the
//! arithmetic). For each edge the kernel reproduces the reference closed form
//! exactly:
//!
//! * `w_sum = wa + wb`; when `w_sum <= 0` the edge is inert (`valid = 0`,
//!   output echoes the input and `lambda`) — both particles are pinned.
//! * `delta = pa - pb`, `length = |delta|`; when `length < EPSILON` the edge is
//!   degenerate (`valid = 0`, output echoes the input) — the endpoints
//!   coincide and the direction is undefined.
//! * `normal = delta / length`, `c = length - rest_length`,
//!   `alpha_tilde = compliance / (dt * dt)`,
//!   `d_lambda = (-c - alpha_tilde * lambda) / (w_sum + alpha_tilde)`,
//!   `correction = normal * d_lambda`; the endpoints move
//!   `pa += correction * wa`, `pb -= correction * wb`, the multiplier updates
//!   `new_lambda = lambda + d_lambda`, and `valid = 1`.
//!
//! There is no loop: each thread performs one guarded projection, so the kernel
//! provably terminates.
//!
//! # Correctness model
//!
//! The correction threads through a subtraction, a `sqrt` and guarded
//! divisions, so `CPU` and `GPU` are not required to be bit-exact. The parity
//! test asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`) on each continuous output; the discrete `valid` flag is
//! compared exactly. Every branch uses an ordered compare (`<=`, `<`), so no
//! bare float equality is involved, and both divisions are reached only behind
//! their ordered guards.
//!
//! # Degenerate inputs
//!
//! A non-positive inverse-mass sum or a coincident endpoint pair reports
//! `valid = 0` with the output left equal to the input and `lambda` echoed
//! unchanged, exactly as the golden returns the incoming multiplier without
//! moving the particles. An empty query batch short-circuits on the host with
//! no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — ordered compares,
//! `length`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no `round`, no float modulo and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! Degeneracy guards use ordered comparisons rather than a bare `x != x` NaN
//! sentinel, so a fast-math driver that folds `x == x` to `true` cannot defeat
//! them.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::distance::project_distance_constraint`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` XPBD distance-projection kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `project`
/// mirrors the `CPU` golden `project_distance_constraint` branch for branch;
/// see the module documentation for the algorithm.
const SOFT_DISTANCE_PROJECT_WGSL: &str = r#"
// Distance-constraint twin: one thread per edge reproduces the compliant XPBD
// stretch projection project_distance_constraint applies to a two-particle
// edge. It mirrors the CPU golden branch for branch, uses only the portable
// core-WGSL subset (ordered compares, length, + - * / plus unsigned index
// math), takes no optional feature, and has no loop, so the kernel provably
// terminates.
//
// Provenance: 孪生自本仓
// prism_physics_core::soft::constraint::distance::project_distance_constraint；
// 无第三方引擎源码或衍生代码。

// Coincidence floor on the edge length, matching the golden EPSILON (f32::EPSILON).
const EPSILON: f32 = 1.1920929e-7;

struct Params {
    // Number of edges in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Endpoint A position (flattened vec3).
    pax: f32,
    pay: f32,
    paz: f32,
    // Endpoint B position (flattened vec3).
    pbx: f32,
    pby: f32,
    pbz: f32,
    // Inverse masses of the two endpoints.
    wa: f32,
    wb: f32,
    // Rest length, compliance, warm-started multiplier and timestep.
    rest_length: f32,
    compliance: f32,
    lambda: f32,
    dt: f32,
}

struct Result {
    // Projected endpoint A position (flattened vec3).
    new_pax: f32,
    new_pay: f32,
    new_paz: f32,
    // Projected endpoint B position (flattened vec3).
    new_pbx: f32,
    new_pby: f32,
    new_pbz: f32,
    // Updated accumulated Lagrange multiplier.
    new_lambda: f32,
    // 1 when the projection ran, 0 for a degenerate edge.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let pa = vec3<f32>(q.pax, q.pay, q.paz);
    let pb = vec3<f32>(q.pbx, q.pby, q.pbz);

    // Default output: unchanged input, lambda echoed, invalid.
    var out: Result;
    out.new_pax = pa.x;
    out.new_pay = pa.y;
    out.new_paz = pa.z;
    out.new_pbx = pb.x;
    out.new_pby = pb.y;
    out.new_pbz = pb.z;
    out.new_lambda = q.lambda;
    out.valid = 0u;

    let w_sum = q.wa + q.wb;
    // Both endpoints pinned: nothing to project.
    if (w_sum <= 0.0) {
        results[idx] = out;
        return;
    }

    let delta = pa - pb;
    let len = length(delta);
    // Coincident endpoints: direction is undefined, skip.
    if (len < EPSILON) {
        results[idx] = out;
        return;
    }

    let seg_dir = delta / len;
    let c = len - q.rest_length;
    let alpha_tilde = q.compliance / (q.dt * q.dt);
    let d_lambda = (-c - alpha_tilde * q.lambda) / (w_sum + alpha_tilde);
    let corr = seg_dir * d_lambda;
    let na = pa + corr * q.wa;
    let nb = pb - corr * q.wb;

    out.new_pax = na.x;
    out.new_pay = na.y;
    out.new_paz = na.z;
    out.new_pbx = nb.x;
    out.new_pby = nb.y;
    out.new_pbz = nb.z;
    out.new_lambda = q.lambda + d_lambda;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SOFT_DISTANCE_PROJECT_WGSL`].
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
/// All `vec3`s are flattened to three scalars so the slot contains no `vec3`
/// and the host and device agree on the array stride byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    pax: f32,
    pay: f32,
    paz: f32,
    pbx: f32,
    pby: f32,
    pbz: f32,
    wa: f32,
    wb: f32,
    rest_length: f32,
    compliance: f32,
    lambda: f32,
    dt: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. All `vec3`s are flattened to three scalars; the trailing
/// `new_lambda` plus `valid` keep the slot an even word count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    new_pax: f32,
    new_pay: f32,
    new_paz: f32,
    new_pbx: f32,
    new_pby: f32,
    new_pbz: f32,
    new_lambda: f32,
    valid: u32,
}

/// One query for the distance-projection twin: a single stretch edge given by
/// its two endpoints, their inverse masses, the rest length, the compliance,
/// the warm-started Lagrange multiplier and the timestep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftDistanceProjectQuery {
    /// Endpoint A position.
    pub pa: [f32; 3],
    /// Endpoint B position.
    pub pb: [f32; 3],
    /// Inverse mass of endpoint A.
    pub wa: f32,
    /// Inverse mass of endpoint B.
    pub wb: f32,
    /// Rest (reference) length of the edge.
    pub rest_length: f32,
    /// Compliance (inverse stiffness); `0` is perfectly rigid.
    pub compliance: f32,
    /// Warm-started accumulated Lagrange multiplier for the current substep.
    pub lambda: f32,
    /// Substep timestep in seconds; must be strictly positive.
    pub dt: f32,
}

impl SoftDistanceProjectQuery {
    /// Builds a query from the two endpoints, their inverse masses, the rest
    /// length, the compliance, the incoming multiplier and the timestep.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the golden project_distance_constraint parameter list"
    )]
    pub fn new(
        pa: [f32; 3],
        pb: [f32; 3],
        wa: f32,
        wb: f32,
        rest_length: f32,
        compliance: f32,
        lambda: f32,
        dt: f32,
    ) -> SoftDistanceProjectQuery {
        SoftDistanceProjectQuery {
            pa,
            pb,
            wa,
            wb,
            rest_length,
            compliance,
            lambda,
            dt,
        }
    }
}

/// One resolved answer for a single edge, mirroring the reference
/// `project_distance_constraint` in-place update and returned multiplier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftDistanceProjectResult {
    /// Projected endpoint A position; equals the input when invalid.
    pub new_pa: [f32; 3],
    /// Projected endpoint B position; equals the input when invalid.
    pub new_pb: [f32; 3],
    /// Updated accumulated Lagrange multiplier; equals the input `lambda` when
    /// invalid.
    pub new_lambda: f32,
    /// `1` when the projection ran, `0` for a degenerate edge (non-positive
    /// inverse-mass sum or coincident endpoints).
    pub valid: u32,
}

/// Encodes one [`SoftDistanceProjectQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SoftDistanceProjectQuery) -> GpuQuery {
    GpuQuery {
        pax: q.pa[0],
        pay: q.pa[1],
        paz: q.pa[2],
        pbx: q.pb[0],
        pby: q.pb[1],
        pbz: q.pb[2],
        wa: q.wa,
        wb: q.wb,
        rest_length: q.rest_length,
        compliance: q.compliance,
        lambda: q.lambda,
        dt: q.dt,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`SoftDistanceProjectResult`].
fn decode_result(raw: &GpuResult) -> SoftDistanceProjectResult {
    SoftDistanceProjectResult {
        new_pa: [raw.new_pax, raw.new_pay, raw.new_paz],
        new_pb: [raw.new_pbx, raw.new_pby, raw.new_pbz],
        new_lambda: raw.new_lambda,
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

/// A compiled, reusable distance-projection compute pipeline, twinning the
/// `CPU` golden `project_distance_constraint`.
pub struct GpuSoftDistanceProject {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSoftDistanceProject {
    /// Compiles the distance-projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftDistanceProject {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_distance_project"),
            source: ShaderSource::Wgsl(SOFT_DISTANCE_PROJECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_soft_distance_project_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_soft_distance_project_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_distance_project_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("project"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftDistanceProject {
            module,
            layout,
            pipeline,
        }
    }

    /// Projects every edge in `queries` and returns one
    /// [`SoftDistanceProjectResult`] per input, in order.
    ///
    /// Each continuous output matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SoftDistanceProjectQuery],
    ) -> Vec<SoftDistanceProjectResult> {
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
            label: Some("prism_volumetric_soft_distance_project_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_distance_project_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_distance_project_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_soft_distance_project_bind_group"),
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
            label: Some("prism_volumetric_soft_distance_project_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_soft_distance_project_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_soft_distance_project_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per edge, flattened to a 1-D dispatch.
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
