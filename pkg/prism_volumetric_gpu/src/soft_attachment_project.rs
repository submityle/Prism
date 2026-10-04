//! `wgpu` compute twin of the single-particle XPBD attachment projection from
//! the `CPU` golden `prism_physics_core::soft::constraint::attachment`
//! (`AttachmentConstraint::project`).
//!
//! An attachment constraint softly drives one particle toward a fixed world
//! anchor. Its constraint function is the raw distance to the target,
//! `C = |p - target|` (zero rest length), with unit gradient
//! `n = (p - target) / |p - target|`, so the `XPBD` denominator is
//! `w + alpha_tilde`. A single compliant projection step updates both the
//! particle position and the accumulated Lagrange multiplier.
//!
//! This module ports that stateless, closed-form projection onto the device:
//! one thread advances one particle. [`GpuSoftAttachmentProject`] is the
//! on-device twin; a passing real-device parity test is direct evidence the
//! kernel takes the same ordered guards and the same compliant-solver
//! arithmetic the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! * `AttachmentConstraint::project` — the whole closed form: the pinned guard
//!   `w <= 0`, the already-at-target guard `|p - target| < EPSILON`, the unit
//!   gradient `n = delta / length`, the compliant denominator
//!   `alpha_tilde = compliance / (dt * dt)`, the Lagrange delta
//!   `delta_lambda = (-length - alpha_tilde * lambda) / (w + alpha_tilde)`, and
//!   the position/multiplier updates `lambda += delta_lambda`,
//!   `p += n * (delta_lambda * w)`.
//!
//! Note the reference constraint value is the distance itself (rest length
//! `0`), so the numerator is `-length`, not `length - rest`.
//!
//! # Result encoding
//!
//! The reference mutates `positions[i]` and `self.lambda` in place and returns
//! nothing. The twin reports the next position (three scalars), the next
//! multiplier and a `valid` flag. `valid` is `1` when the projection ran and
//! `0` when either guard made the step inert (`w <= 0` pinned, or the particle
//! already sits on the target); in the inert case the position and multiplier
//! are returned unchanged, exactly as the reference leaves them.
//!
//! # Correctness model
//!
//! The position and multiplier are continuous and checked with an
//! absolute-or-relative tolerance; `valid` is discrete and checked exactly.
//! The division `/(w + alpha_tilde)` and the `length` call are the only
//! conditioning-sensitive operations, so fixtures and the sweep keep samples
//! clear of the `length = EPSILON` and `w = 0` knees where the discrete `valid`
//! channel could otherwise flip on a rounding tie; dedicated named fixtures pin
//! those degenerate cases.
//!
//! # Degenerate inputs
//!
//! A non-positive inverse mass (`w <= 0`) marks a pinned particle: the step is
//! inert and `valid = 0`. A particle already within `EPSILON` of the target has
//! no defined gradient direction, so the step is inert and `valid = 0`. Both
//! guards use ordered comparisons, so a `NaN` input routes to the inert branch
//! rather than producing a spurious update. An empty query batch short-circuits
//! on the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — ordered compares,
//! `select`, `length`, `max`, division and `vec3` arithmetic — with no `sin`,
//! `cos`, `tan`, `exp`, `log`, `pow`, no `round`, no float modulo and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. There is no bare float equality: every guard is an ordered compare.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::attachment`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` attachment-projection kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `AttachmentConstraint::project` branch for branch;
/// see the module documentation for the algorithm.
const SOFT_ATTACHMENT_PROJECT_WGSL: &str = r#"
// Soft attachment projection twin: one thread per particle runs one compliant
// XPBD projection step toward a fixed target, mirroring
// AttachmentConstraint::project. The constraint value is the raw distance to
// the target (zero rest length), so the numerator is -length.
// Provenance: 孪生自本仓 prism_physics_core::soft::constraint::attachment；无第三方引擎源码或衍生代码。

// Below this distance the particle is treated as already on the target and the
// gradient has no defined direction; matches f32::EPSILON used by the golden.
const EPSILON: f32 = 1.1920929e-7;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Particle position (x, y, z).
    px: f32,
    py: f32,
    pz: f32,
    // Inverse mass; <= 0 marks a pinned particle.
    w: f32,
    // Target anchor (x, y, z).
    tx: f32,
    ty: f32,
    tz: f32,
    // Compliance (inverse stiffness); >= 0.
    compliance: f32,
    // Accumulated Lagrange multiplier for this substep.
    lambda: f32,
    // Substep timestep; strictly > 0.
    dt: f32,
    pad0: f32,
    pad1: f32,
}

struct Result {
    // Next particle position (x, y, z).
    new_px: f32,
    new_py: f32,
    new_pz: f32,
    // Next accumulated Lagrange multiplier.
    new_lambda: f32,
    // 1 when the projection ran, 0 when a guard made the step inert.
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let position = vec3<f32>(q.px, q.py, q.pz);
    let tgt = vec3<f32>(q.tx, q.ty, q.tz);
    let w = q.w;
    let dt = q.dt;

    let delta = position - tgt;
    let dist = length(delta);

    // Ordered guards: a NaN input fails these and routes to the inert branch.
    let w_ok = w > 0.0;
    let len_ok = dist >= EPSILON;
    let is_active = w_ok && len_ok;

    // Guarded unit gradient; the guard only matters on the inert branch.
    let safe_len = max(dist, EPSILON);
    let normal = delta / safe_len;

    let alpha_tilde = q.compliance / (dt * dt);
    let denom = w + alpha_tilde;
    // When active, w > 0 and alpha_tilde >= 0 so denom > 0; guard the inert arm.
    let safe_denom = select(1.0, denom, denom > 0.0);
    let delta_lambda = (-dist - alpha_tilde * q.lambda) / safe_denom;

    let new_lambda_active = q.lambda + delta_lambda;
    let moved = position + normal * (delta_lambda * w);

    var out: Result;
    out.new_px = select(q.px, moved.x, is_active);
    out.new_py = select(q.py, moved.y, is_active);
    out.new_pz = select(q.pz, moved.z, is_active);
    out.new_lambda = select(q.lambda, new_lambda_active, is_active);
    out.valid = select(0u, 1u, is_active);
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;

    results[idx] = out;
}
"#;

/// `repr(C)` `std430` layout of the dispatch parameters.
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
/// Ten payload words plus two padding words keep the stride a flat `48` bytes,
/// a multiple of `16` with every `vec3` flattened to scalars so no
/// vector-alignment surprise can appear.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    px: f32,
    py: f32,
    pz: f32,
    w: f32,
    tx: f32,
    ty: f32,
    tz: f32,
    compliance: f32,
    lambda: f32,
    dt: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Five payload words plus three padding words keep the stride a flat
/// `32` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    new_px: f32,
    new_py: f32,
    new_pz: f32,
    new_lambda: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One attachment-projection query: a particle's position and inverse mass, the
/// fixed target anchor, the compliance, the incoming Lagrange multiplier and
/// the substep timestep, flattened to scalars so the `std430` stride stays an
/// unambiguous flat layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftAttachmentProjectQuery {
    /// Particle position, x component.
    pub px: f32,
    /// Particle position, y component.
    pub py: f32,
    /// Particle position, z component.
    pub pz: f32,
    /// Inverse mass; `<= 0` marks a pinned particle (inert step).
    pub w: f32,
    /// Target anchor, x component.
    pub tx: f32,
    /// Target anchor, y component.
    pub ty: f32,
    /// Target anchor, z component.
    pub tz: f32,
    /// Compliance (inverse stiffness); `>= 0`.
    pub compliance: f32,
    /// Accumulated Lagrange multiplier for this substep.
    pub lambda: f32,
    /// Substep timestep; strictly `> 0`.
    pub dt: f32,
}

impl SoftAttachmentProjectQuery {
    /// Builds an attachment-projection query from the particle state, the
    /// target anchor, the compliance, the incoming multiplier and the timestep.
    #[expect(
        clippy::too_many_arguments,
        reason = "a projection query is a flat scalar record of ten independent fields"
    )]
    #[must_use]
    pub fn new(
        px: f32,
        py: f32,
        pz: f32,
        w: f32,
        tx: f32,
        ty: f32,
        tz: f32,
        compliance: f32,
        lambda: f32,
        dt: f32,
    ) -> SoftAttachmentProjectQuery {
        SoftAttachmentProjectQuery {
            px,
            py,
            pz,
            w,
            tx,
            ty,
            tz,
            compliance,
            lambda,
            dt,
        }
    }
}

/// One resolved answer for a single query: the next particle position, the next
/// Lagrange multiplier and whether the projection actually ran.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftAttachmentProjectResult {
    /// Next particle position, x component.
    pub new_px: f32,
    /// Next particle position, y component.
    pub new_py: f32,
    /// Next particle position, z component.
    pub new_pz: f32,
    /// Next accumulated Lagrange multiplier.
    pub new_lambda: f32,
    /// `1` when the projection ran, `0` when a guard made the step inert.
    pub valid: u32,
}

/// Encodes one [`SoftAttachmentProjectQuery`] into its `std430` [`GpuQuery`].
fn encode_query(q: &SoftAttachmentProjectQuery) -> GpuQuery {
    GpuQuery {
        px: q.px,
        py: q.py,
        pz: q.pz,
        w: q.w,
        tx: q.tx,
        ty: q.ty,
        tz: q.tz,
        compliance: q.compliance,
        lambda: q.lambda,
        dt: q.dt,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`SoftAttachmentProjectResult`].
fn decode_result(raw: &GpuResult) -> SoftAttachmentProjectResult {
    SoftAttachmentProjectResult {
        new_px: raw.new_px,
        new_py: raw.new_py,
        new_pz: raw.new_pz,
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

/// A compiled, reusable attachment-projection compute pipeline, twinning the
/// `CPU` golden `AttachmentConstraint::project`.
pub struct GpuSoftAttachmentProject {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSoftAttachmentProject {
    /// Compiles the attachment-projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftAttachmentProject {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_attachment_project"),
            source: ShaderSource::Wgsl(SOFT_ATTACHMENT_PROJECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_soft_attachment_project_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_soft_attachment_project_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_attachment_project_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftAttachmentProject {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SoftAttachmentProjectResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SoftAttachmentProjectQuery],
    ) -> Vec<SoftAttachmentProjectResult> {
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
            label: Some("prism_volumetric_soft_attachment_project_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_attachment_project_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_attachment_project_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_soft_attachment_project_bind_group"),
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
            label: Some("prism_volumetric_soft_attachment_project_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_soft_attachment_project_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_soft_attachment_project_pass"),
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
