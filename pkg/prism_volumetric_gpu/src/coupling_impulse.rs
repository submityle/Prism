//! `wgpu` compute twin of the particle<->rigid-body two-way coupling impulse
//! contract
//! ([`two_way_coupling`](prism_render_architecture::particle::two_way_coupling),
//! particle design §10).
//!
//! The `CPU` golden
//! [`two_way_coupling`](prism_render_architecture::particle::two_way_coupling)
//! owns the textbook rigid-body contact-impulse exchange that lets a point
//! particle push back on the body it touches while the closed particle+body
//! system conserves linear momentum. This twin reproduces, one thread per
//! contact, the module's pure per-contact function cluster:
//!
//! - [`inv_inertia_world`](prism_render_architecture::particle::two_way_coupling::inv_inertia_world):
//!   the world-space inverse inertia tensor `R * diag(principal) * Rᵀ`.
//! - [`body_point_velocity`](prism_render_architecture::particle::two_way_coupling::body_point_velocity):
//!   the body's surface velocity `v_linear + omega × r` at the contact point.
//! - [`generalized_inverse_mass`](prism_render_architecture::particle::two_way_coupling::generalized_inverse_mass):
//!   `inv_mass + (r × dir) · I_inv (r × dir)`, the effective inverse mass the
//!   body shows a constraint along a unit `direction` at lever arm `r`.
//! - [`coupling_impulse`](prism_render_architecture::particle::two_way_coupling::coupling_impulse):
//!   the contact impulse vector `j * n` with `j = -(1 + e) * vn / w`, zeroed on
//!   a separating, degenerate, or jointly-immovable contact.
//! - [`linear_momentum`](prism_render_architecture::particle::two_way_coupling::linear_momentum):
//!   `mass * velocity`, for conservation accounting.
//!
//! [`GpuCouplingImpulse`] is the on-device twin: a passing real-device parity
//! test is direct evidence the ported kernel solves the same impulse algebra
//! and classifies the same degenerate cases the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! Every per-contact answer the cluster computes is reproduced in one pass: the
//! world inverse inertia tensor, the body surface velocity at the contact, the
//! generalized inverse mass along the (normalized) contact normal, the contact
//! impulse vector, and the body's linear momentum. The in-place state mutators
//! [`apply_coupling`](prism_render_architecture::particle::two_way_coupling::apply_coupling)
//! and
//! [`resolve_coupling`](prism_render_architecture::particle::two_way_coupling::resolve_coupling)
//! are deliberately **not** twinned: they are read-modify-write steps outside
//! the one-thread-per-contact pure-function contract.
//!
//! # Degenerate regimes
//!
//! The impulse mirrors the reference's three zero-impulse identities branch for
//! branch: a degenerate (near-zero-length) contact normal
//! (`length² <= EPS_LEN_SQ`), a *separating* contact (relative normal velocity
//! `>= 0`), and a jointly immovable pair (effective inverse mass at or below
//! `MIN_EFFECTIVE_INV_MASS`). The parity fixtures stay clear of these branch
//! thresholds by rejection sampling so the comparison exercises the live solve.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `dot`, `cross`,
//! `clamp`, `sqrt` (reached only through the guarded normalize, exactly as the
//! reference's [`Vec3`](prism_render_architecture::particle::Vec3) normalization
//! does) and `+ - * /` — with no transcendental call, no `u64` and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There
//! is no loop: each thread performs a fixed sequence of operations.
//!
//! # Correctness model
//!
//! Each contact is a fixed, non-reorderable sequence of multiplies, adds and
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::Vec3;
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

/// The portable core-`WGSL` coupling-impulse kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`two_way_coupling`](prism_render_architecture::particle::two_way_coupling)
/// per-contact function cluster; see the module documentation for the algebra.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
const COUPLING_IMPULSE_WGSL: &str = r#"
// coupling_impulse twin: one thread per contact reproduces the CPU golden
// `particle::two_way_coupling` per-contact pure functions — the world inverse
// inertia tensor R*diag*Rᵀ, the body surface velocity v+omega×r, the
// generalized inverse mass, the contact impulse j*n, and the body linear
// momentum. It mirrors the reference branch for branch, uses only the portable
// core-WGSL subset (dot/cross/clamp/sqrt and + - * /), takes no optional
// feature, and runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::two_way_coupling;
// 无第三方引擎源码或衍生代码。

// Squared-length floor below which a vector is treated as the zero vector so a
// normalize never yields NaN. Matches the reference `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;
// Effective-inverse-mass floor below which a particle+body pair is jointly
// immovable and the impulse divide collapses to zero. Matches the reference
// `MIN_EFFECTIVE_INV_MASS`.
const MIN_EFFECTIVE_INV_MASS: f32 = 1.0e-12;

struct Params {
    // Number of contacts in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Particle contact point (world); a pad lane follows.
    particle_position: vec3<f32>,
    pad0: f32,
    // Particle linear velocity; a pad lane follows.
    particle_velocity: vec3<f32>,
    pad1: f32,
    // Body center of mass (lever-arm origin); a pad lane follows.
    body_center_of_mass: vec3<f32>,
    pad2: f32,
    // Body linear velocity; a pad lane follows.
    body_linear_velocity: vec3<f32>,
    pad3: f32,
    // Body angular velocity; a pad lane follows.
    body_angular_velocity: vec3<f32>,
    pad4: f32,
    // Principal-frame diagonal inverse inertia; a pad lane follows.
    principal_inv_inertia: vec3<f32>,
    pad5: f32,
    // Rotation column x (a principal axis); a pad lane follows.
    rot_col_x: vec3<f32>,
    pad6: f32,
    // Rotation column y (a principal axis); a pad lane follows.
    rot_col_y: vec3<f32>,
    pad7: f32,
    // Rotation column z (a principal axis); a pad lane follows.
    rot_col_z: vec3<f32>,
    pad8: f32,
    // Contact normal (body -> particle, not necessarily unit); a pad follows.
    normal: vec3<f32>,
    pad9: f32,
    // Scalar block, one vec4 slot: particle inverse mass, body inverse mass,
    // restitution, and the body's finite mass for the momentum readout.
    particle_inv_mass: f32,
    body_inv_mass: f32,
    restitution: f32,
    body_mass: f32,
}

struct Result {
    // World inverse inertia tensor, column x; a pad lane follows.
    inv_inertia_col_x: vec3<f32>,
    pad0: f32,
    // World inverse inertia tensor, column y; a pad lane follows.
    inv_inertia_col_y: vec3<f32>,
    pad1: f32,
    // World inverse inertia tensor, column z; a pad lane follows.
    inv_inertia_col_z: vec3<f32>,
    pad2: f32,
    // Body surface velocity at the contact; a pad lane follows.
    body_point_velocity: vec3<f32>,
    pad3: f32,
    // Contact impulse vector applied to the particle; a pad lane follows.
    impulse: vec3<f32>,
    pad4: f32,
    // Body linear momentum mass*velocity; a pad lane follows.
    body_linear_momentum: vec3<f32>,
    pad5: f32,
    // Generalized inverse mass along the normalized contact normal, with three
    // pad lanes filling the trailing vec4 slot.
    generalized_inverse_mass: f32,
    pad6: f32,
    pad7: f32,
    pad8: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Column-major matrix-vector product cols * v.
fn mat_mul_vec(cx: vec3<f32>, cy: vec3<f32>, cz: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return cx * v.x + cy * v.y + cz * v.z;
}

// Unit vector along v, or the zero vector when v is (numerically) zero, exactly
// mirroring the reference `Vec3::normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // inv_inertia_world = R * diag(principal) * Rᵀ. First scale each rotation
    // column by the matching principal entry (R * diag).
    let scaled_x = q.rot_col_x * q.principal_inv_inertia.x;
    let scaled_y = q.rot_col_y * q.principal_inv_inertia.y;
    let scaled_z = q.rot_col_z * q.principal_inv_inertia.z;
    // Columns of Rᵀ are the rows of R.
    let rt_col_x = vec3<f32>(q.rot_col_x.x, q.rot_col_y.x, q.rot_col_z.x);
    let rt_col_y = vec3<f32>(q.rot_col_x.y, q.rot_col_y.y, q.rot_col_z.y);
    let rt_col_z = vec3<f32>(q.rot_col_x.z, q.rot_col_y.z, q.rot_col_z.z);
    // (R*diag) * Rᵀ, column by column.
    let inv_col_x = mat_mul_vec(scaled_x, scaled_y, scaled_z, rt_col_x);
    let inv_col_y = mat_mul_vec(scaled_x, scaled_y, scaled_z, rt_col_y);
    let inv_col_z = mat_mul_vec(scaled_x, scaled_y, scaled_z, rt_col_z);

    // body_point_velocity at the contact point = v_linear + omega × r.
    let contact = q.particle_position;
    let r = contact - q.body_center_of_mass;
    let body_vel = q.body_linear_velocity + cross(q.body_angular_velocity, r);

    // generalized_inverse_mass along the normalized contact normal.
    let n = normalize_or_zero(q.normal);
    let rn = cross(r, n);
    let i_inv_rn = mat_mul_vec(inv_col_x, inv_col_y, inv_col_z, rn);
    let gen_inv_mass = q.body_inv_mass + dot(rn, i_inv_rn);

    // coupling_impulse, mirroring the reference's three zero-impulse identities.
    var impulse = vec3<f32>(0.0, 0.0, 0.0);
    if (dot(q.normal, q.normal) > EPS_LEN_SQ) {
        let relative = q.particle_velocity - body_vel;
        let vn = dot(relative, n);
        if (vn < 0.0) {
            let effective = q.particle_inv_mass + gen_inv_mass;
            if (effective > MIN_EFFECTIVE_INV_MASS) {
                let e = clamp(q.restitution, 0.0, 1.0);
                let magnitude = -(1.0 + e) * vn / effective;
                impulse = n * magnitude;
            }
        }
    }

    // linear_momentum = mass * velocity.
    let momentum = q.body_linear_velocity * q.body_mass;

    var out: Result;
    out.inv_inertia_col_x = inv_col_x;
    out.pad0 = 0.0;
    out.inv_inertia_col_y = inv_col_y;
    out.pad1 = 0.0;
    out.inv_inertia_col_z = inv_col_z;
    out.pad2 = 0.0;
    out.body_point_velocity = body_vel;
    out.pad3 = 0.0;
    out.impulse = impulse;
    out.pad4 = 0.0;
    out.body_linear_momentum = momentum;
    out.pad5 = 0.0;
    out.generalized_inverse_mass = gen_inv_mass;
    out.pad6 = 0.0;
    out.pad7 = 0.0;
    out.pad8 = 0.0;
    results[idx] = out;
}
"#;

/// A column-major 3x3 matrix built from three [`Vec3`] columns, used for the
/// body's rotation (its principal axes as columns) on input and the world-space
/// inverse inertia tensor on output.
///
/// Defined locally rather than reusing the golden `Mat3` so this twin exports
/// only its own `Gpu`-prefixed surface and never re-exports a reference type.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMat3 {
    /// First (x) column.
    pub col_x: Vec3,
    /// Second (y) column.
    pub col_y: Vec3,
    /// Third (z) column.
    pub col_z: Vec3,
}

impl GpuMat3 {
    /// The identity matrix (unit response about every axis).
    pub const IDENTITY: GpuMat3 = GpuMat3 {
        col_x: Vec3::new(1.0, 0.0, 0.0),
        col_y: Vec3::new(0.0, 1.0, 0.0),
        col_z: Vec3::new(0.0, 0.0, 1.0),
    };

    /// Builds a matrix from three explicit columns.
    #[must_use]
    pub const fn from_columns(col_x: Vec3, col_y: Vec3, col_z: Vec3) -> GpuMat3 {
        GpuMat3 {
            col_x,
            col_y,
            col_z,
        }
    }
}

/// One particle<->body coupling contact: the particle state, the body state,
/// the body's principal inverse inertia and orientation, the (not necessarily
/// unit) contact `normal` and the restitution.
///
/// `rotation` carries the body's principal axes as its columns and
/// `principal_inv_inertia` the matching diagonal inverse inertia, together
/// feeding the world-tensor build; `body_mass` is the body's finite mass for
/// the momentum readout.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuCouplingImpulseQuery {
    /// Particle inverse mass (`1 / mass`); `0` is an immovable particle.
    pub particle_inv_mass: f32,
    /// Particle world-space position, also the contact point.
    pub particle_position: Vec3,
    /// Particle world-space linear velocity.
    pub particle_velocity: Vec3,
    /// Body inverse mass (`1 / mass`); `0` is a static body.
    pub body_inv_mass: f32,
    /// Body finite mass used by the `linear_momentum` readout.
    pub body_mass: f32,
    /// Body world-space center of mass (the lever-arm origin).
    pub body_center_of_mass: Vec3,
    /// Body world-space linear velocity.
    pub body_linear_velocity: Vec3,
    /// Body world-space angular velocity.
    pub body_angular_velocity: Vec3,
    /// Principal-frame diagonal inverse inertia.
    pub principal_inv_inertia: Vec3,
    /// Body orientation: its principal axes as the columns of a rotation.
    pub rotation: GpuMat3,
    /// Contact normal (body toward particle); need not be unit length.
    pub normal: Vec3,
    /// Restitution coefficient, clamped to `0..=1` inside the solve.
    pub restitution: f32,
}

/// The resolved per-contact answer, mirroring every value the reference's
/// per-contact function cluster reports.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuCouplingImpulseResult {
    /// World-space inverse inertia tensor `R * diag(principal) * Rᵀ`, matching
    /// `inv_inertia_world`.
    pub inv_inertia_world: GpuMat3,
    /// Body surface velocity at the contact, matching `body_point_velocity`.
    pub body_point_velocity: Vec3,
    /// Generalized inverse mass along the normalized contact normal, matching
    /// `generalized_inverse_mass` with the body's inverse mass.
    pub generalized_inverse_mass: f32,
    /// Contact impulse applied to the particle, matching `coupling_impulse`.
    pub impulse: Vec3,
    /// Body linear momentum `mass * velocity`, matching `linear_momentum`.
    pub body_linear_momentum: Vec3,
}

/// `repr(C)` `std430` image of one packed contact: ten `vec4` slots carrying the
/// vector inputs each on its `16`-byte-aligned lane, then a trailing `vec4` of
/// the four scalar inputs — `176` bytes matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Particle contact point.
    particle_position: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// Particle linear velocity.
    particle_velocity: [f32; 3],
    /// Padding lane.
    pad1: f32,
    /// Body center of mass.
    body_center_of_mass: [f32; 3],
    /// Padding lane.
    pad2: f32,
    /// Body linear velocity.
    body_linear_velocity: [f32; 3],
    /// Padding lane.
    pad3: f32,
    /// Body angular velocity.
    body_angular_velocity: [f32; 3],
    /// Padding lane.
    pad4: f32,
    /// Principal-frame diagonal inverse inertia.
    principal_inv_inertia: [f32; 3],
    /// Padding lane.
    pad5: f32,
    /// Rotation column x.
    rot_col_x: [f32; 3],
    /// Padding lane.
    pad6: f32,
    /// Rotation column y.
    rot_col_y: [f32; 3],
    /// Padding lane.
    pad7: f32,
    /// Rotation column z.
    rot_col_z: [f32; 3],
    /// Padding lane.
    pad8: f32,
    /// Contact normal.
    normal: [f32; 3],
    /// Padding lane.
    pad9: f32,
    /// Particle inverse mass.
    particle_inv_mass: f32,
    /// Body inverse mass.
    body_inv_mass: f32,
    /// Restitution coefficient.
    restitution: f32,
    /// Body finite mass.
    body_mass: f32,
}

impl GpuQuery {
    /// Packs one public query into its `std430` image.
    fn new(query: &GpuCouplingImpulseQuery) -> GpuQuery {
        GpuQuery {
            particle_position: [
                query.particle_position.x,
                query.particle_position.y,
                query.particle_position.z,
            ],
            pad0: 0.0,
            particle_velocity: [
                query.particle_velocity.x,
                query.particle_velocity.y,
                query.particle_velocity.z,
            ],
            pad1: 0.0,
            body_center_of_mass: [
                query.body_center_of_mass.x,
                query.body_center_of_mass.y,
                query.body_center_of_mass.z,
            ],
            pad2: 0.0,
            body_linear_velocity: [
                query.body_linear_velocity.x,
                query.body_linear_velocity.y,
                query.body_linear_velocity.z,
            ],
            pad3: 0.0,
            body_angular_velocity: [
                query.body_angular_velocity.x,
                query.body_angular_velocity.y,
                query.body_angular_velocity.z,
            ],
            pad4: 0.0,
            principal_inv_inertia: [
                query.principal_inv_inertia.x,
                query.principal_inv_inertia.y,
                query.principal_inv_inertia.z,
            ],
            pad5: 0.0,
            rot_col_x: [
                query.rotation.col_x.x,
                query.rotation.col_x.y,
                query.rotation.col_x.z,
            ],
            pad6: 0.0,
            rot_col_y: [
                query.rotation.col_y.x,
                query.rotation.col_y.y,
                query.rotation.col_y.z,
            ],
            pad7: 0.0,
            rot_col_z: [
                query.rotation.col_z.x,
                query.rotation.col_z.y,
                query.rotation.col_z.z,
            ],
            pad8: 0.0,
            normal: [query.normal.x, query.normal.y, query.normal.z],
            pad9: 0.0,
            particle_inv_mass: query.particle_inv_mass,
            body_inv_mass: query.body_inv_mass,
            restitution: query.restitution,
            body_mass: query.body_mass,
        }
    }
}

/// `repr(C)` `std430` image of one result: three `vec4` slots for the world
/// inverse inertia columns, three more for the body point velocity, impulse and
/// momentum, and a trailing `vec4` whose first lane is the generalized inverse
/// mass — `112` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// World inverse inertia column x.
    inv_inertia_col_x: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// World inverse inertia column y.
    inv_inertia_col_y: [f32; 3],
    /// Padding lane.
    pad1: f32,
    /// World inverse inertia column z.
    inv_inertia_col_z: [f32; 3],
    /// Padding lane.
    pad2: f32,
    /// Body surface velocity at the contact.
    body_point_velocity: [f32; 3],
    /// Padding lane.
    pad3: f32,
    /// Contact impulse applied to the particle.
    impulse: [f32; 3],
    /// Padding lane.
    pad4: f32,
    /// Body linear momentum.
    body_linear_momentum: [f32; 3],
    /// Padding lane.
    pad5: f32,
    /// Generalized inverse mass along the normalized contact normal.
    generalized_inverse_mass: f32,
    /// Padding lane.
    pad6: f32,
    /// Padding lane.
    pad7: f32,
    /// Padding lane.
    pad8: f32,
}

/// Uniform parameters for one dispatch: the contact count plus three pad words
/// to fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of contacts in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
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

/// A compiled, reusable coupling-impulse compute pipeline, twinning the `CPU`
/// golden
/// [`two_way_coupling`](prism_render_architecture::particle::two_way_coupling).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
pub struct GpuCouplingImpulse {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCouplingImpulse {
    /// Compiles the coupling-impulse kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset plus the guarded
    /// `sqrt` the reference normalize already uses, so no optional device
    /// feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCouplingImpulse {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_coupling_impulse"),
            source: ShaderSource::Wgsl(COUPLING_IMPULSE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_coupling_impulse_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_coupling_impulse_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_coupling_impulse_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCouplingImpulse {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every coupling contact on-device and returns one
    /// [`GpuCouplingImpulseResult`] per input, in order.
    ///
    /// Each result equals the reference's per-contact cluster
    /// (`inv_inertia_world`, `body_point_velocity`, `generalized_inverse_mass`,
    /// `coupling_impulse`, `linear_momentum`) to within the tolerance documented
    /// on this module. An empty input returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[GpuCouplingImpulseQuery],
    ) -> Vec<GpuCouplingImpulseResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_coupling_impulse_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_coupling_impulse_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_coupling_impulse_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_coupling_impulse_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_coupling_impulse_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_coupling_impulse_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_coupling_impulse_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per contact, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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

/// Decodes one packed [`GpuResult`] into the public [`GpuCouplingImpulseResult`].
fn decode_result(raw: &GpuResult) -> GpuCouplingImpulseResult {
    GpuCouplingImpulseResult {
        inv_inertia_world: GpuMat3::from_columns(
            Vec3::new(
                raw.inv_inertia_col_x[0],
                raw.inv_inertia_col_x[1],
                raw.inv_inertia_col_x[2],
            ),
            Vec3::new(
                raw.inv_inertia_col_y[0],
                raw.inv_inertia_col_y[1],
                raw.inv_inertia_col_y[2],
            ),
            Vec3::new(
                raw.inv_inertia_col_z[0],
                raw.inv_inertia_col_z[1],
                raw.inv_inertia_col_z[2],
            ),
        ),
        body_point_velocity: Vec3::new(
            raw.body_point_velocity[0],
            raw.body_point_velocity[1],
            raw.body_point_velocity[2],
        ),
        generalized_inverse_mass: raw.generalized_inverse_mass,
        impulse: Vec3::new(raw.impulse[0], raw.impulse[1], raw.impulse[2]),
        body_linear_momentum: Vec3::new(
            raw.body_linear_momentum[0],
            raw.body_linear_momentum[1],
            raw.body_linear_momentum[2],
        ),
    }
}
