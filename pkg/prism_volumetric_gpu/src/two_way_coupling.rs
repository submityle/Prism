//! `wgpu` compute twin of the particle<->rigid-body two-way coupling linear
//! algebra
//! ([`two_way_coupling`](prism_render_architecture::particle::two_way_coupling),
//! particle design §10).
//!
//! The `CPU` golden
//! [`two_way_coupling`](prism_render_architecture::particle::two_way_coupling)
//! owns the textbook rigid-body contact-impulse exchange that lets a point
//! particle push back on the body it touches while the closed particle+body
//! system conserves linear momentum. This twin reproduces, one thread per
//! query, the module's pure linear-algebra surface as a tagged batch so one
//! dispatch can mix matrix, inertia, velocity and impulse queries:
//!
//! - the [`Mat3`](prism_render_architecture::particle::two_way_coupling::Mat3)
//!   primitives `mul_vec3`, `transpose` and `mul_mat3`;
//! - [`inv_inertia_world`](prism_render_architecture::particle::two_way_coupling::inv_inertia_world),
//!   the world inverse inertia tensor `R * diag(principal) * Rᵀ`;
//! - [`generalized_inverse_mass`](prism_render_architecture::particle::two_way_coupling::generalized_inverse_mass),
//!   `inv_mass + (r × dir) · I_inv (r × dir)`;
//! - [`body_point_velocity`](prism_render_architecture::particle::two_way_coupling::body_point_velocity),
//!   the body surface velocity `v_linear + omega × r`;
//! - [`coupling_impulse`](prism_render_architecture::particle::two_way_coupling::coupling_impulse),
//!   the contact impulse vector `j * n` with `j = -(1 + e) * vn / w`, zeroed on
//!   a separating, degenerate or jointly-immovable contact;
//! - [`resolve_coupling`](prism_render_architecture::particle::two_way_coupling::resolve_coupling),
//!   returned here as the applied impulse plus the *updated* particle linear,
//!   body linear and body angular velocities;
//! - [`linear_momentum`](prism_render_architecture::particle::two_way_coupling::linear_momentum),
//!   `mass * velocity`, for conservation accounting.
//!
//! [`GpuTwoWayCoupling`] is the on-device twin: a passing real-device parity
//! test is direct evidence the ported kernel solves the same algebra and
//! classifies the same degenerate cases the reference does, not merely that the
//! shader compiles.
//!
//! # What stays on the host
//!
//! The reference
//! [`apply_coupling`](prism_render_architecture::particle::two_way_coupling::apply_coupling)
//! mutates `&mut` particle and body state in place; that read-modify-write step
//! is deliberately **not** twinned. Instead the `ResolveCoupling` query is a
//! pure computation returning the already-updated velocities, which the host
//! writes back. This keeps the kernel inside the one-thread-per-element,
//! no-aliasing contract the rest of the crate uses.
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
//! reference [`Vec3`](prism_render_architecture::particle::Vec3) normalization
//! does) and `+ - * /` — with no transcendental call, no `u64` and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Each
//! thread performs a fixed, bounded sequence with no loop, so it provably
//! terminates.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
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

/// Operation tag: [`Mat3::mul_vec3`] matrix-vector product.
const OP_MAT_MUL_VEC3: u32 = 0;
/// Operation tag: [`Mat3::transpose`].
const OP_MAT_TRANSPOSE: u32 = 1;
/// Operation tag: [`Mat3::mul_mat3`] matrix-matrix product.
const OP_MAT_MUL_MAT3: u32 = 2;
/// Operation tag: `inv_inertia_world` tensor build.
const OP_INV_INERTIA_WORLD: u32 = 3;
/// Operation tag: `generalized_inverse_mass`.
const OP_GENERALIZED_INVERSE_MASS: u32 = 4;
/// Operation tag: `body_point_velocity`.
const OP_BODY_POINT_VELOCITY: u32 = 5;
/// Operation tag: `coupling_impulse`.
const OP_COUPLING_IMPULSE: u32 = 6;
/// Operation tag: `resolve_coupling`.
const OP_RESOLVE_COUPLING: u32 = 7;
/// Operation tag: `linear_momentum`.
const OP_LINEAR_MOMENTUM: u32 = 8;

/// The portable core-`WGSL` two-way-coupling kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` dispatches on
/// a per-query operation tag and mirrors the `CPU` golden
/// [`two_way_coupling`](prism_render_architecture::particle::two_way_coupling)
/// linear algebra; see the module documentation for the formulae.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
const TWO_WAY_COUPLING_WGSL: &str = r#"
// two_way_coupling twin: one thread per query dispatches on an operation tag and
// reproduces the CPU golden `particle::two_way_coupling` linear algebra — the
// Mat3 mul_vec3 / transpose / mul_mat3 primitives, the world inverse inertia
// tensor R*diag*Rᵀ, the generalized inverse mass, the body surface velocity
// v+omega×r, the contact impulse j*n, the resolved (updated) velocities, and the
// linear momentum. It mirrors the reference branch for branch, uses only the
// portable core-WGSL subset (dot/cross/clamp/sqrt and + - * /), takes no
// optional feature, and runs unmodified on Metal, Vulkan and DX12. There is no
// loop, so the kernel provably terminates.
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

// Operation tags, mirroring the host-side OP_* constants.
const OP_MAT_MUL_VEC3: u32 = 0u;
const OP_MAT_TRANSPOSE: u32 = 1u;
const OP_MAT_MUL_MAT3: u32 = 2u;
const OP_INV_INERTIA_WORLD: u32 = 3u;
const OP_GENERALIZED_INVERSE_MASS: u32 = 4u;
const OP_BODY_POINT_VELOCITY: u32 = 5u;
const OP_COUPLING_IMPULSE: u32 = 6u;
const OP_RESOLVE_COUPLING: u32 = 7u;
const OP_LINEAR_MOMENTUM: u32 = 8u;

// Result-kind tags for the decoded union.
const KIND_VECTOR: u32 = 0u;
const KIND_MATRIX: u32 = 1u;
const KIND_SCALAR: u32 = 2u;
const KIND_VELOCITIES: u32 = 3u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation tag selecting which golden function this slot evaluates.
    op: u32,
    pad_op0: u32,
    pad_op1: u32,
    pad_op2: u32,
    // Scalar block, one vec4 slot: particle (or standalone) inverse mass, body
    // inverse mass, restitution, and the finite mass for the momentum readout.
    particle_inv_mass: f32,
    body_inv_mass: f32,
    restitution: f32,
    mass: f32,
    // Matrix A columns: the operand matrix (mul_vec3 / transpose / mul_mat3
    // lhs), the rotation (inv_inertia_world), or the world inverse inertia
    // tensor (generalized_inverse_mass / coupling_impulse / resolve_coupling).
    a_col_x: vec3<f32>,
    pad_ax: f32,
    a_col_y: vec3<f32>,
    pad_ay: f32,
    a_col_z: vec3<f32>,
    pad_az: f32,
    // Matrix B columns: the mul_mat3 right-hand side.
    b_col_x: vec3<f32>,
    pad_bx: f32,
    b_col_y: vec3<f32>,
    pad_by: f32,
    b_col_z: vec3<f32>,
    pad_bz: f32,
    // Primary vector slot: mul_vec3 v / inv_inertia principal / gim lever arm /
    // body_point_velocity point / particle position / linear_momentum velocity.
    position: vec3<f32>,
    pad_pos: f32,
    // Secondary vector slot: gim direction / particle velocity.
    velocity: vec3<f32>,
    pad_vel: f32,
    // Body center of mass (lever-arm origin).
    center_of_mass: vec3<f32>,
    pad_com: f32,
    // Body linear velocity.
    body_linear: vec3<f32>,
    pad_bl: f32,
    // Body angular velocity.
    body_angular: vec3<f32>,
    pad_ba: f32,
    // Contact normal (body -> particle), not necessarily unit length.
    normal: vec3<f32>,
    pad_n: f32,
}

struct Result {
    // Result-kind tag; the host decodes r0..r3 accordingly.
    kind: u32,
    pad_k0: u32,
    pad_k1: u32,
    pad_k2: u32,
    r0: vec3<f32>,
    pad_r0: f32,
    r1: vec3<f32>,
    pad_r1: f32,
    r2: vec3<f32>,
    pad_r2: f32,
    r3: vec3<f32>,
    pad_r3: f32,
}

struct M3 {
    c0: vec3<f32>,
    c1: vec3<f32>,
    c2: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Column-major matrix-vector product m * v, mirroring the reference
// `Mat3::mul_vec3`.
fn m3_mul_vec3(m: M3, v: vec3<f32>) -> vec3<f32> {
    return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z;
}

// Transpose (rows become columns), mirroring the reference `Mat3::transpose`.
fn m3_transpose(m: M3) -> M3 {
    return M3(
        vec3<f32>(m.c0.x, m.c1.x, m.c2.x),
        vec3<f32>(m.c0.y, m.c1.y, m.c2.y),
        vec3<f32>(m.c0.z, m.c1.z, m.c2.z),
    );
}

// Matrix-matrix product a * b, column by column, mirroring `Mat3::mul_mat3`.
fn m3_mul_mat3(a: M3, b: M3) -> M3 {
    return M3(m3_mul_vec3(a, b.c0), m3_mul_vec3(a, b.c1), m3_mul_vec3(a, b.c2));
}

// R * diag(principal) * Rᵀ, mirroring the reference `inv_inertia_world`.
fn inv_inertia_world(principal: vec3<f32>, rot: M3) -> M3 {
    let scaled = M3(rot.c0 * principal.x, rot.c1 * principal.y, rot.c2 * principal.z);
    return m3_mul_mat3(scaled, m3_transpose(rot));
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

// Contact impulse j*n, mirroring the reference `coupling_impulse` branch for
// branch. `inv_inertia` is the body's world inverse inertia tensor.
fn coupling_impulse(q: Query, inv_inertia: M3) -> vec3<f32> {
    if (dot(q.normal, q.normal) <= EPS_LEN_SQ) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let n = normalize_or_zero(q.normal);
    let contact = q.position;
    let r = contact - q.center_of_mass;
    let body_vel = q.body_linear + cross(q.body_angular, r);
    let relative = q.velocity - body_vel;
    let vn = dot(relative, n);
    if (vn >= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let rn = cross(r, n);
    let gim = q.body_inv_mass + dot(rn, m3_mul_vec3(inv_inertia, rn));
    let effective = q.particle_inv_mass + gim;
    if (effective <= MIN_EFFECTIVE_INV_MASS) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let e = clamp(q.restitution, 0.0, 1.0);
    let magnitude = -(1.0 + e) * vn / effective;
    return n * magnitude;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let a = M3(q.a_col_x, q.a_col_y, q.a_col_z);
    let b = M3(q.b_col_x, q.b_col_y, q.b_col_z);

    var out: Result;
    out.kind = KIND_VECTOR;
    out.pad_k0 = 0u;
    out.pad_k1 = 0u;
    out.pad_k2 = 0u;
    out.r0 = vec3<f32>(0.0, 0.0, 0.0);
    out.pad_r0 = 0.0;
    out.r1 = vec3<f32>(0.0, 0.0, 0.0);
    out.pad_r1 = 0.0;
    out.r2 = vec3<f32>(0.0, 0.0, 0.0);
    out.pad_r2 = 0.0;
    out.r3 = vec3<f32>(0.0, 0.0, 0.0);
    out.pad_r3 = 0.0;

    switch q.op {
        case OP_MAT_MUL_VEC3: {
            out.kind = KIND_VECTOR;
            out.r0 = m3_mul_vec3(a, q.position);
        }
        case OP_MAT_TRANSPOSE: {
            out.kind = KIND_MATRIX;
            let t = m3_transpose(a);
            out.r0 = t.c0;
            out.r1 = t.c1;
            out.r2 = t.c2;
        }
        case OP_MAT_MUL_MAT3: {
            out.kind = KIND_MATRIX;
            let m = m3_mul_mat3(a, b);
            out.r0 = m.c0;
            out.r1 = m.c1;
            out.r2 = m.c2;
        }
        case OP_INV_INERTIA_WORLD: {
            out.kind = KIND_MATRIX;
            let m = inv_inertia_world(q.position, a);
            out.r0 = m.c0;
            out.r1 = m.c1;
            out.r2 = m.c2;
        }
        case OP_GENERALIZED_INVERSE_MASS: {
            out.kind = KIND_SCALAR;
            let rn = cross(q.position, q.velocity);
            let w = q.particle_inv_mass + dot(rn, m3_mul_vec3(a, rn));
            out.r0 = vec3<f32>(w, 0.0, 0.0);
        }
        case OP_BODY_POINT_VELOCITY: {
            out.kind = KIND_VECTOR;
            let r = q.position - q.center_of_mass;
            out.r0 = q.body_linear + cross(q.body_angular, r);
        }
        case OP_COUPLING_IMPULSE: {
            out.kind = KIND_VECTOR;
            out.r0 = coupling_impulse(q, a);
        }
        case OP_RESOLVE_COUPLING: {
            out.kind = KIND_VELOCITIES;
            let impulse = coupling_impulse(q, a);
            let r = q.position - q.center_of_mass;
            let reaction = -impulse;
            out.r0 = impulse;
            out.r1 = q.velocity + impulse * q.particle_inv_mass;
            out.r2 = q.body_linear + reaction * q.body_inv_mass;
            out.r3 = q.body_angular + m3_mul_vec3(a, cross(r, reaction));
        }
        case OP_LINEAR_MOMENTUM: {
            out.kind = KIND_VECTOR;
            out.r0 = q.position * q.mass;
        }
        default: {
        }
    }
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in [`TWO_WAY_COUPLING_WGSL`].
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
/// Every `vec3` lane carries a trailing pad word so each stays `16`-byte aligned
/// on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Operation tag selecting which golden function this slot evaluates.
    op: u32,
    /// Padding word.
    pad_op0: u32,
    /// Padding word.
    pad_op1: u32,
    /// Padding word.
    pad_op2: u32,
    /// Particle (or standalone) inverse mass.
    particle_inv_mass: f32,
    /// Body inverse mass.
    body_inv_mass: f32,
    /// Restitution coefficient (clamped to `0..=1` inside the solve).
    restitution: f32,
    /// Finite mass for the `linear_momentum` readout.
    mass: f32,
    /// Matrix A column x.
    a_col_x: [f32; 3],
    /// Pad lane after `a_col_x`.
    pad_ax: f32,
    /// Matrix A column y.
    a_col_y: [f32; 3],
    /// Pad lane after `a_col_y`.
    pad_ay: f32,
    /// Matrix A column z.
    a_col_z: [f32; 3],
    /// Pad lane after `a_col_z`.
    pad_az: f32,
    /// Matrix B column x.
    b_col_x: [f32; 3],
    /// Pad lane after `b_col_x`.
    pad_bx: f32,
    /// Matrix B column y.
    b_col_y: [f32; 3],
    /// Pad lane after `b_col_y`.
    pad_by: f32,
    /// Matrix B column z.
    b_col_z: [f32; 3],
    /// Pad lane after `b_col_z`.
    pad_bz: f32,
    /// Primary vector slot.
    position: [f32; 3],
    /// Pad lane after `position`.
    pad_pos: f32,
    /// Secondary vector slot.
    velocity: [f32; 3],
    /// Pad lane after `velocity`.
    pad_vel: f32,
    /// Body center of mass.
    center_of_mass: [f32; 3],
    /// Pad lane after `center_of_mass`.
    pad_com: f32,
    /// Body linear velocity.
    body_linear: [f32; 3],
    /// Pad lane after `body_linear`.
    pad_bl: f32,
    /// Body angular velocity.
    body_angular: [f32; 3],
    /// Pad lane after `body_angular`.
    pad_ba: f32,
    /// Contact normal (need not be unit length).
    normal: [f32; 3],
    /// Pad lane after `normal`.
    pad_n: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. The four `vec3` lanes carry whichever quantity the result kind
/// selects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Result-kind tag.
    kind: u32,
    /// Padding word.
    pad_k0: u32,
    /// Padding word.
    pad_k1: u32,
    /// Padding word.
    pad_k2: u32,
    /// Result lane 0.
    r0: [f32; 3],
    /// Pad lane after `r0`.
    pad_r0: f32,
    /// Result lane 1.
    r1: [f32; 3],
    /// Pad lane after `r1`.
    pad_r1: f32,
    /// Result lane 2.
    r2: [f32; 3],
    /// Pad lane after `r2`.
    pad_r2: f32,
    /// Result lane 3.
    r3: [f32; 3],
    /// Pad lane after `r3`.
    pad_r3: f32,
}

/// One linear-algebra query against the two-way-coupling twin.
///
/// Matrices are column-major, given as three [`Vec3`] columns `[col_x, col_y,
/// col_z]`. Each variant mirrors one golden function; `ResolveCoupling` returns
/// the *updated* velocities rather than mutating in place, so the host writes
/// them back itself.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TwoWayCouplingQuery {
    /// Matrix-vector product `matrix * vector`
    /// ([`Mat3::mul_vec3`](prism_render_architecture::particle::two_way_coupling::Mat3::mul_vec3)).
    MatMulVec3 {
        /// Column-major matrix columns.
        matrix: [Vec3; 3],
        /// Right-hand vector.
        vector: Vec3,
    },
    /// Matrix transpose
    /// ([`Mat3::transpose`](prism_render_architecture::particle::two_way_coupling::Mat3::transpose)).
    MatTranspose {
        /// Column-major matrix columns.
        matrix: [Vec3; 3],
    },
    /// Matrix-matrix product `lhs * rhs`
    /// ([`Mat3::mul_mat3`](prism_render_architecture::particle::two_way_coupling::Mat3::mul_mat3)).
    MatMulMat3 {
        /// Left operand columns.
        lhs: [Vec3; 3],
        /// Right operand columns.
        rhs: [Vec3; 3],
    },
    /// World inverse inertia tensor `R * diag(principal) * Rᵀ`
    /// ([`inv_inertia_world`](prism_render_architecture::particle::two_way_coupling::inv_inertia_world)).
    InvInertiaWorld {
        /// Principal-frame diagonal inverse inertia.
        principal_inv_inertia: Vec3,
        /// Rotation whose columns are the principal axes.
        rotation: [Vec3; 3],
    },
    /// Generalized inverse mass along `direction` at lever arm `lever_arm`
    /// ([`generalized_inverse_mass`](prism_render_architecture::particle::two_way_coupling::generalized_inverse_mass)).
    GeneralizedInverseMass {
        /// Body inverse mass.
        inv_mass: f32,
        /// World inverse inertia tensor columns.
        inv_inertia_world: [Vec3; 3],
        /// Lever arm `r` from the center of mass to the contact.
        lever_arm: Vec3,
        /// Direction the constraint acts along (need not be unit length).
        direction: Vec3,
    },
    /// Body surface velocity at a world `point`
    /// ([`body_point_velocity`](prism_render_architecture::particle::two_way_coupling::body_point_velocity)).
    BodyPointVelocity {
        /// Body center of mass.
        center_of_mass: Vec3,
        /// Body linear velocity.
        linear_velocity: Vec3,
        /// Body angular velocity.
        angular_velocity: Vec3,
        /// World point to sample.
        point: Vec3,
    },
    /// Contact impulse applied to the particle
    /// ([`coupling_impulse`](prism_render_architecture::particle::two_way_coupling::coupling_impulse)).
    CouplingImpulse {
        /// Particle inverse mass.
        particle_inv_mass: f32,
        /// Particle world position (also the contact point).
        particle_position: Vec3,
        /// Particle linear velocity.
        particle_velocity: Vec3,
        /// Body inverse mass.
        body_inv_mass: f32,
        /// Body world inverse inertia tensor columns.
        body_inv_inertia_world: [Vec3; 3],
        /// Body center of mass.
        body_center_of_mass: Vec3,
        /// Body linear velocity.
        body_linear_velocity: Vec3,
        /// Body angular velocity.
        body_angular_velocity: Vec3,
        /// Contact normal (need not be unit length).
        normal: Vec3,
        /// Restitution coefficient.
        restitution: f32,
    },
    /// Resolve one coupling step, returning the applied impulse plus the updated
    /// velocities
    /// ([`resolve_coupling`](prism_render_architecture::particle::two_way_coupling::resolve_coupling)).
    ResolveCoupling {
        /// Particle inverse mass.
        particle_inv_mass: f32,
        /// Particle world position (also the contact point).
        particle_position: Vec3,
        /// Particle linear velocity.
        particle_velocity: Vec3,
        /// Body inverse mass.
        body_inv_mass: f32,
        /// Body world inverse inertia tensor columns.
        body_inv_inertia_world: [Vec3; 3],
        /// Body center of mass.
        body_center_of_mass: Vec3,
        /// Body linear velocity.
        body_linear_velocity: Vec3,
        /// Body angular velocity.
        body_angular_velocity: Vec3,
        /// Contact normal (need not be unit length).
        normal: Vec3,
        /// Restitution coefficient.
        restitution: f32,
    },
    /// Linear momentum `mass * velocity`
    /// ([`linear_momentum`](prism_render_architecture::particle::two_way_coupling::linear_momentum)).
    LinearMomentum {
        /// Finite mass.
        mass: f32,
        /// Linear velocity.
        velocity: Vec3,
    },
}

/// One resolved answer, mirroring whichever golden function the query selected.
///
/// Matrices are column-major as three [`Vec3`] columns `[col_x, col_y, col_z]`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TwoWayCouplingResult {
    /// A single vector (matrix-vector product, body velocity, impulse,
    /// momentum).
    Vector(Vec3),
    /// A column-major matrix (transpose, matrix product, inverse inertia).
    Matrix([Vec3; 3]),
    /// A single scalar (generalized inverse mass).
    Scalar(f32),
    /// The resolved impulse plus the updated velocities.
    Velocities {
        /// Impulse applied to the particle.
        impulse: Vec3,
        /// Updated particle linear velocity.
        particle_velocity: Vec3,
        /// Updated body linear velocity.
        body_linear_velocity: Vec3,
        /// Updated body angular velocity.
        body_angular_velocity: Vec3,
    },
}

/// Converts a [`Vec3`] into its padded `std430` lane.
fn lane(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// Reads a padded `std430` lane back into a [`Vec3`].
fn unlane(l: [f32; 3]) -> Vec3 {
    Vec3::new(l[0], l[1], l[2])
}

/// A fully zeroed query slot, filled per variant by [`encode_query`].
fn empty_query() -> GpuQuery {
    GpuQuery {
        op: 0,
        pad_op0: 0,
        pad_op1: 0,
        pad_op2: 0,
        particle_inv_mass: 0.0,
        body_inv_mass: 0.0,
        restitution: 0.0,
        mass: 0.0,
        a_col_x: [0.0; 3],
        pad_ax: 0.0,
        a_col_y: [0.0; 3],
        pad_ay: 0.0,
        a_col_z: [0.0; 3],
        pad_az: 0.0,
        b_col_x: [0.0; 3],
        pad_bx: 0.0,
        b_col_y: [0.0; 3],
        pad_by: 0.0,
        b_col_z: [0.0; 3],
        pad_bz: 0.0,
        position: [0.0; 3],
        pad_pos: 0.0,
        velocity: [0.0; 3],
        pad_vel: 0.0,
        center_of_mass: [0.0; 3],
        pad_com: 0.0,
        body_linear: [0.0; 3],
        pad_bl: 0.0,
        body_angular: [0.0; 3],
        pad_ba: 0.0,
        normal: [0.0; 3],
        pad_n: 0.0,
    }
}

/// Writes a column-major matrix into the matrix-A slot of `q`.
fn set_matrix_a(q: &mut GpuQuery, m: [Vec3; 3]) {
    q.a_col_x = lane(m[0]);
    q.a_col_y = lane(m[1]);
    q.a_col_z = lane(m[2]);
}

/// Encodes one [`TwoWayCouplingQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(query: &TwoWayCouplingQuery) -> GpuQuery {
    let mut q = empty_query();
    match query {
        TwoWayCouplingQuery::MatMulVec3 { matrix, vector } => {
            q.op = OP_MAT_MUL_VEC3;
            set_matrix_a(&mut q, *matrix);
            q.position = lane(*vector);
        }
        TwoWayCouplingQuery::MatTranspose { matrix } => {
            q.op = OP_MAT_TRANSPOSE;
            set_matrix_a(&mut q, *matrix);
        }
        TwoWayCouplingQuery::MatMulMat3 { lhs, rhs } => {
            q.op = OP_MAT_MUL_MAT3;
            set_matrix_a(&mut q, *lhs);
            q.b_col_x = lane(rhs[0]);
            q.b_col_y = lane(rhs[1]);
            q.b_col_z = lane(rhs[2]);
        }
        TwoWayCouplingQuery::InvInertiaWorld {
            principal_inv_inertia,
            rotation,
        } => {
            q.op = OP_INV_INERTIA_WORLD;
            set_matrix_a(&mut q, *rotation);
            q.position = lane(*principal_inv_inertia);
        }
        TwoWayCouplingQuery::GeneralizedInverseMass {
            inv_mass,
            inv_inertia_world,
            lever_arm,
            direction,
        } => {
            q.op = OP_GENERALIZED_INVERSE_MASS;
            q.particle_inv_mass = *inv_mass;
            set_matrix_a(&mut q, *inv_inertia_world);
            q.position = lane(*lever_arm);
            q.velocity = lane(*direction);
        }
        TwoWayCouplingQuery::BodyPointVelocity {
            center_of_mass,
            linear_velocity,
            angular_velocity,
            point,
        } => {
            q.op = OP_BODY_POINT_VELOCITY;
            q.center_of_mass = lane(*center_of_mass);
            q.body_linear = lane(*linear_velocity);
            q.body_angular = lane(*angular_velocity);
            q.position = lane(*point);
        }
        TwoWayCouplingQuery::CouplingImpulse {
            particle_inv_mass,
            particle_position,
            particle_velocity,
            body_inv_mass,
            body_inv_inertia_world,
            body_center_of_mass,
            body_linear_velocity,
            body_angular_velocity,
            normal,
            restitution,
        }
        | TwoWayCouplingQuery::ResolveCoupling {
            particle_inv_mass,
            particle_position,
            particle_velocity,
            body_inv_mass,
            body_inv_inertia_world,
            body_center_of_mass,
            body_linear_velocity,
            body_angular_velocity,
            normal,
            restitution,
        } => {
            q.op = match query {
                TwoWayCouplingQuery::ResolveCoupling { .. } => OP_RESOLVE_COUPLING,
                _ => OP_COUPLING_IMPULSE,
            };
            q.particle_inv_mass = *particle_inv_mass;
            q.body_inv_mass = *body_inv_mass;
            q.restitution = *restitution;
            set_matrix_a(&mut q, *body_inv_inertia_world);
            q.position = lane(*particle_position);
            q.velocity = lane(*particle_velocity);
            q.center_of_mass = lane(*body_center_of_mass);
            q.body_linear = lane(*body_linear_velocity);
            q.body_angular = lane(*body_angular_velocity);
            q.normal = lane(*normal);
        }
        TwoWayCouplingQuery::LinearMomentum { mass, velocity } => {
            q.op = OP_LINEAR_MOMENTUM;
            q.mass = *mass;
            q.position = lane(*velocity);
        }
    }
    q
}

/// Decodes one packed [`GpuResult`] into the public [`TwoWayCouplingResult`],
/// using the originating `query` to select the result shape.
fn decode_result(query: &TwoWayCouplingQuery, raw: &GpuResult) -> TwoWayCouplingResult {
    match query {
        TwoWayCouplingQuery::MatMulVec3 { .. }
        | TwoWayCouplingQuery::BodyPointVelocity { .. }
        | TwoWayCouplingQuery::CouplingImpulse { .. }
        | TwoWayCouplingQuery::LinearMomentum { .. } => {
            TwoWayCouplingResult::Vector(unlane(raw.r0))
        }
        TwoWayCouplingQuery::MatTranspose { .. }
        | TwoWayCouplingQuery::MatMulMat3 { .. }
        | TwoWayCouplingQuery::InvInertiaWorld { .. } => {
            TwoWayCouplingResult::Matrix([unlane(raw.r0), unlane(raw.r1), unlane(raw.r2)])
        }
        TwoWayCouplingQuery::GeneralizedInverseMass { .. } => {
            TwoWayCouplingResult::Scalar(raw.r0[0])
        }
        TwoWayCouplingQuery::ResolveCoupling { .. } => TwoWayCouplingResult::Velocities {
            impulse: unlane(raw.r0),
            particle_velocity: unlane(raw.r1),
            body_linear_velocity: unlane(raw.r2),
            body_angular_velocity: unlane(raw.r3),
        },
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

/// A compiled, reusable two-way-coupling compute pipeline, twinning the `CPU`
/// golden
/// [`two_way_coupling`](prism_render_architecture::particle::two_way_coupling).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::two_way_coupling`。
pub struct GpuTwoWayCoupling {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTwoWayCoupling {
    /// Compiles the two-way-coupling kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTwoWayCoupling {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_two_way_coupling"),
            source: ShaderSource::Wgsl(TWO_WAY_COUPLING_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_two_way_coupling_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_two_way_coupling_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_two_way_coupling_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTwoWayCoupling {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one
    /// [`TwoWayCouplingResult`] per input, in order.
    ///
    /// The results match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TwoWayCouplingQuery],
    ) -> Vec<TwoWayCouplingResult> {
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
            label: Some("prism_volumetric_two_way_coupling_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_two_way_coupling_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_two_way_coupling_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_two_way_coupling_bind_group"),
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
            label: Some("prism_volumetric_two_way_coupling_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_two_way_coupling_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_two_way_coupling_pass"),
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(query, result)| decode_result(query, result))
            .collect()
    }
}
