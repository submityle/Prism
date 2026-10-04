//! `wgpu` compute twin of the rotational discrete-element contact law, from the
//! `CPU` golden `prism_physics_core::collider::rotational_contact`'s
//! `rotational_contact_between`.
//!
//! A granular contact between two spheres resists both the sliding of the two
//! surfaces and the grains rolling over one another. The golden evaluates, for
//! one `(a, b)` sphere pair, a normal penalty force, a Cundall–Strack
//! tangential-history friction force clamped to the Coulomb cone, and a rolling
//! resistance couple clamped to `μ_r · R_r · F_n`, advancing the two persistent
//! springs in the process. This module ports that single stateless evaluation
//! onto the device: one thread resolves one contact pair, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same forces, torques and advanced springs the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `rotational_contact_between` for one
//! explicit contact pair: the model `(kₙ, γₙ, k_t, γ_t, μ, k_r, γ_r, μ_r)`, the
//! two centres, radii, linear velocities, angular velocities, the two incoming
//! springs (`tangential`, `rolling`) and the step `dt`. The closed form, in
//! golden operator order, is:
//!
//! * `delta = pos_b − pos_a`, `distance = |delta|`; coincident centres
//!   (`distance ≤ 0`) are a no-op (`valid = 0`).
//! * `overlap = (rad_a + rad_b) − distance`; non-overlapping (`overlap ≤ 0`) is
//!   a no-op (`valid = 0`).
//! * `normal = delta / distance`, the shared contact point splits the overlap:
//!   `arm_a = rad_a − 0.5·overlap`, `arm_b = rad_b − 0.5·overlap`,
//!   `r_a = normal·arm_a`, `r_b = −normal·arm_b`.
//! * surface-relative velocity `rel = (vel_b + ω_b×r_b) − (vel_a + ω_a×r_a)`,
//!   normal split `v_n = rel·normal`, normal force
//!   `F_n = max(kₙ·overlap − γₙ·v_n, 0)`.
//! * tangential history `v_t = rel − v_n·normal`, projected spring update,
//!   force `−k_t·ξ − γ_t·v_t` clamped to `μ·F_n` (sliding when the clamp bites
//!   and `μ·F_n > 0`), parking the spring on the cone.
//! * `force_on_b = F_n·normal + tangential_force`, spin torques
//!   `torque_on_b = r_b×force_on_b`, `torque_on_a = r_a×(−force_on_b)`.
//! * rolling couple over `ω_rel = ω_a − ω_b` with reduced radius
//!   `R_r = rad_a·rad_b / (rad_a + rad_b)`, clamped to `μ_r·R_r·F_n`, added to
//!   `torque_on_a` and subtracted from `torque_on_b`.
//!
//! A no-op pair returns `valid = 0`, zeroed forces, torques and magnitudes,
//! `sliding = 0`, and the two springs echoed back unchanged.
//!
//! # Correctness model
//!
//! Both the reference and the device evaluate the law in `f32`; the host oracle
//! replays the exact golden `if`-branch structure in `f32` so the comparison is
//! against the same closed form, not an arbitrary rewrite. Each continuous
//! quantity is compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`); the discrete `valid` and `sliding` flags are compared
//! exactly, and the parity sweep keeps the two Coulomb clamps well away from
//! their caps so the discrete sliding / clamp decisions cannot be flipped by
//! `f32` rounding.
//!
//! # Degenerate inputs
//!
//! Coincident centres (`distance ≤ 0`) and non-overlapping spheres
//! (`overlap ≤ 0`) are no-ops with `valid = 0`; the `distance` divisor and the
//! two spring-reset divisors (`k_t`, `k_r`) and the reduced-radius denominator
//! are each routed through a `select` so the un-taken branch never divides by
//! zero. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `length`, `sqrt` (via `length`), `cross`, `dot`, `+ - * /`, `select` and
//! unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`, `log`,
//! `pow`, no `round` and no `f32` remainder, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. Every degenerate predicate is an ordered compare fed to
//! `select`; there is no bare `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::rotational_contact`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` rotational-contact kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `rotational_contact_between`; see the module documentation
/// for the closed form.
const ROTATIONAL_CONTACT_FORCE_WGSL: &str = r#"
// Rotational discrete-element contact twin: one thread per query reproduces the
// normal penalty, Cundall-Strack tangential friction clamped to the Coulomb
// cone, and the rolling-resistance couple, advancing the two springs. It uses
// only the portable core-WGSL subset (abs, min, max, length, cross, dot,
// + - * /, select plus unsigned index math), takes no optional feature, and has
// no loop and no branch, so it provably terminates. Every degenerate predicate
// is an ordered compare fed to select; there is no bare f32 equality anywhere.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Model group 0: normal stiffness, normal damping, tangential stiffness,
    // tangential damping.
    model0: vec4<f32>,
    // Model group 1: friction, rolling stiffness, rolling damping, rolling
    // friction.
    model1: vec4<f32>,
    pos_a: vec4<f32>,
    pos_b: vec4<f32>,
    vel_a: vec4<f32>,
    vel_b: vec4<f32>,
    omega_a: vec4<f32>,
    omega_b: vec4<f32>,
    spring_tangential: vec4<f32>,
    spring_rolling: vec4<f32>,
    // rad_a, rad_b, dt, pad.
    scalars: vec4<f32>,
}

struct Result {
    force_on_b: vec4<f32>,
    torque_on_a: vec4<f32>,
    torque_on_b: vec4<f32>,
    spring_tangential_out: vec4<f32>,
    spring_rolling_out: vec4<f32>,
    // overlap, normal magnitude, tangential magnitude, rolling magnitude.
    magnitudes: vec4<f32>,
    // valid, sliding, pad, pad.
    flags: vec4<u32>,
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
    let qd = queries[idx];

    let k_n = qd.model0.x;
    let gamma_n = qd.model0.y;
    let k_t = qd.model0.z;
    let gamma_t = qd.model0.w;
    let friction = qd.model1.x;
    let k_r = qd.model1.y;
    let gamma_r = qd.model1.z;
    let rolling_friction = qd.model1.w;

    let pos_a = qd.pos_a.xyz;
    let pos_b = qd.pos_b.xyz;
    let vel_a = qd.vel_a.xyz;
    let vel_b = qd.vel_b.xyz;
    let omega_a = qd.omega_a.xyz;
    let omega_b = qd.omega_b.xyz;
    let spring_tangential = qd.spring_tangential.xyz;
    let spring_rolling = qd.spring_rolling.xyz;
    let rad_a = qd.scalars.x;
    let rad_b = qd.scalars.y;
    let dt = qd.scalars.z;

    // Step 1-2: contact gating via ordered compares.
    let delta = pos_b - pos_a;
    let distance = length(delta);
    let dist_ok = distance > 0.0;
    let overlap = (rad_a + rad_b) - distance;
    let overlap_ok = overlap > 0.0;
    let valid = dist_ok && overlap_ok;

    // Guard the distance divisor so the un-taken branch never divides by zero.
    let safe_distance = select(1.0, distance, dist_ok);
    let normal = delta / safe_distance;

    // Step 3: shared contact point splits the overlap.
    let arm_a = rad_a - 0.5 * overlap;
    let arm_b = rad_b - 0.5 * overlap;
    let r_a = normal * arm_a;
    let r_b = -normal * arm_b;

    // Step 4: relative surface velocity at the contact point.
    let surf_a = vel_a + cross(omega_a, r_a);
    let surf_b = vel_b + cross(omega_b, r_b);
    let rel = surf_b - surf_a;

    // Step 5: normal penalty response.
    let v_n = dot(rel, normal);
    let normal_force = max(k_n * overlap - gamma_n * v_n, 0.0);

    // Step 6: tangential Cundall-Strack history clamped to the Coulomb cone.
    let v_t = rel - v_n * normal;
    let tang0 = spring_tangential - dot(spring_tangential, normal) * normal + v_t * dt;
    let tf0 = -k_t * tang0 - gamma_t * v_t;
    let tmag0 = length(tf0);
    let max_friction = friction * normal_force;
    let over_cap_t = tmag0 > max_friction;
    let mag_pos_t = tmag0 > 0.0;
    let safe_mag_t = select(1.0, tmag0, mag_pos_t);
    let dir_t = tf0 / safe_mag_t;
    let clamped_force_t = dir_t * max_friction;
    let safe_kt = select(1.0, k_t, k_t > 0.0);
    let clamped_tang = -clamped_force_t / safe_kt;
    let force_if_cap = select(vec3<f32>(0.0, 0.0, 0.0), clamped_force_t, mag_pos_t);
    let tang_if_cap = select(vec3<f32>(0.0, 0.0, 0.0), clamped_tang, mag_pos_t);
    let tf = select(tf0, force_if_cap, over_cap_t);
    let tang = select(tang0, tang_if_cap, over_cap_t);
    let tmag = select(tmag0, max_friction, over_cap_t);
    let sliding = over_cap_t && (max_friction > 0.0);

    // Step 7-8: contact force and its spin torques at the shared point.
    let force_on_b = normal_force * normal + tf;
    let torque_on_b0 = cross(r_b, force_on_b);
    let torque_on_a0 = cross(r_a, -force_on_b);

    // Step 9: rolling resistance couple over the relative rotation.
    let omega_rel = omega_a - omega_b;
    let sum_rad = rad_a + rad_b;
    let safe_sum = select(1.0, sum_rad, sum_rad > 0.0);
    let rolling_radius = (rad_a * rad_b) / safe_sum;
    let rolling0 = spring_rolling + omega_rel * dt;
    let rt0 = -k_r * rolling0 - gamma_r * omega_rel;
    let rmag0 = length(rt0);
    let max_rolling = rolling_friction * rolling_radius * normal_force;
    let over_cap_r = rmag0 > max_rolling;
    let mag_pos_r = rmag0 > 0.0;
    let safe_mag_r = select(1.0, rmag0, mag_pos_r);
    let dir_r = rt0 / safe_mag_r;
    let clamped_rt = dir_r * max_rolling;
    let safe_kr = select(1.0, k_r, k_r > 0.0);
    let clamped_rolling = -clamped_rt / safe_kr;
    let rt_if_cap = select(vec3<f32>(0.0, 0.0, 0.0), clamped_rt, mag_pos_r);
    let rolling_if_cap = select(vec3<f32>(0.0, 0.0, 0.0), clamped_rolling, mag_pos_r);
    let rt = select(rt0, rt_if_cap, over_cap_r);
    let rolling = select(rolling0, rolling_if_cap, over_cap_r);
    let rmag = select(rmag0, max_rolling, over_cap_r);

    // Step 10: fold the rolling couple into both torques.
    let torque_on_a = torque_on_a0 + rt;
    let torque_on_b = torque_on_b0 - rt;

    let zero3 = vec3<f32>(0.0, 0.0, 0.0);
    var out: Result;
    out.force_on_b = vec4<f32>(select(zero3, force_on_b, valid), 0.0);
    out.torque_on_a = vec4<f32>(select(zero3, torque_on_a, valid), 0.0);
    out.torque_on_b = vec4<f32>(select(zero3, torque_on_b, valid), 0.0);
    // A no-op pair echoes the incoming springs unchanged.
    out.spring_tangential_out = vec4<f32>(select(spring_tangential, tang, valid), 0.0);
    out.spring_rolling_out = vec4<f32>(select(spring_rolling, rolling, valid), 0.0);
    out.magnitudes = vec4<f32>(
        select(0.0, overlap, valid),
        select(0.0, normal_force, valid),
        select(0.0, tmag, valid),
        select(0.0, rmag, valid),
    );
    out.flags = vec4<u32>(
        select(0u, 1u, valid),
        select(0u, 1u, valid && sliding),
        0u,
        0u,
    );
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// eleven `vec4` lanes (`176` bytes). Every 3-vector is flattened into the
/// `xyz` of a `vec4` with a padding `w`, so no `vec3` alignment rule applies.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    model0: [f32; 4],
    model1: [f32; 4],
    pos_a: [f32; 4],
    pos_b: [f32; 4],
    vel_a: [f32; 4],
    vel_b: [f32; 4],
    omega_a: [f32; 4],
    omega_b: [f32; 4],
    spring_tangential: [f32; 4],
    spring_rolling: [f32; 4],
    scalars: [f32; 4],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: five `vec4` output vectors, one `vec4` of magnitudes and one `vec4`
/// of `u32` flags — seven lanes (`112` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    force_on_b: [f32; 4],
    torque_on_a: [f32; 4],
    torque_on_b: [f32; 4],
    spring_tangential_out: [f32; 4],
    spring_rolling_out: [f32; 4],
    magnitudes: [f32; 4],
    flags: [u32; 4],
}

/// One rotational-contact query: the model parameters, the `(a, b)` sphere
/// pair's kinematics and the two incoming persistent springs plus the step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotationalContactForceQuery {
    /// Normal penalty stiffness `kₙ`.
    pub normal_stiffness: f32,
    /// Normal viscous damping `γₙ`.
    pub normal_damping: f32,
    /// Tangential spring stiffness `k_t`.
    pub tangential_stiffness: f32,
    /// Tangential viscous damping `γ_t`.
    pub tangential_damping: f32,
    /// Sliding Coulomb friction coefficient `μ`.
    pub friction: f32,
    /// Rolling resistance spring stiffness `k_r`.
    pub rolling_stiffness: f32,
    /// Rolling resistance viscous damping `γ_r`.
    pub rolling_damping: f32,
    /// Rolling resistance coefficient `μ_r`.
    pub rolling_friction: f32,
    /// Centre of sphere `a`.
    pub pos_a: [f32; 3],
    /// Centre of sphere `b`.
    pub pos_b: [f32; 3],
    /// Linear velocity of sphere `a`.
    pub vel_a: [f32; 3],
    /// Linear velocity of sphere `b`.
    pub vel_b: [f32; 3],
    /// Angular velocity of sphere `a`.
    pub omega_a: [f32; 3],
    /// Angular velocity of sphere `b`.
    pub omega_b: [f32; 3],
    /// Incoming tangential-history spring `ξ`.
    pub spring_tangential: [f32; 3],
    /// Incoming rolling spring `θ`.
    pub spring_rolling: [f32; 3],
    /// Radius of sphere `a`.
    pub rad_a: f32,
    /// Radius of sphere `b`.
    pub rad_b: f32,
    /// Integration step `dt`.
    pub dt: f32,
}

impl RotationalContactForceQuery {
    /// Builds a query from the model parameter array
    /// `[kₙ, γₙ, k_t, γ_t, μ, k_r, γ_r, μ_r]`, the two sphere radii, the two
    /// centres, their linear and angular velocities, the two incoming springs
    /// and the step `dt`.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a contact pair is irreducibly many kinematic vectors"
    )]
    pub fn new(
        model: [f32; 8],
        pos_a: [f32; 3],
        pos_b: [f32; 3],
        rad_a: f32,
        rad_b: f32,
        vel_a: [f32; 3],
        vel_b: [f32; 3],
        omega_a: [f32; 3],
        omega_b: [f32; 3],
        spring_tangential: [f32; 3],
        spring_rolling: [f32; 3],
        dt: f32,
    ) -> RotationalContactForceQuery {
        RotationalContactForceQuery {
            normal_stiffness: model[0],
            normal_damping: model[1],
            tangential_stiffness: model[2],
            tangential_damping: model[3],
            friction: model[4],
            rolling_stiffness: model[5],
            rolling_damping: model[6],
            rolling_friction: model[7],
            pos_a,
            pos_b,
            vel_a,
            vel_b,
            omega_a,
            omega_b,
            spring_tangential,
            spring_rolling,
            rad_a,
            rad_b,
            dt,
        }
    }
}

/// One resolved contact, mirroring `rotational_contact_between` for that pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotationalContactForceResult {
    /// `true` when the spheres overlap with distinct centres, else `false`
    /// (a no-op contact).
    pub valid: bool,
    /// Contact force on `b`; `a` feels its negation at the same point.
    pub force_on_b: [f32; 3],
    /// Total torque on `a` (friction-force moment plus rolling couple).
    pub torque_on_a: [f32; 3],
    /// Total torque on `b` (friction-force moment plus rolling couple).
    pub torque_on_b: [f32; 3],
    /// Advanced tangential spring (echoed input when the contact is a no-op).
    pub spring_tangential_out: [f32; 3],
    /// Advanced rolling spring (echoed input when the contact is a no-op).
    pub spring_rolling_out: [f32; 3],
    /// Overlap `δ > 0` resolved by this contact.
    pub overlap: f32,
    /// Normal force magnitude `F_n ≥ 0`.
    pub normal_magnitude: f32,
    /// Tangential friction force magnitude actually applied.
    pub tangential_magnitude: f32,
    /// Rolling resistance torque magnitude actually applied.
    pub rolling_magnitude: f32,
    /// Whether the tangential force reached the Coulomb limit (sliding).
    pub sliding: bool,
}

/// Encodes one [`RotationalContactForceQuery`] into its `std430` [`GpuQuery`]
/// slot, flattening each 3-vector into the `xyz` of a padded `vec4`.
fn encode_query(q: &RotationalContactForceQuery) -> GpuQuery {
    let v4 = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
    GpuQuery {
        model0: [
            q.normal_stiffness,
            q.normal_damping,
            q.tangential_stiffness,
            q.tangential_damping,
        ],
        model1: [
            q.friction,
            q.rolling_stiffness,
            q.rolling_damping,
            q.rolling_friction,
        ],
        pos_a: v4(q.pos_a),
        pos_b: v4(q.pos_b),
        vel_a: v4(q.vel_a),
        vel_b: v4(q.vel_b),
        omega_a: v4(q.omega_a),
        omega_b: v4(q.omega_b),
        spring_tangential: v4(q.spring_tangential),
        spring_rolling: v4(q.spring_rolling),
        scalars: [q.rad_a, q.rad_b, q.dt, 0.0],
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`RotationalContactForceResult`].
fn decode_result(raw: &GpuResult) -> RotationalContactForceResult {
    let v3 = |v: [f32; 4]| [v[0], v[1], v[2]];
    RotationalContactForceResult {
        valid: raw.flags[0] != 0,
        force_on_b: v3(raw.force_on_b),
        torque_on_a: v3(raw.torque_on_a),
        torque_on_b: v3(raw.torque_on_b),
        spring_tangential_out: v3(raw.spring_tangential_out),
        spring_rolling_out: v3(raw.spring_rolling_out),
        overlap: raw.magnitudes[0],
        normal_magnitude: raw.magnitudes[1],
        tangential_magnitude: raw.magnitudes[2],
        rolling_magnitude: raw.magnitudes[3],
        sliding: raw.flags[1] != 0,
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

/// A compiled, reusable rotational-contact compute pipeline, twinning the `CPU`
/// golden `rotational_contact_between`.
pub struct GpuRotationalContactForce {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRotationalContactForce {
    /// Compiles the rotational-contact kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRotationalContactForce {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_rotational_contact_force"),
            source: ShaderSource::Wgsl(ROTATIONAL_CONTACT_FORCE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_rotational_contact_force_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_rotational_contact_force_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_rotational_contact_force_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRotationalContactForce {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`RotationalContactForceResult`] per input, in order.
    ///
    /// The `valid` and `sliding` flags match the reference exactly and the
    /// continuous forces, torques, magnitudes and advanced springs to the
    /// module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RotationalContactForceQuery],
    ) -> Vec<RotationalContactForceResult> {
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
            label: Some("prism_volumetric_rotational_contact_force_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_rotational_contact_force_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_rotational_contact_force_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_rotational_contact_force_bind_group"),
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
            label: Some("prism_volumetric_rotational_contact_force_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_rotational_contact_force_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_rotational_contact_force_pass"),
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
