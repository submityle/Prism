//! `wgpu` compute twin of the per-point grid-fluid, combustion and
//! vorticity-confinement contract
//! ([`fluid`](prism_render_architecture::particle::fluid), particle design §10,
//! §17).
//!
//! The `CPU` golden [`fluid`](prism_render_architecture::particle::fluid) owns
//! the deterministic, per-point numerics of the stable-fluids pyro pipeline: the
//! advection `CFL` scalars
//! ([`cfl_number`](prism_render_architecture::particle::fluid::cfl_number),
//! [`stable_timestep`](prism_render_architecture::particle::fluid::stable_timestep)),
//! the semi-Lagrangian and `MacCormack` advection steps
//! ([`semi_lagrangian_backtrace`](prism_render_architecture::particle::fluid::semi_lagrangian_backtrace),
//! [`maccormack_corrected`](prism_render_architecture::particle::fluid::maccormack_corrected),
//! [`advect_particle`](prism_render_architecture::particle::fluid::advect_particle)),
//! the trilinear interpolation kernel
//! ([`trilinear_weights`](prism_render_architecture::particle::fluid::trilinear_weights),
//! [`trilinear_sample`](prism_render_architecture::particle::fluid::trilinear_sample)),
//! the central-difference field operators
//! ([`central_gradient`](prism_render_architecture::particle::fluid::central_gradient),
//! [`subtract_pressure_gradient`](prism_render_architecture::particle::fluid::subtract_pressure_gradient)),
//! the three-channel combustion coupling
//! ([`step_combustion`](prism_render_architecture::particle::fluid::step_combustion),
//! [`buoyancy_force`](prism_render_architecture::particle::fluid::buoyancy_force),
//! [`blackbody_emission`](prism_render_architecture::particle::fluid::blackbody_emission),
//! [`heat_haze_distortion`](prism_render_architecture::particle::fluid::heat_haze_distortion))
//! and the vorticity estimate and confinement force
//! ([`NeighborVelocities::curl`](prism_render_architecture::particle::fluid::NeighborVelocities::curl),
//! [`vorticity_confinement_force`](prism_render_architecture::particle::fluid::vorticity_confinement_force)).
//! [`GpuFluid`] is the on-device twin: one thread solves one [`FluidQuery`] and
//! writes one [`FluidResult`], so a passing real-device parity test is direct
//! evidence the ported kernel folds the same scalars, velocities, weights and
//! combustion states the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query selects one routine by a `u32` tag and the kernel reproduces it
//! branch for branch: the two `CFL` guards, the three multiply-add advection
//! updates, the eight-corner trilinear weight product and blend, the two
//! central-difference operators, the linear combustion burn / cool update, the
//! buoyancy and heat-haze body terms, the polynomial black-body colour, and the
//! finite-difference curl plus the normalized confinement cross product.
//!
//! # What is left on the host
//!
//! Routines that walk a voxel grid are deliberately *not* twinned here, since a
//! single thread cannot own an unbounded field slice:
//! [`sample_velocity_field`](prism_render_architecture::particle::fluid::sample_velocity_field)
//! and
//! [`advect_particle_in_field`](prism_render_architecture::particle::fluid::advect_particle_in_field)
//! stay on the host, which pre-samples the eight corner velocities and
//! dispatches a [`FluidQuery::TrilinearSample`];
//! [`pressure_residual_l2`](prism_render_architecture::particle::fluid::pressure_residual_l2)
//! and
//! [`jacobi_pressure_solve`](prism_render_architecture::particle::fluid::jacobi_pressure_solve)
//! traverse the whole grid per sweep; and
//! [`GridResolution::linear_index`](prism_render_architecture::particle::fluid::GridResolution::linear_index)
//! and
//! [`GridResolution::voxel_count`](prism_render_architecture::particle::fluid::GridResolution::voxel_count)
//! are host index arithmetic.
//!
//! # No transcendental math
//!
//! Every routine is multiply-add plus at most one `sqrt` (the confinement
//! force's normalize). The kernel uses no `sin`, `cos`, `tan`, `exp`, `log`,
//! `pow`, no inverse trigonometry and no `smoothstep` or `round`; the black-body
//! glow uses the reference's multiply-only polynomial, not a Planck `exp`.
//!
//! # Correctness model
//!
//! The dispatch tag is an integer classification, so the kernel runs exactly the
//! branch the host requested. The continuous entries thread through multiplies,
//! adds, guarded divisions and at most one `sqrt`, so `CPU` and `GPU` are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits. The parity test therefore asserts
//! a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous quantity.
//!
//! # Degenerate inputs
//!
//! A zero cell size, a zero max speed and a flat (zero-gradient) vorticity
//! region all hit the same `EPS_LEN_SQ` guards the reference uses: the `CFL`
//! scalars return zero instead of dividing by a near-zero denominator, and the
//! confinement force normalizes to zero instead of a `NaN`. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `sqrt`, `+ - * /` and unsigned index arithmetic — with no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There
//! is no loop: each thread performs a fixed, bounded sequence of arithmetic, so
//! the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`；无第三方引擎源码或衍生代码。
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
use prism_render_architecture::particle::fluid::{
    advect_particle, blackbody_emission, buoyancy_force, central_gradient, cfl_number,
    heat_haze_distortion, maccormack_corrected, semi_lagrangian_backtrace, stable_timestep,
    step_combustion, subtract_pressure_gradient, trilinear_sample, trilinear_weights,
    vorticity_confinement_force, CombustionParams, CombustionState, EmissionParams,
    NeighborScalars, NeighborVelocities, VorticityParams,
};
use prism_render_architecture::particle::Vec3;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` fluid kernel, embedded inline so the twin ships as a
/// single source file. The single entry point `solve` mirrors the `CPU` golden
/// [`fluid`](prism_render_architecture::particle::fluid) branch for branch; see
/// the module documentation for the algorithm.
const FLUID_WGSL: &str = r#"
// Fluid twin: one thread per query runs the routine its `tag` selects,
// reproducing the CPU golden `particle::fluid` branch for branch. It uses only
// the portable core-WGSL subset (abs/min/max/clamp/sqrt and + - * / plus
// unsigned index math), needs no transcendental call and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12. There is no loop, so
// the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::fluid；无第三方引擎源码
// 或衍生代码。

// Squared-length threshold below which a denominator (cell size, max speed, or
// gradient length) is treated as degenerate; mirrors the reference `EPS_LEN_SQ`.
// Used instead of an f32 `==` compare.
const EPS_LEN_SQ: f32 = 1.0e-12;

// Routine tags; the host casts its query discriminant straight to these codes.
const TAG_CFL_NUMBER: u32 = 0u;
const TAG_STABLE_TIMESTEP: u32 = 1u;
const TAG_SEMI_LAGRANGIAN_BACKTRACE: u32 = 2u;
const TAG_MACCORMACK_CORRECTED: u32 = 3u;
const TAG_ADVECT_PARTICLE: u32 = 4u;
const TAG_TRILINEAR_WEIGHTS: u32 = 5u;
const TAG_TRILINEAR_SAMPLE: u32 = 6u;
const TAG_CENTRAL_GRADIENT: u32 = 7u;
const TAG_SUBTRACT_PRESSURE_GRADIENT: u32 = 8u;
const TAG_STEP_COMBUSTION: u32 = 9u;
const TAG_BUOYANCY_FORCE: u32 = 10u;
const TAG_BLACKBODY_EMISSION: u32 = 11u;
const TAG_HEAT_HAZE_DISTORTION: u32 = 12u;
const TAG_CURL: u32 = 13u;
const TAG_VORTICITY_CONFINEMENT_FORCE: u32 = 14u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Routine selector plus three pad words to fill the std430 16-byte slot.
    tag: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // General scalar lane. cfl: (max_velocity, dt, cell_size, _); stable:
    // (max_velocity, cell_size, cfl_target, _); blackbody: (temperature, low,
    // white, intensity); advection/haze: dt or strength in x; central_gradient
    // and curl: inv_2h in x; vorticity: (epsilon, cell_size, _, _); combustion:
    // dt in x.
    scalar_a: vec4<f32>,
    // Neighbor scalars for central_gradient: (x_plus, x_minus, y_plus, y_minus).
    ns0: vec4<f32>,
    // Neighbor scalars for central_gradient: (z_plus, z_minus, _, _).
    ns1: vec4<f32>,
    // Combustion params: (ignition_temperature, burn_rate, smoke_yield, heat_yield).
    cparams0: vec4<f32>,
    // Combustion params: (cooling_rate, ambient_temperature, buoyancy_alpha, buoyancy_beta).
    cparams1: vec4<f32>,
    // Combustion state: (temperature, fuel, smoke, _).
    cstate: vec4<f32>,
    // General vector operands (xyz; w pad): pos / forward / velocity / curl /
    // gradient depending on the routine.
    vec_a: vec4<f32>,
    vec_b: vec4<f32>,
    vec_c: vec4<f32>,
    // Trilinear corner velocities c0..c7; also reused as the six curl neighbors
    // (c0..c5 = x_plus, x_minus, y_plus, y_minus, z_plus, z_minus).
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    c3: vec4<f32>,
    c4: vec4<f32>,
    c5: vec4<f32>,
    c6: vec4<f32>,
    c7: vec4<f32>,
    // Trilinear weights 0..3 and 4..7.
    w0: vec4<f32>,
    w1: vec4<f32>,
}

struct Result {
    // Scalar lane: cfl / timestep in x.
    scalar: vec4<f32>,
    // Vector lane (xyz): advection, gradient, force, colour or sampled velocity.
    vector: vec4<f32>,
    // Trilinear weight outputs 0..3 and 4..7.
    weights0: vec4<f32>,
    weights1: vec4<f32>,
    // Combustion state output: (temperature, fuel, smoke, _).
    combustion: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Hand-rolled cross product (the core subset does not guarantee a builtin);
// mirrors the reference `Vec3::cross`.
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    );
}

// Unit vector along `v`, or the zero vector when `v` is (numerically) zero;
// mirrors the reference `Vec3::normalize_or_zero`, never yielding a NaN.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
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

    var out: Result;
    out.scalar = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.vector = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.weights0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.weights1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.combustion = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    if (q.tag == TAG_CFL_NUMBER) {
        // |v|*dt / h with a guarded denominator.
        let cell_size = q.scalar_a.z;
        if (abs(cell_size) > EPS_LEN_SQ) {
            out.scalar.x = (q.scalar_a.x * q.scalar_a.y) / cell_size;
        }
    } else if (q.tag == TAG_STABLE_TIMESTEP) {
        // cfl_target*h / |v| with a guarded denominator.
        let max_velocity = q.scalar_a.x;
        if (abs(max_velocity) > EPS_LEN_SQ) {
            out.scalar.x = (q.scalar_a.z * q.scalar_a.y) / max_velocity;
        }
    } else if (q.tag == TAG_SEMI_LAGRANGIAN_BACKTRACE) {
        // pos - velocity*dt.
        out.vector = vec4<f32>(q.vec_a.xyz - q.vec_b.xyz * q.scalar_a.x, 0.0);
    } else if (q.tag == TAG_MACCORMACK_CORRECTED) {
        // forward + 0.5*(original - back).
        out.vector = vec4<f32>(q.vec_a.xyz + (q.vec_b.xyz - q.vec_c.xyz) * 0.5, 0.0);
    } else if (q.tag == TAG_ADVECT_PARTICLE) {
        // pos + velocity*dt.
        out.vector = vec4<f32>(q.vec_a.xyz + q.vec_b.xyz * q.scalar_a.x, 0.0);
    } else if (q.tag == TAG_TRILINEAR_WEIGHTS) {
        let fx = q.vec_a.x;
        let fy = q.vec_a.y;
        let fz = q.vec_a.z;
        let gx = 1.0 - fx;
        let gy = 1.0 - fy;
        let gz = 1.0 - fz;
        out.weights0 = vec4<f32>(gx * gy * gz, fx * gy * gz, gx * fy * gz, fx * fy * gz);
        out.weights1 = vec4<f32>(gx * gy * fz, fx * gy * fz, gx * fy * fz, fx * fy * fz);
    } else if (q.tag == TAG_TRILINEAR_SAMPLE) {
        var acc = vec3<f32>(0.0, 0.0, 0.0);
        acc = acc + q.c0.xyz * q.w0.x;
        acc = acc + q.c1.xyz * q.w0.y;
        acc = acc + q.c2.xyz * q.w0.z;
        acc = acc + q.c3.xyz * q.w0.w;
        acc = acc + q.c4.xyz * q.w1.x;
        acc = acc + q.c5.xyz * q.w1.y;
        acc = acc + q.c6.xyz * q.w1.z;
        acc = acc + q.c7.xyz * q.w1.w;
        out.vector = vec4<f32>(acc, 0.0);
    } else if (q.tag == TAG_CENTRAL_GRADIENT) {
        let inv_2h = q.scalar_a.x;
        out.vector = vec4<f32>(
            (q.ns0.x - q.ns0.y) * inv_2h,
            (q.ns0.z - q.ns0.w) * inv_2h,
            (q.ns1.x - q.ns1.y) * inv_2h,
            0.0,
        );
    } else if (q.tag == TAG_SUBTRACT_PRESSURE_GRADIENT) {
        // velocity - pressure_gradient.
        out.vector = vec4<f32>(q.vec_a.xyz - q.vec_b.xyz, 0.0);
    } else if (q.tag == TAG_STEP_COMBUSTION) {
        var temperature = q.cstate.x;
        var fuel = q.cstate.y;
        var smoke = q.cstate.z;
        let dt = q.scalar_a.x;
        let ignition = q.cparams0.x;
        let burn_rate = q.cparams0.y;
        let smoke_yield = q.cparams0.z;
        let heat_yield = q.cparams0.w;
        let cooling_rate = q.cparams1.x;
        let ambient = q.cparams1.y;
        if (temperature >= ignition && fuel > 0.0) {
            var burned = burn_rate * dt;
            if (burned > fuel) {
                burned = fuel;
            }
            fuel = fuel - burned;
            smoke = smoke + burned * smoke_yield;
            temperature = temperature + burned * heat_yield;
        }
        let cooled = cooling_rate * dt * (temperature - ambient);
        temperature = temperature - cooled;
        out.combustion = vec4<f32>(temperature, fuel, smoke, 0.0);
    } else if (q.tag == TAG_BUOYANCY_FORCE) {
        let ambient = q.cparams1.y;
        let alpha = q.cparams1.z;
        let beta = q.cparams1.w;
        let lift = (q.cstate.x - ambient) * alpha - q.cstate.z * beta;
        out.vector = vec4<f32>(0.0, lift, 0.0, 0.0);
    } else if (q.tag == TAG_BLACKBODY_EMISSION) {
        let temperature = q.scalar_a.x;
        let low = q.scalar_a.y;
        let white = q.scalar_a.z;
        let intensity_scale = q.scalar_a.w;
        let span = white - low;
        var t = 0.0;
        if (span > EPS_LEN_SQ) {
            t = clamp((temperature - low) / span, 0.0, 1.0);
        }
        let intensity = t * intensity_scale;
        let r = clamp(t * 1.6, 0.0, 1.0);
        let g = clamp(t * t * 1.3, 0.0, 1.0);
        let b = clamp(t * t * t, 0.0, 1.0);
        out.vector = vec4<f32>(vec3<f32>(r, g, b) * intensity, 0.0);
    } else if (q.tag == TAG_HEAT_HAZE_DISTORTION) {
        // temperature_gradient * strength.
        out.vector = vec4<f32>(q.vec_a.xyz * q.scalar_a.x, 0.0);
    } else if (q.tag == TAG_CURL) {
        let inv_2h = q.scalar_a.x;
        let x_plus = q.c0.xyz;
        let x_minus = q.c1.xyz;
        let y_plus = q.c2.xyz;
        let y_minus = q.c3.xyz;
        let z_plus = q.c4.xyz;
        let z_minus = q.c5.xyz;
        let dwz_dy = y_plus.z - y_minus.z;
        let dvy_dz = z_plus.y - z_minus.y;
        let dux_dz = z_plus.x - z_minus.x;
        let dwz_dx = x_plus.z - x_minus.z;
        let dvy_dx = x_plus.y - x_minus.y;
        let dux_dy = y_plus.x - y_minus.x;
        out.vector = vec4<f32>(
            (dwz_dy - dvy_dz) * inv_2h,
            (dux_dz - dwz_dx) * inv_2h,
            (dvy_dx - dux_dy) * inv_2h,
            0.0,
        );
    } else if (q.tag == TAG_VORTICITY_CONFINEMENT_FORCE) {
        let curl = q.vec_a.xyz;
        let n = normalize_or_zero(q.vec_b.xyz);
        let force = cross3(n, curl) * (q.scalar_a.x * q.scalar_a.y);
        out.vector = vec4<f32>(force, 0.0);
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`FLUID_WGSL`].
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
/// Every lane is a padded `vec4` so each slot stays `16`-byte aligned on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Routine selector matching the `WGSL` `TAG_*` codes.
    tag: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// General scalar lane (per-routine packing documented in `FLUID_WGSL`).
    scalar_a: [f32; 4],
    /// Neighbor scalars `(x_plus, x_minus, y_plus, y_minus)`.
    ns0: [f32; 4],
    /// Neighbor scalars `(z_plus, z_minus, _, _)`.
    ns1: [f32; 4],
    /// Combustion params `(ignition, burn_rate, smoke_yield, heat_yield)`.
    cparams0: [f32; 4],
    /// Combustion params `(cooling_rate, ambient, buoyancy_alpha, buoyancy_beta)`.
    cparams1: [f32; 4],
    /// Combustion state `(temperature, fuel, smoke, _)`.
    cstate: [f32; 4],
    /// General vector operand `a` (`xyz`; `w` pad).
    vec_a: [f32; 4],
    /// General vector operand `b` (`xyz`; `w` pad).
    vec_b: [f32; 4],
    /// General vector operand `c` (`xyz`; `w` pad).
    vec_c: [f32; 4],
    /// Trilinear corner `0` / curl `x_plus` (`xyz`; `w` pad).
    c0: [f32; 4],
    /// Trilinear corner `1` / curl `x_minus` (`xyz`; `w` pad).
    c1: [f32; 4],
    /// Trilinear corner `2` / curl `y_plus` (`xyz`; `w` pad).
    c2: [f32; 4],
    /// Trilinear corner `3` / curl `y_minus` (`xyz`; `w` pad).
    c3: [f32; 4],
    /// Trilinear corner `4` / curl `z_plus` (`xyz`; `w` pad).
    c4: [f32; 4],
    /// Trilinear corner `5` / curl `z_minus` (`xyz`; `w` pad).
    c5: [f32; 4],
    /// Trilinear corner `6` (`xyz`; `w` pad).
    c6: [f32; 4],
    /// Trilinear corner `7` (`xyz`; `w` pad).
    c7: [f32; 4],
    /// Trilinear weights `0..3`.
    w0: [f32; 4],
    /// Trilinear weights `4..7`.
    w1: [f32; 4],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Scalar lane: `cfl` or timestep in `x`.
    scalar: [f32; 4],
    /// Vector lane (`xyz`): advection, gradient, force, colour or sample.
    vector: [f32; 4],
    /// Trilinear weight outputs `0..3`.
    weights0: [f32; 4],
    /// Trilinear weight outputs `4..7`.
    weights1: [f32; 4],
    /// Combustion state output `(temperature, fuel, smoke, _)`.
    combustion: [f32; 4],
}

/// One query for the fluid twin: a tagged union selecting which golden routine
/// to run with its typed inputs.
///
/// Each variant twins exactly one `CPU` golden function. The golden
/// [`CombustionState`](prism_render_architecture::particle::fluid::CombustionState),
/// [`CombustionParams`](prism_render_architecture::particle::fluid::CombustionParams),
/// [`EmissionParams`](prism_render_architecture::particle::fluid::EmissionParams),
/// [`NeighborScalars`](prism_render_architecture::particle::fluid::NeighborScalars),
/// [`NeighborVelocities`](prism_render_architecture::particle::fluid::NeighborVelocities),
/// [`VorticityParams`](prism_render_architecture::particle::fluid::VorticityParams)
/// and [`Vec3`](prism_render_architecture::particle::Vec3) contract types are
/// reused directly rather than re-declared.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FluidQuery {
    /// The advection `CFL` number, twinning
    /// [`cfl_number`](prism_render_architecture::particle::fluid::cfl_number).
    CflNumber {
        /// Maximum flow speed.
        max_velocity: f32,
        /// Time step.
        dt: f32,
        /// Grid cell size.
        cell_size: f32,
    },
    /// The largest stable time step, twinning
    /// [`stable_timestep`](prism_render_architecture::particle::fluid::stable_timestep).
    StableTimestep {
        /// Maximum flow speed.
        max_velocity: f32,
        /// Grid cell size.
        cell_size: f32,
        /// Target `CFL` number.
        cfl_target: f32,
    },
    /// The semi-Lagrangian back-trace, twinning
    /// [`semi_lagrangian_backtrace`](prism_render_architecture::particle::fluid::semi_lagrangian_backtrace).
    SemiLagrangianBacktrace {
        /// Arrival position.
        pos: Vec3,
        /// Local velocity.
        velocity: Vec3,
        /// Time step.
        dt: f32,
    },
    /// The `MacCormack` diffusion correction, twinning
    /// [`maccormack_corrected`](prism_render_architecture::particle::fluid::maccormack_corrected).
    MaccormackCorrected {
        /// Forward-advected value.
        forward: Vec3,
        /// Original value.
        original: Vec3,
        /// Back-advected value.
        back_advected: Vec3,
    },
    /// The forward particle integration, twinning
    /// [`advect_particle`](prism_render_architecture::particle::fluid::advect_particle).
    AdvectParticle {
        /// Current position.
        pos: Vec3,
        /// Sampled velocity.
        velocity: Vec3,
        /// Time step.
        dt: f32,
    },
    /// The eight trilinear corner weights, twinning
    /// [`trilinear_weights`](prism_render_architecture::particle::fluid::trilinear_weights).
    TrilinearWeights {
        /// In-cell fractions on `[0, 1]` per axis.
        frac: Vec3,
    },
    /// The trilinear corner blend, twinning
    /// [`trilinear_sample`](prism_render_architecture::particle::fluid::trilinear_sample).
    TrilinearSample {
        /// Eight corner velocities, in [`FluidQuery::TrilinearWeights`] order.
        corners: [Vec3; 8],
        /// Eight corner weights, in the same order.
        weights: [f32; 8],
    },
    /// The central-difference gradient, twinning
    /// [`central_gradient`](prism_render_architecture::particle::fluid::central_gradient).
    CentralGradient {
        /// Six axis-aligned scalar neighbor samples.
        neighbors: NeighborScalars,
        /// `1 / (2*h)` for cell size `h`.
        inv_2h: f32,
    },
    /// The pressure-gradient subtraction, twinning
    /// [`subtract_pressure_gradient`](prism_render_architecture::particle::fluid::subtract_pressure_gradient).
    SubtractPressureGradient {
        /// Pre-projection velocity.
        velocity: Vec3,
        /// Pressure gradient to remove.
        pressure_gradient: Vec3,
    },
    /// One combustion step, twinning
    /// [`step_combustion`](prism_render_architecture::particle::fluid::step_combustion).
    StepCombustion {
        /// Current combustion state.
        state: CombustionState,
        /// Combustion coupling parameters.
        params: CombustionParams,
        /// Time step.
        dt: f32,
    },
    /// The buoyancy body force, twinning
    /// [`buoyancy_force`](prism_render_architecture::particle::fluid::buoyancy_force).
    BuoyancyForce {
        /// Current combustion state.
        state: CombustionState,
        /// Combustion coupling parameters.
        params: CombustionParams,
    },
    /// The polynomial black-body colour, twinning
    /// [`blackbody_emission`](prism_render_architecture::particle::fluid::blackbody_emission).
    BlackbodyEmission {
        /// Voxel temperature.
        temperature: f32,
        /// Emission ramp parameters.
        params: EmissionParams,
    },
    /// The heat-haze refraction offset, twinning
    /// [`heat_haze_distortion`](prism_render_architecture::particle::fluid::heat_haze_distortion).
    HeatHazeDistortion {
        /// Local temperature gradient.
        temperature_gradient: Vec3,
        /// Distortion strength.
        strength: f32,
    },
    /// The finite-difference curl, twinning
    /// [`NeighborVelocities::curl`](prism_render_architecture::particle::fluid::NeighborVelocities::curl).
    Curl {
        /// Six axis-aligned velocity neighbor samples.
        neighbors: NeighborVelocities,
        /// `1 / (2*h)` for cell size `h`.
        inv_2h: f32,
    },
    /// The vorticity-confinement force, twinning
    /// [`vorticity_confinement_force`](prism_render_architecture::particle::fluid::vorticity_confinement_force).
    VorticityConfinementForce {
        /// Local curl (vorticity).
        curl: Vec3,
        /// Gradient of the curl magnitude.
        magnitude_gradient: Vec3,
        /// Confinement parameters.
        params: VorticityParams,
        /// Grid cell size.
        cell_size: f32,
    },
}

impl FluidQuery {
    /// Returns the `u32` tag the kernel branches on for this routine.
    #[must_use]
    const fn tag(&self) -> u32 {
        match self {
            FluidQuery::CflNumber { .. } => 0,
            FluidQuery::StableTimestep { .. } => 1,
            FluidQuery::SemiLagrangianBacktrace { .. } => 2,
            FluidQuery::MaccormackCorrected { .. } => 3,
            FluidQuery::AdvectParticle { .. } => 4,
            FluidQuery::TrilinearWeights { .. } => 5,
            FluidQuery::TrilinearSample { .. } => 6,
            FluidQuery::CentralGradient { .. } => 7,
            FluidQuery::SubtractPressureGradient { .. } => 8,
            FluidQuery::StepCombustion { .. } => 9,
            FluidQuery::BuoyancyForce { .. } => 10,
            FluidQuery::BlackbodyEmission { .. } => 11,
            FluidQuery::HeatHazeDistortion { .. } => 12,
            FluidQuery::Curl { .. } => 13,
            FluidQuery::VorticityConfinementForce { .. } => 14,
        }
    }
}

/// One resolved answer for a single query: a tagged union whose variant matches
/// the routine the corresponding [`FluidQuery`] selected.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FluidResult {
    /// A scalar result (`cfl_number` or `stable_timestep`).
    Scalar(f32),
    /// A `3`-vector result (advection, gradient, force, colour or sample).
    Vector(Vec3),
    /// The eight trilinear corner weights (`trilinear_weights`).
    Weights([f32; 8]),
    /// A combustion-state result (`step_combustion`).
    Combustion(CombustionState),
}

/// The `CPU` golden verdict for one query, dispatching to the reference entry
/// points so callers (and the parity test) can pin the twin lane for lane.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`；无第三方引擎源码或衍生代码。
#[must_use]
pub fn cpu_reference(query: &FluidQuery) -> FluidResult {
    match query {
        FluidQuery::CflNumber {
            max_velocity,
            dt,
            cell_size,
        } => FluidResult::Scalar(cfl_number(*max_velocity, *dt, *cell_size)),
        FluidQuery::StableTimestep {
            max_velocity,
            cell_size,
            cfl_target,
        } => FluidResult::Scalar(stable_timestep(*max_velocity, *cell_size, *cfl_target)),
        FluidQuery::SemiLagrangianBacktrace { pos, velocity, dt } => {
            FluidResult::Vector(semi_lagrangian_backtrace(*pos, *velocity, *dt))
        }
        FluidQuery::MaccormackCorrected {
            forward,
            original,
            back_advected,
        } => FluidResult::Vector(maccormack_corrected(*forward, *original, *back_advected)),
        FluidQuery::AdvectParticle { pos, velocity, dt } => {
            FluidResult::Vector(advect_particle(*pos, *velocity, *dt))
        }
        FluidQuery::TrilinearWeights { frac } => FluidResult::Weights(trilinear_weights(*frac)),
        FluidQuery::TrilinearSample { corners, weights } => {
            FluidResult::Vector(trilinear_sample(*corners, *weights))
        }
        FluidQuery::CentralGradient { neighbors, inv_2h } => {
            FluidResult::Vector(central_gradient(*neighbors, *inv_2h))
        }
        FluidQuery::SubtractPressureGradient {
            velocity,
            pressure_gradient,
        } => FluidResult::Vector(subtract_pressure_gradient(*velocity, *pressure_gradient)),
        FluidQuery::StepCombustion { state, params, dt } => {
            FluidResult::Combustion(step_combustion(*state, *params, *dt))
        }
        FluidQuery::BuoyancyForce { state, params } => {
            FluidResult::Vector(buoyancy_force(*state, *params))
        }
        FluidQuery::BlackbodyEmission {
            temperature,
            params,
        } => FluidResult::Vector(blackbody_emission(*temperature, *params)),
        FluidQuery::HeatHazeDistortion {
            temperature_gradient,
            strength,
        } => FluidResult::Vector(heat_haze_distortion(*temperature_gradient, *strength)),
        FluidQuery::Curl { neighbors, inv_2h } => FluidResult::Vector(neighbors.curl(*inv_2h)),
        FluidQuery::VorticityConfinementForce {
            curl,
            magnitude_gradient,
            params,
            cell_size,
        } => FluidResult::Vector(vorticity_confinement_force(
            *curl,
            *magnitude_gradient,
            *params,
            *cell_size,
        )),
    }
}

/// Packs a [`Vec3`] and a trailing scalar into a padded `vec4` lane.
fn v4(v: Vec3, w: f32) -> [f32; 4] {
    [v.x, v.y, v.z, w]
}

/// Encodes one [`FluidQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &FluidQuery) -> GpuQuery {
    let mut g = GpuQuery::zeroed();
    g.tag = q.tag();
    match q {
        FluidQuery::CflNumber {
            max_velocity,
            dt,
            cell_size,
        } => {
            g.scalar_a = [*max_velocity, *dt, *cell_size, 0.0];
        }
        FluidQuery::StableTimestep {
            max_velocity,
            cell_size,
            cfl_target,
        } => {
            g.scalar_a = [*max_velocity, *cell_size, *cfl_target, 0.0];
        }
        FluidQuery::SemiLagrangianBacktrace { pos, velocity, dt }
        | FluidQuery::AdvectParticle { pos, velocity, dt } => {
            g.vec_a = v4(*pos, 0.0);
            g.vec_b = v4(*velocity, 0.0);
            g.scalar_a = [*dt, 0.0, 0.0, 0.0];
        }
        FluidQuery::MaccormackCorrected {
            forward,
            original,
            back_advected,
        } => {
            g.vec_a = v4(*forward, 0.0);
            g.vec_b = v4(*original, 0.0);
            g.vec_c = v4(*back_advected, 0.0);
        }
        FluidQuery::TrilinearWeights { frac } => {
            g.vec_a = v4(*frac, 0.0);
        }
        FluidQuery::TrilinearSample { corners, weights } => {
            g.c0 = v4(corners[0], 0.0);
            g.c1 = v4(corners[1], 0.0);
            g.c2 = v4(corners[2], 0.0);
            g.c3 = v4(corners[3], 0.0);
            g.c4 = v4(corners[4], 0.0);
            g.c5 = v4(corners[5], 0.0);
            g.c6 = v4(corners[6], 0.0);
            g.c7 = v4(corners[7], 0.0);
            g.w0 = [weights[0], weights[1], weights[2], weights[3]];
            g.w1 = [weights[4], weights[5], weights[6], weights[7]];
        }
        FluidQuery::CentralGradient { neighbors, inv_2h } => {
            g.ns0 = [
                neighbors.x_plus,
                neighbors.x_minus,
                neighbors.y_plus,
                neighbors.y_minus,
            ];
            g.ns1 = [neighbors.z_plus, neighbors.z_minus, 0.0, 0.0];
            g.scalar_a = [*inv_2h, 0.0, 0.0, 0.0];
        }
        FluidQuery::SubtractPressureGradient {
            velocity,
            pressure_gradient,
        } => {
            g.vec_a = v4(*velocity, 0.0);
            g.vec_b = v4(*pressure_gradient, 0.0);
        }
        FluidQuery::StepCombustion { state, params, dt } => {
            g.cstate = [state.temperature, state.fuel, state.smoke, 0.0];
            g.cparams0 = [
                params.ignition_temperature,
                params.burn_rate,
                params.smoke_yield,
                params.heat_yield,
            ];
            g.cparams1 = [
                params.cooling_rate,
                params.ambient_temperature,
                params.buoyancy_alpha,
                params.buoyancy_beta,
            ];
            g.scalar_a = [*dt, 0.0, 0.0, 0.0];
        }
        FluidQuery::BuoyancyForce { state, params } => {
            g.cstate = [state.temperature, state.fuel, state.smoke, 0.0];
            g.cparams1 = [
                params.cooling_rate,
                params.ambient_temperature,
                params.buoyancy_alpha,
                params.buoyancy_beta,
            ];
        }
        FluidQuery::BlackbodyEmission {
            temperature,
            params,
        } => {
            g.scalar_a = [
                *temperature,
                params.low_temperature,
                params.white_temperature,
                params.intensity_scale,
            ];
        }
        FluidQuery::HeatHazeDistortion {
            temperature_gradient,
            strength,
        } => {
            g.vec_a = v4(*temperature_gradient, 0.0);
            g.scalar_a = [*strength, 0.0, 0.0, 0.0];
        }
        FluidQuery::Curl { neighbors, inv_2h } => {
            g.c0 = v4(neighbors.x_plus, 0.0);
            g.c1 = v4(neighbors.x_minus, 0.0);
            g.c2 = v4(neighbors.y_plus, 0.0);
            g.c3 = v4(neighbors.y_minus, 0.0);
            g.c4 = v4(neighbors.z_plus, 0.0);
            g.c5 = v4(neighbors.z_minus, 0.0);
            g.scalar_a = [*inv_2h, 0.0, 0.0, 0.0];
        }
        FluidQuery::VorticityConfinementForce {
            curl,
            magnitude_gradient,
            params,
            cell_size,
        } => {
            g.vec_a = v4(*curl, 0.0);
            g.vec_b = v4(*magnitude_gradient, 0.0);
            g.scalar_a = [params.epsilon, *cell_size, 0.0, 0.0];
        }
    }
    g
}

/// Decodes one packed [`GpuResult`] into the public [`FluidResult`], selecting
/// the variant from the query's routine.
fn decode_result(q: &FluidQuery, raw: &GpuResult) -> FluidResult {
    match q {
        FluidQuery::CflNumber { .. } | FluidQuery::StableTimestep { .. } => {
            FluidResult::Scalar(raw.scalar[0])
        }
        FluidQuery::SemiLagrangianBacktrace { .. }
        | FluidQuery::MaccormackCorrected { .. }
        | FluidQuery::AdvectParticle { .. }
        | FluidQuery::TrilinearSample { .. }
        | FluidQuery::CentralGradient { .. }
        | FluidQuery::SubtractPressureGradient { .. }
        | FluidQuery::BuoyancyForce { .. }
        | FluidQuery::BlackbodyEmission { .. }
        | FluidQuery::HeatHazeDistortion { .. }
        | FluidQuery::Curl { .. }
        | FluidQuery::VorticityConfinementForce { .. } => {
            FluidResult::Vector(Vec3::new(raw.vector[0], raw.vector[1], raw.vector[2]))
        }
        FluidQuery::TrilinearWeights { .. } => FluidResult::Weights([
            raw.weights0[0],
            raw.weights0[1],
            raw.weights0[2],
            raw.weights0[3],
            raw.weights1[0],
            raw.weights1[1],
            raw.weights1[2],
            raw.weights1[3],
        ]),
        FluidQuery::StepCombustion { .. } => FluidResult::Combustion(CombustionState::new(
            raw.combustion[0],
            raw.combustion[1],
            raw.combustion[2],
        )),
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

/// A compiled, reusable fluid compute pipeline, twinning the `CPU` golden
/// [`fluid`](prism_render_architecture::particle::fluid).
pub struct GpuFluid {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFluid {
    /// Compiles the fluid kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFluid {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fluid"),
            source: ShaderSource::Wgsl(FLUID_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fluid_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fluid_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fluid_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFluid {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`FluidResult`] per input,
    /// in order.
    ///
    /// The result variant matches the routine each query selected, matching the
    /// reference to within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[FluidQuery]) -> Vec<FluidResult> {
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
            label: Some("prism_volumetric_fluid_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fluid_bind_group"),
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
            label: Some("prism_volumetric_fluid_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fluid_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fluid_pass"),
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
            .map(|(q, r)| decode_result(q, r))
            .collect()
    }
}
