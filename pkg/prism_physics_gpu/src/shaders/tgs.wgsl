// Temporal Gauss-Seidel (TGS) substep distance solver, device kernel.
//
// Twin of the CPU reference in `src/xpbd/tgs.rs`. Each substep integrates
// velocity, runs the biased velocity sweeps per colour, integrates position,
// then runs the bias-free relaxation sweeps per colour. The stages below
// perform the identical f32 arithmetic in the identical order as the CPU
// golden; only device floating-point reassociation (fused multiply-add,
// differing division/sqrt rounding) separates the two, which the parity test
// bounds with a tight tolerance rather than exact equality.
//
// Each solve dispatch runs one colour at a time (see `src/xpbd/coloring.rs`) so
// no two live threads touch the same particle's velocity: the velocity
// read-modify-write and the per-constraint impulse update are therefore
// race-free without atomics.
//
// Provenance: Temporal Gauss-Seidel substepping with soft constraints (PhysX 5
// / Chaos lineage; Catto soft-constraint coefficients). No Unreal Engine source
// or derived code.

struct Params {
    gravity: vec3<f32>,
    h: f32,
    damping_scale: f32,
    bias_rate: f32,
    mass_scale: f32,
    impulse_scale: f32,
    particle_count: u32,
    constraint_count: u32,
    _pad0: u32,
    _pad1: u32,
};

struct ColourParams {
    start: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
};

struct Constraint {
    a: u32,
    b: u32,
    rest_length: f32,
    compliance: f32,
};

const EPSILON: f32 = 1.1920929e-7; // f32::EPSILON, matching the CPU guard.

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(4) var<storage, read> constraints: array<Constraint>;
@group(0) @binding(5) var<storage, read_write> impulses: array<f32>;
// Per-particle integration gate: 1 = integrate this particle this step, 0 =
// frozen (asleep). The dense solver uploads an all-ones mask, which makes the
// extra branch a no-op; the island-aware stepper uploads a real mask.
@group(0) @binding(6) var<storage, read> awake: array<u32>;

@group(1) @binding(0) var<uniform> colour: ColourParams;

// Stage 1: integrate velocity under gravity and linear damping.
@compute @workgroup_size(64)
fn integrate_velocities(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    let w = inverse_masses[i];
    if (w <= 0.0 || awake[i] == 0u) {
        return;
    }
    var v = velocities[i].xyz;
    v = v + params.gravity * params.h;
    v = v * params.damping_scale;
    velocities[i] = vec4<f32>(v, 0.0);
}

// Stage 3: advance positions with the corrected velocity.
@compute @workgroup_size(64)
fn integrate_positions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    let w = inverse_masses[i];
    if (w <= 0.0 || awake[i] == 0u) {
        return;
    }
    positions[i] = vec4<f32>(positions[i].xyz + velocities[i].xyz * params.h, 0.0);
}

// Clears the per-constraint accumulated impulse before a sweep group so the
// soft `impulse_scale` decay acts across that group's iterations only.
@compute @workgroup_size(64)
fn reset_impulses(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.constraint_count) {
        return;
    }
    impulses[i] = 0.0;
}

// Shared soft velocity correction for one constraint `gi`.
fn solve_one(gi: u32, bias_rate: f32, mass_scale: f32, impulse_scale: f32) {
    let con = constraints[gi];
    let wa = inverse_masses[con.a];
    let wb = inverse_masses[con.b];
    let w_sum = wa + wb;
    if (w_sum <= 0.0) {
        return;
    }
    let delta = positions[con.a].xyz - positions[con.b].xyz;
    let length = sqrt(dot(delta, delta));
    if (length < EPSILON) {
        return;
    }
    let normal = delta / length;
    let c = length - con.rest_length;
    let eff_mass = 1.0 / w_sum;
    let v_rel = dot(velocities[con.a].xyz - velocities[con.b].xyz, normal);
    let bias = bias_rate * c;
    let delta_impulse = -eff_mass * mass_scale * (v_rel + bias) - impulse_scale * impulses[gi];
    impulses[gi] = impulses[gi] + delta_impulse;
    let correction = normal * delta_impulse;
    velocities[con.a] = vec4<f32>(velocities[con.a].xyz + correction * wa, 0.0);
    velocities[con.b] = vec4<f32>(velocities[con.b].xyz - correction * wb, 0.0);
}

// Stage 2: biased velocity solve for one colour, using the soft coefficients
// from the global params.
@compute @workgroup_size(64)
fn solve_biased(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= colour.count) {
        return;
    }
    let gi = colour.start + local;
    solve_one(gi, params.bias_rate, params.mass_scale, params.impulse_scale);
}

// Stage 4: bias-free relaxation solve for one colour (no bias, full mass, no
// impulse decay), removing the bias velocity the biased pass injected.
@compute @workgroup_size(64)
fn solve_relax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= colour.count) {
        return;
    }
    let gi = colour.start + local;
    solve_one(gi, 0.0, 1.0, 0.0);
}
