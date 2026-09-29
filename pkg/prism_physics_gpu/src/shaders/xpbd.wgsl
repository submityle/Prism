// Colour-ordered substep XPBD distance solver, device kernel.
//
// Byte-for-byte-intent twin of the CPU reference in `src/xpbd/cpu.rs`. The
// predict / reset / project / finalize stages below perform the identical f32
// arithmetic in the identical order; only floating-point reassociation on the
// device (fused multiply-add, differing division/sqrt rounding) separates the
// two, which the parity test bounds with a tight tolerance rather than exact
// equality.
//
// The projection dispatch runs one colour at a time (see `src/xpbd/coloring.rs`)
// so no two live threads touch the same particle: the position read-modify-write
// is therefore race-free without atomics.
//
// Provenance: substep XPBD (Müller et al.) with the canonical stretch
// constraint. No Unreal Engine source or derived code.

struct Params {
    gravity: vec3<f32>,
    h: f32,
    damping_scale: f32,
    inv_h: f32,
    particle_count: u32,
    constraint_count: u32,
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
@group(0) @binding(3) var<storage, read_write> prev_positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(5) var<storage, read> constraints: array<Constraint>;
@group(0) @binding(6) var<storage, read_write> lambdas: array<f32>;

@group(1) @binding(0) var<uniform> colour: ColourParams;

// Stage 1: snapshot positions, integrate acceleration, damp velocity, advance.
@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    prev_positions[i] = positions[i];
    let w = inverse_masses[i];
    if (w <= 0.0) {
        return;
    }
    var v = velocities[i].xyz;
    v = v + params.gravity * params.h;
    v = v * params.damping_scale;
    velocities[i] = vec4<f32>(v, 0.0);
    positions[i] = vec4<f32>(positions[i].xyz + v * params.h, 0.0);
}

// Stage 2: clear the per-constraint Lagrange multipliers for a fresh substep.
@compute @workgroup_size(64)
fn reset_lambdas(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.constraint_count) {
        return;
    }
    lambdas[i] = 0.0;
}

// Stage 3: project one colour's constraints. `colour.start` indexes the
// reordered constraint and lambda arrays; each thread owns one constraint.
@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= colour.count) {
        return;
    }
    let gi = colour.start + local;
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
    let alpha_tilde = con.compliance / (params.h * params.h);
    let delta_lambda = (-c - alpha_tilde * lambdas[gi]) / (w_sum + alpha_tilde);
    lambdas[gi] = lambdas[gi] + delta_lambda;
    let correction = normal * delta_lambda;
    positions[con.a] = vec4<f32>(positions[con.a].xyz + correction * wa, 0.0);
    positions[con.b] = vec4<f32>(positions[con.b].xyz - correction * wb, 0.0);
}

// Stage 4: recover velocities from the net substep displacement.
@compute @workgroup_size(64)
fn finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    let v = (positions[i].xyz - prev_positions[i].xyz) * params.inv_h;
    velocities[i] = vec4<f32>(v, 0.0);
}
