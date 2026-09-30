// Colour-ordered substep XPBD one-sided contact solver, device kernel.
//
// Byte-for-byte-intent twin of the CPU reference in `src/contacts/cpu.rs`. The
// predict / reset / project / finalize stages below perform the identical f32
// arithmetic in the identical order; only floating-point reassociation on the
// device (fused multiply-add, differing division/sqrt rounding) separates the
// two, which the parity test bounds with a tight tolerance rather than exact
// equality.
//
// The projection dispatch runs one colour at a time (see the shared colouring
// in `src/xpbd/coloring.rs`) so no two live threads touch the same particle: the
// position read-modify-write is therefore race-free without atomics.
//
// The normal projection differs from the distance kernel in exactly two places —
// a separated pair (c >= 0) is skipped, and the accumulated multiplier is
// clamped to be non-negative so a contact can push apart but never pull
// together. Those two lines are the whole of the one-sided normal model.
//
// After the normal correction each contact also applies positional Coulomb
// friction: the tangential drift accumulated this substep (from prev_positions)
// is either fully cancelled (static, inside the cone mu_s * penetration) or
// clamped to the dynamic cone mu_d * penetration. A frictionless contact
// (both coefficients 0) short-circuits, leaving the trajectory unchanged.
//
// A final velocity-level restitution pass (Müller et al. 2020) runs once per
// substep after finalize: it restores a fraction of the pre-solve approach
// speed along the contact normal so a struck pair rebounds instead of sticking.
// It reads the post-prediction velocity snapshot (vel_pre) written by predict.
//
// Provenance: substep XPBD (Müller et al.) with the canonical one-sided contact
// constraint, the positional Coulomb friction of Müller et al. 2020 bounded by
// penetration depth after Macklin et al. 2014, and the substep velocity-level
// restitution of that same 2020 work. No Unreal Engine source or derived code.

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

struct Contact {
    a: u32,
    b: u32,
    rest: f32,
    compliance: f32,
    static_friction: f32,
    dynamic_friction: f32,
    restitution: f32,
};

const EPSILON: f32 = 1.1920929e-7; // f32::EPSILON, matching the CPU guard.

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> prev_positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(5) var<storage, read> contacts: array<Contact>;
@group(0) @binding(6) var<storage, read_write> lambdas: array<f32>;
@group(0) @binding(7) var<storage, read_write> vel_pre: array<vec4<f32>>;

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
        // Pinned: snapshot the (unchanged) velocity so the restitution pass sees
        // the same pre-solve value the CPU twin's clone captures.
        vel_pre[i] = velocities[i];
        return;
    }
    var v = velocities[i].xyz;
    v = v + params.gravity * params.h;
    v = v * params.damping_scale;
    velocities[i] = vec4<f32>(v, 0.0);
    // Snapshot the post-gravity velocity for the restitution pass.
    vel_pre[i] = velocities[i];
    positions[i] = vec4<f32>(positions[i].xyz + v * params.h, 0.0);
}

// Stage 2: clear the per-contact Lagrange multipliers for a fresh substep.
@compute @workgroup_size(64)
fn reset_lambdas(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.constraint_count) {
        return;
    }
    lambdas[i] = 0.0;
}

// Stage 3: project one colour's contacts. `colour.start` indexes the reordered
// contact and lambda arrays; each thread owns one contact.
@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= colour.count) {
        return;
    }
    let gi = colour.start + local;
    let con = contacts[gi];
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
    let c = length - con.rest;
    if (c >= 0.0) {
        return;
    }
    let normal = delta / length;
    let alpha_tilde = con.compliance / (params.h * params.h);
    let delta_lambda = (-c - alpha_tilde * lambdas[gi]) / (w_sum + alpha_tilde);
    let new_lambda = max(lambdas[gi] + delta_lambda, 0.0);
    let applied = new_lambda - lambdas[gi];
    lambdas[gi] = new_lambda;
    let correction = normal * applied;
    positions[con.a] = vec4<f32>(positions[con.a].xyz + correction * wa, 0.0);
    positions[con.b] = vec4<f32>(positions[con.b].xyz - correction * wb, 0.0);

    // Positional Coulomb friction. `penetration = -c > 0` (the pair overlaps);
    // reuse the pre-correction `normal` (b toward a) to strip the normal
    // component from the substep drift so the correction is purely tangential.
    if (con.static_friction <= 0.0 && con.dynamic_friction <= 0.0) {
        return;
    }
    let penetration = -c;
    let da = positions[con.a].xyz - prev_positions[con.a].xyz;
    let db = positions[con.b].xyz - prev_positions[con.b].xyz;
    let relative = da - db;
    let normal_amount = dot(relative, normal);
    let tangent = relative - normal * normal_amount;
    let tangent_len = sqrt(dot(tangent, tangent));
    if (tangent_len < EPSILON) {
        return;
    }
    var scale = 1.0;
    if (tangent_len >= con.static_friction * penetration) {
        scale = min(con.dynamic_friction * penetration / tangent_len, 1.0);
    }
    let friction = tangent * scale;
    positions[con.a] = vec4<f32>(positions[con.a].xyz - friction * (wa / w_sum), 0.0);
    positions[con.b] = vec4<f32>(positions[con.b].xyz + friction * (wb / w_sum), 0.0);
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

// Stage 5: velocity-level restitution for one colour's contacts. Runs after
// finalize; restores a fraction of the pre-solve approach speed along the
// contact normal so a struck pair rebounds. Colour-ordered like project so no
// two threads write the same particle's velocity at once. `colour.start`
// indexes the reordered contact array; each thread owns one contact.
@compute @workgroup_size(64)
fn restitution(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= colour.count) {
        return;
    }
    let gi = colour.start + local;
    let con = contacts[gi];
    if (con.restitution <= 0.0) {
        return;
    }
    // Activity gate on the substep-start snapshot: a pair separated at the
    // start was never a live contact this substep and gets no impulse.
    let delta_pre = prev_positions[con.a].xyz - prev_positions[con.b].xyz;
    let length_pre = sqrt(dot(delta_pre, delta_pre));
    let c_pre = length_pre - con.rest;
    if (c_pre >= 0.0) {
        return;
    }
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
    let relative = velocities[con.a].xyz - velocities[con.b].xyz;
    let v_n = dot(relative, normal);
    let relative_pre = vel_pre[con.a].xyz - vel_pre[con.b].xyz;
    let v_n_pre = dot(relative_pre, normal);
    let target_vn = max(-con.restitution * v_n_pre, 0.0);
    let dv = target_vn - v_n;
    velocities[con.a] = vec4<f32>(velocities[con.a].xyz + normal * dv * (wa / w_sum), 0.0);
    velocities[con.b] = vec4<f32>(velocities[con.b].xyz - normal * dv * (wb / w_sum), 0.0);
}
