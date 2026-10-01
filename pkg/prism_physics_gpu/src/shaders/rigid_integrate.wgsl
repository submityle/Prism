// Explicit-gyroscopic 6-DOF rigid-body integrator, device kernel.
//
// Byte-for-byte-intent twin of the CPU reference in `src/rigid/cpu.rs`. Each
// invocation integrates one body through every substep internally (bodies are
// independent, so there is no cross-thread coupling and no atomics). The
// linear and angular updates below perform the identical f32 arithmetic in the
// identical order as the CPU twin; only floating-point reassociation on the
// device (fused multiply-add, differing division and square-root rounding)
// separates the two, which the parity test bounds with a tight tolerance
// rather than exact equality.
//
// The orientation dynamics are written as scalar operations over the quaternion
// components (x, y, z, w) so they mirror the CPU reference exactly rather than
// relying on any built-in quaternion type.
//
// Provenance: Euler's rigid-body equations with explicit gyroscopic coupling
// and the quaternion kinematic equation (Baraff & Witkin; standard rigid-body
// dynamics). No Unreal Engine source or derived code.

const EPSILON: f32 = 1.1920929e-7; // f32::EPSILON, matching the CPU guard.

struct Params {
    gravity: vec3<f32>,
    h: f32,
    linear_damping_scale: f32,
    angular_damping_scale: f32,
    substeps: u32,
    body_count: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> orientations: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> linear_velocities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> angular_velocities: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(6) var<storage, read> inverse_inertias: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> forces: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> torques: array<vec4<f32>>;

// Hamilton product a * b of two quaternions stored as (x, y, z, w).
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let ax = a.x; let ay = a.y; let az = a.z; let aw = a.w;
    let bx = b.x; let by = b.y; let bz = b.z; let bw = b.w;
    return vec4<f32>(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    );
}

// Conjugate (inverse rotation) of a unit quaternion stored as (x, y, z, w).
fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

// Rotates v by the unit quaternion q via the expanded sandwich product.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = vec3<f32>(q.x, q.y, q.z);
    let s = q.w;
    return u * (2.0 * dot(u, v)) + v * (s * s - dot(u, u)) + cross(u, v) * (2.0 * s);
}

// Normalises a quaternion, falling back to identity below EPSILON length.
fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    let length_squared = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    let length = sqrt(length_squared);
    if (length > EPSILON) {
        return q / length;
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

// Recovers the principal inertia diagonal from its inverse; a zero (locked)
// axis maps to zero inertia and contributes no gyroscopic coupling.
fn principal_inertia(inverse_inertia: vec3<f32>) -> vec3<f32> {
    var inertia = vec3<f32>(0.0, 0.0, 0.0);
    if (inverse_inertia.x > 0.0) { inertia.x = 1.0 / inverse_inertia.x; }
    if (inverse_inertia.y > 0.0) { inertia.y = 1.0 / inverse_inertia.y; }
    if (inverse_inertia.z > 0.0) { inertia.z = 1.0 / inverse_inertia.z; }
    return inertia;
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.body_count) {
        return;
    }

    let h = params.h;
    let half_h = 0.5 * h;
    let force = forces[i].xyz;
    let torque = torques[i].xyz;
    let inverse_mass = inverse_masses[i];
    let inverse_inertia = inverse_inertias[i].xyz;
    let can_translate = inverse_mass > 0.0;
    let can_rotate = inverse_inertia.x > 0.0 || inverse_inertia.y > 0.0 || inverse_inertia.z > 0.0;
    let inertia = principal_inertia(inverse_inertia);

    var position = positions[i].xyz;
    var linear_velocity = linear_velocities[i].xyz;
    var q = orientations[i];
    var angular_velocity = angular_velocities[i].xyz;

    for (var step: u32 = 0u; step < params.substeps; step = step + 1u) {
        if (can_translate) {
            let acceleration = params.gravity + force * inverse_mass;
            linear_velocity = (linear_velocity + acceleration * h) * params.linear_damping_scale;
            position = position + linear_velocity * h;
        }
        if (can_rotate) {
            let conjugate = quat_conj(q);
            let torque_body = quat_rotate(conjugate, torque);
            let angular_acceleration_body = inverse_inertia * torque_body;
            var omega_body = quat_rotate(conjugate, angular_velocity);
            omega_body = omega_body + angular_acceleration_body * h;
            let angular_momentum_body = inertia * omega_body;
            let gyroscopic = cross(omega_body, angular_momentum_body);
            omega_body = omega_body - inverse_inertia * gyroscopic * h;
            angular_velocity = quat_rotate(q, omega_body) * params.angular_damping_scale;
            let omega_quat = vec4<f32>(angular_velocity.x, angular_velocity.y, angular_velocity.z, 0.0);
            let dq = quat_mul(omega_quat, q);
            let integrated = q + dq * half_h;
            q = quat_normalize(integrated);
        }
    }

    positions[i] = vec4<f32>(position, 0.0);
    linear_velocities[i] = vec4<f32>(linear_velocity, 0.0);
    orientations[i] = q;
    angular_velocities[i] = vec4<f32>(angular_velocity, 0.0);
}
