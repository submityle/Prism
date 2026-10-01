// 6-DOF rigid-body integrator, device kernel (explicit or implicit gyroscopic).
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
// The gyroscopic coupling term `omega x (I * omega)` of the angular update is
// integrated either explicitly (evaluate at the start-of-substep angular
// velocity and subtract) or implicitly (solve the backward-Euler coupling with
// Newton iteration). The implicit path reproduces `implicit_gyroscopic_body`
// and `solve_3x3` from `src/rigid/gyroscopic.rs` as the identical sequence of
// scalar multiplies and divides so the two stay in parity.
//
// Provenance: Euler's rigid-body equations with gyroscopic coupling and the
// quaternion kinematic equation (Baraff & Witkin; standard rigid-body
// dynamics). The implicit gyroscopic solve follows Bullet's
// `computeGyroscopicImpulseImplicit_Body` and PhysX's gyroscopic forces option.
// No Unreal Engine source or derived code.

const EPSILON: f32 = 1.1920929e-7; // f32::EPSILON, matching the CPU guard.

// Determinant magnitude below which the Newton Jacobian is treated as singular
// and the step is dropped. Matches `GYRO_DET_EPSILON` in `rigid/gyroscopic.rs`.
const GYRO_DET_EPSILON: f32 = 1.0e-20;

// Value of `Params.gyroscopic_mode` selecting the implicit Newton solve; any
// other value selects the explicit subtraction. Mirrors `GyroscopicMode` on
// the host (`Explicit = 0`, `Implicit = 1`).
const GYRO_MODE_IMPLICIT: u32 = 1u;

struct Params {
    gravity: vec3<f32>,
    h: f32,
    linear_damping_scale: f32,
    angular_damping_scale: f32,
    substeps: u32,
    body_count: u32,
    // Gyroscopic integration scheme: `GYRO_MODE_IMPLICIT` for the Newton solve,
    // anything else for the explicit subtraction.
    gyroscopic_mode: u32,
    // Newton iterations per substep when `gyroscopic_mode` is implicit; clamped
    // to at least one inside the solve. Ignored by the explicit scheme.
    gyroscopic_iterations: u32,
    // Pads the uniform struct to 48 bytes so its size is a multiple of the
    // 16-byte vec3 alignment; mirrored by the host `Params` padding fields.
    pad0: u32,
    pad1: u32,
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

// Solves the 3x3 system `m * x = b` for `x` using an explicit cofactor
// (adjugate over determinant) inverse, where `m` is stored row-major as
// `[m00, m01, m02, m10, m11, m12, m20, m21, m22]`. Returns the zero vector when
// the determinant magnitude is below `GYRO_DET_EPSILON`, dropping the step
// rather than producing a non-finite result. Byte-for-byte-intent twin of
// `solve_3x3` in `rigid/gyroscopic.rs`.
fn solve_3x3(m: array<f32, 9>, bx: f32, by: f32, bz: f32) -> vec3<f32> {
    let c0 = m[4] * m[8] - m[5] * m[7];
    let c1 = m[5] * m[6] - m[3] * m[8];
    let c2 = m[3] * m[7] - m[4] * m[6];
    let det = m[0] * c0 + m[1] * c1 + m[2] * c2;
    if (abs(det) < GYRO_DET_EPSILON) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let inv_det = 1.0 / det;
    let x = (c0 * bx + (m[2] * m[7] - m[1] * m[8]) * by + (m[1] * m[5] - m[2] * m[4]) * bz) * inv_det;
    let y = (c1 * bx + (m[0] * m[8] - m[2] * m[6]) * by + (m[2] * m[3] - m[0] * m[5]) * bz) * inv_det;
    let z = (c2 * bx + (m[1] * m[6] - m[0] * m[7]) * by + (m[0] * m[4] - m[1] * m[3]) * bz) * inv_det;
    return vec3<f32>(x, y, z);
}

// Solves the implicit (backward-Euler) gyroscopic update for the end-of-substep
// body-frame angular velocity `omega1` satisfying
// `I*omega1 + h*(omega1 x (I*omega1)) = I*omega_body` with Newton iteration.
// The caller must pass a strictly positive inertia on every axis so the
// Jacobian is non-singular. Byte-for-byte-intent twin of
// `implicit_gyroscopic_body` in `rigid/gyroscopic.rs`.
fn implicit_gyroscopic_body(omega_body: vec3<f32>, inertia: vec3<f32>, h: f32, iterations: u32) -> vec3<f32> {
    let ix = inertia.x;
    let iy = inertia.y;
    let iz = inertia.z;
    // Constant right-hand-side angular momentum L0 = I * omega0.
    let l0x = ix * omega_body.x;
    let l0y = iy * omega_body.y;
    let l0z = iz * omega_body.z;

    var omega = omega_body;
    var steps = iterations;
    if (steps < 1u) { steps = 1u; }
    var iter: u32 = 0u;
    loop {
        if (iter >= steps) { break; }
        let wx = omega.x;
        let wy = omega.y;
        let wz = omega.z;
        // Current angular momentum L = I * omega.
        let lx = ix * wx;
        let ly = iy * wy;
        let lz = iz * wz;
        // Residual f = I*omega - L0 + h*(omega x L).
        let cross_x = wy * lz - wz * ly;
        let cross_y = wz * lx - wx * lz;
        let cross_z = wx * ly - wy * lx;
        let fx = lx - l0x + h * cross_x;
        let fy = ly - l0y + h * cross_y;
        let fz = lz - l0z + h * cross_z;
        // Jacobian J = I_mat + h*(skew(omega)*I_mat - skew(L)), row-major.
        let a01 = -wz * iy + lz;
        let a02 = wy * iz - ly;
        let a10 = wz * ix - lz;
        let a12 = -wx * iz + lx;
        let a20 = -wy * ix + ly;
        let a21 = wx * iy - lx;
        let m = array<f32, 9>(
            ix, h * a01, h * a02,
            h * a10, iy, h * a12,
            h * a20, h * a21, iz,
        );
        // Solve J * delta = -f.
        let delta = solve_3x3(m, -fx, -fy, -fz);
        omega.x = omega.x + delta.x;
        omega.y = omega.y + delta.y;
        omega.z = omega.z + delta.z;
        iter = iter + 1u;
    }
    return omega;
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
    // The implicit solve needs a strictly positive inertia on every axis so its
    // Newton Jacobian is non-singular; a body with any locked axis falls back
    // to the explicit path. The same guard runs on the CPU twin.
    let use_implicit = params.gyroscopic_mode == GYRO_MODE_IMPLICIT
        && min(inertia.x, min(inertia.y, inertia.z)) > 0.0;

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
            if (use_implicit) {
                // Implicit (backward-Euler) gyroscopic coupling: solve for the
                // end-of-substep body-frame angular velocity.
                omega_body = implicit_gyroscopic_body(omega_body, inertia, h, params.gyroscopic_iterations);
            } else {
                // Explicit gyroscopic coupling: subtract omega x (I * omega),
                // expressed as an angular acceleration via the inverse inertia.
                let angular_momentum_body = inertia * omega_body;
                let gyroscopic = cross(omega_body, angular_momentum_body);
                omega_body = omega_body - inverse_inertia * gyroscopic * h;
            }
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
