// Revolute (hinge) velocity-motor rigid-body joint stepper, device kernel.
//
// Byte-for-byte-intent twin of the CPU golden in
// `src/rigid/joint/revolute_motor_cpu.rs`. It is a full stepper: the host
// unrolls the substep loop into a fixed dispatch sequence that snapshots the
// transforms, predicts the bodies forward under gravity and damping, resets the
// XPBD multipliers, projects each joint's axis-alignment then velocity-motor
// then point-to-point constraints in colour-batch order, then recovers the
// velocities from the net per-substep motion.
//
// Multiplier layout. Each joint owns three Lagrange multipliers in the shared
// `lambda` buffer: slot `3 * j` for the point-to-point positional weld, slot
// `3 * j + 1` for the axis alignment, and slot `3 * j + 2` for the velocity
// motor, matching the CPU golden's `lambda[3k ..= 3k + 2]` split. The buffer is
// `3 * joint_count` long and `reset_lambda` clears all of it each substep.
//
// Scheduling and race-freedom. The `snapshot`, `predict`, and `recover` passes
// are per-body global dispatches; `reset_lambda` is a per-multiplier dispatch;
// all are race-free. The `solve` pass is dispatched once per colour batch; the
// host colours the joint graph so same-batch joints write disjoint movable
// bodies, and static bodies may be shared because their corrections scale by
// zero. Each solve invocation projects the axis alignment first, then the
// velocity motor, then the point-to-point weld for its one joint, matching the
// CPU `solve_one` order.
//
// The motor is the bilateral compliant XPBD equality
// `C = u . (dphi_b - dphi_a) - target_velocity * h`, where `dphi` is each body's
// angular displacement about the world hinge axis `u` since the substep
// snapshot. It reads the shared per-substep snapshot buffer `prev_orientations`
// to form the relative angular displacement about `u`, exactly as the CPU twin's
// `angular_displacement` does. There is no zero-angle reference and no absolute
// hinge angle, so the motor carries no `ref_a` / `ref_b` directions and no
// transcendental.
//
// The arithmetic below performs the identical f32 operations in the identical
// order as the CPU twin; only device floating-point reassociation separates the
// two, which the parity test bounds with a tight tolerance rather than exact
// equality.
//
// Provenance: the point-to-point (ball-socket) constraint and the hinge
// axis-alignment constraint (Müller et al., "Detailed Rigid Body Simulation
// with XPBD"), with the velocity-level motor expressed as a per-substep
// compliant equality on the relative angular displacement about the hinge axis
// (Macklin et al., "XPBD: Position-Based Simulation of Compliant Constrained
// Dynamics"), over the world-space inverse inertia and quaternion kinematics of
// Baraff & Witkin. Standard wgpu compute dispatch. No Unreal Engine source or
// derived code.

const EPSILON: f32 = 1.1920929e-7; // f32::EPSILON, matching the CPU guard.

struct Params {
    gravity: vec3<f32>,
    h: f32,
    inv_h: f32,
    linear_damping_scale: f32,
    angular_damping_scale: f32,
    joint_count: u32,
    body_count: u32,
    // Number of Lagrange multipliers reset each substep. For the revolute motor
    // joint this equals `3 * joint_count` (one point-to-point weld, one axis
    // alignment, and one velocity motor per joint).
    lambda_count: u32,
    _pad0: u32,
    _pad1: u32,
};

struct ColourParams {
    start: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
};

// Device-packed joint; layout matches `GpuRevoluteMotorJoint` in
// `src/rigid/joint/revolute_motor.rs` (96 bytes).
struct Joint {
    anchor_a: vec4<f32>,
    anchor_b: vec4<f32>,
    axis_a: vec4<f32>,
    axis_b: vec4<f32>,
    body_a: u32,
    body_b: u32,
    compliance: f32,
    angular_compliance: f32,
    motor_compliance: f32,
    target_velocity: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> linear_velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> angular_velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> orientations: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(6) var<storage, read> inverse_inertias: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> joints: array<Joint>;
@group(0) @binding(8) var<storage, read_write> lambda: array<f32>;
@group(0) @binding(9) var<storage, read_write> prev_positions: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> prev_orientations: array<vec4<f32>>;

@group(1) @binding(0) var<uniform> colour: ColourParams;

// Conjugate (inverse rotation) of a unit quaternion stored as (x, y, z, w).
fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

// Hamilton product a * b of two quaternions stored as (x, y, z, w), matching
// the CPU twin's array-based `quat_mul`.
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

// Rotates v by the unit quaternion q via the expanded sandwich product
// 2 (u . v) u + (s^2 - u . u) v + 2 s (u x v) with u = q.xyz, s = q.w.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = vec3<f32>(q.x, q.y, q.z);
    let s = q.w;
    return u * (2.0 * dot(u, v)) + v * (s * s - dot(u, u)) + cross(u, v) * (2.0 * s);
}

// Applies the world-space inverse inertia R diag(inv_i) R^T to v.
fn world_inv_inertia_apply(q: vec4<f32>, inv_i: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let body = quat_rotate(quat_conj(q), v);
    return quat_rotate(q, inv_i * body);
}

// Whether the solver may rotate a body with this body-frame inverse inertia.
fn can_rotate(inv_i: vec3<f32>) -> bool {
    return inv_i.x > 0.0 || inv_i.y > 0.0 || inv_i.z > 0.0;
}

// Normalises a quaternion, returning `fallback` when its length is degenerate.
fn normalize_or(q: vec4<f32>, fallback: vec4<f32>) -> vec4<f32> {
    let length = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (length > EPSILON) {
        return q / length;
    }
    return fallback;
}

// Advances an orientation by its world-space angular velocity over h via the
// quaternion kinematic equation q' = normalize(q + 0.5 h [omega, 0] q).
fn integrate_orientation(q: vec4<f32>, omega: vec3<f32>, h: f32) -> vec4<f32> {
    let dq = quat_mul(vec4<f32>(omega, 0.0), q);
    let integrated = q + dq * (0.5 * h);
    return normalize_or(integrated, q);
}

// Applies a direct XPBD rotation delta omega (not scaled by any time step):
// q' = normalize(q + 0.5 [omega, 0] q).
fn apply_rotation_delta(q: vec4<f32>, omega: vec3<f32>) -> vec4<f32> {
    let dq = quat_mul(vec4<f32>(omega, 0.0), q);
    let integrated = q + dq * 0.5;
    return normalize_or(integrated, q);
}

// Relative rotation of a body since the substep snapshot, as a rotation vector:
// twice the imaginary part of the delta quaternion orientation *
// conj(prev_orientation), hemisphere-corrected so the shortest arc is taken.
// Mirrors the CPU twin's `angular_displacement`.
fn angular_displacement(orientation: vec4<f32>, prev_orientation: vec4<f32>) -> vec3<f32> {
    let delta = quat_mul(orientation, quat_conj(prev_orientation));
    var rotvec = vec3<f32>(delta.x, delta.y, delta.z) * 2.0;
    if (delta.w < 0.0) {
        rotvec = -rotvec;
    }
    return rotvec;
}

// Copies the current transforms into the per-substep snapshot the velocity
// recovery differences against.
@compute @workgroup_size(64)
fn snapshot(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.body_count) {
        return;
    }
    prev_positions[i] = positions[i];
    prev_orientations[i] = orientations[i];
}

// Predicts every body forward one substep: semi-implicit Euler with linear
// damping for translation, the quaternion kinematic equation with angular
// damping for rotation.
@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.body_count) {
        return;
    }
    if (inverse_masses[i] > 0.0) {
        let velocity = (linear_velocities[i].xyz + params.gravity * params.h)
            * params.linear_damping_scale;
        linear_velocities[i] = vec4<f32>(velocity, 0.0);
        positions[i] = vec4<f32>(positions[i].xyz + velocity * params.h, 0.0);
    }
    let omega = angular_velocities[i].xyz * params.angular_damping_scale;
    angular_velocities[i] = vec4<f32>(omega, 0.0);
    let inv_i = inverse_inertias[i].xyz;
    if (can_rotate(inv_i)) {
        orientations[i] = integrate_orientation(orientations[i], omega, params.h);
    }
}

// Resets every joint's three XPBD Lagrange multipliers at the start of a
// substep.
@compute @workgroup_size(64)
fn reset_lambda(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= params.lambda_count) {
        return;
    }
    lambda[j] = 0.0;
}

// Projects one colour batch of revolute motor joints for a single sweep: axis
// alignment first (multiplier slot 3j + 1), then the velocity motor (multiplier
// slot 3j + 2), then the point-to-point weld (multiplier slot 3j), matching the
// CPU `solve_one` order.
@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= colour.count) {
        return;
    }
    let j = colour.start + local;

    let joint = joints[j];
    let a = joint.body_a;
    let b = joint.body_b;
    let ii_a = inverse_inertias[a].xyz;
    let ii_b = inverse_inertias[b].xyz;

    // --- Axis-alignment angular constraint (multiplier slot 3j + 1) ---
    {
        let q_a = orientations[a];
        let q_b = orientations[b];
        let axis_a = quat_rotate(q_a, joint.axis_a.xyz);
        let axis_b = quat_rotate(q_b, joint.axis_b.xyz);
        let len_a = length(axis_a);
        let len_b = length(axis_b);
        if (len_a >= EPSILON && len_b >= EPSILON) {
            let u_a = axis_a / len_a;
            let u_b = axis_b / len_b;
            let delta = cross(u_a, u_b);
            let theta = length(delta);
            if (theta >= EPSILON) {
                let n = delta / theta;
                let w_a = dot(n, world_inv_inertia_apply(q_a, ii_a, n));
                let w_b = dot(n, world_inv_inertia_apply(q_b, ii_b, n));
                let w = w_a + w_b;
                if (w >= EPSILON) {
                    let slot = 3u * j + 1u;
                    let alpha_tilde = joint.angular_compliance / (params.h * params.h);
                    let d_lambda = (-theta - alpha_tilde * lambda[slot]) / (w + alpha_tilde);
                    lambda[slot] = lambda[slot] + d_lambda;
                    let p = n * d_lambda;
                    orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
                    orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
                }
            }
        }
    }

    // --- Velocity-motor constraint (multiplier slot 3j + 2) ---
    // The relative angular rate about the world hinge axis u is servoed onto
    // target_velocity with a bilateral compliant XPBD equality on the relative
    // angular displacement since the substep snapshot:
    // C = u . (dphi_b - dphi_a) - target_velocity * h. motor_compliance softens
    // the rigid rate-lock into a finite-torque motor; there is no reference
    // direction and no absolute hinge angle.
    {
        let q_a = orientations[a];
        let q_b = orientations[b];
        let axis = quat_rotate(q_a, joint.axis_a.xyz);
        let axis_len = length(axis);
        if (axis_len >= EPSILON) {
            let u = axis / axis_len;
            let w_a = dot(u, world_inv_inertia_apply(q_a, ii_a, u));
            let w_b = dot(u, world_inv_inertia_apply(q_b, ii_b, u));
            let w = w_a + w_b;
            if (w >= EPSILON) {
                let slot = 3u * j + 2u;
                let angvec_a = angular_displacement(q_a, prev_orientations[a]);
                let angvec_b = angular_displacement(q_b, prev_orientations[b]);
                let dv = dot(u, angvec_b - angvec_a);
                let target_disp = joint.target_velocity * params.h;
                let c = dv - target_disp;
                let alpha_tilde = joint.motor_compliance / (params.h * params.h);
                let d_lambda = (-c - alpha_tilde * lambda[slot]) / (w + alpha_tilde);
                lambda[slot] = lambda[slot] + d_lambda;
                let p = u * d_lambda;
                orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
                orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
            }
        }
    }

    // --- Point-to-point positional constraint (multiplier slot 3j) ---
    {
        let q_a = orientations[a];
        let q_b = orientations[b];
        let r_a = quat_rotate(q_a, joint.anchor_a.xyz);
        let r_b = quat_rotate(q_b, joint.anchor_b.xyz);
        let dx = (positions[a].xyz + r_a) - (positions[b].xyz + r_b);
        let c = length(dx);
        if (c >= EPSILON) {
            let n = dx / c;
            let inv_m_a = inverse_masses[a];
            let inv_m_b = inverse_masses[b];
            let rn_a = cross(r_a, n);
            let rn_b = cross(r_b, n);
            let w_a = inv_m_a + dot(rn_a, world_inv_inertia_apply(q_a, ii_a, rn_a));
            let w_b = inv_m_b + dot(rn_b, world_inv_inertia_apply(q_b, ii_b, rn_b));
            let w = w_a + w_b;
            if (w >= EPSILON) {
                let slot = 3u * j;
                let alpha_tilde = joint.compliance / (params.h * params.h);
                let d_lambda = (-c - alpha_tilde * lambda[slot]) / (w + alpha_tilde);
                lambda[slot] = lambda[slot] + d_lambda;
                let p = n * d_lambda;
                positions[a] = vec4<f32>(positions[a].xyz + p * inv_m_a, 0.0);
                positions[b] = vec4<f32>(positions[b].xyz - p * inv_m_b, 0.0);
                let dw_a = world_inv_inertia_apply(q_a, ii_a, cross(r_a, p));
                orientations[a] = apply_rotation_delta(q_a, dw_a);
                let dw_b = world_inv_inertia_apply(q_b, ii_b, cross(r_b, p));
                orientations[b] = apply_rotation_delta(q_b, -dw_b);
            }
        }
    }
}

// Recovers each body's linear and angular velocity from the net substep motion.
@compute @workgroup_size(64)
fn recover(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.body_count) {
        return;
    }
    if (inverse_masses[i] > 0.0) {
        let velocity = (positions[i].xyz - prev_positions[i].xyz) * params.inv_h;
        linear_velocities[i] = vec4<f32>(velocity, 0.0);
    }
    let inv_i = inverse_inertias[i].xyz;
    if (can_rotate(inv_i)) {
        let delta = quat_mul(orientations[i], quat_conj(prev_orientations[i]));
        var omega = vec3<f32>(delta.x, delta.y, delta.z) * (2.0 * params.inv_h);
        if (delta.w < 0.0) {
            omega = -omega;
        }
        angular_velocities[i] = vec4<f32>(omega, 0.0);
    }
}
