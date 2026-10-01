// Swing-twist (cone-twist) rigid-body joint stepper, device kernel.
//
// Byte-for-byte-intent twin of the CPU golden in
// `src/rigid/joint/swing_twist_cpu.rs`. It is a full stepper: the host unrolls
// the substep loop into a fixed dispatch sequence that snapshots the transforms,
// predicts the bodies forward under gravity and damping, resets the XPBD
// multipliers, projects each joint's swing-cone then twist-limit then
// point-to-point constraints in colour-batch order, then recovers the
// velocities from the net per-substep motion.
//
// Multiplier layout. Each joint owns three Lagrange multipliers in the shared
// `lambda` buffer: slot `3 * j` for the point-to-point positional weld, slot
// `3 * j + 1` for the swing cone, and slot `3 * j + 2` for the twist limit,
// matching the CPU golden's `lambda[3k ..= 3k + 2]` split. The buffer is
// `3 * joint_count` long and `reset_lambda` clears all of it each substep.
//
// Scheduling and race-freedom. The `snapshot`, `predict`, and `recover` passes
// are per-body global dispatches; `reset_lambda` is a per-multiplier dispatch;
// all are race-free. The `solve` pass is dispatched once per colour batch; the
// host colours the joint graph so same-batch joints write disjoint movable
// bodies, and static bodies may be shared because their corrections scale by
// zero. Each solve invocation projects the swing cone first, then the twist
// limit, then the positional weld for its one joint, matching the CPU
// `solve_one` order.
//
// The arithmetic below performs the identical f32 operations in the identical
// order as the CPU twin; only device floating-point reassociation separates the
// two, which the parity test bounds with a tight tolerance rather than exact
// equality.
//
// Provenance: the point-to-point (ball-socket) constraint, the signed angular
// limit shared with the hinge limit, and the cone-swing limit with their
// substep XPBD handling (Müller et al., "Detailed Rigid Body Simulation with
// XPBD"), over the world-space inverse inertia and quaternion kinematics of
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
    // Number of Lagrange multipliers reset each substep. For the swing-twist
    // joint this equals `3 * joint_count` (positional weld, swing cone, and
    // twist limit per joint).
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

// Device-packed joint; layout matches `GpuSwingTwistJoint` in
// `src/rigid/joint/swing_twist.rs` (128 bytes).
struct Joint {
    anchor_a: vec4<f32>,
    anchor_b: vec4<f32>,
    twist_axis_a: vec4<f32>,
    twist_axis_b: vec4<f32>,
    ref_a: vec4<f32>,
    ref_b: vec4<f32>,
    body_a: u32,
    body_b: u32,
    swing_limit: f32,
    twist_min: f32,
    twist_max: f32,
    compliance: f32,
    swing_compliance: f32,
    twist_compliance: f32,
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
// glam's `Quat` multiplication order used by the CPU twin.
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

// Projects one colour batch of swing-twist joints for a single sweep: swing
// cone first (multiplier slot 3j + 1), then the twist limit (slot 3j + 2),
// then the point-to-point weld (slot 3j), matching the CPU `solve_one` order.
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

    // --- Swing-cone angular constraint (multiplier slot 3j + 1) ---
    {
        let q_a = orientations[a];
        let q_b = orientations[b];
        let axis_a = quat_rotate(q_a, joint.twist_axis_a.xyz);
        let axis_b = quat_rotate(q_b, joint.twist_axis_b.xyz);
        let len_a = length(axis_a);
        let len_b = length(axis_b);
        if (len_a >= EPSILON && len_b >= EPSILON) {
            let u_a = axis_a / len_a;
            let u_b = axis_b / len_b;
            // True angle between the twist axes (may exceed 90 degrees, so acos
            // of the dot rather than the cross-product magnitude).
            let swing = acos(clamp(dot(u_a, u_b), -1.0, 1.0));
            if (swing > joint.swing_limit) {
                let c = swing - joint.swing_limit;
                // Rotation axis that closes the gap between the two twist axes.
                let delta = cross(u_a, u_b);
                let delta_len = length(delta);
                if (delta_len >= EPSILON) {
                    let n = delta / delta_len;
                    let w_a = dot(n, world_inv_inertia_apply(q_a, ii_a, n));
                    let w_b = dot(n, world_inv_inertia_apply(q_b, ii_b, n));
                    let w = w_a + w_b;
                    if (w >= EPSILON) {
                        let slot = 3u * j + 1u;
                        let alpha_tilde = joint.swing_compliance / (params.h * params.h);
                        let d_lambda = (-c - alpha_tilde * lambda[slot]) / (w + alpha_tilde);
                        lambda[slot] = lambda[slot] + d_lambda;
                        let p = n * d_lambda;
                        // Gradients: -n on body a, +n on body b. `u_b` is `u_a`
                        // rotated by +swing about n, so rotating a about +n
                        // closes the cone and rotating b about +n widens it.
                        orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
                        orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
                    }
                }
            }
        }
    }

    // --- Twist-limit angular constraint (multiplier slot 3j + 2) ---
    {
        let q_a = orientations[a];
        let q_b = orientations[b];
        let axis = quat_rotate(q_a, joint.twist_axis_a.xyz);
        let axis_len = length(axis);
        if (axis_len >= EPSILON) {
            let u = axis / axis_len;
            let ra = quat_rotate(q_a, joint.ref_a.xyz);
            let rb = quat_rotate(q_b, joint.ref_b.xyz);
            let pa = ra - u * dot(u, ra);
            let pb = rb - u * dot(u, rb);
            let la = length(pa);
            let lb = length(pb);
            if (la >= EPSILON && lb >= EPSILON) {
                let pa_n = pa / la;
                let pb_n = pb / lb;
                // Signed twist angle from `a`'s reference to `b`'s about `u`.
                let sin_theta = dot(cross(pa_n, pb_n), u);
                let cos_theta = dot(pa_n, pb_n);
                let theta = atan2(sin_theta, cos_theta);
                // Signed limit violation with a free dead zone inside the range.
                var c = 0.0;
                var limited = false;
                if (theta < joint.twist_min) {
                    c = theta - joint.twist_min;
                    limited = true;
                } else if (theta > joint.twist_max) {
                    c = theta - joint.twist_max;
                    limited = true;
                }
                if (limited) {
                    let w_a = dot(u, world_inv_inertia_apply(q_a, ii_a, u));
                    let w_b = dot(u, world_inv_inertia_apply(q_b, ii_b, u));
                    let w = w_a + w_b;
                    if (w >= EPSILON) {
                        let slot = 3u * j + 2u;
                        let alpha_tilde = joint.twist_compliance / (params.h * params.h);
                        let d_lambda = (-c - alpha_tilde * lambda[slot]) / (w + alpha_tilde);
                        lambda[slot] = lambda[slot] + d_lambda;
                        let p = u * d_lambda;
                        // Gradients: -u on body a, +u on body b.
                        orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
                        orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
                    }
                }
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
