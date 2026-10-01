// Angular SLERP drive rigid-body joint stepper, device kernel.
//
// Byte-for-byte-intent twin of the CPU golden in
// `src/rigid/joint/angular_slerp_drive_cpu.rs`. It is a full stepper: the host
// unrolls the substep loop into a fixed dispatch sequence that snapshots the
// transforms, predicts the bodies forward under gravity and damping, resets the
// XPBD multipliers, projects each joint's single geodesic angular drive in
// colour-batch order, then recovers the velocities from the net per-substep
// motion.
//
// Multiplier layout. Each joint owns a single Lagrange multiplier in the shared
// `lambda` buffer: slot `j` is the geodesic angular drive. There is no
// positional weld — the drive couples only orientations. The buffer is
// `joint_count` long and `reset_lambda` clears all of it each substep.
//
// Scheduling and race-freedom. The `snapshot`, `predict`, and `recover` passes
// are per-body global dispatches; `reset_lambda` is a per-multiplier dispatch;
// all are race-free. The `solve` pass is dispatched once per colour batch; the
// host colours the joint graph so same-batch joints write disjoint movable
// bodies, and static bodies may be shared because their corrections scale by
// zero.
//
// The drive is the bilateral compliant-and-damped XPBD equality `C = theta`,
// where `theta` is the geodesic angle of the world-space error rotation
// `(q_a * target_rotation) * conj(q_b)` carrying body `b` onto its target, and
// `n` is that rotation's axis. The damping term reads the shared per-substep
// snapshot buffer `prev_orientations` to form the relative angular displacement
// about `n`, exactly as the CPU twin's `angular_displacement` does. Its
// accumulated multiplier is then clamped each sweep to the box
// `[-max_torque * h, +max_torque * h]` — the net angular impulse a torque of
// `max_torque` delivers over the substep — so the drive saturates rather than
// snapping; a non-positive `max_torque` disables the cap.
//
// The arithmetic below performs the identical f32 operations in the identical
// order as the CPU twin; only device floating-point reassociation separates the
// two, which the parity test bounds with a tight tolerance rather than exact
// equality.
//
// Provenance: the relative-orientation geodesic measurement and its substep
// XPBD angular correction (Müller et al., "Detailed Rigid Body Simulation with
// XPBD"), with the bilateral compliant-and-damped drive and its Macklin-style
// damping regularisation (Macklin et al., "XPBD: Position-Based Simulation of
// Compliant Constrained Dynamics") and the box-limited projected-Gauss-Seidel
// torque cap, over the world-space inverse inertia and quaternion kinematics of
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
    // Number of Lagrange multipliers reset each substep. For the angular SLERP
    // drive joint this equals `joint_count` (one geodesic angular drive per
    // joint, with no positional weld).
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

// Device-packed joint; layout matches `GpuAngularSlerpDriveJoint` in
// `src/rigid/joint/angular_slerp_drive.rs` (48 bytes).
struct Joint {
    target_rotation: vec4<f32>,
    body_a: u32,
    body_b: u32,
    drive_compliance: f32,
    drive_damping: f32,
    max_torque: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
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

// Resets every joint's single XPBD Lagrange multiplier at the start of a
// substep.
@compute @workgroup_size(64)
fn reset_lambda(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= params.lambda_count) {
        return;
    }
    lambda[j] = 0.0;
}

// Projects one colour batch of angular SLERP drives for a single sweep:
// measures the geodesic error rotation from `b` to its target
// `q_a * target_rotation`, forms the compliant-and-damped XPBD step, clamps the
// accumulated impulse to the torque cap, and applies the equal-and-opposite
// rotation delta to the two bodies. Multiplier slot `j`.
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

    let q_a = orientations[a];
    let q_b = orientations[b];

    // Target world orientation of body `b`, and the world-space error rotation
    // that carries `b` onto it: `error = target * conj(q_b)`.
    let goal = quat_mul(q_a, joint.target_rotation);
    var error = quat_mul(goal, quat_conj(q_b));
    // Take the shortest-arc representative (non-negative scalar part).
    if (error.w < 0.0) {
        error = -error;
    }

    // The rotation vector is twice the imaginary part of the error quaternion.
    let delta = vec3<f32>(2.0 * error.x, 2.0 * error.y, 2.0 * error.z);
    let theta = length(delta);
    if (theta < EPSILON) {
        return;
    }
    let n = delta / theta;

    let w_a = dot(n, world_inv_inertia_apply(q_a, ii_a, n));
    let w_b = dot(n, world_inv_inertia_apply(q_b, ii_b, n));
    let w = w_a + w_b;
    if (w < EPSILON) {
        return;
    }

    let alpha_tilde = joint.drive_compliance / (params.h * params.h);
    // XPBD damping scale: gamma = compliance * damping / h (Macklin et al.).
    let gamma = (joint.drive_compliance * joint.drive_damping) / params.h;

    // Relative angular displacement that increases `theta` since the substep
    // snapshot: `dv = n . (angvec_a - angvec_b)`.
    let angvec_a = angular_displacement(q_a, prev_orientations[a]);
    let angvec_b = angular_displacement(q_b, prev_orientations[b]);
    let dv = dot(n, angvec_a - angvec_b);

    let d_lambda = (-theta - alpha_tilde * lambda[j] - gamma * dv)
        / ((1.0 + gamma) * w + alpha_tilde);
    // Box-limited projected step: clamp the accumulated impulse to the torque
    // cap, then apply only the admissible change. `max_torque <= 0` disables it.
    let old = lambda[j];
    let unclamped = old + d_lambda;
    var new_lambda: f32;
    if (joint.max_torque > 0.0) {
        let max_impulse = joint.max_torque * params.h;
        new_lambda = clamp(unclamped, -max_impulse, max_impulse);
    } else {
        new_lambda = unclamped;
    }
    lambda[j] = new_lambda;
    let p = n * (new_lambda - old);

    // Equal and opposite angular impulses: rotating `a` by `+I_a^-1 p` and `b`
    // by `-I_b^-1 p` reduces `theta` for the admissible step while conserving
    // angular momentum.
    orientations[a] = apply_rotation_delta(q_a, world_inv_inertia_apply(q_a, ii_a, p));
    orientations[b] = apply_rotation_delta(q_b, -world_inv_inertia_apply(q_b, ii_b, p));
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
