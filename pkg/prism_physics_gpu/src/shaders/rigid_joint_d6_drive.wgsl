// Configurable D6 rigid-body joint stepper with per-axis drives, device kernel.
//
// Byte-for-byte-intent twin of the CPU golden in
// `src/rigid/joint/d6_drive_cpu.rs`. It is a full stepper: the host unrolls the
// substep loop into a fixed dispatch sequence that snapshots the transforms,
// predicts the bodies forward under gravity and damping, resets the XPBD
// multipliers, then — per colour batch, per position iteration — projects each
// joint's six passive axes (two swings, the twist, three linear) exactly as the
// undriven `rigid_joint_d6.wgsl` twin and then its six per-axis drives, before
// recovering the velocities from the net per-substep motion.
//
// Multiplier layout. Each joint owns twelve Lagrange multipliers in the shared
// `lambda` buffer: slots `12 * j + 0 .. 6` the six passive axes (linear x, y,
// z, twist, swing1, swing2) laid out exactly as the passive twin, and slots
// `12 * j + 6 .. 12` the six drives in the same axis order (linear x, y, z,
// twist, swing1, swing2). The buffer is `12 * joint_count` long and
// `reset_lambda` clears all of it each substep.
//
// The drive update. Each drive is a parallel spring (gain `stiffness`, toward
// `target_position`) and damper (gain `damping`, toward `target_velocity`) on
// one scalar axis. A positive-stiffness drive is the compliant, velocity-damped
// XPBD equality `C = coord - target_position` with compliance `1 / stiffness`,
// regularisation `alpha_tilde = compliance / h^2`, and the Macklin damping term
// `gamma = compliance * damping / h` that bleeds the axis rate onto
// `target_velocity`. A zero-stiffness, positive-damping drive is the pure
// velocity motor `C = dv - target_velocity * h`, regularised by
// `alpha_tilde = 1 / (h * damping)` — the continuous `stiffness -> 0` limit of
// the servo. An inert drive (both gains zero) contributes nothing. The drive
// reads the relative axis rate from the per-substep snapshot buffers, so the
// shader binds `prev_positions` and `prev_orientations` read-write like the
// passive twin.
//
// Scheduling and race-freedom match the passive twin: `snapshot`, `predict`,
// and `recover` are per-body global dispatches, `reset_lambda` is per-
// multiplier, and `solve` is dispatched once per colour batch over a graph the
// host colours so same-batch joints write disjoint movable bodies. Each solve
// invocation projects one joint's six passive axes then its six drives,
// matching the CPU `solve_passive`-then-`solve_drives` order.
//
// The arithmetic below performs the identical f32 operations in the identical
// order as the CPU twin; axis normalisation uses an explicit length and
// division (never inverseSqrt) and angles use atan2, so both engines take one
// transcendental path. Only device floating-point reassociation separates the
// two, which the parity test bounds with a tight tolerance rather than exact
// equality.
//
// Provenance: the per-axis spring-damper actuator of a general-purpose
// constraint (Unreal Engine's `FConstraintDrive`, `PhysX`'s `PxD6JointDrive`),
// realised as a compliant, velocity-damped XPBD constraint with Macklin-style
// damping regularisation (Macklin et al., "XPBD: Position-Based Simulation of
// Compliant Constrained Dynamics") over the passive D6 constraints of Müller
// et al. and the world-space inverse inertia and quaternion kinematics of
// Baraff & Witkin. Standard wgpu compute dispatch. No Unreal Engine source or
// derived code: only the public spring-damper drive semantics are mirrored.

const EPSILON: f32 = 1.1920929e-7; // f32::EPSILON, matching the CPU guard.

struct Params {
    gravity: vec3<f32>,
    h: f32,
    inv_h: f32,
    linear_damping_scale: f32,
    angular_damping_scale: f32,
    joint_count: u32,
    body_count: u32,
    // Number of Lagrange multipliers reset each substep. For the driven D6
    // joint this equals `12 * joint_count` (six passive axes then six drives).
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

// Device-packed per-axis drive; layout matches `GpuD6Drive` in
// `src/rigid/joint/d6_drive.rs` (16 bytes): a spring-damper actuator on one
// scalar degree of freedom.
struct Drive {
    stiffness: f32,
    damping: f32,
    target_position: f32,
    target_velocity: f32,
};

// Device-packed driven joint; layout matches `GpuD6DrivenJoint` in
// `src/rigid/joint/d6_drive_gpu.rs` (224 bytes: the 128-byte passive
// `GpuD6Joint` followed by its 96-byte `GpuD6DriveSet` of six 16-byte drives in
// axis order, no interior or trailing padding).
struct Joint {
    anchor_a: vec4<f32>,
    anchor_b: vec4<f32>,
    basis_a: vec4<f32>,
    basis_b: vec4<f32>,
    body_a: u32,
    body_b: u32,
    linear_x: u32,
    linear_y: u32,
    linear_z: u32,
    twist: u32,
    swing1: u32,
    swing2: u32,
    linear_limit: f32,
    twist_min: f32,
    twist_max: f32,
    swing1_limit: f32,
    swing2_limit: f32,
    compliance: f32,
    linear_limit_compliance: f32,
    angular_limit_compliance: f32,
    drive_linear_x: Drive,
    drive_linear_y: Drive,
    drive_linear_z: Drive,
    drive_twist: Drive,
    drive_swing1: Drive,
    drive_swing2: Drive,
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

// Normalises a vector, returning the zero vector when degenerate — the device
// analogue of glam's `Vec3::normalize_or_zero` used by the CPU swing pass.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if (len > EPSILON) {
        return v / len;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// The world-space joint-frame orientation of a body — the body orientation
// composed with the body-local joint basis.
fn frame_world(body_orientation: vec4<f32>, basis: vec4<f32>) -> vec4<f32> {
    return quat_mul(body_orientation, basis);
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

// Resets every joint's twelve XPBD Lagrange multipliers at the start of a substep.
@compute @workgroup_size(64)
fn reset_lambda(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= params.lambda_count) {
        return;
    }
    lambda[j] = 0.0;
}

// Applies a single-axis angular XPBD correction of signed violation c about the
// unit world axis n, with the antisymmetric gradient -n on body a and +n on
// body b. Shared by the twist and the two swing passes; mirrors the CPU
// `apply_angular`.
fn apply_angular(
    a: u32,
    b: u32,
    q_a: vec4<f32>,
    q_b: vec4<f32>,
    ii_a: vec3<f32>,
    ii_b: vec3<f32>,
    n: vec3<f32>,
    c: f32,
    compliance: f32,
    slot: u32,
) {
    let w_a = dot(n, world_inv_inertia_apply(q_a, ii_a, n));
    let w_b = dot(n, world_inv_inertia_apply(q_b, ii_b, n));
    let w = w_a + w_b;
    if (w < EPSILON) {
        return;
    }
    let alpha_tilde = compliance / (params.h * params.h);
    let d_lambda = (-c - alpha_tilde * lambda[slot]) / (w + alpha_tilde);
    lambda[slot] = lambda[slot] + d_lambda;
    let p = n * d_lambda;
    orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
    orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
}

// Projects swing1 (`which == 0`, tilt toward the frame y axis, corrected about
// +e_z) or swing2 (`which == 1`, tilt toward the frame z axis, corrected about
// -e_y) of a D6 joint. Mirrors the CPU `solve_swing`.
fn solve_swing(j: u32, a: u32, b: u32, ii_a: vec3<f32>, ii_b: vec3<f32>, which: u32, slot: u32) {
    let joint = joints[j];
    var motion = joint.swing1;
    if (which == 1u) {
        motion = joint.swing2;
    }
    if (motion == 2u) {
        return;
    }

    let q_a = orientations[a];
    let q_b = orientations[b];
    let frame_a = frame_world(q_a, joint.basis_a);
    let frame_b = frame_world(q_b, joint.basis_b);
    let t_raw = quat_rotate(frame_a, vec3<f32>(1.0, 0.0, 0.0));
    let e_y_raw = quat_rotate(frame_a, vec3<f32>(0.0, 1.0, 0.0));
    let e_z_raw = quat_rotate(frame_a, vec3<f32>(0.0, 0.0, 1.0));
    let t_len = length(t_raw);
    if (t_len < EPSILON) {
        return;
    }
    let t = t_raw / t_len;
    let e_y = normalize_or_zero(e_y_raw);
    let e_z = normalize_or_zero(e_z_raw);
    if (dot(e_y, e_y) < 0.5 || dot(e_z, e_z) < 0.5) {
        return;
    }

    let u_b_raw = quat_rotate(frame_b, vec3<f32>(1.0, 0.0, 0.0));
    let u_b_len = length(u_b_raw);
    if (u_b_len < EPSILON) {
        return;
    }
    let u_b = u_b_raw / u_b_len;

    let x_c = dot(u_b, t);
    var angle = 0.0;
    var n = vec3<f32>(0.0, 0.0, 0.0);
    var limit = 0.0;
    if (which == 0u) {
        let y_c = dot(u_b, e_y);
        angle = atan2(y_c, x_c);
        n = e_z;
        limit = joint.swing1_limit;
    } else {
        let z_c = dot(u_b, e_z);
        angle = atan2(z_c, x_c);
        n = -e_y;
        limit = joint.swing2_limit;
    }

    var c = 0.0;
    var compliance = 0.0;
    if (motion == 0u) {
        c = angle;
        compliance = joint.compliance;
    } else {
        if (angle > limit) {
            c = angle - limit;
            compliance = joint.angular_limit_compliance;
        } else if (angle < -limit) {
            c = angle + limit;
            compliance = joint.angular_limit_compliance;
        } else {
            return;
        }
    }

    apply_angular(a, b, q_a, q_b, ii_a, ii_b, n, c, compliance, slot);
}

// Projects the twist axis of a D6 joint — the signed angle from body a's frame
// y axis to body b's about the world twist axis. Mirrors the CPU `solve_twist`.
fn solve_twist(j: u32, a: u32, b: u32, ii_a: vec3<f32>, ii_b: vec3<f32>, slot: u32) {
    let joint = joints[j];
    if (joint.twist == 2u) {
        return;
    }

    let q_a = orientations[a];
    let q_b = orientations[b];
    let frame_a = frame_world(q_a, joint.basis_a);
    let frame_b = frame_world(q_b, joint.basis_b);
    let t_raw = quat_rotate(frame_a, vec3<f32>(1.0, 0.0, 0.0));
    let t_len = length(t_raw);
    if (t_len < EPSILON) {
        return;
    }
    let t = t_raw / t_len;

    let ref_a = quat_rotate(frame_a, vec3<f32>(0.0, 1.0, 0.0));
    let ref_b = quat_rotate(frame_b, vec3<f32>(0.0, 1.0, 0.0));
    let pa = ref_a - t * dot(t, ref_a);
    let pb = ref_b - t * dot(t, ref_b);
    let la = length(pa);
    let lb = length(pb);
    if (la < EPSILON || lb < EPSILON) {
        return;
    }
    let pa_n = pa / la;
    let pb_n = pb / lb;

    let sin_theta = dot(cross(pa_n, pb_n), t);
    let cos_theta = dot(pa_n, pb_n);
    let theta = atan2(sin_theta, cos_theta);

    var c = 0.0;
    var compliance = 0.0;
    if (joint.twist == 0u) {
        c = theta;
        compliance = joint.compliance;
    } else {
        if (theta < joint.twist_min) {
            c = theta - joint.twist_min;
            compliance = joint.angular_limit_compliance;
        } else if (theta > joint.twist_max) {
            c = theta - joint.twist_max;
            compliance = joint.angular_limit_compliance;
        } else {
            return;
        }
    }

    apply_angular(a, b, q_a, q_b, ii_a, ii_b, t, c, compliance, slot);
}

// Projects the linear axis `axis` (0 = frame x, 1 = y, 2 = z) of a D6 joint —
// the signed anchor separation along the world-space frame axis. Mirrors the
// CPU `solve_linear`.
fn solve_linear(j: u32, a: u32, b: u32, ii_a: vec3<f32>, ii_b: vec3<f32>, axis: u32, slot: u32) {
    let joint = joints[j];
    var motion = joint.linear_x;
    var local = vec3<f32>(1.0, 0.0, 0.0);
    if (axis == 1u) {
        motion = joint.linear_y;
        local = vec3<f32>(0.0, 1.0, 0.0);
    } else if (axis == 2u) {
        motion = joint.linear_z;
        local = vec3<f32>(0.0, 0.0, 1.0);
    }
    if (motion == 2u) {
        return;
    }

    let q_a = orientations[a];
    let q_b = orientations[b];
    let frame_a = frame_world(q_a, joint.basis_a);
    let axis_w = quat_rotate(frame_a, local);
    let axis_len = length(axis_w);
    if (axis_len < EPSILON) {
        return;
    }
    let n = axis_w / axis_len;

    let r_a = quat_rotate(q_a, joint.anchor_a.xyz);
    let r_b = quat_rotate(q_b, joint.anchor_b.xyz);
    let dx = (positions[a].xyz + r_a) - (positions[b].xyz + r_b);
    let s = dot(dx, n);

    let limit = joint.linear_limit;
    var c = 0.0;
    var compliance = 0.0;
    if (motion == 0u) {
        c = s;
        compliance = joint.compliance;
    } else {
        if (s > limit) {
            c = s - limit;
            compliance = joint.linear_limit_compliance;
        } else if (s < -limit) {
            c = s + limit;
            compliance = joint.linear_limit_compliance;
        } else {
            return;
        }
    }

    let inv_m_a = inverse_masses[a];
    let inv_m_b = inverse_masses[b];
    let rn_a = cross(r_a, n);
    let rn_b = cross(r_b, n);
    let w_a = inv_m_a + dot(rn_a, world_inv_inertia_apply(q_a, ii_a, rn_a));
    let w_b = inv_m_b + dot(rn_b, world_inv_inertia_apply(q_b, ii_b, rn_b));
    let w = w_a + w_b;
    if (w < EPSILON) {
        return;
    }

    let alpha_tilde = compliance / (params.h * params.h);
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

// Relative rotation of a body since the substep snapshot, as a rotation vector:
// twice the imaginary part of the delta quaternion `orientation *
// conj(prev_orientation)`, hemisphere-corrected so the shortest arc is taken.
// An angular drive dots this with its correction axis to read the relative
// angular rate. Mirrors the CPU `angular_displacement`.
fn angular_displacement(q: vec4<f32>, prev_q: vec4<f32>) -> vec3<f32> {
    let delta = quat_mul(q, quat_conj(prev_q));
    var rotvec = vec3<f32>(delta.x, delta.y, delta.z) * 2.0;
    if (delta.w < 0.0) {
        rotvec = -rotvec;
    }
    return rotvec;
}

// Computes one axis's drive multiplier increment, accumulates it into
// `lambda[slot]`, and returns the impulse magnitude to apply along the axis.
// `w` is the generalized effective inverse mass about the axis, `coord` the
// current axis coordinate, and `dv` the relative displacement along the axis
// since the substep snapshot. A positive-stiffness drive is the compliant,
// velocity-damped position servo `C = coord - target_position`; a
// zero-stiffness, positive-damping drive is the pure velocity motor
// `C = dv - target_velocity * h`, whose regularisation is the continuous
// `stiffness -> 0` limit of the servo. An inert drive (both gains zero) or a
// singular axis contributes nothing. Mirrors the CPU `drive_delta`.
fn drive_delta(w: f32, coord: f32, dv: f32, drive: Drive, slot: u32) -> f32 {
    if (!(drive.stiffness > 0.0 || drive.damping > 0.0) || w < EPSILON) {
        return 0.0;
    }

    // Relative displacement the drive wants over this substep to hold the
    // commanded rate; the damper resists deviation from it.
    let c_vel = dv - drive.target_velocity * params.h;

    var d_lambda = 0.0;
    if (drive.stiffness > 0.0) {
        let compliance = 1.0 / drive.stiffness;
        let alpha_tilde = compliance / (params.h * params.h);
        // XPBD damping scale gamma = compliance * damping / h (Macklin et al.);
        // it couples the damper to the spring and vanishes with the compliance.
        var gamma = 0.0;
        if (drive.damping > 0.0) {
            gamma = compliance * drive.damping / params.h;
        }
        let c_pos = coord - drive.target_position;
        d_lambda = (-c_pos - alpha_tilde * lambda[slot] - gamma * c_vel)
            / ((1.0 + gamma) * w + alpha_tilde);
    } else {
        // Pure velocity motor: the stiffness -> 0 limit of the servo above, with
        // the damper setting the regularisation 1 / (h * damping).
        let alpha_tilde = 1.0 / (params.h * drive.damping);
        d_lambda = (-c_vel - alpha_tilde * lambda[slot]) / (w + alpha_tilde);
    }

    lambda[slot] = lambda[slot] + d_lambda;
    return d_lambda;
}

// Servos the twist angle onto its drive about the world twist axis
// `t = rotate(frame_a, X)`, measuring the angle exactly as the passive twist
// pass (the signed angle from body a's frame y axis to body b's, both projected
// perpendicular to t). Mirrors the CPU `drive_twist`.
fn drive_twist(j: u32, a: u32, b: u32, ii_a: vec3<f32>, ii_b: vec3<f32>, drive: Drive, slot: u32) {
    if (!(drive.stiffness > 0.0 || drive.damping > 0.0)) {
        return;
    }
    let joint = joints[j];

    let q_a = orientations[a];
    let q_b = orientations[b];
    let frame_a = frame_world(q_a, joint.basis_a);
    let frame_b = frame_world(q_b, joint.basis_b);
    let t_raw = quat_rotate(frame_a, vec3<f32>(1.0, 0.0, 0.0));
    let t_len = length(t_raw);
    if (t_len < EPSILON) {
        return;
    }
    let t = t_raw / t_len;

    let ref_a = quat_rotate(frame_a, vec3<f32>(0.0, 1.0, 0.0));
    let ref_b = quat_rotate(frame_b, vec3<f32>(0.0, 1.0, 0.0));
    let pa = ref_a - t * dot(t, ref_a);
    let pb = ref_b - t * dot(t, ref_b);
    let la = length(pa);
    let lb = length(pb);
    if (la < EPSILON || lb < EPSILON) {
        return;
    }
    let pa_n = pa / la;
    let pb_n = pb / lb;
    let sin_theta = dot(cross(pa_n, pb_n), t);
    let cos_theta = dot(pa_n, pb_n);
    let theta = atan2(sin_theta, cos_theta);

    let w_a = dot(t, world_inv_inertia_apply(q_a, ii_a, t));
    let w_b = dot(t, world_inv_inertia_apply(q_b, ii_b, t));
    let w = w_a + w_b;

    let angvec_a = angular_displacement(q_a, prev_orientations[a]);
    let angvec_b = angular_displacement(q_b, prev_orientations[b]);
    let dv = dot(t, angvec_b - angvec_a);

    let d_lambda = drive_delta(w, theta, dv, drive, slot);
    let p = t * d_lambda;
    orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
    orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
}

// Servos swing1 (`which == 0`, tilt toward the frame y axis, corrected about
// `+e_z`) or swing2 (`which == 1`, tilt toward the frame z axis, corrected about
// `-e_y`) onto its drive, measuring the angle exactly as the passive swing pass.
// Mirrors the CPU `drive_swing`.
fn drive_swing(j: u32, a: u32, b: u32, ii_a: vec3<f32>, ii_b: vec3<f32>, drive: Drive, which: u32, slot: u32) {
    if (!(drive.stiffness > 0.0 || drive.damping > 0.0)) {
        return;
    }
    let joint = joints[j];

    let q_a = orientations[a];
    let q_b = orientations[b];
    let frame_a = frame_world(q_a, joint.basis_a);
    let frame_b = frame_world(q_b, joint.basis_b);
    let t_raw = quat_rotate(frame_a, vec3<f32>(1.0, 0.0, 0.0));
    let e_y_raw = quat_rotate(frame_a, vec3<f32>(0.0, 1.0, 0.0));
    let e_z_raw = quat_rotate(frame_a, vec3<f32>(0.0, 0.0, 1.0));
    let t_len = length(t_raw);
    if (t_len < EPSILON) {
        return;
    }
    let t = t_raw / t_len;
    let e_y = normalize_or_zero(e_y_raw);
    let e_z = normalize_or_zero(e_z_raw);
    if (dot(e_y, e_y) < 0.5 || dot(e_z, e_z) < 0.5) {
        return;
    }

    let u_b_raw = quat_rotate(frame_b, vec3<f32>(1.0, 0.0, 0.0));
    let u_b_len = length(u_b_raw);
    if (u_b_len < EPSILON) {
        return;
    }
    let u_b = u_b_raw / u_b_len;

    let x_c = dot(u_b, t);
    var angle = 0.0;
    var n = vec3<f32>(0.0, 0.0, 0.0);
    if (which == 0u) {
        let y_c = dot(u_b, e_y);
        angle = atan2(y_c, x_c);
        n = e_z;
    } else {
        let z_c = dot(u_b, e_z);
        angle = atan2(z_c, x_c);
        n = -e_y;
    }

    let w_a = dot(n, world_inv_inertia_apply(q_a, ii_a, n));
    let w_b = dot(n, world_inv_inertia_apply(q_b, ii_b, n));
    let w = w_a + w_b;

    let angvec_a = angular_displacement(q_a, prev_orientations[a]);
    let angvec_b = angular_displacement(q_b, prev_orientations[b]);
    let dv = dot(n, angvec_b - angvec_a);

    let d_lambda = drive_delta(w, angle, dv, drive, slot);
    let p = n * d_lambda;
    orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
    orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
}

// Servos the linear axis `axis` (0 = frame x, 1 = y, 2 = z) onto its drive. The
// signed anchor separation along the world frame axis is the coordinate; the
// damper reads the along-axis closing rate of the two material anchor points
// since the snapshot. Mirrors the CPU `drive_linear`.
fn drive_linear(j: u32, a: u32, b: u32, ii_a: vec3<f32>, ii_b: vec3<f32>, drive: Drive, axis: u32, slot: u32) {
    if (!(drive.stiffness > 0.0 || drive.damping > 0.0)) {
        return;
    }
    let joint = joints[j];

    var local = vec3<f32>(1.0, 0.0, 0.0);
    if (axis == 1u) {
        local = vec3<f32>(0.0, 1.0, 0.0);
    } else if (axis == 2u) {
        local = vec3<f32>(0.0, 0.0, 1.0);
    }

    let q_a = orientations[a];
    let q_b = orientations[b];
    let frame_a = frame_world(q_a, joint.basis_a);
    let axis_w = quat_rotate(frame_a, local);
    let axis_len = length(axis_w);
    if (axis_len < EPSILON) {
        return;
    }
    let n = axis_w / axis_len;

    let r_a = quat_rotate(q_a, joint.anchor_a.xyz);
    let r_b = quat_rotate(q_b, joint.anchor_b.xyz);
    let point_a = positions[a].xyz + r_a;
    let point_b = positions[b].xyz + r_b;
    let s = dot(point_a - point_b, n);

    // Relative displacement of the two material anchor points since the snapshot,
    // projected onto the current frame axis, so the damper reads the along-axis
    // closing rate.
    let prev_point_a = prev_positions[a].xyz + quat_rotate(prev_orientations[a], joint.anchor_a.xyz);
    let prev_point_b = prev_positions[b].xyz + quat_rotate(prev_orientations[b], joint.anchor_b.xyz);
    let dv = dot((point_a - prev_point_a) - (point_b - prev_point_b), n);

    let inv_m_a = inverse_masses[a];
    let inv_m_b = inverse_masses[b];
    let rn_a = cross(r_a, n);
    let rn_b = cross(r_b, n);
    let w_a = inv_m_a + dot(rn_a, world_inv_inertia_apply(q_a, ii_a, rn_a));
    let w_b = inv_m_b + dot(rn_b, world_inv_inertia_apply(q_b, ii_b, rn_b));
    let w = w_a + w_b;

    let d_lambda = drive_delta(w, s, dv, drive, slot);
    let p = n * d_lambda;

    positions[a] = vec4<f32>(positions[a].xyz + p * inv_m_a, 0.0);
    positions[b] = vec4<f32>(positions[b].xyz - p * inv_m_b, 0.0);
    let dw_a = world_inv_inertia_apply(q_a, ii_a, cross(r_a, p));
    orientations[a] = apply_rotation_delta(q_a, dw_a);
    let dw_b = world_inv_inertia_apply(q_b, ii_b, cross(r_b, p));
    orientations[b] = apply_rotation_delta(q_b, -dw_b);
}

// Projects one colour batch of driven D6 joints for a single sweep: the six
// passive axes first — swing1 (slot 12j + 4), swing2 (12j + 5), twist
// (12j + 3), then linear x/y/z (12j + 0, 12j + 1, 12j + 2) — exactly as the
// passive twin, then the six drives — swing1 (12j + 10), swing2 (12j + 11),
// twist (12j + 9), then linear x/y/z (12j + 6, 12j + 7, 12j + 8) — matching the
// CPU `solve_passive`-then-`solve_drives` order.
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

    solve_swing(j, a, b, ii_a, ii_b, 0u, 12u * j + 4u);
    solve_swing(j, a, b, ii_a, ii_b, 1u, 12u * j + 5u);
    solve_twist(j, a, b, ii_a, ii_b, 12u * j + 3u);
    solve_linear(j, a, b, ii_a, ii_b, 0u, 12u * j);
    solve_linear(j, a, b, ii_a, ii_b, 1u, 12u * j + 1u);
    solve_linear(j, a, b, ii_a, ii_b, 2u, 12u * j + 2u);

    drive_swing(j, a, b, ii_a, ii_b, joint.drive_swing1, 0u, 12u * j + 10u);
    drive_swing(j, a, b, ii_a, ii_b, joint.drive_swing2, 1u, 12u * j + 11u);
    drive_twist(j, a, b, ii_a, ii_b, joint.drive_twist, 12u * j + 9u);
    drive_linear(j, a, b, ii_a, ii_b, joint.drive_linear_x, 0u, 12u * j + 6u);
    drive_linear(j, a, b, ii_a, ii_b, joint.drive_linear_y, 1u, 12u * j + 7u);
    drive_linear(j, a, b, ii_a, ii_b, joint.drive_linear_z, 2u, 12u * j + 8u);
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
