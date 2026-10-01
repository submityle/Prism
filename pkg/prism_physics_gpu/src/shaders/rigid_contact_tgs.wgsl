// Soft-constraint Temporal Gauss-Seidel (TGS) 6-DOF rigid-body contact stepper,
// device kernel.
//
// Byte-for-byte-intent twin of the CPU golden in
// `src/rigid/contact_tgs_cpu.rs`. Unlike the velocity-only solver, this is a
// full stepper: the host unrolls the substep loop into dispatches that
// integrate the bodies' velocities under gravity, warm start the contacts,
// re-linearise them against the moved geometry, resolve them as damped springs,
// integrate positions and orientations, then run bias-free relaxation sweeps.
// A single restitution pass after the loop restores elastic bounce.
//
// Scheduling and race-freedom. Per-body passes (`integrate_velocities`,
// `integrate_positions`) and the per-contact `relinearise` pass are global
// dispatches: each thread writes only its own body or its own contact, so there
// is no cross-thread write hazard. The impulse-applying passes (`warm_start`,
// `solve_biased`, `solve_relax`, `apply_restitution`) are dispatched once per
// colour batch; the host colours the contact graph so same-batch contacts write
// disjoint movable bodies, and static bodies (zero inverse mass and inertia) may
// be shared because their write guards skip the (zero) store. The restitution
// pass is therefore colour-ordered on both engines so the Gauss-Seidel sweep
// visits shared bodies in the identical order.
//
// The host pre-computes the soft-constraint coefficients (`bias_rate`,
// `mass_scale`, `impulse_scale`) with a single `SoftParams::from_hertz` call and
// uploads them, so the shader never re-derives them and cannot drift from the
// CPU twin by a `from_hertz` rounding difference. The arithmetic below performs
// the identical f32 operations in the identical order as the CPU twin; only
// device floating-point reassociation separates the two, which the parity test
// bounds with a tight tolerance rather than exact equality.
//
// Provenance: the soft-constraint contact of Catto ("Soft Constraints", GDC
// 2011) and the substepping solver loop it feeds (Box2D TGS Soft), over the
// world-space inverse inertia and quaternion kinematics of Baraff & Witkin.
// Standard wgpu compute dispatch. No Unreal Engine source or derived code.

const EPSILON: f32 = 1.1920929e-7; // f32::EPSILON, matching the CPU guard.
const TANGENT_THRESHOLD: f32 = 0.57735026; // ~= 1/sqrt(3), the CPU seed cutoff.
const MAX_RECOVERY_SPEED: f32 = 3.0; // Box2D v3 maxBiasVelocity.

struct Params {
    gravity: vec3<f32>,
    h: f32,
    inv_h: f32,
    linear_damping_scale: f32,
    angular_damping_scale: f32,
    slop: f32,
    bias_rate: f32,
    mass_scale: f32,
    impulse_scale: f32,
    restitution_threshold: f32,
    contact_count: u32,
    body_count: u32,
    _pad0: u32,
    _pad1: u32,
};

struct ColourParams {
    start: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
};

// Device-packed working contact; layout matches `GpuRigidContact` in
// `src/rigid/contact.rs` (80 bytes). The anchors and `penetration` are
// read-write because `relinearise` overwrites them each substep, and the three
// impulse accumulators round-trip the warm-start seed and converged solution.
struct Contact {
    anchor_a: vec4<f32>,
    anchor_b: vec4<f32>,
    normal: vec4<f32>,
    body_a: u32,
    body_b: u32,
    penetration: f32,
    friction: f32,
    restitution: f32,
    normal_impulse: f32,
    tangent_impulse_0: f32,
    tangent_impulse_1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> linear_velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> angular_velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> orientations: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(6) var<storage, read> inverse_inertias: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(8) var<storage, read> initial_positions: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> initial_orientations: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> initial_arm_a: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> initial_arm_b: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> base_separation: array<f32>;
@group(0) @binding(13) var<storage, read> approach_speed: array<f32>;

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

// Rotates v by the unit quaternion q via the expanded sandwich product.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = vec3<f32>(q.x, q.y, q.z);
    let s = q.w;
    return u * (2.0 * dot(u, v)) + v * (s * s - dot(u, u)) + cross(u, v) * (2.0 * s);
}

// Applies the world-space inverse inertia R diag(inv_inertia) R^T to v.
fn world_inv_inertia_apply(q: vec4<f32>, inv_inertia: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let body = quat_rotate(quat_conj(q), v);
    let scaled = inv_inertia * body;
    return quat_rotate(q, scaled);
}

// Right-handed orthonormal tangent basis spanning the plane perpendicular to n.
// Identical branchless construction as the CPU twin.
fn contact_tangents(n: vec3<f32>) -> mat2x3<f32> {
    var t1: vec3<f32>;
    if (abs(n.x) >= TANGENT_THRESHOLD) {
        t1 = vec3<f32>(n.y, -n.x, 0.0);
    } else {
        t1 = vec3<f32>(0.0, n.z, -n.y);
    }
    let len = length(t1);
    if (len > 0.0) {
        t1 = t1 / len;
    } else {
        t1 = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t2 = cross(n, t1);
    return mat2x3<f32>(t1, t2);
}

// Relative velocity of the contact point on body a with respect to body b.
fn relative_velocity(ci: u32) -> vec3<f32> {
    let a = contacts[ci].body_a;
    let b = contacts[ci].body_b;
    let va = linear_velocities[a].xyz + cross(angular_velocities[a].xyz, contacts[ci].anchor_a.xyz);
    let vb = linear_velocities[b].xyz + cross(angular_velocities[b].xyz, contacts[ci].anchor_b.xyz);
    return va - vb;
}

// Scalar effective mass seen along the unit direction d, including both bodies'
// angular arms through their world-space inverse inertia.
fn effective_mass(ci: u32, d: vec3<f32>) -> f32 {
    let a = contacts[ci].body_a;
    let b = contacts[ci].body_b;
    var k = inverse_masses[a] + inverse_masses[b];
    let arm_a = cross(
        world_inv_inertia_apply(orientations[a], inverse_inertias[a].xyz, cross(contacts[ci].anchor_a.xyz, d)),
        contacts[ci].anchor_a.xyz,
    );
    k = k + dot(d, arm_a);
    let arm_b = cross(
        world_inv_inertia_apply(orientations[b], inverse_inertias[b].xyz, cross(contacts[ci].anchor_b.xyz, d)),
        contacts[ci].anchor_b.xyz,
    );
    k = k + dot(d, arm_b);
    return k;
}

// Reports whether a body's inverse inertia has any non-zero (unlocked) axis.
fn can_rotate(index: u32) -> bool {
    let inv = inverse_inertias[index].xyz;
    return inv.x > 0.0 || inv.y > 0.0 || inv.z > 0.0;
}

// Reports whether the solver may write the body: a free translation or a free
// rotation axis. Matches the CPU `movable_mask`.
fn is_movable(index: u32) -> bool {
    return inverse_masses[index] > 0.0 || can_rotate(index);
}

// Applies +p to body a at anchor_a and -p to body b at anchor_b, guarding each
// store so only movable bodies are written (static/locked deltas are zero, so
// skipping the write preserves the value and removes the batch race).
fn apply_impulse(ci: u32, p: vec3<f32>) {
    let a = contacts[ci].body_a;
    let b = contacts[ci].body_b;

    if (inverse_masses[a] > 0.0) {
        linear_velocities[a] = vec4<f32>(linear_velocities[a].xyz + p * inverse_masses[a], 0.0);
    }
    if (can_rotate(a)) {
        let dw_a = world_inv_inertia_apply(
            orientations[a],
            inverse_inertias[a].xyz,
            cross(contacts[ci].anchor_a.xyz, p),
        );
        angular_velocities[a] = vec4<f32>(angular_velocities[a].xyz + dw_a, 0.0);
    }

    if (inverse_masses[b] > 0.0) {
        linear_velocities[b] = vec4<f32>(linear_velocities[b].xyz - p * inverse_masses[b], 0.0);
    }
    if (can_rotate(b)) {
        let dw_b = world_inv_inertia_apply(
            orientations[b],
            inverse_inertias[b].xyz,
            cross(contacts[ci].anchor_b.xyz, p),
        );
        angular_velocities[b] = vec4<f32>(angular_velocities[b].xyz - dw_b, 0.0);
    }
}

// Advances the orientation q by angular velocity omega over h via the
// quaternion kinematic equation q' = normalize(q + 0.5 h (omega_q * q)),
// matching the CPU twin's `integrate_orientation`.
fn integrate_orientation(q: vec4<f32>, omega: vec3<f32>, h: f32) -> vec4<f32> {
    let omega_quat = vec4<f32>(omega.x, omega.y, omega.z, 0.0);
    let dq = quat_mul(omega_quat, q);
    let half_h = 0.5 * h;
    let integrated = vec4<f32>(
        q.x + dq.x * half_h,
        q.y + dq.y * half_h,
        q.z + dq.z * half_h,
        q.w + dq.w * half_h,
    );
    let len = length(integrated);
    if (len > EPSILON) {
        return integrated / len;
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

// Solves one contact's soft normal impulse (biased when `use_bias`, bias-free
// otherwise) then its clamped 2-D friction impulses, mutating the body
// velocities and the contact's accumulated impulses in place.
fn solve_contact(ci: u32, use_bias: bool) {
    let n = contacts[ci].normal.xyz;
    let basis = contact_tangents(n);
    let t1 = basis[0];
    let t2 = basis[1];

    // --- Soft normal impulse ---
    let k_n = effective_mass(ci, n);
    if (k_n > EPSILON) {
        let normal_mass = 1.0 / k_n;
        let separation = -contacts[ci].penetration;
        var bias = 0.0;
        var mass_scale = 1.0;
        var impulse_scale = 0.0;
        if (separation > 0.0) {
            // Speculative: not yet touching, close only the gap this substep.
            bias = separation * params.inv_h;
        } else if (use_bias) {
            // Penetrating: pull the overlap out as a damped spring, leaving
            // `slop` uncorrected and clamping the recovery speed.
            bias = clamp(params.bias_rate * (separation + params.slop), -MAX_RECOVERY_SPEED, 0.0);
            mass_scale = params.mass_scale;
            impulse_scale = params.impulse_scale;
        }
        let vn = dot(relative_velocity(ci), n);
        let impulse = -normal_mass * mass_scale * (vn + bias) - impulse_scale * contacts[ci].normal_impulse;
        let new_impulse = max(contacts[ci].normal_impulse + impulse, 0.0);
        let applied = new_impulse - contacts[ci].normal_impulse;
        contacts[ci].normal_impulse = new_impulse;
        apply_impulse(ci, n * applied);
    }

    // --- Friction impulses (2-D cone, no bias) ---
    let k_t1 = effective_mass(ci, t1);
    let k_t2 = effective_mass(ci, t2);
    if (k_t1 > EPSILON && k_t2 > EPSILON) {
        let v_rel = relative_velocity(ci);
        let delta_0 = -dot(v_rel, t1) / k_t1;
        let delta_1 = -dot(v_rel, t2) / k_t2;
        let max_friction = contacts[ci].friction * contacts[ci].normal_impulse;
        let old_0 = contacts[ci].tangent_impulse_0;
        let old_1 = contacts[ci].tangent_impulse_1;
        var new_0 = old_0 + delta_0;
        var new_1 = old_1 + delta_1;
        let magnitude = sqrt(new_0 * new_0 + new_1 * new_1);
        if (magnitude > max_friction) {
            let scale = max_friction / magnitude;
            new_0 = new_0 * scale;
            new_1 = new_1 * scale;
        }
        let applied_0 = new_0 - old_0;
        let applied_1 = new_1 - old_1;
        contacts[ci].tangent_impulse_0 = new_0;
        contacts[ci].tangent_impulse_1 = new_1;
        apply_impulse(ci, t1 * applied_0 + t2 * applied_1);
    }
}

// Stage 1 (per substep, one thread per body): add gravity * h to every
// translatable body and apply the per-substep linear and angular damping.
@compute @workgroup_size(64)
fn integrate_velocities(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.body_count) {
        return;
    }
    if (!is_movable(i)) {
        return;
    }
    if (inverse_masses[i] > 0.0) {
        let v = (linear_velocities[i].xyz + params.gravity * params.h) * params.linear_damping_scale;
        linear_velocities[i] = vec4<f32>(v, 0.0);
    }
    let w = angular_velocities[i].xyz * params.angular_damping_scale;
    angular_velocities[i] = vec4<f32>(w, 0.0);
}

// Stage 2 (per substep, one thread per contact in a batch): re-apply the
// accumulated impulse to the body velocities, seeding the solve.
@compute @workgroup_size(64)
fn warm_start(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= colour.count) {
        return;
    }
    let ci = colour.start + local;
    let n = contacts[ci].normal.xyz;
    let basis = contact_tangents(n);
    let impulse = n * contacts[ci].normal_impulse
        + basis[0] * contacts[ci].tangent_impulse_0
        + basis[1] * contacts[ci].tangent_impulse_1;
    apply_impulse(ci, impulse);
}

// Stage 3 (per substep, one thread per contact): rotate each contact's world
// arms by its bodies' incremental rotation since the frame start and recompute
// the separation from the bodies' COM translation and arm rotation.
@compute @workgroup_size(64)
fn relinearise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    if (ci >= params.contact_count) {
        return;
    }
    let a = contacts[ci].body_a;
    let b = contacts[ci].body_b;

    let delta_rot_a = quat_mul(orientations[a], quat_conj(initial_orientations[a]));
    let delta_rot_b = quat_mul(orientations[b], quat_conj(initial_orientations[b]));
    let arm_a = quat_rotate(delta_rot_a, initial_arm_a[ci].xyz);
    let arm_b = quat_rotate(delta_rot_b, initial_arm_b[ci].xyz);

    let shift_a = (positions[a].xyz - initial_positions[a].xyz) + (arm_a - initial_arm_a[ci].xyz);
    let shift_b = (positions[b].xyz - initial_positions[b].xyz) + (arm_b - initial_arm_b[ci].xyz);
    let separation = base_separation[ci] + dot(shift_a - shift_b, contacts[ci].normal.xyz);

    contacts[ci].anchor_a = vec4<f32>(arm_a, 0.0);
    contacts[ci].anchor_b = vec4<f32>(arm_b, 0.0);
    contacts[ci].penetration = -separation;
}

// Stage 4 (per substep, one thread per contact in a batch): one biased
// Gauss-Seidel sweep.
@compute @workgroup_size(64)
fn solve_biased(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= colour.count) {
        return;
    }
    solve_contact(colour.start + local, true);
}

// Stage 5 (per substep, one thread per body): advance each body's position and
// orientation by its current velocity over h.
@compute @workgroup_size(64)
fn integrate_positions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.body_count) {
        return;
    }
    if (!is_movable(i)) {
        return;
    }
    if (inverse_masses[i] > 0.0) {
        let p = positions[i].xyz + linear_velocities[i].xyz * params.h;
        positions[i] = vec4<f32>(p, 0.0);
    }
    if (can_rotate(i)) {
        orientations[i] = integrate_orientation(orientations[i], angular_velocities[i].xyz, params.h);
    }
}

// Stage 6 (per substep, one thread per contact in a batch): one bias-free
// relaxation sweep that removes the injected bias velocity.
@compute @workgroup_size(64)
fn solve_relax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= colour.count) {
        return;
    }
    solve_contact(colour.start + local, false);
}

// Final pass (one thread per contact in a batch): restore the elastic bounce
// from the pre-step approach speed for every contact whose approach exceeded the
// restitution threshold and which carried a positive normal impulse.
@compute @workgroup_size(64)
fn apply_restitution(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= colour.count) {
        return;
    }
    let ci = colour.start + local;
    let vn0 = approach_speed[ci];
    if (vn0 >= -params.restitution_threshold || contacts[ci].normal_impulse <= 0.0) {
        return;
    }
    let n = contacts[ci].normal.xyz;
    let k_n = effective_mass(ci, n);
    if (k_n <= EPSILON) {
        return;
    }
    let normal_mass = 1.0 / k_n;
    let vn = dot(relative_velocity(ci), n);
    let impulse = -normal_mass * (vn + contacts[ci].restitution * vn0);
    let new_impulse = max(contacts[ci].normal_impulse + impulse, 0.0);
    let applied = new_impulse - contacts[ci].normal_impulse;
    contacts[ci].normal_impulse = new_impulse;
    apply_impulse(ci, n * applied);
}
