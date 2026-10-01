// Velocity-level sequential-impulse 6-DOF rigid-body contact solver, device
// kernel.
//
// Byte-for-byte-intent twin of the CPU reference in `src/rigid/contact_cpu.rs`.
// The solve is colour-ordered: the host partitions contacts into batches whose
// movable endpoints are disjoint, uploads them grouped by batch, and dispatches
// one `warm_start`/`solve` pass per batch so same-batch threads never write the
// same body. Static bodies (zero inverse mass, zero inverse inertia) may be
// shared across a batch because the write guards below skip them — their
// impulse response is zero, so skipping the store changes no value and removes
// the only cross-thread write hazard.
//
// Three entry points mirror the three phases of the CPU twin:
//   * `prepare`     — one thread per contact captures the pre-impulse approach
//                     speed used by the restitution bias.
//   * `warm_start`  — one thread per contact in a batch re-applies the
//                     accumulated impulse as the solver's initial guess.
//   * `solve`       — one thread per contact in a batch runs a single
//                     sequential-impulse sweep (normal then 2-D friction cone).
//
// The arithmetic below performs the identical f32 operations in the identical
// order as the CPU twin; only device floating-point reassociation (fused
// multiply-add, differing division/sqrt rounding) separates the two, which the
// parity test bounds with a tight tolerance rather than exact equality.
//
// Provenance: velocity-level sequential impulses with a 2-D friction cone,
// Baumgarte position bias, restitution, and warm starting (Catto / Box2D,
// Bullet; standard constrained rigid-body dynamics). Standard wgpu compute
// dispatch. No Unreal Engine source or derived code.

const EPSILON: f32 = 1.1920929e-7; // f32::EPSILON, matching the CPU guard.
const TANGENT_THRESHOLD: f32 = 0.57735026; // ~= 1/sqrt(3), the CPU seed cutoff.

struct Params {
    baumgarte: f32,
    slop: f32,
    restitution_threshold: f32,
    inv_dt: f32,
    contact_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct ColourParams {
    start: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
};

// Device-packed contact; layout matches `GpuRigidContact` in
// `src/rigid/contact.rs` (80 bytes). The three impulse accumulators are
// read-write so warm-start seeds and converged impulses round-trip here.
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
@group(0) @binding(3) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(4) var<storage, read> inverse_inertias: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> orientations: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(7) var<storage, read_write> vn_initial: array<f32>;

@group(1) @binding(0) var<uniform> colour: ColourParams;

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

// Applies the world-space inverse inertia R diag(inv_inertia) R^T to v.
fn world_inv_inertia_apply(q: vec4<f32>, inv_inertia: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let body = quat_rotate(quat_conj(q), v);
    let scaled = inv_inertia * body;
    return quat_rotate(q, scaled);
}

// Right-handed orthonormal tangent basis spanning the plane perpendicular to n.
// Identical branchless Frisvad / Catto construction as the CPU twin.
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

// Contact effective mass along the unit direction d, including both bodies'
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

@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    if (ci >= params.contact_count) {
        return;
    }
    vn_initial[ci] = dot(relative_velocity(ci), contacts[ci].normal.xyz);
}

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

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= colour.count) {
        return;
    }
    let ci = colour.start + local;
    let n = contacts[ci].normal.xyz;
    let basis = contact_tangents(n);
    let t1 = basis[0];
    let t2 = basis[1];

    // --- Normal impulse ---
    let k_n = effective_mass(ci, n);
    if (k_n > EPSILON) {
        let vn = dot(relative_velocity(ci), n);
        let position_bias = params.baumgarte * params.inv_dt
            * max(contacts[ci].penetration - params.slop, 0.0);
        var restitution_bias = 0.0;
        if (vn_initial[ci] < -params.restitution_threshold) {
            restitution_bias = -contacts[ci].restitution * vn_initial[ci];
        }
        let bias_velocity = max(position_bias, restitution_bias);
        let lambda = (bias_velocity - vn) / k_n;
        let new_impulse = max(contacts[ci].normal_impulse + lambda, 0.0);
        let applied = new_impulse - contacts[ci].normal_impulse;
        contacts[ci].normal_impulse = new_impulse;
        apply_impulse(ci, n * applied);
    }

    // --- Friction impulses (2-D cone, clamped to friction * normal_impulse) ---
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
