//! Vertex Block Descent (VBD) cloth solve on the GPU.
//!
//! This is the GPU twin of the CPU reference
//! prism_render_architecture::cloth::vbd::solve_cloth_vbd_colored. VBD advances a
//! cloth patch with an implicit backward-Euler step: each substep predicts an
//! inertial target from the current velocity and gravity, relaxes the mesh with
//! a handful of Gauss-Seidel Newton sweeps, then recovers velocity. Each sweep
//! takes one exact per-vertex Newton step against that vertex's own inertia plus
//! constraint 3x3 Hessian.
//!
//! The natural-order CPU sweep is inherently sequential (a vertex reads its
//! neighbors' current positions). A GPU dispatch honors that ordering per color
//! instead: the vertices are graph-colored so that no two vertices sharing a
//! distance constraint land in the same color, so one color's vertices read only
//! other colors and can be relaxed in parallel with no read-after-write hazard.
//! The host dispatches one sweep per color (parallel within a color, serial
//! across colors), reproducing a valid Gauss-Seidel sweep whose per-vertex
//! result is independent of the intra-color order. That is what makes this
//! kernel and solve_cloth_vbd_colored agree to within float32 rounding.
//!
//! Three compute entry points:
//!   cloth_vbd_predict  - per particle: snapshot previous, predict inertial target.
//!   cloth_vbd_sweep    - per current-color vertex: one Newton relaxation step.
//!   cloth_vbd_velocity - per particle: recover velocity, freeze pinned.
//!
//! The per-vertex constraint accumulation order follows the host CSR adjacency,
//! which mirrors the CPU build_adjacency (ascending constraint index), so the
//! summed gradient and Hessian match the golden term for term.

/// Squared-length floor below which a distance direction is left unnormalized,
/// so normalization never divides by ~0 (mirrors the CPU EPS_LEN_SQ).
const CLOTH_VBD_EPS_LEN_SQ: f32 = 1.0e-12;
/// Determinant floor below which a 3x3 Hessian is treated as singular and the
/// vertex is not moved this sweep (mirrors the CPU EPS_DET).
const CLOTH_VBD_EPS_DET: f32 = 1.0e-20;
/// Upper bound on per-constraint stiffness. A rigid constraint (compliance 0)
/// would imply infinite stiffness; capping keeps the Hessian finite while still
/// dominating inertia (mirrors the CPU MAX_STIFFNESS).
const CLOTH_VBD_MAX_STIFFNESS: f32 = 1.0e9;

/// One-sided constraint-kind tags (LRA / tether). A one-sided edge that is slack
/// or at rest contributes nothing, matching the CPU is_one_sided gate.
const CLOTH_CONSTRAINT_LRA: u32 = 3u;
const CLOTH_CONSTRAINT_TETHER: u32 = 4u;

/// One distance constraint. Byte-compatible with GpuClothConstraint in abi.rs
/// (20 bytes) and ClothConstraint in cloth_sim.wesl.
struct ClothConstraint {
    /// First endpoint particle index.
    a: u32,
    /// Second endpoint particle index.
    b: u32,
    /// Rest length the Newton step drives the edge toward.
    rest_length: f32,
    /// XPBD compliance (inverse stiffness); host pre-clamps negatives to 0.
    compliance: f32,
    /// Constraint-kind tag (CLOTH_CONSTRAINT_*); only one-sided kinds gate.
    kind: u32,
}

/// Per-pass scalars shared by the three kernels. gravity_step and retain feed
/// prediction; dt_sub_sq feeds the Newton step; inv_dt_sub feeds velocity
/// recovery; color_begin / color_end select the current color's slice of the
/// color-major vertex order for the sweep kernel.
struct VbdParams {
    /// Gravity acceleration times dt_sub squared (the per-substep inertial pull).
    gravity_step: vec3<f32>,
    /// Velocity retention 1 - damping applied before prediction.
    retain: f32,
    /// Substep timestep dt / substeps.
    dt_sub: f32,
    /// Reciprocal of dt_sub, used by velocity recovery.
    inv_dt_sub: f32,
    /// dt_sub squared, used to form the inertial coefficient and stiffness.
    dt_sub_sq: f32,
    /// Number of particles (bound of the per-particle kernels).
    particle_count: u32,
    /// First index into color_order for the current color (inclusive).
    color_begin: u32,
    /// One past the last index into color_order for the current color.
    color_end: u32,
    /// Padding to a 16-byte multiple; unread.
    pad0: u32,
    /// Padding to a 16-byte multiple; unread.
    pad1: u32,
}

// Particle world position (xyz) and inverse mass (w; <= 0 = pinned).
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
// Explicit velocity (xyz used).
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
// Pre-substep position snapshot for velocity recovery (xyz used).
@group(0) @binding(2) var<storage, read_write> previous: array<vec4<f32>>;
// Predicted inertial target per particle (xyz used).
@group(0) @binding(3) var<storage, read_write> targets: array<vec4<f32>>;
// Distance constraints (shared with the XPBD path).
@group(0) @binding(4) var<storage, read> constraints: array<ClothConstraint>;
// CSR row offsets into csr_entries; length particle_count + 1.
@group(0) @binding(5) var<storage, read> csr_offsets: array<u32>;
// CSR entries: constraint indices touching each vertex, ascending per vertex.
@group(0) @binding(6) var<storage, read> csr_entries: array<u32>;
// Color-major vertex order; a color is a contiguous slice [color_begin, color_end).
@group(0) @binding(7) var<storage, read> color_order: array<u32>;
// Per-pass scalars and the current color window.
@group(0) @binding(8) var<uniform> params: VbdParams;

/// Effective inverse mass: <= 0 (pinned) reads as 0.
fn cloth_vbd_inverse_mass(w: f32) -> f32 {
    return max(w, 0.0);
}

/// XPBD compliance to VBD energy stiffness k for one substep. compliance is
/// host-clamped to >= 0, so a positive value inverts to a capped stiffness and a
/// rigid (0) constraint reads as the cap (mirrors CPU constraint_stiffness).
fn cloth_vbd_stiffness(compliance: f32, dt_sub_sq: f32) -> f32 {
    if compliance > 0.0 {
        return min(1.0 / (compliance * dt_sub_sq), CLOTH_VBD_MAX_STIFFNESS);
    }
    return CLOTH_VBD_MAX_STIFFNESS;
}

/// Per-particle inertial prediction: snapshot the current position, then predict
/// target y = x + (v * retain) * dt_sub + gravity_step. Mirrors CPU predict_targets.
@compute @workgroup_size(64)
fn cloth_vbd_predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.particle_count {
        return;
    }
    let p = positions[i];
    previous[i] = vec4<f32>(p.xyz, previous[i].w);
    let velocity = velocities[i].xyz * params.retain;
    let predicted = p.xyz + velocity * params.dt_sub + params.gravity_step;
    targets[i] = vec4<f32>(predicted, targets[i].w);
}

/// Per-current-color-vertex Newton relaxation. Reads the vertex's CSR adjacency
/// (ascending constraint index) and takes one exact per-vertex Newton step
/// against inertia + the PSD-projected constraint Hessian. A pinned vertex is
/// left untouched. Mirrors CPU relax_vertex.
@compute @workgroup_size(64)
fn cloth_vbd_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    let slot = params.color_begin + local;
    if slot >= params.color_end {
        return;
    }
    let v = color_order[slot];
    if v >= params.particle_count {
        return;
    }
    let w = positions[v].w;
    if w <= 0.0 {
        return;
    }
    let x = positions[v].xyz;
    let mass = 1.0 / w;
    let inertia = mass / params.dt_sub_sq;

    var grad = (x - targets[v].xyz) * inertia;
    // Row-major 3x3 Hessian entries [m00, m01, m02, m10, m11, m12, m20, m21, m22].
    var m0 = inertia;
    var m1 = 0.0;
    var m2 = 0.0;
    var m3 = 0.0;
    var m4 = inertia;
    var m5 = 0.0;
    var m6 = 0.0;
    var m7 = 0.0;
    var m8 = inertia;

    let row_begin = csr_offsets[v];
    let row_end = csr_offsets[v + 1u];
    for (var e = row_begin; e < row_end; e = e + 1u) {
        let con = constraints[csr_entries[e]];
        var other = con.a;
        if con.a == v {
            other = con.b;
        }
        let d = x - positions[other].xyz;
        let len_sq = d.x * d.x + d.y * d.y + d.z * d.z;
        if len_sq < CLOTH_VBD_EPS_LEN_SQ {
            continue;
        }
        let len = sqrt(len_sq);
        let one_sided = con.kind == CLOTH_CONSTRAINT_LRA || con.kind == CLOTH_CONSTRAINT_TETHER;
        if one_sided && len <= con.rest_length {
            continue;
        }
        let inv_len = 1.0 / len;
        let nx = d.x * inv_len;
        let ny = d.y * inv_len;
        let nz = d.z * inv_len;
        let k = cloth_vbd_stiffness(con.compliance, params.dt_sub_sq);
        let coef = k * (len - con.rest_length);
        grad.x = grad.x + nx * coef;
        grad.y = grad.y + ny * coef;
        grad.z = grad.z + nz * coef;
        // Direct `rest / len` division (one IEEE fdiv), not `rest * inv_len`,
        // so the tangential coefficient is bit-identical to the CPU golden
        // `accumulate_constraint`'s `(1.0 - rest / len).max(0.0)`; the normal
        // above keeps `d * inv_len` to mirror the golden `d.scale(1.0 / len)`.
        let tangential = max(1.0 - con.rest_length / len, 0.0);
        let ktang = k * tangential;
        // k * tangential * I first (mirrors the CPU scaled_identity add).
        m0 = m0 + ktang;
        m4 = m4 + ktang;
        m8 = m8 + ktang;
        // then k * (1 - tangential) * n n^T (mirrors the CPU scaled_outer add).
        let s = k * (1.0 - tangential);
        m0 = m0 + s * nx * nx;
        m1 = m1 + s * nx * ny;
        m2 = m2 + s * nx * nz;
        m3 = m3 + s * ny * nx;
        m4 = m4 + s * ny * ny;
        m5 = m5 + s * ny * nz;
        m6 = m6 + s * nz * nx;
        m7 = m7 + s * nz * ny;
        m8 = m8 + s * nz * nz;
    }

    // Solve m * delta = grad by explicit cofactor inversion (mirrors CPU Mat3::solve).
    let c00 = m4 * m8 - m5 * m7;
    let c01 = m5 * m6 - m3 * m8;
    let c02 = m3 * m7 - m4 * m6;
    let det = m0 * c00 + m1 * c01 + m2 * c02;
    if abs(det) < CLOTH_VBD_EPS_DET {
        return;
    }
    let inv_det = 1.0 / det;
    let c10 = m2 * m7 - m1 * m8;
    let c11 = m0 * m8 - m2 * m6;
    let c12 = m1 * m6 - m0 * m7;
    let c20 = m1 * m5 - m2 * m4;
    let c21 = m2 * m3 - m0 * m5;
    let c22 = m0 * m4 - m1 * m3;
    let dx = (c00 * grad.x + c10 * grad.y + c20 * grad.z) * inv_det;
    let dy = (c01 * grad.x + c11 * grad.y + c21 * grad.z) * inv_det;
    let dz = (c02 * grad.x + c12 * grad.y + c22 * grad.z) * inv_det;
    let finite = (dx == dx) && (dy == dy) && (dz == dz)
        && abs(dx) < 3.4e38 && abs(dy) < 3.4e38 && abs(dz) < 3.4e38;
    if !finite {
        return;
    }
    positions[v] = vec4<f32>(x.x - dx, x.y - dy, x.z - dz, w);
}

/// Per-particle velocity recovery: v = (x - x_prev) / dt_sub for free particles,
/// zero for pinned particles. Mirrors CPU recover_velocity.
@compute @workgroup_size(64)
fn cloth_vbd_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.particle_count {
        return;
    }
    let w = positions[i].w;
    if w <= 0.0 {
        velocities[i] = vec4<f32>(0.0, 0.0, 0.0, velocities[i].w);
        return;
    }
    let recovered = (positions[i].xyz - previous[i].xyz) * params.inv_dt_sub;
    velocities[i] = vec4<f32>(recovered, velocities[i].w);
}
