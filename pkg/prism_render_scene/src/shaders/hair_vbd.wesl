//! GPU Vertex Block Descent (VBD) strand solver kernel (high-fidelity slot).
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::solver::simulate_strand_vbd`. A `GPU` is
//! now available, so this shader is compile- and type-checked through the same
//! render-world `ShaderCache` / `wesl` pipeline the runtime uses (see
//! `shader_tests`), with on-device dispatch/perf calibration still awaiting
//! real hardware. It is the stiff-groom high-fidelity solver slot (design §6.7
//! "多求解器插槽": XPBD 基线 / VBD 高保真): braids, dreadlocks and gel-set
//! styling need much higher effective stiffness than the position-based XPBD
//! path (`hair_sim.wesl`) delivers without ballooning iterations or going soft,
//! so those grooms route here.
//!
//! One invocation owns one whole guide strand and advances it by `params.dt`,
//! split into `params.substeps` semi-implicit substeps. Because a single
//! invocation owns the entire strand, the per-vertex sweeps run in the identical
//! sequential Gauss-Seidel order as the CPU golden — each vertex reads its
//! already-updated neighbours within the same sweep — so this twin stays a
//! bit-faithful mirror of the deterministic CPU reference (unlike the
//! self-collision twin, which is honestly Jacobi-relaxed because it fans out one
//! invocation per particle). Each substep:
//!
//! 1. predicts every vertex's inertial target `y = x + v·retain + g·dt_sub^2`
//!    from the *current* implicit velocity, then snapshots `prev = x` so the
//!    next substep measures velocity from here,
//! 2. runs `params.iterations` Gauss-Seidel sweeps that take one exact
//!    per-vertex Newton step against that vertex's own 3x3 Hessian assembled
//!    from inertia + stretch springs + optional bending,
//! 3. projects every free vertex out of the analytic body colliders once the
//!    interior solve has settled (mirrors the CPU `resolve_strand_collisions`
//!    tail-of-substep ordering, identical to `hair_sim.wesl`).
//!
//! Buffer layout matches `hair_sim.wesl` so a groom can switch solver slots
//! without repacking: every particle is a `vec4<f32>` whose `xyz` is the world
//! position and whose `w` is the inverse mass (`0` = pinned root riding the
//! skinned scalp). `prev_positions.xyz` carries the implicit-velocity history.
//! `targets.xyz` is a per-particle scratch buffer holding the substep inertial
//! prediction (the GPU equivalent of the CPU `targets` `Vec`); it is fully
//! overwritten each substep so its incoming contents are irrelevant.
//! `strands[s]` gives each guide's flat offsets into the shared particle /
//! rest-length arrays; callers must size `rest_lengths` to cover
//! `rest_offset + particle_count - 1`.

/// Segments shorter than the square root of this are treated as zero-length
/// (mirrors the CPU `solver::EPS_LEN_SQ`).
const HAIR_VBD_EPS_LEN_SQ: f32 = 1.0e-24;

/// A symmetric 3x3 solve is skipped when the determinant magnitude is below
/// this, so a degenerate (singular) Hessian never divides by ~0 (mirrors the
/// CPU `solver::EPS_DET`).
const HAIR_VBD_EPS_DET: f32 = 1.0e-20;

/// Squared-length epsilon for the collider projection. Mirrors the CPU
/// `collision::EPS_LEN_SQ` so both stages agree on what "degenerate" means.
const HAIR_VBD_COLLIDER_EPS_LEN_SQ: f32 = 1.0e-24;

/// Global VBD parameters shared by every strand in the dispatch. Mirrors the
/// CPU `VbdParams` plus the strand / collider counts that bound the dispatch.
struct HairVbdParams {
    gravity: vec3<f32>,
    dt: f32,
    stretch_stiffness: f32,
    bending_stiffness: f32,
    damping: f32,
    substeps: u32,
    iterations: u32,
    strand_count: u32,
    collider_count: u32,
}

/// Flat layout descriptor for one guide strand. Byte-identical to the
/// `hair_sim.wesl` descriptor (four `u32`, std430 stride 16) so the SAME
/// strands storage buffer feeds either solver twin — swapping the XPBD slot for
/// the VBD slot rebinds only the kernel, not the descriptor upload. `goal_offset`
/// is unused by VBD (it derives inertial targets from the implicit velocity each
/// substep rather than from a per-strand goal-pose array) and is retained solely
/// to keep the descriptor ABI shared.
struct HairStrand {
    particle_offset: u32,
    particle_count: u32,
    rest_offset: u32,
    goal_offset: u32,
}

/// Analytic collision proxy. Mirrors the CPU `collision::Collider` enum packed
/// into two `vec4`s (identical layout to `hair_sim.wesl`): `center_a.xyz` is
/// the sphere center (or capsule endpoint `a`), `center_a.w` is the inflation
/// radius, `end_b.xyz` is the capsule endpoint `b` (unused for a sphere) and
/// `end_b.w` is the kind flag (`< 0.5` = sphere, otherwise capsule).
struct HairCollider {
    center_a: vec4<f32>,
    end_b: vec4<f32>,
}

/// A dense 3x3 matrix in row-major order; used only for per-vertex Hessians.
/// Mirrors the CPU `solver::Mat3` element order exactly so the cofactor solve
/// stays bit-faithful.
struct HairMat3 {
    m: array<f32, 9>,
}

/// One spring's gradient and PSD-projected Hessian contribution for a vertex.
struct HairSpring {
    grad: vec3<f32>,
    hess: HairMat3,
}

/// Result of a 3x3 solve: `value` is the solution when `ok`, otherwise the
/// caller must leave the vertex unmoved this sweep.
struct HairSolve {
    value: vec3<f32>,
    ok: bool,
}

// Particle state: `xyz` = position, `w` = inverse mass (0 = pinned).
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
// Previous position history for the implicit velocity (`xyz` used).
@group(0) @binding(1) var<storage, read_write> prev_positions: array<vec4<f32>>;
// Per-substep inertial-target scratch (`xyz` used); fully rewritten each substep.
@group(0) @binding(2) var<storage, read_write> targets: array<vec4<f32>>;
// Per-segment rest lengths, flat across all strands.
@group(0) @binding(3) var<storage, read> rest_lengths: array<f32>;
// Per-strand flat offsets.
@group(0) @binding(4) var<storage, read> strands: array<HairStrand>;
// Analytic body colliders applied after every substep's interior solve.
@group(0) @binding(5) var<storage, read> colliders: array<HairCollider>;

var<immediate> params: HairVbdParams;

// A scalar multiple of the identity. Mirrors CPU `Mat3::scaled_identity`.
fn hair_mat3_scaled_identity(s: f32) -> HairMat3 {
    var out: HairMat3;
    out.m[0] = s;
    out.m[1] = 0.0;
    out.m[2] = 0.0;
    out.m[3] = 0.0;
    out.m[4] = s;
    out.m[5] = 0.0;
    out.m[6] = 0.0;
    out.m[7] = 0.0;
    out.m[8] = s;
    return out;
}

// Elementwise sum. Mirrors CPU `Mat3::add`.
fn hair_mat3_add(a: HairMat3, b: HairMat3) -> HairMat3 {
    var out: HairMat3;
    for (var i = 0u; i < 9u; i = i + 1u) {
        out.m[i] = a.m[i] + b.m[i];
    }
    return out;
}

// The outer product `v vᵀ` scaled by `s`. Mirrors CPU `Mat3::scaled_outer`
// (row-major element order).
fn hair_mat3_scaled_outer(v: vec3<f32>, s: f32) -> HairMat3 {
    var out: HairMat3;
    out.m[0] = s * v.x * v.x;
    out.m[1] = s * v.x * v.y;
    out.m[2] = s * v.x * v.z;
    out.m[3] = s * v.y * v.x;
    out.m[4] = s * v.y * v.y;
    out.m[5] = s * v.y * v.z;
    out.m[6] = s * v.z * v.x;
    out.m[7] = s * v.z * v.y;
    out.m[8] = s * v.z * v.z;
    return out;
}

// Solves `mat * x = rhs` by explicit cofactor inversion. Returns `ok = false`
// when the matrix is (near-)singular or the result is non-finite so the caller
// leaves the vertex unmoved instead of producing a bad position. Mirrors CPU
// `Mat3::solve` cofactor formulas and operation order bit-for-bit.
fn hair_mat3_solve(mat: HairMat3, rhs: vec3<f32>) -> HairSolve {
    var out: HairSolve;
    out.value = vec3<f32>(0.0, 0.0, 0.0);
    out.ok = false;

    let m = mat.m;
    let c00 = m[4] * m[8] - m[5] * m[7];
    let c01 = m[5] * m[6] - m[3] * m[8];
    let c02 = m[3] * m[7] - m[4] * m[6];
    let det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    if abs(det) < HAIR_VBD_EPS_DET {
        return out;
    }
    let inv_det = 1.0 / det;
    let c10 = m[2] * m[7] - m[1] * m[8];
    let c11 = m[0] * m[8] - m[2] * m[6];
    let c12 = m[1] * m[6] - m[0] * m[7];
    let c20 = m[1] * m[5] - m[2] * m[4];
    let c21 = m[2] * m[3] - m[0] * m[5];
    let c22 = m[0] * m[4] - m[1] * m[3];
    let x = (c00 * rhs.x + c10 * rhs.y + c20 * rhs.z) * inv_det;
    let y = (c01 * rhs.x + c11 * rhs.y + c21 * rhs.z) * inv_det;
    let z = (c02 * rhs.x + c12 * rhs.y + c22 * rhs.z) * inv_det;
    // Finite guard: NaN fails `v == v`, infinities fail the magnitude bound.
    let finite = (x == x) && (y == y) && (z == z)
        && abs(x) < 3.0e38 && abs(y) < 3.0e38 && abs(z) < 3.0e38;
    if !finite {
        return out;
    }
    out.value = vec3<f32>(x, y, z);
    out.ok = true;
    return out;
}

// Accumulates one spring's gradient and PSD-projected Hessian contribution for
// the vertex at `x`, connected to `other` with rest length `rest` and stiffness
// `k`. Mirrors CPU `accumulate_spring`: a (near-)zero-length spring contributes
// nothing, and the Hessian drops its indefinite part under compression so the
// per-vertex Newton step stays a descent direction.
fn hair_accumulate_spring(x: vec3<f32>, other: vec3<f32>, rest: f32, k: f32) -> HairSpring {
    var out: HairSpring;
    out.grad = vec3<f32>(0.0, 0.0, 0.0);
    out.hess = hair_mat3_scaled_identity(0.0);

    let d = x - other;
    let len_sq = dot(d, d);
    if len_sq < HAIR_VBD_EPS_LEN_SQ {
        return out;
    }
    let len = sqrt(len_sq);
    let n = d * (1.0 / len);
    out.grad = n * (k * (len - rest));
    let tangential = max(1.0 - rest / len, 0.0);
    // k·tangential·I + k·(1 - tangential)·nnᵀ
    let iso = hair_mat3_scaled_identity(k * tangential);
    let outer = hair_mat3_scaled_outer(n, k * (1.0 - tangential));
    out.hess = hair_mat3_add(iso, outer);
    return out;
}

// Pushes `point` out to the surface of the sphere `(center, radius)`. Mirrors
// CPU `collision::push_out_sphere` (identical to `hair_sim.wesl`).
fn hair_push_out_sphere(center: vec3<f32>, radius: f32, point: vec3<f32>) -> vec3<f32> {
    if radius <= 0.0 {
        return point;
    }
    let delta = point - center;
    let dist_sq = dot(delta, delta);
    if dist_sq >= radius * radius {
        return point;
    }
    if dist_sq <= HAIR_VBD_COLLIDER_EPS_LEN_SQ {
        return center + vec3<f32>(0.0, radius, 0.0);
    }
    let dir = delta * (1.0 / sqrt(dist_sq));
    return center + dir * radius;
}

// Returns the point on segment `a`..`b` closest to `point`. Mirrors CPU
// `collision::closest_point_on_segment`; a zero-length segment returns `a`.
fn hair_closest_point_on_segment(a: vec3<f32>, b: vec3<f32>, point: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let len_sq = dot(ab, ab);
    if len_sq <= HAIR_VBD_COLLIDER_EPS_LEN_SQ {
        return a;
    }
    let t = clamp(dot(point - a, ab) / len_sq, 0.0, 1.0);
    return a + ab * t;
}

// Projects `point` out of one analytic collider. Mirrors CPU `Collider::push_out`.
fn hair_push_out_collider(collider: HairCollider, point: vec3<f32>) -> vec3<f32> {
    let radius = collider.center_a.w;
    if collider.end_b.w < 0.5 {
        return hair_push_out_sphere(collider.center_a.xyz, radius, point);
    }
    let closest = hair_closest_point_on_segment(collider.center_a.xyz, collider.end_b.xyz, point);
    return hair_push_out_sphere(closest, radius, point);
}

// Projects every free particle of the strand out of every collider, in order.
// Mirrors CPU `resolve_strand_collisions`: pinned particles are never moved, an
// empty collider set is a no-op, and colliders are applied in array order so
// the last one to push wins (deterministic).
fn hair_resolve_collisions(off: u32, count: u32) {
    if params.collider_count == 0u {
        return;
    }
    for (var i = 0u; i < count; i = i + 1u) {
        let p = off + i;
        if positions[p].w <= 0.0 {
            continue;
        }
        var pos = positions[p].xyz;
        for (var c = 0u; c < params.collider_count; c = c + 1u) {
            pos = hair_push_out_collider(colliders[c], pos);
        }
        positions[p] = vec4<f32>(pos, positions[p].w);
    }
}

@compute @workgroup_size(64)
fn simulate_strand_vbd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if s >= params.strand_count {
        return;
    }
    // Mirror CPU guards: empty strand, zero substeps, or non-positive /
    // non-finite dt is a no-op.
    let dt_finite = (params.dt == params.dt) && abs(params.dt) < 3.0e38;
    if params.substeps == 0u || params.dt <= 0.0 || !dt_finite {
        return;
    }

    let strand = strands[s];
    let count = strand.particle_count;
    if count == 0u {
        return;
    }
    let off = strand.particle_offset;
    let rest_off = strand.rest_offset;

    let sub_dt = params.dt / f32(params.substeps);
    let sub_dt_sq = sub_dt * sub_dt;
    let velocity_retain = 1.0 - clamp(params.damping, 0.0, 1.0);
    let stretch = max(params.stretch_stiffness, 0.0);
    let bending = max(params.bending_stiffness, 0.0);
    let gravity_step = params.gravity * sub_dt_sq;

    for (var step = 0u; step < params.substeps; step = step + 1u) {
        // 1. Predict each vertex's inertial target from the current implicit
        //    velocity, then snapshot prev = position.
        for (var i = 0u; i < count; i = i + 1u) {
            let p = off + i;
            let pos = positions[p].xyz;
            let velocity = pos - prev_positions[p].xyz;
            let y = pos + velocity * velocity_retain + gravity_step;
            targets[p] = vec4<f32>(y, 0.0);
        }
        for (var i = 0u; i < count; i = i + 1u) {
            let p = off + i;
            prev_positions[p] = vec4<f32>(positions[p].xyz, prev_positions[p].w);
        }

        // 2. Gauss-Seidel vertex sweeps: one exact Newton step per free vertex.
        for (var it = 0u; it < params.iterations; it = it + 1u) {
            for (var i = 0u; i < count; i = i + 1u) {
                let p = off + i;
                if positions[p].w <= 0.0 {
                    continue; // pinned: fixed at its skinned pose.
                }
                let x = positions[p].xyz;
                let mass = 1.0 / positions[p].w;
                let inertia = mass / sub_dt_sq;

                var grad = (x - targets[p].xyz) * inertia;
                var hess = hair_mat3_scaled_identity(inertia);

                // Segment to the next particle uses rest_lengths[rest_off + i].
                if i + 1u < count {
                    let spring = hair_accumulate_spring(
                        x,
                        positions[off + i + 1u].xyz,
                        rest_lengths[rest_off + i],
                        stretch,
                    );
                    grad = grad + spring.grad;
                    hess = hair_mat3_add(hess, spring.hess);
                }
                // Segment to the previous particle uses rest_lengths[rest_off + i - 1].
                if i >= 1u {
                    let spring = hair_accumulate_spring(
                        x,
                        positions[off + i - 1u].xyz,
                        rest_lengths[rest_off + i - 1u],
                        stretch,
                    );
                    grad = grad + spring.grad;
                    hess = hair_mat3_add(hess, spring.hess);
                }
                // Bending: pull an interior vertex toward its neighbours' midpoint.
                if bending > 0.0 && i >= 1u && i + 1u < count {
                    let mid = (positions[off + i - 1u].xyz + positions[off + i + 1u].xyz) * 0.5;
                    grad = grad + (x - mid) * bending;
                    hess = hair_mat3_add(hess, hair_mat3_scaled_identity(bending));
                }

                let solved = hair_mat3_solve(hess, grad);
                if solved.ok {
                    positions[p] = vec4<f32>(x - solved.value, positions[p].w);
                }
            }
        }

        // 3. Project free particles out of the body colliders once the interior
        //    solve has settled (mirrors CPU tail-of-substep ordering).
        hair_resolve_collisions(off, count);
    }
}
