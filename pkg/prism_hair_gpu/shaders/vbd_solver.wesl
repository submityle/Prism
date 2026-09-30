// Vertex Block Descent (VBD) strand solver twin: advances on-device the same
// variational strand solve the CPU golden `simulate_strand_vbd` computes, one
// thread per guide strand.
//
// XPBD (see `guide_solver.wesl`) is the cheap default, but *stiff* grooms —
// braids, dreadlocks, wax/gel-set styling — need much higher effective
// stiffness than position-based projection delivers. VBD minimizes the same
// backward-Euler incremental potential a full Newton solve would, but block
// locally: it sweeps the vertices in Gauss-Seidel order and takes one exact
// per-vertex Newton step against that vertex's own 3x3 Hessian each iteration
// (design 6.7 "多求解器插槽"). The CPU golden for one strand is
// `simulate_strand_vbd` (in `prism_render_architecture::hair::solver`); a batch
// runs it on each strand slice. This kernel is the on-device twin: one thread
// owns one strand's contiguous particle range and runs the full
// substep/iteration schedule over it, so a passing real-device parity test is
// direct evidence the ported kernel advances the strands to the same state as
// the reference — not merely that its shader compiles.
//
// This kernel mirrors that reference term for term, per substep:
//
//   predict_targets();      // y = x + (x - prev)*velocity_retain + g*sub_dt^2
//   snapshot_prev();        // prev = x (all particles)
//   for _ in 0..iterations {
//     for each free vertex i {
//       grad = (x - target[i]) * inertia;   hess = inertia*I
//       accumulate_spring(next segment rest_lengths[i]);
//       accumulate_spring(prev segment rest_lengths[i-1]);
//       add bending toward neighbor midpoint (interior only);
//       if hess solvable { x = x - hess^-1 * grad; }
//     }
//   }
//   resolve_strand_collisions();            // analytic body push-out
//
// Because every thread mutates a disjoint particle range, the in-place
// Gauss-Seidel state updates (and the per-strand target scratch writes) are
// race free without any barrier, and the read-after-write ordering inside a
// strand matches the scalar reference sweep for sweep. All host-derived scalars
// (`gravity_step = gravity * sub_dt^2`, `velocity_retain = 1 - clamp(damping)`,
// `sub_dt^2`, the clamped `stretch`/`bending`) are precomputed on the host in
// the reference's evaluation order and uploaded in `Params`, so the device
// never re-derives them and cannot diverge on operation ordering. The
// per-strand rest slice is all-or-nothing in the reference, reproduced here as
// one `has_rest` flag per strand.
//
// Every guard is reproduced: pinned particles (`inverse_mass <= 0`) are never
// swept and never moved by a constraint (their `prev` is still snapshot to
// `pos` each substep, matching the reference loop over all particles), a
// (near-)singular vertex Hessian (`|det| < EPS_DET`) leaves the vertex in place
// for that sweep instead of dividing by ~0, a compressed spring drops the
// indefinite Hessian part (PSD projection) so the Newton step stays a descent
// direction, and the collider push-out reproduces `Collider::push_out` (sphere
// and capsule) exactly.
//
// It uses only the portable core-WGSL subset (sqrt via length / min / max /
// clamp / dot / mul / add) - no exp, no pow, no optional feature - so this twin
// runs unmodified on Metal, Vulkan and DX12. It is not bit-exact with the
// scalar CPU reference: a GPU may fuse a multiply-add the reference leaves
// separate, and the perturbation compounds over the iterated Newton sweep (plus
// a 3x3 cofactor inverse per vertex). The parity test therefore asserts a
// per-component tolerance rather than exact equality.
//
// Provenance: standard Vertex Block Descent strand solver (inertia + stretch +
// bending block-local Newton) plus analytic sphere/capsule push-out; no Unreal
// Engine source or derived code.

// Matches `solver::EPS_LEN_SQ` / `collision::EPS_LEN_SQ`: vectors shorter than
// the square root of this are treated as zero-length.
const EPS_LEN_SQ: f32 = 1.0e-24;
// Matches `solver::EPS_DET`: a symmetric 3x3 solve with a determinant below
// this is skipped so a singular Hessian never divides by ~0.
const EPS_DET: f32 = 1.0e-20;

// Collider kind discriminants, matching the CPU upload encoding.
const KIND_SPHERE: u32 = 0u;
const KIND_CAPSULE: u32 = 1u;

// Solve-wide parameters, all host-derived in the reference's evaluation order.
// `48`-byte scalar-packed layout matching the `bytemuck` upload struct `Params`.
struct Params {
    // `gravity * sub_dt^2`: the per-substep gravity displacement.
    gsx: f32,
    gsy: f32,
    gsz: f32,
    // `1 - clamp(damping, 0, 1)`: implicit-velocity retention fraction.
    velocity_retain: f32,
    // `sub_dt^2 = (dt / substeps)^2`: the inertia term uses `mass / sub_dt^2`.
    sub_dt_sq: f32,
    // `max(stretch_stiffness, 0)`: edge (stretch) energy weight.
    stretch: f32,
    // `max(bending_stiffness, 0)`: bending energy weight (`0` disables it).
    bending: f32,
    // Padding to a `16`-byte multiple.
    pad0: f32,
    // Number of semi-implicit substeps.
    substeps: u32,
    // Gauss-Seidel vertex sweeps per substep.
    iterations: u32,
    // Number of strand descriptors in `strands`.
    strand_count: u32,
    // Number of active colliders in `colliders` (`0` disables collision).
    collider_count: u32,
}

// One guide-strand descriptor. `16`-byte stride matching the `bytemuck` upload
// struct `Strand`: the base particle index of this strand in the flat `state`
// pool, its particle count, whether its rest-length companion slice is present
// (all-or-nothing per strand), and one padding word.
struct Strand {
    point_offset: u32,
    point_count: u32,
    has_rest: u32,
    pad0: u32,
}

// One analytic collider. `32`-byte stride matching the `bytemuck` upload struct
// `GpuCollider`. For a sphere `(ax, ay, az)` is the center; for a capsule
// `(ax, ay, az)`..`(bx, by, bz)` is the segment.
struct Collider {
    kind: u32,
    radius: f32,
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
}

// A dense 3x3 matrix in row-major order; used only for per-vertex Hessians.
struct Mat3 {
    m: array<f32, 9>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> strands: array<Strand>;
// Per-particle state, stride `8` f32: position xyz, prev_position xyz,
// inverse_mass, rest_length (the length of the segment leaving this particle).
@group(0) @binding(2) var<storage, read_write> state: array<f32>;
// Per-particle inertial-target scratch, stride `3` f32. Written then read by
// the owning thread each substep; disjoint ranges keep it race free.
@group(0) @binding(3) var<storage, read_write> targets: array<f32>;
@group(0) @binding(4) var<storage, read> colliders: array<Collider>;

fn load_pos(gi: u32) -> vec3<f32> {
    let b = 8u * gi;
    return vec3<f32>(state[b], state[b + 1u], state[b + 2u]);
}

fn store_pos(gi: u32, v: vec3<f32>) {
    let b = 8u * gi;
    state[b] = v.x;
    state[b + 1u] = v.y;
    state[b + 2u] = v.z;
}

fn load_prev(gi: u32) -> vec3<f32> {
    let b = 8u * gi;
    return vec3<f32>(state[b + 3u], state[b + 4u], state[b + 5u]);
}

fn store_prev(gi: u32, v: vec3<f32>) {
    let b = 8u * gi;
    state[b + 3u] = v.x;
    state[b + 4u] = v.y;
    state[b + 5u] = v.z;
}

fn load_inv_mass(gi: u32) -> f32 {
    return state[8u * gi + 6u];
}

fn load_rest(gi: u32) -> f32 {
    return state[8u * gi + 7u];
}

fn load_target(gi: u32) -> vec3<f32> {
    let b = 3u * gi;
    return vec3<f32>(targets[b], targets[b + 1u], targets[b + 2u]);
}

fn store_target(gi: u32, v: vec3<f32>) {
    let b = 3u * gi;
    targets[b] = v.x;
    targets[b + 1u] = v.y;
    targets[b + 2u] = v.z;
}

// A scalar multiple of the 3x3 identity.
fn mat3_scaled_identity(s: f32) -> Mat3 {
    var out: Mat3;
    out.m = array<f32, 9>(s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s);
    return out;
}

// Componentwise sum of two 3x3 matrices.
fn mat3_add(a: Mat3, b: Mat3) -> Mat3 {
    var out: Mat3;
    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        out.m[i] = a.m[i] + b.m[i];
    }
    return out;
}

// The outer product `v vᵀ` scaled by `s`, row-major.
fn mat3_scaled_outer(v: vec3<f32>, s: f32) -> Mat3 {
    var out: Mat3;
    out.m = array<f32, 9>(
        s * v.x * v.x,
        s * v.x * v.y,
        s * v.x * v.z,
        s * v.y * v.x,
        s * v.y * v.y,
        s * v.y * v.z,
        s * v.z * v.x,
        s * v.z * v.y,
        s * v.z * v.z,
    );
    return out;
}

// The solved vertex delta plus a flag. `ok = 0u` means the Hessian was
// (near-)singular, so the caller must leave the vertex in place this sweep.
struct SolveResult {
    ok: u32,
    v: vec3<f32>,
}

// Solves `m * x = rhs` by explicit cofactor inversion, mirroring `Mat3::solve`.
fn mat3_solve(m: Mat3, rhs: vec3<f32>) -> SolveResult {
    var res: SolveResult;
    res.ok = 0u;
    res.v = vec3<f32>(0.0, 0.0, 0.0);
    let e = m.m;
    let c00 = e[4] * e[8] - e[5] * e[7];
    let c01 = e[5] * e[6] - e[3] * e[8];
    let c02 = e[3] * e[7] - e[4] * e[6];
    let det = e[0] * c00 + e[1] * c01 + e[2] * c02;
    if (abs(det) < EPS_DET) {
        return res;
    }
    let inv_det = 1.0 / det;
    let c10 = e[2] * e[7] - e[1] * e[8];
    let c11 = e[0] * e[8] - e[2] * e[6];
    let c12 = e[1] * e[6] - e[0] * e[7];
    let c20 = e[1] * e[5] - e[2] * e[4];
    let c21 = e[2] * e[3] - e[0] * e[5];
    let c22 = e[0] * e[4] - e[1] * e[3];
    let x = (c00 * rhs.x + c10 * rhs.y + c20 * rhs.z) * inv_det;
    let y = (c01 * rhs.x + c11 * rhs.y + c21 * rhs.z) * inv_det;
    let z = (c02 * rhs.x + c12 * rhs.y + c22 * rhs.z) * inv_det;
    let out = vec3<f32>(x, y, z);
    // Reject a non-finite solve exactly as the reference `is_finite` guard does.
    if (out.x == out.x && out.y == out.y && out.z == out.z) {
        res.ok = 1u;
        res.v = out;
    }
    return res;
}

// Accumulates one spring's gradient and PSD-projected Hessian contribution for
// the vertex at `x`, connected to `other` with rest length `rest` and stiffness
// `k`, mirroring `accumulate_spring`.
fn accumulate_spring(
    grad: ptr<function, vec3<f32>>,
    hess: ptr<function, Mat3>,
    x: vec3<f32>,
    other: vec3<f32>,
    rest: f32,
    k: f32,
) {
    let d = x - other;
    let len_sq = dot(d, d);
    if (len_sq < EPS_LEN_SQ) {
        return;
    }
    let len = sqrt(len_sq);
    let n = d * (1.0 / len);
    (*grad) = (*grad) + n * (k * (len - rest));
    let tangential = max(1.0 - rest / len, 0.0);
    // k·tangential·I + k·(1 - tangential)·nnᵀ (drops the indefinite part when
    // compressed so the Newton step stays a descent direction).
    (*hess) = mat3_add(*hess, mat3_scaled_identity(k * tangential));
    (*hess) = mat3_add(*hess, mat3_scaled_outer(n, k * (1.0 - tangential)));
}

// Unit vector along `v`, or zero when `v` is (numerically) zero, mirroring
// `Vec3::normalize_or_zero` term order (`v * (1 / sqrt(len_sq))`).
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Pushes `point` out to the sphere surface, mirroring `push_out_sphere`.
fn push_out_sphere(center: vec3<f32>, radius: f32, point: vec3<f32>) -> vec3<f32> {
    if (radius <= 0.0) {
        return point;
    }
    let delta = point - center;
    let dist_sq = dot(delta, delta);
    if (dist_sq >= radius * radius) {
        return point;
    }
    if (dist_sq <= EPS_LEN_SQ) {
        return center + vec3<f32>(0.0, radius, 0.0);
    }
    let dir = normalize_or_zero(delta);
    return center + dir * radius;
}

// Point on segment `a`..`b` closest to `point`, mirroring
// `closest_point_on_segment`.
fn closest_point_on_segment(a: vec3<f32>, b: vec3<f32>, point: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let len_sq = dot(ab, ab);
    if (len_sq <= EPS_LEN_SQ) {
        return a;
    }
    let t = clamp(dot(point - a, ab) / len_sq, 0.0, 1.0);
    return a + ab * t;
}

// Projects `point` out of collider `col`, mirroring `Collider::push_out`.
fn push_out(col: Collider, point: vec3<f32>) -> vec3<f32> {
    if (col.kind == KIND_CAPSULE) {
        let a = vec3<f32>(col.ax, col.ay, col.az);
        let b = vec3<f32>(col.bx, col.by, col.bz);
        let closest = closest_point_on_segment(a, b, point);
        return push_out_sphere(closest, col.radius, point);
    }
    let center = vec3<f32>(col.ax, col.ay, col.az);
    return push_out_sphere(center, col.radius, point);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.strand_count) {
        return;
    }
    let strand = strands[s];
    let offset = strand.point_offset;
    let n = strand.point_count;
    if (n == 0u) {
        return;
    }
    let gravity_step = vec3<f32>(params.gsx, params.gsy, params.gsz);

    for (var sub: u32 = 0u; sub < params.substeps; sub = sub + 1u) {
        // 1. Predict each vertex's inertial target from the current implicit
        //    velocity (computed for every particle, matching the reference loop
        //    over `particles.iter()`; pinned targets are stored but unused).
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let gi = offset + i;
            let pos = load_pos(gi);
            let velocity = (pos - load_prev(gi)) * params.velocity_retain;
            store_target(gi, pos + velocity + gravity_step);
        }
        // Snapshot prev = pos for every particle so the next substep measures
        // velocity from here (the reference sets prev on all particles).
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let gi = offset + i;
            store_prev(gi, load_pos(gi));
        }

        // 2. Gauss-Seidel vertex sweeps: one exact Newton step per free vertex.
        for (var it: u32 = 0u; it < params.iterations; it = it + 1u) {
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                let gi = offset + i;
                if (load_inv_mass(gi) <= 0.0) {
                    continue; // pinned: fixed at its skinned pose.
                }
                let x = load_pos(gi);
                let mass = 1.0 / load_inv_mass(gi);
                let inertia = mass / params.sub_dt_sq;

                var grad = (x - load_target(gi)) * inertia;
                var hess = mat3_scaled_identity(inertia);

                // Segment to the next particle uses rest_lengths[i].
                if (strand.has_rest == 1u && i + 1u < n) {
                    accumulate_spring(
                        &grad,
                        &hess,
                        x,
                        load_pos(gi + 1u),
                        load_rest(gi),
                        params.stretch,
                    );
                }
                // Segment to the previous particle uses rest_lengths[i - 1].
                if (strand.has_rest == 1u && i >= 1u) {
                    accumulate_spring(
                        &grad,
                        &hess,
                        x,
                        load_pos(gi - 1u),
                        load_rest(gi - 1u),
                        params.stretch,
                    );
                }
                // Bending: pull an interior vertex toward its neighbors' midpoint.
                if (params.bending > 0.0 && i >= 1u && i + 1u < n) {
                    let mid = (load_pos(gi - 1u) + load_pos(gi + 1u)) * 0.5;
                    grad = grad + (x - mid) * params.bending;
                    hess = mat3_add(hess, mat3_scaled_identity(params.bending));
                }

                let solved = mat3_solve(hess, grad);
                if (solved.ok == 1u) {
                    store_pos(gi, x - solved.v);
                }
            }
        }

        // 3. Project free particles out of every collider, in order, once per
        //    substep after the constraint sweeps have settled.
        if (params.collider_count > 0u) {
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                let gi = offset + i;
                if (load_inv_mass(gi) <= 0.0) {
                    continue;
                }
                var pos = load_pos(gi);
                for (var c: u32 = 0u; c < params.collider_count; c = c + 1u) {
                    pos = push_out(colliders[c], pos);
                }
                store_pos(gi, pos);
            }
        }
    }
}
