// Vertex Block Descent (VBD) GPU sweep — the faithful twin of
// prism_physics_core's VbdSolver::step_colored.
//
// Three kernels drive one full step with positions resident on the GPU:
//   * `predict`      — snapshot prev, form the inertial target, warm-start.
//   * `sweep_color`  — one dispatch per colour: relax every vertex of a colour
//                      with the exact per-vertex 3x3 block-descent solve.
//   * `recover`      — recover velocities with the damping scale.
//
// The host records, per substep: predict, then (iterations x colours) sweeps,
// then recover. Because a colour's vertices share no spring, relaxing them in
// parallel reproduces a Gauss-Seidel pass across colours — the exact schedule
// VbdSolver::sweep_colored runs — so the two agree up to a few ULP of
// division/sqrt rounding.
//
// Every arithmetic step mirrors the golden in structure: the glam Mat3
// column-major inverse, the PSD-projected spring Hessian, and the
// ascending-by-spring-index incident accumulation order.
//
// Provenance: Chen et al., "Vertex Block Descent" (SIGGRAPH 2024); textbook
// greedy graph colouring. No Unreal Engine source or derived code.

// Matches `f32::EPSILON`, the golden's spring-length and determinant floor.
const EPS: f32 = 1.1920929e-7;

struct Params {
    particle_count: u32,
    color_count: u32,
    h: f32,
    inv_h: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    velocity_scale: f32,
};

struct ColorParams {
    color_start: u32,
    color_vertex_count: u32,
    pad0: u32,
    pad1: u32,
};

struct GpuSpring {
    a: u32,
    b: u32,
    rest_length: f32,
    stiffness: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> prev_positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> targets: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(6) var<storage, read> springs: array<GpuSpring>;
@group(0) @binding(7) var<storage, read> vert_offsets: array<u32>;
@group(0) @binding(8) var<storage, read> vert_entries: array<u32>;
@group(0) @binding(9) var<storage, read> order: array<u32>;

@group(1) @binding(0) var<uniform> color_params: ColorParams;

// Outer product `a b^T` as a column-major mat3x3, matching glam's
// `Mat3::from_cols(a * b.x, a * b.y, a * b.z)`.
fn outer(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(a * b.x, a * b.y, a * b.z);
}

// Solves `H dx = f`, bit-faithful to glam `Mat3::inverse() * f` with the
// `|det| <= EPS -> 0` singular guard of `VertexSystem::solve`.
fn solve3(h: mat3x3<f32>, f: vec3<f32>) -> vec3<f32> {
    let c0 = h[0];
    let c1 = h[1];
    let c2 = h[2];
    let t0 = cross(c1, c2);
    let t1 = cross(c2, c0);
    let t2 = cross(c0, c1);
    let det = dot(c2, t2);
    if (abs(det) <= EPS) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let inv_det = 1.0 / det;
    let inv_c0 = vec3<f32>(t0.x, t1.x, t2.x) * inv_det;
    let inv_c1 = vec3<f32>(t0.y, t1.y, t2.y) * inv_det;
    let inv_c2 = vec3<f32>(t0.z, t1.z, t2.z) * inv_det;
    return inv_c0 * f.x + inv_c1 * f.y + inv_c2 * f.z;
}

@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    let pos = positions[i].xyz;
    prev_positions[i] = vec4<f32>(pos, 0.0);
    if (inverse_masses[i] == 0.0) {
        // Pinned: the target is the frozen position, warm-start stays put.
        targets[i] = vec4<f32>(pos, 0.0);
        return;
    }
    let vel = velocities[i].xyz;
    let g = vec3<f32>(params.gravity_x, params.gravity_y, params.gravity_z);
    let y = pos + vel * params.h + g * (params.h * params.h);
    targets[i] = vec4<f32>(y, 0.0);
    positions[i] = vec4<f32>(y, 0.0);
}

@compute @workgroup_size(64)
fn sweep_color(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= color_params.color_vertex_count) {
        return;
    }
    let v = order[color_params.color_start + t];
    if (v >= params.particle_count) {
        return;
    }
    let inv_mass = inverse_masses[v];
    if (inv_mass == 0.0) {
        return;
    }
    let mass = 1.0 / inv_mass;

    let x = positions[v].xyz;
    let y = targets[v].xyz;

    // Inertial term: force -(x - y) * (m/h^2), Hessian (m/h^2) I.
    let coeff = mass / (params.h * params.h);
    var force = -(x - y) * coeff;
    var hessian = mat3x3<f32>(
        vec3<f32>(coeff, 0.0, 0.0),
        vec3<f32>(0.0, coeff, 0.0),
        vec3<f32>(0.0, 0.0, coeff),
    );

    let ident = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );

    // Accumulate incident springs in ascending-by-spring-index CSR order,
    // matching the golden Adjacency reduction order.
    let start = vert_offsets[v];
    let end = vert_offsets[v + 1u];
    for (var e = start; e < end; e = e + 1u) {
        let si = vert_entries[e];
        let s = springs[si];
        // Defensive: never read a malformed out-of-range endpoint. Well-formed
        // inputs (which the solver assumes) never trip this.
        if (s.a >= params.particle_count || s.b >= params.particle_count) {
            continue;
        }
        let pa = positions[s.a].xyz;
        let pb = positions[s.b].xyz;
        let d = pa - pb;
        let l = length(d);
        if (l <= EPS) {
            continue;
        }
        let n = d / l;
        let c = l - s.rest_length;
        let k = s.stiffness;
        let grad_a = n * (k * c);
        var sforce: vec3<f32>;
        if (v == s.a) {
            sforce = -grad_a;
        } else {
            sforce = grad_a;
        }
        force = force + sforce;
        let nnt = outer(n, n);
        let transverse = max(k * c / l, 0.0);
        hessian = hessian + nnt * k + (ident - nnt) * transverse;
    }

    let dx = solve3(hessian, force);
    positions[v] = vec4<f32>(x + dx, 0.0);
}

@compute @workgroup_size(64)
fn recover(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    if (inverse_masses[i] == 0.0) {
        velocities[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }
    let v = (positions[i].xyz - prev_positions[i].xyz) * params.inv_h * params.velocity_scale;
    velocities[i] = vec4<f32>(v, 0.0);
}
