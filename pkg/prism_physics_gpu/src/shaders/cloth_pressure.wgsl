// Closed-mesh cloth pressure (volume preservation / inflation), the real-device
// twin of `prism_physics_core::project_pressure`.
//
// Pressure is a single global compliant-XPBD constraint over the whole shell:
// the signed enclosed volume V = (1/6) sum_tri p0 . (p1 x p2) is driven toward a
// target, and every vertex is pushed along its accumulated volume gradient. One
// solver iteration is six coordinated passes (separate compute passes so each
// observes the previous one's writes):
//
//   1. tri_pass     — per triangle: raw volume term + the three corner gradients
//   2. vol_reduce   — sum the triangle volume terms, scale by 1/6 -> accum[0]
//   3. gather_pass  — per vertex: gather incident corner gradients (CSR) + denom
//   4. denom_reduce — sum the per-vertex denominator terms -> accum[1]
//   5. lambda_pass  — single invocation: delta_lambda and lambda accumulation
//   6. apply_pass   — per vertex: x_i += w_i * delta_lambda * grad_i
//
// The scatter of per-triangle gradients to shared vertices is reformulated as a
// race-free per-vertex gather through the host-built CSR adjacency, so no atomic
// floating-point accumulation is needed.
//
// Provenance: signed-volume-via-divergence-theorem pressure, compliant XPBD
// projection, CSR scatter->gather, and workgroup tree reduction are all standard
// techniques. No Unreal Engine source or derived code.

const INV_SIX: f32 = 1.0 / 6.0;
const EPS_LEN_SQ: f32 = 1.0e-12;
const REDUCE_LANES: u32 = 256u;

struct Params {
    // Target enclosed volume `overpressure * rest_volume`.
    target_volume: f32,
    // XPBD compliance (inverse stiffness); 0 is perfectly rigid.
    compliance: f32,
    // Substep time (seconds).
    dt: f32,
    // Number of addressable vertices (positions length).
    vertex_count: u32,
    // Number of triangles in the closed shell.
    triangle_count: u32,
    // Length of the inverse-mass array (indices at or past it are pinned).
    inv_mass_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> inv_mass: array<f32>;
@group(0) @binding(3) var<storage, read> triangles: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> adj_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> adj_corners: array<u32>;
@group(0) @binding(6) var<storage, read_write> tri_grads: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> tri_vol: array<f32>;
@group(0) @binding(8) var<storage, read_write> vert_grad: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> denom_term: array<f32>;
// accum: [0]=volume, [1]=denominator, [2]=lambda (persists), [3]=delta_lambda.
@group(0) @binding(10) var<storage, read_write> accum: array<f32>;

var<workgroup> reduce_scratch: array<f32, 256u>;

// Reads the inverse mass of vertex `i`, treating out-of-range indices as pinned
// (weight 0), exactly like `inverse_masses.get(i).unwrap_or(0.0)` on the CPU.
fn weight_of(i: u32) -> f32 {
    if (i < params.inv_mass_count) {
        return inv_mass[i];
    }
    return 0.0;
}

// Pass 1: per-triangle raw volume term and the three per-corner gradients. A
// triangle with any out-of-range index contributes zero (matching the CPU skip).
@compute @workgroup_size(64)
fn tri_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= params.triangle_count) {
        return;
    }
    let tri = triangles[t];
    let vc = params.vertex_count;

    var vol = 0.0;
    var g0 = vec3<f32>(0.0, 0.0, 0.0);
    var g1 = vec3<f32>(0.0, 0.0, 0.0);
    var g2 = vec3<f32>(0.0, 0.0, 0.0);
    if (tri.x < vc && tri.y < vc && tri.z < vc) {
        let p0 = positions[tri.x].xyz;
        let p1 = positions[tri.y].xyz;
        let p2 = positions[tri.z].xyz;
        vol = dot(p0, cross(p1, p2));
        g0 = cross(p1, p2) * INV_SIX;
        g1 = cross(p2, p0) * INV_SIX;
        g2 = cross(p0, p1) * INV_SIX;
    }
    tri_vol[t] = vol;
    tri_grads[t * 3u + 0u] = vec4<f32>(g0, 0.0);
    tri_grads[t * 3u + 1u] = vec4<f32>(g1, 0.0);
    tri_grads[t * 3u + 2u] = vec4<f32>(g2, 0.0);
}

// Pass 2: tree-reduce the triangle volume terms into accum[0], scaled by 1/6.
@compute @workgroup_size(256)
fn vol_reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = params.triangle_count;
    var acc = 0.0;
    var i = lid.x;
    loop {
        if (i >= n) {
            break;
        }
        acc = acc + tri_vol[i];
        i = i + REDUCE_LANES;
    }
    reduce_scratch[lid.x] = acc;
    workgroupBarrier();

    var stride = REDUCE_LANES >> 1u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid.x < stride) {
            reduce_scratch[lid.x] = reduce_scratch[lid.x] + reduce_scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid.x == 0u) {
        accum[0] = reduce_scratch[0] * INV_SIX;
    }
}

// Pass 3: per-vertex gather of incident corner gradients (CSR) and the
// per-vertex denominator term w_i * |grad_i|^2 (zero for pinned vertices).
@compute @workgroup_size(64)
fn gather_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= params.vertex_count) {
        return;
    }
    let start = adj_offsets[v];
    let end = adj_offsets[v + 1u];
    var grad = vec3<f32>(0.0, 0.0, 0.0);
    var e = start;
    loop {
        if (e >= end) {
            break;
        }
        let packed = adj_corners[e];
        let t = packed >> 2u;
        let c = packed & 3u;
        grad = grad + tri_grads[t * 3u + c].xyz;
        e = e + 1u;
    }
    vert_grad[v] = vec4<f32>(grad, 0.0);

    let w = weight_of(v);
    var term = 0.0;
    if (w > 0.0) {
        term = w * dot(grad, grad);
    }
    denom_term[v] = term;
}

// Pass 4: tree-reduce the per-vertex denominator terms into accum[1].
@compute @workgroup_size(256)
fn denom_reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = params.vertex_count;
    var acc = 0.0;
    var i = lid.x;
    loop {
        if (i >= n) {
            break;
        }
        acc = acc + denom_term[i];
        i = i + REDUCE_LANES;
    }
    reduce_scratch[lid.x] = acc;
    workgroupBarrier();

    var stride = REDUCE_LANES >> 1u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid.x < stride) {
            reduce_scratch[lid.x] = reduce_scratch[lid.x] + reduce_scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid.x == 0u) {
        accum[1] = reduce_scratch[0];
    }
}

// Pass 5: single-invocation compliant-XPBD scalar update. Mirrors the CPU:
// alpha_tilde = compliance / dt^2; denom = sum + alpha_tilde; a (near) zero
// denominator is a no-op (delta_lambda = 0, lambda unchanged).
@compute @workgroup_size(1)
fn lambda_pass() {
    let volume = accum[0];
    let c = volume - params.target_volume;
    let alpha_tilde = params.compliance / (params.dt * params.dt);
    let denom = accum[1] + alpha_tilde;
    var delta = 0.0;
    if (denom >= EPS_LEN_SQ) {
        let lambda = accum[2];
        delta = (-c - alpha_tilde * lambda) / denom;
        accum[2] = lambda + delta;
    }
    accum[3] = delta;
}

// Pass 6: per-vertex position update x_i += w_i * delta_lambda * grad_i.
@compute @workgroup_size(64)
fn apply_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= params.vertex_count) {
        return;
    }
    let w = weight_of(v);
    if (w <= 0.0) {
        return;
    }
    let delta = accum[3];
    let grad = vert_grad[v].xyz;
    let p = positions[v].xyz + grad * (w * delta);
    positions[v] = vec4<f32>(p, positions[v].w);
}
