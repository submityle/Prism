// Curl (divergence-free) noise twin: computes on-device the wispy/vortical
// advection flow field, mirroring the CPU golden `curl_noise_3d` (in
// `prism_render_architecture::volumetric::noise`).
//
// The cloud advection field for cirrus streaks, wind disturbance and contrail
// curl (design section 4) is the analytic curl of a three-component vector
// potential, each component an independent signed Perlin field decorrelated by
// a spatial offset and a seed. Taking the curl with matched second-order
// central differences yields a field that is divergence-free up to
// floating-point rounding. The CPU golden `curl_noise_3d` owns that math; this
// kernel is the on-device twin that runs one thread per sample point and
// returns the same three-component vector.
//
// The gradient selection is pure integer hash work (identical to the Perlin
// twin) and WGSL unsigned integers wrap on overflow exactly like Rust, so the
// gradients are bit-identical to the reference. Only the float central-
// difference algebra can differ, and only by legal multiply-add contraction.
// It uses only the portable core-WGSL subset - no exp, pow or optional feature
// - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard curl-of-Perlin-potential noise (Bridson et al. 2007); no
// Unreal Engine source or derived code.

const FNV_OFFSET_BASIS: u32 = 0x811c9dc5u;
const FNV_PRIME: u32 = 0x01000193u;

// Curl potential decorrelation offsets and seeds, matching the reference.
const CURL_OFFSET_G: vec3<f32> = vec3<f32>(31.416, 47.853, 12.793);
const CURL_OFFSET_B: vec3<f32> = vec3<f32>(-53.217, 19.271, 83.155);
const CURL_SEED_G: u32 = 0x1b56c4e9u;
const CURL_SEED_B: u32 = 0x7a4f3c1du;
const CURL_EPS: f32 = 0.02;

const GRAD3: array<vec3<f32>, 16> = array<vec3<f32>, 16>(
    vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(-1.0, 1.0, 0.0),
    vec3<f32>(1.0, -1.0, 0.0),
    vec3<f32>(-1.0, -1.0, 0.0),
    vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(-1.0, 0.0, 1.0),
    vec3<f32>(1.0, 0.0, -1.0),
    vec3<f32>(-1.0, 0.0, -1.0),
    vec3<f32>(0.0, 1.0, 1.0),
    vec3<f32>(0.0, -1.0, 1.0),
    vec3<f32>(0.0, 1.0, -1.0),
    vec3<f32>(0.0, -1.0, -1.0),
    vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(0.0, -1.0, 1.0),
    vec3<f32>(-1.0, 1.0, 0.0),
    vec3<f32>(0.0, -1.0, -1.0),
);

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    x: f32,
    y: f32,
    z: f32,
    seed: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
// Three f32 per query (the curl vector's x, y, z), tightly packed.
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn lerp_v(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

fn fnv_word(hash: u32, word: u32) -> u32 {
    var h = hash;
    var shift = 0u;
    while (shift < 32u) {
        h = h ^ ((word >> shift) & 0xFFu);
        h = h * FNV_PRIME;
        shift = shift + 8u;
    }
    return h;
}

fn hash_cell(ix: i32, iy: i32, iz: i32, seed: u32) -> u32 {
    var h = FNV_OFFSET_BASIS;
    h = fnv_word(h, seed);
    h = fnv_word(h, bitcast<u32>(ix));
    h = fnv_word(h, bitcast<u32>(iy));
    h = fnv_word(h, bitcast<u32>(iz));
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}

fn gradient(h: u32) -> vec3<f32> {
    return GRAD3[h & 15u];
}

fn grad_dot(ix: i32, iy: i32, iz: i32, dx: f32, dy: f32, dz: f32, seed: u32) -> f32 {
    let g = gradient(hash_cell(ix, iy, iz, seed));
    return g.x * dx + g.y * dy + g.z * dz;
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn perlin_raw(px: f32, py: f32, pz: f32, seed: u32) -> f32 {
    let xi = floor(px);
    let yi = floor(py);
    let zi = floor(pz);
    let x0 = i32(xi);
    let y0 = i32(yi);
    let z0 = i32(zi);
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let z1 = z0 + 1;
    let fx = px - xi;
    let fy = py - yi;
    let fz = pz - zi;
    let u = fade(fx);
    let v = fade(fy);
    let w = fade(fz);
    let n000 = grad_dot(x0, y0, z0, fx, fy, fz, seed);
    let n100 = grad_dot(x1, y0, z0, fx - 1.0, fy, fz, seed);
    let n010 = grad_dot(x0, y1, z0, fx, fy - 1.0, fz, seed);
    let n110 = grad_dot(x1, y1, z0, fx - 1.0, fy - 1.0, fz, seed);
    let n001 = grad_dot(x0, y0, z1, fx, fy, fz - 1.0, seed);
    let n101 = grad_dot(x1, y0, z1, fx - 1.0, fy, fz - 1.0, seed);
    let n011 = grad_dot(x0, y1, z1, fx, fy - 1.0, fz - 1.0, seed);
    let n111 = grad_dot(x1, y1, z1, fx - 1.0, fy - 1.0, fz - 1.0, seed);
    let x00 = lerp_v(n000, n100, u);
    let x10 = lerp_v(n010, n110, u);
    let x01 = lerp_v(n001, n101, u);
    let x11 = lerp_v(n011, n111, u);
    let y00 = lerp_v(x00, x10, v);
    let y11 = lerp_v(x01, x11, v);
    return lerp_v(y00, y11, w);
}

// Three-component signed Perlin vector potential, mirroring `curl_potential`.
fn curl_potential(p: vec3<f32>, seed: u32) -> vec3<f32> {
    let a = perlin_raw(p.x, p.y, p.z, seed);
    let g = p + CURL_OFFSET_G;
    let b = perlin_raw(g.x, g.y, g.z, seed ^ CURL_SEED_G);
    let c = p + CURL_OFFSET_B;
    let d = perlin_raw(c.x, c.y, c.z, seed ^ CURL_SEED_B);
    return vec3<f32>(a, b, d);
}

// Divergence-free curl of the potential via matched central differences,
// mirroring `curl_noise_3d`.
fn curl_noise_3d(p: vec3<f32>, seed: u32) -> vec3<f32> {
    let e = CURL_EPS;
    let inv = 1.0 / (2.0 * e);
    let px1 = curl_potential(p + vec3<f32>(e, 0.0, 0.0), seed);
    let px0 = curl_potential(p - vec3<f32>(e, 0.0, 0.0), seed);
    let py1 = curl_potential(p + vec3<f32>(0.0, e, 0.0), seed);
    let py0 = curl_potential(p - vec3<f32>(0.0, e, 0.0), seed);
    let pz1 = curl_potential(p + vec3<f32>(0.0, 0.0, e), seed);
    let pz0 = curl_potential(p - vec3<f32>(0.0, 0.0, e), seed);
    let dpz_dy = (py1.z - py0.z) * inv;
    let dpy_dz = (pz1.y - pz0.y) * inv;
    let dpx_dz = (pz1.x - pz0.x) * inv;
    let dpz_dx = (px1.z - px0.z) * inv;
    let dpy_dx = (px1.y - px0.y) * inv;
    let dpx_dy = (py1.x - py0.x) * inv;
    return vec3<f32>(dpz_dy - dpy_dz, dpx_dz - dpz_dx, dpy_dx - dpx_dy);
}

@compute @workgroup_size(64)
fn curl(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let v = curl_noise_3d(vec3<f32>(q.x, q.y, q.z), q.seed);
    let base = idx * 3u;
    results[base + 0u] = v.x;
    results[base + 1u] = v.y;
    results[base + 2u] = v.z;
}
