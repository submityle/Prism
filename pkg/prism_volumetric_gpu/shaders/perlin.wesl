// Perlin gradient-noise twin: computes on-device the [0, 1]-normalised
// improved-Perlin gradient noise, mirroring the CPU golden `perlin_3d` (in
// `prism_render_architecture::volumetric::noise`).
//
// The cloud density field's low-frequency continuity comes from Perlin noise
// (design section 4, Nubis-style modelling): a quintic-fade trilinear blend of
// twelve cube-edge gradients selected by a deterministic integer hash. The CPU
// golden `perlin_3d` owns that math; this kernel is the on-device twin that
// runs one thread per sample point and returns the same value.
//
// The hash is pure integer work (FNV-1a byte mixing plus an xorshift-multiply
// avalanche) and WGSL unsigned integers wrap on overflow exactly like Rust's
// `wrapping_mul`/`^`/`>>`, so the lattice hash is bit-identical to the
// reference. Only the float fade/lerp blend can differ, and only by a legal
// multiply-add contraction. It uses only the portable core-WGSL subset - no
// exp, pow or optional feature - so this twin runs unmodified on Metal, Vulkan
// and DX12.
//
// Provenance: standard improved-Perlin gradient noise (Perlin 2002); no Unreal
// Engine source or derived code.

// FNV-1a 32-bit offset basis and prime, the standard constants.
const FNV_OFFSET_BASIS: u32 = 0x811c9dc5u;
const FNV_PRIME: u32 = 0x01000193u;

// The twelve cube-edge-midpoint gradients padded to sixteen (four repeats) so
// a gradient is picked with a four-bit mask; each has squared length 2.
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
    // Number of valid entries in `queries` / outputs in `results`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One noise query. `16`-byte stride matching the `bytemuck` upload struct.
struct Query {
    x: f32,
    y: f32,
    z: f32,
    seed: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// Clamps `x` into `[0, 1]`, mirroring `math::saturate`.
fn saturate_v(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Linear interpolation `a + (b - a) * t`, mirroring `math::lerp`.
fn lerp_v(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Mixes one 32-bit word into the running FNV-1a hash, byte by byte, mirroring
// `fnv_word`.
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

// Deterministic integer hash of an integer lattice point and a seed, mirroring
// `hash_cell` (FNV-1a mixing plus an xorshift-multiply avalanche).
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

// Selects a lattice gradient from GRAD3 using the low four bits of `h`,
// mirroring `gradient`.
fn gradient(h: u32) -> vec3<f32> {
    return GRAD3[h & 15u];
}

// Dot of the gradient at lattice point `(ix, iy, iz)` with the distance vector,
// mirroring `grad_dot`.
fn grad_dot(ix: i32, iy: i32, iz: i32, dx: f32, dy: f32, dz: f32, seed: u32) -> f32 {
    let g = gradient(hash_cell(ix, iy, iz, seed));
    return g.x * dx + g.y * dy + g.z * dz;
}

// Quintic fade `6t^5 - 15t^4 + 10t^3`, mirroring `fade`.
fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Raw signed gradient noise in roughly [-1, 1], mirroring `perlin_raw`.
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

@compute @workgroup_size(64)
fn perlin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = saturate_v(0.5 + 0.5 * perlin_raw(q.x, q.y, q.z, q.seed));
}
