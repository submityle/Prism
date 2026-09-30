// Fractal-noise (fBm) twin: computes on-device the [0, 1] Perlin fBm and
// billowy inverted-Worley fBm values for an arbitrary octave count, mirroring
// the CPU goldens `fbm` and `worley_fbm` (in
// `prism_render_architecture::volumetric::noise`).
//
// The base cloud shape (design section 4, Nubis-style modelling) stacks several
// octaves of coherent noise with geometric frequency growth (lacunarity) and
// amplitude decay (gain), amplitude-normalised into [0, 1]. The
// `perlin_worley` twin (`perlin_worley.wesl`) already exercises these two
// accumulators, but only at the fixed default octave counts baked into that
// entry point. This kernel exposes the full `(point, seed, octaves)` surface of
// the CPU goldens so their variable-octave behaviour - including the
// `octaves == 0` early-out that yields 0 rather than dividing by zero - is
// directly verified on-device, one thread per query.
//
// Every gradient and feature point is selected by the same deterministic
// integer hash (FNV-1a byte mixing plus an xorshift-multiply avalanche) as the
// Perlin, Worley and Perlin-Worley twins, and WGSL unsigned integers wrap on
// overflow exactly like Rust's `wrapping_mul`/`wrapping_add`/`^`/`>>`, so the
// lattice work is bit-identical to the reference. Only the float fBm
// accumulation and the Worley `sqrt` can differ, and only by legal
// multiply-add contraction. It uses only the portable core-WGSL subset - no
// exp, pow or optional feature - so this twin runs unmodified on Metal, Vulkan
// and DX12.
//
// Provenance: standard Nubis-style Perlin / inverted-Worley fractal cloud
// noise; no Unreal Engine source or derived code.

// FNV-1a 32-bit offset basis and prime, the standard constants.
const FNV_OFFSET_BASIS: u32 = 0x811c9dc5u;
const FNV_PRIME: u32 = 0x01000193u;

// Worley feature-point y/z decorrelation seeds.
const WORLEY_SEED_Y: u32 = 0x68e31da4u;
const WORLEY_SEED_Z: u32 = 0xb5279a45u;

// Per-octave fBm seed stride so stacked octaves decorrelate.
const FBM_SEED_STRIDE: u32 = 0x9e3779b9u;

// Lacunarity/gain shared by the fractal accumulators.
const DEFAULT_LACUNARITY: f32 = 2.0;
const DEFAULT_GAIN: f32 = 0.5;

const EPS: f32 = 1e-6;

// Improved-Perlin gradient set (twelve cube-edge midpoints padded to sixteen).
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

// One fractal-noise query. `24`-byte stride matching the `bytemuck` upload
// struct `NoiseFbmQuery`.
struct Query {
    x: f32,
    y: f32,
    z: f32,
    seed: u32,
    octaves: u32,
    pad0: u32,
}

// One fractal-noise result: the Perlin fBm and the inverted-Worley fBm.
struct Result {
    fbm: f32,
    worley_fbm: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

fn saturate_v(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

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

fn hash_to_unit(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
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

fn perlin_3d(px: f32, py: f32, pz: f32, seed: u32) -> f32 {
    return saturate_v(0.5 + 0.5 * perlin_raw(px, py, pz, seed));
}

fn worley_3d(px: f32, py: f32, pz: f32, seed: u32) -> f32 {
    let xi = floor(px);
    let yi = floor(py);
    let zi = floor(pz);
    let x0 = i32(xi);
    let y0 = i32(yi);
    let z0 = i32(zi);
    let fx = px - xi;
    let fy = py - yi;
    let fz = pz - zi;
    var min_sq = 3.4028235e38;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let cx = x0 + dx;
                let cy = y0 + dy;
                let cz = z0 + dz;
                let ox = hash_to_unit(hash_cell(cx, cy, cz, seed));
                let oy = hash_to_unit(hash_cell(cx, cy, cz, seed ^ WORLEY_SEED_Y));
                let oz = hash_to_unit(hash_cell(cx, cy, cz, seed ^ WORLEY_SEED_Z));
                let fpx = f32(dx) + ox - fx;
                let fpy = f32(dy) + oy - fy;
                let fpz = f32(dz) + oz - fz;
                let d2 = fpx * fpx + fpy * fpy + fpz * fpz;
                min_sq = min(min_sq, d2);
            }
        }
    }
    return saturate_v(sqrt(min_sq));
}

fn worley_inverted(px: f32, py: f32, pz: f32, seed: u32) -> f32 {
    return saturate_v(1.0 - worley_3d(px, py, pz, seed));
}

// Perlin fBm: `octaves` of perlin_3d with geometric frequency growth and
// amplitude decay, amplitude-normalised to [0, 1], mirroring `fbm`.
fn fbm(px: f32, py: f32, pz: f32, seed: u32, octaves: u32) -> f32 {
    var freq = 1.0;
    var amp = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    var i = 0u;
    while (i < octaves) {
        let octave_seed = seed + i * FBM_SEED_STRIDE;
        sum = sum + amp * perlin_3d(px * freq, py * freq, pz * freq, octave_seed);
        norm = norm + amp;
        freq = freq * DEFAULT_LACUNARITY;
        amp = amp * DEFAULT_GAIN;
        i = i + 1u;
    }
    if (norm > EPS) {
        return saturate_v(sum / norm);
    }
    return 0.0;
}

// Inverted-Worley fBm (billowy), mirroring `worley_fbm`.
fn worley_fbm(px: f32, py: f32, pz: f32, seed: u32, octaves: u32) -> f32 {
    var freq = 1.0;
    var amp = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    var i = 0u;
    while (i < octaves) {
        let octave_seed = seed + i * FBM_SEED_STRIDE;
        sum = sum + amp * worley_inverted(px * freq, py * freq, pz * freq, octave_seed);
        norm = norm + amp;
        freq = freq * DEFAULT_LACUNARITY;
        amp = amp * DEFAULT_GAIN;
        i = i + 1u;
    }
    if (norm > EPS) {
        return saturate_v(sum / norm);
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn noise_fbm_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let q = queries[i];
    var r: Result;
    r.fbm = fbm(q.x, q.y, q.z, q.seed, q.octaves);
    r.worley_fbm = worley_fbm(q.x, q.y, q.z, q.seed, q.octaves);
    results[i] = r;
}
