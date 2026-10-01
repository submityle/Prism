// Animated film-grain post-process twin: evaluates the CPU gold standard
// `FilmGrainParams::apply` on-device, one thread per pixel, mirroring
// `prism_render_architecture::particle::film_grain`.
//
// The grain field is pure integer hash plus rational/polynomial algebra with no
// transcendental call: `hash_u32` is an xor-shift + odd-constant-multiply
// avalanche finalizer; `grain_hash01` decorrelates the pixel coordinates and
// the frame index with odd multipliers, mixes them and normalizes the top 24
// bits into [0, 1); `value_noise01` quantizes the pixel grid into integer cells,
// hashes the four cell corners and blends them with a smoothstep bilinear lerp;
// the luminance `response` is a shadow-boosted-over-highlight-rolled-off
// rational polynomial; the final composite is either an additive blend or the
// polynomial (Pegtop) soft light, both clamped to [0, 1].
//
// WGSL u32 arithmetic is defined to wrap on overflow, exactly matching the CPU
// golden's `wrapping_mul` / `wrapping_add`, so the integer hash reproduces the
// bit pattern exactly; the floating noise, response and blend are the same
// closed form evaluated in the same order (up to a legal fused multiply-add).
// The kernel uses only the portable core-WGSL subset (min/max/clamp, + - * /,
// bitwise ^ & | << >>, f32 conversion), so it runs unmodified on Metal, Vulkan
// and DX12.
//
// Provenance: hand-rolled integer-hash value-noise film grain plus wgpu compute
// dispatch; no Unreal Engine source or derived code.

// Odd decorrelation multipliers (bijections modulo 2^32), matching the CPU
// goldens `ODD_X`, `ODD_Y`, `ODD_FRAME`.
const ODD_X: u32 = 0x9e3779b1u;
const ODD_Y: u32 = 0x85ebca77u;
const ODD_FRAME: u32 = 0xc2b2ae3du;

// Odd multiply constants of the `hash_u32` finalizer (`MIX_A`, `MIX_B`).
const MIX_A: u32 = 0x7feb352du;
const MIX_B: u32 = 0x846ca68bu;

// 2^24: the largest power of two exactly representable in an f32 mantissa, used
// as the normalization divisor so every hashed value is exact (`NORM_24`).
const NORM_24: f32 = 16777216.0;

// Rec. 709 luma weights (`LUMA_R`, `LUMA_G`, `LUMA_B`).
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

// Blend discriminant codes, matching `GrainBlend::code`.
const BLEND_ADDITIVE: u32 = 0u;

// Uniform parameters for one dispatch. Layout matches `GrainParamsGpu` in
// `src/film_grain.rs`: three f32 tunables, the u32 cell size, the blend code,
// the frame index, the pixel count and one pad word (two vec4 slots).
struct Params {
    intensity: f32,
    shadow_boost: f32,
    highlight_rolloff: f32,
    cell_size: u32,
    blend_code: u32,
    frame: u32,
    count: u32,
    pad: u32,
}

// One input pixel: the integer coordinate plus the linear RGB color in the
// first three lanes of a vec4. Matches `GpuPixel` in `src/film_grain.rs`.
struct Pixel {
    x: u32,
    y: u32,
    pad0: u32,
    pad1: u32,
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pixels: array<Pixel>;
@group(0) @binding(2) var<storage, read_write> results: array<vec4<f32>>;

// Clamp into the closed unit interval [0, 1].
fn clamp01(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

// Cubic smoothstep easing t*t*(3-2t) after clamping t into [0, 1].
fn smoothstep01(t: f32) -> f32 {
    let c = clamp01(t);
    return c * c * (3.0 - 2.0 * c);
}

// Linear interpolation a + (b - a) * t.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Pure u32 avalanche hash: xor-shift and odd-constant-multiply finalizer. u32
// multiply wraps in WGSL, matching the CPU golden `wrapping_mul`.
fn hash_u32(seed: u32) -> u32 {
    var h = seed;
    h ^= h >> 16u;
    h = h * MIX_A;
    h ^= h >> 15u;
    h = h * MIX_B;
    h ^= h >> 16u;
    return h;
}

// A per-pixel, per-frame hashed value in [0, 1): decorrelate the axes and frame
// with odd multipliers, mix, and normalize the top 24 bits.
fn grain_hash01(x: u32, y: u32, frame: u32) -> f32 {
    let a = x * ODD_X;
    let b = y * ODD_Y;
    let c = frame * ODD_FRAME;
    let h = hash_u32(a ^ b ^ c);
    return f32(h >> 8u) / NORM_24;
}

// The within-cell fraction of `coord` along one axis, in [0, 1).
fn cell_fraction(coord: u32, cell: u32) -> f32 {
    return f32(coord % cell) / f32(cell);
}

// Sized value noise in [0, 1): quantize the pixel grid into integer cells, hash
// the four surrounding corners and smoothstep-bilinear blend them.
fn value_noise01(x: u32, y: u32, frame: u32, cell_size: u32) -> f32 {
    let cell = max(cell_size, 1u);
    let gx = x / cell;
    let gy = y / cell;
    let fx = cell_fraction(x, cell);
    let fy = cell_fraction(y, cell);
    let sx = smoothstep01(fx);
    let sy = smoothstep01(fy);
    let gx1 = gx + 1u;
    let gy1 = gy + 1u;
    let c00 = grain_hash01(gx, gy, frame);
    let c10 = grain_hash01(gx1, gy, frame);
    let c01 = grain_hash01(gx, gy1, frame);
    let c11 = grain_hash01(gx1, gy1, frame);
    let top = lerp(c00, c10, sx);
    let bottom = lerp(c01, c11, sx);
    return lerp(top, bottom, sy);
}

// Signed grain in [-1, 1): the unit-interval value noise mapped zero-centered.
fn signed_grain(x: u32, y: u32, frame: u32, cell_size: u32) -> f32 {
    return value_noise01(x, y, frame, cell_size) * 2.0 - 1.0;
}

// Rec. 709 relative luminance of a linear RGB triple, clamped to [0, 1].
fn luma709(color: vec3<f32>) -> f32 {
    return clamp01(LUMA_R * color.x + LUMA_G * color.y + LUMA_B * color.z);
}

// Luminance response weight: shadow-boosted numerator over highlight-rolled-off
// denominator (a rational polynomial, no transcendental).
fn response(l: f32) -> f32 {
    let lc = clamp01(l);
    let boost = max(params.shadow_boost, 0.0);
    let rolloff = max(params.highlight_rolloff, 0.0);
    let numerator = 1.0 + boost * (1.0 - lc);
    let denominator = 1.0 + rolloff * lc * lc;
    return numerator / denominator;
}

// Composite the signed grain onto one channel using the selected blend. Both
// paths clamp into [0, 1]; the soft-light path is the polynomial Pegtop form.
fn composite_channel(base: f32, amount: f32) -> f32 {
    if (params.blend_code == BLEND_ADDITIVE) {
        return clamp01(base + amount);
    }
    let a = clamp01(base);
    let b = clamp01(0.5 + 0.5 * amount);
    return clamp01((1.0 - 2.0 * b) * a * a + 2.0 * a * b);
}

@compute @workgroup_size(64)
fn film_grain_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let px = pixels[idx];
    let color = vec3<f32>(px.color.x, px.color.y, px.color.z);
    let grain = signed_grain(px.x, px.y, params.frame, params.cell_size);
    let weight = response(luma709(color));
    let amount = grain * weight * params.intensity;
    let out = vec3<f32>(
        composite_channel(color.x, amount),
        composite_channel(color.y, amount),
        composite_channel(color.z, amount)
    );
    results[idx] = vec4<f32>(out.x, out.y, out.z, 0.0);
}
