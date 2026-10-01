// Compute twin of Prism's second-order spherical-harmonic transmittance
// projection (prism_render_architecture::hair::dual_scatter_sh::project_sh,
// which composes sh_basis with a Monte-Carlo coefficient accumulation). This
// is the inverse of eval_sh: eval_sh reconstructs a transmittance from nine
// coefficients, project_sh fits nine coefficients from a batch of directional
// transmittance samples via c_lm = sum(T * Y_lm * w) with the uniform
// solid-angle weight w = 4*pi / N.
//
// One thread per SH coefficient (nine threads in a single workgroup): thread k
// walks every sample in authored order and accumulates
//   c[k] += (sanitize(value) * weight) * basis_k(normalize_or_zero(dir)).
// Because each coefficient is accumulated over the samples in the same order
// the golden uses, the only CPU/GPU divergence is fma fusion of the per-sample
// multiply-add, so parity is an fma tolerance rather than bit-for-bit.
//
// The weight is computed on the host (4*pi / N as an f32) and passed in a
// uniform, so the shader never spells out a pi literal whose bit pattern could
// drift from core::f32::consts::PI. Values are sanitised before the weight
// multiply (matching the golden's `sanitize(value) * weight`) and every
// coefficient is sanitised once more after the accumulation.
//
// Samples are a flat storage array of four floats each (dir.x, dir.y, dir.z,
// value) to dodge the std430 vec3 stride. The basis uses only multiply/add
// plus one sqrt (vector normalisation), so it runs unmodified on Metal, Vulkan
// and DX12.

struct Params {
    sample_count: u32,
    weight: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Transmittance samples, flattened dir.x, dir.y, dir.z, value per sample.
@group(0) @binding(1) var<storage, read> samples: array<f32>;
// Nine band-major SH coefficients projected from the samples.
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const MAX_FINITE_F32: f32 = 3.40282347e+38;
// Squared-length threshold below which a direction is treated as degenerate
// (mirrors the golden `NORM_EPS`).
const NORM_EPS: f32 = 1e-12;

// SH basis normalisation constants (standard orthonormal real SH).
const K0: f32 = 0.282095;
const K1: f32 = 0.488603;
const K2A: f32 = 1.092548;
const K2B: f32 = 0.315392;
const K2C: f32 = 0.546274;

// Mirror the golden `is_finite()` test without a float equality on a literal.
fn is_finite_f32(x: f32) -> bool {
    return x == x && x >= -MAX_FINITE_F32 && x <= MAX_FINITE_F32;
}

// Golden `sanitize`: replace a non-finite value with 0, leaving finite intact.
fn sanitize(x: f32) -> f32 {
    if is_finite_f32(x) {
        return x;
    }
    return 0.0;
}

// Golden `Vec3::normalize_or_zero`: sanitise components, then unit-normalise
// when the squared length is finite and above NORM_EPS, else return zero.
fn normalize_or_zero(vx: f32, vy: f32, vz: f32) -> vec3<f32> {
    let x = sanitize(vx);
    let y = sanitize(vy);
    let z = sanitize(vz);
    let len_sq = x * x + y * y + z * z;
    if is_finite_f32(len_sq) && len_sq > NORM_EPS {
        let len = sqrt(len_sq);
        return vec3<f32>(x / len, y / len, z / len);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// The k-th real second-order SH basis value in Cartesian polynomial form,
// matching the golden `sh_basis` band-major ordering.
fn basis_for(k: u32, x: f32, y: f32, z: f32) -> f32 {
    switch k {
        case 0u: { return K0; }
        case 1u: { return K1 * y; }
        case 2u: { return K1 * z; }
        case 3u: { return K1 * x; }
        case 4u: { return K2A * x * y; }
        case 5u: { return K2A * y * z; }
        case 6u: { return K2B * (3.0 * z * z - 1.0); }
        case 7u: { return K2A * x * z; }
        case 8u: { return K2C * (x * x - y * y); }
        default: { return 0.0; }
    }
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= 9u {
        return;
    }

    let weight = params.weight;
    var accum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.sample_count; i = i + 1u) {
        let base = i * 4u;
        let d = normalize_or_zero(samples[base], samples[base + 1u], samples[base + 2u]);
        let value = sanitize(samples[base + 3u]) * weight;
        accum = accum + value * basis_for(k, d.x, d.y, d.z);
    }

    out[k] = sanitize(accum);
}
