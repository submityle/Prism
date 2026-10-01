// Compute twin of Prism's second-order spherical-harmonic transmittance
// reconstruction (prism_render_architecture::hair::dual_scatter_sh::eval_sh,
// which composes sh_basis with a coefficient dot). One thread per query
// direction normalises the direction (degenerate -> zero, collapsing to the
// pole), evaluates the nine real second-order SH basis functions in Cartesian
// polynomial form (no trigonometry), dots them against the shared coefficient
// set, then sanitises and clamps the result to >= 0 (transmittance is
// non-negative).
//
// The nine coefficients are a shared read-only storage buffer (band-major:
// Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22); the per-thread variation is
// the query direction read from a flat x,y,z storage array (3 floats per
// element, tightly packed to dodge the std430 vec3 stride).
//
// The basis uses only multiply/add plus one sqrt (vector normalisation), so it
// runs unmodified on Metal, Vulkan and DX12. The reconstruction is a mul/add
// chain and a divide a GPU may fuse, so CPU and GPU agree within the documented
// fma tolerance rather than bit-for-bit.

struct Params {
    element_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Nine band-major SH coefficients, shared by every thread.
@group(0) @binding(1) var<storage, read> coeffs: array<f32>;
// Query directions, flattened x,y,z per element (unnormalised is fine).
@group(0) @binding(2) var<storage, read> dirs: array<f32>;
// One reconstructed transmittance per direction, clamped to >= 0.
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

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

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.element_count {
        return;
    }
    let base = idx * 3u;
    let d = normalize_or_zero(dirs[base], dirs[base + 1u], dirs[base + 2u]);
    let x = d.x;
    let y = d.y;
    let z = d.z;

    // Nine real second-order SH basis values in Cartesian polynomial form,
    // dotted against the shared coefficients in the golden's band-major order.
    var accum: f32 = 0.0;
    accum = accum + coeffs[0] * K0;
    accum = accum + coeffs[1] * (K1 * y);
    accum = accum + coeffs[2] * (K1 * z);
    accum = accum + coeffs[3] * (K1 * x);
    accum = accum + coeffs[4] * (K2A * x * y);
    accum = accum + coeffs[5] * (K2A * y * z);
    accum = accum + coeffs[6] * (K2B * (3.0 * z * z - 1.0));
    accum = accum + coeffs[7] * (K2A * x * z);
    accum = accum + coeffs[8] * (K2C * (x * x - y * y));

    out[idx] = max(sanitize(accum), 0.0);
}
