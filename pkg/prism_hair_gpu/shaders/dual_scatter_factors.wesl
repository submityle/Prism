// Compute twin of Prism's Zinke dual-scattering forward/back factor estimator
// (prism_render_architecture::hair::dual_scatter_sh::dual_scatter_factors). The
// factors a_f (forward) and a_b (backward) are the mean transmittance over the
// +z and -z hemispheres respectively, driving the global multiple-scatter term
// a_f^n and the local back-scatter term of the Zinke 2008 approximation.
//
// This is a hemisphere-split averaging reduction, genuinely distinct from both
// the SH projection twin (project_sh) and the integer-power twin
// (forward_scatter_power): here each hemisphere's samples are clamped to [0, 1]
// and arithmetically averaged.
//
// Two live threads in a single workgroup: thread 0 accumulates the +z
// hemisphere into the forward factor, thread 1 the -z hemisphere into the
// backward factor. Each walks every sample in authored order, normalises the
// direction (normalize_or_zero), sanitises and clamps the value to [0, 1], and
// averages its hemisphere (count 0 -> 0). Because each factor is summed over
// its hemisphere in the golden's sample order, the only CPU/GPU divergence is
// the final divide, so parity is an fma tolerance rather than bit-for-bit.
//
// Samples are a flat storage array of four floats each (dir.x, dir.y, dir.z,
// value) to dodge the std430 vec3 stride. The kernel uses only multiply/add,
// one sqrt (vector normalisation) and a divide, so it runs unmodified on Metal,
// Vulkan and DX12.

struct Params {
    sample_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Transmittance samples, flattened dir.x, dir.y, dir.z, value per sample.
@group(0) @binding(1) var<storage, read> samples: array<f32>;
// Two factors: out[0] = forward (a_f), out[1] = backward (a_b).
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const MAX_FINITE_F32: f32 = 3.40282347e+38;
// Squared-length threshold below which a direction is treated as degenerate
// (mirrors the golden `NORM_EPS`).
const NORM_EPS: f32 = 1e-12;

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
    let lane = gid.x;
    if lane >= 2u {
        return;
    }
    // lane 0 collects the +z hemisphere, lane 1 the -z hemisphere.
    let want_forward = lane == 0u;

    var sum: f32 = 0.0;
    var count: u32 = 0u;
    for (var i: u32 = 0u; i < params.sample_count; i = i + 1u) {
        let base = i * 4u;
        let d = normalize_or_zero(samples[base], samples[base + 1u], samples[base + 2u]);
        let value = clamp(sanitize(samples[base + 3u]), 0.0, 1.0);
        var in_hemisphere: bool;
        if want_forward {
            in_hemisphere = d.z > 0.0;
        } else {
            in_hemisphere = d.z < 0.0;
        }
        if in_hemisphere {
            sum = sum + value;
            count = count + 1u;
        }
    }

    if count > 0u {
        out[lane] = clamp(sum / f32(count), 0.0, 1.0);
    } else {
        out[lane] = 0.0;
    }
}
