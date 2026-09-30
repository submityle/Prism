// Rayleigh phase-function twin: evaluates on-device the normalized Rayleigh
// scattering phase function, mirroring the CPU golden `rayleigh_phase` (in
// `prism_render_architecture::volumetric::spectral`).
//
// The spectral sky model (design section 8) scatters short wavelengths off air
// molecules with the classic Rayleigh angular signature:
//   phase = 3/(16*pi) * (1 + cos^2 theta)
// with `cos_theta` clamped to `-1..=1` so out-of-range inputs never leave the
// valid domain. The function integrates to one over the sphere of solid angle
// and is forward/back-scatter symmetric - the signature that makes a clear sky
// blue. The CPU golden `rayleigh_phase` owns that math; this kernel is the
// on-device twin that runs one thread per query and reproduces the same value.
//
// The normalization constant is folded the SAME way the reference folds it
// (3 / (16 * PI) with the f32 value of PI), and the kernel contains no
// transcendental call - a clamp and a multiply-add - so CPU and GPU evaluate
// the same closed-form algebra and the parity test can assert a tight
// tolerance.
//
// It uses only the portable core-WGSL subset - clamp and multiply/add - so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Rayleigh scattering phase function; no Unreal Engine
// source or derived code.

// The f32 value of pi, matching `core::f32::consts::PI` on the CPU side.
const PI: f32 = 3.1415927;
// 3 / (16 * PI), matching `RAYLEIGH_NORM` in the CPU reference.
const RAYLEIGH_NORM: f32 = 3.0 / (16.0 * PI);

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One Rayleigh-phase query: the cosine of the scattering angle. Matches
// `GpuQuery` in `src/rayleigh_phase.rs`.
struct Query {
    cos_theta: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// The normalized Rayleigh scattering phase function, mirroring
// `rayleigh_phase`.
fn rayleigh_phase(cos_theta: f32) -> f32 {
    let c = clamp(cos_theta, -1.0, 1.0);
    return RAYLEIGH_NORM * (1.0 + c * c);
}

@compute @workgroup_size(64)
fn rayleigh_phase_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = rayleigh_phase(q.cos_theta);
}
