// Octave-scatter twin: computes on-device the per-octave scattering
// coefficient, extinction coefficient and phase eccentricity, mirroring the
// CPU golden `octave_scatter`.
//
// The Wrenninge-style octave approximation is a cheap multiple-scattering model
// (design section 7): each successive scattering octave attenuates sigma_s,
// sigma_t and the phase eccentricity g by a fixed geometric factor, so the
// summed contribution forms a convergent series that only ever loses energy.
// The CPU golden `octave_scatter` (in
// `prism_render_architecture::volumetric::scatter`) owns that math; this kernel
// is the on-device twin that runs one thread per query and writes the same
// `(sigma_s, sigma_t, g)` triple the reference returns.
//
// The kernel mirrors the reference exactly: base coefficients are clamped
// non-negative, the base eccentricity is clamped to +/-MAX_ABS_G, the three
// geometric factors are saturated into [0, 1], the octave index is clamped to
// the last octave (`octave_count - 1`, with `octave_count == 0` collapsing to
// octave 0), and each geometric factor is raised to the octave index by the
// same repeated-multiply loop `pow_u`. It uses only the portable core-WGSL
// subset (mul + compare) - no exp, no pow, no optional feature - so this twin
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Wrenninge-style octave multiple-scattering decay; no
// Unreal Engine source or derived code.

// Largest anisotropy magnitude accepted for the base eccentricity.
const MAX_ABS_G: f32 = 0.999;

struct Params {
    // Per-octave decay of the extinction coefficient sigma_t, saturated [0, 1].
    attenuation: f32,
    // Per-octave decay of the scattering coefficient sigma_s, saturated [0, 1].
    contribution: f32,
    // Per-octave decay of the phase eccentricity g, saturated [0, 1].
    eccentricity_attenuation: f32,
    // Number of octaves summed; indices at or beyond this clamp to the last.
    octave_count: u32,
    // Number of valid entries in `queries` / outputs in `results`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One octave query. `16`-byte stride matching the `bytemuck` upload struct.
struct Query {
    base_sigma_s: f32,
    base_sigma_t: f32,
    base_g: f32,
    octave_index: u32,
}

// One octave result. `16`-byte stride: the triple plus one pad word.
struct Octave {
    sigma_s: f32,
    sigma_t: f32,
    g: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Octave>;

// Clamps `x` into `[lo, hi]`, mirroring `math::clamp`.
fn clamp_range(x: f32, lo: f32, hi: f32) -> f32 {
    if (x < lo) {
        return lo;
    } else if (x > hi) {
        return hi;
    }
    return x;
}

// Clamps `x` into `[0, 1]`, mirroring `math::saturate`.
fn saturate_v(x: f32) -> f32 {
    return clamp_range(x, 0.0, 1.0);
}

// `base` raised to the non-negative integer power `exp` by repeated
// multiplication, mirroring the reference's `pow_u` (`pow_u(_, 0) == 1`).
fn pow_u(base: f32, exp: u32) -> f32 {
    var acc = 1.0;
    var i = 0u;
    while (i < exp) {
        acc = acc * base;
        i = i + 1u;
    }
    return acc;
}

@compute @workgroup_size(64)
fn octave(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let sigma_s0 = max(q.base_sigma_s, 0.0);
    let sigma_t0 = max(q.base_sigma_t, 0.0);
    let g0 = clamp_range(q.base_g, -MAX_ABS_G, MAX_ABS_G);

    let attenuation = saturate_v(params.attenuation);
    let contribution = saturate_v(params.contribution);
    let eccentricity = saturate_v(params.eccentricity_attenuation);

    // `octave_count.saturating_sub(1)`: 0 stays 0, otherwise count - 1.
    var last = 0u;
    if (params.octave_count > 0u) {
        last = params.octave_count - 1u;
    }
    var i = q.octave_index;
    if (i > last) {
        i = last;
    }

    var out: Octave;
    out.sigma_s = sigma_s0 * pow_u(contribution, i);
    out.sigma_t = sigma_t0 * pow_u(attenuation, i);
    out.g = g0 * pow_u(eccentricity, i);
    out.pad = 0.0;
    results[idx] = out;
}
