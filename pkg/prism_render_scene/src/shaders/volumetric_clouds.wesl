// Volumetric cloud compute kernels — GPU twin of the CPU golden modules in
// `prism_render_architecture::volumetric`.
//
// The sibling CPU modules own the reference numerics and every per-frame
// decision; this shader is their on-device mirror, one `@compute` entry per
// `prism_render_architecture::volumetric::gpu::kernels::VolumetricKernel`:
//
//   * `volumetric_weather_advect`   (@workgroup_size(8,8,1))  — semi-Lagrangian
//     back-trace of the weather map, mirroring `weather::advect_semi_lagrangian`
//     + `WeatherField::sample_bilinear`.
//   * `volumetric_noise_bake`        (@workgroup_size(4,4,4)) — bakes the
//     Perlin-Worley base + inverted-Worley fBm detail into the 3D density
//     cache, mirroring `noise::{perlin_worley, worley_fbm}`.
//   * `volumetric_modeling`          (@workgroup_size(4,4,4)) — composes the
//     density field (coverage remap / height gradient / detail erosion),
//     mirroring `modeling::compose_density`.
//   * `volumetric_multiscatter_lut_bake` (@workgroup_size(4,4,4)) —
//     pre-integrates the (cos x depth x albedo) energy-gain LUT, mirroring
//     `multiscatter::energy_gain`.
//   * `volumetric_raymarch`          (@workgroup_size(8,8,1)) — adaptive
//     low-res view march with empty-space skip + early-out, mirroring
//     `raymarch::{adaptive_step, integrate_segment, march}`.
//   * `volumetric_scatter_resolve`   (@workgroup_size(8,8,1)) — folds the
//     multi-scatter LUT + dual-lobe phase into the march result, mirroring
//     `scatter::{dual_lobe_phase, octave_scatter}`.
//   * `volumetric_shadow_march`      (@workgroup_size(8,8,1)) — light-space
//     cloud-shadow / AVSM march, mirroring `shadow::{accumulate_shadow,
//     shadow_transmittance}`.
//   * `volumetric_upsample`          (@workgroup_size(8,8,1)) — temporal
//     reprojection + upsample, mirroring `temporal::{active_pixel,
//     clamp_history, variance_clip}`.
//
// Every resource is assigned a unique `@group(0)` binding across the whole
// module (each entry is its own pipeline with its own layout, and unused
// module-scope vars are valid), so no two vars ever share a binding slot. The
// CPU crate policy allows only `sqrt` among the float intrinsics and uses
// rational/polynomial `exp`/`pow` approximations; the GPU has no such limit, so
// this shader uses the native `exp`/`pow`/`sqrt`/`floor`/`log` intrinsics.
// Self-contained (no intra-crate `import`s, matching ssgi.wesl / outline.wesl /
// volumetrics.wesl) so the WESL compile test validates it standalone with no
// GPU present.

// ---------------------------------------------------------------------------
// Shared constants + scalar helpers (mirror `volumetric::math`).
// ---------------------------------------------------------------------------

const VC_PI: f32 = 3.14159265358979323846;
const VC_INV_4PI: f32 = 1.0 / (4.0 * VC_PI);
const VC_EPS: f32 = 1.0e-6;
const VC_MAX_ABS_G: f32 = 0.999;

fn vc_saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn vc_lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Mirrors `math::remap`: collapsed input span yields a zero fraction so the map
// never divides by (near) zero.
fn vc_remap(x: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    let span = in_hi - in_lo;
    var t = 0.0;
    if (abs(span) >= VC_EPS) {
        t = (x - in_lo) / span;
    }
    return out_lo + (out_hi - out_lo) * t;
}

// Mirrors `math::smoothstep`: collapsed edges degrade to a hard step at edge0
// (matching the CPU branch) rather than dividing by zero.
fn vc_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (abs(span) < VC_EPS) {
        if (x < edge0) {
            return 0.0;
        }
        return 1.0;
    }
    let t = vc_saturate((x - edge0) / span);
    return t * t * (3.0 - 2.0 * t);
}

// ---------------------------------------------------------------------------
// Deterministic hash + noise (mirror `volumetric::noise`).
// ---------------------------------------------------------------------------

const VC_FNV_OFFSET_BASIS: u32 = 0x811c9dc5u;
const VC_FNV_PRIME: u32 = 0x01000193u;
const VC_WORLEY_SEED_Y: u32 = 0x68e31da4u;
const VC_WORLEY_SEED_Z: u32 = 0xb5279a45u;
const VC_FBM_SEED_STRIDE: u32 = 0x9e3779b9u;
const VC_PERLIN_WORLEY_WORLEY_FREQ: f32 = 2.0;
const VC_PERLIN_WORLEY_SEED_MIX: u32 = 0x27d4eb2fu;
const VC_CURL_SEED_G: u32 = 0x1b56c4e9u;
const VC_CURL_SEED_B: u32 = 0x7a4f3c1du;
const VC_CURL_EPS: f32 = 0.02;
const VC_LACUNARITY: f32 = 2.0;
const VC_GAIN: f32 = 0.5;

// Improved-Perlin gradient set: the twelve cube-edge midpoints padded to
// sixteen (four repeats) so a gradient is chosen with a 4-bit mask, exactly the
// `noise::GRAD3` table.
fn vc_gradient(h: u32) -> vec3<f32> {
    var g = array<vec3<f32>, 16>(
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
    return g[h & 15u];
}

// Mixes one 32-bit word into the running FNV-1a hash, byte by byte (u32
// multiply wraps by definition, matching the CPU `wrapping_mul`).
fn vc_fnv_word(hash_in: u32, word: u32) -> u32 {
    var hash = hash_in;
    var shift = 0u;
    for (var b = 0u; b < 4u; b = b + 1u) {
        hash = hash ^ ((word >> shift) & 0xFFu);
        hash = hash * VC_FNV_PRIME;
        shift = shift + 8u;
    }
    return hash;
}

// Integer hash of a lattice point + seed; the i32 coordinate is taken by its
// bit pattern (CPU `ix as u32`), then an xorshift-multiply avalanche disperses
// neighbouring cells.
fn vc_hash_cell(ix: i32, iy: i32, iz: i32, seed: u32) -> u32 {
    var h = VC_FNV_OFFSET_BASIS;
    h = vc_fnv_word(h, seed);
    h = vc_fnv_word(h, bitcast<u32>(ix));
    h = vc_fnv_word(h, bitcast<u32>(iy));
    h = vc_fnv_word(h, bitcast<u32>(iz));
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}

// Maps a hash to [0, 1) via its top twenty-four bits (2^24 == 16777216).
fn vc_hash_to_unit(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

// Quintic fade 6t^5 - 15t^4 + 10t^3.
fn vc_fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn vc_grad_dot(ix: i32, iy: i32, iz: i32, dx: f32, dy: f32, dz: f32, seed: u32) -> f32 {
    let g = vc_gradient(vc_hash_cell(ix, iy, iz, seed));
    return g.x * dx + g.y * dy + g.z * dz;
}

// Raw signed gradient noise in roughly [-1, 1] (`noise::perlin_raw`).
fn vc_perlin_raw(p: vec3<f32>, seed: u32) -> f32 {
    let xi = floor(p.x);
    let yi = floor(p.y);
    let zi = floor(p.z);
    let x0 = i32(xi);
    let y0 = i32(yi);
    let z0 = i32(zi);
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let z1 = z0 + 1;
    let fx = p.x - xi;
    let fy = p.y - yi;
    let fz = p.z - zi;
    let u = vc_fade(fx);
    let v = vc_fade(fy);
    let w = vc_fade(fz);
    let n000 = vc_grad_dot(x0, y0, z0, fx, fy, fz, seed);
    let n100 = vc_grad_dot(x1, y0, z0, fx - 1.0, fy, fz, seed);
    let n010 = vc_grad_dot(x0, y1, z0, fx, fy - 1.0, fz, seed);
    let n110 = vc_grad_dot(x1, y1, z0, fx - 1.0, fy - 1.0, fz, seed);
    let n001 = vc_grad_dot(x0, y0, z1, fx, fy, fz - 1.0, seed);
    let n101 = vc_grad_dot(x1, y0, z1, fx - 1.0, fy, fz - 1.0, seed);
    let n011 = vc_grad_dot(x0, y1, z1, fx, fy - 1.0, fz - 1.0, seed);
    let n111 = vc_grad_dot(x1, y1, z1, fx - 1.0, fy - 1.0, fz - 1.0, seed);
    let x00 = vc_lerp(n000, n100, u);
    let x10 = vc_lerp(n010, n110, u);
    let x01 = vc_lerp(n001, n101, u);
    let x11 = vc_lerp(n011, n111, u);
    let y00 = vc_lerp(x00, x10, v);
    let y11 = vc_lerp(x01, x11, v);
    return vc_lerp(y00, y11, w);
}

// Perlin noise normalized to [0, 1] (`noise::perlin_3d`).
fn vc_perlin_3d(p: vec3<f32>, seed: u32) -> f32 {
    return vc_saturate(0.5 + 0.5 * vc_perlin_raw(p, seed));
}

// Worley (cellular) distance noise in [0, 1] (`noise::worley_3d`).
fn vc_worley_3d(p: vec3<f32>, seed: u32) -> f32 {
    let xi = floor(p.x);
    let yi = floor(p.y);
    let zi = floor(p.z);
    let x0 = i32(xi);
    let y0 = i32(yi);
    let z0 = i32(zi);
    let fx = p.x - xi;
    let fy = p.y - yi;
    let fz = p.z - zi;
    var min_sq = 1.0e30;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let cx = x0 + dx;
                let cy = y0 + dy;
                let cz = z0 + dz;
                let ox = vc_hash_to_unit(vc_hash_cell(cx, cy, cz, seed));
                let oy = vc_hash_to_unit(vc_hash_cell(cx, cy, cz, seed ^ VC_WORLEY_SEED_Y));
                let oz = vc_hash_to_unit(vc_hash_cell(cx, cy, cz, seed ^ VC_WORLEY_SEED_Z));
                let fpx = f32(dx) + ox - fx;
                let fpy = f32(dy) + oy - fy;
                let fpz = f32(dz) + oz - fz;
                let d2 = fpx * fpx + fpy * fpy + fpz * fpz;
                if (d2 < min_sq) {
                    min_sq = d2;
                }
            }
        }
    }
    return vc_saturate(sqrt(min_sq));
}

// Inverted Worley `1 - worley_3d` (`noise::worley_inverted`).
fn vc_worley_inverted(p: vec3<f32>, seed: u32) -> f32 {
    return vc_saturate(1.0 - vc_worley_3d(p, seed));
}

// Perlin fBm in [0, 1] (`noise::fbm`).
fn vc_fbm(p: vec3<f32>, seed: u32, octaves: u32) -> f32 {
    var freq = 1.0;
    var amp = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    for (var i = 0u; i < octaves; i = i + 1u) {
        let octave_seed = seed + i * VC_FBM_SEED_STRIDE;
        sum = sum + amp * vc_perlin_3d(p * freq, octave_seed);
        norm = norm + amp;
        freq = freq * VC_LACUNARITY;
        amp = amp * VC_GAIN;
    }
    if (norm > VC_EPS) {
        return vc_saturate(sum / norm);
    }
    return 0.0;
}

// Inverted-Worley fBm in [0, 1] (`noise::worley_fbm`).
fn vc_worley_fbm(p: vec3<f32>, seed: u32, octaves: u32) -> f32 {
    var freq = 1.0;
    var amp = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    for (var i = 0u; i < octaves; i = i + 1u) {
        let octave_seed = seed + i * VC_FBM_SEED_STRIDE;
        sum = sum + amp * vc_worley_inverted(p * freq, octave_seed);
        norm = norm + amp;
        freq = freq * VC_LACUNARITY;
        amp = amp * VC_GAIN;
    }
    if (norm > VC_EPS) {
        return vc_saturate(sum / norm);
    }
    return 0.0;
}

// Perlin-Worley base cloud noise (`noise::perlin_worley`, default 5/3 octaves).
fn vc_perlin_worley(p: vec3<f32>, seed: u32) -> f32 {
    let perlin = vc_fbm(p, seed, 5u);
    let billow = vc_worley_fbm(
        p * VC_PERLIN_WORLEY_WORLEY_FREQ,
        seed ^ VC_PERLIN_WORLEY_SEED_MIX,
        3u,
    );
    return vc_saturate(vc_remap(perlin, billow - 1.0, 1.0, 0.0, 1.0));
}

// Three-component vector potential (`noise::curl_potential`).
fn vc_curl_potential(p: vec3<f32>, seed: u32) -> vec3<f32> {
    let a = vc_perlin_raw(p, seed);
    let b = vc_perlin_raw(p + vec3<f32>(31.416, 47.853, 12.793), seed ^ VC_CURL_SEED_G);
    let c = vc_perlin_raw(p + vec3<f32>(-53.217, 19.271, 83.155), seed ^ VC_CURL_SEED_B);
    return vec3<f32>(a, b, c);
}

// Divergence-free curl noise (`noise::curl_noise_3d`), central differences.
fn vc_curl_noise_3d(p: vec3<f32>, seed: u32) -> vec3<f32> {
    let e = VC_CURL_EPS;
    let inv = 1.0 / (2.0 * e);
    let px1 = vc_curl_potential(p + vec3<f32>(e, 0.0, 0.0), seed);
    let px0 = vc_curl_potential(p - vec3<f32>(e, 0.0, 0.0), seed);
    let py1 = vc_curl_potential(p + vec3<f32>(0.0, e, 0.0), seed);
    let py0 = vc_curl_potential(p - vec3<f32>(0.0, e, 0.0), seed);
    let pz1 = vc_curl_potential(p + vec3<f32>(0.0, 0.0, e), seed);
    let pz0 = vc_curl_potential(p - vec3<f32>(0.0, 0.0, e), seed);
    let dpz_dy = (py1.z - py0.z) * inv;
    let dpy_dz = (pz1.y - pz0.y) * inv;
    let dpx_dz = (pz1.x - pz0.x) * inv;
    let dpz_dx = (px1.z - px0.z) * inv;
    let dpy_dx = (px1.y - px0.y) * inv;
    let dpx_dy = (py1.x - py0.x) * inv;
    return vec3<f32>(dpz_dy - dpy_dz, dpx_dz - dpz_dx, dpy_dx - dpx_dy);
}

// ---------------------------------------------------------------------------
// Density modelling (mirror `volumetric::modeling`).
// ---------------------------------------------------------------------------

// Per-kind height profile `(rise_lo, rise_hi, fall_lo, fall_hi)`
// (`modeling::HeightProfile::for_kind`). kind: 0 Stratus, 1 Cumulus,
// 2 Cirrus, 3 Cumulonimbus.
fn vc_height_profile(kind: u32) -> vec4<f32> {
    switch (kind) {
        case 0u: {
            return vec4<f32>(0.00, 0.10, 0.55, 0.75);
        }
        case 1u: {
            return vec4<f32>(0.08, 0.30, 0.65, 0.95);
        }
        case 2u: {
            return vec4<f32>(0.05, 0.25, 0.55, 0.85);
        }
        case 3u: {
            return vec4<f32>(0.03, 0.12, 0.85, 1.00);
        }
        default: {
            return vec4<f32>(0.08, 0.30, 0.65, 0.95);
        }
    }
}

// `modeling::HeightProfile::weight` / `height_gradient`.
fn vc_height_gradient(height_fraction: f32, kind: u32) -> f32 {
    let prof = vc_height_profile(kind);
    let h = vc_saturate(height_fraction);
    let rising = vc_smoothstep(prof.x, prof.y, h);
    let falling = 1.0 - vc_smoothstep(prof.z, prof.w, h);
    return vc_saturate(rising * falling);
}

// `modeling::coverage_remap`.
fn vc_coverage_remap(base_shape: f32, coverage: f32) -> f32 {
    let base = vc_saturate(base_shape);
    let cov = vc_saturate(coverage);
    return vc_saturate(vc_remap(base, 1.0 - cov, 1.0, 0.0, 1.0));
}

// `modeling::detail_erosion`.
fn vc_detail_erosion(base_density: f32, detail_noise: f32, erosion_strength: f32) -> f32 {
    let base = vc_saturate(base_density);
    let detail = vc_saturate(detail_noise);
    let strength = vc_saturate(erosion_strength);
    let eaten = detail * strength;
    return vc_saturate(vc_remap(base, eaten, 1.0, 0.0, 1.0));
}

// `modeling::cloud_type_shape`.
fn vc_cloud_type_shape(base_low: f32, base_high: f32, cloud_type: f32) -> f32 {
    let low = vc_saturate(base_low);
    let high = vc_saturate(base_high);
    let t = vc_saturate(cloud_type);
    return vc_saturate(vc_lerp(low, high, t));
}

// `modeling::combine_coverage`.
fn vc_combine_coverage(base_coverage: f32, weather_coverage: f32) -> f32 {
    return vc_saturate(vc_saturate(base_coverage) * vc_saturate(weather_coverage));
}

// `modeling::compose_density`.
fn vc_compose_density(
    base_shape: f32,
    coverage: f32,
    height_weight: f32,
    detail_noise: f32,
    erosion_strength: f32,
) -> f32 {
    let covered = vc_coverage_remap(base_shape, coverage);
    let shaped = vc_saturate(covered * vc_saturate(height_weight));
    return vc_detail_erosion(shaped, detail_noise, erosion_strength);
}

// ---------------------------------------------------------------------------
// Scattering phase + octave decay (mirror `volumetric::scatter`).
// ---------------------------------------------------------------------------

// `scatter::hg_phase`.
fn vc_hg_phase(cos_theta: f32, g_in: f32) -> f32 {
    let u = clamp(cos_theta, -1.0, 1.0);
    let g = clamp(g_in, -VC_MAX_ABS_G, VC_MAX_ABS_G);
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * u, VC_EPS);
    return VC_INV_4PI * (1.0 - g2) / (denom * sqrt(denom));
}

// `scatter::dual_lobe_phase` (backward lobe is the lerp `a`, forward the `b`).
fn vc_dual_lobe_phase(cos_theta: f32, g_forward: f32, g_backward: f32, blend: f32) -> f32 {
    let forward = vc_hg_phase(cos_theta, g_forward);
    let backward = vc_hg_phase(cos_theta, g_backward);
    return vc_lerp(backward, forward, vc_saturate(blend));
}

// `scatter::draine_phase`.
fn vc_draine_phase(cos_theta: f32, g_in: f32, alpha_in: f32) -> f32 {
    let u = clamp(cos_theta, -1.0, 1.0);
    let g = clamp(g_in, -VC_MAX_ABS_G, VC_MAX_ABS_G);
    let alpha = max(alpha_in, 0.0);
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * u, VC_EPS);
    let norm = 1.0 + alpha * (1.0 + 2.0 * g2) / 3.0;
    return VC_INV_4PI * (1.0 - g2) * (1.0 + alpha * u * u) / (norm * denom * sqrt(denom));
}

// `scatter::hg_draine_phase`.
fn vc_hg_draine_phase(cos_theta: f32, g_hg: f32, g_draine: f32, alpha: f32, hg_weight: f32) -> f32 {
    let hg = vc_hg_phase(cos_theta, g_hg);
    let draine = vc_draine_phase(cos_theta, g_draine, alpha);
    return vc_lerp(draine, hg, vc_saturate(hg_weight));
}

// `scatter::powder`.
fn vc_powder(density_along_view: f32, strength: f32) -> f32 {
    let d = max(density_along_view, 0.0);
    let s = max(strength, 0.0);
    return vc_saturate(1.0 - exp(-2.0 * d * s));
}

// Integer power by repeated multiplication (`scatter::pow_u`).
fn vc_pow_u(base: f32, e: u32) -> f32 {
    var acc = 1.0;
    for (var i = 0u; i < e; i = i + 1u) {
        acc = acc * base;
    }
    return acc;
}

// `scatter::octave_scatter` -> (sigma_s, sigma_t, g).
fn vc_octave_scatter(
    base_sigma_s: f32,
    base_sigma_t: f32,
    base_g: f32,
    octave_index: u32,
    attenuation: f32,
    contribution: f32,
    eccentricity: f32,
    octave_count: u32,
) -> vec3<f32> {
    let sigma_s0 = max(base_sigma_s, 0.0);
    let sigma_t0 = max(base_sigma_t, 0.0);
    let g0 = clamp(base_g, -VC_MAX_ABS_G, VC_MAX_ABS_G);
    let att = vc_saturate(attenuation);
    let con = vc_saturate(contribution);
    let ecc = vc_saturate(eccentricity);
    var last = 0u;
    if (octave_count > 0u) {
        last = octave_count - 1u;
    }
    let i = min(octave_index, last);
    let sigma_s = sigma_s0 * vc_pow_u(con, i);
    let sigma_t = sigma_t0 * vc_pow_u(att, i);
    let g = g0 * vc_pow_u(ecc, i);
    return vec3<f32>(sigma_s, sigma_t, g);
}

// ---------------------------------------------------------------------------
// Multiple-scatter energy gain (mirror `volumetric::multiscatter`).
// ---------------------------------------------------------------------------

const VC_LUT_G: f32 = 0.7;
const VC_MAX_OPTICAL_DEPTH: f32 = 8.0;

// `multiscatter::axis_fraction`.
fn vc_axis_fraction(value: f32, mn: f32, mx: f32) -> f32 {
    let span = mx - mn;
    if (abs(span) < VC_EPS) {
        return 0.0;
    }
    return vc_saturate((value - mn) / span);
}

// `multiscatter::octave_phase_weight` (LUT forward g == 0.7).
fn vc_octave_phase_weight(
    cos_theta: f32,
    attenuation: f32,
    contribution: f32,
    eccentricity: f32,
    octave_count: u32,
) -> f32 {
    var numerator = 0.0;
    var denominator = 0.0;
    for (var n = 0u; n < octave_count; n = n + 1u) {
        let oct = vc_octave_scatter(
            1.0, 1.0, VC_LUT_G, n, attenuation, contribution, eccentricity, octave_count,
        );
        let sigma_s = oct.x;
        let g = oct.z;
        let phase = vc_hg_phase(cos_theta, g);
        let peak = vc_hg_phase(1.0, g);
        var ratio = 1.0;
        if (peak > VC_EPS) {
            ratio = vc_saturate(phase / peak);
        }
        numerator = numerator + sigma_s * ratio;
        denominator = denominator + sigma_s;
    }
    if (denominator > VC_EPS) {
        return vc_saturate(numerator / denominator);
    }
    return 1.0;
}

// `multiscatter::energy_gain`.
fn vc_energy_gain(
    cos_theta: f32,
    optical_depth: f32,
    albedo: f32,
    attenuation: f32,
    contribution: f32,
    eccentricity: f32,
    octave_count: u32,
) -> f32 {
    let a = vc_saturate(albedo);
    let s = sqrt(vc_saturate(1.0 - a));
    var reflectance = 1.0;
    if (1.0 + s > VC_EPS) {
        reflectance = (1.0 - s) / (1.0 + s);
    }
    let depth = max(optical_depth, 0.0);
    let depth_ramp = vc_saturate(1.0 - exp(-depth));
    let phase_mod = vc_octave_phase_weight(cos_theta, attenuation, contribution, eccentricity, octave_count);
    return vc_saturate(reflectance * depth_ramp * phase_mod);
}

// ---------------------------------------------------------------------------
// Shadow / god-ray (mirror `volumetric::shadow`).
// ---------------------------------------------------------------------------

// `shadow::shadow_transmittance`.
fn vc_shadow_transmittance(optical_depth: f32) -> f32 {
    let od = max(optical_depth, 0.0);
    return vc_saturate(exp(-od));
}

// `shadow::god_ray_weight` (native pow for `decay^index`).
fn vc_god_ray_weight(sample_index: u32, decay: f32, weight: f32) -> f32 {
    let d = vc_saturate(decay);
    let w = vc_saturate(weight);
    return vc_saturate(w * pow(d, f32(sample_index)));
}

// `shadow::scattering_mask`.
fn vc_scattering_mask(shadow_transmittance: f32, density: f32) -> f32 {
    return vc_saturate(shadow_transmittance) * vc_saturate(density);
}

// ---------------------------------------------------------------------------
// Temporal reprojection (mirror `volumetric::temporal`).
// ---------------------------------------------------------------------------

// `temporal::active_pixel`. mode: 0 Full, 1 Checkerboard, 2 QuarterRes.
fn vc_active_pixel(frame_index: u32, x: u32, y: u32, mode: u32) -> bool {
    switch (mode) {
        case 1u: {
            return ((x ^ y ^ frame_index) & 1u) == 0u;
        }
        case 2u: {
            return (x & 1u) == (frame_index & 1u) && (y & 1u) == ((frame_index >> 1u) & 1u);
        }
        default: {
            return true;
        }
    }
}

// `temporal::clamp_history` (orders inverted bounds first).
fn vc_clamp_history(history_sample: f32, history_min: f32, history_max: f32) -> f32 {
    var lo = history_min;
    var hi = history_max;
    if (history_min > history_max) {
        lo = history_max;
        hi = history_min;
    }
    return clamp(history_sample, lo, hi);
}

// `temporal::variance_clip`.
fn vc_variance_clip(history_sample: f32, mean: f32, std_dev: f32, gamma: f32) -> f32 {
    let sd = max(std_dev, 0.0);
    let g = max(gamma, 0.0);
    let half = sd * g;
    return clamp(history_sample, mean - half, mean + half);
}

// ---------------------------------------------------------------------------
// group(0) resources — unique binding per resource across the whole module.
// ---------------------------------------------------------------------------

// Weather advect.
@group(0) @binding(0) var vc_weather_src: texture_2d<f32>;
@group(0) @binding(1) var vc_weather_dst: texture_storage_2d<rgba16float, write>;
// Noise bake.
@group(0) @binding(2) var vc_noise_density_out: texture_storage_3d<rgba16float, write>;
// Modeling.
@group(0) @binding(3) var vc_modeling_noise_in: texture_3d<f32>;
@group(0) @binding(4) var vc_modeling_weather_in: texture_2d<f32>;
@group(0) @binding(5) var vc_modeling_density_out: texture_storage_3d<rgba16float, write>;
// Multiscatter LUT bake.
@group(0) @binding(6) var vc_mslut_out: texture_storage_3d<rgba16float, write>;
// Raymarch.
@group(0) @binding(7) var vc_rm_density_in: texture_3d<f32>;
@group(0) @binding(8) var vc_rm_shadow_in: texture_2d<f32>;
@group(0) @binding(9) var vc_rm_scatter_out: texture_storage_2d<rgba16float, write>;
// Scatter resolve.
@group(0) @binding(10) var vc_sr_lowres_in: texture_2d<f32>;
@group(0) @binding(11) var vc_sr_lut_in: texture_3d<f32>;
@group(0) @binding(12) var vc_sr_out: texture_storage_2d<rgba16float, write>;
// Shadow march.
@group(0) @binding(13) var vc_sm_density_in: texture_3d<f32>;
@group(0) @binding(14) var vc_sm_shadow_out: texture_storage_2d<rgba16float, write>;
// Upsample.
@group(0) @binding(15) var vc_us_lowres_in: texture_2d<f32>;
@group(0) @binding(16) var vc_us_history_in: texture_2d<f32>;
@group(0) @binding(17) var vc_us_out: texture_storage_2d<rgba16float, write>;

// ---------------------------------------------------------------------------
// Kernel 1: weather advect (WeatherTexel domain).
// ---------------------------------------------------------------------------

struct VcWeatherAdvectParams {
    width: u32,
    height: u32,
    wind_x: f32,
    wind_y: f32,
    dt: f32,
}

var<immediate> vc_advect_params: VcWeatherAdvectParams;

// Manual clamp-to-edge bilinear tap of the weather map, mirroring
// `WeatherField::sample_bilinear` (four `textureLoad` taps convex-combined).
fn vc_weather_sample_bilinear(u: f32, v: f32, w: u32, h: u32) -> vec4<f32> {
    let max_x = f32(w - 1u);
    let max_y = f32(h - 1u);
    let cx = clamp(u, 0.0, max_x);
    let cy = clamp(v, 0.0, max_y);
    let x0 = u32(cx);
    let y0 = u32(cy);
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let fx = cx - f32(x0);
    let fy = cy - f32(y0);
    let s00 = textureLoad(vc_weather_src, vec2<i32>(i32(x0), i32(y0)), 0);
    let s10 = textureLoad(vc_weather_src, vec2<i32>(i32(x1), i32(y0)), 0);
    let s01 = textureLoad(vc_weather_src, vec2<i32>(i32(x0), i32(y1)), 0);
    let s11 = textureLoad(vc_weather_src, vec2<i32>(i32(x1), i32(y1)), 0);
    let top = mix(s00, s10, fx);
    let bottom = mix(s01, s11, fx);
    return mix(top, bottom, fy);
}

@compute @workgroup_size(8, 8, 1)
fn volumetric_weather_advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = vc_advect_params.width;
    let h = vc_advect_params.height;
    if (gid.x >= w || gid.y >= h || w == 0u || h == 0u) {
        return;
    }
    // Semi-Lagrangian back-trace: sample the previous field at the departure
    // cell `(x - wind.x*dt, y - wind.y*dt)` in cell units.
    let px = f32(gid.x) - vc_advect_params.wind_x * vc_advect_params.dt;
    let py = f32(gid.y) - vc_advect_params.wind_y * vc_advect_params.dt;
    let advected = vc_weather_sample_bilinear(px, py, w, h);
    textureStore(vc_weather_dst, vec2<i32>(i32(gid.x), i32(gid.y)), advected);
}

// ---------------------------------------------------------------------------
// Kernel 2: noise bake (DensityVoxel domain).
// ---------------------------------------------------------------------------

struct VcNoiseBakeParams {
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    base_freq: f32,
    detail_freq: f32,
    seed: u32,
}

var<immediate> vc_noise_params: VcNoiseBakeParams;

@compute @workgroup_size(4, 4, 4)
fn volumetric_noise_bake(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = vec3<u32>(vc_noise_params.dim_x, vc_noise_params.dim_y, vc_noise_params.dim_z);
    if (gid.x >= dim.x || gid.y >= dim.y || gid.z >= dim.z) {
        return;
    }
    // Voxel-centre position in [0, 1)^3 scaled to noise-space frequency.
    let uvw = (vec3<f32>(f32(gid.x), f32(gid.y), f32(gid.z)) + vec3<f32>(0.5))
        / vec3<f32>(f32(dim.x), f32(dim.y), f32(dim.z));
    let base_p = uvw * vc_noise_params.base_freq;
    let detail_p = uvw * vc_noise_params.detail_freq;
    // Base Perlin-Worley shape and high-frequency inverted-Worley erosion detail.
    let base = vc_perlin_worley(base_p, vc_noise_params.seed);
    let detail = vc_worley_fbm(detail_p, vc_noise_params.seed ^ 0x5bd1e995u, 3u);
    let coord = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
    textureStore(vc_noise_density_out, coord, vec4<f32>(base, detail, 0.0, 1.0));
}

// ---------------------------------------------------------------------------
// Kernel 3: modelling (DensityVoxel domain).
// ---------------------------------------------------------------------------

struct VcModelingParams {
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    coverage: f32,
    cloud_type: f32,
    erosion_strength: f32,
    kind: u32,
}

var<immediate> vc_modeling_params: VcModelingParams;

@compute @workgroup_size(4, 4, 4)
fn volumetric_modeling(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = vec3<u32>(vc_modeling_params.dim_x, vc_modeling_params.dim_y, vc_modeling_params.dim_z);
    if (gid.x >= dim.x || gid.y >= dim.y || gid.z >= dim.z) {
        return;
    }
    let coord = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
    // Baked base shape + detail from the noise pass.
    let noise = textureLoad(vc_modeling_noise_in, coord, 0);
    let base_shape = noise.x;
    let detail = noise.y;
    // Weather coverage/type sampled at this column (weather map is 2D).
    let weather = textureLoad(vc_modeling_weather_in, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let coverage = vc_combine_coverage(vc_modeling_params.coverage, weather.x);
    let cloud_type = vc_cloud_type_shape(base_shape, base_shape, vc_modeling_params.cloud_type);
    // Height fraction is the normalised Z within the layer band.
    let height_fraction = (f32(gid.z) + 0.5) / f32(dim.z);
    let height = vc_height_gradient(height_fraction, vc_modeling_params.kind);
    let density = vc_compose_density(
        cloud_type,
        coverage,
        height,
        detail,
        vc_modeling_params.erosion_strength,
    );
    textureStore(vc_modeling_density_out, coord, vec4<f32>(density, 0.0, 0.0, 1.0));
}

// ---------------------------------------------------------------------------
// Kernel 4: multiscatter LUT bake (MultiscatterLut domain).
// ---------------------------------------------------------------------------

struct VcMsLutParams {
    dim_cos: u32,
    dim_depth: u32,
    dim_albedo: u32,
    attenuation: f32,
    contribution: f32,
    eccentricity: f32,
    octave_count: u32,
}

var<immediate> vc_mslut_params: VcMsLutParams;

@compute @workgroup_size(4, 4, 4)
fn volumetric_multiscatter_lut_bake(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dc = vc_mslut_params.dim_cos;
    let dd = vc_mslut_params.dim_depth;
    let da = vc_mslut_params.dim_albedo;
    if (gid.x >= dc || gid.y >= dd || gid.z >= da) {
        return;
    }
    // Axis reconstruction: cos in [-1, 1] (outer), depth in [0, MAX] (middle),
    // albedo in [0, 1] (inner) — the `MultiScatterLut` fill order.
    var cos_theta = -1.0;
    if (dc > 1u) {
        cos_theta = vc_lerp(-1.0, 1.0, f32(gid.x) / f32(dc - 1u));
    }
    var depth = 0.0;
    if (dd > 1u) {
        depth = vc_lerp(0.0, VC_MAX_OPTICAL_DEPTH, f32(gid.y) / f32(dd - 1u));
    }
    var albedo = 0.0;
    if (da > 1u) {
        albedo = vc_lerp(0.0, 1.0, f32(gid.z) / f32(da - 1u));
    }
    let gain = vc_energy_gain(
        cos_theta,
        depth,
        albedo,
        vc_mslut_params.attenuation,
        vc_mslut_params.contribution,
        vc_mslut_params.eccentricity,
        vc_mslut_params.octave_count,
    );
    let coord = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
    textureStore(vc_mslut_out, coord, vec4<f32>(gain, 0.0, 0.0, 1.0));
}

// ---------------------------------------------------------------------------
// Kernel 5: view ray-march (ScreenTile domain).
// ---------------------------------------------------------------------------

struct VcRaymarchParams {
    screen_w: u32,
    screen_h: u32,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    base_step: f32,
    max_step: f32,
    min_step: f32,
    density_threshold: f32,
    transmittance_cutoff: f32,
    max_steps: u32,
    sigma_t: f32,
    albedo: f32,
    phase_g: f32,
}

var<immediate> vc_raymarch_params: VcRaymarchParams;

// `raymarch::adaptive_step`.
fn vc_adaptive_step(current_density: f32, in_cloud: bool) -> f32 {
    let base = vc_raymarch_params.base_step;
    let mx = vc_raymarch_params.max_step;
    let mn = vc_raymarch_params.min_step;
    if (!in_cloud || current_density < vc_raymarch_params.density_threshold) {
        return clamp(mx, mn, mx);
    }
    let t = vc_saturate(current_density);
    return clamp(vc_lerp(base, mn, t), mn, mx);
}

// Sample the baked density cache along the marching column. The march samples
// the density texture at the tile's (x, y) and the normalised depth `frac`.
fn vc_rm_density_at(tile: vec2<u32>, frac: f32) -> f32 {
    let gz = max(vc_raymarch_params.grid_z, 1u);
    let z = min(u32(vc_saturate(frac) * f32(gz)), gz - 1u);
    let coord = vec3<i32>(i32(tile.x), i32(tile.y), i32(z));
    return textureLoad(vc_rm_density_in, coord, 0).x;
}

@compute @workgroup_size(8, 8, 1)
fn volumetric_raymarch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sw = vc_raymarch_params.screen_w;
    let sh = vc_raymarch_params.screen_h;
    if (gid.x >= sw || gid.y >= sh) {
        return;
    }
    let tile = vec2<u32>(gid.x, gid.y);
    let shadow_light = vc_saturate(textureLoad(vc_rm_shadow_in, vec2<i32>(i32(gid.x), i32(gid.y)), 0).x);
    let phase = vc_hg_phase(1.0, vc_raymarch_params.phase_g);
    let sigma_t_base = max(vc_raymarch_params.sigma_t, 0.0);
    let sigma_s_base = sigma_t_base * vc_saturate(vc_raymarch_params.albedo);

    // The march distance is the full [0, 1] normalised column; step lengths are
    // in the same normalised units so the loop stays bounded by max_steps.
    let distance = 1.0;
    var t = 0.0;
    var transmittance = 1.0;
    var scattered = 0.0;
    var optical_depth = 0.0;
    var steps = 0u;
    loop {
        if (t >= distance || steps >= vc_raymarch_params.max_steps) {
            break;
        }
        if (transmittance < vc_raymarch_params.transmittance_cutoff) {
            break;
        }
        let density = vc_rm_density_at(tile, t);
        let in_cloud = density >= vc_raymarch_params.density_threshold;
        var step = vc_adaptive_step(density, in_cloud);
        let remaining = distance - t;
        if (step > remaining) {
            step = remaining;
        }
        if (step <= VC_EPS) {
            break;
        }
        if (in_cloud) {
            // Analytic homogeneous segment integral (`raymarch::integrate_segment`).
            let sigma_t = sigma_t_base * density;
            let sigma_s = sigma_s_base * density;
            let light = shadow_light;
            let seg_optical = sigma_t * step;
            let seg_trans = exp(-seg_optical);
            var integral = step;
            if (sigma_t > VC_EPS) {
                integral = (1.0 - seg_trans) / sigma_t;
            }
            scattered = scattered + transmittance * light * sigma_s * phase * integral;
            transmittance = vc_saturate(transmittance * seg_trans);
            optical_depth = optical_depth + seg_optical;
        }
        steps = steps + 1u;
        t = t + step;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    textureStore(vc_rm_scatter_out, coord, vec4<f32>(scattered, scattered, scattered, transmittance));
}

// ---------------------------------------------------------------------------
// Kernel 6: octave-scatter resolve (ScreenTile domain).
// ---------------------------------------------------------------------------

struct VcScatterResolveParams {
    screen_w: u32,
    screen_h: u32,
    lut_cos: u32,
    lut_depth: u32,
    lut_albedo: u32,
    forward_g: f32,
    backward_g: f32,
    lobe_blend: f32,
    albedo: f32,
    cos_theta: f32,
}

var<immediate> vc_resolve_params: VcScatterResolveParams;

// Nearest-cell LUT fetch keyed on (cos, depth, albedo) fractions.
fn vc_sr_lut_sample(cos_theta: f32, depth: f32, albedo: f32) -> f32 {
    let fc = vc_axis_fraction(cos_theta, -1.0, 1.0);
    let fd = vc_axis_fraction(depth, 0.0, VC_MAX_OPTICAL_DEPTH);
    let fa = vc_axis_fraction(albedo, 0.0, 1.0);
    let dc = max(vc_resolve_params.lut_cos, 1u);
    let dd = max(vc_resolve_params.lut_depth, 1u);
    let da = max(vc_resolve_params.lut_albedo, 1u);
    let ic = min(u32(fc * f32(dc)), dc - 1u);
    let id = min(u32(fd * f32(dd)), dd - 1u);
    let ia = min(u32(fa * f32(da)), da - 1u);
    return textureLoad(vc_sr_lut_in, vec3<i32>(i32(ic), i32(id), i32(ia)), 0).x;
}

@compute @workgroup_size(8, 8, 1)
fn volumetric_scatter_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sw = vc_resolve_params.screen_w;
    let sh = vc_resolve_params.screen_h;
    if (gid.x >= sw || gid.y >= sh) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let lowres = textureLoad(vc_sr_lowres_in, coord, 0);
    let single = lowres.x;
    let transmittance = lowres.w;
    // Optical depth recovered from the surviving transmittance (`-ln T`).
    let optical_depth = -log(max(transmittance, VC_EPS));
    // Dual-lobe phase weighting + multi-scatter energy gain fold-in.
    let phase = vc_dual_lobe_phase(
        vc_resolve_params.cos_theta,
        vc_resolve_params.forward_g,
        vc_resolve_params.backward_g,
        vc_resolve_params.lobe_blend,
    );
    let gain = vc_sr_lut_sample(vc_resolve_params.cos_theta, optical_depth, vc_resolve_params.albedo);
    let resolved = single * (1.0 + gain) * (phase * 4.0 * VC_PI);
    textureStore(vc_sr_out, coord, vec4<f32>(resolved, resolved, resolved, transmittance));
}

// ---------------------------------------------------------------------------
// Kernel 7: light-space cloud-shadow / AVSM march (ShadowTexel domain).
// ---------------------------------------------------------------------------

struct VcShadowMarchParams {
    shadow_w: u32,
    shadow_h: u32,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    step: f32,
    density_scale: f32,
}

var<immediate> vc_shadow_params: VcShadowMarchParams;

@compute @workgroup_size(8, 8, 1)
fn volumetric_shadow_march(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sw = vc_shadow_params.shadow_w;
    let sh = vc_shadow_params.shadow_h;
    if (gid.x >= sw || gid.y >= sh) {
        return;
    }
    // Map the shadow texel to a density-cache column and march the depth slices
    // toward the light, accumulating optical depth (`shadow::accumulate_shadow`).
    let gx = max(vc_shadow_params.grid_x, 1u);
    let gy = max(vc_shadow_params.grid_y, 1u);
    let gz = max(vc_shadow_params.grid_z, 1u);
    let cx = min(gid.x * gx / max(sw, 1u), gx - 1u);
    let cy = min(gid.y * gy / max(sh, 1u), gy - 1u);
    let step = max(vc_shadow_params.step, 0.0);
    let scale = max(vc_shadow_params.density_scale, 0.0);
    var optical_depth = 0.0;
    for (var z = 0u; z < gz; z = z + 1u) {
        let coord = vec3<i32>(i32(cx), i32(cy), i32(z));
        let density = max(textureLoad(vc_sm_density_in, coord, 0).x, 0.0) * scale;
        optical_depth = optical_depth + density * step;
    }
    let transmittance = vc_shadow_transmittance(optical_depth);
    let out_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    textureStore(vc_sm_shadow_out, out_coord, vec4<f32>(transmittance, optical_depth, 0.0, 1.0));
}

// ---------------------------------------------------------------------------
// Kernel 8: temporal reprojection + upsample (ScreenPixel domain).
// ---------------------------------------------------------------------------

struct VcUpsampleParams {
    screen_w: u32,
    screen_h: u32,
    lowres_w: u32,
    lowres_h: u32,
    frame_index: u32,
    mode: u32,
    variance_gamma: f32,
}

var<immediate> vc_upsample_params: VcUpsampleParams;

@compute @workgroup_size(8, 8, 1)
fn volumetric_upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sw = vc_upsample_params.screen_w;
    let sh = vc_upsample_params.screen_h;
    if (gid.x >= sw || gid.y >= sh) {
        return;
    }
    let out_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    // Map the full-res pixel to its low-res source cell.
    let lw = max(vc_upsample_params.lowres_w, 1u);
    let lh = max(vc_upsample_params.lowres_h, 1u);
    let lx = min(gid.x * lw / max(sw, 1u), lw - 1u);
    let ly = min(gid.y * lh / max(sh, 1u), lh - 1u);
    let current = textureLoad(vc_us_lowres_in, vec2<i32>(i32(lx), i32(ly)), 0);
    let history = textureLoad(vc_us_history_in, out_coord, 0);

    // Freshly-marched pixels this frame take the current sample directly; the
    // rest reconstruct from the variance-clipped reprojected history.
    let fresh = vc_active_pixel(vc_upsample_params.frame_index, gid.x, gid.y, vc_upsample_params.mode);
    var out_color = current;
    if (!fresh) {
        // Anti-ghosting: clip each channel of the history toward the current
        // sample's neighbourhood mean (approximated by the current value) with
        // a small variance window, then blend.
        let mean = current.x;
        let sd = abs(current.x - history.x);
        let clipped_x = vc_variance_clip(history.x, mean, sd, vc_upsample_params.variance_gamma);
        let clamped_x = vc_clamp_history(clipped_x, min(current.x, history.x), max(current.x, history.x));
        let blended = vc_lerp(clamped_x, current.x, 0.5);
        out_color = vec4<f32>(blended, blended, blended, current.w);
    }
    textureStore(vc_us_out, out_coord, out_color);
}
