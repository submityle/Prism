// World-space ReSTIR DI — reservoir seed / per-cell light RIS pass (GPU twin).
//
// One compute invocation per resident reservoir-table slot. The pass reads the
// previous (`src`) reservoir table (whose occupied cells already carry the
// cell's visible-point geometry), draws `candidate_count` light candidates per
// occupied cell under streaming resampled importance sampling (RIS) over the
// frame's light list, caps the confidence `m` and finalizes the surviving
// reservoir's unbiased contribution weight `W`, then writes the seeded
// reservoir into the double-buffered (`dst`) table. The downstream fill pass
// then pools these seeded reservoirs across spatial neighbours under GRIS.
// Empty slots are copied through unchanged so the ping-pong preserves vacancy.
//
// This kernel is an arm-for-arm port of the CPU golden:
//   * RIS reservoir estimator (luminance, geometric term, target function,
//     streaming update, confidence cap, finalize):
//     `prism_render_shading::gi::screen_probe::restir` and
//     `prism_render_shading::gi::world_restir::world_reservoir`.
// The byte layouts of `WorldRestirReservoir`, `WorldRestirLight`, and
// `SeedParams` are frozen by `shading::world_restir::abi` (80-byte five-lane
// slot, 32-byte two-lane light, 32-byte two-lane immediate); the `shader_tests`
// ABI guards assert they stay in lockstep, and the seed CPU mirror asserts this
// estimator is bit-equal to the golden `Reservoir` under a finite-input sweep.
//
// Finite-math parity: WGSL lacks `isNan` / `isFinite`, so the golden's
// `x.is_finite()` guards are reduced to the `x == x` NaN probe and the
// `x > MIN_POSITIVE` / `x > 0` positivity tests (matching the fill twin's
// convention). All seed inputs — cell geometry (from the injection pass),
// light positions / radiance, and the uniform tunables — are finite by
// construction, so the surviving `inf`-only divergences are unreachable and are
// mirrored in Rust under a finite-only sweep, which is the only CPU-checkable
// proof of the port (the sandbox has no GPU).

// Smallest positive normal f32 (golden `f32::MIN_POSITIVE`).
const MIN_POSITIVE: f32 = 1.175494e-38;
// Rec. 709 luminance weights (golden `LUMA_R/G/B`).
const LUMA_R: f32 = 0.212639;
const LUMA_G: f32 = 0.715169;
const LUMA_B: f32 = 0.072192;

// One hashed world-cell reservoir slot — std430 twin of
// `abi::GpuWorldRestirReservoir` (five vec4 lanes, 80 bytes); byte-identical to
// the fill twin's `WorldRestirReservoir`.
struct WorldRestirReservoir {
    // visible_point.xyz + finalised contribution weight W.
    visible_point_w: vec4<f32>,
    // visible_normal.xyz + confidence m.
    visible_normal_m: vec4<f32>,
    // sample_point.xyz + SHARC collision checksum (bitcast to u32).
    sample_point_checksum: vec4<f32>,
    // sample_normal.xyz + valid flag (bitcast to u32).
    sample_normal_valid: vec4<f32>,
    // radiance.xyz + one pad word.
    radiance_pad: vec4<f32>,
}

// One candidate light — std430 twin of `abi::GpuWorldRestirLight` (two vec4
// lanes, 32 bytes).
struct WorldRestirLight {
    // emitter world position.xyz + scalar intensity.
    position_intensity: vec4<f32>,
    // linear RGB colour.xyz + one pad word.
    color_pad: vec4<f32>,
}

// Seed immediate block — std430 twin of `abi::GpuWorldRestirSeedParams`
// (two vec4 lanes, 32 bytes).
struct SeedParams {
    // (capacity, light_count, candidate_count, frame).
    dims: vec4<u32>,
    // (intensity, m_cap, pad, pad).
    tunables: vec4<f32>,
}

// A fully-resolved light sample — twin of `prism_render_shading` `GiSample`.
struct Sample {
    visible_point: vec3<f32>,
    visible_normal: vec3<f32>,
    sample_point: vec3<f32>,
    sample_normal: vec3<f32>,
    radiance: vec3<f32>,
}

@group(0) @binding(0) var<storage, read> src_reservoirs: array<WorldRestirReservoir>;
@group(0) @binding(1) var<storage, read_write> dst_reservoirs: array<WorldRestirReservoir>;
@group(0) @binding(2) var<storage, read> lights: array<WorldRestirLight>;

var<immediate> params: SeedParams;

// ---------------------------------------------------------------------------
// Deterministic per-slot RNG + light pick (GPU-authored; the golden takes
// uniforms as inputs, so any in-[0,1) stream keeps each update arm-for-arm).
// ---------------------------------------------------------------------------

fn fmix32(x_in: u32) -> u32 {
    var x = x_in;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

// Uniform in [0, 1) seeded by (frame, slot, counter).
fn rng01(frame: u32, slot: u32, counter: u32) -> f32 {
    let h = fmix32(frame * 0x9e3779b9u + fmix32(slot * 0x85ebca6bu + counter));
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

// Picks a light index in `[0, light_count)` for candidate `k` (uniform source
// pdf `1 / light_count`). `light_count` is non-zero at every call site.
fn pick_light(frame: u32, slot: u32, k: u32, light_count: u32) -> u32 {
    return fmix32(frame * 0x9e3779b9u + slot * 0x85ebca6bu + k * 0x165667b1u) % light_count;
}

// ---------------------------------------------------------------------------
// RIS estimator (golden `restir` + `world_reservoir`).
// ---------------------------------------------------------------------------

fn luminance(rgb: vec3<f32>) -> f32 {
    return LUMA_R * max(rgb.x, 0.0) + LUMA_G * max(rgb.y, 0.0) + LUMA_B * max(rgb.z, 0.0);
}

// Surface-to-surface geometry factor cos_v * cos_s / dist^2 (golden
// `geometric_term`); zero for degenerate / back-facing / non-finite pairs.
// `dist_sq.sqrt().recip()` (not `inverseSqrt`) reproduces the golden's exact
// IEEE reciprocal-square-root so the Rust float mirror is bit-equal.
fn geometric_term(visible_point: vec3<f32>, visible_normal: vec3<f32>, sample_point: vec3<f32>, sample_normal: vec3<f32>) -> f32 {
    let delta = sample_point - visible_point;
    let dist_sq = dot(delta, delta);
    if (dist_sq > MIN_POSITIVE) {
        let inv_dist = 1.0 / sqrt(dist_sq);
        let dir = delta * inv_dist;
        let cos_v = max(dot(visible_normal, dir), 0.0);
        let cos_s = max(dot(sample_normal, -dir), 0.0);
        let term = cos_v * cos_s / dist_sq;
        // `term == term` rejects NaN (golden `is_finite`); a finite >=0 term
        // passes through.
        return select(0.0, max(term, 0.0), term == term);
    }
    return 0.0;
}

// Scalar resampling target p_hat (golden `target_function`).
fn target_function(s: Sample) -> f32 {
    let g = geometric_term(s.visible_point, s.visible_normal, s.sample_point, s.sample_normal);
    let t = luminance(s.radiance) * g;
    return select(0.0, max(t, 0.0), t == t);
}

// Builds a light candidate from the cell's visible geometry and one emitter
// (golden sample construction): the secondary point is the emitter position,
// its normal faces the visible point, and the radiance is the emitter colour
// scaled by its intensity and the artistic gain.
fn build_sample(visible_point: vec3<f32>, visible_normal: vec3<f32>, light: WorldRestirLight, intensity: f32) -> Sample {
    var s: Sample;
    s.visible_point = visible_point;
    s.visible_normal = visible_normal;
    let sample_point = light.position_intensity.xyz;
    s.sample_point = sample_point;
    let delta = visible_point - sample_point;
    let dist_sq = dot(delta, delta);
    if (dist_sq > MIN_POSITIVE) {
        s.sample_normal = delta * (1.0 / sqrt(dist_sq));
    } else {
        s.sample_normal = vec3<f32>(0.0, 0.0, 1.0);
    }
    s.radiance = light.color_pad.xyz * (light.position_intensity.w * intensity);
    return s;
}

// Resampled-importance weight target_function / source_pdf (golden
// `ris_weight`); zero for a non-positive source pdf or a NaN quotient.
fn ris_weight(s: Sample, source_pdf: f32) -> f32 {
    if (!(source_pdf > 0.0)) {
        return 0.0;
    }
    let w = target_function(s) / source_pdf;
    return select(0.0, max(w, 0.0), w == w);
}

// ---------------------------------------------------------------------------
// Reservoir slot (de)serialisation.
// ---------------------------------------------------------------------------

fn reservoir_valid(r: WorldRestirReservoir) -> bool {
    return bitcast<u32>(r.sample_normal_valid.w) != 0u;
}

fn reservoir_checksum(r: WorldRestirReservoir) -> u32 {
    return bitcast<u32>(r.sample_point_checksum.w);
}

@compute @workgroup_size(64, 1, 1)
fn seed_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let capacity = params.dims.x;
    let slot = gid.x;
    if (slot >= capacity) {
        return;
    }

    let center = src_reservoirs[slot];

    // An empty slot carries no cell geometry to seed from: copy it through so
    // the ping-pong preserves vacancy.
    if (!reservoir_valid(center)) {
        dst_reservoirs[slot] = center;
        return;
    }

    let light_count = params.dims.y;
    let candidate_count = params.dims.z;
    let frame = params.dims.w;
    let intensity = params.tunables.x;
    let m_cap = params.tunables.y;

    // The occupied slot carries this cell's visible-point geometry (the
    // injection pass pre-seeds it); the seeded sample re-exposes the same
    // visible point / normal so the fill pass can reconnect neighbours to it.
    let visible_point = center.visible_point_w.xyz;
    let visible_normal = center.visible_normal_m.xyz;

    // Streaming RIS over the frame's lights (golden `stream_candidate` loop:
    // `update(sample, ris_weight(sample, 1 / light_count), u)` per candidate).
    var selected_sample: Sample;
    var selected: bool = false;
    var w_sum: f32 = 0.0;
    var m: f32 = 0.0;
    if (light_count > 0u) {
        let source_pdf = 1.0 / f32(light_count);
        for (var k: u32 = 0u; k < candidate_count; k = k + 1u) {
            let li = pick_light(frame, slot, k, light_count);
            let candidate = build_sample(visible_point, visible_normal, lights[li], intensity);
            let weight = ris_weight(candidate, source_pdf);
            // Golden `update` discards a non-positive weight without touching
            // state; a finite weight > 0 streams in. Seed inputs are finite, so
            // `weight > 0` matches the golden `is_finite && > 0` guard.
            if (weight > 0.0) {
                w_sum = w_sum + weight;
                m = m + 1.0;
                let u = clamp(rng01(frame, slot, k), 0.0, 1.0);
                if (u * w_sum <= weight) {
                    selected_sample = candidate;
                    selected = true;
                }
            }
        }
    }

    // Cap confidence (golden `cap_confidence`) then finalize `W` from the
    // surviving sample's own target density (golden `finalize`).
    if (m_cap >= 0.0 && m > m_cap) {
        m = m_cap;
    }
    var final_w: f32 = 0.0;
    if (selected) {
        let p_hat = target_function(selected_sample);
        if (m > 0.0 && (w_sum == w_sum) && p_hat > 0.0) {
            let w = (w_sum / m) / p_hat;
            final_w = select(0.0, w, (w == w) && (w >= 0.0));
        }
    }

    // Serialise the seeded reservoir, preserving the slot's SHARC checksum and
    // validity. A slot with no surviving candidate keeps its cell geometry so
    // it stays addressable, with the light sample / energy / W / m cleared.
    let checksum = reservoir_checksum(center);
    var out: WorldRestirReservoir;
    if (selected) {
        out.visible_point_w = vec4<f32>(selected_sample.visible_point, final_w);
        out.visible_normal_m = vec4<f32>(selected_sample.visible_normal, m);
        out.sample_point_checksum = vec4<f32>(selected_sample.sample_point, bitcast<f32>(checksum));
        out.sample_normal_valid = vec4<f32>(selected_sample.sample_normal, bitcast<f32>(1u));
        out.radiance_pad = vec4<f32>(selected_sample.radiance, 0.0);
    } else {
        out.visible_point_w = vec4<f32>(visible_point, 0.0);
        out.visible_normal_m = vec4<f32>(visible_normal, 0.0);
        out.sample_point_checksum = vec4<f32>(vec3<f32>(0.0), bitcast<f32>(checksum));
        out.sample_normal_valid = vec4<f32>(vec3<f32>(0.0), bitcast<f32>(1u));
        out.radiance_pad = vec4<f32>(vec3<f32>(0.0), 0.0);
    }
    dst_reservoirs[slot] = out;
}
