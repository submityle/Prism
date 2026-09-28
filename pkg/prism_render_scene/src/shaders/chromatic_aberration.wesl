// GPU twin of `prism_render_shading::chromatic_aberration` (industry-standard
// lateral chromatic aberration / lens colour fringing on the resolved image:
// each colour channel samples from a radially offset uv, the split growing with
// distance from the optical centre — the Unity PPv2 / Unreal Engine Scene
// Fringe convention).
// Self-contained (no intra-crate imports, matching ssgi.wesl / bloom.wesl /
// color_grade.wesl) so the WESL compile test can validate it standalone. Every
// function mirrors the CPU golden arm-for-arm: same operation order, same
// constants. The radial length uses the `sqrt` builtin (matching the CPU
// `f32::sqrt`), lerps are written out `a + (b - a) * t` and all helpers carry a
// `chromatic_aberration_` prefix to avoid symbol collisions.

// Distance below which the radial direction is undefined and returned as zero,
// keeping the aberration centre a fixed point.
const CHROMATIC_ABERRATION_EPSILON: f32 = 1.0e-8;

// Radial direction (unit) and distance of a uv from the aberration centre.
struct ChromaticAberrationRadial {
    dir: vec2<f32>,
    dist: f32,
}

// Per-channel sample coordinates for one pixel (red / green / blue taps).
struct ChromaticAberrationSamples {
    red: vec2<f32>,
    green: vec2<f32>,
    blue: vec2<f32>,
}

// Artist controls (mirrors the CPU `ChromaticAberrationParams`; the neutral
// field values give an identity effect).
struct ChromaticAberrationParams {
    center: vec2<f32>,
    intensity: f32,
    samples: u32,
}

// Radial direction and distance of `uv` from the aberration `center`. At (or
// within `CHROMATIC_ABERRATION_EPSILON` of) the centre the direction is the
// zero vector, so the centre stays a fixed point. Mirrors the CPU golden.
fn chromatic_aberration_radial(uv: vec2<f32>, center: vec2<f32>) -> ChromaticAberrationRadial {
    let delta = uv - center;
    let dist = sqrt(delta.x * delta.x + delta.y * delta.y);
    var out: ChromaticAberrationRadial;
    out.dist = dist;
    if dist > CHROMATIC_ABERRATION_EPSILON {
        out.dir = delta / dist;
    } else {
        out.dir = vec2<f32>(0.0, 0.0);
    }
    return out;
}

// Signed per-channel scalar offsets along the radial direction, as
// (red, green, blue): red rides outward, green stays put, blue rides inward.
// Neutral `intensity = 0` returns all zeros.
fn chromatic_aberration_sample_offsets(dist: f32, intensity: f32) -> vec3<f32> {
    let split = intensity * dist;
    return vec3<f32>(split, 0.0, -split);
}

// Sample coordinate for one colour channel: advance `uv` along `dir` by the
// channel's scalar offset. `channel_index` in 0..3 (red, green, blue); green
// (index 1) returns `uv` unchanged. Mirrors the CPU golden arm-for-arm.
fn chromatic_aberration_channel_uv(uv: vec2<f32>, dir: vec2<f32>, dist: f32, intensity: f32, channel_index: u32) -> vec2<f32> {
    let offsets = chromatic_aberration_sample_offsets(dist, intensity);
    var o = offsets.y;
    if channel_index == 0u {
        o = offsets.x;
    }
    if channel_index == 2u {
        o = offsets.z;
    }
    return uv + dir * o;
}

// Recombine three per-channel samples, keeping each channel from its own tap:
// `(red.r, green.g, blue.b)`. Equal samples reproduce the input.
fn chromatic_aberration_combine(red_sample: vec3<f32>, green_sample: vec3<f32>, blue_sample: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(red_sample.x, green_sample.y, blue_sample.z);
}

// Smooth polynomial spectral weight ramp for the optional multi-tap form: maps
// `t` in [0, 1] to (red, green, blue) with blue peaking at 0, red at 1, green
// in the middle. Pure quadratics, mirroring the CPU golden.
fn chromatic_aberration_spectral_lut(t: f32) -> vec3<f32> {
    let one_minus = 1.0 - t;
    return vec3<f32>(t * t, 4.0 * t * one_minus, one_minus * one_minus);
}

// Signed scalar offset for a spectral tap at `t` in [0, 1], ramping linearly
// from inward (blue, t=0) to outward (red, t=1). Mid tap t=0.5 is zero offset.
fn chromatic_aberration_spectral_offset(dist: f32, intensity: f32, t: f32) -> f32 {
    return intensity * dist * (2.0 * t - 1.0);
}

// Per-channel sample coordinates for a pixel at `uv`. With `intensity = 0` all
// three equal `uv` (identity). Mirrors the CPU `apply_chromatic_aberration`.
fn chromatic_aberration_apply(uv: vec2<f32>, center: vec2<f32>, intensity: f32) -> ChromaticAberrationSamples {
    let radial = chromatic_aberration_radial(uv, center);
    var out: ChromaticAberrationSamples;
    out.red = chromatic_aberration_channel_uv(uv, radial.dir, radial.dist, intensity, 0u);
    out.green = chromatic_aberration_channel_uv(uv, radial.dir, radial.dist, intensity, 1u);
    out.blue = chromatic_aberration_channel_uv(uv, radial.dir, radial.dist, intensity, 2u);
    return out;
}

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real aberration pass wires the resolved target and
// the params uniform later. References every function so none drifts unused.
@compute @workgroup_size(64, 1, 1)
fn chromatic_aberration_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seed = f32(gid.x) * 0.01;
    let uv = vec2<f32>(0.8 + seed, 0.2);

    var params: ChromaticAberrationParams;
    params.center = vec2<f32>(0.5, 0.5);
    params.intensity = 0.2;
    params.samples = 3u;

    let radial = chromatic_aberration_radial(uv, params.center);
    let offsets = chromatic_aberration_sample_offsets(radial.dist, params.intensity);
    let one_channel = chromatic_aberration_channel_uv(uv, radial.dir, radial.dist, params.intensity, 0u);
    let lut = chromatic_aberration_spectral_lut(0.5);
    let spec = chromatic_aberration_spectral_offset(radial.dist, params.intensity, 0.5);
    let samples = chromatic_aberration_apply(uv, params.center, params.intensity);
    let combined = chromatic_aberration_combine(
        vec3<f32>(samples.red.x, 0.0, 0.0),
        vec3<f32>(0.0, samples.green.y, 0.0),
        vec3<f32>(0.0, 0.0, samples.blue.y),
    );

    _ = radial.dir.x + offsets.x + one_channel.x + lut.x + spec + combined.x + f32(params.samples);
}
