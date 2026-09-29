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
    screen_size: vec2<u32>,
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

// Group-0 resources for the real aberration pass.
// binding 0 = the resolved HDR scene colour (filterable, sampled with the
//             linear-clamp sampler below);
// binding 1 = the linear-clamp sampler for the three per-channel fetches;
// binding 2 = the write-only aberrated output (SCENE_COLOR_FORMAT / rgba16float),
//             copied back over scene_color by the dispatch.
@group(0) @binding(0) var chromatic_aberration_scene_color: texture_2d<f32>;
@group(0) @binding(1) var chromatic_aberration_linear_sampler: sampler;
@group(0) @binding(2) var chromatic_aberration_out: texture_storage_2d<rgba16float, write>;

// Immediate (push-constant) block; mirrors the Rust GpuChromaticAberrationParams
// byte-for-byte (center, intensity, samples, screen_size).
var<immediate> chromatic_aberration_params: ChromaticAberrationParams;

// Full-screen lateral chromatic aberration: one invocation per pixel. Rebuilds
// the pixel's uv, asks the CPU-golden twin for the three radially split
// per-channel sample coordinates, fetches the resolved scene colour at each with
// a trilinear (bilinear + explicit LOD 0) clamp sampler, recombines each channel
// from its own tap and writes the fringed pixel. The green (centre) tap is the
// unshifted uv, so its alpha carries the source coverage unchanged. At the
// aberration centre, or with intensity 0, the three taps collapse to the same
// uv and the pass is the identity.
@compute @workgroup_size(8, 8, 1)
fn chromatic_aberration_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= chromatic_aberration_params.screen_size.x
        || gid.y >= chromatic_aberration_params.screen_size.y {
        return;
    }

    let screen = vec2<f32>(
        f32(chromatic_aberration_params.screen_size.x),
        f32(chromatic_aberration_params.screen_size.y),
    );
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5)) / screen;

    let samples = chromatic_aberration_apply(
        uv,
        chromatic_aberration_params.center,
        chromatic_aberration_params.intensity,
    );

    let red = textureSampleLevel(
        chromatic_aberration_scene_color,
        chromatic_aberration_linear_sampler,
        samples.red,
        0.0,
    );
    let green = textureSampleLevel(
        chromatic_aberration_scene_color,
        chromatic_aberration_linear_sampler,
        samples.green,
        0.0,
    );
    let blue = textureSampleLevel(
        chromatic_aberration_scene_color,
        chromatic_aberration_linear_sampler,
        samples.blue,
        0.0,
    );

    let combined = chromatic_aberration_combine(red.rgb, green.rgb, blue.rgb);
    textureStore(
        chromatic_aberration_out,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(combined, green.a),
    );
}
