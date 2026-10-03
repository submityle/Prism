// Water spectral-dispersion / chromatic-refraction compute kernel
// (entry point `water_dispersion_refract`).
//
// Water refracts short (blue) wavelengths slightly more than long (red) ones,
// so a single refracted ray splits into a faint rainbow fringe. This pass
// turns the classic `Cauchy` two-term index law into per-channel screen-space
// refraction offsets, then resamples the scene colour three times (once per
// channel) along the per-pixel refraction direction to produce the coloured
// fringe, mirroring the CPU golden `dispersion::spectral_iors`,
// `dispersion::channel_transmitted_sine` and `dispersion::dispersion_offsets`
// byte-for-byte.
//
// ABI (matches `WaterKernel::DispersionRefract.descriptor()`: 0 storage
// buffers, 1 uniform, 1 storage texture, 2 sampled textures, 8x8 tile,
// Screen domain):
//   @binding(0) params     : DispersionParams uniform
//   @binding(1) scene_in   : texture_2d<f32>, the scene colour to refract
//   @binding(2) gbuffer_in : texture_2d<f32>, per-pixel
//                            (sin_incidence, refract_dir_x, refract_dir_y,
//                             water_mask)
//   @binding(3) color_out  : texture_storage_2d<rgba32float, write>, the
//                            dispersed refracted colour
//
// Scene resampling uses the same truncate-toward-zero integer offset the CPU
// twin applies (`i32(px)` in `WESL`, `px as i32` in Rust), so the composite is
// bit-exact and not a floating sampler approximation. Only `+ - * /`,
// comparisons, `clamp` and `max` appear — no float intrinsic the workspace
// determinism policy forbids. `EPS` matches the crate's `water::EPS` (1e-6).

const EPS: f32 = 1e-6;

// Reference RGB wavelengths in micrometres (red, green, blue), matching
// `dispersion::RGB_WAVELENGTHS_UM`.
const WAVELENGTH_R: f32 = 0.700;
const WAVELENGTH_G: f32 = 0.546;
const WAVELENGTH_B: f32 = 0.440;

struct DispersionParams {
    cauchy_a: f32,
    cauchy_b: f32,
    strength: f32,
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> params: DispersionParams;
@group(0) @binding(1) var scene_in: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_in: texture_2d<f32>;
@group(0) @binding(3) var color_out: texture_storage_2d<rgba32float, write>;

// `Cauchy` refractive-index law n(lambda) = a + b / lambda^2 with a clamped
// non-positive wavelength guard.
fn cauchy_ior(a: f32, b: f32, wavelength_um: f32) -> f32 {
    let lambda = max(wavelength_um, EPS);
    return a + b / (lambda * lambda);
}

// Transmitted-ray sine for one channel from `Snell`'s law, clamped to 0..=1.
fn channel_transmitted_sine(sin_incidence: f32, ior: f32) -> f32 {
    let n = max(ior, EPS);
    return clamp(clamp(sin_incidence, 0.0, 1.0) / n, 0.0, 1.0);
}

// Scene texel sampled at the per-channel screen-space offset along `dir`,
// nearest-neighbour with truncate-toward-zero integer offset and edge clamp.
fn sample_texel(coord: vec2<i32>, offset_px: f32, dir: vec2<f32>) -> vec4<f32> {
    let px = f32(coord.x) + offset_px * dir.x;
    let py = f32(coord.y) + offset_px * dir.y;
    let w = i32(params.width);
    let h = i32(params.height);
    let sx = clamp(i32(px), 0, w - 1);
    let sy = clamp(i32(py), 0, h - 1);
    return textureLoad(scene_in, vec2<i32>(sx, sy), 0);
}

@compute @workgroup_size(8, 8, 1)
fn water_dispersion_refract(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let scene = textureLoad(scene_in, coord, 0);
    let g = textureLoad(gbuffer_in, coord, 0);
    let sin_incidence = g.x;
    let dir = vec2<f32>(g.y, g.z);
    let mask = g.w;

    // Non-water pixels pass the scene colour straight through.
    if (mask < 0.5) {
        textureStore(color_out, coord, scene);
        return;
    }

    let ior_r = cauchy_ior(params.cauchy_a, params.cauchy_b, WAVELENGTH_R);
    let ior_g = cauchy_ior(params.cauchy_a, params.cauchy_b, WAVELENGTH_G);
    let ior_b = cauchy_ior(params.cauchy_a, params.cauchy_b, WAVELENGTH_B);

    let s = clamp(sin_incidence, 0.0, 1.0);
    let gain = max(params.strength, 0.0);
    let off_r = channel_transmitted_sine(s, ior_r) * gain;
    let off_g = channel_transmitted_sine(s, ior_g) * gain;
    let off_b = channel_transmitted_sine(s, ior_b) * gain;

    let tr = sample_texel(coord, off_r, dir);
    let tg = sample_texel(coord, off_g, dir);
    let tb = sample_texel(coord, off_b, dir);

    textureStore(color_out, coord, vec4<f32>(tr.x, tg.y, tb.z, scene.w));
}
