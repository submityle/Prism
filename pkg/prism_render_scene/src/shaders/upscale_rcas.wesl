// Temporal-upscale RCAS (Robust Contrast-Adaptive Sharpening) pass.
//
// GPU twin of the CPU golden `prism_render_shading::upscale::sharpen` (the
// `rcas` kernel and its `rcas_luma` / `rcas_noise` helpers). After the
// reconstruction + accumulation pass (`upscale_reconstruct.wesl`) resolves a
// display-resolution image the result is slightly soft — the Lanczos-2 kernel
// and the temporal history blend both low-pass the signal — so temporal
// upscalers finish with a sharpening pass. RCAS is the FSR variant tuned for
// *upscaled* input: unlike plain CAS (`cas.wesl`) it uses only the five-tap
// cross (no corners), derives a single shared sharpening *lobe* from the
// per-channel contrast limits, and clamps that lobe to `RCAS_LIMIT` so it lifts
// detail without the ringing haloes a naive unsharp mask leaves on a
// reconstructed edge.
//
// The kernel, per FSR's `FsrRcasF`:
//
//       b
//     d e f      (centre e)
//       h
//
//   1. per-channel min/max over the ring {b, d, f, h};
//   2. limiters hit_min = mn / (4*mx) and hit_max = (1 - mx) / (4*mn - 4),
//      lobe_rgb = max(-hit_min, hit_max);
//   3. one lobe clamp(max_channel(lobe_rgb), -RCAS_LIMIT, 0) * sharp, optionally
//      attenuated by the noise estimate;
//   4. resolve ((b + d + f + h)*lobe + e) / (4*lobe + 1).
//
// `sharp` is the settings' sharpness taken directly as a [0, 1] lobe scale, so
// `sharpness == 0` is an exact identity (no sharpening) and `1` is full RCAS. A
// flat neighbourhood is a fixed point at every sharpness. Every denominator is
// guarded so a black or white flat ring is an exact identity rather than a NaN.
// Everything is plain arithmetic, mirrored bit-for-bit by the golden: WGSL
// `min`, `max`, `clamp` on vec3 are componentwise (matching the per-channel
// Rust) and the resolve is written out longhand rather than with `mix`.
//
// Self-contained (no imports); group 0 is
//   0 = input_color (the reconstructed display-res colour, textureLoad)
//   1 = color_out   (rgba16float storage; the sharpened display colour)
// with an `UpscaleRcasParams` immediate block.

// The FSR RCAS lobe clamp `0.25 - 1/16`. Capping the sharpening lobe here is
// what keeps RCAS "robust": it bounds the overshoot so a high-contrast edge
// cannot ring.
const RCAS_LIMIT: f32 = 0.25 - 1.0 / 16.0;
const EPS: f32 = 1.0e-6;

struct UpscaleRcasParams {
    // Display (output) extent in texels.
    width: u32,
    height: u32,
    // Lobe scale in [0, 1] taken from the settings' sharpness. 0 disables
    // sharpening (exact identity); 1 is full-strength RCAS.
    sharpness: f32,
    // 1 to attenuate the lobe by the noise estimate (pull sharpening back in
    // noisy regions so RCAS does not amplify grain); 0 to skip it.
    denoise: u32,
}

@group(0) @binding(0) var input_color: texture_2d<f32>;
@group(0) @binding(1) var color_out: texture_storage_2d<rgba16float, write>;

var<immediate> params: UpscaleRcasParams;

// Load a display-grid texel, clamping to the edge so the cross always has five
// valid taps at the frame border.
fn load_color(x: i32, y: i32) -> vec3<f32> {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    return textureLoad(input_color, vec2<i32>(cx, cy), 0).rgb;
}

// FSR's cheap luma proxy `g + 0.5*(r + b)` (luma times two), used only for the
// noise estimate so it needs no exact weighting.
fn rcas_luma(rgb: vec3<f32>) -> f32 {
    return rgb.y + 0.5 * (rgb.x + rgb.z);
}

// The RCAS noise attenuation factor in [0.5, 1] for the cross. It measures how
// much the centre luma departs from the ring average relative to the ring's
// luma spread: a lone bright/dark sample (noise) yields a large normalised
// departure and a factor near 0.5 (halve the sharpening), while a coherent edge
// yields a factor near 1. A flat (zero-spread) ring returns 1.
fn rcas_noise(b: vec3<f32>, d: vec3<f32>, e: vec3<f32>, f: vec3<f32>, h: vec3<f32>) -> f32 {
    let lb = rcas_luma(b);
    let ld = rcas_luma(d);
    let le = rcas_luma(e);
    let lf = rcas_luma(f);
    let lh = rcas_luma(h);
    let spread = max(max(max(max(lb, ld), le), lf), lh)
        - min(min(min(min(lb, ld), le), lf), lh);
    if (spread < EPS) {
        return 1.0;
    }
    let departure = 0.25 * (lb + ld + lf + lh) - le;
    let nz = clamp(abs(departure) / spread, 0.0, 1.0);
    return -0.5 * nz + 1.0;
}

// The full RCAS kernel over the five-tap cross [b, d, e, f, h] (up, left,
// centre, right, down) at the given sharpness. Returns the centre unchanged
// when sharpness <= 0 or the neighbourhood is flat. All limiter denominators
// are guarded (preserving their sign: `4*mn - 4` is non-positive for a
// display-referred [0, 1] signal) so a black or white flat ring is an exact
// identity rather than a NaN.
fn rcas(b: vec3<f32>, d: vec3<f32>, e: vec3<f32>, f: vec3<f32>, h: vec3<f32>, sharpness: f32, denoise: bool) -> vec3<f32> {
    let sharp = clamp(sharpness, 0.0, 1.0);
    if (sharp <= 0.0) {
        return e;
    }

    // Per-channel min/max over the ring {b, d, f, h}.
    let mn = min(min(b, d), min(f, h));
    let mx = max(max(b, d), max(f, h));

    // Limiters, denominators guarded away from zero preserving their sign.
    let four_mx = max(4.0 * mx, vec3<f32>(EPS));
    let four_mn_minus_four = min(4.0 * mn - vec3<f32>(4.0), vec3<f32>(-EPS));
    let hit_min = mn / four_mx;
    let hit_max = (vec3<f32>(1.0) - mx) / four_mn_minus_four;
    let lobe_rgb = max(-hit_min, hit_max);

    // One shared lobe: the sharpest (most negative overshoot) channel, clamped
    // into [-RCAS_LIMIT, 0], scaled by the sharpness knob.
    let lobe_channel = max(max(lobe_rgb.x, lobe_rgb.y), lobe_rgb.z);
    var lobe = clamp(lobe_channel, -RCAS_LIMIT, 0.0) * sharp;
    if (denoise) {
        lobe = lobe * rcas_noise(b, d, e, f, h);
    }

    // Resolve. `lobe` is non-positive, so `4*lobe + 1 >= 1 - 4*RCAS_LIMIT = 0.25`
    // stays safely positive.
    let rcp = 1.0 / (4.0 * lobe + 1.0);
    return ((b + d + f + h) * lobe + e) * rcp;
}

@compute @workgroup_size(8, 8, 1)
fn upscale_rcas(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    // Five-tap cross [b, d, e, f, h] = [up, left, centre, right, down].
    let b = load_color(coord.x, coord.y - 1);
    let d = load_color(coord.x - 1, coord.y);
    let e = load_color(coord.x, coord.y);
    let f = load_color(coord.x + 1, coord.y);
    let h = load_color(coord.x, coord.y + 1);

    let sharpened = rcas(b, d, e, f, h, params.sharpness, params.denoise != 0u);
    textureStore(color_out, coord, vec4<f32>(sharpened, 1.0));
}
