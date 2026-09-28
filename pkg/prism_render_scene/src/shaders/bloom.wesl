// GPU twin of `prism_render_shading::bloom` (energy-preserving AAA "dual
// filter" bloom: Karis soft-threshold prefilter, fireflies-suppressing Karis
// average, COD 13-tap downsample, 3x3 tent upsample and the artist combine).
// Self-contained (no intra-crate imports, matching ssgi.wesl / exposure.wesl)
// so the WESL compile test can validate it standalone. Every function mirrors
// the CPU golden arm-for-arm; WGSL `max(vec3, vec3)` is componentwise, matching
// the per-channel Rust maths, and lerps are written `a + (b - a) * t` rather
// than `mix` to stay bit-identical with the golden.

const BLOOM_LUMINANCE_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Rec. 709 relative luminance of a linear RGB radiance sample.
fn bloom_luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, BLOOM_LUMINANCE_WEIGHTS);
}

// Karis soft-threshold prefilter: the fraction of `color` that bleeds into the
// bloom pyramid. Brightness is the max component; a `knee` of 0 is a hard
// threshold; negative knees clamp to 0 so the clamp bounds never invert.
fn bloom_prefilter(color: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    let k = max(knee, 0.0);
    let br = max(max(color.x, color.y), color.z);
    var soft = clamp(br - (threshold - k), 0.0, 2.0 * k);
    soft = soft * soft / (4.0 * k + 1.0e-6);
    let contribution = max(soft, br - threshold) / max(br, 1.0e-6);
    return color * contribution;
}

// Karis fireflies-suppression weight 1 / (1 + luminance) for the first
// downsample. Strictly decreasing in luminance.
fn bloom_karis_average_weight(luminance: f32) -> f32 {
    return 1.0 / (1.0 + max(luminance, 0.0));
}

// Average of a 2x2 box of taps (helper for the 13-tap downsample).
fn bloom_box_avg(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return (a + b + c + d) * 0.25;
}

// COD 13-tap downsample. 5x5 grid layout (see the CPU golden):
//    0 . 1 . 2
//    . 3 . 4 .
//    5 . 6 . 7   (6 = centre)
//    . 8 . 9 .
//   10 .11 .12
// Centre 2x2 box weight 0.5 + four corner boxes weight 0.125 each; sum = 1.
fn bloom_downsample_13tap(samples: array<vec3<f32>, 13>) -> vec3<f32> {
    let center = bloom_box_avg(samples[3], samples[4], samples[8], samples[9]);
    let tl = bloom_box_avg(samples[0], samples[1], samples[5], samples[6]);
    let tr = bloom_box_avg(samples[1], samples[2], samples[6], samples[7]);
    let bl = bloom_box_avg(samples[5], samples[6], samples[10], samples[11]);
    let br = bloom_box_avg(samples[6], samples[7], samples[11], samples[12]);
    return 0.5 * center + 0.125 * (tl + tr + bl + br);
}

// 9-tap 3x3 tent upsample (1/16 kernel: corners 1, edges 2, centre 4), taps
// row-major with 4 = centre. `radius` (clamped to [0, 1]) blends the centre tap
// toward the full tent; both endpoints are weight-1 filters so energy is kept.
fn bloom_upsample_tent(samples: array<vec3<f32>, 9>, radius: f32) -> vec3<f32> {
    let r = clamp(radius, 0.0, 1.0);
    let center = samples[4];
    let corners = samples[0] + samples[2] + samples[6] + samples[8];
    let edges = samples[1] + samples[3] + samples[5] + samples[7];
    let tent = (corners + 2.0 * edges + 4.0 * samples[4]) / 16.0;
    return center + (tent - center) * r;
}

// Blend `bloom` over `scene` by `intensity`: scene + (bloom - scene) * i.
fn bloom_combine(scene: vec3<f32>, bloom: vec3<f32>, intensity: f32) -> vec3<f32> {
    return scene + (bloom - scene) * intensity;
}

// Per-mip contribution weight for a `levels`-deep pyramid, normalised so the
// levels sum to 1: weight[i] = (levels - i) / (levels * (levels + 1) / 2).
fn bloom_mip_weight(index: u32, levels: u32) -> f32 {
    if (levels == 0u || index >= levels) {
        return 0.0;
    }
    let total = f32(levels * (levels + 1u) / 2u);
    return f32(levels - index) / total;
}

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real bloom passes wire mip storage bindings later.
@compute @workgroup_size(64, 1, 1)
fn bloom_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    var down: array<vec3<f32>, 13>;
    for (var i = 0u; i < 13u; i = i + 1u) {
        down[i] = vec3<f32>(f32(gid.x) + f32(i));
    }
    var up: array<vec3<f32>, 9>;
    for (var j = 0u; j < 9u; j = j + 1u) {
        up[j] = vec3<f32>(f32(j));
    }

    let lum = bloom_luminance(vec3<f32>(0.2, 0.5, 0.3));
    let pre = bloom_prefilter(vec3<f32>(1.0, 0.8, 0.6), 1.0, 0.5);
    let w = bloom_karis_average_weight(lum);
    let d = bloom_downsample_13tap(down);
    let u = bloom_upsample_tent(up, 1.0);
    let c = bloom_combine(vec3<f32>(0.3, 0.3, 0.3), d, 0.5);
    let mw = bloom_mip_weight(1u, 5u);

    _ = lum + w + mw + pre.x + d.x + u.x + c.x;
}
