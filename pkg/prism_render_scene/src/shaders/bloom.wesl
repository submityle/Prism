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

// Partial-Karis first downsample (COD/Jimenez): the same five overlapping 2x2
// boxes as `bloom_downsample_13tap`, but each box is weighted by the Karis
// fireflies weight `1 / (1 + luma(box))` before the structural 0.5 / 0.125
// combine, then renormalised by the weighted structural sum. Applied only on
// the first (brightest) downsample so a lone fireflies-bright sub-pixel cannot
// pump the whole pyramid every frame; the deeper downsamples use the plain
// energy-preserving `bloom_downsample_13tap`.
fn bloom_downsample_13tap_karis(s: array<vec3<f32>, 13>) -> vec3<f32> {
    let c = (s[3] + s[4] + s[8] + s[9]) * 0.25;
    let tl = (s[0] + s[1] + s[5] + s[6]) * 0.25;
    let tr = (s[1] + s[2] + s[6] + s[7]) * 0.25;
    let bl = (s[5] + s[6] + s[10] + s[11]) * 0.25;
    let br = (s[6] + s[7] + s[11] + s[12]) * 0.25;
    let kc = 0.5 * bloom_karis_average_weight(bloom_luminance(c));
    let ktl = 0.125 * bloom_karis_average_weight(bloom_luminance(tl));
    let ktr = 0.125 * bloom_karis_average_weight(bloom_luminance(tr));
    let kbl = 0.125 * bloom_karis_average_weight(bloom_luminance(bl));
    let kbr = 0.125 * bloom_karis_average_weight(bloom_luminance(br));
    let sum = kc + ktl + ktr + kbl + kbr;
    return (c * kc + tl * ktl + tr * ktr + bl * kbl + br * kbr) / max(sum, 1.0e-6);
}

// Immediate (push-constant) block shared by every bloom pass. `inv_src` is one
// over the *source* texture extent (for the sampling offsets); the artist
// threshold/knee/intensity/radius drive the prefilter, tent spread and combine;
// `dst_width`/`dst_height` are the *destination* extent for the coverage guard.
struct BloomParams {
    inv_src_width: f32,
    inv_src_height: f32,
    threshold: f32,
    knee: f32,
    intensity: f32,
    radius: f32,
    dst_width: u32,
    dst_height: u32,
}

var<immediate> params: BloomParams;

// ---- Pass 0: full-res copy of scene_color -> bloom base -------------------
// `scene_color` is a write-only `rgba16float` storage image, so the combine
// cannot read and write it in one dispatch. This pass lifts a byte-for-byte
// copy into a separate `base` target the combine reads from, keeping the final
// pass free of any read/write aliasing hazard (the SSR-composite precedent).
@group(0) @binding(0) var copy_src: texture_2d<f32>;
@group(0) @binding(1) var copy_dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn bloom_copy_scene(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.dst_width || id.y >= params.dst_height) {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    textureStore(copy_dst, coord, textureLoad(copy_src, coord, 0));
}

// ---- Pass 1: Karis prefilter + partial-Karis first downsample -------------
// Reads scene_color through a linear-clamp sampler, prefilters each of the 13
// COD taps (isolating the bright tail with a soft knee) and folds them with the
// fireflies-suppressing partial-Karis average into the half-res mip 0.
@group(0) @binding(0) var pf_sampler: sampler;
@group(0) @binding(1) var pf_src: texture_2d<f32>;
@group(0) @binding(2) var pf_dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn bloom_prefilter_downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.dst_width || id.y >= params.dst_height) {
        return;
    }
    // Half-res destination: a destination texel spans two source texels, so its
    // centre in source UV is (dst + 0.5) * 2 * inv_src.
    let inv = vec2<f32>(params.inv_src_width, params.inv_src_height);
    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + vec2<f32>(0.5)) * 2.0 * inv;
    var s: array<vec3<f32>, 13>;
    let o = array<vec2<f32>, 13>(
        vec2<f32>(-2.0, -2.0), vec2<f32>(0.0, -2.0), vec2<f32>(2.0, -2.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-2.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-2.0, 2.0), vec2<f32>(0.0, 2.0), vec2<f32>(2.0, 2.0),
    );
    for (var i = 0u; i < 13u; i = i + 1u) {
        let c = textureSampleLevel(pf_src, pf_sampler, uv + o[i] * inv, 0.0).rgb;
        s[i] = bloom_prefilter(c, params.threshold, params.knee);
    }
    let down = bloom_downsample_13tap_karis(s);
    textureStore(pf_dst, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(down, 1.0));
}

// ---- Pass 2: energy-preserving COD 13-tap downsample ----------------------
@group(0) @binding(0) var ds_sampler: sampler;
@group(0) @binding(1) var ds_src: texture_2d<f32>;
@group(0) @binding(2) var ds_dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn bloom_downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.dst_width || id.y >= params.dst_height) {
        return;
    }
    let inv = vec2<f32>(params.inv_src_width, params.inv_src_height);
    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + vec2<f32>(0.5)) * 2.0 * inv;
    var s: array<vec3<f32>, 13>;
    let o = array<vec2<f32>, 13>(
        vec2<f32>(-2.0, -2.0), vec2<f32>(0.0, -2.0), vec2<f32>(2.0, -2.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-2.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-2.0, 2.0), vec2<f32>(0.0, 2.0), vec2<f32>(2.0, 2.0),
    );
    for (var i = 0u; i < 13u; i = i + 1u) {
        s[i] = textureSampleLevel(ds_src, ds_sampler, uv + o[i] * inv, 0.0).rgb;
    }
    let down = bloom_downsample_13tap(s);
    textureStore(ds_dst, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(down, 1.0));
}

// ---- Pass 3: 3x3 tent upsample of the coarser mip, added to the finer -----
// Progressive dual-filter upsample: tent-filter the coarse mip up to this
// finer resolution and add the finer mip's own detail, accumulating the halo
// from coarse to fine. `radius` widens/narrows the tent without changing its
// total weight, so energy is preserved for any radius.
@group(0) @binding(0) var up_sampler: sampler;
@group(0) @binding(1) var up_coarse: texture_2d<f32>;
@group(0) @binding(2) var up_fine: texture_2d<f32>;
@group(0) @binding(3) var up_dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn bloom_upsample(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.dst_width || id.y >= params.dst_height) {
        return;
    }
    // Coarse source is half this destination's extent; its texel size is
    // `inv_src`. The destination texel centre in coarse UV is
    // (dst + 0.5) * 0.5 * inv_src.
    let inv = vec2<f32>(params.inv_src_width, params.inv_src_height);
    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + vec2<f32>(0.5)) * 0.5 * inv;
    var s: array<vec3<f32>, 9>;
    let o = array<vec2<f32>, 9>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0),
    );
    for (var i = 0u; i < 9u; i = i + 1u) {
        s[i] = textureSampleLevel(up_coarse, up_sampler, uv + o[i] * inv, 0.0).rgb;
    }
    let tent = bloom_upsample_tent(s, params.radius);
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    let fine = textureLoad(up_fine, coord, 0).rgb;
    textureStore(up_dst, coord, vec4<f32>(fine + tent, 1.0));
}

// ---- Pass 4: combine the accumulated bloom over the scene -----------------
// Reads the untouched scene base (the pass-0 copy) and the accumulated bloom
// (mip 0 of the upsample chain, sampled up to full res), blends them by the
// artist intensity and writes the result back into `scene_color`.
@group(0) @binding(0) var cb_sampler: sampler;
@group(0) @binding(1) var cb_base: texture_2d<f32>;
@group(0) @binding(2) var cb_bloom: texture_2d<f32>;
@group(0) @binding(3) var cb_dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn bloom_combine_pass(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.dst_width || id.y >= params.dst_height) {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    let base = textureLoad(cb_base, coord, 0);
    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + vec2<f32>(0.5)) *
        vec2<f32>(params.inv_src_width, params.inv_src_height);
    let bloom = textureSampleLevel(cb_bloom, cb_sampler, uv, 0.0).rgb;
    let combined = bloom_combine(base.rgb, bloom, params.intensity);
    textureStore(cb_dst, coord, vec4<f32>(max(combined, vec3<f32>(0.0)), base.a));
}
