// GPU twin of `prism_render_shading::kuwahara` (the classic four-quadrant
// Kuwahara edge-preserving painterly filter: split the neighbourhood into four
// overlapping quadrants, take each quadrant's RGB mean and luminance variance,
// and output the mean of the lowest-variance quadrant).
// Self-contained (no intra-crate imports, matching ssgi.wesl / bloom.wesl /
// exposure.wesl / color_grade.wesl) so the WESL compile test can validate it
// standalone. Every helper is `kuwahara_`-prefixed to avoid symbol collisions
// with the other standalone kernels, and mirrors the CPU golden arm-for-arm:
// same Rec. 709 luma weights, same variance metric (mean squared deviation,
// no sqrt), same forward quadrant scan with a strict `<` so ties keep the first
// minimum. CPU `if` guards (empty / single / disabled) are rewritten with
// `select` so both sides stay in lockstep. WGSL `min`/`max` are componentwise,
// matching the per-channel Rust.

// Rec. 709 luma weights (linear sRGB primaries), used by the variance metric.
const KUWAHARA_LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Rec. 709 relative luminance of a linear RGB sample.
fn kuwahara_luminance(c: vec3<f32>) -> f32 {
    return dot(c, KUWAHARA_LUMA_WEIGHTS);
}

// Arithmetic mean of the first `count` samples of a fixed four-slot quadrant.
// An empty quadrant (count == 0) has no colour to average, so it returns ZERO,
// mirroring the CPU `region_mean` empty-slice guard via `select`.
fn kuwahara_region_mean(samples: array<vec3<f32>, 4>, count: u32) -> vec3<f32> {
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        sum = sum + samples[i];
    }
    let denom = max(f32(count), 1.0);
    let mean = sum / denom;
    return select(mean, vec3<f32>(0.0, 0.0, 0.0), count == 0u);
}

// Mean squared deviation of the samples' luminance about `mean`'s luminance.
// An empty or single-sample quadrant (count < 2) has no spread and returns 0,
// mirroring the CPU `region_luma_variance` guard via `select`.
fn kuwahara_region_luma_variance(samples: array<vec3<f32>, 4>, count: u32, mean: vec3<f32>) -> f32 {
    let mean_luma = kuwahara_luminance(mean);
    var acc = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let d = kuwahara_luminance(samples[i]) - mean_luma;
        acc = acc + d * d;
    }
    let denom = max(f32(count), 1.0);
    let variance = acc / denom;
    return select(variance, 0.0, count < 2u);
}

// Return the RGB mean of the quadrant with the smallest luminance variance,
// scanning the first `count` of a fixed four-slot region set. A strict `<`
// keeps the first minimum on ties, matching the CPU `select_min_variance`
// forward scan. An empty set (count == 0) returns ZERO.
fn kuwahara_select_min_variance(region_means: array<vec3<f32>, 4>, region_vars: array<f32, 4>, count: u32) -> vec3<f32> {
    var best_mean = vec3<f32>(0.0, 0.0, 0.0);
    var best_var = 0.0;
    var found = false;
    for (var i = 0u; i < count; i = i + 1u) {
        let v = region_vars[i];
        let take = (!found) || (v < best_var);
        best_mean = select(best_mean, region_means[i], take);
        best_var = select(best_var, v, take);
        found = true;
    }
    return select(vec3<f32>(0.0, 0.0, 0.0), best_mean, found);
}

// Painterly Kuwahara controls (mirrors the CPU `KuwaharaParams`). `enabled` is
// the master switch; the default of disabled with radius 2 is the identity.
struct KuwaharaParams {
    radius: u32,
    enabled: u32,
}

// Apply the Kuwahara selection to precomputed quadrant statistics. When the
// filter is disabled, or there are no quadrants, the `center` pixel passes
// through (identity); otherwise the lowest-variance quadrant mean is returned.
// Rewrites the CPU `if !enabled || regions.is_empty()` guard with `select`.
fn kuwahara_apply(region_means: array<vec3<f32>, 4>, region_vars: array<f32, 4>, count: u32, center: vec3<f32>, enabled: bool) -> vec3<f32> {
    let selected = kuwahara_select_min_variance(region_means, region_vars, count);
    let use_center = (!enabled) || (count == 0u);
    return select(selected, center, use_center);
}

// ---------------------------------------------------------------------------
// Device wiring: full-screen compute pass over the resolved HDR scene colour.
// ---------------------------------------------------------------------------
//
// display-referred, stylized NPR filter (edge-preserving painterly smoothing).
// It conceptually belongs after the display tone-map curve, but the current
// Prism chain defers tone-mapping to a Bevy node after the MainPass, so every
// pre-MainPass consumer (this one included) runs on linear HDR radiance. Placed
// as a pre-MainPass linear consumer for chain consistency; may be
// repositioned once tone-map ownership is decided.

// Immediate (push-constant) block driving the Kuwahara pass. Mirrors the Rust
// GpuKuwaharaParams byte-for-byte: the framebuffer extent leads (vec2<u32> for
// its 8-byte alignment), then the golden radius and the master enable flag
// (enabled stored as u32 for uniform layout). 2 * 4 + 2 * 4 = 16 bytes, a
// multiple of 16 with no implicit padding.
struct GpuKuwaharaParams {
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<u32>,
    // Neighbourhood radius r; each quadrant is (r + 1) x (r + 1).
    radius: u32,
    // Master enable; 0 passes the centre pixel through (identity).
    enabled: u32,
}

// The resolved linear HDR scene colour this pass filters (textureLoad-ed per
// pixel across the four overlapping quadrants).
@group(0) @binding(0) var kuwahara_scene_color: texture_2d<f32>;
// The stylized HDR output; copied back over scene_color after the dispatch.
@group(0) @binding(1) var kuwahara_out: texture_storage_2d<rgba16float, write>;

var<immediate> kuwahara_params: GpuKuwaharaParams;

// RGB mean of one overlapping quadrant of the (r + 1) x (r + 1) block anchored
// at the centre. (ox, oy) are the quadrant's per-axis directions (+/-1); the
// scan walks dx, dy in 0..=radius so the centre pixel (dx == dy == 0) is shared
// by all four quadrants, matching the CPU golden's overlapping quadrants. Edge
// texels are clamped to the frame. Inlines the golden region_mean (sum / len)
// over the full quadrant, which the fixed four-slot kuwahara_region_mean helper
// cannot size for radius > 1.
fn kuwahara_quadrant_mean(pc: vec2<i32>, ox: i32, oy: i32, radius: i32, width: i32, height: i32) -> vec3<f32> {
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    var count = 0.0;
    for (var dy = 0; dy <= radius; dy = dy + 1) {
        for (var dx = 0; dx <= radius; dx = dx + 1) {
            let sx = clamp(pc.x + ox * dx, 0, width - 1);
            let sy = clamp(pc.y + oy * dy, 0, height - 1);
            sum = sum + textureLoad(kuwahara_scene_color, vec2<i32>(sx, sy), 0).rgb;
            count = count + 1.0;
        }
    }
    // A quadrant always has at least the centre pixel, so count >= 1.
    return sum / max(count, 1.0);
}

// Mean squared deviation of the quadrant's luminance about mean's luminance.
// Inlines the golden region_luma_variance (sum of squared deviations / len,
// with the len < 2 guard returning 0) over the full (r + 1) x (r + 1) quadrant.
fn kuwahara_quadrant_variance(pc: vec2<i32>, ox: i32, oy: i32, radius: i32, width: i32, height: i32, mean: vec3<f32>) -> f32 {
    let mean_luma = kuwahara_luminance(mean);
    var acc = 0.0;
    var count = 0.0;
    for (var dy = 0; dy <= radius; dy = dy + 1) {
        for (var dx = 0; dx <= radius; dx = dx + 1) {
            let sx = clamp(pc.x + ox * dx, 0, width - 1);
            let sy = clamp(pc.y + oy * dy, 0, height - 1);
            let d = kuwahara_luminance(textureLoad(kuwahara_scene_color, vec2<i32>(sx, sy), 0).rgb) - mean_luma;
            acc = acc + d * d;
            count = count + 1.0;
        }
    }
    // Empty or single-sample quadrant has no spread (golden len < 2 guard).
    return select(acc / max(count, 1.0), 0.0, count < 2.0);
}

// One invocation per pixel. Splits the neighbourhood into the four overlapping
// quadrants (top-left, top-right, bottom-left, bottom-right), computes each
// quadrant's RGB mean and luminance variance, then writes the mean of the
// lowest-variance quadrant via the golden selection (disabled or degenerate
// pixels pass through). Alpha is preserved. Runs on the resolved HDR radiance,
// exactly like the golden.
@compute @workgroup_size(8, 8, 1)
fn kuwahara_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = i32(kuwahara_params.screen_size.x);
    let height = i32(kuwahara_params.screen_size.y);
    let pc = vec2<i32>(i32(gid.x), i32(gid.y));
    if (pc.x >= width || pc.y >= height) {
        return;
    }

    let center = textureLoad(kuwahara_scene_color, pc, 0);

    // Repack the immediate block into the golden KuwaharaParams so the on-device
    // evaluation is arm-for-arm identical to the CPU reference.
    var params: KuwaharaParams;
    params.radius = kuwahara_params.radius;
    params.enabled = kuwahara_params.enabled;
    let radius = i32(params.radius);

    // Four overlapping quadrants sharing the centre pixel; (ox, oy) picks the
    // per-axis direction. Order matches the CPU golden: TL, TR, BL, BR (a strict
    // < in the selection keeps the first minimum on ties).
    let m0 = kuwahara_quadrant_mean(pc, -1, -1, radius, width, height);
    let m1 = kuwahara_quadrant_mean(pc, 1, -1, radius, width, height);
    let m2 = kuwahara_quadrant_mean(pc, -1, 1, radius, width, height);
    let m3 = kuwahara_quadrant_mean(pc, 1, 1, radius, width, height);

    let v0 = kuwahara_quadrant_variance(pc, -1, -1, radius, width, height, m0);
    let v1 = kuwahara_quadrant_variance(pc, 1, -1, radius, width, height, m1);
    let v2 = kuwahara_quadrant_variance(pc, -1, 1, radius, width, height, m2);
    let v3 = kuwahara_quadrant_variance(pc, 1, 1, radius, width, height, m3);

    let region_means = array<vec3<f32>, 4>(m0, m1, m2, m3);
    let region_vars = array<f32, 4>(v0, v1, v2, v3);

    let result = kuwahara_apply(region_means, region_vars, 4u, center.rgb, params.enabled != 0u);

    textureStore(kuwahara_out, pc, vec4<f32>(result, center.a));
}
