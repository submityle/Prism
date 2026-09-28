// Full-screen temporal anti-aliasing resolve.
//
// GPU twin of `prism_render_shading::taa::resolve`. Runs on the composited HDR
// `scene_color` (opaque + SSR, before the forward transparency composite), the
// GPU counterpart of the CPU golden `resolve_taa`. Each frame the jittered
// current colour is blended with the motion-reprojected history so the sub-pixel
// camera jitter integrates into a supersampled image, fighting both ghosting and
// firefly flicker with the AAA trio from Karis' *High Quality Temporal
// Supersampling*:
//
//   * motion reprojection  — history is fetched at `cur_uv - motion` using the
//     resolve pass's motion-vector G-buffer, so it follows camera *and*
//     per-object motion;
//   * `YCoCg` neighbourhood clip — the reprojected history is clipped to the
//     colour-variance box (`mean ± γσ`) of the current 3x3 neighbourhood,
//     computed in `YCoCg` where luma and chroma separate so the box hugs the
//     real signal;
//   * luminance feedback weighting — the current and clipped-history
//     contributions are weighted by `1 / (1 + luma)`, so a single blindingly
//     bright sub-pixel sample cannot dominate the blend and flicker.
//
// A dropped reprojection (off-screen after the motion offset) or an invalid
// history (first frame / resize / camera cut) falls back to the current frame,
// so a cut degrades to the un-accumulated — still jittered but un-smeared —
// frame rather than ghosting.
//
// Self-contained (no imports); group 0 is
//   0 = scene_color     (this frame's composited HDR radiance, textureLoad)
//   1 = history         (previous accumulated colour, sampled at the reprojected UV)
//   2 = history_sampler (bilinear clamp filtering)
//   3 = taa_out         (rgba16float storage; the resolved colour the composite reads)
//   4 = motion_vectors  (rg16float screen-space motion, textureLoad)
// with a `TaaParams` immediate block.

struct TaaParams {
    width: u32,
    height: u32,
    // Fraction of the (clipped) history kept when the sample is trusted.
    history_blend: f32,
    // Half-width of the neighbourhood variance box in standard deviations.
    variance_gamma: f32,
    // 1 when a previous frame is available (no resize / camera cut), else 0.
    valid_history: u32,
}

@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var history: texture_2d<f32>;
@group(0) @binding(2) var history_sampler: sampler;
@group(0) @binding(3) var taa_out: texture_storage_2d<rgba16float, write>;
// Per-pixel screen-space motion (`cur_uv - prev_uv`) from the resolve pass; the
// history lookup adds it back to reach last frame's UV. Zero on pixels the
// resolve did not cover.
@group(0) @binding(4) var motion_vectors: texture_2d<f32>;

var<immediate> params: TaaParams;

// Rec.601-style RGB -> YCoCg (luma + two chroma); exactly invertible by
// `ycocg_to_rgb`. The variance clip box is built in this space so a bright
// specular edge does not drag chroma around.
fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * rgb.x + 0.5 * rgb.y + 0.25 * rgb.z,
        0.5 * rgb.x - 0.5 * rgb.z,
        -0.25 * rgb.x + 0.5 * rgb.y - 0.25 * rgb.z,
    );
}

// Exact inverse of `rgb_to_ycocg`.
fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    let y = ycocg.x;
    let co = ycocg.y;
    let cg = ycocg.z;
    return vec3<f32>(y + co - cg, y + cg, y - co - cg);
}

// The firefly-suppressing feedback weight `1 / (1 + max(luma, 0))`: a very
// bright sample earns a small weight so it cannot dominate the temporal blend.
fn tonemap_weight(luma: f32) -> f32 {
    return 1.0 / (1.0 + max(luma, 0.0));
}

// Variance clip box `mean ± gamma*sigma` from the first two colour moments of
// the 3x3 neighbourhood (`mean` and `mean_sq`, the mean of per-channel squares).
fn variance_box_half(mean: vec3<f32>, mean_sq: vec3<f32>, gamma: f32) -> vec3<f32> {
    let variance = max(mean_sq - mean * mean, vec3<f32>(0.0));
    return sqrt(variance) * max(gamma, 0.0);
}

// AABB-clip (Karis): pull `hist` back onto the surface of the box built from
// `center`/`half`, matching the golden `clip_history_to_aabb`.
fn clip_history_to_aabb(hist: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec3<f32> {
    let center = 0.5 * (box_max + box_min);
    let extent = max(0.5 * (box_max - box_min), vec3<f32>(1.0e-5));
    let dir = hist - center;
    let units = dir / extent;
    let max_unit = max(max(abs(units.x), abs(units.y)), abs(units.z));
    if (max_unit > 1.0) {
        return center + dir / max_unit;
    }
    return hist;
}

// Clamp a neighbour tap coordinate to the framebuffer so the 3x3 gather always
// contributes nine samples (matching the golden's fixed nine-tap neighbourhood
// and its zero-variance fixed point on a flat region).
fn clamp_coord(x: i32, y: i32) -> vec2<i32> {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    return vec2<i32>(cx, cy);
}

@compute @workgroup_size(8, 8, 1)
fn resolve_taa(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    // This frame's jittered composited colour is the anchor.
    let current_texel = textureLoad(scene_color, coord, 0);
    let current_rgb = current_texel.rgb;

    // Gather the first two YCoCg colour moments over the 3x3 neighbourhood for
    // the variance clip box, clamping taps to the edge so the count is always 9.
    var sum = vec3<f32>(0.0);
    var sum_sq = vec3<f32>(0.0);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let tap = textureLoad(scene_color, clamp_coord(coord.x + dx, coord.y + dy), 0).rgb;
            let y = rgb_to_ycocg(tap);
            sum = sum + y;
            sum_sq = sum_sq + y * y;
        }
    }
    let mean = sum / 9.0;
    let mean_sq = sum_sq / 9.0;

    var out_rgb = current_rgb;

    if (params.valid_history != 0u) {
        let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
            / vec2<f32>(f32(params.width), f32(params.height));
        // Motion vector already folds in camera + object motion; add it back to
        // find where this surface sat last frame.
        let motion = textureLoad(motion_vectors, coord, 0).xy;
        let prev_uv = uv - motion;
        if (prev_uv.x >= 0.0 && prev_uv.x <= 1.0
            && prev_uv.y >= 0.0 && prev_uv.y <= 1.0) {
            let hist_rgb = textureSampleLevel(history, history_sampler, prev_uv, 0.0).rgb;

            let current_ycocg = rgb_to_ycocg(current_rgb);
            let history_ycocg = rgb_to_ycocg(hist_rgb);

            let half = variance_box_half(mean, mean_sq, params.variance_gamma);
            let clipped = clip_history_to_aabb(history_ycocg, mean - half, mean + half);

            // Luminance (YCoCg `.x`) feedback weighting: down-weight the brighter
            // contributor so a firefly cannot flicker the accumulation.
            let weight_current = (1.0 - params.history_blend) * tonemap_weight(current_ycocg.x);
            let weight_history = params.history_blend * tonemap_weight(clipped.x);
            let total = weight_current + weight_history;
            if (total > 0.0) {
                let blended = (current_ycocg * weight_current + clipped * weight_history) / total;
                out_rgb = ycocg_to_rgb(blended);
            } else {
                out_rgb = current_rgb;
            }
        }
    }

    textureStore(taa_out, coord, vec4<f32>(out_rgb, current_texel.a));
}
