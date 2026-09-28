// Cross-frame temporal accumulation for screen-space reflections.
//
// GPU twin of `prism_render_shading::screen_space::temporal`. Prism has no
// per-pixel motion-vector G-buffer, so history is reprojected purely from
// camera motion: reconstruct each pixel's world position from its reverse-Z
// device depth and the inverse current view-projection, reproject it through
// the previous frame's view-projection to find last frame's UV, sample the
// accumulated history there, clip it to the current 3x3 neighbourhood colour
// box (to kill ghosting on moving surfaces), then exponentially blend. A
// dropped reprojection (off-screen, disoccluded, background, or a flagged
// history invalidation) falls back to the current frame.
//
// Self-contained (no imports): group 0 is
//   0 = ssr_resolved   (this frame's resolved reflection, textureLoad)
//   1 = scene_depth     (reverse-Z device depth, textureLoad)
//   2 = history         (previous accumulated reflection, sampled)
//   3 = history_sampler (filtering)
//   4 = temporal_out    (rgba16float storage, write)
// with a `TemporalParams` immediate block.

struct TemporalParams {
    // inverse(current clip_from_world): NDC+depth -> world.
    world_from_clip: mat4x4<f32>,
    // previous frame's view-projection: world -> previous clip.
    clip_from_world_prev: mat4x4<f32>,
    width: u32,
    height: u32,
    // fraction of reprojected history kept when the sample is valid.
    history_weight: f32,
    // symmetric colour-box expansion before the neighbourhood clip.
    clamp_expand: f32,
    // 1 when a previous frame is available (no resize / camera cut), else 0.
    valid_history: u32,
    // stddev multiplier for the variance clip box (mean +/- gamma*sigma).
    variance_gamma: f32,
    // floor the adaptive weight decays toward on a disocclusion.
    min_history_weight: f32,
    _pad0: u32,
}

@group(0) @binding(0) var ssr_resolved: texture_2d<f32>;
@group(0) @binding(1) var scene_depth: texture_2d<f32>;
@group(0) @binding(2) var history: texture_2d<f32>;
@group(0) @binding(3) var history_sampler: sampler;
@group(0) @binding(4) var temporal_out: texture_storage_2d<rgba16float, write>;

var<immediate> params: TemporalParams;

// Reconstruct world position from a reverse-Z device depth at pixel-centre `uv`.
fn reconstruct_world(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let h = params.world_from_clip * vec4<f32>(ndc, 1.0);
    return h.xyz / h.w;
}

// AABB-clip (Karis) returning both the clamped colour and the overshoot -- how
// far outside the box the raw history began, in box half-extents past the
// surface (0 when already inside). `xyz` is the clamped colour, `w` the
// overshoot. The overshoot is the disocclusion signal the adaptive weight uses.
fn clip_to_aabb_ex(hist: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec4<f32> {
    let center = 0.5 * (box_max + box_min);
    let extent = max(0.5 * (box_max - box_min), vec3<f32>(1.0e-5));
    let dir = hist - center;
    let units = dir / extent;
    let max_unit = max(max(abs(units.x), abs(units.y)), abs(units.z));
    if (max_unit > 1.0) {
        return vec4<f32>(center + dir / max_unit, max_unit - 1.0);
    }
    return vec4<f32>(hist, 0.0);
}

// Variance clip box mean +/- gamma*sigma from the first two colour moments of
// the 3x3 neighbourhood (`mean` and `mean_sq`, the mean of per-channel squares).
// Non-negative-clamped before sqrt to absorb the mean_sq - mean^2 cancellation.
fn variance_box_min(mean: vec3<f32>, mean_sq: vec3<f32>, gamma: f32) -> vec3<f32> {
    let variance = max(mean_sq - mean * mean, vec3<f32>(0.0));
    return mean - sqrt(variance) * max(gamma, 0.0);
}
fn variance_box_max(mean: vec3<f32>, mean_sq: vec3<f32>, gamma: f32) -> vec3<f32> {
    let variance = max(mean_sq - mean * mean, vec3<f32>(0.0));
    return mean + sqrt(variance) * max(gamma, 0.0);
}

// Decay the blend weight from history_weight toward min_history_weight as the
// reprojected history is clipped further outside the box (saturates at
// overshoot == 1, one full half-extent outside).
fn adaptive_weight(overshoot: f32) -> f32 {
    let base = clamp(params.history_weight, 0.0, 1.0);
    let floor = clamp(params.min_history_weight, 0.0, base);
    let t = clamp(overshoot, 0.0, 1.0);
    return base + (floor - base) * t;
}

@compute @workgroup_size(8, 8, 1)
fn accumulate_ssr(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    // This frame's resolved reflection (radiance + confidence) is the anchor.
    let current = textureLoad(ssr_resolved, coord, 0);

    let device_depth = textureLoad(scene_depth, coord, 0).r;

    // Gather the first two colour moments over the 3x3 neighbourhood to build
    // the variance clip box (mean +/- gamma*sigma), the AAA alternative to a raw
    // min/max box that hugs single outlier taps and flickers.
    var sum = current.rgb;
    var sum_sq = current.rgb * current.rgb;
    var count = 1.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            let nx = coord.x + dx;
            let ny = coord.y + dy;
            if (nx < 0 || ny < 0 || nx >= i32(params.width) || ny >= i32(params.height)) {
                continue;
            }
            let tap = textureLoad(ssr_resolved, vec2<i32>(nx, ny), 0).rgb;
            sum = sum + tap;
            sum_sq = sum_sq + tap * tap;
            count = count + 1.0;
        }
    }
    let mean = sum / count;
    let mean_sq = sum_sq / count;
    let expand = vec3<f32>(max(params.clamp_expand, 0.0));
    let box_min = variance_box_min(mean, mean_sq, params.variance_gamma) - expand;
    let box_max = variance_box_max(mean, mean_sq, params.variance_gamma) + expand;

    var out = current;

    // Reproject and accumulate only for real surfaces with a valid history.
    if (params.valid_history != 0u && device_depth > 0.0) {
        let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
            / vec2<f32>(f32(params.width), f32(params.height));
        let world = reconstruct_world(uv, device_depth);
        let clip_prev = params.clip_from_world_prev * vec4<f32>(world, 1.0);
        if (clip_prev.w > 0.0) {
            let ndc_prev = clip_prev.xyz / clip_prev.w;
            if (ndc_prev.x >= -1.0 && ndc_prev.x <= 1.0
                && ndc_prev.y >= -1.0 && ndc_prev.y <= 1.0) {
                let prev_uv = vec2<f32>(
                    ndc_prev.x * 0.5 + 0.5,
                    1.0 - (ndc_prev.y * 0.5 + 0.5),
                );
                let hist = textureSampleLevel(history, history_sampler, prev_uv, 0.0);
                let clip = clip_to_aabb_ex(hist.rgb, box_min, box_max);
                // Decay the blend toward the floor as the history is clipped
                // further out (a disocclusion or a moving surface) so it sheds
                // stale reflection instead of ghosting.
                let w = adaptive_weight(clip.w);
                let rgb = current.rgb * (1.0 - w) + clip.xyz * w;
                let conf = current.a * (1.0 - w) + hist.a * w;
                out = vec4<f32>(rgb, conf);
            }
        }
    }

    textureStore(temporal_out, coord, out);
}
