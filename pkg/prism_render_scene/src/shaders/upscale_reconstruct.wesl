// Temporal-upscale (FSR2 / UE-TSR class) reconstruction + accumulation pass.
//
// GPU twin of the CPU golden `prism_render_shading::upscale::upscale_pixel`
// (plus the Lanczos-2 reconstruction from `upscale::history` and the
// reprojection / disocclusion tests from `upscale::reproject`). Where TAA
// accumulates a *same-resolution* jittered history, temporal upscaling
// reconstructs a *higher*-resolution image from a stream of *lower*-resolution
// render targets: the camera is jittered over a long Halton sequence, each
// frame is drawn at `render_scale` of the display resolution, and this pass
// resolves those samples onto the full display grid.
//
// Per display pixel:
//
//   1. reconstruct the low-resolution current frame onto this display pixel
//      with a separable Lanczos-2 windowed-sinc kernel (`upscale::history`);
//   2. reproject the display-resolution history through the motion vector and
//      test for disocclusion — off-screen after the motion offset or a linear
//      depth mismatch (`upscale::reproject`);
//   3. clip the reprojected history to the current neighbourhood's `YCoCg`
//      variance box (Karis AABB clip) to kill ghosting, folding the clip
//      overshoot into the disocclusion signal;
//   4. detect thin sub-pixel luminance features and grow/melt a history lock so
//      the reconstruction cannot wash them out (`upscale::robust`);
//   5. grow the per-pixel temporal accumulation (clamped at `max_accumulation`,
//      melted by disocclusion but resisted by the lock) and blend the
//      reconstructed current sample with the clipped history by
//      `alpha = 1 / (accumulation + 1)`, floored by `min_alpha`, with the
//      current sample fully winning on a disocclusion.
//
// A first frame / resize / camera cut (`valid_history == 0` or the settings'
// invalidation dependencies intersecting this frame's events) collapses to the
// freshly reconstructed current sample rather than ghosting.
//
// Self-contained (no imports); group 0 is
//   0 = render_color   (this frame's low-resolution HDR radiance, textureLoad)
//   1 = history_color  (previous accumulated display-res colour, sampled)
//   2 = history_sampler (bilinear clamp filtering for the history colour)
//   3 = motion_vectors (rg screen-space motion `cur_uv - prev_uv`, textureLoad)
//   4 = depth          (current linear view-space depth, render-res, textureLoad)
//   5 = history_depth  (previous linear depth, display-res, textureLoad)
//   6 = history_meta   (previous accumulation .r + lock lifetime .g, textureLoad)
//   7 = color_out      (rgba16float storage; the resolved display colour)
//   8 = meta_out       (rgba16float storage; new accumulation .r + lock .g)
//   9 = depth_out     (r32float storage; this frame's depth for next-frame history)
// with an `UpscaleReconstructParams` immediate block.

const PI: f32 = 3.14159265358979323846;
const FLOAT_MAX: f32 = 3.4028235e38;

struct UpscaleReconstructParams {
    // Low-resolution render target extent (this frame's draw size).
    render_width: u32,
    render_height: u32,
    // Display (output / history) extent.
    display_width: u32,
    display_height: u32,
    // render / display in (0, 1]; the grid-conversion scale.
    render_scale: f32,
    // Relative linear-depth tolerance for the disocclusion test (e.g. 0.025).
    depth_tolerance: f32,
    // Half-width, in standard deviations, of the YCoCg neighbourhood clip box.
    variance_gamma: f32,
    // Minimum neighbourhood luma range that counts as a lockable thin feature.
    thin_feature_contrast: f32,
    // Floor on the current-sample blend weight (e.g. 1/32) so the accumulation
    // never fully freezes.
    min_alpha: f32,
    // Upper bound on the temporal accumulation count, in frames.
    max_accumulation: f32,
    // Upper bound on a history-lock lifetime, in frames.
    max_lock_lifetime: f32,
    // The settings' history-invalidation dependency mask and this frame's
    // invalidation events; a non-empty intersection forces a full reset.
    invalidation_dependencies: u32,
    invalidation_events: u32,
    // 1 when a valid previous frame exists (no first frame / resize / cut).
    valid_history: u32,
}

@group(0) @binding(0) var render_color: texture_2d<f32>;
@group(0) @binding(1) var history_color: texture_2d<f32>;
@group(0) @binding(2) var history_sampler: sampler;
@group(0) @binding(3) var motion_vectors: texture_2d<f32>;
@group(0) @binding(4) var depth: texture_2d<f32>;
@group(0) @binding(5) var history_depth: texture_2d<f32>;
@group(0) @binding(6) var history_meta: texture_2d<f32>;
@group(0) @binding(7) var color_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var meta_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var depth_out: texture_storage_2d<r32float, write>;

var<immediate> params: UpscaleReconstructParams;

// Rec.601-style RGB -> YCoCg (luma + two chroma); exactly invertible by
// `ycocg_to_rgb`. The clip box and thin-feature detector work in this space so
// a bright specular edge does not drag chroma around, matching the golden.
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

// The YCoCg luma of an RGB colour — the perceptual channel the thin-feature
// detector works in.
fn luma(rgb: vec3<f32>) -> f32 {
    return rgb_to_ycocg(rgb).x;
}

// Normalised sinc `sin(pi*x) / (pi*x)`, with the removable singularity at 0
// filled in as 1.
fn sinc(x: f32) -> f32 {
    if (abs(x) < 1.0e-6) {
        return 1.0;
    }
    let px = PI * x;
    return sin(px) / px;
}

// Lanczos-2 windowed-sinc kernel value: `sinc(x)*sinc(x/2)` inside the two-lobe
// support `|x| < 2`, and 0 outside.
fn lanczos2_weight(x: f32) -> f32 {
    if (abs(x) >= 2.0) {
        return 0.0;
    }
    return sinc(x) * sinc(x * 0.5);
}

// The four normalised Lanczos-2 tap weights for a sample sitting a fractional
// `frac` in [0, 1) past the second of four consecutive taps (offsets -1..2).
// Normalised to sum to 1 so a flat input reconstructs exactly; a degenerate
// zero sum falls back to the nearest tap.
fn lanczos2_weights(frac: f32) -> array<f32, 4> {
    let f = clamp(frac, 0.0, 1.0);
    var w = array<f32, 4>(
        lanczos2_weight(f + 1.0),
        lanczos2_weight(f),
        lanczos2_weight(f - 1.0),
        lanczos2_weight(f - 2.0),
    );
    let sum = w[0] + w[1] + w[2] + w[3];
    if (abs(sum) > 1.0e-6) {
        let inv = 1.0 / sum;
        w[0] = w[0] * inv;
        w[1] = w[1] * inv;
        w[2] = w[2] * inv;
        w[3] = w[3] * inv;
    } else {
        w = array<f32, 4>(0.0, 1.0, 0.0, 0.0);
    }
    return w;
}

// Load a render-grid texel, clamping to the edge so gathers always contribute.
fn load_render(x: i32, y: i32) -> vec3<f32> {
    let cx = clamp(x, 0, i32(params.render_width) - 1);
    let cy = clamp(y, 0, i32(params.render_height) - 1);
    return textureLoad(render_color, vec2<i32>(cx, cy), 0).rgb;
}

// Load a render-grid linear depth, clamped to the edge.
fn load_depth(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.render_width) - 1);
    let cy = clamp(y, 0, i32(params.render_height) - 1);
    return textureLoad(depth, vec2<i32>(cx, cy), 0).r;
}

// Variance clip box half-extent `gamma*sigma` from the first two YCoCg colour
// moments of the neighbourhood (`mean` and `mean_sq`).
fn variance_box_half(mean: vec3<f32>, mean_sq: vec3<f32>, gamma: f32) -> vec3<f32> {
    let variance = max(mean_sq - mean * mean, vec3<f32>(0.0));
    return sqrt(variance) * max(gamma, 0.0);
}

// AABB-clip (Karis): pull `hist` back onto the surface of the `[box_min,
// box_max]` box and report how far outside it began (`max_unit - 1`, else 0),
// matching the golden `clip_history_to_aabb_ex`. The overshoot rides in `.w`.
fn clip_history_to_aabb(hist: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec4<f32> {
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

// A per-pixel depth disocclusion signal in [0, 1] from linear (view-space,
// positive) depths, with a soft ramp from `tolerance` (still trusted) to
// `2*tolerance` (fully disoccluded). A non-positive / non-finite depth on
// either side is a full disocclusion, matching the golden.
fn depth_disocclusion(current_depth: f32, history_depth_value: f32, tolerance: f32) -> f32 {
    if (current_depth <= 0.0
        || history_depth_value <= 0.0
        || current_depth != current_depth
        || history_depth_value != history_depth_value
        || current_depth >= FLOAT_MAX
        || history_depth_value >= FLOAT_MAX) {
        return 1.0;
    }
    let tol = max(tolerance, 1.0e-6);
    let denom = max(current_depth, history_depth_value);
    let relative = abs(current_depth - history_depth_value) / denom;
    return clamp((relative - tol) / tol, 0.0, 1.0);
}

// Whether a reprojected UV falls outside the [0, 1]^2 history frame.
fn is_offscreen(uv: vec2<f32>) -> bool {
    return uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0;
}

// Thin-feature strength in [0, 1]: the centre luma at an extreme of a
// high-contrast neighbourhood range. `extremeness = |2t - 1|` scaled by a
// contrast confidence, zero on a flat / low-contrast neighbourhood.
fn thin_feature_strength(center_luma: f32, min_luma: f32, max_luma: f32, contrast_threshold: f32) -> f32 {
    let contrast = max_luma - min_luma;
    let threshold = max(contrast_threshold, 1.0e-6);
    if (contrast <= threshold) {
        return 0.0;
    }
    let t = clamp((center_luma - min_luma) / contrast, 0.0, 1.0);
    let extremeness = abs(2.0 * t - 1.0);
    let confidence = clamp((contrast - threshold) / threshold, 0.0, 1.0);
    return clamp(extremeness * confidence, 0.0, 1.0);
}

// Advance a history lock one frame: a disocclusion melts the surviving lifetime
// proportionally, a still-requested feature refreshes it (clamped to
// `max_lock`), otherwise it decays by one frame toward zero.
fn update_lock(prev_lifetime: f32, requested: bool, disocclusion: f32, max_lock: f32) -> f32 {
    let d = clamp(disocclusion, 0.0, 1.0);
    let survived = max(prev_lifetime, 0.0) * (1.0 - d);
    if (requested) {
        return min(survived + 1.0, max_lock);
    }
    return max(survived - 1.0, 0.0);
}

// Normalised lock strength in [0, 1] (`lifetime / max_lock`).
fn lock_strength(lifetime: f32, max_lock: f32) -> f32 {
    if (max_lock <= 0.0) {
        return 0.0;
    }
    return clamp(lifetime / max_lock, 0.0, 1.0);
}

// Advance the per-pixel temporal accumulation one frame: the surviving history
// is melted by `disocclusion`, but a `lock` resists that melt so a locked
// feature is not reset by a transient mismatch. Clamped to `max_acc`.
fn update_accumulation(prev_frames: f32, disocclusion: f32, lock: f32, max_acc: f32) -> f32 {
    let d = clamp(disocclusion, 0.0, 1.0);
    let l = clamp(lock, 0.0, 1.0);
    let melt = d * (1.0 - l);
    let survived = max(prev_frames, 0.0) * (1.0 - melt);
    return min(survived + 1.0, max_acc);
}

// Current-sample blend weight for an accumulation count: `1 / (accumulation +
// 1)`, floored by `min_alpha` so the accumulation never fully stops responding.
fn history_blend_alpha_clamped(accumulation: f32, min_alpha: f32) -> f32 {
    let alpha = 1.0 / (max(accumulation, 0.0) + 1.0);
    return max(alpha, clamp(min_alpha, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn upscale_reconstruct(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.display_width || gid.y >= params.display_height) {
        return;
    }
    let display_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let display_pos = vec2<f32>(f32(gid.x), f32(gid.y));
    let display_size = vec2<f32>(f32(params.display_width), f32(params.display_height));

    // --- 1. Lanczos-2 reconstruction of the low-res frame onto this pixel. ---
    // Map the display pixel centre into the render grid (`display_to_render`),
    // shifted so pixel centres align, then reconstruct from the 4x4 tap block
    // at integer offsets -1..2 with the separable Lanczos-2 kernel.
    let render_sample = (display_pos + vec2<f32>(0.5)) * params.render_scale - vec2<f32>(0.5);
    let base = floor(render_sample);
    let frac = render_sample - base;
    let base_i = vec2<i32>(i32(base.x), i32(base.y));

    var wx = lanczos2_weights(frac.x);
    var wy = lanczos2_weights(frac.y);
    var current_color = vec3<f32>(0.0);
    for (var row = 0; row < 4; row = row + 1) {
        var row_acc = vec3<f32>(0.0);
        for (var col = 0; col < 4; col = col + 1) {
            let tap = load_render(base_i.x + col - 1, base_i.y + row - 1);
            row_acc = row_acc + tap * wx[col];
        }
        current_color = current_color + row_acc * wy[row];
    }

    // --- Current 3x3 neighbourhood (nearest render texels) for the YCoCg clip
    // box and the thin-feature luma range. ---
    let near = vec2<i32>(i32(round(render_sample.x)), i32(round(render_sample.y)));
    var sum = vec3<f32>(0.0);
    var sum_sq = vec3<f32>(0.0);
    var lo = FLOAT_MAX;
    var hi = -FLOAT_MAX;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let tap = load_render(near.x + dx, near.y + dy);
            let y = rgb_to_ycocg(tap);
            sum = sum + y;
            sum_sq = sum_sq + y * y;
            lo = min(lo, y.x);
            hi = max(hi, y.x);
        }
    }
    let mean = sum / 9.0;
    let mean_sq = sum_sq / 9.0;

    let current_depth = load_depth(near.x, near.y);

    // --- 2-3. Reproject, clip and score disocclusion. ---
    let invalidated = (params.invalidation_dependencies & params.invalidation_events) != 0u;
    var disocclusion = 1.0;
    var clipped_color = current_color;
    var prev_accumulation = 0.0;
    var prev_lock = 0.0;

    if (params.valid_history != 0u && !invalidated) {
        let uv = (display_pos + vec2<f32>(0.5)) / display_size;
        // Motion vector (render-res buffer) already folds camera + object
        // motion; `prev_uv = cur_uv - motion` finds last frame's surface.
        let motion = textureLoad(motion_vectors, vec2<i32>(
            clamp(near.x, 0, i32(params.render_width) - 1),
            clamp(near.y, 0, i32(params.render_height) - 1),
        ), 0).xy;
        let prev_uv = uv - motion;

        if (is_offscreen(prev_uv)) {
            disocclusion = 1.0;
        } else {
            let hist_rgb = textureSampleLevel(history_color, history_sampler, prev_uv, 0.0).rgb;
            let half_box = variance_box_half(mean, mean_sq, params.variance_gamma);
            let clip = clip_history_to_aabb(rgb_to_ycocg(hist_rgb), mean - half_box, mean + half_box);
            clipped_color = ycocg_to_rgb(clip.xyz);
            let overshoot = clip.w;

            let prev_coord = vec2<i32>(
                clamp(i32(prev_uv.x * display_size.x), 0, i32(params.display_width) - 1),
                clamp(i32(prev_uv.y * display_size.y), 0, i32(params.display_height) - 1),
            );
            let hist_depth = textureLoad(history_depth, prev_coord, 0).r;
            let hist_meta = textureLoad(history_meta, prev_coord, 0);
            prev_accumulation = hist_meta.r;
            prev_lock = hist_meta.g;

            var d = depth_disocclusion(current_depth, hist_depth, params.depth_tolerance);
            d = max(d, clamp(overshoot, 0.0, 1.0));
            disocclusion = d;
        }
    }

    // --- 4. Thin-feature lock. ---
    let center_luma = luma(current_color);
    let requested = thin_feature_strength(center_luma, lo, hi, params.thin_feature_contrast) > 0.5;
    let lock = update_lock(prev_lock, requested, disocclusion, params.max_lock_lifetime);
    let lock_amount = lock_strength(lock, params.max_lock_lifetime);

    // --- 5. Temporal accumulation and blend. ---
    let accumulation = update_accumulation(prev_accumulation, disocclusion, lock_amount, params.max_accumulation);
    let alpha = history_blend_alpha_clamped(accumulation, params.min_alpha);
    // The current sample must fully win on a disocclusion, whatever the history.
    let current_weight = clamp(max(alpha, disocclusion), 0.0, 1.0);
    let color = clipped_color + (current_color - clipped_color) * current_weight;

    textureStore(color_out, display_coord, vec4<f32>(color, 1.0));
    textureStore(meta_out, display_coord, vec4<f32>(accumulation, lock, 0.0, 0.0));
    // Persist this display pixel's depth (nearest render texel) so next
    // frame's reprojection has a display-resolution history depth to test
    // against. Written here rather than in a separate pass because the
    // reconstruction already resolved the nearest render-grid depth.
    textureStore(depth_out, display_coord, vec4<f32>(current_depth, 0.0, 0.0, 0.0));
}
