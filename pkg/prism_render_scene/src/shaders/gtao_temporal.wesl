//! Cross-frame temporal accumulation for the denoised GTAO buffer.
//!
//! GPU twin of `prism_render_shading::ao::temporal`. The spatial denoise kills
//! most of the per-pixel grain, but a single-frame GTAO estimate still *boils*
//! under motion: each frame the horizon search lands on slightly different
//! depths, so flat walls shimmer and contact shadows crawl. This kernel removes
//! that boil by accumulating the denoised visibility over time — reprojecting
//! the previous frame's converged AO onto the current pixel, variance-clipping
//! it to the current 3x3 neighbourhood, then exponentially blending a little of
//! the fresh estimate in each frame (`XeGTAO`'s temporal filter).
//!
//! A screen-space AO pass runs *before* the resolve writes its motion-vector
//! G-buffer, so there is no per-object motion here. Instead each pixel's
//! view-space position is reconstructed from its linear view depth and pushed
//! through the *previous* frame's `clip_from_world`. The two matrices collapse
//! into one — `clip_prev_from_view = clip_from_world_prev * world_from_view` —
//! because `world_from_view` is affine (`w = 1`), so the CPU-side merge is exact
//! and the immediate block stays small. A disocclusion, an off-screen
//! reprojection, a background pixel, or a flagged-invalid history falls back to
//! the current frame; the reprojected history is variance-clipped before the
//! blend so a moving occluder cannot drag a stale contact shadow (ghosting).
//!
//! Self-contained (no imports); group 0 matches `ao::temporal` bindings:
//!   0 = denoised_ao      (this frame's spatially denoised AO, textureLoad)
//!   1 = history          (previous accumulated AO, sampled at the reprojected UV)
//!   2 = history_sampler  (filtering, clamp-to-edge)
//!   3 = linear_depth     (positive view depth, textureLoad; reprojection + gate)
//!   4 = ao_out           (r32float storage, write; the resolve reads this)
//!   5 = history_out      (r32float storage, write; next frame's history)
//! with a `GtaoTemporalConfig` immediate block.

// Guards the adaptive-weight knee and the variance square root against zero;
// matches the golden `WEIGHT_EPSILON`.
const GTAO_TEMPORAL_WEIGHT_EPSILON: f32 = 1.0e-6;

// Exponential-accumulation tunables + the merged reprojection matrix, uploaded
// per dispatch. Mirrors the golden `GtaoTemporalParams` plus the derived camera
// state. `clip_prev_from_view` is first so its 64 bytes stay 16-byte aligned.
struct GtaoTemporalConfig {
    // clip_from_world_prev * world_from_view: view-space -> previous clip space.
    clip_prev_from_view: mat4x4<f32>,
    // 1 / |proj_diag|; scales NDC into a view-space slope (reconstruct).
    tan_half_fov_x: f32,
    tan_half_fov_y: f32,
    // fraction of the reprojected history kept when it agrees with the neighbourhood.
    history_weight: f32,
    // floor the adaptive weight decays toward on a disocclusion.
    min_history_weight: f32,
    // stddev multiplier for the variance clip band (mean +/- gamma*sigma).
    variance_gamma: f32,
    width: u32,
    height: u32,
    // 1 when a previous frame is available (no resize / first frame), else 0.
    valid_history: u32,
}

@group(0) @binding(0) var denoised_ao: texture_2d<f32>;
@group(0) @binding(1) var history: texture_2d<f32>;
@group(0) @binding(2) var history_sampler: sampler;
@group(0) @binding(3) var linear_depth_tex: texture_2d<f32>;
@group(0) @binding(4) var ao_out: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var history_out: texture_storage_2d<r32float, write>;

var<immediate> config: GtaoTemporalConfig;

// Positive linear depth at integer coord, or `-1.0` for background/non-finite
// (the golden's `Option::None`). Mirrors `GtaoBuffers::depth_at`.
fn depth_at(coord: vec2<i32>) -> f32 {
    let d = textureLoad(linear_depth_tex, coord, 0).x;
    if d > 0.0 && d < 3.4e38 {
        return d;
    }
    return -1.0;
}

// Variance clip band mean +/- gamma*sigma from the first two moments of the 3x3
// neighbourhood (`mean` and `mean_sq`, the mean of the squared visibilities).
// Non-negative-clamped before sqrt to absorb the mean_sq - mean^2 cancellation.
// Mirrors the golden `variance_clip_band`.
fn variance_band_lo(mean: f32, mean_sq: f32, gamma: f32) -> f32 {
    let variance = max(mean_sq - mean * mean, 0.0);
    return mean - sqrt(variance) * max(gamma, 0.0);
}
fn variance_band_hi(mean: f32, mean_sq: f32, gamma: f32) -> f32 {
    let variance = max(mean_sq - mean * mean, 0.0);
    return mean + sqrt(variance) * max(gamma, 0.0);
}

// Decay the blend weight from history_weight toward min_history_weight as the
// reprojected history is clipped further outside the band (saturates at
// overshoot == 1, one full half-width outside). Mirrors the golden
// `gtao_adaptive_history_weight`. `floor_w` avoids shadowing the `floor` builtin.
fn adaptive_weight(overshoot: f32) -> f32 {
    let base = clamp(config.history_weight, 0.0, 1.0);
    let floor_w = clamp(config.min_history_weight, 0.0, base);
    let t = clamp(overshoot, 0.0, 1.0);
    return base + (floor_w - base) * t;
}

@compute @workgroup_size(8, 8, 1)
fn accumulate_gtao(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= config.width || gid.y >= config.height) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    // This frame's denoised visibility is the anchor.
    let current = textureLoad(denoised_ao, coord, 0).x;
    let depth = depth_at(coord);

    // Gather the first two moments over the 3x3 neighbourhood to build the
    // variance clip band (mean +/- gamma*sigma), the AAA alternative to a raw
    // min/max box that hugs single outlier taps and flickers.
    var sum = current;
    var sum_sq = current * current;
    var count = 1.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            let nx = coord.x + dx;
            let ny = coord.y + dy;
            if (nx < 0 || ny < 0 || nx >= i32(config.width) || ny >= i32(config.height)) {
                continue;
            }
            let tap = textureLoad(denoised_ao, vec2<i32>(nx, ny), 0).x;
            sum = sum + tap;
            sum_sq = sum_sq + tap * tap;
            count = count + 1.0;
        }
    }
    let mean = sum / count;
    let mean_sq = sum_sq / count;

    // Default: no valid history -> pass the clamped current frame through.
    var accumulated = clamp(current, 0.0, 1.0);

    // Reproject and accumulate only for real surfaces with a valid history.
    if (config.valid_history != 0u && depth > 0.0) {
        // Pixel-centre UV (origin top-left, y down; the prepass convention).
        let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
            / vec2<f32>(f32(config.width), f32(config.height));
        let ndc_x = 2.0 * uv.x - 1.0;
        let ndc_y = 1.0 - 2.0 * uv.y;
        // View-space position (RH, camera looking down -Z; matches GtaoCamera).
        let view = vec3<f32>(
            ndc_x * config.tan_half_fov_x * depth,
            ndc_y * config.tan_half_fov_y * depth,
            -depth,
        );
        // Project through last frame's clip_from_world (merged with world_from_view).
        let clip_prev = config.clip_prev_from_view * vec4<f32>(view, 1.0);
        // Behind (or on) the previous near plane: no valid projection.
        if (clip_prev.w > 0.0) {
            let ndc_prev = clip_prev.xyz / clip_prev.w;
            // Inside the previous frustum on x/y (an off-screen reprojection or a
            // disocclusion drops history and keeps the current frame).
            if (ndc_prev.x >= -1.0 && ndc_prev.x <= 1.0
                && ndc_prev.y >= -1.0 && ndc_prev.y <= 1.0) {
                let prev_uv = vec2<f32>(
                    ndc_prev.x * 0.5 + 0.5,
                    1.0 - (ndc_prev.y * 0.5 + 0.5),
                );
                let hist = textureSampleLevel(history, history_sampler, prev_uv, 0.0).x;

                // Variance-clip the history to the current neighbourhood band.
                let lo = variance_band_lo(mean, mean_sq, config.variance_gamma);
                let hi = variance_band_hi(mean, mean_sq, config.variance_gamma);
                let center = 0.5 * (lo + hi);
                let half = max(0.5 * (hi - lo), GTAO_TEMPORAL_WEIGHT_EPSILON);
                let dist = hist - center;
                let unit = abs(dist) / half;
                var clipped = hist;
                var overshoot = 0.0;
                if (unit > 1.0) {
                    clipped = center + dist / unit;
                    overshoot = unit - 1.0;
                }

                // Decay the blend toward the floor as the history is clipped
                // further out (a disocclusion or a moving occluder) so it sheds
                // stale occlusion instead of ghosting.
                let w = adaptive_weight(overshoot);
                accumulated = clamp(current * (1.0 - w) + clipped * w, 0.0, 1.0);
            }
        }
    }

    // The resolve reads `ao_out`; `history_out` becomes next frame's history.
    textureStore(ao_out, coord, vec4<f32>(accumulated, 0.0, 0.0, 1.0));
    textureStore(history_out, coord, vec4<f32>(accumulated, 0.0, 0.0, 1.0));
}
