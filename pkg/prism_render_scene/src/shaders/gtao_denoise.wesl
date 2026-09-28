//! Edge-aware spatial GTAO denoiser (compute twin).
//!
//! GPU twin of `prism_render_shading::ao::denoise_gtao`. A single GTAO pass
//! sweeps only a handful of slices per pixel, so its raw ambient-visibility
//! buffer is grainy in the open and ropey along creases. This kernel is the
//! spatial half of the AAA denoiser: a joint **bilateral** blur that averages
//! each pixel with its `(2 * radius + 1)^2` neighbours while depth *and* normal
//! edge-stopping terms keep occlusion from bleeding across geometry seams
//! (`XeGTAO`'s denoise pass, generalised to an `N x N` kernel).
//!
//! It reads the raw ambient-visibility target the GTAO kernel wrote plus the
//! linear view-depth and view-space normal prepass targets (for edge stopping),
//! and writes the denoised visibility the shading resolve multiplies into its
//! ambient term. Every constant and expression mirrors the CPU golden
//! bit-for-bit within tolerance so the two agree on machines with and without a
//! GPU. The file is self-contained (no `import`s), matching `gtao.wesl`.

// Guards the weight normalisation and the sigma denominators against zero;
// matches the golden `WEIGHT_EPSILON`.
const GTAO_DENOISE_WEIGHT_EPSILON: f32 = 1.0e-6;

// Bilateral tunables, uploaded per dispatch. Mirrors the golden
// `GtaoDenoiseConfig`: the kernel half-width plus the three edge-stop sigmas.
struct GtaoDenoiseConfig {
    radius: u32,
    spatial_sigma: f32,
    depth_sigma: f32,
    normal_power: f32,
}

// raw ambient visibility (1 = unoccluded) written by the GTAO kernel, in `.x`.
@group(0) @binding(0) var raw_ao_tex: texture_2d<f32>;
// linear view depth (positive distance, <= 0 = background) in `.x`.
@group(0) @binding(1) var linear_depth_tex: texture_2d<f32>;
// unit view-space normal in `.xyz`.
@group(0) @binding(2) var view_normal_tex: texture_2d<f32>;
// denoised ambient visibility output (1 = unoccluded) in `.x`.
@group(0) @binding(3) var ao_out: texture_storage_2d<r32float, write>;

var<immediate> config: GtaoDenoiseConfig;

// Positive linear depth at integer coord, or `-1.0` for background/non-finite
// (the golden's `Option::None`). Mirrors `GtaoBuffers::depth_at`.
fn depth_at(coord: vec2<i32>) -> f32 {
    let d = textureLoad(linear_depth_tex, coord, 0).x;
    if d > 0.0 && d < 3.4e38 {
        return d;
    }
    return -1.0;
}

// Denoised ambient visibility for the pixel at integer coord `coord`. Mirrors
// `denoise_gtao_pixel`.
fn denoise_pixel(coord: vec2<i32>) -> f32 {
    let dims = vec2<i32>(textureDimensions(ao_out));
    let center = textureLoad(raw_ao_tex, coord, 0).x;

    // Background/sky has no meaningful surface to blur along: keep it as-is.
    let center_depth = depth_at(coord);
    if center_depth <= 0.0 {
        return center;
    }
    let center_normal = textureLoad(view_normal_tex, coord, 0).xyz;

    let radius = i32(config.radius);
    let spatial_sigma = max(config.spatial_sigma, GTAO_DENOISE_WEIGHT_EPSILON);
    let two_spatial_sq = 2.0 * spatial_sigma * spatial_sigma;
    // Relative depth tolerance: scale by the centre depth so the edge stop is
    // perspective-correct.
    let depth_scale = max(config.depth_sigma, GTAO_DENOISE_WEIGHT_EPSILON) * center_depth;
    let two_depth_sq = 2.0 * depth_scale * depth_scale;
    let normal_power = max(config.normal_power, 0.0);

    var weighted_sum = 0.0;
    var weight_total = 0.0;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let sx = coord.x + dx;
            let sy = coord.y + dy;
            if sx < 0 || sy < 0 || sx >= dims.x || sy >= dims.y {
                continue;
            }
            let sample_coord = vec2<i32>(sx, sy);
            // Neighbours on background carry no valid occlusion; skip them so a
            // silhouette against the sky does not darken the foreground.
            let sample_depth = depth_at(sample_coord);
            if sample_depth <= 0.0 {
                continue;
            }

            let dist_sq = f32(dx * dx + dy * dy);
            let w_spatial = exp(-dist_sq / two_spatial_sq);

            let depth_delta = sample_depth - center_depth;
            let w_depth = exp(-(depth_delta * depth_delta) / two_depth_sq);

            let sample_normal = textureLoad(view_normal_tex, sample_coord, 0).xyz;
            let n_dot = max(dot(center_normal, sample_normal), 0.0);
            var w_normal = 1.0;
            if normal_power > 0.0 {
                w_normal = pow(n_dot, normal_power);
            }

            let weight = w_spatial * w_depth * w_normal;
            weighted_sum = weighted_sum + weight * textureLoad(raw_ao_tex, sample_coord, 0).x;
            weight_total = weight_total + weight;
        }
    }

    if weight_total > GTAO_DENOISE_WEIGHT_EPSILON {
        return clamp(weighted_sum / weight_total, 0.0, 1.0);
    }
    return center;
}

@compute @workgroup_size(8, 8, 1)
fn denoise_gtao(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(ao_out);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    let visibility = denoise_pixel(coord);
    textureStore(ao_out, coord, vec4<f32>(visibility, 0.0, 0.0, 1.0));
}
