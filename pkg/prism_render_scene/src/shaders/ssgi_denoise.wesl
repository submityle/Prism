//! Edge-aware spatial SSGI denoiser (compute twin).
//!
//! GPU twin of `prism_render_shading::screen_space::denoise_ssgi`. A single
//! SSGI gather traces only a handful of cosine-weighted rays per pixel, so its
//! raw indirect-diffuse buffer is heavy with Monte-Carlo noise — grainy on open
//! surfaces and splotchy where few rays find a valid hit. This kernel is the
//! spatial half of the AAA denoiser: a joint **bilateral** blur that averages
//! each pixel with its `(2 * radius + 1)^2` neighbours while depth *and* normal
//! edge-stopping terms keep the indirect radiance from bleeding across geometry
//! seams (the same `XeGTAO`-style kernel `gtao_denoise.wesl` applies to AO,
//! generalised to the four-channel SSGI output).
//!
//! It reads the raw SSGI target the gather wrote (`rgb` = pre-albedo mean
//! indirect radiance, `a` = blend confidence) plus the linear view-depth and
//! view-space normal prepass targets (for edge stopping), and writes the
//! denoised buffer the SSGI composite folds over the IBL/SH ambient. Every
//! constant and expression mirrors the CPU golden bit-for-bit within tolerance
//! so the two agree on machines with and without a GPU. The file is
//! self-contained (no `import`s), matching `gtao_denoise.wesl`.

// Guards the weight normalisation and the sigma denominators against zero;
// matches the golden `WEIGHT_EPSILON`.
const SSGI_DENOISE_WEIGHT_EPSILON: f32 = 1.0e-6;

// Bilateral tunables, uploaded per dispatch. Mirrors the golden
// `SsgiDenoiseConfig`: the kernel half-width plus the three edge-stop sigmas.
struct SsgiDenoiseConfig {
    // Clip -> view (inverse reverse-Z projection): reconstructs the linear
    // view-space depth each edge-stop compares, from the device depth the SSR
    // subsystem already stores. Mirrors `ssgi.wesl`'s `view_from_clip`.
    view_from_clip: mat4x4<f32>,
    // Full-resolution framebuffer extent, converting integer texels to the UV
    // the reconstruction inverse-projects.
    screen_size: vec2<f32>,
    radius: u32,
    spatial_sigma: f32,
    depth_sigma: f32,
    normal_power: f32,
    _pad0: u32,
    _pad1: u32,
}

// raw SSGI (rgb = pre-albedo mean indirect radiance, a = blend confidence).
@group(0) @binding(0) var raw_ssgi_tex: texture_2d<f32>;
// full-resolution device depth (reverse-Z; 0 = far/cleared background) in `.x`,
// reprojected to positive linear view depth for the edge stop.
@group(0) @binding(1) var scene_depth_tex: texture_2d<f32>;
// unit view-space normal in `.xyz`.
@group(0) @binding(2) var view_normal_tex: texture_2d<f32>;
// denoised SSGI output (rgb radiance, a confidence).
@group(0) @binding(3) var ssgi_out: texture_storage_2d<rgba16float, write>;

var<immediate> config: SsgiDenoiseConfig;

// Positive linear view depth at integer coord, or `-1.0` for the reverse-Z
// background (device depth 0, the golden's `Option::None`). The device depth is
// inverse-projected to a view-space position and its `-z` returned, matching
// the linear depth the CPU golden receives in `SsgiDenoiseBuffers`.
fn depth_at(coord: vec2<i32>) -> f32 {
    let device_depth = textureLoad(scene_depth_tex, coord, 0).x;
    // Reverse-Z: device depth 0 is the far plane / cleared background.
    if device_depth <= 0.0 {
        return -1.0;
    }
    let uv = (vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5, 0.5)) / config.screen_size;
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let view = config.view_from_clip * vec4<f32>(ndc, 1.0);
    let view_z = view.z / view.w;
    // View space looks down -Z, so the positive linear depth is `-view_z`.
    let linear = -view_z;
    if linear > 0.0 && linear < 3.4e38 {
        return linear;
    }
    return -1.0;
}

// Denoised SSGI value for the pixel at integer coord `coord`. Mirrors
// `denoise_ssgi_pixel`.
fn denoise_pixel(coord: vec2<i32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(ssgi_out));
    let center = textureLoad(raw_ssgi_tex, coord, 0);

    // Background/sky has no meaningful surface to blur along: keep it as-is.
    let center_depth = depth_at(coord);
    if center_depth <= 0.0 {
        return center;
    }
    let center_normal = textureLoad(view_normal_tex, coord, 0).xyz;

    let radius = i32(config.radius);
    let spatial_sigma = max(config.spatial_sigma, SSGI_DENOISE_WEIGHT_EPSILON);
    let two_spatial_sq = 2.0 * spatial_sigma * spatial_sigma;
    // Relative depth tolerance: scale by the centre depth so the edge stop is
    // perspective-correct.
    let depth_scale = max(config.depth_sigma, SSGI_DENOISE_WEIGHT_EPSILON) * center_depth;
    let two_depth_sq = 2.0 * depth_scale * depth_scale;
    let normal_power = max(config.normal_power, 0.0);

    var weighted_sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weight_total = 0.0;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let sx = coord.x + dx;
            let sy = coord.y + dy;
            if sx < 0 || sy < 0 || sx >= dims.x || sy >= dims.y {
                continue;
            }
            let sample_coord = vec2<i32>(sx, sy);
            // Neighbours on background carry no valid indirect radiance; skip
            // them so a silhouette against the sky does not darken the surface.
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
            weighted_sum = weighted_sum + weight * textureLoad(raw_ssgi_tex, sample_coord, 0);
            weight_total = weight_total + weight;
        }
    }

    if weight_total > SSGI_DENOISE_WEIGHT_EPSILON {
        let inv = 1.0 / weight_total;
        let rgb = max(weighted_sum.xyz * inv, vec3<f32>(0.0, 0.0, 0.0));
        let a = clamp(weighted_sum.w * inv, 0.0, 1.0);
        return vec4<f32>(rgb, a);
    }
    return center;
}

@compute @workgroup_size(8, 8, 1)
fn denoise_ssgi(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(ssgi_out);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    let denoised = denoise_pixel(coord);
    textureStore(ssgi_out, coord, denoised);
}
