//! SSR spatial reconstruction (bilateral resolve) compute twin.
//!
//! GPU twin of [`prism_render_shading::screen_space::resolve_reflection`]. The
//! multi-ray GGX trace (`ssr.wesl`) leaves the reflection buffer noisy on rough
//! surfaces; rather than pay for hundreds of rays, this pass treats every nearby
//! pixel's reflection as an extra sample of the *same* lobe and collapses the
//! noise with an edge-aware neighbourhood filter that respects surface normals,
//! view-space depth and each neighbour's own trace confidence, so silhouettes
//! and depth discontinuities never bleed.
//!
//! For each pixel it gathers a fixed `(2*radius+1)^2` neighbourhood, computes
//! the identical spatial x normal x depth weight the golden's
//! `resolve_geometry_weight` does, and produces `rgb` = the confidence-weighted
//! mean reflected radiance and `a` = the geometry-weighted mean of the taps'
//! confidences. A neighbourhood of mostly-missed rays therefore resolves to a
//! low confidence the composite leans toward the IBL fallback for.
//!
//! The spatial sigma is widened per pixel by the centre roughness (a mirror
//! wants a tight kernel that preserves sharp reflections; a rough surface wants
//! a wide one that smooths its broad lobe). The golden takes the already-scaled
//! sigma, so this file owns the roughness scaling and feeds the same scaled
//! sigma into the shared weight, keeping the two bit-compatible within
//! tolerance.
//!
//! The depth bilateral term compares *linear view-space* depth (not raw
//! reverse-Z device depth) so the golden's relative `depth_sigma` tolerance
//! holds uniformly across the frustum; each tap's view-space z is reconstructed
//! from its device depth through `view_from_clip`, exactly like the trace's
//! `reconstruct_view_position`. The file is self-contained (no `import`s),
//! matching the `ssr_composite.wesl` / `gtao.wesl` precedent.

// Immediate (push-constant) block. Mirrors `GpuSsrResolveParams` (96 bytes: a
// `mat4x4<f32>` inverse projection then the extent, kernel radius and the shared
// bilateral tunables, one trailing `u32` padding to the 16-byte alignment the
// matrix forces).
struct ResolveParams {
    // Clip -> view (inverse projection); reconstructs linear view-space depth.
    view_from_clip: mat4x4<f32>,
    // Full-resolution framebuffer extent in texels.
    width: u32,
    height: u32,
    // Neighbourhood half-width: the kernel spans `(2*radius+1)^2` taps.
    radius: u32,
    // Gaussian sigma of the spatial term in pixels, *before* roughness scaling.
    spatial_sigma: f32,
    // Exponent on `max(dot(n, n_centre), 0)`; larger rejects tilted neighbours.
    normal_power: f32,
    // Relative depth tolerance: term is `exp(-|d - d0| / (sigma * max(|d0|,eps)))`.
    depth_sigma: f32,
    // How strongly centre roughness widens the spatial sigma (see below).
    roughness_sigma_scale: f32,
    _pad0: u32,
}

// Noisy multi-ray trace: rgb = reflected radiance, a = blend confidence.
@group(0) @binding(0) var ssr_in: texture_2d<f32>;
// View-space normal (rgb, biased to [0, 1]) plus perceptual roughness (a).
@group(0) @binding(1) var normal_roughness: texture_2d<f32>;
// Full-resolution reverse-Z device depth; 0 = background.
@group(0) @binding(2) var scene_depth: texture_2d<f32>;
// Output: the resolved reflection (rgb radiance, a confidence).
@group(0) @binding(3) var ssr_resolved: texture_storage_2d<rgba16float, write>;

var<immediate> params: ResolveParams;

// Reconstruct linear view-space z from a pixel's device depth, mirroring the
// trace's `reconstruct_view_position` (reverse-Z NDC, camera looking down -Z).
fn reconstruct_view_z(pixel: vec2<i32>, device_depth: f32) -> f32 {
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5))
        / vec2<f32>(f32(params.width), f32(params.height));
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let view = params.view_from_clip * vec4<f32>(ndc, 1.0);
    return view.z / view.w;
}

@compute @workgroup_size(8, 8, 1)
fn resolve_ssr(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let pixel = vec2<i32>(i32(id.x), i32(id.y));

    let center_device = textureLoad(scene_depth, pixel, 0).x;
    // Reverse-Z: device depth 0 is the far plane / cleared background.
    if center_device <= 0.0 {
        textureStore(ssr_resolved, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let center_nr = textureLoad(normal_roughness, pixel, 0);
    let center_normal = normalize(center_nr.xyz * 2.0 - 1.0);
    let roughness = center_nr.w;
    let center_depth = reconstruct_view_z(pixel, center_device);

    // Roughness-scaled spatial sigma: smooth -> tight (preserve sharp mirror),
    // rough -> wide (smooth the broad lobe). The golden receives this scaled
    // sigma so both sides evaluate the identical Gaussian.
    let sigma = max(
        params.spatial_sigma * (0.5 + params.roughness_sigma_scale * roughness),
        1.0e-4,
    );
    let two_sigma_sq = 2.0 * sigma * sigma;
    let normal_power = max(params.normal_power, 0.0);
    let depth_scale = max(params.depth_sigma, 1.0e-4) * max(abs(center_depth), 1.0e-4);

    var color = vec3<f32>(0.0);
    // Sum of geometry weights (denominator for the resolved confidence).
    var geom_sum = 0.0;
    // Sum of geometry * confidence (denominator for the resolved radiance).
    var radiance_weight_sum = 0.0;

    let radius = i32(params.radius);
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let tap = pixel + vec2<i32>(dx, dy);
            if tap.x < 0 || tap.y < 0 || tap.x >= i32(params.width) || tap.y >= i32(params.height) {
                continue;
            }
            let tap_device = textureLoad(scene_depth, tap, 0).x;
            // Skip background neighbours: they carry no valid surface to filter.
            if tap_device <= 0.0 {
                continue;
            }

            let tap_nr = textureLoad(normal_roughness, tap, 0);
            let tap_normal = normalize(tap_nr.xyz * 2.0 - 1.0);
            let tap_depth = reconstruct_view_z(tap, tap_device);
            let refl = textureLoad(ssr_in, tap, 0);

            // Shared golden `resolve_geometry_weight`: spatial x normal x depth.
            let dist_sq = f32(dx * dx + dy * dy);
            let spatial = exp(-dist_sq / two_sigma_sq);
            let n_dot = max(dot(center_normal, tap_normal), 0.0);
            let normal = pow(n_dot, normal_power);
            let depth = exp(-abs(tap_depth - center_depth) / depth_scale);
            let geom = max(spatial * normal * depth, 0.0);

            let conf = clamp(refl.a, 0.0, 1.0);
            let radiance_weight = geom * conf;
            color = color + refl.rgb * radiance_weight;
            geom_sum = geom_sum + geom;
            radiance_weight_sum = radiance_weight_sum + radiance_weight;
        }
    }

    var reflected = vec3<f32>(0.0);
    var confidence = 0.0;
    if radiance_weight_sum > 1.0e-8 {
        reflected = color / radiance_weight_sum;
    }
    if geom_sum > 1.0e-8 {
        confidence = clamp(radiance_weight_sum / geom_sum, 0.0, 1.0);
    }
    textureStore(ssr_resolved, pixel, vec4<f32>(reflected, confidence));
}
