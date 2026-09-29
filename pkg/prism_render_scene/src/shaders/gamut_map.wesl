// GPU twin of `prism_render_shading::gamut_map` (perceptual gamut mapping in the
// scene-referred linear domain, applied after grading and before the display
// tone-map curve: anchor on the achromatic axis, measure each channel's
// relative distance, compress the out-of-gamut excess along a rational knee and
// reconstruct, preserving the anchor and the hue).
// Self-contained (no intra-crate imports, matching ssgi.wesl / bloom.wesl /
// exposure.wesl / color_grade.wesl) so the WESL compile test can validate it
// standalone. Every function mirrors the CPU golden arm-for-arm: same operation
// order, same constants, `if` guards written with `select` (the color_grade.wesl
// precedent) and lerps written `a + (b - a) * t` rather than `mix` so the GPU
// twin stays bit-identical with the golden. `min`/`max`/`clamp` on a vec3 are
// componentwise in WGSL, matching the per-channel Rust.

// Rec. 709 luma weights (linear sRGB primaries), used by gamut_map_luminance.
const GAMUT_MAP_LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Rec. 709 relative luminance of a linear RGB sample.
fn gamut_map_luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, GAMUT_MAP_LUMA_WEIGHTS);
}

// Largest of the three linear channels.
fn gamut_map_max_channel(rgb: vec3<f32>) -> f32 {
    return max(rgb.x, max(rgb.y, rgb.z));
}

// Smallest of the three linear channels.
fn gamut_map_min_channel(rgb: vec3<f32>) -> f32 {
    return min(rgb.x, min(rgb.y, rgb.z));
}

// Achromatic (neutral-axis) anchor: `max(RGB)` (the ACES gamut-compress
// convention), so the max channel is a fixed point of the whole map.
fn gamut_map_achromatic(rgb: vec3<f32>) -> f32 {
    return gamut_map_max_channel(rgb);
}

// Per-channel relative distance from the achromatic anchor `ac`: `(ac - c) / ac`,
// 0 on the anchor, 1 at a primary and > 1 for an out-of-gamut (negative)
// channel. A non-positive anchor is degenerate, so the distance collapses to 0.
fn gamut_map_distance_from_achromatic(rgb: vec3<f32>, ac: f32) -> vec3<f32> {
    let dist = (vec3<f32>(ac, ac, ac) - rgb) / ac;
    return select(dist, vec3<f32>(0.0, 0.0, 0.0), ac <= 0.0);
}

// Compress a single relative distance along a rational knee. Distances below
// `threshold` (and the degenerate `limit <= threshold`) pass through unchanged;
// above `threshold` the excess `x = dist - threshold` follows
// `threshold + x / (1 + x / (limit - threshold))`, asymptoting to `limit`.
fn gamut_map_compress_distance(dist: f32, threshold: f32, limit: f32) -> f32 {
    let span = limit - threshold;
    let x = dist - threshold;
    let compressed = threshold + x / (1.0 + x / span);
    let knee = select(compressed, dist, dist < threshold);
    return select(knee, dist, span <= 0.0);
}

// Clamp a scalar to the [0, 1] display range.
fn gamut_map_clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Artist controls for the gamut compression (mirrors the CPU `GamutMapParams`;
// `enabled` is a u32 flag for the GPU uniform, 0 = disabled identity).
struct GamutMapParams {
    threshold: vec3<f32>,
    limit: vec3<f32>,
    power: f32,
    enabled: u32,
}

// Apply the full gamut compression in the fixed order
// anchor -> distance -> per-channel compress -> reconstruct, then blend by the
// master enable. In-gamut colours reconstruct to themselves; enabled = 0 is a
// hard identity.
fn gamut_map_apply(rgb: vec3<f32>, params: GamutMapParams) -> vec3<f32> {
    let ac = gamut_map_achromatic(rgb);
    let dist = gamut_map_distance_from_achromatic(rgb, ac);
    let cd = vec3<f32>(
        gamut_map_compress_distance(dist.x, params.threshold.x, params.limit.x),
        gamut_map_compress_distance(dist.y, params.threshold.y, params.limit.y),
        gamut_map_compress_distance(dist.z, params.threshold.z, params.limit.z),
    );
    let compressed = vec3<f32>(ac, ac, ac) - cd * ac;
    let t = select(0.0, 1.0, params.enabled != 0u);
    return rgb + (compressed - rgb) * t;
}

// Immediate (push-constant) block: the golden per-channel `threshold`/`limit`
// and the reserved `power` exponent packed into `vec4` lanes (WGSL gives `vec4`
// a 16-byte alignment, so packing the scalars into the free `w` lane avoids the
// implicit padding a bare `vec3` would force), plus the framebuffer extent and a
// device-only global `scale`. Mirrors the Rust `GpuGamutMapParams` field-for-
// field (48 bytes).
struct GpuGamutMapParams {
    // xyz = per-channel distance threshold (the working gamut), w = the reserved
    // perceptual power exponent.
    threshold_power: vec4<f32>,
    // xyz = per-channel asymptotic limit (the gamut boundary target), w = the
    // device-only global scale in [0, 1].
    limit_scale: vec4<f32>,
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<u32>,
    // Explicit tail padding so the block is a multiple of 16 bytes.
    _pad: vec2<u32>,
}

// The pre-exposed, resolved linear HDR scene colour this pass maps.
@group(0) @binding(0) var gamut_map_scene_color: texture_2d<f32>;
// The compressed HDR output; copied back over `scene_color` after the dispatch.
@group(0) @binding(1) var gamut_map_out: texture_storage_2d<rgba16float, write>;

var<immediate> gamut_map_immediate: GpuGamutMapParams;

// Repack the immediate block into the golden `GamutMapParams` the maths twin
// consumes, so the on-device evaluation is arm-for-arm identical to the CPU
// reference. `enabled` is hard-set to 1: the dispatch only runs when the host
// enabled the pass, and the device-only `scale` (applied on top) is the sole
// non-golden control that fades the compression back toward the input.
fn gamut_map_params_from_immediate() -> GamutMapParams {
    var params: GamutMapParams;
    params.threshold = gamut_map_immediate.threshold_power.xyz;
    params.limit = gamut_map_immediate.limit_scale.xyz;
    params.power = gamut_map_immediate.threshold_power.w;
    params.enabled = 1u;
    return params;
}

// One invocation per pixel. Evaluates the golden gamut compression on the linear
// HDR radiance, fades it toward identity by the global `scale` and writes the
// compressed HDR texel (alpha preserved). Compression runs in the scene-referred
// linear domain, exactly like the golden.
@compute @workgroup_size(8, 8, 1)
fn gamut_map_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = i32(gamut_map_immediate.screen_size.x);
    let height = i32(gamut_map_immediate.screen_size.y);
    let pc = vec2<i32>(i32(gid.x), i32(gid.y));
    if (pc.x >= width || pc.y >= height) {
        return;
    }

    let sample = textureLoad(gamut_map_scene_color, pc, 0);

    let params = gamut_map_params_from_immediate();
    let mapped = gamut_map_apply(sample.rgb, params);
    // Fade the compression toward identity (the input) by the global scale;
    // scale = 1 is the exact golden map, scale = 0 disables the effect per-pixel.
    let scale = clamp(gamut_map_immediate.limit_scale.w, 0.0, 1.0);
    let out_rgb = sample.rgb + (mapped - sample.rgb) * scale;

    textureStore(gamut_map_out, pc, vec4<f32>(out_rgb, sample.a));
}
