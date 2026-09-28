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

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real gamut-map pass wires the HDR target and the
// params uniform later. References every function so none drifts unused.
@compute @workgroup_size(64, 1, 1)
fn gamut_map_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seed = f32(gid.x) * 0.01;
    let rgb = vec3<f32>(1.4 + seed, 0.5, -0.2);

    let lum = gamut_map_luminance(rgb);
    let mx = gamut_map_max_channel(rgb);
    let mn = gamut_map_min_channel(rgb);
    let ac = gamut_map_achromatic(rgb);
    let dist = gamut_map_distance_from_achromatic(rgb, ac);
    let cd = gamut_map_compress_distance(dist.z, 0.6, 1.0);
    let cl = gamut_map_clamp01(seed);

    var params: GamutMapParams;
    params.threshold = vec3<f32>(0.6, 0.6, 0.6);
    params.limit = vec3<f32>(1.0, 1.0, 1.0);
    params.power = 1.0;
    params.enabled = 1u;
    let mapped = gamut_map_apply(rgb, params);

    _ = lum + mx + mn + ac + dist.x + cd + cl + mapped.x + params.power;
}
