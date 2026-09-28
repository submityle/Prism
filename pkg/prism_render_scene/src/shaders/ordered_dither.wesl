// GPU twin of `prism_render_shading::ordered_dither` (NPR ordered-dithering /
// Bayer-threshold posterisation: a fixed 4x4 Bayer matrix biases a per-channel
// quantisation to a small palette / bit depth, then blends back by strength).
// This is ordered dithering, NOT the halftone pass: halftone draws round ink
// dots, this thresholds against the Bayer matrix and snaps to discrete levels.
// Self-contained (no intra-crate imports, matching ssgi.wesl / bloom.wesl /
// color_grade.wesl / exposure.wesl) so the WESL compile test can validate it
// standalone. Every function mirrors the CPU golden arm-for-arm: the Bayer
// matrix is the same 16 values element-for-element, the quantiser is identical
// and lerps are written `a + (b - a) * t`; WGSL componentwise vector ops match
// the per-channel Rust and `if` guards are written with `select`.

// Ordered-dither threshold for pixel (x, y): index the fixed 4x4 Bayer matrix
// by (x mod 4, y mod 4) via `& 3u`, normalise to [0, 1) by /16 and recentre by
// -0.5. Same 16 values, same layout as the CPU `ORDERED_DITHER_BAYER_4X4`.
fn ordered_dither_bayer4x4_threshold(x: u32, y: u32) -> f32 {
    let bayer = array<f32, 16>(
        0.0, 8.0, 2.0, 10.0,
        12.0, 4.0, 14.0, 6.0,
        3.0, 11.0, 1.0, 9.0,
        15.0, 7.0, 13.0, 5.0,
    );
    let ix = x & 3u;
    let iy = y & 3u;
    return bayer[iy * 4u + ix] / 16.0 - 0.5;
}

// Snap `scalar` to the nearest of `levels` evenly spaced steps:
// floor(scalar * (levels - 1) + 0.5) / (levels - 1). `levels` is clamped up to
// 2 so the denominator is never zero (levels = 2 is a black/white threshold).
fn ordered_dither_quantize(scalar: f32, levels: u32) -> f32 {
    let denom = f32(max(levels, 2u) - 1u);
    return floor(scalar * denom + 0.5) / denom;
}

// Artist controls for the full pass (mirrors the CPU `OrderedDitherParams`; the
// disabled default is the identity).
struct OrderedDitherParams {
    levels: u32,
    strength: f32,
    enabled: u32,
}

// Apply ordered dithering to one pixel: bias each channel by the Bayer
// threshold / levels, quantise to `levels` steps, then blend the posterised
// colour over the input by `strength` (a + (b - a) * t). enabled = 0 (or
// strength = 0) is the identity. `select` mirrors the CPU early-out.
fn ordered_dither_apply(scene: vec3<f32>, px: u32, py: u32, params: OrderedDitherParams) -> vec3<f32> {
    let threshold = ordered_dither_bayer4x4_threshold(px, py);
    let offset = threshold / f32(params.levels);
    let dithered = vec3<f32>(
        ordered_dither_quantize(scene.x + offset, params.levels),
        ordered_dither_quantize(scene.y + offset, params.levels),
        ordered_dither_quantize(scene.z + offset, params.levels),
    );
    let mixed = scene + (dithered - scene) * params.strength;
    return select(scene, mixed, params.enabled != 0u);
}

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real dither pass wires the target and the params
// uniform later. References every helper once so none drifts unused.
@compute @workgroup_size(64, 1, 1)
fn ordered_dither_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seed = f32(gid.x) * 0.01;
    let scene = vec3<f32>(0.15 + seed, 0.4, 0.85);

    let t = ordered_dither_bayer4x4_threshold(gid.x, gid.y);
    let q = ordered_dither_quantize(scene.x + seed, 4u);

    var params: OrderedDitherParams;
    params.levels = 4u;
    params.strength = 1.0;
    params.enabled = 0u;
    let out = ordered_dither_apply(scene, gid.x, gid.y, params);

    _ = t + q + out.x + f32(params.levels) + params.strength;
}
