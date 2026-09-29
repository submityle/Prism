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

// Immediate (push-constant) block: the framebuffer extent (leading for its
// 8-byte alignment, and used to bounds-check each invocation) followed by the
// golden `levels` palette size and `strength` blend weight. Mirrors the Rust
// `GpuOrderedDitherParams` field-for-field (16 bytes, no implicit padding).
struct GpuOrderedDitherParams {
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<u32>,
    // Number of discrete quantisation steps per channel (golden `levels`).
    levels: u32,
    // Blend weight of the dithered result over the input (golden `strength`).
    strength: f32,
}

// The pre-exposed, resolved linear HDR scene colour this pass dithers.
@group(0) @binding(0) var ordered_dither_scene_color: texture_2d<f32>;
// The dithered HDR output; copied back over `scene_color` after the dispatch.
@group(0) @binding(1) var ordered_dither_out: texture_storage_2d<rgba16float, write>;

var<immediate> ordered_dither_immediate: GpuOrderedDitherParams;

// Repack the immediate block into the golden `OrderedDitherParams` the maths
// twin consumes, so the on-device evaluation is arm-for-arm identical to the
// CPU reference. `enabled` is hard-set to 1: the dispatch only runs when the
// host enabled the pass.
fn ordered_dither_params_from_immediate() -> OrderedDitherParams {
    var params: OrderedDitherParams;
    params.levels = ordered_dither_immediate.levels;
    params.strength = ordered_dither_immediate.strength;
    params.enabled = 1u;
    return params;
}

// One invocation per pixel. Evaluates the golden ordered dither on the linear
// HDR radiance, indexing the Bayer matrix by the integer pixel coordinates
// (`gid.xy`) exactly as the CPU golden takes `px` / `py`, and writes the
// dithered HDR texel (alpha preserved). Runs in the scene-referred linear
// domain to match the current pre-tonemap post chain, exactly like the golden.
@compute @workgroup_size(8, 8, 1)
fn ordered_dither_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = i32(ordered_dither_immediate.screen_size.x);
    let height = i32(ordered_dither_immediate.screen_size.y);
    let pc = vec2<i32>(i32(gid.x), i32(gid.y));
    if (pc.x >= width || pc.y >= height) {
        return;
    }

    let sample = textureLoad(ordered_dither_scene_color, pc, 0);

    let params = ordered_dither_params_from_immediate();
    // The Bayer matrix is indexed by the unsigned pixel coordinates so the
    // dither pattern tiles every 4 texels, matching the CPU golden's px / py.
    let dithered = ordered_dither_apply(sample.rgb, gid.x, gid.y, params);

    textureStore(ordered_dither_out, pc, vec4<f32>(dithered, sample.a));
}
