//! SSR current-frame scene-colour pyramid mip-chain build.
//!
//! The screen-space trace (`ssr.wesl`) samples the current frame's linear HDR
//! colour at each reflected-ray hit and picks a roughness-selected mip so
//! rougher surfaces read a pre-blurred (wider) reflection. Prism's resolve
//! writes the current frame's shaded colour into `scene_color`; this stage
//! copies it into the trace's colour pyramid and builds that pyramid's full mip
//! chain once per frame, right after the resolve, so the trace samples this
//! frame's own radiance (no cross-frame history, motion vectors, or
//! reprojection are involved).
//!
//! Three entry points, one mip level per dispatch (mirroring `ssr_hzb.wesl`):
//!
//! * `ssr_color_copy`         — lifts the resolve's full-resolution `scene_color`
//!   into pyramid level 0 (a 1:1 copy so level 0 is the untouched shaded colour).
//! * `ssr_color_reduce_karis` — writes the *first* coarser level (mip 0 -> mip 1)
//!   as a Karis luma-weighted 2x2 average so lone very-bright texels do not smear
//!   into fireflies when a rough surface reads this pre-blurred mip.
//! * `ssr_color_reduce`       — writes every *coarser* level (mip >= 2) as the 2x2
//!   plain *average* of the finer one (an energy-preserving box downsample; colour
//!   averages where the depth pyramid takes the reverse-Z max), clamping the
//!   trailing column/row for odd extents so the footprint never reads out of
//!   bounds.
//!
//! Both read one source mip as a sampled `texture_2d<f32>` and write one
//! destination mip as an `rgba16float` storage image; the host binds the
//! matching per-mip views and drives the extents through the immediate
//! `MipParams` block (mirrors `GpuSsrHzbParams` in `shading/ssr/abi.rs` — the
//! identical destination-then-source `vec2<u32>` extent contract).

// Immediate (push-constant) block. Mirrors `GpuSsrHzbParams` (16 bytes: the
// destination then the source extent, each a `vec2<u32>`).
struct MipParams {
    dst_width: u32,
    dst_height: u32,
    src_width: u32,
    src_height: u32,
}

// Source mip (scene colour for the copy, the finer pyramid level for a reduce).
@group(0) @binding(0) var src: texture_2d<f32>;
// Destination mip (pyramid level being written).
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

var<immediate> params: MipParams;

// Lifts the resolve's `scene_color` into history level 0. A 1:1 copy keeps
// level 0 identical to the shaded HDR colour, so the trace's finest (mirror)
// reflections read exactly what was on screen last frame.
@compute @workgroup_size(8, 8, 1)
fn ssr_color_copy(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.dst_width || id.y >= params.dst_height {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    let color = textureLoad(src, coord, 0);
    textureStore(dst, coord, color);
}

// Writes one coarser pyramid level as the 2x2 average (box downsample) of the
// finer level. The 2x2 footprint is clamped to the source's last texel so odd
// source dimensions reuse the trailing column/row rather than reading out of
// bounds, matching the Hi-Z build's clamp.
@compute @workgroup_size(8, 8, 1)
fn ssr_color_reduce(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.dst_width || id.y >= params.dst_height {
        return;
    }
    let max_x = i32(max(params.src_width, 1u) - 1u);
    let max_y = i32(max(params.src_height, 1u) - 1u);
    let base_x = i32(id.x * 2u);
    let base_y = i32(id.y * 2u);

    let x0 = min(base_x, max_x);
    let x1 = min(base_x + 1, max_x);
    let y0 = min(base_y, max_y);
    let y1 = min(base_y + 1, max_y);

    let c00 = textureLoad(src, vec2<i32>(x0, y0), 0);
    let c10 = textureLoad(src, vec2<i32>(x1, y0), 0);
    let c01 = textureLoad(src, vec2<i32>(x0, y1), 0);
    let c11 = textureLoad(src, vec2<i32>(x1, y1), 0);

    let averaged = (c00 + c10 + c01 + c11) * 0.25;
    textureStore(dst, vec2<i32>(i32(id.x), i32(id.y)), averaged);
}

// Rec.709 luma of a linear HDR colour, used to weight the firefly-suppressing
// downsample below.
fn ssr_color_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Writes the *first* coarser pyramid level (mip 0 -> mip 1) as a Karis
// luma-weighted 2x2 average instead of a plain box average. Each source texel
// is weighted by `1 / (1 + luma)`, so a lone very-bright pixel (a specular
// highlight or emitter that would otherwise smear into a firefly when a rough
// surface samples this pre-blurred mip) contributes far less than the
// surrounding, dimmer neighbourhood. This is the standard UE/COD
// "Next Generation Post Processing" partial-Karis trick: it is applied only at
// the source mip where fireflies originate, and coarser levels keep the plain,
// energy-preserving box average (`ssr_color_reduce`) so the pyramid does not
// progressively darken. The 2x2 footprint clamps to the source's last texel
// exactly like the box reduce so odd extents never read out of bounds.
@compute @workgroup_size(8, 8, 1)
fn ssr_color_reduce_karis(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.dst_width || id.y >= params.dst_height {
        return;
    }
    let max_x = i32(max(params.src_width, 1u) - 1u);
    let max_y = i32(max(params.src_height, 1u) - 1u);
    let base_x = i32(id.x * 2u);
    let base_y = i32(id.y * 2u);

    let x0 = min(base_x, max_x);
    let x1 = min(base_x + 1, max_x);
    let y0 = min(base_y, max_y);
    let y1 = min(base_y + 1, max_y);

    let c00 = textureLoad(src, vec2<i32>(x0, y0), 0);
    let c10 = textureLoad(src, vec2<i32>(x1, y0), 0);
    let c01 = textureLoad(src, vec2<i32>(x0, y1), 0);
    let c11 = textureLoad(src, vec2<i32>(x1, y1), 0);

    let w00 = 1.0 / (1.0 + ssr_color_luma(c00.rgb));
    let w10 = 1.0 / (1.0 + ssr_color_luma(c10.rgb));
    let w01 = 1.0 / (1.0 + ssr_color_luma(c01.rgb));
    let w11 = 1.0 / (1.0 + ssr_color_luma(c11.rgb));

    let weighted = c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11;
    let weight_sum = w00 + w10 + w01 + w11;
    // `weight_sum` is >= 4/(1+inf) in the limit but always strictly positive
    // here (each weight is in (0, 1]), so the divide is safe without a guard.
    textureStore(dst, vec2<i32>(i32(id.x), i32(id.y)), weighted / weight_sum);
}
