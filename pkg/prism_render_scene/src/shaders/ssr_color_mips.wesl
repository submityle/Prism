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
//! Two entry points, one mip level per dispatch (mirroring `ssr_hzb.wesl`):
//!
//! * `ssr_color_copy`   — lifts the resolve's full-resolution `scene_color` into
//!   pyramid level 0 (a 1:1 copy so level 0 is the untouched shaded colour).
//! * `ssr_color_reduce` — writes each coarser level as the 2x2 *average* of the
//!   finer one (a box downsample; colour averages where the depth pyramid takes
//!   the reverse-Z max), clamping the trailing column/row for odd extents so the
//!   footprint never reads out of bounds.
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
