//! SSR Hi-Z ("nearest depth") pyramid build.
//!
//! The screen-space trace (`ssr.wesl`, CPU golden `prism_render_shading::
//! screen_space::march`) climbs a reverse-Z depth pyramid: each coarse cell
//! stores the *maximum* device depth (nearest surface, since reverse-Z puts the
//! near plane at `1` and the far plane at `0`) of the finer texels beneath it,
//! so a whole cell can be skipped when the ray segment is nearer than anything
//! in it. This kernel builds that pyramid one level per dispatch:
//!
//! * `ssr_hzb_copy`   — lifts the full-resolution `scene_depth` prepass target
//!   into pyramid level 0 (a straight 1:1 copy so level 0 == device depth).
//! * `ssr_hzb_reduce` — writes one coarser level as the 2x2 max-reduction of
//!   the finer one, clamping the trailing column/row for odd source extents,
//!   exactly reproducing `DepthPyramid::from_nearest_reduction`.
//!
//! Both entry points read a single source mip as a sampled `texture_2d<f32>`
//! and write a single destination mip as an `r32float` storage image; the host
//! binds the matching per-mip views and drives the extents through the
//! immediate `HzbParams` block (mirrors `GpuSsrHzbParams` in `shading/ssr/
//! abi.rs`).

// Immediate (push-constant) block. Mirrors `GpuSsrHzbParams` (16 bytes: the
// destination then the source extent, each a `vec2<u32>`).
struct HzbParams {
    dst_width: u32,
    dst_height: u32,
    src_width: u32,
    src_height: u32,
}

// Source mip (device depth for the copy, the finer pyramid level for a reduce).
@group(0) @binding(0) var src: texture_2d<f32>;
// Destination mip (pyramid level being written).
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

var<immediate> params: HzbParams;

// Lifts `scene_depth` into pyramid level 0. A 1:1 copy keeps level 0 identical
// to the device depth the trace reconstructs positions from, so the finest
// march step compares against the same values the resolve shades.
@compute @workgroup_size(8, 8, 1)
fn ssr_hzb_copy(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.dst_width || id.y >= params.dst_height {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    let depth = textureLoad(src, coord, 0).x;
    textureStore(dst, coord, vec4<f32>(depth, 0.0, 0.0, 0.0));
}

// Writes one coarser pyramid level as the 2x2 max (reverse-Z nearest) of the
// finer level. The 2x2 footprint is clamped to the source's last texel so odd
// source dimensions reuse the trailing column/row, matching the CPU golden's
// `.min(prev_size - 1)` clamp rather than reading out of bounds.
@compute @workgroup_size(8, 8, 1)
fn ssr_hzb_reduce(@builtin(global_invocation_id) id: vec3<u32>) {
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

    let d00 = textureLoad(src, vec2<i32>(x0, y0), 0).x;
    let d10 = textureLoad(src, vec2<i32>(x1, y0), 0).x;
    let d01 = textureLoad(src, vec2<i32>(x0, y1), 0).x;
    let d11 = textureLoad(src, vec2<i32>(x1, y1), 0).x;

    let nearest = max(max(d00, d10), max(d01, d11));
    textureStore(dst, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(nearest, 0.0, 0.0, 0.0));
}
