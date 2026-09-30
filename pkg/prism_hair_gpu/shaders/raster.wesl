// Hair strand sub-pixel raster-path classification twin: evaluates on-device
// the same per-segment routing decision the CPU golden `classify_hair_raster`
// computes, one thread per segment.
//
// A render strand is usually a fraction of a pixel wide, so production hair
// engines (UE5 Groom, AMD `TressFX`) rasterize thin strands in a compute
// visibility pass and reserve the hardware triangle path for the rare near,
// thick strand (design 6.2 / hair raster module). The CPU golden for that
// routing is `classify_hair_raster` (in
// `prism_render_architecture::hair::raster`); this kernel mirrors it exactly:
//
//   min_coverage    = max(params.min_coverage, 0)
//   width_threshold = max(params.width_threshold, 0)
//   if screen_length <= 0 or coverage < min_coverage -> CULLED
//   else if screen_width <= width_threshold          -> SUBPIXEL_SOFTWARE
//   else                                             -> THICK_HARDWARE
//
// The decision is pure comparison arithmetic with no fused multiply-add and no
// transcendental call, so the emitted path code is bit-identical to the
// reference (the parity test asserts exact equality on the integer code). Both
// threshold clamps reproduce the reference's `.max(0.0)` guard so a negative
// tunable behaves identically on both sides. It uses only the portable
// core-`WGSL` subset (max plus comparisons) - no exp, no pow, no optional
// feature - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard analytic thin-strand raster routing; no Unreal Engine
// source or derived code.

// Raster-path discriminants, matching the CPU readback decoding and the
// `HairRasterPath` variant order.
const PATH_SUBPIXEL_SOFTWARE: u32 = 0u;
const PATH_THICK_HARDWARE: u32 = 1u;
const PATH_CULLED: u32 = 2u;

struct Params {
    // Number of valid entries in `queries` / codes in `paths`.
    count: u32,
    pad0: u32,
    // Coverage strictly below this (after a `>= 0` clamp) is culled.
    min_coverage: f32,
    // Screen width at or below this (after a `>= 0` clamp) takes the software
    // path.
    width_threshold: f32,
}

// One projected strand-segment statistic. `16`-byte stride matching the
// `bytemuck` upload struct `RasterQuery`.
struct Query {
    screen_width_px: f32,
    screen_length_px: f32,
    coverage: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> paths: array<u32>;

// Classifies one segment, mirroring `classify_hair_raster`.
fn classify(q: Query) -> u32 {
    let min_coverage = max(params.min_coverage, 0.0);
    let width_threshold = max(params.width_threshold, 0.0);
    if (q.screen_length_px <= 0.0 || q.coverage < min_coverage) {
        return PATH_CULLED;
    }
    if (q.screen_width_px <= width_threshold) {
        return PATH_SUBPIXEL_SOFTWARE;
    }
    return PATH_THICK_HARDWARE;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    paths[i] = classify(queries[i]);
}
