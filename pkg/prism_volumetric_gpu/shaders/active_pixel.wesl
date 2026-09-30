// Active-pixel twin: evaluates on-device the low-resolution raymarch scheduling
// predicate, mirroring the CPU golden `active_pixel` (in
// `prism_render_architecture::volumetric::temporal`).
//
// Volumetric clouds raymarch at a fraction of the output resolution and the
// full image is reconstructed over several frames (design section 10). This
// predicate decides, deterministically, whether pixel (x, y) is raymarched on
// frame `frame_index` under one of three update modes:
//   Full        (mode 0): always active (period 1).
//   Checkerboard(mode 1): ((x ^ y ^ frame_index) & 1) == 0 - half the pixels
//                         each frame, the resolved half swapping every frame,
//                         two consecutive frames covering all (period 2).
//   QuarterRes  (mode 2): (x & 1) == (frame_index & 1)
//                         && (y & 1) == ((frame_index >> 1) & 1) - one pixel of
//                         every 2x2 block each frame, cycling all four members
//                         every four frames (period 4).
// Over one period the per-frame active sets partition, then union to, the whole
// grid, so no pixel is ever starved. The CPU golden owns that logic; this
// kernel is the on-device twin that runs one thread per query.
//
// `mode` is uploaded as a u32 (0 = Full, 1 = Checkerboard, 2 = QuarterRes) and
// the result is a u32 (0 = skip, 1 = active). It uses only the portable
// core-WGSL subset - integer bit ops and comparisons - so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: standard checkerboard / quarter-res upsampling schedule; no
// Unreal Engine source or derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One scheduling query: the frame index, pixel coordinates and the update mode
// ordinal. Matches `GpuQuery` in `src/active_pixel.rs`.
struct Query {
    frame_index: u32,
    x: u32,
    y: u32,
    mode: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

// The scheduling predicate, mirroring `active_pixel`.
fn active_pixel(frame_index: u32, x: u32, y: u32, mode: u32) -> bool {
    if (mode == 0u) {
        // Full: always active.
        return true;
    }
    if (mode == 1u) {
        // Checkerboard.
        return ((x ^ y ^ frame_index) & 1u) == 0u;
    }
    // QuarterRes.
    return (x & 1u) == (frame_index & 1u) && (y & 1u) == ((frame_index >> 1u) & 1u);
}

@compute @workgroup_size(64)
fn active_pixel_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let is_active = active_pixel(q.frame_index, q.x, q.y, q.mode);
    results[idx] = select(0u, 1u, is_active);
}
