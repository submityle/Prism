// `wgpu` compute twin of the dependency-free `CPU` golden still-flood-depth map
// (`prism_render_architecture::water::coastline::flood_depth`, applied per
// cell).
//
// `UE5` Water floods every part of a terrain that sits below the still sea
// level: the per-cell flood depth is `max(0, sea_level - height)`, zero on dry
// land so a coast never reports negative water. This twin evaluates that map on
// device so a terrain-streaming pass can derive the flood field in the same
// place the rest of the water subsystem already lives on the `GPU`.
//
// The map is embarrassingly parallel: one invocation owns one cell, with no
// barrier and no workgroup memory. Each cell is a single subtraction followed
// by a clamp to zero — the identical floating-point operation the `CPU` golden
// performs — so the readback is bit-identical to the golden up to the
// last-place slack of a `GPU` fused multiply-add. The kernel uses only the
// portable core-`WGSL` subset (subtract plus `max`), so it runs unmodified on
// Metal, Vulkan and DX12.

// Uniform parameters for one dispatch. Layout matches `Params` in
// `src/water_coastline_flood_depth.rs` (`16` bytes, 16-byte aligned).
struct Params {
    // Number of terrain samples (`heights.len()`), `> 0`.
    count: u32,
    // Padding so `sea_level` and the struct stay 16-byte aligned.
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Still sea level in meters; land above it stays dry.
    sea_level: f32,
    pad3: f32,
    pad4: f32,
    pad5: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Terrain heights in meters, one per cell.
@group(0) @binding(1) var<storage, read> heights: array<f32>;
// Per-cell still flood depth in meters; `0` on dry land.
@group(0) @binding(2) var<storage, read_write> depth: array<f32>;

@compute @workgroup_size(256)
fn flood_depth_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count {
        return;
    }
    // `max(sea_level - height, 0)` — the golden's clamped subtraction verbatim.
    depth[idx] = max(params.sea_level - heights[idx], 0.0);
}
