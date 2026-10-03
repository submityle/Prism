// Water waterline-mask compute kernel (entry point `water_waterline_mask`).
//
// Rasterizes the soft above/below-water transition and the shallow-water
// shoreline band for every screen pixel, mirroring the CPU golden
// `waterline::waterline_weight` and `waterline::shoreline_band` byte-for-byte.
//
// ABI (matches `WaterKernel::WaterlineMask.descriptor()`: 0 storage buffers,
// 1 uniform, 1 storage texture, 1 sampled texture, 8x8 tile, Screen domain):
//   @binding(0) params      : WaterlineMaskParams uniform (16 bytes)
//   @binding(1) scene_in    : texture_2d<f32>, per-pixel
//                             (sample_y, water_surface_y, water_depth, _)
//   @binding(2) mask_out    : texture_storage_2d<rg32float, write>,
//                             (waterline_weight, shoreline_band)
//
// Only `+ - * /`, comparisons, `clamp` and `max` appear — no float intrinsic
// the workspace determinism policy forbids. `EPS` matches the crate's
// `water::EPS` (1e-6) exactly so the division guards fire on the same inputs
// as the CPU twin.

const EPS: f32 = 1e-6;

struct WaterlineMaskParams {
    transition_half_width: f32,
    shoreline_depth: f32,
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> params: WaterlineMaskParams;
@group(0) @binding(1) var scene_in: texture_2d<f32>;
@group(0) @binding(2) var mask_out: texture_storage_2d<rg32float, write>;

// Soft waterline weight in 0..=1: 0 fully in air, 1 fully submerged.
// `depth` is the signed submersion depth (water_surface_y - sample_y).
fn waterline_weight(depth: f32, half: f32) -> f32 {
    if (half <= EPS) {
        if (depth >= 0.0) {
            return 1.0;
        }
        return 0.0;
    }
    return clamp((depth + half) / (2.0 * half), 0.0, 1.0);
}

// Shoreline-band weight in 0..=1 for a submerged, shallow sample.
fn shoreline_band(depth: f32, water_depth: f32, shoreline_depth: f32) -> f32 {
    // `is_underwater` == depth >= 0.0; dry samples contribute nothing.
    if (depth < 0.0) {
        return 0.0;
    }
    if (shoreline_depth <= EPS) {
        return 0.0;
    }
    let clamped_depth = max(water_depth, 0.0);
    return clamp(1.0 - clamped_depth / shoreline_depth, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn water_waterline_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let s = textureLoad(scene_in, coord, 0);
    let sample_y = s.x;
    let water_surface_y = s.y;
    let water_depth = s.z;

    let depth = water_surface_y - sample_y;
    let w = waterline_weight(depth, params.transition_half_width);
    let band = shoreline_band(depth, water_depth, params.shoreline_depth);

    textureStore(mask_out, coord, vec4<f32>(w, band, 0.0, 0.0));
}
