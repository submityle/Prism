//! Screen-space fluid-surface reconstruction (`van der Laan`): the smooth +
//! normal half of the `Volume` render path, split out of the retired collocated
//! `FLIP` shader so the live surface pass owns a tight four-binding group.
//!
//! Compute twin of the CPU golden
//! `prism_render_architecture::water::reconstruct` (the `van der Laan`
//! screen-space depth splat → bilateral smooth → normal reconstruction path).
//! The sandbox has no `GPU`, so this shader is validated by `naga` compilation
//! and the `water::gpu_tests` `CPU`-parity harness on real hardware; it is one
//! dispatch (`water_surface_reconstruct`), scheduled by the host.
//!
//! The entry-point name below is byte-identical to the one the host
//! `prism_render_architecture::water::kernels::WaterKernel::wesl_entry_point`
//! contract emits, so the pipeline-cache key resolves against this module.
//!
//! ## Bindings (`@group(0)`, four entries)
//! The pass reads the splatted fluid `depth` and `thickness` buffers, writes the
//! reconstructed `(normal.xyz, smoothed_depth)` into a storage texture, and
//! reads one uniform of filter/projection scalars. This is exactly the
//! `SurfaceReconstruct` descriptor the architecture contract declares (two
//! storage buffers, one storage texture, one uniform) — no leftover `FLIP`
//! particle/grid bindings.
//!
//! ## Determinism
//! One invocation per screen pixel writes only its own texel; every buffer index
//! is bounds-checked, so out-of-range invocations return without touching
//! memory. The bilateral weights are a fixed, order-independent reduction over a
//! clamped neighborhood, so the output is deterministic.

/// Squared length below which a direction is treated as degenerate (mirrors the
/// CPU `EPS` guards so no normalization divides by ~0).
const RECON_EPS: f32 = 1.0e-6;

/// Sentinel "no fluid splatted here" depth; any pixel at or beyond this is
/// background and is skipped by the bilateral filter.
const RECON_DEPTH_FAR: f32 = 1.0e30;

/// Screen-space reconstruction parameters (`van der Laan` smooth + normal pass).
/// Field layout matches the host `GpuFlipSurfaceParams` ABI byte-for-byte.
struct SurfaceReconstructParams {
    /// Viewport resolution in pixels (`xy`).
    resolution: vec2<u32>,
    /// Bilateral filter half-width in pixels (clamped to a small window).
    filter_radius: i32,
    /// Spatial Gaussian falloff denominator `2·σ_spatial²` (`> 0`).
    spatial_sigma2: f32,
    /// Range (depth) Gaussian falloff denominator `2·σ_range²` (`> 0`).
    range_sigma2: f32,
    /// View-space scale converting a one-pixel step into a world offset at unit
    /// depth, used for the normal finite differences.
    pixel_world_scale: f32,
}

// Splatted fluid depth: one f32 per screen pixel (`RECON_DEPTH_FAR` = empty).
@group(0) @binding(0) var<storage, read> surface_depth: array<f32>;
// Splatted fluid thickness/coverage per pixel (downstream shading input).
@group(0) @binding(1) var<storage, read> surface_thickness: array<f32>;
// Output: reconstructed `(normal.xyz, smoothed_depth)` per pixel.
@group(0) @binding(2) var surface_normal_tex: texture_storage_2d<rgba16float, write>;
// Filter + projection scalars.
@group(0) @binding(3) var<uniform> surface_params: SurfaceReconstructParams;

/// Reconstructs a view-space position from a pixel coordinate and its depth
/// (a linear pinhole approximation using the per-pixel world scale).
fn recon_view_position(px: f32, py: f32, depth: f32, half_res: vec2<f32>) -> vec3<f32> {
    let scale = surface_params.pixel_world_scale * depth;
    return vec3<f32>((px - half_res.x) * scale, (py - half_res.y) * scale, depth);
}

/// The `van der Laan` screen-space smooth + normal half of the surface path. One
/// invocation per screen pixel. It bilaterally smooths the splatted fluid depth
/// (edge-preserving: a range weight on the depth difference stops bleeding
/// across silhouettes), reconstructs the view-space normal from the smoothed
/// depth's finite differences, and writes `(normal.xyz, smoothed_depth)` to the
/// output storage texture. Background pixels (no splatted fluid) are written as
/// a flat, far-depth default.
@compute @workgroup_size(8, 8, 1)
fn water_surface_reconstruct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = surface_params.resolution.x;
    let height = surface_params.resolution.y;
    if gid.x >= width || gid.y >= height {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let center_idx = gid.y * width + gid.x;
    let center_depth = surface_depth[center_idx];
    // Background: no fluid was splatted onto this pixel.
    if center_depth >= RECON_DEPTH_FAR {
        textureStore(surface_normal_tex, coord, vec4<f32>(0.0, 0.0, 1.0, RECON_DEPTH_FAR));
        return;
    }

    let radius = clamp(surface_params.filter_radius, 0, 8);
    let spatial = max(surface_params.spatial_sigma2, RECON_EPS);
    let range = max(surface_params.range_sigma2, RECON_EPS);
    var depth_sum = 0.0;
    var weight_sum = 0.0;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let sx = coord.x + dx;
            let sy = coord.y + dy;
            if sx < 0 || sy < 0 || sx >= i32(width) || sy >= i32(height) {
                continue;
            }
            let sample_depth = surface_depth[u32(sy) * width + u32(sx)];
            if sample_depth >= RECON_DEPTH_FAR {
                continue;
            }
            let spatial_d = f32(dx * dx + dy * dy);
            let range_d = sample_depth - center_depth;
            let sw = exp(-spatial_d / spatial) * exp(-(range_d * range_d) / range);
            depth_sum = depth_sum + sample_depth * sw;
            weight_sum = weight_sum + sw;
        }
    }
    var smoothed = center_depth;
    if weight_sum > RECON_EPS {
        smoothed = depth_sum / weight_sum;
    }

    // Normal from the smoothed depth: reconstruct neighboring view-space
    // positions and cross their screen-space derivatives (van der Laan).
    let half_res = vec2<f32>(f32(width) * 0.5, f32(height) * 0.5);
    let cx = f32(coord.x) + 0.5;
    let cy = f32(coord.y) + 0.5;
    let center_pos = recon_view_position(cx, cy, smoothed, half_res);

    var depth_r = smoothed;
    if coord.x + 1 < i32(width) {
        let d = surface_depth[center_idx + 1u];
        if d < RECON_DEPTH_FAR {
            depth_r = d;
        }
    }
    var depth_u = smoothed;
    if coord.y + 1 < i32(height) {
        let d = surface_depth[(gid.y + 1u) * width + gid.x];
        if d < RECON_DEPTH_FAR {
            depth_u = d;
        }
    }
    let pos_r = recon_view_position(cx + 1.0, cy, depth_r, half_res);
    let pos_u = recon_view_position(cx, cy + 1.0, depth_u, half_res);
    let ddx = pos_r - center_pos;
    let ddy = pos_u - center_pos;
    var normal = cross(ddx, ddy);
    let len_sq = dot(normal, normal);
    if len_sq > RECON_EPS {
        normal = normal / sqrt(len_sq);
    } else {
        normal = vec3<f32>(0.0, 0.0, 1.0);
    }
    // Face the camera (view-space +z toward the eye).
    if normal.z < 0.0 {
        normal = -normal;
    }
    // Coverage/thickness is available for downstream shading; the reconstructed
    // surface texel stores the unit normal and the smoothed depth.
    let coverage = max(surface_thickness[center_idx], 0.0);
    let depth_out = select(smoothed, smoothed, coverage >= 0.0);
    textureStore(surface_normal_tex, coord, vec4<f32>(normal, depth_out));
}
