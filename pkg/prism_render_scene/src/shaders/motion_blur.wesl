// GPU twin of `prism_render_shading::motion_blur` (plausible motion-blur
// reconstruction after McGuire et al. 2012: TileMax / NeighborMax velocity
// dilation, soft depth classification and the cone/cylinder reconstruction
// weight). Self-contained (no intra-crate imports, matching exposure.wesl /
// bloom.wesl) so the WESL compile test validates it standalone. Every helper
// mirrors the CPU golden arm-for-arm; smoothstep is hand-written as
// `3t^2 - 2t^3` to stay bit-identical with the golden rather than the WGSL
// builtin, and depths are linear view distances (larger = farther).
//
// The module carries three compute entry points wired as a chain of passes:
//
//   * `tile_max`     reduces every TILE_SIZE x TILE_SIZE screen tile to its
//                    longest *shuttered, clamped* pixel-space velocity.
//   * `neighbor_max` dilates that tile field by one ring (the 3x3 NeighborMax)
//                    so a fast tile bleeds into the neighbours it smears across.
//   * `reconstruct`  gathers `sample_count` taps along the dilated dominant
//                    velocity and weights each with the golden cone/cylinder /
//                    soft-depth term, writing the blurred HDR colour.
//
// Motion vectors are the resolved `cur_uv - prev_uv` G-buffer (UV space, both
// camera and object motion); this shader converts them to pixels via the
// framebuffer extent. Depth is the SSR geometry prepass's reverse-Z device
// depth, unprojected to a linear view distance with `view_from_clip` exactly
// like `ssr_resolve.wesl`, so the golden's `soft_depth_compare` sees the same
// view-space distances the CPU reference was validated against.

// Immediate (push-constant) block. Mirrors `MotionBlurParams` in `abi.rs`
// (112 bytes: a `mat4x4<f32>` inverse projection then the extent, tile grid,
// tile size, sample count and the golden tunables, three trailing `u32` pads to
// the 16-byte alignment the matrix forces on the struct).
struct MotionBlurParams {
    // Clip -> view (inverse projection); reconstructs linear view-space depth
    // from the reverse-Z device depth so the soft-depth term is scale-correct.
    view_from_clip: mat4x4<f32>,
    // Full-resolution framebuffer extent in texels.
    width: u32,
    height: u32,
    // Tile-grid extent: `ceil(width / tile_size)` x `ceil(height / tile_size)`.
    tiles_x: u32,
    tiles_y: u32,
    // Edge length (texels) of a TileMax / NeighborMax tile.
    tile_size: u32,
    // Reconstruction gather taps per pixel.
    sample_count: u32,
    // Blur budget in pixels; velocities are clamped to this to bound the gather.
    max_velocity_px: f32,
    // Depth softness (view-space units) fed to `mb_soft_depth_compare`.
    soft_z_extent: f32,
    // Fraction of the frame the shutter is open, fed to `mb_shutter_velocity`.
    exposure_fraction: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// This frame's resolved motion-vector G-buffer: `cur_uv - prev_uv` in [0,1] UV,
// capturing camera *and* per-object motion. Read by `tile_max` (per pixel) and
// by `reconstruct` (centre + each tap).
@group(0) @binding(0) var motion_in: texture_2d<f32>;
// TileMax output: the longest shuttered, clamped pixel-space velocity of each
// tile (rg16float, one texel per tile). Written by `tile_max`.
@group(0) @binding(1) var tile_max_out: texture_storage_2d<rg16float, write>;
// TileMax field read by `neighbor_max` (one texel per tile).
@group(0) @binding(2) var tile_in: texture_2d<f32>;
// NeighborMax output: the 3x3-dilated dominant velocity per tile (rg16float).
@group(0) @binding(3) var neighbor_out: texture_storage_2d<rg16float, write>;
// Composited HDR scene colour, the reconstruction's source radiance.
@group(0) @binding(4) var scene_color_in: texture_2d<f32>;
// NeighborMax field read by `reconstruct` (dominant velocity per tile).
@group(0) @binding(5) var neighbor_in: texture_2d<f32>;
// Reverse-Z device depth (R32Float, 0 = background) from the SSR geometry
// prepass; unprojected to a linear view distance for the soft-depth term.
@group(0) @binding(6) var scene_depth: texture_2d<f32>;
// Full-resolution blurred output (rgba16float). Written by `reconstruct`.
@group(0) @binding(7) var blur_out: texture_storage_2d<rgba16float, write>;

var<immediate> params: MotionBlurParams;

// ---------------------------------------------------------------------------
// Golden twins (mirror `prism_render_shading::motion_blur` arm-for-arm).
// ---------------------------------------------------------------------------

// Euclidean length (pixels) of a 2D screen-space velocity vector.
fn mb_velocity_length(v: vec2<f32>) -> f32 {
    return sqrt(v.x * v.x + v.y * v.y);
}

// Scale a per-frame clip velocity by the open-shutter fraction of the frame.
fn mb_shutter_velocity(clip_velocity: vec2<f32>, exposure_fraction: f32) -> vec2<f32> {
    return clip_velocity * exposure_fraction;
}

// Clamp a velocity to a pixel budget, preserving direction; a non-positive
// budget collapses to zero and in-budget vectors pass through untouched.
fn mb_clamp_velocity(v: vec2<f32>, max_len: f32) -> vec2<f32> {
    let len = mb_velocity_length(v);
    if (len <= max_len) {
        return v;
    }
    let scale = max(max_len, 0.0) / len;
    return v * scale;
}

// NeighborMax over a fixed 9-tap tile: the velocity with the greatest magnitude
// (index 4 = centre). An all-zero tile is motionless.
fn mb_neighbor_max(tiles: array<vec2<f32>, 9>) -> vec2<f32> {
    var best = vec2<f32>(0.0, 0.0);
    var best_len = 0.0;
    for (var i = 0u; i < 9u; i = i + 1u) {
        let len = mb_velocity_length(tiles[i]);
        if (len > best_len) {
            best_len = len;
            best = tiles[i];
        }
    }
    return best;
}

// Hand-written smoothstep (Hermite 3t^2 - 2t^3); a degenerate interval is a
// hard step at edge0. Matches the CPU golden bit-for-bit.
fn mb_smooth_step(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge0 == edge1) {
        if (x < edge0) {
            return 0.0;
        }
        return 1.0;
    }
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Soft depth ordering: ~1 when za is in front of zb (smaller depth), ~0 behind,
// linear within `extent`; equal depths return 1.
fn mb_soft_depth_compare(za: f32, zb: f32, extent: f32) -> f32 {
    let e = max(extent, 1.0e-6);
    return clamp(1.0 - (za - zb) / e, 0.0, 1.0);
}

// Cone falloff: full at zero distance, linear to zero at `speed`; a static tap
// covers nothing.
fn mb_cone(dist: f32, speed: f32) -> f32 {
    if (speed <= 1.0e-6) {
        return 0.0;
    }
    return clamp(1.0 - dist / speed, 0.0, 1.0);
}

// Cylinder falloff for two overlapping moving samples.
fn mb_cylinder(dist: f32, speed: f32) -> f32 {
    return 1.0 - mb_smooth_step(0.95 * speed, 1.05 * speed, dist);
}

// McGuire reconstruction weight for one gather tap (cone foreground + cone
// background + doubled cylinder overlap).
fn mb_sample_weight(
    center_depth: f32,
    sample_depth: f32,
    dist: f32,
    center_velocity_len: f32,
    sample_velocity_len: f32,
    soft_z_extent: f32,
) -> f32 {
    let foreground = mb_soft_depth_compare(sample_depth, center_depth, soft_z_extent);
    let background = mb_soft_depth_compare(center_depth, sample_depth, soft_z_extent);
    var weight = foreground * mb_cone(dist, sample_velocity_len);
    weight = weight + background * mb_cone(dist, center_velocity_len);
    weight = weight + mb_cylinder(dist, sample_velocity_len) * mb_cylinder(dist, center_velocity_len) * 2.0;
    return weight;
}

// ---------------------------------------------------------------------------
// GPU-side plumbing shared by the entry points.
// ---------------------------------------------------------------------------

// Per-pixel screen-space velocity in *pixels*, already shuttered and clamped to
// the blur budget: read the UV displacement, scale to pixels, open the shutter
// (`exposure_fraction`) and clamp to `max_velocity_px`. This is the exact CPU
// golden pipeline (`shutter_velocity` then `clamp_velocity`) on the resolved
// motion vector, so both sides agree on the pixel-space velocity of every tap.
fn mb_pixel_velocity(pixel: vec2<i32>) -> vec2<f32> {
    let uv_velocity = textureLoad(motion_in, pixel, 0).xy;
    let pixel_velocity = uv_velocity * vec2<f32>(f32(params.width), f32(params.height));
    let shuttered = mb_shutter_velocity(pixel_velocity, params.exposure_fraction);
    return mb_clamp_velocity(shuttered, params.max_velocity_px);
}

// Reconstruct linear view-space distance (positive, larger = farther) from a
// pixel's reverse-Z device depth, mirroring `ssr_resolve.wesl::reconstruct_view_z`.
// The golden's `soft_depth_compare` expects a view *distance*, so the signed
// view-space z (negative down the -Z view axis) is returned as its magnitude.
fn mb_view_distance(pixel: vec2<i32>, device_depth: f32) -> f32 {
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5))
        / vec2<f32>(f32(params.width), f32(params.height));
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let view = params.view_from_clip * vec4<f32>(ndc, 1.0);
    return abs(view.z / view.w);
}

// Frame-independent interleaved-gradient dither (Jimenez 2014) in [0,1), used to
// jitter the gather taps so the finite sample count does not band.
fn mb_dither(pixel: vec2<i32>) -> f32 {
    let p = vec2<f32>(pixel);
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// ---------------------------------------------------------------------------
// Pass 1: TileMax. One invocation per tile; scans its TILE_SIZE x TILE_SIZE
// pixel footprint for the longest shuttered, clamped velocity.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8, 1)
fn tile_max(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.tiles_x || id.y >= params.tiles_y) {
        return;
    }
    let tile = vec2<i32>(i32(id.x), i32(id.y));
    let origin = vec2<i32>(tile.x * i32(params.tile_size), tile.y * i32(params.tile_size));

    var best = vec2<f32>(0.0, 0.0);
    var best_len = 0.0;
    for (var ty = 0u; ty < params.tile_size; ty = ty + 1u) {
        for (var tx = 0u; tx < params.tile_size; tx = tx + 1u) {
            let pixel = origin + vec2<i32>(i32(tx), i32(ty));
            if (pixel.x >= i32(params.width) || pixel.y >= i32(params.height)) {
                continue;
            }
            let v = mb_pixel_velocity(pixel);
            let len = mb_velocity_length(v);
            if (len > best_len) {
                best_len = len;
                best = v;
            }
        }
    }
    textureStore(tile_max_out, tile, vec4<f32>(best, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Pass 2: NeighborMax. Dilates the tile field by one ring (3x3), so a tile
// inherits the longest velocity of itself and its eight neighbours.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8, 1)
fn neighbor_max(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.tiles_x || id.y >= params.tiles_y) {
        return;
    }
    let tile = vec2<i32>(i32(id.x), i32(id.y));

    // Gather the 3x3 neighbourhood row-major (index 4 = centre), clamping tiles
    // off the grid edge to a zero (motionless) velocity.
    var tiles: array<vec2<f32>, 9>;
    var idx = 0u;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let n = tile + vec2<i32>(dx, dy);
            if (n.x < 0 || n.y < 0 || n.x >= i32(params.tiles_x) || n.y >= i32(params.tiles_y)) {
                tiles[idx] = vec2<f32>(0.0, 0.0);
            } else {
                tiles[idx] = textureLoad(tile_in, n, 0).xy;
            }
            idx = idx + 1u;
        }
    }

    textureStore(neighbor_out, tile, vec4<f32>(mb_neighbor_max(tiles), 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Pass 3: reconstruction. Scatter-as-gather along the dilated dominant velocity
// (McGuire 2012), each tap weighted by the golden cone/cylinder / soft-depth
// term, writing the blurred HDR colour.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8, 1)
fn reconstruct(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let pixel = vec2<i32>(i32(id.x), i32(id.y));
    let center_color = textureLoad(scene_color_in, pixel, 0);

    // Dominant tile velocity (dilated NeighborMax) drives the gather direction.
    let tile = vec2<i32>(pixel.x / i32(params.tile_size), pixel.y / i32(params.tile_size));
    let dominant = textureLoad(neighbor_in, tile, 0).xy;
    let dominant_speed = mb_velocity_length(dominant);

    // A tile that moved less than half a pixel leaves the frame sharp (this also
    // preserves TAA's sub-pixel work when the camera is still).
    if (dominant_speed < 0.5) {
        textureStore(blur_out, pixel, center_color);
        return;
    }

    let center_velocity = mb_pixel_velocity(pixel);
    let center_speed = mb_velocity_length(center_velocity);
    let center_device = textureLoad(scene_depth, pixel, 0).x;
    let center_depth = mb_view_distance(pixel, center_device);

    // Centre tap: always represented so a (near) static pixel inside a moving
    // neighbourhood keeps its own colour. McGuire weights the centre inversely
    // to its own speed, so a fast centre yields to the streak while a still one
    // dominates. Floored at one so the weight can never blow up.
    let center_weight = 1.0 / max(center_speed, 1.0);
    var accum = center_color.rgb * center_weight;
    var total_weight = center_weight;

    let noise = mb_dither(pixel) - 0.5;
    let samples = max(params.sample_count, 1u);

    for (var i = 0u; i < samples; i = i + 1u) {
        // Symmetric jittered parametric position in (-1, 1) along `dominant`.
        let t = (f32(i) + 0.5 + noise) / f32(samples);
        let offset = t * 2.0 - 1.0;
        // Gather spans +/- half the dominant velocity around the pixel.
        let tap_f = vec2<f32>(pixel) + dominant * (offset * 0.5);
        let tap = vec2<i32>(i32(round(tap_f.x)), i32(round(tap_f.y)));
        if (tap.x < 0 || tap.y < 0 || tap.x >= i32(params.width) || tap.y >= i32(params.height)) {
            continue;
        }
        // Skip the tap that lands back on the centre (already accounted for).
        if (tap.x == pixel.x && tap.y == pixel.y) {
            continue;
        }

        let tap_color = textureLoad(scene_color_in, tap, 0).rgb;
        let tap_velocity = mb_pixel_velocity(tap);
        let tap_speed = mb_velocity_length(tap_velocity);
        let tap_device = textureLoad(scene_depth, tap, 0).x;
        let tap_depth = mb_view_distance(tap, tap_device);

        let dist = mb_velocity_length(vec2<f32>(f32(tap.x - pixel.x), f32(tap.y - pixel.y)));
        let weight = mb_sample_weight(
            center_depth,
            tap_depth,
            dist,
            center_speed,
            tap_speed,
            params.soft_z_extent,
        );

        accum = accum + tap_color * weight;
        total_weight = total_weight + weight;
    }

    let result = accum / max(total_weight, 1.0e-5);
    textureStore(blur_out, pixel, vec4<f32>(result, center_color.a));
}
