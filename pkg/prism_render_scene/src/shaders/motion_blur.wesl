// GPU twin of `prism_render_shading::motion_blur` (plausible motion-blur
// reconstruction after McGuire et al. 2012: TileMax / NeighborMax velocity
// dilation, soft depth classification and the cone/cylinder reconstruction
// weight). Self-contained (no intra-crate imports, matching exposure.wesl /
// bloom.wesl) so the WESL compile test validates it standalone. Every function
// mirrors the CPU golden arm-for-arm; smoothstep is hand-written as
// `3t^2 - 2t^3` to stay bit-identical with the golden rather than the WGSL
// builtin, and depths are linear view distances (larger = farther).

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

// TileMax over a fixed 9-tap tile: the velocity with the greatest magnitude
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

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real pass wires the velocity/depth targets and the
// tile buffers later.
@compute @workgroup_size(64, 1, 1)
fn motion_blur_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    var tiles: array<vec2<f32>, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        tiles[i] = vec2<f32>(f32(gid.x) + f32(i), f32(i));
    }

    let vel = vec2<f32>(3.0, 4.0);
    let len = mb_velocity_length(vel);
    let shut = mb_shutter_velocity(vel, 0.5);
    let clamped = mb_clamp_velocity(vel, 2.5);
    let nmax = mb_neighbor_max(tiles);
    let sdc = mb_soft_depth_compare(1.0, 5.0, 1.0);
    let c = mb_cone(2.0, 4.0);
    let cy = mb_cylinder(1.0, 10.0);
    let w = mb_sample_weight(3.0, 3.0, 1.0, 10.0, 10.0, 1.0);

    _ = len + shut.x + clamped.y + nmax.x + sdc + c + cy + w;
}
