// World-space GI — per-pixel probe interpolation (pass 2 of 2).
//
// One invocation per pixel. Reconstructs the shading point's view-space
// normal and depth, gathers the four surrounding screen probes, resolves the
// bilinear x geometric weights (golden `resolve_weights`), blends their SH
// (golden `blend_sh`) and evaluates the clamped-cosine irradiance (golden
// `evaluate_irradiance`) into the GI export buffer. Arm-for-arm with the CPU
// golden `prism_render_shading::gi::world_space::probe_interpolation`.

const K0: f32 = 0.282094800;
const K1: f32 = 0.488602500;
// Cosine-lobe convolution band factors (golden `A0`/`A1`).
const A0: f32 = 3.141592700;
const A1: f32 = 2.094395200;

struct ResolveParams {
    // Inverse projection (clip -> view), column-major.
    view_from_clip: mat4x4<f32>,
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<f32>,
    // Probe-grid dimensions in probes.
    probe_grid: vec2<u32>,
    // Tile size in pixels per probe.
    tile: u32,
    // Minimum dot(n_probe, n_point) for acceptance.
    normal_threshold: f32,
    // Maximum relative depth difference for acceptance.
    depth_rel_threshold: f32,
    // Artistic gain applied to the resolved irradiance.
    intensity: f32,
}

// One stored screen probe (see the probe-update pass for the layout).
struct Probe {
    sh0: vec4<f32>,
    sh1: vec4<f32>,
    sh2: vec4<f32>,
    sh3: vec4<f32>,
    normal_depth: vec4<f32>,
    flags: vec4<f32>,
}

@group(0) @binding(0) var scene_depth: texture_2d<f32>;
@group(0) @binding(1) var normal_roughness: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> probes: array<Probe>;
@group(0) @binding(3) var gi_out: texture_storage_2d<rgba16float, write>;

var<immediate> params: ResolveParams;

fn normalize_or_up(dir: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(dir, dir);
    if (len_sq > 1.175494e-38) {
        return dir * inverseSqrt(len_sq);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

fn sh_basis_l1(dir: vec3<f32>) -> vec4<f32> {
    let d = normalize_or_up(dir);
    return vec4<f32>(K0, K1 * d.y, K1 * d.z, K1 * d.x);
}

fn reconstruct_view_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = params.view_from_clip * clip;
    return view.xyz / view.w;
}

// Bilinear weights for the 2x2 quad in [(0,0),(1,0),(0,1),(1,1)] order
// (golden `bilinear_weights`). `frac` is clamped to [0, 1].
fn bilinear_weights(frac_x: f32, frac_y: f32) -> vec4<f32> {
    let fx = clamp(frac_x, 0.0, 1.0);
    let fy = clamp(frac_y, 0.0, 1.0);
    return vec4<f32>(
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy,
    );
}

// Splits a continuous probe coordinate into a clamped integer base and the
// fraction toward the next probe (golden `clamp_base`).
fn clamp_base(coord: f32, dim: u32) -> vec2<f32> {
    if (dim == 0u) {
        return vec2<f32>(0.0, 0.0);
    }
    if (dim == 1u) {
        return vec2<f32>(0.0, 0.0);
    }
    let max_base = f32(dim - 2u);
    let fl = floor(coord);
    if (fl < 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    if (fl >= max_base) {
        let frac = clamp(coord - max_base, 0.0, 1.0);
        return vec2<f32>(max_base, frac);
    }
    return vec2<f32>(fl, coord - fl);
}

// Geometric acceptance weight (0 or 1) for a single probe (golden
// `similarity_weight`).
fn similarity_weight(
    valid: bool,
    probe_normal: vec3<f32>,
    probe_depth: f32,
    point_normal: vec3<f32>,
    point_depth: f32,
) -> f32 {
    if (!valid) {
        return 0.0;
    }
    let n_dot = dot(probe_normal, point_normal);
    if (n_dot < params.normal_threshold) {
        return 0.0;
    }
    let denom = max(abs(point_depth), 1e-4);
    let depth_rel = abs(probe_depth - point_depth) / denom;
    if (depth_rel > params.depth_rel_threshold) {
        return 0.0;
    }
    return 1.0;
}

@compute @workgroup_size(8, 8, 1)
fn resolve_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(gi_out);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    let pixel = vec2<i32>(i32(global_id.x), i32(global_id.y));

    let device_depth = textureLoad(scene_depth, pixel, 0).x;
    // Reverse-Z background/sky: no GI contribution.
    if (device_depth <= 0.0) {
        textureStore(gi_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let uv = (vec2<f32>(f32(global_id.x), f32(global_id.y)) + vec2<f32>(0.5, 0.5)) / params.screen_size;
    let point_normal = normalize(textureLoad(normal_roughness, pixel, 0).xyz * 2.0 - 1.0);
    let point_view = reconstruct_view_position(uv, device_depth);
    let point_depth = -point_view.z;

    // Base probe coord + fractional position in the 2x2 quad (golden
    // `probe_bilinear_coords`), evaluated at the pixel centre.
    let tile = f32(max(params.tile, 1u));
    let coord_x = (f32(global_id.x) + 0.5) / tile - 0.5;
    let coord_y = (f32(global_id.y) + 0.5) / tile - 0.5;
    let bx = clamp_base(coord_x, params.probe_grid.x);
    let by = clamp_base(coord_y, params.probe_grid.y);
    let base_x = u32(bx.x);
    let base_y = u32(by.x);
    let frac_x = bx.y;
    let frac_y = by.y;

    // Four neighbours: TL, TR, BL, BR (row-major grid order).
    let coords = array<vec2<u32>, 4>(
        vec2<u32>(base_x, base_y),
        vec2<u32>(base_x + 1u, base_y),
        vec2<u32>(base_x, base_y + 1u),
        vec2<u32>(base_x + 1u, base_y + 1u),
    );

    let bw = bilinear_weights(frac_x, frac_y);
    var combined = vec4<f32>(0.0);
    var sum = 0.0;
    // Cache the loaded neighbour SH so blend/fallback reuse it.
    var n_sh0 = array<vec3<f32>, 4>();
    var n_sh1 = array<vec3<f32>, 4>();
    var n_sh2 = array<vec3<f32>, 4>();
    var n_sh3 = array<vec3<f32>, 4>();
    var n_valid = array<bool, 4>(false, false, false, false);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let c = coords[i];
        let in_range = c.x < params.probe_grid.x && c.y < params.probe_grid.y;
        var s = 0.0;
        if (in_range) {
            let idx = c.y * params.probe_grid.x + c.x;
            let probe = probes[idx];
            let valid = probe.flags.x > 0.5;
            n_valid[i] = valid;
            n_sh0[i] = probe.sh0.xyz;
            n_sh1[i] = probe.sh1.xyz;
            n_sh2[i] = probe.sh2.xyz;
            n_sh3[i] = probe.sh3.xyz;
            s = similarity_weight(valid, probe.normal_depth.xyz, probe.normal_depth.w, point_normal, point_depth);
        }
        let w = bw[i] * s;
        combined[i] = w;
        sum = sum + w;
    }

    // Confidence = pre-normalisation summed weight (matches the golden's
    // renormalise-then-blend behaviour; 0 when only the fallback fired).
    let confidence = sum;

    var weights = vec4<f32>(0.0);
    if (sum > 1.175494e-38) {
        weights = combined * (1.0 / sum);
    } else {
        // Fallback: largest bilinear weight preferring a valid probe (golden
        // `resolve_weights` fallback).
        var best = -1;
        var best_w = -1.0;
        for (var i = 0; i < 4; i = i + 1) {
            if (n_valid[i] && bw[i] > best_w) {
                best_w = bw[i];
                best = i;
            }
        }
        if (best < 0) {
            for (var i = 0; i < 4; i = i + 1) {
                if (bw[i] > best_w) {
                    best_w = bw[i];
                    best = i;
                }
            }
        }
        if (best >= 0) {
            weights[best] = 1.0;
        }
    }

    // Blend the neighbour SH (golden `blend_sh`).
    var sh0 = vec3<f32>(0.0);
    var sh1 = vec3<f32>(0.0);
    var sh2 = vec3<f32>(0.0);
    var sh3 = vec3<f32>(0.0);
    for (var i = 0; i < 4; i = i + 1) {
        let w = weights[i];
        if (w != 0.0) {
            sh0 = sh0 + n_sh0[i] * w;
            sh1 = sh1 + n_sh1[i] * w;
            sh2 = sh2 + n_sh2[i] * w;
            sh3 = sh3 + n_sh3[i] * w;
        }
    }

    // Evaluate the clamped-cosine irradiance along the point normal (golden
    // `evaluate_irradiance`): sum coeff * band * basis, clamped non-negative.
    let basis = sh_basis_l1(point_normal);
    var irradiance = sh0 * (A0 * basis.x)
        + sh1 * (A1 * basis.y)
        + sh2 * (A1 * basis.z)
        + sh3 * (A1 * basis.w);
    irradiance = max(irradiance, vec3<f32>(0.0));
    irradiance = irradiance * params.intensity;

    textureStore(gi_out, pixel, vec4<f32>(irradiance, confidence));
}
