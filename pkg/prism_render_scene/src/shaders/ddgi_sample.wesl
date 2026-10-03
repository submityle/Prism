// DDGI (dynamic diffuse global illumination) — per-pixel irradiance-volume
// sample kernel.
//
// One invocation per framebuffer pixel. Reconstructs the shading point's
// world-space position (from the depth buffer and the clip->world matrix) and
// its world-space normal (from the normal G-buffer), locates the enclosing
// probe cell, and blends the eight surrounding octahedral irradiance probes
// with the combined trilinear x back-face x Chebyshev-visibility weights, then
// writes the clamped-cosine diffuse GI irradiance into the export buffer.
//
// Arm-for-arm with the CPU golden
// `prism_render_shading::gi::irradiance_volume` (Majercik et al. 2019 / RTXGI):
//   * `oct_to_dir`/`dir_to_oct` match `gi::world_space::octahedral`.
//   * `oct_bilinear_irradiance` matches `IrradianceOct::sample`.
//   * `oct_bilinear_depth` matches `DdgiDepthOct::sample`.
//   * `chebyshev_weight` matches `gi::world_space::visibility::chebyshev_weight`.
//   * `trilinear_weights`/`normal_backface_weight`/`interpolation_weights`
//     match `gi::irradiance_volume::relocation`.
//
// The CPU-mirror parity tests in `shading/ddgi/shader_tests.rs` transcribe
// these kernels op-for-op and assert agreement with the golden to 1e-6.

// Std430/uniform twin of `GpuDdgiVolume` (see `shading/ddgi/abi.rs`).
struct DdgiVolume {
    // World-space position of probe coordinate (0, 0, 0).
    origin: vec3<f32>,
    // Interior octahedral resolution of each irradiance probe.
    irradiance_interior: u32,
    // World-space spacing between adjacent probes along each axis.
    spacing: vec3<f32>,
    // Interior octahedral resolution of each depth / visibility probe.
    depth_interior: u32,
    // Probe counts along each axis (each component is >= 1).
    counts: vec3<i32>,
    // Padding to the 16-byte boundary (mirrors the Rust twin).
    pad_counts: u32,
    // World-space normal bias pulling the shading point along its normal.
    normal_bias: f32,
    // World-space self-shadow bias subtracted before the Chebyshev test.
    view_bias: f32,
    // Temporal hysteresis (unused at sample time; kept for ABI parity).
    hysteresis: f32,
    // Depth cosine exponent (unused at sample time; kept for ABI parity).
    depth_sharpness: f32,
    // Artistic gain baked into the resolved irradiance.
    intensity: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

// One stored probe's relocation offset + activity flag.
struct ProbeMeta {
    // World-space relocation offset added to the probe's base position.
    offset: vec3<f32>,
    // Non-zero when the probe is active; inactive probes receive zero weight.
    // (`active` is a reserved WGSL/WESL word, so the field is `active_flag`.)
    active_flag: f32,
}

// Immediate (push-constant) twin of `GpuDdgiSampleParams`.
struct SampleParams {
    // Clip -> world transform used to reconstruct world position from depth.
    clip_to_world: mat4x4<f32>,
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<f32>,
    // Positive near-plane distance (reserved for depth linearisation).
    near: f32,
    // Artistic gain applied on top of the volume intensity.
    intensity: f32,
}

@group(0) @binding(0) var scene_depth: texture_2d<f32>;
@group(0) @binding(1) var normal_gbuffer: texture_2d<f32>;
@group(0) @binding(2) var<uniform> volume: DdgiVolume;
@group(0) @binding(3) var<storage, read> probe_meta: array<ProbeMeta>;
@group(0) @binding(4) var irradiance_atlas: texture_2d<f32>;
@group(0) @binding(5) var depth_atlas: texture_2d<f32>;
@group(0) @binding(6) var gi_out: texture_storage_2d<rgba16float, write>;

var<immediate> params: SampleParams;

// --- Octahedral map (golden `gi::world_space::octahedral`) -------------------

// Branchless sign that maps zero to +1 (golden `sign_not_zero`).
fn sign_not_zero(value: f32) -> f32 {
    return select(-1.0, 1.0, value >= 0.0);
}

// Encode a direction into octahedral UV in [0, 1]^2 (golden `dir_to_oct`).
fn dir_to_oct(dir: vec3<f32>) -> vec2<f32> {
    let l1 = abs(dir.x) + abs(dir.y) + abs(dir.z);
    if (l1 <= 1.175494e-38) {
        return vec2<f32>(0.5, 0.5);
    }
    let inv = 1.0 / l1;
    var p = vec2<f32>(dir.x * inv, dir.y * inv);
    if (dir.z < 0.0) {
        p = vec2<f32>(
            (1.0 - abs(p.y)) * sign_not_zero(p.x),
            (1.0 - abs(p.x)) * sign_not_zero(p.y),
        );
    }
    return p * 0.5 + vec2<f32>(0.5, 0.5);
}

// Normalise `dir`, falling back to +Y for a degenerate input.
fn normalize_or_up(dir: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(dir, dir);
    if (len_sq > 1.175494e-38) {
        return dir * inverseSqrt(len_sq);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

// Clamp a floored, UV-scaled coordinate into 0..=last (golden `clamp_padded`).
fn clamp_padded(value: f32, last: i32) -> i32 {
    if (value <= 0.0) {
        return 0;
    }
    let v = i32(value);
    if (v > last) {
        return last;
    }
    return v;
}

// --- Probe atlas addressing --------------------------------------------------

// Flat probe index for a lattice coordinate (x fastest, then y, then z).
fn probe_linear_index(coord: vec3<i32>) -> u32 {
    let cx = volume.counts.x;
    let cy = volume.counts.y;
    let p = (coord.z * cy + coord.y) * cx + coord.x;
    return u32(max(p, 0));
}

// Top-left texel of probe `p`'s padded tile inside a tiled 2D atlas whose
// width holds `atlas_width / padded` tiles per row.
fn probe_tile_origin(p: u32, padded: u32, atlas_width: u32) -> vec2<u32> {
    let per_row = max(atlas_width / padded, 1u);
    let tx = p % per_row;
    let ty = p / per_row;
    return vec2<u32>(tx * padded, ty * padded);
}

// Clamp a lattice coordinate into the valid 0..counts range per axis
// (golden `ProbeGrid::clamp_coord`).
fn clamp_coord(coord: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(
        clamp(coord.x, 0, volume.counts.x - 1),
        clamp(coord.y, 0, volume.counts.y - 1),
        clamp(coord.z, 0, volume.counts.z - 1),
    );
}

// Bilinearly sample probe `p`'s octahedral irradiance along `normal`
// (golden `IrradianceOct::sample`).
fn oct_bilinear_irradiance(p: u32, normal: vec3<f32>) -> vec3<f32> {
    let interior = volume.irradiance_interior;
    let padded = interior + 2u;
    let dims = textureDimensions(irradiance_atlas);
    let base = probe_tile_origin(p, padded, dims.x);

    let oct = dir_to_oct(normal);
    let n = f32(interior);
    let fx = oct.x * n + 0.5;
    let fy = oct.y * n + 0.5;
    let x0f = floor(fx);
    let y0f = floor(fy);
    let tx = clamp(fx - x0f, 0.0, 1.0);
    let ty = clamp(fy - y0f, 0.0, 1.0);
    let last = i32(padded) - 1;
    let x0 = clamp_padded(x0f, last);
    let y0 = clamp_padded(y0f, last);
    let x1 = clamp_padded(x0f + 1.0, last);
    let y1 = clamp_padded(y0f + 1.0, last);

    let c00 = textureLoad(irradiance_atlas, vec2<i32>(base) + vec2<i32>(x0, y0), 0).rgb;
    let c10 = textureLoad(irradiance_atlas, vec2<i32>(base) + vec2<i32>(x1, y0), 0).rgb;
    let c01 = textureLoad(irradiance_atlas, vec2<i32>(base) + vec2<i32>(x0, y1), 0).rgb;
    let c11 = textureLoad(irradiance_atlas, vec2<i32>(base) + vec2<i32>(x1, y1), 0).rgb;
    let top = c00 + (c10 - c00) * tx;
    let bottom = c01 + (c11 - c01) * tx;
    let out = top + (bottom - top) * ty;
    return max(out, vec3<f32>(0.0, 0.0, 0.0));
}

// Bilinearly sample probe `p`'s octahedral [mean, mean_sq] depth moments along
// `dir` (golden `DdgiDepthOct::sample`).
fn oct_bilinear_depth(p: u32, dir: vec3<f32>) -> vec2<f32> {
    let interior = volume.depth_interior;
    let padded = interior + 2u;
    let dims = textureDimensions(depth_atlas);
    let base = probe_tile_origin(p, padded, dims.x);

    let oct = dir_to_oct(dir);
    let n = f32(interior);
    let fx = oct.x * n + 0.5;
    let fy = oct.y * n + 0.5;
    let x0f = floor(fx);
    let y0f = floor(fy);
    let tx = clamp(fx - x0f, 0.0, 1.0);
    let ty = clamp(fy - y0f, 0.0, 1.0);
    let last = i32(padded) - 1;
    let x0 = clamp_padded(x0f, last);
    let y0 = clamp_padded(y0f, last);
    let x1 = clamp_padded(x0f + 1.0, last);
    let y1 = clamp_padded(y0f + 1.0, last);

    let m00 = textureLoad(depth_atlas, vec2<i32>(base) + vec2<i32>(x0, y0), 0).rg;
    let m10 = textureLoad(depth_atlas, vec2<i32>(base) + vec2<i32>(x1, y0), 0).rg;
    let m01 = textureLoad(depth_atlas, vec2<i32>(base) + vec2<i32>(x0, y1), 0).rg;
    let m11 = textureLoad(depth_atlas, vec2<i32>(base) + vec2<i32>(x1, y1), 0).rg;
    let top = m00 + (m10 - m00) * tx;
    let bottom = m01 + (m11 - m01) * tx;
    let out = top + (bottom - top) * ty;
    return max(out, vec2<f32>(0.0, 0.0));
}

// --- Interpolation weights (golden `relocation`) -----------------------------

// Chebyshev / Cantelli variance visibility weight (golden `chebyshev_weight`).
fn chebyshev_weight(mean: f32, mean_sq: f32, distance: f32) -> f32 {
    if (distance <= mean) {
        return 1.0;
    }
    let variance = max(mean_sq - mean * mean, 0.0);
    let delta = distance - mean;
    let weight = variance / (variance + delta * delta);
    return clamp(weight, 0.0, 1.0);
}

// Smooth DDGI "wrap" back-face weight (golden `normal_backface_weight`).
fn normal_backface_weight(point: vec3<f32>, normal: vec3<f32>, probe_position: vec3<f32>) -> f32 {
    let to_probe = probe_position - point;
    let len_sq = dot(to_probe, to_probe);
    let n_len_sq = dot(normal, normal);
    if (len_sq <= 1.175494e-38 || n_len_sq <= 1.175494e-38) {
        return 1.0;
    }
    let dir = to_probe * inverseSqrt(len_sq);
    let n = normal * inverseSqrt(n_len_sq);
    let wrap = max(0.5 + 0.5 * dot(dir, n), 0.0);
    return wrap * wrap;
}

// World-space base position of a probe lattice coordinate (golden
// `ProbeGrid::probe_base_position`).
fn probe_base_position(coord: vec3<i32>) -> vec3<f32> {
    return volume.origin + vec3<f32>(coord) * volume.spacing;
}

// --- Entry point -------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn sample_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.xy;
    let dims = vec2<u32>(u32(params.screen_size.x), u32(params.screen_size.y));
    if (px.x >= dims.x || px.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(px);
    let depth = textureLoad(scene_depth, coord, 0).r;
    // Reconstruct world-space position from the depth sample.
    let uv = (vec2<f32>(px) + vec2<f32>(0.5, 0.5)) / params.screen_size;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let clip = vec4<f32>(ndc, depth, 1.0);
    let world_h = params.clip_to_world * clip;
    let point = world_h.xyz / world_h.w;

    let normal = normalize_or_up(textureLoad(normal_gbuffer, coord, 0).xyz * 2.0 - 1.0);
    let biased_point = point + normal * volume.normal_bias;

    // Locate the enclosing cell (golden `ProbeGrid::world_to_cell`).
    let inv = vec3<f32>(1.0, 1.0, 1.0) / volume.spacing;
    let g = (biased_point - volume.origin) * inv;
    var base = vec3<i32>(0, 0, 0);
    var frac = vec3<f32>(0.0, 0.0, 0.0);
    for (var a = 0; a < 3; a = a + 1) {
        let max_base = max(volume.counts[a] - 1, 0);
        let gi = g[a];
        let fi = floor(gi);
        let bi = clamp(i32(fi), 0, max(max_base - 1, 0));
        base[a] = bi;
        frac[a] = clamp(gi - f32(bi), 0.0, 1.0);
    }

    let bias = max(volume.view_bias, 0.0);
    let fx = clamp(frac.x, 0.0, 1.0);
    let fy = clamp(frac.y, 0.0, 1.0);
    let fz = clamp(frac.z, 0.0, 1.0);

    var irradiance = vec3<f32>(0.0, 0.0, 0.0);
    var weight_sum = 0.0;
    var combined: array<f32, 8>;

    for (var c = 0; c < 8; c = c + 1) {
        let dx = c & 1;
        let dy = (c >> 1) & 1;
        let dz = (c >> 2) & 1;
        let corner = clamp_coord(base + vec3<i32>(dx, dy, dz));
        let p = probe_linear_index(corner);
        let meta_entry = probe_meta[p];
        let probe_pos = probe_base_position(corner) + meta_entry.offset;

        var w = 0.0;
        if (meta_entry.active_flag != 0.0) {
            let wx = select(1.0 - fx, fx, dx == 1);
            let wy = select(1.0 - fy, fy, dy == 1);
            let wz = select(1.0 - fz, fz, dz == 1);
            let tri = wx * wy * wz;
            let n_w = normal_backface_weight(biased_point, normal, probe_pos);
            let to_point = biased_point - probe_pos;
            let dist = max(length(to_point) - bias, 0.0);
            let moments = oct_bilinear_depth(p, normalize_or_up(to_point));
            let d_w = chebyshev_weight(moments.x, moments.y, dist);
            w = max(tri * n_w * d_w, 0.0);
        }
        combined[c] = w;
        weight_sum = weight_sum + w;
    }

    if (weight_sum > 1.175494e-38) {
        let inv_sum = 1.0 / weight_sum;
        for (var c = 0; c < 8; c = c + 1) {
            let w = combined[c] * inv_sum;
            if (w <= 0.0) {
                continue;
            }
            let dx = c & 1;
            let dy = (c >> 1) & 1;
            let dz = (c >> 2) & 1;
            let corner = clamp_coord(base + vec3<i32>(dx, dy, dz));
            let p = probe_linear_index(corner);
            irradiance = irradiance + oct_bilinear_irradiance(p, normal) * w;
        }
    }

    let gain = volume.intensity * params.intensity;
    textureStore(gi_out, coord, vec4<f32>(irradiance * gain, 1.0));
}
