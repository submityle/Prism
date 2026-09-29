// World-space GI — screen-probe radiance capture (pass 1 of 2).
//
// One invocation per screen probe. The probe owns a `tile` x `tile`
// framebuffer block; this pass reconstructs the probe centre's view-space
// position and normal, gathers the tile's pre-exposed radiance into an L1
// spherical-harmonic probe (arm-for-arm with the CPU golden
// `prism_render_shading::gi::world_space`) and writes it into the probe
// storage buffer for the resolve pass to interpolate.
//
// Math parity notes (all identical to the golden):
//   * SH basis `[K0, K1*y, K1*z, K1*x]`, band factors `[A0, A1, A1, A1]`.
//   * `add_directional_radiance` accumulates `radiance * basis * weight`.
//   * Octahedral snap `oct_to_dir(dir_to_oct(dir))` matches `octahedral.rs`.
//   * Probe placement (tile rect, centre pixel) matches `probe_placement.rs`.
// The golden functions are space-agnostic (dot products + depth compares), so
// running them on view-space normals and the `-view_pos.z` depth metric is
// arm-for-arm correct.

// L0/L1 real SH basis constants (mirror golden `K0`/`K1`).
const K0: f32 = 0.282094800;
const K1: f32 = 0.488602500;

struct ProbeParams {
    // Inverse projection (clip -> view), column-major.
    view_from_clip: mat4x4<f32>,
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<f32>,
    // Probe-grid dimensions in probes.
    probe_grid: vec2<u32>,
    // Tile size in pixels per probe.
    tile: u32,
    // World-space radiance-cache cell size (forwarded for parity).
    cell_size: f32,
    // Artistic gain baked into the captured probe radiance.
    intensity: f32,
    // Positive near-plane distance along -Z.
    near: f32,
}

// One stored screen probe. Six vec4 = 96 bytes, matching the Rust
// `PROBE_STRIDE` and the std430 array stride.
struct Probe {
    // Four L1 SH coefficients (rgb in xyz, w unused).
    sh0: vec4<f32>,
    sh1: vec4<f32>,
    sh2: vec4<f32>,
    sh3: vec4<f32>,
    // xyz = view-space capture normal, w = view-space capture depth (-z).
    normal_depth: vec4<f32>,
    // x = valid flag (1.0 / 0.0); yzw reserved.
    flags: vec4<f32>,
}

@group(0) @binding(0) var scene_depth: texture_2d<f32>;
@group(0) @binding(1) var normal_roughness: texture_2d<f32>;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> probes: array<Probe>;

var<immediate> params: ProbeParams;

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

// Decode an octahedral UV back to a unit direction (golden `oct_to_dir`).
fn oct_to_dir(uv: vec2<f32>) -> vec3<f32> {
    let e = uv * 2.0 - vec2<f32>(1.0, 1.0);
    let z = 1.0 - abs(e.x) - abs(e.y);
    var x = e.x;
    var y = e.y;
    if (z < 0.0) {
        x = (1.0 - abs(e.y)) * sign_not_zero(e.x);
        y = (1.0 - abs(e.x)) * sign_not_zero(e.y);
    }
    return normalize(vec3<f32>(x, y, z));
}

// Normalise `dir`, falling back to +Y for a degenerate input (golden
// `normalize_or_up`).
fn normalize_or_up(dir: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(dir, dir);
    if (len_sq > 1.175494e-38) {
        return dir * inverseSqrt(len_sq);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

// Four L0+L1 SH basis functions in golden `[K0, K1*y, K1*z, K1*x]` order.
fn sh_basis_l1(dir: vec3<f32>) -> vec4<f32> {
    let d = normalize_or_up(dir);
    return vec4<f32>(K0, K1 * d.y, K1 * d.z, K1 * d.x);
}

// Reconstruct a view-space position from a UV + reverse-Z device depth via the
// inverse projection (matches `ssgi.wesl` / golden).
fn reconstruct_view_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = params.view_from_clip * clip;
    return view.xyz / view.w;
}

@compute @workgroup_size(8, 8, 1)
fn probe_update_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let probe = global_id.xy;
    if (probe.x >= params.probe_grid.x || probe.y >= params.probe_grid.y) {
        return;
    }
    let probe_index = probe.y * params.probe_grid.x + probe.x;

    // Tile rectangle owned by this probe, clamped to the viewport (golden
    // `probe_pixel_rect`).
    let tile = max(params.tile, 1u);
    let screen = vec2<u32>(u32(params.screen_size.x), u32(params.screen_size.y));
    let min_pixel = vec2<u32>(probe.x * tile, probe.y * tile);
    let max_x = min((probe.x + 1u) * tile, screen.x);
    let max_y = min((probe.y + 1u) * tile, screen.y);

    // Degenerate (off-screen / empty) tile: store an invalid probe.
    if (max_x <= min_pixel.x || max_y <= min_pixel.y) {
        probes[probe_index].sh0 = vec4<f32>(0.0);
        probes[probe_index].sh1 = vec4<f32>(0.0);
        probes[probe_index].sh2 = vec4<f32>(0.0);
        probes[probe_index].sh3 = vec4<f32>(0.0);
        probes[probe_index].normal_depth = vec4<f32>(0.0);
        probes[probe_index].flags = vec4<f32>(0.0);
        return;
    }

    // Probe centre = centre of the clamped rectangle (golden `PixelRect::center`).
    let size_x = max_x - min_pixel.x;
    let size_y = max_y - min_pixel.y;
    let center = vec2<f32>(
        f32(min_pixel.x) + f32(size_x) * 0.5,
        f32(min_pixel.y) + f32(size_y) * 0.5,
    );
    let center_pixel = vec2<i32>(i32(center.x), i32(center.y));
    let center_uv = center / params.screen_size;

    let center_depth = textureLoad(scene_depth, center_pixel, 0).x;
    // Reverse-Z: depth <= 0 is background/sky — no valid probe surface.
    if (center_depth <= 0.0) {
        probes[probe_index].sh0 = vec4<f32>(0.0);
        probes[probe_index].sh1 = vec4<f32>(0.0);
        probes[probe_index].sh2 = vec4<f32>(0.0);
        probes[probe_index].sh3 = vec4<f32>(0.0);
        probes[probe_index].normal_depth = vec4<f32>(0.0);
        probes[probe_index].flags = vec4<f32>(0.0);
        return;
    }

    let center_normal = normalize(textureLoad(normal_roughness, center_pixel, 0).xyz * 2.0 - 1.0);
    let center_view = reconstruct_view_position(center_uv, center_depth);

    // Gather the tile's radiance into an L1 SH probe. Each covered pixel that
    // hit geometry contributes its scene colour along the direction from the
    // probe centre toward that sample, snapped to the octahedral grid.
    var sh0 = vec3<f32>(0.0);
    var sh1 = vec3<f32>(0.0);
    var sh2 = vec3<f32>(0.0);
    var sh3 = vec3<f32>(0.0);
    for (var py = min_pixel.y; py < max_y; py = py + 1u) {
        for (var px = min_pixel.x; px < max_x; px = px + 1u) {
            let pixel = vec2<i32>(i32(px), i32(py));
            let sample_depth = textureLoad(scene_depth, pixel, 0).x;
            if (sample_depth <= 0.0) {
                continue;
            }
            let radiance = textureLoad(scene_color, pixel, 0).rgb;
            let sample_uv = (vec2<f32>(f32(px), f32(py)) + vec2<f32>(0.5, 0.5)) / params.screen_size;
            let sample_view = reconstruct_view_position(sample_uv, sample_depth);
            var dir = sample_view - center_view;
            // Snap through the octahedral grid (parity with the stored atlas).
            dir = oct_to_dir(dir_to_oct(dir));
            // add_directional_radiance with unit solid-angle weight.
            let basis = sh_basis_l1(dir);
            sh0 = sh0 + radiance * basis.x;
            sh1 = sh1 + radiance * basis.y;
            sh2 = sh2 + radiance * basis.z;
            sh3 = sh3 + radiance * basis.w;
        }
    }

    // Bake the artistic gain into the captured radiance.
    sh0 = sh0 * params.intensity;
    sh1 = sh1 * params.intensity;
    sh2 = sh2 * params.intensity;
    sh3 = sh3 * params.intensity;

    probes[probe_index].sh0 = vec4<f32>(sh0, 0.0);
    probes[probe_index].sh1 = vec4<f32>(sh1, 0.0);
    probes[probe_index].sh2 = vec4<f32>(sh2, 0.0);
    probes[probe_index].sh3 = vec4<f32>(sh3, 0.0);
    // normal_depth = (view normal, view depth = -z); flags.x = valid.
    probes[probe_index].normal_depth = vec4<f32>(center_normal, -center_view.z);
    probes[probe_index].flags = vec4<f32>(1.0, 0.0, 0.0, 0.0);
}
