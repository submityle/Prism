// DDGI (dynamic diffuse global illumination) — per-probe update kernel.
//
// One workgroup per probe, 64 cooperative threads (`@workgroup_size(64)`,
// `dispatch_workgroups(probe_count, 1, 1)`). The pass gathers incoming radiance
// along a Fibonacci-sphere ray budget, re-integrates each probe's octahedral
// irradiance + two-moment depth fields with temporal hysteresis, refreshes the
// one-texel gutter border, and relocates / classifies the probe. It writes the
// two atlases (`texture_storage_2d<rgba16float, write>`), the two persistent
// history storage buffers (interior-only, no padding), and the per-probe
// `ProbeMeta` (relocation offset + activity flag).
//
// Arm-for-arm with the CPU golden
// `prism_render_shading::gi::irradiance_volume` (Majercik et al. 2019 / RTXGI):
//   * `fibonacci_sphere_dir`  ~ `rays::fibonacci_sphere_dir`.
//   * `oct_to_dir`            ~ `world_space::octahedral::oct_to_dir`.
//   * `interior_texel_dir`    ~ `IrradianceOct::interior_texel_dir`.
//   * `cosine_weighted_irradiance` ~ `ddgi_probe::cosine_weighted_irradiance`.
//   * `sharpened_depth_moments`    ~ `visibility::sharpened_depth_moments`.
//   * the gutter copy  ~ `IrradianceOct::copy_border`.
//   * `build_stats`/`relocate_offset`/`classify_probe` ~ `rays::probe_ray_stats`
//     + `relocation::{relocate_offset, classify_probe, transition_state}`.
//
// History is zero-initialised by wgpu, so frame 1 reads zero irradiance (black)
// and the exponential blend converges over subsequent frames. The pass reads
// the *previous* `ProbeMeta` and history, blends, and writes the new state in
// the same frame (persistent per-view resource, no +1-frame latency).
//
// Trace: a screen-space world-march in reverse-Z (device far = 0, near = 1).
// A ray that marches behind the recorded scene depth records that surface's
// pre-exposed scene colour as the incoming radiance, the world-space hit
// distance, and a front / back classification from the surface normal.

// --- Compile-time field dimensions (must match the default lattice) ----------

// Interior octahedral resolution of each irradiance probe (RTXGI 6x6). The
// shared / history layout is sized to this; the CPU-mirror test asserts it
// matches `settings::DEFAULT_IRRADIANCE_INTERIOR`.
const IRR_INTERIOR: u32 = 6u;
// Interior octahedral resolution of each depth / visibility probe (RTXGI
// 16x16).
const DEPTH_INTERIOR: u32 = 16u;
// Padded tile side length (interior + one-texel gutter on each edge).
const IRR_PADDED: u32 = 8u;
const DEPTH_PADDED: u32 = 18u;
// Interior texel counts (6*6, 16*16).
const IRR_TEXELS: u32 = 36u;
const DEPTH_TEXELS: u32 = 256u;
// Cooperative ray budget per probe (one ray per thread).
const RAY_COUNT: u32 = 64u;
// Screen-space march step count along each ray.
const MARCH_STEPS: u32 = 32u;
// Reverse-Z device-depth comparison bias (ray point considered behind the
// recorded surface once its device depth drops this far below it).
const DEPTH_BIAS: f32 = 2.0e-4;

const PI: f32 = 3.1415927;
// Golden `f32::MIN_POSITIVE` sentinel used by the degenerate-direction guards.
const MIN_POSITIVE: f32 = 1.175494e-38;
// Golden weight epsilon.
const WEIGHT_EPSILON: f32 = 1.0e-9;

// --- Shared ABI (uniform / storage twin of `shading/ddgi/abi.rs`) ------------

// std430/uniform twin of `GpuDdgiVolume` (identical to `ddgi_sample.wesl`).
struct DdgiVolume {
    origin: vec3<f32>,
    irradiance_interior: u32,
    spacing: vec3<f32>,
    depth_interior: u32,
    counts: vec3<i32>,
    pad_counts: u32,
    normal_bias: f32,
    view_bias: f32,
    hysteresis: f32,
    depth_sharpness: f32,
    intensity: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

// One stored probe's relocation offset + activity flag (identical to the
// sample pass).
struct ProbeMeta {
    offset: vec3<f32>,
    // Non-zero when the probe is active (`active` is reserved, so `active_flag`).
    active_flag: f32,
}

// Immediate (push-constant) twin of `GpuDdgiUpdateParams` (160 bytes).
struct UpdateParams {
    // World -> clip transform used to project a marched ray point to the screen.
    world_to_clip: mat4x4<f32>,
    // Clip -> world transform used to reconstruct the hit world position.
    clip_to_world: mat4x4<f32>,
    // Back-face fraction above which the probe relocates / deactivates.
    backface_threshold: f32,
    // Minimum stand-off distance kept from the closest front face.
    min_frontface_distance: f32,
    // Front-face distance within which the probe counts as active.
    activity_distance: f32,
    // Bounded per-axis relocation fraction of the cell spacing.
    relocation_limit: f32,
    // World-space ray-march length (`spacing.length() * 2`).
    max_ray_distance: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var scene_depth: texture_2d<f32>;
@group(0) @binding(1) var normal_gbuffer: texture_2d<f32>;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var<uniform> volume: DdgiVolume;
@group(0) @binding(4) var<storage, read_write> probe_meta: array<ProbeMeta>;
@group(0) @binding(5) var<storage, read_write> irradiance_history: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> depth_history: array<vec2<f32>>;
@group(0) @binding(7) var irradiance_atlas: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var depth_atlas: texture_storage_2d<rgba16float, write>;

var<immediate> params: UpdateParams;

// --- Workgroup-shared ray buffers + per-probe state --------------------------

// Per-ray results (filled by phase 1, consumed by phases 2 / 3).
var<workgroup> ray_dir: array<vec3<f32>, 64>;
var<workgroup> ray_radiance: array<vec3<f32>, 64>;
var<workgroup> ray_dist: array<f32, 64>;
// 1 = front hit, -1 = back hit, 0 = miss.
var<workgroup> ray_hit: array<f32, 64>;

// Interior-only re-integrated texels (phase 2 writes, phase 3 border-copies).
var<workgroup> irr_texels: array<vec3<f32>, 36>;
var<workgroup> depth_texels: array<vec2<f32>, 256>;

// Per-probe state computed by thread 0 in phase 0, read by all threads.
var<workgroup> w_probe_pos: vec3<f32>;
var<workgroup> w_prev_offset: vec3<f32>;
var<workgroup> w_prev_active: f32;
var<workgroup> w_eff_hysteresis: f32;

// --- Octahedral map (golden `world_space::octahedral`) -----------------------

fn sign_not_zero(value: f32) -> f32 {
    return select(-1.0, 1.0, value >= 0.0);
}

// Decode an octahedral UV in [0, 1]^2 back to a unit direction (golden
// `oct_to_dir`).
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

// --- Fibonacci-sphere ray directions (golden `rays::fibonacci_sphere_dir`) ---

fn fibonacci_sphere_dir(index: u32, count: u32) -> vec3<f32> {
    if (count == 0u) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let golden = PI * (3.0 - sqrt(5.0));
    let i = f32(index);
    let n = f32(count);
    let z = 1.0 - 2.0 * (i + 0.5) / n;
    let r = sqrt(max(1.0 - z * z, 0.0));
    let theta = golden * i;
    return vec3<f32>(r * cos(theta), r * sin(theta), z);
}

// Interior texel's octahedral direction (golden `interior_texel_dir`).
fn interior_texel_dir(ix: u32, iy: u32, interior: u32) -> vec3<f32> {
    let n = f32(interior);
    let u = (f32(min(ix, interior - 1u)) + 0.5) / n;
    let v = (f32(min(iy, interior - 1u)) + 0.5) / n;
    return oct_to_dir(vec2<f32>(u, v));
}

// World-space base position of a probe lattice coordinate (golden
// `ProbeGrid::probe_base_position`).
fn probe_base_position(coord: vec3<i32>) -> vec3<f32> {
    return volume.origin + vec3<f32>(coord) * volume.spacing;
}

// Top-left texel of probe `p`'s padded tile in the tiled 2D atlas (golden /
// `ddgi_sample.wesl` `probe_tile_origin`).
fn probe_tile_origin(p: u32, padded: u32, atlas_width: u32) -> vec2<u32> {
    let per_row = max(atlas_width / padded, 1u);
    let tx = p % per_row;
    let ty = p / per_row;
    return vec2<u32>(tx * padded, ty * padded);
}

// --- Screen-space probe ray trace (reverse-Z) --------------------------------

struct TraceResult {
    radiance: vec3<f32>,
    distance: f32,
    // 1 = front hit, -1 = back hit, 0 = miss.
    hit: f32,
}

// March `dir` from `origin` up to `max_dist`, returning the first surface the
// ray passes behind (reverse-Z: smaller device depth = farther). Misses carry
// `max_dist` and zero radiance.
fn trace_ray(origin: vec3<f32>, dir: vec3<f32>, screen_size: vec2<f32>, max_dist: f32) -> TraceResult {
    var result: TraceResult;
    result.radiance = vec3<f32>(0.0, 0.0, 0.0);
    result.distance = max_dist;
    result.hit = 0.0;

    let steps = i32(MARCH_STEPS);
    let inv_steps = 1.0 / f32(MARCH_STEPS);
    for (var i = 1; i <= steps; i = i + 1) {
        let t = max_dist * f32(i) * inv_steps;
        let p = origin + dir * t;
        let clip = params.world_to_clip * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) {
            break;
        }
        let ndc = clip.xyz / clip.w;
        if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) {
            continue;
        }
        var uv = ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);
        uv.y = 1.0 - uv.y;
        let pixel = vec2<i32>(uv * screen_size);
        let scene_d = textureLoad(scene_depth, pixel, 0).r;
        if (scene_d <= 0.0) {
            // Sky / far plane in reverse-Z: no occluder along this step.
            continue;
        }
        if (ndc.z < scene_d - DEPTH_BIAS) {
            // Ray point is behind the recorded surface: record the hit.
            let surf_h = params.clip_to_world * vec4<f32>(ndc.xy, scene_d, 1.0);
            let surf = surf_h.xyz / surf_h.w;
            result.distance = length(surf - origin);
            result.radiance = textureLoad(scene_color, pixel, 0).rgb;
            let n = normalize(textureLoad(normal_gbuffer, pixel, 0).xyz * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
            let backface = dot(dir, n) > 0.0;
            result.hit = select(1.0, -1.0, backface);
            return result;
        }
    }
    return result;
}

// --- Field re-integration (golden `cosine_weighted_irradiance` / moments) ----

// Cosine-weighted irradiance of the shared ray budget about `axis_dir` (golden
// `cosine_weighted_irradiance`). Reads `ray_dir` / `ray_radiance`.
fn cosine_weighted_irradiance(axis_dir: vec3<f32>) -> vec3<f32> {
    let axis_len_sq = dot(axis_dir, axis_dir);
    if (axis_len_sq <= MIN_POSITIVE) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let axis = axis_dir * inverseSqrt(axis_len_sq);
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    var weight_sum = 0.0;
    for (var r = 0u; r < RAY_COUNT; r = r + 1u) {
        let d = ray_dir[r];
        let len_sq = dot(d, d);
        if (len_sq <= MIN_POSITIVE) {
            continue;
        }
        let dn = d * inverseSqrt(len_sq);
        let w = max(dot(axis, dn), 0.0);
        if (w <= 0.0) {
            continue;
        }
        sum = sum + ray_radiance[r] * w;
        weight_sum = weight_sum + w;
    }
    if (weight_sum <= WEIGHT_EPSILON) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let out = sum / weight_sum;
    return max(out, vec3<f32>(0.0, 0.0, 0.0));
}

// Sharpened two-moment depth of the shared ray budget about `axis_dir` (golden
// `sharpened_depth_moments`). `.xy` = `[mean, mean_sq]`, `.z` = 1 when a valid
// (non-degenerate) estimate was formed, 0 when the caller should keep `prev`.
fn sharpened_depth_moments(axis_dir: vec3<f32>, sharpness: f32, max_distance: f32) -> vec3<f32> {
    let axis_len_sq = dot(axis_dir, axis_dir);
    if (axis_len_sq <= MIN_POSITIVE) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let axis = axis_dir * inverseSqrt(axis_len_sq);
    let sharp = max(sharpness, 1.0);
    let md = max(max_distance, 0.0);
    var sum_d = 0.0;
    var sum_d2 = 0.0;
    var weight_sum = 0.0;
    for (var r = 0u; r < RAY_COUNT; r = r + 1u) {
        let d = ray_dir[r];
        let len_sq = dot(d, d);
        if (len_sq <= MIN_POSITIVE) {
            continue;
        }
        let dn = d * inverseSqrt(len_sq);
        let c = max(dot(axis, dn), 0.0);
        if (c <= 0.0) {
            continue;
        }
        let w = pow(c, sharp);
        if (w <= 0.0) {
            continue;
        }
        let dist = clamp(ray_dist[r], 0.0, md);
        sum_d = sum_d + dist * w;
        sum_d2 = sum_d2 + dist * dist * w;
        weight_sum = weight_sum + w;
    }
    if (weight_sum <= WEIGHT_EPSILON) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let inv = 1.0 / weight_sum;
    return vec3<f32>(sum_d * inv, sum_d2 * inv, 1.0);
}

// --- Per-probe statistics + relocation (golden `rays` / `relocation`) --------

struct RayStats {
    backface_fraction: f32,
    closest_frontface_distance: f32,
    closest_frontface_dir: vec3<f32>,
    farthest_frontface_distance: f32,
    farthest_frontface_dir: vec3<f32>,
    closest_backface_dir: vec3<f32>,
    closest_backface_distance: f32,
}

// Fold the shared ray budget into relocation / classification statistics
// (golden `probe_ray_stats`). `3.4e38` stands in for the golden `INFINITY`.
fn build_stats() -> RayStats {
    var s: RayStats;
    s.backface_fraction = 0.0;
    s.closest_frontface_distance = 3.4e38;
    s.closest_frontface_dir = vec3<f32>(0.0, 0.0, 0.0);
    s.farthest_frontface_distance = 0.0;
    s.farthest_frontface_dir = vec3<f32>(0.0, 0.0, 0.0);
    s.closest_backface_dir = vec3<f32>(0.0, 0.0, 0.0);
    s.closest_backface_distance = 3.4e38;

    var front = 0u;
    var back = 0u;
    for (var r = 0u; r < RAY_COUNT; r = r + 1u) {
        let dist = max(ray_dist[r], 0.0);
        let hit = ray_hit[r];
        if (hit > 0.5) {
            front = front + 1u;
            if (dist < s.closest_frontface_distance) {
                s.closest_frontface_distance = dist;
                s.closest_frontface_dir = ray_dir[r];
            }
            if (dist > s.farthest_frontface_distance) {
                s.farthest_frontface_distance = dist;
                s.farthest_frontface_dir = ray_dir[r];
            }
        } else if (hit < -0.5) {
            back = back + 1u;
            if (dist < s.closest_backface_distance) {
                s.closest_backface_distance = dist;
                s.closest_backface_dir = ray_dir[r];
            }
        }
    }
    let total = front + back;
    if (total != 0u) {
        s.backface_fraction = f32(back) / f32(total);
    }
    return s;
}

// Sanitise probe spacing per axis (golden `sanitize_spacing`, finite inputs).
fn sanitize_spacing(v: vec3<f32>) -> vec3<f32> {
    let m = 1.0e-6;
    return vec3<f32>(max(abs(v.x), m), max(abs(v.y), m), max(abs(v.z), m));
}

// Normalise or report degenerate: `.xyz` = unit vector, `.w` = 1 when valid
// (golden `normalize_or_none`).
fn normalize_or_none(v: vec3<f32>) -> vec4<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= MIN_POSITIVE) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    return vec4<f32>(v * inverseSqrt(len_sq), 1.0);
}

// Bounded probe relocation offset (golden `relocate_offset`).
fn relocate_offset(prev: vec3<f32>, s: RayStats) -> vec3<f32> {
    let spacing = sanitize_spacing(volume.spacing);
    let min_front = max(params.min_frontface_distance, 0.0);
    let bt = clamp(params.backface_threshold, 0.0, 1.0);
    let limit = clamp(params.relocation_limit, 0.0, 0.4999);

    var offset = prev;
    if (s.backface_fraction > bt) {
        let nd = normalize_or_none(s.closest_backface_dir);
        if (nd.w > 0.5) {
            let push = max(s.closest_backface_distance, 0.0) + min_front;
            offset = prev + nd.xyz * push;
        }
    } else if (s.closest_frontface_distance < min_front) {
        let nd = normalize_or_none(s.farthest_frontface_dir);
        if (nd.w > 0.5) {
            offset = prev + nd.xyz * min_front;
        }
    } else {
        offset = prev * 0.5;
    }

    let max_axis = spacing * limit;
    return clamp(offset, -max_axis, max_axis);
}

// Raw probe activity (golden `classify_probe`): 1 = active, 0 = inactive. The
// `NewlyVacated` transient collapses into the activity flag + the
// `prev_active`-gated hysteresis reset, so it needs no separate stored state.
fn classify_probe_active(s: RayStats) -> f32 {
    let bt = clamp(params.backface_threshold, 0.0, 1.0);
    let ad = max(params.activity_distance, 0.0);
    if (s.backface_fraction > bt) {
        return 0.0;
    }
    if (s.closest_frontface_distance <= ad) {
        return 1.0;
    }
    return 0.0;
}

// --- Gutter border copy (golden `copy_border`) -------------------------------

// Fetch an interior irradiance texel addressed by padded coordinate
// `(px, py)` with `1 <= px, py <= IRR_INTERIOR`.
fn fetch_irr(px: u32, py: u32) -> vec3<f32> {
    return irr_texels[(py - 1u) * IRR_INTERIOR + (px - 1u)];
}

fn fetch_depth(px: u32, py: u32) -> vec2<f32> {
    return depth_texels[(py - 1u) * DEPTH_INTERIOR + (px - 1u)];
}

fn store_irr(origin: vec2<u32>, px: u32, py: u32, value: vec3<f32>) {
    let coord = vec2<i32>(origin) + vec2<i32>(i32(px), i32(py));
    textureStore(irradiance_atlas, coord, vec4<f32>(value, 1.0));
}

fn store_depth(origin: vec2<u32>, px: u32, py: u32, value: vec2<f32>) {
    let coord = vec2<i32>(origin) + vec2<i32>(i32(px), i32(py));
    textureStore(depth_atlas, coord, vec4<f32>(value, 0.0, 1.0));
}

// --- Entry point -------------------------------------------------------------

@compute @workgroup_size(64)
fn probe_update_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let probe = wid.x;

    // Phase 0 (thread 0): decode the probe, read its previous metadata, and
    // publish the probe world position + effective hysteresis.
    if (lid == 0u) {
        let cx = u32(max(volume.counts.x, 1));
        let cy = u32(max(volume.counts.y, 1));
        let px = probe % cx;
        let py = (probe / cx) % cy;
        let pz = probe / (cx * cy);
        let coord = vec3<i32>(i32(px), i32(py), i32(pz));

        let prev = probe_meta[probe];
        w_prev_offset = prev.offset;
        w_prev_active = prev.active_flag;
        w_probe_pos = probe_base_position(coord) + prev.offset;
        // Inactive / first-frame probes reset (hysteresis 0 overwrites with the
        // fresh estimate); active probes blend slowly.
        w_eff_hysteresis = select(0.0, volume.hysteresis, prev.active_flag != 0.0);
    }
    workgroupBarrier();

    // Phase 1 (all 64 threads): trace one Fibonacci-sphere ray each.
    let screen_dims = textureDimensions(scene_depth);
    let screen_size = vec2<f32>(f32(screen_dims.x), f32(screen_dims.y));
    let dir = fibonacci_sphere_dir(lid, RAY_COUNT);
    let res = trace_ray(w_probe_pos, dir, screen_size, params.max_ray_distance);
    ray_dir[lid] = dir;
    ray_radiance[lid] = res.radiance;
    ray_dist[lid] = res.distance;
    ray_hit[lid] = res.hit;
    workgroupBarrier();

    let eff_hysteresis = w_eff_hysteresis;

    // Phase 2a: re-integrate every interior irradiance texel, blend with
    // history, write the history buffer + shared tile + the interior atlas
    // texel (padded coordinate (ix + 1, iy + 1)).
    let irr_dims = textureDimensions(irradiance_atlas);
    let irr_origin = probe_tile_origin(probe, IRR_PADDED, irr_dims.x);
    for (var ti = lid; ti < IRR_TEXELS; ti = ti + RAY_COUNT) {
        let ix = ti % IRR_INTERIOR;
        let iy = ti / IRR_INTERIOR;
        let texel_dir = interior_texel_dir(ix, iy, IRR_INTERIOR);
        let fresh = cosine_weighted_irradiance(texel_dir);
        let hist_idx = probe * IRR_TEXELS + ti;
        let prev = irradiance_history[hist_idx].rgb;
        let blended = fresh + (prev - fresh) * eff_hysteresis;
        let clamped = max(blended, vec3<f32>(0.0, 0.0, 0.0));
        irradiance_history[hist_idx] = vec4<f32>(clamped, 1.0);
        irr_texels[ti] = clamped;
        store_irr(irr_origin, ix + 1u, iy + 1u, clamped);
    }

    // Phase 2b: re-integrate every interior depth texel.
    let depth_dims = textureDimensions(depth_atlas);
    let depth_origin = probe_tile_origin(probe, DEPTH_PADDED, depth_dims.x);
    for (var ti = lid; ti < DEPTH_TEXELS; ti = ti + RAY_COUNT) {
        let ix = ti % DEPTH_INTERIOR;
        let iy = ti / DEPTH_INTERIOR;
        let texel_dir = interior_texel_dir(ix, iy, DEPTH_INTERIOR);
        let moments = sharpened_depth_moments(texel_dir, volume.depth_sharpness, params.max_ray_distance);
        let hist_idx = probe * DEPTH_TEXELS + ti;
        let prev = depth_history[hist_idx];
        var blended = prev;
        if (moments.z > 0.5) {
            let fresh = moments.xy;
            blended = fresh + (prev - fresh) * eff_hysteresis;
        }
        depth_history[hist_idx] = blended;
        depth_texels[ti] = blended;
        store_depth(depth_origin, ix + 1u, iy + 1u, blended);
    }
    workgroupBarrier();

    // Phase 3 (thread 0): copy the gutter border from the shared interior
    // tiles, then relocate / classify the probe and write its metadata.
    if (lid == 0u) {
        // Irradiance gutter (golden `copy_border`, mirror(i) = n + 1 - i).
        let irr_last = IRR_INTERIOR + 1u;
        for (var i = 1u; i <= IRR_INTERIOR; i = i + 1u) {
            let m = IRR_INTERIOR + 1u - i;
            store_irr(irr_origin, i, 0u, fetch_irr(m, 1u));
            store_irr(irr_origin, i, irr_last, fetch_irr(m, IRR_INTERIOR));
            store_irr(irr_origin, 0u, i, fetch_irr(1u, m));
            store_irr(irr_origin, irr_last, i, fetch_irr(IRR_INTERIOR, m));
        }
        store_irr(irr_origin, 0u, 0u, fetch_irr(IRR_INTERIOR, IRR_INTERIOR));
        store_irr(irr_origin, irr_last, 0u, fetch_irr(1u, IRR_INTERIOR));
        store_irr(irr_origin, 0u, irr_last, fetch_irr(IRR_INTERIOR, 1u));
        store_irr(irr_origin, irr_last, irr_last, fetch_irr(1u, 1u));

        // Depth gutter.
        let depth_last = DEPTH_INTERIOR + 1u;
        for (var i = 1u; i <= DEPTH_INTERIOR; i = i + 1u) {
            let m = DEPTH_INTERIOR + 1u - i;
            store_depth(depth_origin, i, 0u, fetch_depth(m, 1u));
            store_depth(depth_origin, i, depth_last, fetch_depth(m, DEPTH_INTERIOR));
            store_depth(depth_origin, 0u, i, fetch_depth(1u, m));
            store_depth(depth_origin, depth_last, i, fetch_depth(DEPTH_INTERIOR, m));
        }
        store_depth(depth_origin, 0u, 0u, fetch_depth(DEPTH_INTERIOR, DEPTH_INTERIOR));
        store_depth(depth_origin, depth_last, 0u, fetch_depth(1u, DEPTH_INTERIOR));
        store_depth(depth_origin, 0u, depth_last, fetch_depth(DEPTH_INTERIOR, 1u));
        store_depth(depth_origin, depth_last, depth_last, fetch_depth(1u, 1u));

        // Relocation + classification from the ray statistics.
        let stats = build_stats();
        let offset = relocate_offset(w_prev_offset, stats);
        let active_flag = classify_probe_active(stats);
        probe_meta[probe] = ProbeMeta(offset, active_flag);
    }
}
