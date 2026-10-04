// World-space ReSTIR DI — resolve pass (GPU twin).
//
// One compute invocation per screen pixel. The producer -> inject -> seed ->
// fill chain has, by this point, filled the resident SHARC reservoir table:
// one open-addressed slot per hashed world cell, each holding the cell's
// surviving light sample (secondary point + normal + RGB radiance) and its
// finalised unbiased contribution weight `W`. This pass turns that world-space
// table back into a screen-space direct-illumination estimate: it reconstructs
// each pixel's world shading point from the SSR prepass (exactly as the
// visible-point producer did), re-hashes it into the identical SHARC cell,
// linear-probes the table for that cell's finalised reservoir, and — on a hit —
// re-evaluates the stored light sample from this pixel's shading point under a
// reconnection shift (golden `reconnection_target` semantics, RGB):
//
//     E_rgb = radiance * geometric_term(x_v, n_v, x_s, n_s) * W * intensity
//
// The result is written to the `gi_out` export (`rgb` = pre-BRDF direct
// irradiance, `a` = `[0, 1]` hit confidence), a GI-style export buffer a
// downstream composite folds into `scene_color` — this pass never touches the
// framebuffer, exactly mirroring `world_space_gi_resolve.wesl`.
//
// Parity: the world-pos/normal reconstruction is byte-identical to
// `world_restir_visible_points.wesl`; the SHARC key/checksum/bucket math is the
// arm-for-arm `vec2<u32>` port shared with `world_restir_inject.wesl`; the
// `geometric_term` and the reservoir (de)serialisation + `valid`/`checksum`
// guards are byte-identical to `world_restir_fill.wesl`. WESL has no includes,
// so these helpers are duplicated verbatim (the established inject/seed/fill
// pattern); the `shader_tests` CPU mirror asserts the resolve estimate matches
// the golden `prism_render_shading::gi::screen_probe::restir::geometric_term`,
// the only CPU-checkable proof of the port (the sandbox has no GPU).

// Smallest positive normal f32 (golden `f32::MIN_POSITIVE`).
const MIN_POSITIVE: f32 = 1.175494e-38;
// Grid level clamp bound (golden `spatial_hash::MAX_LEVEL`).
const MAX_LEVEL: i32 = 24;
// Signed-axis bias folded into 21 bits (golden `COORD_BIAS = 1 << 20`).
const COORD_BIAS: u32 = 1048576u;
// Low-21-bit axis mask (golden `COORD_MASK`).
const COORD_MASK: u32 = 0x1fffffu;
// Linear-probe window (golden `world_reservoir::PROBE_LIMIT`).
const PROBE_LIMIT: u32 = 32u;

// 64-bit mixing constants as (low, high) u32 pairs (identical to inject/fill).
const SEED_KEY: vec2<u32> = vec2<u32>(0x7f4a7c15u, 0x9e3779b9u);
const SEED_SUM: vec2<u32> = vec2<u32>(0x27d4eb4fu, 0xc2b2ae3du);
const MIX_C1: vec2<u32> = vec2<u32>(0xed558ccdu, 0xff51afd7u);
const MIX_C2: vec2<u32> = vec2<u32>(0x1a85ec53u, 0xc4ceb9feu);
const LEVEL_MUL: vec2<u32> = vec2<u32>(0x7f4a7c15u, 0x9e3779b9u);
const NORMAL_MUL: vec2<u32> = vec2<u32>(0xed558ccdu, 0xff51afd7u);

// One hashed world-cell reservoir slot — std430 twin of
// `abi::GpuWorldRestirReservoir` (five vec4 lanes, 80 bytes); byte-identical to
// the inject / seed / fill twins' `WorldRestirReservoir`.
struct WorldRestirReservoir {
    // visible_point.xyz + finalised contribution weight W.
    visible_point_w: vec4<f32>,
    // visible_normal.xyz + confidence m.
    visible_normal_m: vec4<f32>,
    // sample_point.xyz + SHARC collision checksum (bitcast to u32).
    sample_point_checksum: vec4<f32>,
    // sample_normal.xyz + valid flag (bitcast to u32).
    sample_normal_valid: vec4<f32>,
    // radiance.xyz + one pad word.
    radiance_pad: vec4<f32>,
}

// Resolve immediate block — std430 twin of `abi::GpuWorldRestirResolveParams`
// (two mat4x4 + four vec4 lanes = 192 bytes, no implicit padding).
struct ResolveParams {
    // Clip -> view (inverse reverse-Z projection); reconstructs the view-space
    // shading point from a device depth.
    view_from_clip: mat4x4<f32>,
    // View -> world; lifts the reconstructed point and its normal into world
    // space for the SHARC hash.
    world_from_view: mat4x4<f32>,
    // camera_position.xyz + base_cell_size.
    camera_base: vec4<f32>,
    // jitter.xyz + level_scale.
    jitter_level: vec4<f32>,
    // (capacity, normal_resolution, screen_width, screen_height).
    dims: vec4<u32>,
    // (intensity, 0, 0, 0): artistic gain multiplying the resolved irradiance.
    intensity_pad: vec4<f32>,
}

// SSR prepass reverse-Z device depth (R32Float); 0 is the cleared background.
@group(0) @binding(0) var depth_tex: texture_2d<f32>;
// SSR packed view-space normal + roughness (Rgba16Float); `.xyz * 2 - 1` is the
// unit view-space normal (matching the visible-point producer's decode).
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
// The resident finalised reservoir table (fill pass output), read-only.
@group(0) @binding(2) var<storage, read> reservoirs: array<WorldRestirReservoir>;
// Direct-illumination export: rgb = pre-BRDF irradiance, a = hit confidence.
@group(0) @binding(3) var gi_out: texture_storage_2d<rgba16float, write>;

var<immediate> params: ResolveParams;

// A fully-resolved SHARC hash-grid cell identity (golden `HashGridKey`).
struct Key {
    coord: vec3<i32>,
    level: i32,
    normal_bin: u32,
}

// ---------------------------------------------------------------------------
// 64-bit (u64-as-2x u32) arithmetic — reproduces the golden `u64` hash math.
// ---------------------------------------------------------------------------

fn mul32x32_64(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xffffu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xffffu;
    let b_hi = b >> 16u;
    let t0 = a_lo * b_lo;
    let t1 = a_hi * b_lo + (t0 >> 16u);
    let t2 = a_lo * b_hi + (t1 & 0xffffu);
    let lo = (t2 << 16u) | (t0 & 0xffffu);
    let hi = a_hi * b_hi + (t1 >> 16u) + (t2 >> 16u);
    return vec2<u32>(lo, hi);
}

fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = mul32x32_64(a.x, b.x);
    return vec2<u32>(p.x, p.y + a.x * b.y + a.y * b.x);
}

fn u64_shr33(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.y >> 1u, 0u);
}

fn u64_xor(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

fn fmix64(x_in: vec2<u32>) -> vec2<u32> {
    var x = x_in;
    x = u64_xor(x, u64_shr33(x));
    x = u64_mul(x, MIX_C1);
    x = u64_xor(x, u64_shr33(x));
    x = u64_mul(x, MIX_C2);
    x = u64_xor(x, u64_shr33(x));
    return x;
}

fn encode_axis(v: i32) -> u32 {
    return (bitcast<u32>(v) + COORD_BIAS) & COORD_MASK;
}

fn pack_key(key: Key) -> vec2<u32> {
    let ex = encode_axis(key.coord.x);
    let ey = encode_axis(key.coord.y);
    let ez = encode_axis(key.coord.z);
    let coord = vec2<u32>(ex | (ey << 21u), (ey >> 11u) | (ez << 10u));
    let level = u64_mul(vec2<u32>(u32(key.level), 0u), LEVEL_MUL);
    let normal = u64_mul(vec2<u32>(key.normal_bin, 0u), NORMAL_MUL);
    return u64_xor(u64_xor(coord, level), normal);
}

fn hash_key(key: Key) -> vec2<u32> {
    return fmix64(u64_xor(pack_key(key), SEED_KEY));
}

fn key_checksum(key: Key) -> u32 {
    let h = fmix64(u64_xor(pack_key(key), SEED_SUM));
    return h.y ^ h.x;
}

fn bucket_index(key: Key, capacity: u32) -> u32 {
    let m = max(capacity, 1u);
    if (m <= 1u) {
        return 0u;
    }
    let h = hash_key(key);
    var rem: u32 = 0u;
    for (var i: i32 = 31; i >= 0; i = i - 1) {
        let bit = (h.y >> u32(i)) & 1u;
        rem = (rem << 1u) | bit;
        if (rem >= m) {
            rem = rem - m;
        }
    }
    for (var i: i32 = 31; i >= 0; i = i - 1) {
        let bit = (h.x >> u32(i)) & 1u;
        rem = (rem << 1u) | bit;
        if (rem >= m) {
            rem = rem - m;
        }
    }
    return rem;
}

// ---------------------------------------------------------------------------
// Octahedral normal binning (golden `octahedral` + `quantize_normal`).
// ---------------------------------------------------------------------------

fn sign_not_zero(value: f32) -> f32 {
    return select(-1.0, 1.0, value >= 0.0);
}

fn dir_to_oct(dir: vec3<f32>) -> vec2<f32> {
    let l1 = abs(dir.x) + abs(dir.y) + abs(dir.z);
    if (l1 <= MIN_POSITIVE) {
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

fn normalize_or_z(n: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(n, n);
    if (len_sq > MIN_POSITIVE) {
        return n * inverseSqrt(len_sq);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn quantize_normal(normal: vec3<f32>, resolution: u32) -> u32 {
    let res = max(resolution, 1u);
    let uv = dir_to_oct(normalize_or_z(normal));
    let res_f = f32(res);
    let bx = u32(clamp(floor(uv.x * res_f), 0.0, res_f - 1.0));
    let by = u32(clamp(floor(uv.y * res_f), 0.0, res_f - 1.0));
    return by * res + bx;
}

// ---------------------------------------------------------------------------
// Distance-scaled grid quantisation (golden `spatial_hash`).
// ---------------------------------------------------------------------------

fn safe_cell_size(cell_size: f32) -> f32 {
    return select(MIN_POSITIVE, cell_size, cell_size > MIN_POSITIVE);
}

fn cell_size_at_level(base_cell_size: f32, level: i32) -> f32 {
    let base = safe_cell_size(base_cell_size);
    let lvl = f32(clamp(level, 0, MAX_LEVEL));
    return base * exp2(lvl);
}

fn grid_level(position: vec3<f32>, camera: vec3<f32>, base_cell_size: f32, level_scale: f32) -> i32 {
    let base = safe_cell_size(base_cell_size);
    if (!(level_scale > 0.0)) {
        return 0;
    }
    let dist = length(position - camera);
    if (!(dist > base)) {
        return 0;
    }
    let ratio = max(dist * level_scale / base, 1.0);
    let level = floor(log2(ratio));
    return clamp(i32(level), 0, MAX_LEVEL);
}

fn quantize_position(position: vec3<f32>, camera: vec3<f32>, base_cell_size: f32, level_scale: f32, jitter: vec3<f32>) -> Key {
    var key: Key;
    key.level = grid_level(position, camera, base_cell_size, level_scale);
    key.normal_bin = 0u;
    let finite = (position.x == position.x) && (position.y == position.y) && (position.z == position.z);
    if (!finite) {
        key.coord = vec3<i32>(0, 0, 0);
        return key;
    }
    let size = cell_size_at_level(base_cell_size, key.level);
    let inv = 1.0 / size;
    key.coord = vec3<i32>(
        i32(floor(position.x * inv + jitter.x)),
        i32(floor(position.y * inv + jitter.y)),
        i32(floor(position.z * inv + jitter.z)),
    );
    return key;
}

// ---------------------------------------------------------------------------
// Reservoir guards + geometry factor (byte-identical to the fill twin).
// ---------------------------------------------------------------------------

fn reservoir_valid(r: WorldRestirReservoir) -> bool {
    return bitcast<u32>(r.sample_normal_valid.w) != 0u;
}

fn reservoir_checksum(r: WorldRestirReservoir) -> u32 {
    return bitcast<u32>(r.sample_point_checksum.w);
}

// Surface-to-surface geometry factor cos_v * cos_s / dist^2 (golden
// `geometric_term`); zero for degenerate / back-facing / non-finite pairs.
fn geometric_term(visible_point: vec3<f32>, visible_normal: vec3<f32>, sample_point: vec3<f32>, sample_normal: vec3<f32>) -> f32 {
    let delta = sample_point - visible_point;
    let dist_sq = dot(delta, delta);
    if (dist_sq > MIN_POSITIVE) {
        let inv_dist = inverseSqrt(dist_sq);
        let dir = delta * inv_dist;
        let cos_v = max(dot(visible_normal, dir), 0.0);
        let cos_s = max(dot(sample_normal, -dir), 0.0);
        let term = cos_v * cos_s / dist_sq;
        return select(0.0, max(term, 0.0), term == term);
    }
    return 0.0;
}

// ---------------------------------------------------------------------------
// View-position reconstruction (byte-identical to the visible-point producer).
// ---------------------------------------------------------------------------

fn reconstruct_view_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = params.view_from_clip * clip;
    return view.xyz / view.w;
}

// Component-wise finite test for an RGB estimate (golden `is_finite`).
fn rgb_finite(v: vec3<f32>) -> bool {
    return (v.x == v.x) && (v.y == v.y) && (v.z == v.z);
}

@compute @workgroup_size(8, 8, 1)
fn resolve_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let screen_w = params.dims.z;
    let screen_h = params.dims.w;
    if (gid.x >= screen_w || gid.y >= screen_h) {
        return;
    }
    let pixel = vec2<i32>(i32(gid.x), i32(gid.y));

    let device_depth = textureLoad(depth_tex, pixel, 0).x;
    // Reverse-Z: device depth 0 is the cleared background — no surface to light.
    if (!(device_depth > 0.0)) {
        textureStore(gi_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let camera = params.camera_base.xyz;
    let base_cell_size = params.camera_base.w;
    let jitter = params.jitter_level.xyz;
    let level_scale = params.jitter_level.w;
    let capacity = params.dims.x;
    let normal_resolution = params.dims.y;
    let intensity = params.intensity_pad.x;

    // Reconstruct the world shading point + normal exactly as the producer did,
    // so this pixel re-hashes into the identical SHARC cell the inject pass
    // claimed for it.
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(screen_w), f32(screen_h));
    let view_pos = reconstruct_view_position(uv, device_depth);
    let world_pos = (params.world_from_view * vec4<f32>(view_pos, 1.0)).xyz;

    let packed = textureLoad(normal_tex, pixel, 0).xyz;
    let view_normal = normalize(packed * 2.0 - 1.0);
    let c0 = params.world_from_view[0].xyz;
    let c1 = params.world_from_view[1].xyz;
    let c2 = params.world_from_view[2].xyz;
    let world_normal_raw = c0 * view_normal.x + c1 * view_normal.y + c2 * view_normal.z;
    let n_len = length(world_normal_raw);
    let world_normal = select(vec3<f32>(0.0, 1.0, 0.0), world_normal_raw / n_len, n_len > 1e-8);

    var key = quantize_position(world_pos, camera, base_cell_size, level_scale, jitter);
    key.normal_bin = quantize_normal(world_normal, normal_resolution);

    let cs = key_checksum(key);
    let base = bucket_index(key, capacity);
    let steps = min(PROBE_LIMIT, capacity);

    // Open-addressed linear probe mirroring the inject claim: stop at the first
    // invalid (empty) slot — the table is first-wins contiguous — and match on
    // the SHARC checksum (same raw `key_checksum` all passes key on).
    var hit = false;
    var slot: WorldRestirReservoir;
    for (var i: u32 = 0u; i < steps; i = i + 1u) {
        let idx = (base + i) % capacity;
        let cand = reservoirs[idx];
        if (!reservoir_valid(cand)) {
            break;
        }
        if (reservoir_checksum(cand) == cs) {
            slot = cand;
            hit = true;
            break;
        }
    }

    if (!hit) {
        textureStore(gi_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Reconnection shift (golden `reconnection_target`, RGB): keep the stored
    // secondary sample (point / normal / radiance) and the finalised weight W,
    // re-evaluate the geometry factor from *this pixel's* shading point.
    let W = slot.visible_point_w.w;
    let sample_point = slot.sample_point_checksum.xyz;
    let sample_normal = slot.sample_normal_valid.xyz;
    let radiance = slot.radiance_pad.xyz;
    let g = geometric_term(world_pos, world_normal, sample_point, sample_normal);
    let estimate = radiance * (g * W * intensity);
    let energy = max(estimate, vec3<f32>(0.0));

    // Confidence is a valid-hit blend weight: a finite estimate from a matched
    // slot resolves at full confidence, a NaN/Inf estimate falls back to 0 so
    // the composite leaves the pixel's clustered direct lighting untouched.
    let confidence = select(0.0, 1.0, rgb_finite(estimate) && (W == W));
    textureStore(gi_out, pixel, vec4<f32>(energy, confidence));
}
