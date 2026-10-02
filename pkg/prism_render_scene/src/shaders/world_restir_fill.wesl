// World-space ReSTIR DI — reservoir fill / GRIS spatial-reuse pass (GPU twin).
//
// One compute invocation per resident reservoir-table slot. The pass reads the
// previous (`src`) reservoir table, pools each occupied cell with a jittered
// ring of spatial neighbours under generalized resampled importance sampling
// (GRIS), re-finalizes the pooled reservoir's unbiased contribution weight `W`,
// and writes the result into the double-buffered (`dst`) table. Empty slots are
// copied through unchanged so the ping-pong preserves temporal history.
//
// This kernel is an arm-for-arm port of the CPU golden:
//   * SHARC spatial-hash key + checksum + bucket:
//     `prism_render_shading::gi::world_restir::spatial_hash`.
//   * octahedral normal binning:
//     `prism_render_shading::gi::world_space::octahedral::dir_to_oct`.
//   * RIS/GRIS reservoir estimator (luminance, geometric term, target function,
//     merge, finalize): `prism_render_shading::gi::screen_probe::restir` and
//     `prism_render_shading::gi::world_restir::world_reservoir`.
// The byte layouts of `WorldRestirReservoir` and `FillParams` are frozen by
// `shading::world_restir::abi` (80-byte five-lane slot, 64-byte four-lane
// immediate); the `shader_tests` ABI guards assert they stay in lockstep.
//
// u64 parity: WGSL has no 64-bit integers, so the SHARC hash — which the golden
// evaluates in `u64` — is reproduced here with `vec2<u32>` (`.x` = low 32 bits,
// `.y` = high 32 bits) and hand-rolled 64-bit multiply / shift / xor. A Rust
// mirror of this exact 2x32 arithmetic is asserted bit-equal against the golden
// `hash_key` / `checksum` / `bucket_index` in `shader_tests`, which is the only
// CPU-checkable proof of the port (the sandbox has no GPU). The bucket modulo
// assumes `capacity <= 2^31` (default 131072) so `rem * 2 + bit` never overflows
// a `u32`.
//
// Finite-math parity: WGSL lacks `isNan` / `isFinite`, so the golden's
// `x.is_finite() && x > MIN_POSITIVE` guards are reduced (matching the
// `world_space_gi_probe_update.wesl` convention) to `x > MIN_POSITIVE` and the
// `t.is_finite()` acceptance tests to the `t == t` NaN probe; `!(a > b)` carries
// the golden's `NaN`/`<=` fallthrough. Stored reservoir geometry is finite by
// construction (produced by the seed pass), so the surviving `inf`-only
// divergences are unreachable in practice and are noted at each site.

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
// Rec. 709 luminance weights (golden `LUMA_R/G/B`).
const LUMA_R: f32 = 0.212639;
const LUMA_G: f32 = 0.715169;
const LUMA_B: f32 = 0.072192;

// 64-bit mixing constants as (low, high) u32 pairs.
//   SEED_KEY  = 0x9e3779b97f4a7c15 (bucket hash seed + level multiplier)
//   SEED_SUM  = 0xc2b2ae3d27d4eb4f (checksum seed)
//   MIX_C1    = 0xff51afd7ed558ccd (fmix64 c1 + normal multiplier)
//   MIX_C2    = 0xc4ceb9fe1a85ec53 (fmix64 c2)
const SEED_KEY: vec2<u32> = vec2<u32>(0x7f4a7c15u, 0x9e3779b9u);
const SEED_SUM: vec2<u32> = vec2<u32>(0x27d4eb4fu, 0xc2b2ae3du);
const MIX_C1: vec2<u32> = vec2<u32>(0xed558ccdu, 0xff51afd7u);
const MIX_C2: vec2<u32> = vec2<u32>(0x1a85ec53u, 0xc4ceb9feu);
const LEVEL_MUL: vec2<u32> = vec2<u32>(0x7f4a7c15u, 0x9e3779b9u);
const NORMAL_MUL: vec2<u32> = vec2<u32>(0xed558ccdu, 0xff51afd7u);

// One hashed world-cell reservoir slot — std430 twin of
// `abi::GpuWorldRestirReservoir` (five vec4 lanes, 80 bytes).
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

// Fill immediate block — std430 twin of `abi::GpuWorldRestirFillParams`
// (four vec4 lanes, 64 bytes).
struct FillParams {
    // camera_position.xyz + capacity.
    camera_capacity: vec4<f32>,
    // jitter.xyz + light_count.
    jitter_lights: vec4<f32>,
    // base_cell_size, level_scale, normal_resolution, m_cap.
    grid_tunables: vec4<f32>,
    // intensity, frame, spatial_samples, spatial_radius.
    reuse_tunables: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> src_reservoirs: array<WorldRestirReservoir>;
@group(0) @binding(1) var<storage, read_write> dst_reservoirs: array<WorldRestirReservoir>;

var<immediate> params: FillParams;

// A fully-resolved SHARC hash-grid cell identity (golden `HashGridKey`).
struct Key {
    coord: vec3<i32>,
    level: i32,
    normal_bin: u32,
}

// A GI path sample payload (golden `GiSample`).
struct Sample {
    visible_point: vec3<f32>,
    visible_normal: vec3<f32>,
    sample_point: vec3<f32>,
    sample_normal: vec3<f32>,
    radiance: vec3<f32>,
}

// ---------------------------------------------------------------------------
// 64-bit (u64-as-2x u32) arithmetic — reproduces the golden `u64` hash math.
// ---------------------------------------------------------------------------

// Full 32x32 -> 64 bit unsigned product via 16-bit schoolbook decomposition.
// Every intermediate is proven to stay inside a u32 (max carry case
// 65535*65535 + 65535 + 65535 = 2^32 - 1), so WGSL's wrapping u32 arithmetic is
// exact here.
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

// Low 64 bits of a 64x64 unsigned product (golden `wrapping_mul`): the two
// cross terms only reach the high word, matching u64 wraparound.
fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = mul32x32_64(a.x, b.x);
    return vec2<u32>(p.x, p.y + a.x * b.y + a.y * b.x);
}

// Logical right shift of a 64-bit value by 33 bits (golden `x >> 33`): the
// result keeps only bits 33..63, which all live in the source high word.
fn u64_shr33(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.y >> 1u, 0u);
}

fn u64_xor(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

// MurmurHash3 64-bit finalizer (golden `fmix64`).
fn fmix64(x_in: vec2<u32>) -> vec2<u32> {
    var x = x_in;
    x = u64_xor(x, u64_shr33(x));
    x = u64_mul(x, MIX_C1);
    x = u64_xor(x, u64_shr33(x));
    x = u64_mul(x, MIX_C2);
    x = u64_xor(x, u64_shr33(x));
    return x;
}

// Biases a signed cell axis into its low 21 bits (golden `encode`): the low 21
// bits of `(bitcast<u32>(v) + COORD_BIAS)` equal those of the golden i64 math
// because addition's low bits are width independent.
fn encode_axis(v: i32) -> u32 {
    return (bitcast<u32>(v) + COORD_BIAS) & COORD_MASK;
}

// Packs a key's fields into a 64-bit value prior to hashing (golden `pack`).
// Axis layout: x in bits [0,20], y in [21,41], z in [42,62].
fn pack_key(key: Key) -> vec2<u32> {
    let ex = encode_axis(key.coord.x);
    let ey = encode_axis(key.coord.y);
    let ez = encode_axis(key.coord.z);
    let coord = vec2<u32>(ex | (ey << 21u), (ey >> 11u) | (ez << 10u));
    let level = u64_mul(vec2<u32>(u32(key.level), 0u), LEVEL_MUL);
    let normal = u64_mul(vec2<u32>(key.normal_bin, 0u), NORMAL_MUL);
    return u64_xor(u64_xor(coord, level), normal);
}

// Full 64-bit bucket hash of a key (golden `hash_key`).
fn hash_key(key: Key) -> vec2<u32> {
    return fmix64(u64_xor(pack_key(key), SEED_KEY));
}

// Independent 32-bit collision checksum (golden `checksum`): the avalanched
// high and low words are folded together.
fn key_checksum(key: Key) -> u32 {
    let h = fmix64(u64_xor(pack_key(key), SEED_SUM));
    return h.y ^ h.x;
}

// Maps a 64-bit hash onto a bucket in `capacity` slots (golden `bucket_index`
// `hash % max(capacity, 1)`). Bit-serial long division from the most
// significant bit; assumes `capacity <= 2^31` so the running remainder never
// overflows a u32.
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

// Normalises `n`, falling back to +Z for a degenerate input (golden
// `normalize_or_z`).
fn normalize_or_z(n: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(n, n);
    if (len_sq > MIN_POSITIVE) {
        return n * inverseSqrt(len_sq);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

// Quantises a normal into an octahedral bin in [0, resolution^2) (golden
// `quantize_normal`).
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

// Edge length of a cell at `level` (golden `cell_size_at_level`).
fn cell_size_at_level(base_cell_size: f32, level: i32) -> f32 {
    let base = safe_cell_size(base_cell_size);
    let lvl = f32(clamp(level, 0, MAX_LEVEL));
    return base * exp2(lvl);
}

// Chooses the grid level for a position given the camera (golden `grid_level`).
// `!(level_scale > 0)` / `!(dist > base)` carry the golden's NaN + in-cell
// fallthrough to level 0.
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

// Quantises a world position into its (cell_coord, level) pair (golden
// `quantize_position`). A NaN position collapses to the origin cell.
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
// RIS/GRIS estimator (golden `restir` + `world_reservoir`).
// ---------------------------------------------------------------------------

fn luminance(rgb: vec3<f32>) -> f32 {
    return LUMA_R * max(rgb.x, 0.0) + LUMA_G * max(rgb.y, 0.0) + LUMA_B * max(rgb.z, 0.0);
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
        // `term == term` rejects NaN (golden `is_finite`); a finite >=0 term
        // passes through.
        return select(0.0, max(term, 0.0), term == term);
    }
    return 0.0;
}

// Scalar resampling target p_hat (golden `target_function`).
fn target_function(s: Sample) -> f32 {
    let g = geometric_term(s.visible_point, s.visible_normal, s.sample_point, s.sample_normal);
    let t = luminance(s.radiance) * g;
    return select(0.0, max(t, 0.0), t == t);
}

// Reconnection-shift target: keep the neighbour's secondary point/normal/
// radiance, re-evaluate from the destination's visible point (golden
// `reconnection_target`).
fn reconnection_target(neighbor: Sample, visible_point: vec3<f32>, visible_normal: vec3<f32>) -> f32 {
    var shifted = neighbor;
    shifted.visible_point = visible_point;
    shifted.visible_normal = visible_normal;
    return target_function(shifted);
}

// ---------------------------------------------------------------------------
// Deterministic per-slot RNG (GPU-authored; the golden takes uniforms as
// inputs, so any in-[0,1) stream keeps each merge arm-for-arm).
// ---------------------------------------------------------------------------

fn fmix32(x_in: u32) -> u32 {
    var x = x_in;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

// Uniform in [0, 1) seeded by (frame, slot, counter).
fn rng01(frame: u32, slot: u32, counter: u32) -> f32 {
    let h = fmix32(frame * 0x9e3779b9u + fmix32(slot * 0x85ebca6bu + counter));
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

// ---------------------------------------------------------------------------
// Reservoir slot (de)serialisation.
// ---------------------------------------------------------------------------

fn load_sample(r: WorldRestirReservoir) -> Sample {
    var s: Sample;
    s.visible_point = r.visible_point_w.xyz;
    s.visible_normal = r.visible_normal_m.xyz;
    s.sample_point = r.sample_point_checksum.xyz;
    s.sample_normal = r.sample_normal_valid.xyz;
    s.radiance = r.radiance_pad.xyz;
    return s;
}

fn reservoir_valid(r: WorldRestirReservoir) -> bool {
    return bitcast<u32>(r.sample_normal_valid.w) != 0u;
}

fn reservoir_checksum(r: WorldRestirReservoir) -> u32 {
    return bitcast<u32>(r.sample_point_checksum.w);
}

@compute @workgroup_size(64, 1, 1)
fn fill_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Integer tunables share the float lanes of the frozen `FillParams`
    // ABI (`abi::GpuWorldRestirFillParams`), which the host sends as raw
    // u32/i32 bit patterns; reinterpret the bits with `bitcast` rather than
    // a numeric `u32(...)` conversion (which would truncate the float value
    // of the bit pattern to a wrong count).
    let capacity = bitcast<u32>(params.camera_capacity.w);
    let slot = gid.x;
    if (slot >= capacity) {
        return;
    }

    let center = src_reservoirs[slot];

    // An empty slot carries no sample to pool neighbours into: copy it through
    // so the ping-pong preserves vacancy / seeded temporal state.
    if (!reservoir_valid(center)) {
        dst_reservoirs[slot] = center;
        return;
    }

    let camera = params.camera_capacity.xyz;
    let jitter = params.jitter_lights.xyz;
    let base_cell_size = params.grid_tunables.x;
    let level_scale = params.grid_tunables.y;
    let normal_resolution = bitcast<u32>(params.grid_tunables.z);
    let m_cap = params.grid_tunables.w;
    let frame = bitcast<u32>(params.reuse_tunables.y);
    let spatial_samples = bitcast<u32>(params.reuse_tunables.z);
    let spatial_radius = bitcast<i32>(params.reuse_tunables.w);

    // Recompute this cell's SHARC key from its stored geometry so neighbour
    // keys can be addressed in the ring below.
    let visible_point = center.visible_point_w.xyz;
    let visible_normal = center.visible_normal_m.xyz;
    var center_key = quantize_position(visible_point, camera, base_cell_size, level_scale, jitter);
    center_key.normal_bin = quantize_normal(visible_normal, normal_resolution);

    // Accumulator reservoir seeded from the center (golden `spatial_reuse` start
    // from a copy of the center). The ABI stores only `W` and `m`, so the
    // streaming `w_sum` is reconstructed exactly via the finalize inverse
    // `w_sum = W * m * target_function(selected)`.
    var acc_sample = load_sample(center);
    var acc_m = center.visible_normal_m.w;
    let center_W = center.visible_point_w.w;
    let center_p_hat = target_function(acc_sample);
    var acc_w_sum = center_W * acc_m * center_p_hat;

    // Pool a jittered ring of spatial neighbours under GRIS reuse.
    let radius = max(spatial_radius, 0);
    for (var k: u32 = 0u; k < spatial_samples; k = k + 1u) {
        if (radius <= 0) {
            break;
        }
        // Jitter a non-zero integer offset within [-radius, radius]^3.
        let span = u32(2 * radius + 1);
        let rx = i32(fmix32(frame * 0x27d4eb4fu + slot * 0x9e3779b9u + k * 0x165667b1u) % span) - radius;
        let ry = i32(fmix32(frame * 0x85ebca6bu + slot * 0xc2b2ae35u + k * 0x27d4eb2fu) % span) - radius;
        let rz = i32(fmix32(frame * 0xc2b2ae35u + slot * 0x165667b1u + k * 0x85ebca6bu) % span) - radius;
        var offset = vec3<i32>(rx, ry, rz);
        if (offset.x == 0 && offset.y == 0 && offset.z == 0) {
            offset.x = 1; // nudge off-center so a neighbour is always distinct
        }

        var nkey = center_key;
        nkey.coord = center_key.coord + offset;
        let ncs = key_checksum(nkey);
        let base = bucket_index(nkey, capacity);

        // Open-addressed linear probe for the neighbour slot (golden
        // `find_slot`): stop at the first vacancy, accept on checksum match.
        var found: bool = false;
        var neighbor: WorldRestirReservoir;
        let steps = min(PROBE_LIMIT, capacity);
        for (var i: u32 = 0u; i < steps; i = i + 1u) {
            let idx = (base + i) % capacity;
            let cand = src_reservoirs[idx];
            if (!reservoir_valid(cand)) {
                break;
            }
            if (reservoir_checksum(cand) == ncs) {
                neighbor = cand;
                found = true;
                break;
            }
        }
        if (!found) {
            continue;
        }

        // GRIS merge (golden `Reservoir::merge` via `merge_spatial`): confidence
        // always accumulates; the induced weight `m * p_hat * W` drives the
        // replacement test.
        let neighbor_sample = load_sample(neighbor);
        let neighbor_m = neighbor.visible_normal_m.w;
        let neighbor_W = neighbor.visible_point_w.w;
        if (!(neighbor_m > 0.0)) {
            continue;
        }
        acc_m = acc_m + neighbor_m;
        let p_hat = reconnection_target(neighbor_sample, visible_point, visible_normal);
        let rw = neighbor_m * max(p_hat, 0.0) * max(neighbor_W, 0.0);
        if (!(rw > 0.0)) {
            continue;
        }
        acc_w_sum = acc_w_sum + rw;
        let u = rng01(frame, slot, k);
        if (u * acc_w_sum <= rw) {
            acc_sample = neighbor_sample;
        }
    }

    // Cap confidence (golden `cap_confidence`) then finalize `W` from the
    // surviving sample's own target density (golden `finalize`).
    if (m_cap >= 0.0 && acc_m > m_cap) {
        acc_m = m_cap;
    }
    let final_p_hat = target_function(acc_sample);
    var final_w: f32 = 0.0;
    if (acc_m > 0.0 && (acc_w_sum == acc_w_sum) && final_p_hat > 0.0) {
        let w = (acc_w_sum / acc_m) / final_p_hat;
        final_w = select(0.0, w, (w == w) && (w >= 0.0));
    }

    // Serialise the pooled reservoir into the destination slot, preserving the
    // slot's SHARC checksum and validity.
    var out: WorldRestirReservoir;
    out.visible_point_w = vec4<f32>(acc_sample.visible_point, final_w);
    out.visible_normal_m = vec4<f32>(acc_sample.visible_normal, acc_m);
    out.sample_point_checksum = vec4<f32>(acc_sample.sample_point, bitcast<f32>(reservoir_checksum(center)));
    out.sample_normal_valid = vec4<f32>(acc_sample.sample_normal, bitcast<f32>(1u));
    out.radiance_pad = vec4<f32>(acc_sample.radiance, 0.0);
    dst_reservoirs[slot] = out;
}
