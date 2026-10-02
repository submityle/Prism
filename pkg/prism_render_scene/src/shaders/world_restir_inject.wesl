// World-space ReSTIR DI — reservoir injection / hash-grid claim pass (GPU twin).
//
// One compute invocation per visible shading point. The pass hashes each point
// into its SHARC world cell, claims (or re-finds) the open-addressed reservoir
// slot owning that cell under a linear-probe window, and pre-seeds the slot
// with the cell's visible-point geometry (position + normal) and a `valid`
// flag. It writes no light sample or energy: the downstream seed pass reads
// these injected cells, streams light candidates under RIS, and the fill pass
// then pools them across spatial neighbours under GRIS. Injection is therefore
// the upstream producer that makes a slot addressable at all — without it the
// seed/fill tables stay empty.
//
// This kernel is an arm-for-arm port of the CPU golden:
//   * SHARC spatial-hash key + checksum + bucket:
//     `prism_render_shading::gi::world_restir::spatial_hash`.
//   * octahedral normal binning:
//     `prism_render_shading::gi::world_space::octahedral::dir_to_oct`.
//   * open-addressed slot claim (linear probe, first-wins, drop-on-full):
//     `prism_render_shading::gi::world_restir::world_reservoir`
//     (`WorldHashGrid::find_or_alloc` / `PROBE_LIMIT`).
// The byte layouts of `WorldRestirReservoir`, `InjectPoint`, and `InjectParams`
// are frozen by `shading::world_restir::abi` (80-byte five-lane slot, 32-byte
// two-lane point, 48-byte three-lane immediate); the `shader_tests` ABI guards
// assert they stay in lockstep, and the inject CPU mirror asserts the claim
// sequence (slot index + fresh/reuse/drop) is bit-identical to the golden grid.
//
// u64 parity: WGSL has no 64-bit integers, so the SHARC hash — which the golden
// evaluates in `u64` — is reproduced here with `vec2<u32>` (`.x` = low 32 bits,
// `.y` = high 32 bits) and hand-rolled 64-bit multiply / shift / xor, byte for
// byte identical to the fill/seed twins. A Rust mirror of this exact 2x32
// arithmetic is asserted bit-equal against the golden `hash_key` / `checksum` /
// `bucket_index` in `shader_tests`, the only CPU-checkable proof of the port
// (the sandbox has no GPU). The bucket modulo assumes `capacity <= 2^31`
// (default 131072) so `rem * 2 + bit` never overflows a `u32`.
//
// Race safety: the slot-state array is a parallel `array<atomic<u32>>` of
// per-slot checksum guards. A fresh claim is an `atomicCompareExchangeWeak`
// from `EMPTY_SLOT` to the key checksum; a spurious failure (reported old value
// still `EMPTY_SLOT`) retries the same slot, a genuine occupant breaks to the
// first-wins check so two invocations hashing to the same cell converge on one
// slot without double-claiming. A full probe window drops the point rather than
// evicting a live cell, matching the golden `find_or_alloc` returning `None`.

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

// Sentinel for an unclaimed slot-state entry. A cell whose raw checksum is
// exactly `0` is indistinguishable from empty (the golden `WorldHashGrid` keeps
// a separate `occupied` bool, so it has no such degeneracy); this `2^-32`
// collision is accepted rather than remapped, keeping inject / seed / fill all
// keyed on the identical raw `key_checksum`. The inject CPU mirror asserts the
// test keys never hit a zero checksum, so the claim order stays bit-exact.
const EMPTY_SLOT: u32 = 0u;

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
// `abi::GpuWorldRestirReservoir` (five vec4 lanes, 80 bytes); byte-identical to
// the seed / fill twins' `WorldRestirReservoir`.
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

// One visible shading point to inject — std430 twin of
// `abi::GpuWorldRestirInjectPoint` (two vec4 lanes, 32 bytes).
struct InjectPoint {
    // world position.xyz + one pad word.
    position_pad: vec4<f32>,
    // world normal.xyz + one pad word.
    normal_pad: vec4<f32>,
}

// Inject immediate block — std430 twin of `abi::GpuWorldRestirInjectParams`
// (three vec4 lanes, 48 bytes). The u32 lane is kept pure (no float-encoded
// counts) so the host ABI needs no bitcast.
struct InjectParams {
    // camera_position.xyz + base_cell_size.
    camera_base: vec4<f32>,
    // jitter.xyz + level_scale.
    jitter_level: vec4<f32>,
    // (capacity, point_count, normal_resolution, frame).
    dims: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> points: array<InjectPoint>;
@group(0) @binding(1) var<storage, read_write> reservoirs: array<WorldRestirReservoir>;
@group(0) @binding(2) var<storage, read_write> slot_state: array<atomic<u32>>;

var<immediate> params: InjectParams;

// A fully-resolved SHARC hash-grid cell identity (golden `HashGridKey`).
struct Key {
    coord: vec3<i32>,
    level: i32,
    normal_bin: u32,
}

// ---------------------------------------------------------------------------
// 64-bit (u64-as-2x u32) arithmetic — reproduces the golden `u64` hash math.
// ---------------------------------------------------------------------------

// Full 32x32 -> 64 bit unsigned product via 16-bit schoolbook decomposition.
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

// Low 64 bits of a 64x64 unsigned product (golden `wrapping_mul`).
fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = mul32x32_64(a.x, b.x);
    return vec2<u32>(p.x, p.y + a.x * b.y + a.y * b.x);
}

// Logical right shift of a 64-bit value by 33 bits (golden `x >> 33`).
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

// Biases a signed cell axis into its low 21 bits (golden `encode`).
fn encode_axis(v: i32) -> u32 {
    return (bitcast<u32>(v) + COORD_BIAS) & COORD_MASK;
}

// Packs a key's fields into a 64-bit value prior to hashing (golden `pack`).
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

// Independent 32-bit collision checksum (golden `checksum`).
fn key_checksum(key: Key) -> u32 {
    let h = fmix64(u64_xor(pack_key(key), SEED_SUM));
    return h.y ^ h.x;
}

// Maps a 64-bit hash onto a bucket in `capacity` slots (golden `bucket_index`).
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
// Injection: hash the point, claim its slot, pre-seed the cell geometry.
// ---------------------------------------------------------------------------

// Builds the injected reservoir: the cell's visible-point geometry plus the
// slot's SHARC checksum and the `valid` flag. No light sample / energy / W / m
// is written — the seed pass fills those from the frame's light list. The
// field layout is byte-for-byte what `seed_main` reads back
// (`visible_point_w.xyz` / `visible_normal_m.xyz` / `sample_point_checksum.w` /
// `sample_normal_valid.w`).
fn make_reservoir(position: vec3<f32>, normal: vec3<f32>, cs: u32) -> WorldRestirReservoir {
    var out: WorldRestirReservoir;
    out.visible_point_w = vec4<f32>(position, 0.0);
    out.visible_normal_m = vec4<f32>(normal, 0.0);
    out.sample_point_checksum = vec4<f32>(vec3<f32>(0.0), bitcast<f32>(cs));
    out.sample_normal_valid = vec4<f32>(vec3<f32>(0.0), bitcast<f32>(1u));
    out.radiance_pad = vec4<f32>(vec3<f32>(0.0), 0.0);
    return out;
}

@compute @workgroup_size(64, 1, 1)
fn inject_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    let point_count = params.dims.y;
    if (pid >= point_count) {
        return;
    }

    let capacity = params.dims.x;
    let normal_resolution = params.dims.z;
    let camera = params.camera_base.xyz;
    let base_cell_size = params.camera_base.w;
    let jitter = params.jitter_level.xyz;
    let level_scale = params.jitter_level.w;

    let point = points[pid];
    let position = point.position_pad.xyz;
    let normal = point.normal_pad.xyz;

    var key = quantize_position(position, camera, base_cell_size, level_scale, jitter);
    key.normal_bin = quantize_normal(normal, normal_resolution);

    let cs = key_checksum(key);
    let base = bucket_index(key, capacity);
    let steps = min(PROBE_LIMIT, capacity);

    // Open-addressed linear probe (golden `find_or_alloc`): claim the first
    // empty slot, or re-find the slot already owning this cell; a full window
    // drops the point.
    for (var i: u32 = 0u; i < steps; i = i + 1u) {
        let idx = (base + i) % capacity;
        var owner: u32 = EMPTY_SLOT;
        // Resolve this slot's ownership, retrying only `atomicCompareExchangeWeak`
        // spurious failures (reported old value still `EMPTY_SLOT`).
        loop {
            let claim = atomicCompareExchangeWeak(&slot_state[idx], EMPTY_SLOT, cs);
            if (claim.exchanged) {
                // Fresh claim: this invocation owns the slot, pre-seed it.
                reservoirs[idx] = make_reservoir(position, normal, cs);
                return;
            }
            if (claim.old_value != EMPTY_SLOT) {
                owner = claim.old_value;
                break;
            }
            // Spurious failure (slot still empty): retry the same slot.
        }
        if (owner == cs) {
            // This cell is already injected (first-wins): do not overwrite.
            return;
        }
        // Genuine collision with another cell: advance the probe window.
    }
    // Probe window full: drop the point rather than evict a live cell.
}
