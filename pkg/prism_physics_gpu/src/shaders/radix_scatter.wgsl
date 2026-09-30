// Stable scatter for one least-significant-digit radix pass.
//
// Runs after the count pass has built the per-block digit histogram and the
// host has exclusive-scanned it in digit-major order into `offsets`, so
// `offsets[digit * num_blocks + block]` is the global output index where this
// block's first key carrying `digit` must land. Each workgroup reprocesses its
// own TILE-wide block, recomputes every lane's digit, and derives a stable
// local rank — the number of earlier lanes in this block that share the same
// digit — then writes the key and its payload to `offsets[...] + local_rank`.
//
// The local rank is counted in lane order rather than allocated atomically:
// atomics would hand out ranks in a nondeterministic order and break the
// stability that the multi-pass LSD scheme depends on. Because the write index
// of every element is a deterministic function of the scanned offsets and the
// in-block order, the scatter is a pure permutation and matches the
// `cpu_radix_sort_*` golden twins bit-for-bit.
//
// Provenance: stable per-block scatter with a scanned digit-major histogram is
// a classical, openly published GPU radix-sort technique (Satish, Harris,
// Garland 2009). No Unreal Engine source or derived code.

// Digit width in bits; must match `RADIX_BITS` in `radix/config.rs`.
const RADIX_BITS: u32 = 8u;
// Buckets per pass; must match `RADIX` in `radix/config.rs`.
const RADIX: u32 = 256u;
// Digit mask; must match `RADIX_MASK` in `radix/config.rs`.
const RADIX_MASK: u32 = RADIX - 1u;
// Keys per block; must match `TILE` in `radix/config.rs`.
const TILE: u32 = 256u;
// Sentinel digit for out-of-range lanes; distinct from every real 8-bit digit.
const INVALID_DIGIT: u32 = RADIX;

struct Params {
    // Number of keys.
    n: u32,
    // Which digit this pass extracts (0 = least significant).
    pass_index: u32,
    // Number of blocks the keys are tiled into.
    num_blocks: u32,
    // Padding to a 16-byte boundary.
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Keys entering this pass.
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
// Payloads entering this pass, one per key.
@group(0) @binding(2) var<storage, read> vals_in: array<u32>;
// Exclusive-scanned per-(digit, block) output bases, digit-major.
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
// Keys leaving this pass, in stable digit order.
@group(0) @binding(4) var<storage, read_write> keys_out: array<u32>;
// Payloads leaving this pass, following their keys.
@group(0) @binding(5) var<storage, read_write> vals_out: array<u32>;

// Every lane's digit for this block, so ranks can be counted in lane order.
var<workgroup> sdigit: array<u32, 256>;

@compute @workgroup_size(256)
fn scatter(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let t = lid.x;
    let block = wgid.x;
    let idx = block * TILE + t;
    let valid = idx < params.n;

    // Out-of-range lanes take the sentinel so they never match a real digit and
    // never contribute to another lane's rank.
    var d = INVALID_DIGIT;
    if (valid) {
        d = (keys_in[idx] >> (params.pass_index * RADIX_BITS)) & RADIX_MASK;
    }
    sdigit[t] = d;
    workgroupBarrier();

    if (valid) {
        // Stable rank: count earlier lanes in this block sharing this digit.
        var local_rank = 0u;
        for (var j = 0u; j < t; j = j + 1u) {
            if (sdigit[j] == d) {
                local_rank = local_rank + 1u;
            }
        }
        let pos = offsets[d * params.num_blocks + block] + local_rank;
        keys_out[pos] = keys_in[idx];
        vals_out[pos] = vals_in[idx];
    }
}
