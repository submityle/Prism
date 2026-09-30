// Per-block digit histogram for one least-significant-digit radix pass.
//
// Each workgroup owns one TILE-wide block of keys and tallies how many carry
// each of the RADIX digit values in the current pass. The counts are written in
// digit-major order — `block_hist[digit * num_blocks + block]` — so a single
// exclusive scan of the whole array yields the global output base of every
// (digit, block) group: all of digit 0's blocks precede digit 1's, and so on,
// which is exactly the stable bucket layout the scatter pass writes into.
//
// The histogram is exact integer counting, so the kernel matches the counting
// step of the `cpu_radix_sort_*` golden twins bit-for-bit.
//
// Provenance: the per-block histogram decomposition of an LSD radix sort is a
// classical, openly published GPU technique (Satish, Harris, Garland 2009). No
// Unreal Engine source or derived code.

// Digit width in bits; must match `RADIX_BITS` in `radix/config.rs`.
const RADIX_BITS: u32 = 8u;
// Buckets per pass; must match `RADIX` in `radix/config.rs`.
const RADIX: u32 = 256u;
// Digit mask; must match `RADIX_MASK` in `radix/config.rs`.
const RADIX_MASK: u32 = RADIX - 1u;
// Keys per block; must match `TILE` in `radix/config.rs`.
const TILE: u32 = 256u;

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
// Keys to histogram.
@group(0) @binding(1) var<storage, read> keys: array<u32>;
// Per-(digit, block) counts in digit-major order.
@group(0) @binding(2) var<storage, read_write> block_hist: array<u32>;

// One atomic bin per digit, shared across the block.
var<workgroup> hist: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn count(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let t = lid.x;
    let block = wgid.x;

    // The workgroup width equals RADIX, so each lane clears exactly one bin.
    atomicStore(&hist[t], 0u);
    workgroupBarrier();

    let idx = block * TILE + t;
    if (idx < params.n) {
        let d = (keys[idx] >> (params.pass_index * RADIX_BITS)) & RADIX_MASK;
        atomicAdd(&hist[d], 1u);
    }
    workgroupBarrier();

    // Export the block's tally for digit `t` in digit-major order.
    block_hist[t * params.num_blocks + block] = atomicLoad(&hist[t]);
}
