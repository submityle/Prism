// Per-block digit histogram for one least-significant-digit radix pass of the
// SER ray-coherence sort.
//
// Each workgroup owns one TILE-wide block of 64-bit CoherenceKeys (carried as
// two u32 words, `lo` and `hi`, because WGSL has no native 64-bit integer) and
// tallies how many carry each of the RADIX digit values in the current pass.
// `params.use_hi` selects which word the digit is read from and
// `params.digit_shift` selects the byte within that word, so one kernel serves
// all eight passes. The counts are written in digit-major order —
// `block_hist[digit * num_blocks + block]` — so a single exclusive scan of the
// whole array yields the global output base of every (digit, block) group,
// which is exactly the stable bucket layout the scatter pass writes into.
//
// The histogram is exact integer counting, so the kernel matches the counting
// step of the `radix_order` CPU golden in prism_render_architecture
// bit-for-bit.
//
// Provenance: the per-block histogram decomposition of an LSD radix sort is a
// classical, openly published GPU technique (Satish, Harris, Garland 2009). No
// Unreal Engine source or derived code.

// Buckets per pass; must match `RADIX` in `radix/config.rs`.
const RADIX: u32 = 256u;
// Digit mask; must match `RADIX_MASK` in `radix/config.rs`.
const RADIX_MASK: u32 = RADIX - 1u;
// Keys per block; must match `TILE` in `radix/config.rs`.
const TILE: u32 = 256u;

struct Params {
    // Number of keys.
    n: u32,
    // Bit offset of the digit within its 32-bit word (0, 8, 16, or 24).
    digit_shift: u32,
    // 0 selects the low word, 1 selects the high word.
    use_hi: u32,
    // Number of blocks the keys are tiled into.
    num_blocks: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Low 32-bit words of the keys to histogram.
@group(0) @binding(1) var<storage, read> keys_lo: array<u32>;
// High 32-bit words of the keys to histogram.
@group(0) @binding(2) var<storage, read> keys_hi: array<u32>;
// Per-(digit, block) counts in digit-major order.
@group(0) @binding(3) var<storage, read_write> block_hist: array<u32>;

// One atomic bin per digit, shared across the block.
var<workgroup> hist: array<atomic<u32>, 256>;

fn key_digit(idx: u32) -> u32 {
    var word = keys_lo[idx];
    if (params.use_hi != 0u) {
        word = keys_hi[idx];
    }
    return (word >> params.digit_shift) & RADIX_MASK;
}

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
        atomicAdd(&hist[key_digit(idx)], 1u);
    }
    workgroupBarrier();

    // Export the block's tally for digit `t` in digit-major order.
    block_hist[t * params.num_blocks + block] = atomicLoad(&hist[t]);
}
