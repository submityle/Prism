// Work-efficient exclusive prefix sum (scan) over one level of the pyramid.
//
// Two kernels cooperate under host orchestration:
//
//   * `scan_block` exclusively scans one BLOCK-wide slice per workgroup in
//     shared memory with the Blelloch up-sweep/down-sweep, writes the per-block
//     total to `block_sums[workgroup]`, and writes the block-local exclusive
//     prefixes back in place. Two elements are processed per thread.
//   * `add_offsets` adds the globally scanned offset of each block back into its
//     elements, turning the block-local prefixes of `scan_block` into a single
//     global exclusive scan. The host runs `scan_block` down the level pyramid
//     and `add_offsets` back up it.
//
// The scan folds with u32 addition, which is associative and wraps identically
// on host and device, so the kernels match the `cpu_exclusive_scan` golden twin
// bit-for-bit rather than within a tolerance. Stream compaction is layered on
// top by `shaders/scan_scatter.wgsl`, which consumes this scan's offsets.
//
// Provenance: the Blelloch work-efficient scan is a classical, openly published
// parallel primitive (Blelloch 1990; Harris, Sengupta, Owens, GPU Gems 3,
// 2007). No Unreal Engine source or derived code.

// Elements scanned per workgroup; must match `BLOCK` in `scan/config.rs`.
const BLOCK: u32 = 512u;
// Half a block: threads per workgroup, and the stride between the two elements
// each thread owns.
const HALF: u32 = BLOCK / 2u;

struct Params {
    // Number of elements in the level this dispatch scans.
    n: u32,
    // Padding to a 16-byte boundary.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// The level being scanned in place: input on entry, block-local exclusive
// prefixes on exit.
@group(0) @binding(1) var<storage, read_write> values: array<u32>;
// Per-workgroup totals (written by `scan_block`) or globally scanned block
// offsets (read by `add_offsets`).
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

// Shared staging for one block's Blelloch scan.
var<workgroup> temp: array<u32, 512>;

@compute @workgroup_size(256)
fn scan_block(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let t = lid.x;
    let block = wgid.x;
    let base = block * BLOCK;

    // Each thread loads its two elements; lanes past the end seed 0 so a partial
    // final block scans as if zero-padded.
    let ai = t;
    let bi = t + HALF;
    var va = 0u;
    var vb = 0u;
    if (base + ai < params.n) {
        va = values[base + ai];
    }
    if (base + bi < params.n) {
        vb = values[base + bi];
    }
    temp[ai] = va;
    temp[bi] = vb;

    // Up-sweep: reduce pairs into partial sums. `d` and `offset` are uniform, so
    // the barriers are reached by every lane; only the writes are guarded.
    var offset = 1u;
    for (var d = HALF; d > 0u; d = d >> 1u) {
        workgroupBarrier();
        if (t < d) {
            let l = offset * (2u * t + 1u) - 1u;
            let r = offset * (2u * t + 2u) - 1u;
            temp[r] = temp[r] + temp[l];
        }
        offset = offset * 2u;
    }

    // Record the block total and clear the root for the exclusive down-sweep.
    workgroupBarrier();
    if (t == 0u) {
        block_sums[block] = temp[BLOCK - 1u];
        temp[BLOCK - 1u] = 0u;
    }

    // Down-sweep: propagate partials back down into exclusive prefixes.
    for (var d = 1u; d < BLOCK; d = d * 2u) {
        offset = offset >> 1u;
        workgroupBarrier();
        if (t < d) {
            let l = offset * (2u * t + 1u) - 1u;
            let r = offset * (2u * t + 2u) - 1u;
            let carry = temp[l];
            temp[l] = temp[r];
            temp[r] = temp[r] + carry;
        }
    }

    workgroupBarrier();
    if (base + ai < params.n) {
        values[base + ai] = temp[ai];
    }
    if (base + bi < params.n) {
        values[base + bi] = temp[bi];
    }
}

@compute @workgroup_size(256)
fn add_offsets(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let t = lid.x;
    let block = wgid.x;
    let base = block * BLOCK;
    // `block_sums` here holds the globally scanned offset of each block.
    let add = block_sums[block];

    let ai = base + t;
    let bi = base + t + HALF;
    if (ai < params.n) {
        values[ai] = values[ai] + add;
    }
    if (bi < params.n) {
        values[bi] = values[bi] + add;
    }
}
