// Flag-driven stream compaction: the scatter half of a compact primitive.
//
// Given a 0/1 keep-flag stream and its exclusive prefix scan (computed by
// `shaders/scan.wgsl`), the `scatter` kernel gathers every flagged `data` entry
// into a dense `out_compact` array. The exclusive scan of the flags is exactly
// the destination index of each kept element, so a single guarded write per
// invocation performs the compaction with no atomics and no contention.
//
// This kernel lives in its own module because its bindings differ from the scan
// kernels': it needs read-only flags, offsets, and payload plus a writable
// dense output, which cannot share the scan module's @group(0) binding numbers.
//
// Provenance: flag-driven stream compaction is a classical, openly published
// parallel primitive (Harris, Sengupta, Owens, GPU Gems 3, 2007). No Unreal
// Engine source or derived code.

struct Params {
    // Number of input elements (equal for flags, offsets, and data).
    n: u32,
    // Padding to a 16-byte boundary.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Keep flags: a non-zero entry keeps the element.
@group(0) @binding(1) var<storage, read> flags: array<u32>;
// Exclusive scan of `flags`: the destination index of each kept element.
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
// Source payload, one entry per flag.
@group(0) @binding(3) var<storage, read> data: array<u32>;
// Dense output; its length is the number of set flags (the scan grand total).
@group(0) @binding(4) var<storage, read_write> out_compact: array<u32>;

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    if (flags[i] != 0u) {
        out_compact[offsets[i]] = data[i];
    }
}
