// Physical page-data indexed-gather twin: reads single words out of physical
// pool slots on-device, mirroring the CPU golden `PageStorage::fetch`.
//
// After `PageStorage::upload` has placed page contents into their slots, the
// rasterizer reads one word by `(slot, word)`. This kernel resolves a whole
// batch of such reads at once: query `i` fetches word `fetch_words[i]` of slot
// `fetch_slots[i]` and writes it to `out[i]`. One thread per query reproduces
// the reference's addressing `slot * page_words + word` exactly, so every
// gathered word is bit-identical to `PageStorage::fetch`.
//
// A query whose slot or word is out of range writes `0`, so a caller that only
// issues in-range reads (the reference's `fetch` returns `Some` for those) gets
// a bit-exact match, while a stray out-of-range read is defined rather than
// undefined.
//
// All arithmetic is 32-bit integer indexing in the portable core-WGSL subset -
// no optional feature - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard indexed gather and `wgpu` compute dispatch; no Unreal
// Engine source or derived code.

struct Params {
    // Number of gather queries in this batch.
    fetch_count: u32,
    // Words per physical slot; the pool's fixed page stride.
    page_words: u32,
    // Number of slots the pool holds; reads at or beyond this yield `0`.
    capacity: u32,
    // Padding to a 16-byte uniform tail.
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Source physical slot for each query.
@group(0) @binding(1) var<storage, read> fetch_slots: array<u32>;
// Word offset within the slot for each query.
@group(0) @binding(2) var<storage, read> fetch_words: array<u32>;
// The flat physical pool, slot-major: slot `s` word `w` at `s * page_words + w`.
@group(0) @binding(3) var<storage, read> pool: array<u32>;
// One gathered word per query.
@group(0) @binding(4) var<storage, read_write> out: array<u32>;

// One thread per gather query.
@compute @workgroup_size(64, 1, 1)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.fetch_count) {
        return;
    }
    let s = fetch_slots[i];
    let w = fetch_words[i];
    // Out-of-range reads are defined to yield `0`, matching a caller that only
    // issues the in-range reads for which the reference `fetch` returns `Some`.
    if (s >= params.capacity || w >= params.page_words) {
        out[i] = 0u;
        return;
    }
    out[i] = pool[s * params.page_words + w];
}
