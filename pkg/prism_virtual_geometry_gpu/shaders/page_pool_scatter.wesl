// Physical page-data scatter-upload twin: places streamed page contents into
// their physical pool slots on-device, mirroring the CPU golden
// `PageStorage::upload`.
//
// `PageStorage` (in `prism_render_architecture::paging`) is a flat pool of
// `capacity` slots, each `page_words` 32-bit words, so slot `s`'s data is the
// contiguous span `[s * page_words, (s + 1) * page_words)`. Streaming a page in
// writes one slot's span. This kernel applies a whole upload batch at once:
// record `r` targets slot `slots[r]` and carries its `page_words` words
// row-major at `data[r * page_words + w]`. One thread per `(record, word)`
// copies a single word into the pool, so the buffer image after the dispatch is
// bit-identical to the reference's `as_words()` for every uploaded slot.
//
// A target slot at or beyond `capacity` is skipped, matching the reference's
// out-of-range rejection: no word is written for it.
//
// All arithmetic is 32-bit integer indexing in the portable core-WGSL subset -
// no optional feature - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard indexed scatter and `wgpu` compute dispatch; no Unreal
// Engine source or derived code.

struct Params {
    // Number of upload records (slot + page payload) in this batch.
    record_count: u32,
    // Words per physical slot; the pool's fixed page stride.
    page_words: u32,
    // Number of slots the pool holds; targets at or beyond this are skipped.
    capacity: u32,
    // Padding to a 16-byte uniform tail.
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Target physical slot for each upload record.
@group(0) @binding(1) var<storage, read> slots: array<u32>;
// Row-major page payloads: record `r` word `w` at `data[r * page_words + w]`.
@group(0) @binding(2) var<storage, read> data: array<u32>;
// The flat physical pool, slot-major: slot `s` word `w` at `s * page_words + w`.
@group(0) @binding(3) var<storage, read_write> pool: array<u32>;

// One thread per `(record, word)` pair, flattened row-major over the batch.
@compute @workgroup_size(64, 1, 1)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.record_count * params.page_words;
    let t = gid.x;
    if (t >= total) {
        return;
    }
    let r = t / params.page_words;
    let w = t % params.page_words;
    let dst = slots[r];
    // Out-of-range target slots are skipped, exactly as the reference rejects
    // an upload whose slot is beyond capacity.
    if (dst >= params.capacity) {
        return;
    }
    pool[dst * params.page_words + w] = data[t];
}
