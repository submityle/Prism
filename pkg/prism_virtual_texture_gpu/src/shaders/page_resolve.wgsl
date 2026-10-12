// Virtual-texture material page resolve.
//
// One invocation resolves one requested virtual page. It probes the exact
// page, then progressively coarser covering pages up to `coarsest_mip`.
// Output words are (physical slot, resolved mip, feedback), where MISS in the
// first two words means that no resident page exists and feedback is 1.
// The binary search is intentionally identical to page_lookup.wgsl.

const ENTRY_WORDS: u32 = 4u;
const OUTPUT_WORDS: u32 = 3u;
const MISS: u32 = 0xFFFFFFFFu;

struct Params {
    entry_count: u32,
    query_count: u32,
    coarsest_mip: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> table: array<u32>;
// Three compare words per requested page.
@group(0) @binding(2) var<storage, read> queries: array<u32>;
// Three output words per request: slot, resolved mip, feedback flag.
@group(0) @binding(3) var<storage, read_write> results: array<u32>;

fn compare(e0: u32, e1: u32, e2: u32, t0: u32, t1: u32, t2: u32) -> i32 {
    if (e0 < t0) { return -1; }
    if (e0 > t0) { return 1; }
    if (e1 < t1) { return -1; }
    if (e1 > t1) { return 1; }
    if (e2 < t2) { return -1; }
    if (e2 > t2) { return 1; }
    return 0;
}

fn exact_slot(t0: u32, t1: u32, t2: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = params.entry_count;
    loop {
        if (lo >= hi) { break; }
        let mid = lo + (hi - lo) / 2u;
        let base = mid * ENTRY_WORDS;
        let ord = compare(
            table[base], table[base + 1u], table[base + 2u], t0, t1, t2
        );
        if (ord < 0) {
            lo = mid + 1u;
        } else if (ord > 0) {
            hi = mid;
        } else {
            return table[base + 3u];
        }
    }
    return MISS;
}

fn shifted_page(value: u32, shift: u32) -> u32 {
    if (shift >= 16u) { return 0u; }
    return value >> shift;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if (q >= params.query_count) { return; }

    let qbase = q * 3u;
    let t0 = queries[qbase];
    let requested_w1 = queries[qbase + 1u];
    let requested_w2 = queries[qbase + 2u];
    let requested_mip = requested_w1 >> 24u;
    let layer_bits = requested_w1 & 0x00FFFF00u;
    let requested_x = requested_w2 >> 16u;
    let requested_y = requested_w2 & 0x0000FFFFu;
    let last_mip = max(params.coarsest_mip, requested_mip);

    var mip = requested_mip;
    var slot = MISS;
    var resolved_mip = MISS;
    loop {
        if (mip > last_mip) { break; }
        let delta = mip - requested_mip;
        let candidate_w1 = (mip << 24u) | layer_bits;
        let candidate_w2 = (shifted_page(requested_x, delta) << 16u)
            | shifted_page(requested_y, delta);
        slot = exact_slot(t0, candidate_w1, candidate_w2);
        if (slot != MISS) {
            resolved_mip = mip;
            break;
        }
        // `mip` is encoded as an 8-bit field. Avoid wrapping from 255 to 0
        // when the caller asks for the full representable range.
        if (mip == last_mip) { break; }
        mip = mip + 1u;
    }

    let out = q * OUTPUT_WORDS;
    results[out] = slot;
    results[out + 1u] = resolved_mip;
    // Exact hits require no refinement feedback; a fallback or total miss does.
    results[out + 2u] = select(1u, 0u, resolved_mip == requested_mip);
}
