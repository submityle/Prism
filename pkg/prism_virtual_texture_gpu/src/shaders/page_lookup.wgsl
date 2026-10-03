// Virtual-texture page-table indirection lookup.
//
// One invocation resolves one sampled page coordinate to its physical slot by
// binary-searching the flat, key-ascending page table produced by the CPU
// golden `GpuPageTable`. Each entry is `ENTRY_WORDS` (4) words: three compare
// words `(w0, w1, w2)` followed by the physical slot. The compare words are
// laid out so an unsigned lexicographic compare of `(w0, w1, w2)` orders
// identically to the golden `TexturePageKey` ordering, so this search visits
// the exact same comparison sequence as `GpuPageTable::lookup` and returns the
// identical slot. A miss writes the sentinel `MISS` (0xFFFFFFFF).
//
// Integer-only: no floating point is involved anywhere, so device output equals
// the golden output bit-for-bit (exact equality, not tolerance).

const ENTRY_WORDS: u32 = 4u;
const MISS: u32 = 0xFFFFFFFFu;

struct Params {
    // Number of resident entries in the page table.
    entry_count: u32,
    // Number of page-coordinate queries to resolve.
    query_count: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Flat page table: `entry_count * ENTRY_WORDS` words, key-ascending.
@group(0) @binding(1) var<storage, read> table: array<u32>;
// Query compare words: three per query (`w0, w1, w2`).
@group(0) @binding(2) var<storage, read> queries: array<u32>;
// Resolved physical slot per query, or `MISS`.
@group(0) @binding(3) var<storage, read_write> slots: array<u32>;

// Returns -1 if entry < target, +1 if entry > target, 0 if equal, using the
// same unsigned lexicographic order as the golden `[u32; 3]::cmp`.
fn compare(e0: u32, e1: u32, e2: u32, t0: u32, t1: u32, t2: u32) -> i32 {
    if (e0 < t0) { return -1; }
    if (e0 > t0) { return 1; }
    if (e1 < t1) { return -1; }
    if (e1 > t1) { return 1; }
    if (e2 < t2) { return -1; }
    if (e2 > t2) { return 1; }
    return 0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if (q >= params.query_count) {
        return;
    }

    let qbase = q * 3u;
    let t0 = queries[qbase];
    let t1 = queries[qbase + 1u];
    let t2 = queries[qbase + 2u];

    var lo: u32 = 0u;
    var hi: u32 = params.entry_count;
    var result: u32 = MISS;
    // Bounded loop: the resident set never exceeds `entry_count`, so at most
    // ceil(log2(entry_count)) iterations run; the explicit break on `lo >= hi`
    // mirrors the golden `while lo < hi`.
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = lo + (hi - lo) / 2u;
        let base = mid * ENTRY_WORDS;
        let ord = compare(table[base], table[base + 1u], table[base + 2u], t0, t1, t2);
        if (ord < 0) {
            lo = mid + 1u;
        } else if (ord > 0) {
            hi = mid;
        } else {
            result = table[base + 3u];
            break;
        }
    }

    slots[q] = result;
}
