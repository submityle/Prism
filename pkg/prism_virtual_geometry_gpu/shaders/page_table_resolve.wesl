// Physical page-table resolve twin: maps virtualized page keys to physical pool
// slots on-device, mirroring the CPU golden `PagePool::slot_of`.
//
// `PagePool` (in `prism_render_architecture::paging`) hands every resident page
// a slot in a bounded physical pool and exports the resident map as a
// key-ordered `(key, slot)` list. A GPU-driven pipeline needs the inverse
// lookup per cluster: given a page key, where does its data live? This kernel
// runs one thread per query key and binary-searches the sorted entry table for
// it, writing the matching slot or the unmapped sentinel on a miss.
//
// The key is `GeometryPageKey { asset, page }`, compared lexicographically
// (asset first, then page) exactly as the CPU `Ord` derive orders it, so the
// device and the reference agree on ordering and on every resolved slot. All
// arithmetic is 32-bit integer comparison in the portable core-WGSL subset -
// no optional feature - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard sorted-table binary search; no Unreal Engine source or
// derived code.

struct Params {
    // Number of valid `(key, slot)` rows in `entries`.
    entry_count: u32,
    // Number of query keys in `queries` and outputs in `slots`.
    query_count: u32,
    // Padding to a 16-byte uniform tail.
    pad0: u32,
    pad1: u32,
}

// One resident page mapping: its `GeometryPageKey` split into `asset`/`page`
// and the physical slot backing it. `pad` keeps a 16-byte stride matching the
// `bytemuck` upload struct.
struct Entry {
    asset: u32,
    page: u32,
    slot: u32,
    pad: u32,
}

// One query key: the `GeometryPageKey` to resolve. `8`-byte stride.
struct Query {
    asset: u32,
    page: u32,
}

// Sentinel written for a query that misses the resident table. Must equal the
// CPU `UNMAPPED_SLOT` (`u32::MAX`).
const UNMAPPED_SLOT: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> entries: array<Entry>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> slots: array<u32>;

// True when key `(a_asset, a_page)` sorts strictly before `(b_asset, b_page)`
// under the lexicographic order the CPU `GeometryPageKey` derives.
fn key_less(a_asset: u32, a_page: u32, b_asset: u32, b_page: u32) -> bool {
    return a_asset < b_asset || (a_asset == b_asset && a_page < b_page);
}

// One thread per query key.
@compute @workgroup_size(64, 1, 1)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if (q >= params.query_count) {
        return;
    }

    let qa = queries[q].asset;
    let qp = queries[q].page;

    // Lower-bound binary search over the sorted entry table: find the first
    // entry that does not sort before the query, then confirm an exact match.
    var lo: u32 = 0u;
    var hi: u32 = params.entry_count;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = lo + (hi - lo) / 2u;
        let e = entries[mid];
        if (key_less(e.asset, e.page, qa, qp)) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    var result: u32 = UNMAPPED_SLOT;
    if (lo < params.entry_count) {
        let e = entries[lo];
        if (e.asset == qa && e.page == qp) {
            result = e.slot;
        }
    }
    slots[q] = result;
}
