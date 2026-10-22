// Native buffer-page lookup consumer. A miss is produced by this shader;
// a hit reads the actual physical pool, rather than returning a CPU-fed value.
struct Params { queries: u32, entries: u32, page_words: u32, pool_words: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read> queries: array<u32>;
@group(0) @binding(3) var<storage, read> pool: array<u32>;
@group(0) @binding(4) var<storage, read_write> feedback: array<u32>;

fn less(a: vec3<u32>, b: vec3<u32>) -> bool {
    return a.x < b.x || (a.x == b.x && (a.y < b.y || (a.y == b.y && a.z < b.z)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.queries { return; }
    let q = i * 4u;
    let key = vec3<u32>(queries[q], queries[q + 1u], queries[q + 2u]);
    var lo = 0u;
    var hi = params.entries;
    var slot = 0xffffffffu;
    loop {
        if lo >= hi { break; }
        let mid = lo + (hi - lo) / 2u;
        let at = mid * 4u;
        let candidate = vec3<u32>(table[at], table[at + 1u], table[at + 2u]);
        if all(candidate == key) { slot = table[at + 3u]; break; }
        if less(candidate, key) { lo = mid + 1u; } else { hi = mid; }
    }
    var value = 0u;
    if slot != 0xffffffffu {
        let word = queries[q + 3u];
        // Division guards multiplication overflow as well as the buffer bound.
        if params.page_words == 0u || word >= params.page_words || slot >= params.pool_words / params.page_words {
            slot = 0xffffffffu;
        } else { value = pool[slot * params.page_words + word]; }
    }
    feedback[i * 2u] = slot;
    feedback[i * 2u + 1u] = value;
}
