// GPU_SCATTER_DELETION_VERSION = 1, KEY_ABI_VERSION = 1.
// Sorted by Rust ScatterCandidateKey::Ord, not packed byte order.
// Dispatch covers primary N; only upstream's FINAL compact/count is consumed.
struct Params {
    // 0 N, 1 D, 2 dispatch X, 3 reserved; 4..10 common key namespace,
    // 11..31 reserved zero. Pairs are low/high u32, never floats/shader-i64.
    words: array<vec4<u32>, 8>,
};
struct Record { words: array<u32, 16>, };
struct Key { words: array<u32, 13>, };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> records: array<Record>;
@group(0) @binding(2) var<storage, read> upstream_compact: array<u32>;
@group(0) @binding(3) var<storage, read> upstream_indirect: array<u32, 4>;
@group(0) @binding(4) var<storage, read> deletions: array<Key>;
@group(0) @binding(5) var<storage, read_write> compact: array<u32>;
@group(0) @binding(6) var<storage, read_write> indirect: array<atomic<u32>, 4>;

var<workgroup> scan: array<u32, 64>;
var<workgroup> output_base: u32;

fn param(index: u32) -> u32 { return params.words[index / 4u][index % 4u]; }

fn compare32(a: u32, b: u32) -> i32 {
    if a < b { return -1; }
    if a > b { return 1; }
    return 0;
}

fn compare64(al: u32, ah: u32, bl: u32, bh: u32, signed_value: bool) -> i32 {
    let mask = select(0u, 0x80000000u, signed_value);
    let high = compare32(ah ^ mask, bh ^ mask);
    if high != 0 { return high; }
    return compare32(al, bl);
}

fn compare_key(a: Key, b: Key) -> i32 {
    var comparison = compare32(a.words[0], b.words[0]);
    if comparison != 0 { return comparison; }
    comparison = compare64(a.words[1], a.words[2], b.words[1], b.words[2], false);
    if comparison != 0 { return comparison; }
    comparison = compare32(a.words[3], b.words[3]);
    if comparison != 0 { return comparison; }
    comparison = compare64(a.words[4], a.words[5], b.words[4], b.words[5], false);
    if comparison != 0 { return comparison; }
    comparison = compare32(a.words[6], b.words[6]);
    if comparison != 0 { return comparison; }
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let i = 7u + axis * 2u;
        comparison = compare64(a.words[i], a.words[i+1u], b.words[i], b.words[i+1u], true);
        if comparison != 0 { return comparison; }
    }
    return 0;
}

fn is_deleted(key: Key) -> bool {
    var low = 0u;
    var high = param(1u);
    // At most ceil(log2(D+1)) FULL key comparisons, with immediate equality.
    loop {
        if low >= high { break; }
        let mid = low + (high - low) / 2u;
        let comparison = compare_key(key, deletions[mid]);
        if comparison == 0 { return true; }
        if comparison < 0 { high = mid; }
        else { low = mid + 1u; }
    }
    return false;
}

@compute @workgroup_size(64)
fn scatter_deletion_main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    // Host checks the padded rectangle has <= 2^32 lanes: both group flattening
    // and the last lane index fit u32, including whole rectangular tail groups.
    let index = (group.y * param(2u) + group.x) * 64u + lane;
    var slot = 0u;
    var accepted = 0u;
    if index < param(0u) && index < upstream_indirect[1] {
        slot = upstream_compact[index];
        if slot < param(0u) {
            var key: Key;
            for (var i = 0u; i < 7u; i += 1u) { key.words[i] = param(4u+i); }
            for (var i = 0u; i < 6u; i += 1u) { key.words[7u+i] = records[slot].words[i]; }
            if !is_deleted(key) { accepted = 1u; }
        }
    }
    // No early returns; every lane reaches every scan barrier.
    scan[lane] = accepted;
    if lane == 0u { output_base = 0u; }
    workgroupBarrier();
    for (var stride = 1u; stride < 64u; stride *= 2u) {
        var preceding = 0u;
        if lane >= stride { preceding = scan[lane-stride]; }
        workgroupBarrier();
        scan[lane] += preceding;
        workgroupBarrier();
    }
    if lane == 0u && scan[63] != 0u { output_base = atomicAdd(&indirect[1], scan[63]); }
    workgroupBarrier();
    if accepted != 0u { compact[output_base + scan[lane] - 1u] = slot; }
}
