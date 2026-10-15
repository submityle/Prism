// GPU_HEIGHTFIELD_PLACEMENT_DELETION_VERSION = 1.
// Key ABI is 14 u32 words / 56 bytes:
// placement_version, source kernel, seed u64, recipe version, salt u64,
// spacing, then canonical source grid XYZ as signed i64 pairs.
// Records are the exact placement source ABI; only words 0..5 are identity.
struct Params {
    words: array<vec4<u32>, 8>,
};
struct Record { words: array<u32, 16>, };
struct Key { words: array<u32, 14>, };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> records: array<Record>;
@group(0) @binding(2) var<storage, read> upstream_compact: array<u32>;
@group(0) @binding(3) var<storage, read> upstream_indirect: array<u32, 4>;
@group(0) @binding(4) var<storage, read> deletions: array<Key>;
@group(0) @binding(5) var<storage, read_write> compact: array<u32>;
@group(0) @binding(6) var<storage, read_write> indirect: array<atomic<u32>, 4>;

var<workgroup> scan: array<u32, 64>;
var<workgroup> output_base: u32;

fn param(index: u32) -> u32 {
    return params.words[index / 4u][index % 4u];
}

fn compare32(a: u32, b: u32) -> i32 {
    if a < b { return -1; }
    if a > b { return 1; }
    return 0;
}

fn compare64(a_low: u32, a_high: u32, b_low: u32, b_high: u32, signed_value: bool) -> i32 {
    let mask = select(0u, 0x80000000u, signed_value);
    let high = compare32(a_high ^ mask, b_high ^ mask);
    if high != 0 { return high; }
    return compare32(a_low, b_low);
}

fn compare_key(a: Key, b: Key) -> i32 {
    var result = compare32(a.words[0], b.words[0]);
    if result != 0 { return result; }
    result = compare32(a.words[1], b.words[1]);
    if result != 0 { return result; }
    result = compare64(a.words[2], a.words[3], b.words[2], b.words[3], false);
    if result != 0 { return result; }
    result = compare32(a.words[4], b.words[4]);
    if result != 0 { return result; }
    result = compare64(a.words[5], a.words[6], b.words[5], b.words[6], false);
    if result != 0 { return result; }
    result = compare32(a.words[7], b.words[7]);
    if result != 0 { return result; }
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let index = 8u + axis * 2u;
        result = compare64(
            a.words[index],
            a.words[index + 1u],
            b.words[index],
            b.words[index + 1u],
            true,
        );
        if result != 0 { return result; }
    }
    return 0;
}

fn is_deleted(key: Key) -> bool {
    var low = 0u;
    var high = param(1u);
    loop {
        if low >= high { break; }
        let middle = low + (high - low) / 2u;
        let result = compare_key(key, deletions[middle]);
        if result == 0 { return true; }
        if result < 0 { high = middle; }
        else { low = middle + 1u; }
    }
    return false;
}

@compute @workgroup_size(64)
fn heightfield_placement_deletion_main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    // Host dispatch proves the padded rectangle fits in u32 and covers N slots.
    let index = (group.y * param(2u) + group.x) * 64u + lane;
    var source_slot = 0u;
    var accepted = 0u;
    if index < param(0u) && index < upstream_indirect[1] {
        source_slot = upstream_compact[index];
        if source_slot < param(0u) {
            var key: Key;
            key.words[0] = param(4u);
            key.words[1] = param(5u);
            key.words[2] = param(6u);
            key.words[3] = param(7u);
            key.words[4] = param(8u);
            key.words[5] = param(9u);
            key.words[6] = param(10u);
            key.words[7] = param(11u);
            for (var i = 0u; i < 6u; i += 1u) {
                key.words[8u + i] = records[source_slot].words[i];
            }
            if !is_deleted(key) { accepted = 1u; }
        }
    }

    // Every invocation reaches every barrier, including invalid tail lanes.
    scan[lane] = accepted;
    if lane == 0u { output_base = 0u; }
    workgroupBarrier();
    for (var stride = 1u; stride < 64u; stride *= 2u) {
        var preceding = 0u;
        if lane >= stride { preceding = scan[lane - stride]; }
        workgroupBarrier();
        scan[lane] += preceding;
        workgroupBarrier();
    }
    if lane == 0u && scan[63] != 0u {
        output_base = atomicAdd(&indirect[1], scan[63]);
    }
    workgroupBarrier();
    if accepted != 0u {
        compact[output_base + scan[lane] - 1u] = source_slot;
    }
}
