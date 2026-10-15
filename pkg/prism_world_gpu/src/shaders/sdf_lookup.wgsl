struct Params {
    brick_size: u32,
    brick_count: u32,
    query_count: u32,
    sample_words_per_brick: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> keys: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read> metadata: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> samples: array<u32>;
@group(0) @binding(4) var<storage, read> queries: array<vec4<i32>>;
@group(0) @binding(5) var<storage, read_write> results: array<vec4<i32>>;

fn floor_div(value: i32, divisor: i32) -> i32 {
    let quotient = value / divisor;
    let remainder = value % divisor;
    if (remainder < 0) {
        return quotient - 1;
    }
    return quotient;
}

fn key_less(a: vec4<i32>, b: vec4<i32>) -> bool {
    if (a.x != b.x) { return a.x < b.x; }
    if (a.y != b.y) { return a.y < b.y; }
    if (a.z != b.z) { return a.z < b.z; }
    return a.w < b.w;
}

fn find_key(wanted: vec4<i32>) -> i32 {
    var low: i32 = 0;
    var high: i32 = i32(params.brick_count);
    loop {
        if (low >= high) { break; }
        let middle = (low + high) / 2;
        let candidate = keys[u32(middle)];
        if (key_less(candidate, wanted)) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low < i32(params.brick_count) && all(keys[u32(low)] == wanted)) {
        return low;
    }
    return -1;
}

fn unpack_signed_sample(word: u32, high_half: bool) -> i32 {
    let shift = select(0u, 16u, high_half);
    let raw = (word >> shift) & 0xffffu;
    if (raw >= 32768u) {
        return i32(raw) - 65536;
    }
    return i32(raw);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.query_count) { return; }
    let query = queries[index];
    let size = i32(params.brick_size);
    let brick = vec4<i32>(
        floor_div(query.x, size),
        floor_div(query.y, size),
        floor_div(query.z, size),
        query.w,
    );
    let brick_index = find_key(brick);
    if (brick_index < 0) {
        results[index] = vec4<i32>(0, 0, 0, 0);
        return;
    }
    let local_x = query.x - brick.x * size;
    let local_y = query.y - brick.y * size;
    let local_z = query.z - brick.z * size;
    let sample_index = (local_z * size + local_y) * size + local_x;
    let word_index = metadata[u32(brick_index)].x + u32(sample_index / 2);
    let word = samples[word_index];
    let value = unpack_signed_sample(word, (sample_index & 1) == 1);
    results[index] = vec4<i32>(1, value, 0, 0);
}
