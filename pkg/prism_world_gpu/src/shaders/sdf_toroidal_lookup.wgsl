struct Params {
    brick_size: u32,
    side: u32,
    mip_levels: u32,
    query_count: u32,
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

fn modulo(value: i32, divisor: i32) -> u32 {
    let remainder = value % divisor;
    if (remainder < 0) {
        return u32(remainder + divisor);
    }
    return u32(remainder);
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
    if (query.w < 0 || query.w >= i32(params.mip_levels)) {
        results[index] = vec4<i32>(0, 0, 0, 0);
        return;
    }
    let size = i32(params.brick_size);
    let brick = vec4<i32>(
        floor_div(query.x, size),
        floor_div(query.y, size),
        floor_div(query.z, size),
        query.w,
    );
    let side = i32(params.side);
    let cells_per_mip = params.side * params.side;
    let slot = u32(brick.w) * cells_per_mip
        + modulo(brick.z, side) * params.side
        + modulo(brick.x, side);
    let slot_metadata = metadata[slot];
    if (slot_metadata.y == 0u || any(keys[slot] != brick)) {
        results[index] = vec4<i32>(0, 0, 0, 0);
        return;
    }
    let local_x = i32(modulo(query.x, size));
    let local_y = i32(modulo(query.y, size));
    let local_z = i32(modulo(query.z, size));
    let sample_index = (local_z * size + local_y) * size + local_x;
    let word_index = slot_metadata.x + u32(sample_index / 2);
    let value = unpack_signed_sample(samples[word_index], (sample_index & 1) == 1);
    results[index] = vec4<i32>(1, value, 0, 0);
}
