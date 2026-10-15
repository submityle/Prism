// HEIGHTFIELD_PLACEMENT_VERSION=1 XZ generation. Core u32 only.
// Params match the source scatter layout, with Y lengths/min grid unused.
// No 3D source compaction or raw Y slab is part of placement.
struct Params { words: array<vec4<u32>, 12>, };
struct Record { words: array<u32, 16>, };
struct Candidate {
    grid: array<vec2<u32>, 3>,
    position: array<vec2<u32>, 3>,
    priority: vec2<u32>,
    valid: bool,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> records: array<Record>;

fn param_word(index: u32) -> u32 {
    return params.words[index / 4u][index % 4u];
}

fn param_pair(index: u32) -> vec2<u32> {
    return vec2<u32>(param_word(index), param_word(index + 1u));
}

fn equal64(a: vec2<u32>, b: vec2<u32>) -> bool {
    return all(a == b);
}

fn unsigned_less64(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

fn signed_less64(a: vec2<u32>, b: vec2<u32>) -> bool {
    // Flipping the sign bit maps signed ordering to unsigned ordering.
    let ah = a.y ^ 0x80000000u;
    let bh = b.y ^ 0x80000000u;
    return ah < bh || (ah == bh && a.x < b.x);
}

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low = a.x + b.x;
    return vec2<u32>(low, a.y + b.y + select(0u, 1u, low < a.x));
}

fn subtract64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}

// Exact 32x32 -> 64 multiplication. Each intermediate product uses 16-bit
// limbs; the carry sums below fit u32, so no significant high bits are lost.
fn multiply32_wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let first = a0 * b0;
    let second = a1 * b0 + (first >> 16u);
    let third = a0 * b1 + (second & 0xffffu);
    let low = (third << 16u) | (first & 0xffffu);
    let high = a1 * b1 + (second >> 16u) + (third >> 16u);
    return vec2<u32>(low, high);
}

fn multiply64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low_product = multiply32_wide(a.x, b.x);
    // The omitted high*high term starts at bit 64. Cross terms only need their
    // low u32 halves because this operation is explicitly modulo 2^64.
    return vec2<u32>(low_product.x, low_product.y + a.x * b.y + a.y * b.x);
}

fn shift_right64(value: vec2<u32>, amount: u32) -> vec2<u32> {
    if amount == 0u {
        return value;
    }
    if amount < 32u {
        return vec2<u32>(
            (value.x >> amount) | (value.y << (32u - amount)),
            value.y >> amount,
        );
    }
    if amount < 64u {
        return vec2<u32>(value.y >> (amount - 32u), 0u);
    }
    return vec2<u32>(0u);
}

fn rotate_right32(value: u32, rotation: u32) -> u32 {
    let amount = rotation & 31u;
    // Never evaluate a shift by 32, including when the PCG rotation is zero.
    if amount == 0u {
        return value;
    }
    return (value >> amount) | (value << (32u - amount));
}

// This is pcg.rs::pcg_hash_u32(u64), NOT the separate density hash.
fn pcg_hash_u32(value: vec2<u32>) -> u32 {
    var state = add64(value, vec2<u32>(0x748fea9bu, 0x853c49e6u));
    state = add64(
        multiply64(state, vec2<u32>(0x4c957f2du, 0x5851f42du)),
        vec2<u32>(0xf767814fu, 0x14057b7eu),
    );
    let word = shift_right64(shift_right64(state, 18u) ^ state, 27u).x;
    return rotate_right32(word, state.y >> 27u);
}

fn hash64(value: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(
        pcg_hash_u32(value ^ vec2<u32>(0x7f4a7c15u, 0x9e3779b9u)),
        pcg_hash_u32(value),
    );
}

fn candidate_hash(grid: array<vec2<u32>, 3>) -> vec2<u32> {
    // 0x7363617474657200 XOR SCATTER_KERNEL_VERSION (1).
    var hash = hash64(vec2<u32>(0x74657201u, 0x73636174u));
    hash = hash64(hash ^ param_pair(0u));
    hash = hash64(hash ^ vec2<u32>(param_word(2u), 0u));
    hash = hash64(hash ^ param_pair(4u));
    hash = hash64(hash ^ vec2<u32>(param_word(3u), 0u));
    hash = hash64(hash ^ grid[0]);
    hash = hash64(hash ^ grid[1]);
    hash = hash64(hash ^ grid[2]);
    return hash;
}

// floor((value * spacing) / 2^64), a u32 in [0, spacing). The 96-bit product
// consists of two exact 32x32 products and the carry between their middle words.
fn jitter_offset(value: vec2<u32>, spacing: u32) -> u32 {
    let lower = multiply32_wide(value.x, spacing);
    let upper = multiply32_wide(value.y, spacing);
    let middle = lower.y + upper.x;
    return upper.y + select(0u, 1u, middle < lower.y);
}


fn placement_priority(grid: array<vec2<u32>, 3>) -> vec2<u32> {
    // mix(0x706c6163655f787a XOR placement_version).
    var hash = hash64(vec2<u32>(0x655f787bu, 0x706c6163u));
    hash = hash64(hash ^ vec2<u32>(1u, 0u));
    hash = hash64(hash ^ param_pair(0u));
    hash = hash64(hash ^ vec2<u32>(param_word(2u), 0u));
    hash = hash64(hash ^ param_pair(4u));
    hash = hash64(hash ^ vec2<u32>(param_word(3u), 0u));
    hash = hash64(hash ^ grid[0]);
    return hash64(hash ^ grid[2]);
}

fn candidate(grid: array<vec2<u32>, 3>) -> Candidate {
    var result: Candidate;
    result.grid = grid;
    let world_min = param_pair(30u);
    let world_max = param_pair(32u);
    for (var axis = 0u; axis < 3u; axis += 2u) {
        if signed_less64(grid[axis], world_min) || signed_less64(world_max, grid[axis]) {
            return result;
        }
    }
    let hash = candidate_hash(grid);
    let spacing = param_word(3u);
    let tags = array<vec2<u32>, 3>(
        vec2<u32>(0x85a308d3u, 0x243f6a88u),
        vec2<u32>(0u),
        vec2<u32>(0x299f31d0u, 0xa4093822u),
    );
    var position: array<vec2<u32>, 3>;
    for (var axis = 0u; axis < 3u; axis += 2u) {
        let offset = jitter_offset(hash64(hash ^ tags[axis]), spacing);
        if equal64(grid[axis], world_min) && offset < param_word(34u) {
            return result;
        }
        if equal64(grid[axis], world_max) && offset > param_word(35u) {
            return result;
        }
        position[axis] = add64(
            multiply64(grid[axis], vec2<u32>(spacing, 0u)), vec2<u32>(offset, 0u),
        );
    }
    result.position = position;
    if param_word(6u) != 0u {
        result.priority = placement_priority(grid);
    }
    result.valid = true;
    return result;
}

fn precedes(a: Candidate, b: Candidate) -> bool {
    if !equal64(a.priority, b.priority) {
        return unsigned_less64(a.priority, b.priority);
    }
    if !equal64(a.grid[0], b.grid[0]) {
        return signed_less64(a.grid[0], b.grid[0]);
    }
    return signed_less64(a.grid[2], b.grid[2]);
}

fn strictly_within(a: Candidate, b: Candidate, radius: u32) -> bool {
    var remaining = multiply32_wide(radius, radius);
    for (var axis = 0u; axis < 3u; axis += 2u) {
        var delta: vec2<u32>;
        if signed_less64(a.position[axis], b.position[axis]) {
            delta = subtract64(b.position[axis], a.position[axis]);
        } else {
            delta = subtract64(a.position[axis], b.position[axis]);
        }
        if delta.y != 0u || delta.x >= radius { return false; }
        let squared = multiply32_wide(delta.x, delta.x);
        if !unsigned_less64(squared, remaining) { return false; }
        remaining = subtract64(remaining, squared);
    }
    return true;
}

fn survives(current: Candidate) -> bool {
    let radius = param_word(6u);
    if radius == 0u { return true; }
    let halo = vec2<u32>(param_word(7u), 0u);
    let world_min = param_pair(30u);
    let world_max = param_pair(32u);
    var lower: array<vec2<u32>, 3>;
    var upper: array<vec2<u32>, 3>;
    for (var axis = 0u; axis < 3u; axis += 2u) {
        let coord = current.grid[axis];
        if unsigned_less64(subtract64(coord, world_min), halo) {
            lower[axis] = world_min;
        } else { lower[axis] = subtract64(coord, halo); }
        if unsigned_less64(subtract64(world_max, coord), halo) {
            upper[axis] = world_max;
        } else { upper[axis] = add64(coord, halo); }
    }
    var x = lower[0];
    loop {
        var z = lower[2];
        loop {
            if !equal64(x, current.grid[0]) || !equal64(z, current.grid[2]) {
                let other = candidate(array<vec2<u32>, 3>(x, vec2<u32>(0u), z));
                if other.valid {
                    if precedes(other, current) {
                        if strictly_within(other, current, radius) { return false; }
                    }
                }
            }
            if equal64(z, upper[2]) { break; }
            z = add64(z, vec2<u32>(1u, 0u));
        }
        if equal64(x, upper[0]) { break; }
        x = add64(x, vec2<u32>(1u, 0u));
    }
    return true;
}

@compute @workgroup_size(64)
fn placement_main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    // Host dispatch proof includes padded invocations <= 2^32.
    let index = group.y * param_word(36u) * 64u + group.x * 64u + lane;
    if index >= param_word(11u) { return; }
    let nz = param_word(10u);
    let grid = array<vec2<u32>, 3>(
        add64(param_pair(12u), vec2<u32>(index / nz, 0u)),
        vec2<u32>(0u),
        add64(param_pair(16u), vec2<u32>(index % nz, 0u)),
    );
    let current = candidate(grid);
    var accepted = false;
    if current.valid {
        var owned = true;
        for (var axis = 0u; axis < 3u; axis += 2u) {
            if signed_less64(current.position[axis], param_pair(18u + axis * 2u))
                || !signed_less64(current.position[axis], param_pair(24u + axis * 2u)) {
                owned = false;
            }
        }
        if owned { accepted = survives(current); }
    }
    var record: Record;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        record.words[axis * 2u] = grid[axis].x;
        record.words[axis * 2u + 1u] = grid[axis].y;
        record.words[6u + axis * 2u] = current.position[axis].x;
        record.words[7u + axis * 2u] = current.position[axis].y;
    }
    record.words[12] = select(0u, 1u, accepted);
    record.words[13] = current.priority.x;
    record.words[14] = current.priority.y;
    records[index] = record;
}

