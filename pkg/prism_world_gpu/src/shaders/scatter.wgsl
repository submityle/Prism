// Integer GPU twin of prism_world_residency/src/scatter.rs, kernel revision 1.
// All 64-bit values are vec2<u32>(low, high), including two's-complement i64s.
// No floating-point arithmetic, shader-i64 feature, or request-local RNG is used.
// The host validates spacing, bounds, counts, halo work, buffers and dispatch.
struct Params {
    words: array<vec4<u32>, 12>,
};

// Scalar words deliberately avoid vec2 alignment padding after accepted.
// 0..5: grid XYZ; 6..11: position XYZ; 12: accepted; 13..14: priority;
// 15: zero. Array stride is exactly 64 bytes, priority starts at byte 52.
struct Record {
    words: array<u32, 16>,
};

struct Candidate {
    grid: array<vec2<u32>, 3>,
    position: array<vec2<u32>, 3>,
    priority: vec2<u32>,
    valid: bool,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> records: array<Record>;
@group(0) @binding(2) var<storage, read_write> compact: array<u32>;
@group(0) @binding(3) var<storage, read_write> indirect: array<atomic<u32>>;

// The only declared workgroup data: 64 * 4 + 4 = 260 bytes.
var<workgroup> scan: array<u32, 64>;
var<workgroup> output_base: u32;

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

fn candidate(grid: array<vec2<u32>, 3>) -> Candidate {
    // Function variables are zero-initialized. Do not publish any partial
    // position until every axis has passed the representable-world checks.
    var result: Candidate;
    result.grid = grid;
    let world_min = param_pair(30u);
    let world_max = param_pair(32u);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if signed_less64(grid[axis], world_min) || signed_less64(world_max, grid[axis]) {
            return result;
        }
    }

    let hash = candidate_hash(grid);
    let spacing = param_word(3u);
    let tags = array<vec2<u32>, 3>(
        vec2<u32>(0x85a308d3u, 0x243f6a88u),
        vec2<u32>(0x03707344u, 0x13198a2eu),
        vec2<u32>(0x299f31d0u, 0xa4093822u),
    );
    var position: array<vec2<u32>, 3>;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let offset = jitter_offset(hash64(hash ^ tags[axis]), spacing);
        // MIN = world_min*spacing + MIN.rem_euclid(spacing), and similarly
        // MAX. The minimum edge voxel may start below MIN, so checking only
        // signed overflow of its wrapped product would incorrectly discard it.
        if equal64(grid[axis], world_min) && offset < param_word(34u) {
            return result;
        }
        if equal64(grid[axis], world_max) && offset > param_word(35u) {
            return result;
        }
        // The range/remainder checks prove the mathematical final sum fits i64.
        // Its modulo-64 product and addition therefore have the exact i64 bits.
        position[axis] = add64(
            multiply64(grid[axis], vec2<u32>(spacing, 0u)),
            vec2<u32>(offset, 0u),
        );
    }
    result.position = position;
    result.priority = hash64(hash ^ vec2<u32>(0xec4e6c89u, 0x082efa98u));
    result.valid = true;
    return result;
}

fn region_contains(position: array<vec2<u32>, 3>) -> bool {
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let lower = param_pair(18u + axis * 2u);
        let upper = param_pair(24u + axis * 2u);
        if signed_less64(position[axis], lower) || !signed_less64(position[axis], upper) {
            return false;
        }
    }
    return true;
}

fn precedes(a: Candidate, b: Candidate) -> bool {
    if !equal64(a.priority, b.priority) {
        return unsigned_less64(a.priority, b.priority);
    }
    // Every other structural key field is identical within this dispatch.
    // Priority collisions are broken by signed lexicographic global XYZ grid.
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if !equal64(a.grid[axis], b.grid[axis]) {
            return signed_less64(a.grid[axis], b.grid[axis]);
        }
    }
    return false;
}

fn strictly_within(
    a: array<vec2<u32>, 3>,
    b: array<vec2<u32>, 3>,
    radius: u32,
) -> bool {
    // Subtract each square from a remaining u64 budget instead of adding
    // three near-u64-sized squares. This needs neither u128 nor an overflowed
    // partial sum. Reject equality: a distance exactly at the radius is allowed.
    var remaining = multiply32_wide(radius, radius);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        var delta: vec2<u32>;
        if signed_less64(a[axis], b[axis]) {
            delta = subtract64(b[axis], a[axis]);
        } else {
            delta = subtract64(a[axis], b[axis]);
        }
        // Signed endpoints may differ by as much as u64::MAX. Only narrow
        // after proving delta < radius <= u32::MAX.
        if delta.y != 0u || delta.x >= radius {
            return false;
        }
        let squared = multiply32_wide(delta.x, delta.x);
        if !unsigned_less64(squared, remaining) {
            return false;
        }
        remaining = subtract64(remaining, squared);
    }
    return true;
}

fn survives(current: Candidate) -> bool {
    let radius = param_word(6u);
    if radius == 0u {
        return true;
    }
    let halo = vec2<u32>(param_word(7u), 0u);
    let world_min = param_pair(30u);
    let world_max = param_pair(32u);
    var lower: array<vec2<u32>, 3>;
    var upper: array<vec2<u32>, 3>;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let coord = current.grid[axis];
        // Distances to inclusive world-grid edges fit unsigned u64 even when
        // signed subtraction would overflow. Clip before adding/subtracting
        // the halo so neither bound can wrap across an i64 endpoint.
        if unsigned_less64(subtract64(coord, world_min), halo) {
            lower[axis] = world_min;
        } else {
            lower[axis] = subtract64(coord, halo);
        }
        if unsigned_less64(subtract64(world_max, coord), halo) {
            upper[axis] = world_max;
        } else {
            upper[axis] = add64(coord, halo);
        }
    }

    // Inclusive lexicographic iteration, Z fastest. Test the endpoint before
    // incrementing, including when that endpoint is i64::MAX (spacing = 1).
    var x = lower[0];
    loop {
        var y = lower[1];
        loop {
            var z = lower[2];
            loop {
                let grid = array<vec2<u32>, 3>(x, y, z);
                let is_self = equal64(x, current.grid[0])
                    && equal64(y, current.grid[1]) && equal64(z, current.grid[2]);
                if !is_self {
                    // Matérn-II compares ORIGINAL candidates, not survivors.
                    // Neighbors outside the request (even at world MAX) still
                    // participate; only unrepresentable candidates are absent.
                    let neighbor = candidate(grid);
                    if neighbor.valid {
                        if precedes(neighbor, current) {
                            if strictly_within(neighbor.position, current.position, radius) {
                                return false;
                            }
                        }
                    }
                }
                if equal64(z, upper[2]) {
                    break;
                }
                z = add64(z, vec2<u32>(1u, 0u));
            }
            if equal64(y, upper[1]) {
                break;
            }
            y = add64(y, vec2<u32>(1u, 0u));
        }
        if equal64(x, upper[0]) {
            break;
        }
        x = add64(x, vec2<u32>(1u, 0u));
    }
    return true;
}

@compute @workgroup_size(64)
fn scatter_main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    // The host caps capacity/dispatch so the entire padded 2D dispatch fits
    // u32. Source slots match the CPU's X-major, then Y, then Z enumeration.
    let index = group.y * param_word(36u) * 64u + group.x * 64u + lane;
    var accepted = 0u;
    if index < param_word(11u) {
        let nz = param_word(10u);
        let ny = param_word(9u);
        let yz = index / nz;
        let grid = array<vec2<u32>, 3>(
            add64(param_pair(12u), vec2<u32>(yz / ny, 0u)),
            add64(param_pair(14u), vec2<u32>(yz % ny, 0u)),
            add64(param_pair(16u), vec2<u32>(index % nz, 0u)),
        );
        let current = candidate(grid);
        if current.valid {
            if region_contains(current.position) {
                if survives(current) {
                    accepted = 1u;
                }
            }
        }

        var record: Record;
        for (var axis = 0u; axis < 3u; axis += 1u) {
            record.words[axis * 2u] = grid[axis].x;
            record.words[axis * 2u + 1u] = grid[axis].y;
            record.words[6u + axis * 2u] = current.position[axis].x;
            record.words[7u + axis * 2u] = current.position[axis].y;
        }
        record.words[12] = accepted;
        record.words[13] = current.priority.x;
        record.words[14] = current.priority.y;
        record.words[15] = 0u;
        records[index] = record;
    }

    // Tail lanes contribute zero but reach EVERY barrier. An inclusive shared
    // scan preserves source order within a group without per-point atomics.
    scan[lane] = accepted;
    if lane == 0u {
        output_base = 0u;
    }
    workgroupBarrier();
    for (var stride = 1u; stride < 64u; stride *= 2u) {
        var preceding = 0u;
        if lane >= stride {
            preceding = scan[lane - stride];
        }
        // All reads from the old scan must finish before any lane writes.
        workgroupBarrier();
        scan[lane] = scan[lane] + preceding;
        workgroupBarrier();
    }
    if lane == 0u {
        let count = scan[63];
        if count != 0u {
            // Host encoding clears only word 1; preserve vertex_count and the
            // remaining indirect words. One global atomic per nonempty group.
            output_base = atomicAdd(&indirect[1], count);
        }
    }
    workgroupBarrier();
    if accepted != 0u {
        // Group reservation order is intentionally unspecified, as in
        // proxy_compact. Records retain globally deterministic source order.
        compact[output_base + scan[lane] - 1u] = index;
    }
}
