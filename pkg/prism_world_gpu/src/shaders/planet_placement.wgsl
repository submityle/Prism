struct Params { words: array<vec4<u32>, 12>, };
struct Record { words: array<u32, 20>, };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> records: array<Record>;
@group(0) @binding(2) var<storage, read> heights: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> compact: array<u32>;
@group(0) @binding(4) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> indirect: array<atomic<u32>>;

fn word(index: u32) -> u32 {
    return params.words[index / 4u][index % 4u];
}

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low = a.x + b.x;
    return vec2<u32>(low, a.y + b.y + select(0u, 1u, low < a.x));
}

fn xor64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

fn shift_right64(value: vec2<u32>, amount: u32) -> vec2<u32> {
    if (amount == 0u) { return value; }
    if (amount < 32u) {
        return vec2<u32>((value.x >> amount) | (value.y << (32u - amount)), value.y >> amount);
    }
    if (amount < 64u) { return vec2<u32>(value.y >> (amount - 32u), 0u); }
    return vec2<u32>(0u, 0u);
}

fn multiply32_wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let first = a0 * b0;
    let second = a1 * b0 + (first >> 16u);
    let third = a0 * b1 + (second & 0xffffu);
    return vec2<u32>((third << 16u) | (first & 0xffffu), a1 * b1 + (second >> 16u) + (third >> 16u));
}

fn multiply64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low = multiply32_wide(a.x, b.x);
    return vec2<u32>(low.x, low.y + a.x * b.y + a.y * b.x);
}

fn mix64(value: vec2<u32>) -> vec2<u32> {
    var mixed = xor64(value, shift_right64(value, 30u));
    mixed = multiply64(mixed, vec2<u32>(0x1ce4e5b9u, 0xbf58476du));
    mixed = xor64(mixed, shift_right64(mixed, 27u));
    mixed = multiply64(mixed, vec2<u32>(0x133111ebu, 0x94d049bbu));
    return xor64(mixed, shift_right64(mixed, 31u));
}

fn rank_for(cell_u: u32, cell_v: u32) -> u32 {
    var value = vec2<u32>(word(8u) ^ word(11u) ^ word(10u), word(9u) ^ word(12u) ^ 1u);
    value = mix64(xor64(value, vec2<u32>(word(3u), 0u)));
    value = mix64(xor64(value, vec2<u32>(word(4u), 0u)));
    value = mix64(xor64(value, vec2<u32>(word(5u), 0u)));
    value = mix64(xor64(value, vec2<u32>(word(6u), 0u)));
    value = mix64(xor64(value, vec2<u32>(cell_u, 0u)));
    value = mix64(xor64(value, vec2<u32>(cell_v, 0u)));
    return value.y >> 16u;
}

fn add128(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let x = a.x + b.x;
    let c0 = select(0u, 1u, x < a.x);
    let y = a.y + b.y + c0;
    let c1 = select(0u, 1u, y < a.y || (c0 == 1u && y == a.y));
    let z = a.z + b.z + c1;
    let c2 = select(0u, 1u, z < a.z || (c1 == 1u && z == a.z));
    return vec4<u32>(x, y, z, a.w + b.w + c2);
}

// Decode two's-complement i128 limbs without requiring shader i64/f64 support.
fn signed_i128_f32(value: vec4<u32>) -> f32 {
    let negative = (value.w & 0x80000000u) != 0u;
    var limbs = value;
    if (negative) {
        limbs = ~limbs;
        limbs.x = limbs.x + 1u;
        if (limbs.x == 0u) { limbs.y = limbs.y + 1u; }
        if (limbs.y == 0u) { limbs.z = limbs.z + 1u; }
        if (limbs.z == 0u) { limbs.w = limbs.w + 1u; }
    }
    var result = f32(limbs.x);
    result = result + f32(limbs.y) * 4294967296.0;
    result = result + f32(limbs.z) * 18446744073709551616.0;
    result = result + f32(limbs.w) * 79228162514264337593543950336.0;
    return select(result, -result, negative);
}

fn finite(value: f32) -> bool {
    return value == value && abs(value) < 3.402823466e+38;
}

fn face_cube(face: u32, u: f32, v: f32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>(1.0, v, -u); }
        case 1u: { return vec3<f32>(-1.0, v, u); }
        case 2u: { return vec3<f32>(u, 1.0, -v); }
        case 3u: { return vec3<f32>(u, -1.0, v); }
        case 4u: { return vec3<f32>(u, v, 1.0); }
        default: { return vec3<f32>(-u, v, -1.0); }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let count = word(0u);
    if (index >= count) { return; }
    let width = word(14u);
    if (width == 0u) { return; }
    let cell_u = word(16u) + index % width;
    let cell_v = word(17u) + index / width;
    let sample_side = word(2u);
    let cells = f32(sample_side - 1u);
    let face_tiles = f32(1u << word(4u));
    let u = -1.0 + 2.0 * (f32(word(5u)) + (f32(cell_u) + 0.5) / cells) / face_tiles;
    let v = -1.0 + 2.0 * (f32(word(6u)) + (f32(cell_v) + 0.5) / cells) / face_tiles;
    let cube = face_cube(word(3u), u, v);
    let length = sqrt(dot(cube, cube));
    let normal = cube / length;
    let row = cell_v * sample_side;
    let h0 = heights[row + cell_u];
    let h1 = heights[row + cell_u + 1u];
    let h2 = heights[row + sample_side + cell_u];
    let h3 = heights[row + sample_side + cell_u + 1u];
    let height_numerator = add128(add128(h0, h1), add128(h2, h3));
    let height = signed_i128_f32(height_numerator) * 0.25;
    let radial_base = vec4<u32>(word(20u), word(21u), word(22u), word(23u));
    let radial_numerator = add128(radial_base, height_numerator);
    let radial = signed_i128_f32(radial_numerator) * 0.25;
    let origin = vec3<f32>(
        signed_i128_f32(vec4<u32>(word(24u), word(25u), word(26u), word(27u))),
        signed_i128_f32(vec4<u32>(word(28u), word(29u), word(30u), word(31u))),
        signed_i128_f32(vec4<u32>(word(32u), word(33u), word(34u), word(35u))));
    let position = origin + normal * radial;
    let invalid = !finite(normal.x) || !finite(normal.y) || !finite(normal.z)
        || !finite(height) || !finite(radial) || radial <= 0.0
        || !finite(position.x) || !finite(position.y) || !finite(position.z);
    let rank = rank_for(cell_u, cell_v);
    let keep = rank <= word(7u);
    var record: Record;
    record.words[0] = word(3u); record.words[1] = word(4u);
    record.words[2] = word(5u); record.words[3] = word(6u);
    record.words[4] = cell_u; record.words[5] = cell_v;
    record.words[6] = bitcast<u32>(position.x); record.words[7] = bitcast<u32>(position.y);
    record.words[8] = bitcast<u32>(position.z);
    record.words[9] = bitcast<u32>(normal.x); record.words[10] = bitcast<u32>(normal.y);
    record.words[11] = bitcast<u32>(normal.z); record.words[12] = select(0u, 1u, keep);
    record.words[13] = rank;
    record.words[14] = height_numerator.x; record.words[15] = height_numerator.y;
    record.words[16] = height_numerator.z; record.words[17] = height_numerator.w;
    record.words[18] = select(0u, 1u, invalid); record.words[19] = 0u;
    records[index] = record;
    if (keep && !invalid) {
        let slot = atomicAdd(&status[0], 1u);
        compact[slot] = index;
    } else if (keep && invalid) {
        atomicAdd(&status[1], 1u);
    }
}

@compute @workgroup_size(1)
fn finalize_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x == 0u) {
        let invalid = atomicLoad(&status[1]);
        atomicStore(&indirect[1], select(0u, atomicLoad(&status[0]), invalid == 0u));
    }
}
