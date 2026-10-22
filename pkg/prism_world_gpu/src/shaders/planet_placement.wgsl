struct Params { words: array<vec4<u32>, 12>, };
struct Record { words: array<u32, 29>, };
struct Df { hi: f32, lo: f32, };

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
        if (limbs.x == 0u) {
            limbs.y = limbs.y + 1u;
            if (limbs.y == 0u) {
                limbs.z = limbs.z + 1u;
                if (limbs.z == 0u) { limbs.w = limbs.w + 1u; }
            }
        }
    }
    var result = f32(limbs.x);
    result = result + f32(limbs.y) * 4294967296.0;
    result = result + f32(limbs.z) * 18446744073709551616.0;
    result = result + f32(limbs.w) * 79228162514264337593543950336.0;
    return select(result, -result, negative);
}

// World origins are authored as i64 and widened to i128 only for a shared
// upload ABI. Decode the actual two-limb domain directly so small negative
// origins do not pass through high sign-extension limbs during float conversion.
fn signed_i64_f32(value: vec2<u32>) -> f32 {
    let negative = (value.y & 0x80000000u) != 0u;
    var limbs = value;
    if (negative) {
        limbs = ~limbs;
        limbs.x = limbs.x + 1u;
        if (limbs.x == 0u) { limbs.y = limbs.y + 1u; }
    }
    let result = f32(limbs.x) + f32(limbs.y) * 4294967296.0;
    return select(result, -result, negative);
}

fn df_from_f32(value: f32) -> Df { return Df(value, 0.0); }

fn df_from_u32(value: u32) -> Df {
    let high = f32(value >> 24u) * 16777216.0;
    let middle = f32((value >> 16u) & 0xffu) * 65536.0;
    let low = f32(value & 0xffffu);
    return df_add(df_add(df_from_f32(high), df_from_f32(middle)), df_from_f32(low));
}

fn df_normalize(hi: f32, lo: f32) -> Df {
    let sum = hi + lo;
    // General TwoSum: unlike FastTwoSum, this remains error-free when
    // cancellation makes |sum| smaller than |lo|.
    let virtual_lo = sum - hi;
    let error = (hi - (sum - virtual_lo)) + (lo - virtual_lo);
    return Df(sum, error);
}

fn df_add(a: Df, b: Df) -> Df {
    let sum = a.hi + b.hi;
    let virtual_b = sum - a.hi;
    let error = (a.hi - (sum - virtual_b)) + (b.hi - virtual_b) + a.lo + b.lo;
    return df_normalize(sum, error);
}

fn df_neg(value: Df) -> Df { return Df(-value.hi, -value.lo); }
fn df_sub(a: Df, b: Df) -> Df { return df_add(a, df_neg(b)); }

fn df_mul(a: Df, b: Df) -> Df {
    let product = a.hi * b.hi;
    let error = fma(a.hi, b.hi, -product) + a.hi * b.lo + a.lo * b.hi + a.lo * b.lo;
    return df_normalize(product, error);
}

fn df_div(a: Df, b: Df) -> Df {
    let first = a.hi / b.hi;
    let remainder = df_sub(a, df_mul(b, df_from_f32(first)));
    let second = (remainder.hi + remainder.lo) / b.hi;
    let quotient = df_add(df_from_f32(first), df_from_f32(second));
    let remainder2 = df_sub(a, df_mul(b, quotient));
    let third = (remainder2.hi + remainder2.lo) / b.hi;
    return df_add(quotient, df_from_f32(third));
}

fn df_sqrt(value: Df) -> Df {
    let first = sqrt(value.hi);
    let remainder = df_sub(value, df_mul(df_from_f32(first), df_from_f32(first)));
    let correction = (remainder.hi + remainder.lo) / (2.0 * first);
    return df_normalize(first, correction);
}

fn df_i64(value: vec2<u32>) -> Df {
    let negative = (value.y & 0x80000000u) != 0u;
    var limbs = value;
    if (negative) {
        limbs = ~limbs;
        limbs.x = limbs.x + 1u;
        if (limbs.x == 0u) { limbs.y = limbs.y + 1u; }
    }
    var result = df_add(df_from_u32(limbs.x), df_mul(df_from_u32(limbs.y), df_from_f32(4294967296.0)));
    if (negative) { result = df_neg(result); }
    return result;
}

fn df_i128(value: vec4<u32>) -> Df {
    let negative = (value.w & 0x80000000u) != 0u;
    var limbs = value;
    if (negative) {
        limbs = ~limbs;
        limbs.x = limbs.x + 1u;
        if (limbs.x == 0u) {
            limbs.y = limbs.y + 1u;
            if (limbs.y == 0u) {
                limbs.z = limbs.z + 1u;
                if (limbs.z == 0u) { limbs.w = limbs.w + 1u; }
            }
        }
    }
    var result = df_from_u32(limbs.x);
    result = df_add(result, df_mul(df_from_u32(limbs.y), df_from_f32(4294967296.0)));
    result = df_add(result, df_mul(df_from_u32(limbs.z), df_from_f32(18446744073709551616.0)));
    result = df_add(result, df_mul(df_from_u32(limbs.w), df_from_f32(79228162514264337593543950336.0)));
    if (negative) { result = df_neg(result); }
    return result;
}

fn df_face_cube(face: u32, u: Df, v: Df) -> array<Df, 3> {
    let one = df_from_f32(1.0);
    let minus_one = df_from_f32(-1.0);
    switch face {
        case 0u: { return array<Df, 3>(one, v, df_neg(u)); }
        case 1u: { return array<Df, 3>(minus_one, v, u); }
        case 2u: { return array<Df, 3>(u, one, df_neg(v)); }
        case 3u: { return array<Df, 3>(u, minus_one, v); }
        case 4u: { return array<Df, 3>(u, v, one); }
        default: { return array<Df, 3>(df_neg(u), v, minus_one); }
    }
}

fn finite(value: f32) -> bool {
    return value == value && abs(value) < 3.402823466e+38;
}

fn i256_to_f32_q84(value: I256) -> f32 {
    let m = value.magnitude.limb;
    let result = f32(m[0]) * 5.421010862427522e-26
        + f32(m[1]) * 2.220446049250313e-16
        + f32(m[2]) * 9.5367431640625e-7
        + f32(m[3]) * 4096.0;
    return select(result, -result, value.negative);
}

fn signed_less_than_zero(value: vec4<u32>) -> bool { return (value.w & 0x80000000u) != 0u; }

fn i256_fits_i128(value: I256) -> bool {
    if (value.magnitude.limb[4] != 0u || value.magnitude.limb[5] != 0u
        || value.magnitude.limb[6] != 0u || value.magnitude.limb[7] != 0u) { return false; }
    if (value.negative) {
        return value.magnitude.limb[3] < 0x80000000u
            || (value.magnitude.limb[3] == 0x80000000u
                && (value.magnitude.limb[2] | value.magnitude.limb[1] | value.magnitude.limb[0]) == 0u);
    }
    return value.magnitude.limb[3] < 0x80000000u
        || (value.magnitude.limb[3] == 0x80000000u
            && (value.magnitude.limb[2] | value.magnitude.limb[1] | value.magnitude.limb[0]) == 0u);
}

fn i256_to_twos_complement_i128(value: I256) -> vec4<u32> {
    var limbs = vec4<u32>(value.magnitude.limb[0], value.magnitude.limb[1], value.magnitude.limb[2], value.magnitude.limb[3]);
    if (value.negative) {
        limbs = ~limbs;
        limbs.x += 1u;
        if (limbs.x == 0u) {
            limbs.y += 1u;
            if (limbs.y == 0u) {
                limbs.z += 1u;
                if (limbs.z == 0u) { limbs.w += 1u; }
            }
        }
    }
    return limbs;
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
    let cells = sample_side - 1u;
    let cube = cube_numerators_for_cell(word(3u), vec2<u32>(word(5u), word(6u)), vec2<u32>(cell_u, cell_v), cells, word(4u));
    let normal_q84 = cube_direction_q84(cube);
    let normal = vec3<f32>(i256_to_f32_q84(normal_q84[0]), i256_to_f32_q84(normal_q84[1]), i256_to_f32_q84(normal_q84[2]));
    let row = cell_v * sample_side;
    let h0 = heights[row + cell_u];
    let h1 = heights[row + cell_u + 1u];
    let h2 = heights[row + sample_side + cell_u];
    let h3 = heights[row + sample_side + cell_u + 1u];
    let height_numerator = add128(add128(h0, h1), add128(h2, h3));
    let radial_base = vec4<u32>(word(20u), word(21u), word(22u), word(23u));
    let radial_numerator = add128(radial_base, height_numerator);
    let radial_q4 = i256_from_i128_words(radial_numerator);
    let origins = array<vec4<u32>, 3>(
        vec4<u32>(word(24u), word(25u), word(26u), word(27u)),
        vec4<u32>(word(28u), word(29u), word(30u), word(31u)),
        vec4<u32>(word(32u), word(33u), word(34u), word(35u)));
    var positions: array<vec4<u32>, 3>;
    var projection_invalid = radial_numerator.w >= 0x80000000u || all(radial_numerator == vec4<u32>(0u));
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let projected = project_q84_to_mm(normal_q84[axis], radial_q4);
        let local = i256_add(projected, i256_from_i128_words(origins[axis]));
        positions[axis] = i256_to_twos_complement_i128(local);
        projection_invalid = projection_invalid || !i256_fits_i128(local);
    }
    let invalid = projection_invalid;
    let rank = rank_for(cell_u, cell_v);
    let keep = rank <= word(7u);
    var record: Record;
    record.words[0] = word(3u); record.words[1] = word(4u);
    record.words[2] = word(5u); record.words[3] = word(6u);
    record.words[4] = cell_u; record.words[5] = cell_v;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        record.words[6u + axis * 4u] = positions[axis].x;
        record.words[7u + axis * 4u] = positions[axis].y;
        record.words[8u + axis * 4u] = positions[axis].z;
        record.words[9u + axis * 4u] = positions[axis].w;
    }
    record.words[18] = bitcast<u32>(normal.x); record.words[19] = bitcast<u32>(normal.y);
    record.words[20] = bitcast<u32>(normal.z); record.words[21] = select(0u, 1u, keep);
    record.words[22] = rank;
    record.words[23] = height_numerator.x; record.words[24] = height_numerator.y;
    record.words[25] = height_numerator.z; record.words[26] = height_numerator.w;
    record.words[27] = select(0u, 1u, invalid); record.words[28] = 0u;
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
