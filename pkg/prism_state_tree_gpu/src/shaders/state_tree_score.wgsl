// StateTree fixed-point score twin.
//
// The batch plan revision is a validated batch-level uniform. Keeping it out
// of the 40-byte input row preserves the existing ECS ABI while the output
// row carries all four 32-bit revision limbs for exact host reconstruction.

struct Params {
    row_count: u32,
    revision_0: u32,
    revision_1: u32,
    revision_2: u32,
    revision_3: u32,
};

struct InputRow {
    words: array<u32, 10>,
};

struct OutputRow {
    words: array<u32, 12>,
};

struct Wide {
    lo: u32,
    hi: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_rows: array<InputRow>;
@group(0) @binding(2) var<storage, read_write> output_rows: array<OutputRow>;

fn mul_u32(left: u32, right: u32) -> Wide {
    let left_lo = left & 0xffffu;
    let left_hi = left >> 16u;
    let right_lo = right & 0xffffu;
    let right_hi = right >> 16u;
    let word_0 = left_lo * right_lo;
    let word_1 = left_lo * right_hi;
    let word_2 = left_hi * right_lo;
    let word_3 = left_hi * right_hi;
    // Split the middle carry before adding the two cross products. Each
    // intermediate stays below 2^32, avoiding WGSL u32 overflow.
    let middle = (word_0 >> 16u) + (word_1 & 0xffffu) + (word_2 & 0xffffu);
    let low = (middle << 16u) | (word_0 & 0xffffu);
    let high = word_3 + (word_1 >> 16u) + (word_2 >> 16u) + (middle >> 16u);
    return Wide(low, high);
}

fn negate_wide(value: Wide) -> Wide {
    let low = (~value.lo) + 1u;
    return Wide(low, (~value.hi) + select(0u, 1u, low == 0u));
}

fn signed_product_shift(left: i32, right: i32) -> Wide {
    let left_bits = bitcast<u32>(left);
    let right_bits = bitcast<u32>(right);
    let left_magnitude = select(left_bits, (~left_bits) + 1u, left < 0);
    let right_magnitude = select(right_bits, (~right_bits) + 1u, right < 0);
    let product = mul_u32(left_magnitude, right_magnitude);
    var shifted = Wide((product.lo >> 16u) | (product.hi << 16u), product.hi >> 16u);
    if ((left < 0) != (right < 0)) {
        shifted = negate_wide(shifted);
    }
    return shifted;
}

fn add_i32(value: Wide, addend: i32) -> Wide {
    let low_addend = bitcast<u32>(addend);
    let low = value.lo + low_addend;
    let carry = select(0u, 1u, low < value.lo);
    let high_addend = select(0xffffffffu, 0u, addend >= 0);
    return Wide(low, value.hi + high_addend + carry);
}

fn score_and_status(value: Wide) -> vec2<u32> {
    if (value.hi == 0u && value.lo <= 0x7fffffffu) {
        return vec2<u32>(value.lo, 0u);
    }
    if (value.hi == 0xffffffffu && value.lo >= 0x80000000u) {
        return vec2<u32>(value.lo, 0u);
    }
    if ((value.hi & 0x80000000u) == 0u) {
        return vec2<u32>(0x7fffffffu, 1u);
    }
    return vec2<u32>(0x80000000u, 1u);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.row_count) {
        return;
    }

    let input = input_rows[index].words;
    let shifted_product = signed_product_shift(bitcast<i32>(input[4]), bitcast<i32>(input[5]));
    let score = score_and_status(add_i32(shifted_product, bitcast<i32>(input[6])));
    var output = output_rows[index].words;
    output[0] = input[0];
    output[1] = input[1];
    output[2] = input[2];
    output[3] = params.revision_0;
    output[4] = params.revision_1;
    output[5] = params.revision_2;
    output[6] = params.revision_3;
    output[7] = input[3];
    output[8] = score.x;
    output[9] = score.y;
    output[10] = 0u;
    output[11] = 0u;
    output_rows[index].words = output;
}
