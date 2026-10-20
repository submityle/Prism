// Integer GPU twin of HEIGHTFIELD_PLACEMENT_VERSION = 1 and
// HEIGHTFIELD_SURFACE_VERSION = 1. Arithmetic uses core u32 WGSL only;
// i32 is used only for the storage ABI of sign-extended encoded i16 heights.
// Limbs are least significant first; signed values use two's complement.
//
// Host preconditions: positive spacing/positive i64 scale, side >= 2,
// side*side <= u32::MAX, ordered height interval and nonzero slope run.
// The nonempty request XZ footprint is covered by the validated half-open
// surface. Source records belong to that request, have canonical grid Y = 0,
// and flag 12 identifies original XZ ownership/minimum-distance survivors.
// Buffers cover count slots (heights cover side*side); dispatch Z = 1 and
// groups_x matches dispatch X. The ordered reset_main compute pass clears
// status and indirect[1] before this shader. Submit project_main, then existing
// field_main at the projected positions, then finalize_main in ordered
// dispatches. No inter-workgroup barriers or optional GPU features are needed.
// All four candidate passes are skipped when count = 0; reset still executes.
// After failure only flags/indirect count are invalidated: compact tails and
// projected positions/samples must not be interpreted as a successful result.
struct Params {
    // 0 count, 1 groups_x, 2 terrain side, 3 terrain spacing,
    // 4..5 origin X, 6..7 origin Y, 8..9 origin Z,
    // 10..11 positive i64 height scale,
    // 12..13 unoffset surface height lower, 14..15 upper,
    // 16..17 signed vertical offset, 18..19 zero,
    // 20 slope rise, 21 slope run, 22..31 zero.
    words: array<vec4<u32>, 8>,
};

struct Record {
    // 64-byte source ABI: 0..5 grid XYZ i64 pairs (Y = 0),
    // 6..11 position XYZ i64 pairs, 12 accepted, 13..14 priority, 15 reserved.
    words: array<u32, 16>,
};

struct Sample {
    // 64 bytes, exact signed i128 values; denominator is Params spacing.
    // Height excludes vertical offset; normal is the upward face normal.
    height: vec4<u32>,
    normal_x: vec4<u32>,
    normal_y: vec4<u32>,
    normal_z: vec4<u32>,
};

struct Status {
    overflow_count: atomic<u32>,
};

struct DivRem64 {
    quotient: vec2<u32>,
    remainder: u32,
};

struct DivRem128 {
    quotient: vec4<u32>,
    remainder: u32,
};

struct Product128 {
    value: vec4<u32>,
    // The fifth limb of the exact unsigned 128x32 product.
    carry: u32,
};

struct RoundedY {
    value: vec2<u32>,
    valid: bool,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> records: array<Record>;
@group(0) @binding(2) var<storage, read> heights: array<i32>;
@group(0) @binding(3) var<storage, read_write> samples: array<Sample>;
@group(0) @binding(4) var<storage, read_write> status: Status;
@group(0) @binding(5) var<storage, read_write> indirect: array<atomic<u32>, 4>;
@group(0) @binding(6) var<storage, read_write> attrs: array<vec4<u32>>;

fn param_word(index: u32) -> u32 {
    return params.words[index / 4u][index % 4u];
}

fn param_pair(index: u32) -> vec2<u32> {
    return vec2<u32>(param_word(index), param_word(index + 1u));
}

// These exact limb helpers match scatter_heightfield.wgsl.
fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let low = a.x + b.x;
    return vec2<u32>(low, a.y + b.y + select(0u, 1u, low < a.x));
}

fn subtract64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}

// Exact 32x32 -> 64. The 16-bit partial products and carry sums fit u32.
fn multiply32_wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let first = a0 * b0;
    let second = a1 * b0 + (first >> 16u);
    let third = a0 * b1 + (second & 0xffffu);
    return vec2<u32>(
        (third << 16u) | (first & 0xffffu),
        a1 * b1 + (second >> 16u) + (third >> 16u),
    );
}

fn add128(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var result: vec4<u32>;
    var carry = 0u;
    for (var i = 0u; i < 4u; i += 1u) {
        let sum = a[i] + b[i];
        let with_carry = sum + carry;
        // At most one carry bit exits a word, even for MAX + MAX + 1.
        carry = select(0u, 1u, sum < a[i] || with_carry < sum);
        result[i] = with_carry;
    }
    return result;
}

fn subtract128(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var result: vec4<u32>;
    var borrow = 0u;
    for (var i = 0u; i < 4u; i += 1u) {
        let difference = a[i] - b[i];
        result[i] = difference - borrow;
        borrow = select(0u, 1u, a[i] < b[i] || difference < borrow);
    }
    return result;
}

fn negate128(value: vec4<u32>) -> vec4<u32> {
    return subtract128(vec4<u32>(0u), value);
}

fn unsigned_less128(a: vec4<u32>, b: vec4<u32>) -> bool {
    if a.w != b.w {
        return a.w < b.w;
    }
    if a.z != b.z {
        return a.z < b.z;
    }
    if a.y != b.y {
        return a.y < b.y;
    }
    return a.x < b.x;
}

fn signed_less128(a: vec4<u32>, b: vec4<u32>) -> bool {
    // Flip only the sign bit to map signed ordering onto unsigned ordering.
    return unsigned_less128(
        vec4<u32>(a.xyz, a.w ^ 0x80000000u),
        vec4<u32>(b.xyz, b.w ^ 0x80000000u),
    );
}

fn extend_signed64(value: vec2<u32>) -> vec4<u32> {
    let fill = select(0u, 0xffffffffu, (value.y & 0x80000000u) != 0u);
    return vec4<u32>(value, fill, fill);
}

fn absolute128(value: vec4<u32>) -> vec4<u32> {
    if (value.w & 0x80000000u) != 0u {
        return negate128(value);
    }
    return value;
}

// Exact unsigned 128x32 -> 160. The carry entering each word is <= factor-1
// (zero when factor is zero). Product high + one carry bit therefore fits u32.
// Signed callers use the low 128 bits, i.e. two's-complement multiplication;
// their mathematical signed results are separately proven to fit i128.
fn multiply128_by32(value: vec4<u32>, factor: u32) -> Product128 {
    var result: vec4<u32>;
    var carry = 0u;
    for (var i = 0u; i < 4u; i += 1u) {
        let product = multiply32_wide(value[i], factor);
        let low = product.x + carry;
        result[i] = low;
        carry = product.y + select(0u, 1u, low < product.x);
    }
    return Product128(result, carry);
}

// Exact unsigned 64x64 -> 128, including both middle-limb carries. Adding
// shifted nonnegative partial products cannot overflow the final 128 bits.
fn multiply64_wide(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    let p00 = multiply32_wide(a.x, b.x);
    let p01 = multiply32_wide(a.x, b.y);
    let p10 = multiply32_wide(a.y, b.x);
    let p11 = multiply32_wide(a.y, b.y);
    var result = vec4<u32>(p00, 0u, 0u);
    result = add128(result, vec4<u32>(0u, p01, 0u));
    result = add128(result, vec4<u32>(0u, p10, 0u));
    return add128(result, vec4<u32>(0u, 0u, p11));
}

// Full unsigned 64/32 division, not a narrowing of a potentially >i64 delta.
// Preserve the 33rd remainder bit before shifting. For divisors > 2^31,
// wrapped subtraction when that bit is set gives the exact reduced remainder.
fn unsigned_divmod64_by32(value: vec2<u32>, denominator: u32) -> DivRem64 {
    if value.y == 0u {
        return DivRem64(vec2<u32>(value.x / denominator, 0u), value.x % denominator);
    }
    var quotient = vec2<u32>(0u, value.y / denominator);
    var remainder = value.y % denominator;
    var bit = 32u;
    loop {
        if bit == 0u {
            break;
        }
        bit -= 1u;
        let carry = remainder >> 31u;
        remainder = (remainder << 1u) | ((value.x >> bit) & 1u);
        if carry != 0u || remainder >= denominator {
            remainder -= denominator;
            quotient.x |= 1u << bit;
        }
    }
    return DivRem64(quotient, remainder);
}

// Exact unsigned 128/32 division. Divide the top limb natively, then bring
// down every bit of the remaining three limbs. Before each bit remainder < d;
// after shifting it is <= 2*d-1, so one subtraction suffices. Preserve the
// 33rd bit, just as in unsigned_divmod64_by32, even when d = u32::MAX.
fn unsigned_divmod128_by32(value: vec4<u32>, denominator: u32) -> DivRem128 {
    var quotient = vec4<u32>(0u, 0u, 0u, value.w / denominator);
    var remainder = value.w % denominator;
    var limb = 3u;
    loop {
        if limb == 0u {
            break;
        }
        limb -= 1u;
        var bit = 32u;
        loop {
            if bit == 0u {
                break;
            }
            bit -= 1u;
            let carry = remainder >> 31u;
            remainder = (remainder << 1u) | ((value[limb] >> bit) & 1u);
            if carry != 0u || remainder >= denominator {
                remainder -= denominator;
                quotient[limb] |= 1u << bit;
            }
        }
    }
    return DivRem128(quotient, remainder);
}

// Round the ENTIRE signed surface+offset numerator, not the surface alone:
// ties away from zero are not translation-invariant across zero. Keep all
// quotient limbs and propagate the rounding carry before checking i64 range.
fn rounded_y(numerator: vec4<u32>, spacing: u32) -> RoundedY {
    let negative = (numerator.w & 0x80000000u) != 0u;
    let division = unsigned_divmod128_by32(absolute128(numerator), spacing);
    var magnitude = division.quotient;
    // Equivalent to 2*remainder >= spacing without overflowing a u32.
    if division.remainder >= spacing - division.remainder {
        magnitude = add128(magnitude, vec4<u32>(1u, 0u, 0u, 0u));
    }
    var limit = vec4<u32>(0xffffffffu, 0x7fffffffu, 0u, 0u);
    if negative {
        // abs(i64::MIN) is representable as unsigned magnitude 2^63.
        limit = vec4<u32>(0u, 0x80000000u, 0u, 0u);
    }
    if unsigned_less128(limit, magnitude) {
        return RoundedY(vec2<u32>(0u), false);
    }
    if negative {
        return RoundedY(negate128(magnitude).xy, true);
    }
    return RoundedY(magnitude.xy, true);
}

fn decode_height(encoded: i32) -> vec4<u32> {
    let bits = bitcast<u32>(encoded);
    let negative = (bits & 0x80000000u) != 0u;
    let magnitude = select(bits, 0u - bits, negative);
    // Encoded i16 magnitude <= 32768 and positive scale <= i64::MAX:
    // decoded magnitude < 2^78, including the -32768 endpoint.
    let product = multiply128_by32(vec4<u32>(param_pair(10u), 0u, 0u), magnitude).value;
    if negative {
        return negate128(product);
    }
    return product;
}

fn within_interval(value: vec4<u32>, lower_word: u32, spacing: u32) -> bool {
    let lower = multiply128_by32(extend_signed64(param_pair(lower_word)), spacing).value;
    let upper = multiply128_by32(extend_signed64(param_pair(lower_word + 2u)), spacing).value;
    return !signed_less128(value, lower) && !signed_less128(upper, value);
}

fn slope_within(dx: vec4<u32>, dz: vec4<u32>, spacing: u32) -> bool {
    let limit = multiply32_wide(spacing, param_word(20u));
    let limit128 = vec4<u32>(limit, 0u, 0u);
    let x = multiply128_by32(absolute128(dx), param_word(21u));
    let z = multiply128_by32(absolute128(dz), param_word(21u));
    // dx/dz magnitude < 2^79; multiplying by run stays below 2^111.
    // Check full products before narrowing so no large component can wrap
    // into acceptance, even if a future input widens those proven bounds.
    if x.carry != 0u || z.carry != 0u
        || unsigned_less128(limit128, x.value) || unsigned_less128(limit128, z.value) {
        return false;
    }
    // Both scaled components <= spacing*rise < 2^64. Their squares fit u128;
    // x^2 <= limit^2 proves subtraction has no borrow. Do not form x^2+z^2.
    let remaining = subtract128(
        multiply64_wide(limit, limit),
        multiply64_wide(x.value.xy, x.value.xy),
    );
    return !unsigned_less128(remaining, multiply64_wide(z.value.xy, z.value.xy));
}

// Return count as an out-of-range sentinel. Flatten padded 2D workgroups in
// 64 bits BEFORE multiplying by 64: rectangular padding can exceed u32 even
// when count fits. The last live group's final lane is <= u32::MAX.
fn invocation_index(group: vec3<u32>, lane: u32) -> u32 {
    let count = param_word(0u);
    let groups_needed = count / 64u + select(0u, 1u, count % 64u != 0u);
    let linear_group = add64(multiply32_wide(group.y, param_word(1u)), vec2<u32>(group.x, 0u));
    if linear_group.y != 0u || linear_group.x >= groups_needed {
        return count;
    }
    return linear_group.x * 64u + lane;
}

@compute @workgroup_size(64)
fn project_main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let index = invocation_index(group, lane);
    if index >= param_word(0u) {
        return;
    }
    // Overwrite every live sample, including original source rejects and
    // projection overflows, so a reused buffer never exposes stale samples.
    samples[index] = Sample(vec4<u32>(0u), vec4<u32>(0u), vec4<u32>(0u), vec4<u32>(0u));
    let source = records[index];
    if source.words[12] != 1u {
        return;
    }

    let spacing = param_word(3u);
    let side = param_word(2u);
    // Coverage proves nonnegative differences <= u64::MAX, including deltas
    // larger than i64::MAX. Modulo subtraction retains the exact unsigned bits.
    let local_x = subtract64(vec2<u32>(source.words[6], source.words[7]), param_pair(4u));
    let local_z = subtract64(vec2<u32>(source.words[10], source.words[11]), param_pair(8u));
    let x_axis = unsigned_divmod64_by32(local_x, spacing);
    let z_axis = unsigned_divmod64_by32(local_z, spacing);
    // Defensive guard; validated half-open coverage proves this cannot reject.
    if x_axis.quotient.y != 0u || z_axis.quotient.y != 0u
        || x_axis.quotient.x >= side - 1u || z_axis.quotient.x >= side - 1u {
        records[index].words[12] = 0u;
        return;
    }
    let base = z_axis.quotient.x * side + x_axis.quotient.x;
    // Four actual encoded corner reads, row-major X within Z, matching CPU.
    // side*side <= u32::MAX and cell < side-1 prove every index fits u32.
    let a = decode_height(heights[base]);
    let b = decode_height(heights[base + 1u]);
    let c = decode_height(heights[base + side]);
    let d = decode_height(heights[base + side + 1u]);
    let rx = x_axis.remainder;
    let rz = z_axis.remainder;
    var local_height: vec4<u32>;
    var dx: vec4<u32>;
    var dz: vec4<u32>;
    // Avoid rx+rz overflow. The a-c-b triangle owns the diagonal, as on CPU.
    if rx <= spacing - rz {
        local_height = add128(
            add128(
                multiply128_by32(a, spacing - rz - rx).value,
                multiply128_by32(b, rx).value,
            ),
            multiply128_by32(c, rz).value,
        );
        dx = subtract128(b, a);
        dz = subtract128(c, a);
    } else {
        local_height = add128(
            add128(
                multiply128_by32(b, spacing - rz).value,
                multiply128_by32(c, spacing - rx).value,
            ),
            multiply128_by32(d, rx - (spacing - rz)).value,
        );
        dx = subtract128(d, c);
        dz = subtract128(d, b);
    }
    // Nonnegative weights sum to spacing < 2^32, so each weighted term,
    // partial sum and the local numerator have magnitude < 2^110. OriginY
    // and offset each contribute magnitude < 2^95. Both the unoffset height
    // and full offset numerator have magnitude < 2^111, as on CPU in
    // scatter_placement.rs. All signed intermediates fit i128; absolute value
    // and the rounding increment cannot approach unsigned 128-bit overflow.
    let height = add128(
        multiply128_by32(extend_signed64(param_pair(6u)), spacing).value,
        local_height,
    );
    let numerator = add128(
        height,
        multiply128_by32(extend_signed64(param_pair(16u)), spacing).value,
    );
    let y = rounded_y(numerator, spacing);
    // CPU projects BEFORE height/slope/field filters: even a candidate those
    // filters would reject must make the entire request fail on overflow.
    if !y.valid {
        atomicAdd(&status.overflow_count, 1u);
        records[index].words[12] = 0u;
        return;
    }

    records[index].words[8] = y.value.x;
    records[index].words[9] = y.value.y;
    samples[index] = Sample(
        height,
        negate128(dx),
        vec4<u32>(spacing, 0u, 0u, 0u),
        negate128(dz),
    );
    // Valid samples and projected Y are retained even for terrain rejects.
    let accepted = within_interval(height, 12u, spacing) && slope_within(dx, dz, spacing);
    records[index].words[12] = select(0u, 1u, accepted);
}

@compute @workgroup_size(64)
fn finalize_main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let index = invocation_index(group, lane);
    if index >= param_word(0u) {
        return;
    }
    // Separate ordered dispatch observes all project_main atomics and all
    // field_main writes. Never poll status within project_main: no workgroup
    // can know that another workgroup has finished reporting its overflows.
    if atomicLoad(&status.overflow_count) != 0u {
        records[index].words[12] = 0u;
        attrs[index].z = 0u;
        if index == 0u {
            // Preserve indirect words 0, 2 and 3. Cleared count invalidates
            // the compact prefix without clearing or interpreting its tail.
            atomicStore(&indirect[1], 0u);
        }
    }
}
