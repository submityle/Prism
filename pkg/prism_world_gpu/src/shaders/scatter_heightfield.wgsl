// Integer GPU twin of HEIGHTFIELD_SURFACE_VERSION = 1 and
// HEIGHTFIELD_SCATTER_FILTER_VERSION = 1. All arithmetic is core u32 WGSL;
// i32 is used only for the storage ABI of sign-extended encoded i16 samples.
// Limb order is least significant first; signed values use two's complement.
//
// Host preconditions: positive spacing/positive i64 height scale, side >= 2,
// side*side <= u32::MAX, valid ordered filter intervals and nonzero slope run.
// The entire nonempty request XZ footprint lies within the validated half-open
// surface. Source/field records are from the same request. All storage buffers
// cover primary_count slots (heights cover side*side), dispatch Z = 1, and the
// host clears indirect[1] before dispatch, preserving indirect[0,2,3].
// Only the indirect-count compact prefix is live; compact tails are unspecified.
// Every primary attribute slot is overwritten, including rejected slots.
struct Params {
    // 0 primary_count, 1 dispatch X, 2 side, 3 spacing,
    // 4..5 origin X, 6..7 origin Y, 8..9 origin Z,
    // 10..11 positive i64 height scale,
    // 12..13 height lower, 14..15 height upper,
    // 16..17 clearance lower, 18..19 clearance upper,
    // 20 slope rise, 21 slope run, 22..31 reserved zero.
    words: array<vec4<u32>, 8>,
};

struct Record {
    // 64-byte source ABI: 6..11 position XYZ pairs, 12 source accepted.
    words: array<u32, 16>,
};

struct DivRem64 {
    quotient: vec2<u32>,
    remainder: u32,
};

struct Product128 {
    value: vec4<u32>,
    // The fifth limb of the exact unsigned 128x32 product.
    carry: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<Record>;
@group(0) @binding(2) var<storage, read> field_attributes: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> heights: array<i32>;
@group(0) @binding(4) var<storage, read_write> attributes: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> compact: array<u32>;
@group(0) @binding(6) var<storage, read_write> indirect: array<atomic<u32>, 4>;

// Existing inclusive scan: 256 bytes plus a four-byte group reservation.
var<workgroup> scan: array<u32, 64>;
var<workgroup> output_base: u32;

fn param_word(index: u32) -> u32 {
    return params.words[index / 4u][index % 4u];
}

fn param_pair(index: u32) -> vec2<u32> {
    return vec2<u32>(param_word(index), param_word(index + 1u));
}

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
    // Check the full products before narrowing. Even if a future wider input
    // overflowed 128 bits, its carry would reject rather than wrap to a pass.
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

fn terrain_accepts(source: Record) -> bool {
    let spacing = param_word(3u);
    let side = param_word(2u);
    // Host coverage proves these mathematical signed-64 endpoint differences
    // are nonnegative and <= u64::MAX. Modulo subtraction retains exact bits.
    let local_x = subtract64(vec2<u32>(source.words[6], source.words[7]), param_pair(4u));
    let local_z = subtract64(vec2<u32>(source.words[10], source.words[11]), param_pair(8u));
    let x_axis = unsigned_divmod64_by32(local_x, spacing);
    let z_axis = unsigned_divmod64_by32(local_z, spacing);
    // Coverage proves high quotient limbs are zero and cell < side-1.
    // Keep a defensive index guard as well; exclusive surface edges reject.
    if x_axis.quotient.y != 0u || z_axis.quotient.y != 0u
        || x_axis.quotient.x >= side - 1u || z_axis.quotient.x >= side - 1u {
        return false;
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
    // Equivalent to rx+rz <= spacing without overflowing u32 at large spacing.
    // The first (a-c-b) triangle owns the exact diagonal, as on CPU.
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
    // Nonnegative triangle weights sum to spacing < 2^32, so local height
    // numerator has magnitude < 2^110. OriginY*spacing is < 2^95; signed
    // height, clearance, interval bounds and intermediates all fit i128.
    let height = add128(
        multiply128_by32(extend_signed64(param_pair(6u)), spacing).value,
        local_height,
    );
    let point_y = extend_signed64(vec2<u32>(source.words[8], source.words[9]));
    let clearance = subtract128(multiply128_by32(point_y, spacing).value, height);
    return within_interval(height, 12u, spacing)
        && within_interval(clearance, 16u, spacing)
        && slope_within(dx, dz, spacing);
}

@compute @workgroup_size(64)
fn terrain_main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let primary_count = param_word(0u);
    let groups_needed = primary_count / 64u + select(0u, 1u, primary_count % 64u != 0u);
    // Match field_main: flatten rectangular padding in 64 bits and reject
    // entire padding groups BEFORE multiplying by 64. Even at primary_count
    // == u32::MAX, the last live group's last index is at most u32::MAX.
    let linear_group = add64(multiply32_wide(group.y, param_word(1u)), vec2<u32>(group.x, 0u));
    var index = 0u;
    var accepted = 0u;
    if linear_group.y == 0u && linear_group.x < groups_needed {
        index = linear_group.x * 64u + lane;
        if index < primary_count {
            let field = field_attributes[index];
            let source = sources[index];
            if source.words[12] == 1u && field.z != 0u {
                if terrain_accepts(source) {
                    accepted = 1u;
                }
            }
            // Preserve density/species/rank even for field or terrain rejects.
            // Every live slot is written, including source-rejected zero attrs.
            attributes[index] = vec4<u32>(field.xy, accepted, field.w);
        }
    }

    // No early returns: padding lanes and rejects reach every uniform barrier.
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
        // Finish all old-scan reads before publishing the next scan values.
        workgroupBarrier();
        scan[lane] = scan[lane] + preceding;
        workgroupBarrier();
    }
    if lane == 0u {
        let count = scan[63];
        if count != 0u {
            output_base = atomicAdd(&indirect[1], count);
        }
    }
    workgroupBarrier();
    if accepted != 0u {
        // Source slot indices stay unchanged. Within-group order is stable;
        // group reservation order is unspecified, matching the source shaders.
        compact[output_base + scan[lane] - 1u] = index;
    }
}
