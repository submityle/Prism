// Standalone, integer-only unsigned 256-bit arithmetic for WGSL.
//
// Representation: limb[0] is least significant; value = sum(limb[i] * 2^(32*i)).
// No floating-point values, atomics, or bindings are used. This file is deliberately
// not included by a production shader.
//
// Bounds and contracts:
// - add/sub/mul/shift return the low 256 bits and report discarded carry/borrow/bits.
// - mul performs full 8x8 schoolbook multiplication. Each 32x32 partial product is
//   decomposed into two u32 words; carry propagation is explicit.
// - div_rem accepts any nonzero divisor and returns quotient/remainder. For its
//   planetary use, numerator width is <=230 bits and divisor width <=147 bits, so
//   the per-step remainder shift cannot overflow U256. Callers must enforce these
//   bounds; an overflow result is not a valid quotient/remainder pair.
// - isqrt returns floor(sqrt(value)); its restoring algorithm uses 128 radix-4 steps.
// - Fixed loop bounds are independent of the operand values. No operation traps.

struct U256 {
    limb: array<u32, 8>,
};

struct U256Carry {
    value: U256,
    overflow: bool,
};

struct U256Borrow {
    value: U256,
    underflow: bool,
};

struct U256DivRem {
    quotient: U256,
    remainder: U256,
    division_by_zero: bool,
};

struct I256 {
    negative: bool,
    magnitude: U256,
};

fn u256_zero() -> U256 {
    return U256(array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));
}

fn u256_from_u32(value: u32) -> U256 {
    return U256(array<u32, 8>(value, 0u, 0u, 0u, 0u, 0u, 0u, 0u));
}

fn u256_is_zero(value: U256) -> bool {
    var any_set = false;
    for (var i = 0u; i < 8u; i += 1u) {
        any_set = any_set || value.limb[i] != 0u;
    }
    return !any_set;
}

// Returns -1, 0, +1 for a < b, a == b, a > b respectively.
fn u256_cmp(a: U256, b: U256) -> i32 {
    for (var i = 8u; i > 0u; i -= 1u) {
        let index = i - 1u;
        if (a.limb[index] < b.limb[index]) { return -1; }
        if (a.limb[index] > b.limb[index]) { return 1; }
    }
    return 0;
}

fn u256_add(a: U256, b: U256) -> U256Carry {
    var out = u256_zero();
    var carry = 0u;
    for (var i = 0u; i < 8u; i += 1u) {
        let first = a.limb[i] + b.limb[i];
        let carry_first = first < a.limb[i];
        let second = first + carry;
        let carry_second = second < first;
        out.limb[i] = second;
        carry = select(0u, 1u, carry_first || carry_second);
    }
    return U256Carry(out, carry != 0u);
}

// Subtraction is modulo 2^256; underflow reports a < b.
fn u256_sub(a: U256, b: U256) -> U256Borrow {
    var out = u256_zero();
    var borrow = 0u;
    for (var i = 0u; i < 8u; i += 1u) {
        let first = a.limb[i] - b.limb[i];
        let borrow_first = a.limb[i] < b.limb[i];
        let second = first - borrow;
        let borrow_second = first < borrow;
        out.limb[i] = second;
        borrow = select(0u, 1u, borrow_first || borrow_second);
    }
    return U256Borrow(out, borrow != 0u);
}

// Exact 32x32 -> (high, low), using only WGSL u32 arithmetic.
fn u256_mul_word(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 65535u;
    let a1 = a >> 16u;
    let b0 = b & 65535u;
    let b1 = b >> 16u;
    let t0 = a0 * b0;
    let w0 = t0 & 65535u;
    let k0 = t0 >> 16u;
    let t1 = a1 * b0 + k0;
    let w1 = t1 & 65535u;
    let k1 = t1 >> 16u;
    let t2 = a0 * b1 + w1;
    let high = a1 * b1 + k1 + (t2 >> 16u);
    let low = (t2 << 16u) | w0;
    return vec2<u32>(high, low);
}

fn u256_add_word_at(value: ptr<function, U256>, index: u32, word: u32) -> bool {
    var carry = word;
    var i = index;
    loop {
        if (carry == 0u) { return false; }
        if (i >= 8u) { return true; }
        let old = (*value).limb[i];
        let sum = old + carry;
        (*value).limb[i] = sum;
        carry = select(0u, 1u, sum < old);
        i += 1u;
    }
    return false;
}

// Full schoolbook product, retaining low 256 bits and reporting any high bits.
fn u256_mul(a: U256, b: U256) -> U256Carry {
    var out = u256_zero();
    var overflow = false;
    for (var i = 0u; i < 8u; i += 1u) {
        for (var j = 0u; j < 8u; j += 1u) {
            let product = u256_mul_word(a.limb[i], b.limb[j]);
            let index = i + j;
            if (index < 8u) {
                overflow = u256_add_word_at(&out, index, product.y) || overflow;
            } else {
                overflow = overflow || product.y != 0u;
            }
            if (index + 1u < 8u) {
                overflow = u256_add_word_at(&out, index + 1u, product.x) || overflow;
            } else {
                overflow = overflow || product.x != 0u;
            }
        }
    }
    return U256Carry(out, overflow);
}

fn u256_bit(value: U256, bit_index: u32) -> u32 {
    return (value.limb[bit_index >> 5u] >> (bit_index & 31u)) & 1u;
}

fn u256_set_bit(value: ptr<function, U256>, bit_index: u32) {
    let limb_index = bit_index >> 5u;
    (*value).limb[limb_index] = (*value).limb[limb_index] | (1u << (bit_index & 31u));
}

fn u256_bit_length(value: U256) -> u32 {
    for (var cursor = 8u; cursor > 0u; cursor -= 1u) {
        let limb = value.limb[cursor - 1u];
        if (limb != 0u) { return (cursor - 1u) * 32u + 32u - countLeadingZeros(limb); }
    }
    return 0u;
}

fn u256_shr_one(value: U256) -> U256 {
    var out = u256_zero();
    for (var cursor = 8u; cursor > 0u; cursor -= 1u) {
        let index = cursor - 1u;
        out.limb[index] = value.limb[index] >> 1u;
        if (index < 7u) { out.limb[index] = out.limb[index] | (value.limb[index + 1u] << 31u); }
    }
    return out;
}

fn u256_shr(value: U256, count: u32) -> U256 {
    var result = u256_zero();
    if (count >= 256u) { return result; }
    let word_shift = count >> 5u;
    let bit_shift = count & 31u;
    for (var destination = 0u; destination < 8u; destination += 1u) {
        let source = destination + word_shift;
        if (source < 8u) {
            result.limb[destination] = value.limb[source] >> bit_shift;
            if (bit_shift != 0u && source + 1u < 8u) {
                result.limb[destination] = result.limb[destination] | (value.limb[source + 1u] << (32u - bit_shift));
            }
        }
    }
    return result;
}

fn u256_low_bits(value: U256, count: u32) -> U256 {
    if (count >= 256u) { return value; }
    var result = u256_zero();
    let full_limbs = count >> 5u;
    let remaining_bits = count & 31u;
    for (var limb = 0u; limb < full_limbs; limb += 1u) {
        result.limb[limb] = value.limb[limb];
    }
    if (remaining_bits != 0u) { result.limb[full_limbs] = value.limb[full_limbs] & ((1u << remaining_bits) - 1u); }
    return result;
}

// Shift left by count. count may be >= 256; all shifted-out nonzero bits are reported.
fn u256_shl(value: U256, count: u32) -> U256Carry {
    var out = u256_zero();
    if (count >= 256u) {
        return U256Carry(out, !u256_is_zero(value));
    }
    let word_shift = count >> 5u;
    let bit_shift = count & 31u;
    var overflow = false;
    for (var i = 0u; i < 8u; i += 1u) {
        let destination = i + word_shift;
        if (destination < 8u) {
            out.limb[destination] = out.limb[destination] | (value.limb[i] << bit_shift);
        } else {
            overflow = overflow || value.limb[i] != 0u;
        }
        if (bit_shift != 0u) {
            let upper = value.limb[i] >> (32u - bit_shift);
            if (destination + 1u < 8u) {
                out.limb[destination + 1u] = out.limb[destination + 1u] | upper;
            } else {
                overflow = overflow || upper != 0u;
            }
        }
    }
    return U256Carry(out, overflow);
}

fn u256_div_rem(numerator: U256, denominator: U256) -> U256DivRem {
    var quotient = u256_zero();
    var remainder = u256_zero();
    if (u256_is_zero(denominator)) {
        return U256DivRem(quotient, remainder, true);
    }
    let numerator_bits = u256_bit_length(numerator);
    let denominator_bits = u256_bit_length(denominator);
    if (numerator_bits < denominator_bits) { return U256DivRem(quotient, numerator, false); }
    let shift = numerator_bits - denominator_bits;
    var shifted_divisor = u256_shl(denominator, shift);
    if (shifted_divisor.overflow) { return U256DivRem(u256_zero(), u256_zero(), false); }
    remainder = numerator;
    for (var cursor = shift + 1u; cursor > 0u; cursor -= 1u) {
        let bit_index = cursor - 1u;
        if (u256_cmp(remainder, shifted_divisor.value) >= 0) {
            remainder = u256_sub(remainder, shifted_divisor.value).value;
            u256_set_bit(&quotient, bit_index);
        }
        shifted_divisor.value = u256_shr_one(shifted_divisor.value);
    }
    return U256DivRem(quotient, remainder, false);
}

// Restoring radix-4 square root, returning floor(sqrt(value)).
fn u256_isqrt(value: U256) -> U256 {
    var root = u256_zero();
    var remainder = u256_zero();
    for (var cursor = 128u; cursor > 0u; cursor -= 1u) {
        let pair_index = cursor - 1u;
        let shifted_rem = u256_shl(remainder, 2u);
        remainder = shifted_rem.value;
        let pair = (u256_bit(value, pair_index * 2u + 1u) << 1u) | u256_bit(value, pair_index * 2u);
        remainder.limb[0] = remainder.limb[0] | pair;

        let shifted_root = u256_shl(root, 1u);
        root = shifted_root.value;
        let trial_shift = u256_shl(root, 1u);
        var trial = trial_shift.value;
        trial.limb[0] = trial.limb[0] | 1u;
        if (u256_cmp(remainder, trial) >= 0) {
            remainder = u256_sub(remainder, trial).value;
            root = u256_add(root, u256_from_u32(1u)).value;
        }
    }
    return root;
}

// Restoring square root of `value << shift` without materializing the shifted
// radicand. `shift` must be even and the resulting root must fit U256. This is
// used by Q84 cube-sphere normalization: Q < 2^126 and shift=168, so the
// conceptual radicand is <2^294 while the root remains <2^147.
fn u256_isqrt_scaled(value: U256, shift: u32) -> U256 {
    let value_bits = u256_bit_length(value);
    if (value_bits == 0u) { return u256_zero(); }
    let total_bits = value_bits + shift;
    let pair_count = (total_bits + 1u) / 2u;
    var root = u256_zero();
    var remainder = u256_zero();
    for (var cursor = pair_count; cursor > 0u; cursor -= 1u) {
        let pair_index = cursor - 1u;
        let bit_low = pair_index * 2u;
        var incoming = 0u;
        if (bit_low >= shift) {
            let source_low = bit_low - shift;
            if (source_low < 256u) { incoming = u256_bit(value, source_low); }
            if (source_low + 1u < 256u) { incoming = incoming | (u256_bit(value, source_low + 1u) << 1u); }
        } else if (bit_low + 1u >= shift) {
            let source = bit_low + 1u - shift;
            if (source < 256u) { incoming = u256_bit(value, source) << 1u; }
        }
        remainder = u256_shl(remainder, 2u).value;
        remainder.limb[0] = remainder.limb[0] | incoming;
        root = u256_shl(root, 1u).value;
        var trial = u256_shl(root, 1u).value;
        trial.limb[0] = trial.limb[0] | 1u;
        if (u256_cmp(remainder, trial) >= 0) {
            remainder = u256_sub(remainder, trial).value;
            root = u256_add(root, u256_from_u32(1u)).value;
        }
    }
    return root;
}

// Normalize integer cube numerators to signed Qprecision_bits. For planetary
// inputs |cube[i]| < 2^62 and precision_bits <= 84; sum(cube[i]^2) < 3*2^124,
// sqrt(sum << 168) < 2^147, and each shifted component < 2^230. Lower adaptive
// precision only reduces shifts and result bit widths. Every operation fits U256.
fn i256_canonical(value: I256) -> I256 {
    return I256(value.negative && !u256_is_zero(value.magnitude), value.magnitude);
}

fn u256_add_checked(a: U256, b: U256) -> U256 {
    return u256_add(a, b).value;
}

fn u256_sub_checked(a: U256, b: U256) -> U256 {
    return u256_sub(a, b).value;
}

fn i256_from_i64_words(value: vec2<u32>) -> I256 {
    let negative = (value.y & 0x80000000u) != 0u;
    var limbs = value;
    if (negative) {
        limbs = ~limbs;
        limbs.x += 1u;
        if (limbs.x == 0u) { limbs.y += 1u; }
    }
    var magnitude = u256_zero();
    magnitude.limb[0] = limbs.x;
    magnitude.limb[1] = limbs.y;
    return i256_canonical(I256(negative, magnitude));
}

fn i256_from_i128_words(value: vec4<u32>) -> I256 {
    let negative = (value.w & 0x80000000u) != 0u;
    var limbs = value;
    if (negative) {
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
    var magnitude = u256_zero();
    magnitude.limb[0] = limbs.x;
    magnitude.limb[1] = limbs.y;
    magnitude.limb[2] = limbs.z;
    magnitude.limb[3] = limbs.w;
    return i256_canonical(I256(negative, magnitude));
}

fn i256_add(a: I256, b: I256) -> I256 {
    if (a.negative == b.negative) {
        return i256_canonical(I256(a.negative, u256_add_checked(a.magnitude, b.magnitude)));
    }
    if (u256_cmp(a.magnitude, b.magnitude) >= 0) {
        return i256_canonical(I256(a.negative, u256_sub_checked(a.magnitude, b.magnitude)));
    }
    return i256_canonical(I256(b.negative, u256_sub_checked(b.magnitude, a.magnitude)));
}

fn project_q_to_mm(normal: I256, radial_quarters_mm: I256, precision_bits: u32) -> I256 {
    let product = u256_mul(normal.magnitude, radial_quarters_mm.magnitude).value;
    let scale_bits = precision_bits + 2u;
    let quotient = u256_shr(product, scale_bits);
    let remainder = u256_low_bits(product, scale_bits);
    var rounded = quotient;
    let half = u256_from_u32(1u);
    let half_shifted = u256_shl(half, scale_bits - 1u).value;
    if (u256_cmp(remainder, half_shifted) >= 0) {
        rounded = u256_add_checked(rounded, u256_from_u32(1u));
    }
    return i256_canonical(I256(normal.negative != radial_quarters_mm.negative, rounded));
}

fn cube_direction_q(cube: array<I256, 3>, precision_bits: u32) -> array<I256, 3> {
    var norm_squared = u256_zero();
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let square = u256_mul(cube[axis].magnitude, cube[axis].magnitude);
        norm_squared = u256_add_checked(norm_squared, square.value);
    }
    let doubled_precision = precision_bits * 2u;
    let length_q = u256_isqrt_scaled(norm_squared, doubled_precision);
    var result: array<I256, 3>;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let numerator = u256_shl(cube[axis].magnitude, doubled_precision).value;
        let division = u256_div_rem(numerator, length_q);
        var magnitude = division.quotient;
        if (division.remainder.limb[0] != 0u || division.remainder.limb[1] != 0u
            || division.remainder.limb[2] != 0u || division.remainder.limb[3] != 0u
            || division.remainder.limb[4] != 0u || division.remainder.limb[5] != 0u
            || division.remainder.limb[6] != 0u || division.remainder.limb[7] != 0u) {
            let complement = u256_sub_checked(length_q, division.remainder);
            if (u256_cmp(division.remainder, complement) >= 0) {
                magnitude = u256_add_checked(magnitude, u256_from_u32(1u));
            }
        }
        result[axis] = i256_canonical(I256(cube[axis].negative, magnitude));
    }
    return result;
}

// Exact doubled cell-centre numerators for the cube-sphere map. The common
// denominator cells * 2^level cancels during normalization. Valid inputs are
// constrained by the placement contract: level <= 30, cells < 2^32, cell < cells.
fn cell_axis_numerator_q84_input(tile: u32, cell: u32, cells: u32, level: u32) -> I256 {
    let tile_product = u256_mul(u256_from_u32(cells), u256_from_u32(tile)).value;
    let global_twice_base = u256_shl(tile_product, 1u).value;
    let cell_twice_plus_one = u256_add(
        u256_shl(u256_from_u32(cell), 1u).value,
        u256_from_u32(1u),
    ).value;
    let global_twice = u256_add_checked(global_twice_base, cell_twice_plus_one);
    let face_tiles = u256_shl(u256_from_u32(1u), level).value;
    let denominator = u256_mul(u256_from_u32(cells), face_tiles).value;
    if (u256_cmp(global_twice, denominator) >= 0) {
        return I256(false, u256_sub_checked(global_twice, denominator));
    }
    return I256(true, u256_sub_checked(denominator, global_twice));
}

fn cube_numerators_for_cell(face: u32, tile_xy: vec2<u32>, cell_xy: vec2<u32>, cells: u32, level: u32) -> array<I256, 3> {
    let u = cell_axis_numerator_q84_input(tile_xy.x, cell_xy.x, cells, level);
    let v = cell_axis_numerator_q84_input(tile_xy.y, cell_xy.y, cells, level);
    let d = u256_mul(u256_from_u32(cells), u256_shl(u256_from_u32(1u), level).value).value;
    let positive_d = I256(false, d);
    let negative_d = I256(true, d);
    switch face {
        case 0u: { return array<I256, 3>(positive_d, v, I256(!u.negative, u.magnitude)); }
        case 1u: { return array<I256, 3>(negative_d, v, u); }
        case 2u: { return array<I256, 3>(u, positive_d, I256(!v.negative, v.magnitude)); }
        case 3u: { return array<I256, 3>(u, negative_d, v); }
        case 4u: { return array<I256, 3>(u, v, positive_d); }
        default: { return array<I256, 3>(I256(!u.negative, u.magnitude), v, negative_d); }
    }
}
