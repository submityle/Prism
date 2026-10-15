// Integer GPU twin of scatter_field.rs, SCATTER_FIELD_VERSION = 1.
// Pairs are (low, high), including two's-complement signed coordinates.
// Only core WGSL u32 arithmetic is used: no floats, i64, or optional features.
// The host validates the recipe, sorts brushes by ID, sizes all buffers, uses
// dispatch Z = 1, and clears indirect[1] before dispatch. Source flags are never
// mutated. Uniform words 19..31 and brush word 15 are reserved zeroes.
struct Params {
    // 0..1 field seed, 2 base density, 3 base species, 4 noise enabled,
    // 5 cell size, 6 amplitude, 7 warp cell size, 8 warp amplitude,
    // 9 brush count, 10 primary count, 11 dispatch X,
    // 12..13 scatter seed, 14 scatter version, 15 spacing,
    // 16..17 scatter salt, 18 scatter kernel version (1).
    words: array<vec4<u32>, 8>,
};

struct Record {
    // Scalar words keep the storage array stride exactly 64 bytes.
    // Source: 0..5 grid XYZ pairs, 6..11 position XYZ pairs, 12 accepted.
    // Brush: 0..1 ID (ordering only), 2..7 centre XYZ pairs, 8 inner radius,
    // 9 outer radius, 10 opacity, 11 op (0 replace / 1 add), 12 value,
    // 13 species enabled, 14 species. Add's value is an i32 bit pattern.
    words: array<u32, 16>,
};

struct DivRem64 {
    quotient: vec2<u32>,
    remainder: u32,
};

struct LatticeAxis {
    grid: vec2<u32>,
    remainder: u32,
};

struct SignedOffset {
    magnitude: u32,
    negative: bool,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sources: array<Record>;
@group(0) @binding(2) var<storage, read> brushes: array<Record>;
@group(0) @binding(3) var<storage, read_write> attributes: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> compact: array<u32>;
@group(0) @binding(5) var<storage, read_write> indirect: array<atomic<u32>, 4>;

// Same 260-byte inclusive scan and single group reservation as scatter.wgsl.
var<workgroup> scan: array<u32, 64>;
var<workgroup> output_base: u32;

fn param_word(index: u32) -> u32 {
    return params.words[index / 4u][index % 4u];
}

fn param_pair(index: u32) -> vec2<u32> {
    return vec2<u32>(param_word(index), param_word(index + 1u));
}

fn unsigned_less64(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

fn signed_less64(a: vec2<u32>, b: vec2<u32>) -> bool {
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

fn negate64(value: vec2<u32>) -> vec2<u32> {
    return subtract64(vec2<u32>(0u), value);
}

// Exact 32x32 -> 64, using 16-bit limbs. Each product and carry sum fits u32.
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
    let lower = multiply32_wide(a.x, b.x);
    // Deliberately modulo 2^64, used only for hashes/lattice coordinates.
    return vec2<u32>(lower.x, lower.y + a.x * b.y + a.y * b.x);
}

// Exact 64x32 -> 96, low limb first. No high bits are discarded.
fn multiply64_by32_to96(a: vec2<u32>, b: u32) -> vec3<u32> {
    let lower = multiply32_wide(a.x, b);
    let upper = multiply32_wide(a.y, b);
    let middle = lower.y + upper.x;
    return vec3<u32>(lower.x, middle, upper.y + select(0u, 1u, middle < lower.y));
}

fn unsigned_less96(a: vec3<u32>, b: vec3<u32>) -> bool {
    if a.z != b.z {
        return a.z < b.z;
    }
    if a.y != b.y {
        return a.y < b.y;
    }
    return a.x < b.x;
}

// Unsigned long division by a positive u32. Divide the high limb directly,
// then bring down 32 low bits. The mathematical shifted remainder is at most
// 2*denominator-1 (33 bits); retain its carry before shifting, so denominators
// above 2^31 work too. One subtraction restores remainder < denominator.
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
            // If carry is set, wrapped subtraction produces the exact reduced
            // remainder: the omitted 2^32 cancels the subtraction's borrow.
            remainder -= denominator;
            quotient.x |= 1u << bit;
        }
    }
    return DivRem64(quotient, remainder);
}

// Signed i64 / positive u32 with Euclidean (floor) quotient and remainder.
// Unsigned magnitude represents abs(i64::MIN) exactly; never negate an i64.
fn signed_floor_divrem64_by32(value: vec2<u32>, denominator: u32) -> LatticeAxis {
    let negative = (value.y & 0x80000000u) != 0u;
    var magnitude = value;
    if negative {
        magnitude = negate64(value);
    }
    let division = unsigned_divmod64_by32(magnitude, denominator);
    if !negative {
        return LatticeAxis(division.quotient, division.remainder);
    }
    var quotient = negate64(division.quotient);
    var remainder = division.remainder;
    if remainder != 0u {
        quotient = subtract64(quotient, vec2<u32>(1u, 0u));
        remainder = denominator - remainder;
    }
    return LatticeAxis(quotient, remainder);
}

// Compute floor((original + offset)/size) without ever constructing a wrapped
// or clamped virtual position. If original = q*size+r, divide r+offset, then
// add its quotient to q. Here r+offset is in [-u32::MAX, 2*u32::MAX-1], so its
// signed-64 representation is exact even at original i64 MIN/MAX. Only the
// final lattice grid is reduced modulo 2^64, as the CPU hash contract requires.
fn lattice_axis(original: vec2<u32>, size: u32, offset: SignedOffset) -> LatticeAxis {
    let initial = signed_floor_divrem64_by32(original, size);
    if offset.magnitude == 0u {
        return initial;
    }
    var shifted_remainder = vec2<u32>(initial.remainder, 0u);
    if offset.negative {
        shifted_remainder = subtract64(shifted_remainder, vec2<u32>(offset.magnitude, 0u));
    } else {
        shifted_remainder = add64(shifted_remainder, vec2<u32>(offset.magnitude, 0u));
    }
    let adjustment = signed_floor_divrem64_by32(shifted_remainder, size);
    return LatticeAxis(add64(initial.grid, adjustment.grid), adjustment.remainder);
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
    if amount == 0u {
        return value;
    }
    return (value >> amount) | (value << (32u - amount));
}

// Copied integer PCG/mix arithmetic from scatter.wgsl; this is the CPU's
// pcg_hash_u32(u64), not a substitute 32-bit avalanche or density hash.
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

// Nearest-half-up, NOT interpolation of a signed difference. The two weights
// sum to ONE, so the weighted sum plus 32767 <= ONE^2+32767 < 2^32.
fn lerp16(a: u32, b: u32, weight: u32) -> u32 {
    return (a * (65535u - weight) + b * weight + 32767u) / 65535u;
}

// Exact t^2*(3*ONE-2*t)/ONE^2 with nearest-half-up rounding. t^2 fits u32,
// the numerator <= ONE^3 < 2^48, and ONE^2 = 4294836225 fits a u32 divisor.
fn smoothstep16(weight: u32) -> u32 {
    let numerator = multiply32_wide(weight * weight, 196605u - 2u * weight);
    let rounded = add64(numerator, vec2<u32>(2147418112u, 0u));
    return unsigned_divmod64_by32(rounded, 4294836225u).quotient.x;
}

fn value_noise(
    original: array<vec2<u32>, 3>,
    cell_size: u32,
    channel: u32,
    offsets: array<SignedOffset, 3>,
) -> u32 {
    var grid: array<vec2<u32>, 3>;
    var weights: array<u32, 3>;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let coordinate = lattice_axis(original[axis], cell_size, offsets[axis]);
        grid[axis] = coordinate.grid;
        // Floor the lattice fraction BEFORE smoothstep, exactly as the CPU.
        let fraction = unsigned_divmod64_by32(
            multiply32_wide(coordinate.remainder, 65535u), cell_size,
        ).quotient.x;
        weights[axis] = smoothstep16(fraction);
    }
    let tag = multiply64(vec2<u32>(channel, 0u), vec2<u32>(0xd192ed03u, 0xd1b54a32u));
    let base = hash64(param_pair(0u) ^ tag);
    var corners: array<u32, 8>;
    for (var x = 0u; x < 2u; x += 1u) {
        let hash_x = hash64(base ^ add64(grid[0], vec2<u32>(x, 0u)));
        for (var y = 0u; y < 2u; y += 1u) {
            let hash_xy = hash64(hash_x ^ add64(grid[1], vec2<u32>(y, 0u)));
            for (var z = 0u; z < 2u; z += 1u) {
                let hash = hash64(hash_xy ^ add64(grid[2], vec2<u32>(z, 0u)));
                corners[x * 4u + y * 2u + z] = pcg_hash_u32(hash) >> 16u;
            }
        }
    }
    // Z -> Y -> X, rounding at EVERY lerp, not one eight-corner rational sum.
    let x0 = lerp16(
        lerp16(corners[0], corners[1], weights[2]),
        lerp16(corners[2], corners[3], weights[2]),
        weights[1],
    );
    let x1 = lerp16(
        lerp16(corners[4], corners[5], weights[2]),
        lerp16(corners[6], corners[7], weights[2]),
        weights[1],
    );
    return lerp16(x0, x1, weights[0]);
}

// (2*value-ONE)*amplitude/ONE, truncating toward zero. Splitting sign from
// magnitude reproduces signed division without i64; the product is at most
// ONE*u32::MAX (< 2^48) and the quotient never exceeds amplitude (u32).
fn noise_offset(value: u32, amplitude: u32) -> SignedOffset {
    let doubled = 2u * value;
    let negative = doubled < 65535u;
    var magnitude = 0u;
    if negative {
        magnitude = 65535u - doubled;
    } else {
        magnitude = doubled - 65535u;
    }
    let scaled = unsigned_divmod64_by32(multiply32_wide(magnitude, amplitude), 65535u);
    return SignedOffset(scaled.quotient.x, negative);
}

// Used only for density edits, where magnitude <= ONE. Neither subtraction
// underflow nor a signed cast is needed; the positive sum is at most 131070.
fn apply_density_offset(density: u32, offset: SignedOffset) -> u32 {
    if offset.negative {
        if offset.magnitude >= density {
            return 0u;
        }
        return density - offset.magnitude;
    }
    return min(65535u, density + offset.magnitude);
}

// floor(numerator*ONE/denominator), with 0 <= numerator < denominator.
// The scaled numerator can need 80 bits. Build the 16-bit quotient from high
// to low (binary search), comparing exact 96-bit products; neither a wrapped
// u64 product nor an approximate/rounded ratio ever enters brush falloff.
fn radial_ratio16(numerator: vec2<u32>, denominator: vec2<u32>) -> u32 {
    let scaled = multiply64_by32_to96(numerator, 65535u);
    var quotient = 0u;
    var bit = 16u;
    loop {
        if bit == 0u {
            break;
        }
        bit -= 1u;
        let trial = quotient | (1u << bit);
        if !unsigned_less96(scaled, multiply64_by32_to96(denominator, trial)) {
            quotient = trial;
        }
    }
    return quotient;
}

fn brush_weight(brush: Record, position: array<vec2<u32>, 3>) -> u32 {
    let opacity = brush.words[10];
    if opacity == 0u {
        return 0u;
    }
    let radius = brush.words[9];
    let outer = multiply32_wide(radius, radius);
    var remaining = outer;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        let center = vec2<u32>(brush.words[2u + axis * 2u], brush.words[3u + axis * 2u]);
        var delta: vec2<u32>;
        if signed_less64(position[axis], center) {
            delta = subtract64(center, position[axis]);
        } else {
            delta = subtract64(position[axis], center);
        }
        // Opposite signed endpoints can differ by u64::MAX. Compare BEFORE
        // narrowing/squaring; distance >= radius always has zero weight.
        if delta.y != 0u || delta.x >= radius {
            return 0u;
        }
        let square = multiply32_wide(delta.x, delta.x);
        // Strict outer boundary. Subtract squares from the remaining budget,
        // so the three-axis squared distance cannot overflow an intermediate.
        if !unsigned_less64(square, remaining) {
            return 0u;
        }
        remaining = subtract64(remaining, square);
    }
    let inner = multiply32_wide(brush.words[8], brush.words[8]);
    let distance = subtract64(outer, remaining);
    var radial = 65535u;
    if unsigned_less64(inner, distance) {
        // This branch proves inner < distance < outer, hence nonzero divisor
        // and remaining < outer-inner. Equal radii take the hard-brush path.
        radial = smoothstep16(radial_ratio16(remaining, subtract64(outer, inner)));
    }
    // Product <= ONE^2; adding 32767 still fits u32. Nearest-half-up opacity.
    return (radial * opacity + 32767u) / 65535u;
}

fn sample_field(position: array<vec2<u32>, 3>) -> vec2<u32> {
    var density = param_word(2u);
    if param_word(4u) != 0u {
        // Function variables start at zero: no displacement for the warp
        // lookups themselves, each of which reads the SAME original position.
        var zero_offsets: array<SignedOffset, 3>;
        var offsets: array<SignedOffset, 3>;
        let warp_amplitude = param_word(8u);
        if warp_amplitude != 0u {
            for (var axis = 0u; axis < 3u; axis += 1u) {
                let value = value_noise(position, param_word(7u), axis + 1u, zero_offsets);
                offsets[axis] = noise_offset(value, warp_amplitude);
            }
        }
        let value = value_noise(position, param_word(5u), 0u, offsets);
        density = apply_density_offset(density, noise_offset(value, param_word(6u)));
    }
    var species = param_word(3u);
    // The host uploads canonical ascending-ID order. IDs do not enter math.
    for (var brush_index = 0u; brush_index < param_word(9u); brush_index += 1u) {
        let brush = brushes[brush_index];
        let weight = brush_weight(brush, position);
        if weight == 0u {
            continue;
        }
        if brush.words[11] == 0u {
            density = lerp16(density, brush.words[12], weight);
        } else {
            let encoded = brush.words[12];
            let negative = (encoded & 0x80000000u) != 0u;
            var magnitude = encoded;
            if negative {
                magnitude = 0u - encoded;
            }
            // Validated |Add| <= ONE: product <= ONE^2 fits u32. Unsigned
            // division of the magnitude gives signed truncation toward zero.
            let delta = SignedOffset(magnitude * weight / 65535u, negative);
            density = apply_density_offset(density, delta);
        }
        if weight >= 32768u && brush.words[13] != 0u {
            species = brush.words[14];
        }
    }
    return vec2<u32>(density, species);
}

fn candidate_rank(grid: array<vec2<u32>, 3>) -> u32 {
    // CPU candidate_rank: field seed, domain tag, then EVERY structural key
    // field in order. No source slot, position, density, or request-local data.
    var hash = hash64(param_pair(0u) ^ vec2<u32>(0x645f726bu, 0x6669656cu));
    hash = hash64(hash ^ vec2<u32>(param_word(18u), 0u));
    hash = hash64(hash ^ param_pair(12u));
    hash = hash64(hash ^ vec2<u32>(param_word(14u), 0u));
    hash = hash64(hash ^ param_pair(16u));
    hash = hash64(hash ^ vec2<u32>(param_word(15u), 0u));
    hash = hash64(hash ^ grid[0]);
    hash = hash64(hash ^ grid[1]);
    hash = hash64(hash ^ grid[2]);
    return pcg_hash_u32(hash);
}

@compute @workgroup_size(64)
fn field_main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let primary_count = param_word(10u);
    let groups_needed = primary_count / 64u + select(0u, 1u, primary_count % 64u != 0u);
    // Rectangular dispatch may contain entire padding groups. Reject them
    // BEFORE multiplying by 64; wide group flattening also prevents overflow
    // from wrapping a padding group back onto a real source slot.
    let linear_group = add64(multiply32_wide(group.y, param_word(11u)), vec2<u32>(group.x, 0u));
    var index = 0u;
    var accepted = 0u;
    if linear_group.y == 0u && linear_group.x < groups_needed {
        index = linear_group.x * 64u + lane;
        if index < primary_count {
            // Rejected source slots are ALWAYS zero, with no field/rank work.
            attributes[index] = vec4<u32>(0u);
            let source = sources[index];
            if source.words[12] == 1u {
                var grid: array<vec2<u32>, 3>;
                var position: array<vec2<u32>, 3>;
                for (var axis = 0u; axis < 3u; axis += 1u) {
                    grid[axis] = vec2<u32>(source.words[axis * 2u], source.words[axis * 2u + 1u]);
                    position[axis] = vec2<u32>(source.words[6u + axis * 2u], source.words[7u + axis * 2u]);
                }
                let sample = sample_field(position);
                let rank = candidate_rank(grid);
                // Explicit full density accepts rank == u32::MAX too.
                if sample.x == 65535u || (sample.x != 0u && rank < sample.x * 65537u) {
                    accepted = 1u;
                }
                // Keep sampled density/species/rank even on field rejection.
                attributes[index] = vec4<u32>(sample.x, sample.y, accepted, rank);
            }
        }
    }

    // No invocation returns early. All lanes, including rectangular padding
    // and source/field rejects, reach every uniform workgroup barrier.
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
        // Finish every old-scan read before any lane writes its new value.
        workgroupBarrier();
        scan[lane] = scan[lane] + preceding;
        workgroupBarrier();
    }
    if lane == 0u {
        let count = scan[63];
        if count != 0u {
            // Only instance_count changes; host preserves indirect[0,2,3].
            output_base = atomicAdd(&indirect[1], count);
        }
    }
    workgroupBarrier();
    if accepted != 0u {
        // Unique source slots; source order within a group, unspecified group
        // reservation order. The host provides primary_count output capacity.
        compact[output_base + scan[lane] - 1u] = index;
    }
}
