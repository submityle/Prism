// Motion-vector encode/pack: compact per-pixel velocity target writer.
//
// One invocation encodes one pixel's MotionSample into the compact texel the
// temporal resolve reads: a signed 16-bit velocity pair (snorm16, normalised by
// the encoding full-scale) plus a u32 of unorm8 reactive/transparency/
// confidence bytes with the top byte reserved for per-pixel flags.
//
// Every arithmetic step mirrors the CPU golden
// `prism_render_architecture::motion::encode` exactly:
//   * clamp_signed_unit / clamp01 branch on NaN first (clamp never sees NaN),
//   * snorm16 rounds via an explicit +/-0.5 half-step bias then truncates
//     toward zero (NOT round-half-even), so there is no rounding-mode mismatch,
//   * unorm8 scales by 255, adds 0.5, truncates,
//   * the bytes are packed with the identical shifts (0/8/16/24).
// The clamp bounds guarantee the biased value is always inside the integer
// range, so the truncating cast never saturates or hits undefined behaviour.
// The device output therefore equals the golden bit-for-bit (exact i32/u32
// equality, not a tolerance).
//
// Provenance: standard fixed-point snorm16/unorm8 quantisation and bit packing;
// no neural, learned, or data-driven components. No Unreal Engine source or
// derived code.

const SNORM16_SCALE: f32 = 32767.0;
// flags::TRANSPARENT = 1 << 1 in the golden `encode::flags` module.
const FLAG_TRANSPARENT: u32 = 2u;

struct Params {
    // Number of pixels to encode.
    count: u32,
    // Velocity full-scale in pixels (already clamped up to >= 1 on the host via
    // VelocityEncoding::new); maps to +/-1.0 normalised.
    max_velocity_pixels: f32,
    pad0: u32,
    pad1: u32,
};

// The compact per-pixel output record. vx/vy hold the snorm16 pair widened to
// i32; masks holds the packed unorm8 + flags u32.
struct Encoded {
    vx: i32,
    vy: i32,
    masks: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Per-pixel pixel-space velocity (prev - curr).
@group(0) @binding(1) var<storage, read> velocity: array<vec2<f32>>;
// Per-pixel (reactive, transparency, confidence, unused), each in [0, 1].
@group(0) @binding(2) var<storage, read> masks_in: array<vec4<f32>>;
// Per-pixel extra flag bits supplied by the caller (low byte meaningful).
@group(0) @binding(3) var<storage, read> flags_in: array<u32>;
// Compact output records.
@group(0) @binding(4) var<storage, read_write> out_enc: array<Encoded>;

// IEEE-754 NaN test by bit pattern. Metal compiles WGSL with fast-math, under
// which `x != x` is optimised to `false` and never detects a NaN, so NaN
// resolution must inspect the raw bits: a NaN has every exponent bit set and a
// nonzero mantissa, i.e. its magnitude bits exceed the +inf pattern 0x7F800000.
fn is_nan_bits(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7FFFFFFFu) > 0x7F800000u;
}

// Clamp to [-1, 1], resolving NaN to 0 (clamp never sees a NaN).
fn clamp_signed_unit(x: f32) -> f32 {
    if (is_nan_bits(x)) {
        return 0.0;
    }
    return clamp(x, -1.0, 1.0);
}

// Clamp to [0, 1], resolving NaN (and negatives) to 0 before any clamp.
fn clamp01(x: f32) -> f32 {
    if (is_nan_bits(x) || x < 0.0) {
        return 0.0;
    }
    if (x > 1.0) {
        return 1.0;
    }
    return x;
}

// snorm16 encode via half-step bias + truncate toward zero.
fn encode_snorm16(normalized: f32) -> i32 {
    let clamped = clamp_signed_unit(normalized);
    let scaled = clamped * SNORM16_SCALE;
    var biased: f32;
    if (scaled >= 0.0) {
        biased = scaled + 0.5;
    } else {
        biased = scaled - 0.5;
    }
    return i32(biased);
}

// unorm8 encode via scale-by-255 + 0.5 bias + truncate.
fn encode_unorm8(value: f32) -> u32 {
    let clamped = clamp01(value);
    let scaled = clamped * 255.0 + 0.5;
    return u32(scaled);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }

    let v = velocity[i];
    let vx = encode_snorm16(v.x / params.max_velocity_pixels);
    let vy = encode_snorm16(v.y / params.max_velocity_pixels);

    let m = masks_in[i];
    let reactive = m.x;
    let transparency = m.y;
    let confidence = m.z;

    var flag_bits = flags_in[i];
    if (transparency > 0.0) {
        flag_bits = flag_bits | FLAG_TRANSPARENT;
    }

    let r = encode_unorm8(reactive) << 0u;
    let t = encode_unorm8(transparency) << 8u;
    let c = encode_unorm8(confidence) << 16u;
    let f = (flag_bits & 0xFFu) << 24u;

    out_enc[i] = Encoded(vx, vy, r | t | c | f, 0u);
}
