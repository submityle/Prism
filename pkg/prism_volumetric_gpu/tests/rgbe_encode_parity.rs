//! Real-device parity for the shared-exponent `HDR` packer twin:
//! [`GpuRgbeEncode`](prism_volumetric_gpu::rgbe_encode::GpuRgbeEncode) must
//! reproduce the `CPU` golden
//! [`rgbe_encode`](prism_render_architecture::particle::rgbe_encode) element for
//! element across Radiance `RGBE` encode/decode, Khronos `RGB9E5` encode/decode,
//! and the shared `IEEE754` primitives (`max_channel`, `channel_byte`,
//! `floor_log2`, `pow2_i32`).
//!
//! The color fixtures cover the black sentinel `[0, 0, 0]`, the exact white
//! `[1, 1, 1]`, exact powers of two, sub-`RGBE_MIN` inputs that collapse to
//! black, negative and `NaN` channels guarded to `0.0`, a dim channel beside a
//! bright one that quantizes away under the shared exponent, single-channel
//! colors, the `RGB9E5` saturation point `65408.0` and a value far above it,
//! and a broad spread of pseudo-random positive radiances built from a pure
//! integer `LCG` (no `f32` transcendental method is used to synthesize
//! fixtures). The decode fixtures feed random `RGBE` quads and `RGB9E5` words
//! directly, and the primitive fixture sweeps the full `pow2_i32` exponent
//! range `-160..=160` so every normal, subnormal, over-range and underflow
//! branch is exercised.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! `floor_log2`, `pow2_i32`, the byte quantization and the final bit assembly
//! are pure integer / `bitcast` work, and `WGSL` unsigned integers wrap exactly
//! like Rust's `wrapping_*`, so the packed `RGBE` bytes, the `RGB9E5` word, the
//! `channel_byte` and the `floor_log2` are bit-identical and asserted with exact
//! `==`; `pow2` is compared on its raw bits so an infinity matches an infinity.
//! The decoded continuous channels can diverge only by a legal rounding of a few
//! units in the last place, so they are compared with `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (relative floor `1e-6`), with a bit-exact fast path first.
//! Several scenarios additionally assert a non-trivial code so a degenerate
//! all-zero kernel could not pass.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::rgbe_encode`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::rgbe_encode as gold;
use prism_volumetric_gpu::rgbe_encode::{GpuRgbeEncode, RgbePrimQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for reconstructed-value parity.
const ABS_TOL: f32 = 1.0e-4;

/// Relative tolerance for reconstructed-value parity.
const REL_TOL: f32 = 1.0e-3;

/// Relative-tolerance floor so near-zero magnitudes keep a usable scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Radiance `RGBE` exponent bias; mirrors the golden `RGBE_EXP_BIAS`.
const RGBE_EXP_BIAS: i32 = 128;

/// Radiance `RGBE` per-channel mantissa width; mirrors the golden
/// `RGBE_MANTISSA_BITS`.
const RGBE_MANTISSA_BITS: i32 = 8;

/// Reconstructed-value closeness: a bit-exact fast path (so an infinity matches
/// an infinity) then an absolute-or-relative tolerance, because the reference
/// contract forbids `f32` equality.
fn close(a: f32, b: f32) -> bool {
    if a.to_bits() == b.to_bits() {
        return true;
    }
    let diff = (a - b).abs();
    if diff <= ABS_TOL {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff <= REL_TOL * scale
}

/// Rebuilds `2^n` as an exact `f32` purely by integer bit assembly, replicating
/// the golden private `pow2_i32` so parity can assert the on-device integer path
/// bit for bit. Uses only `f32::from_bits` and integer work, so no `f32`
/// transcendental appears in the fixture.
fn ref_pow2_i32(n: i32) -> f32 {
    if (-126..=127).contains(&n) {
        let biased = (n + 127) as u32;
        f32::from_bits(biased << 23)
    } else if n > 127 {
        f32::INFINITY
    } else if (-149..=-127).contains(&n) {
        let shift = (n + 149) as u32;
        f32::from_bits(1_u32 << shift)
    } else {
        0.0
    }
}

/// Extracts the unbiased base-2 exponent straight from the `IEEE754` field,
/// replicating the golden private `floor_log2`. Pure integer work.
fn ref_floor_log2(x: f32) -> i32 {
    let biased = (x.to_bits() >> 23) & 0x0000_00ff;
    if biased == 0 {
        -128
    } else {
        biased as i32 - 127
    }
}

/// Floors a non-negative product into a color byte, clamping to `0..=255`,
/// replicating the golden private `channel_byte`. Uses only `f32::clamp`, so no
/// transcendental appears.
fn ref_channel_byte(v: f32) -> u8 {
    let clamped = v.clamp(0.0, 255.0);
    clamped as u8
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a unit value in `[0, 1)`.
fn lcg_unit(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A raw 32-bit draw from the same generator, used to synthesize random `RGBE`
/// quads and `RGB9E5` words for the decode paths.
fn lcg_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// A broad fixture of linear `HDR` colors for the encode paths: the black
/// sentinel, exact white, exact powers of two, a sub-`RGBE_MIN` input, negative
/// and `NaN` channels (sanitized on encode), a dim-beside-bright pair, single
/// channels, the `RGB9E5` saturation point and a value far above it, and a
/// spread of pseudo-random positive radiances from a pure-integer `LCG`.
fn color_fixture() -> Vec<[f32; 3]> {
    let mut v = vec![
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.25, 0.125],
        [1.0e-34, 1.0e-35, 0.0],
        [-4.0, 2.0, -1.0],
        [-1.0, 2.0, f32::NAN],
        [f32::NAN, f32::NAN, f32::NAN],
        [12.0, 10.0, 8.0],
        [100.0, 90.0, 110.0],
        [3.5, 3.4, 3.6],
        [0.03, 0.028, 0.031],
        [8.0, 0.01, 0.0],
        [2000.0, 0.01, 0.0],
        [500.0, 0.0, 0.0],
        [0.0, 500.0, 0.0],
        [0.0, 0.0, 500.0],
        [65_408.0, 65_408.0, 65_408.0],
        [1.0e30, 1.0e30, 1.0e30],
        [1000.0, 500.0, 250.0],
        [0.9, 0.8, 0.7],
    ];
    let mut state: u64 = 0x1234_5678_9abc_def1;
    let mut count: u32 = 0;
    while count < 100 {
        let r = lcg_unit(&mut state) * 300.0;
        let g = lcg_unit(&mut state) * 50.0;
        let b = lcg_unit(&mut state) * 5.0;
        v.push([r, g, b]);
        count += 1;
    }
    v
}

/// Random `RGBE` quads plus a few known codes (the black sentinel, exact white
/// and the half/quarter/eighth code) for the decode path.
fn quad_fixture() -> Vec<[u8; 4]> {
    let mut v = vec![[0, 0, 0, 0], [128, 128, 128, 129], [128, 64, 32, 128]];
    let mut state: u64 = 0x0bad_c0de_1337_f00d;
    let mut count: u32 = 0;
    while count < 128 {
        v.push(lcg_u32(&mut state).to_le_bytes());
        count += 1;
    }
    v
}

/// Random `RGB9E5` words plus a few known codes for the decode path.
fn word_fixture() -> Vec<u32> {
    let mut v = vec![0u32, gold::rgb9e5_encode([1.0, 1.0, 1.0])];
    let mut state: u64 = 0xfeed_face_dead_beef;
    let mut count: u32 = 0;
    while count < 128 {
        v.push(lcg_u32(&mut state));
        count += 1;
    }
    v
}

/// Probes for the shared `IEEE754` primitives. The exponent fed to `pow2_i32`
/// sweeps `-160..=160` so every branch (normal, subnormal, over-range to
/// infinity, underflow to zero) is covered, while the other three fields cycle
/// through representative values. `max_rgb` holds only finite values because the
/// kernel's `max_channel` runs before any sanitization, matching the golden.
fn prim_fixture() -> Vec<RgbePrimQuery> {
    let rgbs = [
        [1.0f32, 2.0, 3.0],
        [5.0, 1.0, 2.0],
        [0.0, 0.0, 0.0],
        [100.0, 50.0, 25.0],
        [0.25, 0.5, 0.125],
        [-1.0, 2.0, -3.0],
    ];
    let channel_vals = [-5.0f32, 0.0, 0.5, 1.0, 42.3, 127.5, 200.0, 255.0, 300.0];
    let log2_inputs = [1.0f32, 2.0, 3.9, 0.5, 0.0, 1024.0, 0.001, 7.5];
    let mut v = Vec::new();
    for (i, n) in (-160i32..=160).enumerate() {
        v.push(RgbePrimQuery {
            max_rgb: rgbs[i % rgbs.len()],
            channel_value: channel_vals[i % channel_vals.len()],
            log2_input: log2_inputs[i % log2_inputs.len()],
            pow2_exponent: n,
        });
    }
    v
}

#[test]
fn rgbe_encode_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbeEncode::new(&ctx);
    let colors = color_fixture();
    let got = gpu.rgbe_encode(&ctx, &colors);
    assert_eq!(got.len(), colors.len());
    for (idx, &c) in colors.iter().enumerate() {
        assert_eq!(
            got[idx],
            gold::rgbe_encode(c),
            "rgbe_encode mismatch at {idx} for color {c:?}"
        );
    }
}

#[test]
fn rgbe_decode_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbeEncode::new(&ctx);
    let quads = quad_fixture();
    let got = gpu.rgbe_decode(&ctx, &quads);
    assert_eq!(got.len(), quads.len());
    for (idx, &q) in quads.iter().enumerate() {
        let expected = gold::rgbe_decode(q);
        for (channel, (&g, &e)) in got[idx].iter().zip(expected.iter()).enumerate() {
            assert!(
                close(g, e),
                "rgbe_decode mismatch at {idx} channel {channel} for quad {q:?}: gpu {g} cpu {e}"
            );
        }
    }
}

#[test]
fn rgb9e5_encode_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbeEncode::new(&ctx);
    let colors = color_fixture();
    let got = gpu.rgb9e5_encode(&ctx, &colors);
    assert_eq!(got.len(), colors.len());
    for (idx, &c) in colors.iter().enumerate() {
        assert_eq!(
            got[idx],
            gold::rgb9e5_encode(c),
            "rgb9e5_encode mismatch at {idx} for color {c:?}"
        );
    }
}

#[test]
fn rgb9e5_decode_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbeEncode::new(&ctx);
    let words = word_fixture();
    let got = gpu.rgb9e5_decode(&ctx, &words);
    assert_eq!(got.len(), words.len());
    for (idx, &w) in words.iter().enumerate() {
        let expected = gold::rgb9e5_decode(w);
        for (channel, (&g, &e)) in got[idx].iter().zip(expected.iter()).enumerate() {
            assert!(
                close(g, e),
                "rgb9e5_decode mismatch at {idx} channel {channel} for word {w:#010x}: gpu {g} cpu {e}"
            );
        }
    }
}

#[test]
fn primitives_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbeEncode::new(&ctx);
    let queries = prim_fixture();
    let got = gpu.primitives(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, q) in queries.iter().enumerate() {
        let result = got[idx];
        // channel_byte and floor_log2 are integers: exact equality.
        assert_eq!(
            result.channel_byte,
            ref_channel_byte(q.channel_value),
            "channel_byte mismatch at {idx} for value {}",
            q.channel_value
        );
        assert_eq!(
            result.floor_log2,
            ref_floor_log2(q.log2_input),
            "floor_log2 mismatch at {idx} for input {}",
            q.log2_input
        );
        // pow2 is reproduced to the bit, so an infinity matches an infinity.
        assert_eq!(
            result.pow2.to_bits(),
            ref_pow2_i32(q.pow2_exponent).to_bits(),
            "pow2 mismatch at {idx} for exponent {}",
            q.pow2_exponent
        );
        // max_channel is a plain selection, compared with a continuous bound.
        let expected_max = gold::max_channel(q.max_rgb);
        assert!(
            close(result.max_channel, expected_max),
            "max_channel mismatch at {idx} for {:?}: gpu {} cpu {expected_max}",
            q.max_rgb,
            result.max_channel
        );
    }
}

#[test]
fn rgbe_known_vectors_are_non_trivial() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbeEncode::new(&ctx);
    let colors = [[1.0f32, 1.0, 1.0], [0.5, 0.25, 0.125], [0.0, 0.0, 0.0]];
    let got = gpu.rgbe_encode(&ctx, &colors);
    assert_eq!(got[0], [128, 128, 128, 129], "white RGBE code");
    assert_eq!(got[1], [128, 64, 32, 128], "half/quarter/eighth RGBE code");
    assert_eq!(got[2], [0, 0, 0, 0], "black sentinel");
    // A degenerate all-zero kernel could not produce the bright white code.
    assert_ne!(got[0], [0, 0, 0, 0], "white must not collapse to black");
}

#[test]
fn rgb9e5_known_layout_is_non_trivial() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbeEncode::new(&ctx);
    // Saturation: MAXVAL and a value far above it must pack to the same word,
    // and that word must be non-zero (a degenerate kernel could not produce it).
    let colors = [[65_408.0f32, 65_408.0, 65_408.0], [1.0e30, 1.0e30, 1.0e30]];
    let got = gpu.rgb9e5_encode(&ctx, &colors);
    assert_eq!(got[0], got[1], "RGB9E5 saturates to one code");
    assert_eq!(
        got[0],
        gold::rgb9e5_encode(colors[0]),
        "RGB9E5 saturation matches reference"
    );
    assert_ne!(got[0], 0, "saturated RGB9E5 word must be non-zero");
}

#[test]
fn rgbe_decode_round_trips_the_scale() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbeEncode::new(&ctx);
    // Decoding the half/quarter/eighth code reproduces exact powers of two, so
    // the shared scale (pow2) is wired correctly end to end.
    let got = gpu.rgbe_decode(&ctx, &[[128, 64, 32, 128]]);
    assert!(close(got[0][0], 0.5), "red channel");
    assert!(close(got[0][1], 0.25), "green channel");
    assert!(close(got[0][2], 0.125), "blue channel");
    // Independently confirm the scale factor used by the decode path.
    let scale = ref_pow2_i32(128 - RGBE_EXP_BIAS - RGBE_MANTISSA_BITS);
    assert!(
        close(scale, 1.0 / 256.0),
        "exponent byte 128 scales the mantissa by 2^-8"
    );
}

#[test]
fn large_random_batch_spans_workgroups() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbeEncode::new(&ctx);
    // Far more than one workgroup of 64, so the dispatch covers several groups.
    let mut colors = Vec::new();
    let mut state: u64 = 0xa5a5_5a5a_c3c3_3c3c;
    let mut count: u32 = 0;
    while count < 500 {
        let r = lcg_unit(&mut state) * 1000.0;
        let g = lcg_unit(&mut state) * 1000.0;
        let b = lcg_unit(&mut state) * 1000.0;
        colors.push([r, g, b]);
        count += 1;
    }
    let rgbe = gpu.rgbe_encode(&ctx, &colors);
    let rgb9e5 = gpu.rgb9e5_encode(&ctx, &colors);
    assert_eq!(rgbe.len(), colors.len());
    assert_eq!(rgb9e5.len(), colors.len());
    for (idx, &c) in colors.iter().enumerate() {
        assert_eq!(rgbe[idx], gold::rgbe_encode(c), "rgbe batch at {idx}");
        assert_eq!(rgb9e5[idx], gold::rgb9e5_encode(c), "rgb9e5 batch at {idx}");
    }
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRgbeEncode::new(&ctx);
    assert!(gpu.rgbe_encode(&ctx, &[]).is_empty());
    assert!(gpu.rgbe_decode(&ctx, &[]).is_empty());
    assert!(gpu.rgb9e5_encode(&ctx, &[]).is_empty());
    assert!(gpu.rgb9e5_decode(&ctx, &[]).is_empty());
    assert!(gpu.primitives(&ctx, &[]).is_empty());
}
