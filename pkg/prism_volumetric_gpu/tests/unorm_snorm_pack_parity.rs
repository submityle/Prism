//! Real-device parity for the normalized fixed-point quantizer twin:
//! [`GpuUnormSnormPack`](prism_volumetric_gpu::unorm_snorm_pack::GpuUnormSnormPack)
//! must reproduce the `CPU` golden
//! [`unorm_snorm_pack`](prism_render_architecture::particle::unorm_snorm_pack)
//! element for element across the `UNORM`/`SNORM` scalar pack and unpack and the
//! `RGBA` vector helpers.
//!
//! The fixtures cover the endpoints `0.0`, `1.0` and `-1.0`, the midpoint `0.5`,
//! out-of-range values that must be clamped, round-half-up tie points (where
//! `x * MAX + 0.5` lands exactly on an integer), negative `SNORM` values, and a
//! broad spread of pseudo-random values built with pure integer/arithmetic
//! `LCG` steps (the crate has no `bevy_math`, so no `f32` transcendental method
//! is used to synthesize fixtures).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Pack produces an integer code, so the comparison is an exact `==` on every
//! code (integers compare exactly); any mismatch is a genuine port bug. Unpack
//! produces an `f32`, so parity is asserted with the tolerance
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (relative floor `1e-6`) because the
//! reference contract forbids `f32` equality. The `UNORM`/`SNORM` round-half
//! boundary is validated through the pack codes, which are integers and thus
//! compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::unorm_snorm_pack`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::unorm_snorm_pack as gold;
use prism_volumetric_gpu::unorm_snorm_pack::GpuUnormSnormPack;
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for reconstructed-value parity.
const ABS_TOL: f32 = 1e-4;
/// Relative tolerance for reconstructed-value parity.
const REL_TOL: f32 = 1e-3;
/// Relative-tolerance floor so near-zero magnitudes keep a usable scale.
const REL_FLOOR: f32 = 1e-6;

/// Reconstructed-value closeness: absolute or relative tolerance (`f32`
/// equality is forbidden by the reference contract).
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_TOL {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff <= REL_TOL * scale
}

/// A broad fixture of real values: endpoints, the midpoint, out-of-range
/// clamp cases, round-half tie points, negative `SNORM` values, and a spread of
/// pseudo-random values from a pure-integer `LCG` (no transcendental method).
fn value_fixture() -> Vec<f32> {
    let mut v = vec![
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        0.25,
        -0.25,
        0.75,
        -0.75,
        // Out-of-range values the quantizers must clamp.
        2.0,
        -2.0,
        1.5,
        -1.5,
        10.0,
        -10.0,
        // UNORM8 round-half-up ties: x * 255 has fractional part exactly 0.5.
        0.5 / 255.0,
        1.5 / 255.0,
        2.5 / 255.0,
        127.5 / 255.0,
        254.5 / 255.0,
        // SNORM8 ties (magnitude), both signs.
        0.5 / 127.0,
        1.5 / 127.0,
        126.5 / 127.0,
        -0.5 / 127.0,
        -1.5 / 127.0,
        -126.5 / 127.0,
        // Near endpoints.
        0.999,
        -0.999,
        0.001,
        -0.001,
    ];
    let mut state: u32 = 0x1234_5678;
    let mut count: u32 = 0;
    while count < 96 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let u = ((state >> 8) as f32) / ((1u32 << 24) as f32);
        v.push(u);
        v.push(u * 2.0 - 1.0);
        count += 1;
    }
    v
}

#[test]
fn pack_unorm8_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    let xs = value_fixture();
    let got = gpu.pack_unorm8(&ctx, &xs);
    assert_eq!(got.len(), xs.len());
    for (idx, &x) in xs.iter().enumerate() {
        assert_eq!(
            got[idx],
            u32::from(gold::pack_unorm8(x)),
            "pack_unorm8 mismatch at {idx} for x {x}"
        );
    }
}

#[test]
fn pack_unorm16_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    let xs = value_fixture();
    let got = gpu.pack_unorm16(&ctx, &xs);
    assert_eq!(got.len(), xs.len());
    for (idx, &x) in xs.iter().enumerate() {
        assert_eq!(
            got[idx],
            u32::from(gold::pack_unorm16(x)),
            "pack_unorm16 mismatch at {idx} for x {x}"
        );
    }
}

#[test]
fn pack_snorm8_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    let xs = value_fixture();
    let got = gpu.pack_snorm8(&ctx, &xs);
    assert_eq!(got.len(), xs.len());
    for (idx, &x) in xs.iter().enumerate() {
        assert_eq!(
            got[idx],
            i32::from(gold::pack_snorm8(x)),
            "pack_snorm8 mismatch at {idx} for x {x}"
        );
    }
}

#[test]
fn pack_snorm16_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    let xs = value_fixture();
    let got = gpu.pack_snorm16(&ctx, &xs);
    assert_eq!(got.len(), xs.len());
    for (idx, &x) in xs.iter().enumerate() {
        assert_eq!(
            got[idx],
            i32::from(gold::pack_snorm16(x)),
            "pack_snorm16 mismatch at {idx} for x {x}"
        );
    }
}

#[test]
fn unpack_unorm8_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    let codes: Vec<u32> = (0u32..=255).collect();
    let got = gpu.unpack_unorm8(&ctx, &codes);
    assert_eq!(got.len(), codes.len());
    for (idx, &code) in codes.iter().enumerate() {
        let expected = gold::unpack_unorm8(code as u8);
        assert!(
            close(got[idx], expected),
            "unpack_unorm8 mismatch at code {code}: gpu {} cpu {expected}",
            got[idx]
        );
    }
}

#[test]
fn unpack_unorm16_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    let codes: Vec<u32> = [0u32, 1, 128, 255, 256, 32768, 40000, 65534, 65535].to_vec();
    let got = gpu.unpack_unorm16(&ctx, &codes);
    assert_eq!(got.len(), codes.len());
    for (idx, &code) in codes.iter().enumerate() {
        let expected = gold::unpack_unorm16(code as u16);
        assert!(
            close(got[idx], expected),
            "unpack_unorm16 mismatch at code {code}: gpu {} cpu {expected}",
            got[idx]
        );
    }
}

#[test]
fn unpack_snorm8_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    // Every valid code plus the reserved -128.
    let codes: Vec<i32> = (-128i32..=127).collect();
    let got = gpu.unpack_snorm8(&ctx, &codes);
    assert_eq!(got.len(), codes.len());
    for (idx, &code) in codes.iter().enumerate() {
        let expected = gold::unpack_snorm8(code as i8);
        assert!(
            close(got[idx], expected),
            "unpack_snorm8 mismatch at code {code}: gpu {} cpu {expected}",
            got[idx]
        );
    }
}

#[test]
fn unpack_snorm16_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    let codes: Vec<i32> = [-32768i32, -32767, -20000, -1, 0, 1, 20000, 32767].to_vec();
    let got = gpu.unpack_snorm16(&ctx, &codes);
    assert_eq!(got.len(), codes.len());
    for (idx, &code) in codes.iter().enumerate() {
        let expected = gold::unpack_snorm16(code as i16);
        assert!(
            close(got[idx], expected),
            "unpack_snorm16 mismatch at code {code}: gpu {} cpu {expected}",
            got[idx]
        );
    }
}

#[test]
fn pack_unorm8x4_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    let texels = [
        [0.0f32, 1.0, 0.5, 0.25],
        [2.0, -1.0, 0.5 / 255.0, 254.5 / 255.0],
        [
            0x11 as f32 / 255.0,
            0x22 as f32 / 255.0,
            0x33 as f32 / 255.0,
            0x44 as f32 / 255.0,
        ],
        [0.999, 0.001, 0.75, 0.125],
    ];
    let got = gpu.pack_unorm8x4(&ctx, &texels);
    assert_eq!(got.len(), texels.len());
    for (idx, texel) in texels.iter().enumerate() {
        assert_eq!(
            got[idx],
            gold::pack_unorm8x4(*texel),
            "pack_unorm8x4 mismatch at {idx}"
        );
    }
    // LSB-first byte order: red in the LSB, alpha in the MSB.
    assert_eq!(got[2], 0x4433_2211, "pack_unorm8x4 byte order");
}

#[test]
fn unpack_unorm8x4_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    let packed = [0x4433_2211u32, 0x0000_0000, 0xFFFF_FFFF, 0x1234_ABCD];
    let got = gpu.unpack_unorm8x4(&ctx, &packed);
    assert_eq!(got.len(), packed.len());
    for (idx, &p) in packed.iter().enumerate() {
        let expected = gold::unpack_unorm8x4(p);
        for (channel, (&g, &e)) in got[idx].iter().zip(expected.iter()).enumerate() {
            assert!(
                close(g, e),
                "unpack_unorm8x4 mismatch at {idx} channel {channel}: gpu {g} cpu {e}"
            );
        }
    }
}

#[test]
fn pack_unorm16x2_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    let pairs = [
        [0.0f32, 1.0],
        [0.5, 0.25],
        [2.0, -1.0],
        [0x1234 as f32 / 65535.0, 0xABCD as f32 / 65535.0],
    ];
    let got = gpu.pack_unorm16x2(&ctx, &pairs);
    assert_eq!(got.len(), pairs.len());
    for (idx, pair) in pairs.iter().enumerate() {
        assert_eq!(
            got[idx],
            gold::pack_unorm16x2(*pair),
            "pack_unorm16x2 mismatch at {idx}"
        );
    }
    // Channel 0 occupies the low 16 bits, channel 1 the high 16 bits.
    assert_eq!(got[3] & 0xFFFF, 0x1234, "pack_unorm16x2 low channel");
    assert_eq!(got[3] >> 16, 0xABCD, "pack_unorm16x2 high channel");
}

#[test]
fn unpack_unorm16x2_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    let packed = [0xABCD_1234u32, 0x0000_0000, 0xFFFF_FFFF, 0x9C40_4E20];
    let got = gpu.unpack_unorm16x2(&ctx, &packed);
    assert_eq!(got.len(), packed.len());
    for (idx, &p) in packed.iter().enumerate() {
        let expected = gold::unpack_unorm16x2(p);
        for (channel, (&g, &e)) in got[idx].iter().zip(expected.iter()).enumerate() {
            assert!(
                close(g, e),
                "unpack_unorm16x2 mismatch at {idx} channel {channel}: gpu {g} cpu {e}"
            );
        }
    }
}

#[test]
fn round_trip_codes_are_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    // The round trip is fundamentally an integer code: unpack then re-pack on
    // the GPU must return the original code exactly for every valid code.
    let u8_codes: Vec<u32> = (0u32..=255).collect();
    let re_u8 = gpu.pack_unorm8(&ctx, &gpu.unpack_unorm8(&ctx, &u8_codes));
    assert_eq!(re_u8, u8_codes, "unorm8 code round trip");

    let s8_codes: Vec<i32> = (-127i32..=127).collect();
    let re_s8 = gpu.pack_snorm8(&ctx, &gpu.unpack_snorm8(&ctx, &s8_codes));
    assert_eq!(re_s8, s8_codes, "snorm8 code round trip");

    let u16_codes: Vec<u32> = [0u32, 1, 128, 32768, 40000, 65534, 65535].to_vec();
    let re_u16 = gpu.pack_unorm16(&ctx, &gpu.unpack_unorm16(&ctx, &u16_codes));
    assert_eq!(re_u16, u16_codes, "unorm16 code round trip");

    let s16_codes: Vec<i32> = [-32767i32, -20000, -1, 0, 1, 20000, 32767].to_vec();
    let re_s16 = gpu.pack_snorm16(&ctx, &gpu.unpack_snorm16(&ctx, &s16_codes));
    assert_eq!(re_s16, s16_codes, "snorm16 code round trip");
}

#[test]
fn snorm_rounding_is_symmetric_about_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    // Equal magnitudes map to negated codes (round half away from zero).
    let x = 50.4 / 127.0;
    let got = gpu.pack_snorm8(&ctx, &[x, -x]);
    assert_eq!(got, vec![50, -50], "snorm8 symmetric rounding");
    // Endpoints must land on the symmetric range, never the reserved code.
    let ends = gpu.pack_snorm8(&ctx, &[1.0, -1.0, 2.0, -2.0]);
    assert_eq!(ends, vec![127, -127, 127, -127], "snorm8 clamped endpoints");
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuUnormSnormPack::new(&ctx);
    assert!(gpu.pack_unorm8(&ctx, &[]).is_empty());
    assert!(gpu.pack_unorm16(&ctx, &[]).is_empty());
    assert!(gpu.pack_snorm8(&ctx, &[]).is_empty());
    assert!(gpu.pack_snorm16(&ctx, &[]).is_empty());
    assert!(gpu.unpack_unorm8(&ctx, &[]).is_empty());
    assert!(gpu.unpack_unorm16(&ctx, &[]).is_empty());
    assert!(gpu.unpack_snorm8(&ctx, &[]).is_empty());
    assert!(gpu.unpack_snorm16(&ctx, &[]).is_empty());
    assert!(gpu.pack_unorm8x4(&ctx, &[]).is_empty());
    assert!(gpu.unpack_unorm8x4(&ctx, &[]).is_empty());
    assert!(gpu.pack_unorm16x2(&ctx, &[]).is_empty());
    assert!(gpu.unpack_unorm16x2(&ctx, &[]).is_empty());
}
