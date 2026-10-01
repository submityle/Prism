//! Real-device parity for the `Morton` bit-interleave twin:
//! [`GpuMortonCode`](prism_volumetric_gpu::morton_code::GpuMortonCode) must
//! reproduce the `CPU` golden
//! [`morton_code`](prism_render_architecture::particle::morton_code) element for
//! element across the 2D encode/decode and 3D encode/decode transforms.
//!
//! The fixtures cover the full set of boundary bit patterns the interleave can
//! hit: zero, a single low bit, the maximum coordinate (`0xFFFF` for 2D,
//! `0x3FF` for 3D), the alternating `0x5555`/`0xAAAA` masks, hand-picked
//! interleaved patterns and a dense Cartesian sweep so every axis exercises its
//! own bit lane. Encode-then-decode is checked for the exact round-trip.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Every transform is pure `u32` bit algebra with no rounding anywhere, so
//! `CPU` and `GPU` must agree bit for bit. The comparison is an exact `==` on
//! every output, with no tolerance: any mismatch is a genuine port bug. `WGSL`
//! has no `u64`, and the reference is already `u16`/`u32`-only, so nothing is
//! out of scope here.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::morton_code`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::morton_code::{
    morton_decode_2d, morton_decode_3d, morton_encode_2d, morton_encode_3d,
};
use prism_volumetric_gpu::morton_code::GpuMortonCode;
use prism_volumetric_gpu::GpuContext;

/// A broad fixture of 16-bit coordinates: zero, single bits, the alternating
/// even/odd masks, the full-range boundary and a few well-mixed patterns.
fn coords_2d() -> Vec<u16> {
    vec![
        0u16, 1, 2, 3, 4, 8, 0x0010, 0x0100, 0x1000, 0x8000, 0x5555, 0xAAAA, 0x0F0F, 0xF0F0,
        0x1234, 0x7A2C, 0x0F31, 0xFFFE, 0xFFFF,
    ]
}

/// A broad fixture of 10-bit coordinates (`0..=0x3FF`): zero, single bits, the
/// interleave masks and the full-range boundary.
fn coords_3d() -> Vec<u32> {
    vec![
        0u32, 1, 2, 3, 4, 7, 8, 0x010, 0x040, 0x100, 0x155, 0x2AA, 0x123, 0x2AB, 0x1FF, 0x200,
        0x3FE, 0x3FF,
    ]
}

#[test]
fn encode_2d_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMortonCode::new(&ctx);
    let base = coords_2d();
    let mut xs: Vec<u16> = Vec::new();
    let mut ys: Vec<u16> = Vec::new();
    for &x in &base {
        for &y in &base {
            xs.push(x);
            ys.push(y);
        }
    }
    let got = gpu.encode_2d(&ctx, &xs, &ys);
    assert_eq!(got.len(), xs.len());
    for (idx, (&x, &y)) in xs.iter().zip(ys.iter()).enumerate() {
        assert_eq!(
            got[idx],
            morton_encode_2d(x, y),
            "encode_2d mismatch at {idx} for (x={x:#06x}, y={y:#06x})"
        );
    }
}

#[test]
fn decode_2d_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMortonCode::new(&ctx);
    // Sweep codes derived from the coordinate fixture plus boundary codes.
    let base = coords_2d();
    let mut codes: Vec<u32> = Vec::new();
    for &x in &base {
        for &y in &base {
            codes.push(morton_encode_2d(x, y));
        }
    }
    codes.extend([0u32, 1, 2, 3, 0x5555_5555, 0xAAAA_AAAA, u32::MAX]);
    let got = gpu.decode_2d(&ctx, &codes);
    assert_eq!(got.len(), codes.len());
    for (idx, &code) in codes.iter().enumerate() {
        assert_eq!(
            got[idx],
            morton_decode_2d(code),
            "decode_2d mismatch at {idx} for code {code:#010x}"
        );
    }
}

#[test]
fn roundtrip_2d_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMortonCode::new(&ctx);
    let base = coords_2d();
    let mut xs: Vec<u16> = Vec::new();
    let mut ys: Vec<u16> = Vec::new();
    for &x in &base {
        for &y in &base {
            xs.push(x);
            ys.push(y);
        }
    }
    let codes = gpu.encode_2d(&ctx, &xs, &ys);
    let back = gpu.decode_2d(&ctx, &codes);
    assert_eq!(back.len(), xs.len());
    for (idx, (&x, &y)) in xs.iter().zip(ys.iter()).enumerate() {
        assert_eq!(
            back[idx],
            (x, y),
            "roundtrip_2d mismatch at {idx} for (x={x:#06x}, y={y:#06x})"
        );
    }
}

#[test]
fn encode_3d_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMortonCode::new(&ctx);
    let base = coords_3d();
    let mut xs: Vec<u32> = Vec::new();
    let mut ys: Vec<u32> = Vec::new();
    let mut zs: Vec<u32> = Vec::new();
    for &x in &base {
        for &y in &base {
            for &z in &base {
                xs.push(x);
                ys.push(y);
                zs.push(z);
            }
        }
    }
    let got = gpu.encode_3d(&ctx, &xs, &ys, &zs);
    assert_eq!(got.len(), xs.len());
    for (idx, ((&x, &y), &z)) in xs.iter().zip(ys.iter()).zip(zs.iter()).enumerate() {
        assert_eq!(
            got[idx],
            morton_encode_3d(x, y, z),
            "encode_3d mismatch at {idx} for (x={x:#05x}, y={y:#05x}, z={z:#05x})"
        );
    }
}

#[test]
fn decode_3d_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMortonCode::new(&ctx);
    let base = coords_3d();
    let mut codes: Vec<u32> = Vec::new();
    for &x in &base {
        for &y in &base {
            for &z in &base {
                codes.push(morton_encode_3d(x, y, z));
            }
        }
    }
    // Boundary codes, including the full low-30-bit fill and an all-ones word.
    codes.extend([0u32, 1, 2, 4, 7, 0x0924_9249, 0x3FFF_FFFF, u32::MAX]);
    let got = gpu.decode_3d(&ctx, &codes);
    assert_eq!(got.len(), codes.len());
    for (idx, &code) in codes.iter().enumerate() {
        assert_eq!(
            got[idx],
            morton_decode_3d(code),
            "decode_3d mismatch at {idx} for code {code:#010x}"
        );
    }
}

#[test]
fn roundtrip_3d_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMortonCode::new(&ctx);
    let base = coords_3d();
    let mut xs: Vec<u32> = Vec::new();
    let mut ys: Vec<u32> = Vec::new();
    let mut zs: Vec<u32> = Vec::new();
    for &x in &base {
        for &y in &base {
            for &z in &base {
                xs.push(x);
                ys.push(y);
                zs.push(z);
            }
        }
    }
    let codes = gpu.encode_3d(&ctx, &xs, &ys, &zs);
    let back = gpu.decode_3d(&ctx, &codes);
    assert_eq!(back.len(), xs.len());
    for (idx, ((&x, &y), &z)) in xs.iter().zip(ys.iter()).zip(zs.iter()).enumerate() {
        assert_eq!(
            back[idx],
            (x, y, z),
            "roundtrip_3d mismatch at {idx} for (x={x:#05x}, y={y:#05x}, z={z:#05x})"
        );
    }
}

#[test]
fn encode_3d_ignores_bits_above_ten() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMortonCode::new(&ctx);
    // High bits above bit 9 must drop on both sides, matching `part1by2`.
    let xs = vec![0xFFFF_FC00u32, 0x0000_0400 | 1, 0x7FFF_FFFF];
    let ys = vec![0u32, 0x0000_0401, 0x0000_03FF];
    let zs = vec![0u32, 0x0000_0400, 0xFFFF_FFFF];
    let got = gpu.encode_3d(&ctx, &xs, &ys, &zs);
    assert_eq!(got.len(), xs.len());
    for (idx, ((&x, &y), &z)) in xs.iter().zip(ys.iter()).zip(zs.iter()).enumerate() {
        assert_eq!(
            got[idx],
            morton_encode_3d(x, y, z),
            "encode_3d high-bit mismatch at {idx}"
        );
    }
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMortonCode::new(&ctx);
    // No dispatch is issued and every entry point returns an empty vector.
    assert!(gpu.encode_2d(&ctx, &[], &[]).is_empty());
    assert!(gpu.decode_2d(&ctx, &[]).is_empty());
    assert!(gpu.encode_3d(&ctx, &[], &[], &[]).is_empty());
    assert!(gpu.decode_3d(&ctx, &[]).is_empty());
}
