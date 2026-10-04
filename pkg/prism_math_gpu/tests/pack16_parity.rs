//! Real-device parity for the §24.1 16-bit-per-channel vertex-attribute
//! (`unorm16x2` / `snorm16x2`) pack/unpack shader mirror.
//!
//! The pack kernels fold each `vec2<f32>` lane into one `u32` of two 16-bit
//! channels on a real `GPU` from the single-sourced
//! [`WGSL_PACK16`](prism_math::shader_mirror::WGSL_PACK16) fragment (the WGSL
//! built-ins `pack2x16unorm` / `pack2x16snorm`) and the unpack kernels widen
//! them back (`unpack2x16unorm` / `unpack2x16snorm`). Both are diffed against
//! the CPU reference [`prism_math::pack16`].
//!
//! WGSL defines the quantizers as `⌊0.5 + N·clamp(c)⌋` (`N` = 65535 unorm /
//! 32767 snorm), exactly like the CPU reference, so for **exactly-representable
//! quantized** inputs the packed half-words match [`prism_math::pack16`]
//! **bit-for-bit**; the exact sweeps assert that. Arbitrary inputs may differ
//! by at most one code at a rounding tie, a documented honest boundary checked
//! with a one-code tolerance. The widening (`unpack`) direction is exact on
//! both sides. The suite skips gracefully when no adapter is available.

use prism_math::pack16::{
    pack_snorm2x16, pack_unorm2x16, unpack_snorm2x16, unpack_unorm2x16,
};
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuPack16;

/// Acquires a device, or prints a skip note and returns `None` on hosts without
/// a usable adapter.
#[expect(
    clippy::print_stderr,
    reason = "test-only skip note when no GPU adapter is present"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// A small deterministic linear-congruential sequence.
fn lcg(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

/// Maximum absolute difference, in integer codes, between two packed `u32`s,
/// taken per half-word on the raw bits (`unorm` interpretation).
fn max_unorm_code_diff(a: u32, b: u32) -> i32 {
    let mut m = 0;
    for s in [0, 16] {
        let x = ((a >> s) & 0xFFFF) as i32;
        let y = ((b >> s) & 0xFFFF) as i32;
        m = m.max((x - y).abs());
    }
    m
}

/// Maximum absolute difference, in integer codes, between two packed `u32`s,
/// interpreting each half-word as a two's-complement `i16` (`snorm`).
fn max_snorm_code_diff(a: u32, b: u32) -> i32 {
    let mut m = 0;
    for s in [0, 16] {
        let x = i32::from(((a >> s) & 0xFFFF) as u16 as i16);
        let y = i32::from(((b >> s) & 0xFFFF) as u16 as i16);
        m = m.max((x - y).abs());
    }
    m
}

#[test]
fn pack_unorm_matches_cpu_on_exact_quantized_sweep() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPack16::new(&ctx);

    // Each half-word maps to an exactly-representable dequantized value h/65535;
    // packing it must land back on half-word h on both sides, bit-for-bit.
    let mut src: Vec<[f32; 2]> = Vec::new();
    let mut h: u32 = 0;
    while h <= 0xFFFF {
        src.push([h as f32 / 65535.0, (0xFFFF - h) as f32 / 65535.0]);
        h += 3;
    }

    let gpu = kernel.pack_unorm2x16(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (v, &g) in src.iter().zip(gpu.iter()) {
        assert_eq!(g, pack_unorm2x16(*v), "unorm pack drift at {v:?}");
    }
}

#[test]
fn pack_snorm_matches_cpu_on_exact_quantized_sweep() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPack16::new(&ctx);

    // Each encodable signed code in [-32767, 32767] maps to exact k/32767;
    // packing it must land back on code k bit-for-bit on both sides.
    let mut src: Vec<[f32; 2]> = Vec::new();
    let mut k: i32 = -32767;
    while k <= 32767 {
        src.push([k as f32 / 32767.0, -k as f32 / 32767.0]);
        k += 3;
    }

    let gpu = kernel.pack_snorm2x16(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (v, &g) in src.iter().zip(gpu.iter()) {
        assert_eq!(g, pack_snorm2x16(*v), "snorm pack drift at {v:?}");
    }
}

#[test]
fn unpack_unorm_matches_cpu_bit_exact() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPack16::new(&ctx);

    // Widening is exact division by 65535 and must agree bit-for-bit.
    let mut keys: Vec<u32> = Vec::new();
    let mut seed = 0x0ac8_1234u32;
    for _ in 0..8192 {
        keys.push(lcg(&mut seed));
    }
    let mut h: u32 = 0;
    while h <= 0xFFFF {
        keys.push(h | (h << 16));
        h += 101;
    }

    let gpu = kernel.unpack_unorm2x16(&ctx, &keys);
    assert_eq!(gpu.len(), keys.len());
    for (&k, got) in keys.iter().zip(gpu.iter()) {
        let cpu = unpack_unorm2x16(k);
        for lane in 0..2 {
            assert_eq!(
                got[lane].to_bits(),
                cpu[lane].to_bits(),
                "unorm unpack drift for {k:#010x} lane {lane}"
            );
        }
    }
}

#[test]
fn unpack_snorm_matches_cpu_bit_exact() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPack16::new(&ctx);

    // The CPU reference clamps -32768 up to -1.0 (max(i16/32767, -1)); build
    // keys that exercise 0x8000 so the clamp path is covered.
    let mut keys: Vec<u32> = Vec::new();
    let mut seed = 0x51ed_600du32;
    for _ in 0..8192 {
        keys.push(lcg(&mut seed));
    }
    keys.push(0x0000_8000); // low half-word = -32768
    keys.push(0x8000_0000); // high half-word = -32768

    let gpu = kernel.unpack_snorm2x16(&ctx, &keys);
    assert_eq!(gpu.len(), keys.len());
    for (&k, got) in keys.iter().zip(gpu.iter()) {
        let cpu = unpack_snorm2x16(k);
        for lane in 0..2 {
            assert_eq!(
                got[lane].to_bits(),
                cpu[lane].to_bits(),
                "snorm unpack drift for {k:#010x} lane {lane}"
            );
        }
    }
}

#[test]
fn pack_unorm_arbitrary_within_one_code() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPack16::new(&ctx);

    let mut src: Vec<[f32; 2]> = Vec::new();
    let mut seed = 0x1357_9bdfu32;
    for _ in 0..4096 {
        let c = |s: &mut u32| (lcg(s) % 1_000_001) as f32 / 1_000_000.0;
        src.push([c(&mut seed), c(&mut seed)]);
    }
    src.push([-0.5, 1.5]);

    let gpu = kernel.pack_unorm2x16(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (v, &g) in src.iter().zip(gpu.iter()) {
        let cpu = pack_unorm2x16(*v);
        assert!(
            max_unorm_code_diff(g, cpu) <= 1,
            "unorm pack off by >1 code at {v:?}: gpu {g:#010x} cpu {cpu:#010x}"
        );
    }
}

#[test]
fn pack_snorm_arbitrary_within_one_code() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPack16::new(&ctx);

    let mut src: Vec<[f32; 2]> = Vec::new();
    let mut seed = 0x2468_ace0u32;
    for _ in 0..4096 {
        let c = |s: &mut u32| (lcg(s) % 2_000_001) as f32 / 1_000_000.0 - 1.0;
        src.push([c(&mut seed), c(&mut seed)]);
    }
    src.push([-1.5, 1.5]);

    let gpu = kernel.pack_snorm2x16(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (v, &g) in src.iter().zip(gpu.iter()) {
        let cpu = pack_snorm2x16(*v);
        assert!(
            max_snorm_code_diff(g, cpu) <= 1,
            "snorm pack off by >1 code at {v:?}: gpu {g:#010x} cpu {cpu:#010x}"
        );
    }
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPack16::new(&ctx);
    assert!(kernel.pack_unorm2x16(&ctx, &[]).is_empty());
    assert!(kernel.pack_snorm2x16(&ctx, &[]).is_empty());
    assert!(kernel.unpack_unorm2x16(&ctx, &[]).is_empty());
    assert!(kernel.unpack_snorm2x16(&ctx, &[]).is_empty());
}
