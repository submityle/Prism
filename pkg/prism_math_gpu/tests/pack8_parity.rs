//! Real-device parity for the §24.1 8-bit-per-channel vertex-attribute
//! (`unorm8x4` / `snorm8x4`) pack/unpack shader mirror.
//!
//! The pack kernels fold each `vec4<f32>` lane into one `u32` of four 8-bit
//! channels on a real `GPU` from the single-sourced
//! [`WGSL_PACK8`](prism_math::shader_mirror::WGSL_PACK8) fragment (the WGSL
//! built-ins `pack4x8unorm` / `pack4x8snorm`) and the unpack kernels widen them
//! back (`unpack4x8unorm` / `unpack4x8snorm`). Both are diffed against the CPU
//! reference [`prism_math::pack8`].
//!
//! WGSL defines the quantizers as `⌊0.5 + N·clamp(c)⌋` (`N` = 255 unorm /
//! 127 snorm), exactly like the CPU reference, so for **exactly-representable
//! quantized** inputs the packed bytes match [`prism_math::pack8`]
//! **byte-for-byte**; the byte-exact sweeps assert that. Arbitrary inputs may
//! differ by at most one code at a rounding tie (implementations may round
//! halves differently), a documented honest boundary checked with a one-code
//! tolerance. The widening (`unpack`) direction is exact on both sides. The
//! suite skips gracefully when no adapter is available.

use prism_math::pack8::{
    pack_snorm4x8, pack_unorm4x8, unpack_snorm4x8, unpack_unorm4x8,
};
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuPack8;

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
/// taken channel-by-channel on the raw bytes (`unorm` interpretation).
fn max_unorm_code_diff(a: u32, b: u32) -> i32 {
    let mut m = 0;
    for s in [0, 8, 16, 24] {
        let x = ((a >> s) & 0xFF) as i32;
        let y = ((b >> s) & 0xFF) as i32;
        m = m.max((x - y).abs());
    }
    m
}

/// Maximum absolute difference, in integer codes, between two packed `u32`s,
/// interpreting each byte as a two's-complement `i8` (`snorm` interpretation).
fn max_snorm_code_diff(a: u32, b: u32) -> i32 {
    let mut m = 0;
    for s in [0, 8, 16, 24] {
        let x = i32::from(((a >> s) & 0xFF) as u8 as i8);
        let y = i32::from(((b >> s) & 0xFF) as u8 as i8);
        m = m.max((x - y).abs());
    }
    m
}

#[test]
fn pack_unorm_matches_cpu_on_exact_quantized_sweep() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPack8::new(&ctx);

    // Every one of the 256 unorm bytes maps to an exactly-representable
    // dequantized value k/255; packing it must land back on byte k on both
    // sides, byte-for-byte. Sweep all four channels through distinct bytes.
    let mut src: Vec<[f32; 4]> = Vec::new();
    for k in 0..=255u32 {
        src.push([
            k as f32 / 255.0,
            (255 - k) as f32 / 255.0,
            ((k + 85) % 256) as f32 / 255.0,
            ((k + 170) % 256) as f32 / 255.0,
        ]);
    }

    let gpu = kernel.pack_unorm4x8(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (v, &g) in src.iter().zip(gpu.iter()) {
        assert_eq!(g, pack_unorm4x8(*v), "unorm pack drift at {v:?}");
    }
}

#[test]
fn pack_snorm_matches_cpu_on_exact_quantized_sweep() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPack8::new(&ctx);

    // Every encodable signed code in [-127, 127] maps to exact k/127; packing
    // it must land back on code k byte-for-byte on both sides.
    let mut src: Vec<[f32; 4]> = Vec::new();
    for k in -127i32..=127 {
        src.push([
            k as f32 / 127.0,
            -k as f32 / 127.0,
            (k.rem_euclid(64) - 32) as f32 / 127.0,
            (127 - k.abs()) as f32 / 127.0,
        ]);
    }

    let gpu = kernel.pack_snorm4x8(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (v, &g) in src.iter().zip(gpu.iter()) {
        assert_eq!(g, pack_snorm4x8(*v), "snorm pack drift at {v:?}");
    }
}

#[test]
fn unpack_unorm_matches_cpu_bit_exact() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPack8::new(&ctx);

    // Sweep a wide spread of packed keys; widening is exact division by 255 and
    // must agree bit-for-bit with the CPU reference.
    let mut keys: Vec<u32> = Vec::new();
    let mut seed = 0x0ac8_1234u32;
    for _ in 0..4096 {
        keys.push(lcg(&mut seed));
    }
    for k in 0..=255u32 {
        keys.push(k | (k << 8) | (k << 16) | (k << 24));
    }

    let gpu = kernel.unpack_unorm4x8(&ctx, &keys);
    assert_eq!(gpu.len(), keys.len());
    for (&k, got) in keys.iter().zip(gpu.iter()) {
        let cpu = unpack_unorm4x8(k);
        for lane in 0..4 {
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
    let kernel = GpuPack8::new(&ctx);

    // The CPU reference clamps -128 up to -1.0 (max(i8/127, -1)); build keys
    // that exercise every byte including 0x80 so the clamp path is covered.
    let mut keys: Vec<u32> = Vec::new();
    let mut seed = 0x51ed_600du32;
    for _ in 0..4096 {
        keys.push(lcg(&mut seed));
    }
    for k in 0..=255u32 {
        keys.push(k | (k << 8) | (k << 16) | (k << 24));
    }

    let gpu = kernel.unpack_snorm4x8(&ctx, &keys);
    assert_eq!(gpu.len(), keys.len());
    for (&k, got) in keys.iter().zip(gpu.iter()) {
        let cpu = unpack_snorm4x8(k);
        for lane in 0..4 {
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
    let kernel = GpuPack8::new(&ctx);

    // Arbitrary [0, 1] inputs (plus a few out-of-range values to exercise the
    // clamp); allow a one-code tie-rounding difference per the honest boundary.
    let mut src: Vec<[f32; 4]> = Vec::new();
    let mut seed = 0x1357_9bdfu32;
    for _ in 0..2048 {
        let c = |s: &mut u32| (lcg(s) % 100_001) as f32 / 100_000.0;
        src.push([c(&mut seed), c(&mut seed), c(&mut seed), c(&mut seed)]);
    }
    src.push([-0.5, 1.5, 2.0, -3.0]);

    let gpu = kernel.pack_unorm4x8(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (v, &g) in src.iter().zip(gpu.iter()) {
        let cpu = pack_unorm4x8(*v);
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
    let kernel = GpuPack8::new(&ctx);

    let mut src: Vec<[f32; 4]> = Vec::new();
    let mut seed = 0x2468_ace0u32;
    for _ in 0..2048 {
        let c = |s: &mut u32| (lcg(s) % 200_001) as f32 / 100_000.0 - 1.0;
        src.push([c(&mut seed), c(&mut seed), c(&mut seed), c(&mut seed)]);
    }
    src.push([-1.5, 1.5, 2.0, -3.0]);

    let gpu = kernel.pack_snorm4x8(&ctx, &src);
    assert_eq!(gpu.len(), src.len());
    for (v, &g) in src.iter().zip(gpu.iter()) {
        let cpu = pack_snorm4x8(*v);
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
    let kernel = GpuPack8::new(&ctx);
    assert!(kernel.pack_unorm4x8(&ctx, &[]).is_empty());
    assert!(kernel.pack_snorm4x8(&ctx, &[]).is_empty());
    assert!(kernel.unpack_unorm4x8(&ctx, &[]).is_empty());
    assert!(kernel.unpack_snorm4x8(&ctx, &[]).is_empty());
}
