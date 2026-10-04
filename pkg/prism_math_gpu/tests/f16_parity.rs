//! Real-device parity for the §24.1 / §24.3 half-precision (`binary16`)
//! pack/unpack shader mirror.
//!
//! The pack kernel folds each `f32` lane pair into one `u32` of two packed
//! `binary16` values on a real `GPU` from the single-sourced
//! [`WGSL_F16`](prism_math::shader_mirror::WGSL_F16) fragment (the WGSL builtin
//! `pack2x16float`) and the unpack kernel widens them back
//! (`unpack2x16float`). Both are diffed against the CPU reference
//! [`prism_math::f16::F16`].
//!
//! WGSL defines `pack2x16float` as round-to-nearest-even, exactly like the CPU
//! reference, so for finite values inside the `f16` **normal** range the packed
//! 16 bits match [`F16::from_f32`](prism_math::f16::F16::from_f32)
//! **bit-for-bit**; these tests assert that. Subnormals (which GPUs may flush to
//! zero) and overflow / `NaN` (implementation-defined for `pack2x16float`) are a
//! documented honest boundary and are not asserted bit-exact. The
//! `binary16 -> f32` unpack direction is exact widening on both sides. The
//! suite skips gracefully when no adapter is available.

use prism_math::f16::F16;
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuF16Pack;

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

/// CPU reference: fold two `f32` lanes into one packed `u32` (low = `a`).
fn cpu_pack(a: f32, b: f32) -> u32 {
    let lo = u32::from(F16::from_f32(a).to_bits());
    let hi = u32::from(F16::from_f32(b).to_bits());
    lo | (hi << 16)
}

/// Builds a batch of `f32` values that all land in the `f16` **normal** range
/// (`|x|` in `[2^-14, 65504]`), mixing exactly-representable `f16` values with
/// values that force a round-to-nearest-even step.
fn normal_range_values() -> Vec<f32> {
    let mut v: Vec<f32> = Vec::new();
    // Exactly-representable f16 normals, both signs, swept across the exponent
    // and mantissa space (step keeps the batch compact but wide).
    let mut h: u32 = 0x0400;
    while h <= 0x7BFF {
        v.push(F16::from_bits(h as u16).to_f32());
        v.push(F16::from_bits((h as u16) | 0x8000).to_f32());
        h += 7;
    }
    // Arithmetic-derived values that are generally not exact f16, forcing a
    // rounding step on both sides; kept inside the normal range.
    for i in 1..=2000u32 {
        let x = i as f32 * 15.0; // up to 30000
        v.push(x);
        v.push(-x);
        let y = i as f32 * 0.0007; // down to ~7e-4 (well above 2^-14)
        v.push(y);
        v.push(-y);
    }
    v.push(0.0); // +0 is exact (low bits 0x0000)
    v
}

#[test]
fn pack2_matches_cpu_reference_in_normal_range() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuF16Pack::new(&ctx);

    let vals = normal_range_values();
    // Pair consecutive values; pad to an even count.
    let mut pairs: Vec<[f32; 2]> = Vec::new();
    let mut i = 0;
    while i + 1 < vals.len() {
        pairs.push([vals[i], vals[i + 1]]);
        i += 2;
    }

    let gpu = kernel.pack2(&ctx, &pairs);
    assert_eq!(gpu.len(), pairs.len());
    for (p, &g) in pairs.iter().zip(gpu.iter()) {
        let cpu = cpu_pack(p[0], p[1]);
        assert_eq!(g, cpu, "pack2 drift at ({}, {})", p[0], p[1]);
    }
}

#[test]
fn unpack2_matches_cpu_reference_in_normal_range() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuF16Pack::new(&ctx);

    // Build packed keys directly from f16 normal bit patterns, both halves,
    // then check the GPU widening equals the exact CPU widening bit-for-bit.
    let mut keys: Vec<u32> = Vec::new();
    let mut h: u32 = 0x0400;
    while h <= 0x7BFF {
        let lo = h as u16;
        let hi = (h as u16) ^ 0x8000; // flip sign for the high lane
        keys.push(u32::from(lo) | (u32::from(hi) << 16));
        h += 3;
    }

    let gpu = kernel.unpack2(&ctx, &keys);
    assert_eq!(gpu.len(), keys.len());
    for (&k, got) in keys.iter().zip(gpu.iter()) {
        let lo = F16::from_bits(k as u16).to_f32();
        let hi = F16::from_bits((k >> 16) as u16).to_f32();
        assert_eq!(got[0].to_bits(), lo.to_bits(), "unpack2 low drift for {k:#010x}");
        assert_eq!(got[1].to_bits(), hi.to_bits(), "unpack2 high drift for {k:#010x}");
    }
}

#[test]
fn pack_unpack_round_trip_in_normal_range() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuF16Pack::new(&ctx);

    let mut pairs: Vec<[f32; 2]> = Vec::new();
    let mut seed = 0x51ed_600du32;
    for _ in 0..512 {
        // Random normal-range magnitudes in [2^-14, 65504], random signs.
        let a = (lcg(&mut seed) % 60000) as f32 + 0.001;
        let b = (lcg(&mut seed) % 60000) as f32 + 0.001;
        let sa = if lcg(&mut seed) & 1 == 0 { 1.0 } else { -1.0 };
        let sb = if lcg(&mut seed) & 1 == 0 { 1.0 } else { -1.0 };
        pairs.push([a * sa, b * sb]);
    }

    let keys = kernel.pack2(&ctx, &pairs);
    let back = kernel.unpack2(&ctx, &keys);
    assert_eq!(back.len(), pairs.len());
    for (p, got) in pairs.iter().zip(back.iter()) {
        // Round-trip equals the CPU's quantize-then-widen of the same input.
        let want_lo = F16::from_f32(p[0]).to_f32();
        let want_hi = F16::from_f32(p[1]).to_f32();
        assert_eq!(got[0].to_bits(), want_lo.to_bits(), "round-trip low drift");
        assert_eq!(got[1].to_bits(), want_hi.to_bits(), "round-trip high drift");
    }
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuF16Pack::new(&ctx);
    assert!(kernel.pack2(&ctx, &[]).is_empty());
    assert!(kernel.unpack2(&ctx, &[]).is_empty());
}
