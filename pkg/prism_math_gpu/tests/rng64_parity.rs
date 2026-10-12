//! Real-device parity for the §24.7 GPU `SplitMix64` / `xoshiro256**` shader
//! mirror.
//!
//! Each test builds a batch of `u64` seeds, draws the first output on the CPU
//! with [`prism_math::rng::SplitMix64`] / [`prism_math::rng::Xoshiro256StarStar`]
//! and runs the same draw on a real `GPU` from the shared
//! [`WGSL_U64`](prism_math::shader_mirror::WGSL_U64) emulation plus the
//! single-sourced
//! [`WGSL_SPLITMIX64`](prism_math::shader_mirror::WGSL_SPLITMIX64) /
//! [`WGSL_XOSHIRO256SS`](prism_math::shader_mirror::WGSL_XOSHIRO256SS)
//! fragments.
//!
//! Both generators are 64-bit integer state; the kernel emulates the 64-bit
//! arithmetic with 32×32 partial products, which is **exact** integer
//! arithmetic. The device draws are therefore bit-identical to the CPU, so the
//! suite compares with strict equality. It skips gracefully when no adapter is
//! available so it still passes on a device-less CI image while running the full
//! dispatch on a real `GPU`.

use prism_math::rng::{SplitMix64, Xoshiro256StarStar};
use prism_math_gpu::{GpuContext, GpuRng64, Seed64};

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

/// Seeds exercising low/high bits and the extremes.
fn seeds() -> [u64; 10] {
    [
        0,
        1,
        42,
        0x1234_5678_9ABC_DEF0,
        0xFFFF_FFFF_FFFF_FFFF,
        0xDEAD_BEEF_CAFE_BABE,
        0x0000_0000_FFFF_FFFF,
        0xFFFF_FFFF_0000_0000,
        0x8000_0000_0000_0001,
        0x0FED_CBA9_8765_4321,
    ]
}

#[test]
fn splitmix64_first_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuRng64::new(&ctx);
    let raw = seeds();
    let inputs: Vec<Seed64> = raw.iter().map(|&s| Seed64::new(s)).collect();

    let got = kernel.splitmix64_first(&ctx, &inputs);
    assert_eq!(got.len(), raw.len());
    for (i, &seed) in raw.iter().enumerate() {
        let expected = SplitMix64::new(seed).next();
        assert_eq!(
            got[i], expected,
            "splitmix64[{i}] seed={seed:#x} gpu={:#x} cpu={expected:#x}",
            got[i]
        );
    }
}

#[test]
fn xoshiro256ss_first_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuRng64::new(&ctx);
    let raw = seeds();
    let inputs: Vec<Seed64> = raw.iter().map(|&s| Seed64::new(s)).collect();

    let got = kernel.xoshiro256ss_first(&ctx, &inputs);
    assert_eq!(got.len(), raw.len());
    for (i, &seed) in raw.iter().enumerate() {
        let expected = Xoshiro256StarStar::new(seed).next();
        assert_eq!(
            got[i], expected,
            "xoshiro256**[{i}] seed={seed:#x} gpu={:#x} cpu={expected:#x}",
            got[i]
        );
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuRng64::new(&ctx);
    assert!(kernel.splitmix64_first(&ctx, &[]).is_empty());
    assert!(kernel.xoshiro256ss_first(&ctx, &[]).is_empty());
}
