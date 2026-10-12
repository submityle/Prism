//! Real-device parity for the §24.7 GPU `PCG32` shader mirror.
//!
//! Each test builds a batch of `(seed, stream)` selectors, draws the first
//! 32-bit output on the CPU with
//! [`prism_math::rng::pcg::Pcg32::with_stream`] and runs the same draw on a real
//! `GPU` from the single-sourced
//! [`WGSL_PCG32`](prism_math::shader_mirror::WGSL_PCG32) fragment.
//!
//! `PCG32` is integer 64-bit LCG state with a 32-bit output; the kernel emulates
//! the 64-bit multiply with 32×32 partial products, which is **exact** integer
//! arithmetic. The device output is therefore bit-identical to the CPU for
//! every seed, so the suite compares with strict equality — a stronger
//! guarantee than any float op. It skips gracefully when no adapter is available
//! so it still passes on a device-less CI image while running the full dispatch
//! on a real `GPU`.

use prism_math::rng::Pcg32;
use prism_math_gpu::{GpuContext, GpuPcg32, PcgSeed};

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

#[test]
fn pcg32_first_draw_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPcg32::new(&ctx);

    // Small integer seeds, a high-bit seed, distinct streams, and the maximum
    // u64 values to exercise the full 64-bit multiply/permutation.
    let cases: &[(u64, u64)] = &[
        (0, 0),
        (1, 0),
        (42, 0),
        (42, 7),
        (0x1234_5678_9ABC_DEF0, 0),
        (0xFFFF_FFFF_FFFF_FFFF, 0),
        (0, 0xFFFF_FFFF_FFFF_FFFF),
        (0xDEAD_BEEF_CAFE_BABE, 0x0FED_CBA9_8765_4321),
        (0, 1),
        (0, 2),
        (7, 7),
        (u64::MAX, u64::MAX),
    ];

    let seeds: Vec<PcgSeed> = cases.iter().map(|&(s, t)| PcgSeed::new(s, t)).collect();
    let got = kernel.first_u32(&ctx, &seeds);
    assert_eq!(got.len(), cases.len());

    for (i, &(seed, stream)) in cases.iter().enumerate() {
        let mut cpu = Pcg32::with_stream(seed, stream);
        let expected = cpu.next();
        assert_eq!(
            got[i], expected,
            "pcg32[{i}] seed={seed:#x} stream={stream:#x} gpu={:#x} cpu={expected:#x}",
            got[i]
        );
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuPcg32::new(&ctx);
    assert!(kernel.first_u32(&ctx, &[]).is_empty());
}
