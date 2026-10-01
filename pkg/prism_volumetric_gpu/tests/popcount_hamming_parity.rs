//! Real-device parity for the population-count / `Hamming` / `parity` twin:
//! [`GpuPopcountHamming`](prism_volumetric_gpu::popcount_hamming::GpuPopcountHamming)
//! must reproduce the `CPU` golden
//! [`popcount_hamming`](prism_render_architecture::particle::popcount_hamming)
//! `u32`-domain functions
//! ([`popcount_u32`](prism_render_architecture::particle::popcount_hamming::popcount_u32),
//! [`hamming_distance_u32`](prism_render_architecture::particle::popcount_hamming::hamming_distance_u32),
//! [`parity_u32`](prism_render_architecture::particle::popcount_hamming::parity_u32))
//! element for element, and the host-summed per-element counts must reproduce
//! the whole-slice weight and distance.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Every operation is pure unsigned integer arithmetic with no reordering, so
//! `CPU` and `GPU` are **bit-exact**: the comparison is a precise `==` with no
//! tolerance. There is no `u64` type in `WGSL`, so the `u64` golden variants are
//! not twinned and not tested here.
//!
//! # Fixtures
//!
//! The sweep covers `0`, `u32::MAX`, every single-bit word, the alternating
//! patterns `0xAAAAAAAA` / `0x55555555`, and a pseudo-random `LCG` batch.
//! `Hamming` distance covers `a == b` (distance `0`) and fully-complementary
//! inputs (distance `32`); `parity` covers an even and an odd half.
//!
//! Provenance: twinned from this repository's
//! `prism_render_architecture::particle::popcount_hamming`; no third-party
//! engine source or derived code.

use prism_render_architecture::particle::popcount_hamming::{
    hamming_distance_u32, parity_u32, popcount_u32,
};
use prism_volumetric_gpu::popcount_hamming::GpuPopcountHamming;
use prism_volumetric_gpu::GpuContext;

/// A tiny inline linear congruential generator for pseudo-random sweeps, using
/// the same Numerical Recipes constants as the golden's test generator so the
/// two suites explore comparable words. Test-only, no external deps.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Builds the shared fixture of `u32` words: `0`, `u32::MAX`, every single-bit
/// value, the two alternating patterns, and a pseudo-random `LCG` batch.
fn fixture_words() -> Vec<u32> {
    let mut words = vec![0u32, u32::MAX, 0xAAAA_AAAA, 0x5555_5555];
    for i in 0..32u32 {
        words.push(1u32 << i);
    }
    let mut state = 0x1234_5678_9abc_def0u64;
    for _ in 0..256 {
        words.push(lcg_next(&mut state) as u32);
    }
    words
}

#[test]
fn popcount_matches_reference_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPopcountHamming::new(&ctx);
    let words = fixture_words();
    let got = gpu.popcount(&ctx, &words);
    assert_eq!(got.len(), words.len());
    for (i, &w) in words.iter().enumerate() {
        // Bit-exact: pin against the golden SWAR popcount and the intrinsic.
        assert_eq!(
            got[i],
            popcount_u32(w),
            "popcount mismatch at word {w:#010x}"
        );
        assert_eq!(got[i], w.count_ones());
    }
}

#[test]
fn popcount_known_values() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPopcountHamming::new(&ctx);
    let words = vec![0u32, 1, 0xFFFF_FFFF, 0xAAAA_AAAA, 0x5555_5555, 0x0F0F_0F0F];
    let got = gpu.popcount(&ctx, &words);
    assert_eq!(got, vec![0, 1, 32, 16, 16, 16]);
}

#[test]
fn popcount_empty_input_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPopcountHamming::new(&ctx);
    let got = gpu.popcount(&ctx, &[]);
    assert!(got.is_empty());
}

#[test]
fn popcount_sum_matches_slice_weight() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPopcountHamming::new(&ctx);
    // The whole-array weight is the host sum of the per-element counts, which
    // must equal the CPU sum of the golden popcount over the same words.
    let words = fixture_words();
    let got = gpu.popcount(&ctx, &words);
    let gpu_total: u64 = got.iter().map(|&c| u64::from(c)).sum();
    let cpu_total: u64 = words.iter().map(|&w| u64::from(popcount_u32(w))).sum();
    assert_eq!(gpu_total, cpu_total);
}

#[test]
fn hamming_distance_matches_reference_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPopcountHamming::new(&ctx);
    let mut state = 0x4242_4242_2424_2424u64;
    let mut a = Vec::new();
    let mut b = Vec::new();
    // Random pairs plus the two extreme cases: a == b (distance 0) and fully
    // complementary a / !a (distance 32).
    for _ in 0..256 {
        a.push(lcg_next(&mut state) as u32);
        b.push(lcg_next(&mut state) as u32);
    }
    for &w in &[0u32, u32::MAX, 0xAAAA_AAAA, 0x1357_9bdf] {
        a.push(w);
        b.push(w); // a == b -> distance 0
        a.push(w);
        b.push(!w); // fully complementary -> distance 32
    }
    let got = gpu.hamming_distance(&ctx, &a, &b);
    assert_eq!(got.len(), a.len());
    for ((&x, &y), &dist) in a.iter().zip(b.iter()).zip(got.iter()) {
        assert_eq!(dist, hamming_distance_u32(x, y));
        assert_eq!(dist, (x ^ y).count_ones());
    }
}

#[test]
fn hamming_distance_extremes_are_zero_and_thirty_two() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPopcountHamming::new(&ctx);
    let a = vec![0x1234_5678u32, 0u32, u32::MAX];
    let b = vec![0x1234_5678u32, u32::MAX, 0u32];
    let got = gpu.hamming_distance(&ctx, &a, &b);
    assert_eq!(got, vec![0, 32, 32]);
}

#[test]
fn hamming_distance_empty_input_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPopcountHamming::new(&ctx);
    let got = gpu.hamming_distance(&ctx, &[], &[]);
    assert!(got.is_empty());
}

#[test]
fn hamming_distance_sum_matches_slice_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPopcountHamming::new(&ctx);
    let mut state = 0x0f0f_f0f0_0f0f_f0f0u64;
    let mut a = Vec::new();
    let mut b = Vec::new();
    for _ in 0..300 {
        a.push(lcg_next(&mut state) as u32);
        b.push(lcg_next(&mut state) as u32);
    }
    let got = gpu.hamming_distance(&ctx, &a, &b);
    let gpu_total: u64 = got.iter().map(|&c| u64::from(c)).sum();
    let cpu_total: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| u64::from(hamming_distance_u32(x, y)))
        .sum();
    assert_eq!(gpu_total, cpu_total);
}

#[test]
#[should_panic(expected = "equal-length")]
fn hamming_distance_length_mismatch_panics() {
    let Some(ctx) = GpuContext::try_headless() else {
        // Force the panic even on a host with no adapter so the test is not
        // vacuously passing: the length check runs before any device work.
        panic!("equal-length inputs required (no adapter)");
    };
    let gpu = GpuPopcountHamming::new(&ctx);
    let _ = gpu.hamming_distance(&ctx, &[1, 2, 3], &[1, 2]);
}

#[test]
fn parity_matches_reference_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPopcountHamming::new(&ctx);
    let words = fixture_words();
    let got = gpu.parity(&ctx, &words);
    assert_eq!(got.len(), words.len());
    for (i, &w) in words.iter().enumerate() {
        // GPU emits 1 for odd popcount, 0 for even; map the golden bool the
        // same way for a bit-exact comparison.
        assert_eq!(
            got[i],
            u32::from(parity_u32(w)),
            "parity mismatch at {w:#010x}"
        );
        assert_eq!(got[i], w.count_ones() & 1);
    }
}

#[test]
fn parity_known_odd_and_even_halves() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPopcountHamming::new(&ctx);
    // Even-parity words first, then odd-parity words, so both halves are
    // exercised explicitly.
    let even = vec![0u32, 0b11, u32::MAX, 0x0F0F_0F0F];
    let odd = vec![1u32, 0b111, 0x8000_0000, 0x0000_0007];
    let mut words = even.clone();
    words.extend_from_slice(&odd);
    let got = gpu.parity(&ctx, &words);
    for &p in &got[..even.len()] {
        assert_eq!(p, 0, "every word in the even half must have even parity");
    }
    for &p in &got[even.len()..] {
        assert_eq!(p, 1, "every word in the odd half must have odd parity");
    }
}

#[test]
fn parity_single_bits_all_odd() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPopcountHamming::new(&ctx);
    let words: Vec<u32> = (0..32u32).map(|i| 1u32 << i).collect();
    let got = gpu.parity(&ctx, &words);
    for (i, &p) in got.iter().enumerate() {
        assert_eq!(p, 1, "single-bit word 1 << {i} must have odd parity");
    }
}
