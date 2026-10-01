//! Real-device parity for the `32`-bit Fibonacci `LFSR` `u32` twin:
//! [`GpuFibonacciLfsr`](prism_volumetric_gpu::fibonacci_lfsr::GpuFibonacciLfsr)
//! must reproduce the `CPU` golden
//! [`fibonacci_lfsr`](prism_render_architecture::particle::fibonacci_lfsr)
//! element for element across the advance-state, nth output bit and nth
//! `32`-bit word transforms.
//!
//! The fixtures cover a spread of seeds (including the degenerate all-zero
//! state and the single-bit `1` seed used by the golden hard vectors), advance
//! counts of `0`, `1`, `32` and large step counts, the first `next_bit`
//! sequence, consecutive `next_u32` words, periodicity sampling and a full
//! [`next_array`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::next_array)
//! comparison.
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
//! every `u32` output, with no tolerance: any mismatch is a genuine port bug.
//! `WGSL` has no `u64`, and the reference is already `u32`-only, so nothing is
//! out of scope here.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fibonacci_lfsr`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr;
use prism_volumetric_gpu::fibonacci_lfsr::GpuFibonacciLfsr;
use prism_volumetric_gpu::GpuContext;

/// A broad fixture of seed registers: the degenerate `0`, the golden `1` seed,
/// the full-range boundary `u32::MAX`, a few single-bit and hand-picked
/// patterns so the kernels see edge and well-mixed seeds alike.
fn seed_fixture() -> Vec<u32> {
    vec![
        0u32,
        1,
        2,
        3,
        0x8000_0000,
        u32::MAX,
        u32::MAX - 1,
        0x9E37_79B9,
        0xDEAD_BEEF,
        0x0BAD_F00D,
        0xABCD_1234,
        0x5555_AAAA,
        0x0F1E_2D3C,
        0x1357_9BDF,
        0x0000_0001,
        0x0123_4567,
    ]
}

/// `CPU` golden: raw `state` after seeding with `seed` and advancing `n`
/// single-bit `next_bit` shifts.
fn cpu_advance(seed: u32, n: u32) -> u32 {
    let mut rng = FibonacciLfsr::from_state(seed);
    for _ in 0..n {
        rng.next_bit();
    }
    rng.state()
}

/// `CPU` golden: output bit of the `n`-th (`0`-indexed) `next_bit` call.
fn cpu_bit_at(seed: u32, n: u32) -> u32 {
    let mut rng = FibonacciLfsr::from_state(seed);
    let mut bit = 0u32;
    for _ in 0..=n {
        bit = rng.next_bit();
    }
    bit
}

/// `CPU` golden: the `n`-th (`0`-indexed) `next_u32` word.
fn cpu_word_at(seed: u32, n: u32) -> u32 {
    let mut rng = FibonacciLfsr::from_state(seed);
    for _ in 0..n {
        rng.next_u32();
    }
    rng.next_u32()
}

#[test]
fn advance_state_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFibonacciLfsr::new(&ctx);
    let seeds = seed_fixture();
    // 0 (identity), 1, a full word, two full words and a large non-aligned run.
    let steps: Vec<u32> = [0u32, 1, 32, 64, 1000, 65_537]
        .into_iter()
        .cycle()
        .take(seeds.len())
        .collect();
    let got = gpu.advance_state(&ctx, &seeds, &steps);
    assert_eq!(got.len(), seeds.len());
    for (idx, (&seed, &n)) in seeds.iter().zip(steps.iter()).enumerate() {
        assert_eq!(
            got[idx],
            cpu_advance(seed, n),
            "advance_state mismatch at {idx} for seed {seed:#010x} steps {n}"
        );
    }
}

#[test]
fn advance_zero_steps_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFibonacciLfsr::new(&ctx);
    let seeds = seed_fixture();
    let steps = vec![0u32; seeds.len()];
    // Zero advances must return the seed unchanged, exactly like `.state()`.
    assert_eq!(gpu.advance_state(&ctx, &seeds, &steps), seeds);
}

#[test]
fn next_bit_sequence_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFibonacciLfsr::new(&ctx);
    // The first 64 output bits of a well-mixed seed, index by index.
    let seed = 0x9E37_79B9u32;
    let seeds = vec![seed; 64];
    let indices: Vec<u32> = (0..64u32).collect();
    let got = gpu.next_bit_at(&ctx, &seeds, &indices);
    assert_eq!(got.len(), 64);
    for (idx, &n) in indices.iter().enumerate() {
        let bit = got[idx];
        assert!(bit == 0 || bit == 1, "bit must be 0 or 1 at {idx}");
        assert_eq!(bit, cpu_bit_at(seed, n), "next_bit mismatch at index {n}");
    }
}

#[test]
fn next_bit_hard_vector_from_state_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFibonacciLfsr::new(&ctx);
    // Golden OUTBITS for from_state(1): first bit 1, then seven zeros.
    let expected = [1u32, 0, 0, 0, 0, 0, 0, 0];
    let seeds = vec![1u32; expected.len()];
    let indices: Vec<u32> = (0..expected.len() as u32).collect();
    let got = gpu.next_bit_at(&ctx, &seeds, &indices);
    assert_eq!(got, expected, "next_bit hard vector for from_state(1)");
}

#[test]
fn advance_state_hard_vector_snapshots() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFibonacciLfsr::new(&ctx);
    // Golden SNAPSHOTS: register value after each of the first eight next_bit
    // shifts of from_state(1).
    let expected = [
        0x8000_0000u32,
        0xC000_0000,
        0x6000_0000,
        0xB000_0000,
        0xD800_0000,
        0x6C00_0000,
        0xB600_0000,
        0xDB00_0000,
    ];
    let seeds = vec![1u32; expected.len()];
    // state after (k+1) shifts for k = 0..8.
    let steps: Vec<u32> = (1..=expected.len() as u32).collect();
    let got = gpu.advance_state(&ctx, &seeds, &steps);
    assert_eq!(got, expected, "advance_state snapshots for from_state(1)");
}

#[test]
fn next_u32_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFibonacciLfsr::new(&ctx);
    // For every fixture seed, the first consecutive next_u32 words, word by word.
    let base_seeds = seed_fixture();
    let words_per_seed = 6u32;
    let mut seeds: Vec<u32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for &seed in &base_seeds {
        for n in 0..words_per_seed {
            seeds.push(seed);
            indices.push(n);
        }
    }
    let got = gpu.next_u32_at(&ctx, &seeds, &indices);
    assert_eq!(got.len(), seeds.len());
    for (idx, (&seed, &n)) in seeds.iter().zip(indices.iter()).enumerate() {
        assert_eq!(
            got[idx],
            cpu_word_at(seed, n),
            "next_u32 mismatch at {idx} for seed {seed:#010x} word {n}"
        );
    }
}

#[test]
fn next_u32_hard_vector_from_state_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFibonacciLfsr::new(&ctx);
    // Golden OUT4: four consecutive next_u32 draws for from_state(1).
    let expected = [0x8000_0000u32, 0xDB6D_B451, 0xE790_9909, 0x5C9D_4B22];
    let seeds = vec![1u32; expected.len()];
    let indices: Vec<u32> = (0..expected.len() as u32).collect();
    let got = gpu.next_u32_at(&ctx, &seeds, &indices);
    assert_eq!(got, expected, "next_u32 hard vector OUT4 for from_state(1)");
}

#[test]
fn next_array_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFibonacciLfsr::new(&ctx);
    // Mapping word index 0..N over next_u32_at must equal the reference
    // next_array::<N>() element for element.
    const N: usize = 8;
    for seed in seed_fixture() {
        let seeds = vec![seed; N];
        let indices: Vec<u32> = (0..N as u32).collect();
        let got = gpu.next_u32_at(&ctx, &seeds, &indices);
        let mut rng = FibonacciLfsr::from_state(seed);
        let expected: [u32; N] = rng.next_array();
        assert_eq!(got, expected.to_vec(), "next_array mismatch for seed {seed:#010x}");
    }
}

#[test]
fn periodicity_sampling_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFibonacciLfsr::new(&ctx);
    // Sample the stream at scattered large step counts; parity must hold at
    // every tap regardless of how far the register has advanced.
    let seed = 0xABCD_1234u32;
    let steps = vec![100u32, 500, 1024, 4096, 10_000, 50_000, 99_991, 131_072];
    let seeds = vec![seed; steps.len()];
    let got = gpu.advance_state(&ctx, &seeds, &steps);
    assert_eq!(got.len(), steps.len());
    for (idx, &n) in steps.iter().enumerate() {
        assert_eq!(
            got[idx],
            cpu_advance(seed, n),
            "periodicity mismatch at step {n}"
        );
    }
}

#[test]
fn zero_seed_is_degenerate_fixed_point() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFibonacciLfsr::new(&ctx);
    // The all-zero state only ever emits zeros and shifts to itself; the twin
    // reproduces this degenerate stream exactly, matching the reference.
    let steps = vec![0u32, 1, 32, 123, 10_000];
    let seeds = vec![0u32; steps.len()];
    assert_eq!(
        gpu.advance_state(&ctx, &seeds, &steps),
        vec![0u32; steps.len()]
    );
    let indices: Vec<u32> = (0..32u32).collect();
    let zero_seeds = vec![0u32; indices.len()];
    assert_eq!(
        gpu.next_bit_at(&ctx, &zero_seeds, &indices),
        vec![0u32; indices.len()]
    );
    assert_eq!(
        gpu.next_u32_at(&ctx, &zero_seeds, &indices),
        vec![0u32; indices.len()]
    );
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFibonacciLfsr::new(&ctx);
    // No dispatch is issued and every entry point returns an empty vector.
    assert!(gpu.advance_state(&ctx, &[], &[]).is_empty());
    assert!(gpu.next_bit_at(&ctx, &[], &[]).is_empty());
    assert!(gpu.next_u32_at(&ctx, &[], &[]).is_empty());
}
