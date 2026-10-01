//! Real-device parity for the `32`-bit bit-reversal `u32` twin:
//! [`GpuBitReversalU32`](prism_volumetric_gpu::bit_reversal_u32::GpuBitReversalU32)
//! must reproduce the `CPU` golden
//! [`bit_reversal_u32`](prism_render_architecture::particle::bit_reversal_u32)
//! element for element across the reverse-bits, reverse-lowest-bits,
//! bit-reversed increment and palindrome transforms.
//!
//! The fixtures cover a spread of words (`0`, `u32::MAX`, single-bit lanes and
//! well-mixed patterns), every `bits` width in `0..=32` for the low-lane
//! reversal and palindrome, full bit-reversed counter sweeps for small widths,
//! and the golden small-case palindrome vectors.
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
//! every `u32`/`bool` output, with no tolerance: any mismatch is a genuine port
//! bug. `WGSL` has no `u64`, and only the `u32` core is twinned, so nothing is
//! out of scope here.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_reversal_u32`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::bit_reversal_u32 as golden;
use prism_volumetric_gpu::bit_reversal_u32::{
    host_bit_reverse_increment, host_is_bit_reversal_palindrome, host_reverse_bits_u32,
    host_reverse_lowest_bits, GpuBitReversalU32,
};
use prism_volumetric_gpu::GpuContext;

/// A broad fixture of words: the boundaries `0` and `u32::MAX`, single-bit
/// lanes at the `LSB`, `MSB` and a middle lane, and hand-picked well-mixed
/// patterns so the kernels see edge and dense inputs alike.
fn word_fixture() -> Vec<u32> {
    vec![
        0u32,
        1,
        2,
        3,
        0x8000_0000,
        0x0001_0000,
        u32::MAX,
        u32::MAX - 1,
        0x9E37_79B9,
        0xDEAD_BEEF,
        0x0BAD_F00D,
        0xABCD_1234,
        0x5555_AAAA,
        0xAAAA_5555,
        0x0F1E_2D3C,
        0x1357_9BDF,
        0x0123_4567,
        0x89AB_CDEF,
        0xFFFF_F105,
        0xFFFF_F104,
    ]
}

/// Sanity-couples the in-crate host mirror to the golden module so a drift in
/// either reference is caught before it masks a kernel bug.
#[test]
fn host_mirror_tracks_golden() {
    for &x in &word_fixture() {
        assert_eq!(host_reverse_bits_u32(x), golden::reverse_bits_u32(x));
        let mut bits = 0u32;
        while bits <= 32 {
            assert_eq!(
                host_reverse_lowest_bits(x, bits),
                golden::reverse_lowest_bits(x, bits),
                "reverse_lowest_bits host/golden drift at x {x:#010x} bits {bits}"
            );
            assert_eq!(
                host_is_bit_reversal_palindrome(x, bits),
                golden::is_bit_reversal_palindrome(x, bits),
                "palindrome host/golden drift at x {x:#010x} bits {bits}"
            );
            bits += 1;
        }
        let mut b = 1u32;
        while b <= 32 {
            assert_eq!(
                host_bit_reverse_increment(x, b),
                golden::bit_reverse_increment(x, b),
                "increment host/golden drift at x {x:#010x} bits {b}"
            );
            b += 1;
        }
    }
}

#[test]
fn reverse_bits_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitReversalU32::new(&ctx);
    let xs = word_fixture();
    let got = gpu.reverse_bits(&ctx, &xs);
    assert_eq!(got.len(), xs.len());
    for (idx, &x) in xs.iter().enumerate() {
        assert_eq!(
            got[idx],
            golden::reverse_bits_u32(x),
            "reverse_bits mismatch at {idx} for x {x:#010x}"
        );
    }
}

#[test]
fn reverse_bits_hard_vectors() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitReversalU32::new(&ctx);
    // Single-lane words map to the mirrored lane across the full width.
    let xs = vec![0x0000_0001u32, 0x8000_0000, 0x0000_0003, 0u32, u32::MAX];
    let expected = vec![0x8000_0000u32, 0x0000_0001, 0xC000_0000, 0u32, u32::MAX];
    assert_eq!(gpu.reverse_bits(&ctx, &xs), expected);
}

#[test]
fn reverse_lowest_bits_all_widths() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitReversalU32::new(&ctx);
    // Pair every fixture word with every width in 0..=32.
    let base = word_fixture();
    let mut xs: Vec<u32> = Vec::new();
    let mut bits: Vec<u32> = Vec::new();
    for &x in &base {
        for b in 0..=32u32 {
            xs.push(x);
            bits.push(b);
        }
    }
    let got = gpu.reverse_lowest_bits(&ctx, &xs, &bits);
    assert_eq!(got.len(), xs.len());
    for (idx, (&x, &b)) in xs.iter().zip(bits.iter()).enumerate() {
        assert_eq!(
            got[idx],
            golden::reverse_lowest_bits(x, b),
            "reverse_lowest_bits mismatch at {idx} for x {x:#010x} bits {b}"
        );
    }
}

#[test]
fn reverse_lowest_bits_known_bits3() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitReversalU32::new(&ctx);
    // The 2^3-point permutation: i -> reverse of its low three lanes.
    let xs: Vec<u32> = (0..8u32).collect();
    let bits = vec![3u32; xs.len()];
    let expected = vec![0u32, 4, 2, 6, 1, 5, 3, 7];
    assert_eq!(gpu.reverse_lowest_bits(&ctx, &xs, &bits), expected);
}

#[test]
fn bit_reverse_increment_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitReversalU32::new(&ctx);
    // Pair every fixture word with every width in 1..=32.
    let base = word_fixture();
    let mut indices: Vec<u32> = Vec::new();
    let mut bits: Vec<u32> = Vec::new();
    for &x in &base {
        for b in 1..=32u32 {
            indices.push(x);
            bits.push(b);
        }
    }
    let got = gpu.bit_reverse_increment(&ctx, &indices, &bits);
    assert_eq!(got.len(), indices.len());
    for (idx, (&x, &b)) in indices.iter().zip(bits.iter()).enumerate() {
        assert_eq!(
            got[idx],
            golden::bit_reverse_increment(x, b),
            "bit_reverse_increment mismatch at {idx} for index {x:#010x} bits {b}"
        );
    }
}

#[test]
fn bit_reverse_increment_full_sweep_small_widths() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitReversalU32::new(&ctx);
    // Starting at 0, applying the increment 2^bits - 1 times visits every index
    // exactly once in bit-reversed order, matching the reference counter.
    for bits in 1..=10u32 {
        let total = 1u32 << bits;
        let mut idx = 0u32;
        let mut seen = vec![false; total as usize];
        for step in 0..total {
            assert!(!seen[idx as usize], "index {idx} revisited at bits {bits}");
            seen[idx as usize] = true;
            // Compare GPU and golden on the counter value at this step.
            assert_eq!(idx, golden::reverse_lowest_bits(step, bits));
            let next = gpu.bit_reverse_increment(&ctx, &[idx], &[bits]);
            assert_eq!(
                next[0],
                golden::bit_reverse_increment(idx, bits),
                "increment sweep mismatch at bits {bits} idx {idx}"
            );
            idx = next[0];
        }
        assert_eq!(
            idx, 0,
            "counter must wrap to 0 after full sweep at bits {bits}"
        );
        assert!(
            seen.iter().all(|&b| b),
            "sweep missed an index at bits {bits}"
        );
    }
}

#[test]
fn is_bit_reversal_palindrome_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitReversalU32::new(&ctx);
    // Pair every fixture word with every width in 0..=32.
    let base = word_fixture();
    let mut xs: Vec<u32> = Vec::new();
    let mut bits: Vec<u32> = Vec::new();
    for &x in &base {
        for b in 0..=32u32 {
            xs.push(x);
            bits.push(b);
        }
    }
    let got = gpu.is_bit_reversal_palindrome(&ctx, &xs, &bits);
    assert_eq!(got.len(), xs.len());
    for (idx, (&x, &b)) in xs.iter().zip(bits.iter()).enumerate() {
        assert_eq!(
            got[idx],
            golden::is_bit_reversal_palindrome(x, b),
            "palindrome mismatch at {idx} for x {x:#010x} bits {b}"
        );
    }
}

#[test]
fn is_bit_reversal_palindrome_known_small_cases() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitReversalU32::new(&ctx);
    // Golden small cases: width-3 palindromes and non-palindromes, plus the
    // high-bit-ignoring and zero-width vacuous cases.
    let xs = vec![
        0b101u32,
        0b000,
        0b111,
        0b010,
        0b100,
        0b001,
        0xFFFF_F105,
        0xFFFF_F104,
        0xDEAD_BEEF,
    ];
    let bits = vec![3u32, 3, 3, 3, 3, 3, 3, 3, 0];
    let expected = vec![true, true, true, true, false, false, true, false, true];
    assert_eq!(gpu.is_bit_reversal_palindrome(&ctx, &xs, &bits), expected);
}

#[test]
fn full_width_edges_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitReversalU32::new(&ctx);
    // bits == 32 exercises the full-width mask and the zero-shift reversal path.
    let xs = vec![0u32, u32::MAX, 0x8000_0001, 0x8000_0000, 0x0000_0001];
    let bits = vec![32u32; xs.len()];
    let rev = gpu.reverse_lowest_bits(&ctx, &xs, &bits);
    let pal = gpu.is_bit_reversal_palindrome(&ctx, &xs, &bits);
    for (idx, &x) in xs.iter().enumerate() {
        assert_eq!(rev[idx], golden::reverse_lowest_bits(x, 32));
        assert_eq!(pal[idx], golden::is_bit_reversal_palindrome(x, 32));
    }
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBitReversalU32::new(&ctx);
    // No dispatch is issued and every entry point returns an empty vector.
    assert!(gpu.reverse_bits(&ctx, &[]).is_empty());
    assert!(gpu.reverse_lowest_bits(&ctx, &[], &[]).is_empty());
    assert!(gpu.bit_reverse_increment(&ctx, &[], &[]).is_empty());
    assert!(gpu.is_bit_reversal_palindrome(&ctx, &[], &[]).is_empty());
}
