//! Real-device parity for the `Gray`-code `u32` twin:
//! [`GpuGrayCode`](prism_volumetric_gpu::gray_code::GpuGrayCode) must reproduce
//! the `CPU` golden
//! [`gray_code`](prism_render_architecture::particle::gray_code) element for
//! element across the encode, decode, next, prev and diff transforms.
//!
//! The fixtures cover the full-range boundaries (`0`, `1`, `u32::MAX`), every
//! single-bit value, adjacent `Gray` pairs (so the `next`/`prev` round trip and
//! the single-bit adjacency diff are both exercised), the diff `None` cases
//! (identical and non-adjacent codes) and a `LCG`-generated batch for the
//! `binary <-> gray` round trip.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Every transform is pure integer bit algebra with no rounding anywhere, so
//! `CPU` and `GPU` must agree bit for bit. The comparison is an exact `==` on
//! every `u32` output and on the recovered `Option<u32>` for the diff, with no
//! tolerance: any mismatch is a genuine port bug. `WGSL` has no `u64`, so the
//! `u64` reference variants are out of scope here.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gray_code`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::gray_code::{
    binary_to_gray_u32, gray_diff_bit_index, gray_to_binary_u32, next_gray_u32, prev_gray_u32,
};
use prism_volumetric_gpu::gray_code::GpuGrayCode;
use prism_volumetric_gpu::GpuContext;

/// A deterministic linear-congruential generator so the randomized fixtures are
/// reproducible bit for bit across runs and platforms (the same constants the
/// `CPU` golden tests use).
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// A broad fixture of `u32` values: the `0`/`1`/`u32::MAX` boundaries, every
/// single-bit value `1 << k`, a few hand-picked patterns and a `LCG`-generated
/// tail, so the kernels see edge and random inputs alike.
fn fixture() -> Vec<u32> {
    let mut values = vec![
        0u32,
        1,
        2,
        3,
        u32::MAX,
        u32::MAX - 1,
        0xDEAD_BEEF,
        0x00C0_FFEE,
    ];
    for k in 0..32 {
        values.push(1u32 << k);
    }
    let mut state = 0x0123_4567_89AB_CDEF_u64;
    for _ in 0..4096 {
        values.push((lcg_next(&mut state) >> 32) as u32);
    }
    values
}

#[test]
fn binary_to_gray_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGrayCode::new(&ctx);
    let input = fixture();
    let got = gpu.binary_to_gray(&ctx, &input);
    assert_eq!(got.len(), input.len());
    for (idx, &n) in input.iter().enumerate() {
        assert_eq!(
            got[idx],
            binary_to_gray_u32(n),
            "binary_to_gray mismatch at {idx} for input {n:#010x}"
        );
    }
}

#[test]
fn gray_to_binary_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGrayCode::new(&ctx);
    let input = fixture();
    let got = gpu.gray_to_binary(&ctx, &input);
    assert_eq!(got.len(), input.len());
    for (idx, &g) in input.iter().enumerate() {
        assert_eq!(
            got[idx],
            gray_to_binary_u32(g),
            "gray_to_binary mismatch at {idx} for code {g:#010x}"
        );
    }
}

#[test]
fn binary_gray_binary_round_trip_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGrayCode::new(&ctx);
    let input = fixture();
    // Encode then decode on the device must return the original counter exactly.
    let gray = gpu.binary_to_gray(&ctx, &input);
    let back = gpu.gray_to_binary(&ctx, &gray);
    assert_eq!(back, input, "binary -> gray -> binary must be the identity");
}

#[test]
fn next_gray_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGrayCode::new(&ctx);
    // Feed Gray codes (encode a counter sweep) so next_gray walks real codes,
    // including the wrap at the top of the cycle.
    let counters: Vec<u32> = (0..256u32)
        .chain([u32::MAX - 1, u32::MAX])
        .map(binary_to_gray_u32)
        .collect();
    let got = gpu.next_gray(&ctx, &counters);
    assert_eq!(got.len(), counters.len());
    for (idx, &g) in counters.iter().enumerate() {
        assert_eq!(got[idx], next_gray_u32(g), "next_gray mismatch at {idx}");
    }
}

#[test]
fn prev_gray_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGrayCode::new(&ctx);
    // Includes gray(0) == 0 so the wrap to binary_to_gray(u32::MAX) is covered.
    let codes: Vec<u32> = (0..256u32)
        .chain([u32::MAX])
        .map(binary_to_gray_u32)
        .collect();
    let got = gpu.prev_gray(&ctx, &codes);
    assert_eq!(got.len(), codes.len());
    for (idx, &g) in codes.iter().enumerate() {
        assert_eq!(got[idx], prev_gray_u32(g), "prev_gray mismatch at {idx}");
    }
}

#[test]
fn next_prev_round_trip_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGrayCode::new(&ctx);
    let input = fixture();
    // prev(next(g)) == g and next(prev(g)) == g for every code, on-device.
    let forward_back = gpu.prev_gray(&ctx, &gpu.next_gray(&ctx, &input));
    assert_eq!(forward_back, input, "prev . next must be the identity");
    let back_forward = gpu.next_gray(&ctx, &gpu.prev_gray(&ctx, &input));
    assert_eq!(back_forward, input, "next . prev must be the identity");
}

#[test]
fn gray_diff_bit_index_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGrayCode::new(&ctx);

    let mut a: Vec<u32> = Vec::new();
    let mut b: Vec<u32> = Vec::new();

    // None: identical codes.
    a.push(0);
    b.push(0);
    a.push(0xDEAD_BEEF);
    b.push(0xDEAD_BEEF);

    // Some(k): differ only in bit k, for every single-bit position.
    for k in 0..32 {
        a.push(0);
        b.push(1u32 << k);
    }

    // Some: adjacent counters produce single-bit-adjacent codes.
    for n in 0..512u32 {
        a.push(binary_to_gray_u32(n));
        b.push(binary_to_gray_u32(n + 1));
    }

    // None: non-adjacent codes (two or more differing bits).
    a.push(0b0000);
    b.push(0b0011);
    a.push(binary_to_gray_u32(10));
    b.push(binary_to_gray_u32(12));

    let got = gpu.gray_diff_bit_index(&ctx, &a, &b);
    assert_eq!(got.len(), a.len());
    for (idx, (&ga, &gb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            got[idx],
            gray_diff_bit_index(ga, gb),
            "gray_diff mismatch at {idx} for ({ga:#010x}, {gb:#010x})"
        );
    }
}

#[test]
fn boundary_values_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGrayCode::new(&ctx);
    let input = vec![0u32, u32::MAX, 0x8000_0000];
    assert_eq!(
        gpu.binary_to_gray(&ctx, &input),
        vec![0, 0x8000_0000, binary_to_gray_u32(0x8000_0000)]
    );
    assert_eq!(
        gpu.gray_to_binary(&ctx, &[0u32, 0x8000_0000]),
        vec![0, u32::MAX]
    );
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGrayCode::new(&ctx);
    // No dispatch is issued and every entry point returns an empty vector.
    assert!(gpu.binary_to_gray(&ctx, &[]).is_empty());
    assert!(gpu.gray_to_binary(&ctx, &[]).is_empty());
    assert!(gpu.next_gray(&ctx, &[]).is_empty());
    assert!(gpu.prev_gray(&ctx, &[]).is_empty());
    assert!(gpu.gray_diff_bit_index(&ctx, &[], &[]).is_empty());
}
