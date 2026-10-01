//! Real-device parity for the `radix`-`histogram` twin:
//! [`GpuRadixHistogram`](prism_volumetric_gpu::gpu_radix_histogram::GpuRadixHistogram)
//! must reproduce the `CPU` golden
//! [`histogram`](prism_render_architecture::particle::gpu_radix_histogram::histogram)
//! across an empty batch (the all-zero `histogram`), a single key, a batch that
//! collapses entirely into one bucket, a batch spread one-per-bucket across the
//! whole `digit` range, a large pseudo-random batch, and a sweep over several
//! `pass`/`bits` combinations.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! A `histogram` bucket count is a `u32` integer, so parity is asserted with
//! **exact per-bucket equality**, not a float tolerance. There is no float math
//! anywhere in the kernel — `digit` extraction is a shift and a mask and
//! counting is `atomicAdd` — so the comparison is bit-exact by construction and
//! has no `ULP`-boundary degenerate region to avoid.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_radix_histogram`；
//! 无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::gpu_radix_histogram::{histogram, RadixConfig};
use prism_volumetric_gpu::gpu_radix_histogram::{GpuRadixHistogram, RadixHistogramQuery};
use prism_volumetric_gpu::GpuContext;

/// A tiny integer linear-congruential generator; only integer arithmetic, so no
/// transcendental appears. Returns the full 32-bit high word of the state.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    fn fill(&mut self, len: usize) -> Vec<u32> {
        (0..len).map(|_| self.next_u32()).collect()
    }
}

/// Runs the `GPU` `histogram` and asserts exact per-bucket parity against the
/// `CPU` golden [`histogram`], returning the counts for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuRadixHistogram,
    keys: &[u32],
    pass: u32,
    bits: u32,
) -> Vec<u32> {
    let config = RadixConfig::new(bits);
    let query = RadixHistogramQuery {
        keys: keys.to_vec(),
        config,
        pass,
    };
    let got = gpu.eval(ctx, &query);
    let want = histogram(keys, pass, bits);

    assert_eq!(
        got.len(),
        want.len(),
        "histogram length must equal bucket_count (bits {bits})"
    );
    for (bucket, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g, w,
            "bucket {bucket}: gpu count {g} vs cpu count {w} (pass {pass} bits {bits})"
        );
    }
    got
}

#[test]
fn empty_input_is_the_zero_histogram() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixHistogram::new(&ctx);
    // No key issues no dispatch; the result is the correctly sized zero
    // histogram and matches the reference exactly.
    let got = check(&ctx, &gpu, &[], 0, 4);
    assert_eq!(got.len(), 16, "a 4-bit pass has 16 buckets");
    assert!(got.iter().all(|&c| c == 0), "every bucket is empty");
}

#[test]
fn single_key_lands_in_one_bucket() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixHistogram::new(&ctx);
    // Key 0x0000_0023, pass 0, 4-bit digit -> nibble 0x3 == bucket 3.
    let got = check(&ctx, &gpu, &[0x0000_0023], 0, 4);
    assert_eq!(got[3], 1, "the lone key falls in bucket 3");
    assert_eq!(
        got.iter().map(|&c| u64::from(c)).sum::<u64>(),
        1,
        "exactly one key is counted"
    );
}

#[test]
fn all_keys_collapse_to_a_single_bucket() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixHistogram::new(&ctx);
    // All keys share pass-0 4-bit digit 0x5 (low nibble), so every one counts
    // into bucket 5 and all other buckets stay empty.
    let keys = [0x0000_0005u32, 0x0000_0015, 0x0000_00A5, 0x0000_0FF5, 0x1234_5675];
    let got = check(&ctx, &gpu, &keys, 0, 4);
    assert_eq!(got[5], keys.len() as u32, "every key lands in bucket 5");
    for (bucket, &count) in got.iter().enumerate() {
        if bucket != 5 {
            assert_eq!(count, 0, "bucket {bucket} must stay empty");
        }
    }
}

#[test]
fn one_key_per_bucket_fills_the_digit_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixHistogram::new(&ctx);
    // Keys 0..16 have distinct pass-0 4-bit digits 0..15, so each of the 16
    // buckets receives exactly one key.
    let keys: Vec<u32> = (0u32..16).collect();
    let got = check(&ctx, &gpu, &keys, 0, 4);
    assert!(
        got.iter().all(|&c| c == 1),
        "each bucket receives exactly one key: {got:?}"
    );
}

#[test]
fn uniform_spread_counts_evenly() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixHistogram::new(&ctx);
    // 256 keys cycling 0..16 give each 4-bit pass-0 bucket exactly 16 keys.
    let keys: Vec<u32> = (0u32..256).map(|k| k % 16).collect();
    let got = check(&ctx, &gpu, &keys, 0, 4);
    assert!(
        got.iter().all(|&c| c == 16),
        "each bucket receives exactly 16 keys: {got:?}"
    );
}

#[test]
fn large_random_batch_conserves_totals() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixHistogram::new(&ctx);
    let mut lcg = Lcg::new(0xC0FF_EE01_1357_9BDF);
    let keys = lcg.fill(4096);
    // A spread of pass/bits layouts, each compared bucket-for-bucket exactly.
    for bits in [1u32, 2, 4, 8] {
        for pass in [0u32, 1, 3, 7] {
            let got = check(&ctx, &gpu, &keys, pass, bits);
            assert_eq!(
                got.iter().map(|&c| u64::from(c)).sum::<u64>(),
                keys.len() as u64,
                "every key is counted exactly once (pass {pass} bits {bits})"
            );
            assert_eq!(
                got.len(),
                1usize << bits,
                "bucket_count is 1 << bits (bits {bits})"
            );
        }
    }
}

#[test]
fn high_pass_sees_zero_padded_digits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixHistogram::new(&ctx);
    // At 8 bits, pass 4 would shift by 32 and reach past the top bit, so every
    // key's digit is the zero-padded 0 and all keys pile into bucket 0 — exactly
    // the golden's behaviour, verified by the shared check against `histogram`.
    let mut lcg = Lcg::new(0x0BAD_F00D_FEED_BEEF);
    let keys = lcg.fill(300);
    let got = check(&ctx, &gpu, &keys, 4, 8);
    assert_eq!(got[0], keys.len() as u32, "the zero-padded pass counts into bucket 0");
    assert!(
        got.iter().skip(1).all(|&c| c == 0),
        "no key escapes bucket 0 on a fully zero-padded pass"
    );
}

#[test]
fn boundary_keys_bin_exactly() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRadixHistogram::new(&ctx);
    // Boundary values (0, u32::MAX, the sign bit, duplicates) across every
    // supported bit width and the first few passes, each compared exactly.
    let keys = [
        u32::MAX,
        0,
        u32::MAX,
        1,
        0,
        0x8000_0000,
        0x7FFF_FFFF,
        42,
        42,
    ];
    for bits in [1u32, 2, 4, 8] {
        for pass in [0u32, 1, 2, 3] {
            check(&ctx, &gpu, &keys, pass, bits);
        }
    }
}
