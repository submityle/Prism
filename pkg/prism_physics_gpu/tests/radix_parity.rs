//! Real-device parity tests for the `GPU` `LSD` radix sort.
//!
//! Each test sorts on the device and asserts bit-for-bit equality with the
//! `CPU` golden twins: sorting `u32` keys (and a `u32` payload) is a pure
//! integer permutation, identical on host and device, so parity is exact rather
//! than within a tolerance. Every test skips cleanly when no adapter is
//! available (for example inside a sandbox) so the suite never fails for lack of
//! a `GPU`.
//!
//! Provenance: exercises Prism's own count and scatter kernels against their
//! `CPU` twins; no Unreal Engine source or derived code.

use std::time::Instant;

use prism_physics_gpu::radix::config::TILE;
use prism_physics_gpu::{cpu_radix_sort_keys, cpu_radix_sort_pairs, GpuContext, GpuRadixSort};

/// A small xorshift generator so the tests are deterministic without pulling in
/// an `RNG` dependency.
struct Rng {
    /// Mutable generator state; never zero.
    state: u64,
}

impl Rng {
    /// Seeds the generator, forcing a non-zero state.
    fn new(seed: u64) -> Rng {
        Rng { state: seed | 1 }
    }

    /// Advances the state and returns the next 64-bit value.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Returns a full-range `u32`.
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Returns a `u32` bounded below `n` (assumes `n > 0`).
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }
}

/// Asserts a `GPU` key sort of `keys` equals the twin, bit-for-bit.
fn assert_keys_match(sort: &GpuRadixSort, ctx: &GpuContext, keys: &[u32]) {
    let want = cpu_radix_sort_keys(keys);
    let got = sort.sort_keys(ctx, keys);
    assert_eq!(got, want, "key sort mismatch for len {}", keys.len());
}

#[test]
fn random_keys_sort_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sort = GpuRadixSort::new(&ctx);
    let mut rng = Rng::new(0xC0FF_EE55);
    let keys: Vec<u32> = (0..5000).map(|_| rng.next_u32()).collect();
    assert_keys_match(&sort, &ctx, &keys);
}

#[test]
fn degenerate_sizes_sort_correctly() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sort = GpuRadixSort::new(&ctx);

    // Empty stays on the host and never touches the device.
    assert_eq!(sort.sort_keys(&ctx, &[]), Vec::<u32>::new());

    let tile = TILE as usize;
    for len in [1usize, tile - 1, tile, tile + 1] {
        let mut rng = Rng::new(0x0051_E1D0 + len as u64);
        let keys: Vec<u32> = (0..len).map(|_| rng.below(1000)).collect();
        assert_keys_match(&sort, &ctx, &keys);
    }
}

#[test]
fn boundary_key_values_sort() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sort = GpuRadixSort::new(&ctx);
    let keys = vec![
        u32::MAX,
        0,
        1,
        u32::MAX - 1,
        0x00FF_00FF,
        0xFF00_FF00,
        0x8000_0000,
        0x7FFF_FFFF,
    ];
    assert_keys_match(&sort, &ctx, &keys);
}

#[test]
fn many_equal_keys_sort_correctly() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sort = GpuRadixSort::new(&ctx);
    // Heavy digit collisions stress the per-block stable ranking.
    let mut rng = Rng::new(0x1357_9BDF);
    let keys: Vec<u32> = (0..4000).map(|_| rng.below(4)).collect();
    assert_keys_match(&sort, &ctx, &keys);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "reports the large-input timing to the test log"
)]
fn million_key_sort_is_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sort = GpuRadixSort::new(&ctx);
    let mut rng = Rng::new(0x1234_5678_9ABC);
    let keys: Vec<u32> = (0..1_000_000).map(|_| rng.next_u32()).collect();

    let started = Instant::now();
    let got = sort.sort_keys(&ctx, &keys);
    let elapsed = started.elapsed();

    let want = cpu_radix_sort_keys(&keys);
    assert_eq!(got, want, "million-key sort mismatch");
    eprintln!("1M key radix sort: {elapsed:?}");
}

#[test]
fn random_pairs_sort_stably() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sort = GpuRadixSort::new(&ctx);
    let mut rng = Rng::new(0xDEAD_BEEF);
    // Small key range forces frequent ties; identity payloads reveal any
    // reordering of equal keys.
    let keys: Vec<u32> = (0..6000).map(|_| rng.below(256)).collect();
    let values: Vec<u32> = (0..keys.len() as u32).collect();

    let (got_k, got_v) = sort.sort_pairs(&ctx, &keys, &values);
    let (want_k, want_v) = cpu_radix_sort_pairs(&keys, &values);
    assert_eq!(got_k, want_k, "pair key sort mismatch");
    assert_eq!(got_v, want_v, "pair payload sort mismatch (stability)");
}

#[test]
fn large_pairs_sort_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let sort = GpuRadixSort::new(&ctx);
    let mut rng = Rng::new(0xABCD_1234_5678);
    let keys: Vec<u32> = (0..500_000).map(|_| rng.next_u32()).collect();
    let values: Vec<u32> = (0..keys.len() as u32).collect();

    let (got_k, got_v) = sort.sort_pairs(&ctx, &keys, &values);
    let (want_k, want_v) = cpu_radix_sort_pairs(&keys, &values);
    assert_eq!(got_k, want_k, "large pair key sort mismatch");
    assert_eq!(got_v, want_v, "large pair payload sort mismatch");
}
