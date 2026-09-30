//! Real-device parity tests for the `GPU` exclusive scan and stream compaction.
//!
//! Each test scans or compacts on the device and asserts bit-for-bit equality
//! with the `CPU` golden twins: the fold is wrapping `u32` addition, which is
//! associative and identical on host and device, so parity is exact rather than
//! within a tolerance. Every test skips cleanly when no adapter is available
//! (for example inside a sandbox) so the suite never fails for lack of a `GPU`.
//!
//! Provenance: exercises Prism's own scan and scatter kernels against their
//! `CPU` twins; no Unreal Engine source or derived code.

use std::time::Instant;

use prism_physics_gpu::scan::config::BLOCK;
use prism_physics_gpu::{cpu_compact, cpu_exclusive_scan, GpuContext, GpuScan};

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

    /// Returns a `u32` bounded below `n` (assumes `n > 0`).
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % u64::from(n)) as u32
    }
}

/// Asserts a `GPU` exclusive scan of `values` equals the twin, bit-for-bit.
fn assert_scan_matches(scan: &GpuScan, ctx: &GpuContext, values: &[u32]) {
    let (want, want_total) = cpu_exclusive_scan(values);
    let (got, got_total) = scan.exclusive_scan(ctx, values);
    assert_eq!(got, want, "scan mismatch for len {}", values.len());
    assert_eq!(
        got_total,
        want_total,
        "total mismatch for len {}",
        values.len()
    );
}

#[test]
fn random_scan_is_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scan = GpuScan::new(&ctx);
    let mut rng = Rng::new(0xC0FF_EE12);
    let values: Vec<u32> = (0..3000).map(|_| rng.below(1000)).collect();
    assert_scan_matches(&scan, &ctx, &values);
}

#[test]
fn degenerate_sizes_scan_correctly() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scan = GpuScan::new(&ctx);

    // Empty stays on the host and never touches the device.
    assert_eq!(scan.exclusive_scan(&ctx, &[]), (Vec::new(), 0));

    let block = BLOCK as usize;
    for len in [1usize, block - 1, block, block + 1] {
        let mut rng = Rng::new(0x0051_E1D0 + len as u64);
        let values: Vec<u32> = (0..len).map(|_| rng.below(64)).collect();
        assert_scan_matches(&scan, &ctx, &values);
    }
}

#[test]
fn scan_wraps_like_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scan = GpuScan::new(&ctx);
    // Values large enough that the running total wraps past u32::MAX.
    let values = vec![u32::MAX, 3, u32::MAX - 1, 10];
    assert_scan_matches(&scan, &ctx, &values);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "reports the large-input timing to the test log"
)]
fn million_element_scan_is_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scan = GpuScan::new(&ctx);
    let mut rng = Rng::new(0x1234_5678_9ABC);
    let values: Vec<u32> = (0..1_000_000).map(|_| rng.below(4)).collect();

    let started = Instant::now();
    let (got, got_total) = scan.exclusive_scan(&ctx, &values);
    let elapsed = started.elapsed();

    let (want, want_total) = cpu_exclusive_scan(&values);
    assert_eq!(got, want, "million-element scan mismatch");
    assert_eq!(got_total, want_total, "million-element total mismatch");
    eprintln!("1M exclusive scan: {elapsed:?}");
}

#[test]
fn three_level_pyramid_scan_is_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scan = GpuScan::new(&ctx);
    // 300_000 > BLOCK^2 (262_144), forcing a three-scan-level pyramid.
    let mut rng = Rng::new(0xBEEF_F00D);
    let values: Vec<u32> = (0..300_000).map(|_| rng.below(8)).collect();
    assert_scan_matches(&scan, &ctx, &values);
}

#[test]
fn random_compact_matches_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scan = GpuScan::new(&ctx);
    let mut rng = Rng::new(0xDEAD_BEEF);
    let n = 5000usize;
    let data: Vec<u32> = (0..n as u32).collect();
    let flags: Vec<u32> = (0..n).map(|_| rng.below(2)).collect();

    let want = cpu_compact(&data, &flags);
    let got = scan.compact(&ctx, &data, &flags);
    assert_eq!(got, want, "compact mismatch for len {n}");
}

#[test]
fn compact_all_and_none_kept() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scan = GpuScan::new(&ctx);
    let data: Vec<u32> = (0..2000u32).collect();

    let all_ones = vec![1u32; data.len()];
    assert_eq!(
        scan.compact(&ctx, &data, &all_ones),
        cpu_compact(&data, &all_ones),
    );

    let all_zeros = vec![0u32; data.len()];
    assert!(scan.compact(&ctx, &data, &all_zeros).is_empty());

    // Empty stays on the host.
    assert!(scan.compact(&ctx, &[], &[]).is_empty());
}

#[test]
fn large_compact_matches_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scan = GpuScan::new(&ctx);
    let mut rng = Rng::new(0x0BAD_CAFE);
    let n = 600_000usize;
    let data: Vec<u32> = (0..n as u32)
        .map(|v| v.wrapping_mul(2_654_435_761))
        .collect();
    let flags: Vec<u32> = (0..n).map(|_| u32::from(rng.below(3) == 0)).collect();

    let want = cpu_compact(&data, &flags);
    let got = scan.compact(&ctx, &data, &flags);
    assert_eq!(got.len(), want.len(), "compact length mismatch");
    assert_eq!(got, want, "large compact mismatch");
}
