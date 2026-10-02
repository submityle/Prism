//! Real-device parity for the segmented-scan twin:
//! [`GpuSegmentedScan`](prism_volumetric_gpu::gpu_scan_segmented::GpuSegmentedScan)
//! must reproduce the `CPU` golden
//! [`gpu_scan_segmented`](prism_render_architecture::particle::gpu_scan_segmented)
//! across an empty input (host short-circuit), a single element, an all-head
//! mask (every element its own singleton segment), an all-zero flag mask
//! (degenerates to a plain scan), a mixed multi-segment mask, a ragged flag
//! array shorter than the values, a wrapping accumulation past `u32::MAX`, a
//! carry that must flow across block boundaries inside one segment, a head
//! landing exactly on a block boundary (severing the carry), and a large
//! pseudo-random sweep over several lengths and block sizes.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every segmented-scan output is a `u32`: a per-segment prefix offset or a
//! per-block carry-in. Parity is therefore asserted with **exact `==`**, not a
//! float tolerance. There is no float math anywhere in the kernel — prefixes are
//! integer additions and the carry fold is a bounded block-local segment scan —
//! so the comparison is bit-exact by construction and has no `ULP`-boundary
//! degenerate region to avoid.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_scan_segmented`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::gpu_scan_segmented::{
    naive_segmented_exclusive, naive_segmented_inclusive, segment_head_count,
    segment_start_indices, segmented_exclusive_scan, segmented_inclusive_scan, SegmentedScanConfig,
};
use prism_volumetric_gpu::gpu_scan_segmented::{
    GpuSegmentedScan, GpuSegmentedScanConfig, GpuSegmentedScanQuery,
};
use prism_volumetric_gpu::GpuContext;

/// A tiny integer linear-congruential generator; only integer arithmetic, so no
/// transcendental appears. Returns the full 32-bit high word of the state.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// Fills a length-`len` value array with draws reduced modulo `modulo`
    /// (`0` means the full 32-bit range).
    fn fill_values(&mut self, len: usize, modulo: u32) -> Vec<u32> {
        (0..len)
            .map(|_| {
                let sample = self.next_u32();
                if modulo == 0 {
                    sample
                } else {
                    sample % modulo
                }
            })
            .collect()
    }

    /// Fills a length-`len` head-flag array: a flag is set roughly once in six
    /// draws, giving varied uneven segments well away from any boundary.
    fn fill_flags(&mut self, len: usize) -> Vec<u32> {
        (0..len)
            .map(|_| u32::from(self.next_u32().is_multiple_of(6)))
            .collect()
    }
}

/// Runs the `GPU` segmented scan for both variants and asserts exact parity
/// against the `CPU` golden: `scanned` matches the golden element for element
/// and `block_carry_ins` matches the golden block for block.
fn check(ctx: &GpuContext, gpu: &GpuSegmentedScan, values: &[u32], flags: &[u32], block_size: u32) {
    let exclusive = gpu.evaluate(
        ctx,
        &GpuSegmentedScanQuery {
            values: values.to_vec(),
            flags: flags.to_vec(),
            block_size,
            inclusive: false,
        },
    );
    let inclusive = gpu.evaluate(
        ctx,
        &GpuSegmentedScanQuery {
            values: values.to_vec(),
            flags: flags.to_vec(),
            block_size,
            inclusive: true,
        },
    );

    let golden_ex = segmented_exclusive_scan(values, flags, block_size as usize);
    let golden_in = segmented_inclusive_scan(values, flags, block_size as usize);

    assert_eq!(
        exclusive.scanned, golden_ex.scanned,
        "exclusive scanned must match golden (block_size {block_size})"
    );
    assert_eq!(
        exclusive.scanned,
        naive_segmented_exclusive(values, flags),
        "exclusive scanned must match the naive oracle (block_size {block_size})"
    );
    assert_eq!(
        exclusive.block_carry_ins, golden_ex.block_carry_ins,
        "exclusive block carry-ins must match golden (block_size {block_size})"
    );
    assert_eq!(
        inclusive.scanned, golden_in.scanned,
        "inclusive scanned must match golden (block_size {block_size})"
    );
    assert_eq!(
        inclusive.scanned,
        naive_segmented_inclusive(values, flags),
        "inclusive scanned must match the naive oracle (block_size {block_size})"
    );
    assert_eq!(
        inclusive.block_carry_ins, golden_in.block_carry_ins,
        "inclusive block carry-ins must match golden (block_size {block_size})"
    );
}

#[test]
fn empty_input_is_degenerate_but_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentedScan::new(&ctx);
    // No element issues no dispatch; both outputs are empty and match golden.
    let result = gpu.evaluate(
        &ctx,
        &GpuSegmentedScanQuery {
            values: Vec::new(),
            flags: Vec::new(),
            block_size: 8,
            inclusive: false,
        },
    );
    assert!(result.scanned.is_empty(), "empty input scans to nothing");
    assert!(
        result.block_carry_ins.is_empty(),
        "empty input has no block carry-ins"
    );
}

#[test]
fn single_element_input() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentedScan::new(&ctx);
    // A single head element scans exclusively to the neutral element and
    // inclusively to its own value.
    check(&ctx, &gpu, &[42], &[1], 4);
    check(&ctx, &gpu, &[42], &[0], 4);
}

#[test]
fn every_element_is_a_head() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentedScan::new(&ctx);
    let values = [7u32, 11, 13, 17, 19, 23, 29];
    let flags = [1u32, 1, 1, 1, 1, 1, 1];
    for &bs in &[1u32, 2, 4, 7, 16] {
        check(&ctx, &gpu, &values, &flags, bs);
    }
}

#[test]
fn all_zero_flags_degenerate_to_plain_scan() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentedScan::new(&ctx);
    let mut lcg = Lcg::new(0x1111_2222_3333_4444);
    let values = lcg.fill_values(40, 100);
    let flags = vec![0u32; values.len()];
    for &bs in &[1u32, 2, 3, 4, 8, 16, 64] {
        check(&ctx, &gpu, &values, &flags, bs);
    }
}

#[test]
fn multiple_segments_reset_independently() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentedScan::new(&ctx);
    // Segments: [1,2,3] [4,5] [6] with heads at 0, 3, 5.
    let values = [1u32, 2, 3, 4, 5, 6];
    let flags = [1u32, 0, 0, 1, 0, 1];
    for &bs in &[1u32, 2, 3, 4, 8] {
        check(&ctx, &gpu, &values, &flags, bs);
    }
}

#[test]
fn first_head_may_be_absent() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentedScan::new(&ctx);
    // No flag at index 0: the array still opens an implicit first segment.
    let values = [3u32, 4, 5, 6];
    let flags = [0u32, 0, 1, 0];
    for &bs in &[1u32, 2, 3, 4] {
        check(&ctx, &gpu, &values, &flags, bs);
    }
}

#[test]
fn ragged_flag_array_reads_missing_as_non_head() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentedScan::new(&ctx);
    // Flags shorter than values: positions past the flag end are not heads.
    let values = [1u32, 2, 3, 4, 5];
    let flags = [1u32, 0];
    for &bs in &[1u32, 2, 3, 4, 5] {
        check(&ctx, &gpu, &values, &flags, bs);
    }
}

#[test]
fn wrapping_accumulation_within_segment() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentedScan::new(&ctx);
    // Values that overflow u32 aggregate; both paths wrap identically.
    let values = vec![u32::MAX; 10];
    let flags = vec![0u32; 10];
    for &bs in &[1u32, 2, 4, 8] {
        check(&ctx, &gpu, &values, &flags, bs);
    }
}

#[test]
fn carry_flows_across_block_boundary_inside_a_segment() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentedScan::new(&ctx);
    // One long segment (no interior heads) spanning many blocks must carry the
    // running sum across every block boundary, matching the plain scan.
    let mut lcg = Lcg::new(0x5151_5151_2626_2626);
    let values = lcg.fill_values(100, 50);
    let mut flags = vec![0u32; values.len()];
    flags[0] = 1;
    for &bs in &[1u32, 2, 3, 8, 16, 32] {
        check(&ctx, &gpu, &values, &flags, bs);
    }
}

#[test]
fn head_at_block_start_severs_the_carry() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentedScan::new(&ctx);
    // Block size 4: a head exactly on the second block's first element must
    // ignore the carry from block 0 and restart at the neutral element.
    let values = [10u32, 20, 30, 40, 5, 6, 7, 8];
    let mut flags = vec![0u32; values.len()];
    flags[0] = 1;
    flags[4] = 1;
    check(&ctx, &gpu, &values, &flags, 4);
}

#[test]
fn matches_golden_on_random_arrays_across_block_sizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSegmentedScan::new(&ctx);
    for &len in &[5usize, 7, 13, 31, 33, 100, 257, 512] {
        let mut lcg = Lcg::new(0xdead_beef_0000_0001 ^ (len as u64));
        let values = lcg.fill_values(len, 1000);
        let flags = lcg.fill_flags(len);
        for &bs in &[1u32, 2, 4, 5, 8, 16, 32, 64] {
            check(&ctx, &gpu, &values, &flags, bs);
        }
    }
}

#[test]
fn config_helpers_match_golden() {
    // Pure-CPU config parity: no GPU adapter required.
    let gpu = GpuSegmentedScanConfig::new(0);
    assert_eq!(gpu.block_size, 1, "zero block size clamps to one");
    assert!(gpu.is_power_of_two_block());

    for &bs in &[1u32, 2, 7, 8, 16] {
        let gpu = GpuSegmentedScanConfig::new(bs);
        let golden = SegmentedScanConfig::new(bs as usize);
        assert_eq!(gpu.is_power_of_two_block(), golden.is_power_of_two_block());
        for &len in &[0usize, 1, 9, 10, 17, 100] {
            assert_eq!(
                gpu.num_blocks(len),
                golden.num_blocks(len),
                "len {len} bs {bs}"
            );
            assert_eq!(gpu.scanned_bytes(len), golden.scanned_bytes(len));
            assert_eq!(gpu.flag_bytes(len), golden.flag_bytes(len));
            assert_eq!(gpu.block_carry_bytes(len), golden.block_carry_bytes(len));
        }
    }
}

#[test]
fn segment_start_indices_and_head_count_cross_checked() {
    // Pure-CPU golden sanity used to shape fixtures: no GPU adapter required.
    let flags = [1u32, 0, 0, 1, 0, 1, 0];
    assert_eq!(segment_start_indices(&flags), vec![0, 3, 5]);
    assert_eq!(segment_head_count(&flags), 3);
}
