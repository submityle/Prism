//! Real-device parity for the stream-compaction twin:
//! [`GpuCompact`](prism_volumetric_gpu::gpu_compact::GpuCompact) must reproduce
//! the `CPU` golden
//! [`gpu_compact`](prism_render_architecture::particle::gpu_compact) across an
//! empty input (host short-circuit), an all-keep mask (identity indices), an
//! all-drop mask (empty compaction), an alternating mask, block-boundary
//! aligned and ragged lengths, non-zero flags that still count as keep, and a
//! large pseudo-random mask over several block sizes.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every compaction output is a `u32`: a `scatter` destination offset or a
//! compacted original index. Parity is therefore asserted with **exact `==`**,
//! not a float tolerance. There is no float math anywhere in the kernel —
//! `scatter` destinations are integer additions and the compaction is a bounded
//! block-local `scan` plus a scatter — so the comparison is bit-exact by
//! construction and has no `ULP`-boundary degenerate region to avoid.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_compact`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::gpu_compact::{
    compact_indices, compacted_count, scatter_offsets,
};
use prism_volumetric_gpu::gpu_compact::{GpuCompact, GpuCompactQuery};
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

    /// Fills a length-`len` keep mask whose bits are the top bit of each draw,
    /// giving a roughly even keep/drop split well away from any boundary.
    fn fill_mask(&mut self, len: usize) -> Vec<u32> {
        (0..len).map(|_| self.next_u32() >> 31).collect()
    }
}

/// Runs the `GPU` compaction and asserts exact parity against the `CPU` golden:
/// every per-element `scatter` destination equals [`scatter_offsets`], and the
/// dense compacted list equals [`compact_indices`].
fn check(ctx: &GpuContext, gpu: &GpuCompact, flags: &[u32], block_size: u32) -> Vec<u32> {
    let query = GpuCompactQuery {
        flags: flags.to_vec(),
        block_size,
    };
    let results = gpu.evaluate(ctx, &query);
    let compacted = gpu.compact(ctx, &query);

    let want_scatter = scatter_offsets(flags, block_size as usize);
    let want_compact = compact_indices(flags, block_size as usize);

    assert_eq!(
        results.len(),
        flags.len(),
        "one result per element (block_size {block_size})"
    );
    for (index, (res, &dst)) in results.iter().zip(want_scatter.iter()).enumerate() {
        let want_keep = u32::from(flags[index] != 0);
        assert_eq!(
            res.keep, want_keep,
            "element {index}: gpu keep {} vs cpu keep {want_keep} (block_size {block_size})",
            res.keep
        );
        assert_eq!(
            res.destination, dst,
            "element {index}: gpu dest {} vs cpu dest {dst} (block_size {block_size})",
            res.destination
        );
    }

    assert_eq!(
        compacted, want_compact,
        "compacted index list must match the golden (block_size {block_size})"
    );
    assert_eq!(
        compacted.len(),
        compacted_count(flags),
        "compacted length equals the survivor count (block_size {block_size})"
    );
    compacted
}

#[test]
fn empty_input_is_degenerate_but_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompact::new(&ctx);
    // No element issues no dispatch; both outputs are empty and match the
    // golden exactly.
    let compacted = check(&ctx, &gpu, &[], 8);
    assert!(compacted.is_empty(), "empty input compacts to nothing");
}

#[test]
fn all_kept_returns_identity_indices() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompact::new(&ctx);
    let flags = [1u32, 1, 1, 1, 1];
    let compacted = check(&ctx, &gpu, &flags, 2);
    assert_eq!(
        compacted,
        [0, 1, 2, 3, 4],
        "every element survives in order"
    );
}

#[test]
fn all_dropped_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompact::new(&ctx);
    let flags = [0u32, 0, 0, 0];
    let compacted = check(&ctx, &gpu, &flags, 2);
    assert!(
        compacted.is_empty(),
        "no survivor means an empty compaction"
    );
}

#[test]
fn alternating_predicate_packs_survivors() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompact::new(&ctx);
    let flags = [1u32, 0, 1, 0, 1, 0, 1];
    let compacted = check(&ctx, &gpu, &flags, 3);
    assert_eq!(compacted, [0, 2, 4, 6], "the even indices survive");
}

#[test]
fn nonzero_flag_counts_as_keep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompact::new(&ctx);
    // Any non-zero flag is a keep, matching the golden `keep_bit`.
    let flags = [5u32, 0, 9, 0, 42];
    let compacted = check(&ctx, &gpu, &flags, 4);
    assert_eq!(compacted, [0, 2, 4], "the non-zero flags survive");
}

#[test]
fn block_boundary_aligned_length() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompact::new(&ctx);
    // 8 elements, block_size 4 -> two full blocks (aligned boundary).
    let flags = [1u32, 1, 0, 1, 0, 1, 1, 0];
    let compacted = check(&ctx, &gpu, &flags, 4);
    assert_eq!(
        compacted,
        [0, 1, 3, 5, 6],
        "survivors pack densely across blocks"
    );
}

#[test]
fn block_boundary_ragged_last_block() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompact::new(&ctx);
    // 10 elements, block_size 4 -> 3 blocks, the last block holds only 2.
    let flags = [1u32, 0, 1, 1, 0, 0, 1, 0, 1, 1];
    let compacted = check(&ctx, &gpu, &flags, 4);
    assert_eq!(
        compacted,
        [0, 2, 3, 6, 8, 9],
        "the ragged last block packs too"
    );
}

#[test]
fn single_block_covers_whole_input_when_block_exceeds_len() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompact::new(&ctx);
    // A block larger than the input is a single block covering everything.
    let flags = [0u32, 1, 1, 0, 1];
    let compacted = check(&ctx, &gpu, &flags, 1024);
    assert_eq!(compacted, [1, 2, 4], "one block compacts the whole input");
}

#[test]
fn block_size_one_is_each_element_its_own_block() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompact::new(&ctx);
    // block_size 1: every element is its own block, so each block-local offset
    // is zero and the whole prefix lives in the host-uploaded bases.
    let flags = [0u32, 1, 1, 0, 1, 1];
    let compacted = check(&ctx, &gpu, &flags, 1);
    assert_eq!(compacted, [1, 2, 4, 5], "survivors still pack in order");
}

#[test]
fn large_random_mask_over_several_block_sizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCompact::new(&ctx);
    let mut lcg = Lcg::new(0xC0FF_EE11_2233_4455);
    let flags = lcg.fill_mask(4096);
    // A spread of block sizes, each compared element-for-element exactly.
    for block_size in [1u32, 2, 7, 16, 64, 256, 1000] {
        let compacted = check(&ctx, &gpu, &flags, block_size);
        assert_eq!(
            compacted.len(),
            compacted_count(&flags),
            "survivor count is independent of block size (block_size {block_size})"
        );
    }
}
