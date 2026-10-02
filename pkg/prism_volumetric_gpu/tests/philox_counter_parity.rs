//! Real-device parity for the `Philox4x32-10` twin:
//! [`GpuPhiloxCounter`](prism_volumetric_gpu::philox_counter::GpuPhiloxCounter)
//! must reproduce the `CPU` golden
//! [`philox_counter`](prism_render_architecture::particle::philox_counter)
//! bit for bit, both for the pure block function
//! [`philox4x32`](prism_render_architecture::particle::philox_counter::philox4x32)
//! and for the counter-walking
//! [`Philox4x32::next_block`](prism_render_architecture::particle::philox_counter::Philox4x32::next_block)
//! semantics.
//!
//! The fixtures cover the hard Random123 reference vectors (the all-zero block
//! and the digits-of-pi block), a large batch of `(ctr, key)` pairs drawn from
//! a host-side `u64` `LCG`, the counter-carry boundary counters (word `0`, word
//! `1` and word `2` saturated so the little-endian increment carries up), and a
//! walked stream whose host-pre-generated counters are fed as queries so the
//! `GPU` reproduces the reference `next_block` sequence element for element. All
//! fixtures are plain integer words drawn from an integer `LCG`, so they stay
//! pure and need no external math library and no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every value is a `u32` output word with no rounding anywhere on the path, so
//! `CPU` and `GPU` must agree exactly: the comparison is an exact `==` on each
//! of the four output words with no tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::philox_counter`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::philox_counter::{philox4x32, Philox4x32};
use prism_volumetric_gpu::philox_counter::{GpuPhiloxCounter, PhiloxCounterQuery};
use prism_volumetric_gpu::GpuContext;

/// A minimal host-side `u64` linear congruential generator used only to draw
/// fixture counters and keys. Pure integer arithmetic, no transcendental math.
struct Lcg {
    /// Current `64`-bit state.
    state: u64,
}

impl Lcg {
    /// Seeds the generator.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the state (Knuth `MMIX` multiplier and increment) and returns it.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Draws the next `u32` from the high bits of the state (better mixed).
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Draws a `128`-bit counter as four little-endian `u32` words.
    fn next_ctr(&mut self) -> [u32; 4] {
        [
            self.next_u32(),
            self.next_u32(),
            self.next_u32(),
            self.next_u32(),
        ]
    }

    /// Draws a `64`-bit key as two `u32` words.
    fn next_key(&mut self) -> [u32; 2] {
        [self.next_u32(), self.next_u32()]
    }
}

/// Asserts the `GPU` block for each query equals the `CPU` golden `philox4x32`
/// exactly, in one batched dispatch.
fn assert_parity(gpu: &GpuPhiloxCounter, ctx: &GpuContext, queries: &[PhiloxCounterQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let expected = philox4x32(q.ctr, q.key);
        assert_eq!(
            r.block, expected,
            "philox4x32 mismatch for ctr {:?} key {:?}: gpu {:?} vs cpu {:?}",
            q.ctr, q.key, r.block, expected
        );
    }
}

#[test]
fn hard_reference_vectors_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPhiloxCounter::new(&ctx);
    // The two canonical Random123 vectors: the all-zero block and the
    // digits-of-pi block. Exact outputs are asserted directly so the kernel is
    // pinned to the published vectors, not merely to the host golden.
    let pi_ctr = [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344];
    let pi_key = [0xa409_3822, 0x299f_31d0];
    let queries = [
        PhiloxCounterQuery {
            ctr: [0, 0, 0, 0],
            key: [0, 0],
        },
        PhiloxCounterQuery {
            ctr: pi_ctr,
            key: pi_key,
        },
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), 2);
    assert_eq!(
        got[0].block,
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
    );
    assert_eq!(
        got[1].block,
        [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
    );
    // And they still agree with the host golden.
    assert_parity(&gpu, &ctx, &queries);
}

#[test]
fn random_lcg_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPhiloxCounter::new(&ctx);
    // A large batch of random (ctr, key) pairs exercises the full multiply-mix
    // path (no branches, so random inputs need no critical-region avoidance)
    // and the one-thread-per-query flattening across more than one workgroup.
    let mut rng = Lcg::new(0x0123_4567_89ab_cdef);
    let queries: Vec<PhiloxCounterQuery> = (0..300)
        .map(|_| PhiloxCounterQuery {
            ctr: rng.next_ctr(),
            key: rng.next_key(),
        })
        .collect();
    assert_parity(&gpu, &ctx, &queries);
}

#[test]
fn high_multiplier_product_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPhiloxCounter::new(&ctx);
    // Counters whose lanes are all-ones drive the mulhilo carry chain to its
    // longest, the exact corner the u32 32x32->64 reconstruction must get
    // right. Several keys are mixed in to vary the XOR lanes.
    let queries = [
        PhiloxCounterQuery {
            ctr: [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF],
            key: [0xFFFF_FFFF, 0xFFFF_FFFF],
        },
        PhiloxCounterQuery {
            ctr: [0xFFFF_FFFF, 0, 0xFFFF_FFFF, 0],
            key: [0, 0],
        },
        PhiloxCounterQuery {
            ctr: [0x8000_0000, 0x8000_0000, 0x8000_0000, 0x8000_0000],
            key: [0x8000_0000, 0x8000_0000],
        },
        PhiloxCounterQuery {
            ctr: [0x0001_0000, 0, 0x0001_0000, 0],
            key: [0, 0],
        },
    ];
    assert_parity(&gpu, &ctx, &queries);
}

#[test]
fn carry_boundary_counters_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPhiloxCounter::new(&ctx);
    // The little-endian counter-increment carry boundaries: the host
    // pre-generates each counter that `next_block` would walk through at a
    // carry, and the kernel reproduces philox4x32 of each exactly.
    let key = [0x1111_2222, 0x3333_4444];
    let queries = [
        PhiloxCounterQuery {
            ctr: [0xFFFF_FFFF, 0, 0, 0],
            key,
        },
        PhiloxCounterQuery {
            ctr: [0, 1, 0, 0],
            key,
        },
        PhiloxCounterQuery {
            ctr: [0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0],
            key,
        },
        PhiloxCounterQuery {
            ctr: [0, 0, 1, 0],
            key,
        },
        PhiloxCounterQuery {
            ctr: [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0],
            key,
        },
        PhiloxCounterQuery {
            ctr: [0, 0, 0, 1],
            key,
        },
    ];
    assert_parity(&gpu, &ctx, &queries);
}

#[test]
fn walked_stream_matches_next_block() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPhiloxCounter::new(&ctx);
    // Drive the reference counter-walking stream on the host, feeding each
    // current counter as a query. The i-th walked block equals philox4x32 of
    // the i-th counter, which the kernel must reproduce element for element.
    let key = [0xDEAD_BEEF, 0x0BAD_F00D];
    let mut stream = Philox4x32::new(key);
    let mut queries = Vec::new();
    let mut expected = Vec::new();
    for _ in 0..128 {
        queries.push(PhiloxCounterQuery {
            ctr: stream.counter(),
            key,
        });
        expected.push(stream.next_block());
    }
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), expected.len());
    for (i, (r, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(r.block, *e, "walked block {i} mismatch");
    }
    // And the queries themselves agree with the pure golden function.
    assert_parity(&gpu, &ctx, &queries);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPhiloxCounter::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
