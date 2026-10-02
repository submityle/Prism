//! Real-device parity for the `GPU` timestamp-query-pool slot-layout twin:
//! [`GpuTimerQueryLayout`] must reproduce the `CPU` golden
//! [`gpu_timer_query`](prism_render_architecture::particle::gpu_timer_query)
//! pure-`u32` slot-layout functions bit for bit across random batches, the
//! odd/even/zero capacity cases, the in-range and out-of-range pair cases, the
//! resolve-buffer size, a single query, and the degenerate empty batch.
//!
//! The tests skip (with a printed notice on the first) when the host has no
//! `wgpu` adapter, so the suite stays green everywhere while still exercising
//! the full dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned operation is exact, non-overflowing integer arithmetic, so the
//! `CPU` and `GPU` agree bit for bit. The parity test therefore asserts a strict
//! `==` on both the value word and the validity flag with no tolerance: any
//! mismatch is a genuine port defect.
//!
//! # Oracle
//!
//! The expected `(value, present)` for each query is the golden
//! [`cpu_reference`](prism_volumetric_gpu::gpu_timer_query::cpu_reference)
//! evaluated on the same capacity and pair, so the twin is pinned directly
//! against the reference
//! [`TimestampQueryPool`](prism_render_architecture::particle::gpu_timer_query::TimestampQueryPool)
//! rather than a re-transcribed copy.
//!
//! # Fixtures
//!
//! The deterministic `u64` `LCG` fixtures draw capacities from several regimes —
//! zero, small even, small odd, and mid-range values — and pair indices from
//! both inside and outside the pair capacity, so the indexing operations
//! exercise both their `Some` and `None` paths. Every capacity stays well inside
//! the restricted resolve-size subset. Integer parity has no boundary fuzz, so
//! no reject-sampling margin is needed. The fixtures use pure integer arithmetic
//! with no external math library and no transcendental method.
//!
//! Provenance: 孪生自本仓
//! `prism_render_architecture::particle::gpu_timer_query` 的纯 `u32` 槽位布局子集
//! 真机 parity；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::gpu_timer_query::{
    cpu_reference, GpuTimerQueryLayout, GpuTimerQueryLayoutQuery, GpuTimerQueryOp,
};
use prism_volumetric_gpu::GpuContext;

/// Every operation the twin supports, in a fixed order so a random batch cycles
/// through all five with each dispatch.
const OP_TABLE: [GpuTimerQueryOp; 5] = [
    GpuTimerQueryOp::QueryCount,
    GpuTimerQueryOp::PairCapacity,
    GpuTimerQueryOp::BeginQueryIndex,
    GpuTimerQueryOp::EndQueryIndex,
    GpuTimerQueryOp::ResolveBufferBytes,
];

/// A tiny deterministic `64`-bit linear-congruential generator so the "random"
/// fixtures are reproducible run to run without pulling in an external crate.
/// The constants are the common `PCG`/`MMIX` `LCG` multiplier and increment.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }

    /// Advances the generator and returns the next raw word.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// A reproducible pool capacity drawn from one of several regimes: zero, a
    /// small even value, a small odd value, and a mid-range value. Every value
    /// stays well inside the restricted resolve-size subset.
    fn next_capacity(&mut self) -> u32 {
        let regime = self.next_u64() % 4;
        match regime {
            0 => 0,
            1 => ((self.next_u64() % 64) as u32) * 2,
            2 => ((self.next_u64() % 64) as u32) * 2 + 1,
            _ => (self.next_u64() % 4_096) as u32,
        }
    }

    /// A reproducible pair index that may fall inside or outside the pair
    /// capacity, so the indexing operations exercise both verdicts.
    fn next_pair(&mut self, capacity: u32) -> u32 {
        let span = capacity + 4;
        (self.next_u64() % u64::from(span)) as u32
    }
}

/// Runs one batch on the device and asserts every lane matches the golden
/// reference exactly in both the value word and the validity flag.
fn check_batch(
    label: &str,
    engine: &GpuTimerQueryLayout,
    ctx: &GpuContext,
    queries: &[GpuTimerQueryLayoutQuery],
) {
    let gpu = engine.run(ctx, queries);
    assert_eq!(gpu.len(), queries.len(), "{label}: result count");
    for (i, q) in queries.iter().enumerate() {
        let (value, present) = cpu_reference(q);
        assert_eq!(
            gpu[i].present,
            present,
            "{label}[{i}] op {op:?} capacity={cap} pair={pair}: present cpu {present}, gpu {g}",
            op = q.op,
            cap = q.capacity,
            pair = q.pair,
            g = gpu[i].present,
        );
        // The value word is only meaningful when the lane is present; the golden
        // reports zero for an absent lane and the kernel does the same.
        assert_eq!(
            gpu[i].value,
            value,
            "{label}[{i}] op {op:?} capacity={cap} pair={pair}: value cpu {value}, gpu {g}",
            op = q.op,
            cap = q.capacity,
            pair = q.pair,
            g = gpu[i].value,
        );
        assert_eq!(
            gpu[i].as_option(),
            if present { Some(value) } else { None },
            "{label}[{i}]: Option round trip",
        );
    }
}

/// Builds a reproducible batch of `count` queries, cycling through all five
/// operations and drawing each capacity and pair from the mixed-regime
/// generator.
fn random_batch(rng: &mut Lcg, count: usize) -> Vec<GpuTimerQueryLayoutQuery> {
    let mut queries = Vec::with_capacity(count);
    for i in 0..count {
        let op = OP_TABLE[i % OP_TABLE.len()];
        let capacity = rng.next_capacity();
        let pair = rng.next_pair(capacity);
        queries.push(GpuTimerQueryLayoutQuery { capacity, pair, op });
    }
    queries
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_random_batches() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping gpu-timer-query parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuTimerQueryLayout::new(&ctx);
    for (seed, count) in [(1usize, 70usize), (2, 128), (3, 257), (4, 64)] {
        let mut rng = Lcg::new(0x7117_0000_u64.wrapping_add(seed as u64));
        let queries = random_batch(&mut rng, count);
        check_batch(
            &format!("random batch {seed} (count={count})"),
            &engine,
            &ctx,
            &queries,
        );
    }
}

#[test]
fn gpu_matches_cpu_on_capacity_parity_cases() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuTimerQueryLayout::new(&ctx);
    // Zero capacity (no pairs), odd capacity (trailing slot dropped), and even
    // capacity, each probed for count, pair capacity and the resolve-buffer
    // size.
    let mut queries = Vec::new();
    for capacity in [0u32, 1, 2, 3, 6, 7, 8, 255, 256] {
        for op in [
            GpuTimerQueryOp::QueryCount,
            GpuTimerQueryOp::PairCapacity,
            GpuTimerQueryOp::ResolveBufferBytes,
        ] {
            queries.push(GpuTimerQueryLayoutQuery {
                capacity,
                pair: 0,
                op,
            });
        }
    }
    check_batch("capacity parity cases", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_on_pair_index_in_and_out_of_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuTimerQueryLayout::new(&ctx);
    // Capacity 6 => pair capacity 3: pairs 0,1,2 resolve, pair 3 and u32::MAX are
    // out of range and must report None. Capacity 7 drops its trailing slot, so
    // pair 3 is still out of range.
    let mut queries = Vec::new();
    for capacity in [0u32, 6, 7] {
        for pair in [0u32, 1, 2, 3, u32::MAX] {
            queries.push(GpuTimerQueryLayoutQuery {
                capacity,
                pair,
                op: GpuTimerQueryOp::BeginQueryIndex,
            });
            queries.push(GpuTimerQueryLayoutQuery {
                capacity,
                pair,
                op: GpuTimerQueryOp::EndQueryIndex,
            });
        }
    }
    check_batch("pair index range", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_on_single_query() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuTimerQueryLayout::new(&ctx);
    let queries = vec![GpuTimerQueryLayoutQuery {
        capacity: 8,
        pair: 2,
        op: GpuTimerQueryOp::EndQueryIndex,
    }];
    check_batch("single query", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_on_empty_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuTimerQueryLayout::new(&ctx);
    let out = engine.run(&ctx, &[]);
    assert!(
        out.is_empty(),
        "empty batch must short-circuit to an empty vec"
    );
}
