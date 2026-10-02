//! Real-device parity for the statistics-readback reduction twin:
//! [`GpuReadbackReduce`](prism_volumetric_gpu::readback_reduce::GpuReadbackReduce)
//! must reproduce the `CPU` golden
//! [`readback`](prism_render_architecture::particle::readback) across an empty
//! partial list (host short-circuit), a single partial (identity), several
//! partials, partials that drive a field to `u32::MAX` saturation, and a record
//! whose cull total exceeds the alive count so `alive_after_cull` clamps to
//! zero.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every counter is a `u32` saturating sum, which is exactly associative, so
//! the `GPU` reduced record equals the golden
//! [`reduce_partials`](prism_render_architecture::particle::readback::reduce_partials)
//! field for field and the derived accessors equal the golden
//! [`StatSnapshot`](prism_render_architecture::particle::readback::StatSnapshot)
//! accessors. Everything is asserted with **exact `==`** and no tolerance;
//! there is no `f32` quantity anywhere in this contract.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::readback`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use prism_render_architecture::particle::readback::{
    reduce_partials, ParticleStatField, StatSnapshot, STAT_FIELD_COUNT,
};
use prism_volumetric_gpu::readback_reduce::{GpuReadbackReduce, GpuReadbackReduceQuery};
use prism_volumetric_gpu::GpuContext;

/// A tiny integer linear-congruential generator; only integer arithmetic, so no
/// transcendental appears.
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
}

/// Runs the reduction on the device and asserts exact parity: the reduced
/// record equals the golden `reduce_partials`, and both derived accessors equal
/// the golden `StatSnapshot` accessors.
fn check(ctx: &GpuContext, gpu: &GpuReadbackReduce, partials: &[[u32; STAT_FIELD_COUNT]]) {
    let query = GpuReadbackReduceQuery {
        partials: partials.to_vec(),
    };
    let result = gpu.evaluate(ctx, &query);

    let golden_reduced = reduce_partials(partials);
    assert_eq!(
        result.reduced, golden_reduced,
        "gpu reduced record vs golden reduce_partials"
    );

    let snapshot = StatSnapshot::from_partials(partials);
    assert_eq!(
        result.total_culled,
        snapshot.total_culled(),
        "gpu total_culled vs golden StatSnapshot::total_culled"
    );
    assert_eq!(
        result.alive_after_cull,
        snapshot.alive_after_cull(),
        "gpu alive_after_cull vs golden StatSnapshot::alive_after_cull"
    );

    // The reduced record is also the field-wise golden read of the snapshot.
    for field in ParticleStatField::ALL {
        assert_eq!(
            result.reduced[field.index()],
            snapshot.get(field),
            "gpu reduced field {field:?} vs golden snapshot read"
        );
    }
}

#[test]
fn empty_partials_short_circuits_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReadbackReduce::new(&ctx);
    check(&ctx, &gpu, &[]);

    // The host short-circuit must also surface the derived accessors as zero.
    let result = gpu.evaluate(
        &ctx,
        &GpuReadbackReduceQuery {
            partials: Vec::new(),
        },
    );
    assert_eq!(result.reduced, [0u32; STAT_FIELD_COUNT]);
    assert_eq!(result.total_culled, 0);
    assert_eq!(result.alive_after_cull, 0);
}

#[test]
fn single_partial_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReadbackReduce::new(&ctx);
    let mut partial = [0u32; STAT_FIELD_COUNT];
    for (i, slot) in partial.iter_mut().enumerate() {
        *slot = i as u32 + 1;
    }
    check(&ctx, &gpu, &[partial]);
}

#[test]
fn multiple_partials_sum_per_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReadbackReduce::new(&ctx);
    // AliveCount high, three cull fields modest so alive_after_cull is positive.
    let a = [1000, 2, 3, 4, 5, 6, 7];
    let b = [10, 20, 30, 40, 50, 60, 70];
    let c = [100, 200, 300, 400, 500, 600, 700];
    check(&ctx, &gpu, &[a, b, c]);
}

#[test]
fn saturates_near_u32_max() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReadbackReduce::new(&ctx);
    let big = [u32::MAX; STAT_FIELD_COUNT];
    let one = [1u32; STAT_FIELD_COUNT];
    // MAX + MAX + 1 would wrap; the saturating fold pins every field at MAX.
    check(&ctx, &gpu, &[big, big, one]);
    // A single near-max partial plus a small one that tips exactly one field.
    let near = [u32::MAX - 2, 0, 0, u32::MAX - 1, 0, 0, 0];
    let tip = [5, 0, 0, 3, 0, 0, 0];
    check(&ctx, &gpu, &[near, tip]);
}

#[test]
fn cull_exceeds_alive_clamps_alive_after_cull_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReadbackReduce::new(&ctx);
    // AliveCount 10 but the three cull counters sum to 60, so alive_after_cull
    // must clamp to zero rather than underflow.
    let partial = [10, 0, 0, 20, 20, 20, 0];
    check(&ctx, &gpu, &[partial]);
    assert_eq!(ParticleStatField::ALL.len(), STAT_FIELD_COUNT);
}

#[test]
fn large_random_partials() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReadbackReduce::new(&ctx);
    let mut rng = Lcg::new(0xdead_c0de_1234_5678);
    let partials: Vec<[u32; STAT_FIELD_COUNT]> = (0..257)
        .map(|_| {
            let mut record = [0u32; STAT_FIELD_COUNT];
            for slot in record.iter_mut() {
                *slot = rng.next_u32() % 1_000_000;
            }
            record
        })
        .collect();
    check(&ctx, &gpu, &partials);
}

#[test]
fn mixed_saturating_and_exact_fields() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuReadbackReduce::new(&ctx);
    // Field 0 saturates, the rest stay exact, so the per-field independence of
    // the one-thread-per-field reduction is exercised in a single record set.
    let a = [u32::MAX, 1, 2, 3, 4, 5, 6];
    let b = [u32::MAX, 10, 20, 30, 40, 50, 60];
    let c = [2, 100, 200, 300, 400, 500, 600];
    check(&ctx, &gpu, &[a, b, c]);
}
