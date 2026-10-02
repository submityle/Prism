//! Real-device parity for the event-scatter bind-group twin:
//! [`GpuEventPassBuffers`](prism_volumetric_gpu::event_pass_buffers::GpuEventPassBuffers)
//! must reproduce the `CPU` golden
//! [`particle::event_pass_buffers`](prism_render_architecture::particle::event_pass_buffers)
//! across all six per-buffer answers — `binding`, `stride`, `access`,
//! `is_output`, `element_count` and `byte_size` — for the single
//! [`EventScatterBuffer`](prism_render_architecture::particle::event_pass_buffers::EventScatterBuffer)
//! enum, over an exhaustive cross-product of every variant and a spread of pool
//! extents (including the empty pool), plus a large random batch spanning many
//! workgroups.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every value is a `u32` binding / stride / classification code or a discrete
//! `1`/`0` flag with no rounding anywhere on the path, so the outputs are
//! bit-identical and asserted with exact `==` and no tolerance. Fixtures keep
//! `stride * capacity` well below `2^31`, so the device `u32` multiply never
//! wraps where the golden `saturating_mul` would otherwise clamp. Several
//! scenarios additionally assert a non-trivial mix of writable and read-only
//! flags and a spread of byte sizes, so a degenerate constant kernel could not
//! pass.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::event_pass_buffers`；无第三方引擎源码或衍生代码。

extern crate alloc;

use prism_render_architecture::particle::event_pass_buffers::{
    EventScatterBuffer, ParticleEventExtent,
};
use prism_render_architecture::particle::gpu_layout::ParticleBufferAccess;
use prism_volumetric_gpu::event_pass_buffers::{
    GpuEventPassBufferQuery, GpuEventPassBufferResult, GpuEventPassBuffers,
};
use prism_volumetric_gpu::GpuContext;

/// Number of variants in `EventScatterBuffer::ALL`.
const VARIANT_COUNT: u32 = 4;

/// Encodes a [`ParticleBufferAccess`] variant into the `0`/`1` code the twin
/// consumes, in the golden enum's declaration order.
fn access_code(access: ParticleBufferAccess) -> u32 {
    match access {
        ParticleBufferAccess::Read => 0,
        ParticleBufferAccess::ReadWrite => 1,
    }
}

/// Computes the golden answer for one query by calling the `CPU` reference
/// directly, matching exactly what the twin must reproduce.
fn golden_result(q: GpuEventPassBufferQuery) -> GpuEventPassBufferResult {
    let extent = ParticleEventExtent {
        source_event_capacity: q.source_event_capacity,
        channel_count: q.channel_count,
        scattered_capacity: q.scattered_capacity,
    };
    let buffer = EventScatterBuffer::ALL[q.variant_code as usize];
    GpuEventPassBufferResult {
        binding: buffer.binding(),
        stride: buffer.stride() as u32,
        access_code: access_code(buffer.access()),
        is_output: u32::from(buffer.access().is_writable()),
        element_count: buffer.element_count(&extent),
        byte_size: buffer.byte_size(&extent) as u32,
    }
}

/// Runs the twin over `queries` and asserts per-element parity against the
/// golden, returning the device results for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuEventPassBuffers,
    queries: &[GpuEventPassBufferQuery],
) -> Vec<GpuEventPassBufferResult> {
    let results = gpu.evaluate(ctx, queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (&q, &r) in queries.iter().zip(results.iter()) {
        assert_eq!(r, golden_result(q), "mismatch for query {q:?}");
    }
    results
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns a raw `u64` state word.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Draws a random in-contract query: a valid variant code `0..4` and three
/// moderate extent counts. Each count stays below `2^20`, so with the max
/// stride of `32` the product is below `2^25` and neither the device multiply
/// nor the golden `saturating_mul` clamps.
fn event_query(state: &mut u64) -> GpuEventPassBufferQuery {
    let variant_code = (lcg(state) >> 40) as u32 % VARIANT_COUNT;
    let source_event_capacity = ((lcg(state) >> 32) as u32) & ((1u32 << 20) - 1);
    let channel_count = ((lcg(state) >> 32) as u32) & ((1u32 << 20) - 1);
    let scattered_capacity = ((lcg(state) >> 32) as u32) & ((1u32 << 20) - 1);
    GpuEventPassBufferQuery {
        variant_code,
        source_event_capacity,
        channel_count,
        scattered_capacity,
    }
}

/// Builds the exhaustive cross-product of every variant code and a spread of
/// extent counts (including the empty pool and a large value).
fn structured_queries() -> Vec<GpuEventPassBufferQuery> {
    let caps = [0u32, 1, 2, 7, 64, 255, 256, 1024, 4096, 65_536, 1_000_000];
    let channels = [0u32, 1, 8, 64];
    let mut queries = Vec::new();
    for variant_code in 0u32..VARIANT_COUNT {
        for &cap in &caps {
            for &channel_count in &channels {
                queries.push(GpuEventPassBufferQuery {
                    variant_code,
                    source_event_capacity: cap,
                    channel_count,
                    scattered_capacity: cap,
                });
            }
        }
    }
    queries
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_structured_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping event-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuEventPassBuffers::new(&ctx);
    let queries = structured_queries();
    let results = check(&ctx, &gpu, &queries);

    // Non-trivial: both writability classes appear (EventCounters and
    // ScatteredEvents are outputs; SourceEvents and ChannelOffsets are inputs),
    // and the byte sizes span a range, so a degenerate constant kernel could
    // not pass.
    let writable = results.iter().filter(|r| r.is_output == 1).count();
    let read_only = results.iter().filter(|r| r.is_output == 0).count();
    assert!(writable > 0, "fixture must include an output buffer");
    assert!(read_only > 0, "fixture must include read-only buffers");
    let min_bytes = results.iter().map(|r| r.byte_size).min().unwrap_or(0);
    let max_bytes = results.iter().map(|r| r.byte_size).max().unwrap_or(0);
    assert!(
        min_bytes < max_bytes,
        "fixture should span a range of byte sizes, got [{min_bytes}, {max_bytes}]"
    );
    // Both the per-element strides appear (u32=4 and the 32-byte record).
    let distinct_strides = results
        .iter()
        .map(|r| r.stride)
        .collect::<alloc::collections::BTreeSet<_>>();
    assert!(
        distinct_strides.len() >= 2,
        "fixture should exercise every distinct stride, got {distinct_strides:?}"
    );
    // The extent-driven element counts are not pinned to one: a populated pool
    // yields counts far above a single element, so the kernel cannot be a
    // constant element_count twin.
    let max_elems = results.iter().map(|r| r.element_count).max().unwrap_or(0);
    assert!(
        max_elems > 1,
        "fixture should drive element_count above one, got {max_elems}"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_empty_pool_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping event-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuEventPassBuffers::new(&ctx);

    // A zero extent drives every extent-sized element_count to zero, yet the
    // byte_size still clamps up to one record, so byte_size equals the stride.
    let mut queries = Vec::new();
    for variant_code in 0u32..VARIANT_COUNT {
        queries.push(GpuEventPassBufferQuery {
            variant_code,
            source_event_capacity: 0,
            channel_count: 0,
            scattered_capacity: 0,
        });
    }
    let results = check(&ctx, &gpu, &queries);
    for r in &results {
        assert_eq!(
            r.element_count, 0,
            "empty extent drives element_count to zero (no fixed single-element block)"
        );
        assert_eq!(
            r.byte_size, r.stride,
            "empty pool still reserves exactly one element"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_large_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping event-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuEventPassBuffers::new(&ctx);

    // Several thousand in-contract queries spanning many workgroups.
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let count = 8192usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        queries.push(event_query(&mut state));
    }
    let results = check(&ctx, &gpu, &queries);

    // The batch mixes both output and read-only buffers, so a degenerate
    // single-class kernel could not pass.
    let any_output = results.iter().any(|r| r.is_output == 1);
    let any_read_only = results.iter().any(|r| r.is_output == 0);
    assert!(any_output, "random batch should include an output buffer");
    assert!(
        any_read_only,
        "random batch should include read-only buffers"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping event-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuEventPassBuffers::new(&ctx);
    let results = gpu.evaluate(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield empty output");
}
