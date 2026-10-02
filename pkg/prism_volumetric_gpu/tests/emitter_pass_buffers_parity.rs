//! Real-device parity for the emitter-update bind-group twin:
//! [`GpuEmitterPassBuffers`](prism_volumetric_gpu::emitter_pass_buffers::GpuEmitterPassBuffers)
//! must reproduce the `CPU` golden
//! [`particle::emitter_pass_buffers`](prism_render_architecture::particle::emitter_pass_buffers)
//! across all six per-buffer answers — `binding`, `stride`, `access`,
//! `is_output`, `element_count` and `byte_size` — for every
//! [`EmitterUpdateBuffer`](prism_render_architecture::particle::emitter_pass_buffers::EmitterUpdateBuffer)
//! variant, over an exhaustive cross-product of every variant and a spread of
//! frame extents (including the empty extent), plus a large random batch
//! spanning many workgroups.
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
//! `stride * count` well below `2^31`, so the device `u32` multiply never wraps
//! where the golden `saturating_mul` would otherwise clamp. The `element_count`
//! is checked raw (not clamped), so the empty-extent cases assert the exact
//! `0` the extent-sized arrays report versus the fixed `1` the append counter
//! reports. Several scenarios additionally assert a non-trivial mix of writable
//! and read-only flags and a spread of byte sizes, so a degenerate constant
//! kernel could not pass.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::emitter_pass_buffers`；无第三方引擎源码或衍生代码。

extern crate alloc;

use prism_render_architecture::particle::emitter_pass_buffers::{
    EmitterUpdateBuffer, ParticleEmitterExtent,
};
use prism_render_architecture::particle::gpu_layout::ParticleBufferAccess;
use prism_volumetric_gpu::emitter_pass_buffers::{
    GpuEmitterPassBufferQuery, GpuEmitterPassBufferResult, GpuEmitterPassBuffers,
};
use prism_volumetric_gpu::GpuContext;

/// Encodes a [`ParticleBufferAccess`] variant into the `0`/`1` code the twin
/// consumes, in the golden enum's declaration order.
fn access_code(access: ParticleBufferAccess) -> u32 {
    match access {
        ParticleBufferAccess::Read => 0,
        ParticleBufferAccess::ReadWrite => 1,
    }
}

/// Builds an in-contract query from a variant and extent, with the pad word
/// zeroed.
fn make_query(
    variant_code: u32,
    emitter_count: u32,
    spawn_request_capacity: u32,
) -> GpuEmitterPassBufferQuery {
    GpuEmitterPassBufferQuery {
        variant_code,
        emitter_count,
        spawn_request_capacity,
        pad0: 0,
    }
}

/// Computes the golden answer for one query by calling the `CPU` reference
/// directly, matching exactly what the twin must reproduce.
fn golden_result(q: GpuEmitterPassBufferQuery) -> GpuEmitterPassBufferResult {
    let extent = ParticleEmitterExtent {
        emitter_count: q.emitter_count,
        spawn_request_capacity: q.spawn_request_capacity,
    };
    let buffer = EmitterUpdateBuffer::ALL[q.variant_code as usize];
    GpuEmitterPassBufferResult {
        binding: buffer.binding(),
        stride: buffer.stride() as u32,
        access_code: access_code(buffer.access()),
        is_output: u32::from(buffer.access().is_writable()),
        element_count: buffer.element_count(extent) as u32,
        byte_size: buffer.byte_size(extent) as u32,
    }
}

/// Runs the twin over `queries` and asserts per-element parity against the
/// golden, returning the device results for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuEmitterPassBuffers,
    queries: &[GpuEmitterPassBufferQuery],
) -> Vec<GpuEmitterPassBufferResult> {
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

/// Draws a random in-contract query: a valid variant plus an emitter count and
/// spawn-request capacity bounded well below `2^20` so the `u32` multiply never
/// wraps.
fn emitter_query(state: &mut u64) -> GpuEmitterPassBufferQuery {
    let variant_code = (lcg(state) >> 40) as u32 % 4;
    let emitter_count = ((lcg(state) >> 32) as u32) & ((1u32 << 20) - 1);
    let spawn_request_capacity = ((lcg(state) >> 32) as u32) & ((1u32 << 20) - 1);
    make_query(variant_code, emitter_count, spawn_request_capacity)
}

/// Every variant crossed with a spread of extents, including the empty extent.
fn structured_queries() -> Vec<GpuEmitterPassBufferQuery> {
    let emitter_counts = [0u32, 1, 2, 7, 64, 255, 256, 1024, 4096, 65_536, 1_000_000];
    let capacities = [0u32, 1, 3, 64, 1024, 65_536, 1_000_000];
    let mut queries = Vec::new();
    for variant_code in 0u32..4 {
        for &emitter_count in &emitter_counts {
            for &spawn_request_capacity in &capacities {
                queries.push(make_query(
                    variant_code,
                    emitter_count,
                    spawn_request_capacity,
                ));
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
        eprintln!("skipping emitter-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuEmitterPassBuffers::new(&ctx);
    let queries = structured_queries();
    let results = check(&ctx, &gpu, &queries);

    // Non-trivial: both writability classes appear (only EmitterParams is
    // read-only), and the byte sizes span a range, so a degenerate constant
    // kernel could not pass.
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
    // Every distinct per-element stride appears (u32=4, request=16, state=32,
    // params=64).
    let distinct_strides = results
        .iter()
        .map(|r| r.stride)
        .collect::<alloc::collections::BTreeSet<_>>();
    assert!(
        distinct_strides.len() >= 4,
        "fixture should exercise every distinct stride, got {distinct_strides:?}"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_empty_extent_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping emitter-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuEmitterPassBuffers::new(&ctx);

    // A zero-sized extent: byte_size clamps up to one stride for every buffer,
    // but element_count is reported raw — 0 for the extent-sized arrays and the
    // fixed 1 for the append counter.
    let mut queries = Vec::new();
    for variant_code in 0u32..4 {
        queries.push(make_query(variant_code, 0, 0));
    }
    let results = check(&ctx, &gpu, &queries);
    for r in &results {
        assert_eq!(
            r.byte_size, r.stride,
            "empty extent reserves exactly one element for byte_size"
        );
    }
    // The extent-sized arrays report 0 elements; only the counter reports 1.
    assert_eq!(results[0].element_count, 0, "EmitterParams element_count");
    assert_eq!(results[1].element_count, 0, "EmitterState element_count");
    assert_eq!(results[2].element_count, 0, "SpawnRequests element_count");
    assert_eq!(
        results[3].element_count, 1,
        "SpawnRequestCounter is a fixed single-element block"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_counter_fixed_regardless_of_extent() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping emitter-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuEmitterPassBuffers::new(&ctx);

    // The SpawnRequestCounter (variant 3) is a single-element block whatever the
    // extent is, so its element_count and byte_size never scale.
    let queries = [
        make_query(3, 0, 0),
        make_query(3, 1_000_000, 999_999),
        make_query(3, 7, 4096),
    ];
    let results = check(&ctx, &gpu, &queries);
    for r in &results {
        assert_eq!(r.element_count, 1, "counter is always a single element");
        assert_eq!(r.stride, 4, "counter stride is a scalar u32");
        assert_eq!(r.byte_size, 4, "counter byte_size is a single u32");
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_large_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping emitter-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuEmitterPassBuffers::new(&ctx);

    // Several thousand in-contract queries spanning many workgroups.
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let count = 8192usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        queries.push(emitter_query(&mut state));
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
        eprintln!("skipping emitter-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuEmitterPassBuffers::new(&ctx);
    let results = gpu.evaluate(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield empty output");
}
