//! Real-device parity for the `Spawn`/`Emit`-pass bind-group twin:
//! [`GpuSpawnPassBuffers`](prism_volumetric_gpu::spawn_pass_buffers::GpuSpawnPassBuffers)
//! must reproduce the `CPU` golden
//! [`particle::spawn_pass_buffers`](prism_render_architecture::particle::spawn_pass_buffers)
//! across all seven per-buffer answers — `binding`, `stride`, `kind`,
//! `access`, `element_count`, `byte_size` and the pass-level
//! `total_storage_bytes` — for every
//! [`SpawnPassBuffer`](prism_render_architecture::particle::spawn_pass_buffers::SpawnPassBuffer)
//! binding, over an exhaustive cross-product of every variant and a spread of
//! pool extents (including the empty pool), plus a large random batch spanning
//! many workgroups.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every value is a `u32` binding / stride / classification code or a discrete
//! count with no rounding anywhere on the path, so the outputs are bit-identical
//! and asserted with exact `==` and no tolerance. Fixtures keep
//! `stride * capacity` and the summed `total_storage_bytes` well below `2^31`,
//! so the device `u32` multiply and add never wrap where the golden
//! `saturating_mul`/`saturating_add` would otherwise clamp. Several scenarios
//! additionally assert a non-trivial mix of storage and uniform kinds, writable
//! and read-only access, and a spread of byte sizes, so a degenerate constant
//! kernel could not pass.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::spawn_pass_buffers`；无第三方引擎源码或衍生代码。

extern crate alloc;

use prism_render_architecture::particle::gpu_layout::ParticleBufferAccess;
use prism_render_architecture::particle::spawn_pass_buffers::{
    total_storage_bytes, SpawnBindingKind, SpawnPassBuffer, SpawnPassExtent,
};
use prism_volumetric_gpu::spawn_pass_buffers::{
    GpuSpawnPassBuffers, GpuSpawnPassBuffersQuery, GpuSpawnPassBuffersResult,
};
use prism_volumetric_gpu::GpuContext;

/// Number of distinct `Spawn`-pass bindings (the length of `SpawnPassBuffer::ALL`).
const VARIANT_COUNT: u32 = 6;

/// Encodes a [`SpawnBindingKind`] into the `0`/`1` code the twin consumes, in
/// the golden enum's declaration order.
fn kind_code(kind: SpawnBindingKind) -> u32 {
    match kind {
        SpawnBindingKind::Storage => 0,
        SpawnBindingKind::Uniform => 1,
    }
}

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
fn golden_result(q: GpuSpawnPassBuffersQuery) -> GpuSpawnPassBuffersResult {
    let extent = SpawnPassExtent {
        capacity: q.capacity,
        spawn_count: q.spawn_count,
        emitter_count: q.emitter_count,
    };
    let buffer = SpawnPassBuffer::ALL[q.variant_code as usize];
    GpuSpawnPassBuffersResult {
        binding: buffer.binding(),
        stride: buffer.stride() as u32,
        kind_code: kind_code(buffer.kind()),
        access_code: access_code(buffer.access()),
        element_count: buffer.element_count(extent) as u32,
        byte_size: buffer.byte_size(extent) as u32,
        total_storage_bytes: total_storage_bytes(extent) as u32,
    }
}

/// Runs the twin over `queries` and asserts per-element parity against the
/// golden, returning the device results for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuSpawnPassBuffers,
    queries: &[GpuSpawnPassBuffersQuery],
) -> Vec<GpuSpawnPassBuffersResult> {
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

/// Draws a random in-contract query: a valid variant code `0..6`, a moderate
/// pool capacity, spawn count and emitter count. The capacity stays below
/// `2^20`, so with the summed storage strides the `total_storage_bytes` stays
/// below `2^26` and neither the device multiply/add nor the golden
/// `saturating_mul`/`saturating_add` clamps.
fn spawn_query(state: &mut u64) -> GpuSpawnPassBuffersQuery {
    let variant_code = (lcg(state) >> 40) as u32 % VARIANT_COUNT;
    let capacity = ((lcg(state) >> 32) as u32) & ((1u32 << 20) - 1);
    let spawn_count = ((lcg(state) >> 32) as u32) & ((1u32 << 18) - 1);
    let emitter_count = ((lcg(state) >> 40) as u32) & ((1u32 << 8) - 1);
    GpuSpawnPassBuffersQuery {
        variant_code,
        capacity,
        spawn_count,
        emitter_count,
    }
}

/// Builds the exhaustive cross-product of every variant code and a spread of
/// pool extents (including the empty pool and a large value).
fn structured_queries() -> Vec<GpuSpawnPassBuffersQuery> {
    let capacities = [0u32, 1, 2, 7, 64, 255, 256, 1024, 4096, 65_536, 1_000_000];
    let spawn_counts = [0u32, 1, 256, 4096];
    let emitter_counts = [0u32, 1, 3, 42];
    let mut queries = Vec::new();
    for variant_code in 0u32..VARIANT_COUNT {
        for &capacity in &capacities {
            for &spawn_count in &spawn_counts {
                for &emitter_count in &emitter_counts {
                    queries.push(GpuSpawnPassBuffersQuery {
                        variant_code,
                        capacity,
                        spawn_count,
                        emitter_count,
                    });
                }
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
        eprintln!("skipping spawn-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSpawnPassBuffers::new(&ctx);
    let queries = structured_queries();
    let results = check(&ctx, &gpu, &queries);

    // Non-trivial: both binding kinds appear (only SpawnParams is a uniform),
    // both access classes appear, and the byte sizes span a range, so a
    // degenerate constant kernel could not pass.
    let storage = results.iter().filter(|r| r.kind_code == 0).count();
    let uniform = results.iter().filter(|r| r.kind_code == 1).count();
    assert!(storage > 0, "fixture must include storage buffers");
    assert!(uniform > 0, "fixture must include the uniform buffer");
    let writable = results.iter().filter(|r| r.access_code == 1).count();
    let read_only = results.iter().filter(|r| r.access_code == 0).count();
    assert!(writable > 0, "fixture must include writable buffers");
    assert!(read_only > 0, "fixture must include the read-only uniform");
    let min_bytes = results.iter().map(|r| r.byte_size).min().unwrap_or(0);
    let max_bytes = results.iter().map(|r| r.byte_size).max().unwrap_or(0);
    assert!(
        min_bytes < max_bytes,
        "fixture should span a range of byte sizes, got [{min_bytes}, {max_bytes}]"
    );
    // All three distinct strides appear (u32=4, vec4=16, uniform=32).
    let distinct_strides = results
        .iter()
        .map(|r| r.stride)
        .collect::<alloc::collections::BTreeSet<_>>();
    assert!(
        distinct_strides.len() >= 3,
        "fixture should exercise every distinct stride, got {distinct_strides:?}"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_empty_pool_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping spawn-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSpawnPassBuffers::new(&ctx);

    // A zero-everything extent still reserves one element per buffer, so
    // byte_size equals the stride for the extent-driven buffers; the counter
    // block keeps its fixed four-element footprint.
    let mut queries = Vec::new();
    for variant_code in 0u32..VARIANT_COUNT {
        queries.push(GpuSpawnPassBuffersQuery {
            variant_code,
            capacity: 0,
            spawn_count: 0,
            emitter_count: 0,
        });
    }
    let results = check(&ctx, &gpu, &queries);
    for (&q, r) in queries.iter().zip(results.iter()) {
        // The fixed counter block (variant 1) spans four elements; every other
        // buffer collapses to a single clamped element on the empty extent.
        if q.variant_code == 1 {
            assert_eq!(
                r.byte_size,
                r.stride * 4,
                "counters keep the fixed four-element footprint"
            );
        } else {
            assert_eq!(
                r.byte_size, r.stride,
                "empty pool reserves exactly one element"
            );
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_large_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping spawn-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSpawnPassBuffers::new(&ctx);

    // Several thousand in-contract queries spanning many workgroups.
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let count = 8192usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        queries.push(spawn_query(&mut state));
    }
    let results = check(&ctx, &gpu, &queries);

    // The batch mixes both binding kinds, so a degenerate single-class kernel
    // could not pass.
    let any_storage = results.iter().any(|r| r.kind_code == 0);
    let any_uniform = results.iter().any(|r| r.kind_code == 1);
    assert!(any_storage, "random batch should include storage buffers");
    assert!(
        any_uniform,
        "random batch should include the uniform buffer"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping spawn-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSpawnPassBuffers::new(&ctx);
    let results = gpu.evaluate(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield empty output");
}
