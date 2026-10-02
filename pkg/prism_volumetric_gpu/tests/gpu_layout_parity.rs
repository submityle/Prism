//! Real-device parity for the `std430` byte-layout twin:
//! [`GpuLayout`](prism_volumetric_gpu::gpu_layout::GpuLayout) must reproduce the
//! `CPU` golden
//! [`particle::gpu_layout`](prism_render_architecture::particle::gpu_layout)
//! across its two per-query answers — the clamped total byte size
//! [`storage_bytes`](prism_render_architecture::particle::gpu_layout::storage_bytes)
//! and the writability flag
//! [`ParticleBufferAccess::is_writable`](prism_render_architecture::particle::gpu_layout::ParticleBufferAccess::is_writable)
//! — over an exhaustive cross-product of the `std430` strides, a spread of
//! element counts (including the empty pool) and every access variant, plus a
//! large random batch spanning many workgroups.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every value is a `u32` byte size or a discrete `1`/`0` writability flag with
//! no rounding anywhere on the path, so the outputs are bit-identical and
//! asserted with exact `==` and no tolerance. Fixtures keep `stride * count`
//! well below `2^31`, so the device `u32` multiply never wraps where the golden
//! `saturating_mul` would otherwise clamp. Several scenarios additionally assert
//! a non-trivial mix of writable and read-only flags and a spread of byte
//! sizes, so a degenerate constant kernel could not pass.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_layout`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::gpu_layout::{
    storage_bytes, ParticleBufferAccess, U32_STRIDE, VEC2_STRIDE, VEC4_STRIDE,
};
use prism_volumetric_gpu::gpu_layout::{GpuLayout, GpuLayoutQuery, GpuLayoutResult};
use prism_volumetric_gpu::GpuContext;

/// Every [`ParticleBufferAccess`] variant in the golden enum's declaration
/// order; the index into this array is the `access_code` the twin consumes.
const ALL_ACCESS: [ParticleBufferAccess; 2] =
    [ParticleBufferAccess::Read, ParticleBufferAccess::ReadWrite];

/// Decodes an `access_code` back into its [`ParticleBufferAccess`] variant so
/// the host can call the golden predicate on it.
fn access_from_code(code: u32) -> ParticleBufferAccess {
    ALL_ACCESS[code as usize]
}

/// Computes the golden answer for one query by calling the `CPU` reference
/// directly, matching exactly what the twin must reproduce.
fn golden_result(q: GpuLayoutQuery) -> GpuLayoutResult {
    let bytes = storage_bytes(q.stride as usize, q.count as usize) as u32;
    let writable = u32::from(access_from_code(q.access_code).is_writable());
    GpuLayoutResult {
        storage_bytes: bytes,
        is_writable: writable,
    }
}

/// Runs the twin over `queries` and asserts per-element parity against the
/// golden, returning the device results for extra assertions.
fn check(ctx: &GpuContext, gpu: &GpuLayout, queries: &[GpuLayoutQuery]) -> Vec<GpuLayoutResult> {
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

/// Draws a random in-contract query: a `std430` stride, a moderate element
/// count and an access code covering both variants. The product stays well
/// below `2^31`, so neither the device multiply nor the golden `saturating_mul`
/// clamps.
fn draw_query(state: &mut u64) -> GpuLayoutQuery {
    let strides = [U32_STRIDE as u32, VEC2_STRIDE as u32, VEC4_STRIDE as u32];
    let stride = strides[(lcg(state) >> 40) as usize % strides.len()];
    // Count in [0, 2^18): with a max stride of 16 the product is below 2^22.
    let count = ((lcg(state) >> 32) as u32) & ((1u32 << 18) - 1);
    let access_code = ((lcg(state) >> 48) as u32) % (ALL_ACCESS.len() as u32);
    GpuLayoutQuery {
        stride,
        count,
        access_code,
    }
}

/// Builds the exhaustive cross-product of the `std430` strides, a spread of
/// element counts (including the empty pool) and every access variant.
fn structured_queries() -> Vec<GpuLayoutQuery> {
    let strides = [U32_STRIDE as u32, VEC2_STRIDE as u32, VEC4_STRIDE as u32];
    let counts = [0u32, 1, 2, 7, 10, 64, 255, 256, 1024, 4096, 65_536];
    let mut queries = Vec::new();
    for &stride in &strides {
        for &count in &counts {
            for access_code in 0u32..ALL_ACCESS.len() as u32 {
                queries.push(GpuLayoutQuery {
                    stride,
                    count,
                    access_code,
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
        eprintln!("skipping gpu-layout parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuLayout::new(&ctx);
    let queries = structured_queries();
    let results = check(&ctx, &gpu, &queries);

    // Non-trivial: both writability classes appear, and the byte sizes span a
    // range, so a degenerate constant kernel could not pass.
    let writable = results.iter().filter(|r| r.is_writable == 1).count();
    let read_only = results.iter().filter(|r| r.is_writable == 0).count();
    assert!(writable > 0, "fixture must include writable buffers");
    assert!(read_only > 0, "fixture must include read-only buffers");
    let min_bytes = results.iter().map(|r| r.storage_bytes).min().unwrap_or(0);
    let max_bytes = results.iter().map(|r| r.storage_bytes).max().unwrap_or(0);
    assert!(
        min_bytes < max_bytes,
        "fixture should span a range of byte sizes, got [{min_bytes}, {max_bytes}]"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_empty_pool_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping gpu-layout parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuLayout::new(&ctx);

    // A zero-count pool still reserves one element, so storage_bytes == stride.
    let queries: Vec<GpuLayoutQuery> = [U32_STRIDE, VEC2_STRIDE, VEC4_STRIDE]
        .into_iter()
        .map(|stride| GpuLayoutQuery {
            stride: stride as u32,
            count: 0,
            access_code: 0,
        })
        .collect();
    let results = check(&ctx, &gpu, &queries);
    for (q, r) in queries.iter().zip(results.iter()) {
        assert_eq!(
            r.storage_bytes, q.stride,
            "empty pool reserves exactly one element"
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
        eprintln!("skipping gpu-layout parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuLayout::new(&ctx);

    // Several thousand in-contract queries spanning many workgroups.
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let count = 8192usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        queries.push(draw_query(&mut state));
    }
    let results = check(&ctx, &gpu, &queries);

    // The batch mixes both access variants, so a degenerate single-class kernel
    // could not pass.
    let any_writable = results.iter().any(|r| r.is_writable == 1);
    let any_read_only = results.iter().any(|r| r.is_writable == 0);
    assert!(any_writable, "random batch should include writable buffers");
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
        eprintln!("skipping gpu-layout parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuLayout::new(&ctx);
    let results = gpu.evaluate(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield empty output");
}
