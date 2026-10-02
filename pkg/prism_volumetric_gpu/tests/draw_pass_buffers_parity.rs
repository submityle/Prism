//! Real-device parity for the draw-pass bind-group twin:
//! [`GpuDrawPassBuffers`](prism_volumetric_gpu::draw_pass_buffers::GpuDrawPassBuffers)
//! must reproduce the `CPU` golden
//! [`particle::draw_pass_buffers`](prism_render_architecture::particle::draw_pass_buffers)
//! across all six per-buffer answers — `binding`, `stride`, `access`,
//! `is_output`, `element_count` and `byte_size` — for both the
//! [`FillDrawArgsBuffer`](prism_render_architecture::particle::draw_pass_buffers::FillDrawArgsBuffer)
//! and
//! [`RenderDrawBuffer`](prism_render_architecture::particle::draw_pass_buffers::RenderDrawBuffer)
//! passes, over an exhaustive cross-product of every `(which-enum, variant)`
//! pair and a spread of pool extents (including the empty pool), plus a large
//! random batch spanning many workgroups.
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
//! flags, both passes and a spread of byte sizes, so a degenerate constant
//! kernel could not pass.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::draw_pass_buffers`；无第三方引擎源码或衍生代码。

extern crate alloc;

use prism_render_architecture::particle::draw_pass_buffers::{
    FillDrawArgsBuffer, ParticleDrawExtent, RenderDrawBuffer,
};
use prism_render_architecture::particle::gpu_layout::ParticleBufferAccess;
use prism_volumetric_gpu::draw_pass_buffers::{
    GpuDrawPassBufferQuery, GpuDrawPassBufferResult, GpuDrawPassBuffers,
};
use prism_volumetric_gpu::GpuContext;

/// `which_enum` selector for the `FillDrawArgs` pass.
const WHICH_FILL_DRAW_ARGS: u32 = 0;
/// `which_enum` selector for the `RenderDraw` pass.
const WHICH_RENDER_DRAW: u32 = 1;

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
fn golden_result(q: GpuDrawPassBufferQuery) -> GpuDrawPassBufferResult {
    let extent = ParticleDrawExtent {
        capacity: q.capacity,
        live_count: q.live_count,
    };
    match q.which_enum {
        WHICH_FILL_DRAW_ARGS => {
            let buffer = FillDrawArgsBuffer::ALL[q.variant_code as usize];
            GpuDrawPassBufferResult {
                binding: buffer.binding(),
                stride: buffer.stride() as u32,
                access_code: access_code(buffer.access()),
                is_output: u32::from(buffer.is_output()),
                element_count: buffer.element_count(&extent),
                byte_size: buffer.byte_size(&extent) as u32,
            }
        }
        _ => {
            let buffer = RenderDrawBuffer::ALL[q.variant_code as usize];
            GpuDrawPassBufferResult {
                binding: buffer.binding(),
                stride: buffer.stride() as u32,
                access_code: access_code(buffer.access()),
                is_output: u32::from(buffer.is_output()),
                element_count: buffer.element_count(&extent),
                byte_size: buffer.byte_size(&extent) as u32,
            }
        }
    }
}

/// Runs the twin over `queries` and asserts per-element parity against the
/// golden, returning the device results for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuDrawPassBuffers,
    queries: &[GpuDrawPassBufferQuery],
) -> Vec<GpuDrawPassBufferResult> {
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

/// Draws a random in-contract query: a `which-enum` selector, a valid variant
/// code `0..3`, a moderate pool capacity and a moderate live count. The
/// capacity stays below `2^20`, so with the max stride of `64` the product is
/// below `2^26` and neither the device multiply nor the golden `saturating_mul`
/// clamps.
fn draw_query(state: &mut u64) -> GpuDrawPassBufferQuery {
    let which_enum = (lcg(state) >> 40) as u32 % 2;
    let variant_code = (lcg(state) >> 40) as u32 % 3;
    let capacity = ((lcg(state) >> 32) as u32) & ((1u32 << 20) - 1);
    let live_count = ((lcg(state) >> 32) as u32) & ((1u32 << 20) - 1);
    GpuDrawPassBufferQuery {
        which_enum,
        variant_code,
        capacity,
        live_count,
    }
}

/// Builds the exhaustive cross-product of both passes, every variant code and a
/// spread of pool capacities (including the empty pool and a large value) with
/// a couple of live counts.
fn structured_queries() -> Vec<GpuDrawPassBufferQuery> {
    let capacities = [0u32, 1, 2, 7, 64, 255, 256, 1024, 4096, 65_536, 1_000_000];
    let live_counts = [0u32, 1500];
    let mut queries = Vec::new();
    for which_enum in [WHICH_FILL_DRAW_ARGS, WHICH_RENDER_DRAW] {
        for variant_code in 0u32..3 {
            for &capacity in &capacities {
                for &live_count in &live_counts {
                    queries.push(GpuDrawPassBufferQuery {
                        which_enum,
                        variant_code,
                        capacity,
                        live_count,
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
        eprintln!("skipping draw-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuDrawPassBuffers::new(&ctx);
    let queries = structured_queries();
    let results = check(&ctx, &gpu, &queries);

    // Non-trivial: both writability classes appear (only FillDrawArgs::DrawArgs
    // is an output), and the byte sizes span a range, so a degenerate constant
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
    // Both the per-element strides appear (u32=4, instance=64, record=20).
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
        eprintln!("skipping draw-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuDrawPassBuffers::new(&ctx);

    // A zero-capacity pool still reserves one element per buffer, so byte_size
    // equals the stride and element_count-driven arrays collapse to one.
    let mut queries = Vec::new();
    for which_enum in [WHICH_FILL_DRAW_ARGS, WHICH_RENDER_DRAW] {
        for variant_code in 0u32..3 {
            queries.push(GpuDrawPassBufferQuery {
                which_enum,
                variant_code,
                capacity: 0,
                live_count: 0,
            });
        }
    }
    let results = check(&ctx, &gpu, &queries);
    for r in &results {
        assert_eq!(
            r.byte_size, r.stride,
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
        eprintln!("skipping draw-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuDrawPassBuffers::new(&ctx);

    // Several thousand in-contract queries spanning many workgroups.
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let count = 8192usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        queries.push(draw_query(&mut state));
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
        eprintln!("skipping draw-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuDrawPassBuffers::new(&ctx);
    let results = gpu.evaluate(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield empty output");
}
