//! Real-device parity for the sort/cull bind-group twin:
//! [`GpuSortPassBuffers`](prism_volumetric_gpu::sort_pass_buffers::GpuSortPassBuffers)
//! must reproduce the `CPU` golden
//! [`particle::sort_pass_buffers`](prism_render_architecture::particle::sort_pass_buffers)
//! across all six per-buffer answers — `binding`, `stride`, `access`,
//! `is_output`, `element_count` and `byte_size` — for every one of the four
//! passes
//! ([`CompactionBuffer`](prism_render_architecture::particle::sort_pass_buffers::CompactionBuffer),
//! [`BoundsBuffer`](prism_render_architecture::particle::sort_pass_buffers::BoundsBuffer),
//! [`CullBuffer`](prism_render_architecture::particle::sort_pass_buffers::CullBuffer)
//! and
//! [`SortBuffer`](prism_render_architecture::particle::sort_pass_buffers::SortBuffer)),
//! over an exhaustive cross-product of every `(which-enum, variant)` pair and a
//! spread of extents (including the empty extent and degenerate zero knobs),
//! plus a large random batch spanning many workgroups.
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
//! every element count and the widest `stride * count` product well below
//! `2^31`, so the device `u32` multiply never wraps where the golden
//! `saturating_mul` would otherwise clamp. Several scenarios additionally
//! assert a non-trivial mix of writable and read-only flags, all four passes
//! and a spread of byte sizes, so a degenerate constant kernel could not pass.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sort_pass_buffers`；无第三方引擎源码或衍生代码。

extern crate alloc;

use prism_render_architecture::particle::gpu_layout::ParticleBufferAccess;
use prism_render_architecture::particle::sort_pass_buffers::{
    BoundsBuffer, CompactionBuffer, CullBuffer, SortBuffer, SortCullExtent,
};
use prism_volumetric_gpu::sort_pass_buffers::{
    GpuSortPassBufferQuery, GpuSortPassBufferResult, GpuSortPassBuffers,
};
use prism_volumetric_gpu::GpuContext;

/// `which_enum` selector for the `Compaction` pass.
const WHICH_COMPACTION: u32 = 0;
/// `which_enum` selector for the `Bounds` pass.
const WHICH_BOUNDS: u32 = 1;
/// `which_enum` selector for the `Cull` pass.
const WHICH_CULL: u32 = 2;
/// `which_enum` selector for the `Sort` pass.
const WHICH_SORT: u32 = 3;

/// Number of variants (and `@binding` slots) each pass's enum declares, indexed
/// by the `which_enum` selector.
fn variant_count(which_enum: u32) -> u32 {
    match which_enum {
        WHICH_COMPACTION => CompactionBuffer::ALL.len() as u32,
        WHICH_BOUNDS => BoundsBuffer::ALL.len() as u32,
        WHICH_CULL => CullBuffer::ALL.len() as u32,
        _ => SortBuffer::ALL.len() as u32,
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
fn golden_result(q: GpuSortPassBufferQuery) -> GpuSortPassBufferResult {
    let extent = SortCullExtent {
        particle_count: q.particle_count,
        candidate_count: q.candidate_count,
        radix_buckets: q.radix_buckets,
        workgroup_count: q.workgroup_count,
    };
    let variant = q.variant_code as usize;
    match q.which_enum {
        WHICH_COMPACTION => {
            let buffer = CompactionBuffer::ALL[variant];
            GpuSortPassBufferResult {
                binding: buffer.binding(),
                stride: buffer.stride() as u32,
                access_code: access_code(buffer.access()),
                is_output: u32::from(buffer.is_output()),
                element_count: buffer.element_count(extent),
                byte_size: buffer.byte_size(extent) as u32,
            }
        }
        WHICH_BOUNDS => {
            let buffer = BoundsBuffer::ALL[variant];
            GpuSortPassBufferResult {
                binding: buffer.binding(),
                stride: buffer.stride() as u32,
                access_code: access_code(buffer.access()),
                is_output: u32::from(buffer.is_output()),
                element_count: buffer.element_count(extent),
                byte_size: buffer.byte_size(extent) as u32,
            }
        }
        WHICH_CULL => {
            let buffer = CullBuffer::ALL[variant];
            GpuSortPassBufferResult {
                binding: buffer.binding(),
                stride: buffer.stride() as u32,
                access_code: access_code(buffer.access()),
                is_output: u32::from(buffer.is_output()),
                element_count: buffer.element_count(extent),
                byte_size: buffer.byte_size(extent) as u32,
            }
        }
        _ => {
            let buffer = SortBuffer::ALL[variant];
            GpuSortPassBufferResult {
                binding: buffer.binding(),
                stride: buffer.stride() as u32,
                access_code: access_code(buffer.access()),
                is_output: u32::from(buffer.is_output()),
                element_count: buffer.element_count(extent),
                byte_size: buffer.byte_size(extent) as u32,
            }
        }
    }
}

/// Runs the twin over `queries` and asserts per-element parity against the
/// golden, returning the device results for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuSortPassBuffers,
    queries: &[GpuSortPassBufferQuery],
) -> Vec<GpuSortPassBufferResult> {
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
/// code for that enum, and four moderate extent counts. The per-particle and
/// per-candidate counts stay below `2^20` and the `radix` / workgroup knobs
/// below `2^9`, so with the widest stride of `128` every product is below
/// `2^28` and neither the device multiply nor the golden `saturating_mul`
/// clamps.
fn sort_query(state: &mut u64) -> GpuSortPassBufferQuery {
    let which_enum = (lcg(state) >> 40) as u32 % 4;
    let variant_code = (lcg(state) >> 40) as u32 % variant_count(which_enum);
    let particle_count = ((lcg(state) >> 32) as u32) & ((1u32 << 20) - 1);
    let candidate_count = ((lcg(state) >> 32) as u32) & ((1u32 << 20) - 1);
    let radix_buckets = ((lcg(state) >> 32) as u32) & ((1u32 << 9) - 1);
    let workgroup_count = ((lcg(state) >> 32) as u32) & ((1u32 << 9) - 1);
    GpuSortPassBufferQuery {
        which_enum,
        variant_code,
        particle_count,
        candidate_count,
        radix_buckets,
        workgroup_count,
    }
}

/// Builds the exhaustive cross-product of all four passes, every variant code
/// and a spread of extents (including the empty extent, degenerate zero knobs
/// and large values).
fn structured_queries() -> Vec<GpuSortPassBufferQuery> {
    let extents = [
        SortCullExtent {
            particle_count: 0,
            candidate_count: 0,
            radix_buckets: 0,
            workgroup_count: 0,
        },
        SortCullExtent {
            particle_count: 1,
            candidate_count: 1,
            radix_buckets: 1,
            workgroup_count: 1,
        },
        SortCullExtent {
            particle_count: 7,
            candidate_count: 3,
            radix_buckets: 16,
            workgroup_count: 2,
        },
        SortCullExtent {
            particle_count: 4096,
            candidate_count: 3000,
            radix_buckets: 256,
            workgroup_count: 16,
        },
        SortCullExtent {
            particle_count: 65_536,
            candidate_count: 50_000,
            radix_buckets: 256,
            workgroup_count: 64,
        },
        SortCullExtent {
            particle_count: 1_000_000,
            candidate_count: 500_000,
            radix_buckets: 16,
            workgroup_count: 8,
        },
    ];
    let mut queries = Vec::new();
    for which_enum in [WHICH_COMPACTION, WHICH_BOUNDS, WHICH_CULL, WHICH_SORT] {
        for variant_code in 0..variant_count(which_enum) {
            for &extent in &extents {
                queries.push(GpuSortPassBufferQuery {
                    which_enum,
                    variant_code,
                    particle_count: extent.particle_count,
                    candidate_count: extent.candidate_count,
                    radix_buckets: extent.radix_buckets,
                    workgroup_count: extent.workgroup_count,
                });
            }
        }
    }
    queries
}

/// Every `(which-enum, variant)` pair, holding the extent fixed, so a scenario
/// can sweep all bindings of all four passes against one extent.
fn all_pairs(extent: SortCullExtent) -> Vec<GpuSortPassBufferQuery> {
    let mut queries = Vec::new();
    for which_enum in [WHICH_COMPACTION, WHICH_BOUNDS, WHICH_CULL, WHICH_SORT] {
        for variant_code in 0..variant_count(which_enum) {
            queries.push(GpuSortPassBufferQuery {
                which_enum,
                variant_code,
                particle_count: extent.particle_count,
                candidate_count: extent.candidate_count,
                radix_buckets: extent.radix_buckets,
                workgroup_count: extent.workgroup_count,
            });
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
        eprintln!("skipping sort-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSortPassBuffers::new(&ctx);
    let queries = structured_queries();
    let results = check(&ctx, &gpu, &queries);

    // Non-trivial: both writability classes appear, and the byte sizes span a
    // range, so a degenerate constant kernel could not pass.
    let writable = results.iter().filter(|r| r.is_output == 1).count();
    let read_only = results.iter().filter(|r| r.is_output == 0).count();
    assert!(writable > 0, "fixture must include output buffers");
    assert!(read_only > 0, "fixture must include read-only buffers");
    let min_bytes = results.iter().map(|r| r.byte_size).min().unwrap_or(0);
    let max_bytes = results.iter().map(|r| r.byte_size).max().unwrap_or(0);
    assert!(
        min_bytes < max_bytes,
        "fixture should span a range of byte sizes, got [{min_bytes}, {max_bytes}]"
    );
    // Every distinct stride appears (u32=4, vec2=8, vec4=16, cull-params=128).
    let distinct_strides = results
        .iter()
        .map(|r| r.stride)
        .collect::<alloc::collections::BTreeSet<_>>();
    assert_eq!(
        distinct_strides,
        [4u32, 8, 16, 128].into_iter().collect(),
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
        eprintln!("skipping sort-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSortPassBuffers::new(&ctx);

    // A zero extent clamps every buffer to its one-element floor, so each
    // byte_size is a whole number of strides and never drops below a single
    // element. Extent-driven arrays (particle/candidate counts) collapse to
    // exactly one element, while knob-driven scratch keeps its fixed shape:
    // the partial-bounds buffer reserves `effective_workgroup_count() * 2`
    // (i.e. two) elements even when the extent is empty.
    let queries = all_pairs(SortCullExtent::default());
    let results = check(&ctx, &gpu, &queries);
    for r in &results {
        assert!(
            r.byte_size >= r.stride,
            "empty extent reserves at least one element"
        );
        assert_eq!(
            r.byte_size % r.stride,
            0,
            "byte_size must be a whole number of strides"
        );
    }
    assert!(
        results.iter().any(|r| r.byte_size == r.stride),
        "empty extent must collapse at least one buffer to a single element"
    );
    assert!(
        results.iter().any(|r| r.byte_size == 2 * r.stride),
        "the partial-bounds buffer keeps its two-element shape under an empty extent"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_degenerate_knobs() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sort-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSortPassBuffers::new(&ctx);

    // Non-empty domains but zero radix/workgroup knobs: the histogram and the
    // partial-bounds scratch must clamp their knob factors up to one.
    let extent = SortCullExtent {
        particle_count: 10,
        candidate_count: 10,
        radix_buckets: 0,
        workgroup_count: 0,
    };
    let queries = all_pairs(extent);
    let results = check(&ctx, &gpu, &queries);

    // Histogram = effective_buckets(1) * effective_workgroups(1) = 1 element.
    let histogram = results[all_histogram_index()];
    assert_eq!(
        histogram.element_count, 1,
        "histogram clamps to one element"
    );
    // Partial bounds = effective_workgroups(1) * 2 = 2 elements.
    let partial = results[all_partial_bounds_index()];
    assert_eq!(
        partial.element_count, 2,
        "partial bounds clamps to two elements"
    );
}

/// Index of the `BoundsBuffer::PartialBounds` query inside [`all_pairs`]:
/// Compaction has three variants, then Bounds variant `1`.
fn all_partial_bounds_index() -> usize {
    CompactionBuffer::ALL.len() + 1
}

/// Index of the `SortBuffer::Histogram` query inside [`all_pairs`]: all of
/// Compaction, Bounds and Cull, then Sort variant `2`.
fn all_histogram_index() -> usize {
    CompactionBuffer::ALL.len() + BoundsBuffer::ALL.len() + CullBuffer::ALL.len() + 2
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_large_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sort-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSortPassBuffers::new(&ctx);

    // Several thousand in-contract queries spanning many workgroups.
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let count = 8192usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        queries.push(sort_query(&mut state));
    }
    let results = check(&ctx, &gpu, &queries);

    // The batch mixes both output and read-only buffers, so a degenerate
    // single-class kernel could not pass.
    let any_output = results.iter().any(|r| r.is_output == 1);
    let any_read_only = results.iter().any(|r| r.is_output == 0);
    assert!(any_output, "random batch should include output buffers");
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
        eprintln!("skipping sort-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSortPassBuffers::new(&ctx);
    let results = gpu.evaluate(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield empty output");
}
