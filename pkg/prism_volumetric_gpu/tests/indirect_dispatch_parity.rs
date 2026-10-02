//! Real-device parity for the indirect-dispatch argument twin:
//! [`GpuIndirectDispatch`](prism_volumetric_gpu::indirect_dispatch::GpuIndirectDispatch)
//! must reproduce the `CPU` golden
//! [`particle::indirect_dispatch`](prism_render_architecture::particle::indirect_dispatch)
//! across the three workgroup counts `x`, `y`, `z`, the `is_empty` flag and the
//! host-folded `u64` `total_workgroups` product, over a structured spread of
//! (`element_count`, `workgroup_size`) pairs — exact multiples, one-over and
//! one-under the `ceil`-division boundary, sub-group counts, the zero-element
//! degenerate, and the zero-`workgroup_size` guard — plus a large random batch
//! spanning many workgroups.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned value is a `u32` workgroup count or a discrete `1`/`0` flag
//! with no rounding anywhere on the path, so the outputs are bit-identical and
//! asserted with exact `==` and no tolerance. The `total_workgroups` aggregate
//! is folded on the host in `u64` from the same device `u32` words the golden
//! multiplies, so it is likewise exact. Fixtures keep every `element_count` and
//! `workgroup_size` well inside `u32` and cover both the exact-multiple and the
//! remainder `ceil`-division cases, so a degenerate constant kernel could not
//! pass.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_dispatch`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::indirect_dispatch::DispatchIndirectCommand;
use prism_volumetric_gpu::indirect_dispatch::{
    GpuIndirectDispatch, GpuIndirectDispatchQuery, GpuIndirectDispatchResult,
};
use prism_volumetric_gpu::GpuContext;

/// Computes the golden answer for one query by calling the `CPU` reference
/// directly, matching exactly what the twin must reproduce.
fn golden_result(q: GpuIndirectDispatchQuery) -> GpuIndirectDispatchResult {
    let command = DispatchIndirectCommand::from_element_count(q.element_count, q.workgroup_size);
    GpuIndirectDispatchResult {
        x: command.x,
        y: command.y,
        z: command.z,
        is_empty: command.is_empty(),
        total_workgroups: command.total_workgroups(),
    }
}

/// Runs the twin over `queries` and asserts per-element parity against the
/// golden, returning the device results for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuIndirectDispatch,
    queries: &[GpuIndirectDispatchQuery],
) -> Vec<GpuIndirectDispatchResult> {
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

/// Draws a random in-contract query: an `element_count` in `[0, 2^20)` and a
/// non-zero `workgroup_size` in `[1, 1024]`, both well inside `u32` so neither
/// the overflow-safe `ceil`-division nor the `u64` product can wrap and the
/// result stays away from the `ceil`-division tie ambiguity.
fn dispatch_query(state: &mut u64) -> GpuIndirectDispatchQuery {
    let element_count = ((lcg(state) >> 32) as u32) & ((1u32 << 20) - 1);
    let workgroup_size = ((lcg(state) >> 40) as u32) % 1024 + 1;
    GpuIndirectDispatchQuery {
        element_count,
        workgroup_size,
    }
}

/// Builds a structured spread of queries covering the `ceil`-division
/// boundaries (exact multiple, one-over, one-under, sub-group) across several
/// workgroup sizes, plus the zero-element degenerate and the zero-`workgroup_size`
/// guard.
fn structured_queries() -> Vec<GpuIndirectDispatchQuery> {
    let workgroup_sizes = [1u32, 32, 64, 128, 256];
    let mut queries = Vec::new();
    for &workgroup_size in &workgroup_sizes {
        // Zero elements -> no workgroups (an empty launch).
        queries.push(GpuIndirectDispatchQuery::new(0, workgroup_size));
        // One element always needs exactly one group.
        queries.push(GpuIndirectDispatchQuery::new(1, workgroup_size));
        // Exact multiples and the one-over / one-under neighbours around them.
        for multiple in [1u32, 2, 7, 64] {
            let exact = workgroup_size * multiple;
            queries.push(GpuIndirectDispatchQuery::new(exact, workgroup_size));
            queries.push(GpuIndirectDispatchQuery::new(exact + 1, workgroup_size));
            if exact > 0 {
                queries.push(GpuIndirectDispatchQuery::new(exact - 1, workgroup_size));
            }
        }
    }
    // The zero-workgroup-size guard: any element count yields an empty launch.
    for element_count in [0u32, 1, 1_000, 1_000_000] {
        queries.push(GpuIndirectDispatchQuery::new(element_count, 0));
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
        eprintln!("skipping indirect-dispatch parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuIndirectDispatch::new(&ctx);
    let queries = structured_queries();
    let results = check(&ctx, &gpu, &queries);

    // Non-trivial: both empty and non-empty launches appear, and the workgroup
    // counts span a range, so a degenerate constant kernel could not pass.
    let empty = results.iter().filter(|r| r.is_empty).count();
    let non_empty = results.iter().filter(|r| !r.is_empty).count();
    assert!(empty > 0, "fixture must include empty launches");
    assert!(non_empty > 0, "fixture must include non-empty launches");
    let min_x = results.iter().map(|r| r.x).min().unwrap_or(0);
    let max_x = results.iter().map(|r| r.x).max().unwrap_or(0);
    assert!(
        min_x < max_x,
        "fixture should span a range of workgroup counts, got [{min_x}, {max_x}]"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_exact_and_remainder_boundaries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping indirect-dispatch parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuIndirectDispatch::new(&ctx);

    // Deterministic assertions on both sides of the ceil-division tie: 256 at
    // size 64 is an exact multiple (4 groups); 257 and 255 straddle it (5 and 4
    // groups), so exact-divide and remainder paths are each pinned.
    let queries = [
        GpuIndirectDispatchQuery::new(256, 64),
        GpuIndirectDispatchQuery::new(257, 64),
        GpuIndirectDispatchQuery::new(255, 64),
    ];
    let results = check(&ctx, &gpu, &queries);
    assert_eq!(results[0].x, 4, "exact multiple divides cleanly");
    assert_eq!(results[1].x, 5, "one over rounds up");
    assert_eq!(results[2].x, 4, "one under still needs the same group");
    for r in &results {
        assert_eq!((r.y, r.z), (1, 1), "linear launches keep y = z = 1");
        assert!(!r.is_empty, "a non-zero launch is not empty");
        assert_eq!(
            r.total_workgroups,
            u64::from(r.x),
            "total_workgroups folds to x for a 1-D launch"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_degenerate_launches() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping indirect-dispatch parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuIndirectDispatch::new(&ctx);

    // Zero elements and the zero-workgroup-size guard both collapse to an empty
    // (0, 1, 1) launch with a zero workgroup product.
    let queries = [
        GpuIndirectDispatchQuery::new(0, 64),
        GpuIndirectDispatchQuery::new(1_000, 0),
        GpuIndirectDispatchQuery::new(0, 0),
    ];
    let results = check(&ctx, &gpu, &queries);
    for r in &results {
        assert_eq!((r.x, r.y, r.z), (0, 1, 1), "degenerate launch is (0, 1, 1)");
        assert!(r.is_empty, "a zero-x launch reads as empty");
        assert_eq!(r.total_workgroups, 0, "an empty launch does no work");
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_large_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping indirect-dispatch parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuIndirectDispatch::new(&ctx);

    // Several thousand in-contract queries spanning many workgroups.
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let count = 8192usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        queries.push(dispatch_query(&mut state));
    }
    let results = check(&ctx, &gpu, &queries);

    // The batch mixes empty and non-empty launches (element_count can draw 0),
    // so a degenerate single-class kernel could not pass.
    let any_non_empty = results.iter().any(|r| !r.is_empty);
    assert!(
        any_non_empty,
        "random batch should include non-empty launches"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping indirect-dispatch parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuIndirectDispatch::new(&ctx);
    let results = gpu.evaluate(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield empty output");
}
