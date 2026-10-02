//! Real-device parity for the particle §9 dispatch-contract twin:
//! [`GpuDispatch`](prism_volumetric_gpu::GpuDispatch) must reproduce the `CPU`
//! golden
//! [`gpu_dispatch`](prism_render_architecture::particle::gpu_dispatch)
//! classification and ceil-division — the pass `order`, 1-D `workgroup_size`,
//! direct/indirect flag and `workgroup_count` — element for element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every lane is pure unsigned integer arithmetic with no reordering, so `CPU`
//! and `GPU` are **bit-exact**: the comparison is a precise `==` with no
//! tolerance on all four lanes.
//!
//! # Fixtures
//!
//! The sweep covers all nine pass codes and, for each, an `element_count` of
//! `0` (empty domain), an exact multiple of the pass workgroup size, a
//! one-past-a-multiple remainder, a one-below-a-multiple remainder and a large
//! value (kept below `2^31` so the kernel's `(ec + ws - 1)` ceiling never
//! wraps). A separate host-side case pins the degenerate zero-workgroup-size
//! guard the kernel mirrors.
//!
//! Provenance: twinned from this repository's
//! `prism_render_architecture::particle::gpu_dispatch`; no third-party engine
//! source or derived code.

use prism_render_architecture::particle::gpu_dispatch::{workgroup_count, ParticleComputePass};
use prism_volumetric_gpu::GpuContext;
use prism_volumetric_gpu::{GpuDispatch, GpuDispatchQuery, GpuDispatchResult};

/// The golden expectation for one query, built straight from the `CPU` golden
/// methods and function, so the parity assertion pins the twin lane for lane.
fn expected(query: &GpuDispatchQuery) -> GpuDispatchResult {
    let pass = ParticleComputePass::ALL[query.pass_code as usize];
    let workgroup_size = pass.workgroup_size();
    GpuDispatchResult {
        order: pass.order(),
        workgroup_size,
        is_indirect: u32::from(pass.is_indirect()),
        workgroup_count: workgroup_count(query.element_count, workgroup_size),
    }
}

/// Builds the shared fixture: for every pass code, a spread of element counts
/// that exercises the empty domain, exact multiples, both remainder sides and a
/// large value. Element counts stay below `2^31` so the kernel's open-coded
/// ceiling never overflows.
fn fixture_queries() -> Vec<GpuDispatchQuery> {
    let mut queries = Vec::new();
    for pass in ParticleComputePass::ALL {
        let pass_code = pass.order();
        let ws = pass.workgroup_size();
        let counts = [
            0,             // empty domain -> 0 groups
            ws,            // exactly one group
            ws * 4,        // exact multiple
            ws * 4 + 1,    // one past a multiple -> rounds up
            ws * 7 - 1,    // one below a multiple -> rounds up
            1,             // single element -> one group
            1_000_000,     // mid-size
            2_000_000_000, // large, still < 2^31
        ];
        for element_count in counts {
            queries.push(GpuDispatchQuery {
                pass_code,
                element_count,
            });
        }
    }
    queries
}

#[test]
fn dispatch_matches_reference_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDispatch::new(&ctx);
    let queries = fixture_queries();
    let got = gpu.eval(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, g) in queries.iter().zip(got.iter()) {
        // Every lane is an integer or classification code: exact equality.
        assert_eq!(
            *g,
            expected(q),
            "dispatch mismatch for pass_code {} element_count {}",
            q.pass_code,
            q.element_count
        );
    }
}

#[test]
fn dispatch_covers_every_pass_variant() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDispatch::new(&ctx);
    // One query per pass code with a fixed element count, asserting the exact
    // order / size / indirect classification each pass must emit.
    let queries: Vec<GpuDispatchQuery> = ParticleComputePass::ALL
        .into_iter()
        .map(|pass| GpuDispatchQuery {
            pass_code: pass.order(),
            element_count: 100,
        })
        .collect();
    let got = gpu.eval(&ctx, &queries);
    assert_eq!(got.len(), 9, "all nine passes present");
    for (pass, g) in ParticleComputePass::ALL.into_iter().zip(got.iter()) {
        assert_eq!(g.order, pass.order());
        assert_eq!(g.workgroup_size, pass.workgroup_size());
        assert_eq!(g.is_indirect, u32::from(pass.is_indirect()));
        assert_eq!(
            g.workgroup_count,
            workgroup_count(100, pass.workgroup_size())
        );
    }
    // Sort is the only wider pass and is indirect; FillDrawArgs is narrow and
    // direct, so the batch is not trivially uniform.
    let sort = got[ParticleComputePass::Sort.order() as usize];
    assert_eq!(sort.workgroup_size, 256);
    assert_eq!(sort.is_indirect, 1);
    let fill = got[ParticleComputePass::FillDrawArgs.order() as usize];
    assert_eq!(fill.workgroup_size, 64);
    assert_eq!(fill.is_indirect, 0);
}

#[test]
fn dispatch_workgroup_count_known_values() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDispatch::new(&ctx);
    // Simulate (indirect, ws 64) and Sort (indirect, ws 256) exercise both
    // group sizes with empty, exact and remainder counts.
    let sim = ParticleComputePass::Simulate.order();
    let sort = ParticleComputePass::Sort.order();
    let queries = vec![
        GpuDispatchQuery {
            pass_code: sim,
            element_count: 0,
        },
        GpuDispatchQuery {
            pass_code: sim,
            element_count: 64,
        },
        GpuDispatchQuery {
            pass_code: sim,
            element_count: 65,
        },
        GpuDispatchQuery {
            pass_code: sort,
            element_count: 256,
        },
        GpuDispatchQuery {
            pass_code: sort,
            element_count: 257,
        },
    ];
    let got = gpu.eval(&ctx, &queries);
    let counts: Vec<u32> = got.iter().map(|r| r.workgroup_count).collect();
    assert_eq!(counts, vec![0, 1, 2, 1, 2]);
}

#[test]
fn dispatch_empty_input_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDispatch::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty());
}

#[test]
fn workgroup_count_zero_size_is_degenerate_zero() {
    // The kernel's ceiling guard mirrors the golden `workgroup_count`: a zero
    // workgroup size yields `0` rather than dividing by zero. The pass-derived
    // sizes are always `64` or `256`, so this degenerate input is pinned
    // directly against the golden function (the only lane a pass code cannot
    // reach on the device).
    assert_eq!(workgroup_count(1000, 0), 0);
    assert_eq!(workgroup_count(0, 0), 0);
}
