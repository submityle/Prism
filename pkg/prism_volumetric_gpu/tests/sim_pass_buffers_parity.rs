//! Real-device parity for the per-pass buffer-contract twin:
//! [`GpuSimPassBuffers`](prism_volumetric_gpu::sim_pass_buffers::GpuSimPassBuffers)
//! must reproduce the `CPU` golden
//! [`particle::sim_pass_buffers`](prism_render_architecture::particle::sim_pass_buffers)
//! across all nine per-slot answers — `binding`, `stride`, access code,
//! `is_output`, `persists_across_frames`, `aliases_persistent_pool`,
//! `element_count`, `byte_size` and the pass-level `transient_scratch_bytes` —
//! over an exhaustive cross-product of every buffer slot and a spread of
//! `SimPassExtent` values (including the empty extent), plus a large random
//! batch spanning many workgroups.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every value is a `u32` count or a discrete `1`/`0` classification flag with
//! no rounding anywhere on the path, so the outputs are bit-identical and
//! asserted with exact `==` and no tolerance. Fixtures keep each extent count
//! well below `2^18`, so with the widest stride of `64` neither a device `u32`
//! multiply nor the summed transient total wraps where the golden
//! `saturating_mul` would otherwise clamp. Several scenarios additionally assert
//! a non-trivial mix of writable and read-only, persistent and transient slots
//! and a spread of byte sizes, so a degenerate constant kernel could not pass.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sim_pass_buffers`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::gpu_layout::ParticleBufferAccess;
use prism_render_architecture::particle::sim_pass_buffers::{
    transient_scratch_bytes, SimPassBuffer, SimPassExtent,
};
use prism_volumetric_gpu::sim_pass_buffers::{
    GpuSimPassBuffers, GpuSimPassBuffersQuery, GpuSimPassBuffersResult,
};
use prism_volumetric_gpu::GpuContext;

/// Number of buffer slots the `SimulationStages` pass binds (`@binding` `0..8`).
const SLOT_COUNT: u32 = SimPassBuffer::ALL.len() as u32;

/// Encodes a [`ParticleBufferAccess`] into the twin's access code, matching the
/// golden enum's declaration order (`Read` == `0`, `ReadWrite` == `1`).
fn access_code(access: ParticleBufferAccess) -> u32 {
    match access {
        ParticleBufferAccess::Read => 0,
        ParticleBufferAccess::ReadWrite => 1,
    }
}

/// Builds the [`SimPassExtent`] a query carries.
fn extent_of(q: GpuSimPassBuffersQuery) -> SimPassExtent {
    SimPassExtent {
        particle_capacity: q.particle_capacity,
        grid_cell_count: q.grid_cell_count,
        constraint_count: q.constraint_count,
    }
}

/// Computes the golden answer for one query by calling the `CPU` reference
/// directly, matching exactly what the twin must reproduce.
fn golden_result(q: GpuSimPassBuffersQuery) -> GpuSimPassBuffersResult {
    let buffer = SimPassBuffer::ALL[q.slot_code as usize];
    let extent = extent_of(q);
    GpuSimPassBuffersResult {
        binding: buffer.binding(),
        stride: buffer.stride() as u32,
        access_code: access_code(buffer.access()),
        is_output: u32::from(buffer.is_output()),
        persists_across_frames: u32::from(buffer.persists_across_frames()),
        aliases_persistent_pool: u32::from(buffer.aliases_persistent_pool()),
        element_count: buffer.element_count(extent),
        byte_size: buffer.byte_size(extent) as u32,
        transient_scratch_bytes: transient_scratch_bytes(extent) as u32,
    }
}

/// Runs the twin over `queries` and asserts per-element parity against the
/// golden, returning the device results for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuSimPassBuffers,
    queries: &[GpuSimPassBuffersQuery],
) -> Vec<GpuSimPassBuffersResult> {
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

/// Draws a random in-contract query: a random buffer slot and three
/// `SimPassExtent` counts in `[1, 2^18)`. With the widest stride of `64` the
/// per-slot product and the summed transient total both stay below `2^25`, so
/// neither the device multiply nor the golden `saturating_mul` clamps.
fn draw_query(state: &mut u64) -> GpuSimPassBuffersQuery {
    let slot_code = ((lcg(state) >> 40) as u32) % SLOT_COUNT;
    let mask = (1u32 << 18) - 1;
    let particle_capacity = (((lcg(state) >> 20) as u32) & mask) + 1;
    let grid_cell_count = (((lcg(state) >> 20) as u32) & mask) + 1;
    let constraint_count = (((lcg(state) >> 20) as u32) & mask) + 1;
    GpuSimPassBuffersQuery {
        slot_code,
        particle_capacity,
        grid_cell_count,
        constraint_count,
    }
}

/// Builds the exhaustive cross-product of every buffer slot and a spread of
/// `SimPassExtent` values, including the empty extent and the all-ones extent.
fn structured_queries() -> Vec<GpuSimPassBuffersQuery> {
    let extents = [
        (0u32, 0u32, 0u32),
        (1, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1024, 512, 200),
        (4096, 2048, 777),
        (65_536, 32_768, 4_096),
    ];
    let mut queries = Vec::new();
    for slot_code in 0..SLOT_COUNT {
        for &(particle_capacity, grid_cell_count, constraint_count) in &extents {
            queries.push(GpuSimPassBuffersQuery {
                slot_code,
                particle_capacity,
                grid_cell_count,
                constraint_count,
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
        eprintln!("skipping sim-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSimPassBuffers::new(&ctx);
    let queries = structured_queries();
    let results = check(&ctx, &gpu, &queries);

    // Non-trivial: both writability classes, both persistence classes and a
    // spread of byte sizes appear, so a degenerate constant kernel could not
    // pass.
    let writable = results.iter().filter(|r| r.is_output == 1).count();
    let read_only = results.iter().filter(|r| r.is_output == 0).count();
    assert!(writable > 0, "fixture must include writable buffers");
    assert!(read_only > 0, "fixture must include read-only buffers");
    let persistent = results
        .iter()
        .filter(|r| r.persists_across_frames == 1)
        .count();
    let transient = results
        .iter()
        .filter(|r| r.persists_across_frames == 0)
        .count();
    assert!(persistent > 0, "fixture must include persistent pools");
    assert!(transient > 0, "fixture must include transient buffers");
    let min_bytes = results.iter().map(|r| r.byte_size).min().unwrap_or(0);
    let max_bytes = results.iter().map(|r| r.byte_size).max().unwrap_or(0);
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
fn gpu_matches_cpu_on_empty_extent_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sim-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSimPassBuffers::new(&ctx);

    // A zero extent clamps every buffer's *byte size* up to one element's stride
    // (so the WebGPU binding stays non-empty), while the raw element count stays
    // unclamped: 0 for the extent-driven pools and 1 for the single-element uniform.
    let queries: Vec<GpuSimPassBuffersQuery> = (0..SLOT_COUNT)
        .map(|slot_code| GpuSimPassBuffersQuery {
            slot_code,
            particle_capacity: 0,
            grid_cell_count: 0,
            constraint_count: 0,
        })
        .collect();
    let results = check(&ctx, &gpu, &queries);
    for (q, r) in queries.iter().zip(results.iter()) {
        assert_eq!(
            r.byte_size, r.stride,
            "empty extent reserves exactly one element's worth of bytes for slot {}",
            q.slot_code
        );
        assert!(
            r.element_count <= 1,
            "a zero extent leaves the extent-driven buffers empty (0) and the              single-element uniform at 1 for slot {}, got {}",
            q.slot_code, r.element_count
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
        eprintln!("skipping sim-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSimPassBuffers::new(&ctx);

    // Several thousand in-contract queries spanning many workgroups.
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let count = 8192usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        queries.push(draw_query(&mut state));
    }
    let results = check(&ctx, &gpu, &queries);

    // The batch mixes writable / read-only and persistent / transient slots, so
    // a degenerate single-class kernel could not pass.
    assert!(
        results.iter().any(|r| r.is_output == 1),
        "random batch should include writable buffers"
    );
    assert!(
        results.iter().any(|r| r.is_output == 0),
        "random batch should include read-only buffers"
    );
    assert!(
        results.iter().any(|r| r.persists_across_frames == 1),
        "random batch should include persistent pools"
    );
    assert!(
        results.iter().any(|r| r.persists_across_frames == 0),
        "random batch should include transient buffers"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sim-pass-buffers parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSimPassBuffers::new(&ctx);
    let results = gpu.evaluate(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield empty output");
}
