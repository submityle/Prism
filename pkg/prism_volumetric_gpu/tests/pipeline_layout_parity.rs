//! Real-device parity for the cross-pass bind-group sizing twin:
//! [`GpuPipelineLayout`](prism_volumetric_gpu::pipeline_layout::GpuPipelineLayout)
//! must reproduce the `CPU` golden
//! [`particle::pipeline_layout`](prism_render_architecture::particle::pipeline_layout)
//! across its three per-query answers — the `@group(0)` binding count
//! [`pass_binding_count`](prism_render_architecture::particle::pipeline_layout::pass_binding_count),
//! the clamped total declared byte size
//! [`pass_declared_bytes`](prism_render_architecture::particle::pipeline_layout::pass_declared_bytes)
//! and the handoff stride-consistency flag
//! [`Handoff::is_consistent`](prism_render_architecture::particle::pipeline_layout::Handoff::is_consistent)
//! — over every one of the ten
//! [`FramePass`](prism_render_architecture::particle::frame_pipeline::FramePass)
//! steps, a spread of whole-frame extents (including the fully-zero frame and
//! the clamp-to-one edges) and a large random batch spanning many workgroups.
//! The whole-frame roll-ups
//! [`frame_binding_count`](prism_render_architecture::particle::pipeline_layout::frame_binding_count)
//! and
//! [`frame_upper_bound_bytes`](prism_render_architecture::particle::pipeline_layout::frame_upper_bound_bytes)
//! are checked against their device folds.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every value is a `u32` count, a `u32` byte size or a discrete `1`/`0` flag
//! with no rounding anywhere on the path, so the outputs are bit-identical and
//! asserted with exact `==` and no tolerance. Fixtures mask every element count
//! to sixteen bits and the `radix` bucket and workgroup counts to ten bits, so
//! every per-buffer product, every per-pass sum and the whole-frame roll-up all
//! stay well below `2^31` and the device `u32` arithmetic never wraps where the
//! golden `saturating_mul` / `saturating_add` would otherwise clamp. Several
//! scenarios additionally assert a non-trivial spread of binding counts, byte
//! sizes and both consistency-flag values, so a degenerate constant kernel
//! could not pass.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::pipeline_layout`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::frame_pipeline::FramePass;
use prism_render_architecture::particle::pipeline_layout::{
    frame_binding_count, frame_upper_bound_bytes, pass_binding_count, pass_declared_bytes, Handoff,
    HandoffResource, ParticleFrameExtent,
};
use prism_volumetric_gpu::pipeline_layout::{
    GpuPipelineLayout, GpuPipelineLayoutQuery, GpuPipelineLayoutResult,
};
use prism_volumetric_gpu::GpuContext;

/// The whole-frame `@group(0)` binding tally the golden
/// [`frame_binding_count`](prism_render_architecture::particle::pipeline_layout::frame_binding_count)
/// returns: `4 + 6 + 8 + 4 + 3 + 2 + 4 + 4 + 3 + 3`.
const FRAME_BINDING_TOTAL: u32 = 41;

/// Rebuilds the whole-frame
/// [`ParticleFrameExtent`](prism_render_architecture::particle::pipeline_layout::ParticleFrameExtent)
/// a query carries, so the host can call the golden sizing reference on it.
fn extent_of(q: GpuPipelineLayoutQuery) -> ParticleFrameExtent {
    ParticleFrameExtent {
        emitter_count: q.emitter_count,
        particle_capacity: q.particle_capacity,
        spawn_count: q.spawn_count,
        alive_count: q.alive_count,
        visible_count: q.visible_count,
        grid_cell_count: q.grid_cell_count,
        constraint_count: q.constraint_count,
        event_source_capacity: q.event_source_capacity,
        event_channel_count: q.event_channel_count,
        event_scattered_capacity: q.event_scattered_capacity,
        radix_buckets: q.radix_buckets,
        workgroup_count: q.workgroup_count,
    }
}

/// Projects an extent to the twelve-scalar array the frame roll-up helpers
/// consume, in `ParticleFrameExtent` declaration order.
fn extent_array(e: ParticleFrameExtent) -> [u32; 12] {
    [
        e.emitter_count,
        e.particle_capacity,
        e.spawn_count,
        e.alive_count,
        e.visible_count,
        e.grid_cell_count,
        e.constraint_count,
        e.event_source_capacity,
        e.event_channel_count,
        e.event_scattered_capacity,
        e.radix_buckets,
        e.workgroup_count,
    ]
}

/// Builds one query for `pass_code` over `extent`, with the two probe strides
/// driving the consistency flag.
fn make_query(
    extent: ParticleFrameExtent,
    pass_code: u32,
    a_stride: u32,
    b_stride: u32,
) -> GpuPipelineLayoutQuery {
    GpuPipelineLayoutQuery {
        pass_code,
        a_stride,
        b_stride,
        emitter_count: extent.emitter_count,
        particle_capacity: extent.particle_capacity,
        spawn_count: extent.spawn_count,
        alive_count: extent.alive_count,
        visible_count: extent.visible_count,
        grid_cell_count: extent.grid_cell_count,
        constraint_count: extent.constraint_count,
        event_source_capacity: extent.event_source_capacity,
        event_channel_count: extent.event_channel_count,
        event_scattered_capacity: extent.event_scattered_capacity,
        radix_buckets: extent.radix_buckets,
        workgroup_count: extent.workgroup_count,
    }
}

/// The golden stride-consistency flag, driven through the real
/// [`Handoff::is_consistent`](prism_render_architecture::particle::pipeline_layout::Handoff::is_consistent)
/// predicate the kernel twins.
fn golden_consistent(a_stride: u32, b_stride: u32) -> u32 {
    let handoff = Handoff {
        resource: HandoffResource::PoolPositions,
        producer: FramePass::SimulationStages,
        consumer: FramePass::Bounds,
        producer_stride: a_stride as usize,
        consumer_stride: b_stride as usize,
    };
    u32::from(handoff.is_consistent())
}

/// Computes the golden answer for one query by calling the `CPU` reference
/// directly, matching exactly what the twin must reproduce.
fn golden_result(q: GpuPipelineLayoutQuery) -> GpuPipelineLayoutResult {
    let pass = FramePass::ALL[q.pass_code as usize];
    let extent = extent_of(q);
    GpuPipelineLayoutResult {
        binding_count: pass_binding_count(pass),
        declared_bytes: pass_declared_bytes(pass, extent) as u32,
        consistent: golden_consistent(q.a_stride, q.b_stride),
    }
}

/// Runs the twin over `queries` and asserts per-element parity against the
/// golden, returning the device results for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuPipelineLayout,
    queries: &[GpuPipelineLayoutQuery],
) -> Vec<GpuPipelineLayoutResult> {
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

/// Draws a `bits`-wide unsigned scalar from the generator.
fn next_masked(state: &mut u64, bits: u32) -> u32 {
    ((lcg(state) >> 29) as u32) & ((1u32 << bits) - 1)
}

/// Draws a random in-contract whole-frame extent. Element counts are masked to
/// sixteen bits and the `radix` bucket / workgroup knobs to ten bits, so the
/// largest per-pass sum (the sort histograms) stays near `9e6` and the whole
/// frame near `4e7` — far below `2^31`, so neither the device multiply nor the
/// golden `saturating_mul` clamps.
fn draw_extent(state: &mut u64) -> ParticleFrameExtent {
    ParticleFrameExtent {
        emitter_count: next_masked(state, 16),
        particle_capacity: next_masked(state, 16),
        spawn_count: next_masked(state, 16),
        alive_count: next_masked(state, 16),
        visible_count: next_masked(state, 16),
        grid_cell_count: next_masked(state, 16),
        constraint_count: next_masked(state, 16),
        event_source_capacity: next_masked(state, 16),
        event_channel_count: next_masked(state, 16),
        event_scattered_capacity: next_masked(state, 16),
        radix_buckets: next_masked(state, 10),
        workgroup_count: next_masked(state, 10),
    }
}

/// The probe strides the fixtures choose between; they cover the `std430`
/// scalar / record widths the pipeline actually declares, so both the
/// consistent and inconsistent branches of the flag are exercised.
const PROBE_STRIDES: [u32; 7] = [4, 8, 16, 20, 32, 64, 128];

/// Draws a random single-pass query: a random pass code, a random extent and a
/// random pair of probe strides.
fn draw_query(state: &mut u64) -> GpuPipelineLayoutQuery {
    let extent = draw_extent(state);
    let pass_code = (lcg(state) >> 40) as u32 % (FramePass::ALL.len() as u32);
    let a = PROBE_STRIDES[(lcg(state) >> 37) as usize % PROBE_STRIDES.len()];
    let b = PROBE_STRIDES[(lcg(state) >> 41) as usize % PROBE_STRIDES.len()];
    make_query(extent, pass_code, a, b)
}

/// A spread of whole-frame extents covering the fully-zero frame (every buffer
/// clamps up to one element), the all-ones edge, a production-typical frame and
/// frames with the `radix` / workgroup knobs zeroed so the sort / bounds
/// clamp-to-one paths are exercised.
fn structured_extents() -> Vec<ParticleFrameExtent> {
    let zero = ParticleFrameExtent::default();
    let ones = ParticleFrameExtent {
        emitter_count: 1,
        particle_capacity: 1,
        spawn_count: 1,
        alive_count: 1,
        visible_count: 1,
        grid_cell_count: 1,
        constraint_count: 1,
        event_source_capacity: 1,
        event_channel_count: 1,
        event_scattered_capacity: 1,
        radix_buckets: 1,
        workgroup_count: 1,
    };
    let typical = ParticleFrameExtent {
        emitter_count: 4,
        particle_capacity: 4096,
        spawn_count: 128,
        alive_count: 2048,
        visible_count: 1024,
        grid_cell_count: 512,
        constraint_count: 256,
        event_source_capacity: 1024,
        event_channel_count: 8,
        event_scattered_capacity: 2048,
        radix_buckets: 256,
        workgroup_count: 16,
    };
    let zero_knobs = ParticleFrameExtent {
        radix_buckets: 0,
        workgroup_count: 0,
        ..typical
    };
    let wide = ParticleFrameExtent {
        emitter_count: 37,
        particle_capacity: 60_000,
        spawn_count: 9_000,
        alive_count: 55_000,
        visible_count: 40_000,
        grid_cell_count: 8_192,
        constraint_count: 12_000,
        event_source_capacity: 30_000,
        event_channel_count: 64,
        event_scattered_capacity: 50_000,
        radix_buckets: 256,
        workgroup_count: 1_000,
    };
    vec![zero, ones, typical, zero_knobs, wide]
}

/// Builds the cross-product of every structured extent and every pass, pairing
/// each with alternating equal / unequal probe strides so both consistency-flag
/// values appear across the batch.
fn structured_queries() -> Vec<GpuPipelineLayoutQuery> {
    let mut queries = Vec::new();
    for (i, extent) in structured_extents().into_iter().enumerate() {
        for pass_code in 0u32..(FramePass::ALL.len() as u32) {
            // Alternate between a consistent (equal) and an inconsistent
            // (unequal) stride probe so the flag takes both values.
            let (a, b) = if (i as u32 + pass_code).is_multiple_of(2) {
                (16, 16)
            } else {
                (16, 8)
            };
            queries.push(make_query(extent, pass_code, a, b));
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
        eprintln!("skipping pipeline-layout parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuPipelineLayout::new(&ctx);
    let queries = structured_queries();
    let results = check(&ctx, &gpu, &queries);

    // Non-trivial: the binding counts span a range (passes declare 2..=8
    // bindings), the byte sizes span a range, and both consistency-flag values
    // appear, so a degenerate constant kernel could not pass.
    let min_bind = results.iter().map(|r| r.binding_count).min().unwrap_or(0);
    let max_bind = results.iter().map(|r| r.binding_count).max().unwrap_or(0);
    assert!(
        min_bind < max_bind,
        "fixture should span a range of binding counts, got [{min_bind}, {max_bind}]"
    );
    let min_bytes = results.iter().map(|r| r.declared_bytes).min().unwrap_or(0);
    let max_bytes = results.iter().map(|r| r.declared_bytes).max().unwrap_or(0);
    assert!(
        min_bytes < max_bytes,
        "fixture should span a range of byte sizes, got [{min_bytes}, {max_bytes}]"
    );
    let consistent = results.iter().filter(|r| r.consistent == 1).count();
    let inconsistent = results.iter().filter(|r| r.consistent == 0).count();
    assert!(consistent > 0, "fixture must include consistent handoffs");
    assert!(
        inconsistent > 0,
        "fixture must include inconsistent handoffs"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_zero_frame_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping pipeline-layout parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuPipelineLayout::new(&ctx);

    // A fully-zero frame: every buffer still clamps up to one element, so every
    // pass declares a positive byte size and never zero.
    let extent = ParticleFrameExtent::default();
    let queries: Vec<GpuPipelineLayoutQuery> = (0u32..(FramePass::ALL.len() as u32))
        .map(|pass_code| make_query(extent, pass_code, 16, 16))
        .collect();
    let results = check(&ctx, &gpu, &queries);
    for (q, r) in queries.iter().zip(results.iter()) {
        assert!(
            r.declared_bytes > 0,
            "pass {} must clamp to a non-empty binding",
            q.pass_code
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_stride_consistency_flag() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping pipeline-layout parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuPipelineLayout::new(&ctx);

    let extent = structured_extents()[2]; // the production-typical frame
    let queries = vec![
        make_query(extent, 2, 16, 16),  // equal strides -> consistent
        make_query(extent, 2, 16, 8),   // unequal -> inconsistent
        make_query(extent, 5, 32, 32),  // equal -> consistent
        make_query(extent, 7, 128, 20), // unequal -> inconsistent
    ];
    let results = check(&ctx, &gpu, &queries);
    assert_eq!(results[0].consistent, 1, "equal strides are consistent");
    assert_eq!(results[1].consistent, 0, "unequal strides are inconsistent");
    assert_eq!(results[2].consistent, 1, "equal strides are consistent");
    assert_eq!(results[3].consistent, 0, "unequal strides are inconsistent");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_on_large_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping pipeline-layout parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuPipelineLayout::new(&ctx);

    // Several thousand in-contract queries spanning many workgroups.
    let mut state = 0x0BAD_F00D_1234_5678u64;
    let count = 8192usize;
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        queries.push(draw_query(&mut state));
    }
    let results = check(&ctx, &gpu, &queries);

    // The batch mixes every pass and both consistency values, so a degenerate
    // single-class kernel could not pass.
    let min_bind = results.iter().map(|r| r.binding_count).min().unwrap_or(0);
    let max_bind = results.iter().map(|r| r.binding_count).max().unwrap_or(0);
    assert!(
        min_bind < max_bind,
        "random batch should span binding counts"
    );
    assert!(
        results.iter().any(|r| r.consistent == 1),
        "random batch should include consistent handoffs"
    );
    assert!(
        results.iter().any(|r| r.consistent == 0),
        "random batch should include inconsistent handoffs"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_frame_rollups_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping pipeline-layout parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuPipelineLayout::new(&ctx);

    for extent in structured_extents() {
        let arr = extent_array(extent);
        let device_bindings = gpu.frame_binding_count(&ctx, arr);
        assert_eq!(
            device_bindings,
            frame_binding_count(),
            "device frame binding count must match the golden"
        );
        assert_eq!(
            device_bindings, FRAME_BINDING_TOTAL,
            "the whole-frame binding tally is a fixed 41"
        );
        let device_bytes = gpu.frame_upper_bound_bytes(&ctx, arr);
        assert_eq!(
            device_bytes,
            frame_upper_bound_bytes(extent) as u32,
            "device frame byte upper bound must match the golden"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping pipeline-layout parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuPipelineLayout::new(&ctx);
    let results = gpu.evaluate(&ctx, &[]);
    assert!(results.is_empty(), "empty input must yield empty output");
}
